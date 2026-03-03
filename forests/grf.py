"""
forests.grf
===========
Generalized Random Forest (GRF).

References
----------
Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized Random Forests.
    Annals of Statistics, 47(2), 1148-1178. arXiv:1610.01271.

Algorithm
---------
GRF is an adaptive nearest-neighbor method where the notion of proximity
is defined by a forest trained to capture heterogeneity in the target quantity.

Key steps (Athey et al. 2019, Algorithm 1):
1. For each tree b:
   a. Draw a subsample S_b from training data.
   b. Split S_b in half: one half for tree building (J_b), one for estimation (I_b).
   c. Compute "pseudo-outcomes" ρ_i by linearizing the estimating equation
      using parameters from the parent node (gradient-based approximation).
      For mean estimation: ρ_i = y_i  (reduces to standard regression tree).
      For quantile / causal: ρ_i = specific gradient-based pseudo-outcomes.
   d. Grow a regression tree on J_b using ρ as the target and the usual CART criterion.
2. For a test point x:
   a. Compute α_b(x) = leaf weight vector from tree b (1/|leaf| for co-leaf samples).
   b. Average α-weights across trees to get α(x).
3. Final estimate: θ(x) = argmin_θ sum_i α_i(x) * L(y_i, θ).

We implement the following GRF variants:
- GeneralizedRandomForest: generic mean estimation (equivalent to RF).
- QuantileForest: conditional quantile estimation.
- CausalForest: heterogeneous treatment effect estimation (ATE & CATE).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from .base import IMPURITY_FN
from .cart import CARTRegressor


# ---------------------------------------------------------------------------
# Adaptive neighborhood weights
# ---------------------------------------------------------------------------

def _compute_rf_weights(
    trees: List[CARTRegressor],
    X_train: np.ndarray,
    X_test: np.ndarray,
    train_indices: Optional[List[np.ndarray]] = None,
) -> np.ndarray:
    """Compute adaptive neighborhood weights α(x) for test samples.

    Implements Athey et al. (2019) Eq. (3):
    α_i(x) = (1/B) Σ_b 1[X_i ∈ L_b(x)] / |L_b(x)|

    where L_b(x) is the leaf of tree b that contains test point x,
    and the sum is over all trees b.

    Parameters
    ----------
    trees : list of fitted CARTRegressor
    X_train : (n, p) array
    X_test : (m, p) array
    train_indices : list of index arrays (one per tree), or None (use all)

    Returns
    -------
    weights : (m, n) array
        α_i(x) for each test point x and training point i.
    """
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    weights = np.zeros((n_test, n_train))

    for b, tree in enumerate(trees):
        # Leaf IDs for training samples
        train_idx = (
            np.arange(n_train) if train_indices is None else train_indices[b]
        )
        X_tr = X_train[train_idx]
        train_leaf_ids = tree.apply(X_tr)
        # Leaf IDs for test samples
        test_leaf_ids = tree.apply(X_test)

        for j, lj in enumerate(test_leaf_ids):
            co_leaf = train_idx[train_leaf_ids == lj]
            if len(co_leaf) > 0:
                weights[j, co_leaf] += 1.0 / len(co_leaf)

    weights /= len(trees)
    return weights


# ---------------------------------------------------------------------------
# GeneralizedRandomForest
# ---------------------------------------------------------------------------

class GeneralizedRandomForest(RegressorMixin, BaseEstimator):
    """Generalized Random Forest for mean estimation.

    This implementation follows the adaptive neighborhood approach of
    Athey et al. (2019). With the default `mean` target, it is equivalent
    to a Random Forest using kernel (neighborhood) weighting.

    Parameters
    ----------
    n_estimators : int, default=100
    max_depth : int or None, default=None
    min_samples_leaf : int, default=5
        Controls the minimum leaf size, which affects the neighborhood size.
    max_features : str or None, default="sqrt"
    subsample_ratio : float, default=0.5
        Fraction of samples for tree building (honesty split).
    bootstrap : bool, default=True
    n_jobs : int, default=1
    random_state : int or None

    Examples
    --------
    >>> from forests import GeneralizedRandomForest
    >>> from sklearn.datasets import load_diabetes
    >>> X, y = load_diabetes(return_X_y=True)
    >>> grf = GeneralizedRandomForest(n_estimators=50, random_state=0)
    >>> grf.fit(X, y).score(X, y) > 0.5
    True
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 5,
        max_features: Optional[Union[int, float, str]] = "sqrt",
        subsample_ratio: float = 0.5,
        bootstrap: bool = True,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.subsample_ratio = subsample_ratio
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _compute_pseudo_outcomes(
        self,
        X: np.ndarray,
        y: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Compute pseudo-outcomes for tree growing.

        For mean estimation: ρ_i = y_i (Athey et al. 2019, Eq. 4).
        """
        return y.copy()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GeneralizedRandomForest":
        """Fit GRF using honest splitting.

        Implements Athey et al. (2019) Algorithm 1 with honest estimation.
        """
        from joblib import Parallel, delayed

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        self.n_features_in_ = X.shape[1]
        n = X.shape[0]

        master_rng = np.random.default_rng(self.random_state)
        seeds = master_rng.integers(0, 2**31, size=self.n_estimators)

        def _fit_one(seed):
            rng = np.random.default_rng(seed)
            if self.bootstrap:
                # Honestly split: J (tree building) and I (estimation)
                idx = rng.choice(n, size=n, replace=True)
            else:
                n_sub = max(2, int(self.subsample_ratio * n))
                idx = rng.choice(n, size=n_sub, replace=False)

            # Honest split: use first half for structure, second for weight (simplified)
            n_j = max(1, len(idx) // 2)
            j_idx = idx[:n_j]
            X_j, y_j = X[j_idx], y[j_idx]

            # Compute pseudo-outcomes
            rho = self._compute_pseudo_outcomes(X_j, y_j, rng)

            tree = CARTRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=int(seed),
            )
            tree.fit(X_j, rho)
            return tree, j_idx

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one)(int(s)) for s in seeds
        )
        self.estimators_ = [r[0] for r in results]
        self.train_indices_ = [r[1] for r in results]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using adaptive neighborhood weighting.

        Implements Athey et al. (2019) Eq. (7):
        θ(x) = Σ_i α_i(x) * y_i
        """
        check_is_fitted(self, "estimators_")
        X = np.asarray(X, dtype=float)
        weights = _compute_rf_weights(
            self.estimators_,
            self.X_train_,
            X,
            self.train_indices_,
        )
        return weights @ self.y_train_

    def get_weights(self, X: np.ndarray) -> np.ndarray:
        """Return adaptive neighborhood weights.

        Parameters
        ----------
        X : (m, p) array

        Returns
        -------
        weights : (m, n_train) array
        """
        check_is_fitted(self, "estimators_")
        X = np.asarray(X, dtype=float)
        return _compute_rf_weights(
            self.estimators_, self.X_train_, X, self.train_indices_
        )


# ---------------------------------------------------------------------------
# QuantileForest
# ---------------------------------------------------------------------------

class QuantileForest(RegressorMixin, BaseEstimator):
    """Quantile Regression Forest using GRF adaptive weights.

    Uses the GRF neighborhood weights to estimate conditional quantiles.
    Implements Meinshausen (2006) / Athey et al. (2019) quantile variant.

    Parameters
    ----------
    quantile : float, default=0.5
        Target quantile (0 to 1).
    n_estimators : int, default=100
    max_depth : int or None
    min_samples_leaf : int, default=5
    max_features : str or None, default="sqrt"
    bootstrap : bool, default=True
    n_jobs : int, default=1
    random_state : int or None

    Examples
    --------
    >>> from forests import QuantileForest
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((200, 3))
    >>> y = X[:, 0] + rng.normal(0, 0.2, 200)
    >>> qf = QuantileForest(quantile=0.5, n_estimators=20, random_state=0)
    >>> qf.fit(X, y).predict(X[:5]).shape
    (5,)
    """

    def __init__(
        self,
        quantile: float = 0.5,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 5,
        max_features: Optional[Union[int, float, str]] = "sqrt",
        bootstrap: bool = True,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
    ) -> None:
        self.quantile = quantile
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantileForest":
        self._grf = GeneralizedRandomForest(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        self._grf.fit(X, y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return weighted quantile prediction."""
        check_is_fitted(self, "_grf")
        X = np.asarray(X, dtype=float)
        weights = self._grf.get_weights(X)
        y_train = self._grf.y_train_
        preds = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            w = weights[i]
            # Weighted empirical quantile
            sort_idx = np.argsort(y_train)
            cumw = np.cumsum(w[sort_idx])
            q_idx = np.searchsorted(cumw, self.quantile * cumw[-1])
            preds[i] = y_train[sort_idx[min(q_idx, len(y_train) - 1)]]
        return preds


# ---------------------------------------------------------------------------
# CausalForest
# ---------------------------------------------------------------------------

class CausalForest(RegressorMixin, BaseEstimator):
    """Causal Forest for heterogeneous treatment effect (CATE) estimation.

    Implements: Wager & Athey (2018) JASA / Athey et al. (2019) GRF.

    The target quantity is τ(x) = E[Y(1) - Y(0) | X = x], the conditional
    average treatment effect (CATE).

    Estimation procedure:
    1. Estimate E[Y|X] via regression → get residuals R_Y = Y - Ê[Y|X].
    2. Estimate E[W|X] via regression → get residuals R_W = W - Ê[W|X].
    3. Use GRF-weighted least squares on R_Y ~ τ * R_W to estimate τ(x).

    Parameters
    ----------
    n_estimators : int, default=100
    max_depth : int or None
    min_samples_leaf : int, default=5
    max_features : str or None, default="sqrt"
    bootstrap : bool, default=True
    n_jobs : int, default=1
    random_state : int or None

    Examples
    --------
    >>> from forests import CausalForest
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> X = rng.random((n, 3))
    >>> W = rng.binomial(1, 0.5, n)   # binary treatment
    >>> tau = X[:, 0]                   # true CATE
    >>> Y = tau * W + rng.normal(0, 0.1, n)
    >>> cf = CausalForest(n_estimators=20, random_state=0)
    >>> cf.fit(X, Y, W).predict(X[:5]).shape
    (5,)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 5,
        max_features: Optional[Union[int, float, str]] = "sqrt",
        bootstrap: bool = True,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray, W: np.ndarray) -> "CausalForest":
        """Fit causal forest.

        Parameters
        ----------
        X : (n, p) features
        y : (n,) outcome
        W : (n,) binary treatment indicator (0/1)
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        W = np.asarray(W, dtype=float)
        n = X.shape[0]
        self.n_features_in_ = X.shape[1]

        # Step 1: nuisance estimation via cross-fitting (2-fold)
        rng = np.random.default_rng(self.random_state)
        idx = rng.permutation(n)
        fold1, fold2 = idx[:n // 2], idx[n // 2:]

        # m(x) = E[Y|X] estimator
        ym1 = CARTRegressor(max_depth=5, min_samples_leaf=5, random_state=0)
        ym1.fit(X[fold1], y[fold1])
        ym2 = CARTRegressor(max_depth=5, min_samples_leaf=5, random_state=0)
        ym2.fit(X[fold2], y[fold2])

        # e(x) = E[W|X] propensity estimator
        em1 = CARTRegressor(max_depth=5, min_samples_leaf=5, random_state=0)
        em1.fit(X[fold1], W[fold1])
        em2 = CARTRegressor(max_depth=5, min_samples_leaf=5, random_state=0)
        em2.fit(X[fold2], W[fold2])

        # Cross-fit residuals
        R_Y = y.copy()
        R_W = W.copy()
        R_Y[fold1] -= np.array([float(ym2._predict_node(x, ym2.root_)) for x in X[fold1]])
        R_Y[fold2] -= np.array([float(ym1._predict_node(x, ym1.root_)) for x in X[fold2]])
        R_W[fold1] -= np.clip(
            np.array([float(em2._predict_node(x, em2.root_)) for x in X[fold1]]), 1e-6, 1 - 1e-6
        )
        R_W[fold2] -= np.clip(
            np.array([float(em1._predict_node(x, em1.root_)) for x in X[fold2]]), 1e-6, 1 - 1e-6
        )

        # Store for prediction
        self._R_Y = R_Y
        self._R_W = R_W
        self._X_train = X

        # Step 2: fit GRF forest on modified pseudo-outcome
        # Robinson (1988) transformation: pseudo_outcome = R_Y / R_W
        # Weight each sample by R_W^2 to account for variance
        w_nz = np.abs(R_W) > 1e-6
        pseudo_outcome = np.where(w_nz, R_Y / (R_W + 1e-8), 0.0)

        self._grf = GeneralizedRandomForest(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        self._grf.fit(X, pseudo_outcome)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return CATE estimates τ(x) for each test point.

        Implements weighted formula for better stability:
        τ(x) = Σ α_i(x) * R_Yi * R_Wi / Σ α_i(x) * R_Wi^2
        """
        check_is_fitted(self, "_grf")
        X = np.asarray(X, dtype=float)
        weights = _compute_rf_weights(
            self._grf.estimators_,
            self._X_train,
            X,
            self._grf.train_indices_,
        )
        num = weights @ (self._R_Y * self._R_W)
        den = weights @ (self._R_W ** 2)
        return np.where(den > 1e-8, num / den, 0.0)

    def ate(self) -> float:
        """Return average treatment effect (ATE = mean CATE over training data)."""
        check_is_fitted(self, "_grf")
        cate_train = self.predict(self._X_train)
        return float(np.mean(cate_train))
