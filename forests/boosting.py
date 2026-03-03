"""
forests.boosting
================
Gradient Boosted Trees (GBT) from scratch.

References
----------
Friedman, J.H. (2001). Greedy function approximation: A gradient boosting machine.
    Annals of Statistics, 29(5), 1189-1232.

Friedman, J.H. (2002). Stochastic gradient boosting.
    Computational Statistics & Data Analysis, 38(4), 367-378.

Algorithm
---------
1. Initialize: F_0(x) = argmin_γ Σ L(y_i, γ)  (mean for MSE, log-odds for cross-entropy)
2. For m = 1..M:
   a. Compute pseudo-residuals: r_im = -[∂L(y_i, F(x_i))/∂F(x_i)]
   b. Fit a regression tree h_m to residuals.
   c. Compute optimal leaf weights γ_jm by line search.
   d. Update: F_m(x) = F_{m-1}(x) + η * Σ_j γ_jm * 1[x ∈ R_jm]
3. Predict: F_M(x)

Key design choices:
- Subsampling (stochastic GBM): each tree uses fraction `subsample` of data.
- Shrinkage (learning rate η): controls step size.
- Max features per split: controls diversity.
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from .cart import CARTRegressor


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def _mse_negative_gradient(y: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Negative gradient of MSE loss = residuals."""
    return y - F


def _logistic_negative_gradient(y: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Negative gradient of binary cross-entropy.

    L = -[y log σ(F) + (1-y) log(1-σ(F))]
    -∂L/∂F = y - σ(F)
    """
    sigma = 1.0 / (1.0 + np.exp(-np.clip(F, -30, 30)))
    return y - sigma


def _softmax(F: np.ndarray) -> np.ndarray:
    """Softmax of (n, K) array."""
    F_max = F.max(axis=1, keepdims=True)
    exp_F = np.exp(F - F_max)
    return exp_F / exp_F.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# GradientBoostedRegressor
# ---------------------------------------------------------------------------

class GradientBoostedRegressor(RegressorMixin, BaseEstimator):
    """Gradient Boosted Trees for regression.

    Implements: Friedman (2001) Annals of Statistics.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting stages.
    learning_rate : float, default=0.1
        Shrinkage factor η ∈ (0, 1].
    max_depth : int, default=3
        Maximum depth of each regression tree.
    min_samples_leaf : int, default=5
    max_features : str or None, default="sqrt"
    subsample : float, default=1.0
        Fraction of samples used per tree (stochastic GBM).
    loss : {"mse", "mae"}, default="mse"
    random_state : int or None

    Examples
    --------
    >>> from forests import GradientBoostedRegressor
    >>> from sklearn.datasets import load_diabetes
    >>> X, y = load_diabetes(return_X_y=True)
    >>> gbr = GradientBoostedRegressor(n_estimators=100, random_state=0)
    >>> gbr.fit(X, y).score(X, y) > 0.8
    True
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_leaf: int = 5,
        max_features: Optional[Union[int, float, str]] = "sqrt",
        subsample: float = 1.0,
        loss: str = "mse",
        random_state: Optional[int] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.subsample = subsample
        self.loss = loss
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostedRegressor":
        """Fit GBT regressor via stagewise forward fitting."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]
        n = X.shape[0]
        rng = np.random.default_rng(self.random_state)

        # Initialize F_0 = mean(y)
        self.init_pred_ = float(np.mean(y))
        F = np.full(n, self.init_pred_)
        self.estimators_: List[CARTRegressor] = []
        self.train_scores_: List[float] = []

        for m in range(self.n_estimators):
            # a) pseudo-residuals
            residuals = _mse_negative_gradient(y, F)

            # b) subsampling (stochastic GBM, Friedman 2002)
            if self.subsample < 1.0:
                sample_idx = rng.choice(n, size=max(1, int(self.subsample * n)), replace=False)
                X_sub, r_sub = X[sample_idx], residuals[sample_idx]
            else:
                X_sub, r_sub = X, residuals

            seed = int(rng.integers(0, 2**31))
            tree = CARTRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=seed,
            )
            tree.fit(X_sub, r_sub)

            # c) leaf weight update (leaf-wise line search for MSE: optimal = mean residual in leaf)
            leaf_ids_all = tree.apply(X)
            for lid in np.unique(leaf_ids_all):
                mask = leaf_ids_all == lid
                # Find the leaf node and update value
                for node in tree._iter_nodes(tree.root_):
                    if node.is_leaf and node.leaf_id == lid:
                        node.value = np.array([float(residuals[mask].mean())])
                        break

            # Tree fits residuals. Preds are node.value[0]
            step = np.array([float(tree._predict_node(x, tree.root_)[0]) for x in X])
            F += self.learning_rate * step
            self.estimators_.append(tree)
            self.train_scores_.append(float(np.mean((y - F) ** 2)))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions."""
        check_is_fitted(self, "estimators_")
        X = np.asarray(X, dtype=float)
        F = np.full(X.shape[0], self.init_pred_)
        for tree in self.estimators_:
            step = np.array([float(tree._predict_node(x, tree.root_)[0]) for x in X])
            F += self.learning_rate * step
        return F

    def staged_predict(self, X: np.ndarray):
        """Yield predictions after each stage (for learning curve analysis)."""
        check_is_fitted(self, "estimators_")
        X = np.asarray(X, dtype=float)
        F = np.full(X.shape[0], self.init_pred_)
        for tree in self.estimators_:
            step = np.array([float(tree._predict_node(x, tree.root_)[0]) for x in X])
            F = F + self.learning_rate * step
            yield F.copy()


# ---------------------------------------------------------------------------
# GradientBoostedClassifier
# ---------------------------------------------------------------------------

class GradientBoostedClassifier(ClassifierMixin, BaseEstimator):
    """Gradient Boosted Trees for classification.

    Uses one-vs-all (OvR) strategy with softmax output for multi-class.
    Binary case: single tree sequence with logistic loss.

    Implements: Friedman (2001) §4.6 (multi-class GBM).

    Parameters
    ----------
    n_estimators : int, default=100
    learning_rate : float, default=0.1
    max_depth : int, default=3
    min_samples_leaf : int, default=5
    max_features : str or None, default="sqrt"
    subsample : float, default=1.0
    random_state : int or None

    Examples
    --------
    >>> from forests import GradientBoostedClassifier
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> gbc = GradientBoostedClassifier(n_estimators=100, random_state=0)
    >>> gbc.fit(X, y).score(X, y) > 0.95
    True
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_leaf: int = 5,
        max_features: Optional[Union[int, float, str]] = "sqrt",
        subsample: float = 1.0,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostedClassifier":
        """Fit GBT classifier.

        For K>2 classes: fits K trees per stage (one per class), using
        multinomial logistic loss and softmax (Friedman 2001, Eq. 19).
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        K = self.n_classes_
        n = X.shape[0]
        self.n_features_in_ = X.shape[1]
        rng = np.random.default_rng(self.random_state)

        label_map = {c: i for i, c in enumerate(self.classes_)}
        y_int = np.array([label_map[yi] for yi in y])
        Y = np.eye(K)[y_int]  # one-hot (n, K)

        # Initialize F: (n, K) — uniform class prob
        self.init_log_odds_ = np.log(np.bincount(y_int, minlength=K) / n + 1e-9)
        F = np.tile(self.init_log_odds_, (n, 1))  # (n, K)

        # Estimators: list of K-length lists (one list per stage)
        self.estimators_: List[List[Optional[CARTRegressor]]] = []

        for m in range(self.n_estimators):
            proba = _softmax(F)  # (n, K)
            stage_trees = []

            for k in range(K):
                # Pseudo-residuals for class k (Friedman 2001, Eq. 18)
                r_k = Y[:, k] - proba[:, k]

                if self.subsample < 1.0:
                    idx = rng.choice(n, size=max(1, int(self.subsample * n)), replace=False)
                    X_sub, r_sub = X[idx], r_k[idx]
                else:
                    X_sub, r_sub = X, r_k

                seed = int(rng.integers(0, 2**31))
                tree = CARTRegressor(
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    max_features=self.max_features,
                    random_state=seed,
                )
                tree.fit(X_sub, r_sub)

                # Leaf weight scaling (Friedman 2001, Eq. 19)
                leaf_ids = tree.apply(X)
                for lid in np.unique(leaf_ids):
                    mask = leaf_ids == lid
                    r_leaf = r_k[mask]
                    p_leaf = proba[mask, k]
                    sum_r = r_leaf.sum()
                    sum_p = (p_leaf * (1 - p_leaf)).sum()
                    gamma = (K / (K - 1)) * sum_r / (sum_p + 1e-8)
                    for node in tree._iter_nodes(tree.root_):
                        if node.is_leaf and node.leaf_id == lid:
                            node.value = np.array([gamma])
                            break

                step_k = np.array([float(tree._predict_node(x, tree.root_)[0]) for x in X])
                F[:, k] += self.learning_rate * step_k
                stage_trees.append(tree)

            self.estimators_.append(stage_trees)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities."""
        check_is_fitted(self, "estimators_")
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        F = np.tile(self.init_log_odds_, (n, 1))
        for stage_trees in self.estimators_:
            for k, tree in enumerate(stage_trees):
                step_k = np.array([float(tree._predict_node(x, tree.root_)[0]) for x in X])
                F[:, k] += self.learning_rate * step_k
        return _softmax(F)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class label predictions."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
