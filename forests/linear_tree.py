"""
forests.linear_tree
===================
LinearTree, LinearForest, and LinearBoost.

A LinearTree fits a linear regression model at each leaf of a decision tree.
This creates a piecewise-linear model that is more expressive than simple
step-function trees while maintaining tree interpretability.

References
----------
Potts, D., & Sammut, C. (2005). Incremental learning of linear model trees.
    Machine Learning, 61(1), 5-48.

Fraiman, R., Justel, A., & Svarc, M. (2008). A new regression tree method
    with linear models at the leaves. Lecture Notes in Statistics.

Algorithm
---------
1. Build a decision tree (CART splits) on X to define regions/leaves.
2. For each leaf region, fit an OLS linear regression model on the samples
   in that region.
3. Prediction: route x to its leaf, then apply that leaf's linear model.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from .base import BaseForest, BaseTree, Node, IMPURITY_FN
from .cart import CARTRegressor


# ---------------------------------------------------------------------------
# OLS helper
# ---------------------------------------------------------------------------

def _fit_ols(X: np.ndarray, y: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """Fit OLS (with optional ridge) and return coefficient vector [w, b].

    Solves: min ||Xw + b - y||^2 + ridge * ||w||^2
    """
    n, p = X.shape
    # Augment for bias
    Xa = np.column_stack([X, np.ones(n)])
    # Ridge on features only (not bias)
    R = ridge * np.eye(p + 1)
    R[-1, -1] = 0.0  # no penalty on bias
    coef = np.linalg.solve(Xa.T @ Xa + R, Xa.T @ y)
    return coef  # shape (p+1,): [w_0, ..., w_{p-1}, b]


# ---------------------------------------------------------------------------
# LinearTree
# ---------------------------------------------------------------------------

class LinearTree(RegressorMixin, BaseEstimator):
    """Decision Tree with linear models at the leaves.

    Builds a CART tree for partitioning, then fits an OLS linear
    regression on the leaf samples. Prediction applies the linear
    model of the matched leaf.

    Parameters
    ----------
    max_depth : int or None, default=4
    min_samples_leaf : int, default=10
        Minimum samples to fit a leaf linear model reliably.
    max_features : int, float, str, or None, default=None
    ridge : float, default=1e-3
        Ridge regularization for leaf OLS.
    criterion : {"mse", "mae"}, default="mse"
    random_state : int or None

    Examples
    --------
    >>> from forests import LinearTree
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((100, 3))
    >>> y = 2 * X[:, 0] - X[:, 1] + rng.normal(0, 0.1, 100)
    >>> lt = LinearTree(max_depth=3, min_samples_leaf=5, random_state=0)
    >>> lt.fit(X, y).score(X, y) > 0.9
    True
    """

    def __init__(
        self,
        max_depth: Optional[int] = 4,
        min_samples_leaf: int = 10,
        max_features: Optional[Union[int, float, str]] = None,
        ridge: float = 1e-3,
        criterion: str = "mse",
        random_state: Optional[int] = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.ridge = ridge
        self.criterion = criterion
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearTree":
        """Fit tree structure then fit leaf linear models."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]

        # Step 1: grow CART tree for partitioning
        self._tree = CARTRegressor(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
        )
        self._tree.fit(X, y)

        # Step 2: fit leaf linear models
        leaf_ids = self._tree.apply(X)
        self._leaf_models: Dict[int, np.ndarray] = {}
        for lid in np.unique(leaf_ids):
            mask = leaf_ids == lid
            X_leaf, y_leaf = X[mask], y[mask]
            if X_leaf.shape[0] < 2:
                # Fallback: constant model
                self._leaf_models[lid] = np.append(
                    np.zeros(X.shape[1]), float(y_leaf.mean()) if len(y_leaf) > 0 else 0.0
                )
            else:
                self._leaf_models[lid] = _fit_ols(X_leaf, y_leaf, self.ridge)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the leaf linear model."""
        check_is_fitted(self, "_tree")
        X = np.asarray(X, dtype=float)
        leaf_ids = self._tree.apply(X)
        preds = np.zeros(X.shape[0])
        for i, (x, lid) in enumerate(zip(X, leaf_ids)):
            coef = self._leaf_models[lid]
            # coef = [w_0, ..., w_{p-1}, b]
            preds[i] = float(x @ coef[:-1] + coef[-1])
        return preds


# ---------------------------------------------------------------------------
# LinearForest
# ---------------------------------------------------------------------------

class LinearForest(RegressorMixin, BaseEstimator):
    """Forest of LinearTrees.

    Averages predictions from multiple LinearTree models, each trained
    on a bootstrap sample of the data.

    Parameters
    ----------
    n_estimators : int, default=100
    max_depth : int or None, default=4
    min_samples_leaf : int, default=10
    max_features : int, float, str, or None, default="sqrt"
    ridge : float, default=1e-3
    bootstrap : bool, default=True
    n_jobs : int, default=1
    random_state : int or None

    Examples
    --------
    >>> from forests import LinearForest
    >>> from sklearn.datasets import load_diabetes
    >>> X, y = load_diabetes(return_X_y=True)
    >>> lf = LinearForest(n_estimators=10, random_state=0)
    >>> lf.fit(X, y).score(X, y) > 0.6
    True
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 4,
        min_samples_leaf: int = 10,
        max_features: Optional[Union[int, float, str]] = "sqrt",
        ridge: float = 1e-3,
        bootstrap: bool = True,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.ridge = ridge
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearForest":
        from joblib import Parallel, delayed
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]
        master_rng = np.random.default_rng(self.random_state)
        seeds = master_rng.integers(0, 2**31, size=self.n_estimators)

        def _fit_one(seed):
            rng = np.random.default_rng(seed)
            if self.bootstrap:
                idx = rng.integers(0, X.shape[0], size=X.shape[0])
                X_s, y_s = X[idx], y[idx]
            else:
                X_s, y_s = X, y
            lt = LinearTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                ridge=self.ridge,
                random_state=int(seed),
            )
            lt.fit(X_s, y_s)
            return lt

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one)(int(s)) for s in seeds
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "estimators_")
        X = np.asarray(X, dtype=float)
        return np.mean([lt.predict(X) for lt in self.estimators_], axis=0)


# ---------------------------------------------------------------------------
# LinearBoost
# ---------------------------------------------------------------------------

class LinearBoost(RegressorMixin, BaseEstimator):
    """Linear Boosting: sequential LinearTrees on residuals.

    Parameters
    ----------
    n_estimators : int, default=50
    learning_rate : float, default=0.1
    max_depth : int or None, default=3
    min_samples_leaf : int, default=5
    max_features : int, float, str, or None, default="sqrt"
    ridge : float, default=1e-3
    random_state : int or None

    Examples
    --------
    >>> from forests import LinearBoost
    >>> from sklearn.datasets import load_diabetes
    >>> X, y = load_diabetes(return_X_y=True)
    >>> lb = LinearBoost(n_estimators=30, learning_rate=0.1, random_state=0)
    >>> lb.fit(X, y).score(X, y) > 0.5
    True
    """

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 0.1,
        max_depth: Optional[int] = 3,
        min_samples_leaf: int = 5,
        max_features: Optional[Union[int, float, str]] = "sqrt",
        ridge: float = 1e-3,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.ridge = ridge
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearBoost":
        """Fit LinearBoost via stagewise boosting on residuals."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.random_state)

        self.init_pred_ = float(np.mean(y))
        self.estimators_: List[LinearTree] = []
        self.n_features_in_ = X.shape[1]

        residuals = y - self.init_pred_

        for t in range(self.n_estimators):
            seed = int(rng.integers(0, 2**31))
            lt = LinearTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                ridge=self.ridge,
                random_state=seed,
            )
            lt.fit(X, residuals)
            pred = lt.predict(X)
            residuals = residuals - self.learning_rate * pred
            self.estimators_.append(lt)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "estimators_")
        X = np.asarray(X, dtype=float)
        pred = np.full(X.shape[0], self.init_pred_)
        for lt in self.estimators_:
            pred += self.learning_rate * lt.predict(X)
        return pred
