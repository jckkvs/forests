"""
forests.cart
============
CART (Classification and Regression Trees) implementation.

References
----------
Breiman, L., Friedman, J.H., Olshen, R.A., & Stone, C.J. (1984).
Classification and Regression Trees. Wadsworth & Brooks/Cole.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from .base import (
    BaseTree,
    Node,
    IMPURITY_FN,
    gini_impurity,
    entropy_impurity,
    mse_impurity,
    mae_impurity,
)


# ---------------------------------------------------------------------------
# Internal: axis-aligned split search
# ---------------------------------------------------------------------------

def _best_split_axis(
    X: np.ndarray,
    y: np.ndarray,
    feature_indices: np.ndarray,
    impurity_fn,
    min_samples_leaf: int,
    n_classes: Optional[int],
) -> Tuple[Optional[int], Optional[float], float]:
    """Exhaustive axis-aligned split search.

    Parameters
    ----------
    X : (n, p) array
    y : (n,) array
    feature_indices : subset of feature indices to search
    impurity_fn : callable(y) -> float
    min_samples_leaf : int
    n_classes : int or None (for classification)

    Returns
    -------
    best_feature, best_threshold, best_gain
    """
    n = len(y)
    base_impurity = impurity_fn(y) if n_classes is None else impurity_fn(y, n_classes)
    best_gain = 0.0
    best_feature: Optional[int] = None
    best_threshold: Optional[float] = None

    for f in feature_indices:
        vals = X[:, f]
        sort_idx = np.argsort(vals)
        vals_sorted = vals[sort_idx]
        y_sorted = y[sort_idx]

        # All candidate thresholds: midpoints between consecutive unique values
        unique_vals = np.unique(vals_sorted)
        if len(unique_vals) < 2:
            continue
        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

        for thr in thresholds:
            left_mask_sorted = vals_sorted <= thr
            n_left = left_mask_sorted.sum()
            n_right = n - n_left
            if n_left < min_samples_leaf or n_right < min_samples_leaf:
                continue

            y_left = y_sorted[left_mask_sorted]
            y_right = y_sorted[~left_mask_sorted]

            if n_classes is not None:
                imp_left = impurity_fn(y_left, n_classes)
                imp_right = impurity_fn(y_right, n_classes)
            else:
                imp_left = impurity_fn(y_left)
                imp_right = impurity_fn(y_right)

            gain = base_impurity - (n_left / n) * imp_left - (n_right / n) * imp_right

            if gain > best_gain:
                best_gain = gain
                best_feature = int(f)
                best_threshold = float(thr)

    return best_feature, best_threshold, best_gain


# ---------------------------------------------------------------------------
# CART Classifier
# ---------------------------------------------------------------------------

class CARTClassifier(BaseTree, ClassifierMixin):
    """CART Decision Tree for classification.

    Implements: Breiman et al. (1984) CART.
    Split criterion: Gini impurity or Information Gain (entropy).

    Parameters
    ----------
    criterion : {"gini", "entropy"}
        Splitting criterion.
    max_depth : int or None
        Maximum depth.
    min_samples_split : int
        Minimum samples to split a node.
    min_samples_leaf : int
        Minimum samples in each leaf.
    min_impurity_decrease : float
        Minimum impurity decrease to make a split.
    max_features : int, float, str, or None
        Features to consider at each split.
    class_weight : dict or "balanced" or None
        Class weights.
    random_state : int or None

    Examples
    --------
    >>> from forests import CARTClassifier
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = CARTClassifier(max_depth=3, random_state=0)
    >>> clf.fit(X, y).predict(X[:5])
    array([0, 0, 0, 0, 0])
    """

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        max_features: Optional[Union[int, float, str]] = None,
        class_weight=None,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            max_features=max_features,
            random_state=random_state,
        )
        self.criterion = criterion
        self.class_weight = class_weight

    def _impurity(self, y: np.ndarray) -> float:
        fn = IMPURITY_FN[self.criterion]
        return fn(y, self.n_classes_)

    def _node_value(self, y: np.ndarray) -> np.ndarray:
        """Return normalized class probability vector."""
        counts = np.bincount(y.astype(int), minlength=self.n_classes_)
        total = counts.sum()
        if total == 0:
            return np.ones(self.n_classes_) / self.n_classes_
        return counts / total

    def _find_best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_indices: np.ndarray,
        rng: np.random.Generator,
        **kwargs,
    ) -> Tuple[Optional[int], Optional[float], float]:
        fn = IMPURITY_FN[self.criterion]
        return _best_split_axis(
            X, y, feature_indices, fn, self.min_samples_leaf, self.n_classes_
        )

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "CARTClassifier":
        """Fit the classifier."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y)
        self.n_classes_: int = len(self.classes_)
        # Remap labels to 0..K-1
        label_map = {c: i for i, c in enumerate(self.classes_)}
        y_mapped = np.array([label_map[yi] for yi in y])
        super().fit(X, y_mapped, **kwargs)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability matrix.

        Parameters
        ----------
        X : (n, p) array

        Returns
        -------
        proba : (n, n_classes) array
        """
        if not hasattr(self, 'root_') or self.root_ is None:
            raise RuntimeError("Model is not fitted yet. Call 'fit' first.")
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_node(x, self.root_) for x in X])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class label predictions."""
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]


# ---------------------------------------------------------------------------
# CART Regressor
# ---------------------------------------------------------------------------

class CARTRegressor(BaseTree, RegressorMixin):
    """CART Decision Tree for regression.

    Implements: Breiman et al. (1984) CART.
    Split criterion: MSE or MAE.

    Parameters
    ----------
    criterion : {"mse", "mae", "friedman_mse"}
        Splitting criterion.
    max_depth : int or None
    min_samples_split : int
    min_samples_leaf : int
    min_impurity_decrease : float
    max_features : int, float, str, or None
    random_state : int or None

    Examples
    --------
    >>> from forests import CARTRegressor
    >>> import numpy as np
    >>> X = np.arange(20).reshape(10, 2).astype(float)
    >>> y = X[:, 0] * 2 + 1
    >>> reg = CARTRegressor(max_depth=3, random_state=0)
    >>> reg.fit(X, y)
    CARTRegressor(max_depth=3, random_state=0)
    """

    def __init__(
        self,
        criterion: str = "mse",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        max_features: Optional[Union[int, float, str]] = None,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            max_features=max_features,
            random_state=random_state,
        )
        self.criterion = criterion

    def _impurity(self, y: np.ndarray) -> float:
        return IMPURITY_FN[self.criterion](y)

    def _node_value(self, y: np.ndarray) -> np.ndarray:
        """Return mean value as 1-element array."""
        return np.array([np.mean(y) if len(y) > 0 else 0.0])

    def _find_best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_indices: np.ndarray,
        rng: np.random.Generator,
        **kwargs,
    ) -> Tuple[Optional[int], Optional[float], float]:
        fn = IMPURITY_FN[self.criterion]
        return _best_split_axis(
            X, y, feature_indices, fn, self.min_samples_leaf, None
        )

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "CARTRegressor":
        """Fit the regressor."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        super().fit(X, y, **kwargs)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return continuous predictions.

        Parameters
        ----------
        X : (n, p) array

        Returns
        -------
        y_pred : (n,) array
        """
        if not hasattr(self, 'root_') or self.root_ is None:
            raise RuntimeError("Model is not fitted yet. Call 'fit' first.")
        X = np.asarray(X, dtype=float)
        return np.array([float(self._predict_node(x, self.root_)) for x in X])
