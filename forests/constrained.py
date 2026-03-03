"""
forests.constrained
===================
Monotonic and linearity-constrained forest models.

Monotonic constraint: forces a feature's split direction to satisfy
    sign(f) > 0 → only keep splits where left_mean < right_mean
    sign(f) < 0 → only keep splits where left_mean > right_mean

Linearity constraint: penalizes splits on feature j if the feature
relationship is already well-captured by a linear model. The split gain
is discounted by the fraction of variance explained by a linear fit on j.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_is_fitted

from .base import BaseForest, BaseTree, Node, RegressorForestMixin, IMPURITY_FN
from .cart import CARTRegressor


# ---------------------------------------------------------------------------
# Monotonic constraint helpers
# ---------------------------------------------------------------------------

def _monotone_split(
    X: np.ndarray,
    y: np.ndarray,
    feature_indices: np.ndarray,
    impurity_fn,
    min_samples_leaf: int,
    monotone_constraints: Dict[int, int],  # {feature_idx: +1 or -1}
) -> Tuple[Optional[int], Optional[float], float]:
    """Axis-aligned split with monotone constraints.

    For constrained feature j with direction d:
      d=+1 → require mean(y_left) <= mean(y_right)
      d=-1 → require mean(y_left) >= mean(y_right)
    """
    n = len(y)
    base_imp = impurity_fn(y)
    best_gain = 0.0
    best_feature: Optional[int] = None
    best_threshold: Optional[float] = None

    for f in feature_indices:
        vals = X[:, f]
        uniq = np.unique(vals)
        if len(uniq) < 2:
            continue
        thresholds = (uniq[:-1] + uniq[1:]) / 2.0
        direction = monotone_constraints.get(int(f), 0)

        for thr in thresholds:
            lm = vals <= thr
            nl, nr = lm.sum(), n - lm.sum()
            if nl < min_samples_leaf or nr < min_samples_leaf:
                continue

            yl, yr = y[lm], y[~lm]
            # Monotone constraint check
            if direction == 1 and np.mean(yl) > np.mean(yr):
                continue
            if direction == -1 and np.mean(yl) < np.mean(yr):
                continue

            gain = base_imp - (nl / n) * impurity_fn(yl) - (nr / n) * impurity_fn(yr)
            if gain > best_gain:
                best_gain = gain
                best_feature = int(f)
                best_threshold = float(thr)

    return best_feature, best_threshold, best_gain


# ---------------------------------------------------------------------------
# Monotonic Constrained Tree / Forest
# ---------------------------------------------------------------------------

class _MonotonicTree(CARTRegressor):
    """Single regression tree with monotone constraints."""

    def __init__(self, monotone_constraints: Optional[Dict[int, int]] = None, **kwargs):
        super().__init__(**kwargs)
        self.monotone_constraints = monotone_constraints or {}

    def _find_best_split(self, X, y, feature_indices, rng, **kwargs):
        fn = IMPURITY_FN[self.criterion]
        return _monotone_split(
            X, y, feature_indices, fn, self.min_samples_leaf,
            self.monotone_constraints
        )


class MonotonicConstrainedForest(RegressorForestMixin, BaseForest):
    """Random Forest with per-feature monotone constraints.

    Parameters
    ----------
    monotone_constraints : dict of {int: int}
        Mapping from feature index to constraint direction.
        +1 = monotone increasing, -1 = monotone decreasing.
    n_estimators : int, default=100
    criterion : {"mse", "mae"}, default="mse"
    max_depth : int or None
    min_samples_split : int, default=2
    min_samples_leaf : int, default=1
    max_features : int, float, str, or None, default="sqrt"
    bootstrap : bool, default=True
    n_jobs : int, default=1
    random_state : int or None

    Examples
    --------
    >>> from forests import MonotonicConstrainedForest
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((100, 2))
    >>> y = X[:, 0] + rng.normal(0, 0.1, 100)  # y increases with X[:,0]
    >>> clf = MonotonicConstrainedForest(
    ...     monotone_constraints={0: 1}, n_estimators=10, random_state=0
    ... )
    >>> clf.fit(X, y)
    MonotonicConstrainedForest(...)
    """

    def __init__(
        self,
        monotone_constraints: Optional[Dict[int, int]] = None,
        n_estimators: int = 100,
        criterion: str = "mse",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        max_features: Union[int, float, str, None] = "sqrt",
        bootstrap: bool = True,
        max_samples=None,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            bootstrap=bootstrap,
            max_samples=max_samples,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )
        self.monotone_constraints = monotone_constraints or {}
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features

    def _make_estimator(self, random_state: int) -> _MonotonicTree:
        return _MonotonicTree(
            monotone_constraints=self.monotone_constraints,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            max_features=self.max_features,
            random_state=random_state,
        )


# ---------------------------------------------------------------------------
# Linearity constraint
# ---------------------------------------------------------------------------

def _linearity_penalized_split(
    X: np.ndarray,
    y: np.ndarray,
    feature_indices: np.ndarray,
    impurity_fn,
    min_samples_leaf: int,
    linear_features: List[int],  # features to apply linearity constraint
    linearity_lambda: float,       # strength of the penalty
) -> Tuple[Optional[int], Optional[float], float]:
    """Split with linearity constraint.

    For constrained features, discount gain by R² of linear fit:
      effective_gain = gain * (1 - linearity_lambda * R2_j)

    A high R² means the feature is already well-described linearly,
    so the split is discounted to prefer linear modeling over recursive splitting.
    """
    n = len(y)
    base_imp = impurity_fn(y)
    best_gain = 0.0
    best_feature: Optional[int] = None
    best_threshold: Optional[float] = None

    # Pre-compute R² for each constrained feature
    r2_cache: Dict[int, float] = {}
    for f in linear_features:
        xf = X[:, f]
        # Simple OLS R²
        xm, ym = xf.mean(), y.mean()
        ss_res_num = np.sum((xf - xm) * (y - ym))
        ss_x = np.sum((xf - xm) ** 2)
        if ss_x < 1e-12:
            r2_cache[f] = 0.0
        else:
            b1 = ss_res_num / ss_x
            y_pred = ym + b1 * (xf - xm)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - ym) ** 2)
            r2_cache[f] = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0

    for f in feature_indices:
        vals = X[:, f]
        uniq = np.unique(vals)
        if len(uniq) < 2:
            continue
        thresholds = (uniq[:-1] + uniq[1:]) / 2.0

        r2 = r2_cache.get(int(f), 0.0)
        discount = 1.0 - linearity_lambda * max(0.0, r2)

        for thr in thresholds:
            lm = vals <= thr
            nl, nr = lm.sum(), n - lm.sum()
            if nl < min_samples_leaf or nr < min_samples_leaf:
                continue
            gain = base_imp - (nl / n) * impurity_fn(y[lm]) - (nr / n) * impurity_fn(y[~lm])
            effective_gain = gain * discount
            if effective_gain > best_gain:
                best_gain = effective_gain
                best_feature = int(f)
                best_threshold = float(thr)

    return best_feature, best_threshold, best_gain


class _LinearityTree(CARTRegressor):
    """Single tree with linearity constraint."""

    def __init__(
        self,
        linear_features: Optional[List[int]] = None,
        linearity_lambda: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.linear_features = linear_features or []
        self.linearity_lambda = linearity_lambda

    def _find_best_split(self, X, y, feature_indices, rng, **kwargs):
        fn = IMPURITY_FN[self.criterion]
        return _linearity_penalized_split(
            X, y, feature_indices, fn, self.min_samples_leaf,
            self.linear_features, self.linearity_lambda
        )


class LinearConstrainedForest(RegressorForestMixin, BaseForest):
    """Random Forest with linearity constraints on specified features.

    For features in `linear_features`, the split gain is discounted
    proportional to the local R² of a linear fit, encouraging the
    model to preserve linear relationships where they exist.

    Parameters
    ----------
    linear_features : list of int
        Feature indices to apply linearity constraint.
    linearity_lambda : float, default=0.5
        Strength of linearity penalty (0=no effect, 1=fully linear).
    n_estimators : int, default=100
    criterion : {"mse", "mae"}, default="mse"
    max_depth : int or None
    min_samples_split : int, default=2
    min_samples_leaf : int, default=1
    max_features : int, float, str, or None, default="sqrt"
    bootstrap : bool, default=True
    n_jobs : int, default=1
    random_state : int or None

    Examples
    --------
    >>> from forests import LinearConstrainedForest
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((100, 3))
    >>> y = 2 * X[:, 0] + rng.normal(0, 0.1, 100)
    >>> reg = LinearConstrainedForest(linear_features=[0], linearity_lambda=0.8,
    ...                               n_estimators=10, random_state=0)
    >>> reg.fit(X, y)
    LinearConstrainedForest(...)
    """

    def __init__(
        self,
        linear_features: Optional[List[int]] = None,
        linearity_lambda: float = 0.5,
        n_estimators: int = 100,
        criterion: str = "mse",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        max_features: Union[int, float, str, None] = "sqrt",
        bootstrap: bool = True,
        max_samples=None,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            bootstrap=bootstrap,
            max_samples=max_samples,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )
        self.linear_features = linear_features or []
        self.linearity_lambda = linearity_lambda
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features

    def _make_estimator(self, random_state: int) -> _LinearityTree:
        return _LinearityTree(
            linear_features=self.linear_features,
            linearity_lambda=self.linearity_lambda,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            max_features=self.max_features,
            random_state=random_state,
        )
