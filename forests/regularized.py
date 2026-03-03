"""
forests.regularized
===================
Regularized decision trees and forests.

Two regularization schemes:
1. Variable-reuse penalty  — encourages use of previously-used features
   by penalizing features that have NOT been used yet (inverse usage).
2. Leaf-weight regularization — L1/L2 penalty on leaf weight magnitudes.

References
----------
Johnson, R., & Zhang, T. (2014). Learning Nonlinear Functions Using Regularized
Greedy Forest. IEEE TPAMI, 36(5), 942-954.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from .base import BaseForest, BaseTree, Node, ClassifierForestMixin, RegressorForestMixin, IMPURITY_FN
from .cart import CARTClassifier, CARTRegressor


# ---------------------------------------------------------------------------
# Variable-reuse penalty split search
# ---------------------------------------------------------------------------

def _penalized_split(
    X: np.ndarray,
    y: np.ndarray,
    feature_indices: np.ndarray,
    impurity_fn,
    min_samples_leaf: int,
    n_classes: Optional[int],
    usage_count: np.ndarray,    # int array shape (n_features,)
    reuse_alpha: float,         # penalty coefficient
) -> Tuple[Optional[int], Optional[float], float]:
    """Axis-aligned split with variable-reuse penalty.

    Features used less get a negative boost (penalty) on their effective gain.
    penalty_j = reuse_alpha * (max_usage - usage_count[j])

    This makes already-used features relatively more attractive.

    Parameters
    ----------
    usage_count : np.ndarray of shape (n_features,)
        Number of times each feature has been used as a split so far.
    reuse_alpha : float
        Penalty weight. Larger → stronger preference for re-used features.
    """
    n = len(y)
    if n_classes is not None:
        base_imp = impurity_fn(y, n_classes)
    else:
        base_imp = impurity_fn(y)

    best_gain = -np.inf
    best_feature: Optional[int] = None
    best_threshold: Optional[float] = None
    max_usage = usage_count.max() if usage_count.max() > 0 else 1

    for f in feature_indices:
        vals = X[:, f]
        uniq = np.unique(vals)
        if len(uniq) < 2:
            continue
        thresholds = (uniq[:-1] + uniq[1:]) / 2.0

        # Penalty: features not used receive negative bonus
        penalty = reuse_alpha * (max_usage - usage_count[f])

        for thr in thresholds:
            lm = vals <= thr
            nl, nr = lm.sum(), n - lm.sum()
            if nl < min_samples_leaf or nr < min_samples_leaf:
                continue
            if n_classes is not None:
                imp_l = impurity_fn(y[lm], n_classes)
                imp_r = impurity_fn(y[~lm], n_classes)
            else:
                imp_l = impurity_fn(y[lm])
                imp_r = impurity_fn(y[~lm])
            raw_gain = base_imp - (nl / n) * imp_l - (nr / n) * imp_r
            effective_gain = raw_gain - penalty
            if effective_gain > best_gain:
                best_gain = effective_gain
                best_feature = int(f)
                best_threshold = float(thr)

    return best_feature, best_threshold, max(0.0, best_gain)


# ---------------------------------------------------------------------------
# VariablePenaltyTree (single tree)
# ---------------------------------------------------------------------------

class _VPClassifierTree(CARTClassifier):
    """Classification tree with variable-reuse penalty.

    Implements the usage-count-based penalty described in regularized.py docstring.
    """

    def __init__(self, reuse_alpha: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.reuse_alpha = reuse_alpha

    def _build(self, X, y, depth, rng, **kwargs):
        """Override to pass usage_count through the tree."""
        n_features = X.shape[1]
        usage_count = kwargs.pop("usage_count", np.zeros(n_features, dtype=int))
        node = super()._build(X, y, depth, rng, usage_count=usage_count, **kwargs)
        return node

    def _find_best_split(self, X, y, feature_indices, rng, **kwargs):
        usage_count = kwargs.get("usage_count", np.zeros(X.shape[1], dtype=int))
        fn = IMPURITY_FN[self.criterion]
        f, thr, gain = _penalized_split(
            X, y, feature_indices, fn, self.min_samples_leaf,
            self.n_classes_, usage_count, self.reuse_alpha
        )
        if f is not None:
            usage_count[f] += 1
        return f, thr, gain

    def fit(self, X, y, **kwargs):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        label_map = {c: i for i, c in enumerate(self.classes_)}
        y_mapped = np.array([label_map[yi] for yi in y])
        # Initialize usage count
        kwargs.setdefault("usage_count", np.zeros(X.shape[1], dtype=int))
        from .base import BaseTree
        BaseTree.fit(self, X, y_mapped, **kwargs)
        return self


class _VPRegressorTree(CARTRegressor):
    """Regression tree with variable-reuse penalty."""

    def __init__(self, reuse_alpha: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.reuse_alpha = reuse_alpha

    def _find_best_split(self, X, y, feature_indices, rng, **kwargs):
        usage_count = kwargs.get("usage_count", np.zeros(X.shape[1], dtype=int))
        fn = IMPURITY_FN[self.criterion]
        f, thr, gain = _penalized_split(
            X, y, feature_indices, fn, self.min_samples_leaf,
            None, usage_count, self.reuse_alpha
        )
        if f is not None:
            usage_count[f] += 1
        return f, thr, gain

    def fit(self, X, y, **kwargs):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        kwargs.setdefault("usage_count", np.zeros(X.shape[1], dtype=int))
        from .base import BaseTree
        BaseTree.fit(self, X, y, **kwargs)
        return self


# ---------------------------------------------------------------------------
# VariablePenaltyForest
# ---------------------------------------------------------------------------

class VariablePenaltyForest(ClassifierForestMixin, BaseForest):
    """Random Forest with variable-reuse penalty.

    Features that have been used as split variables more often get lower
    penalty (i.e., they remain attractive for re-use). Unused features
    receive a penalty proportional to `reuse_alpha * usage_deficit`.

    This is a custom regularization scheme that encourages focused use
    of informative features rather than spreading splits across all features.

    Parameters
    ----------
    n_estimators : int, default=100
    reuse_alpha : float, default=0.1
        Penalty weight on unused features.
    criterion : {"gini", "entropy"}, default="gini"
    max_depth : int or None
    min_samples_split : int, default=2
    min_samples_leaf : int, default=1
    max_features : int, float, str, or None, default="sqrt"
    bootstrap : bool, default=True
    n_jobs : int, default=1
    random_state : int or None

    Examples
    --------
    >>> from forests import VariablePenaltyForest
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = VariablePenaltyForest(n_estimators=10, reuse_alpha=0.05, random_state=0)
    >>> clf.fit(X, y).score(X, y) > 0.8
    True
    """

    def __init__(
        self,
        n_estimators: int = 100,
        reuse_alpha: float = 0.1,
        criterion: str = "gini",
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
        self.reuse_alpha = reuse_alpha
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features

    def _make_estimator(self, random_state: int) -> _VPClassifierTree:
        return _VPClassifierTree(
            reuse_alpha=self.reuse_alpha,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            max_features=self.max_features,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "VariablePenaltyForest":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        super().fit(X, y, **kwargs)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "estimators_")
        X = np.asarray(X, dtype=float)
        all_proba = [
            np.array([tree._predict_node(x, tree.root_) for x in X])
            for tree in self.estimators_
        ]
        return np.mean(all_proba, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]


# ---------------------------------------------------------------------------
# Leaf-Weight Regularized Forest
# ---------------------------------------------------------------------------

class LeafWeightRegularizedForest(RegressorForestMixin, BaseForest):
    """Random Forest with post-hoc leaf weight regularization (L1 or L2).

    After each tree is grown, leaf weights (mean predictions) are shrunk
    toward zero via L1 (soft-thresholding) or L2 (ridge shrinkage).

    L2 shrinkage: w_leaf = w_leaf / (1 + alpha)
    L1 shrinkage: w_leaf = sign(w_leaf) * max(|w_leaf| - alpha, 0)

    Parameters
    ----------
    n_estimators : int, default=100
    leaf_reg : {"none", "l1", "l2"}, default="l2"
        Regularization type.
    alpha : float, default=0.1
        Regularization strength.
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
    >>> from forests import LeafWeightRegularizedForest
    >>> from sklearn.datasets import load_diabetes
    >>> X, y = load_diabetes(return_X_y=True)
    >>> reg = LeafWeightRegularizedForest(n_estimators=10, alpha=0.1, random_state=0)
    >>> reg.fit(X, y).score(X, y) > 0.5
    True
    """

    def __init__(
        self,
        n_estimators: int = 100,
        leaf_reg: str = "l2",
        alpha: float = 0.1,
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
        self.leaf_reg = leaf_reg
        self.alpha = alpha
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features

    def _make_estimator(self, random_state: int) -> CARTRegressor:
        return CARTRegressor(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            max_features=self.max_features,
            random_state=random_state,
        )

    def _regularize_leaves(self, tree: CARTRegressor) -> None:
        """Apply leaf weight shrinkage in-place."""
        from .base import Node
        for node in tree._iter_nodes(tree.root_):
            if node.is_leaf and node.value is not None:
                w = float(node.value[0])
                if self.leaf_reg == "l2":
                    node.value = np.array([w / (1.0 + self.alpha)])
                elif self.leaf_reg == "l1":
                    node.value = np.array([np.sign(w) * max(abs(w) - self.alpha, 0.0)])
                # "none": no change

    def _fit_single(self, seed, X, y, fit_kwargs):
        tree = super()._fit_single(seed, X, y, fit_kwargs)
        self._regularize_leaves(tree)
        return tree
