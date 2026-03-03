"""
forests.sporf
=============
Sparse Projection Oblique Randomer Forest (SPORF).

References
----------
Tomita, T.M., et al. (2020). Sparse Projection Oblique Randomer Forests.
    Journal of Machine Learning Research, 21(104), 1-39.

Algorithm
---------
At each node, instead of selecting a random subset of features (like RF),
SPORF draws a *sparse* random projection vector w where only `d` out of p
features have nonzero weights (drawn from {-1, +1} with equal probability).
The projected signal X @ w is then used as a 1D split variable.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from .base import (
    BaseForest, BaseTree, ClassifierForestMixin, RegressorForestMixin, IMPURITY_FN
)
from .cart import CARTClassifier, CARTRegressor
from .oblique import _ObliqueClassifierTree, _ObliqueRegressorTree


# ---------------------------------------------------------------------------
# SPORF sparse projection split
# ---------------------------------------------------------------------------

def _sporf_split(
    X: np.ndarray,
    y: np.ndarray,
    impurity_fn,
    min_samples_leaf: int,
    n_classes: Optional[int],
    rng: np.random.Generator,
    n_projections: int,
    density: float,
) -> Tuple[Optional[np.ndarray], Optional[float], float]:
    """SPORF: sparse {-1, +1} projection split search.

    Implements Tomita et al. (2020) JMLR, Algorithm 1.
    Each projection vector w is drawn with each element independently:
        w_j = 0     with prob 1 - density
        w_j = ±1    with prob density / 2 each

    Parameters
    ----------
    density : float
        Expected fraction of nonzero weights (lambda / p in the paper).
    n_projections : int
        Number of random projections to try (like n_features in std RF).
    """
    n, p = X.shape
    if n_classes is not None:
        base_imp = impurity_fn(y, n_classes)
    else:
        base_imp = impurity_fn(y)

    best_gain = 0.0
    best_w: Optional[np.ndarray] = None
    best_thr: Optional[float] = None

    for _ in range(n_projections):
        # Sparse {-1, +1} vector
        nonzero_mask = rng.random(p) < density
        if not nonzero_mask.any():
            nonzero_mask[rng.integers(p)] = True
        signs = rng.choice([-1.0, 1.0], size=p)
        w = np.where(nonzero_mask, signs, 0.0)

        proj = X @ w
        uniq = np.unique(proj)
        if len(uniq) < 2:
            continue
        thresholds = (uniq[:-1] + uniq[1:]) / 2.0
        if len(thresholds) > 20:
            thresholds = rng.choice(thresholds, size=20, replace=False)

        for thr in thresholds:
            lm = proj <= thr
            nl, nr = lm.sum(), n - lm.sum()
            if nl < min_samples_leaf or nr < min_samples_leaf:
                continue
            if n_classes is not None:
                imp_l = impurity_fn(y[lm], n_classes)
                imp_r = impurity_fn(y[~lm], n_classes)
            else:
                imp_l = impurity_fn(y[lm])
                imp_r = impurity_fn(y[~lm])
            gain = base_imp - (nl / n) * imp_l - (nr / n) * imp_r
            if gain > best_gain:
                best_gain = gain
                best_w = w.copy()
                best_thr = float(thr)

    return best_w, best_thr, best_gain


# ---------------------------------------------------------------------------
# SPORF Tree
# ---------------------------------------------------------------------------

def _make_sporf_build(base_cls, is_classifier: bool):
    """Factory to create SPORF tree class inheriting from base_cls."""

    class SPORFTree(base_cls):
        """Single SPORF tree."""

        def __init__(self, n_projections: int = 10, density: float = 0.1, **kwargs):
            super().__init__(**kwargs)
            self.n_projections = n_projections
            self.density = density

        def _find_best_split(self, X, y, feature_indices, rng, **kwargs):
            fn = IMPURITY_FN[self.criterion]
            nc = self.n_classes_ if is_classifier else None
            return _sporf_split(
                X, y, fn, self.min_samples_leaf, nc, rng,
                self.n_projections, self.density
            )

        def _build(self, X, y, depth, rng, **kwargs):
            n_samples, n_features = X.shape
            from .base import Node
            impurity = self._impurity(y)
            node = Node(
                value=self._node_value(y),
                impurity=impurity,
                n_samples=n_samples,
                depth=depth,
            )
            too_deep = self.max_depth is not None and depth >= self.max_depth
            too_few = n_samples < self.min_samples_split
            pure = impurity == 0.0
            if too_deep or too_few or pure:
                node.leaf_id = self._leaf_counter
                self._leaf_counter += 1
                return node

            feat_idx = self._select_features(n_features, rng)
            w, thr, gain = self._find_best_split(X, y, feat_idx, rng, **kwargs)
            if w is None or gain < self.min_impurity_decrease:
                node.leaf_id = self._leaf_counter
                self._leaf_counter += 1
                return node

            proj = X @ w
            mask = proj <= thr
            lX, ly = X[mask], y[mask]
            rX, ry = X[~mask], y[~mask]
            if len(ly) < self.min_samples_leaf or len(ry) < self.min_samples_leaf:
                node.leaf_id = self._leaf_counter
                self._leaf_counter += 1
                return node

            node.extra["oblique_w"] = w
            node.threshold = thr
            node.feature = -1
            node.left = self._build(lX, ly, depth + 1, rng, **kwargs)
            node.right = self._build(rX, ry, depth + 1, rng, **kwargs)
            return node

        def _predict_node(self, x, node):
            if node.is_leaf:
                return node.value
            w = node.extra.get("oblique_w")
            proj = float(x @ w) if w is not None else float(x[node.feature])
            if proj <= node.threshold:
                return self._predict_node(x, node.left)
            return self._predict_node(x, node.right)

        def _apply_node(self, x, node):
            if node.is_leaf:
                return node.leaf_id
            w = node.extra.get("oblique_w")
            proj = float(x @ w) if w is not None else float(x[node.feature])
            if proj <= node.threshold:
                return self._apply_node(x, node.left)
            return self._apply_node(x, node.right)

    return SPORFTree


_SPORFClassifierTree = _make_sporf_build(CARTClassifier, True)
_SPORFRegressorTree = _make_sporf_build(CARTRegressor, False)


# ---------------------------------------------------------------------------
# SPORFClassifier / SPORFRegressor
# ---------------------------------------------------------------------------

class SPORFClassifier(ClassifierForestMixin, BaseForest):
    """Sparse Projection Oblique Randomer Forest - Classifier.

    Implements: Tomita et al. (2020) JMLR, Algorithm 1.

    Parameters
    ----------
    n_estimators : int, default=100
    n_projections : int, default=10
        Number of random projections per node.
    density : float, default=0.1
        Expected fraction of nonzero elements in projection vector.
    criterion : {"gini", "entropy"}, default="gini"
    max_depth : int or None
    min_samples_split : int, default=2
    min_samples_leaf : int, default=1
    bootstrap : bool, default=True
    n_jobs : int, default=1
    random_state : int or None

    Examples
    --------
    >>> from forests import SPORFClassifier
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = SPORFClassifier(n_estimators=10, random_state=0)
    >>> clf.fit(X, y).score(X, y) > 0.8
    True
    """

    def __init__(
        self,
        n_estimators: int = 100,
        n_projections: int = 10,
        density: float = 0.1,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
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
        self.n_projections = n_projections
        self.density = density
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease

    def _make_estimator(self, random_state: int):
        return _SPORFClassifierTree(
            n_projections=self.n_projections,
            density=self.density,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            random_state=random_state,
        )

    def fit(self, X, y, **kwargs):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        super().fit(X, y, **kwargs)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "estimators_")
        X = np.asarray(X, dtype=float)
        all_proba = [
            np.array([tree._predict_node(x, tree.root_) for x in X])
            for tree in self.estimators_
        ]
        return np.mean(all_proba, axis=0)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class SPORFRegressor(RegressorForestMixin, BaseForest):
    """Sparse Projection Oblique Randomer Forest - Regressor.

    Implements: Tomita et al. (2020) JMLR.

    Parameters
    ----------
    n_estimators : int, default=100
    n_projections : int, default=10
    density : float, default=0.1
    criterion : {"mse", "mae"}, default="mse"
    max_depth : int or None
    min_samples_split : int, default=2
    min_samples_leaf : int, default=1
    bootstrap : bool, default=True
    n_jobs : int, default=1
    random_state : int or None

    Examples
    --------
    >>> from forests import SPORFRegressor
    >>> from sklearn.datasets import load_diabetes
    >>> X, y = load_diabetes(return_X_y=True)
    >>> reg = SPORFRegressor(n_estimators=10, random_state=0)
    >>> reg.fit(X, y).score(X, y) > 0.5
    True
    """

    def __init__(
        self,
        n_estimators: int = 100,
        n_projections: int = 10,
        density: float = 0.1,
        criterion: str = "mse",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
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
        self.n_projections = n_projections
        self.density = density
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease

    def _make_estimator(self, random_state: int):
        return _SPORFRegressorTree(
            n_projections=self.n_projections,
            density=self.density,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            random_state=random_state,
        )
