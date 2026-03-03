"""
forests.oblique
===============
Oblique (non-axis-aligned) split decision forests:
- ObliqueForest     : Linear combination splits (CART-LC style)
- RotationForest    : PCA-based rotation preprocessing (Rodriguez et al. 2006)
- RandomRotationForest : Random orthogonal rotation matrix preprocessing

References
----------
Rodriguez, J.J., Kuncheva, L.I., & Alonso, C.J. (2006). Rotation forest:
    A new classifier ensemble method. IEEE TPAMI, 28(10), 1619–1630.

Murthy, S.K., Kasif, S., & Salzberg, S. (1994). A system for induction
    of oblique decision trees. JAIR, 2, 1–32.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_is_fitted

from .base import BaseForest, BaseTree, Node, ClassifierForestMixin, RegressorForestMixin, IMPURITY_FN
from .cart import CARTClassifier, CARTRegressor


# ---------------------------------------------------------------------------
# Oblique split: linear combinations of features
# ---------------------------------------------------------------------------

def _random_oblique_split(
    X: np.ndarray,
    y: np.ndarray,
    feature_indices: np.ndarray,
    impurity_fn,
    min_samples_leaf: int,
    n_classes: Optional[int],
    rng: np.random.Generator,
    n_directions: int = 5,
) -> Tuple[Optional[np.ndarray], Optional[float], float]:
    """Search best oblique (linear combination) split.

    Generates `n_directions` random weight vectors in the feature subspace,
    projects X onto each, then does a 1D threshold search.

    Returns
    -------
    best_w : np.ndarray of shape (n_features,) or None
    best_thr : float or None
    best_gain : float
    """
    n = len(y)
    n_features = X.shape[1]
    if n_classes is not None:
        base_imp = impurity_fn(y, n_classes)
    else:
        base_imp = impurity_fn(y)

    best_gain = 0.0
    best_w: Optional[np.ndarray] = None
    best_thr: Optional[float] = None

    k = len(feature_indices)
    for _ in range(n_directions):
        # Sparse random weight vector on selected features
        w_sub = rng.standard_normal(k)
        w_sub /= np.linalg.norm(w_sub) + 1e-12
        w = np.zeros(n_features)
        w[feature_indices] = w_sub

        proj = X @ w
        uniq = np.unique(proj)
        if len(uniq) < 2:
            continue
        thresholds = (uniq[:-1] + uniq[1:]) / 2.0
        # Subsample thresholds for speed
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
# ObliqueTree: stores weight vector in node.extra["oblique_w"]
# ---------------------------------------------------------------------------

class _ObliqueClassifierTree(CARTClassifier):
    """Single oblique decision tree for classification."""

    def __init__(self, n_directions: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.n_directions = n_directions

    def _find_best_split(self, X, y, feature_indices, rng, **kwargs):
        fn = IMPURITY_FN[self.criterion]
        return _random_oblique_split(
            X, y, feature_indices, fn, self.min_samples_leaf,
            self.n_classes_, rng, self.n_directions
        )

    def _build(self, X, y, depth, rng, **kwargs):
        """Override to handle oblique weight storage."""
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
        left_X, left_y = X[mask], y[mask]
        right_X, right_y = X[~mask], y[~mask]
        if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
            node.leaf_id = self._leaf_counter
            self._leaf_counter += 1
            return node

        # Store oblique weight instead of axis-split feature
        node.extra["oblique_w"] = w
        node.threshold = thr
        node.feature = -1  # sentinel for oblique
        node.left = self._build(left_X, left_y, depth + 1, rng, **kwargs)
        node.right = self._build(right_X, right_y, depth + 1, rng, **kwargs)
        return node

    def _predict_node(self, x, node):
        if node.is_leaf:
            return node.value
        w = node.extra.get("oblique_w")
        if w is not None:
            proj = float(x @ w)
        else:
            proj = float(x[node.feature])
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


class _ObliqueRegressorTree(CARTRegressor):
    """Single oblique decision tree for regression."""

    def __init__(self, n_directions: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.n_directions = n_directions

    def _find_best_split(self, X, y, feature_indices, rng, **kwargs):
        fn = IMPURITY_FN[self.criterion]
        return _random_oblique_split(
            X, y, feature_indices, fn, self.min_samples_leaf,
            None, rng, self.n_directions
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
        left_X, left_y = X[mask], y[mask]
        right_X, right_y = X[~mask], y[~mask]
        if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
            node.leaf_id = self._leaf_counter
            self._leaf_counter += 1
            return node

        node.extra["oblique_w"] = w
        node.threshold = thr
        node.feature = -1
        node.left = self._build(left_X, left_y, depth + 1, rng, **kwargs)
        node.right = self._build(right_X, right_y, depth + 1, rng, **kwargs)
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


# ---------------------------------------------------------------------------
# ObliqueForest
# ---------------------------------------------------------------------------

class ObliqueForest(ClassifierForestMixin, BaseForest):
    """Oblique Random Forest Classifier using linear combination splits."""

    def __init__(
        self,
        n_estimators: int = 100,
        n_directions: int = 5,
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
        self.n_directions = n_directions
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features

    def _make_estimator(self, random_state: int) -> _ObliqueClassifierTree:
        return _ObliqueClassifierTree(
            n_directions=self.n_directions,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            max_features=self.max_features,
            random_state=random_state,
        )

    def fit(self, X, y, **kwargs):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        super().fit(X, y, **kwargs)
        return self


class ObliqueForestRegressor(RegressorForestMixin, BaseForest):
    """Oblique Random Forest Regressor using linear combination splits."""

    def __init__(
        self,
        n_estimators: int = 100,
        n_directions: int = 5,
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
        self.n_directions = n_directions
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features

    def _make_estimator(self, random_state: int) -> _ObliqueRegressorTree:
        return _ObliqueRegressorTree(
            n_directions=self.n_directions,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            max_features=self.max_features,
            random_state=random_state,
        )



# ---------------------------------------------------------------------------
# RotationForest: PCA-based rotation (Rodriguez et al., 2006)
# ---------------------------------------------------------------------------

class RotationForest(ClassifierForestMixin, BaseForest):
    """Rotation Forest Classifier."""

    def __init__(
        self,
        n_estimators: int = 100,
        n_feature_groups: int = 3,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        pca_subsample: float = 0.75,
        max_features: Union[int, float, str, None] = None,
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
        self.n_feature_groups = n_feature_groups
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.pca_subsample = pca_subsample
        self.max_features = max_features

    def _make_rotation_matrix(self, X, rng):
        n, p = X.shape
        group_indices = np.array_split(rng.permutation(p), self.n_feature_groups)
        R_blocks = []
        for grp in group_indices:
            if len(grp) == 0: continue
            X_grp = X[:, grp]
            n_sub = max(1, int(self.pca_subsample * n))
            idx = rng.choice(n, size=n_sub, replace=False)
            pca = PCA(n_components=min(len(grp), n_sub))
            pca.fit(X_grp[idx])
            R_blocks.append((grp, pca.components_.T))
        R = np.zeros((p, p))
        col_ptr = 0
        for grp, block in R_blocks:
            R[np.ix_(grp, range(col_ptr, col_ptr + block.shape[1]))] = block
            col_ptr += block.shape[1]
        if col_ptr < p:
            for i in range(col_ptr, p): R[i, i] = 1.0
        return R

    def _make_estimator(self, random_state: int) -> CARTClassifier:
        return CARTClassifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            max_features=self.max_features,
            random_state=random_state,
        )

    def _fit_single(self, seed, X, y, fit_kwargs):
        rng = np.random.default_rng(seed)
        X_s, y_s = self._sample_data(X, y, rng)
        R = self._make_rotation_matrix(X_s, rng)
        tree = self._make_estimator(seed)
        tree.fit(X_s @ R, y_s, **fit_kwargs)
        tree.extra_ = {"rotation_matrix": R}
        return tree

    def fit(self, X, y, **kwargs):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]
        from joblib import Parallel, delayed
        seeds = np.random.default_rng(self.random_state).integers(0, 2**31, size=self.n_estimators)
        self.estimators_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(self._fit_single)(int(s), X, y, kwargs) for s in seeds
        )
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "estimators_")
        X = np.asarray(X, dtype=float)
        all_proba = []
        for tree in self.estimators_:
            X_rot = X @ tree.extra_["rotation_matrix"]
            all_proba.append(np.array([tree._predict_node(x, tree.root_) for x in X_rot]))
        return np.mean(all_proba, axis=0)


class RotationForestRegressor(RegressorForestMixin, BaseForest):
    """Rotation Forest Regressor."""

    def __init__(
        self,
        n_estimators: int = 100,
        n_feature_groups: int = 3,
        criterion: str = "mse",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        pca_subsample: float = 0.75,
        max_features: Union[int, float, str, None] = None,
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
        self.n_feature_groups = n_feature_groups
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.pca_subsample = pca_subsample
        self.max_features = max_features

    def _make_rotation_matrix(self, X, rng):
        n, p = X.shape
        group_indices = np.array_split(rng.permutation(p), self.n_feature_groups)
        R_blocks = []
        for grp in group_indices:
            if len(grp) == 0: continue
            X_grp = X[:, grp]
            n_sub = max(1, int(self.pca_subsample * n))
            idx = rng.choice(n, size=n_sub, replace=False)
            pca = PCA(n_components=min(len(grp), n_sub))
            pca.fit(X_grp[idx])
            R_blocks.append((grp, pca.components_.T))
        R = np.zeros((p, p))
        col_ptr = 0
        for grp, block in R_blocks:
            R[np.ix_(grp, range(col_ptr, col_ptr + block.shape[1]))] = block
            col_ptr += block.shape[1]
        if col_ptr < p:
            for i in range(col_ptr, p): R[i, i] = 1.0
        return R

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

    def _fit_single(self, seed, X, y, fit_kwargs):
        rng = np.random.default_rng(seed)
        X_s, y_s = self._sample_data(X, y, rng)
        R = self._make_rotation_matrix(X_s, rng)
        tree = self._make_estimator(seed)
        tree.fit(X_s @ R, y_s, **fit_kwargs)
        tree.extra_ = {"rotation_matrix": R}
        return tree

    def fit(self, X, y, **kwargs):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]
        from joblib import Parallel, delayed
        seeds = np.random.default_rng(self.random_state).integers(0, 2**31, size=self.n_estimators)
        self.estimators_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(self._fit_single)(int(s), X, y, kwargs) for s in seeds
        )
        return self



# ---------------------------------------------------------------------------
# RandomRotationForest: random orthogonal matrix
# ---------------------------------------------------------------------------

class RandomRotationForest(ClassifierForestMixin, BaseForest):
    """Random Rotation Forest Classifier."""

    def __init__(
        self,
        n_estimators: int = 100,
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
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features

    @staticmethod
    def _random_orthogonal(p: int, rng: np.random.Generator) -> np.ndarray:
        G = rng.standard_normal((p, p))
        Q, _ = np.linalg.qr(G)
        return Q

    def _make_estimator(self, random_state: int) -> CARTClassifier:
        return CARTClassifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            max_features=self.max_features,
            random_state=random_state,
        )

    def _fit_single(self, seed, X, y, fit_kwargs):
        rng = np.random.default_rng(seed)
        X_s, y_s = self._sample_data(X, y, rng)
        Q = self._random_orthogonal(X.shape[1], rng)
        tree = self._make_estimator(seed)
        tree.fit(X_s @ Q, y_s, **fit_kwargs)
        tree.extra_ = {"rotation_matrix": Q}
        return tree

    def fit(self, X, y, **kwargs):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]
        from joblib import Parallel, delayed
        seeds = np.random.default_rng(self.random_state).integers(0, 2**31, size=self.n_estimators)
        self.estimators_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(self._fit_single)(int(s), X, y, kwargs) for s in seeds
        )
        return self


class RandomRotationForestRegressor(RegressorForestMixin, BaseForest):
    """Random Rotation Forest Regressor."""

    def __init__(
        self,
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
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features

    @staticmethod
    def _random_orthogonal(p: int, rng: np.random.Generator) -> np.ndarray:
        G = rng.standard_normal((p, p))
        Q, _ = np.linalg.qr(G)
        return Q

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

    def _fit_single(self, seed, X, y, fit_kwargs):
        rng = np.random.default_rng(seed)
        X_s, y_s = self._sample_data(X, y, rng)
        Q = self._random_orthogonal(X.shape[1], rng)
        tree = self._make_estimator(seed)
        tree.fit(X_s @ Q, y_s, **fit_kwargs)
        tree.extra_ = {"rotation_matrix": Q}
        return tree

    def fit(self, X, y, **kwargs):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]
        from joblib import Parallel, delayed
        seeds = np.random.default_rng(self.random_state).integers(0, 2**31, size=self.n_estimators)
        self.estimators_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(self._fit_single)(int(s), X, y, kwargs) for s in seeds
        )
        return self

