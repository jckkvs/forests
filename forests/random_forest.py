"""
forests.random_forest
=====================
Random Forest and Extremely Randomized Trees (ExtraTrees).

References
----------
Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
Geurts, P., Ernst, D., & Wehenkel, L. (2006). Extremely randomized trees.
    Machine Learning, 63(1), 3-42.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from .base import (
    BaseForest,
    BaseTree,
    ClassifierForestMixin,
    RegressorForestMixin,
    IMPURITY_FN,
)
from .cart import CARTClassifier, CARTRegressor, _best_split_axis


# ---------------------------------------------------------------------------
# Extra Trees: random threshold selection (Geurts et al., 2006)
# ---------------------------------------------------------------------------

class _ExtraTreeClassifier(CARTClassifier):
    """Single extra-randomized tree for classification.

    Instead of finding the optimal split threshold, it draws K random
    thresholds and picks the best among those.
    Implements: Geurts et al. (2006) Eq. (1) random cut generation.
    """

    def _find_best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_indices: np.ndarray,
        rng: np.random.Generator,
        **kwargs,
    ) -> Tuple[Optional[int], Optional[float], float]:
        n = len(y)
        fn = IMPURITY_FN[self.criterion]
        base_imp = fn(y, self.n_classes_)
        best_gain = 0.0
        best_feature: Optional[int] = None
        best_threshold: Optional[float] = None

        for f in feature_indices:
            vals = X[:, f]
            v_min, v_max = vals.min(), vals.max()
            if v_min == v_max:
                continue
            # Draw one random threshold in [v_min, v_max)
            thr = float(rng.uniform(v_min, v_max))
            left_mask = vals <= thr
            n_left = left_mask.sum()
            n_right = n - n_left
            if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                continue
            imp_l = fn(y[left_mask], self.n_classes_)
            imp_r = fn(y[~left_mask], self.n_classes_)
            gain = base_imp - (n_left / n) * imp_l - (n_right / n) * imp_r
            if gain > best_gain:
                best_gain = gain
                best_feature = int(f)
                best_threshold = thr

        return best_feature, best_threshold, best_gain


class _ExtraTreeRegressor(CARTRegressor):
    """Single extra-randomized tree for regression.

    Implements: Geurts et al. (2006) random cut for continuous targets.
    """

    def _find_best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_indices: np.ndarray,
        rng: np.random.Generator,
        **kwargs,
    ) -> Tuple[Optional[int], Optional[float], float]:
        n = len(y)
        fn = IMPURITY_FN[self.criterion]
        base_imp = fn(y)
        best_gain = 0.0
        best_feature: Optional[int] = None
        best_threshold: Optional[float] = None

        for f in feature_indices:
            vals = X[:, f]
            v_min, v_max = vals.min(), vals.max()
            if v_min == v_max:
                continue
            thr = float(rng.uniform(v_min, v_max))
            left_mask = vals <= thr
            n_left = left_mask.sum()
            n_right = n - n_left
            if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                continue
            imp_l = fn(y[left_mask])
            imp_r = fn(y[~left_mask])
            gain = base_imp - (n_left / n) * imp_l - (n_right / n) * imp_r
            if gain > best_gain:
                best_gain = gain
                best_feature = int(f)
                best_threshold = thr

        return best_feature, best_threshold, best_gain


# ---------------------------------------------------------------------------
# Random Forest Classifier
# ---------------------------------------------------------------------------

class RandomForestClassifier(ClassifierForestMixin, BaseForest):
    """Random Forest Classifier.

    Implements: Breiman (2001). Bootstrap sampling + sqrt(p) features per split.

    Parameters
    ----------
    n_estimators : int, default=100
    criterion : {"gini", "entropy"}, default="gini"
    max_depth : int or None
    min_samples_split : int, default=2
    min_samples_leaf : int, default=1
    min_impurity_decrease : float, default=0.0
    max_features : int, float, str, or None, default="sqrt"
    bootstrap : bool, default=True
    max_samples : int, float, or None
    class_weight : None
    n_jobs : int, default=1
    random_state : int or None
    verbose : int, default=0

    Examples
    --------
    >>> from forests import RandomForestClassifier
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = RandomForestClassifier(n_estimators=10, random_state=0)
    >>> clf.fit(X, y).score(X, y) > 0.9
    True
    """

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
        class_weight=None,
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
        self.class_weight = class_weight

    def _make_estimator(self, random_state: int) -> CARTClassifier:
        return CARTClassifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            max_features=self.max_features,
            class_weight=self.class_weight,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "RandomForestClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        super().fit(X, y, **kwargs)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "estimators_")
        X = np.asarray(X, dtype=float)
        all_proba = []
        for tree in self.estimators_:
            proba = np.array([tree._predict_node(x, tree.root_) for x in X])
            all_proba.append(proba)
        return np.mean(all_proba, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]


# ---------------------------------------------------------------------------
# Random Forest Regressor
# ---------------------------------------------------------------------------

class RandomForestRegressor(RegressorForestMixin, BaseForest):
    """Random Forest Regressor.

    Implements: Breiman (2001). Bootstrap sampling + sqrt(p) features.

    Parameters
    ----------
    n_estimators : int, default=100
    criterion : {"mse", "mae"}, default="mse"
    max_depth : int or None
    min_samples_split : int, default=2
    min_samples_leaf : int, default=1
    min_impurity_decrease : float, default=0.0
    max_features : int, float, str, or None, default="sqrt"
    bootstrap : bool, default=True
    max_samples : int, float, or None
    n_jobs : int, default=1
    random_state : int or None
    verbose : int, default=0

    Examples
    --------
    >>> from forests import RandomForestRegressor
    >>> from sklearn.datasets import load_diabetes
    >>> X, y = load_diabetes(return_X_y=True)
    >>> reg = RandomForestRegressor(n_estimators=10, random_state=0)
    >>> reg.fit(X, y).score(X, y) > 0.7
    True
    """

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


# ---------------------------------------------------------------------------
# ExtraTrees Classifier / Regressor
# ---------------------------------------------------------------------------

class ExtraTreesClassifier(ClassifierForestMixin, BaseForest):
    """Extremely Randomized Trees Classifier.

    Implements: Geurts et al. (2006). Random thresholds at each split.

    Parameters
    ----------
    n_estimators : int, default=100
    criterion : {"gini", "entropy"}, default="gini"
    max_depth : int or None
    min_samples_split : int, default=2
    min_samples_leaf : int, default=1
    min_impurity_decrease : float, default=0.0
    max_features : int, float, str, or None, default="sqrt"
    bootstrap : bool, default=False
    n_jobs : int, default=1
    random_state : int or None

    Examples
    --------
    >>> from forests import ExtraTreesClassifier
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = ExtraTreesClassifier(n_estimators=10, random_state=0)
    >>> clf.fit(X, y).score(X, y) > 0.9
    True
    """

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        max_features: Union[int, float, str, None] = "sqrt",
        bootstrap: bool = False,
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

    def _make_estimator(self, random_state: int) -> _ExtraTreeClassifier:
        return _ExtraTreeClassifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            max_features=self.max_features,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "ExtraTreesClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        super().fit(X, y, **kwargs)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "estimators_")
        X = np.asarray(X, dtype=float)
        all_proba = []
        for tree in self.estimators_:
            proba = np.array([tree._predict_node(x, tree.root_) for x in X])
            all_proba.append(proba)
        return np.mean(all_proba, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]


class ExtraTreesRegressor(RegressorForestMixin, BaseForest):
    """Extremely Randomized Trees Regressor.

    Implements: Geurts et al. (2006).

    Parameters
    ----------
    n_estimators : int, default=100
    criterion : {"mse", "mae"}, default="mse"
    max_depth : int or None
    min_samples_split : int, default=2
    min_samples_leaf : int, default=1
    min_impurity_decrease : float, default=0.0
    max_features : int, float, str, or None, default="sqrt"
    bootstrap : bool, default=False
    n_jobs : int, default=1
    random_state : int or None

    Examples
    --------
    >>> from forests import ExtraTreesRegressor
    >>> from sklearn.datasets import load_diabetes
    >>> X, y = load_diabetes(return_X_y=True)
    >>> reg = ExtraTreesRegressor(n_estimators=10, random_state=0)
    >>> reg.fit(X, y).score(X, y) > 0.7
    True
    """

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = "mse",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        max_features: Union[int, float, str, None] = "sqrt",
        bootstrap: bool = False,
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

    def _make_estimator(self, random_state: int) -> _ExtraTreeRegressor:
        return _ExtraTreeRegressor(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            max_features=self.max_features,
            random_state=random_state,
        )
