"""
forests.bernoulli_rf
====================
Bernoulli Random Forest.

Each feature is independently included in each tree with probability p
(Bernoulli sampling), rather than selecting a fixed k features as in RF.

References
----------
Inspired by Bernoulli feature sampling in:
Denil, M., Matheson, D., & de Freitas, N. (2014).
    Narrowing the gap: Random forests in theory and in practice. ICML.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
from sklearn.utils.validation import check_is_fitted

from .base import BaseForest, ClassifierForestMixin, RegressorForestMixin
from .cart import CARTClassifier, CARTRegressor, _best_split_axis
from .base import IMPURITY_FN


class _BernoulliClassifierTree(CARTClassifier):
    """Classifier tree with Bernoulli feature sampling."""

    def __init__(self, feature_prob: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.feature_prob = feature_prob

    def _select_features(self, n_features: int, rng: np.random.Generator) -> np.ndarray:
        """Sample features with Bernoulli probability."""
        mask = rng.random(n_features) < self.feature_prob
        selected = np.where(mask)[0]
        if len(selected) == 0:
            # Guarantee at least one feature
            selected = rng.integers(0, n_features, size=1)
        return selected


class _BernoulliRegressorTree(CARTRegressor):
    """Regressor tree with Bernoulli feature sampling."""

    def __init__(self, feature_prob: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.feature_prob = feature_prob

    def _select_features(self, n_features: int, rng: np.random.Generator) -> np.ndarray:
        mask = rng.random(n_features) < self.feature_prob
        selected = np.where(mask)[0]
        if len(selected) == 0:
            selected = rng.integers(0, n_features, size=1)
        return selected


class BernoulliRandomForest(ClassifierForestMixin, BaseForest):
    """Random Forest with Bernoulli feature sampling.

    At each split, each feature is independently included with probability
    `feature_prob` (Bernoulli distribution), rather than fixing max_features.
    This provides a different exploration-exploitation tradeoff than RF.

    Parameters
    ----------
    n_estimators : int, default=100
    feature_prob : float, default=0.5
        Probability of including each feature at each split.
    criterion : {"gini", "entropy"}, default="gini"
    max_depth : int or None
    min_samples_split : int, default=2
    min_samples_leaf : int, default=1
    bootstrap : bool, default=True
    n_jobs : int, default=1
    random_state : int or None

    Examples
    --------
    >>> from forests import BernoulliRandomForest
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = BernoulliRandomForest(n_estimators=10, feature_prob=0.5, random_state=0)
    >>> clf.fit(X, y).score(X, y) > 0.8
    True
    """

    def __init__(
        self,
        n_estimators: int = 100,
        feature_prob: float = 0.5,
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
        self.feature_prob = feature_prob
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease

    def _make_estimator(self, random_state: int) -> _BernoulliClassifierTree:
        return _BernoulliClassifierTree(
            feature_prob=self.feature_prob,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "BernoulliRandomForest":
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
        return self.classes_[np.argmax(proba, axis=1)]
