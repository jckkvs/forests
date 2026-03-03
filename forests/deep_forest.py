"""
forests.deep_forest
===================
Deep Forest (gcForest) – Cascading forest architecture.

References
----------
Zhou, Z.H., & Feng, J. (2017). Deep Forest: Towards an Alternative to Deep Neural Networks.
    IJCAI 2017. arXiv:1702.08835.

Algorithm (Multi-Grained Scanning + Cascade Forest)
-----------------------------------------------------
Step 1 – Multi-Grained Scanning (optional, for structured data):
    Slide windows of different sizes over feature vector, produce feature
    maps, fit small forests on each window, use forest outputs as features.

Step 2 – Cascade Forest:
    Build L levels of forests. At each level:
    a. Each forest produces a class probability vector → concatenated.
    b. Augmented features = [original features, all forest prob vectors].
    c. Next level trains on augmented features.
    d. Early stopping: if validation performance doesn't improve, stop.

We implement the Cascade Forest (Step 2) as the core.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted

from .random_forest import RandomForestClassifier
from .extras import MondrianForest


class DeepForest(ClassifierMixin, BaseEstimator):
    """Deep Forest (gcForest) – Cascade architecture.

    Each cascade level contains multiple forests. The probability outputs
    of all forests are concatenated to the original features for the next level.
    Early stopping on held-out fold performance.

    Parameters
    ----------
    n_estimators_per_forest : int, default=100
        Trees per forest in each level.
    n_forests_per_level : int, default=4
        Number of forests per cascade level (diverse forests).
    max_levels : int, default=10
        Maximum cascade levels.
    n_folds : int, default=3
        Cross-validation folds for class probability generation.
    min_improvement : float, default=1e-4
        Minimum accuracy improvement to continue adding levels.
    random_state : int or None

    Examples
    --------
    >>> from forests import DeepForest
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> X_norm = (X - X.min(0)) / (X.max(0) - X.min(0) + 1e-8)
    >>> df = DeepForest(n_estimators_per_forest=20, n_forests_per_level=2,
    ...                  max_levels=5, random_state=0)
    >>> df.fit(X_norm, y).score(X_norm, y) > 0.9
    True
    """

    def __init__(
        self,
        n_estimators_per_forest: int = 100,
        n_forests_per_level: int = 4,
        max_levels: int = 10,
        n_folds: int = 3,
        min_improvement: float = 1e-4,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_estimators_per_forest = n_estimators_per_forest
        self.n_forests_per_level = n_forests_per_level
        self.max_levels = max_levels
        self.n_folds = n_folds
        self.min_improvement = min_improvement
        self.random_state = random_state

    def _build_forest(self, seed: int) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=self.n_estimators_per_forest,
            random_state=seed,
        )

    def _level_transform(
        self,
        forests: List[RandomForestClassifier],
        X: np.ndarray,
    ) -> np.ndarray:
        """Concatenate probability outputs from all forests in a level."""
        proba_list = [f.predict_proba(X) for f in forests]
        return np.hstack(proba_list)  # (n, n_forests * n_classes)

    def _level_fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        level_seed: int,
    ) -> Tuple[List[RandomForestClassifier], np.ndarray, float]:
        """Fit forests using k-fold cross-fitting, return OOF proba vectors + accuracy."""
        n = X.shape[0]
        rng = np.random.default_rng(level_seed)
        seeds = rng.integers(0, 2**31, size=self.n_forests_per_level)

        # OOF (out-of-fold) probability matrix for each forest
        oof_proba = np.zeros((n, self.n_forests_per_level * self.n_classes_))
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True,
                              random_state=int(rng.integers(0, 2**31)))

        for f_idx in range(self.n_forests_per_level):
            for train_idx, val_idx in skf.split(X, y):
                forest = self._build_forest(int(rng.integers(0, 2**31)))
                forest.fit(X[train_idx], y[train_idx])
                oof_proba[val_idx,
                          f_idx * self.n_classes_: (f_idx + 1) * self.n_classes_] = \
                    forest.predict_proba(X[val_idx])

        # Retrain all forests on full data
        trained_forests = []
        for f_idx, seed in enumerate(seeds):
            forest = self._build_forest(int(seed))
            forest.fit(X, y)
            trained_forests.append(forest)

        # OOF accuracy (mean class prob → argmax)
        oof_avg = oof_proba.reshape(n, self.n_forests_per_level, self.n_classes_).mean(axis=1)
        oof_pred = self.classes_[np.argmax(oof_avg, axis=1)]
        oof_acc = float((oof_pred == y).mean())

        return trained_forests, oof_proba, oof_acc

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DeepForest":
        """Fit cascade forest.

        Implements Zhou & Feng (2017) Algorithm 1 (cascade part).
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]
        rng = np.random.default_rng(self.random_state)

        self.cascade_: List[List[RandomForestClassifier]] = []
        self.level_accuracies_: List[float] = []

        X_aug = X.copy()
        best_acc = 0.0

        for level in range(self.max_levels):
            level_seed = int(rng.integers(0, 2**31))
            level_forests, oof_proba, oof_acc = self._level_fit_transform(X_aug, y, level_seed)
            self.cascade_.append(level_forests)
            self.level_accuracies_.append(oof_acc)

            # Early stopping: stop if improvement too small
            if oof_acc - best_acc < self.min_improvement and level > 0:
                break
            if oof_acc > best_acc:
                best_acc = oof_acc

            # Augment features for next level
            X_aug = np.hstack([X, oof_proba])

        self.best_level_ = np.argmax(self.level_accuracies_)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities using the best cascade level."""
        check_is_fitted(self, "cascade_")
        X = np.asarray(X, dtype=float)

        X_aug = X.copy()
        for level, level_forests in enumerate(self.cascade_):
            level_proba = self._level_transform(level_forests, X_aug)
            if level == len(self.cascade_) - 1:
                # Last level: average proba across forests
                proba_stacked = level_proba.reshape(
                    X.shape[0], self.n_forests_per_level, self.n_classes_
                )
                return proba_stacked.mean(axis=1)
            X_aug = np.hstack([X, level_proba])

        # Fallback
        return np.ones((X.shape[0], self.n_classes_)) / self.n_classes_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class label predictions."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
