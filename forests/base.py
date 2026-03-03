"""
forests.base
============
Common base classes for all tree and forest models.

References
----------
Breiman, L. et al. (1984). Classification and Regression Trees. Wadsworth.
"""

from __future__ import annotations

import abc
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

@dataclass
class Node:
    """A single node in a decision tree.

    Parameters
    ----------
    feature : int, optional
        Feature index used for splitting. None for leaf nodes.
    threshold : float, optional
        Split threshold value. None for leaf nodes.
    value : np.ndarray, optional
        Aggregated value at this node (class counts or mean).
    impurity : float
        Impurity measure at this node.
    n_samples : int
        Number of training samples that reached this node.
    left : Node, optional
        Left child (condition: X[:, feature] <= threshold).
    right : Node, optional
        Right child (condition: X[:, feature] > threshold).
    depth : int
        Depth of this node in the tree (root = 0).
    leaf_id : int
        Unique leaf identifier (set during tree construction, -1 for internal).
    weight : float
        Leaf weight used in regularized models.
    extra : dict
        Additional metadata stored by specific algorithms.
    """

    feature: Optional[int] = None
    threshold: Optional[float] = None
    value: Optional[np.ndarray] = None
    impurity: float = 0.0
    n_samples: int = 0
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    depth: int = 0
    leaf_id: int = -1
    weight: float = 1.0
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_leaf(self) -> bool:
        """Return True if this node is a leaf."""
        return self.left is None and self.right is None


# ---------------------------------------------------------------------------
# Impurity helpers
# ---------------------------------------------------------------------------

def gini_impurity(y: np.ndarray, n_classes: int) -> float:
    """Compute Gini impurity for label array y.

    Implements: gini = 1 - sum(p_k^2)

    Parameters
    ----------
    y : np.ndarray of shape (n_samples,)
    n_classes : int

    Returns
    -------
    float
    """
    if len(y) == 0:
        return 0.0
    counts = np.bincount(y.astype(int), minlength=n_classes)
    probs = counts / counts.sum()
    return 1.0 - float(np.sum(probs ** 2))


def entropy_impurity(y: np.ndarray, n_classes: int) -> float:
    """Compute entropy impurity for label array y.

    Implements: H = -sum(p_k * log2(p_k))
    """
    if len(y) == 0:
        return 0.0
    counts = np.bincount(y.astype(int), minlength=n_classes)
    probs = counts[counts > 0] / counts.sum()
    return float(-np.sum(probs * np.log2(probs)))


def mse_impurity(y: np.ndarray) -> float:
    """Compute mean squared error impurity.

    Implements: MSE = mean((y - mean(y))^2)
    """
    if len(y) == 0:
        return 0.0
    return float(np.mean((y - np.mean(y)) ** 2))


def mae_impurity(y: np.ndarray) -> float:
    """Compute mean absolute error impurity."""
    if len(y) == 0:
        return 0.0
    return float(np.mean(np.abs(y - np.median(y))))


IMPURITY_FN = {
    "gini": gini_impurity,
    "entropy": entropy_impurity,
    "mse": mse_impurity,
    "mae": mae_impurity,
    "friedman_mse": mse_impurity,  # alias
}


# ---------------------------------------------------------------------------
# BaseTree
# ---------------------------------------------------------------------------

class BaseTree(BaseEstimator, abc.ABC):
    """Abstract base class for single decision trees.

    Subclasses must implement :meth:`_find_best_split`.

    Parameters
    ----------
    max_depth : int or None
        Maximum depth of the tree. None means unlimited.
    min_samples_split : int
        Minimum samples to consider a node for splitting.
    min_samples_leaf : int
        Minimum samples each child node must have.
    min_impurity_decrease : float
        Minimum impurity decrease to accept a split.
    max_features : int, float, str, or None
        Number of features to consider at each split.
        int → exact count, float → fraction, "sqrt" / "log2" → functions.
        None → all features.
    random_state : int or None
        Random seed.
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        max_features: Optional[Union[int, float, str]] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.random_state = random_state
        self.root_: Optional[Node] = None
        self._leaf_counter: int = 0

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _find_best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_indices: np.ndarray,
        rng: np.random.Generator,
        **kwargs: Any,
    ) -> Tuple[Optional[int], Optional[float], float]:
        """Find the best split for a node.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)
        feature_indices : np.ndarray
            Subset of feature indices to search.
        rng : np.random.Generator

        Returns
        -------
        best_feature : int or None
        best_threshold : float or None
        best_gain : float
        """

    @abc.abstractmethod
    def _node_value(self, y: np.ndarray) -> np.ndarray:
        """Compute the prediction value stored at a node/leaf."""

    @abc.abstractmethod
    def _impurity(self, y: np.ndarray) -> float:
        """Compute the impurity of a label array."""

    # ------------------------------------------------------------------
    # Feature selection helper
    # ------------------------------------------------------------------

    def _select_features(
        self,
        n_features: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Select feature indices for a split candidate search."""
        mf = self.max_features
        if mf is None:
            k = n_features
        elif isinstance(mf, int):
            k = min(mf, n_features)
        elif isinstance(mf, float):
            k = max(1, int(mf * n_features))
        elif mf == "sqrt":
            k = max(1, int(np.sqrt(n_features)))
        elif mf == "log2":
            k = max(1, int(np.log2(n_features)))
        else:
            raise ValueError(f"Unknown max_features: {mf!r}")
        return rng.choice(n_features, size=k, replace=False)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depth: int,
        rng: np.random.Generator,
        **kwargs: Any,
    ) -> Node:
        """Recursively build the tree."""
        n_samples, n_features = X.shape
        impurity = self._impurity(y)
        node = Node(
            value=self._node_value(y),
            impurity=impurity,
            n_samples=n_samples,
            depth=depth,
        )

        # Stopping criteria
        too_deep = self.max_depth is not None and depth >= self.max_depth
        too_few = n_samples < self.min_samples_split
        pure = impurity == 0.0

        if too_deep or too_few or pure:
            node.leaf_id = self._leaf_counter
            self._leaf_counter += 1
            return node

        # Feature sampling
        feature_indices = self._select_features(n_features, rng)

        # Find best split
        best_feature, best_threshold, best_gain = self._find_best_split(
            X, y, feature_indices, rng, **kwargs
        )

        if best_feature is None or best_gain < self.min_impurity_decrease:
            node.leaf_id = self._leaf_counter
            self._leaf_counter += 1
            return node

        # Split data
        mask = X[:, best_feature] <= best_threshold
        left_X, left_y = X[mask], y[mask]
        right_X, right_y = X[~mask], y[~mask]

        if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
            node.leaf_id = self._leaf_counter
            self._leaf_counter += 1
            return node

        node.feature = best_feature
        node.threshold = best_threshold
        node.left = self._build(left_X, left_y, depth + 1, rng, **kwargs)
        node.right = self._build(right_X, right_y, depth + 1, rng, **kwargs)
        return node

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs: Any,
    ) -> "BaseTree":
        """Fit the decision tree to training data."""
        rng = np.random.default_rng(self.random_state)
        self._leaf_counter = 0
        self.n_features_in_: int = X.shape[1]
        self.root_ = self._build(X, y, depth=0, rng=rng, **kwargs)
        self.n_leaves_: int = self._leaf_counter
        return self

    # ------------------------------------------------------------------
    # Predict helpers
    # ------------------------------------------------------------------

    def _predict_node(self, x: np.ndarray, node: Node) -> np.ndarray:
        """Traverse tree for a single sample, return leaf value."""
        if node.is_leaf:
            return node.value  # type: ignore[return-value]
        assert node.feature is not None and node.threshold is not None
        if x[node.feature] <= node.threshold:
            return self._predict_node(x, node.left)  # type: ignore[arg-type]
        else:
            return self._predict_node(x, node.right)  # type: ignore[arg-type]

    def _apply_node(self, x: np.ndarray, node: Node) -> int:
        """Return leaf_id for a single sample."""
        if node.is_leaf:
            return node.leaf_id
        assert node.feature is not None and node.threshold is not None
        if x[node.feature] <= node.threshold:
            return self._apply_node(x, node.left)  # type: ignore[arg-type]
        else:
            return self._apply_node(x, node.right)  # type: ignore[arg-type]

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Return leaf IDs for each sample.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        leaf_ids : np.ndarray of shape (n_samples,)
        """
        if not hasattr(self, 'root_') or self.root_ is None:
            raise RuntimeError("Model is not fitted yet. Call 'fit' first.")
        return np.array([self._apply_node(x, self.root_) for x in X])

    def get_depth(self) -> int:
        """Return the maximum depth of the fitted tree."""
        if not hasattr(self, 'root_') or self.root_ is None:
            raise RuntimeError("Model is not fitted yet. Call 'fit' first.")

        def _depth(node: Optional[Node]) -> int:
            if node is None or node.is_leaf:
                return 0
            return 1 + max(_depth(node.left), _depth(node.right))

        return _depth(self.root_)

    def get_n_leaves(self) -> int:
        """Return the number of leaves."""
        if not hasattr(self, 'n_leaves_'):
            raise RuntimeError("Model is not fitted yet. Call 'fit' first.")
        return self.n_leaves_

    def _iter_nodes(self, node: Optional[Node]):
        """Yield all nodes in depth-first order."""
        if node is None:
            return
        yield node
        yield from self._iter_nodes(node.left)
        yield from self._iter_nodes(node.right)

    def get_leaves(self) -> List[Node]:
        """Return all leaf nodes."""
        if not hasattr(self, 'root_') or self.root_ is None:
            raise RuntimeError("Model is not fitted yet. Call 'fit' first.")
        return [n for n in self._iter_nodes(self.root_) if n.is_leaf]


# ---------------------------------------------------------------------------
# BaseForest
# ---------------------------------------------------------------------------

class BaseForest(BaseEstimator, abc.ABC):
    """Abstract base class for ensemble forest models.

    Provides parallel fitting via joblib, sklearn-compatible API,
    and leaf-ID extraction for the RFSimilarity / RFKernel modules.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the forest.
    bootstrap : bool
        Whether to use bootstrap sampling.
    max_samples : int, float, or None
        Number/fraction of samples to use per tree (when bootstrap=True).
    n_jobs : int
        Number of parallel jobs (-1 = all CPUs).
    random_state : int or None
        Master random seed.
    verbose : int
        Verbosity level.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        bootstrap: bool = True,
        max_samples: Optional[Union[int, float]] = None,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ) -> None:
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    @abc.abstractmethod
    def _make_estimator(self, random_state: int) -> BaseTree:
        """Instantiate a single tree estimator."""

    def _sample_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (bootstrapped or full) sample of (X, y)."""
        n = X.shape[0]
        if not self.bootstrap:
            return X, y
        if self.max_samples is None:
            k = n
        elif isinstance(self.max_samples, float):
            k = max(1, int(self.max_samples * n))
        else:
            k = int(self.max_samples)
        idx = rng.integers(0, n, size=k)
        return X[idx], y[idx]

    def _fit_single(
        self,
        seed: int,
        X: np.ndarray,
        y: np.ndarray,
        fit_kwargs: Dict[str, Any],
    ) -> BaseTree:
        """Fit one tree (called in parallel)."""
        rng = np.random.default_rng(seed)
        X_s, y_s = self._sample_data(X, y, rng)
        tree = self._make_estimator(seed)
        tree.fit(X_s, y_s, **fit_kwargs)
        return tree

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **fit_kwargs: Any,
    ) -> "BaseForest":
        """Fit all trees in the forest."""
        from joblib import Parallel, delayed

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.n_features_in_: int = X.shape[1]
        self._validate_fit_params(X, y)

        master_rng = np.random.default_rng(self.random_state)
        seeds = master_rng.integers(0, 2**31, size=self.n_estimators)

        self.estimators_: List[BaseTree] = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(self._fit_single)(int(s), X, y, fit_kwargs)
            for s in seeds
        )
        return self

    def _validate_fit_params(self, X: np.ndarray, y: np.ndarray) -> None:
        """Override to add extra validation logic."""

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Return leaf IDs for each tree.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        leaf_ids : np.ndarray of shape (n_samples, n_estimators)
        """
        check_is_fitted(self, "estimators_")
        X = np.asarray(X, dtype=float)
        return np.column_stack([tree.apply(X) for tree in self.estimators_])

    def _aggregate_predict(self, X: np.ndarray) -> np.ndarray:
        """Stack raw predictions from all estimators."""
        check_is_fitted(self, "estimators_")
        X = np.asarray(X, dtype=float)
        return np.array([tree._predict_node(x, tree.root_) for tree in self.estimators_ for x in X]).reshape(
            len(self.estimators_), X.shape[0], -1
        )


# ---------------------------------------------------------------------------
# Mixin helpers
# ---------------------------------------------------------------------------

class ClassifierForestMixin(ClassifierMixin):
    """Mixin for forest classifiers."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
        """
        check_is_fitted(self, "estimators_")
        X = np.asarray(X, dtype=float)
        # Collect per-tree probabilities and average
        all_proba = []
        for tree in self.estimators_:  # type: ignore[attr-defined]
            proba = np.array([tree._predict_node(x, tree.root_) for x in X])
            all_proba.append(proba)
        return np.mean(all_proba, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


class RegressorForestMixin(RegressorMixin):
    """Mixin for forest regressors."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return averaged predictions from all trees.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
        """
        check_is_fitted(self, "estimators_")
        X = np.asarray(X, dtype=float)
        preds = np.array([
            [tree._predict_node(x, tree.root_).item() if hasattr(tree._predict_node(x, tree.root_), 'item') else float(tree._predict_node(x, tree.root_)) for x in X]
            for tree in self.estimators_  # type: ignore[attr-defined]
        ])
        return np.mean(preds, axis=0)
