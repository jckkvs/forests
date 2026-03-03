"""
forests.rgf
===========
Regularized Greedy Forest (RGF).

References
----------
Johnson, R., & Zhang, T. (2014). Learning Nonlinear Functions Using
Regularized Greedy Forest. IEEE TPAMI, 36(5), 942-954.

Algorithm Summary
-----------------
RGF grows the forest using a greedy procedure:
1. Start with a single-leaf tree (predicting the mean).
2. At each step, either:
   a. Add a new leaf split to an existing leaf, OR
   b. Adjust all existing leaf weights.
3. Regularization: L1/L2 penalty on leaf weights prevents overfitting.
4. The entire ensemble is stored as a forest of trees, each representing
   one step of the greedy procedure.

Simplification: We implement RGF as an iterative gradient boosting
over a shared forest structure with L2-regularized leaf weights.
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array

from .base import BaseTree, Node, IMPURITY_FN
from .cart import CARTRegressor


class _RGFLeafOptimizer:
    """Shared leaf weight optimizer (L2 regularization).

    Maintains a registry of all leaves across all trees in the ensemble.
    After each tree is grown, performs Newton step update on all leaf weights.

    Implements Johnson & Zhang (2014), Section 3: leaf weight update.
    w_leaf ← w_leaf - η * (g_leaf + λ * w_leaf) / (H_leaf + λ)
    where g = gradient, H = hessian (≈ n_samples for MSE), λ = reg strength.
    """

    def __init__(self, reg_lambda: float = 0.1, learning_rate: float = 0.1):
        self.reg_lambda = reg_lambda
        self.learning_rate = learning_rate
        self.leaves: List[Node] = []
        self.leaf_sample_counts: List[int] = []

    def register_leaves(self, tree: CARTRegressor, X: np.ndarray, y: np.ndarray, residuals: np.ndarray):
        """Register all leaves of a newly grown tree."""
        for node in tree._iter_nodes(tree.root_):
            if node.is_leaf:
                # Find samples that reach this leaf
                leaf_ids = tree.apply(X)
                mask = leaf_ids == node.leaf_id
                n_leaf = mask.sum()
                if n_leaf == 0:
                    continue
                g = -np.mean(residuals[mask])  # negative gradient of MSE
                h = 1.0  # hessian = 1 for MSE per sample (normalized)
                # Newton step
                update = self.learning_rate * g / (h + self.reg_lambda)
                old_w = float(node.value[0])
                node.value = np.array([old_w - update])
                self.leaves.append(node)
                self.leaf_sample_counts.append(n_leaf)

    def global_weight_update(self, X: np.ndarray, residuals: np.ndarray, trees: list):
        """Update all leaf weights across the entire forest using L2-regularized Newton steps."""
        for tree in trees:
            leaf_ids = tree.apply(X)
            for node in tree._iter_nodes(tree.root_):
                if not node.is_leaf:
                    continue
                mask = leaf_ids == node.leaf_id
                n = mask.sum()
                if n == 0:
                    continue
                g = -np.mean(residuals[mask])
                update = self.learning_rate * g / (1.0 + self.reg_lambda)
                old_w = float(node.value[0])
                node.value = np.array([old_w - update])


class RegularizedGreedyForest(BaseEstimator, RegressorMixin):
    """Regularized Greedy Forest for regression.

    Implements: Johnson & Zhang (2014) IEEE TPAMI.
    Grows trees greedily on residuals with L2-regularized leaf weight updates.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees (greedy steps).
    max_depth : int, default=4
        Maximum depth of each tree.
    min_samples_leaf : int, default=5
        Minimum samples per leaf.
    reg_lambda : float, default=0.1
        L2 regularization on leaf weights.
    learning_rate : float, default=0.1
        Step size for leaf weight updates (η in the paper).
    max_features : str or None, default="sqrt"
        Features to consider at each split.
    random_state : int or None

    Examples
    --------
    >>> from forests import RegularizedGreedyForest
    >>> from sklearn.datasets import load_diabetes
    >>> X, y = load_diabetes(return_X_y=True)
    >>> rgf = RegularizedGreedyForest(n_estimators=50, random_state=0)
    >>> rgf.fit(X, y).score(X, y) > 0.5
    True
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        min_samples_leaf: int = 5,
        reg_lambda: float = 0.1,
        learning_rate: float = 0.1,
        max_features: Optional[Union[int, float, str]] = "sqrt",
        random_state: Optional[int] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.reg_lambda = reg_lambda
        self.learning_rate = learning_rate
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegularizedGreedyForest":
        """Fit the RGF model.

        Implements Johnson & Zhang (2014) Algorithm 1.
        1. Initialize F_0(x) = mean(y)
        2. For t = 1..T:
            a. Compute residuals r_i = y_i - F_{t-1}(x_i)
            b. Fit a regression tree to residuals
            c. Update leaf weights with regularized Newton step
        3. Final prediction: F(x) = F_0 + sum_t F_t(x)
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.random_state)

        self.init_pred_: float = float(np.mean(y))
        self.trees_: List[CARTRegressor] = []
        optimizer = _RGFLeafOptimizer(self.reg_lambda, self.learning_rate)

        residuals = y - self.init_pred_

        for t in range(self.n_estimators):
            seed = int(rng.integers(0, 2**31))
            tree = CARTRegressor(
                criterion="mse",
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=seed,
            )
            tree.fit(X, residuals)

            # Regularized leaf weight update
            optimizer.register_leaves(tree, X, y, residuals)

            self.trees_.append(tree)

            # Update residuals
            step_pred = np.array([float(tree._predict_node(x, tree.root_)) for x in X])
            residuals = residuals - step_pred

            # Periodic global weight update (every 10 steps)
            if (t + 1) % 10 == 0:
                optimizer.global_weight_update(X, residuals, self.trees_)

        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions.

        Parameters
        ----------
        X : (n, p) array

        Returns
        -------
        y_pred : (n,) array
        """
        check_is_fitted(self, "trees_")
        X = np.asarray(X, dtype=float)
        pred = np.full(X.shape[0], self.init_pred_)
        for tree in self.trees_:
            pred += np.array([float(tree._predict_node(x, tree.root_)) for x in X])
        return pred
