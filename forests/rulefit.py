"""
forests.rulefit
===============
RuleFit: Rule-based feature engineering + Lasso.

References
----------
Friedman, J.H., & Popescu, B.E. (2008). Predictive learning via rule ensembles.
    Annals of Applied Statistics, 2(3), 916-954.

Algorithm
---------
1. Grow an ensemble of decision trees.
2. Extract all decision rules from the trees (each leaf path = 1 rule).
3. Create a binary rule matrix R where R[i,j] = 1 if sample i satisfies rule j.
4. Augment features: [X_scaled, R_scaled].
5. Fit Lasso regression on the augmented features.
6. Prediction = Lasso prediction on augmented features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from .base import Node, IMPURITY_FN
from .cart import CARTRegressor


# ---------------------------------------------------------------------------
# Rule extraction
# ---------------------------------------------------------------------------

@dataclass
class Rule:
    """A single decision rule (conjunction of conditions).

    Attributes
    ----------
    conditions : list of (feature_idx, operator, threshold)
        e.g. [(0, "<=", 3.5), (1, ">", 2.1)]
    support : float
        Fraction of training samples satisfying this rule.
    """
    conditions: List[Tuple[int, str, float]] = field(default_factory=list)
    support: float = 1.0

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Return boolean mask of samples satisfying this rule."""
        mask = np.ones(X.shape[0], dtype=bool)
        for feat, op, thr in self.conditions:
            if op == "<=":
                mask &= X[:, feat] <= thr
            else:
                mask &= X[:, feat] > thr
        return mask

    def __str__(self) -> str:
        parts = [f"X[{f}]{op}{t:.4g}" for f, op, t in self.conditions]
        return " & ".join(parts)


def _extract_rules_from_node(node: Node, conditions: List, rules: List[Rule], X: np.ndarray) -> None:
    """Recursively extract rules by traversing tree paths."""
    if node.is_leaf:
        if len(conditions) > 0:
            r = Rule(conditions=list(conditions))
            # Compute support
            mask = np.ones(X.shape[0], dtype=bool)
            for feat, op, thr in conditions:
                if op == "<=":
                    mask &= X[:, feat] <= thr
                else:
                    mask &= X[:, feat] > thr
            r.support = mask.mean()
            if 0.01 < r.support < 0.99:  # skip trivial rules
                rules.append(r)
        return

    if node.feature is None or node.threshold is None:
        return

    # Left branch: feature <= threshold
    conditions.append((node.feature, "<=", node.threshold))
    _extract_rules_from_node(node.left, conditions, rules, X)
    conditions.pop()

    # Right branch: feature > threshold
    conditions.append((node.feature, ">", node.threshold))
    _extract_rules_from_node(node.right, conditions, rules, X)
    conditions.pop()


# ---------------------------------------------------------------------------
# Simple Lasso solver (coordinate descent)
# ---------------------------------------------------------------------------

def _lasso_coordinate_descent(
    A: np.ndarray,
    b: np.ndarray,
    alpha: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> np.ndarray:
    """Lasso via coordinate descent.

    Minimizes: 0.5 * ||Aw - b||^2 + alpha * ||w||_1

    Parameters
    ----------
    A : (n, p) design matrix
    b : (n,) target
    alpha : Lasso penalty
    max_iter : int
    tol : convergence tolerance

    Returns
    -------
    w : (p,) coefficient vector
    """
    n, p = A.shape
    w = np.zeros(p)
    AtA_diag = (A ** 2).sum(axis=0) / n  # diagonal of A^T A / n

    for iteration in range(max_iter):
        w_old = w.copy()
        for j in range(p):
            # Partial residual
            r = b - A @ w + A[:, j] * w[j]
            # OLS update for feature j
            rho_j = float(A[:, j] @ r) / n
            # Soft-threshold
            if AtA_diag[j] < 1e-12:
                w[j] = 0.0
            else:
                w[j] = np.sign(rho_j) * max(abs(rho_j) - alpha, 0.0) / AtA_diag[j]

        if np.max(np.abs(w - w_old)) < tol:
            break

    return w


# ---------------------------------------------------------------------------
# RuleFit
# ---------------------------------------------------------------------------

class RuleFit(RegressorMixin, BaseEstimator):
    """RuleFit: Rule extraction + Sparse linear model.

    Implements: Friedman & Popescu (2008) Annals of Applied Statistics.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees to generate rules from.
    max_depth : int, default=3
        Maximum tree depth (controls rule complexity).
    min_samples_leaf : int, default=5
    tree_learning_rate : float, default=0.1
        Shrinkage for gradient boosted trees (for diverse rules).
    alpha : float, default=0.01
        Lasso regularization strength.
    lasso_max_iter : int, default=2000
    feature_names : list of str or None
        Optional feature names for rule display.
    random_state : int or None

    Attributes
    ----------
    rules_ : list of Rule
        All extracted rules.
    coef_ : np.ndarray
        Lasso coefficients for [linear_features..., rules...].

    Examples
    --------
    >>> from forests import RuleFit
    >>> from sklearn.datasets import load_diabetes
    >>> X, y = load_diabetes(return_X_y=True)
    >>> rf = RuleFit(n_estimators=20, alpha=0.01, random_state=0)
    >>> rf.fit(X, y).score(X, y) > 0.4
    True
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        min_samples_leaf: int = 5,
        tree_learning_rate: float = 0.1,
        alpha: float = 0.01,
        lasso_max_iter: int = 2000,
        feature_names: Optional[List[str]] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree_learning_rate = tree_learning_rate
        self.alpha = alpha
        self.lasso_max_iter = lasso_max_iter
        self.feature_names = feature_names
        self.random_state = random_state

    def _grow_trees(self, X: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> List[CARTRegressor]:
        """Grow stage-wise trees (gradient boosting on residuals) for diverse rule generation."""
        trees = []
        residuals = y - float(np.mean(y))
        for t in range(self.n_estimators):
            seed = int(rng.integers(0, 2**31))
            tree = CARTRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=seed,
            )
            tree.fit(X, residuals)
            pred = np.array([float(tree._predict_node(x, tree.root_)[0]) for x in X])
            residuals -= self.tree_learning_rate * pred
            trees.append(tree)
        return trees

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RuleFit":
        """Fit RuleFit.

        Steps:
        1. Grow trees on residuals.
        2. Extract all decision rules.
        3. Build augmented feature matrix [X_scaled, R_scaled].
        4. Fit Lasso on augmented features.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.random_state)
        n, p = X.shape
        self.n_features_in_ = p

        # Step 1: grow trees
        trees = self._grow_trees(X, y, rng)

        # Step 2: extract rules
        self.rules_: List[Rule] = []
        for tree in trees:
            _extract_rules_from_node(tree.root_, [], self.rules_, X)

        # Step 3: build rule matrix
        if len(self.rules_) > 0:
            R = np.column_stack([r.apply(X).astype(float) for r in self.rules_])
        else:
            R = np.empty((n, 0))

        # Scale features and rules
        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0) + 1e-8
        X_scaled = (X - self._x_mean) / self._x_std

        if R.shape[1] > 0:
            self._r_std = R.std(axis=0) + 1e-8
            R_scaled = R / self._r_std
        else:
            self._r_std = np.array([])
            R_scaled = R

        # Augmented matrix: [1_bias, X_scaled, R_scaled]
        one = np.ones((n, 1))
        A = np.column_stack([one, X_scaled, R_scaled])

        # Step 4: Lasso
        y_centered = y - y.mean()
        self.y_mean_ = float(y.mean())
        self.coef_ = _lasso_coordinate_descent(A, y_centered, self.alpha, self.lasso_max_iter)
        self.n_rules_ = len(self.rules_)
        return self

    def _augment(self, X: np.ndarray) -> np.ndarray:
        """Build augmented feature matrix for prediction."""
        n = X.shape[0]
        X_scaled = (X - self._x_mean) / self._x_std
        if len(self.rules_) > 0:
            R = np.column_stack([r.apply(X).astype(float) for r in self.rules_])
            R_scaled = R / self._r_std
        else:
            R_scaled = np.empty((n, 0))
        one = np.ones((n, 1))
        return np.column_stack([one, X_scaled, R_scaled])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions."""
        check_is_fitted(self, "coef_")
        X = np.asarray(X, dtype=float)
        A = self._augment(X)
        return self.y_mean_ + A @ self.coef_

    def get_rules(self, top_n: Optional[int] = None) -> List[Tuple[Rule, float]]:
        """Return rules sorted by absolute coefficient value.

        Parameters
        ----------
        top_n : int or None
            Number of top rules to return. None = all.

        Returns
        -------
        list of (Rule, coef)
        """
        check_is_fitted(self, "coef_")
        # coef_[0] = bias, [1:n_features+1] = linear, [n_features+1:] = rules
        rule_coefs = self.coef_[1 + self.n_features_in_:]
        pairs = list(zip(self.rules_, rule_coefs))
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        if top_n is not None:
            return pairs[:top_n]
        return pairs
