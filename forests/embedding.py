"""
forests.embedding
=================
TotallyRandomTreesEmbedding – Unsupervised feature learning via random trees.

References
----------
Geurts, P., Ernst, D., & Wehenkel, L. (2006).
    Extremely randomized trees. Machine Learning, 63(1), 3-42.

Moosmann, F., Triggs, B., & Jurie, F. (2007).
    Fast discriminative visual codebooks using randomized clustering forests.
    NIPS 2007.

Algorithm
---------
1. Grow T completely random trees (no label needed, random splits on [min, max]).
2. For a sample x, traverse each tree to find its leaf → binary indicator vector.
3. Concatenate all leaf indicators → sparse binary feature vector of length L
   (where L = total number of leaves across all trees).
4. This embedding captures local neighborhood structure for any downstream task.

FuzzyDecisionTree
-----------------
References
----------
Umano, M., et al. (1994). Fuzzy decision trees by fuzzy ID3 algorithm and
    its application to diagnosis systems. In IEEE Conf. on Fuzzy Systems.

Algorithm:
- Each node computes a soft membership: μ(x) = σ((x[f] - threshold) / β)
  where β is the fuzzy bandwidth (larger β → softer boundary).
- The output at x is a weighted sum of all leaf values:
  F(x) = Σ_leaf π_leaf(x) * v_leaf
  where π_leaf = product of soft gate probabilities along the path.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from .base import Node


# ---------------------------------------------------------------------------
# TotallyRandomTreesEmbedding
# ---------------------------------------------------------------------------

class _TotallyRandomTree:
    """A single completely random tree for embedding (unsupervised)."""

    def __init__(self, max_depth: int = 5, random_state: Optional[int] = None):
        self.max_depth = max_depth
        self.random_state = random_state
        self.root_: Optional[Node] = None
        self._leaf_counter = 0

    def _build(self, X: np.ndarray, depth: int, rng: np.random.Generator) -> Node:
        n, p = X.shape
        node = Node(n_samples=n, depth=depth)
        if depth >= self.max_depth or n <= 1:
            node.leaf_id = self._leaf_counter
            self._leaf_counter += 1
            return node

        # Totally random: pick random feature and random threshold
        f = int(rng.integers(0, p))
        v_min, v_max = X[:, f].min(), X[:, f].max()
        if v_min == v_max:
            node.leaf_id = self._leaf_counter
            self._leaf_counter += 1
            return node

        thr = float(rng.uniform(v_min, v_max))
        mask = X[:, f] <= thr
        node.feature = f
        node.threshold = thr
        node.left = self._build(X[mask], depth + 1, rng)
        node.right = self._build(X[~mask], depth + 1, rng)
        return node

    def fit(self, X: np.ndarray) -> "_TotallyRandomTree":
        rng = np.random.default_rng(self.random_state)
        self._leaf_counter = 0
        self.root_ = self._build(X, 0, rng)
        self.n_leaves_ = self._leaf_counter
        return self

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Return leaf IDs for each sample."""
        def _traverse(x, node):
            if node.is_leaf or node.feature is None:
                return node.leaf_id
            if x[node.feature] <= node.threshold:
                return _traverse(x, node.left)
            return _traverse(x, node.right)
        return np.array([_traverse(x, self.root_) for x in X])


class TotallyRandomTreesEmbedding(TransformerMixin, BaseEstimator):
    """Unsupervised random trees embedding.

    Maps samples to a sparse binary representation based on which leaves
    they fall into across a forest of totally random trees.

    Useful as a nonparametric basis for downstream classification or regression.

    Parameters
    ----------
    n_estimators : int, default=100
    max_depth : int, default=5
        Controls embedding dimensionality (roughly 2^max_depth per tree).
    sparse_output : bool, default=True
        Return scipy sparse matrix.
    random_state : int or None

    Examples
    --------
    >>> from forests import TotallyRandomTreesEmbedding
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((100, 4))
    >>> emb = TotallyRandomTreesEmbedding(n_estimators=10, max_depth=3, random_state=0)
    >>> X_emb = emb.fit_transform(X)
    >>> X_emb.shape[0] == 100
    True
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        sparse_output: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.sparse_output = sparse_output
        self.random_state = random_state

    def fit(self, X: np.ndarray, y=None) -> "TotallyRandomTreesEmbedding":
        """Build totally random trees."""
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        rng = np.random.default_rng(self.random_state)
        seeds = rng.integers(0, 2**31, size=self.n_estimators)

        self.trees_: List[_TotallyRandomTree] = []
        self.leaf_offsets_: List[int] = []
        offset = 0
        for seed in seeds:
            tree = _TotallyRandomTree(max_depth=self.max_depth, random_state=int(seed))
            tree.fit(X)
            self.trees_.append(tree)
            self.leaf_offsets_.append(offset)
            offset += tree.n_leaves_

        self.n_outputs_ = offset
        return self

    def transform(self, X: np.ndarray):
        """Return binary embedding matrix of shape (n_samples, n_leaves_total).

        Parameters
        ----------
        X : (n, p) array

        Returns
        -------
        X_emb : (n, n_leaves_total) sparse or dense binary array
        """
        check_is_fitted(self, "trees_")
        X = np.asarray(X, dtype=float)
        n = X.shape[0]

        rows, cols = [], []
        for tree, offset in zip(self.trees_, self.leaf_offsets_):
            leaf_ids = tree.apply(X)
            for i, lid in enumerate(leaf_ids):
                rows.append(i)
                cols.append(offset + lid)

        data = np.ones(len(rows), dtype=np.float32)
        mat = csr_matrix((data, (rows, cols)), shape=(n, self.n_outputs_))

        if self.sparse_output:
            return mat
        return mat.toarray()

    def fit_transform(self, X: np.ndarray, y=None):
        return self.fit(X).transform(X)


# ---------------------------------------------------------------------------
# FuzzyDecisionTree
# ---------------------------------------------------------------------------

def _fuzzy_sigmoid(x_val: float, threshold: float, beta: float) -> float:
    """Fuzzy membership: probability of going right.

    μ_right(x) = σ((x - threshold) / beta)
    β: bandwidth. β → 0: approaches hard split.
    """
    z = (x_val - threshold) / max(beta, 1e-8)
    return float(1.0 / (1.0 + np.exp(-np.clip(z, -30, 30))))


class FuzzyDecisionTree(RegressorMixin, BaseEstimator):
    """Fuzzy Decision Tree for regression.

    Uses soft (fuzzy) membership at each split node:
        μ_right(x) = σ((x[f] - threshold) / β)

    Each sample is distributed across all leaves with soft weights,
    making the prediction a weighted average of leaf values.

    Trained top-down like CART but with fuzzy prediction.

    Parameters
    ----------
    max_depth : int, default=4
    min_samples_leaf : int, default=5
    beta : float, default=0.5
        Fuzzy bandwidth. Larger → softer decisions.
    criterion : {"mse"}, default="mse"
    random_state : int or None

    Examples
    --------
    >>> from forests import FuzzyDecisionTree
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((200, 3))
    >>> y = 2 * X[:, 0] - X[:, 1] + rng.normal(0, 0.1, 200)
    >>> fdt = FuzzyDecisionTree(max_depth=4, beta=0.3, random_state=0)
    >>> fdt.fit(X, y).score(X, y) > 0.8
    True
    """

    def __init__(
        self,
        max_depth: int = 4,
        min_samples_leaf: int = 5,
        beta: float = 0.5,
        criterion: str = "mse",
        random_state: Optional[int] = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.beta = beta
        self.criterion = criterion
        self.random_state = random_state

    def _find_split(self, X: np.ndarray, y: np.ndarray):
        """Axis-aligned CART split (structure finding; hard split for tree building)."""
        n, p = X.shape
        base = np.mean((y - np.mean(y)) ** 2) if len(y) > 0 else 0.0
        best_gain, best_f, best_thr = 0.0, None, None

        for f in range(p):
            vals = np.unique(X[:, f])
            if len(vals) < 2:
                continue
            for thr in (vals[:-1] + vals[1:]) / 2.0:
                lm = X[:, f] <= thr
                nl, nr = lm.sum(), n - lm.sum()
                if nl < self.min_samples_leaf or nr < self.min_samples_leaf:
                    continue
                imp_l = np.mean((y[lm] - np.mean(y[lm])) ** 2) if nl > 0 else 0.0
                imp_r = np.mean((y[~lm] - np.mean(y[~lm])) ** 2) if nr > 0 else 0.0
                gain = base - (nl / n) * imp_l - (nr / n) * imp_r
                if gain > best_gain:
                    best_gain, best_f, best_thr = gain, int(f), float(thr)

        return best_f, best_thr

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        node = Node(n_samples=len(y), depth=depth, value=np.array([np.mean(y)]))
        if depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf:
            return node

        best_f, best_thr = self._find_split(X, y)
        if best_f is None:
            return node

        node.feature = best_f
        node.threshold = best_thr
        # Hard split for tree structure (soft membership used at predict time)
        mask = X[:, best_f] <= best_thr
        node.left = self._build(X[mask], y[mask], depth + 1)
        node.right = self._build(X[~mask], y[~mask], depth + 1)
        return node

    def _predict_fuzzy(self, x: np.ndarray, node: Node, pi: float) -> float:
        """Recursively compute weighted leaf contributions.

        Returns cumulative contribution pi * value at leaves.
        """
        if node.is_leaf or node.feature is None:
            return pi * float(node.value[0])
        mu_right = _fuzzy_sigmoid(float(x[node.feature]), node.threshold, self.beta)
        mu_left = 1.0 - mu_right
        left_val = self._predict_fuzzy(x, node.left, pi * mu_left)
        right_val = self._predict_fuzzy(x, node.right, pi * mu_right)
        return left_val + right_val

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FuzzyDecisionTree":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.root_ = self._build(X, y, 0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, 'root_') or self.root_ is None:
            raise RuntimeError("Call fit() first.")
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_fuzzy(x, self.root_, 1.0) for x in X])
