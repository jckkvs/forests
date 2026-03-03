"""
forests.soft_tree
=================
Soft (Sigmoid-Gated) Decision Trees and Forests.

References
----------
Irsoy, O., Yildiz, O.T., & Alpaydin, E. (2012). Soft decision trees.
    Proceedings of ICPR, 1143–1146.

Frosst, N., & Hinton, G. (2017). Distilling a neural network into a soft
    decision tree. CEUR Workshop Proceedings, NIPS 2017 Workshop.

Jordan, M.I., & Jacobs, R.A. (1994). Hierarchical mixtures of experts.
    Neural Computation, 6(2), 181–214.

Algorithm
---------
A soft decision tree uses a sigmoid gate at each internal node:
    p(go_right | x, node_i) = σ(x^T w_i + b_i)
The prediction at leaf l is:
    F(x) = sum_over_leaves l: π_l(x) * value_l
where π_l(x) = product of gate probabilities along the path to l.
    (go_right prob or 1 - go_right prob at each ancestor)

Parameters are learned via gradient descent to minimize MSE (regression)
or cross-entropy (classification).
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted


# ---------------------------------------------------------------------------
# Internal node structure for soft tree
# ---------------------------------------------------------------------------

class SoftNode:
    """A node in a soft decision tree.

    Attributes
    ----------
    w : np.ndarray of shape (n_features,)
        Weight vector for the sigmoid gate.
    b : float
        Bias term.
    value : np.ndarray
        Leaf prediction (only for leaf nodes).
    left : SoftNode or None
    right : SoftNode or None
    depth : int
    """

    def __init__(self, depth: int = 0) -> None:
        self.depth = depth
        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0
        self.value: Optional[np.ndarray] = None
        self.left: Optional["SoftNode"] = None
        self.right: Optional["SoftNode"] = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


# ---------------------------------------------------------------------------
# SoftDecisionTree
# ---------------------------------------------------------------------------

class SoftDecisionTree(BaseEstimator):
    """Soft (Sigmoid-Gated) Decision Tree.

    Implements: Frosst & Hinton (2017) / Irsoy et al. (2012).

    Each internal node computes:
        p_right(x) = σ(x^T w + b)
    Each leaf contributes to the final prediction, weighted by the
    path probability π_l(x) = Π_{ancestors} p or (1-p).

    Parameters
    ----------
    max_depth : int, default=4
        Tree depth. Number of leaves = 2^max_depth.
    learning_rate : float, default=0.01
        SGD learning rate for gate parameters.
    n_epochs : int, default=100
        Number of gradient descent epochs.
    batch_size : int, default=64
        Mini-batch size for SGD.
    temperature : float, default=1.0
        Softmax temperature at leaves (1.0 = no temperature scaling).
    task : {"classification", "regression"}
        Learning task.
    n_classes : int, default=1
        Number of classes (for classification).
    random_state : int or None

    Examples
    --------
    >>> from forests import SoftDecisionTree
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> X = X / X.max(axis=0)  # normalize
    >>> sdt = SoftDecisionTree(max_depth=3, task="classification",
    ...                         n_classes=3, n_epochs=50, random_state=0)
    >>> sdt.fit(X, y).score(X, y) > 0.8
    True
    """

    def __init__(
        self,
        max_depth: int = 4,
        learning_rate: float = 0.01,
        n_epochs: int = 100,
        batch_size: int = 64,
        temperature: float = 1.0,
        task: str = "classification",
        n_classes: int = 1,
        random_state: Optional[int] = None,
    ) -> None:
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.temperature = temperature
        self.task = task
        self.n_classes = n_classes
        self.random_state = random_state

    def _build_tree(self, depth: int, rng: np.random.Generator, n_features: int) -> SoftNode:
        """Recursively build tree structure with random initial weights."""
        node = SoftNode(depth=depth)
        if depth >= self.max_depth:
            # Leaf: initialize value
            if self.task == "classification":
                # Xavier-like init for leaf weights
                node.value = rng.standard_normal(self.n_classes_) * 0.1
            else:
                node.value = np.zeros(1)
        else:
            # Better initial weights for gates
            node.w = rng.standard_normal(n_features) * (1.0 / np.sqrt(n_features))
            node.b = 0.0
            node.left = self._build_tree(depth + 1, rng, n_features)
            node.right = self._build_tree(depth + 1, rng, n_features)
        return node

    def _get_all_leaves(self) -> List[SoftNode]:
        """Return all leaf nodes."""
        leaves = []

        def _collect(node):
            if node is None:
                return
            if node.is_leaf:
                leaves.append(node)
            else:
                _collect(node.left)
                _collect(node.right)

        _collect(self.root_)
        return leaves

    def _get_all_internal(self) -> List[SoftNode]:
        """Return all internal (non-leaf) nodes."""
        nodes = []

        def _collect(node):
            if node is None or node.is_leaf:
                return
            nodes.append(node)
            _collect(node.left)
            _collect(node.right)

        _collect(self.root_)
        return nodes

    def _path_probs(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple]]:
        """Compute path probabilities for all leaves.

        Returns
        -------
        leaf_probs : list of (n,) arrays
            π_l(x) for each leaf.
        leaf_nodes : list of SoftNode
        """
        n = X.shape[0]
        leaves = []

        def _recurse(node: SoftNode, pi: np.ndarray):
            """pi: path probability up to this node, shape (n,)"""
            if node.is_leaf:
                leaves.append((pi, node))
                return
            # Gate probability
            logit = X @ node.w + node.b
            p_right = _sigmoid(logit)
            p_left = 1.0 - p_right
            _recurse(node.left, pi * p_left)
            _recurse(node.right, pi * p_right)

        _recurse(self.root_, np.ones(n))
        path_probs = [lp for lp, _ in leaves]
        leaf_nodes = [ln for _, ln in leaves]
        return path_probs, leaf_nodes

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Return weighted sum of leaf values.

        Implements: F(x) = sum_l π_l(x) * v_l
        shape: (n, out_dim)
        """
        path_probs, leaf_nodes = self._path_probs(X)
        n = X.shape[0]
        out = np.zeros((n, len(leaf_nodes[0].value)))
        for pi, node in zip(path_probs, leaf_nodes):
            out += pi[:, None] * node.value[None, :]
        return out

    def _loss_and_grad(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute loss and perform backward pass.

        Returns mean loss. Updates parameters in-place.
        """
        n = X.shape[0]
        path_probs, leaf_nodes = self._path_probs(X)
        # Prediction
        out = np.zeros((n, len(leaf_nodes[0].value)))
        for pi, node in zip(path_probs, leaf_nodes):
            out += pi[:, None] * node.value[None, :]

        # Loss and output gradient
        if self.task == "classification":
            # Softmax cross-entropy
            out_max = out.max(axis=1, keepdims=True)
            exp_o = np.exp(out - out_max)
            proba = exp_o / exp_o.sum(axis=1, keepdims=True)
            loss = -np.mean(np.log(proba[np.arange(n), y] + 1e-12))
            # gradient of loss w.r.t. out: (proba - one_hot) / n
            d_out = proba.copy()
            d_out[np.arange(n), y] -= 1.0
            d_out /= n
        else:
            # MSE
            loss = float(np.mean((out[:, 0] - y) ** 2))
            d_out = 2.0 * (out[:, 0] - y)[:, None] / n

        # Update leaf values
        for pi, node in zip(path_probs, leaf_nodes):
            # d_loss / d_value_l = sum_i pi_l(x_i) * d_out_i
            grad_v = (pi[:, None] * d_out).sum(axis=0)
            node.value -= self.learning_rate * grad_v

        # Update gate parameters (backprop through path probabilities)
        # We compute the gradient of loss w.r.t. gate logits
        # d_loss / d_logit_node = sum over leaves descendant of node:
        #   sign (+/- for right or left branch) * (pi_l / p_gate) * d_loss/d_v_l
        def _backprop_node(node: SoftNode, pi: np.ndarray, chain_grad: np.ndarray):
            """chain_grad: d_loss/d_pi, shape (n,)"""
            if node.is_leaf:
                return
            logit = X @ node.w + node.b
            p_right = _sigmoid(logit)
            p_left = 1.0 - p_right

            # Recurse to get downstream chain gradients
            # Left child: pi * p_left; right child: pi * p_right
            _backprop_node(node.left, pi * p_left, chain_grad)
            _backprop_node(node.right, pi * p_right, chain_grad)

            # Gradient of loss w.r.t. logit
            # d_loss/d_logit = sum_i chain_grad_i * pi_i * sigmoid'(logit_i)
            # But chain_grad is aggregated from leaves, not directly available here.
            # Use accumulated leaf contributions:
            # We switch to a simpler finite-difference-free approach:
            # d_pi_right/d_logit = p_right*(1-p_right)*pi
            # d_pi_left/d_logit = -p_right*(1-p_right)*pi
            dsigma = p_right * p_left  # σ'(z) = σ(1-σ)
            # compute difference of right vs left contributions weighted by chain
            # Using the marginal contribution approach:
            diff = chain_grad * pi * dsigma  # shape (n,)
            grad_b = float(diff.sum())
            grad_w = (diff[:, None] * X).sum(axis=0)
            node.w -= self.learning_rate * grad_w
            node.b -= self.learning_rate * grad_b

        # Aggregate chain_grad from output gradient
        # chain_grad = d_loss/d_pi_leaf * value_leaf aggregated over leaves
        # => For each internal node, chain_grad contributes via leaf descendants.
        # We compute a simplified version: chain_grad_i = d_loss/d_pi total
        # summed over all leaves descendant.
        # Building this requires another traversal:
        leaf_chain = []
        for pi, node in zip(path_probs, leaf_nodes):
            # d_loss / d_pi_l = (d_out * node.value).sum(axis=1)
            leaf_chain.append((d_out * node.value[None, :]).sum(axis=1))

        def _bprop(node, pi, idx):
            if node.is_leaf:
                return leaf_chain[idx[0]], idx[0] + 1
            logit = X @ node.w + node.b
            p_right = _sigmoid(logit)
            p_left = 1.0 - p_right
            cg_left, idx[0] = _bprop(node.left, pi * p_left, idx)
            cg_right, idx[0] = _bprop(node.right, pi * p_right, idx)
            cg = cg_left + cg_right
            dsigma = p_right * p_left
            diff = cg * pi * dsigma
            node.w -= self.learning_rate * (diff[:, None] * X).sum(axis=0)
            node.b -= float(self.learning_rate * diff.sum())
            return cg, idx[0]

        _bprop(self.root_, np.ones(n), [0])
        return loss

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SoftDecisionTree":
        """Fit the soft tree using mini-batch SGD."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)

        if self.task == "classification":
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)
            label_map = {c: i for i, c in enumerate(self.classes_)}
            y = np.array([label_map[yi] for yi in y])
        else:
            self.n_classes_ = 1
            y = y.astype(float)

        n_features = X.shape[1]
        self.n_features_in_ = n_features
        self.root_ = self._build_tree(0, rng, n_features)
        self.loss_history_: List[float] = []

        n = X.shape[0]
        for epoch in range(self.n_epochs):
            idx = rng.permutation(n)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n, self.batch_size):
                batch = idx[start: start + self.batch_size]
                loss = self._loss_and_grad(X[batch], y[batch])
                epoch_loss += loss
                n_batches += 1
            self.loss_history_.append(epoch_loss / max(n_batches, 1))

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities (classification only)."""
        check_is_fitted(self, "root_")
        X = np.asarray(X, dtype=float)
        raw = self._predict_raw(X)
        # Softmax
        raw_max = raw.max(axis=1, keepdims=True)
        exp_r = np.exp(raw - raw_max)
        return exp_r / exp_r.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions."""
        check_is_fitted(self, "root_")
        X = np.asarray(X, dtype=float)
        raw = self._predict_raw(X)
        if self.task == "classification":
            idx = np.argmax(raw, axis=1)
            return self.classes_[idx]
        else:
            return raw[:, 0]

    def score(self, X, y):
        from sklearn.metrics import accuracy_score, r2_score
        y_pred = self.predict(X)
        if self.task == "classification":
            return accuracy_score(y, y_pred)
        return r2_score(y, y_pred)


# ---------------------------------------------------------------------------
# SoftDecisionForest: ensemble of SoftDecisionTrees
# ---------------------------------------------------------------------------

class SoftDecisionForest(BaseEstimator):
    """Ensemble of Soft Decision Trees.

    Parameters
    ----------
    n_estimators : int, default=10
    max_depth : int, default=4
    learning_rate : float, default=0.01
    n_epochs : int, default=100
    batch_size : int, default=64
    task : {"classification", "regression"}
    n_classes : int, default=1
    bootstrap : bool, default=True
    n_jobs : int, default=1
    random_state : int or None

    Examples
    --------
    >>> from forests import SoftDecisionForest
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> X = X / X.max(axis=0)
    >>> sdf = SoftDecisionForest(n_estimators=5, max_depth=3,
    ...                           task="classification", n_epochs=50, random_state=0)
    >>> sdf.fit(X, y).score(X, y) > 0.8
    True
    """

    def __init__(
        self,
        n_estimators: int = 10,
        max_depth: int = 4,
        learning_rate: float = 0.01,
        n_epochs: int = 100,
        batch_size: int = 64,
        task: str = "classification",
        n_classes: int = 1,
        bootstrap: bool = True,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.task = task
        self.n_classes = n_classes
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SoftDecisionForest":
        from joblib import Parallel, delayed

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if self.task == "classification":
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)
        else:
            self.n_classes_ = 1

        master_rng = np.random.default_rng(self.random_state)
        seeds = master_rng.integers(0, 2**31, size=self.n_estimators)

        def _fit_one(seed):
            rng = np.random.default_rng(seed)
            if self.bootstrap:
                idx = rng.integers(0, X.shape[0], size=X.shape[0])
                X_s, y_s = X[idx], y[idx]
            else:
                X_s, y_s = X, y
            tree = SoftDecisionTree(
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                task=self.task,
                n_classes=self.n_classes_,
                random_state=int(seed),
            )
            tree.fit(X_s, y_s)
            return tree

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one)(int(s)) for s in seeds
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "estimators_")
        X = np.asarray(X, dtype=float)
        return np.mean([t.predict_proba(X) for t in self.estimators_], axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "estimators_")
        if self.task == "classification":
            proba = self.predict_proba(X)
            return self.classes_[np.argmax(proba, axis=1)]
        preds = np.mean([t.predict(X) for t in self.estimators_], axis=0)
        return preds

    def score(self, X, y):
        from sklearn.metrics import accuracy_score, r2_score
        y_pred = self.predict(X)
        if self.task == "classification":
            return accuracy_score(y, y_pred)
        return r2_score(y, y_pred)
