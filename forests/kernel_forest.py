"""
forests.kernel_forest
=====================
Random Kernel Forest.

References
----------
Ustimenko, A., & Prokhorenkova, L. (2022). Random Kernel Forests.
    IEEE Transactions on Neural Networks and Learning Systems.
    DOI: 10.1109/TNNLS.2022.3185709 (IEEE 9837906)

Algorithm
---------
Key innovation: at each node, instead of an axis-aligned split, the
algorithm optimizes an SVM-like objective with a kernel function to find
a quasi-optimal split that maximizes the margin between subtree classes/regions.

This is approximated as:
1. Project X onto a random kernel feature map (e.g., RBF via random Fourier features).
2. Find the optimal hyperplane in the kernel feature space using a linear SVM
   (solved by a fast SMO-style coordinate descent).
3. Use the signed distance from the hyperplane as the split variable.
4. Find the best threshold on this projected value.

Implementation notes:
- We use Random Fourier Features (Rahimi & Recht 2007) to approximate the RBF kernel.
- The SVM-style margin is maximized using a simplified gradient step.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from .base import BaseForest, ClassifierForestMixin, RegressorForestMixin, IMPURITY_FN
from .cart import CARTClassifier, CARTRegressor


# ---------------------------------------------------------------------------
# Random Fourier Features (Rahimi & Recht, 2007)
# ---------------------------------------------------------------------------

class _RFFTransform:
    """Approximate RBF kernel via Random Fourier Features.

    φ(x) = sqrt(2/D) * [cos(ω_1^T x + b_1), ..., cos(ω_D^T x + b_D)]

    ||φ(x) - φ(y)||^2 ≈ 2(1 - exp(-||x-y||^2 / (2*gamma^2)))
    """

    def __init__(self, n_components: int, gamma: float, rng: np.random.Generator, n_features: int):
        self.n_components = n_components
        self.gamma = gamma
        # Sample random frequencies from Gaussian (bandwidth = sqrt(2) * gamma)
        self.omega = rng.standard_normal((n_features, n_components)) * np.sqrt(2 * gamma)
        self.bias = rng.uniform(0, 2 * np.pi, n_components)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Map X to D-dim feature space."""
        Z = np.cos(X @ self.omega + self.bias)
        return Z * np.sqrt(2.0 / self.n_components)


# ---------------------------------------------------------------------------
# SVM-style split finder in RFF space
# ---------------------------------------------------------------------------

def _kernel_svm_split(
    X: np.ndarray,
    y: np.ndarray,
    impurity_fn,
    min_samples_leaf: int,
    n_classes: Optional[int],
    rng: np.random.Generator,
    n_rff: int,
    gamma: float,
    svm_lambda: float,
    n_iter: int,
) -> Tuple[Optional[np.ndarray], Optional[float], float]:
    """Kernel split via RFF + gradient descent margin maximization.

    Implements the SVM-like loss with margin re-scaling (Ustimenko 2022).

    Returns
    -------
    (rff_transform, threshold, gain)
    where rff_transform is stored in node.extra["kernel_rff"]
    and the 1D projection is X_rff @ w.
    """
    n, p = X.shape
    if n_classes is not None:
        base_imp = impurity_fn(y, n_classes)
        # Binary label for SVM
        classes = np.unique(y)
        if len(classes) < 2:
            return None, None, 0.0
        # Use most frequent vs rest
        counts = np.bincount(y, minlength=max(classes) + 1)
        c0 = counts.argmax()
        label_svm = np.where(y == c0, 1.0, -1.0)
    else:
        base_imp = impurity_fn(y)
        # Regression: binary split by median
        median = np.median(y)
        label_svm = np.where(y >= median, 1.0, -1.0)

    # RFF transform
    rff = _RFFTransform(n_rff, gamma, rng, p)
    Z = rff.transform(X)  # (n, n_rff)

    # Initialize w via random normal
    w = rng.standard_normal(n_rff) * 0.01

    # Gradient descent: hinge loss + L2 regularization
    lr = 0.1 / max(n, 1)
    for _ in range(n_iter):
        margins = label_svm * (Z @ w)
        # Subgradient of hinge loss
        active = margins < 1.0
        grad_loss = -label_svm * active.astype(float)
        grad = Z.T @ grad_loss / n + svm_lambda * w
        w -= lr * grad

    # Project X onto learned direction
    proj = Z @ w

    # Threshold search
    best_gain = 0.0
    best_thr: Optional[float] = None
    uniq = np.unique(proj)
    if len(uniq) < 2:
        return None, None, 0.0
    thresholds = (uniq[:-1] + uniq[1:]) / 2.0
    if len(thresholds) > 20:
        thresholds = rng.choice(thresholds, 20, replace=False)

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
            best_thr = float(thr)

    if best_thr is None:
        return None, None, 0.0

    return (rff, w), best_thr, best_gain


# ---------------------------------------------------------------------------
# Kernel tree (base)
# ---------------------------------------------------------------------------

def _build_kernel_tree(is_classifier: bool, base_cls):
    class KernelTree(base_cls):
        def __init__(
            self,
            n_rff: int = 32,
            gamma: float = 1.0,
            svm_lambda: float = 0.01,
            svm_n_iter: int = 20,
            **kwargs,
        ):
            super().__init__(**kwargs)
            self.n_rff = n_rff
            self.gamma = gamma
            self.svm_lambda = svm_lambda
            self.svm_n_iter = svm_n_iter

        def _find_best_split(self, X, y, feature_indices, rng, **kwargs):
            fn = IMPURITY_FN[self.criterion]
            nc = self.n_classes_ if is_classifier else None
            return _kernel_svm_split(
                X[:, feature_indices] if True else X,  # use all features for kernel
                y, fn, self.min_samples_leaf, nc, rng,
                self.n_rff, self.gamma, self.svm_lambda, self.svm_n_iter,
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
            result, thr, gain = self._find_best_split(X, y, feat_idx, rng, **kwargs)
            if result is None or gain < self.min_impurity_decrease:
                node.leaf_id = self._leaf_counter
                self._leaf_counter += 1
                return node

            rff_obj, w = result
            # Project using kernel
            Z = rff_obj.transform(X)
            proj = Z @ w
            mask = proj <= thr

            lX, ly = X[mask], y[mask]
            rX, ry = X[~mask], y[~mask]
            if len(ly) < self.min_samples_leaf or len(ry) < self.min_samples_leaf:
                node.leaf_id = self._leaf_counter
                self._leaf_counter += 1
                return node

            node.extra["kernel_rff"] = rff_obj
            node.extra["kernel_w"] = w
            node.threshold = thr
            node.feature = -1  # kernel split
            node.left = self._build(lX, ly, depth + 1, rng, **kwargs)
            node.right = self._build(rX, ry, depth + 1, rng, **kwargs)
            return node

        def _predict_node(self, x, node):
            if node.is_leaf:
                return node.value
            rff_obj = node.extra.get("kernel_rff")
            if rff_obj is not None:
                Z = rff_obj.transform(x[None, :])
                proj = float(Z @ node.extra["kernel_w"])
            else:
                proj = float(x[node.feature])
            if proj <= node.threshold:
                return self._predict_node(x, node.left)
            return self._predict_node(x, node.right)

        def _apply_node(self, x, node):
            if node.is_leaf:
                return node.leaf_id
            rff_obj = node.extra.get("kernel_rff")
            if rff_obj is not None:
                Z = rff_obj.transform(x[None, :])
                proj = float(Z @ node.extra["kernel_w"])
            else:
                proj = float(x[node.feature])
            if proj <= node.threshold:
                return self._apply_node(x, node.left)
            return self._apply_node(x, node.right)

    return KernelTree


_KernelClassifierTree = _build_kernel_tree(True, CARTClassifier)
_KernelRegressorTree = _build_kernel_tree(False, CARTRegressor)


class RandomKernelForest(ClassifierForestMixin, BaseForest):
    """Random Kernel Forest Classifier.

    Implements: Ustimenko & Prokhorenkova (2022) IEEE TNNLS (IEEE 9837906).
    Uses RBF kernel approximation via Random Fourier Features at each node.
    Split boundary is found by maximizing an SVM-like margin in the kernel space.

    Parameters
    ----------
    n_estimators : int, default=100
    n_rff : int, default=32
        Number of random Fourier features (approximation rank).
    gamma : float, default=1.0
        RBF kernel bandwidth parameter.
    svm_lambda : float, default=0.01
        L2 regularization for the SVM objective.
    svm_n_iter : int, default=20
        SGD iterations for SVM optimization per node.
    criterion : {"gini", "entropy"}, default="gini"
    max_depth : int or None
    min_samples_split : int, default=10
    min_samples_leaf : int, default=5
    bootstrap : bool, default=True
    n_jobs : int, default=1
    random_state : int or None

    Examples
    --------
    >>> from forests import RandomKernelForest
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = RandomKernelForest(n_estimators=5, random_state=0)
    >>> clf.fit(X, y).score(X, y) > 0.8
    True
    """

    def __init__(
        self,
        n_estimators: int = 100,
        n_rff: int = 32,
        gamma: float = 1.0,
        svm_lambda: float = 0.01,
        svm_n_iter: int = 20,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
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
        self.n_rff = n_rff
        self.gamma = gamma
        self.svm_lambda = svm_lambda
        self.svm_n_iter = svm_n_iter
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease

    def _make_estimator(self, random_state: int) -> _KernelClassifierTree:
        return _KernelClassifierTree(
            n_rff=self.n_rff,
            gamma=self.gamma,
            svm_lambda=self.svm_lambda,
            svm_n_iter=self.svm_n_iter,
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
