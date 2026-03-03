"""
forests.similarity
==================
RF-based similarity and kernel methods.

References
----------
Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
    (Section 8: Proximities)

Criminisi, A., Shotton, J., & Konukoglu, E. (2011). Decision Forests for
    Classification, Regression, Density Estimation, Manifold Learning and
    Semi-Supervised Learning. Microsoft Research TR-2011-114.

Algorithm
---------
For a fitted forest, two samples X_i and X_j are "similar" if they fall
into the same leaf node in many trees:

    sim(X_i, X_j) = (1/B) Σ_b 1[leaf_b(X_i) == leaf_b(X_j)]

This gives an (n × n) symmetric similarity matrix S ∈ [0, 1].
The RF kernel matrix K = S (or a normalized/RBF transform of S).
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class RFSimilarity(BaseEstimator, TransformerMixin):
    """Random Forest Proximity / Similarity Matrix.

    Computes pairwise similarity between samples based on how often they
    share the same leaf node across all trees in a fitted forest.

    Can be used with any forest model that has an `apply(X)` method
    returning (n_samples, n_estimators) leaf-id array.

    Parameters
    ----------
    forest : fitted forest estimator
        Must have `apply(X)` returning (n_samples, n_estimators) array.
    normalize : bool, default=True
        Whether to normalize each row of the similarity matrix to [0,1].

    Attributes
    ----------
    similarity_matrix_ : np.ndarray of shape (n_train, n_train)
        Pairwise similarity among training samples (fitted via fit_transform).

    Examples
    --------
    >>> from forests import RandomForestClassifier, RFSimilarity
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> rf = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    >>> sim = RFSimilarity(rf)
    >>> S = sim.fit_transform(X)
    >>> S.shape
    (150, 150)
    >>> (S >= 0).all() and (S <= 1).all()
    True
    """

    def __init__(self, forest=None, normalize: bool = True) -> None:
        self.forest = forest
        self.normalize = normalize

    def fit(self, X: np.ndarray, y=None) -> "RFSimilarity":
        """Compute and store the training similarity matrix.

        Parameters
        ----------
        X : (n, p) array

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        # leaf_ids: (n, n_estimators)
        leaf_ids = self.forest.apply(X)
        n, B = leaf_ids.shape
        self.X_train_ = X.copy()
        self.train_leaf_ids_ = leaf_ids

        # S[i, j] = fraction of trees where i and j share the same leaf
        # Vectorized computation
        S = np.zeros((n, n))
        for b in range(B):
            lids = leaf_ids[:, b]
            # Outer equality check
            same = (lids[:, None] == lids[None, :]).astype(float)
            S += same
        S /= B  # normalize by # trees
        self.similarity_matrix_ = S
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Compute similarity of new samples against training data.

        Parameters
        ----------
        X : (m, p) array

        Returns
        -------
        sim : (m, n_train) array
            sim[i, j] = fraction of trees where X[i] co-leafs with X_train[j].
        """
        check_is_fitted(self, "train_leaf_ids_")
        X = np.asarray(X, dtype=float)
        test_leaf_ids = self.forest.apply(X)  # (m, B)
        m, B = test_leaf_ids.shape
        n = self.train_leaf_ids_.shape[0]
        S = np.zeros((m, n))
        for b in range(B):
            S += (test_leaf_ids[:, b:b+1] == self.train_leaf_ids_[:, b][None, :]).astype(float)
        S /= B
        return S

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Fit on X and return the (n, n) training similarity matrix."""
        self.fit(X)
        return self.similarity_matrix_

    def get_similarity(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute pairwise RF similarity.

        Parameters
        ----------
        X1 : (m, p) array
        X2 : (n, p) array or None
            If None, return X1 vs X1 (symmetric).

        Returns
        -------
        S : (m, n) similarity matrix
        """
        if X2 is None:
            X2_ids = self.forest.apply(np.asarray(X1, dtype=float))
            X1_ids = X2_ids
        else:
            X1_ids = self.forest.apply(np.asarray(X1, dtype=float))
            X2_ids = self.forest.apply(np.asarray(X2, dtype=float))

        B = X1_ids.shape[1]
        S = np.zeros((X1_ids.shape[0], X2_ids.shape[0]))
        for b in range(B):
            S += (X1_ids[:, b:b+1] == X2_ids[:, b][None, :]).astype(float)
        return S / B


# ---------------------------------------------------------------------------
# RFKernel
# ---------------------------------------------------------------------------

class RFKernel(BaseEstimator):
    """Random Forest Kernel Matrix for use with kernel methods (e.g., SVM, GP).

    Normalizes the RF proximity matrix to form a valid positive semi-definite
    kernel:

        K(x_i, x_j) = sim(x_i, x_j) / sqrt(sim(x_i, x_i) * sim(x_j, x_j))

    (Cosine normalization of the proximity)

    Or uses an RBF-like transformation:
        K(x_i, x_j) = exp(-gamma * (1 - sim(x_i, x_j)))

    Parameters
    ----------
    forest : fitted forest estimator
    mode : {"cosine", "rbf", "raw"}, default="cosine"
        Kernel normalization mode.
    gamma : float, default=1.0
        RBF parameter (only used when mode="rbf").

    Examples
    --------
    >>> from forests import RandomForestClassifier, RFKernel
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> rf = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    >>> kernel = RFKernel(rf, mode="cosine")
    >>> K = kernel.fit_transform(X)
    >>> K.shape
    (150, 150)
    >>> # Use with sklearn SVM
    >>> from sklearn.svm import SVC
    >>> svm = SVC(kernel="precomputed")
    >>> svm.fit(K, y).score(K, y)  # doctest: +ELLIPSIS
    ...
    """

    def __init__(
        self,
        forest=None,
        mode: str = "cosine",
        gamma: float = 1.0,
    ) -> None:
        self.forest = forest
        self.mode = mode
        self.gamma = gamma

    def _sim_to_kernel(self, S: np.ndarray) -> np.ndarray:
        if self.mode == "raw":
            return S
        elif self.mode == "rbf":
            return np.exp(-self.gamma * (1.0 - S))
        elif self.mode == "cosine":
            if S.shape[0] == S.shape[1]:
                diag = np.sqrt(np.diag(S))
                diag = np.where(diag < 1e-12, 1e-12, diag)
                return S / diag[:, None] / diag[None, :]
            else:
                # For non-square matrices (e.g. test-train), we can't get diag from S.
                # Since RF proximity diag is always 1.0, normalization is identity.
                return S
        else:
            raise ValueError(f"Unknown mode: {self.mode!r}")

    def fit(self, X: np.ndarray, y=None) -> "RFKernel":
        """Compute kernel matrix on training data."""
        X = np.asarray(X, dtype=float)
        self._sim = RFSimilarity(self.forest)
        S = self._sim.fit_transform(X)
        self.K_train_ = self._sim_to_kernel(S)
        self.X_train_ = X.copy()
        self.diag_train_ = np.diag(S).copy()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return test-vs-train kernel matrix.

        Parameters
        ----------
        X : (m, p) array

        Returns
        -------
        K : (m, n_train) array
        """
        check_is_fitted(self, "K_train_")
        X = np.asarray(X, dtype=float)
        S_test = self._sim.transform(X)
        if self.mode == "raw":
            return S_test
        elif self.mode == "rbf":
            return np.exp(-self.gamma * (1.0 - S_test))
        elif self.mode == "cosine":
            # Diagonal of test-test similarity (approx via 1 for normalized)
            diag_train = np.where(self.diag_train_ < 1e-12, 1e-12, self.diag_train_)
            # Compute test diag
            test_leaf_ids = self.forest.apply(X)
            B = test_leaf_ids.shape[1]
            # Diagonal = fraction of trees where test point co-leafs with itself = 1.0
            diag_test = np.ones(X.shape[0])
            return S_test / np.sqrt(diag_test[:, None]) / np.sqrt(diag_train[None, :])
        raise ValueError(f"Unknown mode: {self.mode!r}")

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Fit and return (n, n) kernel matrix."""
        self.fit(X)
        return self.K_train_

    def get_kernel_matrix(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        """Return kernel matrix between X1 and X2.

        Parameters
        ----------
        X1 : (m, p) array
        X2 : (n, p) array or None → symmetric X1 x X1

        Returns
        -------
        K : (m, n) array
        """
        sim = RFSimilarity(self.forest)
        S = sim.get_similarity(X1, X2)
        return self._sim_to_kernel(S)
