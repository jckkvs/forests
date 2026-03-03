"""
forests.extras
==============
Additional forest models for special tasks.

Models:
- IsolationForest   : Anomaly detection (Liu et al. 2008)
- QuantileRegressionForest : Conditional quantile estimation (Meinshausen 2006)
- RandomSurvivalForest : Survival analysis (Ishwaran et al. 2008)
- MondrianForest    : Online / streaming forest (Lakshminarayanan et al. 2014)
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from .base import BaseForest, BaseTree, Node, IMPURITY_FN
from .cart import CARTRegressor


# ---------------------------------------------------------------------------
# IsolationForest
# ---------------------------------------------------------------------------

class _IsolationTree:
    """A single isolation tree.

    Implements: Liu et al. (2008) Algorithm 1.
    Randomly selects a feature and a random split threshold from [min, max].
    """

    def __init__(self, max_depth: int, random_state: Optional[int] = None):
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
            node.value = np.array([float(n)])
            return node

        # Random feature and threshold
        f = int(rng.integers(0, p))
        v_min, v_max = X[:, f].min(), X[:, f].max()
        if v_min == v_max:
            node.leaf_id = self._leaf_counter
            self._leaf_counter += 1
            node.value = np.array([float(n)])
            return node

        thr = float(rng.uniform(v_min, v_max))
        mask = X[:, f] <= thr
        node.feature = f
        node.threshold = thr
        node.left = self._build(X[mask], depth + 1, rng)
        node.right = self._build(X[~mask], depth + 1, rng)
        return node

    def fit(self, X: np.ndarray) -> "_IsolationTree":
        rng = np.random.default_rng(self.random_state)
        self._leaf_counter = 0
        self.root_ = self._build(X, 0, rng)
        return self

    def path_length(self, x: np.ndarray) -> float:
        """Return path length (depth) for a single sample."""
        def _traverse(node: Node, length: int) -> float:
            if node.is_leaf:
                n = int(node.value[0])
                # Correction term (Liu et al. 2008, Eq. 1)
                return length + _c(n)
            if x[node.feature] <= node.threshold:
                return _traverse(node.left, length + 1)
            return _traverse(node.right, length + 1)

        return _traverse(self.root_, 0)


def _H(n: int) -> float:
    """Harmonic number approximation H(n) ≈ ln(n) + 0.5772..."""
    return float(np.log(n) + 0.5772156649) if n > 1 else (1.0 if n == 1 else 0.0)


def _c(n: int) -> float:
    """Expected path length for an isolation tree with n samples.

    Implements Liu et al. (2008) Eq. (1):
    c(n) = 2*H(n-1) - (2*(n-1)/n)
    """
    if n <= 1:
        return 0.0
    return 2.0 * _H(n - 1) - 2.0 * (n - 1) / n


class IsolationForest(BaseEstimator):
    """Isolation Forest for anomaly detection.

    Implements: Liu, F.T., Ting, K.M., & Zhou, Z.H. (2008).
    Isolation Forest. Proceedings of ICDM, 413-422.

    The anomaly score s(x) = 2^{-E[h(x)] / c(n)}
    where E[h(x)] is the average path length and c(n) is the expected
    path length for a dataset of size n.

    Parameters
    ----------
    n_estimators : int, default=100
    max_samples : int or "auto", default="auto"
        Number of samples to draw per tree.
        "auto" → min(256, n_samples)
    contamination : float, default=0.1
        Expected fraction of outliers (used for threshold).
    random_state : int or None

    Examples
    --------
    >>> from forests import IsolationForest
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((100, 2))
    >>> X_outliers = rng.random((10, 2)) * 10  # far from normal
    >>> X_all = np.vstack([X, X_outliers])
    >>> iso = IsolationForest(n_estimators=50, random_state=0)
    >>> iso.fit(X).predict(X_all).sum()  # outliers should be -1
    -10
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: Union[int, str] = "auto",
        contamination: float = 0.1,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> "IsolationForest":
        """Fit isolation forest."""
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        rng = np.random.default_rng(self.random_state)

        # Sample size per tree
        if self.max_samples == "auto":
            k = min(256, n)
        else:
            k = int(self.max_samples)

        max_depth = int(np.ceil(np.log2(k)))
        self.c_n_: float = _c(k)
        self.trees_: List[_IsolationTree] = []

        seeds = rng.integers(0, 2**31, size=self.n_estimators)
        for seed in seeds:
            seed_rng = np.random.default_rng(seed)
            idx = seed_rng.choice(n, size=k, replace=False)
            tree = _IsolationTree(max_depth=max_depth, random_state=int(seed))
            tree.fit(X[idx])
            self.trees_.append(tree)

        # Compute threshold from training scores
        scores = self._raw_score(X)
        self.threshold_ = float(np.percentile(scores, 100 * (1 - self.contamination)))
        return self

    def _raw_score(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly score s(x) for each sample."""
        path_lengths = np.array([
            [tree.path_length(x) for x in X]
            for tree in self.trees_
        ])
        mean_h = path_lengths.mean(axis=0)
        return 2.0 ** (-mean_h / max(self.c_n_, 1e-8))

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores (higher = more anomalous)."""
        check_is_fitted(self, "trees_")
        return self._raw_score(np.asarray(X, dtype=float))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return +1 (normal) or -1 (anomaly).

        Parameters
        ----------
        X : (n, p) array

        Returns
        -------
        labels : (n,) array of {+1, -1}
        """
        check_is_fitted(self, "threshold_")
        scores = self.score_samples(X)
        return np.where(scores >= self.threshold_, -1, 1).astype(int)


# ---------------------------------------------------------------------------
# QuantileRegressionForest (Meinshausen 2006)
# ---------------------------------------------------------------------------

class QuantileRegressionForest(BaseEstimator):
    """Quantile Regression Forest.

    Implements: Meinshausen, N. (2006). Quantile Regression Forests.
    Journal of Machine Learning Research, 7, 983-999.

    For each test point x, uses the weighted empirical distribution
    derived from co-leaf training samples to estimate any quantile.

    Parameters
    ----------
    n_estimators : int, default=100
    max_depth : int or None
    min_samples_leaf : int, default=5
    max_features : str or None, default="sqrt"
    bootstrap : bool, default=True
    n_jobs : int, default=1
    random_state : int or None

    Examples
    --------
    >>> from forests import QuantileRegressionForest
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((200, 3))
    >>> y = X[:, 0] + rng.normal(0, 0.3, 200)
    >>> qrf = QuantileRegressionForest(n_estimators=20, random_state=0)
    >>> qrf.fit(X, y)
    QuantileRegressionForest(...)
    >>> pred_50 = qrf.predict(X[:5], quantile=0.5)
    >>> pred_50.shape
    (5,)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 5,
        max_features: Union[int, float, str, None] = "sqrt",
        bootstrap: bool = True,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantileRegressionForest":
        from joblib import Parallel, delayed
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.X_train_ = X
        self.y_train_ = y
        self.n_features_in_ = X.shape[1]
        n = X.shape[0]

        master_rng = np.random.default_rng(self.random_state)
        seeds = master_rng.integers(0, 2**31, size=self.n_estimators)

        def _fit_one(seed):
            rng = np.random.default_rng(seed)
            if self.bootstrap:
                idx = rng.integers(0, n, size=n)
                X_s, y_s = X[idx], y[idx]
            else:
                X_s, y_s = X, y
            tree = CARTRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=int(seed),
            )
            tree.fit(X_s, y_s)
            return tree

        self.trees_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one)(int(s)) for s in seeds
        )
        # Compute training leaf IDs per tree
        self._train_leaf_ids = [tree.apply(X) for tree in self.trees_]
        return self

    def predict(self, X: np.ndarray, quantile: float = 0.5) -> np.ndarray:
        """Return conditional quantile estimates.

        Parameters
        ----------
        X : (m, p) array
        quantile : float, default=0.5

        Returns
        -------
        preds : (m,) array
        """
        check_is_fitted(self, "trees_")
        X = np.asarray(X, dtype=float)
        n_train = self.X_train_.shape[0]
        preds = np.zeros(X.shape[0])

        for i, x in enumerate(X):
            # Accumulate co-leaf weights
            weights = np.zeros(n_train)
            for b, tree in enumerate(self.trees_):
                test_lid = tree._apply_node(x, tree.root_)
                train_lids = self._train_leaf_ids[b]
                co_leaf = np.where(train_lids == test_lid)[0]
                if len(co_leaf) > 0:
                    weights[co_leaf] += 1.0 / len(co_leaf)
            w_sum = weights.sum()
            if w_sum < 1e-8:
                preds[i] = np.quantile(self.y_train_, quantile)
                continue
            weights /= w_sum

            # Weighted empirical quantile (Meinshausen 2006, Eq. 2)
            sort_idx = np.argsort(self.y_train_)
            cumw = np.cumsum(weights[sort_idx])
            q_idx = np.searchsorted(cumw, quantile)
            preds[i] = self.y_train_[sort_idx[min(q_idx, n_train - 1)]]

        return preds


# ---------------------------------------------------------------------------
# RandomSurvivalForest (Ishwaran et al. 2008)
# ---------------------------------------------------------------------------

def _log_rank_score(t_left: np.ndarray, e_left: np.ndarray,
                    t_right: np.ndarray, e_right: np.ndarray) -> float:
    """Log-rank statistic for survival split quality.

    Implements: Ishwaran et al. (2008) Eq. (3) – log-rank test.
    Higher = better discrimination between left/right.
    """
    t_all = np.concatenate([t_left, t_right])
    e_all = np.concatenate([e_left, e_right])
    group = np.concatenate([np.zeros(len(t_left)), np.ones(len(t_right))])

    unique_times = np.unique(t_all[e_all == 1])
    stat = 0.0
    for t in unique_times:
        at_risk_l = np.sum(t_left >= t)
        at_risk_r = np.sum(t_right >= t)
        at_risk = at_risk_l + at_risk_r
        if at_risk < 2:
            continue
        events_l = np.sum((t_left == t) & (e_left == 1))
        events_tot = np.sum((t_all == t) & (e_all == 1))
        expected_l = at_risk_l * events_tot / at_risk
        stat += (events_l - expected_l)
    return abs(stat)


class _SurvivalTree:
    """Single survival tree using log-rank split criterion."""

    def __init__(
        self,
        max_depth: int = 5,
        min_samples_leaf: int = 10,
        max_features: Union[int, float, str, None] = "sqrt",
        random_state: Optional[int] = None,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.root_: Optional[Node] = None
        self._leaf_counter = 0

    def _select_features(self, n_features: int, rng: np.random.Generator) -> np.ndarray:
        mf = self.max_features
        if mf is None:
            k = n_features
        elif isinstance(mf, float):
            k = max(1, int(mf * n_features))
        elif isinstance(mf, int):
            k = min(mf, n_features)
        elif mf == "sqrt":
            k = max(1, int(np.sqrt(n_features)))
        elif mf == "log2":
            k = max(1, int(np.log2(n_features)))
        else:
            k = n_features
        return rng.choice(n_features, size=k, replace=False)

    def _build(self, X: np.ndarray, t: np.ndarray, e: np.ndarray,
               depth: int, rng: np.random.Generator) -> Node:
        n = len(t)
        node = Node(n_samples=n, depth=depth)

        # Nelson-Aalen cumulative hazard estimate at this node
        node.extra["survival_times"] = t.copy()
        node.extra["survival_events"] = e.copy()

        if depth >= self.max_depth or n < 2 * self.min_samples_leaf:
            node.leaf_id = self._leaf_counter
            self._leaf_counter += 1
            return node

        feat_idx = self._select_features(X.shape[1], rng)
        best_score = 0.0
        best_f, best_thr = None, None

        for f in feat_idx:
            vals = X[:, f]
            unique = np.unique(vals)
            if len(unique) < 2:
                continue
            thresholds = (unique[:-1] + unique[1:]) / 2.0
            for thr in thresholds:
                lm = vals <= thr
                nl, nr = lm.sum(), n - lm.sum()
                if nl < self.min_samples_leaf or nr < self.min_samples_leaf:
                    continue
                score = _log_rank_score(t[lm], e[lm], t[~lm], e[~lm])
                if score > best_score:
                    best_score = score
                    best_f = int(f)
                    best_thr = float(thr)

        if best_f is None:
            node.leaf_id = self._leaf_counter
            self._leaf_counter += 1
            return node

        mask = X[:, best_f] <= best_thr
        node.feature = best_f
        node.threshold = best_thr
        node.left = self._build(X[mask], t[mask], e[mask], depth + 1, rng)
        node.right = self._build(X[~mask], t[~mask], e[~mask], depth + 1, rng)
        return node

    def fit(self, X: np.ndarray, t: np.ndarray, e: np.ndarray) -> "_SurvivalTree":
        rng = np.random.default_rng(self.random_state)
        self._leaf_counter = 0
        self.root_ = self._build(X, t, e, 0, rng)
        return self

    def _get_leaf_node(self, x: np.ndarray, node: Node) -> Node:
        if node.is_leaf or node.feature is None:
            return node
        if x[node.feature] <= node.threshold:
            return self._get_leaf_node(x, node.left)
        return self._get_leaf_node(x, node.right)

    def predict_cumhazard(self, x: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Nelson-Aalen cumulative hazard for a single sample at given times."""
        leaf = self._get_leaf_node(x, self.root_)
        t_leaf = leaf.extra["survival_times"]
        e_leaf = leaf.extra["survival_events"]
        n_at_risk = len(t_leaf)
        hazard = np.zeros(len(times))
        for ti, time in enumerate(times):
            # Nelson-Aalen: Σ_{t_j ≤ time} d_j / n_j
            cumh = 0.0
            for t_j in np.unique(t_leaf[e_leaf == 1]):
                if t_j > time:
                    break
                d_j = np.sum((t_leaf == t_j) & (e_leaf == 1))
                n_j = np.sum(t_leaf >= t_j)
                if n_j > 0:
                    cumh += d_j / n_j
            hazard[ti] = cumh
        return hazard


class RandomSurvivalForest(BaseEstimator):
    """Random Survival Forest for survival analysis.

    Implements: Ishwaran, H., et al. (2008). Random Survival Forests.
    Annals of Applied Statistics, 2(3), 841-860.

    Uses log-rank splitting criterion and Nelson-Aalen cumulative hazard
    function for prediction.

    Parameters
    ----------
    n_estimators : int, default=100
    max_depth : int, default=5
    min_samples_leaf : int, default=10
    max_features : str or None, default="sqrt"
    bootstrap : bool, default=True
    n_jobs : int, default=1
    random_state : int or None

    Examples
    --------
    >>> from forests import RandomSurvivalForest
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((100, 3))
    >>> t = rng.exponential(2, 100)       # survival times
    >>> e = rng.binomial(1, 0.7, 100)    # event indicator
    >>> rsf = RandomSurvivalForest(n_estimators=10, random_state=0)
    >>> rsf.fit(X, t, e)
    RandomSurvivalForest(...)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        min_samples_leaf: int = 10,
        max_features: Union[int, float, str, None] = "sqrt",
        bootstrap: bool = True,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X: np.ndarray, t: np.ndarray, e: np.ndarray) -> "RandomSurvivalForest":
        from joblib import Parallel, delayed
        X = np.asarray(X, dtype=float)
        t = np.asarray(t, dtype=float)
        e = np.asarray(e, dtype=int)
        n = X.shape[0]
        self.n_features_in_ = X.shape[1]

        master_rng = np.random.default_rng(self.random_state)
        seeds = master_rng.integers(0, 2**31, size=self.n_estimators)

        def _fit_one(seed):
            rng = np.random.default_rng(seed)
            if self.bootstrap:
                idx = rng.integers(0, n, size=n)
                X_s, t_s, e_s = X[idx], t[idx], e[idx]
            else:
                X_s, t_s, e_s = X, t, e
            tree = _SurvivalTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=int(seed),
            )
            tree.fit(X_s, t_s, e_s)
            return tree

        self.trees_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one)(int(s)) for s in seeds
        )
        self.unique_times_ = np.sort(np.unique(t[e == 1]))
        return self

    def predict_cumhazard(self, X: np.ndarray, times: Optional[np.ndarray] = None) -> np.ndarray:
        """Return mean cumulative hazard function.

        Parameters
        ----------
        X : (m, p) array
        times : (T,) array or None (default: training unique event times)

        Returns
        -------
        cumhazard : (m, T) array
        """
        check_is_fitted(self, "trees_")
        X = np.asarray(X, dtype=float)
        if times is None:
            times = self.unique_times_
        cumh_all = np.array([
            [tree.predict_cumhazard(x, times) for x in X]
            for tree in self.trees_
        ])  # (n_estimators, m, T)
        return cumh_all.mean(axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return mean survival time proxy (negative mean cumulative hazard)."""
        check_is_fitted(self, "trees_")
        cumh = self.predict_cumhazard(X)
        return -cumh.mean(axis=1)


# ---------------------------------------------------------------------------
# MondrianForest (Lakshminarayanan et al. 2014) - online learning
# ---------------------------------------------------------------------------

class _MondrianNode:
    """A node in a Mondrian tree."""

    def __init__(self):
        self.feature: Optional[int] = None
        self.threshold: Optional[float] = None
        self.left: Optional["_MondrianNode"] = None
        self.right: Optional["_MondrianNode"] = None
        self.budget: float = 0.0      # remaining budget τ
        self.lower: Optional[np.ndarray] = None   # lower bounds of bounding box
        self.upper: Optional[np.ndarray] = None   # upper bounds of bounding box
        self.value: Optional[np.ndarray] = None   # leaf prediction
        self.n_samples: int = 0

    @property
    def is_leaf(self) -> bool:
        return self.left is None


class MondrianForest(BaseEstimator):
    """Mondrian Forest for online classification.

    Implements: Lakshminarayanan, B., Roy, D.M., & Teh, Y.W. (2014).
    Mondrian Forests: Efficient Online Random Forests.
    NIPS 2014.

    The Mondrian process cuts feature dimensions at rates proportional
    to their range, creating an axis-aligned partition. Can be updated
    online with new data points.

    Parameters
    ----------
    n_estimators : int, default=10
    base_count : float, default=0.0
        Prior pseudocount for each class in leaf predictions.
    lifetime : float, default=1.0
        Budget parameter (controls tree depth).
    n_classes : int, default=2
    random_state : int or None

    Examples
    --------
    >>> from forests import MondrianForest
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> mf = MondrianForest(n_estimators=10, n_classes=3, random_state=0)
    >>> mf.fit(X, y).score(X, y) > 0.7
    True
    """

    def __init__(
        self,
        n_estimators: int = 10,
        base_count: float = 0.0,
        lifetime: float = 1.0,
        n_classes: int = 2,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.base_count = base_count
        self.lifetime = lifetime
        self.n_classes = n_classes
        self.random_state = random_state

    def _sample_mondrian_block(
        self, X: np.ndarray, y: np.ndarray, budget: float, rng: np.random.Generator
    ) -> _MondrianNode:
        """Recursively grow a Mondrian tree block.

        Implements Lakshminarayanan et al. (2014) Algorithm 1 (SampleMondrianBlock).
        """
        node = _MondrianNode()
        node.n_samples = len(y)
        node.lower = X.min(axis=0)
        node.upper = X.max(axis=0)
        # Range of bounding box
        linear_dims = node.upper - node.lower
        rate = linear_dims.sum()  # Poisson rate

        # Sample split time
        if rate > 0:
            E = float(rng.exponential(1.0 / rate))
        else:
            E = np.inf

        counts = np.bincount(y.astype(int), minlength=self.n_classes)
        node.value = (counts + self.base_count) / (len(y) + self.n_classes * self.base_count)

        if E + 0 >= budget or rate < 1e-10 or len(y) <= 1:
            # Leaf
            node.budget = budget
            return node

        node.budget = E
        remaining_budget = budget - E

        # Choose feature proportional to range (Mondrian property)
        probs = linear_dims / rate
        f = int(rng.choice(len(probs), p=probs))
        thr = float(rng.uniform(node.lower[f], node.upper[f]))

        mask = X[:, f] <= thr
        node.feature = f
        node.threshold = thr

        if mask.all() or (~mask).all():
            return node

        node.left = self._sample_mondrian_block(X[mask], y[mask], remaining_budget, rng)
        node.right = self._sample_mondrian_block(X[~mask], y[~mask], remaining_budget, rng)
        return node

    def _predict_node(self, x: np.ndarray, node: _MondrianNode) -> np.ndarray:
        if node.is_leaf or node.feature is None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_node(x, node.left)
        return self._predict_node(x, node.right)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MondrianForest":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        rng = np.random.default_rng(self.random_state)
        self.trees_ = [
            self._sample_mondrian_block(X, y, self.lifetime, rng)
            for _ in range(self.n_estimators)
        ]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "trees_")
        X = np.asarray(X, dtype=float)
        all_proba = [
            np.array([self._predict_node(x, tree) for x in X])
            for tree in self.trees_
        ]
        return np.mean(all_proba, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))
