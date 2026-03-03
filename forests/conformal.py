"""
forests.conformal
=================
Conformal Prediction Forest.

References
----------
Vovk, V., Gammerman, A., & Shafer, G. (2005).
    Algorithmic Learning in a Random World. Springer.

Angelopoulos, A.N., & Bates, S. (2023).
    Conformal Risk Control. ICLR 2023. arXiv:2208.02814.

Romano, Y., Patterson, E., & Candès, E.J. (2019).
    Conformalized Quantile Regression. NeurIPS 2019.

Algorithm (Split-Conformal Prediction)
---------------------------------------
1. Split training data into proper training set D_tr and calibration set D_cal.
2. Fit a base forest model on D_tr.
3. Compute nonconformity scores s_i = |y_i - ŷ_i| for each (X_i, y_i) in D_cal.
4. For a new test point x:
   q_hat = (1 - α)(1 + 1/|D_cal|) quantile of {s_i}
   Prediction interval: [ŷ(x) - q_hat, ŷ(x) + q_hat]
5. Coverage guarantee: P(y ∈ C(x)) ≥ 1 - α (marginally).

Classification variant:
   Nonconformity score: s_i = 1 - ŷ_i[y_i]  (1 - predicted prob for true class)
   Prediction set: C(x) = {y : ŷ(x)[y] ≥ 1 - q_hat}
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from .random_forest import RandomForestClassifier, RandomForestRegressor


class ConformalForestRegressor(RegressorMixin, BaseEstimator):
    """Conformal Prediction Forest for regression.

    Provides prediction intervals with guaranteed marginal coverage:
        P(y ∈ [ŷ(x) - q̂, ŷ(x) + q̂]) ≥ 1 - α

    Parameters
    ----------
    base_forest : forest estimator or None
        Fitted or unfitted forest. If None, uses RandomForestRegressor.
    alpha : float, default=0.1
        Target miscoverage rate (1 - alpha = coverage level).
    calib_size : float, default=0.2
        Fraction of training data used for calibration.
    n_estimators : int, default=100
    max_depth : int or None
    random_state : int or None

    Attributes
    ----------
    q_hat_ : float
        Calibrated nonconformity quantile.
    coverage_guarantee_ : float
        Theoretical coverage = 1 - alpha.

    Examples
    --------
    >>> from forests import ConformalForestRegressor
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((300, 3))
    >>> y = 2 * X[:, 0] + rng.normal(0, 0.3, 300)
    >>> cfr = ConformalForestRegressor(alpha=0.1, random_state=0)
    >>> cfr.fit(X, y)
    ConformalForestRegressor(...)
    >>> intervals = cfr.predict_interval(X[:5])
    >>> intervals.shape
    (5, 2)
    >>> # Coverage should be ≥ 90% on calibration set
    >>> cfr.empirical_coverage_ >= 0.85
    True
    """

    def __init__(
        self,
        base_forest=None,
        alpha: float = 0.1,
        calib_size: float = 0.2,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.base_forest = base_forest
        self.alpha = alpha
        self.calib_size = calib_size
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConformalForestRegressor":
        """Fit base forest and calibrate conformal scores.

        Implements split-conformal calibration (Vovk et al. 2005).
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]
        n = X.shape[0]
        rng = np.random.default_rng(self.random_state)

        # Split into proper train / calibration
        idx = rng.permutation(n)
        n_cal = max(1, int(self.calib_size * n))
        cal_idx, tr_idx = idx[:n_cal], idx[n_cal:]

        if self.base_forest is not None:
            self.forest_ = self.base_forest
            self.forest_.fit(X[tr_idx], y[tr_idx])
        else:
            self.forest_ = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
            )
            self.forest_.fit(X[tr_idx], y[tr_idx])

        # Calibration: nonconformity scores = |y - ŷ|
        y_cal_pred = self.forest_.predict(X[cal_idx])
        scores = np.abs(y[cal_idx] - y_cal_pred)
        self.calibration_scores_ = scores

        # q̂ = (1 - α)(1 + 1/n_cal) quantile (finite-sample correction)
        q_level = min((1 - self.alpha) * (1 + 1.0 / n_cal), 1.0)
        self.q_hat_ = float(np.quantile(scores, q_level))
        self.coverage_guarantee_ = 1.0 - self.alpha

        # Empirical coverage on calibration set
        self.empirical_coverage_ = float(
            np.mean(np.abs(y[cal_idx] - y_cal_pred) <= self.q_hat_)
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return point predictions."""
        check_is_fitted(self, "forest_")
        return self.forest_.predict(X)

    def predict_interval(self, X: np.ndarray) -> np.ndarray:
        """Return prediction intervals [ŷ - q̂, ŷ + q̂].

        Parameters
        ----------
        X : (n, p) array

        Returns
        -------
        intervals : (n, 2) array
            intervals[:, 0] = lower bound
            intervals[:, 1] = upper bound
        """
        check_is_fitted(self, "q_hat_")
        X = np.asarray(X, dtype=float)
        y_hat = self.predict(X)
        return np.column_stack([y_hat - self.q_hat_, y_hat + self.q_hat_])

    def coverage_on(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute empirical coverage on a test set."""
        intervals = self.predict_interval(X)
        y = np.asarray(y, dtype=float)
        return float(np.mean((y >= intervals[:, 0]) & (y <= intervals[:, 1])))


class ConformalForestClassifier(ClassifierMixin, BaseEstimator):
    """Conformal Prediction Forest for classification.

    Returns prediction sets (not single label) with guaranteed coverage:
        P(y_true ∈ C(x)) ≥ 1 - α

    Parameters
    ----------
    base_forest : forest estimator or None
    alpha : float, default=0.1
    calib_size : float, default=0.2
    n_estimators : int, default=100
    random_state : int or None

    Examples
    --------
    >>> from forests import ConformalForestClassifier
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> cfc = ConformalForestClassifier(alpha=0.1, random_state=0)
    >>> cfc.fit(X, y)
    ConformalForestClassifier(...)
    >>> sets = cfc.predict_set(X[:5])
    >>> # Each prediction set contains the true class with ≥ 90% probability
    """

    def __init__(
        self,
        base_forest=None,
        alpha: float = 0.1,
        calib_size: float = 0.2,
        n_estimators: int = 100,
        random_state: Optional[int] = None,
    ) -> None:
        self.base_forest = base_forest
        self.alpha = alpha
        self.calib_size = calib_size
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConformalForestClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        n = X.shape[0]
        rng = np.random.default_rng(self.random_state)

        idx = rng.permutation(n)
        n_cal = max(1, int(self.calib_size * n))
        cal_idx, tr_idx = idx[:n_cal], idx[n_cal:]

        if self.base_forest is not None:
            self.forest_ = self.base_forest
            self.forest_.fit(X[tr_idx], y[tr_idx])
        else:
            self.forest_ = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
            )
            self.forest_.fit(X[tr_idx], y[tr_idx])

        # Nonconformity score: 1 - ŷ[y_true]
        proba_cal = self.forest_.predict_proba(X[cal_idx])
        label_map = {c: i for i, c in enumerate(self.forest_.classes_)}
        scores = np.array([
            1.0 - proba_cal[i, label_map[yi]]
            for i, yi in enumerate(y[cal_idx])
        ])
        self.calibration_scores_ = scores

        q_level = min((1 - self.alpha) * (1 + 1.0 / n_cal), 1.0)
        self.q_hat_ = float(np.quantile(scores, q_level))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "forest_")
        return self.forest_.predict(X)

    def predict_set(self, X: np.ndarray) -> List[np.ndarray]:
        """Return conformal prediction sets.

        Parameters
        ----------
        X : (n, p) array

        Returns
        -------
        prediction_sets : list of n arrays
            Each array contains the class labels in the prediction set.
        """
        check_is_fitted(self, "q_hat_")
        X = np.asarray(X, dtype=float)
        proba = self.forest_.predict_proba(X)
        sets = []
        for proba_i in proba:
            # Include class k if nonconformity score ≤ q̂ ↔ 1 - p_k ≤ q̂ ↔ p_k ≥ 1 - q̂
            included = self.forest_.classes_[proba_i >= (1 - self.q_hat_)]
            if len(included) == 0:
                included = self.forest_.classes_[[np.argmax(proba_i)]]
            sets.append(included)
        return sets

    def coverage_on(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute empirical coverage (fraction where y_true ∈ prediction set)."""
        sets = self.predict_set(X)
        y = np.asarray(y)
        return float(np.mean([yi in s for s, yi in zip(sets, y)]))
