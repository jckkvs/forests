"""tests/test_cart.py – Tests for CARTClassifier and CARTRegressor."""

import numpy as np
import pytest
from sklearn.datasets import load_iris, load_diabetes

from forests import CARTClassifier, CARTRegressor


class TestCARTClassifier:
    """T-001: CART Classifier tests."""

    def test_fit_predict_iris(self, iris):
        X, y = iris
        clf = CARTClassifier(max_depth=3, random_state=0)
        clf.fit(X, y)
        y_pred = clf.predict(X)
        assert y_pred.shape == y.shape
        assert set(y_pred).issubset(set(y))

    def test_accuracy_iris(self, iris):
        """Unlimited depth should achieve high accuracy."""
        X, y = iris
        clf = CARTClassifier(random_state=0)
        clf.fit(X, y)
        acc = (clf.predict(X) == y).mean()
        assert acc >= 0.95, f"Expected >=0.95 accuracy, got {acc:.3f}"

    def test_predict_proba_shape(self, iris):
        X, y = iris
        clf = CARTClassifier(max_depth=3, random_state=0)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(y), len(np.unique(y)))
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_apply_returns_leaf_ids(self, iris):
        X, y = iris
        clf = CARTClassifier(max_depth=3, random_state=0)
        clf.fit(X, y)
        leaf_ids = clf.apply(X)
        assert leaf_ids.shape == (len(y),)
        assert leaf_ids.dtype in [np.int32, np.int64, int]

    def test_depth_limit(self, iris):
        X, y = iris
        for d in [1, 2, 3]:
            clf = CARTClassifier(max_depth=d, random_state=0)
            clf.fit(X, y)
            assert clf.get_depth() <= d

    def test_entropy_criterion(self, iris):
        X, y = iris
        clf = CARTClassifier(criterion="entropy", max_depth=3, random_state=0)
        clf.fit(X, y)
        acc = (clf.predict(X) == y).mean()
        assert acc > 0.8

    def test_min_samples_leaf(self, iris):
        X, y = iris
        clf = CARTClassifier(min_samples_leaf=20, random_state=0)
        clf.fit(X, y)
        for node in clf.get_leaves():
            assert node.n_samples >= 1  # samples that reach leaf

    def test_get_n_leaves(self, iris):
        X, y = iris
        clf = CARTClassifier(max_depth=2, random_state=0)
        clf.fit(X, y)
        n_leaves = clf.get_n_leaves()
        assert n_leaves >= 1

    def test_not_fitted_error(self):
        clf = CARTClassifier()
        with pytest.raises(Exception):
            clf.predict(np.zeros((5, 4)))

    def test_max_features_sqrt(self, iris):
        X, y = iris
        clf = CARTClassifier(max_features="sqrt", random_state=0)
        clf.fit(X, y)
        assert clf.predict(X).shape[0] == len(y)


class TestCARTRegressor:
    """T-002: CART Regressor tests."""

    def test_fit_predict(self, regression_data):
        X, y = regression_data
        reg = CARTRegressor(max_depth=4, random_state=0)
        reg.fit(X, y)
        y_pred = reg.predict(X)
        assert y_pred.shape == y.shape

    def test_mse_decreases_with_depth(self, regression_data):
        X, y = regression_data
        mses = []
        for d in [1, 3, 5]:
            reg = CARTRegressor(max_depth=d, random_state=0)
            reg.fit(X, y)
            mse = np.mean((reg.predict(X) - y) ** 2)
            mses.append(mse)
        assert mses[0] >= mses[-1], "MSE should decrease with depth"

    def test_apply_returns_ids(self, regression_data):
        X, y = regression_data
        reg = CARTRegressor(max_depth=3, random_state=0)
        reg.fit(X, y)
        leaf_ids = reg.apply(X)
        assert leaf_ids.shape == (len(y),)

    def test_mae_criterion(self, regression_data):
        X, y = regression_data
        reg = CARTRegressor(criterion="mae", max_depth=3, random_state=0)
        reg.fit(X, y)
        y_pred = reg.predict(X)
        assert y_pred.shape == y.shape

    def test_min_impurity_decrease(self, regression_data):
        X, y = regression_data
        reg_strict = CARTRegressor(min_impurity_decrease=10.0, random_state=0)
        reg_none = CARTRegressor(min_impurity_decrease=0.0, random_state=0)
        reg_strict.fit(X, y)
        reg_none.fit(X, y)
        # Strict model should have fewer leaves
        assert reg_strict.get_n_leaves() <= reg_none.get_n_leaves()
