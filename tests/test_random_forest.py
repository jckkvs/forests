"""tests/test_random_forest.py – Tests for RF and ExtraTrees."""

import numpy as np
import pytest

from forests import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
)


class TestRandomForestClassifier:
    """T-010: RandomForest Classifier tests."""

    def test_score_iris(self, iris):
        X, y = iris
        clf = RandomForestClassifier(n_estimators=20, random_state=0)
        clf.fit(X, y)
        assert clf.score(X, y) > 0.9

    def test_predict_proba(self, iris):
        X, y = iris
        clf = RandomForestClassifier(n_estimators=10, random_state=0)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(y), 3)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_apply_shape(self, iris):
        X, y = iris
        clf = RandomForestClassifier(n_estimators=10, random_state=0)
        clf.fit(X, y)
        leaf_ids = clf.apply(X)
        assert leaf_ids.shape == (len(y), 10)

    def test_bootstrap_false(self, iris):
        X, y = iris
        clf = RandomForestClassifier(n_estimators=10, bootstrap=False, random_state=0)
        clf.fit(X, y)
        assert clf.score(X, y) > 0.9

    def test_n_estimators(self, iris):
        X, y = iris
        for n in [5, 10, 20]:
            clf = RandomForestClassifier(n_estimators=n, random_state=0)
            clf.fit(X, y)
            assert len(clf.estimators_) == n

    def test_predict_shape(self, binary):
        X, y = binary
        clf = RandomForestClassifier(n_estimators=10, random_state=0)
        clf.fit(X, y)
        y_pred = clf.predict(X)
        assert y_pred.shape == y.shape


class TestRandomForestRegressor:
    """T-011: RandomForest Regressor tests."""

    def test_score_diabetes(self, diabetes):
        X, y = diabetes
        reg = RandomForestRegressor(n_estimators=20, random_state=0)
        reg.fit(X, y)
        assert reg.score(X, y) > 0.7

    def test_predict_shape(self, regression_data):
        X, y = regression_data
        reg = RandomForestRegressor(n_estimators=10, random_state=0)
        reg.fit(X, y)
        y_pred = reg.predict(X)
        assert y_pred.shape == y.shape


class TestExtraTrees:
    """T-012: ExtraTrees tests."""

    def test_extra_trees_classifier(self, iris):
        X, y = iris
        clf = ExtraTreesClassifier(n_estimators=20, random_state=0)
        clf.fit(X, y)
        assert clf.score(X, y) > 0.8

    def test_extra_trees_regressor(self, diabetes):
        X, y = diabetes
        reg = ExtraTreesRegressor(n_estimators=20, random_state=0)
        reg.fit(X, y)
        assert reg.score(X, y) > 0.6

    def test_extra_trees_apply(self, iris):
        X, y = iris
        clf = ExtraTreesClassifier(n_estimators=10, random_state=0)
        clf.fit(X, y)
        leaf_ids = clf.apply(X)
        assert leaf_ids.shape == (len(y), 10)
