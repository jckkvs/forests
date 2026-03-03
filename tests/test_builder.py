"""tests/test_builder.py – Tests for ForestBuilder unified interface."""

import warnings
import numpy as np
import pytest

from forests import ForestBuilder, IncompatibleOptionsWarning


class TestForestBuilderClassification:
    """T-060: ForestBuilder for classification tasks."""

    def test_default_rf(self, iris):
        X, y = iris
        fb = ForestBuilder(n_estimators=10, random_state=0, task="classification")
        fb.fit(X, y)
        assert fb.score(X, y) > 0.9
        assert fb.model_type_ == "RandomForestClassifier"

    def test_rotation_forest(self, iris):
        X, y = iris
        fb = ForestBuilder(n_estimators=10, rotation=True, random_state=0, task="classification")
        fb.fit(X, y)
        assert fb.score(X, y) > 0.8

    def test_random_rotation_forest(self, iris):
        X, y = iris
        fb = ForestBuilder(n_estimators=10, random_rotation=True, random_state=0, task="classification")
        fb.fit(X, y)
        assert fb.score(X, y) > 0.7

    def test_oblique_forest(self, iris):
        X, y = iris
        fb = ForestBuilder(n_estimators=10, split_type="oblique", random_state=0, task="classification")
        fb.fit(X, y)
        assert fb.score(X, y) > 0.7

    def test_sporf(self, iris):
        X, y = iris
        fb = ForestBuilder(n_estimators=10, split_type="oblique",
                           sparse_projection=True, random_state=0, task="classification")
        fb.fit(X, y)
        assert fb.score(X, y) > 0.7

    def test_variable_penalty(self, iris):
        X, y = iris
        fb = ForestBuilder(n_estimators=10, variable_reuse_penalty=0.05, random_state=0, task="classification")
        fb.fit(X, y)
        assert fb.score(X, y) > 0.8

    def test_bernoulli_bootstrap(self, iris):
        X, y = iris
        fb = ForestBuilder(n_estimators=10, bootstrap="bernoulli",
                           feature_prob=0.5, random_state=0, task="classification")
        fb.fit(X, y)
        assert fb.score(X, y) > 0.7

    def test_predict_proba_available(self, iris):
        X, y = iris
        fb = ForestBuilder(n_estimators=10, random_state=0, task="classification")
        fb.fit(X, y)
        proba = fb.predict_proba(X)
        assert proba.shape == (len(y), 3)

    def test_apply_available(self, iris):
        X, y = iris
        fb = ForestBuilder(n_estimators=10, random_state=0, task="classification")
        fb.fit(X, y)
        leaf_ids = fb.apply(X)
        assert leaf_ids.shape == (len(y), 10)


class TestForestBuilderRegression:
    """T-061: ForestBuilder for regression tasks."""

    def test_default_rfr(self, regression_data):
        X, y = regression_data
        fb = ForestBuilder(n_estimators=10, random_state=0, task="regression")
        fb.fit(X, y)
        assert fb.score(X, y) > 0.7
        assert fb.model_type_ == "RandomForestRegressor"

    def test_monotone_constraint(self, regression_data):
        X, y = regression_data
        fb = ForestBuilder(n_estimators=10, monotone_constraints={0: 1},
                           random_state=0, task="regression")
        fb.fit(X, y)
        assert fb.score(X, y) > 0.3

    def test_linearity_constraint(self, regression_data):
        X, y = regression_data
        fb = ForestBuilder(n_estimators=10, linear_features=[0],
                           linearity_lambda=0.8, random_state=0, task="regression")
        fb.fit(X, y)
        assert fb.score(X, y) > 0.3

    def test_leaf_l2_regularization(self, regression_data):
        X, y = regression_data
        fb = ForestBuilder(n_estimators=10, leaf_regularization="l2",
                           leaf_reg_alpha=0.1, random_state=0, task="regression")
        fb.fit(X, y)
        assert fb.score(X, y) > 0.3

    def test_linear_leaf(self, regression_data):
        X, y = regression_data
        fb = ForestBuilder(n_estimators=10, linear_leaf=True,
                           random_state=0, task="regression")
        fb.fit(X, y)
        assert fb.predict(X).shape == y.shape


class TestIncompatibilityWarnings:
    """T-062: Incompatible option warnings."""

    def test_rotation_oblique_warning(self, iris):
        X, y = iris
        with pytest.warns(IncompatibleOptionsWarning):
            fb = ForestBuilder(n_estimators=5, rotation=True,
                               split_type="oblique", random_state=0, task="classification")
            fb.fit(X, y)

    def test_rotation_random_rotation_warning(self, iris):
        X, y = iris
        with pytest.warns(IncompatibleOptionsWarning):
            fb = ForestBuilder(n_estimators=5, rotation=True,
                               random_rotation=True, random_state=0, task="classification")
            fb.fit(X, y)

    def test_soft_tree_linear_leaf_warning(self, iris):
        X, y = iris
        with pytest.warns(IncompatibleOptionsWarning):
            fb = ForestBuilder(n_estimators=3, soft_tree=True,
                               linear_leaf=True, random_state=0, task="classification")
            fb.fit(X, y)

    def test_rotation_sparse_projection_warning(self, iris):
        X, y = iris
        with pytest.warns(IncompatibleOptionsWarning):
            fb = ForestBuilder(n_estimators=5, rotation=True,
                               sparse_projection=True, split_type="oblique",
                               random_state=0, task="classification")
            fb.fit(X, y)
