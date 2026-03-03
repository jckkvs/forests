"""tests/test_builder.py – Tests for ForestsClassifier unified interface."""

import warnings
import numpy as np
import pytest

from forests import ForestsClassifier, ForestsRegressor, IncompatibleOptionsWarning


class TestForestsClassifier:
    """T-060: ForestsClassifier for classification tasks."""

    def test_default_rf(self, iris):
        X, y = iris
        fb = ForestsClassifier(n_estimators=10, random_state=0)
        fb.fit(X, y)
        assert fb.score(X, y) > 0.9
        assert fb.model_type_ == "RandomForestClassifier"

    def test_rotation_forest(self, iris):
        X, y = iris
        fb = ForestsClassifier(n_estimators=10, rotation=True, random_state=0)
        fb.fit(X, y)
        assert fb.score(X, y) > 0.8

    def test_random_rotation_forest(self, iris):
        X, y = iris
        fb = ForestsClassifier(n_estimators=10, random_rotation=True, random_state=0)
        fb.fit(X, y)
        assert fb.score(X, y) > 0.7

    def test_oblique_forest(self, iris):
        X, y = iris
        fb = ForestsClassifier(n_estimators=10, split_type="oblique", random_state=0)
        fb.fit(X, y)
        assert fb.score(X, y) > 0.7

    def test_sporf(self, iris):
        X, y = iris
        fb = ForestsClassifier(n_estimators=10, split_type="oblique",
                           sparse_projection=True, random_state=0)
        fb.fit(X, y)
        assert fb.score(X, y) > 0.7

    def test_variable_penalty(self, iris):
        X, y = iris
        fb = ForestsClassifier(n_estimators=10, variable_reuse_penalty=0.05, random_state=0)
        fb.fit(X, y)
        assert fb.score(X, y) > 0.8

    def test_bernoulli_bootstrap(self, iris):
        X, y = iris
        fb = ForestsClassifier(n_estimators=10, bootstrap="bernoulli",
                           feature_prob=0.5, random_state=0)
        fb.fit(X, y)
        assert fb.score(X, y) > 0.7

    def test_predict_proba_available(self, iris):
        X, y = iris
        fb = ForestsClassifier(n_estimators=10, random_state=0)
        fb.fit(X, y)
        proba = fb.predict_proba(X)
        assert proba.shape == (len(y), 3)

    def test_apply_available(self, iris):
        X, y = iris
        fb = ForestsClassifier(n_estimators=10, random_state=0)
        fb.fit(X, y)
        leaf_ids = fb.apply(X)
        assert leaf_ids.shape == (len(y), 10)

    def test_extra_trees(self, iris):
        X, y = iris
        fb = ForestsClassifier(n_estimators=10, extra_trees=True, random_state=0)
        fb.fit(X, y)
        assert fb.score(X, y) > 0.8
        assert fb.model_type_ == "ExtraTreesClassifier"

    def test_boosting(self, iris):
        X, y = iris
        fb = ForestsClassifier(n_estimators=10, boosting=True, random_state=0)
        fb.fit(X, y)
        assert fb.score(X, y) > 0.8
        assert fb.model_type_ == "GradientBoostedClassifier"

    def test_rulefit(self, iris):
        X, y = iris
        fb = ForestsClassifier(n_estimators=10, rulefit=True, random_state=0)
        fb.fit(X, y)
        assert fb.score(X, y) > 0.8
        assert fb.model_type_ == "RuleFit"

    def test_rgf(self, iris):
        X, y = iris
        fb = ForestsClassifier(n_estimators=10, rgf=True, random_state=0)
        fb.fit(X, y)
        assert fb.model_type_ == "RegularizedGreedyForest"

    def test_deep_forest(self, iris):
        X, y = iris
        fb = ForestsClassifier(n_estimators=10, deep_forest=True, max_depth=2, random_state=0)
        fb.fit(X, y)
        assert fb.score(X, y) > 0.8
        assert fb.model_type_ == "DeepForest"

    def test_conformal(self, iris):
        X, y = iris
        fb = ForestsClassifier(n_estimators=10, conformal=True, random_state=0)
        fb.fit(X, y)
        assert fb.score(X, y) > 0.8
        assert fb.model_type_ == "ConformalForestClassifier"

    def test_mondrian(self, iris):
        X, y = iris
        fb = ForestsClassifier(n_estimators=10, mondrian=True, random_state=0)
        fb.fit(X, y)
        assert fb.score(X, y) > 0.8
        assert fb.model_type_ == "MondrianForest"

    def test_isolation(self, iris):
        X, y = iris
        fb = ForestsClassifier(n_estimators=10, isolation=True, random_state=0)
        fb.fit(X)
        # score is not applicable, just verify it runs
        assert fb.model_type_ == "IsolationForest"


class TestForestsRegressor:
    """T-061: ForestsClassifier for regression tasks."""

    def test_default_rfr(self, regression_data):
        X, y = regression_data
        fb = ForestsRegressor(n_estimators=10, random_state=0)
        fb.fit(X, y)
        assert fb.score(X, y) > 0.7
        assert fb.model_type_ == "RandomForestRegressor"

    def test_monotone_constraint(self, regression_data):
        X, y = regression_data
        fb = ForestsRegressor(n_estimators=10, monotone_constraints={0: 1},
                           random_state=0)
        fb.fit(X, y)
        assert fb.score(X, y) > 0.3

    def test_linearity_constraint(self, regression_data):
        X, y = regression_data
        fb = ForestsRegressor(n_estimators=10, linear_features=[0],
                           linearity_lambda=0.8, random_state=0)
        fb.fit(X, y)
        assert fb.score(X, y) > 0.3

    def test_leaf_l2_regularization(self, regression_data):
        X, y = regression_data
        fb = ForestsRegressor(n_estimators=10, leaf_regularization="l2",
                           leaf_reg_alpha=0.1, random_state=0)
        fb.fit(X, y)
        assert fb.score(X, y) > 0.3

    def test_linear_leaf(self, regression_data):
        X, y = regression_data
        fb = ForestsRegressor(n_estimators=10, linear_leaf=True,
                           random_state=0)
        fb.fit(X, y)
        assert fb.predict(X).shape == y.shape

    def test_extra_trees(self, regression_data):
        X, y = regression_data
        fb = ForestsRegressor(n_estimators=10, extra_trees=True, random_state=0)
        fb.fit(X, y)
        assert fb.score(X, y) > 0.5
        assert fb.model_type_ == "ExtraTreesRegressor"

    def test_boosting(self, regression_data):
        X, y = regression_data
        fb = ForestsRegressor(n_estimators=10, boosting=True, random_state=0)
        fb.fit(X, y)
        assert fb.score(X, y) > 0.3
        assert fb.model_type_ == "GradientBoostedRegressor"

    def test_rulefit(self, regression_data):
        X, y = regression_data
        fb = ForestsRegressor(n_estimators=10, rulefit=True, random_state=0)
        fb.fit(X, y)
        assert fb.score(X, y) > 0.5
        assert fb.model_type_ == "RuleFit"

    def test_rgf(self, regression_data):
        X, y = regression_data
        fb = ForestsRegressor(n_estimators=10, rgf=True, random_state=0)
        fb.fit(X, y)
        assert hasattr(fb.model_, "predict")
        assert fb.model_type_ == "RegularizedGreedyForest"

    def test_conformal(self, regression_data):
        X, y = regression_data
        fb = ForestsRegressor(n_estimators=10, conformal=True, random_state=0)
        fb.fit(X, y)
        assert fb.score(X, y) > 0.3
        assert fb.model_type_ == "ConformalForestRegressor"

    def test_task_incompatibilities(self, regression_data):
        X, y = regression_data
        with pytest.raises(ValueError, match="DeepForest does not support regression tasks."):
            fb = ForestsRegressor(n_estimators=10, deep_forest=True, random_state=0)
            fb.fit(X, y)

        with pytest.raises(ValueError, match="MondrianForest does not support regression tasks."):
            fb = ForestsRegressor(n_estimators=10, mondrian=True, random_state=0)
            fb.fit(X, y)

    def test_random_survival(self, regression_data):
        X, y = regression_data
        # fake survival e array
        rng = np.random.default_rng(0)
        e = rng.binomial(1, 0.7, len(y))
        
        fb = ForestsRegressor(n_estimators=10, survival=True, random_state=0)
        fb.fit(X, y, e=e)
        
        assert hasattr(fb.model_, "predict")
        assert fb.model_type_ == "RandomSurvivalForest"


class TestIncompatibilityWarnings:
    """T-062: Incompatible option warnings."""

    def test_rotation_oblique_warning(self, iris):
        X, y = iris
        with pytest.warns(IncompatibleOptionsWarning):
            fb = ForestsClassifier(n_estimators=5, rotation=True,
                               split_type="oblique", random_state=0)
            fb.fit(X, y)

    def test_rotation_random_rotation_warning(self, iris):
        X, y = iris
        with pytest.warns(IncompatibleOptionsWarning):
            fb = ForestsClassifier(n_estimators=5, rotation=True,
                               random_rotation=True, random_state=0)
            fb.fit(X, y)

    def test_soft_tree_linear_leaf_warning(self, iris):
        X, y = iris
        with pytest.warns(IncompatibleOptionsWarning):
            fb = ForestsClassifier(n_estimators=3, soft_tree=True,
                               linear_leaf=True, random_state=0)
            fb.fit(X, y)

    def test_rotation_sparse_projection_warning(self, iris):
        X, y = iris
        with pytest.warns(IncompatibleOptionsWarning):
            fb = ForestsClassifier(n_estimators=5, rotation=True,
                               sparse_projection=True, split_type="oblique",
                               random_state=0)
            fb.fit(X, y)
