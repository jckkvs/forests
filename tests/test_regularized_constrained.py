import numpy as np
import pytest
from forests.regularized import VariablePenaltyForest, LeafWeightRegularizedForest
from forests.constrained import MonotonicConstrainedForest, LinearConstrainedForest

def test_variable_penalty_forest(iris):
    X, y = iris
    # Test that it runs
    model = VariablePenaltyForest(n_estimators=5, reuse_alpha=0.5, random_state=42)
    model.fit(X, y)
    assert model.score(X, y) > 0.8

def test_leaf_weight_regularized_l1(diabetes):
    X, y = diabetes
    model = LeafWeightRegularizedForest(n_estimators=5, leaf_reg="l1", alpha=0.1, random_state=42)
    model.fit(X, y)
    assert model.score(X, y) > 0.0

def test_leaf_weight_regularized_l2(diabetes):
    X, y = diabetes
    model = LeafWeightRegularizedForest(n_estimators=5, leaf_reg="l2", alpha=0.1, random_state=42)
    model.fit(X, y)
    assert model.score(X, y) > 0.0

def test_monotonic_constrained_forest():
    # Simple data where y is monotone increasing with x
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = X.ravel() * 2 + np.random.normal(0, 0.1, 100)
    
    # Positive constraint on feature 0
    model = MonotonicConstrainedForest(
        n_estimators=5, 
        monotone_constraints={0: 1}, 
        random_state=42
    )
    model.fit(X, y)
    
    # Predict on sorted X
    X_test = np.linspace(0, 10, 50).reshape(-1, 1)
    y_pred = model.predict(X_test)
    
    # Check if predictions are monotonically non-decreasing
    assert np.all(np.diff(y_pred) >= -1e-10)

def test_linear_constrained_forest(regression_data):
    X, y = regression_data
    model = LinearConstrainedForest(
        n_estimators=5,
        linear_features=[0],
        linearity_lambda=0.5,
        random_state=42
    )
    model.fit(X, y)
    assert model.score(X, y) > 0.0
