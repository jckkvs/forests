import numpy as np
import pytest
from forests.rgf import RegularizedGreedyForest
from forests.soft_tree import SoftDecisionTree, SoftDecisionForest
from forests.linear_tree import LinearTree, LinearForest, LinearBoost
from forests.rulefit import RuleFit
from forests.bernoulli_rf import BernoulliRandomForest

def test_rgf(regression_data):
    X, y = regression_data
    model = RegularizedGreedyForest(n_estimators=5, reg_lambda=0.1, random_state=42)
    model.fit(X, y)
    assert model.score(X, y) > 0.0

def test_soft_decision_tree_classifier(iris):
    X, y = iris
    X = X / (X.max(axis=0) + 1e-8)  # Normalization for soft tree
    model = SoftDecisionTree(max_depth=2, n_epochs=100, learning_rate=0.05, n_classes=3, task="classification", random_state=42)
    model.fit(X, y)
    assert model.score(X, y) > 0.5

def test_soft_decision_forest_classifier(iris):
    X, y = iris
    X = X / (X.max(axis=0) + 1e-8)
    model = SoftDecisionForest(n_estimators=2, max_depth=2, n_epochs=50, learning_rate=0.05, n_classes=3, task="classification", bootstrap=False, random_state=42)
    model.fit(X, y)
    assert model.score(X, y) > 0.5

def test_linear_tree_regressor(regression_data):
    X, y = regression_data
    model = LinearTree(max_depth=2, random_state=42)
    model.fit(X, y)
    assert model.score(X, y) > 0.5

def test_linear_forest_regressor(regression_data):
    X, y = regression_data
    model = LinearForest(n_estimators=3, max_depth=2, random_state=42)
    model.fit(X, y)
    assert model.score(X, y) > 0.5

def test_linear_boost_regressor(regression_data):
    X, y = regression_data
    model = LinearBoost(n_estimators=5, max_depth=2, random_state=42)
    model.fit(X, y)
    assert model.score(X, y) > 0.4  # Lower tolerance for small ensemble

def test_rulefit_regressor(diabetes):
    X, y = diabetes
    model = RuleFit(n_estimators=5, alpha=0.1, random_state=42)
    model.fit(X, y)
    assert model.score(X, y) > 0.0
    
    # Test rule extraction
    rules = model.get_rules(top_n=3)
    assert len(rules) <= 3

def test_bernoulli_rf_classifier(iris):
    X, y = iris
    model = BernoulliRandomForest(n_estimators=5, feature_prob=0.5, random_state=42)
    model.fit(X, y)
    assert model.score(X, y) > 0.8
