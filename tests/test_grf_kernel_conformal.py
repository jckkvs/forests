import numpy as np
import pytest
from forests.grf import GeneralizedRandomForest, QuantileForest, CausalForest
from forests.kernel_forest import RandomKernelForest
from forests.conformal import ConformalForestRegressor, ConformalForestClassifier

def test_grf_regressor(regression_data):
    X, y = regression_data
    model = GeneralizedRandomForest(n_estimators=5, random_state=42)
    model.fit(X, y)
    assert model.score(X, y) > 0.0

def test_quantile_forest(regression_data):
    X, y = regression_data
    model = QuantileForest(n_estimators=5, quantile=0.9, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)

def test_causal_forest(survival_data):
    X, t, e = survival_data
    # Use binary W for treatment
    W = e # Just reuse e as a treatment indicator
    y = t
    model = CausalForest(n_estimators=5, random_state=42)
    model.fit(X, y, W)
    cate = model.predict(X)
    assert cate.shape == (len(X),)
    assert isinstance(model.ate(), float)

def test_random_kernel_forest(iris):
    X, y = iris
    model = RandomKernelForest(n_estimators=5, n_rff=10, random_state=42)
    model.fit(X, y)
    assert model.score(X, y) > 0.7

def test_conformal_forest_regressor(regression_data):
    X, y = regression_data
    model = ConformalForestRegressor(n_estimators=5, alpha=0.1, random_state=42)
    model.fit(X, y)
    intervals = model.predict_interval(X)
    assert intervals.shape == (len(X), 2)
    cov = model.coverage_on(X, y)
    assert 0.0 <= cov <= 1.0

def test_conformal_forest_classifier(iris):
    X, y = iris
    model = ConformalForestClassifier(n_estimators=5, alpha=0.1, random_state=42)
    model.fit(X, y)
    sets = model.predict_set(X)
    assert len(sets) == len(X)
    cov = model.coverage_on(X, y)
    assert 0.0 <= cov <= 1.0
