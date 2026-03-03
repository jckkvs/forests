import numpy as np
import pytest
from forests.extras import IsolationForest, QuantileRegressionForest, RandomSurvivalForest, MondrianForest

def test_isolation_forest():
    rng = np.random.default_rng(42)
    # Normal data
    X_train = rng.standard_normal((100, 2))
    # Outliers
    X_test = np.vstack([
        rng.standard_normal((10, 2)),
        rng.uniform(5, 10, (5, 2)) # outliers
    ])
    
    model = IsolationForest(n_estimators=50, contamination=0.1, random_state=42)
    model.fit(X_train)
    
    scores = model.score_samples(X_test)
    assert len(scores) == 15
    preds = model.predict(X_test)
    # Outliers should generally have lower path lengths -> higher anomaly scores
    # In this implementation, higher score means more anomalous.
    assert preds.shape == (15,)

def test_quantile_regression_forest(regression_data):
    X, y = regression_data
    model = QuantileRegressionForest(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    q50 = model.predict(X, quantile=0.5)
    q90 = model.predict(X, quantile=0.9)
    assert q50.shape == (len(X),)
    # On training data, q90 should generally be >= q50
    assert np.mean(q90 >= q50) > 0.5

def test_random_survival_forest(survival_data):
    X, t, e = survival_data
    model = RandomSurvivalForest(n_estimators=5, max_depth=3, random_state=42)
    model.fit(X, t, e)
    
    # Check cumulative hazard
    ch = model.predict_cumhazard(X[:2])
    assert ch.ndim == 2
    assert ch.shape[0] == 2
    
    # Check proxy prediction
    preds = model.predict(X[:5])
    assert preds.shape == (5,)

def test_mondrian_forest(iris):
    X, y = iris
    model = MondrianForest(n_estimators=5, n_classes=3, lifetime=1.0, random_state=42)
    model.fit(X, y)
    
    assert model.score(X, y) > 0.7
    proba = model.predict_proba(X[:3])
    assert proba.shape == (3, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
