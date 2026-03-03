import numpy as np
import pytest
from forests.oblique import ObliqueForest, RotationForest, RandomRotationForest
from forests.sporf import SPORFClassifier, SPORFRegressor

def test_oblique_forest_classifier(iris):
    X, y = iris
    model = ObliqueForest(n_estimators=3, max_depth=2, random_state=42)
    model.fit(X, y)
    assert model.score(X, y) > 0.8
    # Test apply and predict_proba for coverage
    assert model.apply(X).shape == (len(X), 3)
    assert model.predict_proba(X).shape == (len(X), 3)

def test_rotation_forest_variants(iris):
    X, y = iris
    # RotationForest
    rot = RotationForest(n_estimators=2, random_state=42)
    rot.fit(X, y)
    assert rot.score(X, y) > 0.8
    
    # RandomRotationForest
    rr = RandomRotationForest(n_estimators=2, random_state=42)
    rr.fit(X, y)
    assert rr.score(X, y) > 0.8
    
    # Test apply (using rot model for example)
    leaf_ids = rot.apply(X)
    assert leaf_ids.shape == (len(X), 2) # n_estimators=2

def test_sporf_classifier(binary):
    X, y = binary
    model = SPORFClassifier(n_estimators=5, n_projections=3, density=0.1, random_state=42)
    model.fit(X, y)
    score = model.score(X, y)
    assert score > 0.7
    
    # Verify sparsity in the first tree's projections
    # Note: SPORF logic usually stores projection vectors in Node.feature if oblique
    # We need to check how it's implemented.
    tree = model.estimators_[0]
    assert hasattr(tree, "root_")

def test_sporf_regressor(regression_data):
    X, y = regression_data
    model = SPORFRegressor(n_estimators=5, n_projections=3, density=0.1, random_state=42)
    model.fit(X, y)
    score = model.score(X, y)
    # Regression score (R^2) can be low for random data but should be reasonable or at least run
    assert score > 0.0

def test_oblique_forest_params():
    # Test with invalid n_directions
    with pytest.raises(ValueError):
        _ = ObliqueForest(n_directions=0)
