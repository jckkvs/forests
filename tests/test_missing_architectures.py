import numpy as np
import pytest
from forests.boosting import GradientBoostedRegressor, GradientBoostedClassifier
from forests.deep_forest import DeepForest
from forests.embedding import TotallyRandomTreesEmbedding, FuzzyDecisionTree
from forests.similarity import RFSimilarity, RFKernel
from forests.random_forest import RandomForestClassifier

def test_boosting_variants(regression_data):
    X, y = regression_data
    # GradientBoostedRegressor
    gb = GradientBoostedRegressor(n_estimators=5, max_depth=2, random_state=42)
    gb.fit(X, y)
    assert gb.score(X, y) > 0.0
    
    # GradientBoostedClassifier
    from sklearn.datasets import load_iris
    Xi, yi = load_iris(return_X_y=True)
    gbc = GradientBoostedClassifier(n_estimators=5, max_depth=2, random_state=42)
    gbc.fit(Xi, yi)
    assert gbc.score(Xi, yi) > 0.8

def test_deep_forest(iris):
    X, y = iris
    # DeepForest requires normalization as per docstring hint
    X_norm = (X - X.min(0)) / (X.max(0) - X.min(0) + 1e-8)
    model = DeepForest(n_estimators_per_forest=5, n_forests_per_level=2, max_levels=2, random_state=42)
    model.fit(X_norm, y)
    assert model.score(X_norm, y) > 0.8

def test_embeddings_and_fuzzy(iris):
    X, y = iris
    # TotallyRandomTreesEmbedding
    rte = TotallyRandomTreesEmbedding(n_estimators=5, max_depth=2, random_state=42)
    rte.fit(X)
    Z = rte.transform(X)
    assert Z.shape[0] == len(X)
    
    # FuzzyDecisionTree
    from forests.embedding import FuzzyDecisionTree
    fdt = FuzzyDecisionTree(max_depth=3, beta=0.3, random_state=42)
    # Binary iris for simplicity of regression test
    y_reg = (y > 0).astype(float)
    fdt.fit(X, y_reg)
    preds = fdt.predict(X)
    assert preds.shape == (len(X),)

def test_similarity_and_kernel(iris):
    X, y = iris
    rf = RandomForestClassifier(n_estimators=5, random_state=42).fit(X, y)
    
    # RFSimilarity: fit_transform and transform
    sim = RFSimilarity(rf)
    S = sim.fit_transform(X[:10])
    assert S.shape == (10, 10)
    
    S_new = sim.transform(X[10:15])
    assert S_new.shape == (5, 10)
    
    # RFKernel variants
    for mode in ["cosine", "rbf", "raw"]:
        kernel = RFKernel(rf, mode=mode, gamma=0.5)
        K = kernel.fit_transform(X[:10])
        assert K.shape == (10, 10)
        
        K_new = kernel.transform(X[10:15])
        assert K_new.shape == (5, 10)
        
        # Test get_kernel_matrix
        Km = kernel.get_kernel_matrix(X[:3], X[5:10])
        assert Km.shape == (3, 5)

    # Error case for unknown mode
    kernel_fail = RFKernel(rf, mode="invalid")
    with pytest.raises(ValueError):
        kernel_fail.fit(X[:5])
