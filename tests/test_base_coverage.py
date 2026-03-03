import numpy as np
import pytest
from sklearn.utils.validation import check_is_fitted
from forests.base import BaseTree, Node, gini_impurity, entropy_impurity, mse_impurity, mae_impurity
from forests.cart import CARTRegressor, CARTClassifier
from forests.random_forest import RandomForestClassifier

def test_impurity_corner_cases():
    # Empty arrays
    y_empty = np.array([])
    assert gini_impurity(y_empty, 2) == 0.0
    assert entropy_impurity(y_empty, 2) == 0.0
    assert mse_impurity(y_empty) == 0.0
    assert mae_impurity(y_empty) == 0.0

def test_base_tree_errors():
    tree = CARTRegressor()
    # Not fitted errors
    with pytest.raises(RuntimeError, match="not fitted"):
        tree.apply(np.zeros((5, 2)))
    with pytest.raises(RuntimeError, match="not fitted"):
        tree.get_depth()
    with pytest.raises(RuntimeError, match="not fitted"):
        tree.get_n_leaves()
    with pytest.raises(RuntimeError, match="not fitted"):
        tree.get_leaves()

def test_max_features_variations():
    X = np.random.rand(20, 5)
    y = np.random.rand(20)
    
    # max_features float
    tree = CARTRegressor(max_features=0.5, random_state=42)
    tree.fit(X, y)
    assert tree.n_features_in_ == 5
    
    # max_features log2
    tree = CARTRegressor(max_features="log2", random_state=42)
    tree.fit(X, y)
    
    # max_features invalid
    tree = CARTRegressor(max_features="invalid_mode")
    with pytest.raises(ValueError, match="Unknown max_features"):
        tree.fit(X, y)

def test_base_forest_max_samples():
    X = np.random.rand(20, 5)
    y = (np.random.rand(20) > 0.5).astype(int)
    
    # bootstrap=True, max_samples=0.5
    rf = RandomForestClassifier(n_estimators=2, bootstrap=True, max_samples=0.5, random_state=42)
    rf.fit(X, y)
    
    # max_samples int
    rf = RandomForestClassifier(n_estimators=2, bootstrap=True, max_samples=5, random_state=42)
    rf.fit(X, y)

def test_base_forest_not_fitted():
    rf = RandomForestClassifier()
    with pytest.raises(check_is_fitted.__globals__['NotFittedError'] if 'NotFittedError' in check_is_fitted.__globals__ else Exception):
        rf.apply(np.zeros((5,2)))
