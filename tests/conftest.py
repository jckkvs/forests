"""
conftest.py – shared fixtures for tests/
"""

import numpy as np
import pytest
from sklearn.datasets import load_iris, load_diabetes, make_classification


@pytest.fixture
def iris():
    X, y = load_iris(return_X_y=True)
    return X.astype(float), y


@pytest.fixture
def diabetes():
    X, y = load_diabetes(return_X_y=True)
    return X.astype(float), y.astype(float)


@pytest.fixture
def binary():
    X, y = make_classification(
        n_samples=200, n_features=10, random_state=42
    )
    return X.astype(float), y.astype(int)


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(42)
    X = rng.random((200, 5))
    y = 2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] + rng.normal(0, 0.1, 200)
    return X, y


@pytest.fixture
def survival_data():
    rng = np.random.default_rng(42)
    X = rng.random((100, 3))
    t = rng.exponential(2, 100)
    e = rng.binomial(1, 0.7, 100)
    return X, t, e
