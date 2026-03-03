"""
forests
=======
A comprehensive Python library for decision trees and forests.

Provides:
- CART decision trees
- Random Forest & ExtraTrees
- Regularized trees (variable-reuse penalty, leaf-weight regularization)
- Monotonic & Linearity constrained forests
- Oblique, Rotation, Random-Rotation forests
- SPORF (Sparse Projection Oblique Randomer Forest)
- Regularized Greedy Forest (RGF)
- Soft (sigmoid-gated) Decision Trees/Forests
- Linear Tree / Linear Forest / Linear Boost
- RuleFit
- Bernoulli Random Forest
- Generalized Random Forest (GRF)
- Random Kernel Forest
- Gradient Boosted Trees (GBT) – from scratch
- Deep Forest (gcForest / Cascade Forest)
- Conformal Prediction Forests (with coverage guarantee)
- Totally Random Trees Embedding (unsupervised)
- Fuzzy Decision Tree
- Extras: IsolationForest, QuantileRegressionForest, RandomSurvivalForest, MondrianForest
- RF Similarity & RF Kernel
- ForestsClassifier, ForestsRegressor: unified interface for all of the above
"""

from .cart import CARTClassifier, CARTRegressor
from .random_forest import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
)
from .regularized import VariablePenaltyForest, LeafWeightRegularizedForest
from .constrained import MonotonicConstrainedForest, LinearConstrainedForest
from .oblique import ObliqueForest, RotationForest, RandomRotationForest
from .sporf import SPORFClassifier, SPORFRegressor
from .rgf import RegularizedGreedyForest
from .soft_tree import SoftDecisionTree, SoftDecisionForest
from .linear_tree import LinearTree, LinearForest, LinearBoost
from .rulefit import RuleFit
from .bernoulli_rf import BernoulliRandomForest
from .grf import GeneralizedRandomForest, QuantileForest, CausalForest
from .kernel_forest import RandomKernelForest
from .extras import (
    IsolationForest,
    QuantileRegressionForest,
    RandomSurvivalForest,
    MondrianForest,
)
from .similarity import RFSimilarity, RFKernel
from .boosting import GradientBoostedRegressor, GradientBoostedClassifier
from .deep_forest import DeepForest
from .conformal import ConformalForestRegressor, ConformalForestClassifier
from .embedding import TotallyRandomTreesEmbedding, FuzzyDecisionTree
from .builder import ForestsClassifier, ForestsRegressor, IncompatibleOptionsWarning

__version__ = "0.1.0"
__all__ = [
    # CART
    "CARTClassifier",
    "CARTRegressor",
    # Random Forest
    "RandomForestClassifier",
    "RandomForestRegressor",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
    # Regularized
    "VariablePenaltyForest",
    "LeafWeightRegularizedForest",
    # Constrained
    "MonotonicConstrainedForest",
    "LinearConstrainedForest",
    # Oblique / Rotation
    "ObliqueForest",
    "RotationForest",
    "RandomRotationForest",
    # SPORF
    "SPORFClassifier",
    "SPORFRegressor",
    # RGF
    "RegularizedGreedyForest",
    # Soft Tree
    "SoftDecisionTree",
    "SoftDecisionForest",
    # Linear Tree
    "LinearTree",
    "LinearForest",
    "LinearBoost",
    # RuleFit
    "RuleFit",
    # Bernoulli RF
    "BernoulliRandomForest",
    # GRF
    "GeneralizedRandomForest",
    "QuantileForest",
    "CausalForest",
    # Kernel Forest
    "RandomKernelForest",
    # Extras
    "IsolationForest",
    "QuantileRegressionForest",
    "RandomSurvivalForest",
    "MondrianForest",
    # Gradient Boosted Trees
    "GradientBoostedRegressor",
    "GradientBoostedClassifier",
    # Deep Forest (gcForest)
    "DeepForest",
    # Conformal Prediction
    "ConformalForestRegressor",
    "ConformalForestClassifier",
    # Unsupervised embedding & Fuzzy
    "TotallyRandomTreesEmbedding",
    "FuzzyDecisionTree",
    # Similarity / Kernel
    "RFSimilarity",
    "RFKernel",
    # Builder
    "ForestsClassifier",
    "ForestsRegressor",
    "IncompatibleOptionsWarning",
]
