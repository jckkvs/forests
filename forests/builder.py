"""
forests.builder
===============
ForestBuilder: Unified interface for all forest variants.

Allows users to compose a forest by specifying algorithm components as arguments.
Incompatible combinations trigger IncompatibleOptionsWarning.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator


class IncompatibleOptionsWarning(UserWarning):
    """Warning issued when incompatible ForestBuilder options are detected."""


# ---------------------------------------------------------------------------
# Compatibility rules
# ---------------------------------------------------------------------------

_INCOMPATIBILITY_RULES: List[Dict[str, Any]] = [
    {
        "conditions": {"rotation": True, "split_type": "oblique"},
        "message": (
            "rotation=True と split_type='oblique' は機能が重複しています。"
            "RotationForestではなくObliqueForestが使用されます。"
            "rotation=Falseを推奨します。"
        ),
    },
    {
        "conditions": {"random_rotation": True, "rotation": True},
        "message": (
            "rotation=True と random_rotation=True の両方が指定されました。"
            "PCAベースのRotationForest (rotation=True) を優先します。"
        ),
    },
    {
        "conditions": {"soft_tree": True, "linear_leaf": True},
        "message": (
            "soft_tree=True と linear_leaf=True は学習方法が競合しています。"
            "soft_tree=True (シグモイドゲート) が優先されます。"
        ),
    },
    {
        "conditions": {"soft_tree": True, "split_type": "oblique"},
        "message": (
            "soft_tree=True と split_type='oblique' を同時に指定しています。"
            "soft_tree=True (ニューラル的ゲート) が優先されます。"
        ),
    },
    {
        "conditions": {"generalized_target": "causal", "soft_tree": True},
        "message": (
            "generalized_target='causal' と soft_tree=True は学習目的が異なります。"
            "CausalForestでは木は通常CARTベースで構築します。"
        ),
    },
    {
        "conditions": {"rotation": True, "sparse_projection": True},
        "message": (
            "rotation=True と sparse_projection=True は変換が競合します。"
            "RotationForest (PCA変換) が優先されます。sparse_projection=Falseを推奨します。"
        ),
    },
]


def _check_incompatibilities(**params) -> None:
    """Check for incompatible option combinations and warn."""
    for rule in _INCOMPATIBILITY_RULES:
        conditions = rule["conditions"]
        match = all(params.get(k) == v for k, v in conditions.items())
        if match:
            warnings.warn(
                f"[ForestBuilder 互換性警告] {rule['message']}",
                IncompatibleOptionsWarning,
                stacklevel=3,
            )


# ---------------------------------------------------------------------------
# ForestBuilder
# ---------------------------------------------------------------------------

class ForestBuilder(BaseEstimator):
    """Unified Forest Builder with composable options.

    Compose any forest model by specifying algorithm components as arguments.
    Incompatible combinations trigger IncompatibleOptionsWarning and the
    first listed option takes priority.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees.

    --- Split type ---
    split_type : {"axis", "oblique", "kernel", "soft"}, default="axis"
        Type of node split:
        - "axis"    : standard axis-aligned split (CART)
        - "oblique" : linear combination split (ObliqueForest / SPORF)
        - "kernel"  : RBF kernel split (RandomKernelForest)
        - "soft"    : sigmoid-gated soft split (SoftDecisionTree)

    --- Rotation transforms ---
    rotation : bool, default=False
        Apply PCA rotation to features before each tree (RotationForest).
    random_rotation : bool, default=False
        Apply random orthogonal rotation (RandomRotationForest).

    --- SPORF sparse projection ---
    sparse_projection : bool, default=False
        Use sparse {-1, +1} projection (SPORF). Activates when split_type="oblique".

    --- Regularization ---
    variable_reuse_penalty : float, default=0.0
        Penalty on unused variables. > 0 encourages reuse of past features.
    leaf_regularization : {"none", "l1", "l2"}, default="none"
        Regularization on leaf weights.
    leaf_reg_alpha : float, default=0.1
        Leaf regularization strength.

    --- Constraints ---
    monotone_constraints : dict or None
        {feature_idx: +1 or -1} monotone constraint per feature.
    linear_features : list of int or None
        Feature indices for linearity constraint.
    linearity_lambda : float, default=0.5
        Linearity constraint strength.

    --- Special modes ---
    soft_tree : bool, default=False
        Use SoftDecisionTree (sigmoid gates, gradient learning).
    linear_leaf : bool, default=False
        Fit a linear model at each leaf (LinearForest).
    generalized_target : {"mean", "quantile", "causal"}, default="mean"
        GRF target type. Activates GeneralizedRandomForest when not "mean".
    quantile : float, default=0.5
        Quantile for generalized_target="quantile".

    --- Ensemble strategy ---
    bootstrap : {"standard", "bernoulli", "none"}, default="standard"
        Sampling strategy per tree.
    feature_prob : float, default=0.5
        Bernoulli feature inclusion probability (when bootstrap="bernoulli"
        or split_type="bernoulli").
    criterion : str, default="auto"
        Split criterion. "auto" → "gini" for classification, "mse" for regression.
    max_depth : int or None
    min_samples_split : int, default=2
    min_samples_leaf : int, default=1
    max_features : int, float, str, or None, default="sqrt"

    --- Extra (kernel forest) ---
    n_rff : int, default=32
    gamma : float, default=1.0
    svm_lambda : float, default=0.01

    --- Extra (oblique) ---
    n_directions : int, default=5
    density : float, default=0.1

    --- Infrastructure ---
    n_jobs : int, default=1
    random_state : int or None
    verbose : int, default=0

    Examples
    --------
    >>> from forests import ForestBuilder
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)

    # Standard Random Forest (axis-aligned splits)
    >>> fb = ForestBuilder(n_estimators=10, random_state=0)
    >>> fb.fit(X, y).score(X, y) > 0.9
    True

    # Rotation Forest
    >>> fb = ForestBuilder(n_estimators=10, rotation=True, random_state=0)
    >>> fb.fit(X, y).score(X, y) > 0.8
    True

    # SPORF
    >>> fb = ForestBuilder(n_estimators=10, split_type="oblique",
    ...                    sparse_projection=True, random_state=0)
    >>> fb.fit(X, y).score(X, y) > 0.8
    True

    # Monotone constrained (regression)
    >>> from sklearn.datasets import make_regression
    >>> X2, y2 = make_regression(n_samples=100, n_features=5, random_state=0)
    >>> fb = ForestBuilder(n_estimators=10, monotone_constraints={0: 1},
    ...                    task="regression", random_state=0)
    >>> fb.fit(X2, y2).score(X2, y2) > 0.5
    True
    """

    def __init__(
        self,
        n_estimators: int = 100,
        split_type: str = "axis",
        rotation: bool = False,
        random_rotation: bool = False,
        sparse_projection: bool = False,
        variable_reuse_penalty: float = 0.0,
        leaf_regularization: str = "none",
        leaf_reg_alpha: float = 0.1,
        monotone_constraints: Optional[Dict[int, int]] = None,
        linear_features: Optional[List[int]] = None,
        linearity_lambda: float = 0.5,
        soft_tree: bool = False,
        linear_leaf: bool = False,
        generalized_target: str = "mean",
        quantile: float = 0.5,
        bootstrap: str = "standard",
        feature_prob: float = 0.5,
        criterion: str = "auto",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        max_features: Optional[Union[int, float, str]] = "sqrt",
        n_rff: int = 32,
        gamma: float = 1.0,
        svm_lambda: float = 0.01,
        n_directions: int = 5,
        density: float = 0.1,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
        verbose: int = 0,
        task: str = "classification",  # "classification" or "regression"
    ) -> None:
        self.n_estimators = n_estimators
        self.split_type = split_type
        self.rotation = rotation
        self.random_rotation = random_rotation
        self.sparse_projection = sparse_projection
        self.variable_reuse_penalty = variable_reuse_penalty
        self.leaf_regularization = leaf_regularization
        self.leaf_reg_alpha = leaf_reg_alpha
        self.monotone_constraints = monotone_constraints
        self.linear_features = linear_features
        self.linearity_lambda = linearity_lambda
        self.soft_tree = soft_tree
        self.linear_leaf = linear_leaf
        self.generalized_target = generalized_target
        self.quantile = quantile
        self.bootstrap = bootstrap
        self.feature_prob = feature_prob
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.n_rff = n_rff
        self.gamma = gamma
        self.svm_lambda = svm_lambda
        self.n_directions = n_directions
        self.density = density
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.task = task

    def _resolve_criterion(self) -> str:
        if self.criterion != "auto":
            return self.criterion
        return "gini" if self.task == "classification" else "mse"

    def _build_model(self):
        """Select and instantiate the appropriate model based on parameters."""
        # Check incompatibilities
        _check_incompatibilities(
            rotation=self.rotation,
            random_rotation=self.random_rotation,
            split_type=self.split_type,
            soft_tree=self.soft_tree,
            linear_leaf=self.linear_leaf,
            sparse_projection=self.sparse_projection,
            generalized_target=self.generalized_target,
        )

        crit = self._resolve_criterion()
        use_bootstrap = self.bootstrap != "none"
        common_kw = dict(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            max_features=self.max_features,
            bootstrap=use_bootstrap,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
        )

        # --- Soft tree ---
        if self.soft_tree:
            from .soft_tree import SoftDecisionForest
            n_cls = getattr(self, "_n_classes", 1)
            return SoftDecisionForest(
                n_estimators=self.n_estimators,
                task=self.task,
                n_classes=n_cls,
                bootstrap=use_bootstrap,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )

        # --- GRF variants ---
        if self.generalized_target == "causal":
            from .grf import CausalForest
            return CausalForest(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                bootstrap=use_bootstrap,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )
        if self.generalized_target == "quantile":
            from .grf import QuantileForest
            return QuantileForest(
                quantile=self.quantile,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                bootstrap=use_bootstrap,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )

        # --- Linear leaf ---
        if self.linear_leaf:
            from .linear_tree import LinearForest
            return LinearForest(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                bootstrap=use_bootstrap,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )

        # --- Monotone constraints ---
        if self.monotone_constraints:
            from .constrained import MonotonicConstrainedForest
            return MonotonicConstrainedForest(
                monotone_constraints=self.monotone_constraints,
                criterion=crit,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                max_features=self.max_features,
                bootstrap=use_bootstrap,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )

        # --- Linearity constraint ---
        if self.linear_features:
            from .constrained import LinearConstrainedForest
            return LinearConstrainedForest(
                linear_features=self.linear_features,
                linearity_lambda=self.linearity_lambda,
                criterion=crit,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                max_features=self.max_features,
                bootstrap=use_bootstrap,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )

        # --- Rotation transforms ---
        if self.rotation:
            from .oblique import RotationForest
            return RotationForest(
                criterion=crit,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                bootstrap=use_bootstrap,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )
        if self.random_rotation:
            from .oblique import RandomRotationForest
            return RandomRotationForest(
                criterion=crit,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                bootstrap=use_bootstrap,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )

        # --- Kernel split ---
        if self.split_type == "kernel":
            from .kernel_forest import RandomKernelForest
            return RandomKernelForest(
                n_estimators=self.n_estimators,
                n_rff=self.n_rff,
                gamma=self.gamma,
                svm_lambda=self.svm_lambda,
                criterion=crit,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                bootstrap=use_bootstrap,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )

        # --- Oblique (SPORF or oblique) ---
        if self.split_type == "oblique":
            if self.sparse_projection:
                from .sporf import SPORFClassifier, SPORFRegressor
                Cls = SPORFClassifier if self.task == "classification" else SPORFRegressor
                return Cls(
                    n_estimators=self.n_estimators,
                    n_projections=self.n_directions,
                    density=self.density,
                    criterion=crit,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    min_impurity_decrease=self.min_impurity_decrease,
                    bootstrap=use_bootstrap,
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                )
            else:
                from .oblique import ObliqueForest
                return ObliqueForest(
                    n_estimators=self.n_estimators,
                    n_directions=self.n_directions,
                    criterion=crit,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    min_impurity_decrease=self.min_impurity_decrease,
                    max_features=self.max_features,
                    bootstrap=use_bootstrap,
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                )

        # --- Variable reuse penalty ---
        if self.variable_reuse_penalty > 0.0:
            from .regularized import VariablePenaltyForest
            return VariablePenaltyForest(
                n_estimators=self.n_estimators,
                reuse_alpha=self.variable_reuse_penalty,
                criterion=crit,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                max_features=self.max_features,
                bootstrap=use_bootstrap,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )

        # --- Leaf weight regularization ---
        if self.leaf_regularization != "none":
            from .regularized import LeafWeightRegularizedForest
            return LeafWeightRegularizedForest(
                n_estimators=self.n_estimators,
                leaf_reg=self.leaf_regularization,
                alpha=self.leaf_reg_alpha,
                criterion=crit,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                max_features=self.max_features,
                bootstrap=use_bootstrap,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )

        # --- Bernoulli bootstrap ---
        if self.bootstrap == "bernoulli":
            from .bernoulli_rf import BernoulliRandomForest
            return BernoulliRandomForest(
                n_estimators=self.n_estimators,
                feature_prob=self.feature_prob,
                criterion=crit,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                bootstrap=True,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )

        # --- Default: Standard Random Forest ---
        if self.task == "classification":
            from .random_forest import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                criterion=crit,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                max_features=self.max_features,
                bootstrap=use_bootstrap,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=self.verbose,
            )
        else:
            from .random_forest import RandomForestRegressor
            return RandomForestRegressor(
                n_estimators=self.n_estimators,
                criterion=crit,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                max_features=self.max_features,
                bootstrap=use_bootstrap,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=self.verbose,
            )

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_kwargs) -> "ForestBuilder":
        """Fit the selected forest model.

        Parameters
        ----------
        X : (n, p) array
        y : (n,) target array
        **fit_kwargs : additional arguments passed to model.fit()
            e.g., W=W for CausalForest

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        # Determine number of classes for soft_tree
        if self.task == "classification":
            self._n_classes = int(len(np.unique(y)))
        else:
            self._n_classes = 1

        self.model_ = self._build_model()
        self.model_.fit(X, y, **fit_kwargs)
        self.n_features_in_ = X.shape[1]
        self.model_type_ = type(self.model_).__name__
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions from the selected model."""
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, "model_")
        return self.model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities (classification only)."""
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, "model_")
        if hasattr(self.model_, "predict_proba"):
            return self.model_.predict_proba(X)
        raise AttributeError(f"{self.model_type_} does not support predict_proba.")

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return model score (accuracy or R²)."""
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, "model_")
        return self.model_.score(X, y)

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Return leaf IDs (if model supports it)."""
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, "model_")
        if hasattr(self.model_, "apply"):
            return self.model_.apply(X)
        raise AttributeError(f"{self.model_type_} does not support apply().")

    def get_model(self):
        """Return the underlying fitted model instance."""
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, "model_")
        return self.model_

    @property
    def estimators_(self):
        """Forward estimators_ attribute from underlying model for RF compatibility."""
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, "model_")
        if hasattr(self.model_, "estimators_"):
            return self.model_.estimators_
        return []
