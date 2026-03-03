import numpy as np
from sklearn.datasets import load_iris, make_regression
from sklearn.model_selection import train_test_split
from forests import ForestsClassifier, ForestsRegressor

# Classification data
X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

# Regression data
X2, y2 = make_regression(n_samples=300, n_features=10, noise=10, random_state=0)
X2_tr, X2_te, y2_tr, y2_te = train_test_split(X2, y2, test_size=0.3, random_state=0)

# Survival data
rng = np.random.default_rng(0)
t_surv = rng.exponential(2, 300)
e_surv = rng.binomial(1, 0.7, 300)
t_surv_tr, t_surv_te, e_surv_tr, e_surv_te = train_test_split(t_surv, e_surv, test_size=0.3, random_state=0)

def test_cls(name, kwargs):
    print(f"Testing {name} (Classification)...")
    clf = ForestsClassifier(n_estimators=10, random_state=0, **kwargs)
    clf.fit(X_tr, y_tr)
    print(f"  Score: {clf.score(X_te, y_te):.3f} [{clf.model_type_}]")

def test_reg(name, kwargs):
    print(f"Testing {name} (Regression)...")
    reg = ForestsRegressor(n_estimators=10, random_state=0, **kwargs)
    reg.fit(X2_tr, y2_tr)
    print(f"  Score: {reg.score(X2_te, y2_te):.3f} [{reg.model_type_}]")

cls_configs = [
    ("RandomForest", dict()),
    ("ExtraTrees", dict(extra_trees=True)),
    ("GradientBoosting", dict(boosting=True)),
    ("RuleFit", dict(rulefit=True)),
    ("RGF", dict(rgf=True)),
    ("DeepForest", dict(deep_forest=True)),
    ("Conformal", dict(conformal=True)),
    ("Mondrian", dict(mondrian=True)),
    ("ObliqueForest", dict(split_type="oblique")),
    ("RotationForest", dict(rotation=True)),
    ("RandomRotationForest", dict(random_rotation=True)),
    ("SPORF", dict(split_type="oblique", sparse_projection=True)),
    ("SoftDecisionForest", dict(soft_tree=True)),
    ("BernoulliRF", dict(bootstrap="bernoulli")),
    ("VariablePenalty", dict(variable_reuse_penalty=0.1)),
    ("KernelForest", dict(split_type="kernel")),
]

reg_configs = [
    ("RandomForest", dict()),
    ("ExtraTrees", dict(extra_trees=True)),
    ("GradientBoosting", dict(boosting=True)),
    ("RuleFit", dict(rulefit=True)),
    ("RGF", dict(rgf=True)),
    ("Conformal", dict(conformal=True)),
    ("MonotonicConstrained", dict(monotone_constraints={0: 1})),
    ("LinearConstrained", dict(linear_features=[0])),
    ("LeafWeightRegularized", dict(leaf_regularization="l2")),
    ("LinearForest", dict(linear_leaf=True)),
    ("LinearBoost", dict(linear_boost=True)),
    ("GRF-Causal", dict(generalized_target="causal")),
    ("GRF-Quantile", dict(generalized_target="quantile")),
    ("QuantileRegForest (Meinshausen)", dict(quantile_reg=True)),
]

print("--- CLASSIFICATION TESTS ---")
for name, kw in cls_configs:
    try:
        test_cls(name, kw)
    except Exception as e:
        print(f"  FAILED: {name} - {e}")

print("\n--- REGRESSION TESTS ---")
for name, kw in reg_configs:
    try:
        test_reg(name, kw)
    except Exception as e:
        print(f"  FAILED: {name} - {e}")

print("\n--- SPECIAL TESTS & ERRORS ---")
# Additional Tests (Survival, Isolation, Errors)
print("Testing DeepForest (Regression)...")
try:
    reg = ForestsRegressor(n_estimators=10, random_state=0, deep_forest=True)
    reg.fit(X2_tr, y2_tr)
except Exception as e:
    print(f"  EXPECTED ERROR: {e}")

print("Testing MondrianForest (Regression)...")
try:
    reg = ForestsRegressor(n_estimators=10, random_state=0, mondrian=True)
    reg.fit(X2_tr, y2_tr)
except Exception as e:
    print(f"  EXPECTED ERROR: {e}")

print("Testing Survival (Regression)...")
try:
    reg = ForestsRegressor(n_estimators=10, random_state=0, survival=True)
    reg.fit(X2_tr, t_surv_tr, e=e_surv_tr)
    print(f"  Score: {reg.score(X2_te, t_surv_te):.3f} [{reg.model_type_}]")
except Exception as e:
    print(f"  FAILED: {e}")
    
print("Testing IsolationForest...")
try:
    iso = ForestsClassifier(n_estimators=10, isolation=True, random_state=0)
    iso.fit(X_tr)
    print(f"  Model Type: {iso.model_type_}")
except Exception as e:
    print(f"  FAILED: {e}")
