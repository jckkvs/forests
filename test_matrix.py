import warnings
import numpy as np
import pandas as pd
from forests import ForestsRegressor, ForestsClassifier, IncompatibleOptionsWarning

# Setup dummy data
X = np.random.rand(20, 4)
y = np.random.rand(20)
# events for survival
e = np.random.binomial(1, 0.7, 20)

# Define 25 triggering options
options = [
    ("extra_trees", {"extra_trees": True}),
    ("boosting", {"boosting": True}),
    ("rulefit", {"rulefit": True}),
    ("rgf", {"rgf": True}),
    ("deep_forest", {"deep_forest": True}),
    ("conformal", {"conformal": True}),
    ("isolation", {"isolation": True}),
    ("mondrian", {"mondrian": True}),
    ("survival", {"survival": True}),
    ("linear_boost", {"linear_boost": True}),
    ("quantile_reg", {"quantile_reg": True}),
    ("soft_tree", {"soft_tree": True}),
    ("linear_leaf", {"linear_leaf": True}),
    ("oblique", {"split_type": "oblique"}),
    ("kernel", {"split_type": "kernel"}),
    ("rotation", {"rotation": True}),
    ("random_rotation", {"random_rotation": True}),
    ("sporf", {"split_type": "oblique", "sparse_projection": True}),
    ("causal", {"generalized_target": "causal"}),
    ("grf_quantile", {"generalized_target": "quantile"}),
    ("var_penalty", {"variable_reuse_penalty": 0.1}),
    ("leaf_reg", {"leaf_regularization": "l2"}),
    ("monotone", {"monotone_constraints": {0: 1}}),
    ("linear_feat", {"linear_features": [0]}),
    ("bernoulli", {"bootstrap": "bernoulli"}),
]

results = []

print(f"Starting Matrix Test: {len(options)} options, {len(options)*(len(options)-1)} combinations...")

for i, (name1, kw1) in enumerate(options):
    for j, (name2, kw2) in enumerate(options):
        if i == j:
            continue
            
        combined_kw = {**kw1, **kw2}
        combined_kw["n_estimators"] = 2
        combined_kw["random_state"] = 42
        
        test_id = f"{name1} + {name2}"
        outcome = "SUCCESS"
        warned = False
        error_msg = ""
        model_type = "N/A"
        
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                # We use Regressor primarily, but fall back or handle for classification only
                # Note: deep_forest and mondrian don't support regression
                is_reg = True
                if "deep_forest" in combined_kw or "mondrian" in combined_kw:
                    build_cls = ForestsClassifier
                    is_reg = False
                else:
                    build_cls = ForestsRegressor
                
                model = build_cls(**combined_kw)
                
                # Try to fit to see which one "wins"
                if combined_kw.get("survival"):
                    model.fit(X, y, e=e)
                elif combined_kw.get("isolation"):
                    model.fit(X)
                elif combined_kw.get("generalized_target") == "causal":
                    model.fit(X, y, W=np.random.binomial(1, 0.5, 20))
                else:
                    model.fit(X, y)
                
                model_type = model.model_type_
                if any(issubclass(warn.category, IncompatibleOptionsWarning) for warn in w):
                    warned = True
                    
        except Exception as e_exc:
            outcome = "FAILED"
            error_msg = str(e_exc)
            
        results.append({
            "Pair": test_id,
            "Outcome": outcome,
            "ModelWon": model_type,
            "Warned": warned,
            "Error": error_msg[:100]
        })

df = pd.DataFrame(results)
df.to_csv("matrix_test_results.csv", index=False)

print("\n--- Summary ---")
print(f"Total combinations tested: {len(results)}")
print(f"Success: {sum(df['Outcome'] == 'SUCCESS')}")
print(f"Failed (Expected or otherwise): {sum(df['Outcome'] == 'FAILED')}")
print(f"Warnings caught: {sum(df['Warned'])}")

# Check top failures
if sum(df['Outcome'] == 'FAILED') > 0:
    print("\nRecent Failures Sample:")
    print(df[df['Outcome'] == 'FAILED'].head(10))

print("\nResults saved to matrix_test_results.csv")
