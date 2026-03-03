"""
examples/02_forest_builder.py
==============================
ForestsClassifier - トッピング式カスタマイズ

ForestsClassifier の引数を変えるだけでアルゴリズムを切り替え可能。
競合する組み合わせを指定すると IncompatibleOptionsWarning が出る。
"""

import warnings
import numpy as np
from sklearn.datasets import load_iris, make_regression
from sklearn.model_selection import train_test_split

from forests import ForestsClassifier, ForestsRegressor, IncompatibleOptionsWarning

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

configs = [
    # (name, kwargs)
    ("RandomForest (default)",   dict()),
    ("RotationForest",           dict(rotation=True)),
    ("RandomRotationForest",     dict(random_rotation=True)),
    ("ObliqueForest",            dict(split_type="oblique")),
    ("SPORF",                    dict(split_type="oblique", sparse_projection=True)),
    ("VariablePenalty α=0.1",    dict(variable_reuse_penalty=0.1)),
    ("BernoulliRF p=0.5",        dict(bootstrap="bernoulli", feature_prob=0.5)),
]

print("=" * 55)
print(f"{'設定名':<28} {'Accuracy':>8}")
print("=" * 55)
for name, kw in configs:
    fb = ForestsClassifier(n_estimators=30, random_state=0, **kw)
    fb.fit(X_tr, y_tr)
    acc = fb.score(X_te, y_te)
    print(f"{name:<28} {acc:>8.3f}  [{fb.model_type_}]")

# ─── 互換性警告デモ ─────────────────────────────────────────────────────────
print("\n▼ 互換性警告のデモ:")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    fb_conflict = ForestsClassifier(
        n_estimators=5,
        rotation=True,
        split_type="oblique",   # ← rotation と競合
        random_state=0,
    )
    fb_conflict.fit(X_tr, y_tr)
    if w:
        print(f"  ⚠️  {w[0].message}")

# ─── 回帰タスク ──────────────────────────────────────────────────────────────
print("\n▼ 回帰タスク:")
X2, y2 = make_regression(n_samples=300, n_features=10, noise=10, random_state=0)
X2_tr, X2_te, y2_tr, y2_te = train_test_split(X2, y2, test_size=0.3, random_state=0)

reg_configs = [
    ("RandomForest",              dict()),
    ("LeafL2 α=0.5",              dict(leaf_regularization="l2", leaf_reg_alpha=0.5)),
    ("LinearLeaf (LinearForest)", dict(linear_leaf=True)),
    ("MonotonicConstraint f0↑",  dict(monotone_constraints={0: 1})),
]
for name, kw in reg_configs:
    fb = ForestsRegressor(n_estimators=30, random_state=0, **kw)
    fb.fit(X2_tr, y2_tr)
    r2 = fb.score(X2_te, y2_te)
    print(f"  {name:<28} R²={r2:.3f}")
