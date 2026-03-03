"""
examples/01_quickstart.py
=========================
forests ライブラリ - クイックスタート

sklearn の決定木・ランダムフォレストと同様の感覚で使えます。
"""

import numpy as np
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split

from forests import (
    CARTClassifier,
    CARTRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
)

# ─── 分類 ───────────────────────────────────────────────────────────────────
X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

# CART
cart = CARTClassifier(max_depth=3, criterion="gini", random_state=0)
cart.fit(X_tr, y_tr)
print(f"[CART]         acc={cart.score(X_te, y_te):.3f}  depth={cart.get_depth()}  leaves={cart.get_n_leaves()}")

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_tr, y_tr)
print(f"[RandomForest] acc={rf.score(X_te, y_te):.3f}  n_trees={len(rf.estimators_)}")

# ExtraTrees
et = ExtraTreesClassifier(n_estimators=100, random_state=0)
et.fit(X_tr, y_tr)
print(f"[ExtraTrees]   acc={et.score(X_te, y_te):.3f}")

# ─── 回帰 ───────────────────────────────────────────────────────────────────
X2, y2 = load_diabetes(return_X_y=True)
X2_tr, X2_te, y2_tr, y2_te = train_test_split(X2, y2, test_size=0.3, random_state=0)

cart_r = CARTRegressor(max_depth=5, criterion="mse", random_state=0)
cart_r.fit(X2_tr, y2_tr)
print(f"\n[CARTRegressor]        R^2={cart_r.score(X2_te, y2_te):.3f}")

rf_r = RandomForestRegressor(n_estimators=100, random_state=0)
rf_r.fit(X2_tr, y2_tr)
print(f"[RandomForestRegressor] R^2={rf_r.score(X2_te, y2_te):.3f}")

# ─── 葉ID取得 (RFSimilarity に使用) ────────────────────────────────────────
leaf_ids = rf.apply(X_te)  # shape: (n_test, n_estimators)
print(f"\napply() shape: {leaf_ids.shape}")
