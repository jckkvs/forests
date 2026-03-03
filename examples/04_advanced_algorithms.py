"""
examples/04_advanced_algorithms.py
==================================
RGF (Regularized Greedy Forest) と Soft Decision Tree (ヒントン) の実演。
"""

import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from forests import RegularizedGreedyForest, SoftDecisionTree, SoftDecisionForest, RuleFit

# 1. Regularized Greedy Forest (Johnson & Zhang 2014)
print("--- 1. Regularized Greedy Forest (RGF) ---")
X, y = load_diabetes(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

rgf = RegularizedGreedyForest(n_estimators=50, reg_lambda=0.1, random_state=42)
rgf.fit(X_tr, y_tr)
print(f"RGF R^2 Score: {rgf.score(X_te, y_te):.3f}")

# 2. Soft Decision Tree (Hinton et al. 2017)
print("\n--- 2. Soft Decision Tree (Hinton) ---")
# SoftTree は SGD を使うため、入力の正規化が極めて重要です。
X_bc, y_bc = load_breast_cancer(return_X_y=True)
scaler = StandardScaler()
X_bc = scaler.fit_transform(X_bc)
X_tr, X_te, y_tr, y_te = train_test_split(X_bc, y_bc, test_size=0.2, random_state=42)

# 学習率とエポック数を適切に設定
st = SoftDecisionTree(max_depth=3, learning_rate=0.1, n_epochs=50, random_state=42)
st.fit(X_tr, y_tr)
print(f"SoftTree Accuracy: {st.score(X_te, y_te):.3f}")

# 3. RuleFit (Friedman & Popescu 2008)
print("\n--- 3. RuleFit ---")
rf = RuleFit(n_estimators=20, alpha=0.01, random_state=42)
rf.fit(X_tr, y_tr)
print(f"RuleFit Accuracy: {rf.score(X_te, y_te):.3f}")
# 解釈可能なルールの抽出
print("Top 3 Rules for Cancer Prediction:")
for rule, coef in rf.get_rules(top_n=3):
    print(f"  {rule}: coef={coef:.4f}")
