"""
examples/06_extras_survival_isolation.py
========================================
Survival Analysis (生存分析) と 異常検知 (Isolation Forest) の実演。
"""

import numpy as np
from forests import RandomSurvivalForest, IsolationForest, MondrianForest

rng = np.random.default_rng(42)

# 1. Random Survival Forest (Ishwaran et al. 2008)
print("--- 1. Random Survival Forest for Time-to-Event Data ---")
n = 200
X = rng.random((n, 3))
# 真の生存時間: X[:, 0] が大きいほど生存時間が短くなる指数分布
t = rng.exponential(1.0 / (0.1 + X[:, 0]), n)
# 打ち切りインジケータ (1: イベント発生, 0: 打ち切り)
e = rng.binomial(1, 0.8, n)

rsf = RandomSurvivalForest(n_estimators=50, max_depth=5, random_state=42)
rsf.fit(X, t, e)

# 生存関数 (累積ハザード) の推定
# 予測値は「平均生存時間プロキシ (負の累積ハザード)」
risk_scores = rsf.predict(X[:5])
print(f"Risk scores (negative cumhazard): {risk_scores}")

# 2. Isolation Forest (Liu et al. 2008)
print("\n--- 2. Isolation Forest for Anomaly Detection ---")
X_normal = rng.normal(0, 1, (100, 2))
X_outliers = rng.uniform(5, 10, (10, 2))  # 外れ値
X_all = np.vstack([X_normal, X_outliers])

iso = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
iso.fit(X_normal)  # 正常データで学習 (または全体で学習)
preds = iso.predict(X_all)

# label 1 = normal, -1 = anomaly
print(f"Detected anomalies: {np.sum(preds == -1)}")
print(f"Outliers correctly identified: {np.sum(preds[100:] == -1)} / 10")

# 3. Mondrian Forest (Lakshminarayanan et al. 2014)
print("\n--- 3. Mondrian Forest (Online Learning Ready) ---")
from sklearn.datasets import load_iris
X_i, y_i = load_iris(return_X_y=True)
mf = MondrianForest(n_estimators=10, n_classes=3, lifetime=1.0, random_state=42)
mf.fit(X_i, y_i)
print(f"Mondrian Forest Accuracy: {mf.score(X_i, y_i):.3f}")
