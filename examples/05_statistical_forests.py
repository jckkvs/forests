"""
examples/05_statistical_forests.py
==================================
Generalized Random Forest (GRF), Causal Forest, Quantile Forest の実演。
"""

import numpy as np
import matplotlib.pyplot as plt
from forests import CausalForest, QuantileForest, GeneralizedRandomForest

# 再現性のためのシード
rng = np.random.default_rng(42)

# 1. 因果推論フォレスト (Causal Forest - Athey et al. 2019)
print("--- 1. Causal Forest for Treatment Effect Estimation ---")
n = 1000
p = 5
X = rng.random((n, p))
# 処置変 W (50%の確率で 1)
W = rng.binomial(1, 0.5, n)
# 真の処置効果 tau(x): 変数 X0 に依存して効果が変わる (異質性)
tau = 2.0 * X[:, 0]
# 観測アウトカム Y = ベースライン + tau(x) * W + ノイズ
Y = (X[:, 1] + X[:, 2]) + (tau * W) + rng.normal(0, 0.1, n)

cf = CausalForest(n_estimators=100, random_state=42)
cf.fit(X, Y, W)

# 異質処置効果 (CATE) の推定
cate_hat = cf.predict(X)
print(f"真の平均処置効果 (ATE): {tau.mean():.3f}")
print(f"推定平均処置効果 (ATE): {cf.ate():.3f}")

# 2. 分位点回帰フォレスト (Quantile Forest)
print("\n--- 2. Quantile Forest for Uncertainty Quantification ---")
# 非線形な関係とヘテロスケダスティシティ (不等分散)
X_reg = np.sort(5 * rng.random((200, 1)), axis=0)
y_reg = np.sin(X_reg).ravel() + rng.normal(0, 0.1 + 0.1 * X_reg.ravel(), 200)

# 中央値(0.5), 下限(0.1), 上限(0.9) の予測
qf_low = QuantileForest(n_estimators=50, quantile=0.1, random_state=42).fit(X_reg, y_reg)
qf_med = QuantileForest(n_estimators=50, quantile=0.5, random_state=42).fit(X_reg, y_reg)
qf_high = QuantileForest(n_estimators=50, quantile=0.9, random_state=42).fit(X_reg, y_reg)

y_low = qf_low.predict(X_reg)
y_med = qf_med.predict(X_reg)
y_high = qf_high.predict(X_reg)

print(f"10% quantile mean: {y_low.mean():.3f}")
print(f"90% quantile mean: {y_high.mean():.3f}")
print("Prediction interval contains point in training: ", np.mean((y_reg >= y_low) & (y_reg <= y_high)))
