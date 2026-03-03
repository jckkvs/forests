"""
examples/03_rf_similarity_kernel.py
=====================================
RFSimilarity / RFKernel - RFカーネルを使ったSVM

ランダムフォレストの葉の一致度をカーネル行列に変換し、
precomputed カーネルとして SVC に与える例。
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from forests import RandomForestClassifier, RFSimilarity, RFKernel

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

# ─── Step 1: フォレストを学習 ─────────────────────────────────────────────
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_tr, y_tr)
print(f"RF accuracy: {rf.score(X_te, y_te):.3f}")

# ─── Step 2: 類似度行列 ──────────────────────────────────────────────────
sim = RFSimilarity(rf)
S_train = sim.fit_transform(X_tr)     # (n_train, n_train)
S_test  = sim.transform(X_te)         # (n_test,  n_train)
print(f"\nSimilarity matrix (train): {S_train.shape}")
print(f"  S[0,0]={S_train[0,0]:.3f} (自分自身=1.0)")
print(f"  S[0トレイン,1テスト]の平均={S_test.mean():.3f}")

# ─── Step 3: RFカーネル + SVM ────────────────────────────────────────────
print("\n▼ RF Kernel (cosine) + SVM:")
kernel = RFKernel(rf, mode="cosine")
K_train = kernel.fit_transform(X_tr)  # (n_train, n_train)
K_test  = kernel.transform(X_te)      # (n_test,  n_train)

svm = SVC(kernel="precomputed", random_state=0)
svm.fit(K_train, y_tr)
print(f"  RF-cosine kernel SVM accuracy: {svm.score(K_test, y_te):.3f}")

print("\n▼ RF Kernel (rbf γ=2.0) + SVM:")
kernel_rbf = RFKernel(rf, mode="rbf", gamma=2.0)
K_train_rbf = kernel_rbf.fit_transform(X_tr)
K_test_rbf  = kernel_rbf.transform(X_te)

svm_rbf = SVC(kernel="precomputed", random_state=0)
svm_rbf.fit(K_train_rbf, y_tr)
print(f"  RF-rbf kernel SVM accuracy:    {svm_rbf.score(K_test_rbf, y_te):.3f}")
