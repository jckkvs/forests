# forests 🌲🌳🌵

**forests** は、決定木およびアンサンブル学習（フォレスト）アルゴリズムをフルスクラッチで実装した、包括的な Python ライブラリです。
標準的なランダムフォレストから、最新の統計的因果推論フォレスト、ソフト決定木、RGF まで、20種類以上のアルゴリズムを単一のライブラリで提供します。

すべてのモデルは **scikit-learn 互換 API** (`fit`, `predict`, `score` 等) を備えており、既存のパイプラインにシームレスに組み込むことが可能です。

---

## ✨ 特徴 (Features)

- **20+ 種類の独自アルゴリズム**: CART, Oblique Forest, RGF, SoftTree, Causal Forest, Mondrian Forest 等。
- **Composable Architecture**: `ForestsClassifier` を通じて、回転、スパース投影、正則化、制約（単調性・線形性）を「トッピング」のように自由に組み合わせて独自のフォレストを構築可能。
- **統計的因果推論 & 高度な推論**: CATE (Conditional Average Treatment Effect) 推論、分位点回帰、生存分析への対応。
- **類似度 & カーネル**: サンプルが同じ葉に落ちる頻度に基づく RF Proximity および RF Kernel を提供。
- **高カバレッジ & 安定性**: テストカバレッジ 88%以上を達成し、多くの論文アルゴリズムを安定して再現。

---

## 📦 インストール (Installation)

```bash
git clone https://github.com/user/forests.git
cd forests
pip install -e .
```

---

## 🚀 クイックスタート (Quickstart)

```python
from forests import RandomForestClassifier, RotationForest
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

# 標準的なランダムフォレスト
rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
print(f"RF Accuracy: {rf.score(X, y):.3f}")

# Rotation Forest (Rodriguez et al. 2006)
rot = RotationForest(n_estimators=50, random_state=42).fit(X, y)
print(f"Rotation Forest Accuracy: {rot.score(X, y):.3f}")
```

---

## 🛠 `ForestsClassifier` によるカスマイズ

`ForestsClassifier` クラスを使用すると、複雑なフォレスト構成を非常に直感的に記述できます。
2025年の最新アップデートにより、以下の主要なアンサンブルモデルもすべてこの単一のインターフェースから呼び出せるようになりました。

### 新たに統合されたアルゴリズムフラグ
`ForestsClassifier` および `ForestsRegressor` に以下の bool フラグを渡すことで、該当のアルゴリズムに切り替わります。

- `extra_trees=True`: Extremely Randomized Trees (ExtraTrees) を使用します。
- `boosting=True`: 勾配ブースティング木 (Gradient Boosted Trees) を使用します。`learning_rate` で学習率を制御します。
- `rulefit=True`: RuleFit アルゴリズムを使用し、抽出したルール特徴量上の疎な線形モデルを学習します。
- `rgf=True`: Regularized Greedy Forest (RGF) を使用します。`l2_leaf_reg` で葉のL2正則化の強さを設定します。
- `deep_forest=True`: カスケードフォレストによる深層学習ベースの森 (Deep Forest) を構築します。
- `conformal=True`: Conformal Prediction Forest を使用し、テストサンプルに対する信頼区間/予測集合を提供します。
- `mondrian=True`: オンライン学習可能な無限次元フォレスト (Mondrian Forest) を構築します。
- `isolation=True`: 教師なし異常検知のための Isolation Forest を構築します (Classifierのみ)。
- `survival=True`: Random Survival Forest を構築します (Regressorのみ。`fit` 時に `e=events` パラメータが必要です)。
- `linear_boost=True`: 線形葉木を用いた勾配ブースティング (LinearBoost) を行います (Regressorのみ)。
- `quantile_reg=True`: Meinshausen (2006) 手法による分位点回帰フォレストを構築します (Regressorのみ)。


```python
from forests import ForestsClassifier, ForestsRegressor

# SPORF (Sparse Projection Oblique Randomer Forest) の構成
fb = ForestsClassifier(
    n_estimators=100,
    split_type="oblique",
    sparse_projection=True,
    density=0.05
).fit(X, y)

# 単調性制約 + 線形葉木 (Linear Tree) の構成
fb_mt = ForestsRegressor(
    n_estimators=20,
    monotonic_constraints=[1, 0, -1, 0],  # 1: 増加, -1: 減少
    linear_leaf=True
).fit(X_reg, y_reg)

# 勾配ブースティング木 (GBT) の構成
fb_boost = ForestsClassifier(
    n_estimators=100,
    boosting=True,
    learning_rate=0.05,
    max_depth=3
).fit(X, y)

# RuleFit の構成
fb_rulefit = ForestsRegressor(
    n_estimators=100,
    rulefit=True,
    learning_rate=0.01
).fit(X_reg, y_reg)
```

---

## 📚 実装アルゴリズム一覧

| カテゴリ | クラス / 機能 | 参照論文 |
|:---|:---|:---|
| **基本モデル** | `CART`, `RandomForest`, `ExtraTrees` | Breiman (1984/2001) |
| **高度な木構造** | `ObliqueForest`, `RotationForest`, `SPORF` | Murthy (1994), Rodriguez (2006) |
| **高度な学習機** | `RegularizedGreedyForest (RGF)` | Johnson & Zhang (2014) |
| **微分配学習** | `SoftDecisionTree (Hinton)`, `SoftDecisionForest` | Frosst & Hinton (2017) |
| **線形・ルール** | `LinearTree`, `LinearBoost`, `RuleFit` | Friedman (2008) |
| **統計的学習** | `GeneralizedRandomForest`, `CausalForest` | Athey et al. (2019) JASA |
| **不確実性** | `QuantileForest`, `ConformalForest` | Meinshausen (2006) |
| **特殊タスク** | `IsolationForest`, `RandomSurvivalForest` | Liu (2008), Ishwaran (2008) |
| **オンライン学習** | `MondrianForest` | Lakshminarayanan (2014) |
| **解析ツール** | `RFSimilarity`, `RFKernel` | Breiman (2001) §8 |

---

## 🧪 テスト・品質管理 (Quality Assurance)

プロジェクトは以下の品質基準（DoD）を満たしています。

- **行カバレッジ**: > 87%
- **分岐カバレッジ**: > 75%
- **依存性**: Scikit-Learn, NumPy, SciPy (最小限)
- **API互換性**: 全クラスで `BaseEstimator` を継承し、Sklearn との完全な互換性を確保

テスト実行：
```bash
pytest tests/ -v --cov=forests --cov-branch
```

---

## 📜 ライセンス

MIT License © 2025 forests contributors
