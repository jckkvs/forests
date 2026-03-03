# REPRODUCE_PROMPT.md
# forests ライブラリ 完全再現設計書
# 
# 目的: この文書を読んだ別の生成AIが、forests ライブラリを完全に再現できること。
# 対象: GPT-4, Claude, Gemini など、Pythonコードを生成できる言語モデル

---

# 0. 概要・設計哲学

```
ライブラリ名 : forests
バージョン   : 0.1.0
言語         : Python 3.9+
依存関係     : numpy, scipy, scikit-learn, joblib
              (torch/lightgbm/xgboostは使用しない)
設計思想     : 
  1. 全アルゴリズムをフルスクラッチで実装（既存ライブラリへの依存なし）
  2. scikit-learn互換API (fit/predict/score/predict_proba/apply)
  3. ForestBuilder総合クラスで全アルゴリズムを引数一つで切り替え可能
  4. 互換性のない引数の組み合わせはIncompatibleOptionsWarningで警告
```

---

# 1. ディレクトリ構造

```
forests/                    ← パッケージルート（PyPI配布単位）
├── pyproject.toml
├── README.md
├── REPRODUCE_PROMPT.md     ← この文書
├── forests/                ← Pythonパッケージ
│   ├── __init__.py
│   ├── base.py
│   ├── cart.py
│   ├── random_forest.py
│   ├── regularized.py
│   ├── constrained.py
│   ├── oblique.py
│   ├── sporf.py
│   ├── rgf.py
│   ├── soft_tree.py
│   ├── linear_tree.py
│   ├── rulefit.py
│   ├── bernoulli_rf.py
│   ├── grf.py
│   ├── kernel_forest.py
│   ├── extras.py
│   ├── boosting.py
│   ├── deep_forest.py
│   ├── conformal.py
│   ├── embedding.py
│   ├── similarity.py
│   └── builder.py
├── tests/
│   ├── conftest.py
│   ├── test_cart.py
│   ├── test_random_forest.py
│   ├── test_builder.py
│   └── test_extras.py
└── examples/
    ├── 01_quickstart.py
    ├── 02_forest_builder.py
    ├── 03_rf_similarity_kernel.py
    ├── 04_advanced_algorithms.py
    └── 05_conformal_gbt.py
```

---

# 2. pyproject.toml

```toml
[build-system]
requires = ["setuptools>=65", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "forests"
version = "0.1.0"
requires-python = ">=3.9"
dependencies = [
    "numpy>=1.21",
    "scipy>=1.7",
    "scikit-learn>=1.0",
    "joblib>=1.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-cov>=4.0", "pytest-xdist>=3.0", "mutmut>=2.4"]
```

---

# 3. forests/base.py の設計

## 3.1 Node（データクラス）

```python
@dataclass
class Node:
    feature: Optional[int] = None      # 分割特徴量インデックス（葉ではNone）
    threshold: Optional[float] = None  # 分割しきい値（葉ではNone）
    value: Optional[np.ndarray] = None # 予測値（分類: 確率ベクトル, 回帰: [mean]）
    impurity: float = 0.0
    n_samples: int = 0
    left: Optional["Node"] = None      # X[:,feature] <= threshold → left
    right: Optional["Node"] = None
    depth: int = 0
    leaf_id: int = -1                  # 葉のID（内部ノードは-1）
    weight: float = 1.0               # 正則化モデルで使用
    extra: dict = field(default_factory=dict)
    
    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None
```

## 3.2 不純度関数

```python
# gini_impurity(y, n_classes) → float:  1 - Σ p_k^2
# entropy_impurity(y, n_classes) → float: -Σ p_k log2(p_k)
# mse_impurity(y) → float: mean((y - mean(y))^2)
# mae_impurity(y) → float: mean(|y - median(y)|)
IMPURITY_FN = {"gini": gini_impurity, "entropy": entropy_impurity,
               "mse": mse_impurity, "mae": mae_impurity, "friedman_mse": mse_impurity}
```

## 3.3 BaseTree (abc.ABC + BaseEstimator)

重要: **BaseTree は BaseEstimator と abc.ABC の両方を継承する**。
sklearn 1.6+ の `__sklearn_tags__` MRO問題を回避するため。

```python
class BaseTree(BaseEstimator, abc.ABC):
    # 抽象メソッド（サブクラスで実装必須）:
    #   _find_best_split(X, y, feature_indices, rng, **kwargs)
    #     → (best_feature, best_threshold, best_gain)
    #   _node_value(y) → np.ndarray
    #   _impurity(y) → float
    
    # 具体メソッド:
    #   _select_features(n_features, rng) → indices  ← max_features を尊重
    #   _build(X, y, depth, rng) → Node              ← 再帰的な木構築
    #   fit(X, y) → self                              ← root_ をセット
    #   _predict_node(x, node) → np.ndarray          ← 1サンプル推論
    #   _apply_node(x, node) → int                   ← leaf_id を返す
    #   apply(X) → np.ndarray                        ← 全サンプルの leaf_id
    #   get_depth() / get_n_leaves() / get_leaves()
    
    # 注意: check_is_fitted は使わない（MRO問題）
    # 代わりに hasattr(self, 'root_') を使う
```

## 3.4 BaseForest (BaseEstimator + abc.ABC)

```python
class BaseForest(BaseEstimator, abc.ABC):
    # 抽象メソッド:
    #   _make_estimator(random_state) → BaseTree
    
    # 具体メソッド:
    #   _sample_data(X, y, rng) → (X_boot, y_boot)   ← bootstrap or full
    #   _fit_single(seed, X, y, fit_kwargs) → BaseTree ← 1本の木を並列fit
    #   fit(X, y) → self   ← joblib.Parallel で全木を並列学習
    #   apply(X) → (n, n_estimators)  ← 全木の leaf_id を concat
    #   _aggregate_predict(X) → (n_estimators, n, value_dim)
```

## 3.5 Mixin クラス

```python
class ClassifierForestMixin(ClassifierMixin):
    # predict_proba(X): 全木の確率ベクトルを平均
    # predict(X): argmax(predict_proba)
    # クラスは 0..K-1 にマップされる（classes_ 属性で元ラベルを保持）

class RegressorForestMixin(RegressorMixin):
    # predict(X): 全木の予測値を平均
    # 注意: _predict_node が返す1要素配列から .item() または float() で取り出す
```

---

# 4. forests/cart.py の設計

## 4.1 _best_split_axis(X, y, feature_indices, impurity_fn, min_samples_leaf, n_classes)

```
入力: X (n,p), y (n,), feature_indices, impurity_fn, min_samples_leaf, n_classes(Noneなら回帰)
出力: (best_feature, best_threshold, best_gain)

アルゴリズム:
  For 各特徴量 f in feature_indices:
    thresholds = midpoints between consecutive unique values
    For 各 threshold t:
      gain = impurity(y) - n_left/n * impurity(y_left) - n_right/n * impurity(y_right)
      if gain > best_gain and n_left >= min_samples_leaf and n_right >= min_samples_leaf:
        update best
```

## 4.2 CARTClassifier (BaseTree + ClassifierMixin)

```
fit(X, y):
  self.classes_ = np.unique(y)  ← ソート済みユニークラベル
  y_mapped = [label_map[yi] for yi in y]  ← 0..K-1 にリマップ
  super().fit(X, y_mapped)   ← BaseTree.fit

predict_proba(X) → (n, K):
  各サンプルを _predict_node でトラバース → 確率ベクトル

predict(X):
  classes_[argmax(predict_proba(X), axis=1)]

_node_value(y): counts / counts.sum()  ← 正規化済み確率
_impurity(y): IMPURITY_FN[criterion](y, n_classes_)
```

## 4.3 CARTRegressor (BaseTree + RegressorMixin)

```
_node_value(y): np.array([np.mean(y)])
_impurity(y): IMPURITY_FN[criterion](y)   ← n_classesなし
predict(X) → (n,): float(_predict_node(x, root_)[0])
```

---

# 5. forests/random_forest.py の設計

```
RandomForestClassifier(BaseForest, ClassifierForestMixin)
  _make_estimator(seed):
    return CARTClassifier(criterion, max_depth, min_samples_split,
                          min_samples_leaf, max_features, random_state=seed)
  
  _validate_fit_params(X, y):
    self.classes_ = np.unique(y)   ← forestレベルで保持
    各treeのfitにはy_mappedを渡さず、treeの内部でmappingする
  
  predict_proba(X):
    各treeのpredict_proba(X) を平均 → (n, K)
  
  predict(X):
    classes_[argmax(predict_proba)]
  
  apply(X) → (n, n_estimators):
    np.column_stack([tree.apply(X) for tree in estimators_])

RandomForestRegressor: RegressorForestMixinを使用
ExtraTreesClassifier: max_features="sqrt"でCARTClassifierを使用するが、
                      _find_best_split内でランダムに1閾値のみ試す（ExtraTree分割）
ExtraTreesRegressor: 同上

注意: ExtraTreesでは分割探索時に全thresholdを試さず、
     一様乱数でthresholdを1つ選ぶ（Geurts et al. 2006 の本質的な差異）
```

---

# 6. forests/regularized.py の設計

## VariablePenaltyForest

```
目的: 使われていない変数を優先的に使うよう促す正則化。
      reuse_alpha: 使われた変数のgainに対するペナルティ係数。

実装:
  カスタム _CARTClassifierWithPenalty クラスを作成。
  _find_best_split をオーバーライド:
    gain を計算後、feature が already_used_ にあれば:
    penalized_gain = gain * (1 - reuse_alpha)
    already_used_ を更新してフォレスト全体で共有はしない（木ごとに独立）。
```

## LeafWeightRegularizedForest

```
目的: 葉の値をL1/L2正則化。
leaf_reg: "l1" or "l2"
alpha: 正則化強度

フィット後に各木の全葉ノードの value を shrink:
  L1: sign(v) * max(0, |v| - α)   (soft thresholding)
  L2: v / (1 + α)
```

---

# 7. forests/constrained.py の設計

## MonotonicConstrainedForest

```
monotone_constraints: {feature_idx: +1 or -1}
  +1: 特徴量が増えると予測値が上昇してほしい
  -1: 特徴量が増えると予測値が下降してほしい

実装:
  分割探索時に制約チェック:
  for split (f, t):
    if f in monotone_constraints:
      方向 = monotone_constraints[f]
      left_mean, right_mean = mean(y_left), mean(y_right)
      if 方向 == +1 and left_mean > right_mean: スキップ
      if 方向 == -1 and left_mean < right_mean: スキップ
```

## LinearConstrainedForest

```
linear_features: [f1, f2, ...]  線形に動作させたい特徴量インデックスリスト
linearity_lambda: 線形性強度 ∈ [0, 1]

実装:
  各葉でGain計算時に線形性ペナルティを付与:
  penalized_gain = (1 - linearity_lambda) * gain + linearity_lambda * linear_gain
  linear_gain: 線形性特徴量上の分割の場合に高い報酬を与える
```

---

# 8. forests/oblique.py の設計

## ObliqueForest

```
n_directions: int  各ノードで試みる斜め分割方向数

分割探索:
  For i in range(n_directions):
    w = random unit vector (rng.standard_normal(p), normalized)
    z = X @ w  ← 1次元投影
    threshold候補でGainを計算 → _best_split_axis の1次元版
  全方向で最良の (w, threshold, gain) を返す
  
予測: 各ノードの通過は x @ feature_vector <= threshold で判定
```

## RotationForest (Rodriguez et al. 2006)

```
前処理: 各木のfit前に PCA 変換行列 R を計算:
  グループ数 K に特徴量を分割、各グループに PCA
  R = block_diag(PCA_1, PCA_2, ..., PCA_K)
  X_rot = X @ R.T

分割探索: 通常のCARTを X_rot 上で実行
予測:     x_rot = x @ R.T として通常予測
```

## RandomRotationForest

```
RotationForest と同じだが、PCA の代わりに
ランダム直交行列 Q (Gram-Schmidt 正規化) を使用。
より低コストで多様性を高める。
```

---

# 9. forests/sporf.py の設計

```
参考: Tomita et al. (2020) Sparse Projection Oblique Randomer Forest. JMLR.

分割探索:
  n_projections: 試みる射影数
  density: 射影ベクトルの非ゼロ割合

  各射影 i:
    s ~ Rademacher({-1, +1}) * Bernoulli(density)
    w = s / ||s||  ← スパースランダムベクトル
    z = X @ w
    z上でbest splitを探索

実装: SPORFClassifier, SPORFRegressor
  CARTと同じ BaseTree 継承、_find_best_split をオーバーライド
```

---

# 10. forests/rgf.py の設計

```
参考: Johnson & Zhang (2014). Learning Nonlinear Functions Using Regularized Greedy Forest. TPAMI.

アルゴリズム:
  1. F_0 = 空の木（定数予測）
  2. For m = 1..M:
     a. 1本の木を構築して追加（葉を1個増やす操作）
     b. 全既存葉の重みをL2正則化つきで最適化:
        w* = argmin Σ L(y_i, F(x_i)) + λ||w||^2
        → 閉形式: w* = (Σ x_j^2 + λ)^{-1} Σ x_j y_j  （MSEの場合）
        
実装の要点:
  - 木の追加は正確にはCARTの通常分割だが、
    葉のweightは毎イテレーション λ正則化CLSで更新される
  - RegularizedGreedyForest クラス（回帰専用）
```

---

# 11. forests/soft_tree.py の設計

```
参考: Frosst & Hinton (2017). Distilling a Neural Network Into a Soft Decision Tree. arXiv:1711.09784.

SoftNode: シグモイドゲートの内部ノード
  gate(x) = σ(x[feature] * weight + bias)
  P(right | x, node) = gate(x)
  P(left  | x, node) = 1 - gate(x)

葉の到達確率:
  π_leaf(x) = Π_{node on path} P(direction | x, node)

予測: F(x) = Σ_leaf π_leaf(x) * W_leaf  （W_leaf: 葉の出力ベクトル）

学習: SGD + バックプロパゲーション (PyTorchなし → Numpy自動微分)
  損失: 分類なら -Σ_leaf π_leaf(x) * log(softmax(W_leaf))_[y]
       回帰なら MSE

実装:
  SoftDecisionTree: 1本の木。fit()でNumpy-SGD。
  SoftDecisionForest: n_estimators本のSoftDecisionTreeをjoblibで並列学習。
```

---

# 12. forests/linear_tree.py の設計

```
LinearTree:
  各葉でOLS線形モデルを保持。
  木構造: CARTで分割探索（通常MSEベース）
  葉: LinearRegression（OLS）を学習
  predict(x): 木をトラバース→葉の線形モデルで予測

LinearForest:
  n本のLinearTreeのBootstrapアンサンブル。
  predict: 全木の予測の平均。

LinearBoost:
  順番に LinearTree を学習（残差ブースティング）:
  r_0 = y
  For m = 1..M:
    tree_m.fit(X, r_m)
    r_{m+1} = r_m - lr * tree_m.predict(X)
  predict: lr * Σ_m tree_m.predict(X)
```

---

# 13. forests/rulefit.py の設計

```
参考: Friedman & Popescu (2008). Predictive Learning via Rule Ensembles.
      Annals of Applied Statistics.

アルゴリズム:
  1. n_estimators本の浅い木（max_depth=2〜4）を学習
  2. 全木の全内部ノードからルールを抽出:
     rule = 「x[f1] <= t1 AND x[f2] > t2 AND ...」
  3. 各サンプルについてルールの充足 (0/1) を特徴量化
  4. 元の特徴量 + ルール特徴量 を結合
  5. Lasso（座標降下法）でスパース線形モデルを学習

実装の要点:
  - ルール抽出: 全ノードのパスを再帰的に列挙
  - 重複ルールは削除
  - get_rules(top_n): 重要なルールを返す
```

---

# 14. forests/bernoulli_rf.py の設計

```
参考: Denil et al. (2014). Narrowing the Gap: Random Forests In Theory and In Practice. ICML.

BernoulliRandomForest:
  各ノードで特徴量をBernoulli(feature_prob)で独立にサンプリング。
  feature_probが全特徴量に対して独立にBernoulli分布から選ぶ点が
  通常のmax_features=sqrt(p)と異なる。

  "最低1変数が選ばれることを保証するリトライロジック"が必要。
```

---

# 15. forests/grf.py の設計

```
参考: Athey, S., Tibshirani, J., & Wager, S. (2019).
      Generalized Random Forests. Annals of Statistics.

コアアイデア: 通常のRFは平均を推定するが、GRFは任意のモーメント条件
  E[ψ(Y, θ(x), ν(x)) | X=x] = 0 を解く。

adaptive neighborhood weight:
  α_i(x) = (1/n_trees) * Σ_tree [1(i ∈ leaf(x)) / |leaf(x)|]
  
実装:
  GeneralizedRandomForest: θ(x) = Σ_i α_i(x) * Y_i  （mean estimation）
  QuantileForest: θ(x) の代わりに α_i(x) で重み付けしたY_iのquantile
  CausalForest:
    分割基準: 通常のGainではなくτ(x)の異質性を最大化するGain
    τ(x) = Σ_i α_i(x) * (Y_i - Ȳ) * (W_i - W̄) / Σ_i α_i(x) * (W_i - W̄)^2
    ate() → τ̄ = (1/n) * Σ_i τ(X_i)
```

---

# 16. forests/kernel_forest.py の設計

```
参考: Ustimenko & Prokhorenkova (2022). 
      Random Kernel Forests. IEEE TNNLS. (DOI: 10.1109/TNNLS.2022.3156958)

RandomKernelForest:
  各ノード分割を RFF（Random Fourier Features）で近似したRBFカーネル空間で行う。
  
  RFF: z(x) = sqrt(2/D) * cos(Wω * x + b)
    ωd ~ N(0, γ)
    b  ~ Uniform(0, 2π)
  
  分割探索:
    zを計算し、その1次元射影上でSVM的な最大マージン分割を行う。
    具体的には: w_svm = Σ_i (y_i - ȳ) * z_i を分割方向とする。
    threshold: 中央値 or 最大Gain点

  predict: 通常のCART予測と同じ（構造だけ変わる）
```

---

# 17. forests/extras.py の設計

## IsolationForest

```
アルゴリズム (Liu et al. 2008):
  完全ランダムな木でランダム分割。
  異常値は少ない分割でisolateされる（パスが短い）。
  
  anomaly_score(x) = 2^{-E[h(x)] / c(n)}
  c(n) = 2*H(n-1) - 2(n-1)/n  ← 二分木の平均パス長

属性:
  score_samples(X) → [0,1] (高いほど異常)
  predict(X) → {-1, +1} (contamination割合でthreshold)
```

## QuantileRegressionForest (Meinshausen 2006)

```
各葉のY値を記録し、予測時に重み付きempirical quantileを計算。
predict(X, quantile=0.5): 
  leaf_ids = apply(X)  → (n, n_trees)
  For each test point x:
    training Y values in same leaf across all trees → weighted quantile
```

## RandomSurvivalForest (Ishwaran et al. 2008)

```
分割基準: log-rank test statistic（生存データ用）
  LR = Σ_time (d_Lj - n_Lj * d_j/n_j)^2 / (n_Lj/n_j * (1-n_Lj/n_j) * d_j * (n_j-d_j)/(n_j-1))

fit(X, t, e):  t=event時間, e=イベントフラグ(1=イベント発生)
predict(X): cumulative hazard function の平均
predict_cumhazard(X) → (n, len(unique_times_))
```

## MondrianForest (Lakshminarayanan et al. 2014)

```
各木: Mondrian Process による階層的分割
  - 各次元で独立に分割候補を生成（Poisson点過程）
  - 分割コスト ~ Exp(λ * range(dimension))

online learning: 新データが来たら木を更新可能
fit(X, y, n_classes): バッチ学習版
predict/predict_proba: Mondrian Kernelにより滑らかな確率
```

---

# 18. forests/boosting.py の設計

```
参考: Friedman, J.H. (2001). Greedy function approximation: A gradient boosting machine.
      Annals of Statistics, 29(5), 1189-1232.

GradientBoostedRegressor:
  損失: MSE → negative gradient = y - F(x)
  
  アルゴリズム:
    F_0 = mean(y)
    For m = 1..M:
      r = y - F(x)   ← 残差 = MSEの負勾配
      tree.fit(X_sub, r)  ← subsample ∈ (0, 1] でStochastic GBM
      各葉でγ_leaf = mean(残差 in 葉)  ← 葉ごとの最適ステップ
      F(x) += lr * tree.predict(x)
  
  staged_predict(X): ジェネレータ、各ステージ後の予測を yield

GradientBoostedClassifier (多クラス):
  損失: 多項ロジスティック（softmax cross entropy）
  K本の木を1ステージで学習（one-vs-all）
  負勾配: y_k - softmax(F(x))_k
  葉の重みスケーリング（Friedman 2001 Eq. 19）:
    γ_{jk} = (K/(K-1)) * Σ_j r_{ijk} / Σ_j p_{ijk}(1-p_{ijk})
```

---

# 19. forests/deep_forest.py の設計

```
参考: Zhou, Z.H., & Feng, J. (2017). Deep Forest. IJCAI. arXiv:1702.08835.

DeepForest (Cascade Forest部分):
  
  各レベル:
    n_forests_per_level 個のRandomForestを学習
    各ForestはStratifiedKFold交差検証でOOF（Out-of-Fold）確率ベクトルを生成
    → (n, n_forests * K) の拡張特徴量
  
  次レベルの入力: [元の特徴量, 全Forestのproba] を concat
  
  早期停止: OOF accuracyが min_improvement 以上改善しない場合停止
  
  predict_proba: 最終レベルの全ForestのprobaをStack→平均
```

---

# 20. forests/conformal.py の設計

```
参考: Vovk et al. (2005). Algorithmic Learning in a Random World. Springer.
      Romano et al. (2019). Conformalized Quantile Regression. NeurIPS.

ConformalForestRegressor:
  アルゴリズム (Split-Conformal):
    1. データを D_tr（80%）と D_cal（20%）に分割
    2. D_tr でRandomForestを学習
    3. D_cal で nonconformity score = |y - ŷ| を計算
    4. q̂ = (1-α)(1 + 1/n_cal) 分位点の s_i 値
    5. 予測区間: [ŷ(x) - q̂, ŷ(x) + q̂]
  
  カバレッジ保証: P(y ∈ C(x)) ≥ 1 - α （マルジナル）
  
  メソッド:
    predict_interval(X) → (n, 2)
    coverage_on(X, y) → float

ConformalForestClassifier:
  nonconformity score = 1 - ŷ[y_true]
  予測集合: C(x) = {k : ŷ_k ≥ 1 - q̂}
  
  メソッド:
    predict_set(X) → list of arrays
    coverage_on(X, y) → float  ← P(y ∈ C(x))
```

---

# 21. forests/embedding.py の設計

```
TotallyRandomTreesEmbedding:
  完全ランダム木（ラベルなし）で各サンプルを葉に割り当て。
  特徴量: one-hot(leaf_id) を全木でconcat → スパースバイナリ行列
  
  fit(X): 各木でランダム分割（特徴量とthresholdを完全ランダムに選択）
  transform(X) → scipy.sparse.csr_matrix (n, Σ n_leaves_per_tree)
  
  用途: 教師なし特徴学習→downstream分類・クラスタリング

FuzzyDecisionTree:
  各内部ノードで fuzzy membership を使用:
    μ_right(x) = σ((x[feature] - threshold) / β)  ← β: fuzzy bandwidth
  
  サンプルは全葉に確率的に割り当てられ、予測は加重平均:
    F(x) = Σ_leaf π_leaf(x) * v_leaf
    π_leaf = 各内部ノードのμの積
  
  訓練: CARTと同じ構造探索（ハード分割）、予測のみソフト
```

---

# 22. forests/similarity.py の設計

```
RFSimilarity:
  近傍行列の計算 (Breiman 2001, §8):
  S[i,j] = (1/n_trees) * Σ_tree 1[leaf_id(x_i) == leaf_id(x_j)]
  
  fit_transform(X) → (n, n) 類似度行列 [0, 1]
  transform(X) → (n_test, n_train) 類似度行列
  
  実装の最適化: 木ごとにapply()でleaf_idを取得し、行列演算で比較

RFKernel:
  mode: "raw"    → S（そのまま）
        "cosine" → S(x,z) / sqrt(S(x,x) * S(z,z))
        "rbf"    → exp(-gamma * (1 - S))
  
  fit_transform(X) → カーネル行列 K (n, n)
  transform(X) → (n_test, n_train)
  
  典型的な使い方: SVC(kernel="precomputed").fit(K_train, y)
```

---

# 23. forests/builder.py の設計

## ForestBuilder の引数解決優先順位

```
1. soft_tree=True          → SoftDecisionForest
2. generalized_target="causal" → CausalForest
3. generalized_target="quantile" → QuantileForest
4. linear_leaf=True        → LinearForest
5. monotone_constraints    → MonotonicConstrainedForest
6. linear_features         → LinearConstrainedForest
7. rotation=True           → RotationForest
8. random_rotation=True    → RandomRotationForest
9. split_type="kernel"     → RandomKernelForest
10. split_type="oblique" + sparse_projection=True → SPORF
11. split_type="oblique"   → ObliqueForest
12. variable_reuse_penalty > 0 → VariablePenaltyForest
13. leaf_regularization != "none" → LeafWeightRegularizedForest
14. bootstrap="bernoulli"  → BernoulliRandomForest
15. デフォルト             → RandomForestClassifier / RandomForestRegressor
```

## IncompatibilityWarning の条件

```python
_INCOMPATIBILITY_RULES = [
    {"conditions": {"rotation": True, "split_type": "oblique"},
     "message": "rotation と oblique は機能重複"},
    {"conditions": {"rotation": True, "random_rotation": True},
     "message": "両方の回転を同時に指定"},
    {"conditions": {"soft_tree": True, "linear_leaf": True},
     "message": "学習方式が競合"},
    {"conditions": {"soft_tree": True, "split_type": "oblique"},
     "message": "分割の哲学が競合"},
    {"conditions": {"generalized_target": "causal", "soft_tree": True},
     "message": "CausalForestはCARTベース前提"},
    {"conditions": {"rotation": True, "sparse_projection": True},
     "message": "PCA変換とスパース射影が競合"},
]
```

---

# 24. テスト設計

```
tests/conftest.py:
  fixtures: iris, diabetes, binary, regression_data, survival_data

tests/test_cart.py:
  TestCARTClassifier: 10テスト
    - fit_predict, accuracy, predict_proba, apply, depth_limit,
      entropy_criterion, min_samples_leaf, get_n_leaves,
      not_fitted_error, max_features_sqrt
  TestCARTRegressor: 5テスト
    - fit_predict, mse_decreases_with_depth, apply, mae_criterion,
      min_impurity_decrease

tests/test_random_forest.py:
  TestRandomForestClassifier: 6テスト
  TestRandomForestRegressor: 2テスト
  TestExtraTrees: 3テスト

tests/test_builder.py:
  TestForestBuilderClassification: 9テスト（全アルゴリズムのfit/score）
  TestForestBuilderRegression: 5テスト
  TestIncompatibilityWarnings: 4テスト（pytest.warns使用）

tests/test_extras.py:
  TestIsolationForest: 3テスト
  TestQuantileRegressionForest: 2テスト
  TestRandomSurvivalForest: 2テスト
  TestMondrianForest: 2テスト
```

---

# 25. 既知の実装上の注意点

```
1. sklearn 1.6+ MRO問題:
   BaseTree は必ず BaseEstimator から継承すること。
   check_is_fitted() の代わりに hasattr(self, 'root_') を使うこと。
   理由: check_is_fitted → get_tags → __sklearn_tags__ のMROで
         abc.ABC と BaseEstimator の多重継承が競合する。

2. NumPy 1.25+ 非推奨警告:
   0次元配列からfloat()で変換すると DeprecationWarning が出る。
   代わりに .item() を使うこと:
   float(node.value) → node.value.item()

3. pyproject.toml のbuild-backend:
   "setuptools.backends.legacy:build" は古い pip では動かない。
   "setuptools.build_meta" を使うこと。

4. Windows + PowerShell:
   tail コマンドは存在しない。パイプ先を Write-Host などにすること。

5. Parallel fitting:
   joblib.Parallel(n_jobs=n_jobs)(delayed(func)(seed, ...) for seed in seeds)
   各ツリーに固有の seed を渡すことで再現性を確保。
```

---

# 26. アルゴリズム全一覧・論文対応表

| クラス | 論文 | 発表年 | ファイル |
|---|---|---|---|
| CARTClassifier/Regressor | Breiman et al. CART | 1984 | cart.py |
| RandomForestClassifier/Regressor | Breiman. Machine Learning | 2001 | random_forest.py |
| ExtraTreesClassifier/Regressor | Geurts et al. Machine Learning | 2006 | random_forest.py |
| VariablePenaltyForest | カスタム | — | regularized.py |
| LeafWeightRegularizedForest | カスタム | — | regularized.py |
| MonotonicConstrainedForest | カスタム | — | constrained.py |
| LinearConstrainedForest | カスタム | — | constrained.py |
| ObliqueForest | Murthy et al. JAIR | 1994 | oblique.py |
| RotationForest | Rodriguez et al. TPAMI | 2006 | oblique.py |
| RandomRotationForest | カスタム | — | oblique.py |
| SPORFClassifier/Regressor | Tomita et al. JMLR | 2020 | sporf.py |
| RegularizedGreedyForest | Johnson & Zhang. TPAMI | 2014 | rgf.py |
| SoftDecisionTree/Forest | Frosst & Hinton. arXiv | 2017 | soft_tree.py |
| LinearTree/Forest/Boost | Potts & Sammut. ECML | 2005 | linear_tree.py |
| RuleFit | Friedman & Popescu. Ann.Appl.Stat | 2008 | rulefit.py |
| BernoulliRandomForest | Denil et al. ICML | 2014 | bernoulli_rf.py |
| GeneralizedRandomForest | Athey et al. Ann.Stat | 2019 | grf.py |
| QuantileForest | Athey et al. Ann.Stat | 2019 | grf.py |
| CausalForest | Wager & Athey. JASA | 2018 | grf.py |
| RandomKernelForest | Ustimenko & Prokhorenkova. TNNLS | 2022 | kernel_forest.py |
| IsolationForest | Liu et al. ICDM | 2008 | extras.py |
| QuantileRegressionForest | Meinshausen. JMLR | 2006 | extras.py |
| RandomSurvivalForest | Ishwaran et al. Ann.Appl.Stat | 2008 | extras.py |
| MondrianForest | Lakshminarayanan et al. NIPS | 2014 | extras.py |
| GradientBoostedRegressor | Friedman. Ann.Stat | 2001 | boosting.py |
| GradientBoostedClassifier | Friedman. Ann.Stat | 2001 | boosting.py |
| DeepForest | Zhou & Feng. IJCAI | 2017 | deep_forest.py |
| ConformalForestRegressor | Vovk et al. Springer | 2005 | conformal.py |
| ConformalForestClassifier | Angelopoulos & Bates. ICLR | 2023 | conformal.py |
| TotallyRandomTreesEmbedding | Geurts et al. Machine Learning | 2006 | embedding.py |
| FuzzyDecisionTree | Umano et al. IEEE Fuzzy Sys | 1994 | embedding.py |
| RFSimilarity | Breiman. Machine Learning §8 | 2001 | similarity.py |
| RFKernel | Breiman. Machine Learning §8 | 2001 | similarity.py |
| ForestBuilder | カスタム | — | builder.py |
