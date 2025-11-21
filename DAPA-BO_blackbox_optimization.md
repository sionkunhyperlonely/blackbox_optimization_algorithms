# DAPA-BO: Discrete Adaptive Patch Aggregation for Black-box Optimization

## 概要

**DAPA-BO (Discrete Adaptive Patch Aggregation for Black-box Optimization)** は、
評価コストが高いブラックボックス関数の最適化を目的とした、
**動的な空間分割＋局所モデル＋バンディット** を組み合わせた新しいアルゴリズムです。

主な特徴は次の3点です。

1. 探索空間を **パッチ（タイル）** に分割し、サンプリングが進むにつれて
   **有望な領域だけ細かく分割（refine）** していく。
2. 各パッチごとに **局所モデル（ローカルレグレッサ＋統計量）** を持ち、
   パッチ内部はモデルベース最適化（例：EI最大化）で効率的に探索する。
3. パッチ単位で **バンディット（UCB 型）のスコア** を計算し、
   「どのパッチを次に調べるか」と「そのパッチのどこを調べるか」を分離して決める
   2階構造になっている。

これにより、

- 高次元空間でも「局所的には低次元で滑らか」という問題に適応しやすい
- 有望な領域に計算資源を集中しつつ、未探索領域も一定程度カバーできる

という利点があります。

---

## 問題設定

最大化問題を考えます。

\[
\max_{x \in \mathcal{X}} f(x)
\]

- ドメイン: \( \mathcal{X} \subset \mathbb{R}^d \)（例：\([0,1]^d\) に正規化）
- 関数: \( f \) はブラックボックス
  - 値は \( y = f(x) + \epsilon \) としてのみ観測可能
  - \( \epsilon \) はノイズ（0でもよい）
- 微分不可能・ノイズありでも良いが、
  - **局所的にはある程度なめらか（連続、あるいはリプシッツ的）** であることを仮定。

---

## アルゴリズム全体像

### 高レベルな流れ

1. **初期パッチ分割**
   - 探索空間 \(\mathcal{X}\) を粗くパッチ分割する。
     - 例：\([0,1]^d\) を各次元 2 分割 → \(2^d\) 個のパッチ。
   - 各パッチ \(P_k\) に以下を持たせる：
     - 評価点の集合: \( \mathcal{D}_k = \{(x_i, y_i)\} \)
     - ローカルモデル: \( M_k \)（線形回帰・小さなGP・ランダムフォレストなど）
     - パッチの価値指標: \( V_k \)
     - サンプル数: \( n_k \)

2. **反復ステップ**
   1. **パッチ選択**
      - 各パッチの価値指標 \(V_k\) を、UCB 的なスコアで評価：
        \[
        S_k = \hat{\mu}_k + \beta_t \cdot \hat{\sigma}_k + \gamma_t \cdot \text{UncertaintyTerm}_k
        \]
        - \(\hat{\mu}_k\): パッチ内の最大観測値 or モデル最大予測値
        - \(\hat{\sigma}_k\): パッチ内性能のばらつき or モデル不確実性
        - UncertaintyTerm: サンプル数が少ないパッチへの探索ボーナス（例：\(1/\sqrt{n_k+1}\)）
      - \(S_k\) が最大のパッチ \(P_{k^*}\) を選択。

   2. **パッチ内部候補点生成**
      - 選ばれたパッチ \(P_{k^*}\) の内部で候補点 \(x^{(t)}\) を生成：
        - exploitation：ローカルモデル \(M_{k^*}\) 上で Expected Improvement (EI) などを最大化
        - exploration：パッチ内で一様サンプリング or ラテン超方格サンプリング
      - 例：
        - 70% の確率でローカルEI最大化
        - 30% の確率でパッチ内一様乱数

   3. **評価**
      - \(y^{(t)} = f(x^{(t)})\) を観測し、対応するパッチのデータ集合に追加。

   4. **ローカルモデル更新**
      - パッチ \(P_{k^*}\) のデータ \(\mathcal{D}_{k^*}\) を使って
        ローカルモデル \(M_{k^*}\) を再学習または増分更新。
      - \(\hat{\mu}_{k^*}, \hat{\sigma}_{k^*}, n_{k^*}\) を更新。

   5. **パッチの分割・統合（Refinement / Pruning）**
      - 分割条件（例）：
        - \(n_{k^*} \ge n_{\text{split}}\)
        - かつ \(\hat{\sigma}_{k^*}\) が閾値以上、またはパッチ体積がまだ大きい
      - 条件を満たしたら \(P_{k^*}\) をサブパッチに分割し、
        各サブパッチにローカルモデルを再構築する。
      - 統合・プルーニング条件（例）：
        - パッチの上限推定値が、全体のベスト値より十分に低い
        - → そのパッチを探索候補から外す（active=false）

3. **停止条件**
   - 評価回数が予算 \(T\) に到達
   - あるいはパッチ全体の不確実性が閾値以下
   - 出力：全評価点の中で最大値を与える \(x\)

---

## 擬似コード

```pseudo
Input:
  - Domain X ⊂ R^d
  - Budget T (max number of evaluations)
  - Initial partition level L0 (e.g., 1 split per dimension)
  - Split threshold n_split
  - Hyperparameters β_t, γ_t schedules

Initialize:
  P = { P_1, ..., P_K }    // initial patches (e.g., a regular grid)
  For each patch P_k:
    D_k = ∅
    M_k = null
    μ_k = -∞
    σ_k = +∞
    n_k = 0
    active_k = true

// Optional: initial space-filling
for t = 1,...,T_init:
  x_t = LatinHypercubeSample(X)
  y_t = f(x_t)
  find patch P_k containing x_t
  update_patch(P_k, x_t, y_t)

Main loop:
for t = T_init+1,...,T:

  // 1. Patch selection (bandit-like)
  candidates = { k | active_k = true }
  For each k in candidates:
    μ_k = max_y_in(D_k) or max_predicted_value(M_k)
    σ_k = std_of_y_in(D_k) or mean_model_uncertainty(M_k)
    U_k = μ_k + β_t * σ_k + γ_t * (1 / sqrt(n_k + 1))
  Choose k* = argmax_k U_k

  // 2. Within-patch candidate generation
  With probability p_exploit:
    x_t = argmax_x∈P_k* EI_Mk*(x)
  else:
    x_t = RandomUniform(P_k*)

  // 3. Evaluate
  y_t = f(x_t)

  // 4. Update local data and model
  D_k* = D_k* ∪ {(x_t, y_t)}
  n_k* = n_k* + 1
  Update local model M_k* using D_k*
  Update μ_k*, σ_k*, etc.

  // 5. Patch refinement / pruning
  if n_k* ≥ n_split and volume(P_k*) > min_volume:
    if σ_k* ≥ sigma_split_threshold:
      Subdivide P_k* into {P_k*_1, ..., P_k*_J}
      For each j:
        D_k*_j = { (x,y) ∈ D_k* | x ∈ P_k*_j }
        Train M_k*_j on D_k*_j
        Initialize μ_k*_j, σ_k*_j, n_k*_j
      active_k* = false  // parent patch is replaced by children

  if n_k* ≥ n_prune and (μ_k* + β_t * σ_k*) << global_best_y:
    active_k* = false   // prune unpromising region

Output:
  x_best = argmax_x,y over all collected data
```

---

## 既存手法との関係と違い

### ベイズ最適化（GP + EI/UCB）との比較

- 標準的なベイズ最適化は、
  - 1つのグローバルサロゲートモデル（例：全空間をカバーする GP）を使う。
  - 高次元・サンプル数増加時に、計算コストと表現力の両面で厳しくなる。
- DAPA-BO は：
  - **パッチごとに独立したローカルモデル \(M_k\)** を持つ。
  - 各モデルは小規模で済むため、計算負荷が分散される。
  - 有望なパッチだけ細かく分割するため、
    複雑な地形は局所的に精度を上げ、平坦な領域は粗いままにしておける。

### DIRECT, HOO などの空間分割型手法との比較

- DIRECT/HOO系も空間分割＋UCB 的な枠組みを持つが：
  - DAPA-BO では各パッチ内に **明示的なローカル回帰モデル** を持つのが特徴。
  - パッチ選択だけでなく、パッチ内の点選択もモデルベース最適化で行う。
- イメージとして：
  - 上位レベル：HOO/DIRECT に似た「どの領域を掘るか」選択
  - 下位レベル：BO/EI のような「その領域でどこを試すか」の最適化

### CMA-ES や進化戦略との比較

- CMA-ES は、1つあるいは少数の個体群の分布（ガウス分布など）を更新していく。
- DAPA-BO は、
  - 各パッチが「局所的な探索分布＋モデル」に相当し、
  - 複数の有望な谷（局所最適候補）を **並列に追跡** できる。
- マルチモーダルな関数において、
  - 1つの分布で全てをカバーするよりも柔軟に振る舞える。

---

## 実装上の具体的な選択肢

実用的かつ実装しやすい構成の一例：

- ドメイン：\([0,1]^d\) に線形スケーリングで正規化
- 初期パッチ：
  - 次元数がそれほど大きくない場合：各次元 2 分割 → \(2^d\) パッチ
  - 高次元の場合：まずは 1 分割（＝1パッチ）から開始し、サンプルが溜まり次第分割
- ローカルモデル \(M_k\)：
  - サンプル数が少ないうちは「平均 & 分散のみ」
  - \(n_k\) が閾値を超えたらランダムフォレスト or 小さい GP を構築
- パッチ内の候補点生成：
  - EI をランダム多点サンプリング＋最大値で近似
  - または簡単な局所最適化アルゴリズム（L-BFGS など）を利用
- パッチ選択 UCB スコア（簡易版）：
  \[
  U_k = \max_{(x,y) \in \mathcal{D}_k} y
        + \beta_t \cdot \mathrm{std}(\{y\})
        + \gamma_t \cdot \frac{1}{\sqrt{n_k + 1}}
  \]
- 分割戦略：
  - \(\sigma_k\) （ばらつき）が大きいパッチから優先的に分割
  - 分割方向：
    - パッチ内のサンプルで、
      もっとも変動が大きい次元を1つ選び、その次元だけを2分割
- プルーニング戦略：
  - パッチ上限 \(U_k\) がグローバルベスト \(y_{\text{best}}\) より
    \(\delta\) だけ小さい（例：\(U_k < y_{\text{best}} - \delta\)）なら active=false にする。

---

## 理論的性質（直観レベル）

- HOO系アルゴリズムと同様、
  - パッチ分割が進めば「真の最適解を含むパッチ」を十分小さく切り出せる。
- 各パッチ内をモデルベースで探索することで、
  - 同じ評価回数でも局所情報の活用効率が高くなることが期待される。
- 一方できちんとした収束レートの証明には、
  - バンディット理論＋モデル誤差（回帰誤差）を組み合わせた解析が必要であり、
  - これは独立した理論研究テーマとなる。

---

## 想定される適用分野

- 評価コストが高いシミュレーションベース最適化
  - CFD（流体解析）、FEM（構造解析）などの設計最適化
- 複雑なハイパーパラメータ最適化
  - 深層学習モデル、強化学習エージェントのハイパーパラメータ探索
- ロボティクス・制御系
  - 実機評価が高価な制御ポリシーのチューニング
- マルチモーダルな目的関数
  - 複数の有望な谷が存在するような問題設定

---

## まとめ

DAPA-BO は、

- 探索空間の **動的パッチ分割**
- パッチごとの **ローカルモデル**
- パッチ選択に対する **バンディット的 UCB スコア**

を組み合わせた、2階構造の新しいブラックボックス最適化アルゴリズムです。

**上位レベル**では「どのパッチを掘るか」を決め、  
**下位レベル**では「そのパッチのどこを掘るか」をモデルベースに決めることで、

- 高コストな評価を節約しつつ、
- マルチモーダル＆高次元な問題にも実用的に対応できることを狙っています。

この設計に基づき、Python などで実装し、
標準的なベイズ最適化や CMA-ES 等と比較実験することで、
具体的な性能評価やハイパーパラメータ設計の指針が得られます。
