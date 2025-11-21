
# 全く新しい実用的なブラックボックス最適化アルゴリズム: PFSBO

## 1. 概要

この文書では、既存の代表的な枠組み（BO, CMA-ES, NES, TPE, SMAC など）とは振る舞いが異なる、
**実装して試せるレベルの新しいブラックボックス最適化アルゴリズム案**を説明する。

提案手法は、

- 内側で: **多目的最適化（パレート前線）に基づく集団型探索**
- 外側で: **複数の探索レジーム（局所探索・大域探索・モデルベース提案など）の使用頻度をマルチアームドバンディットで制御**

という二階構造を持つ。ここではこのアルゴリズムを

> **PFSBO: Pareto-Front Scheduled Black-box Optimization**

と呼ぶ。

---

## 2. 問題設定

- 探索空間: \( X \subseteq \mathbb{R}^d \)（連続空間。離散への拡張も可能）
- 目的関数: \( f: X \to \mathbb{R} \)  
  - 評価コストが高く、勾配が利用できないブラックボックス
- 評価予算: 最大評価回数 \( T \)

目的は

\[
\min_{x \in X} f(x)
\]

とする。

---

## 3. アルゴリズムのアイデア

### 3.1 多目的化

通常の単一目的ブラックボックス最適化では、`f(x)` だけで点を評価する。
ここでは発想を変え、「探索の質」に関する補助的な目的を導入し、
**多目的最適化問題として扱う**。

代表的な構成例:

1. 実目的

   \[
   F_1(x) = f(x)
   \]

2. 多様性目的

   集団 \( P = \{x_1, \dots, x_n\} \) に対して、各点 `x` の最近傍距離を用いる。

   \[
   F_2(x) = - \min_{y \in P\setminus \{x\}} \|x - y\|
   \]

   多様な点ほど「良い」とみなしたいので、距離にマイナスをつけて
   「小さいほど良い」目的（最小化）として扱う。

3. （任意）不確実性目的

   簡易サロゲートモデル \( m \) をオンザフライで学習できるなら、モデル不確実性に基づく

   \[
   F_3(x) = - \text{uncertainty}_m(x)
   \]

   を導入できる（例: ランダムフォレストの木間分散、ガウス過程の分散など）。

結果として、多目的関数

\[
F(x) = (F_1(x), F_2(x), F_3(x), \dots)
\]

を最小化する問題になる。

---

### 3.2 パレート前線とランク付け

候補集合 \( P \) に対して、非支配ソートで **パレートランク** を定義する。

- ランク1: ほかのどの点にも支配されない点（パレート前線）
- ランク2: ランク1を除いた集合の中で非支配な点
- …

さらに、同一ランク内では `crowding distance` のような密度指標を用いて二次的なスコア付けを行う。
これにより、多目的上の良さと多様性の両方を考慮した**生存選択**が可能になる。

---

### 3.3 探索レジーム＋マルチアームドバンディット

次に、新しいサンプル点（子個体）の生成方法を複数定義する。例:

1. `LocalGaussian`: パレート前線上の良い点の近傍にガウス変異
2. `GlobalUniform`: 探索空間全体からの一様サンプリング
3. `ModelSuggest`: 簡易サロゲートモデルに基づく提案（EI 風など）
4. `Crossover`: ランダムに選んだ 2 点の線形結合など

これらをアーム \( a \in \{1, \dots, K\} \) とみなし、
**各アームが生み出したサンプルの「有用度」を報酬としてマルチアームドバンディットで学習**する。

#### 報酬設計の例

アーム `a` によって生成された点 `x_new` について:

- `x_new` がランク1に入ったか？
- 既知の最良値 \( f_{best} \) を更新したか？
- 多様性がどれだけ増したか？

などに基づいて、報酬 \( r_t \) を設計できる。例:

\[
r_t = \alpha \cdot \mathbb{I}[f(x_{new}) < f_{best}^{(t-1)}]
    + \beta \cdot \max(0, \text{rank}_{old} - \text{rank}_{new})
    + \gamma \cdot \Delta \text{diversity}
\]

バンディットアルゴリズムとしては、UCB1 や Thompson Sampling などの標準手法を利用する。

---

## 4. 形式的定義

### 4.1 多目的関数

- 実目的: \( F_1(x) = f(x) \)
- 多様性目的（例）:

  \[
  F_2(x) = - \min_{y \in P\setminus \{x\}} \|x - y\|
  \]

- 不確実性目的（任意）:

  \[
  F_3(x) = - \text{uncertainty}_m(x)
  \]

これらをまとめて

\[
F(x) = (F_1(x), F_2(x), F_3(x), \dots)
\]

を最小化する。

### 4.2 パレートランクと密度

- 非支配ソートにより、各点にランク `rank(x)` を割り当てる。
- 同ランク内では crowding distance などで密度を評価し、
  選択（サバイバル）時に多様性を維持する。

### 4.3 探索レジームのバンディット制御

- 各レジームをアーム `a` とみなし、
- アーム `a` が使われた回数 `N_a[a]` と推定報酬平均 `Q_a[a]` を保持
- UCB1 の例:

  \[
  a_t = \arg\max_a \left( Q_a[a] + c \sqrt{\frac{\log t}{N_a[a] + 1}} \right)
  \]

  などでアーム選択を行い、得られた報酬で `Q_a[a]` を逐次更新する。

---

## 5. PFSBO の疑似コード

### 5.1 初期化

```pseudo
Input:
    f        // ブラックボックス関数
    X        // 探索空間 (制約つきでもOK、投影で対応)
    T        // 最大評価回数
    n_init   // 初期サンプル数
    N        // 各世代の母集団サイズ
    K        // 探索レジームの数
Output:
    x_best, f_best

D = {}  // 評価履歴: (x, f(x)) の集合

// 初期サンプル生成
P = {}
for i = 1 .. n_init:
    x_i ~ InitialSampler(X)      // 一様 or ラテン超方格など
    y_i = f(x_i)
    D.add((x_i, y_i))
    P.add(x_i)

t = n_init

// バンディットパラメータの初期化
for a = 1..K:
    N_a[a] = 0      // アーム使用回数
    Q_a[a] = 0.0    // 推定報酬平均

(x_best, f_best) = argmin_{(x, y) in D} y
```

---

### 5.2 主ループ

```pseudo
while t < T:

    // 1. 多目的値 F(x) の更新
    compute F_i(x) for all x in P using D
        // F1: f(x)
        // F2: -nearest_neighbor_distance
        // F3: (optional) -uncertainty from surrogate model

    // 2. 非支配ソート
    rank, crowding = NonDominatedSorting(P, F)

    // 3. 生存選択 (elitism)
    P = SelectByRankAndCrowding(P, rank, crowding, N)

    // 4. サロゲートモデル更新 (任意)
    m = FitSurrogate(D)   // 軽量モデルでOK。使わないなら skip

    // 5. 新しい候補点の生成
    new_points = {}
    while |new_points| < N:
        // 5-1. バンディットによる探索レジーム選択
        a = SelectArmByUCB(Q_a, N_a, total_plays = t)

        // 5-2. 選ばれたレジームで新しい点を生成
        x_parent = SampleParentFromFront(P, rank)
        x_new = GenerateCandidate(a, x_parent, P, m, X)

        // 5-3. 制約外なら投影 or リジェクト
        x_new = ProjectToFeasible(x_new, X)

        // 5-4. 評価
        y_new = f(x_new)
        D.add((x_new, y_new))
        new_points.add(x_new)
        t = t + 1

        // 5-5. ベスト更新
        if y_new < f_best:
            f_best = y_new
            x_best = x_new

        // 5-6. 報酬計算用の一時ランク更新
        tempP = P ∪ {x_new}
        F_with_x_new = UpdateFValuesWithNewPoint(F, x_new, tempP, D, m)
        temp_rank, _ = NonDominatedSorting(tempP, F_with_x_new)

        rank_new = temp_rank[x_new]
        reward = ComputeReward(f_best, y_new, rank_new, rank, P, x_new)

        // 5-7. バンディット更新
        N_a[a] = N_a[a] + 1
        Q_a[a] = Q_a[a] + (reward - Q_a[a]) / N_a[a]

        if t >= T:
            break

    // 6. 世代更新
    P = P ∪ new_points
end while

return x_best, f_best
```

---

### 5.3 探索レジーム `GenerateCandidate` の例

```pseudo
function GenerateCandidate(a, x_parent, P, m, X):

    switch a:

        case LocalGaussian:
            // 局所ガウス変異
            σ = AdaptedScaleFromPopulation(P)   // P の分散などからスケールを推定
            ε ~ N(0, σ^2 I)
            return x_parent + ε

        case GlobalUniform:
            // 探索空間全体から一様サンプル
            return SampleUniform(X)

        case ModelSuggest:
            // サロゲートモデルを使用 (任意)
            // 例: ランダム候補集合 C に対して EI 風スコアを計算し、最大のものを返す
            C = {c_1, ..., c_M} ~ Uniform(X)
            return argmax_{c in C} Acquisition(c; m)

        case Crossover:
            // 2点の線形結合
            x2 = RandomChoice(P)
            λ ~ Uniform(0, 1)
            return λ * x_parent + (1 - λ) * x2
```

---

## 6. 実用上の性質

### 6.1 メリット

1. **単一スカラーへの還元を避ける**  
   - 実目的だけでなく、多様性や不確実性を明示的な目的として扱うため、
     局所解に早期に収束しにくい。

2. **探索戦略の自動適応**  
   - 「どの程度グローバル探索 vs ローカル探索を行うか」を
     バンディットがオンラインで学習するため、ユーザが手動で調節する負担が軽減される。

3. **既存手法との統合が容易**  
   - 各レジームの中身として、CMA-ES 風変異や、ベイズ最適化の獲得関数、
     Nelder–Mead 的局所探索などをそのまま利用できる。
   - PFSBO はそれらを切り替える「メタ戦略」として働く。

### 6.2 計算量とスケーラビリティ

- 各世代の主な計算コストは非支配ソート（通常 \( O(M^2) \), M: 集団サイズ）。
- 評価コストが支配的な高価な実験（数秒〜数分/評価）では、
  内部計算コストは十分小さいことが多い。
- 集団サイズは実用上、数十〜数百程度に抑えるのが無難。

---

## 7. シンプル実装案

最初のプロトタイプを簡単に実装する場合は、以下のように簡略化できる。

1. 多目的は `F1 = f(x)` と `F2 = -nearest_neighbor_distance` の 2 つのみ。
2. サロゲートモデルは **無し** とする。
3. 探索レジームは以下の 3 種類:
   - `LocalGaussian`
   - `GlobalUniform`
   - `Crossover`
4. バンディットは UCB1 のみ。

この構成なら、Python で数百行程度のコードで十分実装可能である。

---

## 8. 拡張アイデア

- **雑音の多い評価値への対応**  
  - 同一点の再評価、信頼区間によるロバストな報酬設計など。

- **制約付き最適化**  
  - 制約違反量を別目的として追加し、多目的最適化で扱う。

- **高次元問題**  
  - 探索レジーム内でサブスペース探索やランダム射影を用いる。

- **並列化**  
  - 集団型アルゴリズムなので、評価バッチを自然に並列実行できる。

---

## 9. まとめ

PFSBO (Pareto-Front Scheduled Black-box Optimization) は、

- 多目的な観点（目的値 + 多様性 + 不確実性など）で候補点を評価し、
- パレート前線を用いて集団を維持しつつ、
- 複数の探索レジームの使用頻度をマルチアームドバンディットで適応的に制御する、

という二階構造のブラックボックス最適化アルゴリズムである。

既存の最適化アルゴリズムを「探索レジーム」として組み込み、
PFSBO をその上にかぶせることで、探索戦略を自動的に切り替えながら最適化を進めることができる。
