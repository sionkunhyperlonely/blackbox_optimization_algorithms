# RLSO: Reinforced Local Subspace Optimizer
**全く新しい実用的なブラックボックス最適化アルゴリズムの設計メモ（Markdown 版）**

---

## 1. 問題設定

- 目的：  
  \[
    \min_{x \in \Omega \subset \mathbb{R}^d} f(x)
  \]
- 前提：
  - \( f(x) \) はブラックボックス
  - 導関数情報なし
  - 評価コストが高い（1回の評価が重い）
  - ノイズを含む場合もある
- 制約：
  - まずは単純なボックス制約 \( \Omega = [l, u] \) を想定
  - 一般制約はペナルティや別モデルで拡張可能

---

## 2. アルゴリズム概要：RLSO

### 2.1 名前とコンセプト

- 名称：**RLSO – Reinforced Local Subspace Optimizer**
- 直感的なアイデア：
  1. 「良さそうな領域」を **アンカー（anchor）** として複数保持
  2. 各アンカーの近傍に対して、**低次元の有効部分空間（Active Subspace）** を推定
  3. その部分空間上で、**信頼領域 + 局所二次モデル** に基づくサロゲート最適化を実行
  4. 「どのアンカー（あるいはグローバル探索）を使うか」を **多腕バンディット** で選択

### 2.2 特徴のまとめ

- 高次元でも、実際に効いている方向だけを抽出して探索
- ローカル探索とグローバル探索の配分をバンディットで自動調整
- サロゲートは単純な二次回帰（リッジ回帰）で、実装が軽い
- 並列評価・リスタート戦略とも相性が良い

---

## 3. データ構造

### 3.1 評価履歴アーカイブ

- 評価履歴：
  \[
    \mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}
  \]
  - \( x_i \in \mathbb{R}^d \)：サンプル点
  - \( y_i = f(x_i) \)：評価値

### 3.2 アンカー

- アンカーを最大 \(K_{\max}\) 個まで保持
- 各アンカー \(k = 1,\dots,K\) について：
  - 中心：\( c_k \in \mathbb{R}^d \)
  - 部分空間基底：\( B_k \in \mathbb{R}^{d \times r_k} \)
    - 直交列
    - \( r_k \ll d \) （有効次元）
  - 信頼領域半径：\( \Delta_k > 0 \)
  - 近傍データインデックス集合：\( I_k \subset \{1,\dots,N\} \)
  - バンディット用スコア：\( s_k \in \mathbb{R} \)
  - 失敗カウンタ：\( f_k \in \mathbb{N} \)

### 3.3 グローバル探索腕

- バンディットの 1 腕として、グローバル探索用の「擬似アンカー」を持つ
- 中心や部分空間は持たず、「サンプリング戦略」だけを表現する

---

## 4. アルゴリズム各ステップ

### 4.1 初期化

1. ラテンハイパーキューブなどで \( n_0 \) 点サンプリングし、評価：  
   \[
     \{(x_i, y_i)\}_{i=1}^{n_0}
   \]
2. その中から良好な \( K_0 \) 点を選び、各点をアンカー中心 \( c_k \) とする
3. 各アンカーに対して：
   - 初期部分空間 \( B_k \)
     - 近傍点の PCA などで求めるか、単位ベクトル＋ランダム直交ベクトルで近似
   - 初期信頼領域半径 \( \Delta_k = \Delta_{\text{init}} \)
   - バンディット用スコア \( s_k = 0 \)、失敗カウンタ \( f_k = 0 \)

---

### 4.2 バンディットによるアンカー／腕の選択

- 腕の集合：
  - アンカー 1..K
  - グローバル探索腕 1本
- 各腕 \( j \) について：
  - 平均改善量推定 \( \hat{g}_j \)
  - 選択回数 \( n_j \)
- 例として、ソフトマックス + UCB 的ルール：
  \[
    p_j \propto \exp\left(\beta\, \hat{g}_j + \alpha \sqrt{\frac{\ln t}{n_j + 1}}\right)
  \]
  - \( t \)：全体のイテレーション
  - \(\alpha, \beta\)：ハイパーパラメータ
- 確率 \(p_j\) に従って腕 \( j \) をサンプリングして選ぶ

---

### 4.3 アンカーの局所モデル構築

アンカー \( k \) が選ばれたとする。

1. 近傍点集合の抽出：
   \[
     I_k = \{ i \mid \|x_i - c_k\|_2 \le R_k \}
   \]
2. 各点の部分空間への射影：
   \[
     z_i = B_k^\top (x_i - c_k) \in \mathbb{R}^{r_k}
   \]
3. 射影データ \( (z_i, y_i) \) に対して、**二次モデル**をリッジ回帰でフィット：
   \[
     m_k(z) = a + g^\top z + \tfrac{1}{2} z^\top H z
   \]
   - \( a \in \mathbb{R} \)：定数項
   - \( g \in \mathbb{R}^{r_k} \)：勾配近似
   - \( H \in \mathbb{R}^{r_k \times r_k} \)：ヘッセ近似
4. データ数が少なければ：
   - 線形モデルのみ（\( H = 0 \)）にフォールバック
   - それも不安定ならランダムサンプリングに切り替え

---

### 4.4 部分空間上の信頼領域サブ問題

- 信頼領域半径 \(\Delta_k\) のもとで：
  \[
    \min_{z \in \mathbb{R}^{r_k}} m_k(z) \quad \text{s.t. } \|z\|_2 \le \Delta_k
  \]
- 解き方の例：
  - \(H\) が正定値の場合：
    - 無制約解 \( z^* = -H^{-1} g \) を求め、\(\|z^*\|_2 \le \Delta_k\) ならそのまま
    - そうでなければ、半径 \(\Delta_k\) 上にクリップ
  - 一般の場合：トランケーテッド CG などで近似解
- \( z^* \) から元の空間の候補点：
  \[
    x_{\text{new}} = \Pi_\Omega \bigl( c_k + B_k z^* \bigr)
  \]
  - \(\Pi_\Omega\)：ボックス制約 \(\Omega\) への射影（単なるクリップ）

---

### 4.5 グローバル探索腕

グローバル探索腕が選ばれた場合：

- 一様ランダムサンプリング：
  \[
    x_{\text{new}} \sim \mathcal{U}([l,u])
  \]
- もしくは、「既存点のクラスタリングに基づいて、あまりサンプルされていない領域」を狙うサンプリング
- 得られた点が十分良ければ、新規アンカーの候補として扱う

---

### 4.6 評価と信頼領域の更新

1. 新しい点を評価：
   \[
     y_{\text{new}} = f(x_{\text{new}})
   \]
2. 予測改善（局所モデル上）：
   \[
     \hat{\Delta} = m_k(0) - m_k(z^*)
   \]
3. 実際の改善：
   \[
     \Delta = f_{\text{best, old}} - y_{\text{new}}
   \]
4. 信頼領域比：
   \[
     \rho = \frac{\Delta}{\hat{\Delta} + \varepsilon}
   \]
   - \(\varepsilon\)：0 除算防止の小さな定数
5. 信頼領域半径の更新ルール（例）：
   - \(\rho > \eta_1\) なら「成功」：
     \[
       \Delta_k \leftarrow \gamma_{\text{inc}} \Delta_k
     \]
   - \(\rho < \eta_0\) なら「失敗」：
     \[
       \Delta_k \leftarrow \gamma_{\text{dec}} \Delta_k
     \]
   - それ以外は半径そのまま

6. バンディットの更新：
   - 腕 \(k\)（あるいはグローバル腕）の平均改善 \(\hat{g}_k\) を移動平均などで更新
   - 成功回数・失敗回数を更新

7. アンカーのリスタート：
   - 連続失敗回数 \( f_k \) が閾値を超えたら、
     - そのアンカーを廃棄
     - 良好な点から新しいアンカーを作成

---

### 4.7 部分空間 \(B_k\) の更新（能動次元推定）

アンカー \(k\) の近傍点集合 \( I_k \) を用いて、  
「評価値がよく変動する方向」を推定する簡便な方法：

1. 各 \( i \in I_k \) について：
   - \( d_i = x_i - c_k \)
   - 正規化：
     \[
       \tilde{d}_i = \frac{d_i}{\|d_i\|_2}
     \]
   - 重み：
     \[
       w_i \propto |y_i - \bar{y}|
     \]
     - \(\bar{y}\)：近傍点の平均評価値
2. 加重共分散行列：
   \[
     C_k = \sum_{i \in I_k} w_i \tilde{d}_i \tilde{d}_i^\top
   \]
3. 固有分解：
   - \( C_k \) の固有値分解を行い、上位 \( r_k \) 個の固有ベクトルを列に並べて \( B_k \) を構成
   - これにより、「値が変わりやすい方向」に沿った部分空間を抽出

---

## 5. 擬似コード

```pseudo
Input: budget T, initial sample size n0, max anchors K_max
Initialize D with n0 Latin-hypercube samples
Select K0 best points -> anchors {c_k}
For each anchor k:
    initialize B_k, Δ_k, s_k, f_k, stats for bandit

for t = 1..T:
    # 1. choose arm (anchor or global)
    compute probs p_j for all arms j using bandit rule
    sample arm j ~ Categorical(p)

    if j is global arm:
        x_new <- global_sample(D)   # e.g. exploration strategy
        k_used <- "global"
    else:
        k <- j
        # 2. build local model around anchor k
        I_k <- {i : ||x_i - c_k|| <= R_k}
        if |I_k| < p_min:
            x_new <- random_around(c_k, Δ_k, B_k)
        else:
            fit quadratic model m_k(z) on projected data z_i = B_k^T (x_i - c_k)
            z_star <- solve_trust_region_subproblem(m_k, Δ_k)
            x_new <- project_to_box(c_k + B_k * z_star)
        k_used <- k

    # 3. evaluate
    y_new <- f(x_new)
    add (x_new, y_new) to D

    # 4. update best
    if y_new < f_best:
        f_best <- y_new
        x_best <- x_new

    # 5. update local structures and bandit stats
    if k_used != "global":
        update_active_subspace_B_k(k_used, D)
        compute predicted_improvement hatΔ and actual improvement Δ
        ρ <- Δ / (hatΔ + ε)
        if ρ > η1: Δ_k <- γ_inc * Δ_k; success = 1
        else if ρ < η0: Δ_k <- γ_dec * Δ_k; success = 0
        else: success = 0.5
        update_bandit_reward(k_used, Δ)
        update_failure_counter_and_rebuild_anchor_if_needed(k_used, success)
    else:
        update_bandit_reward(global_arm, Δ)
        # possibly create new anchor from x_new if good
        if y_new in top_q_percent(D):
            create_or_replace_anchor(x_new)

return x_best, f_best
```

---

## 6. 実用面での考察

### 6.1 メリット

- 高次元でも有効次元が低ければ効率的に探索可能
- ローカル探索（信頼領域）とグローバル探索（リスタート）をバンディットで自動調整
- サロゲートモデルは単純な二次回帰で、実装・計算コストが軽い
- 並列化しやすく、複数候補点を同時に評価する拡張も容易

### 6.2 計算コストの目安

- 各イテレーションの主なコスト：
  - 近傍点抽出：KD-tree 等で高速化可能
  - 二次回帰フィット：サンプル数 \(m\)、部分空間次元 \(r\) として \(O(m r^2)\)
  - 部分空間更新（固有分解）：頻度を下げることでコスト調整可能

### 6.3 ノイズへの対処

- 改善量 \(\Delta\) をそのまま使わず、移動平均・中央値・分位点などでロバスト化
- 高ノイズ環境では、同一点の複数評価により分散を推定し、信頼度に応じた重み付けを行う

### 6.4 制約付き問題への拡張

- 制約関数もブラックボックスの場合：
  - 実行可能/不可能のラベル付きデータを使って分類器を学習
  - 予測上「実行可能」となる点のみを候補として受け入れる
- あるいはペナルティ法：
  \[
    f'(x) = f(x) + \lambda \cdot \text{violation}(x)
  \]
  を目的関数として扱う

---

## 7. 既存手法との差別化ポイント（ざっくり）

- **CMA-ES との違い**
  - CMA-ES：1つのグローバルなガウス分布＋共分散行列を更新
  - RLSO：
    - 複数アンカー
    - 各アンカーごとに局所部分空間＋信頼領域＋二次モデル
    - どのアンカーを使うかをバンディットで決定

- **ベイズ最適化との違い**
  - ベイズ最適化：グローバルなサロゲート（GP など）と取得関数最適化が中心
  - RLSO：
    - 「部分空間 × ローカル信頼領域」に基づく局所サロゲート最適化を主役とし
    - どの局所領域を掘るかをバンディットで決める構造

- **直接探索／パターンサーチとの違い**
  - 直接探索：格子や固定パターンで探索することが多い
  - RLSO：
    - 評価データから有効部分空間を能動的に学習し
    - 導関数なしでも「効いている方向」に沿って動く

---

## 8. まとめ

- RLSO は、
  - **複数アンカー**
  - **局所部分空間（Active Subspace）**
  - **信頼領域に基づく二次サロゲート最適化**
  - **多腕バンディットによるアンカー選択とグローバル探索**
  の4要素を組み合わせた、実装可能なブラックボックス最適化アルゴリズムである。
- 評価コストが高い高次元問題に対して、
  - 有効次元が低い状況で特に効果が期待できる。
- 実装段階では：
  - 近傍点探索（KD-tree 等）
  - リッジ回帰による二次モデル
  - シンプルな UCB/ソフトマックスバンディット
  を用いれば、現場利用にも耐える形で運用可能である。
