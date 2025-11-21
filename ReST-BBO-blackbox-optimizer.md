# ReST-BBO: Region-wise Surrogate-guided Trust-region Black-box Optimizer

## 0. 問題設定

- 黒箱関数 \( f: \mathbb{R}^d \to \mathbb{R} \) を **最小化** する。
- 勾配は利用不可。
- 評価コストは高い（＝少ない評価回数でそこそこ良い解を見つけたい）。
- 制約は簡単のため、箱型制約 \(x \in [l, u]^d\) を想定。  
  一般制約への拡張は後述。

> 注: 「全く新しい」ことを数学的に保証することはできないが、  
> 少なくとも典型的な CMA-ES / ベイズ最適化 / PSO / DIRECT などの  
> 教科書的アルゴリズムの単純な焼き直しではないように設計している。

---

## 1. アルゴリズムの全体像

**名前:** **ReST-BBO**  
(**Re**gion-wise **S**urrogate-guided **T**rust-region **B**lack-**B**ox **O**ptimizer)

### コアアイデア

1. **探索空間を複数の「局所領域 (region)」に分割**
   - 各領域は「中心点 \(c_r\)」と「半径 \(\Delta_r\)」を持つ球状のトラストリージョン。
   - 各領域ごとにローカル代理モデル（サロゲート）を構築。

2. **領域ごとに軽量なローカル回帰モデルを構築**
   - 例: RBF（Radial Basis Function）やランダム特徴写像＋リッジ回帰。
   - 領域内のデータのみを使い、「局所的に滑らかな近似」を狙う。

3. **どの領域を次に探索するかをバンディット的に決定**
   - 各領域は「最近どれくらい改善を出したか」の報酬を持つ。
   - UCB 的指標で「実績もあるし、まだ不確実性もある」領域を優先。

4. **選ばれた領域の中で、代理モデルに基づいて点を提案**
   - 「予測値」と「不確実性」から LCB (Lower Confidence Bound) 風の獲得関数を作る。
   - ランダムサーチ＋ローカル探索でその獲得関数を最小化。

5. **代理モデルの予測と実測の“一致度”でトラストリージョンを拡大／縮小**
   - よく当たるなら半径を広げる（大胆な探索）。
   - 外してばかりなら半径を縮める（保守的な探索）。

6. **良い点が遠くに見つかったら、新しい領域を生成**
   - 既存領域から遠く、かつ評価値の良い点を中心に新しい領域を作る。

---

## 2. データ構造

### 2.1 グローバルアーカイブ

これまで評価した全ての点の集合:
\[
\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N
\]

### 2.2 領域（region）の集合

領域 \(r = 1, \dots, R\) それぞれについて、以下を管理する:

- 中心: \(c_r \in \mathbb{R}^d\)
- 半径: \(\Delta_r > 0\)
- ローカルデータインデックス: \(I_r \subset \{1, \dots, N\}\)  
  （「距離が近い点」や「領域内の点」など）
- 代理モデル: \(m_r(x) \approx f(x)\) for \(x\) near \(c_r\)
- 予測不確実性指標: \(s_r(x)\)（モデルから計算）
- バンディット用統計:
  - 評価回数: \(n_r\)
  - 平均報酬: \(\hat{\mu}_r\)（例: 1 step あたりの平均改善量）
- 老朽化指標: \(a_r\)（最後に有意な改善を出してから何ステップ経過したか）

---

## 3. ローカル代理モデル

### 3.1 RBF + リッジ回帰モデルの例

領域 \(r\) のローカルデータ:
\[
\{(x_j, y_j)\}_{j \in I_r}
\]

中心 \(c_r\)、半径 \(\Delta_r\) を使ってスケーリングした距離を:
\[
z_j = \frac{x_j - c_r}{\Delta_r}
\]

RBF 特徴:
\[
\phi_j(x) = \exp\left(-\frac{\left\| (x - c_r)/\Delta_r - z_j \right\|^2}{2 \sigma^2}\right)
\]

モデル:
\[
m_r(x) = w_0 + \sum_{j \in I_r} w_j \, \phi_j(x)
\]

行列 \(\Phi \in \mathbb{R}^{|I_r| \times |I_r|}\) を構成し、リッジ回帰で
\[
\hat{w} = (\Phi^\top \Phi + \lambda I)^{-1} \Phi^\top y
\]
を求める。実装時は効率化の余地あり。

### 3.2 不確実性指標 \(s_r(x)\)

厳密なベイズ推論は避け、実用的な近似指標を用いる。

#### 距離ベース

最近傍ローカル点との距離:
\[
s_r(x) = \min_{j\in I_r} \|x - x_j\|
\]

#### RBF の自己相関ベース

\[
s_r(x) = 1 - \max_{j\in I_r} \phi_j(x)
\]

など、実装の簡単さ・安定性で選択すれば良い。

---

## 4. 獲得関数と候補点生成

### 4.1 LCB 風獲得関数

現在までのベスト値:
\[
y^\* = \min_i y_i
\]

領域 \(r\) の候補点 \(x\) に対して:

- 予測値:
  \[
  \mu_r(x) = m_r(x)
  \]
- 不確実性:
  \[
  u_r(x) = s_r(x)
  \]

から LCB（Lower Confidence Bound）風の獲得関数を定義する:

\[
\text{LCB}_r(x) = \mu_r(x) - \kappa \cdot u_r(x)
\]

- \(\kappa > 0\) は探索性（不確実性をどれだけ重視するか）を制御するパラメータ。
- 小さいほど「良い候補」と見なす。

### 4.2 領域内での候補点生成

領域 \(r\) 内で次の候補点 \(x_{\text{cand}}\) を生成する:

1. 領域内で一様乱数サンプリング  
   \[
   x^{(m)} = c_r + \Delta_r \cdot \xi^{(m)},\quad \xi^{(m)} \sim \text{Uniform}([-1,1]^d)
   \]
   - 箱制約 \([l, u]^d\) からはみ出した場合はクリッピング。
   - \(m = 1, \dots, M\) の複数サンプルを生成。

2. 各 \(x^{(m)}\) に対し \(\text{LCB}_r(x^{(m)})\) を計算し、最小のものを候補点とする:
   \[
   x_{\text{cand}} = \arg\min_{m} \text{LCB}_r(x^{(m)})
   \]

3. オプションとして、  
   \(x_{\text{cand}}\) を初期値に代理モデル上での局所最適化（数ステップの勾配ベース最適化など）を行い、さらに LCB を下げた点を採用することもできる。

---

## 5. 領域選択：バンディット式 UCB

### 5.1 領域スコア

時刻 \(t\) における領域 \(r\) のスコアを

\[
\text{Score}_r(t) = \hat{\mu}_r + \beta \sqrt{\frac{\log t}{n_r + 1}} + \gamma \cdot U_r
\]

と定義する。

- \(\hat{\mu}_r\): 領域 \(r\) がこれまでに出した平均改善量（報酬）
- \(n_r\): 領域 \(r\) が選ばれた回数
- \(U_r\): 領域 \(r\) の「平均不確実性」  
  例: 領域内の代表点集合における \(u_r(x)\) の平均
- \(\beta, \gamma > 0\): 探索・不確実性の重み付けパラメータ

### 5.2 領域選択ルール

- 確率 \(1 - \epsilon\) で
  \[
  r^\* = \arg\max_r \text{Score}_r(t)
  \]
- 確率 \(\epsilon\) でランダムな領域を選択（純探索）

\(\epsilon\) は例として 0.1 程度とし、探索と利用のバランスを取る。

---

## 6. トラストリージョン更新ルール

領域 \(r^\*\) 内の候補点 \(x_{\text{cand}}\) を評価したとする。

- 旧ベスト値: \(y_{\text{old}}^\*\)
- 新しい評価値: \(y_{\text{cand}} = f(x_{\text{cand}})\)
- 代理モデル予測値: \(\tilde{y} = m_{r^\*}(x_{\text{cand}})\)

改善量:

\[
\Delta f_{\text{real}} = y_{\text{old}}^\* - y_{\text{cand}}
\]
\[
\Delta f_{\text{pred}} = y_{\text{old}}^\* - \tilde{y}
\]

比率:

\[
\rho = 
\begin{cases}
\frac{\Delta f_{\text{real}}}{\Delta f_{\text{pred}}} & \text{if } \Delta f_{\text{pred}} > 0\\
0 & \text{otherwise}
\end{cases}
\]

### 6.1 半径更新

- もし \( \Delta f_{\text{real}} > 0 \) かつ \( \rho > \eta_1 \)（例: 0.75）なら
  - 実際に改善しており、モデル予測も良好 → 半径を拡大:
    \[
    \Delta_{r^\*} \leftarrow \min(\gamma_{\text{inc}} \Delta_{r^\*}, \Delta_{\max})
    \]

- もし \( \rho < \eta_2 \)（例: 0.25）なら
  - モデルの信頼性が低い → 半径を縮小:
    \[
    \Delta_{r^\*} \leftarrow \max(\gamma_{\text{dec}} \Delta_{r^\*}, \Delta_{\min})
    \]

- それ以外の場合は半径は維持。

### 6.2 バンディット報酬の更新

報酬:
\[
r_{\text{bandit}} = \max(\Delta f_{\text{real}}, 0)
\]

移動平均で更新:
\[
\hat{\mu}_{r^\*} \leftarrow \frac{n_{r^\*} \hat{\mu}_{r^\*} + r_{\text{bandit}}}{n_{r^\*} + 1}
\]

そして \(n_{r^\*} \leftarrow n_{r^\*} + 1\)、老朽化指標 \(a_{r^\*} \leftarrow 0\) とする。

他の領域 \(r \neq r^\*\) については \(a_r \leftarrow a_r + 1\)。

---

## 7. 新規領域生成と古い領域削除

### 7.1 新規領域生成

新しい点 \(x_{\text{cand}}\) が

- 評価値として十分良い（例: \(y_{\text{cand}} \le y^\* + \delta\)）
- 全ての既存領域中心 \(c_r\) からの距離が大きい
  \[
  \min_r \|x_{\text{cand}} - c_r\| > \alpha \cdot \Delta_r
  \]

という条件を満たす場合、新しい領域を生成する:

- 中心:
  \[
  c_{\text{new}} = x_{\text{cand}}
  \]
- 半径:
  \[
  \Delta_{\text{new}} = \Delta_{\text{init}}
  \]
- 近傍点を集めてローカルデータ \(I_{\text{new}}\) を構築し、代理モデル \(m_{\text{new}}\) をフィット。

### 7.2 領域の削除・統合

- 領域 \(r\) が長期間改善を出していない（例: \(a_r > A_{\max}\) 評価分）場合、削除候補とする。
- 中心の近い領域同士を統合し、データとモデルを再構築することもできる。

---

## 8. アルゴリズムの擬似コード

以下は Python 風の簡略擬似コード。

```text
Input: f, box [l, u], eval_budget
Hyperparams: n_init, R_max, M, κ, β, γ, 
             Δ_init, Δ_min, Δ_max, 
             γ_inc, γ_dec, η1, η2, ε, ...
--------------------------------------------------
1. 初期サンプリング
   D = {}
   for i in 1..n_init:
       x_i = ラテン超立方 or 一様乱数 in [l, u]
       y_i = f(x_i)
       D.add((x_i, y_i))

2. 初期領域の構築
   - グローバルベスト x*, y* を D から選ぶ
   - R = 1
   - region_1:
       c_1 = x*
       Δ_1 = Δ_init
       I_1 = 近傍点をいくつか
       m_1 = fit_surrogate(D[I_1])
       n_1 = |I_1|
       μ̂_1 = 0
       a_1 = 0

3. for t in (n_init+1)..eval_budget:
   3.1 領域選択
       with prob ε:
           r = ランダムな領域
       else:
           for each region r:
               U_r = 領域代表点での不確実性平均
               Score_r = μ̂_r + β * sqrt(log(t) / (n_r + 1)) + γ * U_r
           r = argmax Score_r

   3.2 領域 r 内で候補点生成
       候補集合 C = {}
       for m in 1..M:
           ξ ~ Uniform([-1,1]^d)
           x = c_r + Δ_r * ξ
           x を [l, u] にクリップ
           C.add(x)

       各 x in C について LCB_r(x) を計算
       x_cand = argmin_x LCB_r(x)

       (オプション) x_cand を初期値に代理モデル上の局所最適化

   3.3 評価
       y_cand = f(x_cand)
       D.add((x_cand, y_cand))

   3.4 トラストリージョン更新
       y_old_best = y*
       y* = min(y*, y_cand)

       y_pred = m_r(x_cand)
       Δf_real = y_old_best - y_cand
       Δf_pred = y_old_best - y_pred

       if Δf_pred > 0:
           ρ = Δf_real / Δf_pred
       else:
           ρ = 0

       if Δf_real > 0 and ρ > η1:
           Δ_r = min(γ_inc * Δ_r, Δ_max)
       elif ρ < η2:
           Δ_r = max(γ_dec * Δ_r, Δ_min)

       # バンディット報酬更新
       reward = max(Δf_real, 0)
       μ̂_r = ( (n_r * μ̂_r) + reward ) / (n_r + 1)
       n_r += 1
       a_r = 0

   3.5 領域データと代理モデルの更新
       I_r = D から「中心 c_r に近い k 点」を再選択
       m_r = fit_surrogate(D[I_r])

       他の領域 r' の a_r' をインクリメント

   3.6 新規領域生成 / 古い領域削除
       if 条件を満たすなら:
           新しい region_new を作る
       改善がずっとない領域を削除

4. return 最良の (x*, y*)
```

---

## 9. 実装上のポイント

### 9.1 推奨ハイパーパラメータの目安

- 初期点数: \(n_{\text{init}} \approx 5d \sim 10d\)
- 領域あたりローカル点数: \(k \approx 10d\)（最大で 200〜300 程度）
- 候補サンプル数: \(M \approx 50 \sim 200\)
- トラストリージョン半径（変数が [0,1]^d に正規化されていると仮定）:
  - \(\Delta_{\text{init}} = 0.25\)
  - \(\Delta_{\min} = 0.05\)
  - \(\Delta_{\max} = 0.5\)
- パラメータ例:
  - \(\kappa \approx 1.0 \sim 3.0\)
  - \(\beta \approx 0.5 \sim 2.0\)
  - \(\gamma \approx 0.5 \sim 1.0\)
  - \(\gamma_{\text{inc}} = 1.5,\ \gamma_{\text{dec}} = 0.5\)
  - \(\eta_1 = 0.75,\ \eta_2 = 0.25\)
  - \(\epsilon = 0.1\)

### 9.2 ノイズのある関数への拡張

- 同一点を複数回評価し平均をとる。
- 代理モデルのフィットにロバスト回帰（Huber loss など）を用いる。
- 不確実性指標に観測ノイズの分散を組み込む。

などで対応できる。

### 9.3 並列バッチ評価

- 各イテレーションで複数候補点を提案し、並列で評価するバッチ版も容易に構成可能。
- 異なる領域から 1 点ずつ選んだり、同一領域で LCB に少量ノイズを加えて複数点を生成したりといったバリエーションが考えられる。

---

## 10. 既存手法との比較（ざっくり）

- **CMA-ES 等**
  - 通常は単一ガウス分布で探索するが、ReST-BBO は  
    「複数トラストリージョン＋ローカルサロゲート」を採用。
- **標準的ベイズ最適化 (GP + EI/LCB)**
  - 1 つのグローバル GP を維持する代わりに、「多数の軽量ローカルモデル」を使う。
  - 高次元にもある程度スケールしやすい（局所的な次元効果の利用）。
- **DIRECT / MADS 系**
  - メッシュや矩形分割を基本とするのに対し、本手法は  
    連続的な球状トラストリージョン＋統計モデルに基づく。

---

## 11. 拡張案

- **実コード実装**  
  - NumPy / SciPy / scikit-learn を用いた Python 実装。
- **ベンチマーク実験**  
  - Sphere, Rosenbrock, Ackley など標準関数での性能評価。
- **離散・混合変数への拡張**  
  - 離散部分はカテゴリ変数として one-hot 化し、距離関数を工夫。
- **制約付き最適化**  
  - ペナルティ関数を導入する、あるいは可行性を予測するサロゲートモデルを追加。

---

以上が、ReST-BBO の Markdown 形式による仕様まとめである。
