# CABS: Cell-based Adaptive Bandit & Subspace Optimization

## 概要

**CABS (Cell-based Adaptive Bandit & Subspace optimization)** は、  
連続領域上のブラックボックス関数を対象とした「セル分割 × バンディット × ローカル低次元化 × サロゲート」のハイブリッド最適化アルゴリズムです。

対象問題は、例えば次のような連続値最適化問題を想定します。

\[
\min_{x \in [l,u]^d} f(x)
\]

- 勾配情報なし（ブラックボックス）
- ノイズあり評価にも対応可能
- 高次元の場合でも、アクティブサブスペースにより実効次元を低減して探索

---

## 1. 基本アイデア

CABS のコアとなるアイデアは以下の通りです。

1. **探索空間を「セル（部分領域）」に分割**しながら、
2. どのセルを重点的に探索するかを **バンディット（UCB風スコア）** で選択し、
3. 選択されたセル内では、
   - そのセルに属する評価点から **ローカルサロゲートモデル** を構築し、
   - 良好な評価を持つ点群から **ローカルアクティブサブスペース（低次元部分空間）** を抽出し、
   - その低次元空間に沿って **CMA-ES風のガウスサンプリング** により新しい候補点を生成する。

これにより、

- 全体としては DIRECT 系のような「セル分割＋グローバル探索」構造を持ちつつ、
- 各セルでは CMA-ES 的な局所探索を **自動次元削減付き** で行う、

という特徴を持ちます。

---

## 2. データ構造

### 2.1 評価アーカイブ

アルゴリズム全体で共有する評価履歴として、アーカイブ

\[
A = \{(x_i, y_i)\}_{i=1}^N
\]

を保持します。

- \( x_i \in \mathbb{R}^d \)：入力
- \( y_i = f(x_i) \)：評価値

### 2.2 セル（Cell）

セル \(h\) は、探索空間の部分領域を表すオブジェクトで、次を保持します。

- `B_h = [l_h, u_h]` : d 次元ハイパーレクト角（部分領域）
- `indices_h` : アーカイブ内でこのセルに属する点のインデックス集合
- 統計量  
  - \( f_{\min}(h) = \min_{i \in indices_h} y_i \)  
  - \( \mu_h = \text{mean}_{i \in indices_h} y_i \)  
  - \( \sigma_h = \text{std}_{i \in indices_h} y_i \)（セル内のばらつき・ノイズ指標）
- `n_eval(h)` : セル内での評価回数
- `diam(h)` : セルの対角線長（セルの大きさの指標）

セル全体の集合を \( \mathcal{H} \) とします。

---

## 3. アルゴリズム全体の流れ

### 3.1 初期化

1. 元の探索空間 \([l,u]^d\) を線形変換して、内部表現を \([0,1]^d\) にスケーリングする。
2. ラテン超方格サンプリング (LHS) などで、初期点 \(N_0\) 個を生成：
   \[
   x_1, \dots, x_{N_0} \sim \text{LHS}([0,1]^d)
   \]
3. 各点を元の空間に逆変換して評価し、アーカイブに格納：
   \[
   y_i = f(x_i)
   \]
4. 探索空間全体をカバーする 1 つのルートセル \(h_{\text{root}}\) を作成：
   - `B_root = [0,1]^d`
   - `indices_root = {1, ..., N0}`
5. セル集合を \(\mathcal{H} = \{h_{\text{root}}\}\) とする。

### 3.2 反復ステップ（メインループ）

評価予算 \(N_{\max}\) に到達するまで以下を繰り返す。

1. 各セル \(h\) に対して **バンディット風スコア \(S(h)\)** を計算。
2. スコアが最も有望なセルを \(K\) 個選択。
3. 各選択セル内で
   - ローカルサロゲート構築
   - アクティブサブスペース抽出
   - 低次元ガウスサンプリング＋候補点生成
   - サロゲートによるフィルタ（オプション）
   - 実評価とアーカイブ更新
4. 必要に応じてセル分割（Refinement）を行い、\(\mathcal{H}\) を更新。

---

## 4. セルのスコアリング（バンディット的 UCB）

セル \(h\) のスコア \(S(h)\) を、次のように定義します。

\[
S(h) = f_{\min}(h)
      - \alpha \sqrt{\frac{2 \log N_{\text{tot}}}{n_{\text{eval}}(h) + 1}}
      - \beta \cdot diam(h)
\]

- 第1項：**利用（exploitation）**  
  - \( f_{\min}(h) \) が小さいセルほど良い。
- 第2項：**探索（exploration）**  
  - 評価回数 \(n_{\text{eval}}(h)\) が少ないセルほどボーナスが大きい。
- 第3項：**未分割の広さのボーナス**  
  - まだ大きな領域をカバーしているセルに対して、探索優先度を高める。

ここで、

- \( N_{\text{tot}} \)：これまで実行した評価の総数
- \(\alpha, \beta > 0\)：ハイパーパラメータ（探索・利用バランスの調整）

各反復で、**スコアが最小のセル**（＝「まだあまり探索されておらず、良い値も出ていて、かつ広い」セル）を \(K\) 個選択します。

---

## 5. セル内のローカル探索

選ばれたセル \(h\) ごとに、次の処理を行います。

### 5.1 ローカルサロゲートモデル

セル \(h\) の点集合

\[
\{x_i, y_i\}_{i \in indices_h}
\]

から近傍点 \(M\) 個を使い、簡単なローカルサロゲートを構築します。

例：

- セル内の評価値が小さい順に \(M\) 点を選択。
- 以下のようなモデルをローカルにフィットする。

  - **ローカル線形モデル**  
    \[
    \hat{f}_h(x) \approx a_0 + a^\top x
    \]
  - **RBF（放射基底関数）モデル**  
    \[
    \hat{f}_h(x) \approx w_0 + \sum_{j=1}^M w_j \phi(\|x - x_{i_j}\|)
    \]

実装としては「線形回帰＋L2正則化」程度でも十分です。

このサロゲートは、

- 勾配方向（おおよその傾き）の推定
- 「どの方向に良さそうか」の粗い判断

に使います。

### 5.2 ローカルアクティブサブスペースの抽出

セル内の“良い点”の集合を

\[
X_{\text{good}} = \{x_i \mid i \in indices_h,\ y_i \text{ がセル内下位 }q\%\}
\]

とします（例：下位 30% の点を採用）。

これらの平均 \(\mu_h\) を引いた差分から共分散行列を構築します。

\[
C_h = \frac{1}{|X_{\text{good}}|} \sum_{x \in X_{\text{good}}} (x - \mu_h)(x - \mu_h)^\top
\]

これを固有値分解し、

\[
C_h = U \Lambda U^\top
\]

とおくと、上位 \(r\) 個の固有ベクトルからなる

\[
U_r \in \mathbb{R}^{d \times r}
\]

を **アクティブサブスペース** と見なします。

- \(r\) は 2〜5 程度の小さな整数（ハイパーパラメータ）。
- 固有値の落ち方を見て \(r\) を自動決定するバリアントも可能です。

### 5.3 サブスペースに沿ったガウスサンプリング

セル \(h\) 内の現在の最良点を \(x^*_h\) とします。

1. アクティブサブスペース内でガウス分布を定義：
   \[
   z \sim \mathcal{N}(0, \Sigma_z)
   \]
   - 例えば \(\Sigma_z = \text{diag}(\lambda_1, ..., \lambda_r)\)（固有値をそのままスケールとして使用）。

2. 実空間への射影：
   \[
   x_{\text{cand}} = x^*_h + U_r z + \epsilon
   \]
   - \(\epsilon \sim \mathcal{N}(0, \sigma_\perp^2 I_d)\)：サブスペース以外の方向への小さなノイズ。

3. セル境界への射影：
   - \(x_{\text{cand}}\) を `B_h ∩ [0,1]^d` にクリップ（はみ出した場合は端点に押し戻す）。

これを \(n_{\text{new}}\) 回（もしくはそれ以上）繰り返し、新しい候補点集合を生成します。

### 5.4 サロゲートによるフィルタリング（オプション）

生成した候補点群 \(\{x_{\text{cand},j}\}\) に対し、

- ローカルサロゲート \(\hat{f}_h(x)\) で予測値を計算。
- 予測値が低い上位 \(n_{\text{eval}}\) 点のみ実際に関数評価を行う。

これにより、実評価のコストを抑えつつサロゲートを「ガイド」として活用できます。

---

## 6. セル分割（Refinement）

セル \(h\) において評価が十分に溜まった場合、そのセルをさらに細かく分割して、局所的に探索を強化します。

### 6.1 分割条件

例えば以下のような条件で分割を行います。

- `n_eval(h) ≥ n_split_min`
- かつ `σ_h`（セル内標準偏差）が所定の閾値以上（局所的な変動がまだ大きい）

### 6.2 分割方向の選択

ローカルサロゲートから求めた **近似勾配** \(g_h\)（線形モデルの係数など）を使用し、  
「範囲が広く、かつ変化が大きい」次元でセルを分割します。

\[
j^* = \arg\max_j \left( (u_{h,j} - l_{h,j}) \cdot |g_{h,j}| \right)
\]

- \((u_{h,j} - l_{h,j})\)：セル内の \(j\) 次元の長さ
- \(|g_{h,j}|\)：サロゲート上の勾配の大きさ（変化の度合い）

### 6.3 分割点と子セル生成

1. セル内の点群 \(\{x_i\}_{i \in indices_h}\) の、次元 \(j^*\) の値の中央値 \(m\) を計算。
2. 境界を以下のように分けて 2 つの子セル \(h_1, h_2\) を生成。
   - \(h_1\)：\(x_{j^*} \le m\) を含む領域
   - \(h_2\)：\(x_{j^*} > m\) を含む領域
3. `indices_h` 内の点を \(h_1, h_2\) に振り分ける。
4. 親セル \(h\) を \(\mathcal{H}\) から削除し、代わりに \(h_1, h_2\) を追加する。

---

## 7. 擬似コード（Python風）

以下は CABS の全体像を示す擬似コードです（実装の骨格イメージ）。

```python
def CABS_optimize(f, bounds, d, 
                  N0=20, N_max=1000,
                  K=3, n_new=10, 
                  r=3, alpha=1.0, beta=0.1,
                  n_split_min=30):

    # 0. スケーリングして [0,1]^d へ
    # bounds: list of (l, u)
    def scale(x_real):
        return [(x_real[j] - bounds[j][0]) / (bounds[j][1] - bounds[j][0]) for j in range(d)]

    def descale(x_unit):
        return [bounds[j][0] + x_unit[j] * (bounds[j][1] - bounds[j][0]) for j in range(d)]

    # 1. 初期サンプル（LHSなど）
    X = lhs_sampling(d, N0)  # in [0,1]^d
    Y = [f(descale(x)) for x in X]
    N_tot = N0

    # 2. セルの初期化
    root = Cell(bounds=[np.zeros(d), np.ones(d)],
                indices=list(range(N0)))
    H = [root]

    while N_tot < N_max:
        # 3. セルスコア計算
        scores = []
        for h in H:
            Yh = [Y[i] for i in h.indices]
            fmin = min(Yh)
            n_eval = len(h.indices)
            diam = np.linalg.norm(h.bounds[1] - h.bounds[0])
            explore = np.sqrt(2 * np.log(N_tot + 1) / (n_eval + 1))
            S = fmin - alpha * explore - beta * diam
            scores.append((S, h))

        # 4. スコア上位 K セルを選択
        scores.sort(key=lambda t: t[0])
        target_cells = [h for _, h in scores[:K]]

        new_points = []

        # 5. 各セルでローカル探索
        for h in target_cells:
            Xh = [X[i] for i in h.indices]
            Yh = [Y[i] for i in h.indices]

            # 5.1 ローカルサロゲート構築（線形モデルなど）
            surrogate = fit_local_model(Xh, Yh)

            # 5.2 アクティブサブスペース
            X_good = select_good_points(Xh, Yh, quantile=0.3)
            U_r, lambdas = local_active_subspace(X_good, r=r)

            x_best = Xh[np.argmin(Yh)]

            # 5.3 候補生成（サブスペースに沿ったガウスサンプリング）
            candidates = []
            for _ in range(5 * n_new):  # 多めに生成
                z = np.random.normal(size=r) * np.sqrt(lambdas[:r])
                eps = np.random.normal(scale=0.01, size=d)
                x_cand = x_best + U_r @ z + eps
                x_cand = project_to_box(x_cand, h.bounds)
                candidates.append(x_cand)

            # 5.4 サロゲートでフィルタして評価対象を決定
            preds = [surrogate(xc) for xc in candidates]
            idx = np.argsort(preds)[:n_new]
            selected = [candidates[i] for i in idx]

            # 実評価予定としてバッファに積む
            for xc in selected:
                new_points.append((xc, h))

        # 6. 実評価＆アーカイブ更新
        for xc, h in new_points:
            yc = f(descale(xc))
            X.append(xc)
            Y.append(yc)
            new_idx = len(X) - 1
            h.indices.append(new_idx)

        N_tot = len(X)

        # 7. セル分割
        new_H = []
        for h in H:
            if len(h.indices) >= n_split_min:
                Yh = [Y[i] for i in h.indices]
                if np.std(Yh) > some_threshold:
                    h1, h2 = split_cell(h, X, Y)
                    new_H.extend([h1, h2])
                else:
                    new_H.append(h)
            else:
                new_H.append(h)

        H = new_H

    # 8. 最良点を返す
    best_idx = int(np.argmin(Y))
    return descale(X[best_idx]), Y[best_idx]
```

※ `lhs_sampling`, `Cell`, `fit_local_model`, `local_active_subspace`, `select_good_points`, `project_to_box`, `split_cell` などは、  
  実装時に状況に合わせて定義する必要があります。

---

## 8. 特徴・実用上のポイント

### 8.1 長所

- **グローバル探索性**  
  - セル分割＋バンディットスコアによって、未踏領域や有望領域をバランスよく探索できる。
- **ローカル収束性**  
  - 各セル内でアクティブサブスペースに沿ったガウスサンプリングを行うため、局所最適への収束性能が高い。
- **高次元対応**  
  - 実質的に効いている方向（アクティブサブスペース）を抽出することで、問題の「実効次元」を自動的に圧縮できる。
- **ノイズへの頑健性**  
  - セル内の分散やサロゲートを用いて、ノイズレベルを推定しつつ UCB の探索度合いを調整可能。
  - 同一点再評価戦略と組み合わせることで、平均化によるノイズ低減も行える。

### 8.2 注意点・弱点

- セル構造・サロゲート・PCA（固有値分解）など、実装がやや重い。
- 次元が極端に高い（例：100+）場合、アクティブサブスペースの推定に十分なデータが必要。
- 離散変数や組合せ最適化には、そのままでは適用しづらい（拡張が必要）。

---

## 9. 拡張アイデア

- **並列化**  
  - 複数セルでの候補生成・評価をバッチ化して並列実行できる。
- **混合ドメインへの拡張**  
  - 離散変数をワンホット埋め込みしてサブスペース抽出を行う。
  - もしくはセルごとに離散部分を条件付きで固定し、連続部分のみ最適化するバリアント。
- **サロゲートの高度化**  
  - 線形モデルや単純RBFの代わりに、ローカルGP、ランダムフォレスト、ニューラルネットなどを使用。
- **アクティブサブスペース次元 r の自動決定**  
  - 固有値の落ち方（スペクトルギャップ）を見て、情報量の高い固有値までを採用するようにする。

---

## 10. まとめ

CABS は、

- DIRECT/HOO のような **階層的セル分割＋グローバル探索**、
- CMA-ES やアクティブサブスペース法のような **局所低次元探索**、
- バンディットUCB のような **探索・搾取バランス制御**、

を一つの枠組みに統合した、  
**実用的なブラックボックス最適化アルゴリズムの新しい設計案**です。

実装時には、対象問題（次元、評価コスト、ノイズの有無、連続/離散/混合など）に応じて、

- セル分割条件
- UCBパラメータ \(\alpha, \beta\)
- アクティブサブスペース次元 \(r\)
- サロゲートモデルの種類

を調整することで、さまざまな応用に適用できます。
