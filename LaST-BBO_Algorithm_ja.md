# LaST-BBO: Latent-Space Trust-region Black-Box Optimization

## 1. 問題設定

- 目的：  
  \(\min_{x \in \mathcal{X}} f(x)\)
- 条件：
  - \(\mathcal{X} \subset \mathbb{R}^d\) は有界な箱型領域（各次元ごとに下限・上限あり）
  - \(f(x)\) はブラックボックス関数で、勾配は不明
  - 評価コストが高い（シミュレーションや実験など）
  - ノイズありの評価も想定
  - 次元数 \(d\) は中〜高次元（例：20〜200）も想定

本アルゴリズム **LaST-BBO**（Latent-Space Trust-region Black-Box Optimization）は、
「評価履歴から低次元の潜在空間を学習し、その潜在空間上で信頼領域付き最適化を行う」
実用志向のブラックボックス最適化手法である。

---

## 2. コアアイデアの概要

1. **潜在空間への埋め込み**
   - 高次元空間 \(\mathbb{R}^d\) で直接モデル化する代わりに、
     - 位置が近い点同士
     - 評価値が近い点同士
     が潜在空間で近くなるような埋め込み \(Z \in \mathbb{R}^{n \times m}\) を構成する（\(m \ll d\)）。

2. **潜在空間上での surrogate モデル + ベイズ的探索**
   - 潜在空間上でガウス過程やランダムフォレスト等の surrogate モデル \(g(z)\) を学習し、
     UCB/LCB などの獲得関数を最適化する。

3. **信頼領域 (trust region) による局所探索**
   - 良い評価値を持つ点をクラスタリングし、各クラスタごとに潜在空間上の信頼領域（球）を定義。
   - 各信頼領域内で獲得関数を最適化することで、局所探索を安定させる。

4. **潜在空間 → 元空間への写像**
   - 新しい潜在点 \(\hat{z}\) に対して、既存点の加重平均（バリセントリック写像）＋小ノイズによって
     元の空間の点 \(x_{\text{new}}\) を生成する。
   - これにより、既知の良い点の「周辺」を滑らかに探索できる。

5. **大域探索と局所探索のバランス**
   - 改善が停滞したときは信頼領域を広げる・探索度合いを増やすなど、
     ハイパーパラメータを動的に調整することで、大域探索と局所探索のトレードオフを制御する。

---

## 3. アルゴリズムの状態

- 評価履歴
  - 入力点行列：  
    \(X = [x_1, \dots, x_n]^\top \in \mathbb{R}^{n \times d}\)
  - 評価値ベクトル：  
    \(y = [f(x_1), \dots, f(x_n)]^\top\)

- 潜在空間の座標
  - \(Z = [z_1, \dots, z_n]^\top \in \mathbb{R}^{n \times m}\)

- surrogate モデル
  - 潜在空間上の回帰モデル：  
    \(g: \mathbb{R}^m \rightarrow \mathbb{R}\)  
    （例：ガウス過程、ランダムフォレスト、XGBoost 等）

- 信頼領域情報
  - 良好な点をクラスタリングしたクラスタ中心：  
    \(c_j \in \mathbb{R}^m\) （\(j=1,\dots,k\)）
  - 各クラスタに対応する半径：  
    \(r_j > 0\)

---

## 4. 潜在空間埋め込みの構成

### 4.1 初期サンプリング

1. 領域 \(\mathcal{X}\) 上で LHS（ラテン超方格）や一様乱数により \(n_0\) 点をサンプリング：  
   \(X \in \mathbb{R}^{n_0 \times d}\)
2. 各点で \(f(x)\) を評価し、\(y\) を得る。

### 4.2 重み行列の構成

評価済み点同士の類似度に基づき重み行列 \(W\) を定義：

\[
W_{ij} = \exp\left(
  -\frac{\lVert x_i - x_j \rVert^2}{\sigma_x^2}
  -\frac{(y_i - y_j)^2}{\sigma_y^2}
\right)
\]

- 位置が近く評価値も近い組は重みが大きくなる。
- 評価が大きく異なる組は重みが小さい（潜在空間で離れやすい）。

### 4.3 ラプラシアン固有マップによる埋め込み

1. 度数行列 \(D\) を  
   \[
   D_{ii} = \sum_j W_{ij}
   \]
   と定義。
2. グラフラプラシアンを  
   \[
   L = D - W
   \]
   とする。
3. 固有値問題  
   \[
   L u = \lambda D u
   \]
   を解き、\(\lambda\) が小さい方から \(m\) 個の固有ベクトルを取り出す。
4. それらを並べて  
   \[
   Z = [u_1, \dots, u_m] \in \mathbb{R}^{n \times m}
   \]
   とし、これを潜在空間の座標とする。

※ 実装上は scikit-learn の `SpectralEmbedding` や類似手法で代用可能。

---

## 5. 潜在空間での surrogate 学習

潜在空間上で回帰モデルを学習する：

- 入力：\(Z \in \mathbb{R}^{n \times m}\)
- 出力：\(y \in \mathbb{R}^n\)

例：ガウス過程回帰（ノイズあり）  
\[
y = g(Z) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma_n^2 I).
\]

これにより、任意の潜在点 \(z\) に対して予測平均 \(\mu(z)\) と予測分散 \(\sigma^2(z)\) が得られる。

---

## 6. 信頼領域（Trust Region）の構築

### 6.1 良好な点集合の抽出

1. 評価値 \(y\) でソートし、上位 \(p\%\) を「良い点集合」 \(X_{\text{good}}\) とする。
2. 以下のインデックス集合を定義：  
   \[
   I_{\text{good}} = \{ i \mid y_i \text{ が小さい方から } p\% \text{ に入る} \}
   \]
3. 対応する潜在空間座標集合：  
   \[
   Z_{\text{good}} = \{ z_i \mid i \in I_{\text{good}} \}
   \]

### 6.2 クラスタリング

- \(Z_{\text{good}}\) に対して k-means によるクラスタリングを実行し、\(k\) 個のクラスタに分割。
- 各クラスタ \(C_j\) の中心を  
  \[
  c_j \in \mathbb{R}^m, \quad j = 1,\dots,k
  \]
  とする。

### 6.3 半径の決定

クラスタ \(C_j\) 内の最大距離をもとに半径を設定：

\[
r_j = \alpha \cdot \max_{i \in C_j} \lVert z_i - c_j \rVert
\]

- \(\alpha > 1\) として少し広げることで、過度に局所に閉じないようにする。

---

## 7. 潜在空間での獲得関数最適化

### 7.1 LCB (Lower Confidence Bound) の定義

潜在空間の任意点 \(z\) に対して、

\[
\text{LCB}(z) = \mu(z) - \beta \sigma(z)
\]

- \(\mu(z)\)：surrogate モデルの予測平均
- \(\sigma(z)\)：予測標準偏差
- \(\beta > 0\)：探索度合いパラメータ（大きいほど探索寄り）

### 7.2 各信頼領域内での最適化

各クラスタ \(j\) について：

1. 初期点としてクラスタ中心 \(c_j\) 付近の点（複数）を用意。
2. 制約 \(\lVert z - c_j \rVert \le r_j\) の球制約付きで \(\text{LCB}(z)\) を最小化。
   - 例：ランダムスタート + L-BFGS-B + 投影。
3. 得られた最適点を \(\hat{z}_j\) とする。

### 7.3 大域探索候補の追加（任意）

- 潜在空間全体で LCB を最適化し、既存クラスタから離れた点をいくつかサンプリングすることで、
  大域探索用の候補を追加する。

---

## 8. 潜在空間 → 元空間への写像

新しい潜在点 \(\hat{z}\) を元の空間の点 \(x_{\text{new}}\) に変換する。

### 8.1 バリセントリック写像

1. 距離に基づく重みを定義：  
   \[
   w_i(\hat{z}) = \exp\left(-\frac{\lVert \hat{z} - z_i \rVert^2}{\rho^2}\right)
   \]
   - \(\rho\) はスケールパラメータ。

2. 正規化：  
   \[
   \tilde{w}_i = \frac{w_i}{\sum_{k=1}^n w_k}
   \]

3. 元空間での代表点：  
   \[
   \bar{x}(\hat{z}) = \sum_{i=1}^n \tilde{w}_i x_i
   \]

4. 小さなガウスノイズを付加し、探索の多様性を確保：  
   \[
   x_{\text{new}} = \Pi_{\mathcal{X}}(\bar{x}(\hat{z}) + \epsilon),
   \quad \epsilon \sim \mathcal{N}(0, \tau^2 I_d)
   \]
   - \(\Pi_{\mathcal{X}}\) は領域への射影（上下限クリッピング）。

### 8.2 バッチ生成

複数の \(\hat{z}\)（各信頼領域、あるいは大域探索由来）から同様の写像を行い、
バッチ候補 \(\{x_{\text{new}}^{(1)}, \dots, x_{\text{new}}^{(B)}\}\) を生成する。

---

## 9. 評価と更新ループ

### 9.1 ループ全体の流れ

1. 初期サンプルを評価し、\(X, y\) を得る。
2. 潜在空間埋め込み \(Z\) を構築。
3. surrogate モデル \(g\) を学習。
4. 良い点集合のクラスタリング → 信頼領域 \((c_j, r_j)\) の設定。
5. 各信頼領域・大域探索用で LCB を最適化し、潜在候補 \(\hat{z}\) を得る。
6. 潜在候補を元空間へ写像し、候補点 \(x_{\text{new}}\) を生成。
7. \(f(x_{\text{new}})\) を評価し、\(X, y\) を更新。
8. 予算（評価回数）に達するまで 2〜7 を繰り返す。

### 9.2 停滞時の適応的調整

一定の評価ウィンドウ W で最良値がほとんど改善しない場合、以下を検討：

- 各半径 \(r_j\) を拡大 → 大域探索を強める
- \(\beta\)（探索係数）を増やす → 不確実な領域を優先
- 潜在次元 \(m\) の変更 → 地形の複雑さに合わせる
- 埋め込みのための点集合をサブサンプリング → 計算コスト削減

---

## 10. 疑似コード

```python
def LaST_BBO(
    f, bounds, budget,
    n_init=20,
    m=8,          # latent dimension
    k=3,          # number of clusters
    beta=2.0,
    p_good=0.3,
    batch_size=5
):
    # 0. 初期サンプリング
    X = latin_hypercube(bounds, n_init)   # shape: (n_init, d)
    y = [f(x) for x in X]
    n_eval = n_init

    while n_eval < budget:
        # 1. 潜在空間埋め込み
        Z = build_latent_embedding(X, y, m)

        # 2. surrogate 学習（GP, RF など）
        surrogate = fit_surrogate(Z, y)

        # 3. 良い点の抽出
        n_good = max(int(len(y) * p_good), k)
        idx_sorted = np.argsort(y)
        good_idx = idx_sorted[:n_good]
        Z_good = Z[good_idx]

        # 4. クラスタリングと信頼領域
        clusters = kmeans(Z_good, k)
        centers = clusters.centers      # shape: (k, m)
        radii = estimate_radii(Z_good, clusters)  # list of length k

        cand_Z = []

        # 5. 各信頼領域で acquisition 最適化
        for j in range(k):
            cj = centers[j]
            rj = radii[j]
            zj = optimize_acquisition_in_ball(
                surrogate,
                center=cj,
                radius=rj,
                beta=beta    # LCB 用パラメータ
            )
            cand_Z.append(zj)

        # 6. 大域探索用の追加候補（任意）
        cand_Z += global_exploration_candidates(surrogate, Z, beta)

        # 7. 潜在空間 → 元空間
        cand_X = []
        for z_hat in cand_Z[:batch_size]:
            x_new = latent_to_original(z_hat, Z, X)
            x_new = project_to_bounds(x_new, bounds)
            cand_X.append(x_new)

        # 8. 評価して履歴更新
        y_new = [f(x) for x in cand_X]
        X = np.vstack([X, cand_X])
        y = np.concatenate([y, y_new])
        n_eval += len(cand_X)

        # 9. 停滞検知・ハイパーパラメータ調整（必要であれば）
        #    adjust_beta_and_radii(...)

    best_idx = np.argmin(y)
    return X[best_idx], y[best_idx]
```

---

## 11. 特徴・利点（実用性の観点）

1. **高次元への対応**
   - 評価履歴から「本質的な自由度」を抽出し、低次元の潜在空間で最適化するため、
     高次元でも GP などのモデルが扱いやすい。

2. **データ駆動の低次元空間**
   - REMBO のようにランダム射影を事前に固定するのではなく、
     評価値を含むデータから潜在空間を構成する点が特徴。

3. **局所探索と大域探索の自然な統合**
   - 信頼領域（半径 \(r_j\)）や \(\beta\) の調整で、
     攻める（局所）/ 広く探す（大域）のバランスを直感的に制御可能。

4. **実装容易性**
   - 潜在空間埋め込み：`SpectralEmbedding` 等
   - surrogate：`GaussianProcessRegressor` や `RandomForestRegressor`
   - 球制約付き最適化：ランダムスタート + 投影付き L-BFGS-B など

5. **ノイズへのロバスト性**
   - surrogate をノイズあり回帰として扱うことで、観測ノイズを吸収可能。
   - 埋め込みで評価値差 \((y_i - y_j)^2\) を用いているため、
     ノイズに対してもある程度平滑な構造が得られる。

---

## 12. バリエーションと拡張

- surrogate モデルの変更
  - ガウス過程 → ランダムフォレスト / XGBoost / 小さな MLP など
  - 評価回数が多い場合にもスケールさせやすい。

- 潜在→元空間写像の改良
  - バリセントリック写像の代わりに
    - 局所線形回帰（LLR）
    - RBF 補間
    などを使うことで、より構造を反映した写像も可能。

- バッチ最適化
  - 各信頼領域から複数の \(\hat{z}\) をサンプリングし、
    まとめて評価することで並列計算資源を活用できる。

---

このファイルをベースに、実際のコード実装（Python 等）や、
特定の応用領域（ハイパーパラメータ探索 / 実験条件最適化など）向けの
チューニングガイドも追加していくことができます。
