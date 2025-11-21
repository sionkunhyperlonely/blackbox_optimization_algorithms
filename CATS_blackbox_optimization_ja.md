# CATS: Cluster-Aware Trust-region Surrogate Optimizer
（クラスタ構造を利用したトラストリージョン型ブラックボックス最適化アルゴリズム）

## 1. 問題設定

連続ブラックボックス最適化問題を扱う。

- 目的関数：  
  \[
    \min_{x \in \mathcal{X}} f(x),\quad \mathcal{X} = [0,1]^d
  \]
- 特徴
  - 導関数が利用できない（真の中身はブラックボックス）
  - 評価コストが高く、許される評価回数は数百〜数万回程度
  - 次元 \(d\) は中程度（例：5〜50 次元）
  - 目的関数値には小さなノイズが含まれていてもよい

目的：有限回の関数評価で、できるだけ良い近似解 \(x^\*\) を見つける。

---

## 2. アルゴリズム概要

**アルゴリズム名：CATS（Cluster-Aware Trust-region Surrogate optimizer）**

評価済みの点集合を自動的にクラスタリングし、各クラスタごとに局所サロゲートモデルとトラストリージョンを持つ。クラスタ単位で「局所改良」と「大域探索」をバランス良く行うことで、多峰性を持つ目的関数に対してもロバストに動作することを狙う。

### 基本アイデア

1. **クラスタリング**  
   - 評価済みの点 \(\{x_i\}\) をクラスタリングし、複数の「局所領域（谷）」を自動抽出。
2. **クラスタごとの局所サロゲート**  
   - 各クラスタごとに軽量なサロゲートモデル（例：ランダムフォレスト、RBF、浅いNN）を学習。
3. **トラストリージョン**  
   - 各クラスタにトラストリージョン（TR）サイズ \(\rho_k\) を持たせ、改善状況に応じて拡大／縮小。
4. **局所探索 + 大域探索**  
   - 各クラスタのTR内をサロゲートの獲得関数で最適化しつつ、別途大域的な候補点も生成して未知領域を探索。
5. **クラスタの誕生と死**  
   - 改善が見込めないクラスタのTRは縮小し、一定以下になれば破棄。有望な点の周辺に新しいクラスタを生成。

---

## 3. データ構造

常に管理する情報：

- 評価履歴  
  \[
    D = \{(x_i, y_i)\}_{i=1}^n, \quad y_i = f(x_i) + \text{noise}
  \]

- クラスタ集合  
  \[
    \mathcal{C} = \{C_1, \dots, C_K\}
  \]
  各クラスタ \(C_k\) は以下を持つ：
  - メンバーインデックス集合：\( I_k \subset \{1,\dots,n\} \)
  - 重心：\( \mu_k \in \mathbb{R}^d \)
  - トラストリージョンサイズ：\( \rho_k > 0 \)
  - 連続改善カウンタ：\( s_k \)
  - 連続失敗カウンタ：\( f_k \)

クラスタ \(C_k\) の「有望度」の一例として、クラスタ内最小値

\[
  q_k = \min_{i \in I_k} y_i
\]

を用い、これに基づいて評価予算の配分を行う。

---

## 4. CATS アルゴリズム詳細

### 4.1 初期化

1. **初期サンプリング**
   - LHS（ラテン超方格）などで \(N_0\) 点を一様サンプリング：  
     \[
       x_1, \dots, x_{N_0} \sim \text{LHS}([0,1]^d)
     \]
   - すべて評価し、\( y_i = f(x_i) \) を得る。

2. **初期クラスタリング**
   - 評価済み点 \(\{x_i\}\) に対して k-means などで \(K\) クラスタに分割。
   - 各クラスタの重心 \(\mu_k\)、初期TRサイズ \(\rho_k = \rho_{\text{init}}\) を設定。
   - カウンタ \(s_k, f_k = 0\) とする。

---

### 4.2 メインループ

最大評価回数 \(N_{\text{max}}\) に達するまで繰り返す。

#### ステップ 1：リクラスタリング

1. 最新の評価履歴 \(D\) に基づき、クラスタリングを更新。
   - 完全にゼロから再実行する代わりに、前回中心を初期値にした数ステップの k-means を回すなど、インクリメンタルな更新を想定。
2. 各クラスタの重心 \(\mu_k\) とメンバー \(I_k\) を更新。
3. 各クラスタの有望度 \(q_k\) を更新し、後続の予算配分に利用。

#### ステップ 2：クラスタごとのサロゲート学習

各クラスタ \(C_k\) について：

1. サブデータ  
   \(D_k = \{(x_i, y_i) \mid i \in I_k\}\) を取り出す。
2. 軽量サロゲートモデル \(s_k(x)\) を学習。例：  
   - ランダムフォレスト回帰
   - RBFネットワーク
   - 浅いニューラルネットワークなど
3. サロゲートが予測分散 \(\sigma_k(x)\) を提供できない場合、ランダムフォレストの木間分散などで擬似的な不確実性指標を定義する。

#### ステップ 3：トラストリージョン内での局所候補生成

クラスタ \(k\) に対し、トラストリージョンを

\[
  \text{TR}_k = \{ x \mid \|x - \mu_k\|_\infty \le \rho_k \} \cap [0,1]^d
\]

のように定義する（L∞ ノルムによるハイパーキューブ）。

1. バッチサイズ \(B\) に対して、クラスタごとの候補数 \(B_k\) を有望度に応じて配分：  
   \[
     B_k \propto \frac{1}{\text{rank}(q_k)} + \varepsilon
   \]
2. 各クラスタで獲得関数を定義：  
   - 例：UCB（Upper Confidence Bound）
     \[
       \text{UCB}_k(x) = s_k(x) - \beta \sigma_k(x)
     \]
3. TR\(_k\) 内で \(\text{UCB}_k(x)\) を最大化するような候補点を生成：
   - 簡易CMA-ES、DE、ランダム再始動付き局所探索などを利用。
   - サロゲート最適化で得た候補 \(B_k^{\text{local}}\) に加え、TR内の一様サンプリング由来の候補 \(B_k^{\text{rand}}\) も生成し、
     \(B_k^{\text{local}} + B_k^{\text{rand}} = B_k\) となるよう調整する。

#### ステップ 4：大域探索候補の追加

- 評価履歴から上位 \(P\) 個の良い点を抽出し、それらにガウス混合モデル（GMM）をフィット。
- GMM からサンプリングした点、および完全一様サンプルを組み合わせて、
  **大域探索候補**として \(B_{\text{global}}\) 点程度を生成する。
- これにより、既知のクラスタ外の領域へもジャンプしやすくなる。

#### ステップ 5：評価するバッチの選択

- 全候補集合：  
  \[
    \mathcal{X}_{\text{cand}} = \bigcup_k \{x^{\text{local}}_{k,j}\} \cup \text{global candidates}
  \]
- ここから、例えば
  - UCB や EI の値が良い点を優先しつつ、
  - 既評価点からの距離による多様性も考慮して
- 最大 \(B\) 点を選択し、実際に評価する。

#### ステップ 6：評価と履歴更新

- 選択された点 \(x\) を並列に評価し、新たな値 \(y = f(x)\) を取得。
- これらを評価履歴 \(D\) に追加する。

#### ステップ 7：トラストリージョンサイズの適応

各クラスタ \(C_k\) に対して：

- そのクラスタ内での最良値が更新されたかをチェック：
  - 改善あり：  
    \(s_k \leftarrow s_k + 1\), \(f_k \leftarrow 0\)
  - 改善なし：  
    \(f_k \leftarrow f_k + 1\)

- TR サイズの更新ルールの一例：
  - 一定回数改善がない場合（例：\(f_k \ge F_{\text{shrink}}\)）：  
    \[
      \rho_k \leftarrow \alpha_{\text{shrink}} \rho_k, \quad 0 < \alpha_{\text{shrink}} < 1
    \]
  - 改善が続いた場合（例：\(s_k \ge S_{\text{expand}}\)）：  
    \[
      \rho_k \leftarrow \min(\alpha_{\text{expand}} \rho_k, \rho_{\max})
    \]

- \(\rho_k\) が極端に小さくなり、かつ改善も見込めないクラスタは **kill** し、
  - 最近見つかった良好な点周辺に新しいクラスタを生成して、探索資源を再配置する。

---

## 5. 疑似コード

以下は、Python風の疑似コードである。実装言語に応じて書き換え可能。

```python
def CATS_optimize(f, d, N_max, B, K,
                  N0=20, rho_init=0.25, rho_min=1e-3,
                  alpha_shrink=0.5, alpha_expand=1.5):

    # --- 初期サンプリング ---
    X = lhs_sampling(N0, d)         # shape (N0, d)
    y = np.array([f(x) for x in X]) # shape (N0,)

    # --- 初期クラスタリング ---
    clusters = init_clusters(X, y, K, rho_init)

    while len(X) < N_max:
        # 1. 軽いリクラスタリング
        clusters = recluster(X, y, clusters, K)

        candidates = []

        # 2. 各クラスタでサロゲート → 候補生成
        for C in clusters:
            Xk, yk = X[C.indices], y[C.indices]
            surrogate = fit_surrogate(Xk, yk)  # RF/RBF/NN 等
            Bk = allocate_budget(C, clusters, B)  # 有望度に応じ配分

            local_cands = optimize_acquisition_in_TR(
                surrogate, C.center, C.rho, Bk
            )
            rand_cands  = random_in_TR(
                C.center, C.rho, n=max(1, Bk // 4)
            )

            candidates.extend(local_cands + rand_cands)

        # 3. 大域探索候補
        global_cands = global_exploration_candidates(
            X, y, B_global=max(1, B // 5)
        )
        candidates.extend(global_cands)

        # 4. 評価するバッチの選択（多様性も考慮）
        eval_points = select_batch(candidates, B, X)

        # 5. 評価
        y_new = np.array([f(x) for x in eval_points])

        # 6. 履歴更新
        X = np.vstack([X, eval_points])
        y = np.concatenate([y, y_new])

        # 7. TR の適応・クラスタの生死更新
        clusters = update_trust_regions(
            clusters, X, y, rho_min,
            alpha_shrink, alpha_expand
        )

    best_idx = np.argmin(y)
    return X[best_idx], y[best_idx]
```

上記の補助関数の役割の例：

- `lhs_sampling(N0, d)`  
  - LHS で \(N_0\) 点を生成。
- `init_clusters(X, y, K, rho_init)`  
  - k-means でクラスタを作り、重心・TRサイズ・カウンタを初期化。
- `recluster(X, y, clusters, K)`  
  - 既存クラスタ中心を初期値として再クラスタリング。
- `fit_surrogate(Xk, yk)`  
  - 軽量サロゲート（RF 等）を学習。
- `allocate_budget(C, clusters, B)`  
  - 有望度 \(q_k\) に基づいて、各クラスタに割り当てる候補数を決定。
- `optimize_acquisition_in_TR(surrogate, center, rho, Bk)`  
  - TR内で獲得関数の最適化を行い、局所候補点を \(B_k\) 個生成。
- `random_in_TR(center, rho, n)`  
  - TR内に一様分布で \(n\) 点をサンプリング。
- `global_exploration_candidates(X, y, B_global)`  
  - 上位少数点に GMM をフィットし、そこからサンプリング＋一様サンプルを混ぜて大域候補生成。
- `select_batch(candidates, B, X)`  
  - 既存点との距離や獲得関数値に基づいて、多様性のあるバッチを選択。
- `update_trust_regions(clusters, X, y, rho_min, alpha_shrink, alpha_expand)`  
  - 改善/非改善カウンタに基づき TR サイズを更新し、必要ならクラスタの kill / 新設を行う。

---

## 6. 実用的な実装メモ

### 6.1 ライブラリ例（Python）

- NumPy / SciPy：数値計算
- scikit-learn：
  - `KMeans` （クラスタリング）
  - `RandomForestRegressor`（サロゲート）
  - `GaussianMixture`（GMM）
- `scipy.optimize.differential_evolution` など：TR内獲得関数の最適化

### 6.2 推奨初期設定の一例

- 次元 \(d \le 10\)、評価上限 \(N_{\max} \approx 2000\)
- クラスタ数：\(K = 3\)〜5
- 初期点数：\(N_0 \approx 10d\)
- バッチサイズ：\(B = 5\)〜20
- 初期TRサイズ：\(\rho_{\text{init}} = 0.25\)
- TR最小サイズ：\(\rho_{\min} = 10^{-3}\)

### 6.3 アルゴリズムの性質

- **並列評価に適している**  
  毎イテレーションでバッチを生成するため、そのまま並列実行しやすい。
- **高次元に比較的スケール**  
  全空間に巨大なGPを張らず、クラスタごとに小さなサロゲートを学習するだけで済む。
- **多峰性へのロバスト性**  
  複数クラスタ + TR により、複数の「谷」を並列追跡でき、ダメな谷は縮小・破棄、有望な谷にリソース集中という挙動が自然に出る。
- **実装容易性**  
  すべて既存ライブラリで実装可能であり、アルゴリズム構造もシンプル。

---

## 7. まとめ

CATS は、

- クラスタリングにより評価済み点の構造を明示的に捉え、
- 各クラスタごとにサロゲートとトラストリージョンを持たせ、
- 局所探索と大域探索をバランス良く行う

という設計の、ブラックボックス最適化アルゴリズムである。

特に、

- **多峰性の強い関数**
- **中程度の次元**
- **評価回数が限られた設定**
- **並列評価環境**

において実用的に利用できることを目指した構成になっている。
