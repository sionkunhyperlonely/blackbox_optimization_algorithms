# MASSBO: Mixture of Adaptive Subspace Surrogate Black-box Optimizer

## 0. 問題設定

ブラックボックス関数 \( f: \mathbb{R}^d \to \mathbb{R} \) を最小化する問題を考える。

- 導関数なし（ブラックボックス）
- 評価コストが高い（1 回の評価が遅い）
- ノイズあり／なし両対応
- 探索領域はボックス制約 \( x \in [l, u]^d \)

目的は

\[
\min_{x \in [l,u]^d} f(x)
\]

を解くことである。


## 1. アルゴリズムのコアアイデア

MASSBO の特徴は、次の 5 点を**同時に**行うことにある。

1. **複数の局所ガウス分布（ローカルコンポーネント）による探索**
   - 各局所 j に対して分布 \(\mathcal{N}(m_j, \Sigma_j)\) と信頼領域半径 \(r_j\) を持つ。

2. **各局所で「低次元部分空間」を自動学習して探索**
   - 高次元空間でも、実質的には低次元構造が存在するという前提。
   - PCA やローカル線形モデルにより有効次元を推定。

3. **ローカルごとに簡易サロゲートモデルを構築**
   - 例: RBF、ランダムフォレスト、軽量 GP。
   - 部分空間上で Expected Improvement (EI) や Thompson Sampling により候補点を生成。

4. **ローカルへの計算資源配分をバンディットで決定**
   - 各ローカルの「最近の改善量」を報酬として UCB/TS により評価回数を配分。

5. **分布更新はランキングベースでノイズに頑健**
   - 各ローカル内の上位 \(q\%\) サンプルのみを用いて \(m_j, \Sigma_j\) を更新。
   - CMA-ES 的な更新だが、「複数局所＋低ランク＋サロゲート＋バンディット」という組み合わせが新しい。


## 2. データ構造

### グローバルアーカイブ

- \( \mathcal{A} = \{(x_i, f_i)\}_{i=1}^N \)

### ローカルコンポーネント（クラスタ） \( j = 1, \dots, K \)

- 中心: \(m_j \in \mathbb{R}^d\)
- 共分散: \(\Sigma_j \in \mathbb{R}^{d \times d}\)（低ランク表現）  
  \[
  \Sigma_j = U_j \Lambda_j U_j^\top + \sigma_j^2 I
  \]
  - \(U_j \in \mathbb{R}^{d \times k}\): 部分空間の基底（\(k \ll d\)）
- 信頼領域半径: \(r_j > 0\)
- サロゲートモデル: \(s_j(y)\)
- 性能統計:
  - 直近 \(M\) 回の平均改善量 \(\Delta_j\)
  - 評価回数 \(n_j\)


## 3. 部分空間学習とサロゲート構築

### 部分空間学習

ローカル j に対して、アーカイブから近傍点を抽出：

\[
\mathcal{A}_j = \{(x_i, f_i) \in \mathcal{A} \mid \|x_i - m_j\| \le c r_j\}
\]

その上で

- \(x_i - m_j\) を行ベクトルとする行列 \(X_j\) を作成。
- PCA（あるいは局所線形回帰の設計行列）で固有分解。
- 上位 k 個の固有ベクトルを \(U_j = [u_{j,1}, \dots, u_{j,k}]\) とする。

すると、任意の点は近似的に

\[
x \approx m_j + U_j y, \quad y \in \mathbb{R}^k
\]

と表現でき、探索は主に y 空間で行う。

### サロゲートモデル

- 入力: \(y_i = U_j^\top (x_i - m_j)\)
- 出力: \(f_i\)
- モデル例:
  - RBF: \( s_j(y) = \sum_\ell \alpha_\ell \phi(\|y-y_\ell\|) \)
  - ランダムフォレスト回帰
  - 軽量 Gaussian Process (GP)

候補点生成は、y 空間で EI 最大化（多スタートランダム探索）か Thompson Sampling により行う。


## 4. 1 ステップの処理フロー

1. **ローカル選択（バンディット）**
   - 各ローカル j の「報酬」 \(R_j\) を最近の改善量から算出。
   - 例えば UCB スコア：
     \[
     \text{score}_j = \hat{\mu}_j + \beta \sqrt{\frac{\ln t}{n_j + 1}}
     \]
   - 最大スコアの j を選択。

2. **部分空間・サロゲート更新**
   - 選ばれたローカル j について \(U_j\) と \(s_j\) を更新。

3. **ローカル j で候補点を B 個生成**
   1. y 空間で候補 \(y^{(b)}\) を EI/TS によりサンプリング。
   2. 元の空間に写像：
      \[
      x^{(b)} = \operatorname{clip}_{[l,u]}(m_j + U_j y^{(b)})
      \]
   3. 信頼領域制約：\(\|x^{(b)} - m_j\| \le r_j\) を満たさなければ棄却または縮小。

4. **目的関数評価とアーカイブ更新**
   - 各候補点について \(f(x^{(b)})\) を評価し、アーカイブに追加。

5. **ローカル分布更新（ランキングベース）**
   - ローカル j の最新 L 点から上位 \(q\%\) を選ぶ。
   - その平均 \(\bar{x}_j\) と共分散 \(\widehat{\mathrm{Cov}}_j\) を計算。
   - 更新：
     \[
     m_j \leftarrow (1-\eta_m)m_j + \eta_m \bar{x}_j
     \]
     \[
     \Sigma_j \leftarrow (1-\eta_\Sigma)\Sigma_j + \eta_\Sigma \widehat{\mathrm{Cov}}_j
     \]
   - PCA により再度 \(U_j, \Lambda_j\) を抽出。

6. **信頼領域半径の調整**
   - ベスト値が改善した場合：
     \[
     r_j \leftarrow \min(\gamma_{\text{inc}} r_j, r_{\max})
     \]
   - 改善しなかった場合：
     \[
     r_j \leftarrow \max(\gamma_{\text{dec}} r_j, r_{\min})
     \]

7. **ローカルの生成・削除**
   - グローバルベストから離れた領域に良い点が見つかった場合、新しいローカルを生成。
   - 長期間改善のないローカルは削除。


## 5. 擬似コード（Python 風）

```python
import numpy as np

def MASSBO(
    f,                  # ブラックボックス関数
    bounds,             # (l, u)
    d,                  # 次元
    K_init=3,           # 初期ローカル数
    k_subspace=5,       # 部分空間次元
    budget=1000,        # 総評価回数
    B_per_iter=5        # 1 iteration あたり評価回数
):
    # 1. 初期化
    A = []  # archive: list of (x, f(x))

    locals_ = []
    for j in range(K_init):
        m_j = np.random.uniform(bounds[0], bounds[1], size=d)
        Sigma_j = np.eye(d)
        r_j = 0.25 * np.linalg.norm(bounds[1] - bounds[0])
        U_j = np.eye(d)[:, :k_subspace]  # 部分空間の初期値
        stats = {"n": 0, "recent_improvements": []}
        locals_.append({
            "m": m_j, "Sigma": Sigma_j, "U": U_j,
            "r": r_j, "stats": stats, "surrogate": None
        })

    evals = 0
    best_x, best_f = None, np.inf

    while evals < budget:
        t = evals // B_per_iter + 1

        # 2. ローカル選択 (UCB)
        scores = []
        for loc in locals_:
            n_j = loc["stats"]["n"]
            if n_j == 0:
                mu_hat = 0.0
            else:
                recent = loc["stats"]["recent_improvements"][-10:]
                mu_hat = np.mean(recent) if len(recent) > 0 else 0.0
            ucb = mu_hat + 0.1 * np.sqrt(np.log(t) / (n_j + 1))
            scores.append(ucb)
        j = int(np.argmax(scores))
        loc = locals_[j]

        # 3. 部分空間学習・サロゲート更新
        #    (A から近傍点を抽出して U_j, surrogate を更新する処理を書く)

        # 4. 候補生成 & 評価
        new_points = []
        local_best_f_before = best_f

        for _ in range(B_per_iter):
            y = sample_y_from_surrogate(loc)  # EI/TS 最大化を行う部分
            x = project_to_bounds(loc["m"] + loc["U"] @ y, bounds)

            # 信頼領域制約
            diff = x - loc["m"]
            norm_diff = np.linalg.norm(diff)
            if norm_diff > loc["r"]:
                x = loc["m"] + diff * (loc["r"] / norm_diff)

            fx = f(x)
            evals += 1
            A.append((x, fx))
            new_points.append((x, fx))

            if fx < best_f:
                best_f, best_x = fx, x

            if evals >= budget:
                break

        # 5. 改善量を記録
        improvement = max(0.0, local_best_f_before - best_f)
        loc["stats"]["recent_improvements"].append(improvement)
        loc["stats"]["n"] += len(new_points)

        # 6. ローカル更新 (ランキングベース)
        update_local_distribution(loc, A)  # 上位 q% から m, Sigma, U を更新

        # 7. 信頼領域の更新
        if improvement > 0:
            loc["r"] = min(1.5 * loc["r"], max_radius(bounds))
        else:
            loc["r"] = max(0.5 * loc["r"], min_radius(bounds))

        # 8. ローカル生成/削除は適宜実装

    return best_x, best_f
```

補助関数 `sample_y_from_surrogate`, `project_to_bounds`, `update_local_distribution`,  
`max_radius`, `min_radius` などは、問題設定や実装環境に応じて具体化する。


## 6. 推奨ハイパーパラメータ

- 初期ローカル数: \(K_{\text{init}} = 3 \sim 5\)
- 部分空間次元: \(k_{\text{subspace}} = 3 \sim 10\)（次元 d に依存）
- 1 ステップあたり評価数: \(B_{\text{per\_iter}} = 2 \sim 10\)
- 信頼領域半径:
  - \(r_{\max} = 0.5 \|u - l\|\)
  - \(r_{\min} = 0.01 \|u - l\|\)
  - 拡大率: \(\gamma_{\text{inc}} = 1.5\)
  - 縮小率: \(\gamma_{\text{dec}} = 0.5\)
- ランキングベース更新:
  - 上位 20% のサンプルを利用
  - 学習率: \(\eta_m = 0.2, \eta_\Sigma = 0.1\)


## 7. 既存手法との比較

### CMA-ES との比較

- CMA-ES: 単一のガウス分布で全空間を探索。
- MASSBO:
  - 複数のガウス分布（複数局所）
  - 低ランク共分散＋部分空間学習
  - ローカルごとにサロゲートを構築
  - バンディットで計算資源を配分

多峰性・高次元・並列評価に対してより柔軟に対応できる。

### ベイズ最適化（単一 GP + EI）との比較

- 単一 GP は
  - 高次元・多峰性に弱い
  - 評価点が増えると計算コストも高くなる。
- MASSBO は
  - 「局所ごとの小さなサロゲート＋部分空間」でスケールさせる。
  - 多峰性への対応として複数の局所モデルを同時に運用。

### TuRBO / REMBO 系との関係

- TuRBO: 信頼領域を動かす高次元ベイズ最適化。
- REMBO: ランダム低次元埋め込みに基づく高次元 BO。
- MASSBO:
  - 信頼領域＋低次元埋め込みの考え方を共有しつつ、
  - 「複数局所＋ランキング ES 更新＋バンディット配分」という新しい組み合わせ。


## 8. 想定ユースケース

- 評価コストが「中〜重い」ブラックボックス（数秒〜数分）。
- 次元数 \(d\) が 20〜200 程度。
- 多峰性があり、局所解が多数存在する可能性。
- 並列計算資源があり、B 個ずつまとめて評価したい。

このような条件下で、MASSBO は

- 全域をラフにカバーしつつ、
- 有望な局所に探索資源を集中し、
- 部分空間学習で高次元性を克服し、
- サロゲートで評価回数を削減

といった振る舞いが期待できる。


## 9. 拡張の方向性

- **実装レベル**:
  - NumPy / PyTorch ベースでのリファレンス実装。
  - 並列評価・分散環境対応。

- **離散・組合せ最適化への拡張**:
  - 離散変数の one-hot 埋め込み。
  - Gumbel-softmax 的な連続緩和。

- **ノイズが大きい場合の強化**:
  - リサンプリング戦略。
  - ロバスト統計（メディアン、順位統計）を使った更新。

これらを組み合わせることで、より広いクラスのブラックボックス最適化問題に適用可能となる。
