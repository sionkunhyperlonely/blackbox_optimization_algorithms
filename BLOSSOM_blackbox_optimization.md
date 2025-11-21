# BLOSSOM: Bandit-guided Local Surrogate & State-space Mixture Optimization

このドキュメントでは、ブラックボックス最適化のための新しい実用的アルゴリズム **BLOSSOM**  
(*Bandit-guided LOcal Surrogate & State-space Mixture Optimization*) をまとめます。

---

## 1. 問題設定

- 最小化したいブラックボックス関数  
  \( f : \mathcal{X} \subset \mathbb{R}^d \to \mathbb{R} \)
- 勾配情報なし
- 評価コストは高い可能性がある
- ノイズを含む評価も許容
- 領域 \(\mathcal{X}\) はボックス制約 \([\ell, u]^d\) とする

目的は、限られた評価回数の下で、真の最小値に近い点 \(x^*\) を見つけることです。

---

## 2. アルゴリズムの基本アイデア

BLOSSOM は、探索空間を「**ローカルな専門家 (expert)**」の集合として扱います。

- 各専門家は、自分の担当領域に対して **軽量な局所サロゲートモデル**（線形 + RBF など）を持つ
- 専門家ごとに「次に評価すべき候補点」と「期待される改善量」を計算する
- **マルチアームドバンディット（UCB）** の枠組みで、どの専門家を選ぶかを決定
- 評価結果を使って、各専門家の中心・半径を更新
- 必要に応じて **新しい専門家の生成** や **不要な専門家の統合** を行う

ベイズ最適化のような重いグローバルGPを避け、
CMA-ES のような単一分布依存でもない、
「**複数の局所モデル + バンディット制御**」というハイブリッド構造が特徴です。

---

## 3. データ構造

### 3.1 アーカイブ

評価済み点の集合

\[
  \mathcal{A} = \{(x_i, y_i)\}_{i=1}^N, \quad y_i = f(x_i)
\]

### 3.2 専門家（ローカルモデル）の集合

各専門家 \(E_k\) は以下を持つ：

- 中心: \( c_k \in \mathcal{X} \)
- 半径: \( r_k > 0 \)
- 担当データ:
  \[
    \mathcal{A}_k = \{(x_i, y_i) \in \mathcal{A} : \|x_i - c_k\|_2 \le r_k\}
  \]
- ローカルサロゲートモデル: \( \hat{f}_k(x) \approx f(x) \)
- 不確実性推定: \( s_k(x) \ge 0 \)
- 選択回数: \( n_k \)（何回この専門家から候補点を出したか）

さらに、**グローバル探索用の疑似専門家 \(E_0\)** を一つ用意します。  
\(E_0\) はサロゲートモデルを持たず、アーカイブが疎な領域を探す役割を持ちます。

---

## 4. ローカルサロゲートモデル

各専門家 \(E_k\) の担当データ集合 \(\mathcal{A}_k\) に対して、軽量なサロゲートモデルを構築します。

### 4.1 入力の再スケーリング

計算安定性のため、事前に \(\mathcal{X}\) を \([0,1]^d\) に線形変換しておきます。

### 4.2 基底関数の例

- 低次数多項式: \(1, x_j, x_j^2\) など
- ランダムRBF:
  \[
    \phi_m(x) = \exp(-\gamma_m \|x - u_m\|_2^2)
  \]
  - \(u_m\): \(\mathcal{A}_k\) からランダムに選ぶ中心
  - \(\gamma_m\): 適当なスケールパラメータ

これらを組み合わせて、数十〜百程度の次元の特徴ベクトル \(\phi(x)\) を作ります。

### 4.3 リッジ回帰による当てはめ

ローカルサロゲートを線形モデル \(\hat{f}_k(x) = w_k^\top \phi(x)\) で表現し、リッジ回帰でパラメータを推定します：

\[
  w_k = \arg\min_w \sum_{(x_i,y_i)\in \mathcal{A}_k} (y_i - w^\top \phi(x_i))^2 + \lambda\|w\|^2
\]

### 4.4 不確実性推定（簡易版）

- 残差分散 \(\hat{\sigma}_k^2\) を計算
- 疑似的な標準偏差として
  \[
    s_k(x) = \hat{\sigma}_k \sqrt{ \phi(x)^\top (X_k^\top X_k + \lambda I)^{-1} \phi(x) }
  \]
  を用いる（厳密なベイズ解釈でなくてもよい）

これにより、GPほど重くないが、「予測平均 + 不確実性」を出せるサロゲートが得られます。

---

## 5. ローカル獲得関数（Acquisition）

最小化問題として、専門家 \(E_k\) の局所領域

\[
  \mathcal{B}_k = \{x \in \mathcal{X} : \|x - c_k\|_2 \le r_k\}
\]

内での候補点を決めるために、**ローワー・コンフィデンス・バンド (LCB)** を使います。

### 5.1 グローバルベスト

\[
  f^\star = \min_i y_i
\]

### 5.2 LCB の定義

\[
  \text{LCB}_k(x) = \hat{f}_k(x) - \kappa\, s_k(x)
\]

- \(\kappa > 0\)：探索と活用のトレードオフを調整する係数

### 5.3 局所候補点

\[
  x_k^{\text{cand}} = \arg\min_{x \in \mathcal{B}_k} \text{LCB}_k(x)
\]

これは連続最適化問題なので、次のような軽量手法の組み合わせで近似的に解きます。

- ランダムサンプリングで初期点を数十個生成し LCB を評価
- 下位の数点から Nelder–Mead / パウエル法 / 小規模 CMA-ES などで局所探索
- 最も LCB が小さい点を \(x_k^{\text{cand}}\) とする

### 5.4 専門家の期待改善度

専門家 \(E_k\) の期待改善度スコアを

\[
  q_k = f^\star - \text{LCB}_k(x_k^{\text{cand}})
\]

と定義します。LCB が小さいほど \(q_k\) が大きくなり、「改善が期待できる」ことを意味します。

---

## 6. 専門家選択：UCB によるマルチアームドバンディット

時刻 \(t\) までに各専門家 \(E_k\) が選ばれた回数を \(n_k\) とします。  
専門家 \(E_k\) の **バンディットスコア** を

\[
  S_k = q_k + \alpha \sqrt{\frac{2 \log t}{n_k + 1}}
\]

で定義します。

- \(q_k\): サロゲートに基づく期待改善量
- 第2項: あまり選ばれていない専門家を楽観的に評価するボーナス項
- \(\alpha\): 探索の強さを制御するハイパーパラメータ

### 6.1 グローバル探索専門家 \(E_0\)

\(E_0\) については、例えば次のようにします。

- アーカイブの「局所密度」が最も低い点を候補とする
- その点に対する \(f\) の値を単純な k-NN などで近似し、LCB に相当する値を定義
- それを基に \(q_0\) を計算し、他の専門家と同様に UCB スコアを付ける

最終的に

\[
  k^\star = \arg\max_k S_k
\]

を選び、その専門家の候補点 \(x_{k^\star}^{\text{cand}}\) を実際に評価します。

---

## 7. 専門家の更新・生成・統合

新しい評価点 \((x_{\text{new}}, y_{\text{new}})\) を得たら、次の更新を行います。

### 7.1 アーカイブ更新

\[
  \mathcal{A} \leftarrow \mathcal{A} \cup \{(x_{\text{new}}, y_{\text{new}})\}
\]

グローバルベストも更新：

- もし \(y_{\text{new}} < f^\star\) なら
  - \(f^\star \leftarrow y_{\text{new}}\)
  - ベスト点 \(x^\star \leftarrow x_{\text{new}}\)

### 7.2 最も近い専門家の探索

\[
  k_{\text{near}} = \arg\min_k \| x_{\text{new}} - c_k \|_2
\]

- もし \(\| x_{\text{new}} - c_{k_{\text{near}}} \|_2 \le r_{k_{\text{near}}}\) なら、その専門家の担当領域とみなす
- そうでなければ、「どの専門家にも属さない新しい領域」と判断

### 7.3 信頼領域風の半径更新

専門家 \(E_k\) の担当領域内でのベスト値を \(y_k^{\min}\) とする。

- もし \(y_{\text{new}}\) が \(y_k^{\min}\) を十分改善していたら（例: \(y_{\text{new}} \le y_k^{\min} - \epsilon\)）
  - \(r_k \leftarrow \min(\gamma_{\text{grow}} r_k, r_{\max})\)
  - 例: \(\gamma_{\text{grow}} = 1.2\)
- 改善がなければ
  - \(r_k \leftarrow \max(\gamma_{\text{shrink}} r_k, r_{\min})\)
  - 例: \(\gamma_{\text{shrink}} = 0.7\)

### 7.4 中心の更新

担当データ \(\mathcal{A}_k\) のうち、上位 p% 良い点の重心を \(\bar{x}_k\) として、

\[
  c_k \leftarrow (1 - \eta) c_k + \eta \bar{x}_k
\]

のように、良い点に少しずつ中心を寄せていきます。

### 7.5 新しい専門家の生成

次のような場合は新しい専門家を作成します。

- 「どの専門家にも属さない」位置に評価点が現れた
- 既存専門家のサロゲートの予測誤差が大きい状況が頻発する：  
  \(|y_{\text{new}} - \hat{f}_k(x_{\text{new}})| > \tau s_k(x_{\text{new}})\) など

このとき

- 中心: \(c_{\text{new}} = x_{\text{new}}\)
- 半径: \(r_{\text{new}} = r_{\text{init}}\)（小さめの固定値）

を持つ新専門家を追加します。

### 7.6 不要な専門家の統合・削除

- 中心が互いに近く、半径が大きく重なり、担当データがほぼ同じ場合は統合する
- 長期間選ばれず（\(n_k\) が小さいまま）かつ \(q_k\) が常に低い専門家は削除してよい

これにより、専門家の数が増えすぎて計算コストが膨らむのを防ぎます。

---

## 8. BLOSSOM の擬似コード

Python 風の擬似コードを示します（概要）。

```python
def BLOSSOM(f, X_bounds, budget, n_init):
    # X_bounds: shape (d, 2), each row is [l_j, u_j]
    # budget: max number of function evaluations
    # n_init: initial evaluations

    A = []  # archive: list of (x, y)

    # 1. initial sampling (e.g., Latin Hypercube)
    X_init = latin_hypercube_sampling(n_init, X_bounds)
    for x in X_init:
        y = f(x)
        A.append((x, y))

    # 2. initial best
    x_best, y_best = min(A, key=lambda xy: xy[1])

    # 3. initialize experts
    Experts = []

    # local expert centered at best point
    E1 = Expert(center=x_best,
                radius=0.5 * np.linalg.norm(X_bounds[:, 1] - X_bounds[:, 0]),
                data_indices=list(range(len(A))),
                n_select=1)
    Experts.append(E1)

    # global exploration expert E0
    E0 = GlobalExplorationExpert()
    Experts.append(E0)

    # 4. main loop
    for t in range(len(A) + 1, budget + 1):

        q_list = []
        x_cand_list = []

        # 4-1. build local surrogates and optimize LCB (for k != 0)
        for k, Ek in enumerate(Experts):
            if isinstance(Ek, GlobalExplorationExpert):
                x_cand, q = Ek.propose(A, X_bounds, y_best)
                x_cand_list.append(x_cand)
                q_list.append(q)
            else:
                A_k = subset_of_A_within_radius(A, Ek.center, Ek.radius)
                Ek.fit_surrogate(A_k)
                x_cand = optimize_LCB(Ek, X_bounds)
                lcb_val = Ek.lcb(x_cand)
                q = y_best - lcb_val

                x_cand_list.append(x_cand)
                q_list.append(q)

        # 4-2. compute bandit scores
        S_list = []
        for k, Ek in enumerate(Experts):
            n_k = Ek.n_select
            t_float = float(t)
            S_k = q_list[k] + alpha * np.sqrt(2.0 * np.log(t_float) / (n_k + 1.0))
            S_list.append(S_k)

        # 4-3. select expert
        k_star = int(np.argmax(S_list))
        Ek_star = Experts[k_star]
        x_new = x_cand_list[k_star]

        # 4-4. evaluate
        y_new = f(x_new)
        A.append((x_new, y_new))
        Ek_star.n_select += 1

        # 4-5. update global best
        if y_new < y_best:
            y_best = y_new
            x_best = x_new

        # 4-6. update experts: center, radius, data membership, etc.
        update_experts(Experts, A, x_new, y_new,
                       r_min, r_max,
                       gamma_grow, gamma_shrink,
                       eta, tau)

    return x_best, y_best
```

実装時には、

- `latin_hypercube_sampling`
- `subset_of_A_within_radius`
- `optimize_LCB`
- `update_experts`
- `GlobalExplorationExpert` / `Expert` クラス

などを個別に設計する必要がありますが、上記の流れが BLOSSOM の全体像です。

---

## 9. 実用性と拡張

### 9.1 計算量・スケーラビリティ

- 各サロゲートは **局所** かつ **線形 + RBF** 型なので、
  - \(|\mathcal{A}_k|\) を数十〜百程度に制限すれば十分軽量
- 高次元問題でも、各専門家の領域が小さくなれば、
  実質的な局所有効次元が下がることを期待できます。

### 9.2 拡張の方向

- 離散・カテゴリカル変数が含まれる場合
  - One-hot 表現と Hamming 距離ベースの RBF
  - あるいはカテゴリごとに別専門家を用意
- 並列化
  - バンディットスコアの上位 \(M\) 個の専門家から候補点を生成し、
    並列に評価することで容易にマルチコア / マシン並列に対応

### 9.3 既存手法との違い（概略）

- ベイズ最適化 (GP)
  - 通常は 1 つのグローバル GP で全空間をカバー
  - BLOSSOM は **多数の局所的軽量モデル** をバンディット制御
- CMA-ES / 進化戦略
  - 1 つ（もしくは少数）の分布で全体を表現
  - BLOSSOM は **複数の局所領域** を並列に動かす
- TuRBO 型 BO
  - 少数のトラストリージョン + GP に基づく BO
  - BLOSSOM は **多数のローカル専門家 + バンディット + 軽量サロゲート** という組み合わせ

---

## 10. まとめ

BLOSSOM は、

- **複数の局所サロゲート専門家**
- **UCB によるバンディット的領域選択**
- **信頼領域風の半径更新と専門家の生成・統合**

を組み合わせた、実装を意識したブラックボックス最適化アルゴリズムです。

実際にコード化する際は、本ドキュメントの設計をたたき台として、
- 問題次元
- 評価コスト
- ノイズレベル
- 評価予算

などに応じて、サロゲートの種類やハイパーパラメータを調整していくことができます。
