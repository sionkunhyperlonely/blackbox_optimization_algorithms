# kNN-Trust-Region Evolutionary Search (KTRES)
**（kNN 信頼領域進化探索アルゴリズム）**

この文書では、完全ブラックボックス関数を対象とした、新しい実用的最適化アルゴリズム
**kNN-Trust-Region Evolutionary Search（KTRES; ケトレス）** を簡潔にまとめる。

---

## 0. 問題設定

- 最小化したいブラックボックス関数:  f: X ⊂ R^d → R
- 勾配情報は利用できない（完全ブラックボックス）
- 関数評価コストは中〜高（評価回数は少ない方がよい）
- ノイズは軽微（オプションで対処可能）
- 探索領域はボックス制約 X = [l1, u1] × ... × [ld, ud]

---

## 1. コンセプト概要

KTRES は次の 3 つの要素から成るブラックボックス最適化アルゴリズムである。

1. **ローカル探索：信頼領域 + 局所 2 次サロゲート**
   - 良好な解の周囲に球状の信頼領域を定義し、その内部のサンプルから 2 次回帰モデルを構築する。
   - 構築した局所 2 次モデルの最小点を、次のローカル探索候補とする。

2. **グローバル探索：kNN ベースのスコアリング**
   - 全アーカイブから k 近傍を用いた粗い予測値（平均）と、既存点からの距離（未探索度）を計算。
   - 「予測値が良い」かつ「未探索な領域」を高スコアとする指標 S(x) を定義し、
     それに基づきグローバル候補点を選択する。

3. **二重メモリと適応制御**
   - 全ての評価履歴を保持するアーカイブ A と、複数の信頼領域集合 R を同時に管理する。
   - 各信頼領域は、改善があれば半径を拡大し、停滞すれば半径を縮小する TR ライクな適応を行う。

Bayesian Optimization のような重い GP 学習を避けつつ、CMA-ES とは異なり、
1 つのガウス分布ではなく複数の局所構造と kNN スコアを組み合わせて探索する点が特徴である。

---

## 2. 主なデータ構造とハイパーパラメータ

### 2.1 データ構造

- **アーカイブ A = {(xi, yi)}**
  - すべての評価点と目的値を保存する。
- **信頼領域集合 R = {(cj, rj)}**
  - cj: 信頼領域 j の中心（アーカイブ中の点）
  - rj: 半径
- **エリート集合 E**
  - A のうち目的値が良い上位 Ne 点からなる集合。

### 2.2 代表的なハイパーパラメータ

- 初期サンプル数 N0 = 20 + 4d
- エリート数 Ne = min(0.2 × |A|, 40)
- 最大信頼領域数 M = 5
- 初期半径 r_init = 0.25 × (u − l)（各次元幅の約 1/4）
- 半径拡大率 γ_inc = 1.5, 縮小率 γ_dec = 0.6
- kNN の近傍数 K_nn = min(10, |A|)
- グローバル候補母集団サイズ K_global = 200
- 1 ステップの評価数 B_local + B_global（例：ローカル 3 / グローバル 7）

---

## 3. アルゴリズムの流れ

### 3.1 初期化

1. 探索領域 X に対して、Latin Hypercube Sampling などで N0 点サンプリング。
2. 各点 xi に対して yi = f(xi) を評価し、アーカイブ A に追加。
3. A を目的値昇順にソートし、上位 Ne 点をエリート集合 E とする。
4. E を k-means で最大 M 個のクラスタに分割し、各クラスタの最良点を中心 cj とする。
5. 各中心に初期半径 r_init を割り当てて信頼領域集合 R を構成する。

### 3.2 各イテレーションの処理概要

1. **アーカイブ更新とエリート更新**
   - 新評価点を A に追加し、再ソートして E と x_best を更新する。

2. **信頼領域の再構成**
   - E を再度クラスタリングし、各クラスタ最良点を新しい中心 cj とする。
   - 過去の半径 rj を引き継ぎつつ、必要に応じて調整する。

3. **ローカル 2 次サロゲートによる候補点生成**
   - 各信頼領域 Rj 内のデータ Sj を抽出する。
   - |Sj| < d + 1 の場合は、Rj 内から一様乱数で候補を生成。
   - 十分な点数があれば、中心 cj 周りで 2 次回帰モデル qj(z) をフィットし、
     その最小点をボックス制約＋球制約内に射影してローカル候補 x_j^loc を得る。

4. **kNN ベースのグローバル候補生成**
   - 領域全体から一様に K_global 点をサンプルし、それぞれについて：
     - A からの k 近傍による予測値 f_hat(z)（近傍の y の平均）を計算。
     - 既存点からの最小距離 d_min(z) を計算。
   - これらを正規化し、
     S(z) = λ · score_f(z) + (1 − λ) · score_d(z)
     を定義して総合スコアとする（λ は exploitation / exploration のバランス）。
   - S(z) が大きい上位 B_global 点をグローバル候補とする。

5. **評価と信頼領域半径の適応**
   - ローカル候補およびグローバル候補から選んだ点 X_new を評価し、A に追加。
   - 各信頼領域ごとに、中心 cj の値 y_c と新ローカル候補の値 y_new から改善率
     ρj = (y_c − y_new) / max(|y_c|, 1)
     を計算し、
     - ρj が十分大きければ rj を γ_inc 倍（改善 → 拡大）、
     - ρj が小さければ rj を γ_dec 倍（停滞 → 縮小）する。

6. **再起動（任意）**
   - 一定ステップ改善がない場合、性能の悪い信頼領域を捨て、
     エリート集合の多様な点から新しい信頼領域を生成する。

---

## 4. 擬似コード（概要）

```text
Input: f, bounds [l, u], eval_budget

# Initialization
A = {}
X0 = LatinHypercube(l, u, N0)
for x in X0:
    y = f(x)
    A.add((x, y))

while evals(A) < eval_budget:
    # 1. elites
    sort A by y ascending
    E = top Ne points from A

    # 2. trust regions from E
    clusters = kmeans(positions(E), k = min(M, |E|))
    R = {}
    for each cluster c:
        center = best point in c
        radius = previous_radius_or_r_init(center)
        R.add((center, radius))

    # 3. local candidates via quadratic surrogate
    local_candidates = []
    for (center, radius) in R:
        Sj = { (x, y) in A | ||x - center|| <= radius }
        if |Sj| < d + 1:
            x_loc = random_in_ball(center, radius, bounds)
        else:
            q = fit_quadratic_surrogate(Sj, center)
            x_loc = argmin_quadratic_in_ball(q, center, radius, bounds)
        local_candidates.add(x_loc)

    # 4. global candidates via kNN-based score
    Z = uniform_sample(bounds, K_global)
    compute f_hat(z) and d_min(z) using kNN on A
    normalize to scores and compute S(z)
    global_candidates = top B_global z by S(z)

    # 5. evaluate new points and adapt radii
    X_new = select B_local from local_candidates + global_candidates
    for x in X_new:
        y = f(x)
        A.add((x, y))
    adapt_radius_of_each_region_based_on_improvement()

return best x in A
```

---

## 5. 既存手法との違いと特徴

- Bayesian Optimization と異なり、グローバルサロゲートに GP や TPE を使わず、
  kNN 平均と距離を組み合わせた軽量なスコアのみで探索する。
- CMA-ES のように全空間を 1 つの分布でモデル化せず、複数の信頼領域に分けて局所モデルを構築。
- 古典的 Trust Region 法と異なり、勾配を使わない完全ブラックボックスかつ多信頼領域構造。
- グローバル側は「精度の高い回帰モデル」ではなく、安価な kNN 近似を使うため、
  実装と計算のコストが低い。

---

## 6. 実装のポイント（Python 例）

- アーカイブ管理: `numpy` 配列や `pandas.DataFrame`
- 初期サンプリング: `scipy.stats.qmc.LatinHypercube` など
- k-means: `sklearn.cluster.KMeans`
- 2 次回帰: 特徴展開 + `sklearn.linear_model.Ridge`
- kNN: `sklearn.neighbors.NearestNeighbors`

評価コストが非常に高い場合は、
- `K_global` を小さめにし、`B_local` を増やして局所探索を強化する、
といったチューニングが有効である。

---

以上が、KTRES アルゴリズムの Markdown 形式による要約である。