# CLOUD-BO: Contrastive Local-Global Decomposition Bayesian Optimizer

## 1. 問題設定

目的関数 \( f: \mathbb{R}^d \to \mathbb{R} \) は以下のようなブラックボックス関数とする：

- 微分不可能（勾配が使えない）
- 評価が高コスト（シミュレーション、実験など）
- ノイズを含む可能性あり
- 制約は「箱型制約」\(x \in [l,u]^d\) とする（一般制約は後述）

**目標：**  
限られた評価回数 \(N\) で、良好な近似最適解 \(x^\*\) を見つけるブラックボックス最適化アルゴリズムを設計する。

---

## 2. コンセプト概要

アルゴリズム名案：  
**CLOUD-BO: Contrastive Local-Global Decomposition Bayesian Optimizer**

### 2.1 背景

既存のブラックボックス最適化では大きく

- **グローバル探索型**：ベイズ最適化（GP + acquisition）、ランダム探索、進化戦略など
- **ローカル探索型**：信頼領域法（trust-region BO, TuRBO 系）、局所サロゲート + CMA-ES など

に分かれるが、多くは

> 「グローバル」と「ローカル」を *スケジュール* で切り替える  
> （前半グローバル、後半ローカル など）

という設計になっている。

### 2.2 基本アイデア

**CLOUD-BO の主眼は**「グローバル」と「ローカル」の **“意見の違い” を積極的に利用する探索** にある。

1. サロゲートモデルを「グローバル」と「ローカルクラスタごと」に **複数** 持つ
2. 候補点を  
   - 「グローバルモデルで良さそうだが、ローカルモデルとは意見が違う点」  
   - 「ローカルモデルが強く良いと主張する点」  
   を、**対比（contrastive）** させながら選ぶ
3. その結果、
   - 「局所解にハマって動けなくなる」
   - 「グローバルに薄く広く探索して終わる」
   の両方を避けることを狙う。

---

## 3. アルゴリズムの全体フロー

反復 \(t = 1,2,\dots\) で、以下を繰り返す：

1. 既存データをクラスタリング（入力空間上のクラスタ）
2. 各クラスタごとにローカルサロゲート（軽量なモデル）を構築
3. 全データでグローバルサロゲートを構築
4. グローバル獲得関数 \(a_\text{global}(x)\) と  
   各ローカル獲得関数 \(a_k(x)\) を定義
5. **コントラストスコア**
   \[
   C(x) = \alpha \cdot a_\text{global}(x) + \beta \cdot \max_k a_k(x) 
          + \gamma \cdot D(x)
   \]
   を最大化する点を候補として生成（\(D(x)\) は「グローバルとローカルの意見の差」）
6. 上位 \(B\) 個の候補点を評価し、データセットに追加

---

## 4. 詳細設計

### 4.1 初期点の生成

- \(n_\text{init}\) 個の初期点を  
  Latin Hypercube Sampling (LHS) や Sobol sequence で生成
- それぞれ評価して \(\mathcal{D} = \{(x_i, y_i)\}\) を初期化

ここで \(y_i = f(x_i)\)。

---

### 4.2 クラスタリング（ローカル領域の定義）

入力点 \(\{x_i\}\) を \(K\) クラスタに分け、  
各クラスタ \(k\) を局所領域と見なす。

- アルゴリズム：k-means などの標準クラスタリング
- 点が少ないときは \(K\) を小さくするか、クラスタリング自体をスキップ
- 例：\(\lvert \mathcal{D}\rvert < 5K\) の場合は \(K\) を減らす

---

### 4.3 サロゲートモデル

サロゲートとしては、計算コストとスケーラビリティを考慮し、

- 中〜高次元（\(d \sim 5-50\)）でも動く
- 再学習が高速

であることを重視する。

#### グローバルモデル \(M_\text{global}\)

- 入力：すべてのデータ \(\mathcal{D}\)
- 出力：予測平均 \(\mu_g(x)\)、不確実性 \(\sigma_g(x)\)
- 候補：
  - ランダムフォレスト（木の分散 → 不確実性）
  - ライトGBM
  - 小型 MLP ＋ MC dropout で不確実性推定

#### ローカルモデル \(M_k\)

- クラスタ \(k\) 内のデータのみで学習
- \(\mu_k(x), \sigma_k(x)\) を出す
- 候補：
  - 低次元近似（PCA など）＋ガウス過程
  - 小さなランダムフォレスト

---

### 4.4 獲得関数

一般的な Expected Improvement (EI) を利用する：

- ベスト値 \(y_\text{best} = \min_i y_i\)
- EI の定義：
  \[
  \text{EI}(x; \mu, \sigma) = \mathbb{E}[(y_\text{best} - Y(x))_+]
  \]

実装では、標準正規分布の CDF と PDF を用いた閉形式を使用する。

定義：

- グローバル獲得関数  
  \[
  a_\text{global}(x) = \text{EI}(x; \mu_g, \sigma_g)
  \]
- ローカル獲得関数  
  \[
  a_k(x) = \text{EI}(x; \mu_k, \sigma_k)
  \]

---

### 4.5 コントラスト項 \(D(x)\)

「グローバルとローカルの意見の違い」を表すスコア。

例として：

\[
D(x) = \left| \mu_g(x) - \min_k \mu_k(x) \right|
\]

- グローバルはあまり良くないと言っているが、あるローカルモデルが良いと言っている
- その逆

といった「ねじれ」を持つ点を優遇することで、  
谷の向こう側・新しい局所解を見つけやすくする。

---

### 4.6 最終スコア \(C(x)\)

それぞれのスコアを [0,1] にスケーリングしたうえで線形結合する：

\[
C(x) = \alpha \cdot \widehat{a}_\text{global}(x)
     + \beta \cdot \widehat{a}_\text{local}(x)
     + \gamma \cdot \widehat{D}(x)
\]

- \(\widehat{\cdot}\) は min-max 正規化された量
- \(\widehat{a}_\text{local}(x) = \max_k \widehat{a}_k(x)\)

ハイパーパラメータ例：

- \(\alpha = 0.5\)（グローバル EI の寄与）
- \(\beta = 0.3\)（ローカル EI の寄与）
- \(\gamma = 0.2\)（コントラストの寄与）

---

### 4.7 候補点の最適化（バッチ取得）

1. サンプル候補点集合 \(X_\text{cand}\) を多数生成  
   - Sobol シーケンスや一様乱数で \(n_\text{cand} \sim 10^3–10^4\) 点
2. 各点について \(C(x)\) を計算
3. \(C(x)\) の高い順に \(B\) 個を選択
4. それらを実際のブラックボックス関数 \(f\) に投げて評価
5. 得られた \((x, y)\) をデータセット \(\mathcal{D}\) に追加

計算資源に余裕がある場合、  
候補点の周囲で CMA-ES 等による短いローカル最適化を行い、さらに \(C(x)\) を高めることも可能。

---

## 5. 疑似コード（Python風）

```python
import numpy as np

def CLOUD_BO(f, bounds, N_eval,
             n_init=10, K=3, B=4,
             alpha=0.5, beta=0.3, gamma=0.2):
    '''
    f      : 評価関数 f(x)
    bounds : [(l1, u1), ..., (ld, ud)] のリスト
    N_eval : 総評価回数の上限
    n_init : 初期サンプル数
    K      : 最大クラスタ数
    B      : 1イテレーションあたりのバッチサイズ
    '''

    # 1. 初期サンプルの生成
    X = latin_hypercube(bounds, n_init)  # shape: (n_init, d)
    y = np.array([f(x) for x in X])

    while len(y) < N_eval:
        # 2. クラスタリング
        K_eff = min(K, max(1, len(y) // 5))  # データ数に応じて調整
        labels = kmeans(X, K_eff)            # returns cluster id per point

        # 3. サロゲート学習
        M_global = fit_surrogate(X, y)       # 例: ランダムフォレスト
        M_locals = []
        for k in range(K_eff):
            Xk = X[labels == k]
            yk = y[labels == k]
            if len(yk) < 3:
                M_locals.append(None)
            else:
                M_locals.append(fit_surrogate(Xk, yk))

        # 4. 候補点集合の生成
        X_cand = sobol_sampling(bounds, n_cand=2000)

        mu_g, sig_g = M_global.predict(X_cand, return_std=True)
        a_g = expected_improvement(mu_g, sig_g, y.min())

        a_locals_list = []
        mu_locals_list = []

        for Mk in M_locals:
            if Mk is None:
                a_locals_list.append(np.zeros(len(X_cand)))
                mu_locals_list.append(mu_g)  # ダミーとしてグローバル平均を再利用
            else:
                muk, sigk = Mk.predict(X_cand, return_std=True)
                a_locals_list.append(expected_improvement(muk, sigk, y.min()))
                mu_locals_list.append(muk)

        a_locals = np.max(np.vstack(a_locals_list), axis=0)
        mu_locals_min = np.min(np.vstack(mu_locals_list), axis=0)

        # 5. コントラスト項 D(x)
        D = np.abs(mu_g - mu_locals_min)

        # 6. 正規化関数
        def norm(v):
            v_min, v_max = v.min(), v.max()
            if v_max == v_min:
                return np.zeros_like(v)
            return (v - v_min) / (v_max - v_min)

        C = alpha * norm(a_g) + beta * norm(a_locals) + gamma * norm(D)

        # 7. 上位 B 点を選んで評価
        idx = np.argsort(-C)[:B]
        X_new = X_cand[idx]
        y_new = np.array([f(x) for x in X_new])

        # データ更新
        X = np.vstack([X, X_new])
        y = np.concatenate([y, y_new])

    best_idx = np.argmin(y)
    return X[best_idx], y[best_idx]
```

補助関数例：

- `latin_hypercube(bounds, n)`
- `kmeans(X, K)`
- `fit_surrogate(X, y)`（ランダムフォレストなど）
- `sobol_sampling(bounds, n_cand)`
- `expected_improvement(mu, sigma, y_best)`

などは、scikit-learn, scipy, sobol シーケンス実装などを使って実装できる。

---

## 6. 実用面での特徴

### 6.1 長所（意図）

1. **マルチモーダル問題に強い**  
   - クラスタごとのローカルモデルとグローバルモデルの「意見のズレ」を利用することで、  
     新しい谷・山を発見しやすい。

2. **中〜高次元にもある程度スケール**  
   - GP に依存せず、RF/GBDT/MLP などを使うことで  
     \(d \sim 50\) くらいまでは現実的。

3. **バッチ並列評価に自然対応**  
   - コントラストスコア上位 \(B\) 点をまとめて評価するだけでバッチ BO になる。

4. **実装が比較的シンプル**  
   - 必須要素は「クラスタリング＋複数サロゲート＋EI＋スコア合成」のみ。

---

### 6.2 想定される弱点・課題

- クラスタ数 \(K\) や重み \(\alpha,\beta,\gamma\) のチューニングが必要
- ノイズが非常に強い場合、クラスタリングとローカル学習が不安定になる可能性
- 複雑な制約条件を扱うには、
  - 可行領域推定モデル（feasibility classifier）を別途学習し、
  - 「可行確率 × \(C(x)\)」を最大化する、
  などの工夫が必要。

---

## 7. 発展バリアント案

さらなる新規性を加えたい場合の拡張案：

1. **クラスタごとに低次元潜在空間を学習**  
   - ローカルオートエンコーダで次元圧縮し、その潜在空間上で BO を回す
   - グローバルは元空間 or 別の潜在空間

2. **コントラストを「順位の不一致」で定義**  
   - 例：グローバル EI ランキングで上位だが、ローカルでは下位  
   - あるいはその逆を優先的にサンプリング

3. **意見の“合意点”を捨てる探索**  
   - グローバルと全ローカルが一致して「良い」と言う点は敢えて候補から外し、
   - 「どこかが反対している」点だけ探索することで多様性を上げる

---

## 8. 次のステップ

- 単純なベンチマーク関数（Branin, Hartmann, Ackley など）で Python 実装を検証
- ランダム探索や標準 BO, CMA-ES と比較して性能評価
- 実用タスク（ハイパーパラメータ最適化、シミュレーションチューニングなど）への適用

これにより、CLOUD-BO の有効性や特性（収束速度、多様性、ロバスト性）を定量的に評価できる。
