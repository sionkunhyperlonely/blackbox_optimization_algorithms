# BALE-Opt: Bandit-guided Adaptive Local Ensemble Optimization

## 1. 問題設定

最適化したい関数を

\[
\min_{x \in \mathcal{X}} f(x)
\]

とする。ここで:

- \(\mathcal{X} \subset \mathbb{R}^d\) は **有界なボックス領域**（各次元に上下限あり）
- \(f(x)\) は
  - 黒箱（勾配や内部構造は不明）
  - 評価が高コスト（シミュレーション、実験など）
  - ノイズを含んでもよい（平均的な性能の改善を目指す）

このような状況で、**サンプル効率が高く、実装が容易で、並列化しやすいブラックボックス最適化アルゴリズム**を設計する。

---

## 2. アルゴリズム概要

アルゴリズム名（仮）: **BALE-Opt**  
**B**andit-guided **A**daptive **L**ocal **E**nsemble Optimization

中核となるアイデアは以下の 4 つの要素を統合することにある。

1. **アンサンブル・サロゲートモデル**
   - Random Forest（RF）
   - 局所 kNN 回帰
   - RBF カーネル回帰 or 小規模 MLP
2. **複数の獲得関数（acquisition function）**
   - EI（Expected Improvement）
   - LCB（Lower Confidence Bound）
   - Pure Exploitation（予測値の単純最小化）
3. **マルチアームドバンディット（MAB）**
   - 「どのサロゲート × 獲得関数」が実際に改善につながっているかをオンライン学習
4. **局所トラストリージョン**
   - ベスト解周りに複数の局所探索領域を維持し、拡大・縮小を自動調整
   - グローバル探索と局所探索を自動的にバランス

---

## 3. 構成要素の詳細

### 3.1 観測データとサロゲートアンサンブル

観測データ集合:

\[
\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N,\quad y_i = f(x_i)
\]

これに対して 3 種類のサロゲートモデルを構築する。

1. **Random Forest レグレッサ**
   - 非線形・ロバスト・実装容易
   - 予測値 \(\hat f_{\mathrm{RF}}(x)\)、木ごとのばらつきから分散 \(\sigma_{\mathrm{RF}}(x)\) を推定

2. **kNN ローカル回帰**
   - 近傍 \(k\) 点の加重平均
   - \(\hat f_{\mathrm{kNN}}(x)\)、局所ばらつきから分散 \(\sigma_{\mathrm{kNN}}(x)\)

3. **RBF 回帰 / 小さな MLP**
   - 滑らかなグローバル近似
   - \(\hat f_{\mathrm{RBF}}(x)\)、ドロップアウトなどから不確実性指標 \(\sigma_{\mathrm{RBF}}(x)\)

これらをモデルインデックス \(m \in \{1,2,3\}\) で識別する。

---

### 3.2 獲得関数

現在のベスト値を

\[
f_{\min} = \min_{(x_i, y_i) \in \mathcal{D}} y_i
\]

とする。

モデル \(m\) の予測を \(F_m(x)\)（平均 & 分散を持つ確率変数）として、次の獲得関数を定義する。

#### (1) Expected Improvement (EI)

\[
a_{\mathrm{EI}}(x; m) = \mathbb{E}\big[(f_{\min} - F_m(x))_+\big]
\]

ここで \((u)_+ = \max(u, 0)\)。  
正規分布を仮定すると、標準的な閉形式で計算できる。

#### (2) Lower Confidence Bound (LCB)

最小化問題なので、下側信頼限界を用いる:

\[
a_{\mathrm{LCB}}(x; m) = \hat f_m(x) - \kappa \, \sigma_m(x)
\]

\(\kappa > 0\) は探索・活用のトレードオフを決めるパラメータ。

#### (3) Pure Exploitation (PE)

\[
a_{\mathrm{PE}}(x; m) = \hat f_m(x)
\]

予測値の最小化のみを行う、完全活用型。

獲得関数インデックスを \(k \in \{\mathrm{EI}, \mathrm{LCB}, \mathrm{PE}\}\) とする。

---

### 3.3 マルチアームドバンディット層

**腕（アーム） = (サロゲートモデル m, 獲得関数 k)** の組み合わせ:

\[
\mathcal{A} = \{(m,k)\}
\]

各腕 \(a \in \mathcal{A}\) に対して:

- 使用回数
- 累積報酬
- 報酬分布の推定パラメータ（例: 正規分布の平均・分散）

を保持する。

#### 報酬設計

イテレーション \(t\) で腕 \(a\) を選び、点 \(x_t\) を評価した結果 \(y_t = f(x_t)\) を得るとする。  
評価前のベスト値を \(f_{\min}^{\mathrm{old}}\)、評価後のベスト値を \(f_{\min}^{\mathrm{new}}\) とすると、報酬を

\[
r_t = \max\left(0,\; f_{\min}^{\mathrm{old}} - f_{\min}^{\mathrm{new}}\right)
\]

と定義する。つまり「ベスト値がどれだけ改善したか」を報酬とする。

この報酬を使って、各腕の「有効性」をオンラインに学習する。  
更新には Thompson Sampling や UCB1 などの標準的 MAB 手法を利用できる。

---

### 3.4 トラストリージョン

局所探索を行うために、**複数の局所トラストリージョン**を維持する。

各リージョン \(R_j\) は:

- 中心: \(c_j \in \mathbb{R}^d\)
- スケール: \(\Delta_j \in \mathbb{R}^d_+\)（各次元ごとの半径）

で定義され、探索領域は

\[
x \in \bigl[c_j - \Delta_j,\; c_j + \Delta_j\bigr] \cap \mathcal{X}
\]

となる。

#### 更新ルールの例

- リージョン内で一定回数連続して改善が観測された場合:  
  \(\Delta_j \leftarrow \alpha_{\mathrm{grow}} \Delta_j\) （\(\alpha_{\mathrm{grow}} > 1\)）
- しばらく改善がない場合:  
  \(\Delta_j \leftarrow \alpha_{\mathrm{shrink}} \Delta_j\) （\(\alpha_{\mathrm{shrink}} < 1\)）
- \(\Delta_j\) がある閾値以下に縮小したリージョンは削除
- 新たな良好解が得られた場合、その点を中心とする新リージョンを追加

各リージョンごとに独立したバンディットを持ち、**どのサロゲート×獲得関数がその局所で有効か**を学習する。

---

## 4. BALE-Opt のアルゴリズムフロー

### 4.1 初期化フェーズ

1. **初期サンプリング**
   - Latin Hypercube Sampling（LHS）や一様乱数などで \(n_{\mathrm{init}}\) 点をサンプリング
   - それぞれについて \(y_i = f(x_i)\) を評価し、\(\mathcal{D}\) を構成

2. **サロゲート学習**
   - RF, kNN, RBF/MLP を \(\mathcal{D}\) に対して学習し、アンサンブルモデルを得る

3. **トラストリージョン初期化**
   - \(\mathcal{D}\) の上位 \(K\) 個の良好解を中心として、同一サイズのトラストリージョンを作る

4. **MAB の初期化**
   - 腕集合 \(\mathcal{A} = \{(m,k)\}\) を定義
   - グローバル用バンディットと、各トラストリージョン用バンディットを用意し、報酬統計を初期化

---

### 4.2 1 イテレーションの詳細

以下を最大評価回数（`max_evals`）や時間制限に達するまで繰り返す。

1. **モード選択: グローバル探索 vs 局所探索**
   - 確率 \(p_{\mathrm{global}}\) で全域探索モード
   - 確率 \(1 - p_{\mathrm{global}}\) で、トラストリージョンの一つを選んで局所探索
   - \(p_{\mathrm{global}}\) はイテレーションが進むにつれ徐々に小さくしてもよい

2. **該当モードのバンディットから腕 \(a^* = (m^*, k^*)\) を選択**
   - 例: Thompson Sampling によって、報酬分布のサンプルが最大の腕を選択

3. **候補点生成**
   - 選んだサロゲート \(m^*\) と獲得関数 \(k^*\) を用いて、対象領域内（全域 or 選択リージョン内）で獲得関数
     \[
     a_{k^*}(x; m^*)
     \]
     を最大化する候補点を求める
   - 実装上は、
     - 対象領域内で多数のランダムサンプルを生成し
     - それらに対して獲得値を評価し
     - その中で最良の点を採用する
   - 既存点との最小距離制約 \(\|x - x_i\| \ge \delta_{\min}\) を入れて、多様性を確保してもよい

4. **真の関数評価**
   - 候補点 \(x_{\mathrm{new}}\) に対して真の関数 \(y_{\mathrm{new}} = f(x_{\mathrm{new}})\) を評価
   - \(\mathcal{D} \leftarrow \mathcal{D} \cup \{(x_{\mathrm{new}}, y_{\mathrm{new}})\}\)

5. **報酬計算 & バンディット更新**
   - 評価前のベスト \(f_{\min}^{\mathrm{old}}\)、更新後のベスト \(f_{\min}^{\mathrm{new}}\) を算出
   - 報酬:
     \[
     r = \max(0, f_{\min}^{\mathrm{old}} - f_{\min}^{\mathrm{new}})
     \]
   - 選択した腕 \(a^*\) に対して報酬 \(r\) でバンディットを更新

6. **トラストリージョン更新**
   - 新たなベスト解が得られた場合、その周囲に新リージョンを追加 or 既存リージョンを更新
   - 各リージョンの最近の改善状況に応じて \(\Delta_j\) を拡大・縮小

7. **サロゲート再学習 / インクリメンタル更新**
   - 毎イテレーション or 数ステップごとにアンサンブルモデルを再学習
   - 評価コストが高ければ、例えば「5 ステップごとにのみ再学習」などにしてもよい

---

## 5. 擬似コード（Python 風）

以下はアルゴリズム全体の流れを示す擬似コードである（実行可能な完全コードではないが、実装の骨格として利用できる）。

```python
def bale_optimize(f, bounds, max_evals, n_init=20, K=3, p_global=0.3):
    # 1. 初期サンプリング
    D = []  # list of (x, y)
    for x in latin_hypercube(bounds, n_init):
        y = f(x)
        D.append((x, y))

    # 2. サロゲート学習
    models = train_surrogate_ensemble(D)  # {"RF": ..., "kNN": ..., "RBF": ...}

    # 3. トラストリージョン初期化
    trust_regions = init_trust_regions(D, K, bounds)

    # 4. MAB 初期化（グローバル + 各リージョン用）
    global_bandit = init_bandit(models, acquisitions=["EI", "LCB", "PE"])
    region_bandits = {
        j: init_bandit(models, ["EI", "LCB", "PE"])
        for j in range(len(trust_regions))
    }

    evals = len(D)
    while evals < max_evals:
        # モード決定
        if random.random() < p_global or not trust_regions:
            mode = "global"
            bandit = global_bandit
            region = None
        else:
            mode = "local"
            region_id = select_region(trust_regions)  # 例: 最近の改善に応じた重み付き選択
            bandit = region_bandits[region_id]
            region = trust_regions[region_id]

        # 腕選択（サロゲート × 獲得関数）
        model_name, acq_name = bandit_select(bandit)
        model = models[model_name]

        # 候補点生成
        if mode == "global":
            x_new = propose_candidate_global(model, acq_name, bounds, D)
        else:
            x_new = propose_candidate_region(model, acq_name, region, bounds, D)

        # 真の関数評価
        y_new = f(x_new)
        D.append((x_new, y_new))
        evals += 1

        # 報酬計算
        old_best = min(y for _, y in D[:-1])
        new_best = min(y for _, y in D)
        reward = max(0.0, old_best - new_best)

        # バンディット更新
        bandit_update(bandit, (model_name, acq_name), reward)

        # トラストリージョン更新
        trust_regions = update_trust_regions(trust_regions, D, bounds)

        # サロゲート更新（必要に応じて）
        if evals % 5 == 0:
            models = train_surrogate_ensemble(D)

    # 最終結果
    x_best, y_best = min(D, key=lambda xy: xy[1])
    return x_best, y_best, D
```

---

## 6. 実用面での考察

### 6.1 実装容易性

- サロゲートは scikit-learn 等の汎用ライブラリで実装可能
- 獲得関数最大化は「ランダムサーチ + ベスト選択」から始めれば十分
- MAB も標準的な Thompson Sampling / UCB の実装でよい

### 6.2 並列化

- 1 イテレーションで複数候補点をサンプリングし、独立に評価するよう拡張しやすい
- 評価結果が返ってきた順にバンディットとサロゲートを更新することで、非同期最適化も可能

### 6.3 次元数とスケーラビリティ

- RF や MLP は中程度の次元（数十次元）まで現実的に扱える
- 高次元では、特徴選択・ランダムサブスペース・低次元射影などと組み合わせて拡張する

### 6.4 制約付き最適化への拡張

- 制約違反ペナルティを目的関数に付与する単純な方法
- あるいは制約可否を別サロゲート（分類器）でモデル化し、
  - 「制約を満たす確率」×「改善量（EI/LCB など）」の形式の獲得関数を用いる

---

## 7. 新規性と拡張アイデア

BALE-Opt は、既存の以下のような手法とは構造が異なる。

- 単一 GP + EI/LCB による標準的ベイズ最適化
- RF ベースの BO（例: SMAC 系）
- CMA-ES や他の進化戦略

BALE-Opt では、

1. **複数種類のサロゲートモデル（アンサンブル）**
2. **複数の獲得関数**
3. **それらの組み合わせを「腕」とみなす MAB**
4. **トラストリージョンを用いたローカル探索とグローバル探索の自動切り替え**

を一体化し、「どのモデル・獲得関数がどの局所領域で有効か」をオンラインに学習する点が特徴的である。

研究・実務レベルでの今後の方向性としては:

- 標準ベンチマーク関数（Rastrigin, Ackley, Rosenbrock 等）における既存 BO/CMA-ES との比較
- ノイズレベル・次元数・多峰性の違いに対するロバストネス評価
- MAB の種類（Thompson vs UCB vs ε-greedy）や報酬設計の違いの分析

などが考えられる。

---

以上が、**実用的かつ拡張性の高いブラックボックス最適化アルゴリズム「BALE-Opt」の設計仕様**である。
