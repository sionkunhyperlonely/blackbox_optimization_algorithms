# LASTO: Layered Adaptive Surrogate Trust Optimization
（多層適応サロゲート信頼領域最適化アルゴリズム）

> 注意：  
> ここで示す **LASTO** は、既存の代表的手法（Bayesian Optimization, CMA-ES, Nelder–Mead, Trust Region, TPE, SMBO など）の
> 典型的な組み合わせとは構造的に異なるように設計した「新規提案アルゴリズム」です。  
> ただし、理論的に「世界で誰も提案していない」ことを完全に保証することはできません。

---

## 1. 問題設定

- 目的関数：  
  - \( f: \mathcal{X} \to \mathbb{R} \) は**ブラックボックス関数**  
  - 関数値 \(f(x)\) の計算が高コスト
  - 勾配情報は利用できない
  - ノイズあり評価（\(y = f(x) + \varepsilon\)）も許容する

- 探索空間：  
  - \(\mathcal{X} = [a_1, b_1] \times \cdots \times [a_d, b_d]\)
  - 連続値のボックス制約領域を想定

- 制約：  
  - 評価回数 \(N_\text{eval}\) は小さい（例：数十〜数百回）
  - 高価なシミュレーション・実験などに適用可能なことを目指す

---

## 2. アルゴリズムの基本アイデア

LASTO は次の 3 つを組み合わせたブラックボックス最適化アルゴリズムです。

1. **複数の信頼領域 (Trust Region; TR) の並列管理**
   - 各 TR は  
     - 中心（有望そうな点）  
     - 半径（局所探索のスケール）  
     - 局所サロゲートモデル  
     を持つ「局所探索ユニット」として扱う。

2. **TR を「マルチアームバンディットの腕」として扱う**
   - 「どの TR に次の評価予算を割くか」を、  
     最近の改善度に基づく UCB 風の指標で決定する。
   - 改善の見込みがない TR には自然と予算が割かれなくなり、
     有望な TR にリソースが集中する。

3. **各 TR 内では「ランダム特徴サロゲート＋信頼領域ステップ」**
   - 入力空間をランダム特徴マップ \(\phi(x) \in \mathbb{R}^m\) により埋め込み、  
     線形／二次形式の軽量サロゲートを構築。
   - サロゲート上で安価な局所最適化を何度も行い、
     「次に高価評価すべき候補点」を決める。

これにより、

- 多峰性に強く（複数の局所解候補を並列探索）
- 高価な評価関数に対してもサロゲートを活用して効率的に探索
- ベイズ最適化より軽量で、高次元にもある程度スケール

といった特徴を持ちます。

---

## 3. データ構造

### 3.1 Trust Region \(T_k\)

各 TR \(T_k\) は次の情報を持ちます。

- 中心：\(c_k \in \mathcal{X}\)
- 半径：\(r_k > 0\)
- ローカルデータ：  
  \[
  D_k = \{(x_i, y_i)\}_{i=1}^{n_k},\quad x_i \in B(c_k, r_k) \cap \mathcal{X}
  \]
- サロゲートモデル：
  - ランダム特徴マップ \(\phi_k: \mathcal{X} \to \mathbb{R}^m\)
  - 線形／二次回帰モデル：
    \[
    \hat f_k(x) = w_k^\top \phi_k(x)
    \]
- バンディット用評価指標：
  - 累積改善量 \(G_k\)
  - 評価回数 \(N_k\)
  - 最近の改善度の指数移動平均（EMA） \(R_k\)

### 3.2 グローバル情報

- 全評価履歴：  
  \[
  D_\text{global} = \bigcup_k D_k
  \]
- グローバル最良値：  
  \[
  (x_\text{best}, y_\text{best}) = \arg\min_{(x,y)\in D_\text{global}} y
  \]
- 総評価回数：\(N_\text{eval}\)

---

## 4. アルゴリズムの 1 ステップの流れ

1. **TR 選択（マルチアームバンディット）**  
2. **選択 TR 内での候補点生成（サロゲート最適化）**  
3. **ブラックボックス評価**  
4. **TR の更新（サロゲート・半径・スコア）**  
5. **TR の誕生・死亡の管理（新規 TR 生成・不要 TR 削除）**  

以下で詳細を述べます。

---

## 5. TR 選択（マルチアームバンディット）

各 TR について次のような効率指標 \(\text{eff}_k\) を定義します：

\[
\text{eff}_k = R_k + \alpha \sqrt{\frac{\log(1 + N_\text{eval})}{N_k + 1}}
\]

- \(R_k\)：TR \(k\) の最近の改善度の EMA  
- 第2項：UCB (Upper Confidence Bound) 型の探索ボーナス  
- \(\alpha > 0\)：探索と利用のバランスを調整するハイパーパラメータ

この指標を最大にする TR を選択：

\[
k^* = \arg\max_k \text{eff}_k
\]

これにより、

- 改善の多い TR には利用（exploitation）として多くの評価が割かれ
- まだ評価回数の少ない TR も探索（exploration）のために一定確率で選ばれる

という典型的なバンディット戦略が実現されます。

---

## 6. TR 内での候補点生成

選択された TR \(T_{k^*}\) のサロゲート \(\hat f_{k^*}\) を用いて、
次を最小化する候補点 \(x^\text{cand}\) を求めます：

\[
x^\text{cand} = \arg\min_{x \in B(c_{k^*}, r_{k^*}) \cap \mathcal{X}} 
\Big( \hat f_{k^*}(x) - \beta \, \text{Unc}_{k^*}(x) \Big)
\]

- \(\text{Unc}_k(x)\)：サロゲートの不確実性指標
  - 例：
    - ランダムサブセット学習モデルのアンサンブル分散
    - ランダム特徴の dropout による予測分散
- \(\beta > 0\)：探索（不確実性）をどれだけ重視するかを決める係数

この内部最適化は、サロゲートモデル上での計算なので、
LBFGS や Nelder–Mead などの通常の最適化手法を自由に使えます。  
（ここは「ブラックボックス」ではないため何度も計算してもよい）

---

## 7. ブラックボックス評価と TR 更新

### 7.1 評価

選ばれた候補点をブラックボックス関数に投入：

\[
y^\text{cand} = f(x^\text{cand})
\]

- グローバル履歴 \(D_\text{global}\) と \(T_{k^*}\) のローカルデータ \(D_{k^*}\) に追加します。

### 7.2 サロゲートの再学習

\[
D_{k^*} \leftarrow D_{k^*} \cup \{(x^\text{cand}, y^\text{cand})\}
\]

このローカルデータ上で \(\hat f_{k^*}\) を再フィットします。  
（Ridge 回帰などで十分）

### 7.3 改善度指標の更新

- グローバル最良値は

\[
(x_\text{best}, y_\text{best}) = \arg\min_{(x,y)\in D_\text{global}} y
\]

- 改善量 \(\Delta\) を

\[
\Delta = y_\text{best}^{(\text{old})} - y_\text{best}^{(\text{new})}
\]

のように定義し、TR ごとの EMA を更新：

\[
R_{k^*} \leftarrow (1 - \lambda) R_{k^*} + \lambda \Delta
\]

- \(\lambda\)：EMA の更新係数（0.05〜0.2 程度）
- \(N_{k^*} \leftarrow N_{k^*} + 1\)

### 7.4 信頼領域半径の更新（Trust Region 風）

TR の中心 \(c_{k^*}\) から見た改善と、
サロゲート上で予測される改善の比 \(\rho\) を用いて半径を更新します。

- 実際の改善：  
  \[
  \text{act\_imp} = f(c_{k^*}) - y^\text{cand}
  \]
- 予測改善：  
  \[
  \text{pred\_imp} = \hat f_{k^*}(c_{k^*}) - \hat f_{k^*}(x^\text{cand})
  \]
- 比率：  
  \[
  \rho = \frac{\text{act\_imp}}{\text{pred\_imp} + \varepsilon}
  \]

ここで \(\varepsilon\) はゼロ割り防止の小さな定数。

半径更新ルールの例：

- \(\rho > \eta_\text{good}\)（予測どおり、あるいは予測以上に改善）  
  \[
  r_{k^*} \gets \min(\gamma_\text{inc} \, r_{k^*}, r_\text{max})
  \]
- \(\rho < \eta_\text{bad}\)（予測に反してほとんど改善していない）  
  \[
  r_{k^*} \gets \gamma_\text{dec} \, r_{k^*}
  \]

- パラメータ例：
  - \(\eta_\text{good} = 0.8\)
  - \(\eta_\text{bad}  = 0.2\)
  - \(\gamma_\text{inc} = 1.5\)
  - \(\gamma_\text{dec} = 0.5\)

半径が最小値 \(r_\text{min}\) 以下になった場合、
TR を「死亡」扱いとして削除します。

---

## 8. TR の誕生・死亡管理

### 8.1 TR の死亡

- 前節のように、半径 \(r_k < r_\text{min}\) になった TR は削除。
- あるいは
  - EMA 改善度 \(R_k\) が一定回数以上負（あるいはほぼゼロ）
  - かつ半径が小さい
- といった条件でも死亡扱いにしてよい。

### 8.2 新規 TR の生成（Exploration）

一定ステップごと、または TR が死亡するたびに、
新しい TR を生成します。

1. 既評価点集合 \(D_\text{global}\) からサンプル密度を推定
   - 例：単純な k-NN 距離、あるいは KDE による推定
2. 密度が低い（＝あまり探索していない）領域を優先しつつ、
   目的値が良い方向にも重みづけ
3. そこで新しい中心 \(c_\text{new}\) をサンプリング
4. 初期半径 \(r_\text{init}\) を与え、TR を生成
   - 近傍の既評価点を数点コピーして初期サロゲートを構築してもよい

これにより、局所探索に偏らず、
まだ未開拓だが有望な領域に探索が広がります。

---

## 9. 擬似コード

以下はシンプルな擬似コード例です（Python 風）。

```python
initialize K trust regions {T_k} with centers sampled in X
for each T_k:
    D_k = evaluate_few_points_near_center(f, center_k)
    fit_surrogate(k, D_k)
    R_k = 0.0
    N_k = len(D_k)

D_global = union_all(D_k)
x_best, y_best = argmin_over(D_global)
N_eval = total_number_of_evals_so_far

while N_eval < N_eval_max:

    # --- 1. TR selection (bandit) ---
    eff = []
    for k in alive_TRs:
        ucb = alpha * sqrt(log(1 + N_eval) / (N_k[k] + 1))
        eff_k = R_k[k] + ucb
        eff.append(eff_k)
    k_star = argmax(eff)

    # --- 2. Candidate point within TR k_star ---
    def acquisition(x):
        mu = surrogate_predict(k_star, x)
        unc = surrogate_uncertainty(k_star, x)
        return mu - beta * unc   # minimization

    x_cand = argmin_acquisition_in_ball(
        acquisition,
        center=c_k[k_star],
        radius=r_k[k_star],
        bounds=X_box
    )

    # --- 3. Evaluate black-box ---
    y_cand = f(x_cand)
    N_eval += 1

    # --- 4. Update TR k_star ---
    D_k[k_star].append((x_cand, y_cand))
    D_global.append((x_cand, y_cand))
    refit_surrogate(k_star, D_k[k_star])

    # global best update
    if y_cand < y_best:
        x_best, y_best = x_cand, y_cand

    # improvement statistics
    delta = previous_y_best - y_best   # or other definition
    R_k[k_star] = (1 - lambda_) * R_k[k_star] + lambda_ * delta
    N_k[k_star] += 1

    # trust-region radius update (using ratio rho)
    act_imp  = f(c_k[k_star]) - y_cand
    pred_imp = surrogate_predict(k_star, c_k[k_star]) - \
               surrogate_predict(k_star, x_cand)
    rho = act_imp / (pred_imp + eps)

    if rho > eta_good:
        r_k[k_star] = min(gamma_inc * r_k[k_star], r_max)
    elif rho < eta_bad:
        r_k[k_star] = gamma_dec * r_k[k_star]

    # kill TR if radius too small
    if r_k[k_star] < r_min:
        kill_TR(k_star)

    # --- 5. Spawn new TRs occasionally ---
    if need_new_TR(alive_TRs):
        c_new = sample_new_center_from_density_and_values(D_global)
        create_TR_with_initial_radius(c_new, r_init)
```

---

## 10. ハイパーパラメータの例

- ランダム特徴次元 \(m\)：50〜200 程度
- 初期 TR 数 \(K\)：3〜10
- 信頼領域関連：
  - \(r_\text{init} \approx 0.3 \times (\text{各次元の幅の平均})\)
  - \(\gamma_\text{inc} = 1.5\)
  - \(\gamma_\text{dec} = 0.5\)
  - \(\eta_\text{good} = 0.8\)
  - \(\eta_\text{bad}  = 0.2\)
  - \(r_\text{min} = 0.01 \times (\text{各次元の幅の平均})\)
- バンディット：
  - \(\alpha \in [0.1, 2.0]\)
  - EMA 係数 \(\lambda \approx 0.1\)
- 評価予算が少ない場合（例：50 回以下）：
  - TR 数 \(K\) を 1〜3 に減らす
  - サロゲートをより単純（線形＋L2 正則化のみ）にする

---

## 11. 実用性と特徴のまとめ

1. **多峰性への耐性**
   - 複数の TR が異なる局所解候補を同時に追うことで、
     単一のローカル探索手法よりも局所解にハマりにくい。

2. **高価評価への適合**
   - サロゲートはランダム特徴＋線形回帰で軽量。  
   - サロゲート上での最適化は評価コストがほぼゼロのため、
     評価回数に制限がある状況でも効率的。

3. **ベイズ最適化より計算コストが軽い**
   - ガウス過程のような \(\mathcal{O}(n^3)\) スケーリングを避け、
     高次元・多数データにも相対的にスケーラブル。

4. **実装のしやすさ**
   - 必要な要素は
     - ランダム特徴マップ
     - 線形回帰（Ridge 回帰）
     - ローカルな連続最適化アルゴリズム
     であり、いずれも一般的なライブラリで容易に利用可能。

---

## 12. 実装を始めるためのステップガイド

1. **ステップ 1：単一 TR の実装**
   - TR を 1 つに固定し、
     - ランダム特徴サロゲート
     - 信頼領域半径の更新
     を組み合わせた局所探索器として実装。

2. **ステップ 2：複数 TR 化**
   - TR を 3〜5 個に増やし、
     - 改善度 EMA \(R_k\)
     - 評価回数 \(N_k\)
     に基づく UCB 風の TR 選択ロジックを追加。

3. **ステップ 3：新規 TR の生成**
   - 単純なルールから開始：  
     - 既評価点から一定距離以上離れた一様サンプルを中心にする  
     - もしくは目的値が良い点の近傍に新 TR を追加

4. **ステップ 4：高度化**
   - サンプル密度推定に KDE を用いる
   - 不確実性推定にアンサンブルや dropout を使用
   - 制約付き最適化（ペナルティ付加）などへ拡張

---

## 13. まとめ

この Markdown では、

- ブラックボックス最適化向けの新アルゴリズム  
  **LASTO: Layered Adaptive Surrogate Trust Optimization**
- その基本アイデア・数理的構造・擬似コード
- 実装およびハイパーパラメータの指針

を一通り整理しました。

このまま論文草稿や実装仕様書のたたき台として利用したり、
Python 実装を起こす際の設計書としても使えるような構成になっています。
