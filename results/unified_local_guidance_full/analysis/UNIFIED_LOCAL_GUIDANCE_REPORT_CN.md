# 统一 local-guidance 统计分析报告

统计单位为独立 `(problem, dim, seed)` instance。候选点不是 replicate；sequential step 只在 instance 内汇总为 final 和 AUC。正 effect 统一表示方法 A（新方法）更好。

## 五项批准的 primary contrasts

| Hypothesis | Dataset | n | Effect [95% CI] | Pratt one-sided p | Holm p | Rank-biserial | W/T/L |
|---|---|---:|---:|---:|---:|---:|---:|
| H1_mechanism_normalized_regret_LocalReliability_vs_Geometry | mechanism | 64 | -0.00630 [-0.03279, +0.02071] | 0.4318 | 1 | +0.077 | 7/51/6 |
| H2_mechanism_top10_hit_LocalReliability_vs_Geometry | mechanism | 64 | +0.03125 [-0.03125, +0.09375] | 0.1587 | 0.7933 | +0.500 | 3/60/1 |
| H3_sequential_final_normalized_regret_LocalReliability_vs_Geometry | sequential | 64 | -0.03466 [-0.07497, +0.00713] | 0.9213 | 1 | -0.163 | 18/21/25 |
| H4_sequential_regret_auc_LocalReliability_vs_Geometry | sequential | 64 | -0.14844 [-0.78745, +0.44526] | 0.7047 | 1 | -0.087 | 21/18/25 |
| H5_mechanism_reliability_increment_LocalReliability_vs_LocalNoReliability | mechanism | 64 | +0.00000 [+0.00000, +0.00000] | 1 | 1 | +0.000 | 0/64/0 |

## 方法汇总

| Dataset | Method | Instances | Means |
|---|---|---:|---:|
| mechanism | Target-Only | 64 | normalized_regret=0.637826; top10_hit=0.078125 |
| mechanism | Geometry-Only | 64 | normalized_regret=0.445772; top10_hit=0.296875 |
| mechanism | Local-Rank-No-Reliability | 64 | normalized_regret=0.452071; top10_hit=0.328125 |
| mechanism | Local-Rank+Reliability | 64 | normalized_regret=0.452071; top10_hit=0.328125 |
| mechanism | Reversed-Local-Rank | 64 | normalized_regret=0.532372; top10_hit=0.15625 |
| sequential | Target-Only | 64 | final_normalized_regret=0.473845; auc_normalized_regret=14.0164; total_improvement=7.49218 |
| sequential | Geometry-Only | 64 | final_normalized_regret=0.404869; auc_normalized_regret=12.0032; total_improvement=8.69495 |
| sequential | Local-Rank-No-Reliability | 64 | final_normalized_regret=0.439525; auc_normalized_regret=12.1517; total_improvement=7.981 |
| sequential | Local-Rank+Reliability | 64 | final_normalized_regret=0.439525; auc_normalized_regret=12.1517; total_improvement=7.981 |

## 次要对比

次要对比仅用于描述，不替代五项 primary contrasts。

| Dataset | A | B | Metric | n | Effect [95% CI] | p |
|---|---|---|---|---:|---:|---:|
| mechanism | Local-Rank-No-Reliability | Geometry-Only | normalized_regret | 64 | -0.00630 [-0.03345, +0.02079] | 0.4318 |
| mechanism | Local-Rank+Reliability | Geometry-Only | normalized_regret | 64 | -0.00630 [-0.03360, +0.02018] | 0.4318 |
| mechanism | Reversed-Local-Rank | Geometry-Only | normalized_regret | 64 | -0.08660 [-0.15295, -0.02745] | 0.9913 |
| mechanism | Local-Rank-No-Reliability | Geometry-Only | top10_hit | 64 | +0.03125 [-0.03125, +0.09375] | 0.1587 |
| mechanism | Local-Rank+Reliability | Geometry-Only | top10_hit | 64 | +0.03125 [-0.03125, +0.09375] | 0.1587 |
| mechanism | Reversed-Local-Rank | Geometry-Only | top10_hit | 64 | -0.14062 [-0.23438, -0.06250] | 0.9987 |
| sequential | Local-Rank-No-Reliability | Geometry-Only | final_normalized_regret | 64 | -0.03466 [-0.07514, +0.00721] | 0.9213 |
| sequential | Local-Rank+Reliability | Geometry-Only | final_normalized_regret | 64 | -0.03466 [-0.07628, +0.00629] | 0.9213 |
| sequential | Local-Rank-No-Reliability | Geometry-Only | auc_normalized_regret | 64 | -0.14844 [-0.75859, +0.45593] | 0.7047 |
| sequential | Local-Rank+Reliability | Geometry-Only | auc_normalized_regret | 64 | -0.14844 [-0.77251, +0.43545] | 0.7047 |

Bootstrap replicates: 5000 (from config; full protocol uses 5000).
