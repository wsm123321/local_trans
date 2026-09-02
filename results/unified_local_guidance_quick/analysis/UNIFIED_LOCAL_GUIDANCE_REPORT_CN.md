# 统一 local-guidance 统计分析报告

统计单位为独立 `(problem, dim, seed)` instance。候选点不是 replicate；sequential step 只在 instance 内汇总为 final 和 AUC。正 effect 统一表示方法 A（新方法）更好。

## 五项批准的 primary contrasts

| Hypothesis | Dataset | n | Effect [95% CI] | Pratt one-sided p | Holm p | Rank-biserial | W/T/L |
|---|---|---:|---:|---:|---:|---:|---:|
| H1_mechanism_normalized_regret_LocalReliability_vs_Geometry | mechanism | 4 | +0.00000 [+0.00000, +0.00000] | 1 | 1 | +0.000 | 0/4/0 |
| H2_mechanism_top10_hit_LocalReliability_vs_Geometry | mechanism | 4 | +0.00000 [+0.00000, +0.00000] | 1 | 1 | +0.000 | 0/4/0 |
| H3_sequential_final_normalized_regret_LocalReliability_vs_Geometry | sequential | 4 | +0.02525 [+0.00000, +0.07576] | 0.5 | 1 | +1.000 | 1/3/0 |
| H4_sequential_regret_auc_LocalReliability_vs_Geometry | sequential | 4 | +0.02781 [+0.00000, +0.08342] | 0.5 | 1 | +1.000 | 1/3/0 |
| H5_mechanism_reliability_increment_LocalReliability_vs_LocalNoReliability | mechanism | 4 | +0.00000 [+0.00000, +0.00000] | 1 | 1 | +0.000 | 0/4/0 |

## 方法汇总

| Dataset | Method | Instances | Means |
|---|---|---:|---:|
| mechanism | Target-Only | 4 | normalized_regret=0.721848; top10_hit=0 |
| mechanism | Geometry-Only | 4 | normalized_regret=0.459526; top10_hit=0.5 |
| mechanism | Local-Rank-No-Reliability | 4 | normalized_regret=0.459526; top10_hit=0.5 |
| mechanism | Local-Rank+Reliability | 4 | normalized_regret=0.459526; top10_hit=0.5 |
| mechanism | Reversed-Local-Rank | 4 | normalized_regret=0.501581; top10_hit=0.25 |
| sequential | Target-Only | 4 | final_normalized_regret=0.628328; auc_normalized_regret=4.42038; total_improvement=1.97816 |
| sequential | Geometry-Only | 4 | final_normalized_regret=0.540866; auc_normalized_regret=3.32388; total_improvement=1.98252 |
| sequential | Local-Rank-No-Reliability | 4 | final_normalized_regret=0.515613; auc_normalized_regret=3.29607; total_improvement=2.17556 |
| sequential | Local-Rank+Reliability | 4 | final_normalized_regret=0.515613; auc_normalized_regret=3.29607; total_improvement=2.17556 |

## 次要对比

次要对比仅用于描述，不替代五项 primary contrasts。

| Dataset | A | B | Metric | n | Effect [95% CI] | p |
|---|---|---|---|---:|---:|---:|
| mechanism | Local-Rank-No-Reliability | Geometry-Only | normalized_regret | 4 | +0.00000 [+0.00000, +0.00000] | 1 |
| mechanism | Local-Rank+Reliability | Geometry-Only | normalized_regret | 4 | +0.00000 [+0.00000, +0.00000] | 1 |
| mechanism | Reversed-Local-Rank | Geometry-Only | normalized_regret | 4 | -0.04206 [-0.12617, +0.00000] | 1 |
| mechanism | Local-Rank-No-Reliability | Geometry-Only | top10_hit | 4 | +0.00000 [+0.00000, +0.00000] | 1 |
| mechanism | Local-Rank+Reliability | Geometry-Only | top10_hit | 4 | +0.00000 [+0.00000, +0.00000] | 1 |
| mechanism | Reversed-Local-Rank | Geometry-Only | top10_hit | 4 | -0.25000 [-0.75000, +0.00000] | 1 |
| sequential | Local-Rank-No-Reliability | Geometry-Only | final_normalized_regret | 4 | +0.02525 [+0.00000, +0.07576] | 0.5 |
| sequential | Local-Rank+Reliability | Geometry-Only | final_normalized_regret | 4 | +0.02525 [+0.00000, +0.07576] | 0.5 |
| sequential | Local-Rank-No-Reliability | Geometry-Only | auc_normalized_regret | 4 | +0.02781 [+0.00000, +0.08342] | 0.5 |
| sequential | Local-Rank+Reliability | Geometry-Only | auc_normalized_regret | 4 | +0.02781 [+0.00000, +0.08342] | 0.5 |

Bootstrap replicates: 500 (from config; full protocol uses 5000).
