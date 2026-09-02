# Gate-0 Oracle Local-Model Transfer Quick 结论

**结论标签：`promising_value_or_dual_head_transfer`**

这是 quick、非正式、描述性分析，不是确认性统计结论，也不涉及未知 alignment、候选获取或 online BO。Bootstrap 只按 seed 重采样，candidate 不是独立重复。

## 明确的 Gate-0 规则

- decision input keys：`dimension × panel × seed × relation_or_control × shell × method`；frozen dimensions=`[2]`，panels=`['test']`，shells=`[0.35, 0.7, 1.0]`。缺失 key/panel 或 frozen set 不匹配时 decision failure。
- canonical methods：`target_only, geometry, rank, value, dual`。
- positive：`identity, output_affine, scale_0.7, scale_1.5`。
- negative：`reversal, independent_expert, identity_label_permutation`；boundary：`rotate_45, roughness`。
- decision metrics 仅为 `pairwise_accuracy, ndcg_at_top, normalized_top1_regret`；正 delta 表示 challenger 更好。
- for one head, at least 2/3 positive shells pass at least 2/3 metrics; each metric requires seed-level win rate >= 0.5 against minimum_delta=0 and seed mean delta > minimum_delta; at least 2 metrics pass overall。
- challenger has Rank-beyond increment iff at least 2/3 decision-metric seed means > minimum_delta=0 with seed win rate >= 0.5 AND at least 2/3 shells have at least 2/3 such metrics。
- after shell rows are averaged within seed, identity observed−permuted must have mean > minimum_delta=0 and seed win rate >= 0.5 for at least 2/3 metrics; report descriptive seed bootstrap CI。
- for each negative condition, average shell rows within seed; a metric replicates only when mean > minimum_delta=0 and seed win rate >= 0.5; a condition replicates at least 2/3 metrics, and the overall negative rule fails when at least 2/3 (2 of 3) conditions replicate。
- benefit cannot be confined to shell .35; at least 2 of 3 shells must pass。
- sRMSE 是 secondary，绝不驱动 decision；所有量纲不被混合或取 max。

## 三个 head 相对 Geometry 的 positive 结果

| head | metric | seed mean | 95% seed CI | seed win rate | n_seed_units | n_pair_units | pass |
|---|---|---:|---|---:|---:|---:|---|
| rank | pairwise_accuracy | +0.02223 | [+0.00104, +0.04620] | +0.62500 | 8 | 96 | yes |
| rank | ndcg_at_top | +0.05341 | [+0.02209, +0.08530] | +0.87500 | 8 | 96 | yes |
| rank | normalized_top1_regret | +0.05945 | [+0.01679, +0.11105] | +0.75000 | 8 | 96 | yes |
| value | pairwise_accuracy | +0.10570 | [+0.08082, +0.13342] | +1.00000 | 8 | 96 | yes |
| value | ndcg_at_top | +0.08144 | [+0.05754, +0.10935] | +1.00000 | 8 | 96 | yes |
| value | normalized_top1_regret | +0.07949 | [+0.04364, +0.12228] | +1.00000 | 8 | 96 | yes |
| dual | pairwise_accuracy | +0.07586 | [+0.05399, +0.10022] | +1.00000 | 8 | 96 | yes |
| dual | ndcg_at_top | +0.07380 | [+0.04942, +0.09998] | +1.00000 | 8 | 96 | yes |
| dual | normalized_top1_regret | +0.07769 | [+0.04106, +0.12097] | +1.00000 | 8 | 96 | yes |

## Value/Dual 相对 Rank 的增量（只用原始 results 配对）

| challenger | metric | seed mean | 95% seed CI | seed win rate | n_seed_units | n_pair_units | pair W/T/L | pass |
|---|---|---:|---|---:|---:|---:|---|---|
| value vs rank | pairwise_accuracy | +0.08347 | [+0.07509, +0.09719] | +1.00000 | 8 | 96 | 94/0/2 | yes |
| value vs rank | ndcg_at_top | +0.02804 | [+0.01803, +0.03859] | +1.00000 | 8 | 96 | 85/0/11 | yes |
| value vs rank | normalized_top1_regret | +0.02004 | [+0.00659, +0.03676] | +1.00000 | 8 | 96 | 51/32/13 | yes |
| value | rule | 3/3 metrics; 3/3 shells | — | — | 8 | 96 | — | yes |
| dual vs rank | pairwise_accuracy | +0.05363 | [+0.04886, +0.05965] | +1.00000 | 8 | 96 | 94/0/2 | yes |
| dual vs rank | ndcg_at_top | +0.02039 | [+0.01132, +0.02976] | +1.00000 | 8 | 96 | 89/1/6 | yes |
| dual vs rank | normalized_top1_regret | +0.01824 | [+0.00553, +0.03434] | +0.87500 | 8 | 96 | 30/61/5 | yes |
| dual | rule | 3/3 metrics; 3/3 shells | — | — | 8 | 96 | — | yes |

## Negative 与 permutation 结论

- **rank** negative conditions：reversal (0/3 metrics; condition replicate=no): pairwise_accuracy: mean=+0.00000, seed_win=+0.00000, n_seed=8, n_pair=24, replicate=no, ndcg_at_top: mean=+0.00000, seed_win=+0.00000, n_seed=8, n_pair=24, replicate=no, normalized_top1_regret: mean=+0.00000, seed_win=+0.00000, n_seed=8, n_pair=24, replicate=no; independent_expert (0/3 metrics; condition replicate=no): pairwise_accuracy: mean=-0.03789, seed_win=+0.25000, n_seed=8, n_pair=24, replicate=no, ndcg_at_top: mean=-0.03185, seed_win=+0.37500, n_seed=8, n_pair=24, replicate=no, normalized_top1_regret: mean=-0.09605, seed_win=+0.25000, n_seed=8, n_pair=24, replicate=no; identity_label_permutation (0/3 metrics; condition replicate=no): pairwise_accuracy: mean=-0.04440, seed_win=+0.37500, n_seed=8, n_pair=24, replicate=no, ndcg_at_top: mean=-0.02501, seed_win=+0.50000, n_seed=8, n_pair=24, replicate=no, normalized_top1_regret: mean=-0.06832, seed_win=+0.37500, n_seed=8, n_pair=24, replicate=no。Permutation：pairwise_accuracy: observed−permuted=+0.06271, CI=[+0.02083, +0.12274], seed_win=+0.87500, n_seed=8, n_pair=24, pass=yes; ndcg_at_top: observed−permuted=+0.10473, CI=[+0.04345, +0.18983], seed_win=+1.00000, n_seed=8, n_pair=24, pass=yes; normalized_top1_regret: observed−permuted=+0.14386, CI=[+0.05525, +0.26025], seed_win=+0.75000, n_seed=8, n_pair=24, pass=yes；3/3 identity metrics pass。
- **value** negative conditions：reversal (0/3 metrics; condition replicate=no): pairwise_accuracy: mean=+0.00000, seed_win=+0.00000, n_seed=8, n_pair=24, replicate=no, ndcg_at_top: mean=+0.00000, seed_win=+0.00000, n_seed=8, n_pair=24, replicate=no, normalized_top1_regret: mean=+0.00000, seed_win=+0.00000, n_seed=8, n_pair=24, replicate=no; independent_expert (0/3 metrics; condition replicate=no): pairwise_accuracy: mean=-0.03960, seed_win=+0.25000, n_seed=8, n_pair=24, replicate=no, ndcg_at_top: mean=-0.02956, seed_win=+0.50000, n_seed=8, n_pair=24, replicate=no, normalized_top1_regret: mean=-0.09615, seed_win=+0.37500, n_seed=8, n_pair=24, replicate=no; identity_label_permutation (1/3 metrics; condition replicate=no): pairwise_accuracy: mean=-0.04002, seed_win=+0.37500, n_seed=8, n_pair=24, replicate=no, ndcg_at_top: mean=+0.00035, seed_win=+0.50000, n_seed=8, n_pair=24, replicate=yes, normalized_top1_regret: mean=-0.02887, seed_win=+0.37500, n_seed=8, n_pair=24, replicate=no。Permutation：pairwise_accuracy: observed−permuted=+0.15033, CI=[+0.11910, +0.19346], seed_win=+1.00000, n_seed=8, n_pair=24, pass=yes; ndcg_at_top: observed−permuted=+0.09593, CI=[+0.06282, +0.13128], seed_win=+1.00000, n_seed=8, n_pair=24, pass=yes; normalized_top1_regret: observed−permuted=+0.11049, CI=[+0.05372, +0.16826], seed_win=+0.87500, n_seed=8, n_pair=24, pass=yes；3/3 identity metrics pass。
- **dual** negative conditions：reversal (0/3 metrics; condition replicate=no): pairwise_accuracy: mean=+0.00000, seed_win=+0.00000, n_seed=8, n_pair=24, replicate=no, ndcg_at_top: mean=+0.00000, seed_win=+0.00000, n_seed=8, n_pair=24, replicate=no, normalized_top1_regret: mean=+0.00000, seed_win=+0.00000, n_seed=8, n_pair=24, replicate=no; independent_expert (0/3 metrics; condition replicate=no): pairwise_accuracy: mean=-0.04196, seed_win=+0.25000, n_seed=8, n_pair=24, replicate=no, ndcg_at_top: mean=-0.03107, seed_win=+0.37500, n_seed=8, n_pair=24, replicate=no, normalized_top1_regret: mean=-0.09768, seed_win=+0.25000, n_seed=8, n_pair=24, replicate=no; identity_label_permutation (0/3 metrics; condition replicate=no): pairwise_accuracy: mean=-0.05048, seed_win=+0.37500, n_seed=8, n_pair=24, replicate=no, ndcg_at_top: mean=-0.02403, seed_win=+0.50000, n_seed=8, n_pair=24, replicate=no, normalized_top1_regret: mean=-0.06078, seed_win=+0.37500, n_seed=8, n_pair=24, replicate=no。Permutation：pairwise_accuracy: observed−permuted=+0.12271, CI=[+0.08192, +0.17965], seed_win=+1.00000, n_seed=8, n_pair=24, pass=yes; ndcg_at_top: observed−permuted=+0.11432, CI=[+0.05128, +0.20358], seed_win=+1.00000, n_seed=8, n_pair=24, pass=yes; normalized_top1_regret: observed−permuted=+0.14072, CI=[+0.05150, +0.26224], seed_win=+0.87500, n_seed=8, n_pair=24, pass=yes；3/3 identity metrics pass。

Scale_1.5 是 source-GP 外推关系；其结果不能解释为源域观测范围内的直接证据。

结论严格按独立的 Geometry transfer 规则与 Rank-beyond 增量规则给出；不会因为三头 selection score 相同而默认选择 Rank。
