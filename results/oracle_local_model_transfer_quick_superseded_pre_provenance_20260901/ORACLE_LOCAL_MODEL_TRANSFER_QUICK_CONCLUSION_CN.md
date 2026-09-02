# Gate-0 Oracle Local-Model Transfer Quick 结论

**结论标签：`promising_rank_transfer`**

这是 quick、非正式、描述性分析，不是确认性统计结论，也不涉及未知 alignment、候选获取或 online BO。Bootstrap 只按 seed 重采样，candidate 不是独立重复。

## 明确的 Gate-0 规则

- canonical methods：`target_only, geometry, rank, value, dual`。
- positive：`identity, output_affine, scale_0.7, scale_1.5`。
- negative：`reversal, independent_expert, identity_label_permutation`；boundary：`rotate_45, roughness`。
- decision metrics 仅为 `pairwise_accuracy, ndcg_at_top, normalized_top1_regret`；仅相对 Geometry 配对，正 delta 表示 transfer 更好。
- for one head, at least 2/3 positive shells pass at least 2/3 metrics; each metric requires seed-level win rate >= 0.5 and mean delta > 0; at least 2 metrics pass overall。
- identity observed Geometry-relative delta exceeds identity_label_permutation on at least 2/3 decision metrics。
- negative conditions may not replicate positive decision deltas on at least 2/3 metrics; exact target-only fallback is acceptable。
- benefit cannot be confined to shell .35; at least 2 of 3 shells must pass。
- sRMSE 是 secondary，绝不驱动 decision；所有量纲不被混合或取 max。

summary rows：1080；contrast rows（含 vs Geometry 与 vs Target-Only）：12096。
若任一 head 未同时满足全部规则，保守输出 `no_oracle_headroom_stop_before_alignment`。
