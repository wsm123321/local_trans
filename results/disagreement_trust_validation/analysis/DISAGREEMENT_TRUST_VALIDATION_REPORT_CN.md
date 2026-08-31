# 冲突条件信任最小验证结论

- Stage: `disagreement-conditioned-trust-pilot-v1-r-full`
- 预设判定：**do_not_advance_complex_disagreement_trust**
- Holdout：864 个事件，435 个有效源—目标冲突，119 个真实正收益事件；共同 eligible 支持集 410 个事件。

## 一、五种方法的候选结果

| 方法 | normalized regret↓ | Top-10% 命中↑ | actionable coverage | 接受后负迁移率 | 接受数 |
|---|---:|---:|---:|---:|---:|
| Target-Only | 0.1142 | 0.8495 | 0.0000 | NA | 0 |
| Local Spearman Gate | 0.0677 | 0.8912 | 0.5448 | 0.5570 | 237 |
| Target-Residual Spearman Gate | 0.0852 | 0.8692 | 0.5241 | 0.5921 | 228 |
| Disagreement-Correction Gate | 0.0673 | 0.8912 | 0.5218 | 0.5286 | 227 |
| Oracle Gate | 0.0475 | 0.9201 | 0.2736 | 0.0000 | 119 |

## 二、能否预测下一次源建议有益

| Gate | 共同支持 AUPRC↑ | 共同支持正例率 | 单门 eligible coverage | 全 actionable neutral-imputed AUPRC |
|---|---:|---:|---:|---:|
| Local Spearman Gate | 0.3882 | 0.2732 | 1.0000 | 0.3851 |
| Target-Residual Spearman Gate | 0.3755 | 0.2732 | 1.0000 | 0.3725 |
| Disagreement-Correction Gate | 0.3956 | 0.2732 | 0.9425 | 0.3954 |

## 三、核心配对判断

- 冲突 Gate 相对 Local Spearman 的 normalized-regret 优势：`+0.0003`，seed-cluster bootstrap 95% CI `[-0.0086, +0.0091]`。
- 冲突 Gate 相对 Local Spearman 的 AUPRC 优势：`+0.0074`，95% CI `[-0.0442, +0.0500]`。
- 冲突 Gate 相对 Target-Only 的 normalized-regret 优势：`+0.0469`，95% CI `[+0.0272, +0.0691]`。
- Oracle 可达 regret headroom：`+0.0667`，95% CI `[+0.0474, +0.0889]`。

## 四、预设推进检查

- PASS — `sufficient_effective_events`
- PASS — `auprc_bootstrap_defined`
- FAIL — `correction_auprc_superior`
- FAIL — `correction_regret_superior_by_margin`
- PASS — `coverage_comparable`
- PASS — `top10_noninferior`
- PASS — `accepted_harm_bootstrap_defined`
- PASS — `accepted_harm_noninferior`
- PASS — `correction_improves_target_only`
- PASS — `oracle_has_nontrivial_headroom`

## 五、阈值与公平性边界

Development 阈值只根据连续分数、eligible 状态和 actionable coverage 冻结，未使用最终候选面板标签：

- Local Spearman Gate: threshold `0.648951`；Development actionable coverage `0.500`。
- Target-Residual Spearman Gate: threshold `0.583916`；Development actionable coverage `0.493`。
- Disagreement-Correction Gate: threshold `0.623047`；Development actionable coverage `0.401`。

所有方法共享 source-blind 付费成对诊断反馈、目标历史、Target-Only GP、候选池、源专家、Top-K 适用集合、源提名和真实评价面板；三个可部署 gate 只能接受同一个 `x_S` 或精确回退同一个 `x_T`。主 AUPRC 只在三个 gate 共同 eligible 的支持集上计算；Oracle 只在真值揭示后计算，不进入可部署 AUPRC。

## 六、结论边界

本实验只判断是否值得继续设计这层信任机制。预测比较严格条件化在三个冻结估计器共同 eligible 的最终事件支持集上，并依赖额外付费的 source-blind 成对诊断反馈；它不证明自然 单点反馈下可学，也不证明实际区域检索、未知对齐、多源调度、闭环 BO 预算收益、高维泛化 或普遍无负迁移。
