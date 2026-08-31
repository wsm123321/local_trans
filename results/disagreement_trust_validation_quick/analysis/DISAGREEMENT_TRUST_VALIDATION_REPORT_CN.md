# 冲突条件信任最小验证结论

- Stage: `disagreement-conditioned-trust-pilot-v1-r-quick`
- 预设判定：**do_not_advance_complex_disagreement_trust**
- Holdout：108 个事件，65 个有效源—目标冲突，34 个真实正收益事件；共同 eligible 支持集 49 个事件。

## 一、五种方法的候选结果

| 方法 | normalized regret↓ | Top-10% 命中↑ | actionable coverage | 接受后负迁移率 | 接受数 |
|---|---:|---:|---:|---:|---:|
| Target-Only | 0.2566 | 0.5370 | 0.0000 | NA | 0 |
| Local Spearman Gate | 0.1217 | 0.7130 | 0.6154 | 0.2750 | 40 |
| Target-Residual Spearman Gate | 0.1212 | 0.7222 | 0.5692 | 0.2432 | 37 |
| Disagreement-Correction Gate | 0.1503 | 0.6667 | 0.4154 | 0.3333 | 27 |
| Oracle Gate | 0.1032 | 0.7500 | 0.5231 | 0.0000 | 34 |

## 二、能否预测下一次源建议有益

| Gate | 共同支持 AUPRC↑ | 共同支持正例率 | 单门 eligible coverage | 全 actionable neutral-imputed AUPRC |
|---|---:|---:|---:|---:|
| Local Spearman Gate | 0.6140 | 0.4490 | 1.0000 | 0.6597 |
| Target-Residual Spearman Gate | 0.6808 | 0.4490 | 1.0000 | 0.7643 |
| Disagreement-Correction Gate | 0.6765 | 0.4490 | 0.7538 | 0.6841 |

## 三、核心配对判断

- 冲突 Gate 相对 Local Spearman 的 normalized-regret 优势：`-0.0286`，seed-cluster bootstrap 95% CI `[-0.0529, -0.0053]`。
- 冲突 Gate 相对 Local Spearman 的 AUPRC 优势：`+0.0625`，95% CI `[-0.1268, +0.1635]`。
- 冲突 Gate 相对 Target-Only 的 normalized-regret 优势：`+0.1063`，95% CI `[+0.0571, +0.1632]`。
- Oracle 可达 regret headroom：`+0.1534`，95% CI `[+0.1085, +0.1994]`。

## 四、预设推进检查

- PASS — `sufficient_effective_events`
- PASS — `auprc_bootstrap_defined`
- FAIL — `correction_auprc_superior`
- FAIL — `correction_regret_superior_by_margin`
- FAIL — `coverage_comparable`
- FAIL — `top10_noninferior`
- PASS — `accepted_harm_bootstrap_defined`
- FAIL — `accepted_harm_noninferior`
- PASS — `correction_improves_target_only`
- PASS — `oracle_has_nontrivial_headroom`

## 五、阈值与公平性边界

Development 阈值只根据连续分数、eligible 状态和 actionable coverage 冻结，未使用最终候选面板标签：

- Local Spearman Gate: threshold `0.684524`；Development actionable coverage `0.484`。
- Target-Residual Spearman Gate: threshold `0.642857`；Development actionable coverage `0.516`。
- Disagreement-Correction Gate: threshold `0.656250`；Development actionable coverage `0.323`。

所有方法共享 source-blind 付费成对诊断反馈、目标历史、Target-Only GP、候选池、源专家、Top-K 适用集合、源提名和真实评价面板；三个可部署 gate 只能接受同一个 `x_S` 或精确回退同一个 `x_T`。主 AUPRC 只在三个 gate 共同 eligible 的支持集上计算；Oracle 只在真值揭示后计算，不进入可部署 AUPRC。

## 六、结论边界

本实验只判断是否值得继续设计这层信任机制。预测比较严格条件化在三个冻结估计器共同 eligible 的最终事件支持集上，并依赖额外付费的 source-blind 成对诊断反馈；它不证明自然 单点反馈下可学，也不证明实际区域检索、未知对齐、多源调度、闭环 BO 预算收益、高维泛化 或普遍无负迁移。
