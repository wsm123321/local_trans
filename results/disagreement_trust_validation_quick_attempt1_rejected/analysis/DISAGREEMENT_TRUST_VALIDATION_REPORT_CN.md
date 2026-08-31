# 冲突条件信任最小验证结论

- Stage: `disagreement-conditioned-trust-pilot-v1-quick`
- 预设判定：**do_not_advance_complex_disagreement_trust**
- Holdout：108 个事件，86 个有效源—目标冲突，57 个真实正收益事件。

## 一、五种方法的候选结果

| 方法 | normalized regret↓ | Top-10% 命中↑ | actionable coverage | 接受后负迁移率 | 接受数 |
|---|---:|---:|---:|---:|---:|
| Target-Only | 0.4482 | 0.4074 | 0.0000 | NA | 0 |
| Local Spearman Gate | 0.2006 | 0.6296 | 0.5465 | 0.1277 | 47 |
| Target-Residual Spearman Gate | 0.2223 | 0.6667 | 0.5581 | 0.0833 | 48 |
| Disagreement-Correction Gate | 0.2739 | 0.6204 | 0.4535 | 0.1538 | 39 |
| Oracle Gate | 0.1371 | 0.7037 | 0.6628 | 0.0000 | 57 |

## 二、能否预测下一次源建议有益

| Gate | AUPRC↑ | 正例率基线 | eligible coverage |
|---|---:|---:|---:|
| Local Spearman Gate | 0.9058 | 0.6628 | 1.0000 |
| Target-Residual Spearman Gate | 0.9344 | 0.6628 | 1.0000 |
| Disagreement-Correction Gate | 0.8690 | 0.6628 | 0.8953 |

## 三、核心配对判断

- 冲突 Gate 相对 Local Spearman 的 normalized-regret 优势：`-0.0733`，seed-cluster bootstrap 95% CI `[-0.1689, +0.0079]`。
- 冲突 Gate 相对 Local Spearman 的 AUPRC 优势：`-0.0368`，95% CI `[-0.0996, +0.0150]`。
- 冲突 Gate 相对 Target-Only 的 normalized-regret 优势：`+0.1743`，95% CI `[+0.0968, +0.2492]`。
- Oracle 可达 regret headroom：`+0.3110`，95% CI `[+0.2087, +0.4112]`。

## 四、预设推进检查

- PASS — `sufficient_effective_events`
- FAIL — `correction_auprc_superior`
- FAIL — `correction_regret_superior_by_margin`
- FAIL — `coverage_comparable`
- FAIL — `top10_noninferior`
- FAIL — `accepted_harm_noninferior`
- PASS — `correction_improves_target_only`
- PASS — `oracle_has_nontrivial_headroom`

## 五、阈值与公平性边界

Development 阈值只根据连续分数、eligible 状态和 actionable coverage 冻结，未使用最终候选面板标签：

- Local Spearman Gate: threshold `0.784470`；Development actionable coverage `0.513`。
- Target-Residual Spearman Gate: threshold `0.619529`；Development actionable coverage `0.487`。
- Disagreement-Correction Gate: threshold `0.890625`；Development actionable coverage `0.487`。

所有方法共享目标历史、Target-Only GP、候选池、源专家、Top-K 适用集合、源提名和真实评价面板；三个可部署 gate 只能接受同一个 `x_S` 或精确回退同一个 `x_T`。Oracle 只在真值揭示后计算，不进入可部署 AUPRC。

## 六、结论边界

本实验只判断是否值得继续设计这层信任机制。即使通过，也不证明实际区域检索、未知对齐、多源调度、闭环 BO 预算收益、高维泛化或普遍无负迁移。
