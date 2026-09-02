# Unified Local Guidance 正式决策

## 1. 最终判定

本阶段在相同 source observations、同一个 `SourceLocalStructureLibrary`、相同 Target-GP 配置和相同目标评测预算下，直接比较：

1. `Target-Only`；
2. `Geometry-Only`；
3. `Local-Rank-No-Reliability`；
4. `Local-Rank+Reliability`。

正式判定为：

> **当前局部 rank surrogate 没有在区域几何之上提供可确认的候选决策或 sequential BO 增量；不推进当前 `structure_score` 形式的 model-aware online guidance。保留 Source Local Structure 作为已验证的 source-side 抽取成果，并保留低自由度 Geometry-Only 候选引导作为后续基线。**

这不是“局部模型没有学习到 source structure”的结论。Source Structure 阶段已经证明其 source-side fidelity；本阶段否定的是更窄的主张：在当前无对齐、共享坐标和固定 rank-fusion 机制下，把局部模型分数叠加到区域几何上，没有稳定改善目标候选选择或等预算 BO。

## 2. 正式规模和审计

- 4 个问题族：GMM、Rastrigin、Lunacek、Ackley；
- 2D 和 5D；
- 8 个 seeds；
- 64 个独立 `(problem, dim, seed)` instances；
- 320 条机制决策记录；
- 6,400 条共享机制候选记录；
- 256 条 sequential summaries；
- 5,120 次付费 target evaluations，外加 256 条 step-0 初始状态；
- 81 个抽取出的 source local structures；
- 0 failures；
- 5,000 次 instance-level paired bootstrap；
- one-sided Wilcoxon with Pratt zero handling；
- 五项 primary hypotheses 使用 Holm FWER correction。

`AUDIT.json` 的所有检查均通过，包括配置、协议和 artifact hashes、行数、唯一键、finite metrics、机制候选共享、Target-Only 零基 acquisition rank、fallback 一致性及轨迹长度。

## 3. 五项主检验

所有 effect 均定向为正值有利于新方法。

| 主检验 | 平均 effect | 95% CI | Holm p | 支持 |
|---|---:|---:|---:|---|
| H1 机制 normalized regret：Local+Reliability vs Geometry | -0.00630 | [-0.03279, +0.02071] | 1.0000 | 否 |
| H2 机制 Top-10% hit：Local+Reliability vs Geometry | +0.03125 | [-0.03125, +0.09375] | 0.7933 | 否 |
| H3 Sequential final regret：Local+Reliability vs Geometry | -0.03466 | [-0.07497, +0.00713] | 1.0000 | 否 |
| H4 Sequential regret AUC：Local+Reliability vs Geometry | -0.14844 | [-0.78745, +0.44526] | 1.0000 | 否 |
| H5 机制 reliability 增量：Local+Reliability vs Local-No-Reliability | 0.00000 | [0.00000, 0.00000] | 1.0000 | 否 |

五项主检验均未达到“bootstrap CI 下界大于 0 且 Holm-adjusted p 小于 0.05”的预注册支持标准。

## 4. 方法均值

### 4.1 共享候选机制

| 方法 | normalized regret | Top-10% hit |
|---|---:|---:|
| Target-Only | 0.63783 | 0.07813 |
| Geometry-Only | 0.44577 | 0.29688 |
| Local-Rank-No-Reliability | 0.45207 | 0.32813 |
| Local-Rank+Reliability | 0.45207 | 0.32813 |
| Reversed-Local-Rank | 0.53237 | 0.15625 |

Geometry-Only 相对 Target-Only 的均值方向更好，但该对比不是本阶段五项主检验之一，因此这里只作描述性结果。主问题是局部模型能否超越几何；答案是否定的。

Local-Rank+Reliability 与 Geometry-Only 在 64 个机制实例中有 51 个选择完全相同。发生不同选择时，local model 并没有形成稳定优势：normalized-regret 配对结果为 7 胜、51 平、6 负。

### 4.2 等预算 sequential BO

| 方法 | final normalized regret | regret AUC | total improvement |
|---|---:|---:|---:|
| Target-Only | 0.47385 | 14.01641 | 7.49218 |
| Geometry-Only | 0.40487 | 12.00325 | 8.69495 |
| Local-Rank-No-Reliability | 0.43953 | 12.15168 | 7.98100 |
| Local-Rank+Reliability | 0.43953 | 12.15168 | 7.98100 |

相对 Geometry-Only，Local-Rank+Reliability 的 final regret 平均更差 0.03466，配对为 18 胜、21 平、25 负；regret AUC 平均更差 0.14844，配对为 21 胜、18 平、25 负。两者置信区间均跨 0，因此不能宣称局部模型显著有害，但也没有推进依据。

## 5. Reliability 为什么没有增量

`Local-Rank-No-Reliability` 与 `Local-Rank+Reliability`：

- 64 个共享候选机制实例全部选择相同；
- sequential 中有 22/1,280 个付费 step 的 `selected_y` 不同，23/1,280 个 step 的 proposal hash 不同；
- 但两者的 incumbent `best_y`、normalized-regret 轨迹、final regret 和 AUC 全部相同；
- H5 机制 effect 精确为 0。

这说明当前 reliability 因子偶尔改变了 sequential 候选决策及其后续 proposal，但没有改变任何冻结的性能主指标。主要机制原因可能是：

1. 很多 instance 只抽取一个结构，对该结构整体乘以 reliability 后，再做候选内 rank normalization，排序不变；
2. 多结构 instance 中，reliability 可以改变少量候选或后续 proposal，但没有改变 incumbent；
3. reliability 是每结构的标量，而不是候选条件化可信度。

因此不能把“加入 reliability 字段”表述为已经实现有效门控。若以后重做，应让可靠度影响是否使用某区域或候选，而不是只作为 rank normalization 前的整体缩放。

## 6. Reversed 安全诊断

`Reversed-Local-Rank` 保留相同区域几何，只反转局部模型 quality factor。它相对 Geometry-Only：

- 机制 normalized regret effect 为 -0.08660，95% CI [-0.15295, -0.02745]；
- Top-10% hit effect 为 -0.14063，95% CI [-0.23438, -0.06250]。

负 effect 表示 reversed local knowledge 更差。这说明当前融合机制确实会响应局部模型方向，且错误的局部排序能够造成可观察伤害；正常 local rank 没有增量并不是因为模型分支完全未接入代码。

该 reversal 是显式安全压力测试，不代表自然任务分布，也不构成一般 no-harm 估计。

## 7. 分层描述

局部模型没有在维度分层中显示稳定增量：

- 2D 机制 regret：Geometry 0.47453，Local 0.47305；sequential final：Geometry 0.18219，Local 0.20003；
- 5D 机制 regret：Geometry 0.41701，Local 0.43109；sequential final：Geometry 0.62755，Local 0.67902。

问题族分层中，Local 相对 Geometry 在 Ackley 只有很小的机制方向改善；GMM 基本相同；Lunacek 和 Rastrigin 的 sequential final 均更差。以上均为描述性切片，不替代冻结主检验。

## 8. 研究含义与后续边界

本阶段支持以下决策：

1. **保留 Geometry-Only。** 区域中心、协方差和区域质量已经能够提供较强的候选偏好；相对 Target-Only 的正式确认可另行预注册，但现有均值足以将其保留为基线。
2. **不推进当前 Local-Rank fusion。** Source-side local fidelity 没有自动转化成 target-side candidate utility。
3. **不推进当前 reliability 乘法。** 它在本实验中没有改变任何主决策。
4. **不据此否定对齐或景观模型。** 本实验没有做区域匹配、输入空间对齐、输出校准或 value/curvature transfer；不能推断这些方向无效。
5. **不直接继续 online BO 调参。** 在同一结果上调整 source weight、nomination ratio 或 reliability 公式会变成 post-hoc 优化，必须建立新协议和独立 seeds。

如果后续重新研究局部模型，应先改变可识别性而不是扩大现有 full：

- 使用 candidate-conditioned trust，而不是每结构标量缩放；
- 设计比 Geometry-Only 更有区分力的局部 candidate panel；
- 增加 source-label permutation、cross-seed expert swapping 和 region-level wrong-local-model controls；
- 先做 oracle alignment/value-head upper-bound，再决定是否值得进入新的 sequential study。

## 9. Provenance

该正式 artifact 生成时：

- Git HEAD：`62f53ea7d38ddf859e283b5053431ad1cc597dfa`；
- 工作树为 dirty，包含本阶段新代码、协议、配置、测试、结果以及此前 README 更新；
- Python 3.12.13；NumPy 2.4.3；pandas 2.3.3；SciPy 1.17.1；scikit-learn 1.8.0。

因此结果应按 `run_manifest.json` 中的 config/protocol/runner/companion hashes 引用，不能表述为由干净的当前 Git commit 单独生成。
