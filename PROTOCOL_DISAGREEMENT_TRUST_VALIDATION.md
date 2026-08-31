# Frozen Protocol: Disagreement-Conditioned Target-Feedback Trust Pilot v1-R

## 1. 唯一研究问题

在目标代理已经产生候选并形成自己的首选之后，**冲突条件信任**是否比 **Local Spearman** 更准确地预测：同一个源局部专家的下一次固定候选意见能否改善 Target-Only？

这是二维、固定局部图表、单源专家、共享候选的静态机制验证。为了获得可识别的成对决策反馈，本实验明确支付一组**源域不参与选择的成对诊断评价**；它不是自然单点反馈、等预算在线 BO 或预算节省实验。

## 2. v1-R 修订原因

第一次 Quick 仅用于代码验证，已永久归档为 `results/disagreement_trust_validation_quick_attempt1_rejected/`，不进入最终结论。独立审计指出：

1. 旧反馈对由源专家参与提名，导致反馈采样分布受被检验机制影响；
2. 旧 Local Spearman 使用全部点、冲突信任使用每轮一个决策事件，证据单位不匹配；
3. 旧 AUPRC 把证据不足映射为 0，混淆“未知”和“有害”。

v1-R 在正式运行前封闭修订：反馈对完全由 Target-Only acquisition 排名选出；三个估计器都只消费相同的每轮成对诊断事件；主 AUPRC 只在三者共同 eligible 的事件集合上比较，另报 neutral-imputed 全 actionable 敏感性结果。

## 3. 独立单位和受控关系

一个最终事件由 `(split, seed, relation)` 唯一标识。受控二维局部关系为：

- `identity`；
- `output_affine`；
- `scale_0.7`、`scale_1.5`；
- `rotate_45`；
- `anisotropy_swap`；
- `roughness`；
- `reversal`；
- `independent_expert`。

同一 seed 下各 relation 共享基础随机量，正式统计以 **seed cluster** 为外层重采样单位；关系、候选点、诊断对和面板点均不是独立重复。

Development seeds 仅按无标签 coverage 冻结门槛；Holdout seeds 只做最终判断。Quick-v1-R 仍只用于验证代码、稀疏度与审计产物，正式结论使用 Full-v1-R 的全新 development/holdout seeds。

## 4. 源专家和付费成对诊断反馈

### 4.1 固定源局部专家

源专家仅用源函数上的冻结 Sobol 样本训练。固定核 Matern-5/2 GP 学习源响应的相对 rank-quality，输出越大表示源专家认为越好。除 `independent_expert` 外，各关系使用同一个基础源专家；`independent_expert` 使用独立振荡源函数训练的专家。

局部图表与区域对应由生成器固定，因此结果不能解释为实际区域检索或对齐能力。

### 4.2 Source-blind 成对反馈

每个事件先建立所有 gate 共用的目标历史：

1. 用 source-blind Sobol 初始点拟合 Target-Only GP；
2. 每轮由 Target-Only GP 在新的 source-blind 原始池上计算 EI 并提出候选；
3. 固定诊断对为 acquisition 第 1 名 `x_T` 和配置中冻结的第 `r` 名 `x_D`；**源质量不参与二者选择**；
4. 揭示结果前保存二者的 Target-Only 均值、acquisition 和源质量；
5. 支付两次诊断评价，揭示 `f(x_T),f(x_D)`，并把这两个点都加入下一轮共享目标历史。

每轮始终固定两次评价，不因源专家是否冲突而改变。该轨迹只用于识别“历史冲突中谁更常正确”，不构成等预算优化结果。未来接入真实单点评价 BO 前，必须另行设计随机化日志、shadow evaluation 或其他可识别反馈，且重新核算目标评价预算。

## 5. 三个目标反馈分数

每轮只形成一个成对诊断单位。定义：

```text
source_margin = h_s(x_D) - h_s(x_T)
true_D_advantage = f(x_T) - f(x_D)
residual_D_advantage = [f(x_T)-mu_T(x_T)] - [f(x_D)-mu_T(x_D)]
```

正的 `source_margin` 表示源专家反对 Target-Only 并偏好 `x_D`；正的两个 advantage 表示 `x_D` 更好或比目标模型预期得更好。

### 5.1 Local Spearman Gate

计算所有诊断轮次上的：

```text
rho_local = Spearman(source_margin, true_D_advantage)
score = 0.5 + 0.5 * rho_local * n / (n + shrinkage)
```

它使用所有同单位成对事件，回答“源偏好强度与目标真实局部优势整体是否同向”。

### 5.2 Target-Residual Spearman Gate

计算：

```text
rho_residual = Spearman(source_margin, residual_D_advantage)
```

使用同一收缩形式。所有目标均值在该轮结果揭示前保存，禁止使用训练内残差。

### 5.3 Disagreement-Correction Gate

仅当 Target-Only 严格偏好 `x_T`、源专家严格偏好 `x_D` 时，形成冲突事件：

- success：`x_D` 比 `x_T` 至少好一个冻结容差；
- failure：`x_T` 比 `x_D` 至少好一个冻结容差；
- 实践并列：记录但不进入 success/failure。

连续分数为：

```text
P(p_correction > 0.5 | success, failure),
p_correction ~ Beta(1 + success, 1 + failure)
```

不足冻结最小非并列冲突事件数时 abstain。该比较使三种分数拥有相同反馈轮次、相同诊断对与相同付费预算；但其统计量和使用的信息子集仍不同，因此结论只针对这三个冻结完整估计器，不声称因果隔离了“条件化”本身。

## 6. 最终共享候选事件与五种方法

最终检查点只用共享目标历史拟合 Target-Only GP，并从新的 source-blind 原始池提出候选池 `C`：

- `x_T = argmax EI(x)`；
- 固定源专家只能在 acquisition 前 `K` 名内提名同一个 `x_S`；
- 只有当 Target-Only 严格偏好 `x_T` 且源专家严格偏好不同的 `x_S` 时，事件才 actionable；
- 所有非 Oracle 分数、候选、提名和选择必须先落盘，之后才揭示整个候选面板的真值。

恰好比较：

1. `Target-Only`：始终选择 `x_T`；
2. `Local Spearman Gate`；
3. `Target-Residual Spearman Gate`；
4. `Disagreement-Correction Gate`；
5. `Oracle Gate`：只在揭示后且 `f(x_S)<f(x_T)-epsilon` 时接受。

三个可部署 gate 只能接受同一个 `x_S` 或精确回退同一个 `x_T`；不得改变目标历史、GP、候选、acquisition、源专家、Top-K 适用集合、源提名或候选分数。`x_S==x_T` 为 no-op，全部回退。

## 7. 无标签阈值冻结

三个 gate 的阈值只使用 Development 中的连续分数、eligible 状态和最终事件是否 actionable，不使用最终候选面板标签。

阈值必须严格大于 neutral score `0.5`，并选择使 Development actionable coverage 最接近配置目标的取值；coverage 误差并列时选择更高、更保守的阈值。Holdout 不重新调阈值。证据不足、低于阈值或 no-op 均回退 Target-Only。

## 8. 冻结评价指标

所有方法共享同一真实候选面板。最小化问题中报告：

- **一步候选 raw regret**：`f(x_method)-min(C)`；
- **一步候选 normalized regret**：除以 `q90(f(C))-min(C)`；
- **Top-10% hit**：以名义第 `ceil(0.1|C|)` 名的真实值为 cutoff，并把 cutoff 加冻结容差内的并列候选全部记为 hit，禁止按候选索引拆散并列；
- **overall/actionable acceptance coverage**；
- **接受后负迁移率**：接受事件中 `f(x_S)>f(x_T)+epsilon` 的比例，始终报告分子/分母；无接受为 NA；
- **主 AUPRC**：只在 actionable 且三个 gate 全部 eligible 的共同支持集上，用揭示前连续分数预测 `f(x_S)<f(x_T)-epsilon`；报告共同支持 coverage 与该集合的正例 prevalence；
- **AUPRC 敏感性**：在全部 actionable 事件上把 ineligible 映射为 neutral `0.5`，只作为“选择性预测 + 可用性”的合成诊断，不作为纯预测主证据；
- **Oracle headroom**。

每轮和最终事件的 `epsilon` 都在结果揭示前由已有目标历史计算：

```text
max(abs_tol, rel_tol * (q90(history_y)-min(history_y)))
```

同时持久化 `source_gain/epsilon`，避免只凭 AUPRC 把微小差异解释成实际效用。

## 9. 统计与推进标准

Holdout 采用 seed-cluster bootstrap。主对比：

1. Disagreement Gate 相对 Local Spearman 的 normalized regret 优势；
2. 二者共同支持集 AUPRC 差；
3. coverage、Top-10% hit 和接受后负迁移率；
4. Disagreement Gate 相对 Target-Only；
5. Oracle headroom。

bootstrap 对选择性 harm rate 若出现无接受的未定义重采样，必须报告有限 replicate 数和比例；低于冻结比例时该项判为不确定，不能静默删除后宣称通过。

只有同时满足以下冻结模式，才建议继续复杂化冲突信任：

- 共同支持集样本与正例数达到最低要求；
- AUPRC 相对 Local Spearman 的 cluster-bootstrap 95% CI 下界大于 0，且高于共同支持集 prevalence；
- normalized regret 优势达到实践 margin，且 CI 下界大于 0；
- coverage 可比，Top-10% hit 与接受后负迁移率在冻结容差内不劣；
- 相对 Target-Only 有实际决策收益；
- Oracle 存在非平凡 headroom。

若证据稀疏或选择性风险 bootstrap 大量未定义，结论为 `inconclusive`；若只改善 AUPRC、只靠 coverage 崩塌变安全，或候选结果不改善，则不推进复杂机制。

## 10. 边界

即使通过，本 Pilot 也只支持：在二维、固定图表、单专家、共享候选和付费 source-blind 成对诊断反馈下，冻结的冲突纠错估计器比冻结的 decision-matched Local Spearman 更接近下一次固定源提名的增量价值。它不证明自然单点反馈可学性、实际区域检索、未知对齐、多源调度、在线预算收益、高维泛化或普遍无负迁移。
