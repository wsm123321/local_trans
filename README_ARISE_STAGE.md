# 下一阶段：从“任务相似度”转向“决策条件下的局部可迁移性”

## 0. 阶段判断

现有完整实验已经支持两个事实：

1. 当源域优质局部区域与目标域真正相关时，固定筛选和软重排序能够加速目标搜索；
2. 当前基于全局 Spearman 相关与 elite enrichment 的自适应相似度没有可靠识别能力，固定筛选的收益边界也没有被统计分辨。

因此，下一阶段不应继续微调一个全局相似度阈值，而应改变问题定义：

> **不再问“源任务与目标任务是否相似”，而是问“源域的哪一个局部区域，是否能为当前目标决策提供超出目标代理模型自身预期的增量改进”。**

本压缩包实现 **ARISE-BO：Active Region Identification and Safe Exploitation**。

---

## 1. 为什么常用相似度匹配在这里容易失效

### 1.1 全局任务相似度掩盖局部异质性

常见做法把一个源任务压缩成单一相似度：

- 共享点函数值相关性或排序相关性；
- 任务核/任务协方差；
- 两组样本或表示之间的 MMD、Wasserstein 距离；
- 元特征或任务嵌入距离；
- 源模型在目标样本上的似然或预测误差。

但你的问题中真正可迁移的对象是若干局部区域。同一个源任务可能同时包含：

- 与目标全局最优盆地相似的区域；
- 与目标局部最优相似但会误导搜索的区域；
- 在目标域完全无关的区域。

单个全局分数会把正迁移和负迁移平均掉。

### 1.2 几何相似或响应相关不等于优化决策有用

两个区域中心接近、协方差相似，或者函数值呈正相关，不代表使用该区域会选出比目标采集函数更好的下一个点。

优化真正关心的是：

\[
\text{使用区域后的决策收益}
-
\text{目标模型本来能够得到的决策收益}.
\]

因此，静态几何距离与全局响应相关性和最终候选选择之间存在目标错位。

### 1.3 Few-shot 目标样本导致相关系数极不稳定

目标初期只有少量、非均匀、由优化策略自适应采集的点。直接计算

\[
\rho(r_s(x_i),-y_i)
\]

会受到以下影响：

- 样本没有覆盖源区域；
- 样本集中在目标采集函数偏好的位置；
- 不同区域的证据被混在一起；
- 一个极端样本可以改变相关系数符号；
- 相关性没有表达估计不确定性。

### 1.4 当前相似度没有控制目标代理模型已经知道的信息

如果一个点本来就被目标 GP 判断为优质，那么它同时落在源区域内，并不能证明源区域提供了增量知识。

需要控制目标模型的预测与采集预期，检查源区域是否解释了目标模型之外的“意外好结果”。

### 1.5 被动匹配存在不可避免的冷启动

只有目标点偶然落入源区域后，才能知道该区域是否有效；但如果算法从不访问该区域，就永远没有证据。这是一个主动学习问题，而不只是相似度计算问题。

---

## 2. 新定义：局部区域相似 = 正的超额改进效应

在第 \(t\) 次目标评价前，目标代理模型在候选点 \(x_t\) 给出高斯预测：

\[
Y_t(x_t)\sim\mathcal N(\mu_t(x_t),\sigma_t^2(x_t)).
\]

以当前最好值 \(y_t^*\) 为基准，随机改进量为：

\[
I_t(x)=\max(0,y_t^*-Y_t(x)-\xi).
\]

目标 GP 可以计算其期望和方差：

\[
\mathbb E_t[I_t(x)],\qquad
\operatorname{Var}_t[I_t(x)].
\]

获得真实目标值后，定义标准化超额改进：

\[
e_t=
\frac{
I_t^{\mathrm{real}}(x_t)-\mathbb E_t[I_t(x_t)]
}{
\sqrt{\operatorname{Var}_t[I_t(x_t)]+\epsilon^2}
}.
\]

含义：

- \(e_t>0\)：该点比目标代理模型评价前预期的更好；
- \(e_t<0\)：该点低于目标代理模型的预期；
- 它已经控制了目标模型本身的预测、方差和采集偏好。

设第 \(k\) 个源局部区域对当前目标候选集中点 \(x_t\) 的相对支持度为 \(\tilde s_k(x_t)\in[0,1]\)。若该区域在当前候选集上的绝对支持几乎为零，则该列直接置零。ARISE 拟合：

\[
e_t = b + \sum_{k=1}^{K}\beta_k \tilde s_k(x_t)+\varepsilon_t,
\qquad
\beta_k\sim\mathcal N(0,\tau^2).
\]

于是 \(\beta_k\) 表示：

> **在控制目标代理模型的预期后，落入区域 \(k\) 是否仍倾向于产生额外改进。**

对每个区域维护后验：

\[
p(\beta_k\mid\mathcal D_t),
\]

并计算：

- 后验均值；
- 可信下界 LCB；
- 可信上界 UCB；
- \(P(\beta_k>0)\)；
- 有效覆盖量；
- 指数遗忘后的当前证据（默认旧证据每轮乘 0.95），使“相似性”可以随搜索阶段改变。

区域状态定义为：

- **trusted**：覆盖充分且 LCB \(>0\)；
- **rejected**：覆盖充分且 UCB \(<0\)；
- **uncertain**：证据不足或区间跨越 0。

这给出了问题 1“如何判断局部区域相似”的可检验答案。

---

## 3. 搜索策略：目标回退、可信利用、主动探测

目标代理模型仍然独立生成候选集 \(C_t\)。源域不生成原始候选。

### 3.1 Target fallback

没有可信区域且不进行探测时：

\[
x_t=\arg\max_{x\in C_t}\alpha_t(x).
\]

### 3.2 Safe exploitation

只使用可信区域：

\[
w_{k,t}=\max(0,\operatorname{LCB}(\beta_k)),
\]

\[
g_t(x)=\sum_{k:\,trusted}w_{k,t}\tilde s_k(x),
\]

\[
J_t(x)=\tilde\alpha_t(x)+\lambda g_t(x).
\]

区域引导只能在目标采集排名前一定比例的候选中选择，避免把搜索强行拉到目标模型完全不认可的位置。

### 3.3 Active probing

对于 UCB 仍可能为正、但证据不足的区域，使用区域效应后验协方差 \(\Sigma_{\beta,t}\) 计算一次评价的辨识价值。设候选的未决区域支持向量为 \(z_t(x)\)，则：

\[
V_t(x)=z_t(x)^\top\Sigma_{\beta,t}z_t(x),
\qquad
IG_t(x)=\frac12\log(1+V_t(x)).
\]

这比逐区域方差求和更能处理重叠区域：若两个区域效应高度相关，重复探测不会获得虚假的双倍价值。再以 \(P(\beta_k>0)\) 对仍有正迁移可能的区域进行乐观调节。

在早期按照固定间隔，从目标采集排名靠前的候选中选择兼具采集价值和后验信息增益的点：

\[
J_t^{probe}(x)=\tilde\alpha_t(x)+\eta\,\widetilde{IG}_t(x).
\]

探测得到的真实目标值会更新区域后验。这样解决“没有访问区域就无法判断相似性”的冷启动问题。

这给出了问题 2“如何让源域指导目标域搜索”的完整策略。

---

## 4. 论文故事

### 4.1 研究问题

昂贵迁移优化通常依赖全局任务相似度来决定是否迁移，但目标任务只有极少样本，且源任务内部的局部区域具有异质性。全局相似并不能回答哪个源区域对当前目标决策真正有用。

### 4.2 核心观察

当前实验发现：

- 匹配区域的固定筛选和软重排序能够显著加速；
- 当前全局自适应相似度基本退化为保守的 Target-Only；
- 这说明问题不在于“区域知识是否有用”，而在于“如何在线识别哪一个区域有用”。

### 4.3 方法转折

现有方法问：

> Are the source and target tasks similar?

本文改为：

> Which source local region produces target improvement beyond the target surrogate's own expectation, and when should it be probed or exploited?

### 4.4 方法贡献

1. **决策条件下的局部可迁移性定义**：用超额改进而非几何距离或原始函数相关性定义区域相似；
2. **区域级贝叶斯后验**：分别保留正证据、负证据和不确定性，避免全局平均；
3. **主动相似性识别**：把相似度估计变成受预算约束的主动实验；
4. **安全搜索策略**：可信区域利用、不确定区域探测、无证据时目标回退。

### 4.5 可检验主张

- H1：区域后验 \(P(\beta_k>0)\) 比全局 Spearman、elite enrichment、几何距离更准确地预测真实 counterfactual region gain；
- H2：ARISE 在 mixed-source 场景下优于固定软融合和旧全局自适应规则；
- H3：ARISE 在 wrong-source 场景下接近 Target-Only，负迁移小于固定迁移；
- H4：主动探测降低“识别出第一个可信有效区域”所需的目标评价次数；
- H5：ARISE 的收益来自可信区域利用，而不是额外随机探索。

---

## 5. 实验设计

### 5.1 区域识别实验

每一轮在同一个目标候选集上离线计算各区域专家候选的真实目标值：

\[
G_{k,t}=f_t(x_{0,t})-f_t(x_{k,t}),
\]

其中 \(x_{0,t}\) 是 Target-Only 候选，\(x_{k,t}\) 是区域 \(k\) 指引的候选。\(G_{k,t}>0\) 表示该区域在当前决策中真实有用。

指标：

- AUROC / AUPRC；
- Brier score 与校准曲线；
- 后验均值与真实 \(G_{k,t}\) 的 Spearman；
- trusted precision；
- rejected precision；
- time-to-first-correct-trust；
- 不同目标预算下的识别性能。

### 5.2 等预算优化实验

统一目标初始点和评价预算，比较：

- `target_only`：纯目标 BO；
- `fixed`：所有区域固定软引导；
- `global_adaptive`：复现当前全局 Spearman + elite enrichment 自适应规则；
- `posterior`：只利用可信区域，不主动探测；
- `arise`：完整方法。

场景：

- matching：只有有效区域；
- mixed：有效、随机和错误区域混合；
- wrong：只有错误区域。

指标：

- final normalized simple regret；
- regret AUC；
- 达到阈值所需评价次数；
- paired win/loss rate；
- 负迁移率；
- probe/exploit/target 步数；
- 区域状态随时间的变化。

### 5.3 必做消融

- 去掉超额改进标准化，直接使用真实 \(-y\)；
- 全局单一后验 vs 区域级后验；
- 去掉 active probing；
- 去掉可信下界，只用后验均值；
- 去掉采集排名 gate；
- 不同 prior variance、coverage threshold、probe interval。

---

## 6. 文件说明

```text
src/region_guided_reranking_study/arise_transfer.py
    ARISE-BO 核心算法、区域后验、超额改进、主动探测与安全利用。

scripts/run_arise_similarity_study.py
    运行区域识别与等预算优化实验，保存原始 CSV。

scripts/analyze_arise_study.py
    计算识别指标、配对统计、绘图并生成数据驱动报告。

scripts/run_all_arise_studies.py
    一键运行实验和分析。

configs/arise_quick.json
configs/arise_full.json
    快速验证与完整实验配置。

tests/test_arise_transfer.py
    改进量矩、区域正负后验和 counterfactual gain 测试。
```

---

## 7. 运行方式

将压缩包内容覆盖到仓库根目录后：

```bash
pip install -e .
python -m pytest
```

快速验证：

```bash
python scripts/run_all_arise_studies.py \
  --config configs/arise_quick.json \
  --output results/arise_quick
```

完整实验：

```bash
python scripts/run_all_arise_studies.py \
  --config configs/arise_full.json \
  --output results/arise_full
```

主要输出：

```text
results/arise_full/
├── arise_optimizer_summary.csv
├── arise_optimizer_traces.csv
├── arise_region_identification.csv
├── arise_run_manifest.json
└── analysis/
    ├── ARISE_STUDY_REPORT.md
    ├── arise_identification_metrics.csv
    ├── arise_aggregate_metrics.csv
    ├── arise_convergence.png
    └── arise_identification.png
```

---

## 8. 相关研究脉络

本阶段应在论文中把已有工作归纳为四类，而不是逐个罗列算法：

1. multi-task kernel / task covariance：把相似性放入联合代理模型；
2. source-model ensemble / ranking transfer：按源模型的目标表现加权；
3. distribution or representation matching：用任务分布、元特征或嵌入距离判断相似；
4. source-prior / response transformation：把历史响应面、排序或分位数迁移到目标模型。

可参考的代表性工作包括：

- Shilton et al., *Regret Bounds for Transfer Learning in Bayesian Optimisation*, AISTATS 2017;
- Salinas et al., *A Quantile-based Approach for Hyperparameter Transfer Learning*, ICML 2020;
- multi-task-kernel Bayesian optimization and ranking/model-ensemble transfer methods。

ARISE 与这些工作的区别不是提出另一个任务距离，而是把“可迁移性”定义成区域对当前优化决策的增量效用，并在线主动辨识。
