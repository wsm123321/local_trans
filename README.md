# 源局部区域引导的目标候选重排序研究 (Region-Guided Candidate Reranking Study)

## 1. 核心猜想与研究定位

传统昂贵多任务迁移优化往往尝试直接迁移“源最优点”、“源响应表面模型”或“源高阶局部曲率/Hessian”。然而，在少样本（few-shot）目标优化初期，这些高自由度模型迁移极易带来模型失配与严重的负迁移风险。

本研究验证的新猜想：**将候选生成权与源知识解耦**：
$$
\boxed{\text{目标代理模型负责“产生可能性”} + \text{源优质区域先验负责“提高命中率”}}
$$

### 核心可检验条件
设 $U_t(x)$ 为候选点在目标任务上的真实效用（例如负目标函数值或真实改进量），$\alpha_t(x)$ 为目标代理模型计算的采集得分（Acquisition Score），$r_s(x)$ 为源区域支持得分：
$$
\boxed{I(U_t(x); r_s(x) \mid \alpha_t(x)) > 0}
$$
即：**在已知目标代理模型打分的条件下，源优质区域得分仍能提供统计显著的增量候选质量信息。**

---

## 2. 形式化与数学定义

### 2.1 源区域库表示 (Source Region Library)
从历史源任务的优质采样点中提取 $K$ 个优质区域 $\mathcal{R}_s = \{R_1, \dots, R_K\}$，每个区域参数化为：
$$
R_k = (\mu_k, \Sigma_k, q_k, n_k)
$$
- $\mu_k \in \mathbb{R}^d$: 区域中心位置；
- $\Sigma_k \in \mathbb{R}^{d \times d}$: 区域协方差（有效范围与形状）；
- $q_k \in [0, 1]$: 区域平均归一化质量；
- $n_k \in \mathbb{N}^+$: 区域样本数 / 跨源任务重复出现频次（可信度）。

### 2.2 候选点源区域支持得分 (Source Region Support Score)
对于候选点 $x \in \mathbb{R}^d$：
$$
r_s(x) = \max_{k=1,\dots,K} q_k \exp\left( -\frac{1}{2} (x - \mu_k)^\top (\Sigma_k + \epsilon I)^{-1} (x - \mu_k) \right)
$$

### 2.3 候选池与软重排序融合 (Soft Reranking)
目标代理模型（如 GP）在目标样本 $\mathcal{D}_t$ 上训练，生成宽候选池：
$$
C_t = C_{\text{acq}} \cup C_{\text{global}} \cup C_{\text{diverse}}
$$
对 $x \in C_t$，计算其归一化���集得分 $\tilde{\alpha}_t(x)$ 与归一化源得分 $\tilde{r}_s(x)$：
$$
J_t(x) = \tilde{\alpha}_t(x) + \lambda_t \tilde{r}_s(x)
$$

---

## 3. 六组对照基线设计 (Controlled Comparators)

所有方法**严格共享**：相同目标初始样本、相同目标代理模型、相同候选池 $C_t$、相同随机种子、相同评测预算。

| 编号 | 方法名称 | 排序依据 / 机制 | 预期作用与检验目的 |
| :--- | :--- | :--- | :--- |
| **M1** | **Target-Only** | 仅按目标采集得分 $\tilde{\alpha}_t(x)$ | 零迁移基线（标准 BO 选择） |
| **M2** | **Source-Region (Ours)** | 目标采集 + 匹配源区域得分 $\tilde{\alpha}_t(x) + \lambda \tilde{r}_s(x)$ | 本文核心方法 |
| **M3** | **Random-Region** | 目标采集 + 随机位置区域先验 | 检验增益是否仅来自“额外空间正则/散布偏好” |
| **M4** | **Wrong-Source** | 目标采集 + 明确失配/对抗源区域 | 负迁移压力测试，检验抗干扰能力 |
| **M5** | **Oracle-Target-Region** | 目标采集 + 真实目标最优盆地区域 | 性能上界（理论最佳区域先验） |
| **M6** | **Hard-Filter** | 仅保留源区域覆盖内的候选（硬过滤） | 验证软融合对比硬过滤的安全边界 |

---

## 4. 检验指标体系

1. **条件增量检验 (Conditional Incremental Information)**
   - 偏相关系数 $\rho(U_t, r_s \mid \alpha_t)$；
   - 增量可解释方差 $\Delta R^2 = R^2(U_t \sim \alpha_t + r_s) - R^2(U_t \sim \alpha_t)$；
   - 条件互信息估计 $I(U_t; r_s \mid \alpha_t)$。
2. **候选决策质量 (Candidate Selection Quality)**
   - 最终所选 Top-1 / Top-k 候选的真实 Simple Regret；
   - 单步真实改进量 (One-step Improvement)；
   - Top-10% 优质解命中率 (Top-k Hit Rate)；
   - 真实最优候选在重排序前后的排位变化 (Rank Improvement)。
3. **因果归因判定准则**
   - 满足 $\text{Source-Region} > \text{Random-Region}$（排除随机伪效应）；
   - 满足 $\text{Wrong-Source} \le \text{Target-Only}$ 且在自适应下负迁移可控；
   - 软重排序在目标模型发现新区域时优于 Hard-Filter。
