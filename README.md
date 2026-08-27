# 源局部区域引导的目标候选重排序研究 (Region-Guided Candidate Reranking Study)

本项目为探索**“目标代理模型负责产生候选可能性，源优质局部区域负责提供空间偏好先验进行软重排序”**的机制验证与边界分析原型。

---

## 1. 核心猜想与形式化定义

### 1.1 源区域库表示 (Source Region Library)
从历史源任务的优质采样点中提取 $K$ 个优质区域 $\mathcal{R}_s = \{R_1, \dots, R_K\}$，每个区域参数化为：
$$
R_k = (\mu_k, \Sigma_k, q_k, n_k)
$$
- $\mu_k \in \mathbb{R}^d$: 区域中心位置；
- $\Sigma_k \in \mathbb{R}^{d \times d}$: 区域协方差（有效范围与形状）；
- $q_k \in [0, 1]$: 区域平均归一化质量；
- $n_k \in \mathbb{N}^+$: 区域样本数 / 跨源任务重复出现频次（可信度）。

### 1.2 候选点源区域支持得分 (Source Region Support Score)
对于候选点 $x \in \mathbb{R}^d$：
$$
r_s(x) = \max_{k=1,\dots,K} q_k \exp\left( -\frac{1}{2} (x - \mu_k)^\top \Sigma_k^{-1} (x - \mu_k) \right)
$$

### 1.3 软重排序融合与低信息安全门控 (Soft Reranking with Safety Gating)
对 $x \in C_t$，计算其归一化采集得分 $\tilde{\alpha}_t(x)$ 与归一化源得分 $\tilde{r}_s(x)$（使用 `scipy.stats.rankdata` 处理并列值）：
$$
J_t(x) = \tilde{\alpha}_t(x) + \lambda_t \tilde{r}_s(x)
$$
**安全门控规则**：若源区域得分在候选池上的方差 $\mathrm{Var}(r_s) < \epsilon$（无判别力常数），自动置 $\lambda_t = 0.0$。

---

## 2. 六组对照基线设计 (Controlled Comparators)

所有方法**严格共享相同的��标初始样本、相同的目标 GP 代理模型和相同的排他候选池 $C_t$**（通过 `np.random.SeedSequence` 隔离随机流，候选池严格排除已评测目标点与源数据）。

| 编号 | 方法名称 | 机制说明 |
| :--- | :--- | :--- |
| **M1** | **Target-Only** | 仅按目标采集得分 $\tilde{\alpha}_t(x)$ 排序（零迁移基线） |
| **M2** | **Source-Region** | 目标采集 + 匹配源区域得分软融合 $\tilde{\alpha}_t(x) + \lambda \tilde{r}_s(x)$ |
| **M3** | **Random-Region** | **严格结构匹配随机对照**：保留源区域的真实协方差谱、体积、质量与频次，仅随机平移中心 $\mu_k$ |
| **M4** | **Wrong-Source** | **对抗/失配源对照**：区域中心指向目标函数的欺骗性局部极小或高损失区域 |
| **M5** | **Oracle-Target-Region** | 真实目标全局最优盆地先验（理论上界） |
| **M6** | **Hard-Filter** | **几何卡方置信域过滤**：仅保留位于区域 95% 置信椭球 $(x-\mu)^\top \Sigma^{-1} (x-\mu) \le \chi^2_{d, 0.95}$ 内的候选 |

---

## 3. 实验运行与复现指南

### 3.1 环境安装
```bash
pip install -e .
```

### 3.2 运行自动化测试
```bash
python -m pytest
```

### 3.3 运行全量三阶段实验
```bash
# 阶段一：固定候选池单步重排序机制检验
python scripts/run_mechanism_experiment.py

# 阶段二：连续空间漂移与有效性边界曲线
python scripts/run_drift_curve_experiment.py

# 阶段三：闭环序列贝叶斯优化迭代
python scripts/run_sequential_bo.py

# 生成可视化图表与数据驱动验证报告
python scripts/plot_and_report.py
```

---

## 4. 目录结构

```text
local_trans/
├── pyproject.toml
├── requirements.txt
├── README.md
├── src/
│   └── region_guided_reranking_study/
│       ├── __init__.py
│       ├── landscapes.py
│       ├── source_regions.py
│       ├── surrogate_and_candidates.py
│       ├── rerankers.py
│       ├── metrics.py
│       └── sequential_bo.py
├── scripts/
│   ├── run_mechanism_experiment.py
│   ├── run_drift_curve_experiment.py
│   ├── run_sequential_bo.py
│   └── plot_and_report.py
├── tests/
│   ├── test_random_isolation.py
│   ├── test_rank_ties.py
│   └── test_comparators.py
└── results/
    ├── VERIFICATION_REPORT.md
    ├── mechanism_2d_demonstration.png
    ├── statistical_hypothesis_validation.png
    ├── comparator_regret_comparison.png
    ├── drift_boundary_curve.png
    └── sequential_bo_convergence.png
```
