# 下一阶段：目标代理提出候选，源局部区域负责筛选

本代码包对应以下严格解耦流程：

```text
目标历史数据 Dt
      │
      ▼
训练目标代理模型 GPt
      │
      ▼
从大候选池中提出 M 个目标候选 Pt
      │
      ▼
源域优质样本聚类得到局部区域库 Rs
      │
      ▼
只对 Pt 计算源区域支持度 rs(x)
      │
      ▼
按源区域支持度筛选 Pt，得到 St ⊆ Pt
      │
      ▼
在 St 内按目标采集函数选择真实评价点
      │
      ▼
更新目标数据并进入下一轮
```

关键约束是：**源域区域不能生成新候选，也不能参与目标代理训练。它只能在目标代理已经提出的候选集合内进行筛选。**

## 一、核心算法

新增模块：

```text
src/region_guided_reranking_study/target_region_screening.py
```

核心类：

```python
RegionFilteredTargetBO
```

标准使用方式：

```python
from region_guided_reranking_study import (
    RegionFilteredBOConfig,
    RegionFilteredTargetBO,
    RegionScreeningConfig,
    TargetProposalConfig,
)

config = RegionFilteredBOConfig(
    proposal=TargetProposalConfig(
        raw_pool_size=1500,
        proposal_size=100,
        acquisition="ei",
    ),
    screening=RegionScreeningConfig(
        policy="adaptive",
        geometry="quantile",
        retain_ratio=0.25,
    ),
    random_state=42,
)

optimizer = RegionFilteredTargetBO(bounds, config)
optimizer.fit_source_regions(source_datasets)
result = optimizer.optimize(
    objective=target_function,
    init_X=target_init_X,
    init_y=target_init_y,
    budget=20,
)
```

### 1. 目标候选提出

目标 GP 在目标数据上独立训练。它先生成大候选池，再按照目标采集函数提出 `proposal_size` 个候选：

\[
P_t = \operatorname{TopM}_{x\in C_t}\alpha_t(x).
\]

源域数据和源区域得分不会传入候选提出函数。

### 2. 固定区域筛选

固定筛选保留源支持度最高的一部分目标候选：

\[
S_t=\operatorname{Top}_{\eta |P_t|}\{r_s(x):x\in P_t\},
\]

其中 `retain_ratio = η`。最终仍然按照目标采集函数，在 `S_t` 内选择评价点：

\[
x_t=\arg\max_{x\in S_t}\alpha_t(x).
\]

这与软加权重排序不同：源知识决定候选是否进入短名单，目标模型决定短名单中的最终顺序。

### 3. 目标证据驱动的自适应筛选

自适应策略利用已经获得的目标观测估计源区域兼容性，包括：

- 源区域支持度与目标真实效用的 Spearman 排名相关；
- 目标优质样本相对普通样本的源支持度富集程度。

兼容性较低时保留更多目标候选；兼容性较高时筛选更强：

\[
\eta_t=1-c_t(1-\eta_{\min}),
\]

其中 `c_t∈[0,1]` 是经少样本收缩后的兼容性。若兼容性低于激活阈值，则本轮不进行源区域筛选，退化为 Target-Only。

这不是已经证明有效的负迁移防护，而是下一阶段需要通过实验检验的算法假设。

## 二、科研实验代码

### A. 共享候选机制实验

```bash
python scripts/run_screening_mechanism_study.py \
  --config configs/region_screening_full.json
```

严格共享：

- 相同目标初始数据；
- 相同目标 GP；
- 相同原始候选池；
- 相同目标提出候选集合；
- 相同真实评价预算。

比较方法：

```text
Target-Only
Matching-Fixed-Filter
Matching-Adaptive-Filter
Matching-Soft-Rerank
Random-Adaptive-Filter
Wrong-Adaptive-Filter
Oracle-Fixed-Filter
```

主要回答：

1. 匹配源区域是否能在目标已经提出的候选中筛出更好的点？
2. 收益是否优于结构匹配随机区域和错误源区域？
3. 硬筛选与软重排序的贡献是否不同？
4. 自适应兼容性是否真的能识别源区域有效性？

输出：

```text
results/region_screening/mechanism/
├── screening_mechanism_summary.csv
└── screening_mechanism_details.json
```

### B. 等预算闭环优化实验

```bash
python scripts/run_screening_sequential_study.py \
  --config configs/region_screening_full.json
```

每种方法分别评价自己选择的目标点、更新自己的目标代理，并保持完全相同的目标评价预算。

输出：

```text
results/region_screening/sequential/
├── screening_sequential_summary.csv
└── screening_sequential_traces.csv
```

### C. 源区域漂移边界实验

```bash
python scripts/run_screening_drift_study.py \
  --config configs/region_screening_full.json
```

目标任务、目标初始数据和目标候选集合保持不变，只连续平移源任务优质区域：

```text
δ = 0, 0.25, 0.5, 1, 2, 4
```

同时比较 Fixed-Filter 和 Adaptive-Filter，确定：

- 正迁移区域；
- 结果不确定区域；
- 负迁移区域；
- 自适应筛选是否比固定筛选更安全。

输出：

```text
results/region_screening/drift/
└── screening_drift_summary.csv
```

### D. 动态统计分析

```bash
python scripts/analyze_screening_studies.py \
  --config configs/region_screening_full.json
```

报告结论全部由 CSV 动态计算，不预先写死有效边界或优劣结论。

输出：

```text
results/region_screening/analysis/
├── mechanism_normalized_regret.png
├── sequential_normalized_regret.png
├── drift_transfer_boundary.png
└── SCREENING_STUDY_REPORT.md
```

## 三、运行顺序

先做快速检查：

```bash
pip install -e .
python -m pytest

python scripts/run_screening_mechanism_study.py \
  --config configs/region_screening_quick.json
python scripts/run_screening_sequential_study.py \
  --config configs/region_screening_quick.json
python scripts/run_screening_drift_study.py \
  --config configs/region_screening_quick.json
python scripts/analyze_screening_studies.py \
  --config configs/region_screening_quick.json
```

快速配置正常后，再运行完整配置。

也可以一次运行全部阶段：

```bash
python scripts/run_all_screening_studies.py \
  --config configs/region_screening_quick.json
```

## 四、建议提交到 GitHub 的内容

完成本地实验后提交：

```text
src/region_guided_reranking_study/target_region_screening.py
src/region_guided_reranking_study/screening_research.py
scripts/run_screening_mechanism_study.py
scripts/run_screening_sequential_study.py
scripts/run_screening_drift_study.py
scripts/analyze_screening_studies.py
configs/region_screening_full.json
configs/region_screening_quick.json
tests/test_target_region_screening.py
tests/test_screening_research.py
results/region_screening/
```

建议提交信息：

```text
Add target-proposal source-region screening algorithm and research studies
```

## 五、科研判定标准

不应仅凭 Matching 方法平均值较好就判定方法成立。至少需要同时满足：

1. `Matching-Adaptive-Filter` 相对 `Target-Only` 的配对归一化 regret 改善置信区间主要位于 0 以上；
2. `Matching-Adaptive-Filter` 优于 `Random-Adaptive-Filter`；
3. `Matching-Adaptive-Filter` 优于或明显更安全于 `Wrong-Adaptive-Filter`；
4. 单步共享候选实验和闭环优化实验方向一致；
5. 漂移实验呈现可解释的有效性边界；
6. Adaptive 相比 Fixed 在大漂移或错误源情况下减少负迁移，同时不显著损失匹配源收益。

若样本外全局相关性不强，但 Top-10% 命中率和候选 regret 改善，研究故事应聚焦于：

> 源局部区域不是完整目标响应面的预测模型，而是目标候选集合中的优质尾部筛选器。
