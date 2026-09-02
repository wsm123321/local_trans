# 源局部区域引导的目标候选重排序研究

本项目研究如何在不污染目标代理模型的前提下，利用历史源任务中的优质局部区域、局部排序结构和迁移证据，引导目标任务的候选筛选与重排序。

当前主原则是：

- **Target GP 独立提议候选**：源数据不进入目标 GP，也不负责生成目标候选。
- **Source knowledge 只在候选集内提供建议**：通过区域支持、局部排序或可信度门控影响候选选择。
- **Target-Only 是零迁移基线和精确回退路径**：源证据不足、无判别力或被拒绝时，恢复 Target-Only 决策。
- **抽取有效性、迁移有效性和优化收益分开验证**：source fidelity 不自动等价于 target transferability，也不自动等价于 online BO 收益。

项目已经从早期固定软重排扩展为多个带配置、协议、分析和审计的研究阶段。根目录下的旧三阶段实验仍保留为历史基线，但不代表当前完整研究流程。

---

## 1. 当前研究状态

| 阶段 | 研究问题 | 结果级别 | 当前结论 |
| :--- | :--- | :--- | :--- |
| Legacy candidate reranking | 源区域软重排是否在共享候选池中提供信号 | Legacy / historical | 单步机制中存在正面信号，但闭环收益较弱；保留为历史基线 |
| Target-proposal / source-region screening | 目标先提议、源区域后筛选能否改善候选质量和闭环优化 | **Formal / full** | Matching fixed/soft 有收益；全局 adaptive 规则与稳定 drift boundary 未获统计支持 |
| Source local structure | 能否仅从 source observations 恢复高质量局部结构 | **Formal / full** | 六项预注册主检验均支持；中心与局部排序结构证据较强，但 covariance shape recovery 不优于基线 |
| Unified local guidance | 完整局部 rank surrogate 是否在区域几何之上改善候选决策和等预算 BO | **Formal / full** | 五项主检验均不支持；Geometry-Only 描述性均值较好，但局部 rank 没有增量，结构级 reliability 未改变冻结性能指标；**不推进当前 model-aware online guidance** |
| Oracle local-model transfer Gate-0 | 已知 region 对应和输入对齐时，source rank/value landscape 是否仍有迁移 headroom | Exploratory / quick | Rank、Value、Dual 均在 Geometry 之上显示描述性增量；Value 在三项 decision metrics 上直接优于 Rank，负对照未复制收益；选择 **Value head** 进入正式 Pilot 设计，但不等同于 learned alignment 或 online BO 有效 |
| Local-surrogate transfer Pilot | oracle 区域对应下，校准后的源局部 surrogate 是否优于 target-only local GP | **Formal / full** | 二维中标准化预测误差有小幅改善，但排序 NDCG 无改善，门控覆盖率和 accepted risk 不足；**不进入 online BO** |
| Local transferability map | 哪类任务关系允许局部模型迁移 | Exploratory / quick | 仅二维描述性结果；高局部排序一致性关系较有希望，anisotropy 等失配仍不可靠 |
| ARISE | 能否用区域级超额改进后验区分可信利用、拒绝和主动探测 | Exploratory / quick | 当前只有 quick artifact，识别和优化收益不足以支持正式结论；仓库没有已保存的 full artifact |
| Disagreement-conditioned trust v1-R | 冲突条件 correction 是否优于简单 Local Spearman | **Formal / full** | AUPRC 和 regret 优势均不确定，accepted harm 仍高；正式决定为 `do_not_advance_complex_disagreement_trust` |

权威结果入口：

- Screening：[研究报告](results/region_screening/analysis/SCREENING_STUDY_REPORT.md) · [完整审计](results/region_screening/FULL_RUN_AUDIT.md)
- Source Structure：[研究报告](results/source_structure_stage/analysis/SOURCE_STRUCTURE_REPORT.md) · [完整审计](results/source_structure_stage/FULL_RUN_AUDIT.md)
- Unified Local Guidance：[正式决策](results/unified_local_guidance_full/UNIFIED_LOCAL_GUIDANCE_DECISION_CN.md) · [统计报告](results/unified_local_guidance_full/analysis/UNIFIED_LOCAL_GUIDANCE_REPORT_CN.md) · [完整审计](results/unified_local_guidance_full/FULL_RUN_AUDIT.md)
- Oracle Local-Model Transfer Gate-0：[阶段说明](README_ORACLE_LOCAL_MODEL_TRANSFER_QUICK.md) · [探索性结论](results/oracle_local_model_transfer_quick/ORACLE_LOCAL_MODEL_TRANSFER_QUICK_CONCLUSION_CN.md) · [机器判定](results/oracle_local_model_transfer_quick/analysis/decision.json) · [完整审计](results/oracle_local_model_transfer_quick/FULL_RUN_AUDIT.md)
- Local Surrogate Pilot：[中文决策](results/local_surrogate_transfer_pilot/LOCAL_SURROGATE_TRANSFER_DECISION_CN.md) · [完整审计](results/local_surrogate_transfer_pilot/FULL_RUN_AUDIT.md)
- Transferability Map：[探索性结论](results/local_transferability_map_quick/LOCAL_TRANSFERABILITY_MAP_CONCLUSION_CN.md)
- ARISE quick：[研究报告](results/arise_stage_quick/analysis/ARISE_STUDY_REPORT.md)
- Disagreement Trust：[最终决策](results/disagreement_trust_validation/DISAGREEMENT_TRUST_FINAL_DECISION_CN.md) · [分析报告](results/disagreement_trust_validation/analysis/DISAGREEMENT_TRUST_VALIDATION_REPORT_CN.md) · [机器判定](results/disagreement_trust_validation/analysis/DECISION.json)

---

## 2. 核心机制

### 2.1 源区域库

从历史源任务的优质采样点中提取 $K$ 个区域：

$$
\mathcal{R}_s = \{R_1, \dots, R_K\}, \qquad
R_k = (\mu_k, \Sigma_k, q_k, n_k),
$$

其中：

- $\mu_k \in \mathbb{R}^d$：区域中心；
- $\Sigma_k \in \mathbb{R}^{d \times d}$：区域协方差；
- $q_k \in [0,1]$：区域质量；
- $n_k \in \mathbb{N}^+$：区域样本数或跨源出现频次。

候选点 $x$ 的源区域支持为：

$$
r_s(x) = \max_{k=1,\dots,K}
q_k \exp\left(
-\frac{1}{2}(x-\mu_k)^\top \Sigma_k^{-1}(x-\mu_k)
\right).
$$

### 2.2 候选级软重排与回退

对 Target GP 提出的候选集 $C_t$，将目标采集值与源区域支持分别进行并列感知的秩归一化：

$$
J_t(x) = \widetilde{\alpha}_t(x) + \lambda_t\widetilde{r}_s(x),
\qquad x \in C_t.
$$

若源支持在候选池中没有足够判别力，例如

$$
\operatorname{Var}(r_s) < \epsilon,
$$

则令 $\lambda_t=0$，恢复 Target-Only 排序。后续 screening、local surrogate、ARISE 和 disagreement trust 阶段在此思想上加入了兼容性估计、校准、区域后验或更严格的 exact fallback。

### 2.3 Source local structure

Source Structure 阶段不直接研究 target transfer，而是先验证能否仅从 source observations 抽取：

$$
(\mu_k,\Sigma_k,h_k,\omega_k),
$$

其中 $h_k$ 是区域内的局部排序 surrogate，$\omega_k$ 是由 out-of-fold 诊断支持的结构可靠度。主要流程为：

1. 对 source objective 做 rank normalization；
2. 抽取 elite observations；
3. 用 full-covariance GMM 和 BIC 提取区域；
4. 加入 non-elite boundary context；
5. 在区域坐标中训练局部 GP 或 Random Forest；
6. 用独立 source test 和 OOF diagnostics 验证局部排序 fidelity。

实现见 [`source_local_structure.py`](src/region_guided_reranking_study/source_local_structure.py)，冻结协议见 [`PROTOCOL_SOURCE_STRUCTURE.md`](PROTOCOL_SOURCE_STRUCTURE.md)。

---

## 3. Legacy 六组对照

以下比较器用于早期共享候选池机制研究，仍是理解后续工作的基础，但不是当前全部研究阶段：

| 编号 | 方法 | 机制 |
| :--- | :--- | :--- |
| M1 | **Target-Only** | 仅按目标采集值排序，是零迁移基线 |
| M2 | **Source-Region** | 目标采集值与 matching source-region support 软融合 |
| M3 | **Random-Region** | 保留真实区域的协方差谱、体积、质量和频次，仅随机平移中心 |
| M4 | **Wrong-Source** | 使用失配或对抗源区域 |
| M5 | **Oracle-Target-Region** | 使用真实目标最优盆地先验，作为理论上界 |
| M6 | **Hard-Filter** | 只保留位于区域卡方置信椭球中的候选 |

早期实验使用 `numpy.random.SeedSequence` 隔离随机流，并尽量共享目标初始样本、Target GP 和候选池。旧结果见 [`results/VERIFICATION_REPORT.md`](results/VERIFICATION_REPORT.md)；它是历史报告，不应覆盖后续冻结协议和正式阶段结果。

---

## 4. 环境与安装

项目元数据当前声明 Python 3.9+，依赖包括 NumPy、SciPy、scikit-learn、pandas、Matplotlib 和 pytest。

> 注意：部分较新模块使用 `X | Y` 联合类型语法。为避免 Python 3.9 的解析兼容性问题，当前实际运行建议使用 **Python 3.10+**；项目版本下限仍需在后续代码维护中统一。

```bash
cd region_guided_reranking_study
python -m pip install -e .
python -m pytest -q
```

配置文件位于 [`configs/`](configs/)。正式研究与 quick validation 必须使用不同输出目录，不要把 quick、full 或历史失败尝试合并分析。

---

## 5. 复现入口

### 5.1 Region Screening

一键入口会依次运行共享候选机制、等预算 sequential study、drift study 和分析。

```bash
# Quick：代码路径验证
python scripts/run_all_screening_studies.py \
  --config configs/region_screening_quick.json \
  --output-dir results/region_screening_quick

# Full：正式结果
python scripts/run_all_screening_studies.py \
  --config configs/region_screening_full.json \
  --output-dir results/region_screening
```

详细设计见 [`README_NEXT_STAGE.md`](README_NEXT_STAGE.md)。

### 5.2 Source Structure

```bash
# Quick：代码路径验证
python scripts/run_all_source_structure_studies.py \
  --config configs/source_structure_quick.json \
  --output results/source_structure_stage_quick

# Full：正式 recovery + held-out validation + analysis
python scripts/run_all_source_structure_studies.py \
  --config configs/source_structure_full.json \
  --output results/source_structure_stage
```

拆分分析：

```bash
python scripts/analyze_source_structure_study.py \
  --input results/source_structure_stage \
  --output results/source_structure_stage/analysis
```

阶段说明见 [`README_SOURCE_STRUCTURE_STAGE.md`](README_SOURCE_STRUCTURE_STAGE.md)，冻结协议见 [`PROTOCOL_SOURCE_STRUCTURE.md`](PROTOCOL_SOURCE_STRUCTURE.md)。

### 5.3 Unified Local Guidance

该阶段将同一个 `SourceLocalStructureLibrary` 接入 Target-GP 候选级闭环，直接比较 Target-Only、Geometry-Only、局部 rank surrogate 以及 reliability 增量。

```bash
# Quick：代码与审计验证
python scripts/run_unified_local_guidance_study.py \
  --config configs/unified_local_guidance_quick.json \
  --output results/unified_local_guidance_quick

# Full：正式机制 + 等预算 sequential study
python scripts/run_unified_local_guidance_study.py \
  --config configs/unified_local_guidance_full.json \
  --output results/unified_local_guidance_full

# 正式分析与审计
python scripts/analyze_unified_local_guidance_study.py \
  --input results/unified_local_guidance_full \
  --config results/unified_local_guidance_full/config.json \
  --manifest results/unified_local_guidance_full/run_manifest.json \
  --output results/unified_local_guidance_full/analysis

python scripts/audit_unified_local_guidance_study.py \
  --input results/unified_local_guidance_full \
  --config results/unified_local_guidance_full/config.json \
  --manifest results/unified_local_guidance_full/run_manifest.json
```

Full 的五项主检验均不支持局部 rank 超越 Geometry-Only；当前决策是保留几何候选指导，不推进该 `structure_score` online 路线。详见 [`README_UNIFIED_LOCAL_GUIDANCE.md`](README_UNIFIED_LOCAL_GUIDANCE.md) 和 [`PROTOCOL_UNIFIED_LOCAL_GUIDANCE.md`](PROTOCOL_UNIFIED_LOCAL_GUIDANCE.md)。

### 5.4 Oracle Local-Model Transfer Gate-0

该阶段隔离 region extraction、matching 和 alignment estimation，在已知 target→source chart transform 下分别迁移 Rank、Value 和 Rank+Value heads。

```bash
python scripts/run_oracle_local_model_transfer_quick.py \
  --config configs/oracle_local_model_transfer_quick.json \
  --output results/oracle_local_model_transfer_quick

python scripts/analyze_oracle_local_model_transfer_quick.py \
  --input results/oracle_local_model_transfer_quick

python scripts/audit_oracle_local_model_transfer_quick.py \
  --input results/oracle_local_model_transfer_quick
```

当前 quick 判定为 `promising_value_or_dual_head_transfer`，选择 `value`：Value 相对 Geometry 的 Pairwise、NDCG、Top-1 regret 方向化增量分别为 `+0.10570`、`+0.08144`、`+0.07949`；相对 Rank 的直接增量分别为 `+0.08347`、`+0.02804`、`+0.02004`。这是 2D、8-seed、oracle alignment 下的描述性 headroom，不是正式确认或 online BO 证据。`scale_1.5` 包含 source-GP 外推，必须单独解释。详见 [`README_ORACLE_LOCAL_MODEL_TRANSFER_QUICK.md`](README_ORACLE_LOCAL_MODEL_TRANSFER_QUICK.md)。

### 5.5 Local-Surrogate Transfer Pilot

一键入口会运行 Pilot、统计分析和审计。

```bash
# Quick：代码路径验证
python scripts/run_all_local_surrogate_transfer_pilot.py \
  --config configs/local_surrogate_transfer_quick.json \
  --output results/local_surrogate_transfer_pilot_quick

# Full：正式 confirmatory Pilot
python scripts/run_all_local_surrogate_transfer_pilot.py \
  --config configs/local_surrogate_transfer_full.json \
  --output results/local_surrogate_transfer_pilot
```

本阶段使用 oracle region correspondence 检验局部 surrogate 的预测和排序价值，不包含 unknown alignment、候选生成或 online BO。详见 [`README_LOCAL_SURROGATE_TRANSFER_STAGE.md`](README_LOCAL_SURROGATE_TRANSFER_STAGE.md) 和 [`PROTOCOL_LOCAL_SURROGATE_TRANSFER_PILOT.md`](PROTOCOL_LOCAL_SURROGATE_TRANSFER_PILOT.md)。

### 5.6 Local Transferability Map

该阶段只有二维探索性 quick 配置，没有正式 full study：

```bash
python scripts/run_local_transferability_map_quick.py \
  --config configs/local_transferability_map_quick.json \
  --output results/local_transferability_map_quick
```

### 5.7 ARISE

```bash
# 当前仓库中已有的 quick exploratory 路径
python scripts/run_all_arise_studies.py \
  --config configs/arise_quick.json \
  --output results/arise_stage_quick

# Full 设计入口；仓库当前没有已保存的 full artifact
python scripts/run_all_arise_studies.py \
  --config configs/arise_full.json \
  --output results/arise_stage
```

ARISE 的方法、假设和计划消融见 [`README_ARISE_STAGE.md`](README_ARISE_STAGE.md)。不得将现有 quick 结果表述为 full confirmation。

### 5.8 Disagreement Trust v1-R

```bash
# Quick：修订协议的代码验证
python scripts/run_disagreement_trust_validation.py \
  --config configs/disagreement_trust_validation_quick.json \
  --output results/disagreement_trust_validation_quick

# Full：正式 held-out controlled validation
python scripts/run_disagreement_trust_validation.py \
  --config configs/disagreement_trust_validation_full.json \
  --output results/disagreement_trust_validation
```

对已有正式产物重新执行分析：

```bash
python scripts/analyze_disagreement_trust_validation.py \
  --input results/disagreement_trust_validation \
  --config configs/disagreement_trust_validation_full.json \
  --output results/disagreement_trust_validation/analysis
```

冻结协议见 [`PROTOCOL_DISAGREEMENT_TRUST_VALIDATION.md`](PROTOCOL_DISAGREEMENT_TRUST_VALIDATION.md)。正式结论是不推进更复杂的 disagreement correction，而不是证明该机制安全。

### 5.9 Legacy historical workflow

以下脚本保留用于复查最早的机制、漂移和闭环 BO 结果：

```bash
python scripts/run_mechanism_experiment.py
python scripts/run_drift_curve_experiment.py
python scripts/run_sequential_bo.py
python scripts/plot_and_report.py
```

这些入口没有 quick/full 配置接口，主要规模参数硬编码在脚本中，不是当前正式研究的推荐入口。

---

## 6. 结果状态与证据规则

### Formal / full

只有以下目录代表当前可用于正式结论的 full artifacts：

- [`results/region_screening/`](results/region_screening/)
- [`results/source_structure_stage/`](results/source_structure_stage/)
- [`results/unified_local_guidance_full/`](results/unified_local_guidance_full/)
- [`results/local_surrogate_transfer_pilot/`](results/local_surrogate_transfer_pilot/)
- [`results/disagreement_trust_validation/`](results/disagreement_trust_validation/)

### Quick / code validation / exploratory

以下目录用于代码路径验证或描述性探索，不能与 full 数据合并：

- `results/region_screening_quick/`
- `results/source_structure_stage_quick/`
- `results/unified_local_guidance_quick/`
- `results/oracle_local_model_transfer_quick/`
- `results/local_surrogate_transfer_pilot_quick/`
- `results/disagreement_trust_validation_quick/`
- `results/local_transferability_map_quick/`
- `results/arise_stage_quick/`

其中 Transferability Map 和 ARISE 当前尤其应标为 **exploratory**。

### Rejected / failed / superseded

名称中带有 `attempt*_rejected`、`attempt*_failed` 或 `attempt*_superseded` 的目录只用于审计历史。对应的 `REJECTION.md`、`FAILURE.md` 或 `SUPERSEDED.md` 解释了协议、实现或证据单位问题；这些数据不得进入正式统计结论。

### Legacy

`results/` 根目录下的早期 CSV、JSON、PNG 和 `VERIFICATION_REPORT.md` 属于旧三阶段历史结果。当前正式证据应优先引用各阶段子目录中的 protocol、manifest、analysis、audit 和 decision。

---

## 7. 目录结构

```text
region_guided_reranking_study/
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/                              # 各阶段 quick/full 冻结配置
├── src/region_guided_reranking_study/
│   ├── landscapes.py                     # Benchmark landscapes 与任务族
│   ├── source_regions.py                 # 基础源区域抽取与区域库
│   ├── surrogate_and_candidates.py       # Target GP、采集函数与候选池
│   ├── rerankers.py                      # Legacy 六比较器
│   ├── sequential_bo.py                  # Legacy sequential BO
│   ├── local_region_transfer.py          # 候选级局部区域迁移优化器
│   ├── source_local_structure.py         # Source local structure 抽取
│   ├── source_structure_research.py      # Recovery/validation 研究工具
│   ├── local_structure_guidance.py       # 局部结构候选级统一评分与回退
│   ├── target_region_screening.py        # Target proposal + source screening
│   ├── screening_research.py             # Screening 统计与对照工具
│   ├── local_surrogate_transfer.py       # 校准、残差 GP 与 gate
│   ├── local_surrogate_transfer_research.py
│   ├── oracle_local_model_transfer.py    # Gate-0 oracle Rank/Value 迁移原语
│   ├── arise_transfer.py                 # ARISE 区域证据模型与 BO
│   └── disagreement_trust.py             # Disagreement-conditioned trust
├── scripts/                              # Runner、analyzer、audit 与绘图入口
├── tests/                                # 随机隔离、比较器和各阶段单元测试
├── results/                              # Formal、quick、legacy 与审计归档
├── README_NEXT_STAGE.md                  # Screening 阶段说明
├── README_SOURCE_STRUCTURE_STAGE.md
├── README_UNIFIED_LOCAL_GUIDANCE.md
├── README_ORACLE_LOCAL_MODEL_TRANSFER_QUICK.md
├── README_LOCAL_SURROGATE_TRANSFER_STAGE.md
├── README_ARISE_STAGE.md
├── PROTOCOL_SOURCE_STRUCTURE.md
├── PROTOCOL_UNIFIED_LOCAL_GUIDANCE.md
├── PROTOCOL_LOCAL_SURROGATE_TRANSFER_PILOT.md
├── PROTOCOL_DISAGREEMENT_TRUST_VALIDATION.md
├── MANIFEST_NEXT_STAGE.md
├── MANIFEST_SOURCE_STRUCTURE_STAGE.md
├── MANIFEST_ARISE_STAGE.md
├── LOCAL_VALIDATION.md
└── SHA256SUMS.txt
```

---

## 8. 协议、审计与 provenance

建议按以下证据链阅读每个正式阶段：

1. 阶段 README：研究问题、方法和运行方式；
2. Frozen protocol：数据隔离、主假设、统计单位和推进标准；
3. JSON config：实际运行规模与参数；
4. Run manifest：运行环境、配置摘要、输入输出和生成时 Git 状态；
5. Raw CSV/JSON：候选级、实例级或轨迹级结果；
6. Analysis report：聚合指标、cluster bootstrap、配对检验和多重校正；
7. Audit / decision：完整性检查和最终研究判定。

历史 artifact 的 manifest 可能记录不同于当前 checkout 的 Git commit。这是生成时 provenance，不是错误；引用结果时应使用 manifest 中的 **artifact generation commit**，不能笼统声称所有结果由当前 HEAD 生成。

[`SHA256SUMS.txt`](SHA256SUMS.txt) 目前只覆盖特定阶段交付文件，不是整个仓库的完整 checksum 清单。[`LOCAL_VALIDATION.md`](LOCAL_VALIDATION.md) 记录本地工程验证，也不等同于科学结论。

---

## 9. 结论解释边界

当前结果支持的是一组受控、分层的结论，而不是通用的无负迁移保证：

- Source local structure recovery 有较强证据，但 **source fidelity 不等于 target transferability**。
- Unified Local Guidance 的五项正式主检验全部不支持：当前局部 rank surrogate 没有在 Geometry-Only 之上改善共享候选决策或等预算 BO；结构级 reliability 偶尔改变 sequential 候选，但没有改变任何冻结性能主指标，因此不推进当前 `structure_score` online 路线。
- Matching fixed/soft screening 和本次 Geometry-Only 的描述性均值都显示区域几何有信号，但 adaptive trust、正式 Geometry-vs-Target 确认和 drift boundary 仍需区分处理。
- Oracle Local-Model Transfer Gate-0 表明：在 region 对应和输入对齐已知时，Rank、Value、Dual 都存在超越 Geometry 的描述性 headroom，Value 又在三项 decision metrics 上直接优于 Rank；source-label permutation、reversal 和 independent expert 没有复制同等级收益。该结果支持正式研究 **Value-head 局部景观迁移**，但只有 2D、8 seeds，且 `scale_1.5` 混有 source-GP 外推，不能外推到 learned alignment 或 online BO。
- Local-surrogate Pilot 的正面结果主要限于二维、oracle 区域对应和预测误差；没有排序收益，也没有 online BO 证据。
- Transferability Map 与 ARISE 只有 quick/exploratory 结果。
- Disagreement correction 没有可靠胜过简单 Local Spearman；两者相对 Target-Only 的部分收益不能归因于 correction 独有。
- Wrong-source 任务级失配不一定构成严格的 region-level wrong-local-model 对照。
- 当前结果不能声称 universal transfer、high-dimensional robustness、general no-harm、evaluation-budget saving 或部署安全性。

基于现有证据，Geometry-Only 仍是候选级低自由度基线，Target-Only 仍是必须保留的精确回退。局部模型研究则应进入一个独立的正式 Gate-0 Pilot：以 Value head 为主、Dual 为消融，先确认 oracle alignment 下的 source-specific 增量与外推边界，再研究 matching/alignment 的学习；不应继续调当前未经对齐的 `structure_score` rank-fusion 参数，也不应直接跳到 online BO。
