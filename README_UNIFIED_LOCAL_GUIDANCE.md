# Unified Local Guidance Study

本阶段补上此前缺失的端到端连接：将 `SourceLocalStructure` 中的区域几何和局部 GP/RF 排序代理接入 Target-GP 候选选择，并在相同候选与相同目标评测预算下直接检验局部模型是否超越几何。

本阶段**不做区域匹配或输入对齐**，只在当前 benchmark 的共享坐标假设下研究候选级 guidance。

## 方法

四个主方法共享同一个 `SourceLocalStructureLibrary`、Target-GP 配置和 guidance 参数，只改变 score mode：

- `Target-Only`：仅按 Target GP 的 EI；
- `Geometry-Only`：区域质量与 Mahalanobis membership；
- `Local-Rank-No-Reliability`：几何加局部 rank surrogate；
- `Local-Rank+Reliability`：几何、局部 rank surrogate和结构级 OOF reliability。

`Reversed-Local-Rank` 保留区域几何，只反转局部模型 quality factor，是机制安全压力测试，不是自然任务分布，也不进入 sequential 主比较。

统一评分入口见 [`local_structure_guidance.py`](src/region_guided_reranking_study/local_structure_guidance.py)，冻结协议见 [`PROTOCOL_UNIFIED_LOCAL_GUIDANCE.md`](PROTOCOL_UNIFIED_LOCAL_GUIDANCE.md)。

## 冻结规模

| slice | problems | dimensions | seeds | source | raw pool | proposal | budget | bootstrap |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| quick | GMM, Ackley | 2D | 2 | 80 | 300 | 40 | 5 | 500 |
| full | GMM, Rastrigin, Lunacek, Ackley | 2D, 5D | 8 | 160 | 1500 | 100 | 20 | 5000 |

两者固定 EI 和 `n_init=2d+2`。Full 参数在查看 quick 结果前冻结，没有依据 quick 效果调参。

## 运行

```bash
# Quick：代码与审计验证，不用于科学结论
python scripts/run_unified_local_guidance_study.py \
  --config configs/unified_local_guidance_quick.json \
  --output results/unified_local_guidance_quick

python scripts/analyze_unified_local_guidance_study.py \
  --input results/unified_local_guidance_quick \
  --config results/unified_local_guidance_quick/config.json \
  --manifest results/unified_local_guidance_quick/run_manifest.json \
  --output results/unified_local_guidance_quick/analysis

python scripts/audit_unified_local_guidance_study.py \
  --input results/unified_local_guidance_quick \
  --config results/unified_local_guidance_quick/config.json \
  --manifest results/unified_local_guidance_quick/run_manifest.json
```

```bash
# Full：正式 confirmatory study
python scripts/run_unified_local_guidance_study.py \
  --config configs/unified_local_guidance_full.json \
  --output results/unified_local_guidance_full

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

Runner 拒绝覆盖非空输出目录，也不调用 analyzer 或 audit。

## 正式结果

Full study 包含：

- 64 个独立 `(problem, dim, seed)` instances；
- 320 条机制结果；
- 6,400 条共享候选 panel 记录；
- 256 条 sequential summaries；
- 5,120 次新增 target evaluations；
- 81 个 source local structures；
- 0 failures。

五项预注册主检验全部不支持：

| 主检验 | Effect [95% CI] | Holm p |
|---|---:|---:|
| 机制 regret：Local+Reliability vs Geometry | -0.00630 [-0.03279, +0.02071] | 1.0000 |
| 机制 Top-10% hit：Local+Reliability vs Geometry | +0.03125 [-0.03125, +0.09375] | 0.7933 |
| Sequential final regret：Local+Reliability vs Geometry | -0.03466 [-0.07497, +0.00713] | 1.0000 |
| Sequential regret AUC：Local+Reliability vs Geometry | -0.14844 [-0.78745, +0.44526] | 1.0000 |
| Reliability 增量 | 0.00000 [0.00000, 0.00000] | 1.0000 |

所有 effect 定向为正值有利于新方法。

方法均值显示 Geometry-Only 的机制 normalized regret 为 0.44577，Local-Rank+Reliability 为 0.45207；sequential final regret 分别为 0.40487 和 0.43953。局部模型没有在几何之上形成稳定增量。

Reliability on/off 在 64 个共享候选机制实例中选择完全相同；sequential 中少量 step 的候选或后续 proposal 不同，但 incumbent normalized-regret 轨迹、final regret 和 AUC 全部相同。因此当前结构级标量 reliability 偶尔影响候选，却没有产生冻结性能指标增量。Reversed local rank 明显弱于 Geometry-Only，说明模型分支确实进入决策，而且错误局部排序能够造成伤害。

正式决策：

> **保留 Geometry-Only 候选引导；不推进当前 `structure_score` 形式的 model-aware online guidance。Source Local Structure 仍作为 source-side 抽取成果保留。**

详细证据：

- [正式决策](results/unified_local_guidance_full/UNIFIED_LOCAL_GUIDANCE_DECISION_CN.md)
- [统计报告](results/unified_local_guidance_full/analysis/UNIFIED_LOCAL_GUIDANCE_REPORT_CN.md)
- [主检验](results/unified_local_guidance_full/analysis/PRIMARY_TESTS.csv)
- [完整审计](results/unified_local_guidance_full/FULL_RUN_AUDIT.md)
- [机器审计](results/unified_local_guidance_full/AUDIT.json)

## 解释边界

该结论只针对：

- 当前共享坐标 benchmark；
- 固定 source weight、nominee union 和 rank fusion；
- source-side rank surrogate；
- 无区域匹配、无输入对齐、无输出校准；
- 2D/5D、无 target observation noise 的冻结切片。

它不能用于否定 oracle alignment、value/curvature head、candidate-conditioned trust 或其他局部景观迁移机制；这些方向需要新协议和独立 seeds，不能在本 full 结果上继续 post-hoc 调参。
