# Gate-0 Oracle Local-Model Transfer Quick

这是一个受控、静态、二维 held-out quick study，用来回答局部模型迁移研究最前置的问题：

> 在 source region 对应和 target→source 局部坐标变换均已知时，source 局部模型是否在 Target-Only 与 Geometry-Prior 之上保留可观察的预测和候选排序信息？

本实验不学习 region matching 或 alignment，不生成 BO 候选，也不运行 online BO。它只决定局部模型迁移是否值得进入更严格的正式 Pilot。

## 初步结论

当前机器判定为：

- `label = promising_value_or_dual_head_transfer`
- `selected_head = value`

在四个预设 positive relations 上，相对 Geometry-Prior 的 8-seed 描述性均值为：

| head | Pairwise delta | NDCG delta | Top-1 regret delta |
|---|---:|---:|---:|
| Rank | +0.02223 | +0.05341 | +0.05945 |
| Value | **+0.10570** | **+0.08144** | **+0.07949** |
| Rank+Value | +0.07586 | +0.07380 | +0.07769 |

所有 delta 已统一方向，因此正值表示 transfer 优于 Geometry。Value 相对 Rank 的直接配对增量为：

- Pairwise：`+0.08347`，96 个 relation×shell 配对单元中 W/T/L=`94/0/2`；
- NDCG：`+0.02804`，W/T/L=`85/0/11`；
- Top-1 regret：`+0.02004`，W/T/L=`51/32/13`。

三个指标的 8 个 seed 均值都为正，seed win rate 均为 `1.0`。Dual 也通过 Rank-beyond 规则，但三个直接增量均小于 Value，因此当前 quick 选择 Value head。

负对照没有复制同等级收益：

- reversal 的非负校准精确回退到 Target-Only；
- independent expert 没有在任何 head 上复制 positive 规则；
- identity label permutation 也没有形成 condition-level replication；
- observed identity 相对 permutation 的三项 decision metrics 在 Rank、Value、Dual 三个 head 上均通过 source-specific 检查。

这说明**正确对应并对齐的 source 局部模型存在迁移 headroom，而且 source value landscape 比 rank-only 表达更值得优先研究**。但这是 quick、描述性结论，不是正式统计确认，也不是 learned alignment 或 online BO 的有效性证明。

## 冻结设计

- 维度：2D；seeds：`11, 23, 37, 53, 71, 89, 107, 131`。
- source 训练样本：128。
- target context：12 个点，由三个 shell 各 4 个方向组成。
- test shells：`0.35, 0.7, 1.0`，每 shell 128 个 held-out directions。
- relations：`identity`、`output_affine`、`scale_0.7`、`scale_1.5`、`rotate_45`、`roughness`、`reversal`、`independent_expert`。
- source-specific null：identity 下额外加入 `identity_label_permutation`。
- methods：`Target-Only`、`Geometry-Prior+Residual`、`Oracle-Rank+Residual`、`Oracle-Value+Residual`、`Oracle-Rank+Value+Residual`。

因此：

- `results.csv`：`8 × 9 × 3 × 5 = 1080` 行；
- `prediction_ledger.csv`：`1080 × 128 = 138240` 行；
- `source_expert_diagnostics.csv`：`8 × 9 × 3 = 216` 行。

Analyzer 要求上述完整笛卡尔积；缺失单个 cell、重复 cell、错误 seed、dimension、panel、shell 或 method 都会产生 decision failure。

## Oracle transfer 模型

Source expert 使用同一个 source design 训练：

1. Rank GP：拟合 larger-is-better rank quality，迁移前转换为 `source_cost = 1 - rank_quality`；
2. Value GP：拟合 source cost 的 robust-standardized raw response。

Target 侧使用低自由度非负校准和 residual GP：

$$
\widehat f_t(x)=b+\beta^\top\phi_s(T(x))+g_t(x),
\qquad \beta\ge 0.
$$

校准只读取 12 个 target context points。若 feature 为常数、拟合失败或所有系数为零，则精确回退到与 Target-Only 相同的 GP。

Oracle target→source 变换为：

- identity、output affine、roughness、reversal、independent：$T(z)=z$；
- scale：$T(z)=sz$；
- rotate 45°：$T(z)=zR(\pi/4)^\top$。

Target test truth 不参与 source expert 拟合、target calibration 或模型选择。

## 判定口径

Decision metrics 仅包括：

- Pairwise accuracy；
- NDCG@Top-10%；
- normalized Top-1 regret。

Standardized RMSE、Spearman、Precision@Top-10%、NLL 和区间覆盖率均为 secondary，不参与 head 选择。

统计单位是 seed。relation×shell 行先在 seed 内聚合，bootstrap 只重采样 8 个 seed，不把 128 个 candidate 当作独立重复。Value 或 Dual 只有在 Rank 先通过 Geometry-relative 规则，并且自身同时通过 Geometry-relative 与 Rank-beyond 规则后，才能替代 Rank。

完整规则和逐项数值见：

- [`results/oracle_local_model_transfer_quick/ORACLE_LOCAL_MODEL_TRANSFER_QUICK_CONCLUSION_CN.md`](results/oracle_local_model_transfer_quick/ORACLE_LOCAL_MODEL_TRANSFER_QUICK_CONCLUSION_CN.md)
- [`results/oracle_local_model_transfer_quick/analysis/decision.json`](results/oracle_local_model_transfer_quick/analysis/decision.json)

## 外推与解释限制

`scale_1.5` 不是纯粹的 source-observed-range transfer：target→source 查询会超出 source 训练盒 `[-1,1]^2`。

- context：每个 seed 的 12 个点中平均 `4.875` 个超界，平均比例 `40.625%`；
- shell `0.35`：`0/128` 超界；
- shell `0.7`：平均 `41/128` 超界，范围 `37–45`；
- shell `1.0`：`128/128` 全部超界。

因此 `scale_1.5` 混合了关系迁移与 source-GP extrapolation，不能解释为源观测范围内的直接证据。`rotate_45` 和 `roughness` 作为 boundary relations，不参与 positive-headroom 判定。

本 quick 仍有以下边界：

- 只有 2D 和 8 seeds；
- correspondence 与 alignment 是 oracle 已知的；
- 固定函数族、固定 GP kernel 和很小的 target context；
- 没有 candidate acquisition、evaluation budget 或 online BO trajectory；
- bootstrap interval 仅为描述性区间，不是确认性推断；
- manifest 是声明性 provenance，不是外部签名的对抗性信任锚。

## 运行、分析与审计

```bash
python scripts/run_oracle_local_model_transfer_quick.py \
  --config configs/oracle_local_model_transfer_quick.json \
  --output results/oracle_local_model_transfer_quick

python scripts/analyze_oracle_local_model_transfer_quick.py \
  --input results/oracle_local_model_transfer_quick

python scripts/audit_oracle_local_model_transfer_quick.py \
  --input results/oracle_local_model_transfer_quick
```

Runner 拒绝非空输出目录，不会覆盖已有证据。配置必须与代码中的 frozen canonical config 精确一致。

## 输出与可重建性

```text
results.csv                         # instance × method 指标、校准与 hash
prediction_ledger.csv                # 128 个 candidate 的 truth/prediction/std
source_expert_diagnostics.csv        # rank/value 一致性及 source-query OOB
failures.csv                         # seed 级异常；当前为 0 行
config.json                          # 实际冻结配置
reproducibility_inputs.npz           # 无损设计、真值、变换查询、置换与 seed lineage
run_manifest.json                    # artifact、依赖源码和环境版本 hash
analysis/summary.csv
analysis/contrasts.csv
analysis/decision.json
analysis/analysis_manifest.json      # analyzer、分析输入与输出 hash
analysis/*.png
ORACLE_LOCAL_MODEL_TRANSFER_QUICK_CONCLUSION_CN.md
AUDIT.json
FULL_RUN_AUDIT.md
```

当前审计结果为 `ok=true`，共 92 项检查、0 项失败。审计从 `reproducibility_inputs.npz` 独立重建设计、真值、target→source 查询、permutation、model seed lineage、诊断与数据 hash，并从 candidate ledger 重算 8640 个预测指标。

旧的 provenance 不完整结果已原样保留在：

`results/oracle_local_model_transfer_quick_superseded_pre_provenance_20260901/`

它只用于审计历史，不应作为当前 quick 结论来源。
