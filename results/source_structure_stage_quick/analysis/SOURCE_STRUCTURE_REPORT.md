# Source local-structure extraction and validation report

This report is generated from frozen CSV artifacts. Statistical units are independent source-task instances, not individual candidate points.

## 1. Pre-specified primary tests

| Hypothesis | Pairs | Mean oriented advantage [95% bootstrap CI] | Holm p | Rank-biserial | Supported |
|---|---:|---:|---:|---:|---:|
| H1_Source_NDCG_Proposed_vs_Geometry | 8 | +0.0803 [+0.0078, +0.1834] | 0.125 | +0.250 | no |
| H2_Source_Local_Spearman_Proposed_vs_Geometry | 8 | +0.2297 [+0.0509, +0.5031] | 0.02344 | +1.000 | yes |
| H3_Source_NDCG_Proposed_vs_Permutation | 8 | +0.1940 [+0.0833, +0.3090] | 0.03125 | +0.750 | yes |
| H4_Matching_Target_NDCG_Proposed_vs_Geometry | 4 | +0.1275 [+0.0427, +0.2921] | 0.125 | +1.000 | no |
| H5_Recovery_Recall_Proposed_vs_TopObservations | 8 | +0.3333 [+0.1667, +0.5000] | 0.04688 | +1.000 | yes |
| H6_Recovery_CenterError_Proposed_vs_TopObservations | 8 | +0.1123 [+0.0584, +0.1687] | 0.02344 | +1.000 | yes |

## 2. Controlled recovery summary

| Method | Basin recall | Normalized center error | Shape error |
|---|---:|---:|---:|
| Proposed-Local-Structure | 1.0000 | 0.0156 | 0.1501 |
| Random-Centers | 0.2083 | 0.2520 | 0.1121 |
| Top-Observations | 0.6667 | 0.1279 | 0.1123 |

## 3. Held-out ranking summary

| Domain | Source scenario | Method | NDCG@top | Spearman | Precision@top | Top-1 regret |
|---|---|---|---:|---:|---:|---:|
| source | matching | Geometry-Only | 0.7187 | 0.6380 | 0.3107 | 0.4726 |
| source | matching | Global-Source-GP | 0.9044 | 0.7999 | 0.6607 | 0.0215 |
| source | matching | Label-Permutation | 0.6367 | 0.4436 | 0.2027 | 0.7457 |
| source | matching | Proposed-Local-Structure | 0.8443 | 0.7093 | 0.4424 | 0.2611 |
| source | wrong | Geometry-Only | 0.7873 | 0.6917 | 0.3421 | 0.3164 |
| source | wrong | Global-Source-GP | 0.8264 | 0.7123 | 0.6189 | 0.2517 |
| source | wrong | Label-Permutation | 0.6420 | 0.4340 | 0.1754 | 0.5091 |
| source | wrong | Proposed-Local-Structure | 0.8224 | 0.7167 | 0.3364 | 0.3654 |
| target | matching | Geometry-Only | 0.7321 | 0.6367 | 0.3710 | 0.5193 |
| target | matching | Global-Source-GP | 0.9105 | 0.7969 | 0.6379 | 0.0517 |
| target | matching | Label-Permutation | 0.6375 | 0.4467 | 0.1754 | 0.7631 |
| target | matching | Proposed-Local-Structure | 0.8596 | 0.7080 | 0.4812 | 0.1684 |
| target | wrong | Geometry-Only | 0.7179 | 0.5050 | 0.2215 | 0.5559 |
| target | wrong | Global-Source-GP | 0.7584 | 0.5468 | 0.2230 | 0.3949 |
| target | wrong | Label-Permutation | 0.6706 | 0.4827 | 0.2018 | 0.4385 |
| target | wrong | Proposed-Local-Structure | 0.7577 | 0.5200 | 0.2699 | 0.4456 |

## 4. Extraction diagnostics

Structures evaluated: 13; mean context size: 30.31; mean boundary fraction: 0.682; mean OOF Spearman: 0.427.

## 5. Completeness and failures

Controlled-recovery failures: 0; held-out validation failures: 0. Primary tests use complete paired instances only.

## 6. Data-derived interpretation

Supported primary claims:

- H2_Source_Local_Spearman_Proposed_vs_Geometry
- H3_Source_NDCG_Proposed_vs_Permutation
- H5_Recovery_Recall_Proposed_vs_TopObservations
- H6_Recovery_CenterError_Proposed_vs_TopObservations

Source fidelity and target transferability are reported separately. A structure may be extracted faithfully from its source task while remaining non-transferable to a mismatched target; this distinction must be preserved in any paper claim.
