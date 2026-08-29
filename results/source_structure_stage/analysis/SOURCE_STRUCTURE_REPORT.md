# Source local-structure extraction and validation report

This report is generated from frozen CSV artifacts. Statistical units are independent source-task instances, not individual candidate points.

## 1. Pre-specified primary tests

| Hypothesis | Pairs | Mean oriented advantage [95% bootstrap CI] | Holm p | Rank-biserial | Supported |
|---|---:|---:|---:|---:|---:|
| H1_Source_NDCG_Proposed_vs_Geometry | 128 | +0.0261 [+0.0192, +0.0334] | 7.776e-16 | +0.734 | yes |
| H2_Source_Local_Spearman_Proposed_vs_Geometry | 128 | +0.0809 [+0.0639, +0.0989] | 7.176e-19 | +0.750 | yes |
| H3_Source_NDCG_Proposed_vs_Permutation | 128 | +0.1331 [+0.1134, +0.1543] | 2.854e-22 | +1.000 | yes |
| H4_Matching_Target_NDCG_Proposed_vs_Geometry | 64 | +0.0213 [+0.0116, +0.0319] | 6.162e-06 | +0.531 | yes |
| H5_Recovery_Recall_Proposed_vs_TopObservations | 144 | +0.0694 [+0.0139, +0.1250] | 0.005181 | +0.225 | yes |
| H6_Recovery_CenterError_Proposed_vs_TopObservations | 144 | +0.1064 [+0.0917, +0.1212] | 2.206e-20 | +0.764 | yes |

## 2. Controlled recovery summary

| Method | Basin recall | Normalized center error | Shape error |
|---|---:|---:|---:|
| Proposed-Local-Structure | 0.3889 | 0.0957 | 0.1769 |
| Random-Centers | 0.1273 | 0.2703 | 0.1054 |
| Top-Observations | 0.3194 | 0.2021 | 0.1054 |

## 3. Held-out ranking summary

| Domain | Source scenario | Method | NDCG@top | Spearman | Precision@top | Top-1 regret |
|---|---|---|---:|---:|---:|---:|
| source | matching | Geometry-Only | 0.8201 | 0.6723 | 0.4795 | 0.3605 |
| source | matching | Global-Source-GP | 0.8429 | 0.6541 | 0.5391 | 0.2890 |
| source | matching | Label-Permutation | 0.7129 | 0.4829 | 0.2913 | 0.5471 |
| source | matching | Proposed-Local-Structure | 0.8457 | 0.6975 | 0.5121 | 0.2646 |
| source | wrong | Geometry-Only | 0.8186 | 0.6782 | 0.4793 | 0.4195 |
| source | wrong | Global-Source-GP | 0.8452 | 0.6480 | 0.5346 | 0.2748 |
| source | wrong | Label-Permutation | 0.7116 | 0.4683 | 0.2690 | 0.5799 |
| source | wrong | Proposed-Local-Structure | 0.8451 | 0.7036 | 0.5115 | 0.2995 |
| target | matching | Geometry-Only | 0.8168 | 0.6705 | 0.4735 | 0.3786 |
| target | matching | Global-Source-GP | 0.8277 | 0.6254 | 0.5122 | 0.2981 |
| target | matching | Label-Permutation | 0.7155 | 0.4867 | 0.2947 | 0.5403 |
| target | matching | Proposed-Local-Structure | 0.8381 | 0.6927 | 0.5033 | 0.3071 |
| target | wrong | Geometry-Only | 0.7403 | 0.4757 | 0.2980 | 0.5395 |
| target | wrong | Global-Source-GP | 0.6958 | 0.3502 | 0.2433 | 0.5448 |
| target | wrong | Label-Permutation | 0.7275 | 0.4862 | 0.2953 | 0.5100 |
| target | wrong | Proposed-Local-Structure | 0.7484 | 0.4841 | 0.3048 | 0.5472 |

## 4. Extraction diagnostics

Structures evaluated: 157; mean context size: 78.38; mean boundary fraction: 0.669; mean OOF Spearman: 0.579.

## 5. Completeness and failures

Controlled-recovery failures: 0; held-out validation failures: 0. Primary tests use complete paired instances only.

## 6. Data-derived interpretation

Supported primary claims:

- H1_Source_NDCG_Proposed_vs_Geometry
- H2_Source_Local_Spearman_Proposed_vs_Geometry
- H3_Source_NDCG_Proposed_vs_Permutation
- H4_Matching_Target_NDCG_Proposed_vs_Geometry
- H5_Recovery_Recall_Proposed_vs_TopObservations
- H6_Recovery_CenterError_Proposed_vs_TopObservations

Source fidelity and target transferability are reported separately. A structure may be extracted faithfully from its source task while remaining non-transferable to a mismatched target; this distinction must be preserved in any paper claim.
