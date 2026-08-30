# Source Local-Surrogate Transfer Pilot v1 Report

This report is generated from frozen CSV artifacts. The statistical unit is one `(problem, dimension, seed)` task instance; candidate-panel points are not treated as replicates.

## 1. Scope

This is a controlled 2D static held-out model-transfer Pilot under oracle region correspondence and a frozen isotropic local chart. It does not test unknown alignment, online BO, or a general no-harm guarantee.

## 2. Pre-specified primary tests

| Hypothesis | Pairs | Mean oriented advantage [95% bootstrap CI] | Holm p | Rank-biserial | Supported |
|---|---:|---:|---:|---:|---:|
| H1_Matching_sRMSE_Calibrated_vs_TargetOnly | 8 | -0.0188 [-0.0608, +0.0037] | 1 | +0.000 | no |
| H2_Matching_NDCG_Calibrated_vs_TargetOnly | 8 | -0.0041 [-0.0116, -0.0000] | 1 | -1.000 | no |
| H3_Reversed_sRMSE_Gated_vs_Fixed | 8 | +0.1995 [+0.1287, +0.2841] | 0.01562 | +1.000 | yes |
| H4_GateAcceptance_Matching_vs_Reversed | 8 | +0.3750 [+0.1250, +0.7500] | 0.375 | +1.000 | no |

## 3. Primary-slice model means (context=12)

| Relation | Method | sRMSE | NDCG@top | Pairwise accuracy | Top-1 regret | Negative-transfer rate |
|---|---|---:|---:|---:|---:|---:|
| matching | Calibrated-Source+Residual | 0.4028 | 0.9912 | 0.9102 | 0.0031 | 0.125 |
| matching | Fixed-Source+Residual | 0.4908 | 0.8773 | 0.8832 | 0.3259 | 0.625 |
| matching | Gated-Source+Residual | 0.4028 | 0.9912 | 0.9102 | 0.0031 | 0.125 |
| matching | Source-Affine-Only | 0.9716 | 0.4045 | 0.5604 | 0.8097 | 1.000 |
| matching | Target-Only | 0.3840 | 0.9953 | 0.9156 | 0.0025 | 0.000 |
| reversed | Calibrated-Source+Residual | 0.4068 | 0.9949 | 0.9083 | 0.0025 | 0.250 |
| reversed | Fixed-Source+Residual | 0.5835 | 0.9610 | 0.8487 | 0.1784 | 1.000 |
| reversed | Gated-Source+Residual | 0.3840 | 0.9953 | 0.9156 | 0.0025 | 0.000 |
| reversed | Source-Affine-Only | 1.0258 | 0.3746 | 0.5051 | 0.6661 | 1.000 |
| reversed | Target-Only | 0.3840 | 0.9953 | 0.9156 | 0.0025 | 0.000 |
| wrong | Calibrated-Source+Residual | 0.3845 | 0.9950 | 0.9136 | 0.0025 | 0.250 |
| wrong | Fixed-Source+Residual | 0.5416 | 0.8674 | 0.8669 | 0.2590 | 0.625 |
| wrong | Gated-Source+Residual | 0.3896 | 0.9953 | 0.9134 | 0.0025 | 0.125 |
| wrong | Source-Affine-Only | 0.9854 | 0.4175 | 0.5882 | 0.4690 | 1.000 |
| wrong | Target-Only | 0.3840 | 0.9953 | 0.9156 | 0.0025 | 0.000 |

## 4. Gate behavior (context=12)

| Relation | Acceptance coverage | Accepted instances | Risk among accepted | Intention-to-use risk | Mean sRMSE delta |
|---|---:|---:|---:|---:|---:|
| matching | 0.375 | 3 | 0.333 | 0.125 | +0.0188 |
| reversed | 0.000 | 0 | NA | 0.000 | +0.0000 |
| wrong | 0.125 | 1 | 1.000 | 0.125 | +0.0055 |

## 5. Source-expert and support diagnostics

| Relation | Source held-out NDCG | Source held-out pairwise accuracy | Below-membership-0.05 fraction | Normalized extraction-center error |
|---|---:|---:|---:|---:|
| matching | 0.8262 | 0.7251 | 0.0000 | 2.8200 |
| reversed | 0.1867 | 0.2749 | 0.0000 | 2.8200 |
| wrong | 0.6960 | 0.7744 | 0.0000 | 1.0228 |

## 6. Completeness

Result rows: 240; diagnostic rows: 24; failures: 0. Primary tests use complete paired instances only.

## 7. Data-derived interpretation boundary

Supported Pilot hypotheses:

- H3_Reversed_sRMSE_Gated_vs_Fixed

Any supported matching result applies only to correct oracle region correspondence in this 2D controlled Pilot. Wrong-source behavior and explicit reversal behavior are separate safety stress tests. Gate rejection is not proof that a source is intrinsically non-transferable, and low observed harm is not a universal no-negative-transfer guarantee.
