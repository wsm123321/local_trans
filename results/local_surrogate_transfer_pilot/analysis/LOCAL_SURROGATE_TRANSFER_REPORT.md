# Source Local-Surrogate Transfer Pilot v1 Report

This report is generated from frozen CSV artifacts. The statistical unit is one `(problem, dimension, seed)` task instance; candidate-panel points are not treated as replicates.

## 1. Scope

This is a controlled 2D static held-out model-transfer Pilot under oracle region correspondence and a frozen isotropic local chart. It does not test unknown alignment, online BO, or a general no-harm guarantee.

## 2. Pre-specified primary tests

| Hypothesis | Pairs | Mean oriented advantage [95% bootstrap CI] | Holm p | Rank-biserial | Supported |
|---|---:|---:|---:|---:|---:|
| H1_Matching_sRMSE_Calibrated_vs_TargetOnly | 64 | +0.0127 [+0.0001, +0.0261] | 0.01077 | +0.362 | yes |
| H2_Matching_NDCG_Calibrated_vs_TargetOnly | 64 | -0.0003 [-0.0009, +0.0002] | 0.5869 | +0.021 | no |
| H3_Reversed_sRMSE_Gated_vs_Fixed | 64 | +0.1471 [+0.0989, +0.2012] | 3.865e-08 | +0.639 | yes |
| H4_GateAcceptance_Matching_vs_Reversed | 64 | +0.1875 [+0.0625, +0.3125] | 0.007017 | +0.667 | yes |

## 3. Primary-slice model means (context=12)

| Relation | Method | sRMSE | NDCG@top | Pairwise accuracy | Top-1 regret | Negative-transfer rate |
|---|---|---:|---:|---:|---:|---:|
| matching | Calibrated-Source+Residual | 0.3548 | 0.9970 | 0.9211 | 0.0075 | 0.141 |
| matching | Fixed-Source+Residual | 0.4042 | 0.9904 | 0.9045 | 0.0298 | 0.484 |
| matching | Gated-Source+Residual | 0.3547 | 0.9970 | 0.9214 | 0.0075 | 0.047 |
| matching | Source-Affine-Only | 0.9555 | 0.4318 | 0.5788 | 0.6864 | 1.000 |
| matching | Target-Only | 0.3676 | 0.9973 | 0.9179 | 0.0056 | 0.000 |
| reversed | Calibrated-Source+Residual | 0.3678 | 0.9974 | 0.9174 | 0.0054 | 0.062 |
| reversed | Fixed-Source+Residual | 0.5156 | 0.9481 | 0.8684 | 0.1451 | 0.781 |
| reversed | Gated-Source+Residual | 0.3686 | 0.9974 | 0.9174 | 0.0056 | 0.031 |
| reversed | Source-Affine-Only | 1.0042 | 0.4203 | 0.5077 | 0.7024 | 1.000 |
| reversed | Target-Only | 0.3676 | 0.9973 | 0.9179 | 0.0056 | 0.000 |
| wrong | Calibrated-Source+Residual | 0.3461 | 0.9970 | 0.9228 | 0.0070 | 0.156 |
| wrong | Fixed-Source+Residual | 0.4219 | 0.9717 | 0.8994 | 0.0668 | 0.516 |
| wrong | Gated-Source+Residual | 0.3471 | 0.9969 | 0.9233 | 0.0077 | 0.062 |
| wrong | Source-Affine-Only | 0.9338 | 0.4904 | 0.6035 | 0.6545 | 0.984 |
| wrong | Target-Only | 0.3676 | 0.9973 | 0.9179 | 0.0056 | 0.000 |

## 4. Gate behavior (context=12)

| Relation | Acceptance coverage | Accepted instances | Risk among accepted | Intention-to-use risk | Mean sRMSE delta |
|---|---:|---:|---:|---:|---:|
| matching | 0.234 | 15 | 0.200 | 0.047 | -0.0129 |
| reversed | 0.047 | 3 | 0.667 | 0.031 | +0.0010 |
| wrong | 0.422 | 27 | 0.148 | 0.062 | -0.0205 |

## 5. Source-expert and support diagnostics

| Relation | Source held-out NDCG | Source held-out pairwise accuracy | Below-membership-0.05 fraction | Normalized extraction-center error |
|---|---:|---:|---:|---:|
| matching | 0.7733 | 0.7837 | 0.0000 | 3.0016 |
| reversed | 0.1596 | 0.2163 | 0.0000 | 3.0016 |
| wrong | 0.7807 | 0.8041 | 0.0000 | 2.7020 |

## 6. Completeness

Result rows: 2880; diagnostic rows: 192; failures: 0. Primary tests use complete paired instances only.

## 7. Data-derived interpretation boundary

Supported Pilot hypotheses:

- H1_Matching_sRMSE_Calibrated_vs_TargetOnly
- H3_Reversed_sRMSE_Gated_vs_Fixed
- H4_GateAcceptance_Matching_vs_Reversed

Any supported matching result applies only to correct oracle region correspondence in this 2D controlled Pilot. Wrong-source behavior and explicit reversal behavior are separate safety stress tests. Gate rejection is not proof that a source is intrinsically non-transferable, and low observed harm is not a universal no-negative-transfer guarantee.
