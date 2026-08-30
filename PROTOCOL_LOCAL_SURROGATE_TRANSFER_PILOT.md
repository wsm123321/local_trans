# Frozen Protocol: Source Local-Surrogate Transfer Pilot v1

## 1. Narrow research question

With source and target high-quality regions already placed in a correct, externally fixed correspondence, does an extracted source local relative-rank surrogate add held-out target prediction and ranking value beyond a target-only local GP?

This is a **static held-out model-transfer pilot**. It does not test region discovery, region matching, alignment estimation, candidate filtering, acquisition, sequential BO, or budget savings.

## 2. Separation from the preceding extraction claim

The preceding stage established that source-local centers and relative ordering can be recovered under its protocol, while covariance-shape recovery was not superior to simple baselines. This pilot therefore:

- consumes the already defined `SourceLocalStructure` rank expert;
- fixes source and target global-basin anchors from the controlled generator (oracle correspondence);
- uses translation plus one task-independent isotropic chart radius;
- does **not** use the extracted full covariance as a cross-task alignment estimate;
- treats source fidelity and target incremental value as separate diagnostics.

A positive result supports model transfer only under correct region correspondence. A negative result does not overturn source-structure extraction fidelity.

## 3. Experimental unit and primary slice

The independent primary unit is one controlled task instance:

```text
(problem family, dimension, seed)
```

The confirmatory Pilot-v1 slice is:

- dimension: 2;
- target context size: 12;
- matching or reversed expert relation as specified by each hypothesis;
- every problem/seed instance retained regardless of performance.

Other configured target context sizes and the realistic wrong-source relation are secondary response curves. Test-panel points are correlated measurements within an instance and are never treated as replicates.

## 4. Frozen data flow and leakage controls

For every instance, independent `SeedSequence` children generate:

1. task parameters;
2. matching-source training design;
3. wrong-source training design;
4. target-context Sobol prefix;
5. target-test Sobol panel;
6. extractor/model random states.

Controls:

- source structures use source observations only;
- source and target anchors come from generator metadata, never from target test labels;
- target context sizes are nested prefixes of one frozen Sobol design;
- target context and target test designs use independent scrambles and must not overlap after 12-decimal rounding;
- all methods receive the same target context and target test panel within an instance;
- gate decisions use target-context cross-validation only;
- target test labels are revealed only to the metric function;
- full-study configuration is frozen before its outcomes are inspected;
- Quick outputs are code-validation artifacts and are not confirmatory evidence.

## 5. Local chart and source expert

The generator's source global-basin center is used only to associate one extracted structure with the controlled target region. If `mu_hat_s` is the selected extracted center, `a_t` is the oracle target anchor, and `r` is the frozen scalar radius (`0.04` of the mean domain width in Pilot v1), the shared chart is

```text
x_s(z) = mu_hat_s + r z
x_t(z) = a_t + r z,   z in [-1, 1]^2.
```

Thus the source expert is queried in its native extracted region rather than being forced to extrapolate at the source oracle center. The distance from `mu_hat_s` to the source oracle anchor is persisted as an extraction diagnostic. The expert predicts source relative quality `h_s(x_s(z)) in [0,1]`; larger is better. Its full covariance is used internally by the already fitted source model but is not used to estimate the cross-task chart.

Relations:

- `matching`: matching source from the controlled task suite;
- `wrong`: deliberately mismatched source from the same benchmark family, aligned only at declared global anchors; descriptive stress test;
- `reversed`: the matching expert with quality `1-h_s`; an explicit order-reversal negative control.

The reversed relation is not presented as a natural task distribution. It exists to make failure to reject harmful model knowledge falsifiable.

## 6. Shared target model and interventions

All GP-containing methods use the same fixed Matern-5/2 kernel, chart input, white-noise setting, normalization, and target observations. Hyperparameters are not optimized per method.

Methods:

1. `Target-Only`: shared GP fitted directly to target responses.
2. `Source-Affine-Only`: non-negative ridge-shrunk affine calibration of source cost `1-h_s` to target responses, without a target residual GP.
3. `Fixed-Source+Residual`: an unconditionally positive source prior scaled by target response standard deviation plus the shared target residual GP.
4. `Calibrated-Source+Residual`: a non-negative, ridge-shrunk target calibration plus the shared target residual GP. A non-positive raw association shrinks exactly to Target-Only.
5. `Gated-Source+Residual`: uses the calibrated transfer model only when target-context K-fold evidence passes all frozen conditions; otherwise refits the exact Target-Only model.

The v1 gate requires:

- positive calibrated slope;
- cross-validated transfer RMSE strictly below target-only RMSE;
- target-context source/target pairwise ordering accuracy at least 0.55;
- non-negligible variation in expert scores.

The gate is not tuned on target test results.

## 7. Metrics

On one independent frozen target panel per instance:

- standardized RMSE (primary predictive metric; lower is better);
- NDCG@top (primary decision-ranking metric; higher is better);
- Spearman rank correlation;
- pairwise ordering accuracy;
- precision@top;
- normalized Top-1 regret;
- mean Gaussian negative log likelihood;
- empirical 95% interval coverage.

Gate diagnostics:

- acceptance coverage;
- context CV relative RMSE gain;
- context pairwise agreement;
- calibrated/raw slope;
- test negative-transfer rate using the descriptive harm margin `Delta sRMSE > 0.01` relative to Target-Only.

Risk among accepted instances is reported together with coverage. Rejected instances fall back exactly to Target-Only and are included in all intention-to-use summaries.

## 8. Pre-specified primary hypotheses

All advantages are oriented so positive favors the claimed method. Tests use only the primary slice (`d=2`, target context 12).

- **H1 Matching predictive increment**: `Target-Only sRMSE - Calibrated-Source+Residual sRMSE > 0`.
- **H2 Matching ranking increment**: `Calibrated-Source+Residual NDCG - Target-Only NDCG > 0`.
- **H3 Reversal safety relative to unconditional transfer**: `Fixed-Source+Residual sRMSE - Gated-Source+Residual sRMSE > 0` under the reversed expert.
- **H4 Observable gate discrimination**: gate acceptance indicator under matching minus the paired reversed expert is greater than zero.

H1/H2 test whether target calibration exposes incremental model value. H3 does not establish absolute no-harm; it asks only whether the proposed conservative path is safer than unconditional positive transfer. H4 asks whether harmful reversal is distinguishable from matching knowledge using target context alone.

## 9. Statistical analysis

- paired differences at task-instance level;
- 5,000-sample nonparametric bootstrap CI for the mean paired advantage;
- one-sided paired Wilcoxon test with Pratt zero handling;
- Holm family-wise correction across H1-H4;
- rank-biserial effect size and paired win/tie/loss rates;
- complete-pair analysis with all failures reported.

A primary hypothesis is marked supported only when:

```text
95% bootstrap lower bound > 0
and Holm-adjusted p < 0.05.
```

Because this is a controlled 2D Pilot, even all four supported hypotheses would not establish high-dimensional, noisy, unknown-alignment, online-BO, or general no-negative-transfer claims.

## 10. Failure, audit, and reporting rules

- Every failure is written with the full instance key and exception.
- Failed instances are not silently replaced.
- No instance is removed for poor performance or gate rejection.
- The manifest records configuration, versions, counts, Git identity/status, and SHA-256 hashes of produced tables.
- Quick and full output directories are never merged.
- Any post-full parameter change creates a new protocol/config/output identity.
- Source expert fidelity, matching transfer benefit, realistic wrong-source behavior, explicit reversal safety, gate coverage, and accepted-set risk are reported separately.
