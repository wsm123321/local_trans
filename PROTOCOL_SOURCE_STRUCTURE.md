# Frozen Protocol: Source Local Structure Extraction and Validation

## 1. Research question

Can historical source observations be compressed into a geometrically localized, response-scale-invariant representation that recovers known high-quality basin structure and ranks independent source candidates better than geometry alone?

## 2. Estimand

The primary estimand is the instance-level improvement of the proposed local-structure score over a geometry-only score on independent source test candidates. A secondary estimand is controlled oracle-basin recovery. Cross-target performance is explicitly secondary and is not used to establish source-extraction validity.

## 3. Experimental unit

An independent unit is one tuple of:

```text
(problem, dimension, seed, source scenario, source index)
```

for held-out validation, or:

```text
(dimension, seed, source sample size, noise level)
```

for controlled recovery. Candidate points within an instance are correlated and are never treated as independent replicates in inferential tests.

## 4. Data separation and leakage controls

- Source training samples are used for elite selection, region fitting, context construction, and local-model fitting.
- Source test samples are generated from a separate random stream after fitting.
- The target function is never queried during source-structure extraction.
- Every comparison method receives the identical frozen test candidate set.
- Label-permutation structures are fitted from the same source inputs with permuted responses.
- Configuration files are frozen before inspecting full-study outcomes.
- Any hyperparameter modification after the full run defines a new study version and requires a new output directory.

## 5. Methods

### Proposed-Local-Structure

BIC-selected elite GMM geometry, boundary-expanded local context, covariance whitening, rank-normalized local surrogate, cross-validated reliability, and geometric extrapolation gate.

### Proposed-No-CV-Weight

Same extraction and local surrogate without reliability weighting. This isolates the effect of internal validation weighting.

### Geometry-Only

Region quality multiplied only by Gaussian Mahalanobis support.

### Global-Source-GP

A GP fitted to the full source response. This is a predictive upper-reference on the source task, not the proposed transfer mechanism.

### Best-Point-Distance

Negative standardized distance to the best observed source sample.

### Label-Permutation

The complete extraction pipeline after source response permutation. This tests whether the method extracts signal beyond input geometry and model flexibility.

### Random-Score

Independent random ranking sanity check.

## 6. Primary hypotheses

All advantages are oriented so positive values favor the proposed method.

- **H1**: source-domain NDCG@top of Proposed-Local-Structure exceeds Geometry-Only on all held-out candidates.
- **H2**: source-domain Spearman of Proposed-Local-Structure exceeds Geometry-Only on the frozen local subset.
- **H3**: source-domain NDCG@top of Proposed-Local-Structure exceeds Label-Permutation.
- **H4**: matching-target NDCG@top of Proposed-Local-Structure exceeds Geometry-Only.
- **H5**: controlled oracle-basin recall of Proposed-Local-Structure exceeds Top-Observations.
- **H6**: controlled normalized center error of Proposed-Local-Structure is lower than Top-Observations.

H1, H2, H3, H5, and H6 concern extraction fidelity. H4 concerns transfer utility.

## 7. Metrics

### Ranking metrics

- Spearman rank correlation;
- NDCG at the configured top fraction;
- precision at the configured top fraction;
- top-set quality enrichment;
- normalized Top-1 regret.

### Recovery metrics

- basin recall under a fixed oracle-Mahalanobis threshold;
- mean and median matched Mahalanobis error;
- center error normalized by domain diameter;
- trace-normalized covariance shape error.

## 8. Statistical analysis

- paired differences at the independent-instance level;
- 5,000-sample nonparametric bootstrap confidence interval for the mean paired advantage;
- one-sided paired Wilcoxon test with Pratt treatment of zeros;
- Holm family-wise p-value correction across H1-H6;
- rank-biserial paired effect size;
- complete-pair analysis with all extraction failures reported separately.

A claim is marked supported only when:

```text
95% bootstrap lower bound > 0
and Holm-adjusted p < 0.05
```

## 9. Sample sizes

The exact sample sizes, dimensions, seeds, source-task count, and test-pool composition are stored in the JSON configuration and copied verbatim into each run manifest. The quick configuration is for code validation only. Paper claims must use the frozen full configuration or a subsequently preregistered version.

## 10. Failure and exclusion policy

- Numerical or extraction failures are written to failure CSV files with the full instance key and exception.
- Failed instances are not silently replaced.
- Inferential tests use complete pairs only and report the resulting pair count.
- No instance is removed based on poor performance.
- Empty or constant-score cases produce explicit finite or missing metrics according to the metric definition.

## 11. Reporting rule

Source fidelity, controlled recovery, and target transferability must be reported in separate sections. The manuscript must not infer target transferability solely from successful source recovery, nor describe a mismatched-target failure as a failure to extract the source structure.
