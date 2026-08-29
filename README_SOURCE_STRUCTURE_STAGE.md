# Source Local Structure Extraction Stage

Base repository commit: `917e38a32c947d9394d7d3ff96cbbe39c6236bd0`

This stage addresses one narrow question before further transfer optimization:

> Can high-quality local structure be extracted faithfully from historical source observations?

The code deliberately separates **structure extraction** from **source-target similarity** and **optimization performance**. A structure can be valid on its source task yet not be transferable to a mismatched target. The experiments report these claims separately.

## 1. Structure definition

For source task observations \(D_s=\{(x_i,y_i)\}\), each extracted structure is

\[
\mathcal S_k=(\mu_k,\Sigma_k,h_k,\omega_k),
\]

where:

- \(\mu_k,\Sigma_k\): location and shape of a high-quality local region;
- \(h_k(z)\): a local surrogate of **relative rank quality** in canonical coordinates
  \(z=\Sigma_k^{-1/2}(x-\mu_k)\);
- \(\omega_k\): cross-validated reliability of the local ordering model.

The final source-structure score is geometrically gated:

\[
G_k(x)=q_k\exp\!\left[-\tfrac12(x-\mu_k)^\top\Sigma_k^{-1}(x-\mu_k)\right]
\left[\epsilon_q+(1-\epsilon_q)h_k(x)\right]\omega_k.
\]

It transfers neither raw source values nor a global source response surface.

## 2. Extraction algorithm

1. Rank-normalize source responses, preserving minimization order while removing response scale.
2. Select the elite source observations.
3. Fit full-covariance Gaussian mixtures and choose the number of regions by BIC, subject to a minimum cluster size.
4. For each elite region, add nearby non-elite boundary observations. This prevents a model trained only on truncated elite values.
5. Whiten the region into canonical local coordinates.
6. Train a local GP or random-forest rank surrogate.
7. Estimate out-of-fold Spearman, NDCG, top-set precision, and a reliability weight.
8. Gate local predictions by geometric membership to avoid uncontrolled extrapolation.

## 3. Academic validation design

### Study A: controlled structure recovery

Synthetic Gaussian-basin landscapes expose exact centers and covariance shapes. The extractor is trained from Latin-hypercube source samples under several sample sizes and noise levels.

Baselines:

- best observed source points (`Top-Observations`);
- uniformly random centers (`Random-Centers`).

Metrics:

- oracle-basin recall;
- matched Mahalanobis center error;
- normalized Euclidean center error;
- covariance shape error.

### Study B: held-out source fidelity

For each benchmark source task, training and testing observations use independent random streams. The frozen test set combines global samples and independent samples around the extracted regions. All methods are evaluated on the same set.

Baselines:

- geometry only;
- global source GP;
- distance to the best observed source point;
- label-permutation null;
- random score.

Metrics:

- Spearman rank correlation;
- NDCG at the pre-specified top fraction;
- top-set precision;
- top-set enrichment;
- normalized Top-1 regret.

### Study C: cross-task specificity

The same extracted source structure and frozen candidate set are evaluated against the target function. Matching and deliberately mismatched sources are reported separately. This is **not** used to prove source extraction; it tests whether faithfully extracted source structure is also reusable on a target.

## 4. Pre-specified primary claims

The analysis script tests six frozen hypotheses:

1. proposed source NDCG exceeds geometry-only NDCG;
2. proposed local-subset Spearman exceeds geometry-only Spearman;
3. proposed source NDCG exceeds the label-permutation null;
4. matching-target NDCG exceeds geometry-only NDCG;
5. controlled basin recall exceeds best-point recovery;
6. controlled center error is lower than best-point recovery.

Statistical units are independent task instances, not candidate points. Tests use paired one-sided Wilcoxon statistics, instance-level bootstrap confidence intervals, and Holm family-wise correction. Conclusions are generated from CSV files rather than hardcoded.

The formal frozen protocol is in `PROTOCOL_SOURCE_STRUCTURE.md`.

## 5. Installation and tests

Extract this archive into the repository root, then run:

```bash
pip install -e .
python -m pytest tests/test_source_local_structure.py -q
```

## 6. Quick study

```bash
python scripts/run_all_source_structure_studies.py \
  --config configs/source_structure_quick.json \
  --output results/source_structure_stage_quick
```

## 7. Full study

```bash
python scripts/run_all_source_structure_studies.py \
  --config configs/source_structure_full.json \
  --output results/source_structure_stage
```

Run stages separately when needed:

```bash
python scripts/run_source_structure_recovery.py \
  --config configs/source_structure_full.json \
  --output results/source_structure_stage/recovery

python scripts/run_source_structure_validation.py \
  --config configs/source_structure_full.json \
  --output results/source_structure_stage/validation

python scripts/analyze_source_structure_study.py \
  --input results/source_structure_stage \
  --output results/source_structure_stage/analysis
```

## 8. Expected artifacts

```text
results/source_structure_stage/
├── recovery/
│   ├── source_structure_recovery.csv
│   ├── source_structure_recovery_diagnostics.csv
│   ├── source_structure_recovery_failures.csv
│   └── source_structure_recovery_manifest.json
├── validation/
│   ├── source_structure_validation.csv
│   ├── source_structure_diagnostics.csv
│   ├── source_structure_validation_failures.csv
│   └── source_structure_validation_manifest.json
└── analysis/
    ├── SOURCE_STRUCTURE_REPORT.md
    ├── source_structure_primary_tests.csv
    ├── source_structure_recovery.png
    ├── source_structure_source_ndcg.png
    └── source_structure_target_ndcg.png
```

## 9. Interpretation boundary

The paper may claim that source structure was successfully extracted only when the held-out source and controlled-recovery hypotheses are supported after multiplicity correction. Matching-target performance is a separate transfer claim. Poor performance on mismatched targets does not invalidate extraction; it demonstrates that extraction and similarity selection are different research problems.
