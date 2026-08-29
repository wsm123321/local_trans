# Full Source Local-Structure Study Audit

## Frozen execution scope

- Configuration: `configs/source_structure_full.json`
- Controlled recovery: dimensions {2, 5}, 8 seeds, sample sizes {100, 250, 500}, noise levels {0, 0.02, 0.05}; 144 independent instances.
- Held-out validation: GMM, Rastrigin, Lunacek, Ackley; dimensions {2, 5}; 8 seeds; matching and wrong source scenarios; 128 independent source-task instances.
- Source training samples per validation instance: 160.
- Frozen test design: 2,500 global samples plus 300 samples per extracted structure.
- Statistical unit: independent task instance.
- Bootstrap samples: 5,000.
- Multiplicity control: Holm family-wise correction at alpha=0.05.

## Completeness checks

- Recovery summary: 432 rows = 144 instances × 3 methods.
- Recovery diagnostics: 261 extracted-structure rows.
- Recovery failures: 0.
- Validation summary: 3,584 rows = 128 instances × 2 evaluation domains × 2 subsets × 7 methods.
- Validation diagnostics: 157 extracted-structure rows.
- Validation failures: 0.
- Primary tests: 6 pre-specified paired hypotheses.
- Missing values in recovery, validation, and primary-test tables: 0.
- Automated tests after formal execution: 29 passed.

## Primary statistical results

All six pre-specified hypotheses are supported after Holm adjustment:

1. Source NDCG: proposed local structure exceeds geometry-only; mean advantage +0.0261, 95% CI [0.0192, 0.0334], Holm p=7.776e-16.
2. Source local-subset Spearman: proposed exceeds geometry-only; +0.0809 [0.0639, 0.0989], Holm p=7.176e-19.
3. Source NDCG: proposed exceeds label permutation; +0.1331 [0.1134, 0.1543], Holm p=2.854e-22.
4. Matching-target NDCG: proposed exceeds geometry-only; +0.0213 [0.0116, 0.0319], Holm p=6.162e-06.
5. Controlled basin recall: proposed exceeds top observations; +0.0694 [0.0139, 0.1250], Holm p=0.005181.
6. Controlled center error: proposed is lower than top observations; oriented advantage +0.1064 [0.0917, 0.1212], Holm p=2.206e-20.

## Aggregate findings

- Controlled recovery basin recall: proposed 0.3889, top observations 0.3194, random centers 0.1273.
- Controlled normalized center error: proposed 0.0957, top observations 0.2021, random centers 0.2703.
- Proposed covariance shape error (0.1769) is worse than the top-observation/random-center construction (0.1054); covariance shape was not a pre-specified supported claim and should not be overstated.
- Held-out source NDCG: proposed 0.8457 (matching) and 0.8451 (wrong-source task evaluated on its own source), both above geometry-only.
- Matching-target NDCG: proposed 0.8381 versus geometry-only 0.8168.
- Wrong-target NDCG: proposed 0.7484 versus geometry-only 0.7403; this descriptive result is not a pre-specified transfer-success claim.
- Global Source GP remains competitive or stronger on some held-out metrics. The evidence supports faithful local-structure extraction, not universal dominance over a global source response model.

## Honest interpretation boundary

The formal experiment supports faithful recovery of source-local centers and local relative-ranking information. It also supports a modest matching-target NDCG advantage over geometry alone. These results do not establish that every faithfully extracted structure is transferable, nor that the proposed structure should be used without a separate source-target selection mechanism. Source fidelity and target transferability remain separate claims.

## Artifact SHA256

- `analysis/source_structure_primary_tests.csv`: `10c2825ccf54b6f1e8e4b4dbfd6114b1a7956cfd93eb63d97b2b1e0a10114c0e`
- `analysis/source_structure_recovery.png`: `ad341d9f5830c113194ce315a4e5921a9133e9fe29daa773e284ca1995bfcc15`
- `analysis/SOURCE_STRUCTURE_REPORT.md`: `b5d7f4cbb026dcb1b29ab0ad2469cd26cd8d50f263bdc1ca4f616f2f0e74f120`
- `analysis/source_structure_source_ndcg.png`: `d9ac7eb173ce35f6db24004e0c1d9183492d857b4c714cb59cffd562b796ca43`
- `analysis/source_structure_target_ndcg.png`: `1206de465db91e80d1216defa14abf4d5cf72420d2d6ca2902fe132c96be20cb`
- `recovery/source_structure_recovery.csv`: `66599e78ddae2bc452c6bd652726a0c555c6426eee4410e4bd58d35726b1ad95`
- `recovery/source_structure_recovery_diagnostics.csv`: `07ecd877c334697562f3c8914a9958652fff559677c18401625adec8faab0036`
- `recovery/source_structure_recovery_failures.csv`: `83be83296c4bbe4454738c29d8b6e74cc5229014bc72a711f36ddb47386aabe5`
- `recovery/source_structure_recovery_manifest.json`: `dcbb14857ea2cd82239585fdd3b4e23020eb32617fc9f2118a53734a94de85e9`
- `validation/source_structure_diagnostics.csv`: `37817fbca3d096ded8ece7be12cce883f3487d07a66f60f3b897cc2d16f65c72`
- `validation/source_structure_validation.csv`: `331ce86f42e95a0d6fcdd19da86147f416bbc3ad07f56ffeb95417f508bed8a5`
- `validation/source_structure_validation_failures.csv`: `39b9cc5f3b9a4c02966876691fd50733c2718fb6a418b7a116395bcfce2f581e`
- `validation/source_structure_validation_manifest.json`: `d67963188ed816a6945c80fcb6f7d1e0917af526fa06189f9a07de144191bee6`
