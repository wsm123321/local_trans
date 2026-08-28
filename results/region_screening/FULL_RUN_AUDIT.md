# Full Region-Screening Run Audit

## Frozen execution scope

- Configuration: `configs/region_screening_full.json`
- Problems: GMM, Rastrigin, Lunacek, Ackley
- Dimensions: 2, 5
- Seeds: 42, 101, 2026, 777, 999, 1234, 5678, 8888
- Shared-proposal mechanism instances: 64
- Equal-budget sequential instances: 64
- Drift target instances: 16
- Drift values: 0, 0.25, 0.5, 1, 2, 4
- Target evaluation budget per sequential method: 20

## Completeness checks

- Mechanism summary: 448 rows = 64 instances × 7 methods.
- Sequential summary: 448 rows = 64 instances × 7 methods.
- Sequential traces: 9,408 rows = 448 runs × 21 checkpoints (initial state plus 20 evaluations).
- Drift summary: 192 rows = 16 target instances × 6 drift values × 2 methods.
- Automated tests after the full run: 17 passed.
- Mechanism compatibility NaNs occur only for `Matching-Soft-Rerank`, for which compatibility diagnostics are not defined by design.

## Primary empirical findings

### Shared-proposal mechanism study

- Target-Only mean normalized regret: 0.6742.
- Matching-Fixed-Filter: 0.5578; paired reduction +0.1164, 95% bootstrap CI [0.0432, 0.1946], Wilcoxon p=0.008286.
- Matching-Adaptive-Filter: 0.6694; paired reduction +0.0048, CI crosses zero, Wilcoxon p=0.5751.
- Matching-Soft-Rerank: 0.5106; paired reduction +0.1636, CI [0.0833, 0.2522], Wilcoxon p=0.0006628.
- Random-Adaptive-Filter: 0.6938; no evidence of improvement.

### Equal-budget sequential optimization

- Target-Only final normalized regret: 0.5375.
- Matching-Fixed-Filter: 0.4690; paired reduction +0.0685, CI [0.0114, 0.1281], Wilcoxon p=0.04624.
- Matching-Adaptive-Filter: 0.5010; paired reduction +0.0365, CI crosses zero, Wilcoxon p=0.6169.
- Matching-Soft-Rerank: 0.3947; paired reduction +0.1428, CI [0.0742, 0.2133], Wilcoxon p=0.000615.
- Random-Adaptive-Filter: 0.5402; no evidence of improvement.

### Drift boundary

- Fixed filtering shows a positive mean direction at small drift and a negative mean direction at drift 2 and 4, but every per-drift 95% bootstrap interval crosses zero.
- Adaptive filtering remains near Target-Only across drift levels; all per-drift intervals also include zero.
- Therefore this full experiment does not establish a statistically resolved positive or negative drift boundary.
- The adaptive rule is conservative, but the current data do not establish that it is a superior negative-transfer safeguard.

## Honest scope boundary

The full results support the usefulness of matching fixed filtering in the shared-proposal and sequential comparisons. They do not establish that the current adaptive compatibility rule reliably identifies transferability or yields a resolved drift boundary. The existing soft-reranking baseline remains stronger than both fixed and adaptive filtering in these experiments.

## Artifact SHA256

- `analysis/drift_transfer_boundary.png`: `0b3d37b1d255ecfe9fee014632c9aeafa0cc82865c733adf232d0b0c92e85624`
- `analysis/mechanism_normalized_regret.png`: `0bcf217f39064b9a62882be6c1311901414d47e35a866a84d4ae9dc7848fec6e`
- `analysis/SCREENING_STUDY_REPORT.md`: `99ed048df357fb0bf5a89c403d2e71af136f49b49088aa91d068c88e46eb1bea`
- `analysis/sequential_normalized_regret.png`: `d1359908a5a53d13158c2bb31046972c7c81c0888ad269ccdf9866ddc2b542eb`
- `drift/screening_drift_summary.csv`: `3928f95bcd4af2672eb761f73432eb19b544f677543b018a137457c481eee3a4`
- `mechanism/screening_mechanism_details.json`: `060e015de718324d323d9552cf053edbd4b57a8ae5badb576c0dc5050b4ff8ff`
- `mechanism/screening_mechanism_summary.csv`: `00323482c8c2ee5355927fb5ef6031ab8b758e1775d4c8859e8d4a565fc1dc42`
- `sequential/screening_sequential_summary.csv`: `401f4a61df88f2473de9afb434006ae6e5ebefff4a3f1ff33fdee55edc813af2`
- `sequential/screening_sequential_traces.csv`: `40c8a15d28217ab457654601037427d56b254c7233f256d2b36b70196b893919`
