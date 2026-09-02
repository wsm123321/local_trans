# Unified local-guidance full-run audit

- Overall status: **PASS**
- Stage: `unified-local-guidance-full`

## Checks

| Check | Status | Detail |
|---|---|---|
| approved_artifact_paths | PASS | all canonical files present |
| json_readable | PASS | config and manifest parse as objects |
| csv_readable | PASS | ['failures', 'mechanism', 'panel', 'sequential_summary', 'sequential_traces', 'source_diagnostics'] |
| expected_cartesian_instances_from_config | PASS | instances=64 |
| config_hash | PASS | expected=f02882d8d55392cf80b213e3699bb1938566e1ab2c6f9012569fcf4133a0ef7a, observed=f02882d8d55392cf80b213e3699bb1938566e1ab2c6f9012569fcf4133a0ef7a |
| stage_identity | PASS | config=unified-local-guidance-full, manifest=unified-local-guidance-full |
| protocol_hash | PASS | E:\study\暑假工作\region_guided_reranking_study\PROTOCOL_UNIFIED_LOCAL_GUIDANCE.md |
| runner_hash | PASS | E:\study\暑假工作\region_guided_reranking_study\scripts\run_unified_local_guidance_study.py |
| companion_hash | PASS | E:\study\暑假工作\region_guided_reranking_study\src\region_guided_reranking_study\local_structure_guidance.py |
| canonical_artifact_declarations | PASS | missing=[] |
| artifact_hashes | PASS | all declared hashes match |
| mechanism_method_set | PASS | expected=['Geometry-Only', 'Local-Rank+Reliability', 'Local-Rank-No-Reliability', 'Reversed-Local-Rank', 'Target-Only'], observed=['Geometry-Only', 'Local-Rank+Reliability', 'Local-Rank-No-Reliability', 'Reversed-Local-Rank', 'Target-Only'] |
| sequential_method_set | PASS | expected=['Geometry-Only', 'Local-Rank+Reliability', 'Local-Rank-No-Reliability', 'Target-Only'], observed=['Geometry-Only', 'Local-Rank+Reliability', 'Local-Rank-No-Reliability', 'Target-Only'] |
| trace_method_set | PASS | expected=['Geometry-Only', 'Local-Rank+Reliability', 'Local-Rank-No-Reliability', 'Target-Only'], observed=['Geometry-Only', 'Local-Rank+Reliability', 'Local-Rank-No-Reliability', 'Target-Only'] |
| mechanism_instance_set | PASS | missing=[], extra=[] |
| panel_instance_set | PASS | missing=[], extra=[] |
| sequential_summary_instance_set | PASS | missing=[], extra=[] |
| sequential_trace_instance_set | PASS | missing=[], extra=[] |
| mechanism_unique_keys | PASS | duplicates=0 |
| sequential_summary_unique_keys | PASS | duplicates=0 |
| sequential_trace_unique_keys | PASS | duplicates=0 |
| mechanism_row_count | PASS | expected=320, observed=320 |
| mechanism_panel_row_count | PASS | expected=6400, observed=6400 |
| sequential_summary_row_count | PASS | expected=256, observed=256 |
| sequential_trace_row_count | PASS | expected_paid=5120, observed_paid=5120, initial_rows=256, total=5376 |
| finite_mechanism_values | PASS | {'selected_y': 0, 'truth_min': 0, 'truth_q90': 0, 'raw_regret': 0, 'normalized_regret': 0} |
| finite_summary_values | PASS | {'initial_best_y': 0, 'final_best_y': 0, 'known_optimum_y': 0, 'final_normalized_regret': 0, 'auc_normalized_regret': 0, 'total_improvement': 0} |
| zero_failures | PASS | failure_rows=0 |
| mechanism_panel_semantics | PASS | all panel-derived mechanism fields match |
| trace_and_summary_semantics | PASS | all trace and summary fields match |
| mechanism_candidate_hashes_shared_across_five_methods | PASS | columns=['raw_pool_hash', 'proposal_hash', 'truth_hash'], violations=0 |
| source_structure_diagnostics_present | PASS | rows=81, missing=[], extra=[] |
| target_only_zero_based_acquisition_rank | PASS | runner uses zero-based rank; observed=[0] |
| fallback_consistency | PASS | all fallback selections equal Target-Only |
| analysis_exists | PASS | [] |
| primary_semantics | PASS | five PRIMARY contrasts and Holm/support status match analyzer |

## Warnings

- target_noise_std=0: runner noise stream bug is documented and does not affect this full run
- known optimum is a declared oracle basin-center approximation, not a separately verified global optimum

## Counts

- `instances`: 64
- `mechanism_rows`: 320
- `mechanism_candidate_rows`: 6400
- `sequential_summary_rows`: 256
- `sequential_trace_rows`: 5120
- `sequential_initial_rows`: 256
- `sequential_trace_rows_total`: 5376
- `budget`: 20
- `proposal_size`: 100
- `mechanism_rows_observed`: 320
- `mechanism_panel_rows_observed`: 6400
- `sequential_summary_rows_observed`: 256
- `sequential_trace_rows_observed`: 5376

## Artifact SHA-256

- `results\unified_local_guidance_full\mechanism_results.csv`: `00077ae76e6edd8ca81232da95397ee27ef2036d888a8b02a19eea31c4aec1bd`
- `results\unified_local_guidance_full\mechanism_candidate_panel.csv`: `bfe707550d18f4df11ca0a336bb1d4eba7c671e6a62f8d08966977599736dabe`
- `results\unified_local_guidance_full\source_structure_diagnostics.csv`: `1c6646f210de48af8590ea834391e411ed519edf23737dcd25ef7872579c0711`
- `results\unified_local_guidance_full\sequential_summary.csv`: `f18d98e6fe6a48648789f30ce6cade96c22b9d576f728e9acd83896e6b0f18d3`
- `results\unified_local_guidance_full\sequential_traces.csv`: `e13b29966079494b25ea0db9eac3a795ab1838fec09dc633a5355793fb215f53`
- `results\unified_local_guidance_full\failures.csv`: `1cba120c60350efa5fc22d87c23a24506c0d8292d25777ee122da29b2594dbd2`
- `results\unified_local_guidance_full\config.json`: `1df4c6b307bf34c8b6da611c9996d7b79dd0a5b474f07a551fcc6a33b90e8630`
- `results\unified_local_guidance_full\analysis\PRIMARY_TESTS.csv`: `edfa5a7daf5a8ac5c72ebbcb78ca40a06cca9af11d61da487353c0d81716bab7`
- `results\unified_local_guidance_full\analysis\METHOD_SUMMARY.csv`: `d46ee0b8562f7d14a4543c0309f07f715a9d59e5ed26f2cbe8e5c260a5a66bb2`
- `results\unified_local_guidance_full\analysis\SECONDARY_CONTRASTS.csv`: `724f55f02ed52532d8311d9d643ea65d66b47b362e5cbab7aedd6a1468352076`
- `results\unified_local_guidance_full\analysis\mechanism_regret.png`: `6fded2552bb1220d41adb4117b5c68b6aad8b990ab0d0a5e12c072e718717843`
- `results\unified_local_guidance_full\analysis\sequential_final.png`: `d2fdc6de4477f13f654ffaea4401115bec79ceb91a6c2facb56fba912701507d`
- `results\unified_local_guidance_full\analysis\sequential_auc.png`: `8c465e9b4774d8874751418890acb52ac8f7751051f015e272494183864f9ebd`
