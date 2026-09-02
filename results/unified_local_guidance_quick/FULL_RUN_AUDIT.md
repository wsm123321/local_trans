# Unified local-guidance full-run audit

- Overall status: **PASS**
- Stage: `unified-local-guidance-quick`

## Checks

| Check | Status | Detail |
|---|---|---|
| approved_artifact_paths | PASS | all canonical files present |
| json_readable | PASS | config and manifest parse as objects |
| csv_readable | PASS | ['failures', 'mechanism', 'panel', 'sequential_summary', 'sequential_traces', 'source_diagnostics'] |
| expected_cartesian_instances_from_config | PASS | instances=4 |
| config_hash | PASS | expected=74928eb1ed255df219c81c21589f350d66d87d8b7c7c540e12023237636eebea, observed=74928eb1ed255df219c81c21589f350d66d87d8b7c7c540e12023237636eebea |
| stage_identity | PASS | config=unified-local-guidance-quick, manifest=unified-local-guidance-quick |
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
| mechanism_row_count | PASS | expected=20, observed=20 |
| mechanism_panel_row_count | PASS | expected=160, observed=160 |
| sequential_summary_row_count | PASS | expected=16, observed=16 |
| sequential_trace_row_count | PASS | expected_paid=80, observed_paid=80, initial_rows=16, total=96 |
| finite_mechanism_values | PASS | {'selected_y': 0, 'truth_min': 0, 'truth_q90': 0, 'raw_regret': 0, 'normalized_regret': 0} |
| finite_summary_values | PASS | {'initial_best_y': 0, 'final_best_y': 0, 'known_optimum_y': 0, 'final_normalized_regret': 0, 'auc_normalized_regret': 0, 'total_improvement': 0} |
| zero_failures | PASS | failure_rows=0 |
| mechanism_panel_semantics | PASS | all panel-derived mechanism fields match |
| trace_and_summary_semantics | PASS | all trace and summary fields match |
| mechanism_candidate_hashes_shared_across_five_methods | PASS | columns=['raw_pool_hash', 'proposal_hash', 'truth_hash'], violations=0 |
| source_structure_diagnostics_present | PASS | rows=9, missing=[], extra=[] |
| target_only_zero_based_acquisition_rank | PASS | runner uses zero-based rank; observed=[0] |
| fallback_consistency | PASS | all fallback selections equal Target-Only |
| analysis_exists | PASS | [] |
| primary_semantics | PASS | five PRIMARY contrasts and Holm/support status match analyzer |

## Warnings

- target_noise_std=0: runner noise stream bug is documented and does not affect this full run
- known optimum is a declared oracle basin-center approximation, not a separately verified global optimum

## Counts

- `instances`: 4
- `mechanism_rows`: 20
- `mechanism_candidate_rows`: 160
- `sequential_summary_rows`: 16
- `sequential_trace_rows`: 80
- `sequential_initial_rows`: 16
- `sequential_trace_rows_total`: 96
- `budget`: 5
- `proposal_size`: 40
- `mechanism_rows_observed`: 20
- `mechanism_panel_rows_observed`: 160
- `sequential_summary_rows_observed`: 16
- `sequential_trace_rows_observed`: 96

## Artifact SHA-256

- `results\unified_local_guidance_quick\mechanism_results.csv`: `daedd31b897fad42478acf7e7559040684d87358fd7a6e73cdc6d3e44252781f`
- `results\unified_local_guidance_quick\mechanism_candidate_panel.csv`: `bcaa6849243cdb63c0acead38681c76cc1e76a0b8d8ed383b5b8eaee287c250e`
- `results\unified_local_guidance_quick\source_structure_diagnostics.csv`: `72338fb8f273e88714d394fdc3e5867e285cc325e9a7760c084a1118c033ce5a`
- `results\unified_local_guidance_quick\sequential_summary.csv`: `d0c0cc2adab04cbca07f1881f1a31204074d5bee683d624d405b298725c95ab6`
- `results\unified_local_guidance_quick\sequential_traces.csv`: `1e2f75eb4f2f35340655e778dd0e74d677e86240c242f65edd26c680bafa5e95`
- `results\unified_local_guidance_quick\failures.csv`: `1cba120c60350efa5fc22d87c23a24506c0d8292d25777ee122da29b2594dbd2`
- `results\unified_local_guidance_quick\config.json`: `cbb5f43362c6af6770c7cd65c8f29d05870f49eb732c35725876bc57a983a575`
- `results\unified_local_guidance_quick\analysis\PRIMARY_TESTS.csv`: `e8ed40968a608c1adf9fe188713cbfa318857cca911cb1330b86671aca1dc3fb`
- `results\unified_local_guidance_quick\analysis\METHOD_SUMMARY.csv`: `7dcc1444e49e32d827eaa751e756f1703c9302db5de388c3b2058a750906701a`
- `results\unified_local_guidance_quick\analysis\SECONDARY_CONTRASTS.csv`: `6325433e1a8892c43f7d53ece05a86cf23e28d06b9f407db0c2dd26d5b9d98d4`
- `results\unified_local_guidance_quick\analysis\mechanism_regret.png`: `f0c8a6d3d023e88f5036094fe2f79caff6ca98b488fb63bd077a322a907f144d`
- `results\unified_local_guidance_quick\analysis\sequential_final.png`: `e4ec67e17ac74b2109ae054ae12ca83bdb3ecb8945f9509df87b3e60c699d96b`
- `results\unified_local_guidance_quick\analysis\sequential_auc.png`: `36fef314c831d810e7652e90db14506225fdd8edcdb79653bfeb25a63487794f`
