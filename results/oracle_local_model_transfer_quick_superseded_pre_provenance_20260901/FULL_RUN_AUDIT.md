# Gate-0 Oracle Local-Model Transfer Quick Audit

状态：**PASS**

- PASS `required_manifest`：E:\study\暑假工作\region_guided_reranking_study\results\oracle_local_model_transfer_quick\run_manifest.json
- PASS `required_config`：embedded/missing
- PASS `required_diagnostics`：E:\study\暑假工作\region_guided_reranking_study\results\oracle_local_model_transfer_quick\source_expert_diagnostics.csv
- PASS `required_failures`：E:\study\暑假工作\region_guided_reranking_study\results\oracle_local_model_transfer_quick\failures.csv
- PASS `stage_identity`：manifest=gate-0-oracle-local-model-transfer-quick-v0, config=gate-0-oracle-local-model-transfer-quick-v0
- PASS `artifact_hashes`：all manifest artifacts match
- PASS `config_hash`：expected=e68d4f1ed6aba776bdee048e65b12ceadbf2e4be7297915c29a714ade750c975, observed=e68d4f1ed6aba776bdee048e65b12ceadbf2e4be7297915c29a714ade750c975
- PASS `runner_sha256`：E:\study\暑假工作\region_guided_reranking_study\scripts\run_oracle_local_model_transfer_quick.py
- PASS `core_sha256`：E:\study\暑假工作\region_guided_reranking_study\src\region_guided_reranking_study\oracle_local_model_transfer.py
- PASS `strict_rowcounts`：expected={'result_rows': 1080, 'ledger_rows': 138240, 'diagnostic_rows': 216}, observed={'result_rows': 1080, 'ledger_rows': 138240, 'diagnostic_rows': 216}
- PASS `manifest_rowcounts`：{'result_rows': 1080, 'ledger_rows': 138240, 'diagnostic_rows': 216, 'failure_rows': 0}
- PASS `zero_failures`：failure_rows=0
- PASS `collision_free_condition_key`：relation_or_control required
- PASS `result_keyset_unique`：key=['dimension', 'seed', 'relation_or_control', 'control', 'shell', 'panel', 'method'], duplicates=0
- PASS `ledger_keyset_unique`：key=['dimension', 'seed', 'relation_or_control', 'control', 'shell', 'panel', 'candidate_index', 'method'], duplicates=0
- PASS `panel_sets`：frozen seed/condition/shell/method panel
- PASS `finite_required`：all required numeric values finite
- PASS `shared_context_test_truth_hashes`：violations=0
- PASS `diagnostic_hash_agreement`：violations=0
- PASS `truth_consistency_across_methods`：bad_panels=0
- PASS `ledger_metrics_recomputed`：compared=8640, max_abs_diff=1.631e-12, errors=[]
- PASS `target_only_fallback_exact`：mismatches=0
- PASS `coefficients_nonnegative`：bad_rows=0
- PASS `source_permutation_relation`：bad_rows_or_panels=0
- PASS `analysis_required_files`：[]
- PASS `analysis_summary_recomputed`：observed=(1080, 16), expected=(1080, 16)
- PASS `analysis_contrasts_recomputed`：observed=(12096, 17), expected=(12096, 17)
- PASS `analysis_decision_recomputed`：expected=promising_rank_transfer, observed=promising_rank_transfer
- PASS `conclusion_recomputed`：label=promising_rank_transfer
- PASS `conclusion_hash`：expected=f91f14864a42557a2fe64a74fd01c214df725fe077457adfcfa02ecf9d695de0, observed=f91f14864a42557a2fe64a74fd01c214df725fe077457adfcfa02ecf9d695de0
