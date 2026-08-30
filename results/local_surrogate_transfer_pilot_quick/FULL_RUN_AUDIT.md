# Local-Surrogate Transfer Pilot v1 Audit

- Stage identity: `local-surrogate-transfer-pilot-v1-quick`
- Overall audit status: **PASS**

## Checks

| Check | Status | Detail |
|---|---:|---|
| stage_identity | PASS | manifest=local-surrogate-transfer-pilot-v1-quick, config=local-surrogate-transfer-pilot-v1-quick |
| config_hash | PASS | expected=03d9ddbad16eff4fefe7fc78f828baf5c74f419b33ade03cdcba0bd518a85c4d, observed=03d9ddbad16eff4fefe7fc78f828baf5c74f419b33ade03cdcba0bd518a85c4d |
| protocol_hash | PASS | expected=b548f951b1f22346046cdfe5a5e647053a53cbc311c2f6fe4728567945b454fc, observed=b548f951b1f22346046cdfe5a5e647053a53cbc311c2f6fe4728567945b454fc |
| pre_analysis_artifact_hashes | PASS | all match |
| expected_result_rows | PASS | expected=240, observed=240 |
| expected_diagnostic_rows | PASS | expected=24, observed=24 |
| expected_target_ledger_rows | PASS | expected=2144, observed=2144 |
| zero_failures | PASS | failure_rows=0 |
| unique_result_keys | PASS | duplicates=0 |
| unique_diagnostic_keys | PASS | duplicates=0 |
| unique_ledger_keys | PASS | duplicates=0 |
| complete_method_relation_context_levels | PASS | methods=['Calibrated-Source+Residual', 'Fixed-Source+Residual', 'Gated-Source+Residual', 'Source-Affine-Only', 'Target-Only'], relations=['matching', 'reversed', 'wrong'], contexts=[6, 12] |
| finite_required_result_metrics | PASS | {"interval_coverage_95": 0, "mean_negative_log_likelihood": 0, "ndcg_at_top": 0, "normalized_top1_regret": 0, "pairwise_accuracy": 0, "precision_at_top": 0, "source_fidelity_ndcg": 0, "source_fidelity_pairwise": 0, "source_fidelity_spearman": 0, "source_membership_below_0_05": 0, "source_membership_mean": 0, "source_membership_min": 0, "spearman": 0, "srmse_delta_vs_target_only": 0, "standardized_rmse": 0} |
| four_finite_primary_tests | PASS | rows=4, nonfinite={'n_pairs': 0, 'mean_advantage': 0, 'ci_low': 0, 'ci_high': 0, 'wilcoxon_one_sided_p': 0, 'rank_biserial': 0, 'holm_adjusted_p': 0} |
| target_only_relation_invariance | PASS | maximum metric spread=0.000e+00 |
| target_only_zero_delta | PASS | maximum absolute delta=0.000e+00 |
| rejected_gate_exact_fallback | PASS | rejected=42, maximum metric difference=0.000e+00 |
| nonpositive_calibration_exact_fallback | PASS | nonpositive=25, maximum metric difference=0.000e+00 |
| reversed_is_paired_counterfactual | PASS | paired=8, mismatches=0 |
| context_test_design_disjoint | PASS | instances_with_overlap=0 |
| target_ledger_panel_counts | PASS | context_range=(12,12), test_range=(256,256) |
| zero_noise_ledger_consistency | PASS | maximum difference=0.000e+00 |
| shared_target_artifact_hashes | PASS | violating_instances=0 |

## Counts

- `expected_instances`: 8
- `result_rows`: 240
- `diagnostic_rows`: 24
- `target_ledger_rows`: 2144
- `failure_rows`: 0
- `primary_test_rows`: 4

## Artifact SHA-256

- `results\local_surrogate_transfer_pilot_quick\local_surrogate_transfer_manifest.json`: `af1c663f9b6c8e70e69cf1c72785283ced6c39fbc7c72e1aeb57f60cae9c3da1`
- `results\local_surrogate_transfer_pilot_quick\local_surrogate_transfer_results.csv`: `3f87fb0ca5d1c758ffef22e2f80cc5538d65b65023ecbadcb13028e5be43caa6`
- `results\local_surrogate_transfer_pilot_quick\local_surrogate_transfer_diagnostics.csv`: `e2bd72cc58843a67327ab15b7587cf21278a2ffa8280e82a150ab834b31623bc`
- `results\local_surrogate_transfer_pilot_quick\local_surrogate_transfer_target_ledger.csv`: `547a0d2d86e29c338275ad5d5e25552dd21bf4ecdc337d8ef4ebb18054c3564d`
- `results\local_surrogate_transfer_pilot_quick\local_surrogate_transfer_failures.csv`: `1cba120c60350efa5fc22d87c23a24506c0d8292d25777ee122da29b2594dbd2`
- `results\local_surrogate_transfer_pilot_quick\analysis\local_surrogate_transfer_primary_tests.csv`: `bf3272b7240535128442d0353acbf967a57d9c1a262c2d13db8543e4d9e58c5a`
- `results\local_surrogate_transfer_pilot_quick\analysis\local_surrogate_transfer_summary.csv`: `a0937f814d187caeef6780ea343ab761c5b99a1d8ea0fda44b8f94fdde3a12c2`
- `results\local_surrogate_transfer_pilot_quick\analysis\local_surrogate_transfer_gate_summary.csv`: `258352faa72520e7eb5f669c23483718eb41f55cdbdd5157f10b738bab1b1c5f`
- `results\local_surrogate_transfer_pilot_quick\analysis\LOCAL_SURROGATE_TRANSFER_REPORT.md`: `2f7f03865138b89161d831ce794269cefa848619c943689aa546648f83b10284`
