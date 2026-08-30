# Local-Surrogate Transfer Pilot v1 Audit

- Stage identity: `local-surrogate-transfer-pilot-v1-full`
- Overall audit status: **PASS**

## Checks

| Check | Status | Detail |
|---|---:|---|
| stage_identity | PASS | manifest=local-surrogate-transfer-pilot-v1-full, config=local-surrogate-transfer-pilot-v1-full |
| config_hash | PASS | expected=d3179431b59ac6a16128bf47e40ad2d6ab232ea6d2ba14c768b217a29f229a1d, observed=d3179431b59ac6a16128bf47e40ad2d6ab232ea6d2ba14c768b217a29f229a1d |
| protocol_hash | PASS | expected=b548f951b1f22346046cdfe5a5e647053a53cbc311c2f6fe4728567945b454fc, observed=b548f951b1f22346046cdfe5a5e647053a53cbc311c2f6fe4728567945b454fc |
| pre_analysis_artifact_hashes | PASS | all match |
| expected_result_rows | PASS | expected=2880, observed=2880 |
| expected_diagnostic_rows | PASS | expected=192, observed=192 |
| expected_target_ledger_rows | PASS | expected=50432, observed=50432 |
| zero_failures | PASS | failure_rows=0 |
| unique_result_keys | PASS | duplicates=0 |
| unique_diagnostic_keys | PASS | duplicates=0 |
| unique_ledger_keys | PASS | duplicates=0 |
| complete_method_relation_context_levels | PASS | methods=['Calibrated-Source+Residual', 'Fixed-Source+Residual', 'Gated-Source+Residual', 'Source-Affine-Only', 'Target-Only'], relations=['matching', 'reversed', 'wrong'], contexts=[6, 12, 20] |
| finite_required_result_metrics | PASS | {"interval_coverage_95": 0, "mean_negative_log_likelihood": 0, "ndcg_at_top": 0, "normalized_top1_regret": 0, "pairwise_accuracy": 0, "precision_at_top": 0, "source_fidelity_ndcg": 0, "source_fidelity_pairwise": 0, "source_fidelity_spearman": 0, "source_membership_below_0_05": 0, "source_membership_mean": 0, "source_membership_min": 0, "spearman": 0, "srmse_delta_vs_target_only": 0, "standardized_rmse": 0} |
| four_finite_primary_tests | PASS | rows=4, nonfinite={'n_pairs': 0, 'mean_advantage': 0, 'ci_low': 0, 'ci_high': 0, 'wilcoxon_one_sided_p': 0, 'rank_biserial': 0, 'holm_adjusted_p': 0} |
| target_only_relation_invariance | PASS | maximum metric spread=0.000e+00 |
| target_only_zero_delta | PASS | maximum absolute delta=0.000e+00 |
| rejected_gate_exact_fallback | PASS | rejected=452, maximum metric difference=0.000e+00 |
| nonpositive_calibration_exact_fallback | PASS | nonpositive=263, maximum metric difference=0.000e+00 |
| reversed_is_paired_counterfactual | PASS | paired=64, mismatches=0 |
| context_test_design_disjoint | PASS | instances_with_overlap=0 |
| target_ledger_panel_counts | PASS | context_range=(20,20), test_range=(768,768) |
| zero_noise_ledger_consistency | PASS | maximum difference=0.000e+00 |
| shared_target_artifact_hashes | PASS | violating_instances=0 |

## Counts

- `expected_instances`: 64
- `result_rows`: 2880
- `diagnostic_rows`: 192
- `target_ledger_rows`: 50432
- `failure_rows`: 0
- `primary_test_rows`: 4

## Artifact SHA-256

- `results\local_surrogate_transfer_pilot\local_surrogate_transfer_manifest.json`: `335a869da9e2e697f3968d843b1fed14ab33160ed6a97553bf3e5cf36be85842`
- `results\local_surrogate_transfer_pilot\local_surrogate_transfer_results.csv`: `663871728930fcb89af2f85aad28914d712e21d0fc74765b70fb1e6193f00ecf`
- `results\local_surrogate_transfer_pilot\local_surrogate_transfer_diagnostics.csv`: `fdc8fd5e3e0cc1ac3e7955838a786050e98730dc164746f08952c435f257370f`
- `results\local_surrogate_transfer_pilot\local_surrogate_transfer_target_ledger.csv`: `cd54d1be4c0f681f424b64654258779e9b28cd8f7c8ebbd43e9cab0eb66a0058`
- `results\local_surrogate_transfer_pilot\local_surrogate_transfer_failures.csv`: `1cba120c60350efa5fc22d87c23a24506c0d8292d25777ee122da29b2594dbd2`
- `results\local_surrogate_transfer_pilot\analysis\local_surrogate_transfer_primary_tests.csv`: `5f28916062eb56854a29c09925e5afbd3816c373d42cee6276c4b4101880bf47`
- `results\local_surrogate_transfer_pilot\analysis\local_surrogate_transfer_summary.csv`: `e56fce37db806ec7987961257bcad1b9552f671fa392fc987069a6e2d72c87a7`
- `results\local_surrogate_transfer_pilot\analysis\local_surrogate_transfer_gate_summary.csv`: `e016fddeb5e31cfb30d48cc4b51c1a1a468664eee5731be07d49f84910138508`
- `results\local_surrogate_transfer_pilot\analysis\LOCAL_SURROGATE_TRANSFER_REPORT.md`: `476ac3dff54156b3c49b1089f97e4cf781b35637a13a11cdf611bd3b4974dd8e`
- `results\local_surrogate_transfer_pilot\LOCAL_SURROGATE_TRANSFER_DECISION_CN.md`: `7065a38aa1fb6ff0a6384b47e66cb492e575ff2563acd2ce35898d9b46a77aea`
- `results\local_surrogate_transfer_pilot\analysis\local_surrogate_transfer_gate.png`: `7f070708967d818c9be0f62994f106cffc079b65d294f953b9359141d3be368d`
- `results\local_surrogate_transfer_pilot\analysis\local_surrogate_transfer_matching_curves.png`: `7d879bf26862a6a467b843b538f3e7b9d33bfdbb2f689ef88868edcace892655`
- `results\local_surrogate_transfer_pilot\analysis\local_surrogate_transfer_reversed_curves.png`: `aff030736c801402353a8312f8c711092f831951eba0c9136354172896a60449`
- `results\local_surrogate_transfer_pilot\analysis\local_surrogate_transfer_wrong_curves.png`: `796e90d16d96b81418f7e3fb66050b78255bd6b31f99984b5c2a2344f4c70c55`
