"""Reproducibility and semantic audit for Local-Surrogate Transfer Pilot v1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
METHODS = [
    "Target-Only",
    "Source-Affine-Only",
    "Fixed-Source+Residual",
    "Calibrated-Source+Residual",
    "Gated-Source+Residual",
]
METRIC_COLUMNS = [
    "standardized_rmse",
    "ndcg_at_top",
    "spearman",
    "pairwise_accuracy",
    "precision_at_top",
    "normalized_top1_regret",
    "mean_negative_log_likelihood",
    "interval_coverage_95",
]


def _resolve_repo_path(path: Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (REPO_ROOT / candidate).resolve()


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run_audit(input_dir: Path, config_path: Path) -> Dict:
    input_dir = _resolve_repo_path(input_dir)
    config_path = _resolve_repo_path(config_path)
    config = load_json(config_path)
    pilot = dict(config["pilot"])
    manifest_path = input_dir / "local_surrogate_transfer_manifest.json"
    results_path = input_dir / "local_surrogate_transfer_results.csv"
    diagnostics_path = input_dir / "local_surrogate_transfer_diagnostics.csv"
    ledger_path = input_dir / "local_surrogate_transfer_target_ledger.csv"
    failures_path = input_dir / "local_surrogate_transfer_failures.csv"
    analysis_dir = input_dir / "analysis"
    primary_path = analysis_dir / "local_surrogate_transfer_primary_tests.csv"
    summary_path = analysis_dir / "local_surrogate_transfer_summary.csv"
    gate_summary_path = analysis_dir / "local_surrogate_transfer_gate_summary.csv"
    report_path = analysis_dir / "LOCAL_SURROGATE_TRANSFER_REPORT.md"
    decision_path = input_dir / "LOCAL_SURROGATE_TRANSFER_DECISION_CN.md"
    plot_paths = sorted(analysis_dir.glob("local_surrogate_transfer_*.png"))

    required = [
        manifest_path,
        results_path,
        diagnostics_path,
        ledger_path,
        failures_path,
        primary_path,
        summary_path,
        gate_summary_path,
        report_path,
    ]
    if decision_path.exists():
        required.append(decision_path)
    required.extend(plot_paths)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required artifacts: {missing}")

    manifest = load_json(manifest_path)
    results = pd.read_csv(results_path)
    diagnostics = pd.read_csv(diagnostics_path)
    ledger = pd.read_csv(ledger_path)
    failures = pd.read_csv(failures_path)
    primary = pd.read_csv(primary_path)

    checks: List[Dict] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    problems = [str(value) for value in pilot["problems"]]
    dimensions = [int(value) for value in pilot["dimensions"]]
    seeds = [int(value) for value in pilot["seeds"]]
    relations = [str(value) for value in pilot["relations"]]
    contexts = [int(value) for value in pilot["target_context_sizes"]]
    expected_instances = len(problems) * len(dimensions) * len(seeds)
    expected_results = expected_instances * len(relations) * len(contexts) * len(METHODS)
    expected_diagnostics = expected_instances * len(relations)
    expected_ledger = expected_instances * (
        max(contexts) + int(pilot["target_test_samples"])
    )

    check(
        "stage_identity",
        manifest.get("stage_id") == config.get("stage_id"),
        f"manifest={manifest.get('stage_id')}, config={config.get('stage_id')}",
    )
    config_sha = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()
    check(
        "config_hash",
        manifest.get("config_sha256") == config_sha,
        f"expected={config_sha}, observed={manifest.get('config_sha256')}",
    )
    protocol_path = REPO_ROOT / str(manifest["protocol_path"])
    protocol_sha = file_sha256(protocol_path)
    check(
        "protocol_hash",
        manifest.get("protocol_sha256") == protocol_sha,
        f"expected={protocol_sha}, observed={manifest.get('protocol_sha256')}",
    )

    artifact_hash_errors: List[str] = []
    for relative, expected_hash in manifest.get("artifact_sha256", {}).items():
        path = REPO_ROOT / relative
        if not path.exists() or file_sha256(path) != expected_hash:
            artifact_hash_errors.append(relative)
    check(
        "pre_analysis_artifact_hashes",
        not artifact_hash_errors,
        "all match" if not artifact_hash_errors else str(artifact_hash_errors),
    )

    check(
        "expected_result_rows",
        len(results) == expected_results,
        f"expected={expected_results}, observed={len(results)}",
    )
    check(
        "expected_diagnostic_rows",
        len(diagnostics) == expected_diagnostics,
        f"expected={expected_diagnostics}, observed={len(diagnostics)}",
    )
    check(
        "expected_target_ledger_rows",
        len(ledger) == expected_ledger,
        f"expected={expected_ledger}, observed={len(ledger)}",
    )
    check(
        "zero_failures",
        len(failures) == 0,
        f"failure_rows={len(failures)}",
    )

    result_key = [
        "problem",
        "dim",
        "seed",
        "relation",
        "target_context_size",
        "method",
    ]
    diagnostic_key = ["problem", "dim", "seed", "relation"]
    ledger_key = ["problem", "dim", "seed", "panel", "point_index"]
    check(
        "unique_result_keys",
        not results.duplicated(result_key).any(),
        f"duplicates={int(results.duplicated(result_key).sum())}",
    )
    check(
        "unique_diagnostic_keys",
        not diagnostics.duplicated(diagnostic_key).any(),
        f"duplicates={int(diagnostics.duplicated(diagnostic_key).sum())}",
    )
    check(
        "unique_ledger_keys",
        not ledger.duplicated(ledger_key).any(),
        f"duplicates={int(ledger.duplicated(ledger_key).sum())}",
    )

    observed_methods = set(results["method"].astype(str))
    observed_relations = set(results["relation"].astype(str))
    observed_contexts = set(results["target_context_size"].astype(int))
    check(
        "complete_method_relation_context_levels",
        observed_methods == set(METHODS)
        and observed_relations == set(relations)
        and observed_contexts == set(contexts),
        f"methods={sorted(observed_methods)}, relations={sorted(observed_relations)}, contexts={sorted(observed_contexts)}",
    )

    numeric_required = METRIC_COLUMNS + [
        "srmse_delta_vs_target_only",
        "source_fidelity_ndcg",
        "source_fidelity_spearman",
        "source_fidelity_pairwise",
        "source_membership_mean",
        "source_membership_min",
        "source_membership_below_0_05",
    ]
    nonfinite = {
        column: int((~np.isfinite(pd.to_numeric(results[column], errors="coerce"))).sum())
        for column in numeric_required
    }
    check(
        "finite_required_result_metrics",
        all(value == 0 for value in nonfinite.values()),
        json.dumps(nonfinite, sort_keys=True),
    )

    primary_numeric = [
        "n_pairs",
        "mean_advantage",
        "ci_low",
        "ci_high",
        "wilcoxon_one_sided_p",
        "rank_biserial",
        "holm_adjusted_p",
    ]
    primary_nonfinite = {
        column: int((~np.isfinite(pd.to_numeric(primary[column], errors="coerce"))).sum())
        for column in primary_numeric
    }
    check(
        "four_finite_primary_tests",
        len(primary) == 4 and all(value == 0 for value in primary_nonfinite.values()),
        f"rows={len(primary)}, nonfinite={primary_nonfinite}",
    )

    # The target-only model must be bitwise-equivalent at metric precision across
    # source relations, because source knowledge is not an input to that baseline.
    target_only = results[results["method"] == "Target-Only"].copy()
    target_groups = target_only.groupby(
        ["problem", "dim", "seed", "target_context_size"],
        sort=False,
    )
    max_target_spread = 0.0
    for _, group in target_groups:
        for column in METRIC_COLUMNS:
            values = group[column].astype(float).to_numpy()
            max_target_spread = max(max_target_spread, float(np.ptp(values)))
    check(
        "target_only_relation_invariance",
        max_target_spread <= 1e-14,
        f"maximum metric spread={max_target_spread:.3e}",
    )

    target_delta = np.max(np.abs(target_only["srmse_delta_vs_target_only"].astype(float)))
    check(
        "target_only_zero_delta",
        target_delta <= 1e-14 and not target_only["negative_transfer"].astype(bool).any(),
        f"maximum absolute delta={target_delta:.3e}",
    )

    # Every rejected gate must be the exact Target-Only fallback in all reported
    # predictions and uncertainty metrics.
    gated = results[results["method"] == "Gated-Source+Residual"].copy()
    rejected = gated[~gated["gate_accepted"].astype(bool)]
    target_reference = target_only[result_key[:-1] + METRIC_COLUMNS].copy()
    merged_rejected = rejected.merge(
        target_reference,
        on=result_key[:-1],
        suffixes=("_gated", "_target"),
        how="left",
    )
    max_fallback_difference = 0.0
    for column in METRIC_COLUMNS:
        difference = np.abs(
            merged_rejected[f"{column}_gated"].astype(float)
            - merged_rejected[f"{column}_target"].astype(float)
        )
        if len(difference):
            max_fallback_difference = max(
                max_fallback_difference,
                float(np.max(difference)),
            )
    check(
        "rejected_gate_exact_fallback",
        len(merged_rejected) == len(rejected) and max_fallback_difference <= 1e-14,
        f"rejected={len(rejected)}, maximum metric difference={max_fallback_difference:.3e}",
    )

    # Non-positive calibrated associations must also be exact Target-Only fallbacks.
    calibrated = results[
        results["method"] == "Calibrated-Source+Residual"
    ].copy()
    nonpositive = calibrated[calibrated["calibration_raw_slope"] <= 0.0]
    merged_nonpositive = nonpositive.merge(
        target_reference,
        on=result_key[:-1],
        suffixes=("_calibrated", "_target"),
        how="left",
    )
    max_nonpositive_difference = 0.0
    for column in METRIC_COLUMNS:
        difference = np.abs(
            merged_nonpositive[f"{column}_calibrated"].astype(float)
            - merged_nonpositive[f"{column}_target"].astype(float)
        )
        if len(difference):
            max_nonpositive_difference = max(
                max_nonpositive_difference,
                float(np.max(difference)),
            )
    check(
        "nonpositive_calibration_exact_fallback",
        len(merged_nonpositive) == len(nonpositive)
        and max_nonpositive_difference <= 1e-14,
        f"nonpositive={len(nonpositive)}, maximum metric difference={max_nonpositive_difference:.3e}",
    )

    # Matching and reversed conditions must be counterfactual transformations of
    # the identical fitted source structure, not independently refitted experts.
    if {"matching", "reversed"}.issubset(set(relations)):
        matching_diag = diagnostics[diagnostics["relation"] == "matching"].copy()
        reversed_diag = diagnostics[diagnostics["relation"] == "reversed"].copy()
        pair_keys = ["problem", "dim", "seed"]
        paired_diag = matching_diag.merge(
            reversed_diag,
            on=pair_keys,
            suffixes=("_matching", "_reversed"),
        )
        same_expert = (
            paired_diag["source_region_id_matching"].astype(str)
            == paired_diag["source_region_id_reversed"].astype(str)
        ) & (
            paired_diag["matching_source_data_hash_matching"].astype(str)
            == paired_diag["matching_source_data_hash_reversed"].astype(str)
        )
        check(
            "reversed_is_paired_counterfactual",
            len(paired_diag) == expected_instances and bool(same_expert.all()),
            f"paired={len(paired_diag)}, mismatches={int((~same_expert).sum())}",
        )

    # Audit target design separation directly from persisted coordinates.
    overlap_instances = 0
    for _, group in ledger.groupby(["problem", "dim", "seed"], sort=False):
        context_points = {
            tuple(np.round(json.loads(value), 12))
            for value in group[group["panel"] == "context"]["chart_point"]
        }
        test_points = {
            tuple(np.round(json.loads(value), 12))
            for value in group[group["panel"] == "test"]["chart_point"]
        }
        if context_points.intersection(test_points):
            overlap_instances += 1
    check(
        "context_test_design_disjoint",
        overlap_instances == 0,
        f"instances_with_overlap={overlap_instances}",
    )

    context_counts = ledger[ledger["panel"] == "context"].groupby(
        ["problem", "dim", "seed"]
    ).size()
    test_counts = ledger[ledger["panel"] == "test"].groupby(
        ["problem", "dim", "seed"]
    ).size()
    check(
        "target_ledger_panel_counts",
        len(context_counts) == expected_instances
        and len(test_counts) == expected_instances
        and bool((context_counts == max(contexts)).all())
        and bool((test_counts == int(pilot["target_test_samples"])).all()),
        f"context_range=({context_counts.min()},{context_counts.max()}), test_range=({test_counts.min()},{test_counts.max()})",
    )

    if float(pilot.get("target_observation_noise_std", 0.0)) == 0.0:
        max_noise_difference = float(
            np.max(
                np.abs(
                    ledger["clean_target_y"].astype(float)
                    - ledger["observed_target_y"].astype(float)
                )
            )
        )
        check(
            "zero_noise_ledger_consistency",
            max_noise_difference <= 1e-14,
            f"maximum difference={max_noise_difference:.3e}",
        )

    # Hash columns must be constant within an independent target instance.
    hash_columns = [
        "target_context_design_hash",
        "target_test_design_hash",
        "target_test_truth_hash",
    ]
    hash_violations = 0
    for _, group in results.groupby(["problem", "dim", "seed"], sort=False):
        if any(group[column].nunique(dropna=False) != 1 for column in hash_columns):
            hash_violations += 1
    check(
        "shared_target_artifact_hashes",
        hash_violations == 0,
        f"violating_instances={hash_violations}",
    )

    passed = all(item["passed"] for item in checks)
    hashed_artifacts = [path for path in required if path.exists()]
    artifact_hashes = {
        str(path.relative_to(REPO_ROOT)): file_sha256(path)
        for path in hashed_artifacts
    }
    payload = {
        "ok": passed,
        "stage_id": config.get("stage_id"),
        "checks": checks,
        "counts": {
            "expected_instances": expected_instances,
            "result_rows": len(results),
            "diagnostic_rows": len(diagnostics),
            "target_ledger_rows": len(ledger),
            "failure_rows": len(failures),
            "primary_test_rows": len(primary),
        },
        "artifact_sha256": artifact_hashes,
    }
    audit_json_path = input_dir / "AUDIT.json"
    with audit_json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    _write_audit_markdown(input_dir / "FULL_RUN_AUDIT.md", payload)
    if not passed:
        failed_names = [item["name"] for item in checks if not item["passed"]]
        raise RuntimeError(f"Audit failed: {failed_names}")
    return payload


def _write_audit_markdown(path: Path, payload: Mapping) -> None:
    lines = [
        "# Local-Surrogate Transfer Pilot v1 Audit",
        "",
        f"- Stage identity: `{payload['stage_id']}`",
        f"- Overall audit status: **{'PASS' if payload['ok'] else 'FAIL'}**",
        "",
        "## Checks",
        "",
        "| Check | Status | Detail |",
        "|---|---:|---|",
    ]
    for item in payload["checks"]:
        detail = str(item["detail"]).replace("|", "\\|")
        lines.append(
            f"| {item['name']} | {'PASS' if item['passed'] else 'FAIL'} | {detail} |"
        )
    lines.extend(["", "## Counts", ""])
    for key, value in payload["counts"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Artifact SHA-256", ""])
    for relative, digest in payload["artifact_sha256"].items():
        lines.append(f"- `{relative}`: `{digest}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "results" / "local_surrogate_transfer_pilot_quick",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "local_surrogate_transfer_quick.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_audit(arguments.input, arguments.config)
