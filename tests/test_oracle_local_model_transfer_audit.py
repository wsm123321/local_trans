"""Mutation tests for the Gate-0 quick auditor."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
import audit_oracle_local_model_transfer_quick as auditor  # noqa: E402
from test_oracle_local_model_transfer_analysis import _fixture  # noqa: E402
import analyze_oracle_local_model_transfer_quick as analyzer  # noqa: E402


def _prepared(tmp_path: Path) -> Path:
    d = _fixture(tmp_path)
    # Add the optional outer hashes before analysis so the analyzer's source
    # manifest records the finalized runner manifest bytes.
    manifest_path = d / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = [d / "results.csv", d / "prediction_ledger.csv", d / "source_expert_diagnostics.csv", d / "failures.csv"]
    manifest["artifact_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    analyzer.run_analysis(d)
    return d


def test_audit_passes_clean_outputs(tmp_path: Path) -> None:
    d = _prepared(tmp_path)
    payload = auditor.run_audit(d)
    assert payload["ok"]
    assert all(item["passed"] for item in payload["checks"])


def test_metric_mutation_fails_hash_or_semantic_audit(tmp_path: Path) -> None:
    d = _prepared(tmp_path)
    p = d / "results.csv"
    frame = pd.read_csv(p); frame.loc[0, "standardized_rmse"] += .25; frame.to_csv(p, index=False)
    with pytest.raises(RuntimeError): auditor.run_audit(d)


def test_prediction_mutation_fails_recomputed_metrics(tmp_path: Path) -> None:
    d = _prepared(tmp_path)
    p = d / "prediction_ledger.csv"
    frame = pd.read_csv(p); frame.loc[0, "predicted_mean"] += 5.; frame.to_csv(p, index=False)
    with pytest.raises(RuntimeError): auditor.run_audit(d)


def test_conclusion_mutation_fails_recomputed_decision(tmp_path: Path) -> None:
    d = _prepared(tmp_path)
    p = d / "ORACLE_LOCAL_MODEL_TRANSFER_QUICK_CONCLUSION_CN.md"
    text = p.read_text(encoding="utf-8").replace("quick", "formal", 1)
    p.write_text(text, encoding="utf-8")
    with pytest.raises(RuntimeError): auditor.run_audit(d)


def test_permutation_condition_cannot_be_relabelled_as_identity(tmp_path: Path) -> None:
    d = _prepared(tmp_path)
    p = d / "results.csv"; frame = pd.read_csv(p)
    frame.loc[frame["relation_or_control"] == "identity_label_permutation", "relation"] = "output_affine"
    frame.to_csv(p, index=False)
    with pytest.raises(RuntimeError): auditor.run_audit(d)


def _refresh_artifact_hash(manifest_path: Path, logical: str, path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][logical] = digest
    for declared_path in list(manifest.get("artifact_sha256", {})):
        if Path(declared_path).name == path.name:
            manifest["artifact_sha256"][declared_path] = digest
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    analysis_manifest_path = manifest_path.parent / "analysis" / "analysis_manifest.json"
    if analysis_manifest_path.exists():
        analysis_manifest = json.loads(analysis_manifest_path.read_text(encoding="utf-8"))
        for entry in analysis_manifest.get("source_inputs", {}).values():
            if Path(entry.get("path", "")).name == path.name:
                entry["sha256"] = digest
        analysis_manifest_path.write_text(json.dumps(analysis_manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def test_exact_schema_mutations_fail_with_refreshed_artifact_hash(tmp_path: Path) -> None:
    d = _prepared(tmp_path)
    p = d / "prediction_ledger.csv"; frame = pd.read_csv(p); frame.loc[0, "candidate_index"] = 128; frame.to_csv(p, index=False)
    _refresh_artifact_hash(d / "run_manifest.json", "prediction_ledger", p)
    with pytest.raises(RuntimeError): auditor.run_audit(d)


def test_npz_source_y_mutation_fails_with_refreshed_artifact_hash(tmp_path: Path) -> None:
    d = _prepared(tmp_path); source = d / "reproducibility_inputs.npz"; arrays = dict(np.load(source, allow_pickle=False)); arrays["seed_11_source_y_identity"] = arrays["seed_11_source_y_identity"].copy(); arrays["seed_11_source_y_identity"][0] += 1.; np.savez_compressed(source, **arrays)
    _refresh_artifact_hash(d / "run_manifest.json", "reproducibility_inputs", source)
    with pytest.raises(RuntimeError): auditor.run_audit(d)


def test_diagnostic_oob_mutation_fails_with_refreshed_artifact_hash(tmp_path: Path) -> None:
    d = _prepared(tmp_path); p = d / "source_expert_diagnostics.csv"; frame = pd.read_csv(p); frame.loc[0, "source_test_oob_count"] += 1; frame.to_csv(p, index=False)
    _refresh_artifact_hash(d / "run_manifest.json", "source_expert_diagnostics", p)
    with pytest.raises(RuntimeError): auditor.run_audit(d)


def test_missing_manifest_dependency_fails(tmp_path: Path) -> None:
    d = _prepared(tmp_path); p = d / "run_manifest.json"; manifest = json.loads(p.read_text(encoding="utf-8")); del manifest["dependencies"]["runner"]; p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with pytest.raises(RuntimeError): auditor.run_audit(d)


def test_config_mutation_fails_even_when_manifest_config_is_refreshed(tmp_path: Path) -> None:
    d = _prepared(tmp_path); p = d / "config.json"; config = json.loads(p.read_text(encoding="utf-8")); config["top_fraction"] = .2; p.write_text(json.dumps(config), encoding="utf-8")
    manifest_path = d / "run_manifest.json"; manifest = json.loads(manifest_path.read_text(encoding="utf-8")); manifest["artifacts"]["config"] = hashlib.sha256(p.read_bytes()).hexdigest(); manifest["config_sha256"] = manifest["artifacts"]["config"]; manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with pytest.raises(RuntimeError): auditor.run_audit(d)


def test_relation_relabel_fails_with_refreshed_artifact_hash(tmp_path: Path) -> None:
    d = _prepared(tmp_path); p = d / "results.csv"; frame = pd.read_csv(p); frame.loc[0, "relation_or_control"] = "identity_label_permutation"; frame.to_csv(p, index=False)
    _refresh_artifact_hash(d / "run_manifest.json", "results", p)
    with pytest.raises(RuntimeError): auditor.run_audit(d)


def test_chart_point_swap_fails_semantic_identity_after_hash_refresh(tmp_path: Path) -> None:
    d = _prepared(tmp_path); p = d / "prediction_ledger.csv"; frame = pd.read_csv(p)
    same = (frame.seed == frame.seed.iloc[0]) & (frame.relation_or_control == frame.relation_or_control.iloc[0]) & (frame.shell == frame.shell.iloc[0])
    rows = frame.index[same & (frame.method == "Target-Only")].tolist(); frame.loc[rows[0], "chart_point"], frame.loc[rows[1], "chart_point"] = frame.loc[rows[1], "chart_point"], frame.loc[rows[0], "chart_point"]; frame.to_csv(p, index=False)
    _refresh_artifact_hash(d / "run_manifest.json", "prediction_ledger", p)
    with pytest.raises(RuntimeError): auditor.run_audit(d)


def test_truth_mutation_fails_npz_identity_after_hash_refresh(tmp_path: Path) -> None:
    d = _prepared(tmp_path); p = d / "prediction_ledger.csv"; frame = pd.read_csv(p); frame.loc[0, "truth"] += 1.; frame.to_csv(p, index=False)
    _refresh_artifact_hash(d / "run_manifest.json", "prediction_ledger", p)
    with pytest.raises(RuntimeError): auditor.run_audit(d)


def test_permutation_mutation_and_invalid_permutation_fail(tmp_path: Path) -> None:
    d = _prepared(tmp_path); p = d / "reproducibility_inputs.npz"; arrays = dict(np.load(p, allow_pickle=False)); arrays["seed_11_permutation"] = arrays["seed_11_permutation"].copy(); arrays["seed_11_permutation"][0] = 128; np.savez_compressed(p, **arrays)
    _refresh_artifact_hash(d / "run_manifest.json", "reproducibility_inputs", p)
    with pytest.raises(RuntimeError): auditor.run_audit(d)


def test_source_query_hash_mutation_fails_after_hash_refresh(tmp_path: Path) -> None:
    d = _prepared(tmp_path); p = d / "results.csv"; frame = pd.read_csv(p); frame.loc[0, "source_test_query_hash"] = "tampered"; frame.to_csv(p, index=False)
    _refresh_artifact_hash(d / "run_manifest.json", "results", p)
    with pytest.raises(RuntimeError): auditor.run_audit(d)


def test_old_diagnostic_name_fails_schema_audit(tmp_path: Path) -> None:
    d = _prepared(tmp_path); p = d / "source_expert_diagnostics.csv"; frame = pd.read_csv(p).rename(columns={"source_value_pairwise_accuracy": "source_value_rmse"}); frame.to_csv(p, index=False)
    _refresh_artifact_hash(d / "run_manifest.json", "source_expert_diagnostics", p)
    with pytest.raises(RuntimeError): auditor.run_audit(d)


def test_missing_manifest_artifact_fails_even_when_files_exist(tmp_path: Path) -> None:
    d = _prepared(tmp_path); p = d / "run_manifest.json"; manifest = json.loads(p.read_text(encoding="utf-8")); del manifest["artifacts"]["failures"]; p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with pytest.raises(RuntimeError): auditor.run_audit(d)


def test_model_seed_mutation_fails_lineage_audit(tmp_path: Path) -> None:
    d = _prepared(tmp_path); p = d / "results.csv"; frame = pd.read_csv(p); frame.loc[0, "model_seed"] += 1; frame.to_csv(p, index=False)
    _refresh_artifact_hash(d / "run_manifest.json", "results", p)
    with pytest.raises(RuntimeError): auditor.run_audit(d)


def test_missing_single_result_cell_fails_exact_cartesian_audit(tmp_path: Path) -> None:
    d = _prepared(tmp_path); p = d / "results.csv"; frame = pd.read_csv(p); frame = frame.iloc[1:]; frame.to_csv(p, index=False)
    _refresh_artifact_hash(d / "run_manifest.json", "results", p)
    with pytest.raises(RuntimeError): auditor.run_audit(d)


def test_fallback_prediction_mutation_fails_even_without_gate_flag(tmp_path: Path) -> None:
    d = _prepared(tmp_path)
    p = d / "results.csv"; frame = pd.read_csv(p)
    fallback = (frame["method"] == "Oracle-Rank+Residual").idxmax()
    frame.loc[fallback, "effective_mode"] = "target_only"; frame.to_csv(p, index=False)
    ledger = d / "prediction_ledger.csv"; lf = pd.read_csv(ledger)
    row = frame.loc[fallback]
    mask = (lf["seed"] == row["seed"]) & (lf["relation_or_control"] == row["relation_or_control"]) & (lf["shell"] == row["shell"]) & (lf["method"] == "Oracle-Rank+Residual")
    lf.loc[mask, "predicted_mean"] += 1.; lf.to_csv(ledger, index=False)
    with pytest.raises(RuntimeError): auditor.run_audit(d)
