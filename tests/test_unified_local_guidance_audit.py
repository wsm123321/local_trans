"""Unit and mutation tests for the semantic unified local-guidance audit."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load_script("approved_audit", "audit_unified_local_guidance_study.py")


@pytest.fixture
def canonical_fixture(tmp_path):
    """Copy the existing full fixture and make its manifest self-contained."""
    source = ROOT / "results" / "unified_local_guidance_full"
    for name in (
        "mechanism_results.csv", "mechanism_candidate_panel.csv",
        "source_structure_diagnostics.csv", "sequential_summary.csv",
        "sequential_traces.csv", "failures.csv", "config.json",
        "run_manifest.json",
    ):
        shutil.copy2(source / name, tmp_path / name)
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    analysis_names = (
        "PRIMARY_TESTS.csv", "METHOD_SUMMARY.csv", "SECONDARY_CONTRASTS.csv",
        "mechanism_regret.png", "sequential_final.png", "sequential_auc.png",
    )
    for name in analysis_names:
        shutil.copy2(source / "analysis" / name, analysis / name)
    manifest_path = tmp_path / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_sha256"] = {
        name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        for name in audit.CANONICAL_ARTIFACTS
    }
    manifest["artifact_sha256"].update({
        f"analysis/{name}": hashlib.sha256((analysis / name).read_bytes()).hexdigest()
        for name in analysis_names
    })
    # The code identities in a copied fixture still refer to the real repository.
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return tmp_path


def run_fixture(path):
    return audit.run_audit(path)


def mutate_csv(path: Path, name: str, mutator) -> None:
    artifact_path = path / "analysis" / name if name.startswith("PRIMARY_") else path / name
    frame = pd.read_csv(artifact_path)
    mutator(frame)
    frame.to_csv(artifact_path, index=False)
    manifest_path = path / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_key = f"analysis/{name}" if name.startswith("PRIMARY_") else name
    manifest["artifact_sha256"][manifest_key] = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def failed_checks(path):
    with pytest.raises(RuntimeError):
        run_fixture(path)
    payload = json.loads((path / "AUDIT.json").read_text(encoding="utf-8"))
    return {item["name"] for item in payload["checks"] if not item["passed"]}


def test_exact_method_sets_are_distinct_by_phase():
    assert audit.MECHANISM_METHODS == frozenset({
        "Target-Only", "Geometry-Only", "Local-Rank-No-Reliability",
        "Local-Rank+Reliability", "Reversed-Local-Rank",
    })
    assert audit.SEQUENTIAL_METHODS == frozenset({
        "Target-Only", "Geometry-Only", "Local-Rank-No-Reliability",
        "Local-Rank+Reliability",
    })
    assert audit.SAFETY_METHOD not in audit.SEQUENTIAL_METHODS


def test_target_only_rank_is_zero_based_per_runner_definition():
    frame = pd.DataFrame({"method": ["Target-Only", "Geometry-Only"], "acquisition_rank": [0, 2]})
    ok, detail = audit.target_only_selection_ok(frame)
    assert ok and "zero-based" in detail
    frame.loc[0, "acquisition_rank"] = 1
    ok, _ = audit.target_only_selection_ok(frame)
    assert not ok


def test_shared_hash_checks_only_mechanism_groups():
    frame = pd.DataFrame({
        "problem": ["a"] * 5, "dim": [2] * 5, "seed": [1] * 5,
        "method": sorted(audit.MECHANISM_METHODS),
        "raw_pool_hash": ["pool"] * 5, "proposal_hash": ["proposal"] * 5,
        "truth_hash": ["truth"] * 5,
    })
    assert audit.shared_hash_violations(frame, ["raw_pool_hash", "proposal_hash", "truth_hash"]) == 0
    frame.loc[0, "proposal_hash"] = "different"
    assert audit.shared_hash_violations(frame, ["proposal_hash"]) == 1


def test_trace_length_and_unique_keys_are_paid_steps_only():
    rows = []
    for method in audit.SEQUENTIAL_METHODS:
        for step in range(3):
            rows.append({"problem": "a", "dim": 2, "seed": 1, "method": method, "step": step})
    frame = pd.DataFrame(rows)
    result = audit.trace_length_violations(frame, expected_length=3)
    assert result["bad_groups"] == 0 and result["lengths"] == [3]
    assert audit.duplicate_count(frame, ["problem", "dim", "seed", "method", "step"]) == 0
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    assert audit.duplicate_count(duplicated, ["problem", "dim", "seed", "method", "step"]) == 1


def test_exact_artifact_names_do_not_accept_legacy_aliases(tmp_path):
    (tmp_path / "screening_mechanism_summary.csv").write_text("x\n1\n", encoding="utf-8")
    assert not (tmp_path / "mechanism_results.csv").exists()


def test_original_full_fixture_passes_semantic_audit(canonical_fixture):
    payload = run_fixture(canonical_fixture)
    assert payload["ok"]
    assert any("target_noise_std=0" in value for value in payload["warnings"])
    assert any("oracle basin-center" in value for value in payload["warnings"])


@pytest.mark.parametrize(
    ("artifact", "mutator", "semantic_check"),
    [
        ("mechanism_results.csv", lambda f: f.__setitem__("normalized_regret", f["normalized_regret"] + 0.25), "mechanism_panel_semantics"),
        ("mechanism_results.csv", lambda f: f.__setitem__("selected_y", f["selected_y"] + 0.25), "mechanism_panel_semantics"),
        ("mechanism_candidate_panel.csv", lambda f: f.__setitem__("truth", f["truth"] + 0.25), "mechanism_panel_semantics"),
        ("sequential_summary.csv", lambda f: f.__setitem__("auc_normalized_regret", f["auc_normalized_regret"] + 0.25), "trace_and_summary_semantics"),
    ],
)
def test_semantic_mutations_fail_after_manifest_hash_update(canonical_fixture, artifact, mutator, semantic_check):
    mutate_csv(canonical_fixture, artifact, mutator)
    failed = failed_checks(canonical_fixture)
    assert semantic_check in failed
    assert "artifact_hashes" not in failed


@pytest.mark.parametrize(
    "mutator",
    [
        lambda f: f.__setitem__("method_a", f["method_a"].where(f.index != 0, "Geometry-Only")),
        lambda f: f.__setitem__("higher_is_better", f["higher_is_better"].where(f.index != 0, ~f["higher_is_better"].astype(bool))),
        lambda f: f.__setitem__("holm_adjusted_p", f["holm_adjusted_p"] + 0.25),
        lambda f: f.__setitem__("supported", ~f["supported"].astype(bool)),
    ],
)
def test_primary_mutations_fail_semantic_primary_check(canonical_fixture, mutator):
    mutate_csv(canonical_fixture, "PRIMARY_TESTS.csv", mutator)
    failed = failed_checks(canonical_fixture)
    assert "primary_semantics" in failed
    assert "artifact_hashes" not in failed
