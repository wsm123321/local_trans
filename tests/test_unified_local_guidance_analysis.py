"""Unit tests for the approved unified local-guidance statistics helpers."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analysis = load_script("approved_analysis", "analyze_unified_local_guidance_study.py")


def test_approved_methods_and_five_primary_contrasts_are_frozen():
    assert analysis.MAIN_METHODS == (
        "Target-Only", "Geometry-Only", "Local-Rank-No-Reliability",
        "Local-Rank+Reliability",
    )
    assert analysis.SAFETY_METHOD == "Reversed-Local-Rank"
    assert len(analysis.PRIMARY_CONTRASTS) == 5
    assert [x["hypothesis"] for x in analysis.PRIMARY_CONTRASTS] == [
        "H1_mechanism_normalized_regret_LocalReliability_vs_Geometry",
        "H2_mechanism_top10_hit_LocalReliability_vs_Geometry",
        "H3_sequential_final_normalized_regret_LocalReliability_vs_Geometry",
        "H4_sequential_regret_auc_LocalReliability_vs_Geometry",
        "H5_mechanism_reliability_increment_LocalReliability_vs_LocalNoReliability",
    ]
    assert analysis.PRIMARY_CONTRASTS[4]["metric"] == "normalized_regret"
    assert analysis.PRIMARY_CONTRASTS[4]["higher_is_better"] is False


def test_bootstrap_is_reproducible_and_effect_direction_is_positive_for_lower_better():
    assert analysis.paired_bootstrap_ci([1.0, 2.0, 3.0], 200, seed=7) == analysis.paired_bootstrap_ci([1.0, 2.0, 3.0], 200, seed=7)
    frame = pd.DataFrame({
        "problem": ["a"], "dim": [2], "seed": [1],
        "method": ["x"], "metric": [1.0],
    })
    baseline = frame.copy(); baseline["method"] = "y"; baseline["metric"] = 2.0
    assert analysis.strict_paired_differences(pd.concat([frame, baseline]), "x", "y", "metric", False).tolist() == [1.0]


def test_pratt_ties_rank_biserial_and_holm():
    assert analysis.wilcoxon_pratt_one_sided([0, 0, 0]) == 1.0
    assert analysis.rank_biserial([1, -1, 0]) == 0.0
    assert np.allclose(analysis.holm_adjust([0.01, 0.04, 0.2]), [0.03, 0.08, 0.2])
    stats = analysis.paired_statistics([1, 0, -1], n_bootstrap=50)
    assert (stats["wins"], stats["ties"], stats["losses"]) == (1, 1, 1)


def test_strict_pairing_rejects_duplicate_instance_method_rows():
    frame = pd.DataFrame({
        "problem": ["a", "a", "a"], "dim": [2, 2, 2], "seed": [1, 1, 1],
        "method": ["A", "A", "B"], "metric": [1., 2., 0.],
    })
    with pytest.raises(ValueError, match="duplicate"):
        analysis.strict_paired_differences(frame, "A", "B", "metric", True)


def test_analyzer_requires_exact_filenames_and_writes_five_primary_rows(tmp_path):
    methods = list(analysis.MECHANISM_METHODS)
    mechanism = pd.DataFrame([
        {"problem": "a", "dim": 2, "seed": 1, "method": method,
         "normalized_regret": 1.0 - 0.1 * i, "top10_hit": i % 2,
         "acquisition_rank": 0 if method == "Target-Only" else i + 1,
         "selected_index": i, "true_rank": i, "candidate_count": 10,
         "selected_y": 1.0, "target_truth": 1.0, "proposal_acquisition": 1.0,
         "raw_pool_hash": "pool", "proposal_hash": "proposal", "truth_hash": "truth",
         "fallback": False}
        for i, method in enumerate(methods)
    ])
    sequence = pd.DataFrame([
        {"problem": "a", "dim": 2, "seed": 1, "method": method,
         "final_normalized_regret": 1.0 - 0.1 * i,
         "auc_normalized_regret": 2.0 - 0.1 * i}
        for i, method in enumerate(analysis.MAIN_METHODS)
    ])
    traces = pd.DataFrame([
        {"problem": "a", "dim": 2, "seed": 1, "method": method, "step": step,
         "normalized_regret": 1.0 - 0.1 * i}
        for i, method in enumerate(analysis.MAIN_METHODS) for step in (1, 2)
    ])
    mechanism.to_csv(tmp_path / "mechanism_results.csv", index=False)
    sequence.to_csv(tmp_path / "sequential_summary.csv", index=False)
    traces.to_csv(tmp_path / "sequential_traces.csv", index=False)
    (tmp_path / "run_manifest.json").write_text("{}", encoding="utf-8")
    primary = analysis.run_analysis(tmp_path, output_dir=tmp_path / "analysis")
    assert len(primary) == 5
    assert set(primary["method_a"]) == {"Local-Rank+Reliability"}
    assert (tmp_path / "analysis" / "PRIMARY_TESTS.csv").exists()
    (tmp_path / "mechanism_results.csv").unlink()
    with pytest.raises(FileNotFoundError, match="approved artifact"):
        analysis.run_analysis(tmp_path, output_dir=tmp_path / "analysis2")


def test_trace_auc_is_one_row_per_instance_method_not_step_replicates():
    traces = pd.DataFrame({
        "problem": ["a"] * 4, "dim": [2] * 4, "seed": [1] * 4,
        "method": ["A", "A", "B", "B"], "step": [0, 1, 0, 1],
        "normalized_regret": [2., 1., 2., 2.],
    })
    out = analysis.trace_auc(traces)
    assert len(out) == 2
    assert out.loc[out.method == "A", "auc_normalized_regret"].iat[0] == 1.5
    assert out.loc[out.method == "A", "final_normalized_regret"].iat[0] == 1.0
    paid_out = analysis.trace_auc(pd.concat([traces, pd.DataFrame({"problem": ["a"] * 2, "dim": [2] * 2, "seed": [1] * 2, "method": ["A", "B"], "step": [2, 2], "normalized_regret": [0.5, 1.5]})], ignore_index=True), include_initial=False)
    assert paid_out.loc[paid_out.method == "A", "final_normalized_regret"].iat[0] == 0.5
    with pytest.raises(ValueError, match="duplicate step"):
        analysis.trace_auc(pd.concat([traces, traces.iloc[[0]]], ignore_index=True))
