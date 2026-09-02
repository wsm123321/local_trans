"""Tests for oracle benchmark transfer runner contract, execution, and reproducibility."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_oracle_benchmark_transfer_pilot.py"
SPEC = importlib.util.spec_from_file_location("benchmark_pilot_runner", SCRIPT)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _mini_config():
    return {
        "stage_id": "test-oracle-benchmark-transfer-mini",
        "scope": "test_mini",
        "dimension": 2,
        "seeds": [11],
        "problems": ["GMM", "Rastrigin"],
        "relations": ["matching", "reversed"],
        "target_context_samples": [6],
        "shells": [0.5, 1.0],
        "methods": [
            "Target-Only",
            "Geometry-Prior+Residual",
            "Oracle-Rank+Residual",
            "Oracle-Value+Residual",
            "Oracle-Rank+Value+Residual",
        ],
        "source_train_samples": 24,
        "target_test_samples": 48,  # Total test points across 2 shells -> 24 per shell
        "chart_radius_fraction": 0.04,
        "source_expert": {"length_scale": 0.5, "noise": 0.001},
        "transfer_model": {
            "gp_length_scale": 0.6,
            "gp_noise": 0.001,
            "calibration_ridge": 1.0,
            "fixed_prior_scale": 1.0,  # Should not cause TypeError
        },
        "top_fraction": 0.1,
        "harm_margin_srmse": 0.01,
    }


def test_formal_config_normalization():
    formal_config_path = REPO_ROOT / "configs" / "oracle_benchmark_transfer_pilot.json"
    if formal_config_path.exists():
        raw_formal = json.loads(formal_config_path.read_text(encoding="utf-8"))
        norm_formal = runner.validate_and_normalize_config(raw_formal)
        assert norm_formal["dimension"] == 2
        assert norm_formal["context_sample_sizes"] == [6, 12, 20]
        assert norm_formal["target_test_samples"] == 512
        assert norm_formal["chart_radius_fraction"] == 0.04
        assert set(norm_formal["conditions"]) == {"matching", "reversed", "label_permutation"}


def test_config_validation_and_normalization():
    # Validates default
    norm = runner.validate_and_normalize_config(runner.DEFAULT_PILOT_CONFIG)
    assert norm["dimension"] == 2
    assert set(norm["problems"]) == {"GMM", "Rastrigin", "Lunacek", "Ackley"}
    assert set(norm["conditions"]) == {"matching", "reversed", "label_permutation"}
    assert norm["context_sample_sizes"] == [6, 12, 20]

    # Custom mini config
    mini = _mini_config()
    norm_mini = runner.validate_and_normalize_config(mini)
    assert norm_mini["problems"] == ["GMM", "Rastrigin"]
    assert norm_mini["conditions"] == ["matching", "reversed"]
    assert norm_mini["context_sample_sizes"] == [6]
    assert norm_mini["target_test_samples"] == 48

    # Rejection of invalid configs
    with pytest.raises(ValueError, match="Dimension must be 2"):
        bad = copy.deepcopy(mini)
        bad["dimension"] = 3
        runner.validate_and_normalize_config(bad)

    with pytest.raises(ValueError, match="Unsupported problem"):
        bad = copy.deepcopy(mini)
        bad["problems"] = ["UnknownProblem"]
        runner.validate_and_normalize_config(bad)

    with pytest.raises(ValueError, match="Unsupported condition"):
        bad = copy.deepcopy(mini)
        bad["relations"] = ["bad_condition"]
        runner.validate_and_normalize_config(bad)

    with pytest.raises(ValueError, match="Unsupported method"):
        bad = copy.deepcopy(mini)
        bad["methods"] = ["bad_method"]
        runner.validate_and_normalize_config(bad)

    with pytest.raises(ValueError, match="Context sample size"):
        bad = copy.deepcopy(mini)
        bad["shells"] = [0.2, 0.4, 0.6, 0.8]
        bad["target_context_samples"] = [2]  # < 4 shells
        runner.validate_and_normalize_config(bad)


def test_refuse_nonempty_output_directory(tmp_path):
    output = tmp_path / "nonempty"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("important", encoding="utf-8")

    with pytest.raises(FileExistsError):
        runner.run_benchmark_pilot(_mini_config(), output)

    assert sentinel.read_text(encoding="utf-8") == "important"


def test_mini_pilot_execution_artifacts_and_manifest(tmp_path):
    output = tmp_path / "pilot_run"
    cfg = _mini_config()
    frame = runner.run_benchmark_pilot(cfg, output)

    # Expected counts: 1 seed * 2 problems * 2 conditions * 1 context * 2 shells * 5 methods = 40 rows
    expected_result_rows = 1 * 2 * 2 * 1 * 2 * 5
    assert len(frame) == expected_result_rows
    assert set(frame["problem"]) == {"GMM", "Rastrigin"}
    assert set(frame["condition"]) == {"matching", "reversed"}
    assert set(frame["relation"]) == {"matching", "reversed"}
    assert set(frame["context_size"]) == {6}
    assert set(frame["shell"]) == {0.5, 1.0}
    assert set(frame["method"]) == set(cfg["methods"])

    # Prediction ledger
    # 48 total test samples across 2 shells -> 24 per shell -> 40 * 24 = 960 rows
    ledger = pd.read_csv(output / "prediction_ledger.csv")
    assert len(ledger) == 40 * 24
    assert set(ledger["problem"]) == {"GMM", "Rastrigin"}
    assert "relation" in ledger.columns
    assert "context_size" in ledger.columns

    # Diagnostics
    diagnostics = pd.read_csv(output / "source_expert_diagnostics.csv")
    expected_diag_rows = 1 * 2 * 2 * 1 * 2
    assert len(diagnostics) == expected_diag_rows
    assert "source_value_pairwise_accuracy" in diagnostics
    assert "source_rank_target_agreement" in diagnostics
    assert "source_value_standardized_target_rmse" in diagnostics
    assert "chart_radius" in diagnostics
    # GMM has width 10.0 -> 0.04 * 10 = 0.4; Rastrigin has width 10.24 -> 0.04 * 10.24 = 0.4096
    gmm_diag = diagnostics[diagnostics["problem"] == "GMM"]
    ras_diag = diagnostics[diagnostics["problem"] == "Rastrigin"]
    assert np.isclose(gmm_diag["chart_radius"].iloc[0], 0.4)
    assert np.isclose(ras_diag["chart_radius"].iloc[0], 0.4096)

    # Rastrigin should log exact_rotation transform, GMM should log identity
    assert (ras_diag["source_query_transform"] == "exact_rotation").all()
    assert (gmm_diag["source_query_transform"] == "identity").all()

    # Failures
    failures = pd.read_csv(output / "failures.csv")
    assert len(failures) == 0

    # Manifest
    manifest_path = output / "run_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["counts"]["result_rows"] == expected_result_rows
    assert manifest["counts"]["ledger_rows"] == len(ledger)
    assert manifest["counts"]["diagnostic_rows"] == expected_diag_rows
    assert manifest["counts"]["failure_rows"] == 0
    assert manifest["config_sha256"] == runner._file_hash(output / "config.json")
    assert manifest["artifacts"]["results"] == runner._file_hash(output / "results.csv")
    assert manifest["artifacts"]["reproducibility_inputs"] == runner._file_hash(output / "reproducibility_inputs.npz")
    assert "oracle_benchmark_transfer" in manifest["dependencies"]

    # Reproducibility inputs
    with np.load(output / "reproducibility_inputs.npz", allow_pickle=False) as inputs:
        assert "seed_11_GMM_source_dirs" in inputs
        assert "seed_11_Rastrigin_source_dirs" in inputs
        assert "seed_11_GMM_source_y_raw" in inputs
        assert "seed_11_GMM_source_y_matching" in inputs
        assert "seed_11_GMM_source_y_reversed" in inputs
        # Unit ball source dirs norm <= 1.0
        assert np.all(np.linalg.norm(inputs["seed_11_GMM_source_dirs"], axis=1) <= 1.0 + 1e-12)
        # Reversed source y is negative of raw
        assert np.allclose(inputs["seed_11_GMM_source_y_reversed"], -inputs["seed_11_GMM_source_y_raw"])


def test_mini_pilot_determinism(tmp_path):
    cfg = _mini_config()
    first = runner.run_benchmark_pilot(cfg, tmp_path / "run1")
    second = runner.run_benchmark_pilot(cfg, tmp_path / "run2")

    pd.testing.assert_frame_equal(first, second)
    for name in (
        "results.csv",
        "prediction_ledger.csv",
        "source_expert_diagnostics.csv",
        "failures.csv",
        "config.json",
    ):
        assert (tmp_path / "run1" / name).read_bytes() == (tmp_path / "second" if False else tmp_path / "run2" / name).read_bytes()


def test_failure_is_recorded_and_not_silent(tmp_path, monkeypatch):
    cfg = _mini_config()
    cfg["problems"] = ["GMM"]

    def faulty_create_pair(*args, **kwargs):
        raise RuntimeError("Simulated landscape construction failure")

    monkeypatch.setattr(runner, "create_benchmark_pair", faulty_create_pair)

    output = tmp_path / "failure_run"
    runner.run_benchmark_pilot(cfg, output)

    failures = pd.read_csv(output / "failures.csv")
    assert len(failures) == 1
    assert failures.iloc[0]["error_type"] == "RuntimeError"
    assert "Simulated landscape construction failure" in failures.iloc[0]["error"]
