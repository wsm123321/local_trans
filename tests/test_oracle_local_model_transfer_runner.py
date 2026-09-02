"""Gate-0 oracle local-model transfer runner contract tests."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_oracle_local_model_transfer_quick.py"
SPEC = importlib.util.spec_from_file_location("oracle_quick_runner", SCRIPT)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.fixture
def config():
    return runner.load_config(REPO_ROOT / "configs" / "oracle_local_model_transfer_quick.json")


def _mini_config(config):
    value = copy.deepcopy(config)
    value["seeds"] = [11]
    value["relations"] = ["identity"]
    # Keep both the relation and its explicitly frozen control in this small run.
    value["controls"] = ["identity_label_permutation"]
    return value


def test_frozen_config_and_method_panel(config):
    assert config["dimension"] == 2
    assert config["seeds"] == [11, 23, 37, 53, 71, 89, 107, 131]
    assert config["source_train_samples"] == 128
    assert config["target_context_samples"] == 12
    assert config["target_test_samples_per_shell"] == 128
    assert config["shells"] == [0.35, 0.7, 1.0]
    assert config["relations"] == list(runner.RELATIONS)
    assert config["controls"] == ["identity_label_permutation"]
    assert config["methods"] == list(runner.METHOD_MODES)
    assert config["source_expert"] == {"length_scale": 0.45, "noise": 0.0001}
    assert config["transfer_model"] == {"gp_length_scale": 0.6, "gp_noise": 0.0001, "calibration_ridge": 1.0}
    assert config["top_fraction"] == 0.1
    assert config["harm_margin_srmse"] == 0.01
    assert config["stage_id"] == runner.CANONICAL_CONFIG["stage_id"]
    assert config["scope"] == runner.CANONICAL_CONFIG["scope"]
    runner._validate_config(config)


def test_exact_config_freeze_rejects_any_change(config):
    changed = copy.deepcopy(config)
    changed["harm_margin_srmse"] = 0.02
    with pytest.raises(ValueError, match="exact frozen"):
        runner._validate_config(changed)
    changed = copy.deepcopy(config)
    changed["stage_id"] = "other-stage"
    with pytest.raises(ValueError):
        runner._validate_config(changed)


def test_relation_transforms_and_identity_control_semantics():
    source, target = runner.make_relation("output_affine", 0.2, 1.0)
    points = np.array([[0.1, -0.2], [0.7, 0.3]])
    assert np.allclose(target(points), 4.0 + 2.5 * source(points))
    for relation in ("scale_0.7", "scale_1.5", "rotate_45"):
        source, target = runner.make_relation(relation, 0.2, 1.0)
        transform = runner.relation_transform(relation)
        assert np.allclose(source(transform(points)), target(points))
    source, target = runner.make_relation("output_affine", 0.2, 1.0)
    assert np.allclose(source(points), (target(points) - 4.0) / 2.5)
    source, target = runner.make_relation("reversal", 0.2, 1.0)
    assert np.allclose(target(points), -source(points))


def test_run_refuses_nonempty_output(config, tmp_path):
    output = tmp_path / "nonempty"
    output.mkdir()
    sentinel = output / "do_not_touch.txt"
    sentinel.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        runner.run_quick(_mini_config(config), output)
    assert sentinel.read_text(encoding="utf-8") == "existing"


def test_small_run_row_count_panels_hashes_and_manifest(config, tmp_path, monkeypatch):
    # A reduced isolated run exercises the writer without executing the formal quick.
    monkeypatch.setattr(runner, "_validate_config", lambda _: None)
    output = tmp_path / "run"
    frame = runner.run_quick(_mini_config(config), output)
    assert len(frame) == 1 * 2 * 3 * 5
    assert set(frame["panel"]) == {"test"}
    assert set(frame["shell"]) == {0.35, 0.7, 1.0}
    assert set(frame["method"]) == set(config["methods"])
    assert set(frame["relation_or_control"]) == {"identity", "identity_label_permutation"}

    ledger = pd.read_csv(output / "prediction_ledger.csv")
    assert len(ledger) == 1 * 2 * 3 * 5 * 128
    assert set(ledger["panel"]) == {"test"}
    normal = frame[frame["control"] == "none"]
    permuted = frame[frame["control"] == "identity_label_permutation"]
    assert set(normal["target_context_design_hash"]) == set(permuted["target_context_design_hash"])
    assert set(normal["target_test_design_hash"]) == set(permuted["target_test_design_hash"])
    assert set(normal["target_test_truth_hash"]) == set(permuted["target_test_truth_hash"])
    assert set(normal["source_data_hash"]) != set(permuted["source_data_hash"])

    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["result_rows"] == len(frame)
    assert manifest["counts"]["ledger_rows"] == len(ledger)
    assert manifest["counts"]["diagnostic_rows"] == 1 * 2 * 3
    assert manifest["counts"]["failure_rows"] == 0
    assert manifest["config_sha256"]
    assert set(manifest["artifacts"]) == {
        "results", "prediction_ledger", "source_expert_diagnostics", "failures",
        "config", "reproducibility_inputs"
    }
    assert manifest["artifacts"]["config"] == runner._file_hash(output / "config.json")
    assert manifest["artifacts"]["reproducibility_inputs"] == runner._file_hash(output / "reproducibility_inputs.npz")
    assert set(manifest["dependencies"]) == {
        "runner", "oracle_core", "local_surrogate_transfer_research", "local_surrogate_transfer"
    }
    assert {"scipy", "sklearn"}.issubset(manifest)
    assert "head" in manifest["git"] and "dirty" in manifest["git"]
    assert set(manifest["runner_core_sha256"]) == {"runner", "core"}
    assert all(manifest["runner_core_sha256"].values())
    assert manifest["runner_sha256"] and manifest["core_sha256"]
    assert manifest["runner_path"].endswith("run_oracle_local_model_transfer_quick.py")
    assert manifest["core_path"].endswith("oracle_local_model_transfer.py")

    diagnostics = pd.read_csv(output / "source_expert_diagnostics.csv")
    assert "source_value_pairwise_accuracy" in diagnostics
    assert "source_rank_target_agreement" in diagnostics
    assert "source_value_standardized_target_rmse" in diagnostics
    assert "source_value_rmse" not in diagnostics
    assert "source_rank_pairwise_accuracy" not in diagnostics
    assert {"source_context_oob_count", "source_context_oob_rate", "source_test_oob_count", "source_test_oob_rate"}.issubset(diagnostics)
    assert diagnostics["source_test_oob_count"].notna().all()

    with np.load(output / "reproducibility_inputs.npz", allow_pickle=False) as inputs:
        source_dirs = inputs["seed_11_source_dirs"]
        permutation = inputs["seed_11_permutation"]
        assert source_dirs.shape == (128, 2)
        assert np.array_equal(np.sort(permutation), np.arange(128))
        for relation in ("identity",):
            source_y = inputs[f"seed_11_source_y_{relation}"]
            assert runner._array_hash(source_y) == runner._array_hash(inputs[f"seed_11_source_y_{relation}"])
            truth = inputs[f"seed_11_target_test_truth_{relation}_0.35"]
            assert runner._array_hash(truth) in set(frame["target_test_truth_hash"])


def test_small_run_is_deterministic(config, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_validate_config", lambda _: None)
    first = runner.run_quick(_mini_config(config), tmp_path / "first")
    second = runner.run_quick(_mini_config(config), tmp_path / "second")
    pd.testing.assert_frame_equal(first, second)
    for name in ("results.csv", "prediction_ledger.csv", "source_expert_diagnostics.csv", "failures.csv"):
        assert (tmp_path / "first" / name).read_bytes() == (tmp_path / "second" / name).read_bytes()


def test_api_rejects_missing_prior_and_unknown_mode():
    from region_guided_reranking_study.oracle_local_model_transfer import (
        OracleLocalModelTransfer,
    )

    with pytest.raises(ValueError):
        OracleLocalModelTransfer("not-a-method")
    model = OracleLocalModelTransfer("oracle_rank")
    with pytest.raises(ValueError):
        model.fit(np.ones((4, 2)), np.arange(4, dtype=float))
