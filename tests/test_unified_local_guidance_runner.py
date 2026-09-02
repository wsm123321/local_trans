"""Lightweight contract tests for the unified local-guidance runner."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_unified_local_guidance_study.py"
SPEC = importlib.util.spec_from_file_location("unified_runner", SCRIPT)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.mark.parametrize(
    ("name", "source", "pool", "proposal", "budget", "boot", "problems", "dims", "nseeds"),
    [
        ("unified_local_guidance_quick.json", 80, 300, 40, 5, 500, {"GMM", "Ackley"}, {2}, 2),
        ("unified_local_guidance_full.json", 160, 1500, 100, 20, 5000,
         {"GMM", "Rastrigin", "Lunacek", "Ackley"}, {2, 5}, 8),
    ],
)
def test_frozen_config_scale(name, source, pool, proposal, budget, boot, problems, dims, nseeds):
    config = runner.load_config(REPO_ROOT / "configs" / name)
    study = config["study"]
    assert int(study["source_samples"]) == source
    assert int(study["raw_pool_size"]) == pool
    assert int(study["proposal_size"]) == proposal
    assert int(study["budget"]) == budget
    assert int(config["analysis"]["bootstrap_samples"]) == boot
    assert set(study["problems"]) == problems
    assert {int(d) for d in study["dimensions"]} == dims
    assert len(study["seeds"]) == nseeds
    assert study["acquisition"] == "ei"
    assert study["relations"] == ["matching", "reversed"]
    assert study["methods"] == ["Target-Only", "Geometry-Only", "Local-Rank-No-Reliability", "Local-Rank+Reliability"]
    assert study["safety_diagnostics"] == ["Reversed-Local-Rank"]
    assert set(config["guidance"]) == {"source_weight", "target_nomination_ratio", "source_nomination_ratio", "aggregation"}


def test_run_refuses_nonempty_output(tmp_path):
    config = runner.load_config(REPO_ROOT / "configs" / "unified_local_guidance_quick.json")
    output = tmp_path / "not_empty"
    output.mkdir()
    (output / "do_not_touch.txt").write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        runner.run_study(config, output)
    assert (output / "do_not_touch.txt").read_text(encoding="utf-8") == "existing"


def test_pool_helper_is_deterministic_and_excludes_all_data():
    bounds = np.array([[-1.0, 1.0], [-2.0, 2.0]])
    target = np.array([[0.0, 0.0], [0.2, -0.4]])
    source = np.array([[-0.3, 0.5], [0.7, -1.2]])
    first = runner.generate_unique_pool(bounds, 32, np.random.default_rng(123), [target, source])
    second = runner.generate_unique_pool(bounds, 32, np.random.default_rng(123), [target, source])
    assert np.array_equal(first, second)
    runner.assert_unique_nonoverlap(first, [target, source])


def test_shared_proposal_is_deterministic_and_has_no_overlap():
    raw = np.array([[0.1, 0.1], [0.4, 0.4], [0.2, 0.3], [0.9, 0.9]])
    acquisition = np.array([0.2, 0.8, 0.5, 0.1])
    class DummyGP:
        def compute_acquisition(self, points, acq_type="ei"):
            assert acq_type == "ei"
            return np.asarray(acquisition)[np.array([0, 1, 2, 3])]
    proposal, _, indices = runner.make_ei_proposal(DummyGP(), raw, 2)
    assert indices.tolist() == [1, 2]
    assert np.array_equal(proposal, raw[indices])
    runner.assert_unique_nonoverlap(raw[:1], [raw[2:]])


def test_approved_method_modes_and_required_metric_semantics():
    assert runner.MAIN_METHODS == (
        "Target-Only", "Geometry-Only", "Local-Rank-No-Reliability", "Local-Rank+Reliability"
    )
    assert runner.METHOD_MODES["Geometry-Only"] == "geometry_only"
    assert runner.METHOD_MODES["Local-Rank-No-Reliability"] == "local_rank_no_reliability"
    assert runner.METHOD_MODES["Local-Rank+Reliability"] == "local_rank_reliability"
    assert runner.METHOD_MODES["Reversed-Local-Rank"] == "reversed_local_rank"


def test_mechanism_metrics_use_q90_scale_tie_aware_top10_and_zero_based_rank():
    truth = np.array([0.0, 1.0, 1.0, 10.0])
    metrics = runner.mechanism_metrics(truth, 2, tolerance=1e-12)
    assert metrics["raw_regret"] == 1.0
    assert metrics["normalized_regret"] == 1.0 / (np.quantile(truth, 0.90) - np.min(truth))
    assert metrics["true_rank"] == 2  # stable 0-based position within the tied values
    assert metrics["top10_hit"] is False
    tied = runner.mechanism_metrics(np.array([0.0, 0.0, 1.0, 2.0]), 1)
    assert tied["top10_hit"] is True
    assert tied["true_rank"] == 1
