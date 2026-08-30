"""Tests for target-calibrated source local-surrogate transfer."""

import numpy as np

from region_guided_reranking_study.local_surrogate_transfer import (
    LocalExpertResidualRegressor,
    LocalSurrogateTransferConfig,
    cross_validated_transfer_evidence,
    fit_affine_source_calibration,
    pairwise_order_accuracy,
)
from region_guided_reranking_study.local_surrogate_transfer_research import (
    AlignedLocalExpert,
    evaluate_predictions,
    sobol_chart_design,
)


def transfer_config(seed: int = 42) -> LocalSurrogateTransferConfig:
    return LocalSurrogateTransferConfig(
        gp_length_scale=0.6,
        gp_noise=1e-4,
        calibration_ridge=0.2,
        fixed_prior_scale=1.0,
        cv_folds=4,
        gate_min_relative_rmse_gain=0.0,
        gate_min_pairwise_accuracy=0.55,
        random_state=seed,
    )


def test_pairwise_order_accuracy_handles_matching_reversed_and_ties():
    first = np.array([0.0, 1.0, 2.0, 3.0])
    assert pairwise_order_accuracy(first, first) == 1.0
    assert pairwise_order_accuracy(first, -first) == 0.0
    assert pairwise_order_accuracy(np.ones(4), first) == 0.5


def test_affine_calibration_is_positive_for_matching_and_zero_for_reversal():
    quality = np.linspace(1.0, 0.0, 15)
    target_y = 2.0 + 3.0 * (1.0 - quality)
    matching = fit_affine_source_calibration(quality, target_y, transfer_config())
    reversed_calibration = fit_affine_source_calibration(
        1.0 - quality,
        target_y,
        transfer_config(),
    )

    assert matching is not None
    assert matching.slope > 0.0
    assert matching.raw_slope > 0.0
    assert reversed_calibration is not None
    assert reversed_calibration.raw_slope < 0.0
    assert reversed_calibration.slope == 0.0


def test_calibrated_reversal_falls_back_exactly_to_target_only():
    rng = np.random.default_rng(12)
    X = rng.uniform(-1.0, 1.0, size=(12, 2))
    y = 0.4 + X[:, 0] ** 2 + 0.3 * X[:, 1] ** 2
    matching_quality = 1.0 - (y - np.min(y)) / (np.ptp(y) + 1e-12)
    reversed_quality = 1.0 - matching_quality
    test_X = rng.uniform(-1.0, 1.0, size=(30, 2))
    test_quality = rng.uniform(0.0, 1.0, size=30)

    target = LocalExpertResidualRegressor("target_only", transfer_config(12)).fit(X, y)
    calibrated = LocalExpertResidualRegressor("calibrated", transfer_config(12)).fit(
        X,
        y,
        reversed_quality,
    )

    target_mean, target_std = target.predict(test_X, return_std=True)
    transfer_mean, transfer_std = calibrated.predict(
        test_X,
        test_quality,
        return_std=True,
    )
    assert calibrated.effective_mode_ == "target_only"
    assert np.array_equal(target_mean, transfer_mean)
    assert np.array_equal(target_std, transfer_std)


def test_gate_accepts_exact_matching_expert_and_rejects_reversal():
    X = sobol_chart_design(2, 16, seed=99)
    cost = 0.3 + (X[:, 0] + 0.25) ** 2 + 0.4 * (X[:, 1] - 0.15) ** 2
    quality = 1.0 - (cost - np.min(cost)) / (np.ptp(cost) + 1e-12)
    y = 5.0 + 2.5 * (1.0 - quality)
    config = transfer_config(99)

    matching = cross_validated_transfer_evidence(X, y, quality, config)
    reversal = cross_validated_transfer_evidence(X, y, 1.0 - quality, config)

    assert matching.accepted
    assert matching.relative_rmse_gain > 0.0
    assert matching.pairwise_accuracy == 1.0
    assert not reversal.accepted
    assert reversal.calibration_raw_slope < 0.0
    assert "non_positive_calibrated_slope" in reversal.rejection_reason


def test_gated_rejection_is_exact_target_only_fallback():
    X = sobol_chart_design(2, 12, seed=123)
    y = 1.0 + X[:, 0] ** 2 + 0.5 * X[:, 1] ** 2
    quality = (y - np.min(y)) / (np.ptp(y) + 1e-12)  # larger is worse
    test_X = sobol_chart_design(2, 32, seed=124)
    test_quality = np.linspace(0.0, 1.0, len(test_X))
    config = transfer_config(123)

    target = LocalExpertResidualRegressor("target_only", config).fit(X, y)
    gated = LocalExpertResidualRegressor("gated", config).fit(X, y, quality)
    target_mean, target_std = target.predict(test_X, return_std=True)
    gated_mean, gated_std = gated.predict(test_X, test_quality, return_std=True)

    assert gated.evidence_ is not None
    assert not gated.evidence_.accepted
    assert gated.effective_mode_ == "target_only"
    assert np.array_equal(target_mean, gated_mean)
    assert np.array_equal(target_std, gated_std)


def test_prediction_metrics_reward_perfect_order_and_are_finite():
    truth = np.linspace(0.0, 5.0, 100)
    std = np.full(100, 0.25)
    perfect = evaluate_predictions(truth, truth, std, top_fraction=0.10)
    reversed_metrics = evaluate_predictions(truth, -truth, std, top_fraction=0.10)

    assert perfect.standardized_rmse == 0.0
    assert perfect.ndcg_at_top == 1.0
    assert np.isclose(perfect.spearman, 1.0)
    assert perfect.pairwise_accuracy == 1.0
    assert perfect.precision_at_top == 1.0
    assert perfect.normalized_top1_regret == 0.0
    assert np.isfinite(perfect.mean_negative_log_likelihood)
    assert reversed_metrics.ndcg_at_top < perfect.ndcg_at_top
    assert reversed_metrics.pairwise_accuracy == 0.0


class _DummyStructure:
    dim = 2

    def predict_relative_quality(self, X, return_std=False):
        points = np.asarray(X, dtype=float)
        mean = np.clip(0.5 + 0.1 * points[:, 0], 0.0, 1.0)
        std = np.full(len(points), 0.05)
        return (mean, std) if return_std else (mean, None)

    def membership(self, X):
        points = np.asarray(X, dtype=float)
        return np.exp(-0.5 * np.sum(points**2, axis=1))


def test_aligned_expert_uses_translation_and_scalar_radius_only():
    expert = AlignedLocalExpert(
        structure=_DummyStructure(),
        source_anchor=np.array([2.0, -1.0]),
        target_anchor=np.array([-3.0, 4.0]),
        chart_radius=0.5,
    )
    chart = np.array([[0.0, 0.0], [1.0, -1.0]])
    source_points = expert.source_points_from_chart(chart)
    target_points = expert.target_points_from_chart(chart)
    quality, std = expert.predict_quality_from_chart(chart, return_std=True)

    assert np.allclose(source_points, [[2.0, -1.0], [2.5, -1.5]])
    assert np.allclose(target_points, [[-3.0, 4.0], [-2.5, 3.5]])
    assert np.allclose(quality, [0.7, 0.75])
    assert np.allclose(std, 0.05)
