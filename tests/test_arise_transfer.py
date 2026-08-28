"""Tests for decision-conditional region similarity and ARISE guidance."""

import numpy as np

from region_guided_reranking_study.arise_transfer import (
    ARISEConfig,
    ARISEDecision,
    RegionEvidenceModel,
    counterfactual_region_gains,
    improvement_moments,
)
from region_guided_reranking_study.source_regions import SourceRegion, SourceRegionLibrary
from region_guided_reranking_study.target_region_screening import TargetProposalSet


def test_improvement_moments_match_monte_carlo():
    rng = np.random.default_rng(123)
    mu = 0.4
    sigma = 0.7
    best = 0.0
    xi = 0.01
    expected, variance = improvement_moments(mu, sigma, best, xi=xi)

    samples = rng.normal(mu, sigma, size=300000)
    improvement = np.maximum(0.0, best - samples - xi)
    assert abs(expected - np.mean(improvement)) < 0.01
    assert abs(variance - np.var(improvement)) < 0.02


def test_region_evidence_model_preserves_signed_local_effects():
    config = ARISEConfig(
        prior_effect_variance=4.0,
        credible_z=0.5,
        min_region_coverage=0.5,
    )
    library = SourceRegionLibrary(
        [
            SourceRegion(np.array([0.2]), np.eye(1), 1.0, source_task_id="positive"),
            SourceRegion(np.array([0.8]), np.eye(1), 1.0, source_task_id="negative"),
        ]
    )
    model = RegionEvidenceModel(2, config)

    for _ in range(12):
        model.update(np.array([1.0, 0.0]), residual=1.5)
        model.update(np.array([0.0, 1.0]), residual=-1.5)

    posterior = model.snapshot(library)
    assert posterior[0].mean > 0.0
    assert posterior[0].status == "trusted"
    assert posterior[1].mean < 0.0
    assert posterior[1].status == "rejected"


def test_uncertain_region_remains_uncertain_without_coverage():
    config = ARISEConfig(min_region_coverage=1.0)
    library = SourceRegionLibrary(
        [SourceRegion(np.array([0.5]), np.eye(1), 1.0, source_task_id="unknown")]
    )
    model = RegionEvidenceModel(1, config)
    posterior = model.snapshot(library)[0]
    assert posterior.coverage == 0.0
    assert posterior.status == "uncertain"
    assert 0.49 <= posterior.probability_positive <= 0.51


def test_active_probe_uses_joint_posterior_information_gain():
    config = ARISEConfig(prior_effect_variance=1.0)
    model = RegionEvidenceModel(2, config)
    supports = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    variance = model.predictive_effect_variance(supports)
    assert variance[1] > variance[0] > variance[2]


def test_counterfactual_region_gain_uses_shared_target_proposals():
    proposal = TargetProposalSet(
        points=np.array([[0.0], [0.5], [1.0]]),
        acquisition_scores=np.array([1.0, 0.8, 0.7]),
        raw_pool=np.array([[0.0], [0.5], [1.0]]),
        raw_acquisition_scores=np.array([1.0, 0.8, 0.7]),
        raw_indices=np.array([0, 1, 2]),
        iteration=0,
    )
    decision = ARISEDecision(
        point=np.array([[0.5]]),
        selected_index=1,
        target_top1_index=0,
        selected_target_rank=1,
        selected_region_index=0,
        mode="fixed",
        proposal_set=proposal,
        acquisition_normalized=np.array([1.0, 0.5, 0.0]),
        support_matrix=np.ones((3, 2)),
        support_normalized=np.ones((3, 2)),
        trusted_guidance=np.zeros(3),
        probe_guidance=np.zeros(3),
        combined_scores=np.ones(3),
        region_candidate_indices=np.array([1, 2]),
        posteriors=[],
        improvement_forecast=None,  # type: ignore[arg-type]
        selected_supports=np.ones(2),
        global_compatibility_trust=0.0,
        iteration=0,
    )
    true_y = np.array([0.8, 0.2, 1.1])
    gains = counterfactual_region_gains(decision, true_y)
    assert np.allclose(gains, np.array([0.6, -0.3]))
