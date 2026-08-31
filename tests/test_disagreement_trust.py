"""Tests for the minimal disagreement-conditioned trust mechanism."""

import numpy as np

from region_guided_reranking_study.disagreement_trust import (
    disagreement_correction_trust,
    freeze_coverage_threshold,
    gate_accepts,
    local_spearman_trust,
    nominate_source_candidate,
    paired_margin_spearman_trust,
    practical_outcome_tolerance,
    prediction_score,
    residual_spearman_trust,
    select_target_diagnostic_pair,
    tie_aware_top_fraction_mask,
)


def test_local_spearman_rewards_matching_and_penalizes_reversal():
    quality = np.linspace(1.0, 0.0, 12)
    cost = np.linspace(0.0, 2.0, 12)
    matching = local_spearman_trust(
        quality,
        cost,
        min_points=8,
        shrinkage=8.0,
    )
    reversal = local_spearman_trust(
        1.0 - quality,
        cost,
        min_points=8,
        shrinkage=8.0,
    )

    assert matching.eligible
    assert matching.raw_statistic == 1.0
    assert matching.score > 0.5
    assert reversal.eligible
    assert reversal.raw_statistic == -1.0
    assert reversal.score < 0.5


def test_decision_matched_margin_spearman_uses_common_pair_orientation():
    source_margin = np.linspace(-1.0, 1.0, 8)
    true_diagnostic_advantage = 2.0 * source_margin
    evidence = paired_margin_spearman_trust(
        source_margin,
        true_diagnostic_advantage,
        min_points=6,
        shrinkage=2.0,
        label="test_margin",
    )
    assert evidence.eligible
    assert np.isclose(evidence.raw_statistic, 1.0)
    assert evidence.score > 0.5


def test_residual_spearman_uses_preobservation_error_direction():
    quality = np.linspace(0.0, 1.0, 10)
    target_mean = np.full(10, 1.0)
    # High source quality identifies unexpectedly low target costs.
    target_cost = 1.0 - quality
    evidence = residual_spearman_trust(
        quality,
        target_cost,
        target_mean,
        min_points=6,
        shrinkage=2.0,
    )

    assert evidence.eligible
    assert np.isclose(evidence.raw_statistic, 1.0)
    assert evidence.score > 0.5


def test_sparse_evidence_abstains_and_prediction_score_is_neutral():
    evidence = local_spearman_trust(
        [0.1, 0.2, 0.3],
        [3.0, 2.0, 1.0],
        min_points=8,
        shrinkage=8.0,
    )

    assert not evidence.eligible
    assert evidence.score == 0.5
    assert prediction_score(evidence) == 0.5


def test_disagreement_beta_posterior_separates_correction_and_harm():
    correcting = disagreement_correction_trust(5, 0, min_events=3)
    harmful = disagreement_correction_trust(0, 5, min_events=3)
    sparse = disagreement_correction_trust(2, 0, min_events=3)

    assert correcting.eligible and correcting.score > 0.95
    assert harmful.eligible and harmful.score < 0.05
    assert not sparse.eligible
    assert sparse.score > 0.5  # evidence exists, but the gate must still abstain
    assert prediction_score(sparse) == 0.5


def test_coverage_threshold_uses_actionable_eligibility_and_is_conservative():
    scores = np.array([0.95, 0.85, 0.75, 0.65, 0.55, 0.99])
    eligible = np.array([True, True, True, True, True, False])
    actionable = np.array([True, True, True, True, False, True])
    frozen = freeze_coverage_threshold(
        scores,
        eligible,
        actionable,
        target_coverage=0.5,
        minimum_positive_score=0.5,
    )

    # There are five actionable events. Threshold 0.85 accepts two (40%);
    # threshold 0.75 accepts three (60%). Both are equally close, so the more
    # conservative higher threshold is selected.
    assert frozen.threshold == 0.85
    assert frozen.accepted_actionable_events == 2
    assert np.isclose(frozen.achieved_actionable_coverage, 0.4)


def test_target_diagnostic_pair_is_source_blind_and_rank_frozen():
    acquisition = np.array([0.4, 0.9, 0.7, 0.8])
    target, diagnostic = select_target_diagnostic_pair(
        acquisition,
        alternative_rank=3,
    )
    assert target == 1
    assert diagnostic == 2


def test_nomination_requires_strict_target_source_disagreement():
    acquisition = np.array([1.0, 0.9, 0.8, 0.7])
    source_quality = np.array([0.1, 0.8, 0.9, 1.0])
    target, source, eligible = nominate_source_candidate(
        acquisition,
        source_quality,
        top_k=3,
    )
    assert target == 0
    assert source == 2
    assert np.array_equal(eligible, [0, 1, 2])

    no_conflict_quality = np.array([1.0, 0.8, 0.7, 0.6])
    target, source, _ = nominate_source_candidate(
        acquisition,
        no_conflict_quality,
        top_k=3,
    )
    assert target == source == 0


def test_gate_rejection_is_exact_target_only_fallback_logic():
    evidence = disagreement_correction_trust(5, 0, min_events=3)
    assert gate_accepts(evidence, 0.8, actionable=True)
    assert not gate_accepts(evidence, 0.8, actionable=False)
    sparse = disagreement_correction_trust(2, 0, min_events=3)
    assert not gate_accepts(sparse, 0.8, actionable=True)


def test_top_fraction_mask_includes_boundary_ties():
    values = np.array([0.0, 1.0, 1.0, 2.0, 3.0])
    mask, nominal_count, cutoff = tie_aware_top_fraction_mask(
        values,
        fraction=0.4,
        tolerance=0.0,
    )
    assert nominal_count == 2
    assert cutoff == 1.0
    assert np.array_equal(mask, [True, True, True, False, False])


def test_practical_tolerance_scales_with_observed_output_only():
    base = practical_outcome_tolerance(
        [0.0, 1.0, 2.0],
        relative_fraction=0.01,
        absolute_floor=1e-12,
    )
    affine = practical_outcome_tolerance(
        [4.0, 6.5, 9.0],
        relative_fraction=0.01,
        absolute_floor=1e-12,
    )
    assert base > 0.0
    assert np.isclose(affine / base, 2.5)
