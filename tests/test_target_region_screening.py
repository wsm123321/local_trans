"""Tests for target-surrogate proposal followed by source-region screening."""

import numpy as np

from region_guided_reranking_study.source_regions import SourceRegion, SourceRegionLibrary
from region_guided_reranking_study.target_region_screening import (
    RegionFilteredBOConfig,
    RegionFilteredTargetBO,
    RegionScreeningConfig,
    SourceRegionCandidateFilter,
    TargetCandidateProposer,
    TargetProposalConfig,
    TargetProposalSet,
)


def _proposal(points, acquisition):
    points = np.asarray(points, dtype=float)
    acquisition = np.asarray(acquisition, dtype=float)
    return TargetProposalSet(
        points=points,
        acquisition_scores=acquisition,
        raw_pool=points.copy(),
        raw_acquisition_scores=acquisition.copy(),
        raw_indices=np.arange(len(points)),
        iteration=0,
    )


def _region_library(center):
    return SourceRegionLibrary(
        [
            SourceRegion(
                center=np.asarray(center, dtype=float),
                cov=np.eye(2) * 0.01,
                quality=1.0,
                count=20,
            )
        ]
    )


def test_fixed_filter_selects_best_target_acquisition_inside_source_region():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    proposal = _proposal(
        [[0.1, 0.1], [0.90, 0.90], [0.84, 0.88], [0.5, 0.5]],
        [1.0, 0.9, 0.8, 0.2],
    )
    library = _region_library([0.9, 0.9])
    target_X = np.array([[0.7, 0.7], [0.9, 0.9], [0.8, 0.8]])
    target_y = np.array([0.2, 0.0, 0.1])

    candidate_filter = SourceRegionCandidateFilter(
        bounds,
        RegionScreeningConfig(
            policy="fixed",
            geometry="quantile",
            retain_ratio=0.50,
            min_retained=1,
        ),
    )
    decision = candidate_filter.screen(
        proposal,
        library,
        target_X,
        target_y,
    )

    assert decision.filter_active
    assert decision.selected_indices.tolist() == [1]
    assert not decision.retained_mask[0]
    assert decision.points.shape == (1, 2)


def test_no_filter_exactly_recovers_target_only_top_acquisition():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    proposal = _proposal(
        [[0.1, 0.1], [0.9, 0.9], [0.5, 0.5]],
        [0.4, 1.0, 0.7],
    )
    decision = SourceRegionCandidateFilter(
        bounds,
        RegionScreeningConfig(policy="none"),
    ).screen(
        proposal,
        _region_library([0.1, 0.1]),
        np.array([[0.1, 0.1], [0.3, 0.3]]),
        np.array([1.0, 0.5]),
    )

    assert decision.selected_indices.tolist() == [1]
    assert not decision.filter_active
    assert np.all(decision.retained_mask)


def test_adaptive_filter_disables_incompatible_source_region():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    library = _region_library([0.9, 0.9])
    proposal = _proposal(
        [[0.1, 0.1], [0.9, 0.9], [0.5, 0.5]],
        [1.0, 0.8, 0.4],
    )
    # Target quality improves as points move away from the source region.
    target_X = np.array(
        [[0.90, 0.90], [0.75, 0.75], [0.50, 0.50], [0.25, 0.25], [0.10, 0.10]]
    )
    target_y = np.array([1.0, 0.8, 0.5, 0.2, 0.0])
    screening = RegionScreeningConfig(
        policy="adaptive",
        retain_ratio=0.25,
        min_target_points=2,
        prior_trust=0.0,
        prior_strength=0.0,
        evidence_shrinkage=0.0,
        activation_threshold=0.05,
    )
    decision = SourceRegionCandidateFilter(bounds, screening).screen(
        proposal,
        library,
        target_X,
        target_y,
    )

    assert decision.compatibility.spearman_correlation < 0.0
    assert decision.compatibility.trust == 0.0
    assert not decision.filter_active
    assert decision.selected_indices.tolist() == [0]


def test_adaptive_filter_activates_for_compatible_target_evidence():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    library = _region_library([0.9, 0.9])
    proposal = _proposal(
        [[0.1, 0.1], [0.9, 0.9], [0.82, 0.86], [0.5, 0.5]],
        [1.0, 0.9, 0.8, 0.3],
    )
    target_X = np.array(
        [[0.10, 0.10], [0.25, 0.25], [0.50, 0.50], [0.75, 0.75], [0.90, 0.90]]
    )
    target_y = np.array([1.0, 0.8, 0.5, 0.2, 0.0])
    screening = RegionScreeningConfig(
        policy="adaptive",
        retain_ratio=0.25,
        min_target_points=2,
        prior_trust=0.0,
        prior_strength=0.0,
        evidence_shrinkage=0.0,
        activation_threshold=0.05,
    )
    decision = SourceRegionCandidateFilter(bounds, screening).screen(
        proposal,
        library,
        target_X,
        target_y,
    )

    assert decision.compatibility.spearman_correlation > 0.0
    assert decision.compatibility.trust > 0.0
    assert decision.filter_active
    assert decision.selected_indices.tolist() == [1]


def test_target_proposer_returns_only_target_scored_candidates():
    class DummySurrogate:
        is_fitted = True

        @staticmethod
        def compute_acquisition(X, acq_type="ei"):
            del acq_type
            return -np.sum((X - np.array([0.8, 0.2])) ** 2, axis=1)

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    proposer = TargetCandidateProposer(
        bounds,
        TargetProposalConfig(raw_pool_size=80, proposal_size=12),
        rng=np.random.default_rng(8),
    )
    proposal = proposer.propose(
        DummySurrogate(),
        current_X=np.array([[0.1, 0.1], [0.2, 0.2]]),
    )

    assert proposal.points.shape == (12, 2)
    assert proposal.raw_pool.shape == (80, 2)
    assert np.allclose(
        proposal.acquisition_scores,
        proposal.raw_acquisition_scores[proposal.raw_indices],
    )


def test_complete_optimizer_ask_tell_pipeline():
    rng = np.random.default_rng(22)
    bounds = np.array([[-2.0, 2.0], [-2.0, 2.0]])
    target_center = np.array([0.7, -0.4])

    def objective(X):
        X = np.atleast_2d(X)
        return np.sum((X - target_center) ** 2, axis=1)

    init_X = rng.uniform(bounds[:, 0], bounds[:, 1], size=(6, 2))
    init_y = objective(init_X)
    optimizer = RegionFilteredTargetBO(
        bounds,
        RegionFilteredBOConfig(
            proposal=TargetProposalConfig(raw_pool_size=100, proposal_size=25),
            screening=RegionScreeningConfig(policy="fixed", retain_ratio=0.30),
            random_state=22,
        ),
    )
    optimizer.set_source_region_library(_region_library([0.7, 0.0]))
    optimizer.initialize_target(init_X, init_y)

    decision = optimizer.ask()
    optimizer.tell(decision.points, objective(decision.points))

    assert decision.proposal_set.points.shape == (25, 2)
    assert decision.points.shape == (1, 2)
    assert optimizer.target_X.shape == (7, 2)
    assert optimizer.iteration == 1
