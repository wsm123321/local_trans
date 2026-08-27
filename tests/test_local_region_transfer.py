"""Tests for the complete local-region transfer optimizer."""

import numpy as np

from region_guided_reranking_study.local_region_transfer import (
    LocalRegionTransferConfig,
    LocalRegionTransferOptimizer,
)
from region_guided_reranking_study.source_regions import (
    SourceRegion,
    SourceRegionLibrary,
)


def test_source_region_can_nominate_and_promote_candidate():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    config = LocalRegionTransferConfig(
        pool_size=40,
        source_weight=1.0,
        target_nomination_ratio=0.25,
        source_nomination_ratio=0.25,
        random_state=7,
    )
    optimizer = LocalRegionTransferOptimizer(bounds, config)
    optimizer.set_source_region_library(
        SourceRegionLibrary(
            [
                SourceRegion(
                    center=np.array([0.9, 0.9]),
                    cov=np.eye(2) * 0.01,
                    quality=1.0,
                    count=20,
                )
            ]
        )
    )

    candidates = np.array(
        [
            [0.1, 0.1],  # best target acquisition
            [0.9, 0.9],  # strongest source-region support
            [0.5, 0.5],
            [0.2, 0.8],
        ]
    )
    acquisition = np.array([1.0, 0.9, 0.2, 0.1])

    decision = optimizer.rank_candidate_pool(candidates, acquisition)

    assert decision.selected_indices.tolist() == [1]
    assert decision.shortlist_mask[0]
    assert decision.shortlist_mask[1]
    assert decision.effective_source_weight == 1.0
    assert decision.combined_scores[1] > decision.combined_scores[0]


def test_empty_source_library_reduces_to_target_only_selection():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    optimizer = LocalRegionTransferOptimizer(
        bounds,
        LocalRegionTransferConfig(pool_size=40, source_weight=3.0),
    )

    candidates = np.array(
        [
            [0.1, 0.1],
            [0.9, 0.9],
            [0.5, 0.5],
        ]
    )
    acquisition = np.array([0.3, 0.9, 0.5])

    decision = optimizer.rank_candidate_pool(candidates, acquisition)

    assert decision.selected_indices.tolist() == [1]
    assert decision.effective_source_weight == 0.0
    assert np.allclose(decision.normalized_source, 0.0)
    assert np.allclose(
        decision.combined_scores,
        decision.normalized_acquisition,
    )


def test_complete_ask_tell_pipeline_extracts_regions_and_updates_target():
    rng = np.random.default_rng(12)
    bounds = np.array([[-2.0, 2.0], [-2.0, 2.0]])

    source_center = np.array([0.6, -0.4])
    source_X = rng.uniform(bounds[:, 0], bounds[:, 1], size=(80, 2))
    source_y = np.sum((source_X - source_center) ** 2, axis=1)

    target_center = np.array([0.75, -0.25])

    def target_objective(X):
        X = np.atleast_2d(X)
        return np.sum((X - target_center) ** 2, axis=1)

    init_X = rng.uniform(bounds[:, 0], bounds[:, 1], size=(6, 2))
    init_y = target_objective(init_X)

    optimizer = LocalRegionTransferOptimizer(
        bounds,
        LocalRegionTransferConfig(
            pool_size=100,
            top_ratio=0.25,
            max_clusters=2,
            source_weight=0.8,
            random_state=12,
        ),
    )
    library = optimizer.fit_source_regions([(source_X, source_y)])
    optimizer.initialize_target(init_X, init_y)

    decision = optimizer.ask()
    selected_y = target_objective(decision.points)
    optimizer.tell(decision.points, selected_y)

    assert len(library.regions) >= 1
    assert decision.points.shape == (1, 2)
    assert decision.candidates.shape == (100, 2)
    assert decision.shortlist_mask.any()
    assert optimizer.target_X.shape == (7, 2)
    assert optimizer.target_y.shape == (7,)
    assert optimizer.iteration == 1
    assert np.min(np.linalg.norm(init_X - decision.points[0], axis=1)) > 1e-12


def test_optimize_runs_closed_loop_with_fixed_evaluation_budget():
    rng = np.random.default_rng(21)
    bounds = np.array([[-2.0, 2.0], [-2.0, 2.0]])

    source_X = rng.uniform(bounds[:, 0], bounds[:, 1], size=(100, 2))
    source_y = np.sum((source_X - np.array([0.5, 0.5])) ** 2, axis=1)

    def target_objective(X):
        X = np.atleast_2d(X)
        return np.sum((X - np.array([0.65, 0.55])) ** 2, axis=1)

    init_X = rng.uniform(bounds[:, 0], bounds[:, 1], size=(6, 2))
    init_y = target_objective(init_X)

    optimizer = LocalRegionTransferOptimizer(
        bounds,
        LocalRegionTransferConfig(
            pool_size=80,
            top_ratio=0.20,
            max_clusters=2,
            source_weight=0.7,
            source_weight_decay=0.05,
            random_state=21,
        ),
    )
    optimizer.fit_source_regions([(source_X, source_y)])

    result = optimizer.optimize(
        target_objective,
        init_X=init_X,
        init_y=init_y,
        budget=4,
    )

    assert result.X.shape == (10, 2)
    assert result.y.shape == (10,)
    assert len(result.decisions) == 4
    assert len(result.best_y_trace) == 5
    assert result.best_y == np.min(result.y)
    assert result.best_y <= np.min(init_y)
    assert np.all(np.diff(result.best_y_trace) <= 1e-12)
