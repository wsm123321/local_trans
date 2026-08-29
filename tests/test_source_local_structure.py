"""Tests for source local-structure extraction."""

import numpy as np

from region_guided_reranking_study.source_local_structure import (
    LocalStructureConfig,
    SourceLocalStructureExtractor,
)
from region_guided_reranking_study.source_structure_research import (
    generate_controlled_landscape,
    latin_hypercube_sample,
    recovery_metrics,
)


def fast_config(seed: int = 42, max_regions: int = 3) -> LocalStructureConfig:
    return LocalStructureConfig(
        elite_ratio=0.25,
        max_regions=max_regions,
        min_elite_per_region=4,
        min_context_samples=16,
        max_context_samples=64,
        context_multiplier=3.0,
        min_boundary_fraction=0.25,
        model_type="random_forest",
        random_forest_trees=80,
        cv_folds=3,
        random_state=seed,
    )


def test_affine_response_transform_preserves_extracted_scores():
    rng = np.random.default_rng(12)
    X = rng.uniform(-2.0, 2.0, size=(140, 2))
    y = (X[:, 0] - 0.5) ** 2 + 0.4 * (X[:, 1] + 0.3) ** 2
    test_X = rng.uniform(-2.0, 2.0, size=(50, 2))

    first = SourceLocalStructureExtractor(fast_config(12)).fit_dataset(X, y)
    second = SourceLocalStructureExtractor(fast_config(12)).fit_dataset(
        X,
        17.0 * y + 103.0,
    )

    assert len(first.structures) == len(second.structures)
    assert np.allclose(first.geometry_score(test_X), second.geometry_score(test_X))
    assert np.allclose(first.score(test_X), second.score(test_X))


def test_extracted_structure_scores_true_optimum_above_far_point():
    rng = np.random.default_rng(21)
    X = rng.uniform(-2.0, 2.0, size=(180, 2))
    optimum = np.array([0.55, -0.35])
    y = (
        (X[:, 0] - optimum[0]) ** 2
        + 0.15 * (X[:, 1] - optimum[1]) ** 2
        + 0.1 * (X[:, 0] - optimum[0]) * (X[:, 1] - optimum[1])
    )

    library = SourceLocalStructureExtractor(fast_config(21)).fit_dataset(X, y)
    query = np.vstack([optimum, np.array([-1.8, 1.8])])
    scores = library.score(query, use_reliability=False)

    assert scores.shape == (2,)
    assert scores[0] > scores[1]


def test_context_contains_non_elite_boundary_samples():
    rng = np.random.default_rng(33)
    X = rng.uniform(-3.0, 3.0, size=(160, 2))
    y = np.sum((X - np.array([0.3, 0.7])) ** 2, axis=1)

    library = SourceLocalStructureExtractor(fast_config(33)).fit_dataset(X, y)

    assert len(library.structures) >= 1
    for structure in library.structures:
        assert structure.context_count >= structure.core_count
        assert structure.validation.boundary_fraction > 0.0
        assert structure.context_count >= 16


def test_bic_extractor_recovers_two_separated_elite_clusters():
    rng = np.random.default_rng(44)
    first = rng.normal(loc=np.array([-1.5, -1.2]), scale=0.18, size=(50, 2))
    second = rng.normal(loc=np.array([1.5, 1.2]), scale=0.18, size=(50, 2))
    background = rng.uniform(-3.0, 3.0, size=(120, 2))
    X = np.vstack([first, second, background])
    y = np.minimum(
        np.sum((X - np.array([-1.5, -1.2])) ** 2, axis=1),
        np.sum((X - np.array([1.5, 1.2])) ** 2, axis=1),
    )

    config = fast_config(44, max_regions=2)
    library = SourceLocalStructureExtractor(config).fit_dataset(X, y)

    assert len(library.structures) == 2
    centers = np.vstack([structure.center for structure in library.structures])
    assert np.min(np.linalg.norm(centers - np.array([-1.5, -1.2]), axis=1)) < 0.5
    assert np.min(np.linalg.norm(centers - np.array([1.5, 1.2]), axis=1)) < 0.5


def test_library_component_shapes_and_best_region_indices():
    rng = np.random.default_rng(55)
    X = rng.uniform(-2.0, 2.0, size=(150, 2))
    y = np.minimum(
        np.sum((X - np.array([-0.8, 0.0])) ** 2, axis=1),
        np.sum((X - np.array([0.8, 0.0])) ** 2, axis=1),
    )
    library = SourceLocalStructureExtractor(fast_config(55, 2)).fit_dataset(X, y)
    query = rng.uniform(-2.0, 2.0, size=(23, 2))

    geometry = library.geometry_components(query)
    structure = library.structure_components(query)
    best = library.best_region_indices(query)

    assert geometry.shape == (23, len(library.structures))
    assert structure.shape == geometry.shape
    assert best.shape == (23,)
    assert np.all(best >= 0)


def test_controlled_recovery_metrics_are_finite():
    rng = np.random.default_rng(66)
    landscape = generate_controlled_landscape(2, 3, rng)
    X = latin_hypercube_sample(landscape.bounds, 350, seed=66)
    y = landscape(X)
    library = SourceLocalStructureExtractor(fast_config(66, 3)).fit_dataset(X, y)
    oracle = landscape.oracle_structures()

    metrics = recovery_metrics(
        library.structures,
        [item["center"] for item in oracle],
        [item["covariance"] for item in oracle],
        landscape.bounds,
    )

    assert 0.0 <= metrics.basin_recall <= 1.0
    assert np.isfinite(metrics.mean_matched_mahalanobis)
    assert np.isfinite(metrics.normalized_center_error)


def test_gp_model_path_runs_and_predicts_finite_scores():
    rng = np.random.default_rng(77)
    X = rng.uniform(-2.0, 2.0, size=(90, 2))
    y = np.sum((X - np.array([0.4, -0.2])) ** 2, axis=1)
    config = LocalStructureConfig(
        elite_ratio=0.25,
        max_regions=2,
        min_elite_per_region=4,
        min_context_samples=14,
        max_context_samples=40,
        model_type="gp",
        gp_restarts=0,
        cv_folds=2,
        random_state=77,
    )
    library = SourceLocalStructureExtractor(config).fit_dataset(X, y)
    scores = library.score(rng.uniform(-2.0, 2.0, size=(17, 2)))

    assert scores.shape == (17,)
    assert np.all(np.isfinite(scores))
