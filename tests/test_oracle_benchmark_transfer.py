"""Unit tests for oracle benchmark local chart adapter and primitives."""

import numpy as np
import pytest

from region_guided_reranking_study.landscapes import (
    GaussianMixtureLandscape,
    LunacekBiRastrigin,
    ShiftedAckley,
    ShiftedRotatedRastrigin,
)
from region_guided_reranking_study.oracle_benchmark_transfer import (
    BENCHMARK_CONDITIONS,
    BENCHMARK_METHODS,
    BENCHMARK_PROBLEMS,
    BenchmarkLandscapePair,
    compute_physical_chart_radius,
    create_benchmark_pair,
    derive_benchmark_seed,
    extract_oracle_anchor,
    generate_unit_ball_points,
    generate_unit_sphere_directions,
    partition_context_points,
    rastrigin_chart_transform,
    rastrigin_rotation_matrix,
)


def test_constants_and_supported_sets():
    assert set(BENCHMARK_PROBLEMS) == {"GMM", "Rastrigin", "Lunacek", "Ackley"}
    assert set(BENCHMARK_CONDITIONS) == {"matching", "reversed", "label_permutation"}
    assert "Target-Only" in BENCHMARK_METHODS
    assert "Geometry-Prior+Residual" in BENCHMARK_METHODS
    assert "Oracle-Rank+Residual" in BENCHMARK_METHODS
    assert "Oracle-Value+Residual" in BENCHMARK_METHODS
    assert "Oracle-Rank+Value+Residual" in BENCHMARK_METHODS


def test_anchor_extraction_for_all_benchmark_landscapes():
    # GMM
    gmm = GaussianMixtureLandscape(dim=2, rng=np.random.default_rng(10))
    anchor_gmm = extract_oracle_anchor(gmm)
    assert anchor_gmm.shape == (2,)
    assert np.allclose(anchor_gmm, gmm.centers[0])

    # Rastrigin
    shift = np.array([0.7, -1.1])
    ras = ShiftedRotatedRastrigin(dim=2, shift=shift, rotation_angle=0.4)
    anchor_ras = extract_oracle_anchor(ras)
    assert np.allclose(anchor_ras, shift)

    # Lunacek
    mu1 = np.array([2.1, 2.3])
    luna = LunacekBiRastrigin(dim=2, mu1=mu1)
    anchor_luna = extract_oracle_anchor(luna)
    assert np.allclose(anchor_luna, mu1)

    # Ackley
    ack_shift = np.array([-0.8, 0.5])
    ack = ShiftedAckley(dim=2, shift=ack_shift)
    anchor_ack = extract_oracle_anchor(ack)
    assert np.allclose(anchor_ack, ack_shift)


def test_compute_physical_chart_radius():
    gmm = GaussianMixtureLandscape(dim=2)
    # GMM bounds are [-5.0, 5.0], width = 10.0
    r_gmm = compute_physical_chart_radius(gmm, chart_radius_fraction=0.04)
    assert np.isclose(r_gmm, 0.4)

    ras = ShiftedRotatedRastrigin(dim=2)
    # Rastrigin bounds are [-5.12, 5.12], width = 10.24
    r_ras = compute_physical_chart_radius(ras, chart_radius_fraction=0.04)
    assert np.isclose(r_ras, 0.4096)

    # Explicit chart_radius takes precedence over fraction
    r_explicit = compute_physical_chart_radius(ras, chart_radius=1.5, chart_radius_fraction=0.04)
    assert np.isclose(r_explicit, 1.5)


def test_create_benchmark_pair_for_all_four_problems_with_fraction():
    for problem in BENCHMARK_PROBLEMS:
        pair = create_benchmark_pair(problem, seed=42, dim=2, chart_radius_fraction=0.04)
        assert isinstance(pair, BenchmarkLandscapePair)
        assert pair.problem == problem
        assert pair.dim == 2
        assert pair.target_anchor.shape == (2,)
        assert pair.source_anchor.shape == (2,)
        assert np.all(np.isfinite(pair.target_anchor))
        assert np.all(np.isfinite(pair.source_anchor))
        assert 0.35 <= pair.chart_radius <= 0.45


def test_rastrigin_exact_rotation_coordinate_mapping():
    rng = np.random.default_rng(99)
    target_shift = np.array([1.2, -0.8])
    target_theta = 0.35
    source_shift = np.array([-0.5, 2.1])
    source_theta = 1.15

    target = ShiftedRotatedRastrigin(dim=2, shift=target_shift, rotation_angle=target_theta)
    source = ShiftedRotatedRastrigin(dim=2, shift=source_shift, rotation_angle=source_theta)

    pair = BenchmarkLandscapePair(
        problem="Rastrigin",
        target_landscape=target,
        source_landscape=source,
        target_anchor=target_shift,
        source_anchor=source_shift,
        chart_radius=1.0,
        target_rotation_matrix=target.R,
        source_rotation_matrix=source.R,
    )

    chart_pts = rng.uniform(-0.8, 0.8, size=(10, 2))
    target_vals = pair.evaluate_target(chart_pts)
    source_vals = pair.evaluate_source(chart_pts, condition="matching")

    # The exact rotation mapping ensures exact equivalence in Rastrigin value
    assert np.allclose(target_vals, source_vals, atol=1e-12)

    # Reversed condition produces negative values
    reversed_vals = pair.evaluate_source(chart_pts, condition="reversed")
    assert np.allclose(reversed_vals, -source_vals, atol=1e-12)


def test_generate_unit_sphere_directions_and_angle_binning():
    n_points = 512
    dirs = generate_unit_sphere_directions(n_points, seed=123, dim=2)
    assert dirs.shape == (n_points, 2)
    norms = np.linalg.norm(dirs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-12)

    # Angle binning test: divide [0, 2*pi) into 8 octants
    angles = np.arctan2(dirs[:, 1], dirs[:, 0]) % (2.0 * np.pi)
    bins = np.linspace(0.0, 2.0 * np.pi, 9)
    counts, _ = np.histogram(angles, bins=bins)
    expected_count = n_points / 8  # 64
    # With 1D Sobol angle mapping, points are evenly distributed across octants
    assert np.all(np.abs(counts - expected_count) <= 2)


def test_generate_unit_ball_points_uniform_disk():
    n_points = 512
    pts = generate_unit_ball_points(n_points, seed=456, dim=2)
    assert pts.shape == (n_points, 2)
    norms = np.linalg.norm(pts, axis=1)
    assert np.all(norms <= 1.0 + 1e-12)

    # Radial area test: proportion of points in disk of radius 0.5 should be ~ (0.5)^2 = 0.25
    prop_inner = np.mean(norms <= 0.5)
    assert abs(prop_inner - 0.25) < 0.03


def test_partition_context_points_prefix_stability():
    dirs = generate_unit_sphere_directions(30, seed=77, dim=2)
    shells = [0.35, 0.7, 1.0]

    c6 = partition_context_points(dirs, shells, 6)
    c12 = partition_context_points(dirs, shells, 12)
    c20 = partition_context_points(dirs, shells, 20)

    assert c6.shape == (6, 2)
    assert c12.shape == (12, 2)
    assert c20.shape == (20, 2)

    # Round-robin ensures strict prefix stability:
    # C6 is exact prefix of C12
    assert np.array_equal(c12[:6], c6)
    # C12 is exact prefix of C20
    assert np.array_equal(c20[:12], c12)

    # Check shell assignment: point j has norm equal to shells[j % 3]
    for j in range(20):
        assert np.isclose(np.linalg.norm(c20[j]), shells[j % 3], atol=1e-12)


def test_partition_context_points_validation_errors():
    dirs = generate_unit_sphere_directions(5, seed=1, dim=2)
    with pytest.raises(ValueError, match="need at least"):
        partition_context_points(dirs, [0.5, 1.0], 10)
    with pytest.raises(ValueError, match="shells must be non-empty"):
        partition_context_points(dirs, [], 3)
    with pytest.raises(ValueError, match="must be >="):
        partition_context_points(dirs, [0.2, 0.5, 0.8, 1.0], 2)


def test_derive_benchmark_seed_determinism_and_uniqueness():
    s1 = derive_benchmark_seed(11, "landscape")
    s2 = derive_benchmark_seed(11, "landscape")
    s3 = derive_benchmark_seed(11, "source_expert")
    s4 = derive_benchmark_seed(23, "landscape")

    assert s1 == s2
    assert s1 != s3
    assert s1 != s4
    assert 0 <= s1 < (2**31 - 1)
