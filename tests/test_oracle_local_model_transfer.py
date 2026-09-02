"""Gate-0 oracle local-model transfer tests."""

import numpy as np
import pytest

from region_guided_reranking_study.local_surrogate_transfer import LocalSurrogateTransferConfig
from region_guided_reranking_study.oracle_local_model_transfer import (
    CalibratedFeatureResidualRegressor,
    FixedKernelSourceExpert,
    fit_source_oracle_expert,
    identity_transform,
    oracle_coordinate_transform,
    radial_geometry_feature,
    radial_geometry_features,
    rotate_transform,
    scale_transform,
)


def config():
    return LocalSurrogateTransferConfig(
        gp_length_scale=0.7, gp_noise=1e-4, calibration_ridge=0.05, random_state=17
    )


def test_calibration_recovers_nonnegative_feature_prior_and_combines_residual():
    rng = np.random.default_rng(4)
    X = rng.uniform(-1.0, 1.0, size=(30, 2))
    features = np.column_stack([X[:, 0], X[:, 1] ** 2])
    truth = 2.0 + 1.75 * ((features[:, 0] - features[:, 0].mean()) / features[:, 0].std())
    truth += 0.9 * ((features[:, 1] - features[:, 1].mean()) / features[:, 1].std())
    truth += 0.15 * np.sin(4.0 * X[:, 0])

    model = CalibratedFeatureResidualRegressor(config()).fit(X, truth, features)
    assert model.effective == "calibrated"
    assert not model.fallback
    assert np.all(model.coefficients >= 0.0)
    assert model.intercept_ is not None
    assert np.allclose(model.feature_means, features.mean(axis=0))
    assert np.allclose(model.feature_stds, features.std(axis=0))

    mean, std = model.predict(X[:7], features=features[:7], return_std=True)
    assert mean.shape == (7,)
    assert std.shape == (7,)
    assert np.all(np.isfinite(mean)) and np.all(std > 0.0)
    # The fitted prior is recovered on the training context up to the residual GP.
    standardized = (features - model.feature_means_) / model.feature_stds_
    prior = model.intercept_ + standardized @ model.coefficients_
    assert np.corrcoef(prior, truth)[0, 1] > 0.9


def test_constant_feature_falls_back_exactly_to_same_target_only_gp():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(16, 2))
    y = 0.4 + X[:, 0] ** 2 + 0.2 * X[:, 1]
    test_X = rng.normal(size=(11, 2))
    constant = np.ones((len(X), 1))
    test_features = np.ones((len(test_X), 1))

    target = CalibratedFeatureResidualRegressor(config()).fit(X, y, None)
    fallback = CalibratedFeatureResidualRegressor(config()).fit(X, y, constant)
    target_mean, target_std = target.predict(test_X, return_std=True)
    got_mean, got_std = fallback.predict(test_X, test_features, return_std=True)

    assert fallback.fallback
    assert fallback.fallback_reason_ == "constant_feature"
    assert fallback.effective == "target_only"
    assert np.array_equal(target_mean, got_mean)
    assert np.array_equal(target_std, got_std)
    assert np.all(fallback.coefficients_ == 0.0)


def test_nonpositive_association_has_exact_target_only_fallback():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(20, 2))
    feature = X[:, 0]
    y = 3.0 - 2.0 * feature + 0.05 * X[:, 1]
    test_X = rng.normal(size=(13, 2))
    target = CalibratedFeatureResidualRegressor(config()).fit(X, y)
    model = CalibratedFeatureResidualRegressor(config()).fit(X, y, feature)

    expected = target.predict(test_X, return_std=True)
    actual = model.predict(test_X, np.zeros(len(test_X)), return_std=True)
    assert model.fallback
    assert model.fallback_reason_ == "all_coefficients_zero"
    assert model.effective_mode_ == "target_only"
    assert np.array_equal(expected[0], actual[0])
    assert np.array_equal(expected[1], actual[1])


def test_target_only_is_same_zero_feature_class_and_std_is_residual_gp_std():
    rng = np.random.default_rng(7)
    X = rng.uniform(-1, 1, size=(12, 2))
    y = np.sin(X[:, 0]) + X[:, 1] ** 2
    model = CalibratedFeatureResidualRegressor(config(), target_only=True).fit(X, y)
    mean, std = model.predict(X[:3], return_std=True)
    direct_mean, direct_std = model.gp_.predict(X[:3], return_std=True)
    assert model.n_features_ == 0
    assert model.effective == "target_only"
    assert np.array_equal(mean, direct_mean)
    assert np.array_equal(std, np.maximum(direct_std, 1e-12))


def test_source_expert_fits_rank_and_robust_raw_value_and_returns_std():
    rng = np.random.default_rng(8)
    X = rng.uniform(-1.5, 1.5, size=(25, 2))
    y = 1.0 + X[:, 0] ** 2 + 0.5 * X[:, 1] ** 2
    expert = fit_source_oracle_expert(X, y, config())
    rank, rank_std = expert.predict_rank(X[:5], return_std=True)
    raw, raw_std = expert.predict(X[:5], return_std=True)
    assert rank.shape == rank_std.shape == raw.shape == raw_std.shape == (5,)
    assert np.all((rank >= 0.0) & (rank <= 1.0))
    assert np.all(np.isfinite(raw))
    assert np.all(rank_std > 0.0) and np.all(raw_std > 0.0)
    assert expert.raw_standardizer_ is not None
    assert np.isclose(expert.raw_standardizer_.center, np.median(y))


def test_coordinate_transforms_and_relation_semantics():
    points = np.array([[2.0, 4.0], [-2.0, 6.0]])
    assert np.array_equal(identity_transform(points), points)
    assert np.allclose(scale_transform(points, 2.0), [[4.0, 8.0], [-4.0, 12.0]])
    angle = np.pi / 2.0
    # Gate-0 uses row vectors and T(Z)=Z @ R(angle).T.
    assert np.allclose(rotate_transform(np.array([[1.0, 0.0]]), angle), [[0.0, 1.0]])
    for relation in ("roughness", "reversal", "independent"):
        assert np.array_equal(oracle_coordinate_transform(points, relation), points)
    with pytest.raises(ValueError):
        rotate_transform(np.ones((3, 3)), 0.2)


def test_gate0_scale_and_rotate_transforms_recover_target_truth():
    import importlib.util
    from pathlib import Path

    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_oracle_local_model_transfer_quick.py"
    )
    spec = importlib.util.spec_from_file_location("oracle_transfer_runner", script)
    assert spec and spec.loader
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    make_relation = runner.make_relation
    relation_transform = runner.relation_transform

    points = np.array([[0.12, -0.31], [0.44, 0.27], [-0.2, 0.08]])
    theta = 0.37
    independent_theta = 1.2
    for relation in ("scale_0.7", "scale_1.5", "rotate_45"):
        source_truth, target_truth = make_relation(relation, theta, independent_theta)
        transformed = relation_transform(relation)(points)
        assert np.allclose(source_truth(transformed), target_truth(points), atol=1e-12)

    source_truth, target_truth = make_relation("output_affine", theta, independent_theta)
    assert np.allclose(source_truth(points), (target_truth(points) - 4.0) / 2.5)
    assert np.array_equal(relation_transform("independent_expert")(points), points)


def test_rank_cost_prior_has_positive_direction_and_reversal_is_negative():
    rng = np.random.default_rng(29)
    X = rng.uniform(-1.0, 1.0, size=(32, 2))
    source_cost = X[:, 0] ** 2 + 0.35 * X[:, 1] ** 2
    source_quality = 1.0 - (source_cost - source_cost.min()) / (source_cost.max() - source_cost.min())
    # rank GP's larger-is-better output must become minimization cost.
    rank_cost = 1.0 - source_quality
    aligned_y = 2.0 + 1.8 * rank_cost
    aligned = CalibratedFeatureResidualRegressor(config()).fit(X, aligned_y, rank_cost)
    assert aligned.effective == "calibrated"
    assert aligned.coefficients is not None and aligned.coefficients[0] > 0.0

    reversed_y = 4.0 - 1.8 * rank_cost
    reversed_model = CalibratedFeatureResidualRegressor(config()).fit(X, reversed_y, rank_cost)
    assert reversed_model.effective == "target_only"
    assert reversed_model.fallback
    assert reversed_model.fallback_reason_ == "all_coefficients_zero"


def test_radial_geometry_feature_supports_euclidean_mahalanobis_and_matrix_shape():
    points = np.array([[3.0, 4.0], [0.0, 0.0]])
    assert np.allclose(radial_geometry_feature(points), [5.0, 0.0])
    assert np.allclose(radial_geometry_feature(points, center=[1.0, 0.0]), [np.sqrt(20), 1.0])
    covariance = np.diag([4.0, 1.0])
    assert np.allclose(radial_geometry_feature(points, covariance=covariance), [np.sqrt(9 / 4 + 16), 0.0])
    assert radial_geometry_features(points).shape == (2, 1)


def test_nonfinite_inputs_are_rejected():
    X = np.zeros((4, 2))
    y = np.arange(4.0)
    with pytest.raises(ValueError):
        CalibratedFeatureResidualRegressor(config()).fit(X, y, np.array([[0.0], [1.0], [np.nan], [2.0]]))
    with pytest.raises(ValueError):
        CalibratedFeatureResidualRegressor(config()).fit(np.array([[0.0, np.inf]]), [1.0], None)
    model = CalibratedFeatureResidualRegressor(config()).fit(X, y)
    with pytest.raises(ValueError):
        model.predict(np.array([[np.nan, 0.0]]))
