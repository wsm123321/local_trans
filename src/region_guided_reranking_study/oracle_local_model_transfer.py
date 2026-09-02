"""Gate-0 oracle local-model transfer primitives.

The module is intentionally independent of runners and result files.  It supplies
an oracle source expert (fixed Matern GP rank and robust raw-value models), a
non-negative calibrated target residual model, and explicit coordinate/geometry
helpers.  All target residual and target-only GPs use the exact constructor from
``local_surrogate_transfer._make_gp``.

Coordinate transforms use the target-to-source convention.  For the Gate-0
scale relation ``f_target(Z) = local_cost(Z, scale=s)`` while the source is
``local_cost(Z, scale=1)``, querying the source at a target point multiplies by
``s``.  ``roughness``, ``reversal``, and ``independent`` deliberately retain
identity coordinates: these relations model response mismatch rather than
pretending that coordinates are aligned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import lsq_linear
from sklearn.gaussian_process import GaussianProcessRegressor

from .local_surrogate_transfer import LocalSurrogateTransferConfig, _make_gp
from .local_surrogate_transfer_research import rank_quality

Array = np.ndarray
FeatureName = Literal["rank_quality", "raw_value"]
OracleRelation = Literal[
    "identity", "scale", "rotate", "roughness", "reversal", "independent"
]


@dataclass(frozen=True)
class RobustStandardizer:
    """Median/MAD (with IQR/std safeguards) standardization parameters."""

    center: float
    scale: float

    def transform(self, values: Array) -> Array:
        values = np.asarray(values, dtype=float)
        return (values - self.center) / self.scale

    def inverse_transform(self, values: Array) -> Array:
        values = np.asarray(values, dtype=float)
        return values * self.scale + self.center


@dataclass(frozen=True)
class OracleLocalModelTransferConfig:
    """Frozen numerical settings for Gate-0 models.

    The fields intentionally mirror the fixed-GP settings used by
    :class:`LocalSurrogateTransferConfig`, so ``_make_gp`` receives identical
    parameters.  ``as_local_config`` is useful when a caller wants to pass this
    config to existing transfer code.
    """

    gp_length_scale: float = 0.6
    gp_noise: float = 1e-4
    calibration_ridge: float = 1.0
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.gp_length_scale <= 0:
            raise ValueError("gp_length_scale must be positive")
        if self.gp_noise <= 0:
            raise ValueError("gp_noise must be positive")
        if self.calibration_ridge < 0:
            raise ValueError("calibration_ridge must be non-negative")

    def as_local_config(self) -> LocalSurrogateTransferConfig:
        return LocalSurrogateTransferConfig(
            gp_length_scale=self.gp_length_scale,
            gp_noise=self.gp_noise,
            calibration_ridge=self.calibration_ridge,
            random_state=self.random_state,
        )


@dataclass(frozen=True)
class OracleTransferEvidence:
    """Fitted calibration details retained for an audit/ledger."""

    intercept: float
    coefficients: Tuple[float, ...]
    prior_names: Tuple[str, ...]
    prior_means: Tuple[float, ...]
    prior_stds: Tuple[float, ...]


class FixedKernelSourceExpert:
    """Fixed-kernel source expert for rank quality and robust raw value.

    Both models use the fixed ``_make_gp`` kernel.  The rank target is the
    existing larger-is-better rank quality utility; the raw target is the source
    cost after robust median/MAD standardization.  ``predict`` returns the
    feature's defined units and the corresponding GP standard deviation.
    """

    def __init__(
        self,
        config: Optional[Union[LocalSurrogateTransferConfig, OracleLocalModelTransferConfig]] = None,
    ) -> None:
        self.config = config or LocalSurrogateTransferConfig()
        self.rank_gp_: Optional[GaussianProcessRegressor] = None
        self.raw_gp_: Optional[GaussianProcessRegressor] = None
        self.raw_standardizer_: Optional[RobustStandardizer] = None
        self.dim_: Optional[int] = None

    def fit(self, X: Array, y: Array) -> "FixedKernelSourceExpert":
        self.fit_rank_quality(X, y)
        self.fit_raw_value(X, y)
        return self

    def fit_rank_quality(self, X: Array, y: Array) -> "FixedKernelSourceExpert":
        points = _as_points(X, name="X")
        values = _as_values(y, len(points), name="y")
        self._check_or_set_dim(points)
        self.rank_gp_ = _make_gp(points.shape[1], self.config)
        self.rank_gp_.fit(points, rank_quality(values))
        return self

    def fit_raw_value(self, X: Array, y: Array) -> "FixedKernelSourceExpert":
        points = _as_points(X, name="X")
        values = _as_values(y, len(points), name="y")
        self._check_or_set_dim(points)
        self.raw_standardizer_ = _robust_standardizer(values)
        self.raw_gp_ = _make_gp(points.shape[1], self.config)
        self.raw_gp_.fit(points, self.raw_standardizer_.transform(values))
        return self

    def fit_source_function(
        self, X: Array, source_function: Callable[[Array], Array]
    ) -> "FixedKernelSourceExpert":
        points = _as_points(X, name="X")
        values = _as_values(source_function(points), len(points), name="source_function(X)")
        return self.fit(points, values)

    def predict(
        self,
        X: Array,
        feature: FeatureName = "rank_quality",
        return_std: bool = False,
    ) -> Tuple[Array, Optional[Array]]:
        if feature not in {"rank_quality", "raw_value"}:
            raise ValueError("feature must be 'rank_quality' or 'raw_value'.")
        model = self.rank_gp_ if feature == "rank_quality" else self.raw_gp_
        if model is None or self.dim_ is None:
            raise RuntimeError("The requested source expert must be fitted first.")
        points = _as_points(X, dim=self.dim_, name="X")
        mean, std = model.predict(points, return_std=True)
        mean = np.asarray(mean, dtype=float).reshape(-1)
        if feature == "rank_quality":
            mean = np.clip(mean, 0.0, 1.0)
        std = np.maximum(np.asarray(std, dtype=float).reshape(-1), 1e-12)
        return (mean, std) if return_std else (mean, None)

    def predict_rank_quality(self, X: Array, return_std: bool = False):
        return self.predict(X, feature="rank_quality", return_std=return_std)

    def predict_raw_value(self, X: Array, return_std: bool = False):
        return self.predict(X, feature="raw_value", return_std=return_std)

    @property
    def raw_standardizer(self) -> Optional[RobustStandardizer]:
        return self.raw_standardizer_

    def _check_or_set_dim(self, points: Array) -> None:
        if self.dim_ is None:
            self.dim_ = int(points.shape[1])
        elif points.shape[1] != self.dim_:
            raise ValueError(f"X must have {self.dim_} columns.")


class SourceOracleExpert(FixedKernelSourceExpert):
    """Compatibility facade for the historical source-oracle API.

    It can be constructed empty and fitted with ``fit``/``fit_source_function``.
    For compatibility with the old constructor, passing two fitted GP objects is
    also accepted; in that case raw predictions are already in model units and
    no standardizer is applied.
    """

    def __init__(self, value_model=None, rank_model=None, config=None):
        super().__init__(config=config)
        if value_model is not None or rank_model is not None:
            if value_model is None or rank_model is None:
                raise ValueError("value_model and rank_model must be supplied together.")
            self.raw_gp_ = value_model
            self.rank_gp_ = rank_model
            self.dim_ = int(value_model.X_train_.shape[1])

    def predict(self, X: Array, feature: FeatureName = "raw_value", return_std: bool = False):
        return FixedKernelSourceExpert.predict(self, X, feature=feature, return_std=return_std)

    def predict_rank(self, X: Array, return_std: bool = False):
        return FixedKernelSourceExpert.predict(self, X, feature="rank_quality", return_std=return_std)

    def predict_rank_quality(self, X: Array, return_std: bool = False):
        return FixedKernelSourceExpert.predict(self, X, feature="rank_quality", return_std=return_std)

    def predict_raw_value(self, X: Array, return_std: bool = False):
        return FixedKernelSourceExpert.predict(self, X, feature="raw_value", return_std=return_std)


def fit_source_oracle_expert(
    X: Array,
    y: Array,
    config: Optional[Union[LocalSurrogateTransferConfig, OracleLocalModelTransferConfig]] = None,
    seed: Optional[int] = None,
) -> SourceOracleExpert:
    """Fit fixed source rank and robust-standardized raw-value experts."""
    cfg = config or LocalSurrogateTransferConfig()
    if seed is not None:
        if isinstance(cfg, OracleLocalModelTransferConfig):
            cfg = OracleLocalModelTransferConfig(
                gp_length_scale=cfg.gp_length_scale,
                gp_noise=cfg.gp_noise,
                calibration_ridge=cfg.calibration_ridge,
                random_state=int(seed),
            )
        else:
            cfg = LocalSurrogateTransferConfig(
                gp_length_scale=cfg.gp_length_scale,
                gp_noise=cfg.gp_noise,
                calibration_ridge=cfg.calibration_ridge,
                fixed_prior_scale=cfg.fixed_prior_scale,
                max_slope_in_target_std=cfg.max_slope_in_target_std,
                min_expert_std=cfg.min_expert_std,
                cv_folds=cfg.cv_folds,
                gate_min_relative_rmse_gain=cfg.gate_min_relative_rmse_gain,
                gate_min_pairwise_accuracy=cfg.gate_min_pairwise_accuracy,
                random_state=int(seed),
            )
    expert = SourceOracleExpert(config=cfg)
    expert.fit(X, y)
    return expert


class CalibratedFeatureResidualRegressor:
    """Non-negative feature calibration plus fixed-kernel target residual GP.

    The prior is ``intercept + standardized_features @ coefficients``.  Feature
    standardization is computed from target context.  The intercept is not
    regularized; coefficients are constrained to be non-negative with augmented
    ridge least squares.  Constant features and all-zero fitted coefficients
    fall back to an exactly identical target-only GP.  ``features=None`` is the
    canonical zero-feature target-only fit, so there is only one target-only
    implementation.
    """

    def __init__(
        self,
        config: Optional[Union[LocalSurrogateTransferConfig, OracleLocalModelTransferConfig]] = None,
        target_only: bool = False,
    ) -> None:
        self.config = config or LocalSurrogateTransferConfig()
        self.target_only = bool(target_only)
        self.gp_: Optional[GaussianProcessRegressor] = None
        self.dim_: Optional[int] = None
        self.n_features_: int = 0
        self.coefficients_: Optional[Array] = None
        self.intercept_: Optional[float] = None
        self.feature_means_: Optional[Array] = None
        self.feature_stds_: Optional[Array] = None
        self.effective_: Optional[str] = None
        self.effective_mode_: Optional[str] = None
        self.fallback_: bool = False
        self.fallback_reason_: Optional[str] = None

    def fit(
        self,
        X: Array,
        y: Array,
        features: Optional[Array] = None,
        feature_matrix: Optional[Array] = None,
    ) -> "CalibratedFeatureResidualRegressor":
        if features is not None and feature_matrix is not None:
            raise ValueError("Specify only one of features and feature_matrix.")
        if feature_matrix is not None:
            features = feature_matrix
        points = _as_points(X, name="X")
        values = _as_values(y, len(points), name="y")
        self.dim_ = int(points.shape[1])
        matrix = _as_features(features, len(points), name="features")
        self.n_features_ = int(matrix.shape[1])
        self.feature_means_ = np.mean(matrix, axis=0) if self.n_features_ else np.empty(0)
        raw_stds = np.std(matrix, axis=0, ddof=0) if self.n_features_ else np.empty(0)
        self.feature_stds_ = np.where(raw_stds > 1e-12, raw_stds, 1.0)
        self.coefficients_ = np.zeros(self.n_features_, dtype=float)
        self.intercept_ = float(np.mean(values))
        self.fallback_ = False
        self.fallback_reason_ = None

        if self.target_only or self.n_features_ == 0:
            self._fit_target_only(points, values, "target_only")
            return self
        if np.any(raw_stds <= 1e-12):
            self._fit_target_only(points, values, "constant_feature")
            return self

        standardized = (matrix - self.feature_means_) / self.feature_stds_
        coefficients, intercept = _fit_nonnegative_ridge(
            standardized, values, self._calibration_ridge
        )
        self.coefficients_ = coefficients
        self.intercept_ = float(intercept)
        if not np.any(coefficients > 1e-12):
            self._fit_target_only(points, values, "all_coefficients_zero")
            return self

        prior = self.intercept_ + standardized @ coefficients
        self.gp_ = _make_gp(points.shape[1], self.config)
        self.gp_.fit(points, values - prior)
        self.effective_ = "calibrated"
        self.effective_mode_ = self.effective_
        return self

    def predict(
        self,
        X: Array,
        features: Optional[Array] = None,
        feature_matrix: Optional[Array] = None,
        return_std: bool = False,
    ) -> Tuple[Array, Optional[Array]]:
        if features is not None and feature_matrix is not None:
            raise ValueError("Specify only one of features and feature_matrix.")
        if feature_matrix is not None:
            features = feature_matrix
        if self.gp_ is None or self.dim_ is None or self.effective_ is None:
            raise RuntimeError("The regressor must be fitted before prediction.")
        points = _as_points(X, dim=self.dim_, name="X")
        gp_mean, gp_std = self.gp_.predict(points, return_std=True)
        mean = np.asarray(gp_mean, dtype=float).reshape(-1)
        if self.effective_ == "calibrated":
            matrix = _as_features(features, len(points), name="features")
            if matrix.shape[1] != self.n_features_:
                raise ValueError(f"features must have {self.n_features_} columns.")
            standardized = (matrix - self.feature_means_) / self.feature_stds_
            mean = mean + self.intercept_ + standardized @ self.coefficients_
        std = np.maximum(np.asarray(gp_std, dtype=float).reshape(-1), 1e-12)
        return (mean, std) if return_std else (mean, None)

    @property
    def coefficients(self) -> Optional[Array]:
        return self.coefficients_

    @property
    def feature_means(self) -> Optional[Array]:
        return self.feature_means_

    @property
    def feature_stds(self) -> Optional[Array]:
        return self.feature_stds_

    @property
    def effective(self) -> Optional[str]:
        return self.effective_

    @property
    def fallback(self) -> bool:
        return self.fallback_

    def to_record(self) -> Dict[str, object]:
        return {
            "coefficients": [] if self.coefficients_ is None else self.coefficients_.tolist(),
            "intercept": self.intercept_,
            "feature_means": [] if self.feature_means_ is None else self.feature_means_.tolist(),
            "feature_stds": [] if self.feature_stds_ is None else self.feature_stds_.tolist(),
            "effective": self.effective_,
            "effective_mode": self.effective_mode_,
            "fallback": bool(self.fallback_),
            "fallback_reason": self.fallback_reason_,
            "n_features": int(self.n_features_),
        }

    record = to_record

    @property
    def _calibration_ridge(self) -> float:
        return float(getattr(self.config, "calibration_ridge", 1.0))

    def _fit_target_only(self, points: Array, values: Array, reason: str) -> None:
        self.gp_ = _make_gp(points.shape[1], self.config)
        self.gp_.fit(points, values)
        self.effective_ = "target_only"
        self.effective_mode_ = self.effective_
        self.fallback_ = reason != "target_only"
        self.fallback_reason_ = None if reason == "target_only" else reason


TargetOnlyCalibratedRegressor = CalibratedFeatureResidualRegressor


def fit_target_only(
    X: Array,
    y: Array,
    config: Optional[Union[LocalSurrogateTransferConfig, OracleLocalModelTransferConfig]] = None,
) -> CalibratedFeatureResidualRegressor:
    return CalibratedFeatureResidualRegressor(config=config, target_only=True).fit(X, y)


def _fit_nonnegative_ridge(features: Array, y: Array, ridge: float) -> Tuple[Array, float]:
    n_features = features.shape[1]
    design = np.column_stack([np.ones(len(features)), features])
    if ridge > 0.0:
        penalty = np.zeros((n_features, n_features + 1), dtype=float)
        penalty[:, 1:] = np.sqrt(ridge) * np.eye(n_features)
        design = np.vstack([design, penalty])
        target = np.concatenate([y, np.zeros(n_features, dtype=float)])
    else:
        target = y
    result = lsq_linear(
        design,
        target,
        bounds=(np.r_[-np.inf, np.zeros(n_features)], np.full(n_features + 1, np.inf)),
        lsmr_tol="auto",
        max_iter=1000,
    )
    if not np.all(np.isfinite(result.x)):
        raise ValueError("Unable to fit non-negative ridge calibration.")
    return np.maximum(np.asarray(result.x[1:], dtype=float), 0.0), float(result.x[0])


# ---- Historical multi-prior oracle facade ----------------------------------------
MODES = ("target_only", "geometry_prior", "oracle_rank", "oracle_value", "oracle_rank_value")


class OracleLocalModelTransfer:
    """Historical named facade using :class:`CalibratedFeatureResidualRegressor`."""

    def __init__(self, mode: str, config: Optional[OracleLocalModelTransferConfig] = None):
        if mode not in MODES:
            raise ValueError(f"Unknown oracle transfer mode: {mode}")
        self.mode = mode
        self.config = config or OracleLocalModelTransferConfig()
        self.model_: Optional[CalibratedFeatureResidualRegressor] = None
        self.gp_ = None
        self.dim_: Optional[int] = None
        self.effective_mode_: Optional[str] = None
        self.evidence_: Optional[OracleTransferEvidence] = None

    @property
    def prior_names(self) -> Tuple[str, ...]:
        return {
            "geometry_prior": ("geometry",),
            "oracle_rank": ("oracle_rank",),
            "oracle_value": ("oracle_value",),
            "oracle_rank_value": ("oracle_rank", "oracle_value"),
        }.get(self.mode, ())

    def fit(self, X: Array, y: Array, *, geometry_prior=None, oracle_rank=None, oracle_value=None):
        points = _as_points(X, name="X")
        values = _as_values(y, len(points), name="y")
        priors = self._collect_priors(len(points), geometry_prior, oracle_rank, oracle_value)
        features = np.column_stack([priors[name] for name in self.prior_names]) if priors else None
        self.model_ = CalibratedFeatureResidualRegressor(self.config).fit(points, values, features)
        self.gp_ = self.model_.gp_
        self.dim_ = points.shape[1]
        self.effective_mode_ = self.model_.effective_mode_
        if priors:
            self.evidence_ = OracleTransferEvidence(
                intercept=float(self.model_.intercept_),
                coefficients=tuple(float(v) for v in self.model_.coefficients_),
                prior_names=self.prior_names,
                prior_means=tuple(float(v) for v in self.model_.feature_means_),
                prior_stds=tuple(float(v) for v in self.model_.feature_stds_),
            )
        else:
            self.evidence_ = None
        return self

    def predict(self, X: Array, *, geometry_prior=None, oracle_rank=None, oracle_value=None, return_std=False):
        if self.model_ is None:
            raise RuntimeError("The model must be fitted before prediction")
        priors = self._collect_priors(len(_as_points(X, dim=self.dim_, name="X")), geometry_prior, oracle_rank, oracle_value)
        features = np.column_stack([priors[name] for name in self.prior_names]) if self.prior_names else None
        return self.model_.predict(X, features=features, return_std=return_std)

    def _collect_priors(self, n, geometry_prior, oracle_rank, oracle_value):
        wanted = set(self.prior_names)
        available = {}
        if "geometry" in wanted:
            available["geometry"] = _as_values(geometry_prior, n, "geometry_prior")
        if "oracle_rank" in wanted:
            available["oracle_rank"] = _as_values(oracle_rank, n, "oracle_rank")
        if "oracle_value" in wanted:
            available["oracle_value"] = _as_values(oracle_value, n, "oracle_value")
        return available


def geometry_prior_from_chart(chart: Array) -> Array:
    """Fixed source-independent squared radial geometry prior."""
    points = _as_points(chart, name="chart")
    return np.sum(points * points, axis=1)


# ---- Explicit oracle transforms and geometry -------------------------------------
def identity_transform(X: Array) -> Array:
    points, was_vector = _as_transform_points(X)
    return _restore_shape(points, was_vector)


def scale_transform(X: Array, scale: Union[float, Sequence[float]]) -> Array:
    """Map target coordinates to source coordinates by multiplying by scale."""
    points, was_vector = _as_transform_points(X)
    factor = np.asarray(scale, dtype=float)
    if factor.ndim == 0:
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError("scale must be finite and positive.")
    elif factor.ndim == 1 and factor.shape[0] == points.shape[1]:
        if not np.all(np.isfinite(factor)) or np.any(factor <= 0):
            raise ValueError("scale entries must be finite and positive.")
    else:
        raise ValueError("scale must be a positive scalar or one value per dimension.")
    return _restore_shape(points * factor, was_vector)


def rotate_transform(X: Array, angle: float) -> Array:
    """Map target coordinates to source coordinates for a 2-D rotation.

    For the Gate-0 row-vector convention the target-to-source map is
    ``theta_source = theta_target @ R(angle).T``.  This is the transpose of the
    usual column-vector rotation matrix and is the transform used by the runner.
    """
    points, was_vector = _as_transform_points(X)
    if points.shape[1] != 2:
        raise ValueError("rotate_transform is defined for exactly two dimensions.")
    if not np.isfinite(angle):
        raise ValueError("angle must be finite.")
    c, s = np.cos(float(angle)), np.sin(float(angle))
    rotation = np.array([[c, -s], [s, c]], dtype=float)
    return _restore_shape(points @ rotation.T, was_vector)


def oracle_coordinate_transform(X: Array, relation: OracleRelation = "identity", *, scale=1.0, angle: float = 0.0) -> Array:
    """Apply the target-to-source transform for a Gate-0 relation.

    The named quick-panel relations are accepted directly as convenience aliases;
    all non-coordinate relations intentionally use the identity map.
    """
    relation = str(relation).lower()
    if relation in {"identity", "matching", "output_affine", "roughness", "reversal", "independent", "independent_expert"}:
        return identity_transform(X)
    if relation in {"scale", "scale_0.7", "scale_1.5"}:
        if relation == "scale_0.7":
            scale = 0.7
        elif relation == "scale_1.5":
            scale = 1.5
        return scale_transform(X, scale)
    if relation in {"rotate", "rotate_45"}:
        if relation == "rotate_45":
            angle = np.pi / 4.0
        return rotate_transform(X, angle)
    raise ValueError("relation must be identity, scale, rotate, roughness, reversal, or independent.")


oracle_transform = oracle_coordinate_transform
transform_target_to_source = oracle_coordinate_transform


def make_oracle_transform(relation: OracleRelation = "identity", *, scale=1.0, angle: float = 0.0) -> Callable[[Array], Array]:
    return lambda X: oracle_coordinate_transform(X, relation, scale=scale, angle=angle)


def query_source_equivalent(source_function: Callable[[Array], Array], target_X: Array, relation: OracleRelation = "identity", *, scale=1.0, angle: float = 0.0) -> Array:
    return np.asarray(source_function(oracle_coordinate_transform(target_X, relation, scale=scale, angle=angle)), dtype=float).reshape(-1)


def radial_geometry_feature(X: Array, center: Optional[Array] = None, covariance: Optional[Array] = None) -> Array:
    """Return Euclidean or Mahalanobis radius for each point."""
    points = _as_points(X, name="X")
    anchor = np.zeros(points.shape[1]) if center is None else np.asarray(center, dtype=float).reshape(-1)
    if anchor.shape != (points.shape[1],) or not np.all(np.isfinite(anchor)):
        raise ValueError("center must be finite and match X's dimension.")
    diff = points - anchor[None, :]
    if covariance is None:
        return np.linalg.norm(diff, axis=1)
    cov = np.asarray(covariance, dtype=float)
    if cov.shape != (points.shape[1], points.shape[1]) or not np.all(np.isfinite(cov)):
        raise ValueError("covariance must be a finite square matrix matching X.")
    precision = np.linalg.pinv(0.5 * (cov + cov.T))
    return np.sqrt(np.maximum(np.sum((diff @ precision) * diff, axis=1), 0.0))


def radial_geometry_features(X: Array, center: Optional[Array] = None, covariance: Optional[Array] = None) -> Array:
    return radial_geometry_feature(X, center=center, covariance=covariance).reshape(-1, 1)


def radial_geometry_score(X: Array, center: Optional[Array] = None, covariance: Optional[Array] = None) -> Array:
    """Convert radius to the Gaussian radial membership score."""
    radius = radial_geometry_feature(X, center=center, covariance=covariance)
    return np.exp(-0.5 * radius * radius)


def _robust_standardizer(values: Array) -> RobustStandardizer:
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    scale = 1.4826 * mad
    if scale <= 1e-12:
        q25, q75 = np.quantile(values, [0.25, 0.75])
        scale = float((q75 - q25) / 1.349)
    if scale <= 1e-12:
        scale = float(np.std(values, ddof=0))
    return RobustStandardizer(center=center, scale=max(scale, 1e-12))


def _as_points(X: Array, dim: Optional[int] = None, name: str = "X") -> Array:
    points = np.asarray(X, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.ndim != 2 or len(points) == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional array.")
    if dim is not None and points.shape[1] != dim:
        raise ValueError(f"{name} must have {dim} columns.")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must contain finite values.")
    return points.copy()


def _as_values(values: Optional[Array], n: int, name: str) -> Array:
    if values is None:
        raise ValueError(f"{name} is required for this transfer mode")
    result = np.asarray(values, dtype=float).reshape(-1)
    if len(result) != n or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain exactly {n} finite values")
    return result.copy()


def _as_features(features: Optional[Array], n: int, name: str = "features") -> Array:
    if features is None:
        return np.empty((n, 0), dtype=float)
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim == 1:
        if len(matrix) != n:
            raise ValueError(f"{name} must contain exactly {n} rows")
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2 or matrix.shape[0] != n:
        raise ValueError(f"{name} must be a matrix with {n} rows")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain finite values")
    return matrix.copy()


def _as_transform_points(X: Array) -> Tuple[Array, bool]:
    original = np.asarray(X, dtype=float)
    return _as_points(original, name="X"), original.ndim == 1


def _restore_shape(points: Array, was_vector: bool) -> Array:
    return points.reshape(-1) if was_vector else points


OracleTransferConfig = OracleLocalModelTransferConfig
OracleLocalTransfer = OracleLocalModelTransfer

__all__ = [
    "MODES", "OracleLocalModelTransferConfig", "OracleTransferConfig",
    "OracleTransferEvidence", "OracleLocalModelTransfer", "OracleLocalTransfer",
    "RobustStandardizer", "FixedKernelSourceExpert", "SourceOracleExpert",
    "OracleLocalSourceExpert", "OracleSourceExpert", "fit_source_oracle_expert",
    "CalibratedFeatureResidualRegressor", "TargetOnlyCalibratedRegressor",
    "fit_target_only", "geometry_prior_from_chart", "identity_transform",
    "scale_transform", "rotate_transform", "oracle_coordinate_transform",
    "oracle_transform", "transform_target_to_source", "make_oracle_transform",
    "query_source_equivalent", "radial_geometry_feature", "radial_geometry_features",
    "radial_geometry_score", "rank_quality",
]

OracleLocalSourceExpert = FixedKernelSourceExpert
OracleSourceExpert = SourceOracleExpert
