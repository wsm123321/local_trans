"""Target-calibrated transfer of a source local rank surrogate.

This module contains the deliberately small model intervention used by the first
local-surrogate transfer pilot.  A source expert supplies a scalar relative-quality
score.  Target observations may calibrate that score as a prior mean, while a GP
models target residuals.  The target-only model and every transfer condition share
exactly the same fixed GP kernel and numerical settings.

No region matching or Bayesian-optimization logic is implemented here.  The pilot
uses an externally supplied (oracle in the controlled study) region correspondence
so that model transfer is not confounded with region discovery or alignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import KFold

Array = np.ndarray
TransferMode = Literal["target_only", "fixed", "calibrated", "gated"]


@dataclass(frozen=True)
class LocalSurrogateTransferConfig:
    """Frozen settings for target-side local-surrogate transfer."""

    gp_length_scale: float = 1.0
    gp_noise: float = 1e-4
    calibration_ridge: float = 1.0
    fixed_prior_scale: float = 1.0
    max_slope_in_target_std: float = 3.0
    min_expert_std: float = 1e-6
    cv_folds: int = 4
    gate_min_relative_rmse_gain: float = 0.0
    gate_min_pairwise_accuracy: float = 0.55
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.gp_length_scale <= 0.0:
            raise ValueError("gp_length_scale must be positive.")
        if self.gp_noise <= 0.0:
            raise ValueError("gp_noise must be positive.")
        if self.calibration_ridge < 0.0:
            raise ValueError("calibration_ridge must be non-negative.")
        if self.fixed_prior_scale < 0.0:
            raise ValueError("fixed_prior_scale must be non-negative.")
        if self.max_slope_in_target_std <= 0.0:
            raise ValueError("max_slope_in_target_std must be positive.")
        if self.min_expert_std < 0.0:
            raise ValueError("min_expert_std must be non-negative.")
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least two.")
        if not 0.0 <= self.gate_min_pairwise_accuracy <= 1.0:
            raise ValueError("gate_min_pairwise_accuracy must lie in [0, 1].")


@dataclass(frozen=True)
class AffineSourceCalibration:
    """Affine map from source relative quality to target response units.

    The source expert reports quality in [0, 1], where larger is better.  The
    calibration works with ``source_cost = 1 - quality`` so that a positive slope
    represents order-consistent transfer for a minimization problem.
    """

    source_cost_mean: float
    source_cost_std: float
    intercept: float
    slope: float
    raw_slope: float

    def predict(self, source_quality: Array) -> Array:
        quality = _as_quality(source_quality, name="source_quality")
        cost = 1.0 - quality
        standardized = (cost - self.source_cost_mean) / self.source_cost_std
        return self.intercept + self.slope * standardized


@dataclass(frozen=True)
class TransferEvidence:
    """Target-context evidence used by the conservative transfer gate."""

    accepted: bool
    cv_target_rmse: float
    cv_transfer_rmse: float
    relative_rmse_gain: float
    pairwise_accuracy: float
    calibration_slope: float
    calibration_raw_slope: float
    expert_std: float
    n_observations: int
    n_folds: int
    rejection_reason: str


class LocalExpertResidualRegressor:
    """Fixed-kernel local GP with an optional source-expert prior mean.

    Modes
    -----
    ``target_only``
        Fit the shared GP directly to target observations.
    ``fixed``
        Use an unconditionally positive source prior whose amplitude is set by the
        target response standard deviation, then fit a residual GP.
    ``calibrated``
        Estimate a non-negative, ridge-shrunk source slope from target observations,
        then fit a residual GP.  A non-positive association shrinks to target-only.
    ``gated``
        Use the calibrated model only when target-context cross-validation and
        pairwise ordering evidence pass the frozen gate; otherwise refit the exact
        target-only model.
    """

    def __init__(
        self,
        mode: TransferMode,
        config: Optional[LocalSurrogateTransferConfig] = None,
    ) -> None:
        if mode not in {"target_only", "fixed", "calibrated", "gated"}:
            raise ValueError(f"Unknown transfer mode: {mode}")
        self.mode = mode
        self.config = config or LocalSurrogateTransferConfig()
        self.effective_mode_: Optional[TransferMode] = None
        self.calibration_: Optional[AffineSourceCalibration] = None
        self.calibration_attempt_: Optional[AffineSourceCalibration] = None
        self.evidence_: Optional[TransferEvidence] = None
        self.gp_: Optional[GaussianProcessRegressor] = None
        self.dim_: Optional[int] = None

    def fit(
        self,
        X: Array,
        y: Array,
        source_quality: Optional[Array] = None,
    ) -> "LocalExpertResidualRegressor":
        points = _as_points(X, name="X")
        values = _as_values(y, len(points), name="y")
        self.dim_ = int(points.shape[1])

        if self.mode == "target_only":
            self._fit_target_only(points, values)
            return self

        quality = _as_quality(source_quality, n=len(points), name="source_quality")
        if self.mode == "gated":
            self.evidence_ = cross_validated_transfer_evidence(
                points,
                values,
                quality,
                self.config,
            )
            if not self.evidence_.accepted:
                self._fit_target_only(points, values)
                return self
            effective: TransferMode = "calibrated"
        else:
            effective = self.mode

        calibration = _fit_source_calibration(
            quality,
            values,
            mode=effective,
            config=self.config,
        )
        self.calibration_attempt_ = calibration
        if calibration is None or calibration.slope <= 0.0:
            self._fit_target_only(points, values)
            return self

        residual = values - calibration.predict(quality)
        self.gp_ = _make_gp(self.dim_, self.config)
        self.gp_.fit(points, residual)
        self.calibration_ = calibration
        self.effective_mode_ = effective
        return self

    def predict(
        self,
        X: Array,
        source_quality: Optional[Array] = None,
        return_std: bool = False,
    ) -> Tuple[Array, Optional[Array]]:
        if self.gp_ is None or self.effective_mode_ is None or self.dim_ is None:
            raise RuntimeError("The regressor must be fitted before prediction.")
        points = _as_points(X, dim=self.dim_, name="X")

        gp_mean, gp_std = self.gp_.predict(points, return_std=True)
        mean = np.asarray(gp_mean, dtype=float).reshape(-1)
        if self.effective_mode_ != "target_only":
            quality = _as_quality(source_quality, n=len(points), name="source_quality")
            if self.calibration_ is None:
                raise RuntimeError("Transfer calibration is missing.")
            mean = mean + self.calibration_.predict(quality)

        std = np.maximum(np.asarray(gp_std, dtype=float).reshape(-1), 1e-12)
        return (mean, std) if return_std else (mean, None)

    def _fit_target_only(self, X: Array, y: Array) -> None:
        self.gp_ = _make_gp(X.shape[1], self.config)
        self.gp_.fit(X, y)
        self.calibration_ = None
        self.effective_mode_ = "target_only"


def cross_validated_transfer_evidence(
    X: Array,
    y: Array,
    source_quality: Array,
    config: Optional[LocalSurrogateTransferConfig] = None,
) -> TransferEvidence:
    """Estimate pre-decision target evidence without using held-out test labels."""

    cfg = config or LocalSurrogateTransferConfig()
    points = _as_points(X, name="X")
    values = _as_values(y, len(points), name="y")
    quality = _as_quality(source_quality, n=len(points), name="source_quality")
    expert_std = float(np.std(quality, ddof=0))

    if len(points) < 4:
        return _rejected_evidence(
            values,
            quality,
            expert_std,
            reason="fewer_than_four_target_observations",
        )
    if expert_std < cfg.min_expert_std:
        return _rejected_evidence(
            values,
            quality,
            expert_std,
            reason="expert_score_has_negligible_variance",
        )

    n_folds = min(cfg.cv_folds, max(2, len(points) // 2))
    splitter = KFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=cfg.random_state,
    )
    target_prediction = np.full(len(points), np.nan, dtype=float)
    transfer_prediction = np.full(len(points), np.nan, dtype=float)

    for train_index, validation_index in splitter.split(points):
        target_model = LocalExpertResidualRegressor("target_only", cfg).fit(
            points[train_index],
            values[train_index],
        )
        transfer_model = LocalExpertResidualRegressor("calibrated", cfg).fit(
            points[train_index],
            values[train_index],
            quality[train_index],
        )
        target_prediction[validation_index] = target_model.predict(
            points[validation_index]
        )[0]
        transfer_prediction[validation_index] = transfer_model.predict(
            points[validation_index],
            quality[validation_index],
        )[0]

    target_rmse = float(np.sqrt(np.mean((target_prediction - values) ** 2)))
    transfer_rmse = float(np.sqrt(np.mean((transfer_prediction - values) ** 2)))
    relative_gain = float(
        (target_rmse - transfer_rmse) / max(target_rmse, 1e-12)
    )
    pairwise = pairwise_order_accuracy(1.0 - quality, values)
    calibration = _fit_source_calibration(
        quality,
        values,
        mode="calibrated",
        config=cfg,
    )
    slope = 0.0 if calibration is None else float(calibration.slope)
    raw_slope = 0.0 if calibration is None else float(calibration.raw_slope)

    reasons = []
    if slope <= 0.0:
        reasons.append("non_positive_calibrated_slope")
    if relative_gain <= cfg.gate_min_relative_rmse_gain:
        reasons.append("insufficient_cv_rmse_gain")
    if pairwise < cfg.gate_min_pairwise_accuracy:
        reasons.append("insufficient_pairwise_agreement")
    accepted = not reasons
    return TransferEvidence(
        accepted=accepted,
        cv_target_rmse=target_rmse,
        cv_transfer_rmse=transfer_rmse,
        relative_rmse_gain=relative_gain,
        pairwise_accuracy=pairwise,
        calibration_slope=slope,
        calibration_raw_slope=raw_slope,
        expert_std=expert_std,
        n_observations=int(len(points)),
        n_folds=int(n_folds),
        rejection_reason="accepted" if accepted else ";".join(reasons),
    )


def fit_affine_source_calibration(
    source_quality: Array,
    y: Array,
    config: Optional[LocalSurrogateTransferConfig] = None,
    fixed: bool = False,
) -> Optional[AffineSourceCalibration]:
    """Fit the public source-only affine calibration used by study baselines."""

    cfg = config or LocalSurrogateTransferConfig()
    return _fit_source_calibration(
        source_quality,
        y,
        mode="fixed" if fixed else "calibrated",
        config=cfg,
    )


def pairwise_order_accuracy(first_cost: Array, second_cost: Array) -> float:
    """Fraction of non-tied pairs with the same minimization ordering."""

    first = np.asarray(first_cost, dtype=float).reshape(-1)
    second = np.asarray(second_cost, dtype=float).reshape(-1)
    if len(first) != len(second):
        raise ValueError("Pairwise arrays must have equal length.")
    if len(first) < 2:
        return 0.5
    i, j = np.triu_indices(len(first), k=1)
    first_difference = first[i] - first[j]
    second_difference = second[i] - second[j]
    usable = (np.abs(first_difference) > 1e-12) & (
        np.abs(second_difference) > 1e-12
    )
    if not np.any(usable):
        return 0.5
    concordant = np.sign(first_difference[usable]) == np.sign(
        second_difference[usable]
    )
    return float(np.mean(concordant))


def _fit_source_calibration(
    source_quality: Array,
    y: Array,
    mode: TransferMode,
    config: LocalSurrogateTransferConfig,
) -> Optional[AffineSourceCalibration]:
    quality = _as_quality(source_quality, name="source_quality")
    values = _as_values(y, len(quality), name="y")
    source_cost = 1.0 - quality
    cost_mean = float(np.mean(source_cost))
    cost_std = float(np.std(source_cost, ddof=0))
    if cost_std < config.min_expert_std:
        return None

    standardized_cost = (source_cost - cost_mean) / cost_std
    intercept = float(np.mean(values))
    target_centered = values - intercept
    target_std = max(float(np.std(values, ddof=0)), 1e-12)

    if mode == "fixed":
        raw_slope = config.fixed_prior_scale * target_std
        slope = raw_slope
    elif mode == "calibrated":
        denominator = float(
            standardized_cost @ standardized_cost + config.calibration_ridge
        )
        raw_slope = float(standardized_cost @ target_centered / denominator)
        slope = float(
            np.clip(
                raw_slope,
                0.0,
                config.max_slope_in_target_std * target_std,
            )
        )
    else:
        raise ValueError("Calibration mode must be 'fixed' or 'calibrated'.")

    return AffineSourceCalibration(
        source_cost_mean=cost_mean,
        source_cost_std=cost_std,
        intercept=intercept,
        slope=slope,
        raw_slope=raw_slope,
    )


def _make_gp(
    dim: int,
    config: LocalSurrogateTransferConfig,
) -> GaussianProcessRegressor:
    kernel = (
        ConstantKernel(1.0, constant_value_bounds="fixed")
        * Matern(
            length_scale=np.full(dim, config.gp_length_scale, dtype=float),
            length_scale_bounds="fixed",
            nu=2.5,
        )
        + WhiteKernel(
            noise_level=config.gp_noise,
            noise_level_bounds="fixed",
        )
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-10,
        optimizer=None,
        normalize_y=True,
        random_state=config.random_state,
    )


def _rejected_evidence(
    y: Array,
    quality: Array,
    expert_std: float,
    reason: str,
) -> TransferEvidence:
    baseline = float(np.sqrt(np.mean((y - np.mean(y)) ** 2)))
    return TransferEvidence(
        accepted=False,
        cv_target_rmse=baseline,
        cv_transfer_rmse=baseline,
        relative_rmse_gain=0.0,
        pairwise_accuracy=pairwise_order_accuracy(1.0 - quality, y),
        calibration_slope=0.0,
        calibration_raw_slope=0.0,
        expert_std=expert_std,
        n_observations=int(len(y)),
        n_folds=0,
        rejection_reason=reason,
    )


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


def _as_values(y: Array, n: int, name: str = "y") -> Array:
    values = np.asarray(y, dtype=float).reshape(-1)
    if len(values) != n:
        raise ValueError(f"{name} must contain exactly {n} values.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain finite values.")
    return values.copy()


def _as_quality(
    source_quality: Optional[Array],
    n: Optional[int] = None,
    name: str = "source_quality",
) -> Array:
    if source_quality is None:
        raise ValueError(f"{name} is required for a transfer model.")
    quality = np.asarray(source_quality, dtype=float).reshape(-1)
    if n is not None and len(quality) != n:
        raise ValueError(f"{name} must contain exactly {n} values.")
    if len(quality) == 0 or not np.all(np.isfinite(quality)):
        raise ValueError(f"{name} must contain finite values.")
    return np.clip(quality, 0.0, 1.0)


__all__ = [
    "AffineSourceCalibration",
    "LocalExpertResidualRegressor",
    "LocalSurrogateTransferConfig",
    "TransferEvidence",
    "cross_validated_transfer_evidence",
    "fit_affine_source_calibration",
    "pairwise_order_accuracy",
]
