"""Controlled-study utilities for source-local-surrogate transfer.

The controlled pilot fixes region correspondence and chart scale externally.  It
therefore asks only whether an already extracted source local rank model supplies
incremental target prediction information.  Region discovery, region selection,
alignment estimation, acquisition, and sequential optimization are out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

import numpy as np
from scipy.stats import qmc, rankdata, spearmanr

from .source_local_structure import SourceLocalStructure, SourceLocalStructureLibrary

Array = np.ndarray


@dataclass(frozen=True)
class AlignedLocalExpert:
    """A source structure queried through a frozen local chart correspondence.

    Only a translation and a task-independent scalar chart radius are used.  The
    extracted full covariance is intentionally *not* used for source-target
    alignment because the preceding recovery study did not establish superior
    covariance-shape recovery.
    """

    structure: SourceLocalStructure
    source_anchor: Array
    target_anchor: Array
    chart_radius: float
    reverse_quality: bool = False

    def __post_init__(self) -> None:
        source = np.asarray(self.source_anchor, dtype=float).reshape(-1)
        target = np.asarray(self.target_anchor, dtype=float).reshape(-1)
        if len(source) != self.structure.dim or len(target) != self.structure.dim:
            raise ValueError("Expert anchors must match the structure dimension.")
        if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
            raise ValueError("Expert anchors must be finite.")
        if self.chart_radius <= 0.0:
            raise ValueError("chart_radius must be positive.")
        object.__setattr__(self, "source_anchor", source.copy())
        object.__setattr__(self, "target_anchor", target.copy())

    @property
    def dim(self) -> int:
        return self.structure.dim

    def target_points_from_chart(self, Z: Array) -> Array:
        chart = _as_points(Z, self.dim, name="Z")
        return self.target_anchor[None, :] + self.chart_radius * chart

    def source_points_from_chart(self, Z: Array) -> Array:
        chart = _as_points(Z, self.dim, name="Z")
        return self.source_anchor[None, :] + self.chart_radius * chart

    def predict_quality_from_chart(
        self,
        Z: Array,
        return_std: bool = False,
    ) -> tuple[Array, Optional[Array]]:
        source_points = self.source_points_from_chart(Z)
        mean, std = self.structure.predict_relative_quality(
            source_points,
            return_std=return_std,
        )
        if self.reverse_quality:
            mean = 1.0 - mean
        return mean, std

    def source_membership_from_chart(self, Z: Array) -> Array:
        return self.structure.membership(self.source_points_from_chart(Z))


@dataclass(frozen=True)
class PredictionMetrics:
    standardized_rmse: float
    ndcg_at_top: float
    spearman: float
    pairwise_accuracy: float
    precision_at_top: float
    normalized_top1_regret: float
    mean_negative_log_likelihood: float
    interval_coverage_95: float
    n_test: int


def select_structure_near_anchor(
    library: SourceLocalStructureLibrary,
    anchor: Array,
) -> SourceLocalStructure:
    """Oracle association used only to remove region-selection confounding."""

    if not library.structures:
        raise ValueError("The source structure library is empty.")
    point = np.asarray(anchor, dtype=float).reshape(-1)
    candidates = [
        np.linalg.norm(structure.center - point)
        for structure in library.structures
    ]
    return library.structures[int(np.argmin(candidates))]


def oracle_global_anchor(landscape: object) -> Array:
    """Return the declared global-basin center of a controlled landscape."""

    basins = landscape.get_oracle_basins()
    if not basins:
        raise ValueError("Landscape does not expose oracle basins.")
    global_basins = [item for item in basins if bool(item.get("is_global", False))]
    selected = global_basins[0] if global_basins else max(
        basins,
        key=lambda item: float(item.get("weight", 0.0)),
    )
    return np.asarray(selected["center"], dtype=float).reshape(-1)


def sobol_chart_design(
    dim: int,
    n_points: int,
    seed: int,
    lower: float = -1.0,
    upper: float = 1.0,
) -> Array:
    """Generate a reproducible prefix-stable scrambled Sobol chart design."""

    if dim < 1 or n_points < 1:
        raise ValueError("dim and n_points must be positive.")
    if not lower < upper:
        raise ValueError("lower must be smaller than upper.")
    exponent = int(np.ceil(np.log2(n_points)))
    unit = qmc.Sobol(d=dim, scramble=True, seed=seed).random_base2(exponent)
    unit = unit[:n_points]
    return qmc.scale(
        unit,
        np.full(dim, lower, dtype=float),
        np.full(dim, upper, dtype=float),
    )


def evaluate_predictions(
    true_y: Array,
    predicted_mean: Array,
    predicted_std: Optional[Array] = None,
    top_fraction: float = 0.10,
) -> PredictionMetrics:
    """Evaluate one frozen target panel; panel points are not replicates."""

    truth = np.asarray(true_y, dtype=float).reshape(-1)
    prediction = np.asarray(predicted_mean, dtype=float).reshape(-1)
    if len(truth) != len(prediction) or len(truth) < 3:
        raise ValueError("Prediction metrics require equal arrays of length >= 3.")
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(prediction)):
        raise ValueError("Prediction metrics require finite values.")
    if not 0.0 < top_fraction < 0.5:
        raise ValueError("top_fraction must lie in (0, 0.5).")

    scale = max(float(np.std(truth, ddof=0)), 1e-12)
    standardized_rmse = float(
        np.sqrt(np.mean((prediction - truth) ** 2)) / scale
    )
    quality = rank_quality(truth)
    score = -prediction
    spearman = safe_spearman(score, quality)
    pairwise = pairwise_cost_accuracy(prediction, truth)

    k = max(1, int(np.ceil(len(truth) * top_fraction)))
    selected = np.argsort(-score, kind="stable")[:k]
    ideal = np.argsort(-quality, kind="stable")[:k]
    discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=float))
    gains = np.power(2.0, quality) - 1.0
    dcg = float(np.sum(gains[selected] * discounts))
    ideal_dcg = float(np.sum(gains[ideal] * discounts))
    ndcg = dcg / ideal_dcg if ideal_dcg > 1e-12 else 0.0
    precision = float(len(set(selected.tolist()).intersection(ideal.tolist())) / k)

    selected_index = int(np.argmin(prediction))
    oracle = float(np.min(truth))
    regret_scale = max(float(np.quantile(truth, 0.90) - oracle), 1e-12)
    normalized_regret = float((truth[selected_index] - oracle) / regret_scale)

    if predicted_std is None:
        mean_nll = float("nan")
        coverage = float("nan")
    else:
        std = np.asarray(predicted_std, dtype=float).reshape(-1)
        if len(std) != len(truth) or not np.all(np.isfinite(std)):
            raise ValueError("predicted_std must match true_y and be finite.")
        std = np.maximum(std, 1e-9)
        residual = truth - prediction
        mean_nll = float(
            np.mean(0.5 * np.log(2.0 * np.pi * std**2) + 0.5 * (residual / std) ** 2)
        )
        coverage = float(np.mean(np.abs(residual) <= 1.96 * std))

    return PredictionMetrics(
        standardized_rmse=standardized_rmse,
        ndcg_at_top=float(ndcg),
        spearman=float(spearman),
        pairwise_accuracy=float(pairwise),
        precision_at_top=precision,
        normalized_top1_regret=normalized_regret,
        mean_negative_log_likelihood=mean_nll,
        interval_coverage_95=coverage,
        n_test=int(len(truth)),
    )


def rank_quality(y: Array) -> Array:
    values = np.asarray(y, dtype=float).reshape(-1)
    if len(values) <= 1:
        return np.ones_like(values)
    ranks = rankdata(values, method="average")
    return 1.0 - (ranks - 1.0) / (len(values) - 1.0)


def safe_spearman(a: Array, b: Array) -> float:
    first = np.asarray(a, dtype=float).reshape(-1)
    second = np.asarray(b, dtype=float).reshape(-1)
    if len(first) < 3 or np.std(first) < 1e-12 or np.std(second) < 1e-12:
        return 0.0
    value = spearmanr(first, second).statistic
    return float(value) if np.isfinite(value) else 0.0


def pairwise_cost_accuracy(first: Array, second: Array) -> float:
    first_values = np.asarray(first, dtype=float).reshape(-1)
    second_values = np.asarray(second, dtype=float).reshape(-1)
    if len(first_values) != len(second_values):
        raise ValueError("Pairwise inputs must have equal length.")
    if len(first_values) < 2:
        return 0.5
    i, j = np.triu_indices(len(first_values), k=1)
    first_difference = first_values[i] - first_values[j]
    second_difference = second_values[i] - second_values[j]
    usable = (np.abs(first_difference) > 1e-12) & (
        np.abs(second_difference) > 1e-12
    )
    if not np.any(usable):
        return 0.5
    return float(
        np.mean(
            np.sign(first_difference[usable])
            == np.sign(second_difference[usable])
        )
    )


def bounded_chart_radius(bounds: Array, fraction: float) -> float:
    bounds_array = np.asarray(bounds, dtype=float)
    if bounds_array.ndim != 2 or bounds_array.shape[1] != 2:
        raise ValueError("bounds must have shape (d, 2).")
    if not 0.0 < fraction < 0.5:
        raise ValueError("fraction must lie in (0, 0.5).")
    return float(fraction * np.mean(bounds_array[:, 1] - bounds_array[:, 0]))


def chart_evaluator(
    function: Callable[[Array], Array],
    anchor: Array,
    radius: float,
) -> Callable[[Array], Array]:
    center = np.asarray(anchor, dtype=float).reshape(-1)
    if radius <= 0.0:
        raise ValueError("radius must be positive.")

    def evaluate(Z: Array) -> Array:
        chart = _as_points(Z, len(center), name="Z")
        points = center[None, :] + radius * chart
        return np.asarray(function(points), dtype=float).reshape(-1)

    return evaluate


def source_support_diagnostics(
    expert: AlignedLocalExpert,
    chart_sets: Sequence[Array],
) -> Dict[str, float]:
    membership = np.concatenate(
        [expert.source_membership_from_chart(chart) for chart in chart_sets]
    )
    return {
        "source_membership_mean": float(np.mean(membership)),
        "source_membership_min": float(np.min(membership)),
        "source_membership_below_0_05": float(np.mean(membership < 0.05)),
    }


def _as_points(X: Array, dim: int, name: str) -> Array:
    points = np.asarray(X, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.ndim != 2 or points.shape[1] != dim or len(points) == 0:
        raise ValueError(f"{name} must have shape (n, {dim}).")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must contain finite values.")
    return points.copy()


__all__ = [
    "AlignedLocalExpert",
    "PredictionMetrics",
    "bounded_chart_radius",
    "chart_evaluator",
    "evaluate_predictions",
    "oracle_global_anchor",
    "pairwise_cost_accuracy",
    "rank_quality",
    "safe_spearman",
    "select_structure_near_anchor",
    "sobol_chart_design",
    "source_support_diagnostics",
]
