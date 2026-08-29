"""Research utilities for source local-structure validation experiments.

The functions in this module enforce shared evaluation sets, independent random
streams, and instance-level metrics.  They are kept separate from the extractor so
that evaluation code cannot accidentally influence structure fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import qmc, rankdata, spearmanr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from .source_local_structure import (
    SourceLocalStructure,
    SourceLocalStructureLibrary,
)

Array = np.ndarray


@dataclass(frozen=True)
class RankingMetrics:
    spearman: float
    ndcg_at_top: float
    precision_at_top: float
    top_mean_quality: float
    enrichment: float
    normalized_top1_regret: float
    selected_y: float
    n_points: int


@dataclass(frozen=True)
class RecoveryMetrics:
    method: str
    basin_recall: float
    mean_matched_mahalanobis: float
    median_matched_mahalanobis: float
    normalized_center_error: float
    normalized_shape_error: float
    extracted_count: int
    oracle_count: int


class ControlledGaussianBasinLandscape:
    """Gaussian-mixture landscape with known local basin geometry."""

    def __init__(
        self,
        centers: Sequence[Array],
        covariances: Sequence[Array],
        weights: Sequence[float],
        bounds: Array,
        noise_std: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.centers = [np.asarray(center, dtype=float).reshape(-1) for center in centers]
        self.covariances = [np.asarray(cov, dtype=float) for cov in covariances]
        self.weights = np.asarray(weights, dtype=float).reshape(-1)
        self.bounds = np.asarray(bounds, dtype=float)
        self.dim = int(self.bounds.shape[0])
        self.noise_std = float(noise_std)
        self.rng = rng if rng is not None else np.random.default_rng(42)
        if not (
            len(self.centers)
            == len(self.covariances)
            == len(self.weights)
        ):
            raise ValueError("centers, covariances, and weights must have equal length.")
        self._precisions = [np.linalg.pinv(cov) for cov in self.covariances]

    def __call__(self, X: Array) -> Array:
        points = np.atleast_2d(np.asarray(X, dtype=float))
        values = np.zeros(len(points), dtype=float)
        for center, precision, weight in zip(
            self.centers,
            self._precisions,
            self.weights,
        ):
            diff = points - center[None, :]
            distance = np.sum((diff @ precision) * diff, axis=1)
            values += weight * np.exp(-0.5 * np.maximum(distance, 0.0))
        result = -values
        if self.noise_std > 0.0:
            result = result + self.rng.normal(0.0, self.noise_std, size=len(points))
        return result

    def oracle_structures(self) -> List[Dict]:
        return [
            {
                "center": center.copy(),
                "covariance": covariance.copy(),
                "weight": float(weight),
            }
            for center, covariance, weight in zip(
                self.centers,
                self.covariances,
                self.weights,
            )
        ]


def generate_controlled_landscape(
    dim: int,
    n_basins: int,
    rng: np.random.Generator,
    noise_std: float = 0.0,
) -> ControlledGaussianBasinLandscape:
    bounds = np.column_stack(
        [np.full(dim, -5.0, dtype=float), np.full(dim, 5.0, dtype=float)]
    )
    centers: List[Array] = []
    minimum_distance = 2.5 * np.sqrt(dim / 2.0)
    for _ in range(n_basins):
        for _attempt in range(10000):
            candidate = rng.uniform(-3.5, 3.5, size=dim)
            if not centers or min(np.linalg.norm(candidate - c) for c in centers) >= minimum_distance:
                centers.append(candidate)
                break
        else:
            raise RuntimeError("Unable to generate sufficiently separated basins.")

    covariances: List[Array] = []
    for _ in range(n_basins):
        matrix = rng.normal(size=(dim, dim))
        q_matrix, _ = np.linalg.qr(matrix)
        scales = rng.uniform(0.45, 1.05, size=dim)
        covariance = q_matrix @ np.diag(scales ** 2) @ q_matrix.T
        covariances.append(0.5 * (covariance + covariance.T))

    weights = np.linspace(1.0, 0.70, n_basins)
    return ControlledGaussianBasinLandscape(
        centers=centers,
        covariances=covariances,
        weights=weights,
        bounds=bounds,
        noise_std=noise_std,
        rng=rng,
    )


def latin_hypercube_sample(
    bounds: Array,
    n_samples: int,
    seed: int,
) -> Array:
    bounds_array = np.asarray(bounds, dtype=float)
    sampler = qmc.LatinHypercube(d=bounds_array.shape[0], seed=seed)
    unit = sampler.random(n=n_samples)
    return qmc.scale(unit, bounds_array[:, 0], bounds_array[:, 1])


def make_independent_test_pool(
    bounds: Array,
    library: SourceLocalStructureLibrary,
    rng: np.random.Generator,
    n_global: int,
    n_local_per_structure: int,
    local_covariance_scale: float = 2.0,
) -> Array:
    bounds_array = np.asarray(bounds, dtype=float)
    dim = bounds_array.shape[0]
    components: List[Array] = [
        rng.uniform(bounds_array[:, 0], bounds_array[:, 1], size=(n_global, dim))
    ]
    for structure in library.structures:
        local = rng.multivariate_normal(
            mean=structure.center,
            cov=structure.covariance * float(local_covariance_scale),
            size=n_local_per_structure,
        )
        local = np.clip(local, bounds_array[:, 0], bounds_array[:, 1])
        components.append(local)
    pool = np.vstack(components)
    return _deduplicate(pool)


def fit_global_source_gp(
    X: Array,
    y: Array,
    random_state: int,
) -> GaussianProcessRegressor:
    points = np.asarray(X, dtype=float)
    values = np.asarray(y, dtype=float).reshape(-1)
    dim = points.shape[1]
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(
            length_scale=np.ones(dim),
            length_scale_bounds=(1e-2, 1e2),
            nu=2.5,
        )
        + WhiteKernel(
            noise_level=1e-4,
            noise_level_bounds=(1e-6, 1e-1),
        )
    )
    model = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=1,
        random_state=random_state,
    )
    model.fit(points, values)
    return model


def best_point_distance_score(
    train_X: Array,
    train_y: Array,
    test_X: Array,
) -> Array:
    points = np.asarray(train_X, dtype=float)
    values = np.asarray(train_y, dtype=float).reshape(-1)
    best = points[int(np.argmin(values))]
    scale = np.maximum(np.std(points, axis=0), 1e-8)
    distance = np.linalg.norm((np.asarray(test_X, dtype=float) - best) / scale, axis=1)
    return -distance


def rank_quality(y: Array) -> Array:
    values = np.asarray(y, dtype=float).reshape(-1)
    if len(values) <= 1:
        return np.ones_like(values)
    ranks = rankdata(values, method="average")
    return 1.0 - (ranks - 1.0) / (len(values) - 1.0)


def evaluate_ranking(
    true_y: Array,
    scores: Array,
    top_fraction: float = 0.10,
    subset_mask: Optional[Array] = None,
) -> RankingMetrics:
    values = np.asarray(true_y, dtype=float).reshape(-1)
    prediction = np.asarray(scores, dtype=float).reshape(-1)
    if len(values) != len(prediction):
        raise ValueError("true_y and scores must have equal length.")

    if subset_mask is not None:
        mask = np.asarray(subset_mask, dtype=bool).reshape(-1)
        if len(mask) != len(values):
            raise ValueError("subset_mask has an invalid length.")
        values = values[mask]
        prediction = prediction[mask]
    if len(values) < 3:
        return RankingMetrics(
            spearman=float("nan"),
            ndcg_at_top=float("nan"),
            precision_at_top=float("nan"),
            top_mean_quality=float("nan"),
            enrichment=float("nan"),
            normalized_top1_regret=float("nan"),
            selected_y=float("nan"),
            n_points=int(len(values)),
        )

    quality = rank_quality(values)
    spearman = _safe_spearman(prediction, quality)
    k = max(1, int(np.ceil(len(values) * top_fraction)))
    selected = np.argsort(-prediction, kind="stable")[:k]
    truth = np.argsort(-quality, kind="stable")[:k]

    discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=float))
    gains = np.power(2.0, quality) - 1.0
    dcg = float(np.sum(gains[selected] * discounts))
    ideal = float(np.sum(gains[truth] * discounts))
    ndcg = dcg / ideal if ideal > 1e-12 else 0.0
    precision = float(len(set(selected.tolist()).intersection(truth.tolist())) / k)
    top_mean_quality = float(np.mean(quality[selected]))
    enrichment = float(top_mean_quality - np.mean(quality))

    selected_index = int(np.argmax(prediction))
    oracle = float(np.min(values))
    scale = max(1e-12, float(np.quantile(values, 0.90) - oracle))
    regret = float((values[selected_index] - oracle) / scale)
    return RankingMetrics(
        spearman=float(spearman),
        ndcg_at_top=float(ndcg),
        precision_at_top=precision,
        top_mean_quality=top_mean_quality,
        enrichment=enrichment,
        normalized_top1_regret=regret,
        selected_y=float(values[selected_index]),
        n_points=int(len(values)),
    )


def local_subset_mask(
    geometry_scores: Array,
    fraction: float = 0.35,
    minimum_points: int = 30,
) -> Array:
    scores = np.asarray(geometry_scores, dtype=float).reshape(-1)
    if len(scores) == 0:
        return np.zeros(0, dtype=bool)
    count = min(len(scores), max(minimum_points, int(np.ceil(len(scores) * fraction))))
    order = np.argsort(-scores, kind="stable")
    mask = np.zeros(len(scores), dtype=bool)
    mask[order[:count]] = True
    return mask


def recovery_metrics(
    structures: Sequence[SourceLocalStructure],
    oracle_centers: Sequence[Array],
    oracle_covariances: Sequence[Array],
    bounds: Array,
    method: str = "Proposed-Structure",
    hit_threshold: float = 2.5,
) -> RecoveryMetrics:
    extracted_centers = [np.asarray(item.center, dtype=float) for item in structures]
    extracted_covariances = [np.asarray(item.covariance, dtype=float) for item in structures]
    return recovery_metrics_from_arrays(
        extracted_centers,
        extracted_covariances,
        oracle_centers,
        oracle_covariances,
        bounds,
        method=method,
        hit_threshold=hit_threshold,
    )


def recovery_metrics_from_arrays(
    extracted_centers: Sequence[Array],
    extracted_covariances: Sequence[Array],
    oracle_centers: Sequence[Array],
    oracle_covariances: Sequence[Array],
    bounds: Array,
    method: str,
    hit_threshold: float = 2.5,
) -> RecoveryMetrics:
    estimated = [np.asarray(center, dtype=float) for center in extracted_centers]
    oracle = [np.asarray(center, dtype=float) for center in oracle_centers]
    oracle_cov = [np.asarray(cov, dtype=float) for cov in oracle_covariances]
    estimated_cov = [np.asarray(cov, dtype=float) for cov in extracted_covariances]

    if not estimated or not oracle:
        return RecoveryMetrics(
            method=method,
            basin_recall=0.0,
            mean_matched_mahalanobis=float("inf"),
            median_matched_mahalanobis=float("inf"),
            normalized_center_error=float("inf"),
            normalized_shape_error=float("nan"),
            extracted_count=len(estimated),
            oracle_count=len(oracle),
        )

    cost = np.zeros((len(estimated), len(oracle)), dtype=float)
    euclidean = np.zeros_like(cost)
    for i, estimate in enumerate(estimated):
        for j, center in enumerate(oracle):
            diff = estimate - center
            precision = np.linalg.pinv(oracle_cov[j])
            cost[i, j] = np.sqrt(max(0.0, float(diff @ precision @ diff)))
            euclidean[i, j] = np.linalg.norm(diff)

    row_index, column_index = linear_sum_assignment(cost)
    matched = cost[row_index, column_index]
    hits = int(np.sum(matched <= hit_threshold))
    recall = hits / len(oracle)

    domain_scale = np.linalg.norm(
        np.asarray(bounds, dtype=float)[:, 1] - np.asarray(bounds, dtype=float)[:, 0]
    )
    center_error = float(
        np.mean(euclidean[row_index, column_index]) / max(domain_scale, 1e-12)
    )

    shape_errors: List[float] = []
    for estimated_index, oracle_index in zip(row_index, column_index):
        if estimated_index >= len(estimated_cov):
            continue
        first = estimated_cov[estimated_index]
        second = oracle_cov[oracle_index]
        first = first / max(np.trace(first), 1e-12)
        second = second / max(np.trace(second), 1e-12)
        shape_errors.append(
            float(np.linalg.norm(first - second, ord="fro") / np.sqrt(first.shape[0]))
        )

    return RecoveryMetrics(
        method=method,
        basin_recall=float(recall),
        mean_matched_mahalanobis=float(np.mean(matched)),
        median_matched_mahalanobis=float(np.median(matched)),
        normalized_center_error=center_error,
        normalized_shape_error=(float(np.mean(shape_errors)) if shape_errors else float("nan")),
        extracted_count=len(estimated),
        oracle_count=len(oracle),
    )


def top_observation_baseline(
    X: Array,
    y: Array,
    n_centers: int,
    default_covariance: Array,
) -> Tuple[List[Array], List[Array]]:
    points = np.asarray(X, dtype=float)
    values = np.asarray(y, dtype=float).reshape(-1)
    indices = np.argsort(values, kind="stable")[:n_centers]
    centers = [points[index].copy() for index in indices]
    covariances = [np.asarray(default_covariance, dtype=float).copy() for _ in centers]
    return centers, covariances


def random_center_baseline(
    bounds: Array,
    n_centers: int,
    dim: int,
    rng: np.random.Generator,
) -> Tuple[List[Array], List[Array]]:
    bounds_array = np.asarray(bounds, dtype=float)
    centers = [
        rng.uniform(bounds_array[:, 0], bounds_array[:, 1])
        for _ in range(n_centers)
    ]
    scale = np.mean((bounds_array[:, 1] - bounds_array[:, 0]) ** 2) / 100.0
    covariances = [np.eye(dim) * scale for _ in centers]
    return centers, covariances


def _safe_spearman(a: Array, b: Array) -> float:
    first = np.asarray(a, dtype=float).reshape(-1)
    second = np.asarray(b, dtype=float).reshape(-1)
    if len(first) < 3 or np.std(first) < 1e-12 or np.std(second) < 1e-12:
        return 0.0
    value = spearmanr(first, second).statistic
    return float(value) if np.isfinite(value) else 0.0


def _deduplicate(X: Array, decimals: int = 12) -> Array:
    points = np.asarray(X, dtype=float)
    rounded = np.round(points, decimals=decimals)
    _, indices = np.unique(rounded, axis=0, return_index=True)
    return points[np.sort(indices)]


__all__ = [
    "ControlledGaussianBasinLandscape",
    "RankingMetrics",
    "RecoveryMetrics",
    "best_point_distance_score",
    "evaluate_ranking",
    "fit_global_source_gp",
    "generate_controlled_landscape",
    "latin_hypercube_sample",
    "local_subset_mask",
    "make_independent_test_pool",
    "random_center_baseline",
    "rank_quality",
    "recovery_metrics",
    "recovery_metrics_from_arrays",
    "top_observation_baseline",
]
