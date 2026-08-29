"""Extraction and validation-ready representation of source local structures.

A *source local structure* is represented by two complementary components:

1. a geometric support region, describing where high-quality source solutions
   concentrate; and
2. a local rank surrogate, describing how relative solution quality changes
   inside and near that region.

The local surrogate is trained on rank-normalized quality rather than raw source
function values.  This makes the representation invariant to affine response
scaling and weakens the transfer assumption from response-surface equality to
local ordering similarity.

The module contains no target-task logic.  It is intentionally limited to source
structure extraction so that extraction fidelity can be evaluated independently
from source-target similarity and optimization performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor

Array = np.ndarray
ModelType = Literal["gp", "random_forest"]
AggregationType = Literal["max", "weighted_sum"]


@dataclass(frozen=True)
class LocalStructureConfig:
    """Configuration for source local-structure extraction.

    Parameters are fixed before evaluation and should be stored with every run.
    The extractor uses only source training observations.  Test observations must
    be generated or loaded independently by the experiment runner.
    """

    elite_ratio: float = 0.20
    max_regions: int = 3
    min_elite_per_region: int = 4

    min_context_samples: int = 16
    max_context_samples: int = 96
    context_multiplier: float = 3.0
    min_boundary_fraction: float = 0.25

    covariance_regularization: float = 1e-3
    gmm_reg_covar: float = 1e-4
    use_bic_model_selection: bool = True

    model_type: ModelType = "gp"
    gp_restarts: int = 0
    random_forest_trees: int = 200
    cv_folds: int = 3
    validation_top_fraction: float = 0.10

    quality_floor: float = 0.10
    reliability_floor: float = 0.25
    random_state: int = 42

    def __post_init__(self) -> None:
        if not 0.0 < self.elite_ratio <= 1.0:
            raise ValueError("elite_ratio must lie in (0, 1].")
        if self.max_regions < 1:
            raise ValueError("max_regions must be at least 1.")
        if self.min_elite_per_region < 2:
            raise ValueError("min_elite_per_region must be at least 2.")
        if self.min_context_samples < 4:
            raise ValueError("min_context_samples must be at least 4.")
        if self.max_context_samples < self.min_context_samples:
            raise ValueError("max_context_samples must be >= min_context_samples.")
        if self.context_multiplier < 1.0:
            raise ValueError("context_multiplier must be at least 1.")
        if not 0.0 <= self.min_boundary_fraction < 1.0:
            raise ValueError("min_boundary_fraction must lie in [0, 1).")
        if self.covariance_regularization <= 0.0:
            raise ValueError("covariance_regularization must be positive.")
        if self.gmm_reg_covar <= 0.0:
            raise ValueError("gmm_reg_covar must be positive.")
        if self.model_type not in {"gp", "random_forest"}:
            raise ValueError("model_type must be 'gp' or 'random_forest'.")
        if self.gp_restarts < 0:
            raise ValueError("gp_restarts must be non-negative.")
        if self.random_forest_trees < 10:
            raise ValueError("random_forest_trees must be at least 10.")
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2.")
        if not 0.0 < self.validation_top_fraction < 0.5:
            raise ValueError("validation_top_fraction must lie in (0, 0.5).")
        if not 0.0 <= self.quality_floor <= 1.0:
            raise ValueError("quality_floor must lie in [0, 1].")
        if not 0.0 <= self.reliability_floor <= 1.0:
            raise ValueError("reliability_floor must lie in [0, 1].")


@dataclass(frozen=True)
class LocalStructureValidation:
    """Cross-validated diagnostics of the within-region rank surrogate."""

    oof_spearman: float
    oof_ndcg: float
    oof_precision_at_top: float
    geometry_spearman: float
    geometry_ndcg: float
    reliability: float
    n_context: int
    n_core: int
    boundary_fraction: float


@dataclass
class SourceLocalStructure:
    """One extracted source local structure.

    The model predicts rank-normalized local quality in canonical whitened
    coordinates.  Geometric membership gates the model and prevents unreliable
    extrapolation outside the extracted region.
    """

    task_id: str
    region_id: str
    center: Array
    covariance: Array
    precision: Array
    whitening: Array
    region_quality: float
    core_count: int
    context_count: int
    model: Any
    model_type: ModelType
    validation: LocalStructureValidation
    bic: float
    quality_floor: float = 0.10
    reliability_floor: float = 0.25
    context_indices: Array = field(repr=False, default_factory=lambda: np.empty(0, dtype=int))
    core_indices: Array = field(repr=False, default_factory=lambda: np.empty(0, dtype=int))

    @property
    def dim(self) -> int:
        return int(len(self.center))

    def transform(self, X: Array) -> Array:
        points = _as_points(X, self.dim, name="X")
        return (points - self.center[None, :]) @ self.whitening.T

    def mahalanobis_sq(self, X: Array) -> Array:
        points = _as_points(X, self.dim, name="X")
        diff = points - self.center[None, :]
        values = np.sum((diff @ self.precision) * diff, axis=1)
        return np.maximum(values, 0.0)

    def membership(self, X: Array) -> Array:
        return np.exp(-0.5 * self.mahalanobis_sq(X))

    def predict_relative_quality(
        self,
        X: Array,
        return_std: bool = False,
    ) -> Tuple[Array, Optional[Array]]:
        canonical = self.transform(X)
        if self.model_type == "gp":
            mean, std = self.model.predict(canonical, return_std=True)
        else:
            tree_predictions = np.vstack(
                [tree.predict(canonical) for tree in self.model.estimators_]
            )
            mean = np.mean(tree_predictions, axis=0)
            std = np.std(tree_predictions, axis=0, ddof=0)
        mean = np.clip(np.asarray(mean, dtype=float).reshape(-1), 0.0, 1.0)
        std = np.maximum(np.asarray(std, dtype=float).reshape(-1), 0.0)
        return (mean, std) if return_std else (mean, None)

    def geometry_score(self, X: Array) -> Array:
        return self.region_quality * self.membership(X)

    def structure_score(self, X: Array, use_reliability: bool = True) -> Array:
        membership = self.membership(X)
        quality, _ = self.predict_relative_quality(X, return_std=False)
        quality_factor = self.quality_floor + (1.0 - self.quality_floor) * quality
        reliability = (
            self.reliability_floor
            + (1.0 - self.reliability_floor) * self.validation.reliability
            if use_reliability
            else 1.0
        )
        return self.region_quality * membership * quality_factor * reliability

    def to_record(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "region_id": self.region_id,
            "center": self.center.tolist(),
            "covariance": self.covariance.tolist(),
            "region_quality": float(self.region_quality),
            "core_count": int(self.core_count),
            "context_count": int(self.context_count),
            "bic": float(self.bic),
            "model_type": self.model_type,
            "oof_spearman": self.validation.oof_spearman,
            "oof_ndcg": self.validation.oof_ndcg,
            "oof_precision_at_top": self.validation.oof_precision_at_top,
            "geometry_spearman": self.validation.geometry_spearman,
            "geometry_ndcg": self.validation.geometry_ndcg,
            "reliability": self.validation.reliability,
            "boundary_fraction": self.validation.boundary_fraction,
        }


@dataclass
class SourceLocalStructureLibrary:
    """Collection of extracted local structures."""

    structures: List[SourceLocalStructure] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.structures)

    @property
    def dim(self) -> Optional[int]:
        return self.structures[0].dim if self.structures else None

    def add(self, structure: SourceLocalStructure) -> None:
        if self.structures and structure.dim != self.structures[0].dim:
            raise ValueError("All structures must have the same dimension.")
        self.structures.append(structure)

    def geometry_components(self, X: Array) -> Array:
        if not self.structures:
            points = np.atleast_2d(np.asarray(X, dtype=float))
            return np.empty((len(points), 0), dtype=float)
        return np.column_stack([s.geometry_score(X) for s in self.structures])

    def structure_components(self, X: Array, use_reliability: bool = True) -> Array:
        if not self.structures:
            points = np.atleast_2d(np.asarray(X, dtype=float))
            return np.empty((len(points), 0), dtype=float)
        return np.column_stack(
            [s.structure_score(X, use_reliability=use_reliability) for s in self.structures]
        )

    def geometry_score(
        self,
        X: Array,
        aggregation: AggregationType = "max",
    ) -> Array:
        return _aggregate_components(
            self.geometry_components(X),
            self.structures,
            aggregation,
        )

    def score(
        self,
        X: Array,
        aggregation: AggregationType = "max",
        use_reliability: bool = True,
    ) -> Array:
        return _aggregate_components(
            self.structure_components(X, use_reliability=use_reliability),
            self.structures,
            aggregation,
        )

    def best_region_indices(self, X: Array, use_reliability: bool = True) -> Array:
        components = self.structure_components(X, use_reliability=use_reliability)
        if components.shape[1] == 0:
            return np.full(components.shape[0], -1, dtype=int)
        return np.argmax(components, axis=1)

    def records(self) -> List[Dict[str, Any]]:
        return [structure.to_record() for structure in self.structures]


class SourceLocalStructureExtractor:
    """Extract geometric regions and local rank surrogates from source data."""

    def __init__(self, config: Optional[LocalStructureConfig] = None) -> None:
        self.config = config or LocalStructureConfig()

    def fit_dataset(
        self,
        X: Array,
        y: Array,
        task_id: str = "source_0",
    ) -> SourceLocalStructureLibrary:
        points = _as_points(X, None, name="X")
        values = _as_values(y, len(points), name="y")
        n, dim = points.shape
        if n < max(8, self.config.min_context_samples):
            raise ValueError(
                "The source dataset is too small for local-structure extraction."
            )

        global_quality = _rank_quality(values)
        n_elite = max(
            self.config.min_elite_per_region,
            int(np.ceil(n * self.config.elite_ratio)),
        )
        n_elite = min(n_elite, n)
        elite_indices = np.argsort(values, kind="stable")[:n_elite]
        elite_points = points[elite_indices]

        mixture, elite_labels, selected_bic = self._fit_elite_mixture(
            elite_points,
            dim,
        )

        library = SourceLocalStructureLibrary()
        for component in range(mixture.n_components):
            local_mask = elite_labels == component
            core_indices = elite_indices[local_mask]
            if len(core_indices) < 2:
                continue

            center = np.asarray(mixture.means_[component], dtype=float)
            covariance = _regularize_covariance(
                np.asarray(mixture.covariances_[component], dtype=float),
                points,
                self.config.covariance_regularization,
            )
            precision = np.linalg.pinv(covariance)
            whitening = _whitening_matrix(covariance)

            context_indices = self._build_context(
                points,
                values,
                core_indices,
                center,
                precision,
                elite_indices,
            )
            context_X = points[context_indices]
            context_y = values[context_indices]
            local_quality = _rank_quality(context_y)
            canonical_X = (context_X - center[None, :]) @ whitening.T

            validation = self._cross_validate_local_model(
                canonical_X,
                local_quality,
                context_X,
                center,
                precision,
                core_indices=np.intersect1d(context_indices, core_indices),
                random_state=self.config.random_state + component * 997,
            )
            model, actual_model_type = self._fit_model(
                canonical_X,
                local_quality,
                random_state=self.config.random_state + component * 997,
                optimize=True,
            )

            structure = SourceLocalStructure(
                task_id=str(task_id),
                region_id=f"{task_id}::region_{component}",
                center=center,
                covariance=covariance,
                precision=precision,
                whitening=whitening,
                region_quality=float(np.mean(global_quality[core_indices])),
                core_count=int(len(core_indices)),
                context_count=int(len(context_indices)),
                model=model,
                model_type=actual_model_type,
                validation=validation,
                bic=float(selected_bic),
                quality_floor=self.config.quality_floor,
                reliability_floor=self.config.reliability_floor,
                context_indices=context_indices.copy(),
                core_indices=core_indices.copy(),
            )
            library.add(structure)

        if not library.structures:
            raise RuntimeError("No valid local structure could be extracted.")
        return library

    def fit_multi_source(
        self,
        source_datasets: Sequence[Tuple[Array, Array]],
        task_ids: Optional[Sequence[str]] = None,
    ) -> SourceLocalStructureLibrary:
        datasets = list(source_datasets)
        if not datasets:
            raise ValueError("source_datasets must not be empty.")
        if task_ids is not None and len(task_ids) != len(datasets):
            raise ValueError("task_ids must match source_datasets in length.")

        combined = SourceLocalStructureLibrary()
        for index, (X, y) in enumerate(datasets):
            task_id = (
                str(task_ids[index])
                if task_ids is not None
                else f"source_{index}"
            )
            library = self.fit_dataset(X, y, task_id=task_id)
            for structure in library.structures:
                combined.add(structure)
        return combined

    def _fit_elite_mixture(
        self,
        elite_points: Array,
        dim: int,
    ) -> Tuple[GaussianMixture, Array, float]:
        max_components = min(
            self.config.max_regions,
            max(1, len(elite_points) // self.config.min_elite_per_region),
        )
        candidates: List[Tuple[float, GaussianMixture, Array]] = []

        component_values: Iterable[int]
        if self.config.use_bic_model_selection:
            component_values = range(1, max_components + 1)
        else:
            component_values = [max_components]

        for n_components in component_values:
            try:
                mixture = GaussianMixture(
                    n_components=n_components,
                    covariance_type="full",
                    reg_covar=self.config.gmm_reg_covar,
                    n_init=5,
                    random_state=self.config.random_state,
                )
                labels = mixture.fit_predict(elite_points)
                counts = np.bincount(labels, minlength=n_components)
                if np.min(counts) < self.config.min_elite_per_region:
                    continue
                bic = float(mixture.bic(elite_points))
                candidates.append((bic, mixture, labels))
            except Exception:
                continue

        if candidates:
            bic, mixture, labels = min(candidates, key=lambda item: item[0])
            return mixture, labels, bic

        covariance = _regularize_covariance(
            np.cov(elite_points.T) if len(elite_points) > 1 else np.eye(dim),
            elite_points,
            self.config.covariance_regularization,
        )
        fallback = GaussianMixture(
            n_components=1,
            covariance_type="full",
            reg_covar=self.config.gmm_reg_covar,
            n_init=1,
            random_state=self.config.random_state,
        )
        fallback.fit(elite_points)
        fallback.means_[0] = np.mean(elite_points, axis=0)
        fallback.covariances_[0] = covariance
        labels = np.zeros(len(elite_points), dtype=int)
        return fallback, labels, float(fallback.bic(elite_points))

    def _build_context(
        self,
        X: Array,
        y: Array,
        core_indices: Array,
        center: Array,
        precision: Array,
        elite_indices: Array,
    ) -> Array:
        diff = X - center[None, :]
        distances = np.sum((diff @ precision) * diff, axis=1)
        distances = np.maximum(distances, 0.0)

        requested = max(
            self.config.min_context_samples,
            int(np.ceil(len(core_indices) * self.config.context_multiplier)),
        )
        requested = min(requested, self.config.max_context_samples, len(X))
        nearest = np.argsort(distances, kind="stable")[:requested]
        context = set(int(i) for i in nearest)
        context.update(int(i) for i in core_indices)

        elite_set = set(int(i) for i in elite_indices)
        minimum_boundary = int(
            np.ceil(len(context) * self.config.min_boundary_fraction)
        )
        current_boundary = sum(index not in elite_set for index in context)
        if current_boundary < minimum_boundary:
            non_elite_order = [
                int(index)
                for index in np.argsort(distances, kind="stable")
                if int(index) not in elite_set
            ]
            for index in non_elite_order:
                context.add(index)
                current_boundary += 1
                if current_boundary >= minimum_boundary:
                    break

        ordered = sorted(context, key=lambda index: (distances[index], index))
        if len(ordered) > self.config.max_context_samples:
            mandatory = set(int(i) for i in core_indices)
            retained = list(sorted(mandatory))
            for index in ordered:
                if index not in mandatory:
                    retained.append(index)
                if len(retained) >= self.config.max_context_samples:
                    break
            ordered = retained[: self.config.max_context_samples]

        return np.asarray(sorted(set(ordered)), dtype=int)

    def _fit_model(
        self,
        X: Array,
        y: Array,
        random_state: int,
        optimize: bool,
    ) -> Tuple[Any, ModelType]:
        if self.config.model_type == "random_forest":
            model = RandomForestRegressor(
                n_estimators=self.config.random_forest_trees,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=random_state,
                n_jobs=1,
            )
            model.fit(X, y)
            return model, "random_forest"

        dim = X.shape[1]
        kernel = (
            ConstantKernel(1.0, (1e-2, 1e2))
            * Matern(
                length_scale=np.ones(dim),
                length_scale_bounds=(0.15, 10.0),
                nu=2.5,
            )
            + WhiteKernel(
                noise_level=1e-3,
                noise_level_bounds=(1e-6, 2e-1),
            )
        )
        model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=(self.config.gp_restarts if optimize else 0),
            random_state=random_state,
        )
        try:
            model.fit(X, y)
            return model, "gp"
        except Exception:
            fallback = RandomForestRegressor(
                n_estimators=max(50, self.config.random_forest_trees // 2),
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=random_state,
                n_jobs=1,
            )
            fallback.fit(X, y)
            return fallback, "random_forest"

    def _cross_validate_local_model(
        self,
        canonical_X: Array,
        quality: Array,
        original_X: Array,
        center: Array,
        precision: Array,
        core_indices: Array,
        random_state: int,
    ) -> LocalStructureValidation:
        n = len(canonical_X)
        folds = min(self.config.cv_folds, max(2, n // 4))
        folds = min(folds, n)
        predictions = np.full(n, np.nan, dtype=float)
        splitter = KFold(
            n_splits=folds,
            shuffle=True,
            random_state=random_state,
        )

        for fold_index, (train_index, test_index) in enumerate(splitter.split(canonical_X)):
            model, model_type = self._fit_model(
                canonical_X[train_index],
                quality[train_index],
                random_state=random_state + fold_index + 1,
                optimize=False,
            )
            if model_type == "gp":
                fold_prediction = model.predict(canonical_X[test_index])
            else:
                fold_prediction = model.predict(canonical_X[test_index])
            predictions[test_index] = np.asarray(fold_prediction, dtype=float)

        predictions = np.clip(np.nan_to_num(predictions, nan=np.mean(quality)), 0.0, 1.0)
        diff = original_X - center[None, :]
        geometry = np.exp(
            -0.5 * np.maximum(np.sum((diff @ precision) * diff, axis=1), 0.0)
        )

        oof_spearman = _safe_spearman(predictions, quality)
        geometry_spearman = _safe_spearman(geometry, quality)
        oof_ndcg = _ndcg_at_fraction(
            quality,
            predictions,
            self.config.validation_top_fraction,
        )
        geometry_ndcg = _ndcg_at_fraction(
            quality,
            geometry,
            self.config.validation_top_fraction,
        )
        precision = _precision_at_fraction(
            quality,
            predictions,
            self.config.validation_top_fraction,
        )
        reliability = float(
            np.clip(
                max(0.0, oof_spearman)
                * max(0.0, oof_ndcg),
                0.0,
                1.0,
            )
        )

        core_count = int(len(core_indices))
        boundary_fraction = float(max(0.0, 1.0 - core_count / max(n, 1)))
        return LocalStructureValidation(
            oof_spearman=float(oof_spearman),
            oof_ndcg=float(oof_ndcg),
            oof_precision_at_top=float(precision),
            geometry_spearman=float(geometry_spearman),
            geometry_ndcg=float(geometry_ndcg),
            reliability=reliability,
            n_context=int(n),
            n_core=core_count,
            boundary_fraction=boundary_fraction,
        )


def _aggregate_components(
    components: Array,
    structures: Sequence[SourceLocalStructure],
    aggregation: AggregationType,
) -> Array:
    if components.ndim != 2:
        raise ValueError("components must be two-dimensional.")
    if components.shape[1] == 0:
        return np.zeros(components.shape[0], dtype=float)
    if aggregation == "max":
        return np.max(components, axis=1)
    if aggregation == "weighted_sum":
        weights = np.asarray(
            [max(1, structure.core_count) for structure in structures],
            dtype=float,
        )
        weights /= np.sum(weights)
        return components @ weights
    raise ValueError("aggregation must be 'max' or 'weighted_sum'.")


def _as_points(X: Array, dim: Optional[int], name: str) -> Array:
    points = np.asarray(X, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array.")
    if dim is not None and points.shape[1] != dim:
        raise ValueError(f"{name} must have {dim} columns.")
    if len(points) == 0 or not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must contain finite observations.")
    return points.copy()


def _as_values(y: Array, n: int, name: str) -> Array:
    values = np.asarray(y, dtype=float).reshape(-1)
    if len(values) != n:
        raise ValueError(f"{name} must contain exactly {n} values.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain finite values.")
    return values.copy()


def _rank_quality(y: Array) -> Array:
    values = np.asarray(y, dtype=float).reshape(-1)
    if len(values) <= 1:
        return np.ones_like(values)
    ranks = rankdata(values, method="average")
    return 1.0 - (ranks - 1.0) / (len(values) - 1.0)


def _regularize_covariance(
    covariance: Array,
    reference_X: Array,
    regularization: float,
) -> Array:
    cov = np.asarray(covariance, dtype=float)
    dim = reference_X.shape[1]
    if cov.ndim == 0:
        cov = np.eye(dim) * float(cov)
    if cov.shape != (dim, dim) or not np.all(np.isfinite(cov)):
        cov = np.cov(reference_X.T)
    cov = np.atleast_2d(cov)
    cov = 0.5 * (cov + cov.T)

    global_scale = float(np.mean(np.var(reference_X, axis=0)))
    global_scale = max(global_scale, 1e-6)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    floor = regularization * global_scale
    eigenvalues = np.maximum(eigenvalues, floor)
    regularized = (eigenvectors * eigenvalues) @ eigenvectors.T
    regularized += floor * 1e-3 * np.eye(dim)
    return 0.5 * (regularized + regularized.T)


def _whitening_matrix(covariance: Array) -> Array:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 1e-12)
    return (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T


def _safe_spearman(a: Array, b: Array) -> float:
    first = np.asarray(a, dtype=float).reshape(-1)
    second = np.asarray(b, dtype=float).reshape(-1)
    if len(first) < 3 or np.std(first) < 1e-12 or np.std(second) < 1e-12:
        return 0.0
    estimate = spearmanr(first, second).statistic
    return float(estimate) if np.isfinite(estimate) else 0.0


def _ndcg_at_fraction(true_quality: Array, scores: Array, fraction: float) -> float:
    relevance = np.asarray(true_quality, dtype=float).reshape(-1)
    prediction = np.asarray(scores, dtype=float).reshape(-1)
    if len(relevance) == 0:
        return float("nan")
    k = max(1, int(np.ceil(len(relevance) * fraction)))
    order = np.argsort(-prediction, kind="stable")[:k]
    ideal = np.argsort(-relevance, kind="stable")[:k]
    discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=float))
    gains = np.power(2.0, relevance) - 1.0
    dcg = float(np.sum(gains[order] * discounts))
    ideal_dcg = float(np.sum(gains[ideal] * discounts))
    return dcg / ideal_dcg if ideal_dcg > 1e-12 else 0.0


def _precision_at_fraction(true_quality: Array, scores: Array, fraction: float) -> float:
    relevance = np.asarray(true_quality, dtype=float).reshape(-1)
    prediction = np.asarray(scores, dtype=float).reshape(-1)
    if len(relevance) == 0:
        return float("nan")
    k = max(1, int(np.ceil(len(relevance) * fraction)))
    truth = set(np.argsort(-relevance, kind="stable")[:k].tolist())
    selected = set(np.argsort(-prediction, kind="stable")[:k].tolist())
    return float(len(truth.intersection(selected)) / k)


__all__ = [
    "LocalStructureConfig",
    "LocalStructureValidation",
    "SourceLocalStructure",
    "SourceLocalStructureExtractor",
    "SourceLocalStructureLibrary",
]
