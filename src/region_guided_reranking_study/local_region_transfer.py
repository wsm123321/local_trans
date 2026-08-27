"""Local-region transfer Bayesian optimization.

This module implements the complete algorithmic pipeline proposed by the project:

1. Extract high-quality local regions from one or more historical source tasks.
2. Fit a surrogate model using target-task observations only.
3. Generate a broad target-driven candidate pool.
4. Let the source-region library nominate and rerank promising candidates.
5. Evaluate the selected target candidate(s), update the target data, and repeat.

The source knowledge never participates in fitting the target surrogate or generating
its acquisition values. It is used only after target-driven candidate generation,
which keeps candidate generation and transferred knowledge explicitly decoupled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .rerankers import normalize_scores
from .source_regions import SourceRegionExtractor, SourceRegionLibrary
from .surrogate_and_candidates import CandidatePoolGenerator, TargetGPSurrogate

Array = np.ndarray
SourceDataset = Tuple[Array, Array]
Objective = Callable[[Array], Array]


@dataclass(frozen=True)
class LocalRegionTransferConfig:
    r"""Configuration of :class:`LocalRegionTransferOptimizer`.

    Parameters
    ----------
    top_ratio:
        Fraction of the best observations retained from every source task before
        clustering in decision space.
    max_clusters:
        Maximum number of high-quality local regions extracted from each source.
    min_samples_per_cluster:
        Minimum number of elite samples used to form a source region.
    merge_threshold:
        Euclidean center-distance threshold used by the existing source-region
        extractor to merge nearby regions from different source tasks.
    pool_size:
        Number of target-driven candidates generated at each iteration.
    acquisition:
        Target acquisition function used for final candidate scoring. Supported
        values are ``"ei"``, ``"lcb"``, and ``"pi"``.
    source_weight:
        Base weight :math:`\lambda_0` of the source-region support score.
    source_weight_decay:
        Optional time decay. At target iteration ``t``,
        :math:`\lambda_t = \lambda_0 / (1 + \gamma t)`.
    target_nomination_ratio:
        Fraction of candidates nominated by target acquisition. These candidates
        are always retained, so source information is never a hard filter.
    source_nomination_ratio:
        Fraction of candidates nominated by source-region support.
    shortlist_size:
        Optional maximum number of candidates retained after nomination. ``None``
        keeps the complete union of target and source nominees.
    source_aggregation:
        Region-library aggregation passed to ``SourceRegionLibrary.score``.
    min_source_variation:
        A degenerate-score guard. If source support is essentially constant over
        the candidate pool, transfer is disabled for that iteration.
    min_batch_distance:
        Minimum normalized Euclidean distance between points selected in the same
        batch. It has no effect when one point is requested.
    random_state:
        Master seed used to create reproducible, isolated random streams.
    """

    top_ratio: float = 0.20
    max_clusters: int = 3
    min_samples_per_cluster: int = 3
    merge_threshold: float = 0.5

    pool_size: int = 1000
    acquisition: str = "ei"
    source_weight: float = 1.0
    source_weight_decay: float = 0.0

    target_nomination_ratio: float = 0.20
    source_nomination_ratio: float = 0.20
    shortlist_size: Optional[int] = None

    source_aggregation: str = "max"
    min_source_variation: float = 1e-8
    min_batch_distance: float = 0.02

    ratio_acq: float = 0.40
    ratio_global: float = 0.40
    ratio_diverse: float = 0.20
    gp_noise_level: float = 1e-4
    random_state: int = 42

    def __post_init__(self) -> None:
        if not 0.0 < self.top_ratio <= 1.0:
            raise ValueError("top_ratio must lie in (0, 1].")
        if self.max_clusters < 1:
            raise ValueError("max_clusters must be at least 1.")
        if self.min_samples_per_cluster < 2:
            raise ValueError("min_samples_per_cluster must be at least 2.")
        if self.merge_threshold < 0.0:
            raise ValueError("merge_threshold must be non-negative.")
        if self.pool_size < 10:
            raise ValueError("pool_size must be at least 10.")
        if self.acquisition.lower() not in {"ei", "lcb", "pi"}:
            raise ValueError("acquisition must be one of: 'ei', 'lcb', or 'pi'.")
        if self.source_weight < 0.0:
            raise ValueError("source_weight must be non-negative.")
        if self.source_weight_decay < 0.0:
            raise ValueError("source_weight_decay must be non-negative.")
        if not 0.0 < self.target_nomination_ratio <= 1.0:
            raise ValueError("target_nomination_ratio must lie in (0, 1].")
        if not 0.0 < self.source_nomination_ratio <= 1.0:
            raise ValueError("source_nomination_ratio must lie in (0, 1].")
        if self.shortlist_size is not None and self.shortlist_size < 1:
            raise ValueError("shortlist_size must be positive when provided.")
        if self.source_aggregation not in {"max", "weighted_sum"}:
            raise ValueError("source_aggregation must be 'max' or 'weighted_sum'.")
        if self.min_source_variation < 0.0:
            raise ValueError("min_source_variation must be non-negative.")
        if self.min_batch_distance < 0.0:
            raise ValueError("min_batch_distance must be non-negative.")
        ratios = (self.ratio_acq, self.ratio_global, self.ratio_diverse)
        if any(r < 0.0 for r in ratios) or not np.isclose(sum(ratios), 1.0):
            raise ValueError("ratio_acq + ratio_global + ratio_diverse must equal 1.")
        if self.gp_noise_level <= 0.0:
            raise ValueError("gp_noise_level must be positive.")


@dataclass
class CandidateDecision:
    """Complete diagnostics for one ask operation."""

    points: Array
    selected_indices: Array
    candidates: Array
    acquisition_scores: Array
    source_scores: Array
    normalized_acquisition: Array
    normalized_source: Array
    combined_scores: Array
    shortlist_mask: Array
    effective_source_weight: float
    iteration: int

    @property
    def shortlist_indices(self) -> Array:
        return np.flatnonzero(self.shortlist_mask)


@dataclass
class LocalRegionTransferResult:
    """Result returned by :meth:`LocalRegionTransferOptimizer.optimize`."""

    X: Array
    y: Array
    best_x: Array
    best_y: float
    best_y_trace: List[float]
    decisions: List[CandidateDecision] = field(default_factory=list)


class LocalRegionTransferOptimizer:
    r"""Source-local-region-guided optimizer with an ask/tell interface.

    Notes
    -----
    The algorithm has two deliberately separated information paths:

    * **Target path:** target observations -> target GP -> acquisition -> candidates.
    * **Transfer path:** source observations -> elite clustering -> region support.

    The two paths meet only during candidate nomination and soft reranking:

    .. math::

        J_t(x) = \widetilde{\alpha}_t(x)
                 + \lambda_t\widetilde{r}_s(x).

    Target acquisition nominees are always included in the shortlist. Therefore,
    the source regions guide candidate selection without becoming a hard search
    constraint.
    """

    def __init__(
        self,
        bounds: Array,
        config: Optional[LocalRegionTransferConfig] = None,
    ) -> None:
        self.bounds = self._validate_bounds(bounds)
        self.dim = int(self.bounds.shape[0])
        self.config = config or LocalRegionTransferConfig()

        master_seed = np.random.SeedSequence(self.config.random_state)
        candidate_seed = master_seed.spawn(1)[0]
        self._candidate_rng = np.random.default_rng(candidate_seed)

        self.source_regions = SourceRegionLibrary()
        self._source_X: List[Array] = []

        self.target_X: Optional[Array] = None
        self.target_y: Optional[Array] = None
        self.iteration = 0
        self.last_decision: Optional[CandidateDecision] = None

    @staticmethod
    def _validate_bounds(bounds: Array) -> Array:
        arr = np.asarray(bounds, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("bounds must have shape (dimension, 2).")
        if not np.all(np.isfinite(arr)):
            raise ValueError("bounds must contain only finite values.")
        if np.any(arr[:, 0] >= arr[:, 1]):
            raise ValueError("Every lower bound must be smaller than its upper bound.")
        return arr.copy()

    def _validate_points(self, X: Array, *, name: str) -> Array:
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] != self.dim:
            raise ValueError(f"{name} must have shape (n, {self.dim}).")
        if len(arr) == 0:
            raise ValueError(f"{name} must contain at least one point.")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must contain only finite values.")
        if np.any(arr < self.bounds[:, 0]) or np.any(arr > self.bounds[:, 1]):
            raise ValueError(f"{name} contains points outside bounds.")
        return arr.copy()

    @staticmethod
    def _validate_values(y: Array, n: int, *, name: str) -> Array:
        arr = np.asarray(y, dtype=float).reshape(-1)
        if len(arr) != n:
            raise ValueError(f"{name} must contain exactly {n} values.")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must contain only finite values.")
        return arr.copy()

    def fit_source_regions(
        self,
        source_datasets: Sequence[SourceDataset],
        task_ids: Optional[Sequence[str]] = None,
    ) -> SourceRegionLibrary:
        """Extract and store high-quality local regions from source observations.

        Every source dataset is a pair ``(X_s, y_s)`` for a minimization problem.
        The best ``top_ratio`` observations are clustered in decision space and
        represented by Gaussian local regions.
        """

        datasets = list(source_datasets)
        if not datasets:
            raise ValueError("source_datasets must contain at least one dataset.")
        if task_ids is not None and len(task_ids) != len(datasets):
            raise ValueError("task_ids must have the same length as source_datasets.")

        validated: List[SourceDataset] = []
        self._source_X = []
        for idx, (X, y) in enumerate(datasets):
            X_arr = self._validate_points(X, name=f"source_datasets[{idx}].X")
            y_arr = self._validate_values(y, len(X_arr), name=f"source_datasets[{idx}].y")
            validated.append((X_arr, y_arr))
            self._source_X.append(X_arr)

        extractor = SourceRegionExtractor(
            top_ratio=self.config.top_ratio,
            max_clusters=self.config.max_clusters,
            min_samples_per_cluster=self.config.min_samples_per_cluster,
            random_state=self.config.random_state,
        )
        self.source_regions = extractor.extract_from_multi_sources(
            validated,
            task_ids=list(task_ids) if task_ids is not None else None,
            merge_threshold=self.config.merge_threshold,
        )
        return self.source_regions

    def set_source_region_library(self, library: SourceRegionLibrary) -> None:
        """Inject an already constructed source-region library."""

        for idx, region in enumerate(library.regions):
            if len(region.center) != self.dim:
                raise ValueError(
                    f"Source region {idx} has dimension {len(region.center)}; "
                    f"expected {self.dim}."
                )
        self.source_regions = library

    def initialize_target(self, X: Array, y: Array) -> None:
        """Set the initial target-task observations."""

        X_arr = self._validate_points(X, name="target X")
        y_arr = self._validate_values(y, len(X_arr), name="target y")
        if len(X_arr) < 2:
            raise ValueError("At least two target observations are required.")
        self.target_X = X_arr
        self.target_y = y_arr
        self.iteration = 0
        self.last_decision = None

    @property
    def is_initialized(self) -> bool:
        return self.target_X is not None and self.target_y is not None

    def _effective_source_weight(self, source_scores: Array) -> float:
        if not self.source_regions.regions:
            return 0.0
        if np.ptp(source_scores) < self.config.min_source_variation:
            return 0.0
        if np.var(source_scores) < self.config.min_source_variation:
            return 0.0
        return float(
            self.config.source_weight
            / (1.0 + self.config.source_weight_decay * self.iteration)
        )

    @staticmethod
    def _top_indices(scores: Array, count: int) -> Array:
        count = min(max(0, int(count)), len(scores))
        return np.argsort(-scores, kind="stable")[:count]

    def _build_shortlist(
        self,
        normalized_acquisition: Array,
        normalized_source: Array,
        combined_scores: Array,
        effective_source_weight: float,
        n_points: int,
    ) -> Array:
        total = len(combined_scores)
        target_count = max(
            n_points,
            int(np.ceil(total * self.config.target_nomination_ratio)),
        )
        source_count = max(
            n_points,
            int(np.ceil(total * self.config.source_nomination_ratio)),
        )

        target_nominees = self._top_indices(normalized_acquisition, target_count)
        shortlist = set(int(i) for i in target_nominees)

        source_nominees = np.array([], dtype=int)
        if effective_source_weight > 0.0:
            source_nominees = self._top_indices(normalized_source, source_count)
            shortlist.update(int(i) for i in source_nominees)

        minimum_required = min(n_points, total)
        if len(shortlist) < minimum_required:
            shortlist.update(
                int(i)
                for i in self._top_indices(combined_scores, minimum_required)
            )

        if self.config.shortlist_size is not None and len(shortlist) > self.config.shortlist_size:
            max_size = max(self.config.shortlist_size, minimum_required)
            mandatory = set(int(i) for i in target_nominees[:minimum_required])
            if effective_source_weight > 0.0:
                mandatory.update(int(i) for i in source_nominees[:minimum_required])

            ordered_union = sorted(
                shortlist,
                key=lambda idx: (-combined_scores[idx], idx),
            )
            limited = list(sorted(mandatory))
            for idx in ordered_union:
                if idx not in mandatory:
                    limited.append(idx)
                if len(limited) >= max_size:
                    break
            shortlist = set(limited[:max_size])

        mask = np.zeros(total, dtype=bool)
        if shortlist:
            mask[np.fromiter(sorted(shortlist), dtype=int)] = True
        return mask

    def _select_batch(
        self,
        candidates: Array,
        combined_scores: Array,
        shortlist_mask: Array,
        n_points: int,
    ) -> Array:
        shortlisted = np.flatnonzero(shortlist_mask)
        if len(shortlisted) < n_points:
            missing = n_points - len(shortlisted)
            remaining = np.flatnonzero(~shortlist_mask)
            extra = remaining[np.argsort(-combined_scores[remaining], kind="stable")[:missing]]
            shortlisted = np.concatenate([shortlisted, extra])

        ordered = shortlisted[
            np.argsort(-combined_scores[shortlisted], kind="stable")
        ]
        if n_points == 1:
            return ordered[:1]

        scale = np.maximum(self.bounds[:, 1] - self.bounds[:, 0], 1e-12)
        selected: List[int] = []
        for idx in ordered:
            if not selected:
                selected.append(int(idx))
            else:
                normalized_diff = (
                    candidates[np.asarray(selected)] - candidates[idx]
                ) / scale
                distances = np.linalg.norm(normalized_diff, axis=1)
                if np.min(distances) >= self.config.min_batch_distance:
                    selected.append(int(idx))
            if len(selected) == n_points:
                break

        if len(selected) < n_points:
            for idx in ordered:
                if int(idx) not in selected:
                    selected.append(int(idx))
                if len(selected) == n_points:
                    break

        return np.asarray(selected, dtype=int)

    def rank_candidate_pool(
        self,
        candidates: Array,
        acquisition_scores: Array,
        n_points: int = 1,
    ) -> CandidateDecision:
        """Nominate, rerank, and select candidates from a supplied target pool.

        This method exposes the central transfer mechanism independently of GP
        fitting, which is useful for diagnostics, ablations, and unit tests.
        """

        candidate_arr = self._validate_points(candidates, name="candidates")
        acquisition_arr = self._validate_values(
            acquisition_scores,
            len(candidate_arr),
            name="acquisition_scores",
        )
        if n_points < 1 or n_points > len(candidate_arr):
            raise ValueError("n_points must lie between 1 and the candidate count.")

        source_scores = self.source_regions.score(
            candidate_arr,
            aggregation=self.config.source_aggregation,
        )
        normalized_acquisition = normalize_scores(acquisition_arr, method="rank")
        effective_source_weight = self._effective_source_weight(source_scores)

        if effective_source_weight > 0.0:
            normalized_source = normalize_scores(source_scores, method="rank")
        else:
            normalized_source = np.zeros_like(source_scores)

        combined_scores = (
            normalized_acquisition
            + effective_source_weight * normalized_source
        )
        shortlist_mask = self._build_shortlist(
            normalized_acquisition=normalized_acquisition,
            normalized_source=normalized_source,
            combined_scores=combined_scores,
            effective_source_weight=effective_source_weight,
            n_points=n_points,
        )
        selected_indices = self._select_batch(
            candidates=candidate_arr,
            combined_scores=combined_scores,
            shortlist_mask=shortlist_mask,
            n_points=n_points,
        )

        return CandidateDecision(
            points=candidate_arr[selected_indices].copy(),
            selected_indices=selected_indices,
            candidates=candidate_arr,
            acquisition_scores=acquisition_arr,
            source_scores=source_scores,
            normalized_acquisition=normalized_acquisition,
            normalized_source=normalized_source,
            combined_scores=combined_scores,
            shortlist_mask=shortlist_mask,
            effective_source_weight=effective_source_weight,
            iteration=self.iteration,
        )

    def _sanitize_candidate_pool(self, candidates: Array) -> Array:
        """Remove accidental overlap with evaluated/source points and pool duplicates."""

        candidate_arr = np.asarray(candidates, dtype=float).copy()
        excluded_parts: List[Array] = []
        if self.target_X is not None and len(self.target_X) > 0:
            excluded_parts.append(self.target_X)
        excluded_parts.extend(X for X in self._source_X if len(X) > 0)
        excluded = (
            np.vstack(excluded_parts)
            if excluded_parts
            else np.empty((0, self.dim), dtype=float)
        )

        accepted: List[Array] = []
        tolerance = 1e-8
        for original in candidate_arr:
            point = np.asarray(original, dtype=float).copy()
            for _ in range(1000):
                far_from_excluded = (
                    len(excluded) == 0
                    or np.min(np.linalg.norm(excluded - point, axis=1)) > tolerance
                )
                far_from_pool = (
                    not accepted
                    or np.min(
                        np.linalg.norm(np.asarray(accepted) - point, axis=1)
                    ) > tolerance
                )
                if far_from_excluded and far_from_pool:
                    break
                point = self._candidate_rng.uniform(
                    self.bounds[:, 0],
                    self.bounds[:, 1],
                )
            else:
                raise RuntimeError(
                    "Unable to construct a non-duplicated candidate pool."
                )
            accepted.append(point)

        repaired = np.asarray(accepted, dtype=float)
        if repaired.shape != candidate_arr.shape:
            raise RuntimeError("Candidate-pool repair changed the pool shape.")
        return repaired

    def ask(self, n_points: int = 1) -> CandidateDecision:
        """Generate target candidates and return source-region-guided selections."""

        if not self.is_initialized:
            raise RuntimeError("Call initialize_target before ask.")
        assert self.target_X is not None
        assert self.target_y is not None

        surrogate = TargetGPSurrogate(
            dim=self.dim,
            noise_level=self.config.gp_noise_level,
            random_state=self.config.random_state + self.iteration,
        )
        surrogate.fit(self.target_X, self.target_y)

        generator = CandidatePoolGenerator(
            bounds=self.bounds,
            pool_size=self.config.pool_size,
            ratio_acq=self.config.ratio_acq,
            ratio_global=self.config.ratio_global,
            ratio_diverse=self.config.ratio_diverse,
            rng=self._candidate_rng,
        )
        candidates = generator.generate(
            surrogate=surrogate,
            current_X=self.target_X,
            excluded_datasets=self._source_X,
        )
        candidates = self._sanitize_candidate_pool(candidates)
        acquisition_scores = surrogate.compute_acquisition(
            candidates,
            acq_type=self.config.acquisition,
        )
        decision = self.rank_candidate_pool(
            candidates,
            acquisition_scores,
            n_points=n_points,
        )
        self.last_decision = decision
        return decision

    def tell(self, X: Array, y: Array) -> None:
        """Append newly evaluated target observations."""

        if not self.is_initialized:
            raise RuntimeError("Call initialize_target before tell.")
        assert self.target_X is not None
        assert self.target_y is not None

        X_arr = self._validate_points(X, name="new target X")
        y_arr = self._validate_values(y, len(X_arr), name="new target y")

        pairwise = np.linalg.norm(
            self.target_X[:, np.newaxis, :] - X_arr[np.newaxis, :, :],
            axis=2,
        )
        if np.any(pairwise < 1e-12):
            raise ValueError("tell received a target point that was already evaluated.")

        if len(X_arr) > 1:
            within_batch = np.linalg.norm(
                X_arr[:, np.newaxis, :] - X_arr[np.newaxis, :, :],
                axis=2,
            )
            upper = within_batch[np.triu_indices(len(X_arr), k=1)]
            if np.any(upper < 1e-12):
                raise ValueError("tell received duplicate points within the new batch.")

        self.target_X = np.vstack([self.target_X, X_arr])
        self.target_y = np.concatenate([self.target_y, y_arr])
        self.iteration += 1

    def get_best(self) -> Tuple[Array, float]:
        """Return the best observed target solution."""

        if not self.is_initialized:
            raise RuntimeError("The target task has not been initialized.")
        assert self.target_X is not None
        assert self.target_y is not None
        idx = int(np.argmin(self.target_y))
        return self.target_X[idx].copy(), float(self.target_y[idx])

    @staticmethod
    def _evaluate_objective(objective: Objective, X: Array) -> Array:
        values = np.asarray(objective(X), dtype=float)
        if values.ndim == 0:
            values = values.reshape(1)
        values = values.reshape(-1)
        if len(values) != len(X):
            raise ValueError(
                "objective must return one scalar value for each input point."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("objective returned non-finite values.")
        return values

    def optimize(
        self,
        objective: Objective,
        init_X: Array,
        init_y: Optional[Array] = None,
        budget: int = 20,
        batch_size: int = 1,
    ) -> LocalRegionTransferResult:
        """Run the complete closed-loop local-region transfer algorithm.

        ``budget`` counts new target evaluations, excluding the initial design.
        """

        if budget < 0:
            raise ValueError("budget must be non-negative.")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        init_points = self._validate_points(init_X, name="init_X")
        if init_y is None:
            init_values = self._evaluate_objective(objective, init_points)
        else:
            init_values = self._validate_values(init_y, len(init_points), name="init_y")
        self.initialize_target(init_points, init_values)

        current_best = float(np.min(init_values))
        best_y_trace: List[float] = [current_best]
        decisions: List[CandidateDecision] = []

        remaining = int(budget)
        while remaining > 0:
            n_points = min(batch_size, remaining)
            decision = self.ask(n_points=n_points)
            values = self._evaluate_objective(objective, decision.points)
            self.tell(decision.points, values)
            decisions.append(decision)

            for value in values:
                current_best = min(current_best, float(value))
                best_y_trace.append(current_best)
            remaining -= n_points

        assert self.target_X is not None
        assert self.target_y is not None
        best_x, best_y = self.get_best()
        return LocalRegionTransferResult(
            X=self.target_X.copy(),
            y=self.target_y.copy(),
            best_x=best_x,
            best_y=best_y,
            best_y_trace=best_y_trace,
            decisions=decisions,
        )
