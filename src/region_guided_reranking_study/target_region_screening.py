"""Target-surrogate proposal followed by source-region candidate screening.

This module implements the next-stage algorithm used by the project:

1. Fit a surrogate using target-task observations only.
2. Let the target surrogate independently propose a finite candidate set.
3. Score those target proposals with source-domain high-quality local regions.
4. Screen the target proposals; the source knowledge never creates new candidates.
5. Select the best target acquisition value among the retained proposals.
6. Evaluate on the target task, update the target data, and repeat.

The implementation exposes fixed and target-evidence-adaptive screening policies so
that the algorithmic mechanism and its negative-transfer boundary can be studied
under an equal-budget experimental design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Literal, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import rankdata, spearmanr

from .source_regions import SourceRegionExtractor, SourceRegionLibrary
from .surrogate_and_candidates import CandidatePoolGenerator, TargetGPSurrogate

Array = np.ndarray
Objective = Callable[[Array], Array]
SourceDataset = Tuple[Array, Array]
ScreeningPolicy = Literal["none", "fixed", "adaptive"]
ScreeningGeometry = Literal["quantile", "ellipsoid", "hybrid"]


@dataclass(frozen=True)
class TargetProposalConfig:
    """Configuration of the target-only proposal stage."""

    raw_pool_size: int = 2000
    proposal_size: int = 100
    acquisition: str = "ei"
    ratio_acq: float = 0.40
    ratio_global: float = 0.40
    ratio_diverse: float = 0.20
    proposal_min_distance: float = 0.00
    gp_noise_level: float = 1e-4

    def __post_init__(self) -> None:
        if self.raw_pool_size < 10:
            raise ValueError("raw_pool_size must be at least 10.")
        if not 1 <= self.proposal_size <= self.raw_pool_size:
            raise ValueError("proposal_size must lie in [1, raw_pool_size].")
        if self.acquisition.lower() not in {"ei", "lcb", "pi"}:
            raise ValueError("acquisition must be 'ei', 'lcb', or 'pi'.")
        ratios = (self.ratio_acq, self.ratio_global, self.ratio_diverse)
        if any(r < 0.0 for r in ratios) or not np.isclose(sum(ratios), 1.0):
            raise ValueError(
                "ratio_acq + ratio_global + ratio_diverse must equal 1."
            )
        if self.proposal_min_distance < 0.0:
            raise ValueError("proposal_min_distance must be non-negative.")
        if self.gp_noise_level <= 0.0:
            raise ValueError("gp_noise_level must be positive.")


@dataclass(frozen=True)
class RegionScreeningConfig:
    """Configuration of source-region candidate screening.

    ``policy='fixed'`` always applies the source filter at full strength.
    ``policy='adaptive'`` estimates source-target compatibility from evaluated
    target observations and continuously adjusts the retained fraction.
    ``policy='none'`` is the target-only baseline.
    """

    policy: ScreeningPolicy = "adaptive"
    geometry: ScreeningGeometry = "quantile"
    retain_ratio: float = 0.25
    ellipsoid_confidence: float = 0.95
    source_aggregation: str = "max"

    min_source_variation: float = 1e-10
    min_target_points: int = 5
    elite_ratio: float = 0.30
    prior_trust: float = 0.20
    prior_strength: float = 2.0
    evidence_shrinkage: float = 8.0
    activation_threshold: float = 0.05

    min_retained: int = 1
    batch_min_distance: float = 0.02

    def __post_init__(self) -> None:
        if self.policy not in {"none", "fixed", "adaptive"}:
            raise ValueError("policy must be 'none', 'fixed', or 'adaptive'.")
        if self.geometry not in {"quantile", "ellipsoid", "hybrid"}:
            raise ValueError(
                "geometry must be 'quantile', 'ellipsoid', or 'hybrid'."
            )
        if not 0.0 < self.retain_ratio <= 1.0:
            raise ValueError("retain_ratio must lie in (0, 1].")
        if not 0.0 < self.ellipsoid_confidence < 1.0:
            raise ValueError("ellipsoid_confidence must lie in (0, 1).")
        if self.source_aggregation not in {"max", "weighted_sum"}:
            raise ValueError("source_aggregation must be 'max' or 'weighted_sum'.")
        if self.min_source_variation < 0.0:
            raise ValueError("min_source_variation must be non-negative.")
        if self.min_target_points < 2:
            raise ValueError("min_target_points must be at least 2.")
        if not 0.0 < self.elite_ratio < 0.5:
            raise ValueError("elite_ratio must lie in (0, 0.5).")
        if not 0.0 <= self.prior_trust <= 1.0:
            raise ValueError("prior_trust must lie in [0, 1].")
        if self.prior_strength < 0.0:
            raise ValueError("prior_strength must be non-negative.")
        if self.evidence_shrinkage < 0.0:
            raise ValueError("evidence_shrinkage must be non-negative.")
        if not 0.0 <= self.activation_threshold <= 1.0:
            raise ValueError("activation_threshold must lie in [0, 1].")
        if self.min_retained < 1:
            raise ValueError("min_retained must be at least 1.")
        if self.batch_min_distance < 0.0:
            raise ValueError("batch_min_distance must be non-negative.")


@dataclass(frozen=True)
class RegionFilteredBOConfig:
    """Complete optimizer configuration."""

    top_ratio: float = 0.20
    max_clusters: int = 3
    min_samples_per_cluster: int = 3
    merge_threshold: float = 0.5
    proposal: TargetProposalConfig = field(default_factory=TargetProposalConfig)
    screening: RegionScreeningConfig = field(default_factory=RegionScreeningConfig)
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


@dataclass
class TargetProposalSet:
    """Target-only proposal-stage diagnostics."""

    points: Array
    acquisition_scores: Array
    raw_pool: Array
    raw_acquisition_scores: Array
    raw_indices: Array
    iteration: int


@dataclass(frozen=True)
class CompatibilityEstimate:
    """Target-evidence estimate of source-region compatibility."""

    trust: float
    raw_evidence: float
    spearman_correlation: float
    elite_enrichment: float
    source_variation: float
    n_target: int


@dataclass
class RegionScreeningDecision:
    """Complete diagnostics of one target-proposal screening decision."""

    points: Array
    selected_indices: Array
    proposal_set: TargetProposalSet
    source_scores: Array
    retained_mask: Array
    source_filter_mask: Array
    effective_retain_ratio: float
    source_threshold: float
    filter_active: bool
    target_top1_retained: bool
    compatibility: CompatibilityEstimate
    iteration: int

    @property
    def retained_indices(self) -> Array:
        return np.flatnonzero(self.retained_mask)


@dataclass
class RegionFilteredBOResult:
    """Closed-loop optimization result."""

    X: Array
    y: Array
    best_x: Array
    best_y: float
    best_y_trace: List[float]
    decisions: List[RegionScreeningDecision] = field(default_factory=list)


class TargetCandidateProposer:
    """Generate candidate proposals using only the target surrogate."""

    def __init__(
        self,
        bounds: Array,
        config: TargetProposalConfig,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.bounds = _validate_bounds(bounds)
        self.dim = int(self.bounds.shape[0])
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng(42)

    def _repair_pool(self, pool: Array, current_X: Optional[Array]) -> Array:
        """Remove duplicates and overlap with evaluated target points."""

        points = np.asarray(pool, dtype=float).copy()
        evaluated = (
            np.asarray(current_X, dtype=float).reshape(-1, self.dim)
            if current_X is not None and len(current_X) > 0
            else np.empty((0, self.dim), dtype=float)
        )
        accepted: List[Array] = []
        tolerance = 1e-10

        for original in points:
            point = original.copy()
            for _ in range(2000):
                target_ok = (
                    len(evaluated) == 0
                    or np.min(np.linalg.norm(evaluated - point, axis=1)) > tolerance
                )
                pool_ok = (
                    not accepted
                    or np.min(np.linalg.norm(np.asarray(accepted) - point, axis=1))
                    > tolerance
                )
                if target_ok and pool_ok:
                    break
                point = self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1])
            else:
                raise RuntimeError("Unable to construct a unique target pool.")
            accepted.append(point)

        repaired = np.asarray(accepted, dtype=float)
        if repaired.shape != points.shape:
            raise RuntimeError("Target-pool repair changed the pool shape.")
        return repaired

    def _select_diverse_top(self, pool: Array, scores: Array) -> Array:
        order = np.argsort(-scores, kind="stable")
        requested = self.config.proposal_size
        if requested == len(pool) or self.config.proposal_min_distance <= 0.0:
            return order[:requested]

        scale = np.maximum(self.bounds[:, 1] - self.bounds[:, 0], 1e-12)
        selected: List[int] = []
        for idx in order:
            if not selected:
                selected.append(int(idx))
            else:
                diff = (pool[np.asarray(selected)] - pool[idx]) / scale
                distances = np.linalg.norm(diff, axis=1)
                if np.min(distances) >= self.config.proposal_min_distance:
                    selected.append(int(idx))
            if len(selected) == requested:
                break

        if len(selected) < requested:
            selected_set = set(selected)
            for idx in order:
                if int(idx) not in selected_set:
                    selected.append(int(idx))
                if len(selected) == requested:
                    break
        return np.asarray(selected, dtype=int)

    def propose(
        self,
        surrogate: TargetGPSurrogate,
        current_X: Array,
        iteration: int = 0,
    ) -> TargetProposalSet:
        """Create a target-only proposal set.

        Source observations and source-region scores are intentionally absent from
        this method. They enter only in :class:`SourceRegionCandidateFilter`.
        """

        current = _validate_points(current_X, self.bounds, name="current_X")
        generator = CandidatePoolGenerator(
            bounds=self.bounds,
            pool_size=self.config.raw_pool_size,
            ratio_acq=self.config.ratio_acq,
            ratio_global=self.config.ratio_global,
            ratio_diverse=self.config.ratio_diverse,
            rng=self.rng,
        )
        raw_pool = generator.generate(
            surrogate=surrogate,
            current_X=current,
            excluded_datasets=None,
        )
        raw_pool = self._repair_pool(raw_pool, current)
        raw_scores = np.asarray(
            surrogate.compute_acquisition(
                raw_pool,
                acq_type=self.config.acquisition,
            ),
            dtype=float,
        ).reshape(-1)
        if len(raw_scores) != len(raw_pool):
            raise RuntimeError("Surrogate returned an invalid acquisition vector.")

        proposal_indices = self._select_diverse_top(raw_pool, raw_scores)
        return TargetProposalSet(
            points=raw_pool[proposal_indices].copy(),
            acquisition_scores=raw_scores[proposal_indices].copy(),
            raw_pool=raw_pool,
            raw_acquisition_scores=raw_scores,
            raw_indices=proposal_indices,
            iteration=int(iteration),
        )


class SourceRegionCandidateFilter:
    """Screen target proposals using source-domain high-quality regions."""

    def __init__(
        self,
        bounds: Array,
        config: RegionScreeningConfig,
    ) -> None:
        self.bounds = _validate_bounds(bounds)
        self.dim = int(self.bounds.shape[0])
        self.config = config

    def estimate_compatibility(
        self,
        library: SourceRegionLibrary,
        target_X: Array,
        target_y: Array,
    ) -> CompatibilityEstimate:
        """Estimate whether source-region support agrees with target evidence.

        Two complementary signals are combined:

        * global rank agreement between source support and target utility;
        * elite enrichment: whether the best target observations receive larger
          source-region support than the remaining observations.

        The evidence is shrunk toward a configurable prior in the few-shot regime.
        Negative evidence is not converted into a negative filter; it disables or
        weakens transfer instead.
        """

        X = _validate_points(target_X, self.bounds, name="target_X")
        y = _validate_values(target_y, len(X), name="target_y")
        n = len(X)

        if not library.regions:
            return CompatibilityEstimate(0.0, 0.0, 0.0, 0.0, 0.0, n)

        support = np.asarray(
            library.score(X, aggregation=self.config.source_aggregation),
            dtype=float,
        ).reshape(-1)
        variation = float(np.ptp(support))
        if variation < self.config.min_source_variation or np.var(support) < self.config.min_source_variation:
            return CompatibilityEstimate(0.0, 0.0, 0.0, 0.0, variation, n)

        rho = 0.0
        if n >= 3 and np.std(y) > 1e-12:
            estimate = spearmanr(support, -y).statistic
            if np.isfinite(estimate):
                rho = float(estimate)

        support_rank = _rank_normalize(support)
        elite_count = max(1, int(np.ceil(n * self.config.elite_ratio)))
        elite_idx = np.argsort(y, kind="stable")[:elite_count]
        rest_mask = np.ones(n, dtype=bool)
        rest_mask[elite_idx] = False
        if np.any(rest_mask):
            elite_enrichment = float(
                np.mean(support_rank[elite_idx]) - np.mean(support_rank[rest_mask])
            )
        else:
            elite_enrichment = 0.0

        positive_rho = max(0.0, rho)
        positive_enrichment = max(0.0, elite_enrichment)
        raw_evidence = float(
            np.clip(0.5 * positive_rho + 0.5 * positive_enrichment, 0.0, 1.0)
        )

        if self.config.policy == "fixed":
            trust = 1.0
        elif self.config.policy == "none":
            trust = 0.0
        else:
            usable_n = max(0, n - self.config.min_target_points + 1)
            data_weight = usable_n / (
                usable_n + self.config.evidence_shrinkage + 1e-12
            )
            prior_weight = self.config.prior_strength / (
                self.config.prior_strength + usable_n + 1e-12
            )
            # The two weights intentionally need not sum exactly to one when
            # evidence_shrinkage is positive; remaining mass represents caution.
            trust = (
                prior_weight * self.config.prior_trust
                + data_weight * raw_evidence
            )
            trust = float(np.clip(trust, 0.0, 1.0))

        return CompatibilityEstimate(
            trust=trust,
            raw_evidence=raw_evidence,
            spearman_correlation=rho,
            elite_enrichment=elite_enrichment,
            source_variation=variation,
            n_target=n,
        )

    def _effective_retain_ratio(self, trust: float) -> float:
        if self.config.policy == "none":
            return 1.0
        if self.config.policy == "fixed":
            return self.config.retain_ratio
        # At trust=0 the filter retains everything; at trust=1 it reaches the
        # configured retain ratio. This turns target evidence into filter strength.
        return float(
            1.0 - trust * (1.0 - self.config.retain_ratio)
        )

    @staticmethod
    def _top_support_mask(scores: Array, count: int) -> Tuple[Array, float]:
        count = min(max(1, int(count)), len(scores))
        order = np.argsort(-scores, kind="stable")
        selected = order[:count]
        threshold = float(scores[selected[-1]])
        mask = scores >= threshold
        return mask, threshold

    def _construct_source_mask(
        self,
        library: SourceRegionLibrary,
        proposals: Array,
        source_scores: Array,
        retain_count: int,
    ) -> Tuple[Array, float]:
        quantile_mask, threshold = self._top_support_mask(
            source_scores,
            retain_count,
        )
        if self.config.geometry == "quantile":
            return quantile_mask, threshold

        ellipsoid_mask = library.filter_inside_any_region(
            proposals,
            confidence=self.config.ellipsoid_confidence,
        )
        ellipsoid_mask = np.asarray(ellipsoid_mask, dtype=bool).reshape(-1)

        if self.config.geometry == "ellipsoid":
            mask = ellipsoid_mask
        else:
            mask = ellipsoid_mask & quantile_mask

        if int(np.sum(mask)) < min(retain_count, self.config.min_retained):
            mask = quantile_mask
        return mask, threshold

    def _select_batch(
        self,
        proposals: Array,
        acquisition_scores: Array,
        retained_mask: Array,
        n_points: int,
    ) -> Array:
        retained = np.flatnonzero(retained_mask)
        if len(retained) < n_points:
            raise RuntimeError("The filter retained fewer candidates than requested.")

        ordered = retained[
            np.argsort(-acquisition_scores[retained], kind="stable")
        ]
        if n_points == 1:
            return ordered[:1]

        scale = np.maximum(self.bounds[:, 1] - self.bounds[:, 0], 1e-12)
        selected: List[int] = []
        for idx in ordered:
            if not selected:
                selected.append(int(idx))
            else:
                diff = (proposals[np.asarray(selected)] - proposals[idx]) / scale
                if np.min(np.linalg.norm(diff, axis=1)) >= self.config.batch_min_distance:
                    selected.append(int(idx))
            if len(selected) == n_points:
                break

        if len(selected) < n_points:
            selected_set = set(selected)
            for idx in ordered:
                if int(idx) not in selected_set:
                    selected.append(int(idx))
                if len(selected) == n_points:
                    break
        return np.asarray(selected, dtype=int)

    def screen(
        self,
        proposal_set: TargetProposalSet,
        library: SourceRegionLibrary,
        target_X: Array,
        target_y: Array,
        n_points: int = 1,
    ) -> RegionScreeningDecision:
        """Filter target proposals and select by target acquisition within the filter."""

        proposals = _validate_points(
            proposal_set.points,
            self.bounds,
            name="proposal_set.points",
        )
        acquisition = _validate_values(
            proposal_set.acquisition_scores,
            len(proposals),
            name="proposal_set.acquisition_scores",
        )
        if not 1 <= n_points <= len(proposals):
            raise ValueError("n_points must lie in [1, number of proposals].")

        source_scores = (
            np.asarray(
                library.score(
                    proposals,
                    aggregation=self.config.source_aggregation,
                ),
                dtype=float,
            ).reshape(-1)
            if library.regions
            else np.zeros(len(proposals), dtype=float)
        )
        compatibility = self.estimate_compatibility(
            library,
            target_X,
            target_y,
        )

        score_has_information = (
            library.regions
            and np.ptp(source_scores) >= self.config.min_source_variation
            and np.var(source_scores) >= self.config.min_source_variation
        )
        filter_active = bool(
            self.config.policy != "none"
            and score_has_information
            and compatibility.trust >= self.config.activation_threshold
        )

        effective_ratio = (
            self._effective_retain_ratio(compatibility.trust)
            if filter_active
            else 1.0
        )
        retain_count = max(
            n_points,
            self.config.min_retained,
            int(np.ceil(len(proposals) * effective_ratio)),
        )
        retain_count = min(retain_count, len(proposals))

        if filter_active and retain_count < len(proposals):
            source_mask, source_threshold = self._construct_source_mask(
                library,
                proposals,
                source_scores,
                retain_count,
            )
        else:
            source_mask = np.ones(len(proposals), dtype=bool)
            source_threshold = float(np.min(source_scores)) if len(source_scores) else 0.0

        if int(np.sum(source_mask)) < n_points:
            source_mask, source_threshold = self._top_support_mask(
                source_scores,
                n_points,
            )

        retained_mask = source_mask.copy()
        selected_indices = self._select_batch(
            proposals,
            acquisition,
            retained_mask,
            n_points,
        )
        target_top1 = int(np.argmax(acquisition))

        return RegionScreeningDecision(
            points=proposals[selected_indices].copy(),
            selected_indices=selected_indices,
            proposal_set=proposal_set,
            source_scores=source_scores,
            retained_mask=retained_mask,
            source_filter_mask=source_mask,
            effective_retain_ratio=float(np.mean(retained_mask)),
            source_threshold=source_threshold,
            filter_active=filter_active and not np.all(retained_mask),
            target_top1_retained=bool(retained_mask[target_top1]),
            compatibility=compatibility,
            iteration=proposal_set.iteration,
        )


class RegionFilteredTargetBO:
    """Closed-loop optimizer implementing target proposal then source filtering."""

    def __init__(
        self,
        bounds: Array,
        config: Optional[RegionFilteredBOConfig] = None,
    ) -> None:
        self.bounds = _validate_bounds(bounds)
        self.dim = int(self.bounds.shape[0])
        self.config = config or RegionFilteredBOConfig()

        seeds = np.random.SeedSequence(self.config.random_state).spawn(2)
        self._proposal_rng = np.random.default_rng(seeds[0])
        self._proposer = TargetCandidateProposer(
            self.bounds,
            self.config.proposal,
            rng=self._proposal_rng,
        )
        self._filter = SourceRegionCandidateFilter(
            self.bounds,
            self.config.screening,
        )

        self.source_regions = SourceRegionLibrary()
        self.target_X: Optional[Array] = None
        self.target_y: Optional[Array] = None
        self.iteration = 0
        self.last_decision: Optional[RegionScreeningDecision] = None

    def fit_source_regions(
        self,
        source_datasets: Sequence[SourceDataset],
        task_ids: Optional[Sequence[str]] = None,
    ) -> SourceRegionLibrary:
        datasets = list(source_datasets)
        if not datasets:
            raise ValueError("source_datasets must contain at least one dataset.")
        if task_ids is not None and len(task_ids) != len(datasets):
            raise ValueError("task_ids must match source_datasets in length.")

        validated: List[SourceDataset] = []
        for idx, (X, y) in enumerate(datasets):
            X_arr = _validate_points(
                X,
                self.bounds,
                name=f"source_datasets[{idx}].X",
            )
            y_arr = _validate_values(
                y,
                len(X_arr),
                name=f"source_datasets[{idx}].y",
            )
            validated.append((X_arr, y_arr))

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
        for idx, region in enumerate(library.regions):
            if len(region.center) != self.dim:
                raise ValueError(
                    f"Source region {idx} has dimension {len(region.center)}; "
                    f"expected {self.dim}."
                )
        self.source_regions = library

    def initialize_target(self, X: Array, y: Array) -> None:
        X_arr = _validate_points(X, self.bounds, name="target_X")
        y_arr = _validate_values(y, len(X_arr), name="target_y")
        if len(X_arr) < 2:
            raise ValueError("At least two target observations are required.")
        self.target_X = X_arr
        self.target_y = y_arr
        self.iteration = 0
        self.last_decision = None

    @property
    def is_initialized(self) -> bool:
        return self.target_X is not None and self.target_y is not None

    def ask(self, n_points: int = 1) -> RegionScreeningDecision:
        if not self.is_initialized:
            raise RuntimeError("Call initialize_target before ask.")
        assert self.target_X is not None
        assert self.target_y is not None

        surrogate = TargetGPSurrogate(
            dim=self.dim,
            noise_level=self.config.proposal.gp_noise_level,
            random_state=self.config.random_state + self.iteration,
        )
        surrogate.fit(self.target_X, self.target_y)
        proposal_set = self._proposer.propose(
            surrogate,
            current_X=self.target_X,
            iteration=self.iteration,
        )
        decision = self._filter.screen(
            proposal_set,
            self.source_regions,
            self.target_X,
            self.target_y,
            n_points=n_points,
        )
        self.last_decision = decision
        return decision

    def tell(self, X: Array, y: Array) -> None:
        if not self.is_initialized:
            raise RuntimeError("Call initialize_target before tell.")
        assert self.target_X is not None
        assert self.target_y is not None

        X_arr = _validate_points(X, self.bounds, name="new_target_X")
        y_arr = _validate_values(y, len(X_arr), name="new_target_y")
        distances = np.linalg.norm(
            self.target_X[:, None, :] - X_arr[None, :, :],
            axis=2,
        )
        if np.any(distances < 1e-12):
            raise ValueError("tell received an already evaluated target point.")
        if len(X_arr) > 1:
            within = np.linalg.norm(
                X_arr[:, None, :] - X_arr[None, :, :],
                axis=2,
            )
            upper = within[np.triu_indices(len(X_arr), k=1)]
            if np.any(upper < 1e-12):
                raise ValueError("tell received duplicate points within the batch.")

        self.target_X = np.vstack([self.target_X, X_arr])
        self.target_y = np.concatenate([self.target_y, y_arr])
        self.iteration += 1

    def get_best(self) -> Tuple[Array, float]:
        if not self.is_initialized:
            raise RuntimeError("The target task has not been initialized.")
        assert self.target_X is not None
        assert self.target_y is not None
        idx = int(np.argmin(self.target_y))
        return self.target_X[idx].copy(), float(self.target_y[idx])

    def optimize(
        self,
        objective: Objective,
        init_X: Array,
        init_y: Optional[Array] = None,
        budget: int = 20,
        batch_size: int = 1,
    ) -> RegionFilteredBOResult:
        if budget < 0:
            raise ValueError("budget must be non-negative.")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        initial_X = _validate_points(init_X, self.bounds, name="init_X")
        initial_y = (
            _evaluate_objective(objective, initial_X)
            if init_y is None
            else _validate_values(init_y, len(initial_X), name="init_y")
        )
        self.initialize_target(initial_X, initial_y)

        best = float(np.min(initial_y))
        trace = [best]
        decisions: List[RegionScreeningDecision] = []
        remaining = int(budget)
        while remaining > 0:
            n_points = min(batch_size, remaining)
            decision = self.ask(n_points=n_points)
            values = _evaluate_objective(objective, decision.points)
            self.tell(decision.points, values)
            decisions.append(decision)
            for value in values:
                best = min(best, float(value))
                trace.append(best)
            remaining -= n_points

        assert self.target_X is not None
        assert self.target_y is not None
        best_x, best_y = self.get_best()
        return RegionFilteredBOResult(
            X=self.target_X.copy(),
            y=self.target_y.copy(),
            best_x=best_x,
            best_y=best_y,
            best_y_trace=trace,
            decisions=decisions,
        )


def _validate_bounds(bounds: Array) -> Array:
    arr = np.asarray(bounds, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("bounds must have shape (dimension, 2).")
    if not np.all(np.isfinite(arr)):
        raise ValueError("bounds must contain only finite values.")
    if np.any(arr[:, 0] >= arr[:, 1]):
        raise ValueError("Every lower bound must be smaller than its upper bound.")
    return arr.copy()


def _validate_points(X: Array, bounds: Array, *, name: str) -> Array:
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    dim = bounds.shape[0]
    if arr.ndim != 2 or arr.shape[1] != dim:
        raise ValueError(f"{name} must have shape (n, {dim}).")
    if len(arr) == 0:
        raise ValueError(f"{name} must contain at least one point.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any(arr < bounds[:, 0]) or np.any(arr > bounds[:, 1]):
        raise ValueError(f"{name} contains points outside bounds.")
    return arr.copy()


def _validate_values(y: Array, n: int, *, name: str) -> Array:
    arr = np.asarray(y, dtype=float).reshape(-1)
    if len(arr) != n:
        raise ValueError(f"{name} must contain exactly {n} values.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr.copy()


def _rank_normalize(values: Array) -> Array:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if len(arr) <= 1 or np.ptp(arr) < 1e-14:
        return np.zeros_like(arr)
    ranks = rankdata(arr, method="average")
    return (ranks - 1.0) / (len(arr) - 1.0)


def _evaluate_objective(objective: Objective, X: Array) -> Array:
    values = np.asarray(objective(X), dtype=float)
    if values.ndim == 0:
        values = values.reshape(1)
    values = values.reshape(-1)
    if len(values) != len(X):
        raise ValueError("objective must return one scalar per input point.")
    if not np.all(np.isfinite(values)):
        raise ValueError("objective returned non-finite values.")
    return values
