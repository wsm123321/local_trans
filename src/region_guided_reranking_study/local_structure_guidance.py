"""Unified candidate guidance using a source local-structure library."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from scipy.stats import rankdata

from .source_local_structure import SourceLocalStructureLibrary

Array = np.ndarray
GuidanceMode = Literal[
    "target_only",
    "geometry_only",
    "local_rank_no_reliability",
    "local_rank_reliability",
    "reversed_local_rank",
]


@dataclass
class LocalStructureGuidanceDecision:
    """Diagnostics and selection returned for one candidate-pool ranking."""

    selected_index: int
    selected_indices: Array
    scores: Array
    shortlist: Array
    fallback: bool
    best_region: int
    candidates: Array
    acquisition_scores: Array
    target_scores: Array
    source_scores: Array
    normalized_target_scores: Array
    normalized_source_scores: Array
    target_nominees: Array
    source_nominees: Array
    mode: str
    source_weight: float
    fallback_reason: Optional[str] = None
    best_region_indices: Optional[Array] = None
    best_region_id: Optional[str] = None

    @property
    def combined_scores(self) -> Array:
        return self.scores

    @property
    def selected_points(self) -> Array:
        return self.candidates[self.selected_indices]

    @property
    def shortlist_indices(self) -> Array:
        return self.shortlist

    @property
    def shortlist_mask(self) -> Array:
        mask = np.zeros(len(self.candidates), dtype=bool)
        mask[self.shortlist] = True
        return mask

    @property
    def best_region_index(self) -> int:
        return self.best_region

    @property
    def effective_source_weight(self) -> float:
        return 0.0 if self.fallback else self.source_weight


# Short aliases are useful to callers that treat this result like other ask APIs.
GuidanceDecision = LocalStructureGuidanceDecision
GuidanceResult = LocalStructureGuidanceDecision


def normalize_rank_scores(scores: Array) -> Array:
    """Map higher-is-better scores to [0, 1] using average ranks for ties."""

    values = np.asarray(scores, dtype=float).reshape(-1)
    if values.size == 0:
        return values.copy()
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must contain only finite values.")
    if values.size == 1 or np.ptp(values) < 1e-12:
        return np.zeros_like(values)
    return (rankdata(values, method="average") - 1.0) / (len(values) - 1.0)


def _validate_inputs(
    candidates: Array,
    acquisition_scores: Array,
    library: SourceLocalStructureLibrary,
    n_points: int,
    source_weight: float,
    aggregation: str,
) -> tuple[Array, Array]:
    if not isinstance(library, SourceLocalStructureLibrary):
        raise TypeError("library must be a SourceLocalStructureLibrary.")
    points = np.asarray(candidates, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] == 0:
        raise ValueError("candidates must be a non-empty two-dimensional array.")
    if not np.all(np.isfinite(points)):
        raise ValueError("candidates must contain only finite values.")
    if library.dim is not None and points.shape[1] != library.dim:
        raise ValueError(f"candidates must have {library.dim} columns.")

    acquisition = np.asarray(acquisition_scores, dtype=float).reshape(-1)
    if len(acquisition) != len(points):
        raise ValueError(
            f"acquisition_scores must contain exactly {len(points)} values."
        )
    if not np.all(np.isfinite(acquisition)):
        raise ValueError("acquisition_scores must contain only finite values.")
    if not isinstance(n_points, (int, np.integer)) or not 1 <= int(n_points) <= len(points):
        raise ValueError("n_points must be an integer between 1 and the candidate count.")
    if not np.isfinite(source_weight) or source_weight < 0.0:
        raise ValueError("source_weight must be finite and non-negative.")
    if aggregation not in {"max", "weighted_sum"}:
        raise ValueError("aggregation must be 'max' or 'weighted_sum'.")
    return points.copy(), acquisition.copy()


def _aggregate(components: Array, library: SourceLocalStructureLibrary, aggregation: str) -> Array:
    if components.shape[1] == 0:
        return np.zeros(len(components), dtype=float)
    if aggregation == "max":
        return np.max(components, axis=1)
    weights = np.asarray([max(1, s.core_count) for s in library.structures], dtype=float)
    weights /= np.sum(weights)
    return components @ weights


def _reversed_local_rank_components(
    candidates: Array,
    library: SourceLocalStructureLibrary,
    use_reliability: bool,
) -> Array:
    """Return local components with only each model quality factor reversed."""

    components = []
    for structure in library.structures:
        membership = structure.membership(candidates)
        quality, _ = structure.predict_relative_quality(candidates, return_std=False)
        quality = np.clip(np.asarray(quality, dtype=float).reshape(-1), 0.0, 1.0)
        quality_factor = structure.quality_floor + (1.0 - structure.quality_floor) * (1.0 - quality)
        if use_reliability:
            reliability = (
                structure.reliability_floor
                + (1.0 - structure.reliability_floor) * structure.validation.reliability
            )
        else:
            reliability = 1.0
        components.append(structure.region_quality * membership * quality_factor * reliability)
    return np.column_stack(components) if components else np.empty((len(candidates), 0))


def _reversed_local_rank_scores(
    candidates: Array,
    library: SourceLocalStructureLibrary,
    use_reliability: bool,
    aggregation: str,
) -> Array:
    return _aggregate(
        _reversed_local_rank_components(candidates, library, use_reliability),
        library,
        aggregation,
    )


def _top_indices(scores: Array, count: int) -> Array:
    return np.argsort(-scores, kind="stable")[: min(len(scores), max(0, int(count)))]


def rank_local_structure_candidates(
    candidates: Array,
    acquisition_scores: Array,
    library: SourceLocalStructureLibrary,
    mode: GuidanceMode = "local_rank_reliability",
    source_weight: float = 1.0,
    target_nomination_ratio: float = 0.20,
    source_nomination_ratio: float = 0.20,
    shortlist_size: Optional[int] = None,
    n_points: int = 1,
    aggregation: str = "max",
) -> LocalStructureGuidanceDecision:
    """Nominate and rerank a target candidate pool with local source structure."""

    valid_modes = {
        "target_only",
        "geometry_only",
        "local_rank_no_reliability",
        "local_rank_reliability",
        "reversed_local_rank",
    }
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of: {', '.join(sorted(valid_modes))}.")
    if not 0.0 < target_nomination_ratio <= 1.0:
        raise ValueError("target_nomination_ratio must lie in (0, 1].")
    if not 0.0 < source_nomination_ratio <= 1.0:
        raise ValueError("source_nomination_ratio must lie in (0, 1].")
    if shortlist_size is not None and (
        not isinstance(shortlist_size, (int, np.integer)) or shortlist_size < 1
    ):
        raise ValueError("shortlist_size must be a positive integer when provided.")

    points, acquisition = _validate_inputs(
        candidates, acquisition_scores, library, n_points, source_weight, aggregation
    )
    target_norm = normalize_rank_scores(acquisition)
    target_count = max(int(n_points), int(np.ceil(len(points) * target_nomination_ratio)))
    target_nominees = _top_indices(target_norm, target_count)

    fallback = False
    fallback_reason: Optional[str] = None
    source_raw = np.zeros(len(points), dtype=float)
    source_norm = np.zeros(len(points), dtype=float)
    best_regions = np.full(len(points), -1, dtype=int)

    if mode == "target_only":
        fallback_reason = None
    elif source_weight == 0.0:
        fallback = True
        fallback_reason = "source_weight_zero"
    elif not library.structures:
        fallback = True
        fallback_reason = "empty_library"
    else:
        try:
            if mode == "geometry_only":
                source_raw = np.asarray(
                    library.geometry_score(points, aggregation=aggregation), dtype=float
                ).reshape(-1)
                geometry_components = library.geometry_components(points)
                best_regions = (
                    np.argmax(geometry_components, axis=1)
                    if geometry_components.shape[1]
                    else np.full(len(points), -1, dtype=int)
                )
            elif mode == "local_rank_no_reliability":
                source_raw = np.asarray(
                    library.score(points, aggregation=aggregation, use_reliability=False), dtype=float
                ).reshape(-1)
                best_regions = library.best_region_indices(points, use_reliability=False)
            elif mode == "local_rank_reliability":
                source_raw = np.asarray(
                    library.score(points, aggregation=aggregation, use_reliability=True), dtype=float
                ).reshape(-1)
                best_regions = library.best_region_indices(points, use_reliability=True)
            else:
                source_raw = _reversed_local_rank_scores(
                    points, library, use_reliability=True, aggregation=aggregation
                )
                components = _reversed_local_rank_components(
                    points, library, use_reliability=True
                )
                best_regions = np.argmax(components, axis=1)
        except (ValueError, TypeError, RuntimeError, FloatingPointError):
            fallback = True
            fallback_reason = "nonfinite_source_scores"

        if not fallback:
            if len(source_raw) != len(points) or not np.all(np.isfinite(source_raw)):
                fallback = True
                fallback_reason = "nonfinite_source_scores"
            elif np.ptp(source_raw) < 1e-12:
                fallback = True
                fallback_reason = "constant_source_scores"
            else:
                source_norm = normalize_rank_scores(source_raw)

    if fallback or mode == "target_only":
        source_raw = np.zeros(len(points), dtype=float)
        source_norm = np.zeros(len(points), dtype=float)
        source_nominees = np.empty(0, dtype=int)
        scores = target_norm.copy()
        shortlist = target_nominees.copy()
        best_regions = np.full(len(points), -1, dtype=int) if fallback else best_regions
    else:
        source_count = max(int(n_points), int(np.ceil(len(points) * source_nomination_ratio)))
        source_nominees = _top_indices(source_norm, source_count)
        union = set(target_nominees.tolist()) | set(source_nominees.tolist())
        shortlist = np.asarray(sorted(union), dtype=int)
        scores = target_norm + float(source_weight) * source_norm

    if shortlist_size is not None and len(shortlist) > int(shortlist_size):
        mandatory = set(target_nominees.tolist())
        if not fallback and mode != "target_only":
            mandatory.update(source_nominees.tolist())
        ordered = sorted(shortlist.tolist(), key=lambda i: (-scores[i], i))
        retained = list(sorted(mandatory))
        retained.extend(i for i in ordered if i not in mandatory)
        shortlist = np.asarray(sorted(set(retained[: max(len(mandatory), int(shortlist_size))])), dtype=int)

    ordered = shortlist[np.argsort(-scores[shortlist], kind="stable")]
    selected_indices = ordered[: int(n_points)]
    selected_index = int(selected_indices[0])
    selected_region = int(best_regions[selected_index]) if len(best_regions) else -1

    return LocalStructureGuidanceDecision(
        selected_index=selected_index,
        selected_indices=selected_indices.copy(),
        scores=scores.copy(),
        shortlist=shortlist.copy(),
        fallback=bool(fallback),
        best_region=selected_region,
        candidates=points,
        acquisition_scores=acquisition,
        target_scores=target_norm.copy(),
        source_scores=source_raw.copy(),
        normalized_target_scores=target_norm.copy(),
        normalized_source_scores=source_norm.copy(),
        target_nominees=target_nominees.copy(),
        source_nominees=source_nominees.copy(),
        mode=mode,
        source_weight=float(source_weight),
        fallback_reason=fallback_reason,
        best_region_indices=best_regions.copy(),
        best_region_id=(
            library.structures[selected_region].region_id
            if 0 <= selected_region < len(library.structures)
            else None
        ),
    )


def score_guidance(
    library: SourceLocalStructureLibrary,
    points: Array,
    relation: str = "matching",
    source_weight: float = 1.0,
) -> Array:
    """Return normalized local guidance scores for the study adapter."""

    if relation not in {"matching", "reversed"}:
        raise ValueError("relation must be 'matching' or 'reversed'.")
    point_array = np.asarray(points)
    n_points = 1 if point_array.ndim == 1 else len(point_array)
    mode = "reversed_local_rank" if relation == "reversed" else "local_rank_reliability"
    values = rank_local_structure_candidates(
        points,
        np.zeros(n_points),
        library,
        mode=mode,
        source_weight=source_weight,
        target_nomination_ratio=1.0,
        source_nomination_ratio=1.0,
    )
    return np.clip(values.normalized_source_scores, 0.0, 1.0)


class LocalStructureGuidance:
    """Reusable facade around the unified local-structure ranking core."""

    def __init__(
        self,
        library: SourceLocalStructureLibrary,
        source_weight: float = 1.0,
        target_nomination_ratio: float = 0.20,
        source_nomination_ratio: float = 0.20,
        shortlist_size: Optional[int] = None,
        aggregation: str = "max",
    ) -> None:
        if not isinstance(library, SourceLocalStructureLibrary):
            raise TypeError("library must be a SourceLocalStructureLibrary.")
        self.library = library
        self.source_weight = source_weight
        self.target_nomination_ratio = target_nomination_ratio
        self.source_nomination_ratio = source_nomination_ratio
        self.shortlist_size = shortlist_size
        self.aggregation = aggregation

    def score_candidates(
        self,
        candidates: Array,
        relation: str = "matching",
    ) -> Array:
        return score_guidance(
            self.library,
            candidates,
            relation=relation,
            source_weight=self.source_weight,
        )

    def target_only(self, candidates: Array, acquisition_scores: Array, **kwargs):
        return self.rank(candidates, acquisition_scores, mode="target_only", **kwargs)

    def geometry_only(self, candidates: Array, acquisition_scores: Array, **kwargs):
        return self.rank(candidates, acquisition_scores, mode="geometry_only", **kwargs)

    def local_rank_no_reliability(self, candidates: Array, acquisition_scores: Array, **kwargs):
        return self.rank(candidates, acquisition_scores, mode="local_rank_no_reliability", **kwargs)

    def local_rank_reliability(self, candidates: Array, acquisition_scores: Array, **kwargs):
        return self.rank(candidates, acquisition_scores, mode="local_rank_reliability", **kwargs)

    def reversed_local_rank(self, candidates: Array, acquisition_scores: Array, **kwargs):
        return self.rank(candidates, acquisition_scores, mode="reversed_local_rank", **kwargs)

    def rank(
        self,
        candidates: Array,
        acquisition_scores: Array,
        mode: GuidanceMode = "local_rank_reliability",
        n_points: int = 1,
        **kwargs,
    ) -> LocalStructureGuidanceDecision:
        options = {
            "source_weight": self.source_weight,
            "target_nomination_ratio": self.target_nomination_ratio,
            "source_nomination_ratio": self.source_nomination_ratio,
            "shortlist_size": self.shortlist_size,
            "aggregation": self.aggregation,
        }
        options.update(kwargs)
        return rank_local_structure_candidates(
            candidates, acquisition_scores, self.library, mode=mode,
            n_points=n_points, **options
        )

    __call__ = rank


def guide_local_structure(*args, **kwargs) -> LocalStructureGuidanceDecision:
    return rank_local_structure_candidates(*args, **kwargs)


def select_local_structure_candidate(*args, **kwargs) -> LocalStructureGuidanceDecision:
    return rank_local_structure_candidates(*args, **kwargs)


def target_only(candidates: Array, acquisition_scores: Array, library: Optional[SourceLocalStructureLibrary] = None, **kwargs) -> LocalStructureGuidanceDecision:
    if library is None:
        library = SourceLocalStructureLibrary()
    return rank_local_structure_candidates(candidates, acquisition_scores, library, mode="target_only", **kwargs)


def geometry_only(candidates: Array, acquisition_scores: Array, library: SourceLocalStructureLibrary, **kwargs) -> LocalStructureGuidanceDecision:
    return rank_local_structure_candidates(candidates, acquisition_scores, library, mode="geometry_only", **kwargs)


def local_rank_no_reliability(candidates: Array, acquisition_scores: Array, library: SourceLocalStructureLibrary, **kwargs) -> LocalStructureGuidanceDecision:
    return rank_local_structure_candidates(candidates, acquisition_scores, library, mode="local_rank_no_reliability", **kwargs)


def local_rank_reliability(candidates: Array, acquisition_scores: Array, library: SourceLocalStructureLibrary, **kwargs) -> LocalStructureGuidanceDecision:
    return rank_local_structure_candidates(candidates, acquisition_scores, library, mode="local_rank_reliability", **kwargs)


def reversed_local_rank(candidates: Array, acquisition_scores: Array, library: SourceLocalStructureLibrary, **kwargs) -> LocalStructureGuidanceDecision:
    return rank_local_structure_candidates(candidates, acquisition_scores, library, mode="reversed_local_rank", **kwargs)


__all__ = [
    "GuidanceDecision",
    "GuidanceMode",
    "GuidanceResult",
    "LocalStructureGuidance",
    "LocalStructureGuidanceDecision",
    "geometry_only",
    "guide_local_structure",
    "local_rank_no_reliability",
    "local_rank_reliability",
    "normalize_rank_scores",
    "rank_local_structure_candidates",
    "reversed_local_rank",
    "score_guidance",
    "select_local_structure_candidate",
    "target_only",
]
