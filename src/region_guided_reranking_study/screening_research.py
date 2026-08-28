"""Utilities shared by source-region candidate-screening experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import rankdata

from .rerankers import normalize_scores
from .source_regions import (
    SourceRegionExtractor,
    SourceRegionLibrary,
    create_structure_matched_random_library,
    create_true_oracle_library,
)
from .target_region_screening import (
    RegionScreeningConfig,
    RegionScreeningDecision,
    SourceRegionCandidateFilter,
    TargetProposalSet,
)

Array = np.ndarray
SourceDataset = Tuple[Array, Array]


@dataclass
class SourceLibraryBundle:
    matching: SourceRegionLibrary
    random: SourceRegionLibrary
    wrong: SourceRegionLibrary
    oracle: SourceRegionLibrary
    matching_datasets: List[SourceDataset]
    wrong_datasets: List[SourceDataset]


@dataclass(frozen=True)
class SelectionMetrics:
    selected_y: float
    raw_regret: float
    normalized_regret: float
    true_rank: int
    acquisition_rank: int
    hit_top05: float
    hit_top10: float


def load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Configuration root must be a JSON object.")
    return data


def get_required(config: Mapping[str, Any], key: str) -> Any:
    if key not in config:
        raise KeyError(f"Missing required configuration key: {key}")
    return config[key]


def extract_source_library(
    source_functions: Sequence,
    bounds: Array,
    dim: int,
    n_samples: int,
    rng: np.random.Generator,
    *,
    top_ratio: float,
    max_clusters: int,
    min_samples_per_cluster: int,
    merge_threshold: float,
    random_state: int,
    task_prefix: str,
) -> Tuple[SourceRegionLibrary, List[SourceDataset]]:
    datasets: List[SourceDataset] = []
    for source_idx, source_function in enumerate(source_functions):
        X = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_samples, dim))
        y = np.asarray(source_function(X), dtype=float).reshape(-1)
        datasets.append((X, y))

    extractor = SourceRegionExtractor(
        top_ratio=top_ratio,
        max_clusters=max_clusters,
        min_samples_per_cluster=min_samples_per_cluster,
        random_state=random_state,
    )
    task_ids = [f"{task_prefix}_{idx}" for idx in range(len(datasets))]
    library = extractor.extract_from_multi_sources(
        datasets,
        task_ids=task_ids,
        merge_threshold=merge_threshold,
    )
    return library, datasets


def build_source_library_bundle(
    suite_entry: Mapping[str, Any],
    dim: int,
    source_samples: int,
    match_rng: np.random.Generator,
    wrong_rng: np.random.Generator,
    random_rng: np.random.Generator,
    *,
    top_ratio: float,
    max_clusters: int,
    min_samples_per_cluster: int,
    merge_threshold: float,
    random_state: int,
) -> SourceLibraryBundle:
    bounds = np.asarray(suite_entry["bounds"], dtype=float)
    matching, matching_datasets = extract_source_library(
        suite_entry["matching_sources"],
        bounds,
        dim,
        source_samples,
        match_rng,
        top_ratio=top_ratio,
        max_clusters=max_clusters,
        min_samples_per_cluster=min_samples_per_cluster,
        merge_threshold=merge_threshold,
        random_state=random_state,
        task_prefix="matching",
    )
    wrong, wrong_datasets = extract_source_library(
        suite_entry["mismatched_sources"],
        bounds,
        dim,
        source_samples,
        wrong_rng,
        top_ratio=top_ratio,
        max_clusters=max_clusters,
        min_samples_per_cluster=min_samples_per_cluster,
        merge_threshold=merge_threshold,
        random_state=random_state + 1,
        task_prefix="wrong",
    )
    random_library = create_structure_matched_random_library(
        matching,
        bounds=bounds,
        rng=random_rng,
    )
    oracle = create_true_oracle_library(
        suite_entry["target"].get_oracle_basins(),
        dim=dim,
    )
    return SourceLibraryBundle(
        matching=matching,
        random=random_library,
        wrong=wrong,
        oracle=oracle,
        matching_datasets=matching_datasets,
        wrong_datasets=wrong_datasets,
    )


def known_optimum_value(target_function) -> float:
    basins = target_function.get_oracle_basins()
    global_centers = [
        np.asarray(item["center"], dtype=float)
        for item in basins
        if item.get("is_global", True)
    ]
    if not global_centers:
        raise ValueError("Target landscape does not expose a global basin.")
    values = np.asarray(target_function(np.vstack(global_centers)), dtype=float)
    return float(np.min(values))


def selection_metrics(
    selected_index: int,
    true_y: Array,
    acquisition_scores: Array,
) -> SelectionMetrics:
    y = np.asarray(true_y, dtype=float).reshape(-1)
    acquisition = np.asarray(acquisition_scores, dtype=float).reshape(-1)
    if len(y) != len(acquisition):
        raise ValueError("true_y and acquisition_scores must have equal length.")

    oracle = float(np.min(y))
    selected_y = float(y[selected_index])
    raw_regret = selected_y - oracle
    scale = max(1e-12, float(np.quantile(y, 0.90) - oracle))
    normalized_regret = raw_regret / scale

    true_order = np.argsort(y, kind="stable")
    true_rank = int(np.where(true_order == selected_index)[0][0])
    acquisition_order = np.argsort(-acquisition, kind="stable")
    acquisition_rank = int(np.where(acquisition_order == selected_index)[0][0])

    return SelectionMetrics(
        selected_y=selected_y,
        raw_regret=float(raw_regret),
        normalized_regret=float(normalized_regret),
        true_rank=true_rank,
        acquisition_rank=acquisition_rank,
        hit_top05=float(selected_y <= np.quantile(y, 0.05)),
        hit_top10=float(selected_y <= np.quantile(y, 0.10)),
    )


def soft_rerank_selection(
    proposal_set: TargetProposalSet,
    library: SourceRegionLibrary,
    source_weight: float = 1.0,
    source_aggregation: str = "max",
) -> Tuple[int, Array, Array]:
    acquisition = np.asarray(proposal_set.acquisition_scores, dtype=float)
    source = (
        np.asarray(
            library.score(
                proposal_set.points,
                aggregation=source_aggregation,
            ),
            dtype=float,
        )
        if library.regions
        else np.zeros(len(proposal_set.points), dtype=float)
    )
    acquisition_norm = normalize_scores(acquisition, method="rank")
    source_norm = normalize_scores(source, method="rank")
    combined = acquisition_norm + source_weight * source_norm
    selected = int(np.argmax(combined))
    return selected, source, combined


def make_filter_decision(
    proposal_set: TargetProposalSet,
    library: SourceRegionLibrary,
    bounds: Array,
    target_X: Array,
    target_y: Array,
    screening_config: RegionScreeningConfig,
) -> RegionScreeningDecision:
    candidate_filter = SourceRegionCandidateFilter(bounds, screening_config)
    return candidate_filter.screen(
        proposal_set,
        library,
        target_X,
        target_y,
        n_points=1,
    )


def mean_bootstrap_ci(
    values: Iterable[float],
    *,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    array = np.asarray(list(values), dtype=float)
    if len(array) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(n_bootstrap, len(array)), replace=True)
    boot_means = np.mean(samples, axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.mean(array)),
        float(np.quantile(boot_means, alpha)),
        float(np.quantile(boot_means, 1.0 - alpha)),
    )


def rank_normalize(values: Array) -> Array:
    array = np.asarray(values, dtype=float).reshape(-1)
    if len(array) <= 1 or np.ptp(array) < 1e-14:
        return np.zeros_like(array)
    ranks = rankdata(array, method="average")
    return (ranks - 1.0) / (len(array) - 1.0)
