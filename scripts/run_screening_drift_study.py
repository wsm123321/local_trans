"""Phase C: continuous source-region drift and screening boundary study.

For each target instance, the target observations and target-proposed candidate set are
held fixed while the source landscape is translated by an increasing distance. The
experiment compares fixed and adaptive filtering and therefore identifies when the
source local region changes from useful guidance to harmful exclusion.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from region_guided_reranking_study.landscapes import GaussianMixtureLandscape
from region_guided_reranking_study.screening_research import (
    load_json,
    make_filter_decision,
    selection_metrics,
)
from region_guided_reranking_study.source_regions import SourceRegionExtractor, SourceRegionLibrary
from region_guided_reranking_study.surrogate_and_candidates import TargetGPSurrogate
from region_guided_reranking_study.target_region_screening import (
    RegionScreeningConfig,
    TargetCandidateProposer,
    TargetProposalConfig,
)


def _proposal_config(study: Dict[str, Any]) -> TargetProposalConfig:
    return TargetProposalConfig(
        raw_pool_size=int(study["raw_pool_size"]),
        proposal_size=int(study["proposal_size"]),
        acquisition=study["acquisition"],
        ratio_acq=float(study["ratio_acq"]),
        ratio_global=float(study["ratio_global"]),
        ratio_diverse=float(study["ratio_diverse"]),
        proposal_min_distance=float(study["proposal_min_distance"]),
        gp_noise_level=float(study["gp_noise_level"]),
    )


def _screening_config(study: Dict[str, Any], policy: str) -> RegionScreeningConfig:
    return RegionScreeningConfig(
        policy=policy,
        geometry=study["screening_geometry"],
        retain_ratio=float(study["retain_ratio"]),
        ellipsoid_confidence=float(study["ellipsoid_confidence"]),
        source_aggregation=study["source_aggregation"],
        min_source_variation=float(study["min_source_variation"]),
        min_target_points=int(study["min_target_points"]),
        elite_ratio=float(study["elite_ratio"]),
        prior_trust=float(study["prior_trust"]),
        prior_strength=float(study["prior_strength"]),
        evidence_shrinkage=float(study["evidence_shrinkage"]),
        activation_threshold=float(study["activation_threshold"]),
        min_retained=int(study["min_retained"]),
        batch_min_distance=float(study["batch_min_distance"]),
    )


def _extract_drifted_library(
    target: GaussianMixtureLandscape,
    source_X: np.ndarray,
    delta: float,
    direction: np.ndarray,
    study: Dict[str, Any],
    seed: int,
) -> SourceRegionLibrary:
    shifted_centers = [center + delta * direction for center in target.centers]
    source = GaussianMixtureLandscape(
        dim=target.dim,
        bounds=target.bounds,
        centers=shifted_centers,
        covs=target.covs,
        weights=target.weights,
        rng=np.random.default_rng(seed + 100003),
    )
    source_y = np.asarray(source(source_X), dtype=float)
    extractor = SourceRegionExtractor(
        top_ratio=float(study["top_ratio"]),
        max_clusters=int(study["max_clusters"]),
        min_samples_per_cluster=int(study["min_samples_per_cluster"]),
        random_state=seed,
    )
    return extractor.extract_from_multi_sources(
        [(source_X, source_y)],
        task_ids=[f"drift_{delta:g}"],
        merge_threshold=float(study["merge_threshold"]),
    )


def run_study(config_path: str, output_dir: str | None = None) -> pd.DataFrame:
    config = load_json(config_path)
    study = config["study"]
    output = Path(output_dir or config["output_dir"]) / "drift"
    output.mkdir(parents=True, exist_ok=True)

    dimensions = [int(value) for value in study["drift_dimensions"]]
    seeds = [int(value) for value in study["drift_seeds"]]
    deltas = [float(value) for value in study["drift_values"]]
    records: List[Dict[str, Any]] = []

    total = len(dimensions) * len(seeds)
    completed = 0
    print(f"Starting source-region drift study: {total} target instances")

    for dim in dimensions:
        n_init = int(study["n_init_multiplier"]) * dim + int(
            study["n_init_offset"]
        )
        for seed in seeds:
            completed += 1
            streams = np.random.SeedSequence(seed).spawn(5)
            target_rng = np.random.default_rng(streams[0])
            init_rng = np.random.default_rng(streams[1])
            proposal_rng = np.random.default_rng(streams[2])
            source_design_rng = np.random.default_rng(streams[3])
            direction_rng = np.random.default_rng(streams[4])

            target = GaussianMixtureLandscape(dim=dim, rng=target_rng)
            bounds = target.bounds
            target_X = init_rng.uniform(
                bounds[:, 0], bounds[:, 1], size=(n_init, dim)
            )
            target_y = np.asarray(target(target_X), dtype=float)

            surrogate = TargetGPSurrogate(
                dim=dim,
                noise_level=float(study["gp_noise_level"]),
                random_state=seed,
            )
            surrogate.fit(target_X, target_y)
            proposer = TargetCandidateProposer(
                bounds,
                _proposal_config(study),
                proposal_rng,
            )
            proposal_set = proposer.propose(
                surrogate,
                current_X=target_X,
                iteration=0,
            )
            true_y = np.asarray(target(proposal_set.points), dtype=float)
            target_only_index = int(np.argmax(proposal_set.acquisition_scores))
            target_only_metric = selection_metrics(
                target_only_index,
                true_y,
                proposal_set.acquisition_scores,
            )

            source_X = source_design_rng.uniform(
                bounds[:, 0],
                bounds[:, 1],
                size=(int(study["drift_source_samples"]), dim),
            )
            direction = direction_rng.normal(size=dim)
            direction /= max(1e-12, np.linalg.norm(direction))

            for delta in deltas:
                library = _extract_drifted_library(
                    target,
                    source_X,
                    delta,
                    direction,
                    study,
                    seed,
                )
                for policy, method in [
                    ("fixed", "Fixed-Filter"),
                    ("adaptive", "Adaptive-Filter"),
                ]:
                    decision = make_filter_decision(
                        proposal_set,
                        library,
                        bounds,
                        target_X,
                        target_y,
                        _screening_config(study, policy),
                    )
                    selected_index = int(decision.selected_indices[0])
                    metric = selection_metrics(
                        selected_index,
                        true_y,
                        proposal_set.acquisition_scores,
                    )
                    records.append(
                        {
                            "dim": dim,
                            "seed": seed,
                            "delta": delta,
                            "method": method,
                            "selected_y": metric.selected_y,
                            "normalized_regret": metric.normalized_regret,
                            "target_only_normalized_regret": (
                                target_only_metric.normalized_regret
                            ),
                            "regret_reduction": (
                                target_only_metric.normalized_regret
                                - metric.normalized_regret
                            ),
                            "true_rank": metric.true_rank,
                            "acquisition_rank": metric.acquisition_rank,
                            "hit_top10": metric.hit_top10,
                            "filter_active": float(decision.filter_active),
                            "retained_fraction": float(
                                np.mean(decision.retained_mask)
                            ),
                            "target_top1_retained": float(
                                decision.target_top1_retained
                            ),
                            "compatibility_trust": (
                                decision.compatibility.trust
                            ),
                            "compatibility_raw_evidence": (
                                decision.compatibility.raw_evidence
                            ),
                            "compatibility_spearman": (
                                decision.compatibility.spearman_correlation
                            ),
                            "source_regions": len(library.regions),
                        }
                    )

            if completed % max(1, int(study["progress_every"])) == 0:
                print(f"Progress: {completed}/{total}")

    frame = pd.DataFrame(records)
    frame.to_csv(output / "screening_drift_summary.csv", index=False)
    print(f"Saved drift results to {output}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "region_screening_full.json"),
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    run_study(args.config, args.output_dir)


if __name__ == "__main__":
    main()
