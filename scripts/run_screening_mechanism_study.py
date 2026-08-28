"""Phase A: shared-proposal mechanism study for source-region screening.

Every method receives the exact same target observations, target GP, raw candidate
pool, and target-proposed candidate set. Only the post-proposal screening policy or
region-library source changes. This isolates what source local regions contribute to
candidate decision quality.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from region_guided_reranking_study.landscapes import get_task_suite
from region_guided_reranking_study.screening_research import (
    build_source_library_bundle,
    load_json,
    make_filter_decision,
    selection_metrics,
    soft_rerank_selection,
)
from region_guided_reranking_study.source_regions import SourceRegionLibrary
from region_guided_reranking_study.surrogate_and_candidates import TargetGPSurrogate
from region_guided_reranking_study.target_region_screening import (
    RegionScreeningConfig,
    TargetCandidateProposer,
    TargetProposalConfig,
)


def _screening_config(
    study: Dict[str, Any],
    policy: str,
) -> RegionScreeningConfig:
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


def run_study(config_path: str, output_dir: str | None = None) -> pd.DataFrame:
    config = load_json(config_path)
    study = config["study"]
    output = Path(output_dir or config["output_dir"]) / "mechanism"
    output.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    detailed: List[Dict[str, Any]] = []
    problems = list(study["problems"])
    dimensions = [int(value) for value in study["dimensions"]]
    seeds = [int(value) for value in study["seeds"]]

    total = len(problems) * len(dimensions) * len(seeds)
    completed = 0
    print(f"Starting shared-proposal screening study: {total} instances")

    for dim in dimensions:
        n_init = int(study["n_init_multiplier"]) * dim + int(
            study["n_init_offset"]
        )
        for problem in problems:
            for seed in seeds:
                completed += 1
                streams = np.random.SeedSequence(seed).spawn(7)
                task_rng = np.random.default_rng(streams[0])
                match_rng = np.random.default_rng(streams[1])
                wrong_rng = np.random.default_rng(streams[2])
                random_rng = np.random.default_rng(streams[3])
                init_rng = np.random.default_rng(streams[4])
                proposal_rng = np.random.default_rng(streams[5])

                suite_entry = get_task_suite(dim=dim, rng=task_rng)[problem]
                target_function = suite_entry["target"]
                bounds = np.asarray(suite_entry["bounds"], dtype=float)

                bundle = build_source_library_bundle(
                    suite_entry,
                    dim,
                    int(study["source_samples"]),
                    match_rng,
                    wrong_rng,
                    random_rng,
                    top_ratio=float(study["top_ratio"]),
                    max_clusters=int(study["max_clusters"]),
                    min_samples_per_cluster=int(
                        study["min_samples_per_cluster"]
                    ),
                    merge_threshold=float(study["merge_threshold"]),
                    random_state=seed,
                )

                target_X = init_rng.uniform(
                    bounds[:, 0], bounds[:, 1], size=(n_init, dim)
                )
                target_y = np.asarray(target_function(target_X), dtype=float)
                surrogate = TargetGPSurrogate(
                    dim=dim,
                    noise_level=float(study["gp_noise_level"]),
                    random_state=seed,
                )
                surrogate.fit(target_X, target_y)
                proposer = TargetCandidateProposer(
                    bounds,
                    _proposal_config(study),
                    rng=proposal_rng,
                )
                proposal_set = proposer.propose(
                    surrogate,
                    current_X=target_X,
                    iteration=0,
                )
                proposal_true_y = np.asarray(
                    target_function(proposal_set.points), dtype=float
                )

                method_specs = [
                    (
                        "Target-Only",
                        SourceRegionLibrary(),
                        _screening_config(study, "none"),
                    ),
                    (
                        "Matching-Fixed-Filter",
                        bundle.matching,
                        _screening_config(study, "fixed"),
                    ),
                    (
                        "Matching-Adaptive-Filter",
                        bundle.matching,
                        _screening_config(study, "adaptive"),
                    ),
                    (
                        "Random-Adaptive-Filter",
                        bundle.random,
                        _screening_config(study, "adaptive"),
                    ),
                    (
                        "Wrong-Adaptive-Filter",
                        bundle.wrong,
                        _screening_config(study, "adaptive"),
                    ),
                    (
                        "Oracle-Fixed-Filter",
                        bundle.oracle,
                        _screening_config(study, "fixed"),
                    ),
                ]

                instance_details: Dict[str, Any] = {
                    "problem": problem,
                    "dim": dim,
                    "seed": seed,
                    "n_target_init": n_init,
                    "proposal_size": len(proposal_set.points),
                    "proposal_oracle_y": float(np.min(proposal_true_y)),
                    "methods": {},
                }

                for method, library, screening in method_specs:
                    decision = make_filter_decision(
                        proposal_set,
                        library,
                        bounds,
                        target_X,
                        target_y,
                        screening,
                    )
                    selected_index = int(decision.selected_indices[0])
                    metric = selection_metrics(
                        selected_index,
                        proposal_true_y,
                        proposal_set.acquisition_scores,
                    )
                    row = {
                        "problem": problem,
                        "dim": dim,
                        "seed": seed,
                        "method": method,
                        "selected_y": metric.selected_y,
                        "raw_regret": metric.raw_regret,
                        "normalized_regret": metric.normalized_regret,
                        "true_rank": metric.true_rank,
                        "acquisition_rank": metric.acquisition_rank,
                        "hit_top05": metric.hit_top05,
                        "hit_top10": metric.hit_top10,
                        "filter_active": float(decision.filter_active),
                        "retained_count": int(np.sum(decision.retained_mask)),
                        "retained_fraction": float(
                            np.mean(decision.retained_mask)
                        ),
                        "target_top1_retained": float(
                            decision.target_top1_retained
                        ),
                        "compatibility_trust": decision.compatibility.trust,
                        "compatibility_raw_evidence": (
                            decision.compatibility.raw_evidence
                        ),
                        "compatibility_spearman": (
                            decision.compatibility.spearman_correlation
                        ),
                        "compatibility_elite_enrichment": (
                            decision.compatibility.elite_enrichment
                        ),
                        "source_regions": len(library.regions),
                    }
                    records.append(row)
                    instance_details["methods"][method] = row

                soft_index, soft_source, soft_combined = soft_rerank_selection(
                    proposal_set,
                    bundle.matching,
                    source_weight=float(study["soft_rerank_weight"]),
                    source_aggregation=study["source_aggregation"],
                )
                soft_metric = selection_metrics(
                    soft_index,
                    proposal_true_y,
                    proposal_set.acquisition_scores,
                )
                soft_row = {
                    "problem": problem,
                    "dim": dim,
                    "seed": seed,
                    "method": "Matching-Soft-Rerank",
                    "selected_y": soft_metric.selected_y,
                    "raw_regret": soft_metric.raw_regret,
                    "normalized_regret": soft_metric.normalized_regret,
                    "true_rank": soft_metric.true_rank,
                    "acquisition_rank": soft_metric.acquisition_rank,
                    "hit_top05": soft_metric.hit_top05,
                    "hit_top10": soft_metric.hit_top10,
                    "filter_active": 0.0,
                    "retained_count": len(proposal_set.points),
                    "retained_fraction": 1.0,
                    "target_top1_retained": 1.0,
                    "compatibility_trust": None,
                    "compatibility_raw_evidence": None,
                    "compatibility_spearman": None,
                    "compatibility_elite_enrichment": None,
                    "source_regions": len(bundle.matching.regions),
                }
                records.append(soft_row)
                instance_details["methods"]["Matching-Soft-Rerank"] = soft_row
                instance_details["soft_source_score_range"] = [
                    float(np.min(soft_source)),
                    float(np.max(soft_source)),
                ]
                instance_details["soft_combined_score_range"] = [
                    float(np.min(soft_combined)),
                    float(np.max(soft_combined)),
                ]
                detailed.append(instance_details)

                if completed % max(1, int(study["progress_every"])) == 0:
                    print(f"Progress: {completed}/{total}")

    frame = pd.DataFrame(records)
    frame.to_csv(output / "screening_mechanism_summary.csv", index=False)
    with (output / "screening_mechanism_details.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(detailed, handle, indent=2, ensure_ascii=False)
    print(f"Saved mechanism results to {output}")
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
