"""Phase B: equal-budget closed-loop optimization study.

The experiment compares target-only Bayesian optimization, fixed/adaptive source-region
filters, random/wrong source controls, and the existing soft-reranking baseline.
Every method starts from the same target observations and receives the same number of
expensive target evaluations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from region_guided_reranking_study.landscapes import get_task_suite
from region_guided_reranking_study.local_region_transfer import (
    LocalRegionTransferConfig,
    LocalRegionTransferOptimizer,
)
from region_guided_reranking_study.screening_research import (
    build_source_library_bundle,
    known_optimum_value,
    load_json,
)
from region_guided_reranking_study.source_regions import SourceRegionLibrary
from region_guided_reranking_study.target_region_screening import (
    RegionFilteredBOConfig,
    RegionFilteredTargetBO,
    RegionScreeningConfig,
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


def _optimizer_config(
    study: Dict[str, Any],
    policy: str,
    seed: int,
) -> RegionFilteredBOConfig:
    return RegionFilteredBOConfig(
        top_ratio=float(study["top_ratio"]),
        max_clusters=int(study["max_clusters"]),
        min_samples_per_cluster=int(study["min_samples_per_cluster"]),
        merge_threshold=float(study["merge_threshold"]),
        proposal=_proposal_config(study),
        screening=_screening_config(study, policy),
        random_state=seed,
    )


def _normalized_regret(
    best_value: float,
    initial_best: float,
    optimum: float,
) -> float:
    scale = max(1e-12, initial_best - optimum)
    return float(max(0.0, best_value - optimum) / scale)


def _run_filtered_method(
    method: str,
    library: SourceRegionLibrary,
    policy: str,
    bounds: np.ndarray,
    target_function,
    init_X: np.ndarray,
    init_y: np.ndarray,
    budget: int,
    study: Dict[str, Any],
    seed: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    optimizer = RegionFilteredTargetBO(
        bounds,
        _optimizer_config(study, policy, seed),
    )
    optimizer.set_source_region_library(library)
    result = optimizer.optimize(
        target_function,
        init_X=init_X,
        init_y=init_y,
        budget=budget,
        batch_size=int(study["batch_size"]),
    )

    optimum = known_optimum_value(target_function)
    initial_best = float(np.min(init_y))
    summary = {
        "method": method,
        "initial_best_y": initial_best,
        "final_best_y": result.best_y,
        "known_optimum_y": optimum,
        "total_improvement": initial_best - result.best_y,
        "final_normalized_regret": _normalized_regret(
            result.best_y,
            initial_best,
            optimum,
        ),
        "n_regions": len(library.regions),
        "filter_activation_rate": float(
            np.mean([decision.filter_active for decision in result.decisions])
        )
        if result.decisions
        else 0.0,
        "mean_retained_fraction": float(
            np.mean(
                [decision.effective_retain_ratio for decision in result.decisions]
            )
        )
        if result.decisions
        else 1.0,
        "mean_compatibility_trust": float(
            np.mean(
                [decision.compatibility.trust for decision in result.decisions]
            )
        )
        if result.decisions
        else 0.0,
    }

    traces: List[Dict[str, Any]] = []
    for step, best_value in enumerate(result.best_y_trace):
        decision_idx = min(step, len(result.decisions)) - 1
        if decision_idx >= 0:
            decision = result.decisions[decision_idx]
            filter_active = float(decision.filter_active)
            retained_fraction = decision.effective_retain_ratio
            trust = decision.compatibility.trust
        else:
            filter_active = 0.0
            retained_fraction = 1.0
            trust = 0.0
        traces.append(
            {
                "method": method,
                "step": step,
                "best_y": float(best_value),
                "normalized_regret": _normalized_regret(
                    float(best_value),
                    initial_best,
                    optimum,
                ),
                "filter_active": filter_active,
                "retained_fraction": retained_fraction,
                "compatibility_trust": trust,
            }
        )
    return summary, traces


def _run_soft_rerank_method(
    library: SourceRegionLibrary,
    bounds: np.ndarray,
    target_function,
    init_X: np.ndarray,
    init_y: np.ndarray,
    budget: int,
    study: Dict[str, Any],
    seed: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    optimizer = LocalRegionTransferOptimizer(
        bounds,
        LocalRegionTransferConfig(
            top_ratio=float(study["top_ratio"]),
            max_clusters=int(study["max_clusters"]),
            min_samples_per_cluster=int(study["min_samples_per_cluster"]),
            merge_threshold=float(study["merge_threshold"]),
            pool_size=int(study["raw_pool_size"]),
            acquisition=study["acquisition"],
            source_weight=float(study["soft_rerank_weight"]),
            source_weight_decay=float(study["soft_rerank_decay"]),
            target_nomination_ratio=float(study["soft_target_nomination_ratio"]),
            source_nomination_ratio=float(study["soft_source_nomination_ratio"]),
            source_aggregation=study["source_aggregation"],
            ratio_acq=float(study["ratio_acq"]),
            ratio_global=float(study["ratio_global"]),
            ratio_diverse=float(study["ratio_diverse"]),
            gp_noise_level=float(study["gp_noise_level"]),
            random_state=seed,
        ),
    )
    optimizer.set_source_region_library(library)
    result = optimizer.optimize(
        target_function,
        init_X=init_X,
        init_y=init_y,
        budget=budget,
        batch_size=int(study["batch_size"]),
    )

    optimum = known_optimum_value(target_function)
    initial_best = float(np.min(init_y))
    summary = {
        "method": "Matching-Soft-Rerank",
        "initial_best_y": initial_best,
        "final_best_y": result.best_y,
        "known_optimum_y": optimum,
        "total_improvement": initial_best - result.best_y,
        "final_normalized_regret": _normalized_regret(
            result.best_y,
            initial_best,
            optimum,
        ),
        "n_regions": len(library.regions),
        "filter_activation_rate": 0.0,
        "mean_retained_fraction": 1.0,
        "mean_compatibility_trust": float("nan"),
    }
    traces = [
        {
            "method": "Matching-Soft-Rerank",
            "step": step,
            "best_y": float(best_value),
            "normalized_regret": _normalized_regret(
                float(best_value),
                initial_best,
                optimum,
            ),
            "filter_active": 0.0,
            "retained_fraction": 1.0,
            "compatibility_trust": float("nan"),
        }
        for step, best_value in enumerate(result.best_y_trace)
    ]
    return summary, traces


def run_study(config_path: str, output_dir: str | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    config = load_json(config_path)
    study = config["study"]
    output = Path(output_dir or config["output_dir"]) / "sequential"
    output.mkdir(parents=True, exist_ok=True)

    summary_records: List[Dict[str, Any]] = []
    trace_records: List[Dict[str, Any]] = []
    problems = list(study["problems"])
    dimensions = [int(value) for value in study["dimensions"]]
    seeds = [int(value) for value in study["seeds"]]
    budget = int(study["budget"])

    total = len(problems) * len(dimensions) * len(seeds)
    completed = 0
    print(f"Starting equal-budget sequential screening study: {total} instances")

    for dim in dimensions:
        n_init = int(study["n_init_multiplier"]) * dim + int(
            study["n_init_offset"]
        )
        for problem in problems:
            for seed in seeds:
                completed += 1
                streams = np.random.SeedSequence(seed).spawn(6)
                task_rng = np.random.default_rng(streams[0])
                match_rng = np.random.default_rng(streams[1])
                wrong_rng = np.random.default_rng(streams[2])
                random_rng = np.random.default_rng(streams[3])
                init_rng = np.random.default_rng(streams[4])

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

                init_X = init_rng.uniform(
                    bounds[:, 0], bounds[:, 1], size=(n_init, dim)
                )
                init_y = np.asarray(target_function(init_X), dtype=float)

                method_specs = [
                    ("Target-Only", SourceRegionLibrary(), "none"),
                    ("Matching-Fixed-Filter", bundle.matching, "fixed"),
                    ("Matching-Adaptive-Filter", bundle.matching, "adaptive"),
                    ("Random-Adaptive-Filter", bundle.random, "adaptive"),
                    ("Wrong-Adaptive-Filter", bundle.wrong, "adaptive"),
                    ("Oracle-Fixed-Filter", bundle.oracle, "fixed"),
                ]

                for method, library, policy in method_specs:
                    # Common random numbers: every filtered method starts from the
                    # same proposal RNG state. After methods choose different points,
                    # the same raw random draws are still used at each iteration.
                    method_seed = seed
                    summary, traces = _run_filtered_method(
                        method,
                        library,
                        policy,
                        bounds,
                        target_function,
                        init_X,
                        init_y,
                        budget,
                        study,
                        method_seed,
                    )
                    summary.update(
                        {"problem": problem, "dim": dim, "seed": seed}
                    )
                    summary_records.append(summary)
                    for row in traces:
                        row.update(
                            {"problem": problem, "dim": dim, "seed": seed}
                        )
                        trace_records.append(row)

                soft_summary, soft_traces = _run_soft_rerank_method(
                    bundle.matching,
                    bounds,
                    target_function,
                    init_X,
                    init_y,
                    budget,
                    study,
                    seed,
                )
                soft_summary.update(
                    {"problem": problem, "dim": dim, "seed": seed}
                )
                summary_records.append(soft_summary)
                for row in soft_traces:
                    row.update({"problem": problem, "dim": dim, "seed": seed})
                    trace_records.append(row)

                if completed % max(1, int(study["progress_every"])) == 0:
                    print(f"Progress: {completed}/{total}")

    summary_frame = pd.DataFrame(summary_records)
    trace_frame = pd.DataFrame(trace_records)
    summary_frame.to_csv(
        output / "screening_sequential_summary.csv", index=False
    )
    trace_frame.to_csv(output / "screening_sequential_traces.csv", index=False)
    print(f"Saved sequential results to {output}")
    return summary_frame, trace_frame


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
