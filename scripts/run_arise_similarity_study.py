"""Run ARISE-BO region-identification and equal-budget optimization studies."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from region_guided_reranking_study.arise_transfer import (  # noqa: E402
    ARISEConfig,
    ARISERegionTransferBO,
    counterfactual_region_gains,
)
from region_guided_reranking_study.landscapes import get_task_suite  # noqa: E402
from region_guided_reranking_study.screening_research import (  # noqa: E402
    build_source_library_bundle,
    known_optimum_value,
)
from region_guided_reranking_study.source_regions import (  # noqa: E402
    SourceRegion,
    SourceRegionLibrary,
)
from region_guided_reranking_study.target_region_screening import (  # noqa: E402
    TargetProposalConfig,
)


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Configuration root must be a JSON object.")
    return data


def clone_library(library: SourceRegionLibrary, prefix: str) -> SourceRegionLibrary:
    cloned = []
    for idx, region in enumerate(library.regions):
        original = region.source_task_id or f"region_{idx}"
        cloned.append(
            SourceRegion(
                center=region.center.copy(),
                cov=region.cov.copy(),
                quality=region.quality,
                count=region.count,
                source_task_id=f"{prefix}::{original}",
            )
        )
    return SourceRegionLibrary(cloned)


def combine_libraries(*libraries: SourceRegionLibrary) -> SourceRegionLibrary:
    regions = []
    for library in libraries:
        regions.extend(library.regions)
    return SourceRegionLibrary(regions)


def scenario_library(bundle, scenario: str) -> SourceRegionLibrary:
    matching = clone_library(bundle.matching, "matching")
    random = clone_library(bundle.random, "random")
    wrong = clone_library(bundle.wrong, "wrong")
    if scenario == "matching":
        return matching
    if scenario == "random":
        return random
    if scenario == "wrong":
        return wrong
    if scenario == "mixed":
        return combine_libraries(matching, random, wrong)
    raise ValueError(f"Unknown scenario: {scenario}")


def region_type(source_task_id: Optional[str]) -> str:
    text = source_task_id or "unknown"
    return text.split("::", 1)[0]


def normalized_regret(value: float, optimum: float, initial_best: float) -> float:
    denominator = max(abs(initial_best - optimum), 1e-12)
    return float((value - optimum) / denominator)


def normalized_auc(trace: Iterable[float], optimum: float, initial_best: float) -> float:
    values = np.asarray(list(trace), dtype=float)
    normalized = np.asarray(
        [normalized_regret(v, optimum, initial_best) for v in values],
        dtype=float,
    )
    return float(np.mean(normalized))


def make_optimizer(
    bounds: np.ndarray,
    policy: str,
    seed: int,
    cfg: Mapping,
) -> ARISERegionTransferBO:
    proposal = TargetProposalConfig(
        raw_pool_size=int(cfg["raw_pool_size"]),
        proposal_size=int(cfg["proposal_size"]),
        acquisition=str(cfg.get("acquisition", "ei")),
        ratio_acq=float(cfg.get("ratio_acq", 0.4)),
        ratio_global=float(cfg.get("ratio_global", 0.4)),
        ratio_diverse=float(cfg.get("ratio_diverse", 0.2)),
        proposal_min_distance=float(cfg.get("proposal_min_distance", 0.0)),
        gp_noise_level=float(cfg.get("gp_noise_level", 1e-4)),
    )
    config = ARISEConfig(
        policy=policy,
        top_ratio=float(cfg.get("top_ratio", 0.2)),
        max_clusters=int(cfg.get("max_clusters", 3)),
        min_samples_per_cluster=int(cfg.get("min_samples_per_cluster", 3)),
        merge_threshold=float(cfg.get("merge_threshold", 0.5)),
        proposal=proposal,
        prior_effect_variance=float(cfg.get("prior_effect_variance", 1.0)),
        intercept_prior_variance=float(cfg.get("intercept_prior_variance", 100.0)),
        credible_z=float(cfg.get("credible_z", 1.2815515655446004)),
        min_region_coverage=float(cfg.get("min_region_coverage", 0.75)),
        trust_effect_threshold=float(cfg.get("trust_effect_threshold", 0.0)),
        reject_effect_threshold=float(cfg.get("reject_effect_threshold", 0.0)),
        support_update_floor=float(cfg.get("support_update_floor", 1e-3)),
        improvement_xi=float(cfg.get("improvement_xi", 0.01)),
        residual_variance_floor=float(cfg.get("residual_variance_floor", 0.05)),
        residual_clip=float(cfg.get("residual_clip", 6.0)),
        evidence_decay=float(cfg.get("evidence_decay", 0.95)),
        guidance_weight=float(cfg.get("guidance_weight", 0.8)),
        fixed_guidance_weight=float(cfg.get("fixed_guidance_weight", 0.8)),
        probe_weight=float(cfg.get("probe_weight", 0.25)),
        exploit_acquisition_gate=float(cfg.get("exploit_acquisition_gate", 0.5)),
        probe_acquisition_gate=float(cfg.get("probe_acquisition_gate", 0.25)),
        region_candidate_weight=float(cfg.get("region_candidate_weight", 1.0)),
        active_probe=bool(cfg.get("active_probe", True)),
        probe_interval=int(cfg.get("probe_interval", 3)),
        probe_horizon=int(cfg.get("probe_horizon", 15)),
        probe_ucb_threshold=float(cfg.get("probe_ucb_threshold", 0.0)),
        global_elite_ratio=float(cfg.get("global_elite_ratio", 0.30)),
        global_prior_trust=float(cfg.get("global_prior_trust", 0.20)),
        global_prior_strength=float(cfg.get("global_prior_strength", 2.0)),
        global_evidence_shrinkage=float(cfg.get("global_evidence_shrinkage", 8.0)),
        global_activation_threshold=float(cfg.get("global_activation_threshold", 0.05)),
        random_state=int(seed),
    )
    return ARISERegionTransferBO(bounds, config)


def run_study(config: Mapping, output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)

    problems = list(config["problems"])
    dimensions = [int(v) for v in config["dimensions"]]
    seeds = [int(v) for v in config["seeds"]]
    scenarios = list(config["scenarios"])
    policies = list(config["policies"])
    source_samples = int(config["source_samples"])
    budget = int(config["budget"])

    summary_rows: List[Dict] = []
    trace_rows: List[Dict] = []
    identification_rows: List[Dict] = []

    total = len(problems) * len(dimensions) * len(seeds) * len(scenarios) * len(policies)
    completed = 0

    for dim in dimensions:
        n_init = int(config.get("n_init", 0)) or (2 * dim + 2)
        for problem in problems:
            for seed in seeds:
                seed_sequence = np.random.SeedSequence(seed)
                task_ss, match_ss, wrong_ss, random_ss, init_ss = seed_sequence.spawn(5)
                task_rng = np.random.default_rng(task_ss)
                match_rng = np.random.default_rng(match_ss)
                wrong_rng = np.random.default_rng(wrong_ss)
                random_rng = np.random.default_rng(random_ss)
                init_rng = np.random.default_rng(init_ss)

                suite = get_task_suite(dim=dim, rng=task_rng)[problem]
                bounds = np.asarray(suite["bounds"], dtype=float)
                target = suite["target"]
                optimum = known_optimum_value(target)
                bundle = build_source_library_bundle(
                    suite,
                    dim,
                    source_samples,
                    match_rng,
                    wrong_rng,
                    random_rng,
                    top_ratio=float(config.get("top_ratio", 0.2)),
                    max_clusters=int(config.get("max_clusters", 3)),
                    min_samples_per_cluster=int(config.get("min_samples_per_cluster", 3)),
                    merge_threshold=float(config.get("merge_threshold", 0.5)),
                    random_state=seed,
                )

                init_X = init_rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_init, dim))
                init_y = np.asarray(target(init_X), dtype=float)
                initial_best = float(np.min(init_y))

                for scenario in scenarios:
                    library = scenario_library(bundle, scenario)
                    for policy in policies:
                        optimizer = make_optimizer(bounds, policy, seed, config)
                        optimizer.set_source_region_library(library)
                        optimizer.initialize_target(init_X, init_y)

                        trace = [initial_best]
                        mode_counts = {"target": 0, "fixed": 0, "global": 0, "exploit": 0, "probe": 0}

                        for step in range(budget):
                            decision = optimizer.ask()
                            proposal_true_y = np.asarray(target(decision.proposal_set.points), dtype=float)
                            gains = counterfactual_region_gains(decision, proposal_true_y)

                            for posterior, gain, region_candidate_idx in zip(
                                decision.posteriors,
                                gains,
                                decision.region_candidate_indices,
                            ):
                                identification_rows.append(
                                    {
                                        "problem": problem,
                                        "dim": dim,
                                        "seed": seed,
                                        "scenario": scenario,
                                        "policy": policy,
                                        "step": step,
                                        "region_index": posterior.region_index,
                                        "region_type": region_type(posterior.source_task_id),
                                        "source_task_id": posterior.source_task_id,
                                        "posterior_mean": posterior.mean,
                                        "posterior_std": posterior.std,
                                        "lower_bound": posterior.lower_bound,
                                        "upper_bound": posterior.upper_bound,
                                        "probability_positive": posterior.probability_positive,
                                        "global_compatibility_trust": decision.global_compatibility_trust,
                                        "coverage": posterior.coverage,
                                        "status": posterior.status,
                                        "true_gain": float(gain),
                                        "true_useful": float(gain > 1e-12),
                                        "region_candidate_index": int(region_candidate_idx),
                                        "region_candidate_y": float(proposal_true_y[region_candidate_idx]),
                                        "target_top1_y": float(proposal_true_y[decision.target_top1_index]),
                                        "decision_mode": decision.mode,
                                        "selected_region_index": decision.selected_region_index,
                                    }
                                )

                            selected_y = np.asarray(target(decision.point), dtype=float)
                            residual = optimizer.tell(decision.point, selected_y)
                            best = min(trace[-1], float(selected_y[0]))
                            trace.append(best)
                            mode_counts[decision.mode] += 1

                            current_posteriors = optimizer.get_region_posteriors()
                            trusted_count = sum(p.status == "trusted" for p in current_posteriors)
                            rejected_count = sum(p.status == "rejected" for p in current_posteriors)
                            uncertain_count = sum(p.status == "uncertain" for p in current_posteriors)
                            trace_rows.append(
                                {
                                    "problem": problem,
                                    "dim": dim,
                                    "seed": seed,
                                    "scenario": scenario,
                                    "policy": policy,
                                    "step": step + 1,
                                    "best_y": best,
                                    "normalized_regret": normalized_regret(best, optimum, initial_best),
                                    "mode": decision.mode,
                                    "selected_target_rank": decision.selected_target_rank,
                                    "selected_region_index": decision.selected_region_index,
                                    "excess_improvement_residual": residual,
                                    "global_compatibility_trust": decision.global_compatibility_trust,
                                    "trusted_regions": trusted_count,
                                    "rejected_regions": rejected_count,
                                    "uncertain_regions": uncertain_count,
                                }
                            )

                        best_x, final_best = optimizer.get_best()
                        summary_rows.append(
                            {
                                "problem": problem,
                                "dim": dim,
                                "seed": seed,
                                "scenario": scenario,
                                "policy": policy,
                                "n_regions": len(library.regions),
                                "initial_best_y": initial_best,
                                "known_optimum_y": optimum,
                                "final_best_y": final_best,
                                "final_normalized_regret": normalized_regret(final_best, optimum, initial_best),
                                "normalized_regret_auc": normalized_auc(trace, optimum, initial_best),
                                "total_improvement": initial_best - final_best,
                                "target_steps": mode_counts["target"],
                                "fixed_steps": mode_counts["fixed"],
                                "global_steps": mode_counts["global"],
                                "exploit_steps": mode_counts["exploit"],
                                "probe_steps": mode_counts["probe"],
                                "final_trusted_regions": sum(
                                    p.status == "trusted" for p in optimizer.get_region_posteriors()
                                ),
                                "final_rejected_regions": sum(
                                    p.status == "rejected" for p in optimizer.get_region_posteriors()
                                ),
                            }
                        )

                        completed += 1
                        if completed % max(1, total // 20) == 0:
                            print(f"ARISE progress: {completed}/{total}")

    summary = pd.DataFrame(summary_rows)
    traces = pd.DataFrame(trace_rows)
    identification = pd.DataFrame(identification_rows)

    summary.to_csv(output_dir / "arise_optimizer_summary.csv", index=False)
    traces.to_csv(output_dir / "arise_optimizer_traces.csv", index=False)
    identification.to_csv(output_dir / "arise_region_identification.csv", index=False)

    with (output_dir / "arise_run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": dict(config),
                "optimizer_rows": len(summary),
                "trace_rows": len(traces),
                "identification_rows": len(identification),
            },
            handle,
            indent=2,
        )
    return summary, traces, identification


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "arise_quick.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "arise_stage",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_study(load_config(args.config), args.output)
