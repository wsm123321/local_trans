"""
Full Verification Experiment Runner.
Executes multi-seed, multi-problem, multi-dimension controlled verification of:
"Source Local Region-Guided Candidate Reranking"
"""

import os
import sys
import json
from typing import Dict, List, Any
import numpy as np
import pandas as pd

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, os.path.dirname(current_dir))

from region_guided_reranking_study.landscapes import (
    get_task_suite, GaussianMixtureLandscape, ShiftedRotatedRastrigin,
    LunacekBiRastrigin, ShiftedAckley
)
from region_guided_reranking_study.source_regions import (
    SourceRegionExtractor, SourceRegionLibrary, 
    create_random_region_library, create_oracle_region_library
)
from region_guided_reranking_study.surrogate_and_candidates import (
    TargetGPSurrogate, CandidatePoolGenerator
)
from region_guided_reranking_study.rerankers import create_comparator_suite
from region_guided_reranking_study.metrics import (
    compute_partial_correlation, compute_incremental_r2, evaluate_candidate_selection
)


def run_single_experiment(prob_name: str, dim: int, seed: int, 
                           n_init: int = 8, pool_size: int = 1000,
                           lambda_weight: float = 1.0) -> Dict[str, Any]:
    """
    Run one strictly paired controlled experiment instance.
    All methods share the exact same initial target data, surrogate, candidate pool, and seed.
    """
    rng = np.random.RandomState(seed)
    
    # 1. Setup problem suite
    suite = get_task_suite(dim=dim, seed=seed)[prob_name]
    target_func = suite["target"]
    matching_sources = suite["matching_sources"]
    mismatched_sources = suite["mismatched_sources"]
    bounds = suite["bounds"]
    
    # 2. Simulate offline source tasks & extract region libraries
    extractor = SourceRegionExtractor(top_ratio=0.20, max_clusters=3, random_state=seed)
    
    # Matching source library
    matching_datasets = []
    for i, src_func in enumerate(matching_sources):
        src_X = rng.uniform(bounds[:, 0], bounds[:, 1], size=(50, dim))
        src_y = src_func(src_X)
        matching_datasets.append((src_X, src_y))
    matching_lib = extractor.extract_from_multi_sources(matching_datasets, task_ids=[f"match_{i}" for i in range(len(matching_sources))])
    
    # Mismatched source library
    mismatched_datasets = []
    for i, src_func in enumerate(mismatched_sources):
        src_X = rng.uniform(bounds[:, 0], bounds[:, 1], size=(50, dim))
        src_y = src_func(src_X)
        mismatched_datasets.append((src_X, src_y))
    wrong_lib = extractor.extract_from_multi_sources(mismatched_datasets, task_ids=[f"wrong_{i}" for i in range(len(mismatched_sources))])
    
    # Random library & Oracle library
    random_lib = create_random_region_library(dim=dim, bounds=bounds, num_regions=3, seed=seed+555)
    oracle_basins = target_func.get_oracle_basins()
    oracle_lib = create_oracle_region_library(oracle_basins, dim=dim)
    
    # 3. Sample initial target data D_t (few-shot early stage)
    target_init_X = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_init, dim))
    target_init_y = target_func(target_init_X)
    y_init_best = float(np.min(target_init_y))
    
    # 4. Train Target GP surrogate
    surrogate = TargetGPSurrogate(dim=dim, random_state=seed)
    surrogate.fit(target_init_X, target_init_y)
    
    # 5. Generate shared candidate pool C_t
    pool_gen = CandidatePoolGenerator(bounds=bounds, pool_size=pool_size, random_state=seed)
    candidates = pool_gen.generate(surrogate=surrogate, current_X=target_init_X)
    
    # Compute ground truth utility on all candidates
    true_y_pool = target_func(candidates)
    true_utility = -true_y_pool  # higher utility = lower cost
    oracle_pool_best_y = float(np.min(true_y_pool))
    
    # Target acquisition scores on candidate pool
    acq_scores = surrogate.compute_acquisition(candidates, acq_type="ei")
    
    # 6. Statistical conditional tests (Hypothesis I(U_t; r_s | alpha_t) > 0)
    matching_r_scores = matching_lib.score(candidates)
    partial_corr_match, p_val_match = compute_partial_correlation(true_utility, matching_r_scores, acq_scores)
    inc_r2_match = compute_incremental_r2(true_utility, acq_scores, matching_r_scores)
    
    wrong_r_scores = wrong_lib.score(candidates)
    partial_corr_wrong, p_val_wrong = compute_partial_correlation(true_utility, wrong_r_scores, acq_scores)
    inc_r2_wrong = compute_incremental_r2(true_utility, acq_scores, wrong_r_scores)

    random_r_scores = random_lib.score(candidates)
    partial_corr_rand, p_val_rand = compute_partial_correlation(true_utility, random_r_scores, acq_scores)
    inc_r2_rand = compute_incremental_r2(true_utility, acq_scores, random_r_scores)

    # 7. Run 6 controlled comparators on the exact same pool
    comparators = create_comparator_suite(
        matching_lib=matching_lib,
        random_lib=random_lib,
        wrong_lib=wrong_lib,
        oracle_lib=oracle_lib,
        weight_lambda=lambda_weight
    )
    
    y_global_min = oracle_pool_best_y  # proxy global min in pool
    
    comparator_results = {}
    for name, reranker in comparators.items():
        ranked_idx, comb_scores = reranker.score_and_rank(candidates, acq_scores)
        metrics = evaluate_candidate_selection(
            ranked_indices=ranked_idx,
            true_y_pool=true_y_pool,
            y_init_best=y_init_best,
            y_global_min=y_global_min,
            top_k_list=(1, 3, 5)
        )
        comparator_results[name] = metrics
        
    return {
        "problem": prob_name,
        "dim": dim,
        "seed": seed,
        "n_init": n_init,
        "pool_size": pool_size,
        "y_init_best": y_init_best,
        "oracle_pool_best": oracle_pool_best_y,
        "stats": {
            "matching": {
                "partial_corr": partial_corr_match,
                "p_val": p_val_match,
                "delta_r2": inc_r2_match["delta_r2"],
                "r2_full": inc_r2_match["r2_with_source"]
            },
            "random": {
                "partial_corr": partial_corr_rand,
                "p_val": p_val_rand,
                "delta_r2": inc_r2_rand["delta_r2"],
                "r2_full": inc_r2_rand["r2_with_source"]
            },
            "wrong": {
                "partial_corr": partial_corr_wrong,
                "p_val": p_val_wrong,
                "delta_r2": inc_r2_wrong["delta_r2"],
                "r2_full": inc_r2_wrong["r2_with_source"]
            },
            "r2_target_only": inc_r2_match["r2_target_only"]
        },
        "comparators": comparator_results
    }


def run_full_suite(output_dir: str):
    """Run full verification experiment across problems, dimensions, and seeds."""
    os.makedirs(output_dir, exist_ok=True)
    
    problems = ["GMM", "Rastrigin", "Lunacek", "Ackley"]
    dimensions = [2, 5]
    seeds = [42, 101, 2026, 777, 999]
    
    records = []
    detailed_results = []
    
    total_runs = len(problems) * len(dimensions) * len(seeds)
    run_count = 0
    
    print(f"Starting full verification suite ({total_runs} total runs)...")
    
    for dim in dimensions:
        n_init = 2 * dim + 2
        for prob in problems:
            for seed in seeds:
                run_count += 1
                res = run_single_experiment(
                    prob_name=prob,
                    dim=dim,
                    seed=seed,
                    n_init=n_init,
                    pool_size=1000,
                    lambda_weight=1.0
                )
                detailed_results.append(res)
                
                # Flatten into summary records
                base_info = {
                    "problem": prob,
                    "dim": dim,
                    "seed": seed,
                    "n_init": n_init,
                    "target_only_r2": res["stats"]["r2_target_only"],
                    "match_partial_corr": res["stats"]["matching"]["partial_corr"],
                    "match_delta_r2": res["stats"]["matching"]["delta_r2"],
                    "rand_partial_corr": res["stats"]["random"]["partial_corr"],
                    "rand_delta_r2": res["stats"]["random"]["delta_r2"],
                    "wrong_partial_corr": res["stats"]["wrong"]["partial_corr"],
                    "wrong_delta_r2": res["stats"]["wrong"]["delta_r2"],
                }
                
                for comp_name, comp_m in res["comparators"].items():
                    row = {**base_info}
                    row["method"] = comp_name
                    row["top1_true_y"] = comp_m["top1_true_y"]
                    row["top1_regret_pool"] = comp_m["top1_regret_pool"]
                    row["top1_improvement"] = comp_m["top1_improvement"]
                    row["top1_hit_top05"] = comp_m["top1_hit_top05"]
                    row["top1_hit_top10"] = comp_m["top1_hit_top10"]
                    row["rank_of_pool_oracle"] = comp_m["rank_of_pool_oracle"]
                    row["top3_mean_regret_pool"] = comp_m["top3_mean_regret_pool"]
                    records.append(row)
                    
                if run_count % 5 == 0:
                    print(f"Progress: {run_count}/{total_runs} runs completed...")

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "experiment_summary.csv")
    df.to_csv(csv_path, index=False)
    
    json_path = os.path.join(output_dir, "detailed_experiment_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2)
        
    print(f"Saved summary CSV to {csv_path}")
    print(f"Saved detailed JSON to {json_path}")
    return df, detailed_results


if __name__ == "__main__":
    out_dir = os.path.join(current_dir, "results")
    run_full_suite(out_dir)
