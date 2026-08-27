"""
Phase 1: Controlled Mechanism Experiment Runner.
Executes single-step candidate reranking under strict random stream isolation,
out-of-sample CV delta R^2, permutation testing, scale-free normalized regret,
and structure-matched controls.
"""

import os
import sys
import json
from typing import Dict, List, Any
import numpy as np
import pandas as pd

# Add src to pythonpath
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(current_dir), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from region_guided_reranking_study.landscapes import get_task_suite
from region_guided_reranking_study.source_regions import (
    SourceRegionExtractor, SourceRegionLibrary, 
    create_structure_matched_random_library, create_true_oracle_library
)
from region_guided_reranking_study.surrogate_and_candidates import (
    TargetGPSurrogate, CandidatePoolGenerator
)
from region_guided_reranking_study.rerankers import create_comparator_suite
from region_guided_reranking_study.metrics import (
    compute_partial_correlation, compute_out_of_sample_incremental_r2,
    evaluate_candidate_selection
)


def run_single_mechanism_instance(prob_name: str, dim: int, seed: int, 
                                   n_init: int = 8, pool_size: int = 1000,
                                   lambda_weight: float = 1.0) -> Dict[str, Any]:
    """
    Run one strictly isolated instance of the candidate reranking experiment.
    """
    # 1. Independent Random Streams via SeedSequence
    seed_seq = np.random.SeedSequence(seed)
    task_ss, match_ss, wrong_ss, init_ss, pool_ss, perm_ss = seed_seq.spawn(6)
    
    task_rng = np.random.default_rng(task_ss)
    match_rng = np.random.default_rng(match_ss)
    wrong_rng = np.random.default_rng(wrong_ss)
    init_rng = np.random.default_rng(init_ss)
    pool_rng = np.random.default_rng(pool_ss)
    perm_rng = np.random.default_rng(perm_ss)
    
    # 2. Setup problem suite
    suite = get_task_suite(dim=dim, rng=task_rng)[prob_name]
    target_func = suite["target"]
    matching_sources = suite["matching_sources"]
    mismatched_sources = suite["mismatched_sources"]
    bounds = suite["bounds"]
    
    # 3. Simulate offline source tasks & extract regions
    extractor = SourceRegionExtractor(top_ratio=0.20, max_clusters=3, random_state=int(seed % 10000))
    
    matching_datasets = []
    for i, src_func in enumerate(matching_sources):
        src_X = match_rng.uniform(bounds[:, 0], bounds[:, 1], size=(50, dim))
        src_y = src_func(src_X)
        matching_datasets.append((src_X, src_y))
    matching_lib = extractor.extract_from_multi_sources(
        matching_datasets, task_ids=[f"match_{i}" for i in range(len(matching_sources))]
    )
    
    mismatched_datasets = []
    for i, src_func in enumerate(mismatched_sources):
        src_X = wrong_rng.uniform(bounds[:, 0], bounds[:, 1], size=(50, dim))
        src_y = src_func(src_X)
        mismatched_datasets.append((src_X, src_y))
    wrong_lib = extractor.extract_from_multi_sources(
        mismatched_datasets, task_ids=[f"wrong_{i}" for i in range(len(mismatched_sources))]
    )
    
    # Strict structure-matched random library
    random_lib = create_structure_matched_random_library(matching_lib, bounds=bounds, rng=pool_rng)
    
    # True Oracle library (targets true global basin)
    oracle_basins = target_func.get_oracle_basins()
    oracle_lib = create_true_oracle_library(oracle_basins, dim=dim)
    
    # 4. Sample initial target data D_t (few-shot early stage)
    target_init_X = init_rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_init, dim))
    target_init_y = target_func(target_init_X)
    y_init_best = float(np.min(target_init_y))
    
    # 5. Train Target GP surrogate
    surrogate = TargetGPSurrogate(dim=dim, random_state=int(seed % 10000))
    surrogate.fit(target_init_X, target_init_y)
    
    # 6. Generate candidate pool C_t with strict exclusion of evaluated points
    pool_gen = CandidatePoolGenerator(bounds=bounds, pool_size=pool_size, rng=pool_rng)
    all_source_X = [ds[0] for ds in matching_datasets + mismatched_datasets]
    candidates = pool_gen.generate(
        surrogate=surrogate, 
        current_X=target_init_X,
        excluded_datasets=all_source_X
    )
    
    true_y_pool = target_func(candidates)
    true_utility = -true_y_pool
    oracle_pool_best_y = float(np.min(true_y_pool))
    
    acq_scores = surrogate.compute_acquisition(candidates, acq_type="ei")
    
    # 7. Statistical conditional tests (out-of-sample Delta R^2 and partial correlation)
    match_r_scores = matching_lib.score(candidates)
    partial_corr_match, p_val_match = compute_partial_correlation(true_utility, match_r_scores, acq_scores)
    oos_r2_match = compute_out_of_sample_incremental_r2(true_utility, acq_scores, match_r_scores, rng=perm_rng)
    
    rand_r_scores = random_lib.score(candidates)
    partial_corr_rand, p_val_rand = compute_partial_correlation(true_utility, rand_r_scores, acq_scores)
    oos_r2_rand = compute_out_of_sample_incremental_r2(true_utility, acq_scores, rand_r_scores, rng=perm_rng)
    
    wrong_r_scores = wrong_lib.score(candidates)
    partial_corr_wrong, p_val_wrong = compute_partial_correlation(true_utility, wrong_r_scores, acq_scores)
    oos_r2_wrong = compute_out_of_sample_incremental_r2(true_utility, acq_scores, wrong_r_scores, rng=perm_rng)
    
    # 8. Run 6 controlled comparators on the exact same pool
    comparators = create_comparator_suite(
        matching_lib=matching_lib,
        random_lib=random_lib,
        wrong_lib=wrong_lib,
        oracle_lib=oracle_lib,
        weight_lambda=lambda_weight
    )
    
    comparator_results = {}
    for name, reranker in comparators.items():
        ranked_idx, comb_scores = reranker.score_and_rank(candidates, acq_scores)
        metrics = evaluate_candidate_selection(
            ranked_indices=ranked_idx,
            true_y_pool=true_y_pool,
            y_init_best=y_init_best
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
                "p_val_corr": p_val_match,
                "delta_r2_oos": oos_r2_match["delta_r2_oos"],
                "perm_p_val": oos_r2_match["permutation_p_val"],
                "r2_oos_full": oos_r2_match["r2_oos_full"]
            },
            "random": {
                "partial_corr": partial_corr_rand,
                "p_val_corr": p_val_rand,
                "delta_r2_oos": oos_r2_rand["delta_r2_oos"],
                "perm_p_val": oos_r2_rand["permutation_p_val"],
                "r2_oos_full": oos_r2_rand["r2_oos_full"]
            },
            "wrong": {
                "partial_corr": partial_corr_wrong,
                "p_val_corr": p_val_wrong,
                "delta_r2_oos": oos_r2_wrong["delta_r2_oos"],
                "perm_p_val": oos_r2_wrong["permutation_p_val"],
                "r2_oos_full": oos_r2_wrong["r2_oos_full"]
            },
            "r2_oos_base": oos_r2_match["r2_oos_base"]
        },
        "comparators": comparator_results
    }


def run_mechanism_suite(output_dir: str):
    """Run full verification across problems, dimensions, and seeds."""
    os.makedirs(output_dir, exist_ok=True)
    
    problems = ["GMM", "Rastrigin", "Lunacek", "Ackley"]
    dimensions = [2, 5]
    seeds = [42, 101, 2026, 777, 999, 1234, 5678, 8888]
    
    records = []
    detailed_results = []
    
    total_runs = len(problems) * len(dimensions) * len(seeds)
    run_count = 0
    print(f"Starting Phase 1 Mechanism Verification Suite ({total_runs} total runs)...")
    
    for dim in dimensions:
        n_init = 2 * dim + 2
        for prob in problems:
            for seed in seeds:
                run_count += 1
                res = run_single_mechanism_instance(
                    prob_name=prob,
                    dim=dim,
                    seed=seed,
                    n_init=n_init,
                    pool_size=1000,
                    lambda_weight=1.0
                )
                detailed_results.append(res)
                
                base_info = {
                    "problem": prob,
                    "dim": dim,
                    "seed": seed,
                    "n_init": n_init,
                    "target_only_r2_oos": res["stats"]["r2_oos_base"],
                    "match_partial_corr": res["stats"]["matching"]["partial_corr"],
                    "match_corr_pval": res["stats"]["matching"]["p_val_corr"],
                    "match_delta_r2_oos": res["stats"]["matching"]["delta_r2_oos"],
                    "match_perm_pval": res["stats"]["matching"]["perm_p_val"],
                    "rand_partial_corr": res["stats"]["random"]["partial_corr"],
                    "rand_delta_r2_oos": res["stats"]["random"]["delta_r2_oos"],
                    "wrong_partial_corr": res["stats"]["wrong"]["partial_corr"],
                    "wrong_delta_r2_oos": res["stats"]["wrong"]["delta_r2_oos"],
                }
                
                for comp_name, comp_m in res["comparators"].items():
                    row = {**base_info}
                    row["method"] = comp_name
                    row["top1_true_y"] = comp_m["top1_true_y"]
                    row["top1_raw_regret"] = comp_m["top1_raw_regret"]
                    row["top1_normalized_regret"] = comp_m["top1_normalized_regret"]
                    row["top1_signed_improvement"] = comp_m["top1_signed_improvement"]
                    row["top1_positive_improvement"] = comp_m["top1_positive_improvement"]
                    row["is_improved"] = comp_m["is_improved"]
                    row["top1_hit_top05"] = comp_m["top1_hit_top05"]
                    row["top1_hit_top10"] = comp_m["top1_hit_top10"]
                    row["rank_of_pool_oracle"] = comp_m["rank_of_pool_oracle"]
                    records.append(row)
                    
                if run_count % 8 == 0:
                    print(f"Progress: {run_count}/{total_runs} runs completed...")

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "mechanism_experiment_summary.csv")
    df.to_csv(csv_path, index=False)
    
    json_path = os.path.join(output_dir, "detailed_mechanism_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2)
        
    print(f"Saved Phase 1 Summary CSV to {csv_path}")
    return df, detailed_results


if __name__ == "__main__":
    repo_root = os.path.dirname(current_dir)
    out_dir = os.path.join(repo_root, "results")
    run_mechanism_suite(out_dir)
