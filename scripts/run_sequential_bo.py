"""
Phase 3: Multi-Step Closed-Loop Sequential Bayesian Optimization Experiment.
Executes iterative BO where each strategy evaluates its chosen candidate,
updates its own surrogate, and continues for a fixed evaluation budget.
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
from region_guided_reranking_study.rerankers import create_comparator_suite
from region_guided_reranking_study.sequential_bo import SequentialBOEngine


def run_sequential_bo_suite(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    problems = ["GMM", "Rastrigin", "Lunacek", "Ackley"]
    dimensions = [2, 5]
    seeds = [42, 101, 2026]
    budget = 10
    
    records = []
    trace_records = []
    
    total_runs = len(problems) * len(dimensions) * len(seeds)
    run_count = 0
    print(f"Starting Phase 3 Sequential BO Suite ({total_runs} instances x 6 methods)...")
    
    for dim in dimensions:
        n_init = 2 * dim + 2
        for prob in problems:
            for seed in seeds:
                run_count += 1
                seed_seq = np.random.SeedSequence(seed)
                task_ss, match_ss, wrong_ss, init_ss, pool_ss = seed_seq.spawn(5)
                
                task_rng = np.random.default_rng(task_ss)
                match_rng = np.random.default_rng(match_ss)
                wrong_rng = np.random.default_rng(wrong_ss)
                init_rng = np.random.default_rng(init_ss)
                pool_rng = np.random.default_rng(pool_ss)
                
                suite = get_task_suite(dim=dim, rng=task_rng)[prob]
                target_func = suite["target"]
                matching_sources = suite["matching_sources"]
                mismatched_sources = suite["mismatched_sources"]
                bounds = suite["bounds"]
                
                # Extract libraries
                extractor = SourceRegionExtractor(top_ratio=0.20, max_clusters=3, random_state=int(seed % 10000))
                matching_datasets = []
                for src in matching_sources:
                    s_X = match_rng.uniform(bounds[:, 0], bounds[:, 1], size=(50, dim))
                    matching_datasets.append((s_X, src(s_X)))
                matching_lib = extractor.extract_from_multi_sources(matching_datasets)
                
                mismatched_datasets = []
                for src in mismatched_sources:
                    s_X = wrong_rng.uniform(bounds[:, 0], bounds[:, 1], size=(50, dim))
                    mismatched_datasets.append((s_X, src(s_X)))
                wrong_lib = extractor.extract_from_multi_sources(mismatched_datasets)
                assert len(wrong_lib.regions) > 0, "Wrong-source library must not be empty"
                
                random_lib = create_structure_matched_random_library(matching_lib, bounds=bounds, rng=pool_rng)
                oracle_basins = target_func.get_oracle_basins()
                oracle_lib = create_true_oracle_library(oracle_basins, dim=dim)
                
                # Shared initial target sample
                init_X = init_rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_init, dim))
                init_y = target_func(init_X)
                
                comparators = create_comparator_suite(
                    matching_lib=matching_lib,
                    random_lib=random_lib,
                    wrong_lib=wrong_lib,
                    oracle_lib=oracle_lib,
                    weight_lambda=1.0
                )
                
                for method_name, reranker in comparators.items():
                    engine = SequentialBOEngine(
                        target_func=target_func,
                        bounds=bounds,
                        reranker=reranker,
                        pool_size=200,
                        lambda_decay=0.08,
                        rng=np.random.default_rng(int(seed % 10000))
                    )
                    res = engine.optimize(init_X=init_X, init_y=init_y, budget=budget)
                    
                    records.append({
                        "problem": prob,
                        "dim": dim,
                        "seed": seed,
                        "method": method_name,
                        "initial_best_y": float(np.min(init_y)),
                        "final_best_y": res["final_best_y"],
                        "total_improvement": float(np.min(init_y) - res["final_best_y"]),
                    })
                    
                    # Record per-step trace
                    for step_idx, val in enumerate(res["best_y_trace"]):
                        trace_records.append({
                            "problem": prob,
                            "dim": dim,
                            "seed": seed,
                            "method": method_name,
                            "step": step_idx,
                            "best_y": val,
                        })
                        
                if run_count % 5 == 0:
                    print(f"Sequential BO Progress: {run_count}/{total_runs} runs completed...")

    df_summary = pd.DataFrame(records)
    df_traces = pd.DataFrame(trace_records)
    
    df_summary.to_csv(os.path.join(output_dir, "sequential_bo_summary.csv"), index=False)
    df_traces.to_csv(os.path.join(output_dir, "sequential_bo_traces.csv"), index=False)
    print("Saved Sequential BO Results.")
    return df_summary, df_traces


if __name__ == "__main__":
    repo_root = os.path.dirname(current_dir)
    out_dir = os.path.join(repo_root, "results")
    run_sequential_bo_suite(out_dir)
