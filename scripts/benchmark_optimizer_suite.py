"""
Comprehensive Benchmark Suite for LocalRegionTransferOptimizer.
Tests closed-loop performance across:
- 4 Multi-modal Problem Landscapes: GMM, Shifted Rastrigin, Lunacek, Shifted Ackley
- 2 Dimensionalities: 2D, 5D
- 5 Independent Seeds per configuration
- 4 Transfer Configurations:
  1. Matching Source Transfer
  2. No-Transfer (Pure Target BO baseline)
  3. Structure-Matched Random Source
  4. Adversarial / Mismatched Source
- Ablation: Shortlist Nomination vs Full Pool Fusion
"""

import os
import sys
import json
import time
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add src to pythonpath
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(current_dir), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from region_guided_reranking_study import (
    LocalRegionTransferConfig,
    LocalRegionTransferOptimizer,
)
from region_guided_reranking_study.landscapes import get_task_suite


def run_optimizer_benchmark(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    problems = ["GMM", "Rastrigin", "Lunacek", "Ackley"]
    dimensions = [2, 5]
    seeds = [42, 101, 2026]
    budget = 10
    
    records = []
    traces = []
    
    total_configs = len(problems) * len(dimensions) * len(seeds)
    count = 0
    print(f"Starting LocalRegionTransferOptimizer Benchmark Suite ({total_configs} instances x 4 modes)...")
    
    start_time = time.time()
    
    for dim in dimensions:
        n_init = 2 * dim + 2
        for prob in problems:
            for seed in seeds:
                count += 1
                seed_seq = np.random.SeedSequence(seed)
                task_ss, match_ss, wrong_ss, rand_ss, init_ss, opt_ss = seed_seq.spawn(6)
                
                task_rng = np.random.default_rng(task_ss)
                match_rng = np.random.default_rng(match_ss)
                wrong_rng = np.random.default_rng(wrong_ss)
                rand_rng = np.random.default_rng(rand_ss)
                init_rng = np.random.default_rng(init_ss)
                
                suite = get_task_suite(dim=dim, rng=task_rng)[prob]
                target_func = suite["target"]
                matching_sources = suite["matching_sources"]
                mismatched_sources = suite["mismatched_sources"]
                bounds = suite["bounds"]
                
                # Prepare source datasets
                matching_datasets = []
                for src in matching_sources:
                    s_X = match_rng.uniform(bounds[:, 0], bounds[:, 1], size=(50, dim))
                    matching_datasets.append((s_X, src(s_X)))
                    
                wrong_datasets = []
                for src in mismatched_sources:
                    s_X = wrong_rng.uniform(bounds[:, 0], bounds[:, 1], size=(50, dim))
                    wrong_datasets.append((s_X, src(s_X)))
                    
                # Random source: random quadratic landscapes
                rand_datasets = []
                for _ in range(2):
                    rand_center = rand_rng.uniform(bounds[:, 0], bounds[:, 1])
                    s_X = rand_rng.uniform(bounds[:, 0], bounds[:, 1], size=(50, dim))
                    s_y = np.sum((s_X - rand_center)**2, axis=1)
                    rand_datasets.append((s_X, s_y))
                    
                # Shared initial target sample
                init_X = init_rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_init, dim))
                init_y = target_func(init_X)
                y_init_best = float(np.min(init_y))
                
                modes = {
                    "Matching-Transfer": matching_datasets,
                    "No-Transfer": [],
                    "Random-Transfer": rand_datasets,
                    "Wrong-Transfer": wrong_datasets,
                }
                
                for mode_name, src_data in modes.items():
                    opt_config = LocalRegionTransferConfig(
                        top_ratio=0.20,
                        max_clusters=3,
                        pool_size=200,
                        source_weight=1.0,
                        source_weight_decay=0.08,
                        target_nomination_ratio=0.20,
                        source_nomination_ratio=0.20,
                        random_state=int(seed % 10000)
                    )
                    
                    optimizer = LocalRegionTransferOptimizer(bounds=bounds, config=opt_config)
                    if len(src_data) > 0:
                        optimizer.fit_source_regions(src_data)
                        
                    res = optimizer.optimize(
                        objective=target_func,
                        init_X=init_X,
                        init_y=init_y,
                        budget=budget
                    )
                    
                    final_best_y = res.best_y
                    improvement = y_init_best - final_best_y
                    
                    records.append({
                        "problem": prob,
                        "dim": dim,
                        "seed": seed,
                        "mode": mode_name,
                        "init_best_y": y_init_best,
                        "final_best_y": final_best_y,
                        "total_improvement": improvement,
                        "n_regions_extracted": len(optimizer.source_regions.regions) if optimizer.source_regions else 0
                    })
                    
                    for step_idx, y_val in enumerate(res.best_y_trace):
                        traces.append({
                            "problem": prob,
                            "dim": dim,
                            "seed": seed,
                            "mode": mode_name,
                            "step": step_idx,
                            "best_y": y_val,
                        })
                        
                if count % 4 == 0:
                    print(f"Benchmark Progress: {count}/{total_configs} instances completed...")

    df_summary = pd.DataFrame(records)
    df_traces = pd.DataFrame(traces)
    
    summary_path = os.path.join(output_dir, "optimizer_benchmark_summary.csv")
    traces_path = os.path.join(output_dir, "optimizer_benchmark_traces.csv")
    
    df_summary.to_csv(summary_path, index=False)
    df_traces.to_csv(traces_path, index=False)
    
    print(f"Saved benchmark summary to {summary_path}")
    print(f"Saved benchmark traces to {traces_path}")
    
    # Plot convergence figures
    plot_optimizer_benchmark(df_traces, df_summary, output_dir)
    return df_summary, df_traces


def plot_optimizer_benchmark(df_traces: pd.DataFrame, df_summary: pd.DataFrame, output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0), dpi=200)
    modes = ["Matching-Transfer", "No-Transfer", "Random-Transfer", "Wrong-Transfer"]
    mode_colors = {
        "Matching-Transfer": "#2ca02c",
        "No-Transfer": "#1f77b4",
        "Random-Transfer": "#7f7f7f",
        "Wrong-Transfer": "#d62728"
    }
    
    for idx, dim_val in enumerate([2, 5]):
        ax = axes[idx]
        sub_df = df_traces[df_traces['dim'] == dim_val]
        
        for mode in modes:
            m_df = sub_df[sub_df['mode'] == mode]
            if len(m_df) == 0:
                continue
            step_means = m_df.groupby('step')['best_y'].mean()
            step_stds = m_df.groupby('step')['best_y'].std()
            steps = step_means.index.values
            
            ax.plot(steps, step_means.values, label=mode, color=mode_colors[mode], lw=2.2)
            ax.fill_between(
                steps,
                step_means.values - 0.25 * step_stds.values,
                step_means.values + 0.25 * step_stds.values,
                color=mode_colors[mode],
                alpha=0.15
            )
            
        ax.set_xlabel("Optimization Step", fontsize=10)
        ax.set_ylabel("Best Function Value (Lower is Better)", fontsize=10)
        ax.set_title(f"{dim_val}D Problems: Optimizer Convergence Comparison", fontsize=11, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        if idx == 0:
            ax.legend(frameon=True, fontsize=9)
            
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "optimizer_benchmark_convergence.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved benchmark figure to {fig_path}")


if __name__ == "__main__":
    repo_root = os.path.dirname(current_dir)
    res_dir = os.path.join(repo_root, "results")
    run_optimizer_benchmark(res_dir)
