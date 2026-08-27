"""
Phase 2: Continuous Spatial Drift and Transferability Boundary Experiment.
Evaluates how incremental predictive power (Delta R^2_OOS) and Normalized Regret
transition as source region centers continuously drift away:
delta in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0].
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

from region_guided_reranking_study.landscapes import GaussianMixtureLandscape, ShiftedAckley
from region_guided_reranking_study.source_regions import (
    SourceRegionExtractor, SourceRegionLibrary
)
from region_guided_reranking_study.surrogate_and_candidates import (
    TargetGPSurrogate, CandidatePoolGenerator
)
from region_guided_reranking_study.rerankers import SoftRegionReranker, TargetOnlyReranker
from region_guided_reranking_study.metrics import (
    compute_partial_correlation, compute_out_of_sample_incremental_r2,
    evaluate_candidate_selection
)


def run_drift_experiment(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    deltas = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    seeds = [42, 101, 2026, 777, 999, 1234]
    dim = 2
    n_init = 6
    pool_size = 1000
    
    records = []
    print(f"Starting Phase 2 Continuous Drift Experiment (Deltas={deltas})...")
    
    for delta in deltas:
        for seed in seeds:
            seed_seq = np.random.SeedSequence(seed)
            task_ss, match_ss, init_ss, pool_ss, perm_ss = seed_seq.spawn(5)
            
            task_rng = np.random.default_rng(task_ss)
            match_rng = np.random.default_rng(match_ss)
            init_rng = np.random.default_rng(init_ss)
            pool_rng = np.random.default_rng(pool_ss)
            perm_rng = np.random.default_rng(perm_ss)
            
            # Target GMM
            target_gmm = GaussianMixtureLandscape(dim=dim, rng=task_rng)
            bounds = target_gmm.bounds
            
            # Drifted Source: centers shifted by delta in random direction
            drift_dir = task_rng.normal(size=dim)
            drift_dir /= np.linalg.norm(drift_dir)
            
            shifted_centers = [c + delta * drift_dir for c in target_gmm.centers]
            src_func = GaussianMixtureLandscape(dim=dim, centers=shifted_centers, 
                                                covs=target_gmm.covs, weights=target_gmm.weights, rng=match_rng)
            
            src_X = match_rng.uniform(bounds[:, 0], bounds[:, 1], size=(60, dim))
            src_y = src_func(src_X)
            
            extractor = SourceRegionExtractor(top_ratio=0.2, max_clusters=3, random_state=int(seed % 10000))
            src_lib = extractor.extract_from_multi_sources([(src_X, src_y)])
            
            # Target initial data
            target_init_X = init_rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_init, dim))
            target_init_y = target_gmm(target_init_X)
            y_init_best = float(np.min(target_init_y))
            
            # Target GP
            surrogate = TargetGPSurrogate(dim=dim, random_state=int(seed % 10000))
            surrogate.fit(target_init_X, target_init_y)
            
            # Candidate pool
            pool_gen = CandidatePoolGenerator(bounds=bounds, pool_size=pool_size, rng=pool_rng)
            candidates = pool_gen.generate(surrogate=surrogate, current_X=target_init_X, excluded_datasets=[src_X])
            
            true_y_pool = target_gmm(candidates)
            true_utility = -true_y_pool
            acq_scores = surrogate.compute_acquisition(candidates, acq_type="ei")
            
            # Statistical metrics
            r_scores = src_lib.score(candidates)
            partial_corr, _ = compute_partial_correlation(true_utility, r_scores, acq_scores)
            oos_r2 = compute_out_of_sample_incremental_r2(true_utility, acq_scores, r_scores, rng=perm_rng)
            
            # Reranking metrics
            t_only = TargetOnlyReranker()
            s_rerank = SoftRegionReranker(src_lib, weight_lambda=1.0)
            
            t_idx, _ = t_only.score_and_rank(candidates, acq_scores)
            s_idx, _ = s_rerank.score_and_rank(candidates, acq_scores)
            
            m_target = evaluate_candidate_selection(t_idx, true_y_pool, y_init_best)
            m_source = evaluate_candidate_selection(s_idx, true_y_pool, y_init_best)
            
            records.append({
                "delta": delta,
                "seed": seed,
                "partial_corr": partial_corr,
                "delta_r2_oos": oos_r2["delta_r2_oos"],
                "target_norm_regret": m_target["top1_normalized_regret"],
                "source_norm_regret": m_source["top1_normalized_regret"],
                "target_signed_improvement": m_target["top1_signed_improvement"],
                "source_signed_improvement": m_source["top1_signed_improvement"],
                "regret_reduction": m_target["top1_normalized_regret"] - m_source["top1_normalized_regret"],
            })
            
    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "drift_curve_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved Drift Curve Summary to {csv_path}")
    return df


if __name__ == "__main__":
    repo_root = os.path.dirname(current_dir)
    out_dir = os.path.join(repo_root, "results")
    run_drift_experiment(out_dir)
