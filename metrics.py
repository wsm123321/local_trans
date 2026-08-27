"""
Statistical Verification and Performance Evaluation Metrics.
Implements:
1. Conditional Information & Incremental Explanatory Power:
   - Partial Correlation: rho(U_t, r_s | alpha_t)
   - Incremental R^2: Delta R^2 = R^2(U_t ~ alpha_t + r_s) - R^2(U_t ~ alpha_t)
   - Spearman Rank Correlations
2. Candidate Selection Quality:
   - Simple Regret (vs pool oracle & global minimum)
   - One-step Improvement
   - Top-10% / Top-5% Hit Rate
   - Rank of Global Best Candidate in Reranked List
"""

from typing import Dict, Tuple, Optional
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression


def compute_partial_correlation(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[float, float]:
    """
    Compute partial correlation rho(x, y | z): correlation between x and y controlling for z.
    Returns: (partial_corr, p_value_approx)
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    
    # Numerical guard: constant arrays have 0 variance
    std_x, std_y, std_z = np.std(x), np.std(y), np.std(z)
    if std_x < 1e-8 or std_y < 1e-8:
        return 0.0, 1.0
        
    if std_z < 1e-8:
        # If z is constant, partial correlation reduces to standard pearson correlation
        r_xy, p_xy = pearsonr(x, y)
        return float(np.nan_to_num(r_xy, nan=0.0)), float(np.nan_to_num(p_xy, nan=1.0))
        
    r_xy, _ = pearsonr(x, y)
    r_xz, _ = pearsonr(x, z)
    r_yz, _ = pearsonr(y, z)
    
    r_xy = float(np.nan_to_num(r_xy, nan=0.0))
    r_xz = float(np.nan_to_num(r_xz, nan=0.0))
    r_yz = float(np.nan_to_num(r_yz, nan=0.0))
    
    denom = np.sqrt(max(1e-12, (1.0 - r_xz**2) * (1.0 - r_yz**2)))
    partial_r = (r_xy - r_xz * r_yz) / denom
    partial_r = float(np.clip(np.nan_to_num(partial_r, nan=0.0), -1.0, 1.0))
    
    # Approx t-test p-value
    n = len(x)
    df = n - 3
    if df > 0 and abs(partial_r) < 1.0:
        t_stat = partial_r * np.sqrt(df / max(1e-8, (1.0 - partial_r**2)))
        from scipy.stats import t
        p_val = float(2.0 * (1.0 - t.cdf(abs(t_stat), df=df)))
    else:
        p_val = 0.0
        
    return partial_r, p_val


def compute_incremental_r2(U_t: np.ndarray, alpha_t: np.ndarray, r_s: np.ndarray) -> Dict[str, float]:
    """
    Fit linear models:
      Model 1: U_t ~ alpha_t
      Model 2: U_t ~ alpha_t + r_s
    Compute Delta R^2 = R^2_2 - R^2_1.
    """
    U = np.asarray(U_t, dtype=float).reshape(-1, 1)
    A = np.asarray(alpha_t, dtype=float).reshape(-1, 1)
    AR = np.column_stack([alpha_t, r_s])
    
    reg1 = LinearRegression().fit(A, U)
    r2_base = float(reg1.score(A, U))
    
    reg2 = LinearRegression().fit(AR, U)
    r2_full = float(reg2.score(AR, U))
    
    delta_r2 = max(0.0, r2_full - r2_base)
    
    return {
        "r2_target_only": r2_base,
        "r2_with_source": r2_full,
        "delta_r2": delta_r2
    }


def evaluate_candidate_selection(ranked_indices: np.ndarray, 
                                 true_y_pool: np.ndarray, 
                                 y_init_best: float,
                                 y_global_min: float,
                                 top_k_list: Tuple[int, ...] = (1, 3, 5)) -> Dict[str, float]:
    """
    Evaluate decision quality of the selected candidate(s).
    Assumes minimization (lower y is better).
    """
    ranked_y = true_y_pool[ranked_indices]
    oracle_pool_best = np.min(true_y_pool)
    oracle_pool_best_idx = np.argmin(true_y_pool)
    
    # Rank of the pool's single best candidate in the method's sorted list
    rank_of_oracle = int(np.where(ranked_indices == oracle_pool_best_idx)[0][0])
    
    top1_y = ranked_y[0]
    top1_simple_regret_pool = top1_y - oracle_pool_best
    top1_simple_regret_global = top1_y - y_global_min
    top1_improvement = max(0.0, y_init_best - top1_y)
    
    # Hit rates: is top-1 within top-5% and top-10% of the candidate pool?
    q05 = np.quantile(true_y_pool, 0.05)
    q10 = np.quantile(true_y_pool, 0.10)
    top1_hit_top05 = 1.0 if top1_y <= q05 else 0.0
    top1_hit_top10 = 1.0 if top1_y <= q10 else 0.0
    
    results = {
        "top1_true_y": float(top1_y),
        "top1_regret_pool": float(top1_simple_regret_pool),
        "top1_regret_global": float(top1_simple_regret_global),
        "top1_improvement": float(top1_improvement),
        "top1_hit_top05": float(top1_hit_top05),
        "top1_hit_top10": float(top1_hit_top10),
        "rank_of_pool_oracle": rank_of_oracle,
    }
    
    # Top-K average regrets
    for k in top_k_list:
        k_actual = min(k, len(ranked_y))
        topk_mean_regret = np.mean(ranked_y[:k_actual]) - oracle_pool_best
        results[f"top{k}_mean_regret_pool"] = float(topk_mean_regret)
        
    return results
