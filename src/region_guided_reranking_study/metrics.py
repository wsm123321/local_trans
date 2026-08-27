"""
Statistical Verification and Performance Evaluation Metrics.
Implements:
1. Rigorous Candidate Decision Quality Metrics:
   - Normalized Regret (scale-free across benchmark landscapes)
   - Signed Improvement (preserves negative transfer degradation)
   - Pool Oracle Rank Promotion
   - Top-10% and Top-5% Hit Rates
2. Statistical Information Tests:
   - Partial Correlation rho(U_t, r_s | alpha_t)
   - 5-Fold Cross-Validated Out-of-Sample Delta R^2_OOS
   - Permutation Null Hypothesis Test on Source Region Score
"""

from typing import Dict, Tuple, Optional, List
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold


def compute_partial_correlation(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[float, float]:
    """
    Compute partial correlation rho(x, y | z): correlation between x and y controlling for z.
    Returns: (partial_corr, p_value_approx)
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    
    std_x, std_y, std_z = np.std(x), np.std(y), np.std(z)
    if std_x < 1e-8 or std_y < 1e-8:
        return 0.0, 1.0
        
    if std_z < 1e-8:
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
    
    n = len(x)
    df = n - 3
    if df > 0 and abs(partial_r) < 1.0:
        t_stat = partial_r * np.sqrt(df / max(1e-8, (1.0 - partial_r**2)))
        from scipy.stats import t
        p_val = float(2.0 * (1.0 - t.cdf(abs(t_stat), df=df)))
    else:
        p_val = 1.0
        
    return partial_r, p_val


def compute_out_of_sample_incremental_r2(U_t: np.ndarray, alpha_t: np.ndarray, r_s: np.ndarray,
                                         n_splits: int = 5, n_permutations: int = 50,
                                         rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
    """
    Fast, vectorized 5-Fold Cross-Validated Out-of-Sample Delta R^2_OOS and Permutation p-value.
    Uses closed-form regularized ridge regression for lightning speed.
    """
    if rng is None:
        rng = np.random.default_rng(42)
        
    U = np.asarray(U_t, dtype=float).ravel()
    A = np.asarray(alpha_t, dtype=float).ravel()
    R = np.asarray(r_s, dtype=float).ravel()
    
    # Check variance
    if np.var(U) < 1e-10 or np.var(R) < 1e-10:
        return {
            "r2_oos_base": 0.0,
            "r2_oos_full": 0.0,
            "delta_r2_oos": 0.0,
            "permutation_p_val": 1.0
        }
        
    # Standardize arrays
    u_std = (U - np.mean(U)) / (np.std(U) + 1e-8)
    a_std = (A - np.mean(A)) / (np.std(A) + 1e-8)
    r_std = (R - np.mean(R)) / (np.std(R) + 1e-8)
    
    N = len(U)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = list(kf.split(U))
    
    # Pre-build feature matrices
    X1 = np.column_stack([np.ones(N), a_std])
    X2 = np.column_stack([np.ones(N), a_std, r_std])
    
    pred_base = np.zeros(N)
    pred_full = np.zeros(N)
    
    alpha_ridge = 1.0
    
    for tr, te in folds:
        # Model 1
        X1_tr, y_tr = X1[tr], u_std[tr]
        w1 = np.linalg.solve(X1_tr.T @ X1_tr + alpha_ridge * np.eye(2), X1_tr.T @ y_tr)
        pred_base[te] = X1[te] @ w1
        
        # Model 2
        X2_tr = X2[tr]
        w2 = np.linalg.solve(X2_tr.T @ X2_tr + alpha_ridge * np.eye(3), X2_tr.T @ y_tr)
        pred_full[te] = X2[te] @ w2
        
    ss_tot = np.sum(u_std**2) + 1e-12
    ss_res_base = np.sum((u_std - pred_base)**2)
    ss_res_full = np.sum((u_std - pred_full)**2)
    
    r2_oos_base = max(-1.0, float(1.0 - (ss_res_base / ss_tot)))
    r2_oos_full = max(-1.0, float(1.0 - (ss_res_full / ss_tot)))
    delta_r2_oos = float(np.clip(r2_oos_full - r2_oos_base, -1.0, 1.0))
    
    # Fast permutation test
    perm_delta_r2s = []
    r_shuffled = r_std.copy()
    for _ in range(n_permutations):
        rng.shuffle(r_shuffled)
        X2_perm = np.column_stack([np.ones(N), a_std, r_shuffled])
        pred_perm = np.zeros(N)
        for tr, te in folds:
            X2p_tr = X2_perm[tr]
            wp = np.linalg.solve(X2p_tr.T @ X2p_tr + alpha_ridge * np.eye(3), X2p_tr.T @ u_std[tr])
            pred_perm[te] = X2_perm[te] @ wp
        ss_res_p = np.sum((u_std - pred_perm)**2)
        r2_p = max(-1.0, float(1.0 - (ss_res_p / ss_tot)))
        perm_delta_r2s.append(r2_p - r2_oos_base)
        
    perm_delta_r2s = np.array(perm_delta_r2s)
    p_val = float(np.mean(perm_delta_r2s >= delta_r2_oos))
    
    return {
        "r2_oos_base": float(r2_oos_base),
        "r2_oos_full": float(r2_oos_full),
        "delta_r2_oos": float(delta_r2_oos),
        "permutation_p_val": float(p_val)
    }


def evaluate_candidate_selection(ranked_indices: np.ndarray, 
                                 true_y_pool: np.ndarray, 
                                 y_init_best: float) -> Dict[str, float]:
    """
    Evaluate candidate decision quality.
    Assumes minimization (lower y is better).
    """
    ranked_y = true_y_pool[ranked_indices]
    oracle_pool_best = np.min(true_y_pool)
    oracle_pool_best_idx = np.argmin(true_y_pool)
    
    rank_of_oracle = int(np.where(ranked_indices == oracle_pool_best_idx)[0][0])
    
    top1_y = ranked_y[0]
    top1_raw_regret = top1_y - oracle_pool_best
    
    # Scale-free normalized regret: standardized by candidate pool 90% quantile spread
    pool_scale = max(1e-4, float(np.quantile(true_y_pool, 0.90) - oracle_pool_best))
    normalized_regret = float(top1_raw_regret / pool_scale)
    
    # Signed improvement: preserves negative transfer degradation
    signed_improvement = float(y_init_best - top1_y)
    positive_improvement = float(max(0.0, signed_improvement))
    is_improved = 1.0 if signed_improvement > 0 else 0.0
    
    # Hit rates in candidate pool
    q05 = np.quantile(true_y_pool, 0.05)
    q10 = np.quantile(true_y_pool, 0.10)
    top1_hit_top05 = 1.0 if top1_y <= q05 else 0.0
    top1_hit_top10 = 1.0 if top1_y <= q10 else 0.0
    
    return {
        "top1_true_y": float(top1_y),
        "top1_raw_regret": float(top1_raw_regret),
        "top1_normalized_regret": float(normalized_regret),
        "top1_signed_improvement": float(signed_improvement),
        "top1_positive_improvement": float(positive_improvement),
        "is_improved": float(is_improved),
        "top1_hit_top05": float(top1_hit_top05),
        "top1_hit_top10": float(top1_hit_top10),
        "rank_of_pool_oracle": rank_of_oracle,
    }
