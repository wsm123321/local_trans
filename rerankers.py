"""
Candidate Reranking Mechanisms and Policies.
Implements the 6 controlled comparators under identical candidate pools and surrogate scores:
1. Target-Only (pure acquisition)
2. Source-Region (matching source library soft rerank)
3. Random-Region (random regions baseline)
4. Wrong-Source (mismatched/adversarial source baseline)
5. Oracle-Target-Region (ground truth target basin upper bound)
6. Hard-Filter (strict binary gating baseline)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from .source_regions import SourceRegionLibrary


def normalize_scores(scores: np.ndarray, method: str = "rank") -> np.ndarray:
    """
    Normalize score array of length M to [0, 1] where higher is better.
    method: 'rank' (quantile rank) or 'minmax'
    """
    scores = np.asarray(scores, dtype=float).ravel()
    M = len(scores)
    if M <= 1:
        return np.ones(M)
        
    if method == "rank":
        # Tied ranks handled by argsort of argsort
        ranks = np.argsort(np.argsort(scores))
        return ranks / (M - 1.0)
    elif method == "minmax":
        s_min, s_max = np.min(scores), np.max(scores)
        if s_max - s_min < 1e-12:
            return np.zeros(M)
        return (scores - s_min) / (s_max - s_min)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


class BaseReranker:
    """Base interface for candidate ranking strategies."""
    def __init__(self, name: str):
        self.name = name

    def score_and_rank(self, candidates: np.ndarray, acq_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rank candidates.
        Returns:
            ranked_indices: 1D array of indices in descending order of preference (0 is top-1).
            combined_scores: 1D array of combined scores J_t(x) for each candidate.
        """
        raise NotImplementedError


class TargetOnlyReranker(BaseReranker):
    """M1: Pure Target-Only acquisition function baseline."""
    def __init__(self):
        super().__init__("Target-Only")

    def score_and_rank(self, candidates: np.ndarray, acq_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        combined_scores = normalize_scores(acq_scores, method="rank")
        ranked_indices = np.argsort(combined_scores)[::-1]
        return ranked_indices, combined_scores


class SoftRegionReranker(BaseReranker):
    """
    Soft fusion of target acquisition and region support:
    J_t(x) = alpha_norm(x) + lambda_t * r_norm(x)
    """
    def __init__(self, region_lib: SourceRegionLibrary, weight_lambda: float = 1.0, 
                 norm_method: str = "rank", name: str = "Source-Region"):
        super().__init__(name)
        self.region_lib = region_lib
        self.weight_lambda = weight_lambda
        self.norm_method = norm_method

    def score_and_rank(self, candidates: np.ndarray, acq_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        alpha_norm = normalize_scores(acq_scores, method=self.norm_method)
        raw_r = self.region_lib.score(candidates)
        r_norm = normalize_scores(raw_r, method=self.norm_method)
        
        combined_scores = alpha_norm + self.weight_lambda * r_norm
        ranked_indices = np.argsort(combined_scores)[::-1]
        return ranked_indices, combined_scores


class HardFilterReranker(BaseReranker):
    """
    M6: Hard thresholding - only keeps candidates with region support above threshold.
    Ranks filtered candidates by pure acquisition score.
    """
    def __init__(self, region_lib: SourceRegionLibrary, threshold_ratio: float = 0.5, 
                 name: str = "Hard-Filter"):
        super().__init__(name)
        self.region_lib = region_lib
        self.threshold_ratio = threshold_ratio

    def score_and_rank(self, candidates: np.ndarray, acq_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        M = len(candidates)
        alpha_norm = normalize_scores(acq_scores, method="rank")
        raw_r = self.region_lib.score(candidates)
        
        # Binary mask
        r_thresh = np.quantile(raw_r, self.threshold_ratio)
        mask = raw_r >= r_thresh
        
        combined_scores = alpha_norm.copy()
        if np.sum(mask) > 0:
            # Penalize candidates outside region
            combined_scores[~mask] -= 10.0
            
        ranked_indices = np.argsort(combined_scores)[::-1]
        return ranked_indices, combined_scores


def create_comparator_suite(matching_lib: SourceRegionLibrary,
                             random_lib: SourceRegionLibrary,
                             wrong_lib: SourceRegionLibrary,
                             oracle_lib: SourceRegionLibrary,
                             weight_lambda: float = 1.0) -> Dict[str, BaseReranker]:
    """Assemble all 6 comparators for controlled experiment."""
    return {
        "Target-Only": TargetOnlyReranker(),
        "Source-Region": SoftRegionReranker(matching_lib, weight_lambda=weight_lambda, name="Source-Region"),
        "Random-Region": SoftRegionReranker(random_lib, weight_lambda=weight_lambda, name="Random-Region"),
        "Wrong-Source": SoftRegionReranker(wrong_lib, weight_lambda=weight_lambda, name="Wrong-Source"),
        "Oracle-Target-Region": SoftRegionReranker(oracle_lib, weight_lambda=weight_lambda, name="Oracle-Target-Region"),
        "Hard-Filter": HardFilterReranker(matching_lib, threshold_ratio=0.7, name="Hard-Filter"),
    }
