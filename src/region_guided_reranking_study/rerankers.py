"""
Candidate Reranking Mechanisms and Policies.
Implements the 6 controlled comparators under identical candidate pools and surrogate scores:
1. Target-Only (pure acquisition)
2. Source-Region (soft fusion with zero-variance safety gate)
3. Random-Region (structure-matched random control)
4. Wrong-Source (adversarial/mismatched source baseline)
5. Oracle-Target-Region (true target optimum basin)
6. Hard-Filter (strict chi-square confidence ellipsoid gating)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.stats import rankdata
from .source_regions import SourceRegionLibrary


def normalize_scores(scores: np.ndarray, method: str = "rank") -> np.ndarray:
    """
    Normalize score array of length M to [0, 1] where higher is better.
    Correctly handles tied ranks via average ranking.
    Constant / zero-variance arrays map safely to uniform zeros.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    M = len(scores)
    if M <= 1:
        return np.zeros_like(scores)
        
    if np.ptp(scores) < 1e-12:
        return np.zeros_like(scores)
        
    if method == "rank":
        ranks = rankdata(scores, method="average")
        return (ranks - 1.0) / (M - 1.0)
    elif method == "minmax":
        s_min, s_max = np.min(scores), np.max(scores)
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
        ranked_indices = np.argsort(-combined_scores, kind='stable')
        return ranked_indices, combined_scores


class SoftRegionReranker(BaseReranker):
    """
    Soft fusion of target acquisition and region support with 'No information, no transfer' safety gate:
    J_t(x) = alpha_norm(x) + lambda_t * r_norm(x)
    """
    def __init__(self, region_lib: SourceRegionLibrary, weight_lambda: float = 1.0, 
                 norm_method: str = "rank", min_source_var: float = 1e-8,
                 name: str = "Source-Region"):
        super().__init__(name)
        self.region_lib = region_lib
        self.weight_lambda = float(weight_lambda)
        self.norm_method = norm_method
        self.min_source_var = float(min_source_var)

    def score_and_rank(self, candidates: np.ndarray, acq_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        alpha_norm = normalize_scores(acq_scores, method=self.norm_method)
        raw_r = self.region_lib.score(candidates)
        
        # Zero-variance / Low-information safety gate:
        # If source region score has no variance across the candidate pool, disable transfer (lambda = 0)
        if np.ptp(raw_r) < self.min_source_var or np.var(raw_r) < self.min_source_var:
            effective_lambda = 0.0
            r_norm = np.zeros_like(raw_r)
        else:
            effective_lambda = self.weight_lambda
            r_norm = normalize_scores(raw_r, method=self.norm_method)
            
        combined_scores = alpha_norm + effective_lambda * r_norm
        ranked_indices = np.argsort(-combined_scores, kind='stable')
        return ranked_indices, combined_scores


class HardFilterReranker(BaseReranker):
    """
    M6: Hard geometric filtering using chi-square confidence ellipsoids.
    Candidates outside all source region ellipsoids receive heavy penalty.
    """
    def __init__(self, region_lib: SourceRegionLibrary, confidence: float = 0.95, 
                 name: str = "Hard-Filter"):
        super().__init__(name)
        self.region_lib = region_lib
        self.confidence = float(confidence)

    def score_and_rank(self, candidates: np.ndarray, acq_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        alpha_norm = normalize_scores(acq_scores, method="rank")
        inside_mask = self.region_lib.filter_inside_any_region(candidates, confidence=self.confidence)
        
        combined_scores = alpha_norm.copy()
        if np.sum(inside_mask) > 0:
            # Penalize candidates outside the geometric confidence ellipsoid
            combined_scores[~inside_mask] -= 100.0
            
        ranked_indices = np.argsort(-combined_scores, kind='stable')
        return ranked_indices, combined_scores


class TrueOracleReranker(BaseReranker):
    """M5: Oracle baseline using true target global optimum basin or true utility."""
    def __init__(self, oracle_lib: SourceRegionLibrary, name: str = "Oracle-Target-Region"):
        super().__init__(name)
        self.oracle_lib = oracle_lib

    def score_and_rank(self, candidates: np.ndarray, acq_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        alpha_norm = normalize_scores(acq_scores, method="rank")
        raw_r = self.oracle_lib.score(candidates)
        r_norm = normalize_scores(raw_r, method="rank")
        combined_scores = alpha_norm + 1.0 * r_norm
        ranked_indices = np.argsort(-combined_scores, kind='stable')
        return ranked_indices, combined_scores


def create_comparator_suite(matching_lib: SourceRegionLibrary,
                             random_lib: SourceRegionLibrary,
                             wrong_lib: SourceRegionLibrary,
                             oracle_lib: SourceRegionLibrary,
                             weight_lambda: float = 1.0) -> Dict[str, BaseReranker]:
    """Assemble all 6 comparators."""
    return {
        "Target-Only": TargetOnlyReranker(),
        "Source-Region": SoftRegionReranker(matching_lib, weight_lambda=weight_lambda, name="Source-Region"),
        "Random-Region": SoftRegionReranker(random_lib, weight_lambda=weight_lambda, name="Random-Region"),
        "Wrong-Source": SoftRegionReranker(wrong_lib, weight_lambda=weight_lambda, name="Wrong-Source"),
        "Oracle-Target-Region": TrueOracleReranker(oracle_lib, name="Oracle-Target-Region"),
        "Hard-Filter": HardFilterReranker(matching_lib, confidence=0.95, name="Hard-Filter"),
    }
