"""
Unit tests for tie handling and zero-variance safety gating.
"""

import numpy as np
import pytest
from region_guided_reranking_study.rerankers import normalize_scores, SoftRegionReranker
from region_guided_reranking_study.source_regions import SourceRegionLibrary, SourceRegion


def test_normalize_scores_ties():
    # Identical constant array
    const_scores = np.array([0.0, 0.0, 0.0, 0.0])
    norm_const = normalize_scores(const_scores, method="rank")
    assert np.allclose(norm_const, 0.0), f"Constant scores must normalize to 0.0, got {norm_const}"
    
    # Array with tied values
    scores = np.array([1.0, 3.0, 3.0, 5.0])
    norm_ranks = normalize_scores(scores, method="rank")
    assert norm_ranks[0] == 0.0
    assert norm_ranks[3] == 1.0
    assert norm_ranks[1] == norm_ranks[2] == 0.5, f"Tied values should have equal mid-rank, got {norm_ranks}"


def test_zero_variance_safety_gate():
    # Empty library or library giving 0 support everywhere
    empty_lib = SourceRegionLibrary()
    reranker = SoftRegionReranker(empty_lib, weight_lambda=1.0)
    
    candidates = np.random.uniform(-5, 5, size=(50, 2))
    acq_scores = np.random.uniform(0, 1, size=50)
    
    ranked_idx, comb_scores = reranker.score_and_rank(candidates, acq_scores)
    target_ranks = normalize_scores(acq_scores, method="rank")
    
    # When source has zero variance, comb_scores must match target_ranks exactly
    assert np.allclose(comb_scores, target_ranks), "Zero-variance source should not alter target ranking"
