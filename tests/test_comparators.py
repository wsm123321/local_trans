"""
Unit tests for all 6 comparators execution.
"""

import numpy as np
import pytest
from region_guided_reranking_study.source_regions import SourceRegion, SourceRegionLibrary
from region_guided_reranking_study.rerankers import create_comparator_suite


def test_all_comparators_run():
    dim = 2
    bounds = np.array([[-5.0, 5.0], [-5.0, 5.0]])
    
    r1 = SourceRegion(center=np.array([1.0, 1.0]), cov=np.eye(2), quality=0.9, count=10)
    matching_lib = SourceRegionLibrary([r1])
    
    r_rand = SourceRegion(center=np.array([-2.0, 2.0]), cov=np.eye(2), quality=0.9, count=10)
    random_lib = SourceRegionLibrary([r_rand])
    
    r_wrong = SourceRegion(center=np.array([4.0, -4.0]), cov=np.eye(2), quality=0.9, count=10)
    wrong_lib = SourceRegionLibrary([r_wrong])
    
    oracle_lib = SourceRegionLibrary([r1])
    
    comparators = create_comparator_suite(matching_lib, random_lib, wrong_lib, oracle_lib, weight_lambda=1.0)
    assert len(comparators) == 6
    
    candidates = np.random.uniform(-5, 5, size=(100, dim))
    acq_scores = np.random.uniform(0, 1, size=100)
    
    for name, reranker in comparators.items():
        ranked_idx, scores = reranker.score_and_rank(candidates, acq_scores)
        assert len(ranked_idx) == 100
        assert len(scores) == 100
        assert ranked_idx[0] in range(100)
