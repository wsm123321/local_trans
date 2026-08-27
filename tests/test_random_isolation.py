"""
Unit tests for random stream isolation and candidate pool deduplication.
"""

import numpy as np
import pytest
from region_guided_reranking_study.surrogate_and_candidates import CandidatePoolGenerator


def test_seed_sequence_spawn_isolation():
    seed = 42
    seed_seq = np.random.SeedSequence(seed)
    task_ss, match_ss, wrong_ss, init_ss, pool_ss = seed_seq.spawn(5)
    
    rng1 = np.random.default_rng(match_ss)
    rng2 = np.random.default_rng(pool_ss)
    
    vals1 = rng1.uniform(0, 1, size=10)
    vals2 = rng2.uniform(0, 1, size=10)
    
    assert not np.allclose(vals1, vals2), "Spawned streams should not produce identical values"


def test_candidate_pool_exclusion():
    dim = 2
    bounds = np.array([[-5.0, 5.0], [-5.0, 5.0]])
    rng = np.random.default_rng(123)
    
    target_init_X = rng.uniform(-5.0, 5.0, size=(10, dim))
    source_X = rng.uniform(-5.0, 5.0, size=(50, dim))
    
    gen = CandidatePoolGenerator(bounds=bounds, pool_size=200, rng=rng)
    pool = gen.generate(current_X=target_init_X, excluded_datasets=[source_X])
    
    # Assert no exact overlap
    for pt in pool:
        dists_target = np.linalg.norm(target_init_X - pt[np.newaxis, :], axis=1)
        dists_source = np.linalg.norm(source_X - pt[np.newaxis, :], axis=1)
        assert np.min(dists_target) > 1e-6, "Pool candidate overlaps with target initial point"
        assert np.min(dists_source) > 1e-6, "Pool candidate overlaps with source point"
