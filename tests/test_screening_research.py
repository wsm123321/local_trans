"""Tests for research metrics used by screening experiments."""

import numpy as np

from region_guided_reranking_study.screening_research import (
    mean_bootstrap_ci,
    selection_metrics,
)


def test_selection_metrics_are_scale_free_and_ranked():
    true_y = np.array([5.0, 1.0, 3.0, 2.0])
    acquisition = np.array([1.0, 0.2, 0.8, 0.5])
    metrics = selection_metrics(2, true_y, acquisition)

    assert metrics.selected_y == 3.0
    assert metrics.raw_regret == 2.0
    assert metrics.true_rank == 2
    assert metrics.acquisition_rank == 1
    assert metrics.normalized_regret >= 0.0


def test_bootstrap_ci_contains_constant_mean():
    mean, low, high = mean_bootstrap_ci([0.25] * 20, n_bootstrap=100)
    assert mean == 0.25
    assert low == 0.25
    assert high == 0.25
