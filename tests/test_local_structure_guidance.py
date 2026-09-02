"""Tests for the unified local-structure candidate guidance core."""

from types import SimpleNamespace

import numpy as np
import pytest

from region_guided_reranking_study.local_structure_guidance import (
    geometry_only,
    local_rank_no_reliability,
    local_rank_reliability,
    normalize_rank_scores,
    reversed_local_rank,
    target_only,
)
from region_guided_reranking_study.source_local_structure import SourceLocalStructureLibrary


class FakeStructure:
    def __init__(self, center, quality, reliability=1.0, region_quality=1.0, region_id="r"):
        self.center = np.asarray(center, dtype=float)
        self.quality = np.asarray(quality, dtype=float)
        self.quality_floor = 0.0
        self.reliability_floor = 0.0
        self.region_quality = float(region_quality)
        self.core_count = 1
        self.region_id = region_id
        self.constant_membership = False
        self.validation = SimpleNamespace(reliability=float(reliability))

    @property
    def dim(self):
        return len(self.center)

    def _values(self, X):
        return self.quality[np.arange(len(X))]

    def membership(self, X):
        X = np.asarray(X, dtype=float)
        if self.constant_membership:
            return np.ones(len(X))
        return np.exp(-np.sum((X - self.center) ** 2, axis=1))

    def predict_relative_quality(self, X, return_std=False):
        result = self._values(X)
        return (result, np.zeros_like(result)) if return_std else (result, None)

    def geometry_score(self, X):
        return self.region_quality * self.membership(X)

    def structure_score(self, X, use_reliability=True):
        reliability = self.validation.reliability if use_reliability else 1.0
        return self.geometry_score(X) * self.quality * reliability


def library(*structures):
    return SourceLocalStructureLibrary(list(structures))


def _constant_structure():
    structure = FakeStructure([0.0], [0.5, 0.5])
    structure.constant_membership = True
    return structure


def test_tie_aware_rank_normalization_and_exact_target_selection():
    assert np.allclose(normalize_rank_scores([3.0, 1.0, 1.0, 0.0]), [1.0, 0.5, 0.5, 0.0])
    points = np.arange(8, dtype=float).reshape(-1, 1)
    result = target_only(points, [0.2, 0.9, 0.9, 0.1, 0.4, 0.3, 0.0, 0.8])
    assert result.selected_index == 1
    assert result.fallback is False
    assert np.array_equal(result.source_nominees, np.empty(0, dtype=int))


def test_geometry_and_local_rank_can_differ():
    points = np.array([[0.0], [1.0], [2.0]])
    source = FakeStructure([0.0], [0.05, 0.9, 0.2], region_id="r0")
    result_geometry = geometry_only(points, np.zeros(3), library(source), source_weight=2.0)
    result_local = local_rank_no_reliability(points, np.zeros(3), library(source), source_weight=2.0)
    assert result_geometry.selected_index == 0
    assert result_local.selected_index == 1
    assert not np.allclose(result_geometry.source_scores, result_local.source_scores)


def test_reliability_switch_changes_local_scores():
    points = np.array([[0.0], [1.0]])
    source = FakeStructure([0.0], [0.8, 0.8], reliability=0.2)
    no_reliability = local_rank_no_reliability(points, [0.0, 0.0], library(source))
    with_reliability = local_rank_reliability(points, [0.0, 0.0], library(source))
    assert with_reliability.source_scores[0] < no_reliability.source_scores[0]
    assert with_reliability.source_scores[1] < no_reliability.source_scores[1]


def test_reversed_only_reverses_model_quality_factor():
    points = np.array([[0.0], [1.0]])
    source = FakeStructure([0.0], [0.9, 0.1], reliability=0.7, region_quality=0.8)
    normal = local_rank_reliability(points, [0.0, 0.0], library(source))
    reversed_result = reversed_local_rank(points, [0.0, 0.0], library(source))
    membership = source.membership(points)
    expected = membership * (1.0 - source.quality) * 0.7 * 0.8
    assert np.allclose(reversed_result.source_scores, expected)
    assert np.allclose(reversed_result.normalized_source_scores, normalize_rank_scores(expected))
    assert not np.allclose(reversed_result.source_scores, 1.0 - normal.scores)


@pytest.mark.parametrize("library_value, weight, reason", [
    (SourceLocalStructureLibrary(), 1.0, "empty_library"),
    (library(_constant_structure()), 1.0, "constant_source_scores"),
    (library(FakeStructure([0.0], [np.nan, 0.5])), 1.0, "nonfinite_source_scores"),
    (library(FakeStructure([0.0], [0.1, 0.9])), 0.0, "source_weight_zero"),
])
def test_degenerate_source_guidance_falls_back_exactly_to_target_only(library_value, weight, reason):
    points = np.array([[0.0], [1.0]])
    acquisition = np.array([0.1, 0.9])
    result = local_rank_reliability(points, acquisition, library_value, source_weight=weight)
    target = target_only(points, acquisition)
    assert result.fallback is True
    assert result.fallback_reason == reason
    assert result.selected_index == target.selected_index
    assert np.allclose(result.scores, target.scores)
    assert np.array_equal(result.shortlist, target.shortlist)


def test_target_nominee_is_retained_in_source_target_union():
    points = np.arange(10, dtype=float).reshape(-1, 1)
    source = FakeStructure([0.0], np.full(10, 0.5), region_id="r0")
    result = local_rank_no_reliability(
        points,
        np.array([10.0] + [0.0] * 9),
        library(source),
        source_weight=2.0,
        target_nomination_ratio=0.1,
        source_nomination_ratio=0.2,
        shortlist_size=1,
    )
    assert 0 in result.target_nominees
    assert 0 in result.shortlist


def test_guidance_is_deterministic():
    points = np.array([[0.0], [0.5], [1.0], [1.5]])
    source = FakeStructure([0.5], [0.2, 0.8, 0.3, 0.1], region_id="r0")
    args = (points, np.array([0.2, 0.1, 0.4, 0.3]), library(source))
    first = local_rank_reliability(*args, source_weight=1.7)
    second = local_rank_reliability(*args, source_weight=1.7)
    assert first.selected_index == second.selected_index
    assert np.array_equal(first.shortlist, second.shortlist)
    assert np.allclose(first.scores, second.scores)
