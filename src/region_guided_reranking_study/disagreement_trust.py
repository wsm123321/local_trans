"""Minimal target-feedback trust estimators for source-local candidate advice.

The estimators in this module do not generate candidates and do not alter a target
surrogate.  They score one already fixed source expert from previously revealed
target feedback.  A downstream gate may then accept one fixed source nomination or
fall back to the target-only nomination.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
from scipy.stats import beta as beta_distribution
from scipy.stats import norm, spearmanr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from .local_surrogate_transfer import (
    LocalExpertResidualRegressor,
    LocalSurrogateTransferConfig,
)
from .local_surrogate_transfer_research import rank_quality

Array = np.ndarray
Objective = Callable[[Array], Array]


@dataclass(frozen=True)
class TrustScore:
    """One pre-decision target-feedback score.

    ``score`` is retained even when the estimator is ineligible so that sparse
    evidence remains auditable. Deployable gate logic always checks ``eligible``.
    Prediction sensitivity analyses use :func:`prediction_score`, which maps an
    ineligible estimate to neutral trust rather than inventing negative evidence.
    """

    score: float
    raw_statistic: float
    eligible: bool
    evidence_count: int
    reason: str
    successes: int = 0
    failures: int = 0


@dataclass(frozen=True)
class FrozenGateThreshold:
    """Label-free operating threshold frozen on development score coverage."""

    threshold: float
    target_coverage: float
    achieved_actionable_coverage: float
    actionable_events: int
    eligible_actionable_events: int
    accepted_actionable_events: int


def local_spearman_trust(
    source_quality: Array,
    target_cost: Array,
    *,
    min_points: int,
    shrinkage: float,
) -> TrustScore:
    """Shrink local source/target rank agreement toward neutral trust.

    Source quality is larger-is-better and target cost is smaller-is-better, hence
    the correlation is computed against ``-target_cost``.
    """

    quality, cost = _paired_vectors(source_quality, target_cost)
    return _shrunken_spearman(
        quality,
        -cost,
        min_points=min_points,
        shrinkage=shrinkage,
        label="local_spearman",
    )


def residual_spearman_trust(
    source_quality: Array,
    target_cost: Array,
    pre_observation_target_mean: Array,
    *,
    min_points: int,
    shrinkage: float,
) -> TrustScore:
    """Score whether source quality identifies target-GP underprediction benefit.

    Only means frozen before the corresponding outcomes were observed are valid.
    For minimization, a negative residual ``target_cost - target_mean`` is a better
    outcome than the target model expected; therefore source quality is correlated
    with the negative residual.
    """

    quality, cost = _paired_vectors(source_quality, target_cost)
    means = np.asarray(pre_observation_target_mean, dtype=float).reshape(-1)
    if len(means) != len(cost) or not np.all(np.isfinite(means)):
        raise ValueError("pre_observation_target_mean must match target_cost.")
    residual_benefit = -(cost - means)
    return _shrunken_spearman(
        quality,
        residual_benefit,
        min_points=min_points,
        shrinkage=shrinkage,
        label="residual_spearman",
    )


def paired_margin_spearman_trust(
    source_margin: Array,
    observed_advantage: Array,
    *,
    min_points: int,
    shrinkage: float,
    label: str,
) -> TrustScore:
    """Shrink decision-matched pair-margin agreement toward neutral trust.

    Both inputs use the same orientation: positive means the diagnostic alternative
    is preferred over Target-Only's member of the paid pair.
    """

    source, outcome = _paired_vectors(source_margin, observed_advantage)
    return _shrunken_spearman(
        source,
        outcome,
        min_points=min_points,
        shrinkage=shrinkage,
        label=str(label),
    )


def disagreement_correction_trust(
    successes: int,
    failures: int,
    *,
    min_events: int,
) -> TrustScore:
    """Beta-Binomial trust that source wins target-source disagreement events."""

    success_count = int(successes)
    failure_count = int(failures)
    required = int(min_events)
    if success_count < 0 or failure_count < 0:
        raise ValueError("successes and failures must be non-negative.")
    if required < 1:
        raise ValueError("min_events must be positive.")
    count = success_count + failure_count
    posterior_probability = float(
        beta_distribution.sf(
            0.5,
            1.0 + success_count,
            1.0 + failure_count,
        )
    )
    eligible = count >= required
    return TrustScore(
        score=posterior_probability,
        raw_statistic=(success_count + 1.0) / (count + 2.0),
        eligible=eligible,
        evidence_count=count,
        reason="eligible" if eligible else "insufficient_disagreement_events",
        successes=success_count,
        failures=failure_count,
    )


def prediction_score(evidence: TrustScore) -> float:
    """Neutral-imputed score used only by all-actionable sensitivity analysis."""

    return float(evidence.score) if evidence.eligible else 0.5


def freeze_coverage_threshold(
    scores: Array,
    eligible: Array,
    actionable: Array,
    *,
    target_coverage: float,
    minimum_positive_score: float = 0.5,
) -> FrozenGateThreshold:
    """Freeze a threshold without using future-benefit labels.

    The denominator is every actionable development event.  Ineligible events are
    abstentions.  Candidate thresholds must be strictly above the neutral minimum;
    an additional threshold above one represents accepting no events.  Ties in
    coverage error select the higher, more conservative threshold.
    """

    values = np.asarray(scores, dtype=float).reshape(-1)
    usable = np.asarray(eligible, dtype=bool).reshape(-1)
    opportunities = np.asarray(actionable, dtype=bool).reshape(-1)
    if not (len(values) == len(usable) == len(opportunities)):
        raise ValueError("scores, eligible, and actionable must have equal length.")
    if len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("scores must be a non-empty finite vector.")
    if not 0.0 <= target_coverage <= 1.0:
        raise ValueError("target_coverage must lie in [0, 1].")
    if not 0.0 <= minimum_positive_score <= 1.0:
        raise ValueError("minimum_positive_score must lie in [0, 1].")

    actionable_count = int(np.sum(opportunities))
    eligible_count = int(np.sum(opportunities & usable))
    reject_all = float(1.0 + 1e-12)
    if actionable_count == 0 or eligible_count == 0:
        return FrozenGateThreshold(
            threshold=reject_all,
            target_coverage=float(target_coverage),
            achieved_actionable_coverage=0.0,
            actionable_events=actionable_count,
            eligible_actionable_events=eligible_count,
            accepted_actionable_events=0,
        )

    candidates = values[opportunities & usable]
    candidates = np.unique(candidates[candidates > minimum_positive_score + 1e-15])
    thresholds = np.concatenate(([reject_all], candidates))
    best = None
    for threshold in thresholds:
        accepted = opportunities & usable & (values >= threshold)
        accepted_count = int(np.sum(accepted))
        coverage = accepted_count / actionable_count
        key = (abs(coverage - target_coverage), -float(threshold))
        if best is None or key < best[0]:
            best = (key, float(threshold), accepted_count, coverage)
    if best is None:  # defensive; reject_all always supplies one candidate
        raise RuntimeError("Unable to freeze a coverage threshold.")
    _, threshold, accepted_count, coverage = best
    return FrozenGateThreshold(
        threshold=threshold,
        target_coverage=float(target_coverage),
        achieved_actionable_coverage=float(coverage),
        actionable_events=actionable_count,
        eligible_actionable_events=eligible_count,
        accepted_actionable_events=accepted_count,
    )


def gate_accepts(
    evidence: TrustScore,
    threshold: float,
    *,
    actionable: bool,
) -> bool:
    """Return whether a gate executes the fixed source nomination."""

    return bool(
        actionable
        and evidence.eligible
        and evidence.score >= float(threshold)
    )


def tie_aware_top_fraction_mask(
    true_cost: Array,
    *,
    fraction: float,
    tolerance: float,
) -> Tuple[Array, int, float]:
    """Return tie-inclusive top-fraction membership for a minimization panel."""

    values = np.asarray(true_cost, dtype=float).reshape(-1)
    if len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("true_cost must be a non-empty finite vector.")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1].")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative.")
    nominal_count = max(1, int(np.ceil(fraction * len(values))))
    cutoff = float(np.sort(values, kind="stable")[nominal_count - 1])
    return values <= cutoff + float(tolerance), nominal_count, cutoff


def practical_outcome_tolerance(
    observed_target_cost: Array,
    *,
    relative_fraction: float,
    absolute_floor: float,
) -> float:
    """Compute an outcome tolerance using only already observed target history."""

    values = np.asarray(observed_target_cost, dtype=float).reshape(-1)
    if len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("observed_target_cost must be a non-empty finite vector.")
    if relative_fraction < 0.0 or absolute_floor < 0.0:
        raise ValueError("Tolerance settings must be non-negative.")
    scale = max(float(np.quantile(values, 0.90) - np.min(values)), 0.0)
    return float(max(absolute_floor, relative_fraction * scale))


def select_target_diagnostic_pair(
    acquisition: Array,
    *,
    alternative_rank: int,
) -> Tuple[int, int]:
    """Select a source-blind paid feedback pair using target acquisition only.

    The first member is Target-Only's acquisition maximizer. The second member is
    the candidate at the frozen one-based acquisition rank ``alternative_rank``.
    Both are evaluated in the diagnostic pilot, independently of source advice.
    """

    scores = np.asarray(acquisition, dtype=float).reshape(-1)
    if len(scores) < 2 or not np.all(np.isfinite(scores)):
        raise ValueError("acquisition must contain at least two finite values.")
    rank = int(alternative_rank)
    if rank < 2 or rank > len(scores):
        raise ValueError("alternative_rank must lie in [2, len(acquisition)].")
    order = np.argsort(-scores, kind="stable")
    return int(order[0]), int(order[rank - 1])


def nominate_source_candidate(
    acquisition: Array,
    source_quality: Array,
    *,
    top_k: int,
    relative_tolerance: float = 1e-10,
    absolute_tolerance: float = 1e-14,
) -> Tuple[int, int, Array]:
    """Create one common target choice and one strict source counterproposal.

    The source expert may nominate only inside the target acquisition's top ``K``.
    If target and source do not express strict opposite preferences, the source
    nomination is set equal to the target choice, yielding an auditable no-op.
    """

    target_score, quality = _paired_vectors(acquisition, source_quality)
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    order = np.argsort(-target_score, kind="stable")
    eligible = order[: min(int(top_k), len(order))]
    target_index = int(order[0])
    source_index = int(
        eligible[np.argmax(quality[eligible])]
    )
    if source_index == target_index:
        return target_index, target_index, eligible

    source_scale = max(
        1.0,
        abs(float(quality[source_index])),
        abs(float(quality[target_index])),
    )
    target_scale = max(
        1e-12,
        abs(float(target_score[source_index])),
        abs(float(target_score[target_index])),
    )
    source_margin = absolute_tolerance + relative_tolerance * source_scale
    target_margin = absolute_tolerance + relative_tolerance * target_scale
    source_strict = quality[source_index] > quality[target_index] + source_margin
    target_strict = (
        target_score[target_index] > target_score[source_index] + target_margin
    )
    if not (source_strict and target_strict):
        source_index = target_index
    return target_index, source_index, eligible


def fit_target_only_model(
    X: Array,
    y: Array,
    config: LocalSurrogateTransferConfig,
) -> LocalExpertResidualRegressor:
    """Fit the fixed target-only GP shared by every gate."""

    return LocalExpertResidualRegressor("target_only", config).fit(X, y)


def expected_improvement(
    predicted_mean: Array,
    predicted_std: Array,
    best_observed_cost: float,
) -> Array:
    """Expected improvement for minimization, with no method-specific tuning."""

    mean, std = _paired_vectors(predicted_mean, predicted_std)
    std = np.maximum(std, 1e-12)
    improvement = float(best_observed_cost) - mean
    z = improvement / std
    result = improvement * norm.cdf(z) + std * norm.pdf(z)
    return np.maximum(result, 0.0)


def fit_source_rank_expert(
    X: Array,
    y: Array,
    *,
    length_scale: float,
    noise: float,
    random_state: int,
) -> GaussianProcessRegressor:
    """Fit the frozen source-only local rank GP used by the controlled pilot."""

    points = np.asarray(X, dtype=float)
    values = np.asarray(y, dtype=float).reshape(-1)
    if points.ndim != 2 or len(points) != len(values) or len(points) < 3:
        raise ValueError("Source expert requires matching X/y with at least 3 rows.")
    if length_scale <= 0.0 or noise <= 0.0:
        raise ValueError("Source expert length_scale and noise must be positive.")
    kernel = (
        ConstantKernel(1.0, constant_value_bounds="fixed")
        * Matern(
            length_scale=np.full(points.shape[1], length_scale, dtype=float),
            length_scale_bounds="fixed",
            nu=2.5,
        )
        + WhiteKernel(noise_level=noise, noise_level_bounds="fixed")
    )
    model = GaussianProcessRegressor(
        kernel=kernel,
        optimizer=None,
        normalize_y=True,
        random_state=int(random_state),
    )
    model.fit(points, rank_quality(values))
    return model


def source_quality_prediction(
    model: GaussianProcessRegressor,
    X: Array,
) -> Array:
    """Return the source expert's clipped larger-is-better rank quality."""

    points = np.asarray(X, dtype=float)
    if points.ndim != 2 or len(points) == 0:
        raise ValueError("X must be a non-empty two-dimensional array.")
    prediction = np.asarray(model.predict(points), dtype=float).reshape(-1)
    return np.clip(prediction, 0.0, 1.0)


def rotation(theta: float) -> Array:
    """Two-dimensional rotation matrix for the controlled local relations."""

    cosine, sine = np.cos(theta), np.sin(theta)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=float)


def controlled_local_cost(
    Z: Array,
    *,
    theta: float,
    scale: float = 1.0,
    weights: Tuple[float, float] = (1.0, 0.35),
    ripple: float = 0.07,
    frequencies: Tuple[float, float] = (3.0, 2.0),
) -> Array:
    """Smooth but nontrivial two-dimensional local cost used in the pilot."""

    points = np.asarray(Z, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Z must have shape (n, 2).")
    transformed = scale * (points @ rotation(theta).T)
    quadratic = (
        weights[0] * transformed[:, 0] ** 2
        + weights[1] * transformed[:, 1] ** 2
    )
    oscillation = ripple * (
        1.0 - np.cos(frequencies[0] * np.pi * transformed[:, 0])
    ) + 0.5 * ripple * (
        1.0 - np.cos(frequencies[1] * np.pi * transformed[:, 1])
    )
    return quadratic + oscillation


def make_controlled_relation(
    relation: str,
    *,
    theta: float,
    independent_theta: float,
) -> Tuple[Objective, Objective]:
    """Return ``(source_expert_truth, target_truth)`` for one frozen relation."""

    source = lambda Z: controlled_local_cost(Z, theta=theta)

    def independent(Z: Array) -> Array:
        points = np.asarray(Z, dtype=float)
        transformed = points @ rotation(independent_theta).T
        phase = 0.7 * independent_theta
        return (
            np.sin(3.3 * np.pi * transformed[:, 0] + phase)
            + 0.8 * np.cos(4.1 * np.pi * transformed[:, 1] - phase)
            + 0.35
            * np.sin(2.0 * np.pi * (transformed[:, 0] + transformed[:, 1]))
        )

    if relation == "identity":
        return source, source
    if relation == "output_affine":
        return source, lambda Z: 4.0 + 2.5 * source(Z)
    if relation == "scale_0.7":
        return source, lambda Z: controlled_local_cost(Z, theta=theta, scale=0.7)
    if relation == "scale_1.5":
        return source, lambda Z: controlled_local_cost(Z, theta=theta, scale=1.5)
    if relation == "rotate_45":
        return source, lambda Z: controlled_local_cost(
            Z,
            theta=theta + np.pi / 4.0,
        )
    if relation == "anisotropy_swap":
        return source, lambda Z: controlled_local_cost(
            Z,
            theta=theta,
            weights=(0.35, 1.0),
        )
    if relation == "roughness":
        return source, lambda Z: controlled_local_cost(
            Z,
            theta=theta,
            ripple=0.28,
            frequencies=(7.0, 5.0),
        )
    if relation == "reversal":
        return source, lambda Z: -source(Z)
    if relation == "independent_expert":
        return independent, source
    raise ValueError(f"Unknown controlled relation: {relation}")


def _shrunken_spearman(
    first: Array,
    second: Array,
    *,
    min_points: int,
    shrinkage: float,
    label: str,
) -> TrustScore:
    if min_points < 3:
        raise ValueError("min_points must be at least 3.")
    if shrinkage < 0.0:
        raise ValueError("shrinkage must be non-negative.")
    count = len(first)
    if count < min_points:
        return TrustScore(
            score=0.5,
            raw_statistic=0.0,
            eligible=False,
            evidence_count=count,
            reason=f"{label}:insufficient_points",
        )
    if np.std(first) < 1e-12 or np.std(second) < 1e-12:
        return TrustScore(
            score=0.5,
            raw_statistic=0.0,
            eligible=False,
            evidence_count=count,
            reason=f"{label}:constant_input",
        )
    correlation = spearmanr(first, second).statistic
    if not np.isfinite(correlation):
        return TrustScore(
            score=0.5,
            raw_statistic=0.0,
            eligible=False,
            evidence_count=count,
            reason=f"{label}:undefined",
        )
    weight = count / (count + shrinkage + 1e-12)
    score = float(np.clip(0.5 + 0.5 * float(correlation) * weight, 0.0, 1.0))
    return TrustScore(
        score=score,
        raw_statistic=float(correlation),
        eligible=True,
        evidence_count=count,
        reason="eligible",
    )


def _paired_vectors(first: Array, second: Array) -> Tuple[Array, Array]:
    first_values = np.asarray(first, dtype=float).reshape(-1)
    second_values = np.asarray(second, dtype=float).reshape(-1)
    if (
        len(first_values) == 0
        or len(first_values) != len(second_values)
        or not np.all(np.isfinite(first_values))
        or not np.all(np.isfinite(second_values))
    ):
        raise ValueError("Paired vectors must have equal, non-zero finite length.")
    return first_values, second_values


__all__ = [
    "FrozenGateThreshold",
    "TrustScore",
    "controlled_local_cost",
    "disagreement_correction_trust",
    "expected_improvement",
    "fit_source_rank_expert",
    "fit_target_only_model",
    "freeze_coverage_threshold",
    "gate_accepts",
    "local_spearman_trust",
    "make_controlled_relation",
    "nominate_source_candidate",
    "paired_margin_spearman_trust",
    "practical_outcome_tolerance",
    "prediction_score",
    "residual_spearman_trust",
    "rotation",
    "select_target_diagnostic_pair",
    "source_quality_prediction",
    "tie_aware_top_fraction_mask",
]
