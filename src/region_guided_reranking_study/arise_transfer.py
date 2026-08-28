"""Active Region Identification and Safe Exploitation (ARISE-BO).

ARISE-BO replaces static/global source-target similarity with a region-wise,
decision-conditional transferability posterior.

For every expensive target evaluation, the target-only surrogate provides a
pre-evaluation distribution of improvement.  After the true target value is
observed, ARISE records the standardized *excess improvement*:

    residual = (realized_improvement - expected_improvement)
               / sqrt(variance_of_improvement + floor)

A source local region is transferable only when its support repeatedly predicts
positive excess improvement beyond what the target-only surrogate already
expected.  A Bayesian linear model maintains one signed effect posterior per
source region.  The optimizer then alternates among:

* target-only fallback;
* safe exploitation of regions whose lower credible bound is positive;
* active probing of uncertain regions whose upper credible bound remains
  promising.

The source knowledge never trains the target surrogate and never creates raw
candidates.  It acts only after target-only proposal generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Literal, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import norm, rankdata, spearmanr

from .source_regions import SourceRegionExtractor, SourceRegionLibrary
from .surrogate_and_candidates import TargetGPSurrogate
from .target_region_screening import TargetCandidateProposer, TargetProposalConfig, TargetProposalSet

Array = np.ndarray
Objective = Callable[[Array], Array]
SourceDataset = Tuple[Array, Array]
ARISEPolicy = Literal["target_only", "fixed", "global_adaptive", "posterior", "arise"]
RegionStatus = Literal["trusted", "uncertain", "rejected"]
DecisionMode = Literal["target", "fixed", "global", "exploit", "probe"]


@dataclass(frozen=True)
class ARISEConfig:
    """Configuration for ARISE-BO.

    ``policy`` controls the ablation:

    * ``target_only``: source regions are observed diagnostically but never used;
    * ``fixed``: all source regions guide every decision at fixed strength;
    * ``global_adaptive``: the previous global Spearman/enrichment rule;
    * ``posterior``: exploit only statistically trusted regions, without probes;
    * ``arise``: full trusted exploitation + active uncertainty probing.
    """

    policy: ARISEPolicy = "arise"

    top_ratio: float = 0.20
    max_clusters: int = 3
    min_samples_per_cluster: int = 3
    merge_threshold: float = 0.5
    proposal: TargetProposalConfig = field(default_factory=TargetProposalConfig)

    prior_effect_variance: float = 1.0
    intercept_prior_variance: float = 100.0
    credible_z: float = 1.2815515655446004  # one-sided 90% bound
    min_region_coverage: float = 0.75
    trust_effect_threshold: float = 0.0
    reject_effect_threshold: float = 0.0
    support_update_floor: float = 1e-3

    improvement_xi: float = 0.01
    residual_variance_floor: float = 0.05
    residual_clip: float = 6.0
    evidence_decay: float = 0.95

    guidance_weight: float = 0.80
    fixed_guidance_weight: float = 0.80
    probe_weight: float = 0.25
    exploit_acquisition_gate: float = 0.50
    probe_acquisition_gate: float = 0.25
    region_candidate_weight: float = 1.0

    active_probe: bool = True
    probe_interval: int = 3
    probe_horizon: int = 15
    probe_ucb_threshold: float = 0.0

    global_elite_ratio: float = 0.30
    global_prior_trust: float = 0.20
    global_prior_strength: float = 2.0
    global_evidence_shrinkage: float = 8.0
    global_activation_threshold: float = 0.05

    random_state: int = 42

    def __post_init__(self) -> None:
        if self.policy not in {"target_only", "fixed", "global_adaptive", "posterior", "arise"}:
            raise ValueError("Unsupported ARISE policy.")
        if not 0.0 < self.top_ratio <= 1.0:
            raise ValueError("top_ratio must lie in (0, 1].")
        if self.max_clusters < 1:
            raise ValueError("max_clusters must be positive.")
        if self.min_samples_per_cluster < 2:
            raise ValueError("min_samples_per_cluster must be at least 2.")
        if self.merge_threshold < 0.0:
            raise ValueError("merge_threshold must be non-negative.")
        if self.prior_effect_variance <= 0.0:
            raise ValueError("prior_effect_variance must be positive.")
        if self.intercept_prior_variance <= 0.0:
            raise ValueError("intercept_prior_variance must be positive.")
        if self.credible_z < 0.0:
            raise ValueError("credible_z must be non-negative.")
        if self.min_region_coverage < 0.0:
            raise ValueError("min_region_coverage must be non-negative.")
        if self.support_update_floor < 0.0:
            raise ValueError("support_update_floor must be non-negative.")
        if self.improvement_xi < 0.0:
            raise ValueError("improvement_xi must be non-negative.")
        if self.residual_variance_floor <= 0.0:
            raise ValueError("residual_variance_floor must be positive.")
        if self.residual_clip <= 0.0:
            raise ValueError("residual_clip must be positive.")
        if not 0.0 < self.evidence_decay <= 1.0:
            raise ValueError("evidence_decay must lie in (0, 1].")
        for name, value in {
            "guidance_weight": self.guidance_weight,
            "fixed_guidance_weight": self.fixed_guidance_weight,
            "probe_weight": self.probe_weight,
            "region_candidate_weight": self.region_candidate_weight,
        }.items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative.")
        for name, value in {
            "exploit_acquisition_gate": self.exploit_acquisition_gate,
            "probe_acquisition_gate": self.probe_acquisition_gate,
        }.items():
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1].")
        if self.probe_interval < 1:
            raise ValueError("probe_interval must be at least 1.")
        if self.probe_horizon < 0:
            raise ValueError("probe_horizon must be non-negative.")
        if not 0.0 < self.global_elite_ratio < 0.5:
            raise ValueError("global_elite_ratio must lie in (0, 0.5).")
        if not 0.0 <= self.global_prior_trust <= 1.0:
            raise ValueError("global_prior_trust must lie in [0, 1].")
        if self.global_prior_strength < 0.0 or self.global_evidence_shrinkage < 0.0:
            raise ValueError("global prior/shrinkage parameters must be non-negative.")
        if not 0.0 <= self.global_activation_threshold <= 1.0:
            raise ValueError("global_activation_threshold must lie in [0, 1].")


@dataclass(frozen=True)
class ImprovementMoments:
    expected: float
    variance: float
    predictive_mean: float
    predictive_std: float
    y_best_before: float


@dataclass(frozen=True)
class RegionTransferPosterior:
    region_index: int
    source_task_id: Optional[str]
    mean: float
    std: float
    lower_bound: float
    upper_bound: float
    probability_positive: float
    coverage: float
    status: RegionStatus


@dataclass
class ARISEDecision:
    point: Array
    selected_index: int
    target_top1_index: int
    selected_target_rank: int
    selected_region_index: Optional[int]
    mode: DecisionMode
    proposal_set: TargetProposalSet
    acquisition_normalized: Array
    support_matrix: Array
    support_normalized: Array
    trusted_guidance: Array
    probe_guidance: Array
    combined_scores: Array
    region_candidate_indices: Array
    posteriors: List[RegionTransferPosterior]
    improvement_forecast: ImprovementMoments
    selected_supports: Array
    global_compatibility_trust: float
    iteration: int


@dataclass
class ARISEResult:
    X: Array
    y: Array
    best_x: Array
    best_y: float
    best_y_trace: List[float]
    decisions: List[ARISEDecision] = field(default_factory=list)
    residual_trace: List[float] = field(default_factory=list)


class RegionEvidenceModel:
    """Bayesian region-effect regression on standardized excess improvement."""

    def __init__(self, n_regions: int, config: ARISEConfig) -> None:
        self.n_regions = int(n_regions)
        self.config = config
        self._supports: List[Array] = []
        self._residuals: List[float] = []
        self._weights: List[float] = []

    @property
    def n_observations(self) -> int:
        return len(self._residuals)

    def update(self, supports: Array, residual: float) -> None:
        vector = np.asarray(supports, dtype=float).reshape(-1)
        if len(vector) != self.n_regions:
            raise ValueError("support vector dimension does not match region count.")
        vector = np.where(
            vector >= self.config.support_update_floor,
            vector,
            0.0,
        )
        clipped = float(np.clip(residual, -self.config.residual_clip, self.config.residual_clip))
        self._weights = [weight * self.config.evidence_decay for weight in self._weights]
        self._supports.append(vector)
        self._residuals.append(clipped)
        self._weights.append(1.0)

    def _posterior_parameters(self) -> Tuple[Array, Array, Array]:
        k = self.n_regions
        if k == 0:
            return np.empty(0), np.empty(0), np.empty(0)

        prior_precision = np.empty(k + 1, dtype=float)
        prior_precision[0] = 1.0 / self.config.intercept_prior_variance
        prior_precision[1:] = 1.0 / self.config.prior_effect_variance

        if not self._residuals:
            covariance = np.diag(1.0 / prior_precision)
            mean = np.zeros(k + 1, dtype=float)
            coverage = np.zeros(k, dtype=float)
            return mean[1:], covariance[1:, 1:], coverage

        Z = np.vstack(self._supports)
        y = np.asarray(self._residuals, dtype=float)
        weights = np.asarray(self._weights, dtype=float)
        X = np.column_stack([np.ones(len(Z)), Z])
        weighted_X = X * weights[:, None]
        precision = X.T @ weighted_X + np.diag(prior_precision)
        try:
            covariance = np.linalg.inv(precision)
        except np.linalg.LinAlgError:
            covariance = np.linalg.pinv(precision)
        mean = covariance @ X.T @ (weights * y)
        coverage = np.sum(weights[:, None] * Z * Z, axis=0)
        return mean[1:], covariance[1:, 1:], coverage

    def predictive_effect_variance(self, support_matrix: Array) -> Array:
        """Posterior variance of the local-transfer effect at candidate supports."""

        support = np.asarray(support_matrix, dtype=float)
        if support.ndim != 2 or support.shape[1] != self.n_regions:
            raise ValueError("support_matrix has an invalid shape.")
        if self.n_regions == 0:
            return np.zeros(len(support), dtype=float)
        _, covariance, _ = self._posterior_parameters()
        variance = np.einsum("ij,jk,ik->i", support, covariance, support)
        return np.maximum(variance, 0.0)

    def snapshot(self, library: SourceRegionLibrary) -> List[RegionTransferPosterior]:
        means, covariance, coverage = self._posterior_parameters()
        variances = np.maximum(np.diag(covariance), 1e-12)
        result: List[RegionTransferPosterior] = []
        for k, region in enumerate(library.regions):
            std = float(np.sqrt(variances[k]))
            mean = float(means[k])
            lower = mean - self.config.credible_z * std
            upper = mean + self.config.credible_z * std
            p_positive = float(norm.cdf(mean / max(std, 1e-12)))

            if coverage[k] >= self.config.min_region_coverage and lower > self.config.trust_effect_threshold:
                status: RegionStatus = "trusted"
            elif coverage[k] >= self.config.min_region_coverage and upper < self.config.reject_effect_threshold:
                status = "rejected"
            else:
                status = "uncertain"

            result.append(
                RegionTransferPosterior(
                    region_index=k,
                    source_task_id=region.source_task_id,
                    mean=mean,
                    std=std,
                    lower_bound=float(lower),
                    upper_bound=float(upper),
                    probability_positive=p_positive,
                    coverage=float(coverage[k]),
                    status=status,
                )
            )
        return result


class ARISERegionTransferBO:
    """Target-proposal BO with online region identification and safe guidance."""

    def __init__(
        self,
        bounds: Array,
        config: Optional[ARISEConfig] = None,
    ) -> None:
        self.bounds = _validate_bounds(bounds)
        self.dim = int(self.bounds.shape[0])
        self.config = config or ARISEConfig()

        seed_sequence = np.random.SeedSequence(self.config.random_state)
        proposal_seed, mode_seed = seed_sequence.spawn(2)
        self._proposal_rng = np.random.default_rng(proposal_seed)
        self._mode_rng = np.random.default_rng(mode_seed)
        self._proposer = TargetCandidateProposer(
            self.bounds,
            self.config.proposal,
            rng=self._proposal_rng,
        )

        self.source_regions = SourceRegionLibrary()
        self.evidence = RegionEvidenceModel(0, self.config)
        self.target_X: Optional[Array] = None
        self.target_y: Optional[Array] = None
        self.iteration = 0
        self.last_decision: Optional[ARISEDecision] = None
        self.residual_trace: List[float] = []

    def fit_source_regions(
        self,
        source_datasets: Sequence[SourceDataset],
        task_ids: Optional[Sequence[str]] = None,
    ) -> SourceRegionLibrary:
        datasets = list(source_datasets)
        if not datasets:
            raise ValueError("source_datasets must contain at least one dataset.")
        if task_ids is not None and len(task_ids) != len(datasets):
            raise ValueError("task_ids must match source_datasets in length.")

        validated: List[SourceDataset] = []
        for idx, (X, y) in enumerate(datasets):
            X_arr = _validate_points(X, self.bounds, name=f"source[{idx}].X")
            y_arr = _validate_values(y, len(X_arr), name=f"source[{idx}].y")
            validated.append((X_arr, y_arr))

        extractor = SourceRegionExtractor(
            top_ratio=self.config.top_ratio,
            max_clusters=self.config.max_clusters,
            min_samples_per_cluster=self.config.min_samples_per_cluster,
            random_state=self.config.random_state,
        )
        library = extractor.extract_from_multi_sources(
            validated,
            task_ids=list(task_ids) if task_ids is not None else None,
            merge_threshold=self.config.merge_threshold,
        )
        self.set_source_region_library(library)
        return library

    def set_source_region_library(self, library: SourceRegionLibrary) -> None:
        for idx, region in enumerate(library.regions):
            if len(region.center) != self.dim:
                raise ValueError(f"Region {idx} has the wrong dimension.")
        self.source_regions = library
        self.evidence = RegionEvidenceModel(len(library.regions), self.config)
        self.residual_trace = []

    def initialize_target(self, X: Array, y: Array) -> None:
        X_arr = _validate_points(X, self.bounds, name="target_X")
        y_arr = _validate_values(y, len(X_arr), name="target_y")
        if len(X_arr) < 2:
            raise ValueError("At least two initial target points are required.")
        self.target_X = X_arr
        self.target_y = y_arr
        self.iteration = 0
        self.last_decision = None
        self.evidence = RegionEvidenceModel(len(self.source_regions.regions), self.config)
        self.residual_trace = []

    @property
    def is_initialized(self) -> bool:
        return self.target_X is not None and self.target_y is not None

    def get_region_posteriors(self) -> List[RegionTransferPosterior]:
        return self.evidence.snapshot(self.source_regions)

    def _support_matrix(self, points: Array) -> Array:
        if not self.source_regions.regions:
            return np.zeros((len(points), 0), dtype=float)
        columns = [region.compute_support(points) for region in self.source_regions.regions]
        return np.column_stack(columns)

    def _normalized_support(self, support: Array) -> Array:
        if support.shape[1] == 0:
            return support.copy()
        columns = []
        for k in range(support.shape[1]):
            column = support[:, k]
            # A region that has negligible absolute support on the current target
            # proposal set is unavailable, even if tiny numerical differences can
            # still be ranked.  This prevents high-dimensional underflow from
            # creating false region evidence.
            if np.max(column) < self.config.support_update_floor or np.ptp(column) < 1e-14:
                columns.append(np.zeros_like(column))
            else:
                columns.append(_rank_normalize(column))
        return np.column_stack(columns)

    @staticmethod
    def _gate_mask(acquisition_normalized: Array, ratio: float) -> Array:
        n = len(acquisition_normalized)
        count = max(1, int(np.ceil(n * ratio)))
        order = np.argsort(-acquisition_normalized, kind="stable")
        mask = np.zeros(n, dtype=bool)
        mask[order[:count]] = True
        return mask

    def _region_candidate_indices(
        self,
        acquisition_normalized: Array,
        support_normalized: Array,
    ) -> Array:
        if support_normalized.shape[1] == 0:
            return np.empty(0, dtype=int)
        gate = self._gate_mask(acquisition_normalized, self.config.exploit_acquisition_gate)
        result = []
        for k in range(support_normalized.shape[1]):
            score = acquisition_normalized + self.config.region_candidate_weight * support_normalized[:, k]
            score = np.where(gate, score, -np.inf)
            result.append(int(np.argmax(score)))
        return np.asarray(result, dtype=int)

    def _guidance_vectors(
        self,
        posteriors: Sequence[RegionTransferPosterior],
        support_normalized: Array,
    ) -> Tuple[Array, Array]:
        n = support_normalized.shape[0]
        if support_normalized.shape[1] == 0:
            return np.zeros(n), np.zeros(n)

        trusted_weights = np.asarray(
            [max(0.0, item.lower_bound) if item.status == "trusted" else 0.0 for item in posteriors],
            dtype=float,
        )
        uncertain_mask = np.asarray(
            [
                item.status == "uncertain"
                and item.upper_bound > self.config.probe_ucb_threshold
                for item in posteriors
            ],
            dtype=bool,
        )

        trusted = support_normalized @ trusted_weights
        if np.any(uncertain_mask):
            probe_support = support_normalized * uncertain_mask[None, :]
            effect_variance = self.evidence.predictive_effect_variance(probe_support)
            information_gain = 0.5 * np.log1p(effect_variance)
            positive_weights = np.asarray(
                [item.probability_positive if flag else 0.0 for item, flag in zip(posteriors, uncertain_mask)],
                dtype=float,
            )
            optimistic_overlap = probe_support @ positive_weights
            probe = information_gain * (0.5 + 0.5 * _rank_normalize(optimistic_overlap))
        else:
            probe = np.zeros(n, dtype=float)
        return _rank_normalize(trusted), _rank_normalize(probe)

    def _global_compatibility(self) -> float:
        """Previous global similarity baseline: Spearman + elite enrichment.

        This intentionally reproduces the main limitation studied by ARISE: all
        regions are collapsed into one scalar and negative evidence is clipped.
        """

        if not self.is_initialized or not self.source_regions.regions:
            return 0.0
        assert self.target_X is not None
        assert self.target_y is not None
        support = np.asarray(self.source_regions.score(self.target_X, aggregation="max"), dtype=float)
        if np.ptp(support) < 1e-12 or np.var(support) < 1e-12:
            return 0.0
        rho = 0.0
        if len(support) >= 3 and np.std(self.target_y) > 1e-12:
            value = spearmanr(support, -self.target_y).statistic
            if np.isfinite(value):
                rho = float(value)
        support_rank = _rank_normalize(support)
        elite_count = max(1, int(np.ceil(len(support) * self.config.global_elite_ratio)))
        elite_idx = np.argsort(self.target_y, kind="stable")[:elite_count]
        rest = np.ones(len(support), dtype=bool)
        rest[elite_idx] = False
        enrichment = (
            float(np.mean(support_rank[elite_idx]) - np.mean(support_rank[rest]))
            if np.any(rest)
            else 0.0
        )
        raw = float(np.clip(0.5 * max(0.0, rho) + 0.5 * max(0.0, enrichment), 0.0, 1.0))
        n = len(support)
        data_weight = n / (n + self.config.global_evidence_shrinkage + 1e-12)
        prior_weight = self.config.global_prior_strength / (self.config.global_prior_strength + n + 1e-12)
        return float(np.clip(prior_weight * self.config.global_prior_trust + data_weight * raw, 0.0, 1.0))

    def _choose_mode(
        self,
        posteriors: Sequence[RegionTransferPosterior],
        trusted_guidance: Array,
        probe_guidance: Array,
    ) -> DecisionMode:
        if self.config.policy == "target_only" or not self.source_regions.regions:
            return "target"
        if self.config.policy == "fixed":
            return "fixed"
        if self.config.policy == "global_adaptive":
            return "target"

        has_trusted = any(item.status == "trusted" for item in posteriors) and np.ptp(trusted_guidance) > 1e-14
        if self.config.policy == "posterior":
            return "exploit" if has_trusted else "target"

        has_probe = (
            self.config.active_probe
            and self.iteration < self.config.probe_horizon
            and np.ptp(probe_guidance) > 1e-14
            and any(
                item.status == "uncertain" and item.upper_bound > self.config.probe_ucb_threshold
                for item in posteriors
            )
        )
        # Probe only on the configured schedule.  When no region is trusted,
        # non-probe iterations fall back to target-only BO instead of spending
        # the entire early budget on similarity identification.
        scheduled_probe = (
            has_probe and self.iteration % self.config.probe_interval == 0
        )
        if scheduled_probe:
            return "probe"
        if has_trusted:
            return "exploit"
        return "target"

    def ask(self) -> ARISEDecision:
        if not self.is_initialized:
            raise RuntimeError("Call initialize_target before ask.")
        assert self.target_X is not None
        assert self.target_y is not None

        surrogate = TargetGPSurrogate(
            dim=self.dim,
            noise_level=self.config.proposal.gp_noise_level,
            random_state=self.config.random_state + self.iteration,
        )
        surrogate.fit(self.target_X, self.target_y)
        proposal_set = self._proposer.propose(
            surrogate,
            current_X=self.target_X,
            iteration=self.iteration,
        )

        acquisition = np.asarray(proposal_set.acquisition_scores, dtype=float)
        acquisition_normalized = _rank_normalize(acquisition)
        support = self._support_matrix(proposal_set.points)
        support_normalized = self._normalized_support(support)
        posteriors = self.get_region_posteriors()
        trusted_guidance, probe_guidance = self._guidance_vectors(posteriors, support_normalized)
        global_trust = self._global_compatibility()
        if self.config.policy == "global_adaptive":
            has_global_signal = support_normalized.shape[1] > 0 and np.ptp(np.max(support_normalized, axis=1)) > 1e-14
            mode: DecisionMode = (
                "global"
                if has_global_signal and global_trust >= self.config.global_activation_threshold
                else "target"
            )
        else:
            mode = self._choose_mode(posteriors, trusted_guidance, probe_guidance)

        if mode == "target":
            combined = acquisition_normalized.copy()
        elif mode == "fixed":
            fixed = (
                np.max(support_normalized, axis=1)
                if support_normalized.shape[1]
                else np.zeros(len(acquisition_normalized))
            )
            combined = acquisition_normalized + self.config.fixed_guidance_weight * fixed
            gate = self._gate_mask(acquisition_normalized, self.config.exploit_acquisition_gate)
            combined = np.where(gate, combined, -np.inf)
        elif mode == "global":
            global_guidance = (
                np.max(support_normalized, axis=1)
                if support_normalized.shape[1]
                else np.zeros(len(acquisition_normalized))
            )
            combined = acquisition_normalized + self.config.guidance_weight * global_trust * global_guidance
            gate = self._gate_mask(acquisition_normalized, self.config.exploit_acquisition_gate)
            combined = np.where(gate, combined, -np.inf)
        elif mode == "exploit":
            combined = acquisition_normalized + self.config.guidance_weight * trusted_guidance
            gate = self._gate_mask(acquisition_normalized, self.config.exploit_acquisition_gate)
            combined = np.where(gate, combined, -np.inf)
        else:
            combined = acquisition_normalized + self.config.probe_weight * probe_guidance
            gate = self._gate_mask(acquisition_normalized, self.config.probe_acquisition_gate)
            combined = np.where(gate, combined, -np.inf)

        selected = int(np.argmax(combined))
        target_top1 = int(np.argmax(acquisition))
        target_order = np.argsort(-acquisition, kind="stable")
        target_rank = int(np.where(target_order == selected)[0][0])

        selected_region: Optional[int] = None
        if support_normalized.shape[1] > 0 and mode != "target":
            if mode == "exploit":
                weights = np.asarray(
                    [max(0.0, item.lower_bound) if item.status == "trusted" else 0.0 for item in posteriors]
                )
            elif mode == "probe":
                weights = np.asarray(
                    [
                        item.std * max(item.probability_positive, 0.25)
                        if item.status == "uncertain" and item.upper_bound > self.config.probe_ucb_threshold else 0.0
                        for item in posteriors
                    ]
                )
            else:
                weights = np.ones(support_normalized.shape[1])
            contribution = support_normalized[selected] * weights
            if np.max(contribution) > 0.0:
                selected_region = int(np.argmax(contribution))

        mu, std = surrogate.predict(proposal_set.points[selected:selected + 1], return_std=True)
        y_best = float(np.min(self.target_y))
        expected, variance = improvement_moments(
            float(mu[0]),
            float(std[0]),
            y_best,
            xi=self.config.improvement_xi,
        )
        forecast = ImprovementMoments(
            expected=expected,
            variance=variance,
            predictive_mean=float(mu[0]),
            predictive_std=float(std[0]),
            y_best_before=y_best,
        )

        decision = ARISEDecision(
            point=proposal_set.points[selected:selected + 1].copy(),
            selected_index=selected,
            target_top1_index=target_top1,
            selected_target_rank=target_rank,
            selected_region_index=selected_region,
            mode=mode,
            proposal_set=proposal_set,
            acquisition_normalized=acquisition_normalized,
            support_matrix=support,
            support_normalized=support_normalized,
            trusted_guidance=trusted_guidance,
            probe_guidance=probe_guidance,
            combined_scores=combined,
            region_candidate_indices=self._region_candidate_indices(
                acquisition_normalized,
                support_normalized,
            ),
            posteriors=list(posteriors),
            improvement_forecast=forecast,
            selected_supports=(support_normalized[selected].copy() if support_normalized.shape[1] else np.empty(0)),
            global_compatibility_trust=global_trust,
            iteration=self.iteration,
        )
        self.last_decision = decision
        return decision

    def tell(self, X: Array, y: Array) -> float:
        if not self.is_initialized:
            raise RuntimeError("Call initialize_target before tell.")
        if self.last_decision is None:
            raise RuntimeError("Call ask before tell.")
        assert self.target_X is not None
        assert self.target_y is not None

        X_arr = _validate_points(X, self.bounds, name="new_target_X")
        y_arr = _validate_values(y, len(X_arr), name="new_target_y")
        if len(X_arr) != 1:
            raise ValueError("ARISE currently supports one expensive evaluation per tell call.")
        if not np.allclose(X_arr[0], self.last_decision.point[0], atol=1e-12, rtol=0.0):
            raise ValueError("tell point must match the most recent ask decision.")

        forecast = self.last_decision.improvement_forecast
        realized = max(
            0.0,
            forecast.y_best_before - float(y_arr[0]) - self.config.improvement_xi,
        )
        denominator = np.sqrt(
            max(forecast.variance, 0.0) + self.config.residual_variance_floor ** 2
        )
        residual = float((realized - forecast.expected) / denominator)
        self.evidence.update(self.last_decision.selected_supports, residual)
        self.residual_trace.append(residual)

        if np.min(np.linalg.norm(self.target_X - X_arr[0], axis=1)) < 1e-12:
            raise ValueError("tell received an already evaluated target point.")
        self.target_X = np.vstack([self.target_X, X_arr])
        self.target_y = np.concatenate([self.target_y, y_arr])
        self.iteration += 1
        self.last_decision = None
        return residual

    def get_best(self) -> Tuple[Array, float]:
        if not self.is_initialized:
            raise RuntimeError("Target data are not initialized.")
        assert self.target_X is not None
        assert self.target_y is not None
        idx = int(np.argmin(self.target_y))
        return self.target_X[idx].copy(), float(self.target_y[idx])

    def optimize(
        self,
        objective: Objective,
        init_X: Array,
        init_y: Optional[Array] = None,
        budget: int = 20,
    ) -> ARISEResult:
        if budget < 0:
            raise ValueError("budget must be non-negative.")
        initial_X = _validate_points(init_X, self.bounds, name="init_X")
        initial_y = (
            _evaluate_objective(objective, initial_X)
            if init_y is None
            else _validate_values(init_y, len(initial_X), name="init_y")
        )
        self.initialize_target(initial_X, initial_y)

        trace = [float(np.min(initial_y))]
        decisions: List[ARISEDecision] = []
        for _ in range(int(budget)):
            decision = self.ask()
            value = _evaluate_objective(objective, decision.point)
            self.tell(decision.point, value)
            decisions.append(decision)
            trace.append(min(trace[-1], float(value[0])))

        assert self.target_X is not None
        assert self.target_y is not None
        best_x, best_y = self.get_best()
        return ARISEResult(
            X=self.target_X.copy(),
            y=self.target_y.copy(),
            best_x=best_x,
            best_y=best_y,
            best_y_trace=trace,
            decisions=decisions,
            residual_trace=list(self.residual_trace),
        )


def improvement_moments(
    predictive_mean: float,
    predictive_std: float,
    y_best: float,
    *,
    xi: float = 0.01,
) -> Tuple[float, float]:
    """Mean and variance of one-step improvement under a Gaussian posterior."""

    sigma = max(float(predictive_std), 0.0)
    d = float(y_best - predictive_mean - xi)
    if sigma < 1e-12:
        return max(0.0, d), 0.0
    z = d / sigma
    cdf = float(norm.cdf(z))
    pdf = float(norm.pdf(z))
    mean = d * cdf + sigma * pdf
    second = (d * d + sigma * sigma) * cdf + d * sigma * pdf
    variance = max(0.0, second - mean * mean)
    return float(max(0.0, mean)), float(variance)


def counterfactual_region_gains(
    decision: ARISEDecision,
    true_y: Array,
) -> Array:
    """Offline synthetic-benchmark label: true gain of each region expert.

    Positive values mean the region-specific candidate beats the target-only
    acquisition top-1 on the same target-generated proposal set.
    """

    values = np.asarray(true_y, dtype=float).reshape(-1)
    if len(values) != len(decision.proposal_set.points):
        raise ValueError("true_y must match the proposal-set size.")
    baseline = float(values[decision.target_top1_index])
    return np.asarray(
        [baseline - float(values[idx]) for idx in decision.region_candidate_indices],
        dtype=float,
    )


def _validate_bounds(bounds: Array) -> Array:
    array = np.asarray(bounds, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("bounds must have shape (dimension, 2).")
    if not np.all(np.isfinite(array)) or np.any(array[:, 0] >= array[:, 1]):
        raise ValueError("bounds are invalid.")
    return array.copy()


def _validate_points(X: Array, bounds: Array, *, name: str) -> Array:
    array = np.asarray(X, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != bounds.shape[0] or len(array) == 0:
        raise ValueError(f"{name} has an invalid shape.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    if np.any(array < bounds[:, 0]) or np.any(array > bounds[:, 1]):
        raise ValueError(f"{name} contains points outside bounds.")
    return array.copy()


def _validate_values(y: Array, n: int, *, name: str) -> Array:
    array = np.asarray(y, dtype=float).reshape(-1)
    if len(array) != n or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {n} finite values.")
    return array.copy()


def _rank_normalize(values: Array) -> Array:
    array = np.asarray(values, dtype=float).reshape(-1)
    if len(array) <= 1 or np.ptp(array) < 1e-14:
        return np.zeros_like(array)
    ranks = rankdata(array, method="average")
    return (ranks - 1.0) / (len(array) - 1.0)


def _evaluate_objective(objective: Objective, X: Array) -> Array:
    values = np.asarray(objective(X), dtype=float)
    if values.ndim == 0:
        values = values.reshape(1)
    values = values.reshape(-1)
    if len(values) != len(X) or not np.all(np.isfinite(values)):
        raise ValueError("objective must return one finite scalar per input point.")
    return values
