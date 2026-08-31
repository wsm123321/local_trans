"""Run the frozen shared-candidate target-feedback trust pilot.

Phase A persists every non-oracle score and decision before final candidate-panel
truth is evaluated.  Phase B reconstructs the controlled target function, reveals
that panel, and evaluates exactly five shared-interface methods.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from region_guided_reranking_study.disagreement_trust import (  # noqa: E402
    FrozenGateThreshold,
    TrustScore,
    disagreement_correction_trust,
    expected_improvement,
    fit_source_rank_expert,
    fit_target_only_model,
    freeze_coverage_threshold,
    make_controlled_relation,
    nominate_source_candidate,
    paired_margin_spearman_trust,
    practical_outcome_tolerance,
    prediction_score,
    select_target_diagnostic_pair,
    source_quality_prediction,
    tie_aware_top_fraction_mask,
)
from region_guided_reranking_study.local_surrogate_transfer import (  # noqa: E402
    LocalSurrogateTransferConfig,
)
from region_guided_reranking_study.local_surrogate_transfer_research import (  # noqa: E402
    sobol_chart_design,
)

Array = np.ndarray
DEPLOYABLE_GATES = {
    "Local Spearman Gate": "local_spearman",
    "Target-Residual Spearman Gate": "residual_spearman",
    "Disagreement-Correction Gate": "disagreement_correction",
}
METHODS = [
    "Target-Only",
    "Local Spearman Gate",
    "Target-Residual Spearman Gate",
    "Disagreement-Correction Gate",
    "Oracle Gate",
]


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object.")
    return config


def model_config(config: Mapping[str, Any], random_state: int) -> LocalSurrogateTransferConfig:
    values = dict(config["target_model"])
    values["random_state"] = int(random_state)
    return LocalSurrogateTransferConfig(**values)


def stable_seed(seed: int, *components: int) -> int:
    sequence = np.random.SeedSequence([int(seed), 20260831, *map(int, components)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def rounded_array_hash(*arrays: Array) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(np.round(np.asarray(value, dtype=float), 12))
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_config_hash(config: Mapping[str, Any]) -> str:
    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_identity() -> Dict[str, Any]:
    def capture(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        return completed.stdout.strip()

    try:
        head = capture("rev-parse", "HEAD")
        status = capture("status", "--short")
        branch = capture("branch", "--show-current")
    except (OSError, subprocess.CalledProcessError) as error:
        return {
            "available": False,
            "error": f"{type(error).__name__}: {error}",
        }
    return {
        "available": True,
        "head": head,
        "branch": branch,
        "status_before_output": status,
        "clean_before_output": status == "",
    }


def target_proposals(
    history_X: Array,
    history_y: Array,
    *,
    raw_pool_size: int,
    proposal_size: int,
    design_seed: int,
    target_config: LocalSurrogateTransferConfig,
) -> Tuple[Array, Array, Array, Array]:
    """Build a source-blind target proposal panel from one frozen target GP."""

    if proposal_size > raw_pool_size:
        raise ValueError("proposal_size cannot exceed raw_pool_size.")
    model = fit_target_only_model(history_X, history_y, target_config)
    raw = sobol_chart_design(2, raw_pool_size, seed=design_seed)
    mean, std = model.predict(raw, return_std=True)
    if std is None:
        raise RuntimeError("Target model did not return uncertainty.")
    acquisition = expected_improvement(mean, std, float(np.min(history_y)))
    order = np.argsort(-acquisition, kind="stable")[:proposal_size]
    return (
        raw[order].copy(),
        np.asarray(mean, dtype=float)[order],
        np.asarray(std, dtype=float)[order],
        np.asarray(acquisition, dtype=float)[order],
    )


def score_columns(prefix: str, evidence: TrustScore) -> Dict[str, Any]:
    return {
        f"{prefix}_score": float(evidence.score),
        f"{prefix}_prediction_score": float(prediction_score(evidence)),
        f"{prefix}_raw": float(evidence.raw_statistic),
        f"{prefix}_eligible": bool(evidence.eligible),
        f"{prefix}_evidence_count": int(evidence.evidence_count),
        f"{prefix}_reason": str(evidence.reason),
        f"{prefix}_successes": int(evidence.successes),
        f"{prefix}_failures": int(evidence.failures),
    }


def generate_predecision_artifacts(
    config: Mapping[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate development and holdout features without final-panel labels."""

    event_rows: List[Dict[str, Any]] = []
    history_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    relations = [str(value) for value in config["relations"]]
    source_cfg = config["source_expert"]
    gate_cfg = config["gate"]

    split_specs = [
        (
            "development",
            int(config["development_seed_start"]),
            int(config["development_seed_count"]),
        ),
        (
            "holdout",
            int(config["holdout_seed_start"]),
            int(config["holdout_seed_count"]),
        ),
    ]

    for split, seed_start, seed_count in split_specs:
        for seed_offset, seed in enumerate(range(seed_start, seed_start + seed_count)):
            rng = np.random.default_rng(stable_seed(seed, 1))
            theta = float(rng.uniform(0.0, np.pi))
            independent_theta = float(
                theta + rng.uniform(0.35 * np.pi, 0.85 * np.pi)
            )
            source_X = sobol_chart_design(
                2,
                int(config["source_train_samples"]),
                seed=stable_seed(seed, 2),
            )
            base_source_truth, _ = make_controlled_relation(
                "identity",
                theta=theta,
                independent_theta=independent_theta,
            )
            independent_source_truth, _ = make_controlled_relation(
                "independent_expert",
                theta=theta,
                independent_theta=independent_theta,
            )
            base_expert = fit_source_rank_expert(
                source_X,
                base_source_truth(source_X),
                length_scale=float(source_cfg["length_scale"]),
                noise=float(source_cfg["noise"]),
                random_state=stable_seed(seed, 3),
            )
            independent_expert = fit_source_rank_expert(
                source_X,
                independent_source_truth(source_X),
                length_scale=float(source_cfg["length_scale"]),
                noise=float(source_cfg["noise"]),
                random_state=stable_seed(seed, 4),
            )

            for relation_index, relation in enumerate(relations):
                event_id = f"{split}__{seed}__{relation}"
                _, target_truth = make_controlled_relation(
                    relation,
                    theta=theta,
                    independent_theta=independent_theta,
                )
                expert = (
                    independent_expert
                    if relation == "independent_expert"
                    else base_expert
                )
                initial_X = sobol_chart_design(
                    2,
                    int(config["target_initial_samples"]),
                    seed=stable_seed(seed, 10),
                )
                history_X = initial_X.copy()
                history_y = np.asarray(target_truth(history_X), dtype=float).reshape(-1)
                initial_quality = source_quality_prediction(expert, history_X)

                local_history: List[Dict[str, Any]] = []
                for point_index, (point, cost, quality) in enumerate(
                    zip(history_X, history_y, initial_quality)
                ):
                    local_history.append(
                        {
                            "event_id": event_id,
                            "split": split,
                            "seed": seed,
                            "relation": relation,
                            "history_index": point_index,
                            "stage": "initial",
                            "feedback_round": -1,
                            "feedback_role": "initial",
                            "x0": float(point[0]),
                            "x1": float(point[1]),
                            "target_y": float(cost),
                            "source_quality": float(quality),
                            "pre_target_mean": np.nan,
                            "source_margin": np.nan,
                            "true_diagnostic_advantage": np.nan,
                            "residual_diagnostic_advantage": np.nan,
                            "source_target_disagreement": False,
                        }
                    )

                feedback_pairs: List[Dict[str, Any]] = []
                correction_successes = 0
                correction_failures = 0
                correction_ties = 0
                feedback_disagreements = 0

                for feedback_round in range(int(config["feedback_rounds"])):
                    cfg = model_config(
                        config,
                        stable_seed(seed, 20, relation_index, feedback_round),
                    )
                    proposals, means, _, acquisition = target_proposals(
                        history_X,
                        history_y,
                        raw_pool_size=int(config["feedback_raw_pool_size"]),
                        proposal_size=int(config["proposal_size"]),
                        design_seed=stable_seed(seed, 30, feedback_round),
                        target_config=cfg,
                    )
                    quality = source_quality_prediction(expert, proposals)
                    target_index, diagnostic_index = select_target_diagnostic_pair(
                        acquisition,
                        alternative_rank=int(config["diagnostic_alternative_rank"]),
                    )
                    selected_indices = [target_index, diagnostic_index]
                    selected_roles = ["target", "diagnostic"]
                    selected_X = proposals[np.asarray(selected_indices, dtype=int)]
                    selected_y = np.asarray(
                        target_truth(selected_X), dtype=float
                    ).reshape(-1)
                    tolerance = practical_outcome_tolerance(
                        history_y,
                        relative_fraction=float(
                            gate_cfg["outcome_tolerance_fraction"]
                        ),
                        absolute_floor=float(
                            gate_cfg["outcome_tolerance_absolute"]
                        ),
                    )
                    source_margin = float(
                        quality[diagnostic_index] - quality[target_index]
                    )
                    source_scale = max(
                        1.0,
                        abs(float(quality[diagnostic_index])),
                        abs(float(quality[target_index])),
                    )
                    source_margin_tolerance = 1e-14 + 1e-10 * source_scale
                    true_advantage = float(selected_y[0] - selected_y[1])
                    residual_target = float(
                        selected_y[0] - means[target_index]
                    )
                    residual_diagnostic = float(
                        selected_y[1] - means[diagnostic_index]
                    )
                    residual_advantage = residual_target - residual_diagnostic
                    target_scale = max(
                        1e-12,
                        abs(float(acquisition[target_index])),
                        abs(float(acquisition[diagnostic_index])),
                    )
                    target_margin_tolerance = 1e-14 + 1e-10 * target_scale
                    target_strict_preference = bool(
                        acquisition[target_index]
                        > acquisition[diagnostic_index] + target_margin_tolerance
                    )
                    source_disagreement = bool(
                        target_strict_preference
                        and source_margin > source_margin_tolerance
                    )
                    if source_disagreement:
                        feedback_disagreements += 1
                        if true_advantage > tolerance:
                            correction_successes += 1
                        elif true_advantage < -tolerance:
                            correction_failures += 1
                        else:
                            correction_ties += 1

                    feedback_pairs.append(
                        {
                            "feedback_round": feedback_round,
                            "target_index": int(target_index),
                            "diagnostic_index": int(diagnostic_index),
                            "source_margin": source_margin,
                            "source_margin_tolerance": source_margin_tolerance,
                            "target_margin_tolerance": target_margin_tolerance,
                            "target_strict_preference": target_strict_preference,
                            "true_diagnostic_advantage": true_advantage,
                            "residual_diagnostic_advantage": residual_advantage,
                            "source_target_disagreement": source_disagreement,
                            "outcome_tolerance": tolerance,
                        }
                    )
                    for proposal_index, role, point, cost in zip(
                        selected_indices,
                        selected_roles,
                        selected_X,
                        selected_y,
                    ):
                        local_history.append(
                            {
                                "event_id": event_id,
                                "split": split,
                                "seed": seed,
                                "relation": relation,
                                "history_index": len(local_history),
                                "stage": "paired_diagnostic_feedback",
                                "feedback_round": feedback_round,
                                "feedback_role": role,
                                "x0": float(point[0]),
                                "x1": float(point[1]),
                                "target_y": float(cost),
                                "source_quality": float(quality[proposal_index]),
                                "pre_target_mean": float(means[proposal_index]),
                                "feedback_target_index": int(target_index),
                                "feedback_diagnostic_index": int(diagnostic_index),
                                "feedback_outcome_tolerance": tolerance,
                                "source_margin": source_margin,
                                "true_diagnostic_advantage": true_advantage,
                                "residual_diagnostic_advantage": residual_advantage,
                                "source_target_disagreement": source_disagreement,
                            }
                        )
                    history_X = np.vstack([history_X, selected_X])
                    history_y = np.concatenate([history_y, selected_y])

                local_frame = pd.DataFrame(local_history)
                feedback_frame = pd.DataFrame(feedback_pairs)
                local_evidence = paired_margin_spearman_trust(
                    feedback_frame["source_margin"].to_numpy(dtype=float),
                    feedback_frame["true_diagnostic_advantage"].to_numpy(dtype=float),
                    min_points=int(gate_cfg["spearman_min_points"]),
                    shrinkage=float(gate_cfg["spearman_shrinkage"]),
                    label="decision_matched_local_spearman",
                )
                residual_evidence = paired_margin_spearman_trust(
                    feedback_frame["source_margin"].to_numpy(dtype=float),
                    feedback_frame[
                        "residual_diagnostic_advantage"
                    ].to_numpy(dtype=float),
                    min_points=int(gate_cfg["residual_min_points"]),
                    shrinkage=float(gate_cfg["spearman_shrinkage"]),
                    label="decision_matched_residual_spearman",
                )
                correction_evidence = disagreement_correction_trust(
                    correction_successes,
                    correction_failures,
                    min_events=int(gate_cfg["correction_min_events"]),
                )

                final_cfg = model_config(
                    config,
                    stable_seed(seed, 40, relation_index),
                )
                proposals, means, stds, acquisition = target_proposals(
                    history_X,
                    history_y,
                    raw_pool_size=int(config["final_raw_pool_size"]),
                    proposal_size=int(config["proposal_size"]),
                    design_seed=stable_seed(seed, 50),
                    target_config=final_cfg,
                )
                quality = source_quality_prediction(expert, proposals)
                target_index, source_index, eligible_indices = nominate_source_candidate(
                    acquisition,
                    quality,
                    top_k=int(config["advice_top_k"]),
                )
                actionable = source_index != target_index
                tolerance = practical_outcome_tolerance(
                    history_y,
                    relative_fraction=float(gate_cfg["outcome_tolerance_fraction"]),
                    absolute_floor=float(gate_cfg["outcome_tolerance_absolute"]),
                )
                candidate_hash = rounded_array_hash(
                    proposals,
                    means,
                    stds,
                    acquisition,
                    quality,
                )
                history_hash = rounded_array_hash(history_X, history_y)

                event = {
                    "event_id": event_id,
                    "split": split,
                    "seed": seed,
                    "seed_offset": seed_offset,
                    "relation": relation,
                    "relation_index": relation_index,
                    "theta": theta,
                    "independent_theta": independent_theta,
                    "history_count": len(history_X),
                    "feedback_pair_events": int(len(feedback_frame)),
                    "paid_feedback_evaluations": int(2 * len(feedback_frame)),
                    "feedback_disagreement_rounds": feedback_disagreements,
                    "feedback_correction_successes": correction_successes,
                    "feedback_correction_failures": correction_failures,
                    "feedback_correction_ties": correction_ties,
                    "candidate_count": len(proposals),
                    "target_index": int(target_index),
                    "source_index": int(source_index),
                    "actionable": bool(actionable),
                    "outcome_tolerance": tolerance,
                    "candidate_hash": candidate_hash,
                    "history_hash": history_hash,
                    **score_columns("local_spearman", local_evidence),
                    **score_columns("residual_spearman", residual_evidence),
                    **score_columns("disagreement_correction", correction_evidence),
                }
                event_rows.append(event)
                history_rows.extend(local_history)

                eligible_set = set(map(int, eligible_indices.tolist()))
                for candidate_index, (
                    point,
                    mean,
                    std,
                    acq,
                    expert_quality,
                ) in enumerate(zip(proposals, means, stds, acquisition, quality)):
                    candidate_rows.append(
                        {
                            "event_id": event_id,
                            "split": split,
                            "seed": seed,
                            "relation": relation,
                            "candidate_index": candidate_index,
                            "x0": float(point[0]),
                            "x1": float(point[1]),
                            "target_gp_mean": float(mean),
                            "target_gp_std": float(std),
                            "target_acquisition": float(acq),
                            "source_quality": float(expert_quality),
                            "advice_eligible": candidate_index in eligible_set,
                            "is_target_nomination": candidate_index == target_index,
                            "is_source_nomination": candidate_index == source_index,
                        }
                    )

    return (
        pd.DataFrame(event_rows),
        pd.DataFrame(history_rows),
        pd.DataFrame(candidate_rows),
    )


def persist_predecision(
    events: pd.DataFrame,
    history: pd.DataFrame,
    candidates: pd.DataFrame,
    output: Path,
) -> Dict[str, Path]:
    paths = {
        "events": output / "predecision_events.csv",
        "history": output / "predecision_history.csv",
        "candidates": output / "predecision_candidate_panel.csv",
    }
    events.to_csv(paths["events"], index=False, float_format="%.17g")
    history.to_csv(paths["history"], index=False, float_format="%.17g")
    candidates.to_csv(paths["candidates"], index=False, float_format="%.17g")
    return paths


def freeze_gate_decisions(
    events: pd.DataFrame,
    config: Mapping[str, Any],
    output: Path,
) -> Tuple[Dict[str, Dict[str, Any]], pd.DataFrame]:
    development = events[events["split"] == "development"].copy()
    gate_cfg = config["gate"]
    thresholds: Dict[str, Dict[str, Any]] = {}
    decision_rows: List[Dict[str, Any]] = []

    for method, prefix in DEPLOYABLE_GATES.items():
        frozen = freeze_coverage_threshold(
            development[f"{prefix}_score"].to_numpy(dtype=float),
            development[f"{prefix}_eligible"].to_numpy(dtype=bool),
            development["actionable"].to_numpy(dtype=bool),
            target_coverage=float(
                gate_cfg["development_target_actionable_coverage"]
            ),
            minimum_positive_score=float(gate_cfg["minimum_positive_score"]),
        )
        thresholds[method] = {
            "prefix": prefix,
            **frozen.__dict__,
            "label_free_rule": (
                "development score/eligibility/actionable coverage only; "
                "no final candidate truth"
            ),
        }

    for _, event in events.iterrows():
        decision_rows.append(
            {
                "event_id": event["event_id"],
                "split": event["split"],
                "seed": int(event["seed"]),
                "relation": event["relation"],
                "method": "Target-Only",
                "prefix": "target_only",
                "continuous_score": np.nan,
                "eligible": True,
                "threshold": np.nan,
                "actionable": bool(event["actionable"]),
                "accepted": False,
                "selected_index": int(event["target_index"]),
            }
        )
        for method, prefix in DEPLOYABLE_GATES.items():
            threshold = float(thresholds[method]["threshold"])
            eligible = bool(event[f"{prefix}_eligible"])
            score = float(event[f"{prefix}_score"])
            accepted = bool(
                event["actionable"] and eligible and score >= threshold
            )
            decision_rows.append(
                {
                    "event_id": event["event_id"],
                    "split": event["split"],
                    "seed": int(event["seed"]),
                    "relation": event["relation"],
                    "method": method,
                    "prefix": prefix,
                    "continuous_score": float(
                        event[f"{prefix}_prediction_score"]
                    ),
                    "eligible": eligible,
                    "threshold": threshold,
                    "actionable": bool(event["actionable"]),
                    "accepted": accepted,
                    "selected_index": int(
                        event["source_index"]
                        if accepted
                        else event["target_index"]
                    ),
                }
            )

    threshold_path = output / "frozen_gate_thresholds.json"
    threshold_path.write_text(
        json.dumps(thresholds, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    decisions = pd.DataFrame(decision_rows)
    decisions.to_csv(
        output / "predecision_gate_decisions.csv",
        index=False,
        float_format="%.17g",
    )
    return thresholds, decisions


def reveal_and_evaluate(
    events: pd.DataFrame,
    candidates: pd.DataFrame,
    decisions: pd.DataFrame,
    output: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    revealed_candidates: List[pd.DataFrame] = []
    revealed_events: List[Dict[str, Any]] = []
    method_rows: List[Dict[str, Any]] = []
    decision_lookup = decisions.set_index(["event_id", "method"])

    for _, event in events.iterrows():
        event_id = str(event["event_id"])
        panel = candidates[candidates["event_id"] == event_id].sort_values(
            "candidate_index",
            kind="stable",
        )
        points = panel[["x0", "x1"]].to_numpy(dtype=float)
        _, target_truth = make_controlled_relation(
            str(event["relation"]),
            theta=float(event["theta"]),
            independent_theta=float(event["independent_theta"]),
        )
        true_y = np.asarray(target_truth(points), dtype=float).reshape(-1)
        target_index = int(event["target_index"])
        source_index = int(event["source_index"])
        target_y = float(true_y[target_index])
        source_y = float(true_y[source_index])
        epsilon = float(event["outcome_tolerance"])
        actionable = bool(event["actionable"])
        beneficial = bool(actionable and source_y < target_y - epsilon)
        harmful = bool(actionable and source_y > target_y + epsilon)
        tied = bool(actionable and not beneficial and not harmful)
        oracle_best = float(np.min(true_y))
        normalization_scale = max(
            float(np.quantile(true_y, 0.90) - oracle_best),
            1e-12,
        )
        top_mask, top_count, top_cutoff = tie_aware_top_fraction_mask(
            true_y,
            fraction=0.10,
            tolerance=epsilon,
        )

        candidate_copy = panel.copy()
        candidate_copy["true_target_y"] = true_y
        revealed_candidates.append(candidate_copy)
        revealed_events.append(
            {
                "event_id": event_id,
                "split": event["split"],
                "seed": int(event["seed"]),
                "relation": event["relation"],
                "actionable": actionable,
                "target_index": target_index,
                "source_index": source_index,
                "target_y": target_y,
                "source_y": source_y,
                "source_gain_vs_target": target_y - source_y,
                "source_gain_over_epsilon": (
                    (target_y - source_y) / max(epsilon, 1e-300)
                ),
                "outcome_tolerance": epsilon,
                "source_beneficial": beneficial,
                "source_harmful": harmful,
                "source_tied": tied,
                "candidate_oracle_y": oracle_best,
                "normalization_scale": normalization_scale,
                "candidate_count": len(true_y),
                "top10_nominal_count": top_count,
                "top10_tie_inclusive_count": int(np.sum(top_mask)),
                "top10_cutoff": top_cutoff,
            }
        )

        for method in METHODS:
            if method == "Oracle Gate":
                accepted = beneficial
                selected_index = source_index if accepted else target_index
                prediction = float(beneficial) if actionable else 0.0
                eligible = actionable
                threshold = 0.5
            else:
                decision = decision_lookup.loc[(event_id, method)]
                accepted = bool(decision["accepted"])
                selected_index = int(decision["selected_index"])
                prediction = (
                    float(decision["continuous_score"])
                    if method != "Target-Only"
                    else np.nan
                )
                eligible = bool(decision["eligible"])
                threshold = float(decision["threshold"])
            selected_y = float(true_y[selected_index])
            method_rows.append(
                {
                    "event_id": event_id,
                    "split": event["split"],
                    "seed": int(event["seed"]),
                    "relation": event["relation"],
                    "method": method,
                    "actionable": actionable,
                    "eligible": eligible,
                    "accepted": accepted,
                    "continuous_score": prediction,
                    "threshold": threshold,
                    "selected_index": selected_index,
                    "target_index": target_index,
                    "source_index": source_index,
                    "selected_y": selected_y,
                    "target_y": target_y,
                    "source_y": source_y,
                    "candidate_oracle_y": oracle_best,
                    "raw_regret": selected_y - oracle_best,
                    "normalized_regret": (
                        selected_y - oracle_best
                    ) / normalization_scale,
                    "top10_hit": bool(top_mask[selected_index]),
                    "source_beneficial": beneficial,
                    "source_harmful": harmful,
                    "source_tied": tied,
                    "source_gain_over_epsilon": (
                        (target_y - source_y) / max(epsilon, 1e-300)
                    ),
                    "negative_transfer_on_acceptance": bool(
                        accepted and harmful
                    ),
                    "effective_source_gain": target_y - selected_y,
                    "outcome_tolerance": epsilon,
                }
            )

    revealed_candidate_frame = pd.concat(revealed_candidates, ignore_index=True)
    revealed_event_frame = pd.DataFrame(revealed_events)
    method_frame = pd.DataFrame(method_rows)
    revealed_candidate_frame.to_csv(
        output / "revealed_candidate_panel.csv",
        index=False,
        float_format="%.17g",
    )
    revealed_event_frame.to_csv(
        output / "revealed_event_outcomes.csv",
        index=False,
        float_format="%.17g",
    )
    method_frame.to_csv(
        output / "method_outcomes.csv",
        index=False,
        float_format="%.17g",
    )
    return revealed_candidate_frame, revealed_event_frame, method_frame


def audit_artifacts(
    config: Mapping[str, Any],
    events: pd.DataFrame,
    history: pd.DataFrame,
    candidates: pd.DataFrame,
    decisions: pd.DataFrame,
    revealed_candidates: pd.DataFrame,
    method_outcomes: pd.DataFrame,
    output: Path,
) -> Dict[str, Any]:
    errors: List[str] = []
    expected_events = (
        int(config["development_seed_count"])
        + int(config["holdout_seed_count"])
    ) * len(config["relations"])
    if len(events) != expected_events:
        errors.append(f"expected {expected_events} events, found {len(events)}")
    if events["event_id"].nunique() != len(events):
        errors.append("event_id is not unique")
    dev_seeds = set(events.loc[events["split"] == "development", "seed"])
    holdout_seeds = set(events.loc[events["split"] == "holdout", "seed"])
    if dev_seeds.intersection(holdout_seeds):
        errors.append("development and holdout seeds overlap")
    forbidden = {
        "true_target_y",
        "source_beneficial",
        "source_harmful",
        "candidate_oracle_y",
        "raw_regret",
        "normalized_regret",
        "top10_hit",
    }
    predecision_ledgers = {
        "events": events,
        "history": history,
        "candidates": candidates,
        "decisions": decisions,
    }
    for name, ledger in predecision_ledgers.items():
        if forbidden.intersection(ledger.columns):
            errors.append(f"predecision {name} ledger contains final-panel truth fields")
    if len(decisions) != len(events) * 4:
        errors.append("predecision decision ledger must contain four non-oracle methods")

    threshold_payload = json.loads(
        (output / "frozen_gate_thresholds.json").read_text(encoding="utf-8")
    )
    development = events[events["split"] == "development"]
    reconstructed_thresholds: Dict[str, FrozenGateThreshold] = {}
    for method, prefix in DEPLOYABLE_GATES.items():
        reconstructed = freeze_coverage_threshold(
            development[f"{prefix}_score"].to_numpy(dtype=float),
            development[f"{prefix}_eligible"].to_numpy(dtype=bool),
            development["actionable"].to_numpy(dtype=bool),
            target_coverage=float(
                config["gate"]["development_target_actionable_coverage"]
            ),
            minimum_positive_score=float(
                config["gate"]["minimum_positive_score"]
            ),
        )
        reconstructed_thresholds[method] = reconstructed
        stored = threshold_payload.get(method, {})
        for field, expected_value in reconstructed.__dict__.items():
            stored_value = stored.get(field)
            if isinstance(expected_value, float):
                matches = stored_value is not None and np.isclose(
                    float(stored_value),
                    expected_value,
                    rtol=0.0,
                    atol=1e-15,
                )
            else:
                matches = stored_value == expected_value
            if not matches:
                errors.append(
                    f"threshold reconstruction mismatch for {method}:{field}"
                )
                break
    if len(method_outcomes) != len(events) * len(METHODS):
        errors.append("method outcome ledger does not contain exactly five methods")
    method_counts = method_outcomes.groupby("event_id")["method"].nunique()
    if not np.all(method_counts.to_numpy() == len(METHODS)):
        errors.append("at least one event is missing a method")

    event_lookup = events.set_index("event_id")
    for event_id, event in event_lookup.iterrows():
        candidate_group = candidates[candidates["event_id"] == event_id].sort_values(
            "candidate_index",
            kind="stable",
        )
        candidate_hash = rounded_array_hash(
            candidate_group[["x0", "x1"]].to_numpy(dtype=float),
            candidate_group["target_gp_mean"].to_numpy(dtype=float),
            candidate_group["target_gp_std"].to_numpy(dtype=float),
            candidate_group["target_acquisition"].to_numpy(dtype=float),
            candidate_group["source_quality"].to_numpy(dtype=float),
        )
        if candidate_hash != str(event["candidate_hash"]):
            errors.append(f"candidate hash mismatch for {event_id}")
            break
        history_group = history[history["event_id"] == event_id].sort_values(
            "history_index",
            kind="stable",
        )
        history_hash = rounded_array_hash(
            history_group[["x0", "x1"]].to_numpy(dtype=float),
            history_group["target_y"].to_numpy(dtype=float),
        )
        if history_hash != str(event["history_hash"]):
            errors.append(f"history hash mismatch for {event_id}")
            break

    for _, decision in decisions.iterrows():
        event = event_lookup.loc[decision["event_id"]]
        selected = int(decision["selected_index"])
        target_index = int(event["target_index"])
        source_index = int(event["source_index"])
        if selected not in {target_index, source_index}:
            errors.append("a gate selected outside the fixed target/source nominations")
            break
        method = str(decision["method"])
        if method == "Target-Only":
            reconstructed_accepted = False
            expected_threshold = np.nan
        else:
            prefix = DEPLOYABLE_GATES[method]
            expected_threshold = reconstructed_thresholds[method].threshold
            reconstructed_accepted = bool(
                event["actionable"]
                and event[f"{prefix}_eligible"]
                and event[f"{prefix}_score"] >= expected_threshold
            )
        if bool(decision["accepted"]) != reconstructed_accepted:
            errors.append(f"accepted flag reconstruction mismatch for {method}")
            break
        if method != "Target-Only" and not np.isclose(
            float(decision["threshold"]),
            expected_threshold,
            rtol=0.0,
            atol=1e-15,
        ):
            errors.append(f"decision threshold mismatch for {method}")
            break
        expected = source_index if reconstructed_accepted else target_index
        if selected != expected:
            errors.append("predecision fallback/acceptance selection is inconsistent")
            break

    candidate_counts = candidates.groupby("event_id").size()
    if not np.all(candidate_counts.to_numpy() == int(config["proposal_size"])):
        errors.append("candidate panel size differs from frozen proposal_size")
    if len(revealed_candidates) != len(candidates):
        errors.append("revealed and predecision candidate panels differ in length")

    audit = {
        "ok": not errors,
        "errors": errors,
        "expected_events": expected_events,
        "observed_events": int(len(events)),
        "predecision_history_rows": int(
            pd.read_csv(output / "predecision_history.csv").shape[0]
        ),
        "predecision_candidate_rows": int(len(candidates)),
        "predecision_decision_rows": int(len(decisions)),
        "method_outcome_rows": int(len(method_outcomes)),
        "development_seed_count": len(dev_seeds),
        "holdout_seed_count": len(holdout_seeds),
        "methods": METHODS,
        "final_panel_truth_absent_from_all_predecision_ledgers": all(
            not bool(forbidden.intersection(ledger.columns))
            for ledger in predecision_ledgers.values()
        ),
        "candidate_and_history_hashes_recomputed": not any(
            "hash mismatch" in error for error in errors
        ),
        "thresholds_and_decisions_reconstructed": not any(
            "threshold" in error
            or "accepted flag" in error
            or "fallback/acceptance" in error
            for error in errors
        ),
        "selection_interface": "accept fixed x_S or exact x_T fallback",
        "feedback_interface": "paid source-blind target-acquisition diagnostic pair",
    }
    (output / "AUDIT.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return audit


def write_manifest(
    config: Mapping[str, Any],
    output: Path,
    artifact_paths: Iterable[Path],
    provenance: Mapping[str, Any],
) -> None:
    artifacts = [path for path in artifact_paths if path.exists()]
    manifest = {
        "stage_id": config["stage_id"],
        "scope": config["scope"],
        "protocol": "PROTOCOL_DISAGREEMENT_TRUST_VALIDATION.md",
        "config": config,
        "config_sha256": canonical_config_hash(config),
        "implementation_sha256": {
            "PROTOCOL_DISAGREEMENT_TRUST_VALIDATION.md": file_hash(
                REPO_ROOT / "PROTOCOL_DISAGREEMENT_TRUST_VALIDATION.md"
            ),
            "scripts/run_disagreement_trust_validation.py": file_hash(Path(__file__)),
            "scripts/analyze_disagreement_trust_validation.py": file_hash(
                REPO_ROOT / "scripts" / "analyze_disagreement_trust_validation.py"
            ),
            "src/region_guided_reranking_study/disagreement_trust.py": file_hash(
                SRC_DIR / "region_guided_reranking_study" / "disagreement_trust.py"
            ),
            "src/region_guided_reranking_study/local_surrogate_transfer.py": file_hash(
                SRC_DIR
                / "region_guided_reranking_study"
                / "local_surrogate_transfer.py"
            ),
            "src/region_guided_reranking_study/local_surrogate_transfer_research.py": file_hash(
                SRC_DIR
                / "region_guided_reranking_study"
                / "local_surrogate_transfer_research.py"
            ),
            "tests/test_disagreement_trust.py": file_hash(
                REPO_ROOT / "tests" / "test_disagreement_trust.py"
            ),
        },
        "git": dict(provenance),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
        "artifact_path_base": "run_output_directory",
        "artifact_sha256": {
            str(path.relative_to(output)): file_hash(path) for path in artifacts
        },
        "phase_order": [
            "persist predecision events/history/candidates",
            "freeze thresholds and persist non-oracle decisions",
            "reveal candidate-panel truth",
            "evaluate five methods",
        ],
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run(config_path: Path, output: Path) -> pd.DataFrame:
    config = load_config(config_path)
    provenance = git_identity()
    output.mkdir(parents=True, exist_ok=True)
    print(f"[{config['stage_id']}] Phase A: generating predecision artifacts")
    events, history, candidates = generate_predecision_artifacts(config)
    predecision_paths = persist_predecision(events, history, candidates, output)
    print(
        f"Persisted {len(events)} events, {len(history)} history rows, "
        f"and {len(candidates)} candidate rows before truth reveal."
    )

    print("Phase A2: freezing label-free coverage thresholds and decisions")
    _, decisions = freeze_gate_decisions(events, config, output)

    # Read the persisted artifacts back before truth reveal.  This makes the phase
    # boundary explicit and ensures Phase B evaluates the actual frozen files.
    frozen_events = pd.read_csv(
        predecision_paths["events"], float_precision="round_trip"
    )
    frozen_history = pd.read_csv(
        predecision_paths["history"], float_precision="round_trip"
    )
    frozen_candidates = pd.read_csv(
        predecision_paths["candidates"], float_precision="round_trip"
    )
    frozen_decisions = pd.read_csv(
        output / "predecision_gate_decisions.csv",
        float_precision="round_trip",
    )

    print("Phase B: revealing the common target candidate panels")
    revealed_candidates, _, method_outcomes = reveal_and_evaluate(
        frozen_events,
        frozen_candidates,
        frozen_decisions,
        output,
    )
    audit = audit_artifacts(
        config,
        frozen_events,
        frozen_history,
        frozen_candidates,
        frozen_decisions,
        revealed_candidates,
        method_outcomes,
        output,
    )
    if not audit["ok"]:
        raise RuntimeError(f"Audit failed: {audit['errors']}")

    artifacts = [
        *predecision_paths.values(),
        output / "frozen_gate_thresholds.json",
        output / "predecision_gate_decisions.csv",
        output / "revealed_candidate_panel.csv",
        output / "revealed_event_outcomes.csv",
        output / "method_outcomes.csv",
        output / "AUDIT.json",
    ]
    write_manifest(config, output, artifacts, provenance)
    print(f"Completed and audited output: {output}")
    return method_outcomes


def resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "disagreement_trust_validation_quick.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "disagreement_trust_validation_quick",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(resolve_path(args.config), resolve_path(args.output))
