"""Analyze the held-out disagreement-conditioned trust pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent

DEPLOYABLE = [
    "Local Spearman Gate",
    "Target-Residual Spearman Gate",
    "Disagreement-Correction Gate",
]
METHODS = ["Target-Only", *DEPLOYABLE, "Oracle Gate"]
LOCAL = "Local Spearman Gate"
CORRECTION = "Disagreement-Correction Gate"
TARGET = "Target-Only"
ORACLE = "Oracle Gate"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object.")
    return value


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


def finite_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(labels, dtype=int).reshape(-1)
    prediction = np.asarray(scores, dtype=float).reshape(-1)
    if len(y) == 0 or len(y) != len(prediction) or len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, prediction))


def method_summary(group: pd.DataFrame) -> Dict[str, Any]:
    actionable = group[group["actionable"].astype(bool)]
    accepted = group[group["accepted"].astype(bool)]
    accepted_harm = int(accepted["source_harmful"].astype(bool).sum())
    return {
        "events": int(len(group)),
        "actionable_events": int(len(actionable)),
        "eligible_events": int(group["eligible"].astype(bool).sum()),
        "accepted_events": int(len(accepted)),
        "mean_raw_regret": float(group["raw_regret"].mean()),
        "mean_normalized_regret": float(group["normalized_regret"].mean()),
        "top10_hit_rate": float(group["top10_hit"].astype(float).mean()),
        "overall_acceptance_coverage": float(
            group["accepted"].astype(float).mean()
        ),
        "actionable_acceptance_coverage": float(
            accepted.shape[0] / len(actionable)
        )
        if len(actionable)
        else float("nan"),
        "accepted_harm_count": accepted_harm,
        "accepted_harm_denominator": int(len(accepted)),
        "accepted_negative_transfer_rate": float(
            accepted_harm / len(accepted)
        )
        if len(accepted)
        else float("nan"),
        "mean_effective_source_gain": float(
            group["effective_source_gain"].mean()
        ),
    }


def common_support_event_ids(method_rows: pd.DataFrame) -> set[str]:
    deployable = method_rows[
        method_rows["method"].isin(DEPLOYABLE)
        & method_rows["actionable"].astype(bool)
    ]
    if deployable.empty:
        return set()
    grouped = deployable.groupby("event_id").agg(
        methods=("method", "nunique"),
        all_eligible=("eligible", lambda values: bool(np.all(values.astype(bool)))),
    )
    selected = grouped[
        (grouped["methods"] == len(DEPLOYABLE)) & grouped["all_eligible"]
    ]
    return set(map(str, selected.index.tolist()))


def prediction_metrics(
    method_outcomes: pd.DataFrame,
    revealed_events: pd.DataFrame,
) -> pd.DataFrame:
    holdout_methods = method_outcomes[method_outcomes["split"] == "holdout"].copy()
    holdout_events = revealed_events[
        (revealed_events["split"] == "holdout")
        & revealed_events["actionable"].astype(bool)
    ].copy()
    actionable_prevalence = float(
        holdout_events["source_beneficial"].astype(float).mean()
    )
    common_ids = common_support_event_ids(holdout_methods)
    common_events = holdout_events[
        holdout_events["event_id"].astype(str).isin(common_ids)
    ]
    common_prevalence = float(
        common_events["source_beneficial"].astype(float).mean()
    ) if len(common_events) else float("nan")
    rows: List[Dict[str, Any]] = []
    for method in DEPLOYABLE:
        actionable = holdout_methods[
            (holdout_methods["method"] == method)
            & holdout_methods["actionable"].astype(bool)
        ]
        common = actionable[actionable["event_id"].astype(str).isin(common_ids)]
        rows.append(
            {
                "method": method,
                "actionable_events": int(len(actionable)),
                "positive_events_actionable": int(
                    actionable["source_beneficial"].astype(bool).sum()
                ),
                "actionable_positive_prevalence": actionable_prevalence,
                "eligible_prediction_coverage": float(
                    actionable["eligible"].astype(float).mean()
                ),
                "common_support_events": int(len(common)),
                "common_support_fraction": float(
                    len(common) / len(actionable)
                ) if len(actionable) else float("nan"),
                "common_support_positive_events": int(
                    common["source_beneficial"].astype(bool).sum()
                ),
                "common_support_prevalence": common_prevalence,
                "common_support_auprc": finite_average_precision(
                    common["source_beneficial"].astype(int).to_numpy(),
                    common["continuous_score"].to_numpy(dtype=float),
                ),
                "neutral_imputed_actionable_auprc": finite_average_precision(
                    actionable["source_beneficial"].astype(int).to_numpy(),
                    actionable["continuous_score"].to_numpy(dtype=float),
                ),
            }
        )
    rows.append(
        {
            "method": "Oracle Gate (tautological ceiling)",
            "actionable_events": int(len(holdout_events)),
            "positive_events_actionable": int(
                holdout_events["source_beneficial"].astype(bool).sum()
            ),
            "actionable_positive_prevalence": actionable_prevalence,
            "eligible_prediction_coverage": 1.0,
            "common_support_events": int(len(common_events)),
            "common_support_fraction": float(
                len(common_events) / len(holdout_events)
            ) if len(holdout_events) else float("nan"),
            "common_support_positive_events": int(
                common_events["source_beneficial"].astype(bool).sum()
            ),
            "common_support_prevalence": common_prevalence,
            "common_support_auprc": (
                1.0 if 0.0 < common_prevalence < 1.0 else float("nan")
            ),
            "neutral_imputed_actionable_auprc": (
                1.0 if 0.0 < actionable_prevalence < 1.0 else float("nan")
            ),
        }
    )
    return pd.DataFrame(rows)


def cluster_bootstrap(
    method_outcomes: pd.DataFrame,
    revealed_events: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    holdout = method_outcomes[method_outcomes["split"] == "holdout"].copy()
    event_holdout = revealed_events[
        revealed_events["split"] == "holdout"
    ].copy()
    seeds = np.sort(holdout["seed"].unique())
    if len(seeds) < 2:
        raise ValueError("Cluster bootstrap requires at least two holdout seeds.")
    method_by_seed = {value: holdout[holdout["seed"] == value] for value in seeds}
    event_by_seed = {
        value: event_holdout[event_holdout["seed"] == value] for value in seeds
    }
    rng = np.random.default_rng(int(seed))
    rows: List[Dict[str, Any]] = []

    for bootstrap_index in range(int(n_bootstrap)):
        sampled = rng.choice(seeds, size=len(seeds), replace=True)
        sample_methods = pd.concat(
            [method_by_seed[value] for value in sampled],
            ignore_index=True,
        )
        sample_events = pd.concat(
            [event_by_seed[value] for value in sampled],
            ignore_index=True,
        )
        summaries = {
            method: method_summary(
                sample_methods[sample_methods["method"] == method]
            )
            for method in METHODS
        }
        common_ids = common_support_event_ids(sample_methods)
        common_events = sample_events[
            sample_events["actionable"].astype(bool)
            & sample_events["event_id"].astype(str).isin(common_ids)
        ]
        auprc: Dict[str, float] = {}
        for method in DEPLOYABLE:
            group = sample_methods[
                (sample_methods["method"] == method)
                & sample_methods["actionable"].astype(bool)
                & sample_methods["event_id"].astype(str).isin(common_ids)
            ]
            auprc[method] = finite_average_precision(
                group["source_beneficial"].astype(int).to_numpy(),
                group["continuous_score"].to_numpy(dtype=float),
            )
        prevalence = float(
            common_events["source_beneficial"].astype(float).mean()
        ) if len(common_events) else float("nan")
        rows.append(
            {
                "bootstrap_index": bootstrap_index,
                "regret_advantage_correction_vs_local": (
                    summaries[LOCAL]["mean_normalized_regret"]
                    - summaries[CORRECTION]["mean_normalized_regret"]
                ),
                "regret_advantage_correction_vs_target": (
                    summaries[TARGET]["mean_normalized_regret"]
                    - summaries[CORRECTION]["mean_normalized_regret"]
                ),
                "oracle_regret_headroom": (
                    summaries[TARGET]["mean_normalized_regret"]
                    - summaries[ORACLE]["mean_normalized_regret"]
                ),
                "top10_advantage_correction_vs_local": (
                    summaries[CORRECTION]["top10_hit_rate"]
                    - summaries[LOCAL]["top10_hit_rate"]
                ),
                "top10_advantage_correction_vs_target": (
                    summaries[CORRECTION]["top10_hit_rate"]
                    - summaries[TARGET]["top10_hit_rate"]
                ),
                "coverage_difference_correction_vs_local": (
                    summaries[CORRECTION]["actionable_acceptance_coverage"]
                    - summaries[LOCAL]["actionable_acceptance_coverage"]
                ),
                "harm_rate_advantage_correction_vs_local": (
                    summaries[LOCAL]["accepted_negative_transfer_rate"]
                    - summaries[CORRECTION]["accepted_negative_transfer_rate"]
                ),
                "auprc_advantage_correction_vs_local": (
                    auprc[CORRECTION] - auprc[LOCAL]
                ),
                "correction_auprc_above_prevalence": (
                    auprc[CORRECTION] - prevalence
                ),
            }
        )
    return pd.DataFrame(rows)


def interval(values: pd.Series) -> Tuple[float, float, float]:
    array = values.to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.mean(array)),
        float(np.quantile(array, 0.025)),
        float(np.quantile(array, 0.975)),
    )


def point_contrasts(
    method_summary_frame: pd.DataFrame,
    prediction: pd.DataFrame,
) -> Dict[str, float]:
    summary = method_summary_frame.set_index("method")
    prediction_index = prediction.set_index("method")
    return {
        "regret_advantage_correction_vs_local": float(
            summary.loc[LOCAL, "mean_normalized_regret"]
            - summary.loc[CORRECTION, "mean_normalized_regret"]
        ),
        "regret_advantage_correction_vs_target": float(
            summary.loc[TARGET, "mean_normalized_regret"]
            - summary.loc[CORRECTION, "mean_normalized_regret"]
        ),
        "oracle_regret_headroom": float(
            summary.loc[TARGET, "mean_normalized_regret"]
            - summary.loc[ORACLE, "mean_normalized_regret"]
        ),
        "top10_advantage_correction_vs_local": float(
            summary.loc[CORRECTION, "top10_hit_rate"]
            - summary.loc[LOCAL, "top10_hit_rate"]
        ),
        "top10_advantage_correction_vs_target": float(
            summary.loc[CORRECTION, "top10_hit_rate"]
            - summary.loc[TARGET, "top10_hit_rate"]
        ),
        "coverage_difference_correction_vs_local": float(
            summary.loc[CORRECTION, "actionable_acceptance_coverage"]
            - summary.loc[LOCAL, "actionable_acceptance_coverage"]
        ),
        "harm_rate_advantage_correction_vs_local": float(
            summary.loc[LOCAL, "accepted_negative_transfer_rate"]
            - summary.loc[CORRECTION, "accepted_negative_transfer_rate"]
        ),
        "auprc_advantage_correction_vs_local": float(
            prediction_index.loc[CORRECTION, "common_support_auprc"]
            - prediction_index.loc[LOCAL, "common_support_auprc"]
        ),
        "correction_auprc_above_prevalence": float(
            prediction_index.loc[CORRECTION, "common_support_auprc"]
            - prediction_index.loc[CORRECTION, "common_support_prevalence"]
        ),
    }


def comparison_table(
    point: Mapping[str, float],
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    labels = {
        "regret_advantage_correction_vs_local": "Local regret - Correction regret",
        "regret_advantage_correction_vs_target": "Target regret - Correction regret",
        "oracle_regret_headroom": "Target regret - Oracle regret",
        "top10_advantage_correction_vs_local": "Correction Top10 - Local Top10",
        "top10_advantage_correction_vs_target": "Correction Top10 - Target Top10",
        "coverage_difference_correction_vs_local": "Correction coverage - Local coverage",
        "harm_rate_advantage_correction_vs_local": "Local harm rate - Correction harm rate",
        "auprc_advantage_correction_vs_local": "Correction AUPRC - Local AUPRC",
        "correction_auprc_above_prevalence": "Correction AUPRC - prevalence",
    }
    rows = []
    for metric, description in labels.items():
        _, low, high = interval(bootstrap[metric])
        finite_count = int(np.isfinite(bootstrap[metric].to_numpy(dtype=float)).sum())
        rows.append(
            {
                "metric": metric,
                "description": description,
                "estimate": float(point[metric]),
                "cluster_bootstrap_ci_low": low,
                "cluster_bootstrap_ci_high": high,
                "finite_bootstrap_replicates": finite_count,
                "finite_bootstrap_fraction": finite_count / max(len(bootstrap), 1),
                "positive_favors_correction": metric
                not in {"coverage_difference_correction_vs_local"},
            }
        )
    return pd.DataFrame(rows)


def evaluate_decision(
    method_summary_frame: pd.DataFrame,
    prediction: pd.DataFrame,
    comparisons: pd.DataFrame,
    revealed_events: pd.DataFrame,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    analysis = config["analysis"]
    summary = method_summary_frame.set_index("method")
    prediction_index = prediction.set_index("method")
    comparison = comparisons.set_index("metric")
    holdout_events = revealed_events[revealed_events["split"] == "holdout"]
    actionable_events = int(holdout_events["actionable"].astype(bool).sum())
    positive_events = int(
        (
            holdout_events["actionable"].astype(bool)
            & holdout_events["source_beneficial"].astype(bool)
        ).sum()
    )
    common_support_events = int(
        prediction_index.loc[CORRECTION, "common_support_events"]
    )
    common_support_positive_events = int(
        prediction_index.loc[CORRECTION, "common_support_positive_events"]
    )
    correction_accepts = int(summary.loc[CORRECTION, "accepted_events"])
    local_accepts = int(summary.loc[LOCAL, "accepted_events"])
    sparse_reasons = []
    if actionable_events < int(analysis["minimum_actionable_events"]):
        sparse_reasons.append("too_few_actionable_events")
    if positive_events < int(analysis["minimum_positive_events"]):
        sparse_reasons.append("too_few_positive_benefit_events")
    if common_support_events < int(analysis["minimum_actionable_events"]):
        sparse_reasons.append("too_few_common_support_events")
    if common_support_positive_events < int(analysis["minimum_positive_events"]):
        sparse_reasons.append("too_few_common_support_positive_events")
    if correction_accepts < int(analysis["minimum_accepted_events"]):
        sparse_reasons.append("too_few_correction_acceptances")
    if local_accepts < int(analysis["minimum_accepted_events"]):
        sparse_reasons.append("too_few_local_spearman_acceptances")

    auprc_supported = bool(
        comparison.loc[
            "auprc_advantage_correction_vs_local", "cluster_bootstrap_ci_low"
        ]
        > 0.0
        and comparison.loc[
            "correction_auprc_above_prevalence", "cluster_bootstrap_ci_low"
        ]
        > 0.0
    )
    regret_supported = bool(
        comparison.loc[
            "regret_advantage_correction_vs_local", "estimate"
        ]
        >= float(analysis["practical_normalized_regret_margin"])
        and comparison.loc[
            "regret_advantage_correction_vs_local", "cluster_bootstrap_ci_low"
        ]
        > 0.0
    )
    coverage_supported = bool(
        comparison.loc[
            "coverage_difference_correction_vs_local", "cluster_bootstrap_ci_low"
        ]
        >= -float(analysis["coverage_comparability_tolerance"])
        and comparison.loc[
            "coverage_difference_correction_vs_local", "cluster_bootstrap_ci_high"
        ]
        <= float(analysis["coverage_comparability_tolerance"])
    )
    top10_supported = bool(
        comparison.loc[
            "top10_advantage_correction_vs_local", "cluster_bootstrap_ci_low"
        ]
        >= -float(analysis["top10_noninferiority_margin"])
    )
    harm_bootstrap_defined = bool(
        comparison.loc[
            "harm_rate_advantage_correction_vs_local", "finite_bootstrap_fraction"
        ]
        >= float(analysis["minimum_finite_bootstrap_fraction"])
    )
    harm_supported = bool(
        harm_bootstrap_defined
        and comparison.loc[
            "harm_rate_advantage_correction_vs_local", "cluster_bootstrap_ci_low"
        ]
        >= -float(analysis["harm_rate_noninferiority_margin"])
    )
    actual_utility_supported = bool(
        comparison.loc[
            "regret_advantage_correction_vs_target", "cluster_bootstrap_ci_low"
        ]
        > 0.0
        or comparison.loc[
            "top10_advantage_correction_vs_target", "cluster_bootstrap_ci_low"
        ]
        > 0.0
    )
    oracle_headroom_supported = bool(
        comparison.loc["oracle_regret_headroom", "estimate"]
        >= float(analysis["minimum_oracle_normalized_regret_headroom"])
        and comparison.loc[
            "oracle_regret_headroom", "cluster_bootstrap_ci_low"
        ]
        > 0.0
    )
    checks = {
        "sufficient_effective_events": not sparse_reasons,
        "correction_auprc_superior": auprc_supported,
        "correction_regret_superior_by_margin": regret_supported,
        "coverage_comparable": coverage_supported,
        "top10_noninferior": top10_supported,
        "accepted_harm_bootstrap_defined": harm_bootstrap_defined,
        "accepted_harm_noninferior": harm_supported,
        "correction_improves_target_only": actual_utility_supported,
        "oracle_has_nontrivial_headroom": oracle_headroom_supported,
    }
    if sparse_reasons:
        verdict = "inconclusive_sparse_evidence"
    elif all(checks.values()):
        verdict = "advance_disagreement_conditioned_trust"
    else:
        verdict = "do_not_advance_complex_disagreement_trust"
    return {
        "verdict": verdict,
        "checks": checks,
        "sparse_reasons": sparse_reasons,
        "counts": {
            "holdout_events": int(len(holdout_events)),
            "actionable_events": actionable_events,
            "positive_benefit_events": positive_events,
            "common_support_events": common_support_events,
            "common_support_positive_events": common_support_positive_events,
            "correction_acceptances": correction_accepts,
            "local_spearman_acceptances": local_accepts,
        },
        "thresholds": {
            key: analysis[key]
            for key in [
                "practical_normalized_regret_margin",
                "minimum_actionable_events",
                "minimum_positive_events",
                "minimum_accepted_events",
                "coverage_comparability_tolerance",
                "top10_noninferiority_margin",
                "harm_rate_noninferiority_margin",
                "minimum_oracle_normalized_regret_headroom",
                "minimum_finite_bootstrap_fraction",
            ]
        },
        "headline": {
            "correction_common_support_auprc": float(
                prediction_index.loc[CORRECTION, "common_support_auprc"]
            ),
            "local_spearman_common_support_auprc": float(
                prediction_index.loc[LOCAL, "common_support_auprc"]
            ),
            "common_support_positive_prevalence": float(
                prediction_index.loc[CORRECTION, "common_support_prevalence"]
            ),
        },
    }


def relation_summary(method_outcomes: pd.DataFrame) -> pd.DataFrame:
    holdout = method_outcomes[method_outcomes["split"] == "holdout"]
    rows = []
    for (relation, method), group in holdout.groupby(["relation", "method"]):
        rows.append({"relation": relation, "method": method, **method_summary(group)})
    return pd.DataFrame(rows)


def plot_results(
    summary: pd.DataFrame,
    prediction: pd.DataFrame,
    output: Path,
) -> None:
    indexed = summary.set_index("method").loc[METHODS]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=180)
    axes[0].barh(METHODS, indexed["mean_normalized_regret"])
    axes[0].set_xlabel("Mean normalized one-step regret (lower better)")
    axes[0].invert_yaxis()
    axes[1].barh(METHODS, indexed["top10_hit_rate"])
    axes[1].set_xlabel("Top-10% hit rate (higher better)")
    axes[1].invert_yaxis()
    prediction_plot = prediction[
        prediction["method"].isin(DEPLOYABLE)
    ].set_index("method").loc[DEPLOYABLE]
    axes[2].barh(DEPLOYABLE, prediction_plot["common_support_auprc"])
    prevalence = float(prediction_plot["common_support_prevalence"].iloc[0])
    axes[2].axvline(prevalence, linestyle="--", color="black", label="prevalence")
    axes[2].set_xlabel("Held-out benefit AUPRC")
    axes[2].invert_yaxis()
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(output / "trust_validation_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def format_value(value: float) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.4f}"


def write_report(
    summary: pd.DataFrame,
    prediction: pd.DataFrame,
    comparisons: pd.DataFrame,
    decision: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    output: Path,
    config: Mapping[str, Any],
) -> None:
    indexed = summary.set_index("method")
    prediction_index = prediction.set_index("method")
    comparison = comparisons.set_index("metric")
    correction_regret = comparison.loc["regret_advantage_correction_vs_local"]
    correction_auprc = comparison.loc["auprc_advantage_correction_vs_local"]
    correction_target = comparison.loc["regret_advantage_correction_vs_target"]
    oracle_headroom = comparison.loc["oracle_regret_headroom"]

    lines = [
        "# 冲突条件信任最小验证结论",
        "",
        f"- Stage: `{config['stage_id']}`",
        f"- 预设判定：**{decision['verdict']}**",
        f"- Holdout：{decision['counts']['holdout_events']} 个事件，"
        f"{decision['counts']['actionable_events']} 个有效源—目标冲突，"
        f"{decision['counts']['positive_benefit_events']} 个真实正收益事件；"
        f"共同 eligible 支持集 {decision['counts']['common_support_events']} 个事件。",
        "",
        "## 一、五种方法的候选结果",
        "",
        "| 方法 | normalized regret↓ | Top-10% 命中↑ | actionable coverage | 接受后负迁移率 | 接受数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = indexed.loc[method]
        lines.append(
            f"| {method} | {row['mean_normalized_regret']:.4f} | "
            f"{row['top10_hit_rate']:.4f} | "
            f"{format_value(float(row['actionable_acceptance_coverage']))} | "
            f"{format_value(float(row['accepted_negative_transfer_rate']))} | "
            f"{int(row['accepted_events'])} |"
        )

    lines.extend(
        [
            "",
            "## 二、能否预测下一次源建议有益",
            "",
            "| Gate | 共同支持 AUPRC↑ | 共同支持正例率 | 单门 eligible coverage | 全 actionable neutral-imputed AUPRC |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method in DEPLOYABLE:
        row = prediction_index.loc[method]
        lines.append(
            f"| {method} | {row['common_support_auprc']:.4f} | "
            f"{row['common_support_prevalence']:.4f} | "
            f"{row['eligible_prediction_coverage']:.4f} | "
            f"{row['neutral_imputed_actionable_auprc']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 三、核心配对判断",
            "",
            f"- 冲突 Gate 相对 Local Spearman 的 normalized-regret 优势："
            f"`{correction_regret['estimate']:+.4f}`，seed-cluster bootstrap 95% CI "
            f"`[{correction_regret['cluster_bootstrap_ci_low']:+.4f}, "
            f"{correction_regret['cluster_bootstrap_ci_high']:+.4f}]`。",
            f"- 冲突 Gate 相对 Local Spearman 的 AUPRC 优势："
            f"`{correction_auprc['estimate']:+.4f}`，95% CI "
            f"`[{correction_auprc['cluster_bootstrap_ci_low']:+.4f}, "
            f"{correction_auprc['cluster_bootstrap_ci_high']:+.4f}]`。",
            f"- 冲突 Gate 相对 Target-Only 的 normalized-regret 优势："
            f"`{correction_target['estimate']:+.4f}`，95% CI "
            f"`[{correction_target['cluster_bootstrap_ci_low']:+.4f}, "
            f"{correction_target['cluster_bootstrap_ci_high']:+.4f}]`。",
            f"- Oracle 可达 regret headroom：`{oracle_headroom['estimate']:+.4f}`，"
            f"95% CI `[{oracle_headroom['cluster_bootstrap_ci_low']:+.4f}, "
            f"{oracle_headroom['cluster_bootstrap_ci_high']:+.4f}]`。",
            "",
            "## 四、预设推进检查",
            "",
        ]
    )
    for check, passed in decision["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — `{check}`")
    if decision["sparse_reasons"]:
        lines.append(
            "- 稀疏证据原因：" + ", ".join(decision["sparse_reasons"])
        )

    lines.extend(
        [
            "",
            "## 五、阈值与公平性边界",
            "",
            "Development 阈值只根据连续分数、eligible 状态和 actionable coverage 冻结，未使用最终候选面板标签：",
            "",
        ]
    )
    for method in DEPLOYABLE:
        value = thresholds[method]
        lines.append(
            f"- {method}: threshold `{value['threshold']:.6f}`；"
            f"Development actionable coverage "
            f"`{value['achieved_actionable_coverage']:.3f}`。"
        )
    lines.extend(
        [
            "",
            "所有方法共享 source-blind 付费成对诊断反馈、目标历史、Target-Only GP、候选池、"
            "源专家、Top-K 适用集合、源提名和真实评价面板；三个可部署 gate 只能接受同一个 "
            "`x_S` 或精确回退同一个 `x_T`。主 AUPRC 只在三个 gate 共同 eligible 的支持集上计算；"
            "Oracle 只在真值揭示后计算，不进入可部署 AUPRC。",
            "",
            "## 六、结论边界",
            "",
            "本实验只判断是否值得继续设计这层信任机制。即使通过，也不证明实际区域检索、"
            "未知对齐、多源调度、闭环 BO 预算收益、高维泛化或普遍无负迁移。",
        ]
    )
    (output / "DISAGREEMENT_TRUST_VALIDATION_REPORT_CN.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def analyze(input_dir: Path, config_path: Path, output: Path) -> Dict[str, Any]:
    audit = load_json(input_dir / "AUDIT.json")
    if not audit.get("ok", False):
        raise RuntimeError(f"Refusing to analyze failed audit: {audit.get('errors')}")
    config = load_json(config_path)
    manifest = load_json(input_dir / "run_manifest.json")
    expected_config_hash = canonical_config_hash(config)
    if manifest.get("config_sha256") != expected_config_hash:
        raise RuntimeError("Analysis config does not match the run manifest config hash.")
    manifest_config = manifest.get("config")
    if not isinstance(manifest_config, dict) or canonical_config_hash(
        manifest_config
    ) != expected_config_hash:
        raise RuntimeError("Embedded run-manifest config is missing or inconsistent.")
    for relative_path, expected_hash in manifest.get("artifact_sha256", {}).items():
        artifact = REPO_ROOT / relative_path
        if not artifact.exists() or file_hash(artifact) != expected_hash:
            raise RuntimeError(f"Run artifact hash mismatch: {relative_path}")
    method_outcomes = pd.read_csv(input_dir / "method_outcomes.csv")
    revealed_events = pd.read_csv(input_dir / "revealed_event_outcomes.csv")
    thresholds = load_json(input_dir / "frozen_gate_thresholds.json")
    holdout = method_outcomes[method_outcomes["split"] == "holdout"]

    summary_rows = []
    for method in METHODS:
        group = holdout[holdout["method"] == method]
        summary_rows.append({"method": method, **method_summary(group)})
    summary = pd.DataFrame(summary_rows)
    prediction = prediction_metrics(method_outcomes, revealed_events)
    bootstrap = cluster_bootstrap(
        method_outcomes,
        revealed_events,
        n_bootstrap=int(config["analysis"]["bootstrap_samples"]),
        seed=int(config["analysis"]["bootstrap_seed"]),
    )
    point = point_contrasts(summary, prediction)
    comparisons = comparison_table(point, bootstrap)
    relation = relation_summary(method_outcomes)
    decision = evaluate_decision(
        summary,
        prediction,
        comparisons,
        revealed_events,
        config,
    )

    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "holdout_method_summary.csv", index=False)
    prediction.to_csv(output / "holdout_prediction_metrics.csv", index=False)
    comparisons.to_csv(output / "paired_comparisons.csv", index=False)
    relation.to_csv(output / "holdout_relation_summary.csv", index=False)
    bootstrap.to_csv(output / "cluster_bootstrap_samples.csv", index=False)
    (output / "DECISION.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    plot_results(summary, prediction, output)
    write_report(
        summary,
        prediction,
        comparisons,
        decision,
        thresholds,
        output,
        config,
    )
    artifacts = [
        output / "holdout_method_summary.csv",
        output / "holdout_prediction_metrics.csv",
        output / "paired_comparisons.csv",
        output / "holdout_relation_summary.csv",
        output / "cluster_bootstrap_samples.csv",
        output / "DECISION.json",
        output / "trust_validation_summary.png",
        output / "DISAGREEMENT_TRUST_VALIDATION_REPORT_CN.md",
    ]
    analysis_audit = {
        "ok": all(path.exists() for path in artifacts),
        "input_audit_ok": bool(audit.get("ok", False)),
        "stage_id": config["stage_id"],
        "run_manifest_config_hash_verified": True,
        "run_artifact_hashes_verified": True,
        "config_sha256": expected_config_hash,
        "holdout_method_rows": int(len(holdout)),
        "expected_holdout_method_rows": int(
            int(config["holdout_seed_count"])
            * len(config["relations"])
            * len(METHODS)
        ),
        "bootstrap_samples": int(len(bootstrap)),
        "expected_bootstrap_samples": int(config["analysis"]["bootstrap_samples"]),
        "artifact_sha256": {path.name: file_hash(path) for path in artifacts},
    }
    analysis_audit["ok"] = bool(
        analysis_audit["ok"]
        and analysis_audit["input_audit_ok"]
        and analysis_audit["holdout_method_rows"]
        == analysis_audit["expected_holdout_method_rows"]
        and analysis_audit["bootstrap_samples"]
        == analysis_audit["expected_bootstrap_samples"]
    )
    (output / "ANALYSIS_AUDIT.json").write_text(
        json.dumps(analysis_audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if not analysis_audit["ok"]:
        raise RuntimeError(f"Analysis audit failed: {analysis_audit}")
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    return decision


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "results" / "disagreement_trust_validation_quick",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "disagreement_trust_validation_quick.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPO_ROOT
            / "results"
            / "disagreement_trust_validation_quick"
            / "analysis"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyze(resolve(args.input), resolve(args.config), resolve(args.output))
