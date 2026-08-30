"""Data-driven analysis for Local-Surrogate Transfer Pilot v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent

PRIMARY_HYPOTHESES = [
    {
        "hypothesis": "H1_Matching_sRMSE_Calibrated_vs_TargetOnly",
        "kind": "method",
        "relation": "matching",
        "metric": "standardized_rmse",
        "method_a": "Calibrated-Source+Residual",
        "method_b": "Target-Only",
        "higher_is_better": False,
    },
    {
        "hypothesis": "H2_Matching_NDCG_Calibrated_vs_TargetOnly",
        "kind": "method",
        "relation": "matching",
        "metric": "ndcg_at_top",
        "method_a": "Calibrated-Source+Residual",
        "method_b": "Target-Only",
        "higher_is_better": True,
    },
    {
        "hypothesis": "H3_Reversed_sRMSE_Gated_vs_Fixed",
        "kind": "method",
        "relation": "reversed",
        "metric": "standardized_rmse",
        "method_a": "Gated-Source+Residual",
        "method_b": "Fixed-Source+Residual",
        "higher_is_better": False,
    },
    {
        "hypothesis": "H4_GateAcceptance_Matching_vs_Reversed",
        "kind": "relation",
        "method": "Gated-Source+Residual",
        "metric": "gate_accepted",
        "relation_a": "matching",
        "relation_b": "reversed",
        "higher_is_better": True,
    },
]


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object.")
    return value


def mean_bootstrap_ci(
    values: Iterable[float],
    n_bootstrap: int,
    seed: int = 42,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bootstrap = rng.choice(array, size=(n_bootstrap, len(array)), replace=True)
    estimates = np.mean(bootstrap, axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.mean(array)),
        float(np.quantile(estimates, alpha)),
        float(np.quantile(estimates, 1.0 - alpha)),
    )


def safe_wilcoxon_greater(values: Sequence[float]) -> float:
    differences = np.asarray(values, dtype=float)
    differences = differences[np.isfinite(differences)]
    if len(differences) == 0 or np.all(np.abs(differences) < 1e-15):
        return 1.0
    try:
        return float(
            wilcoxon(
                differences,
                zero_method="pratt",
                alternative="greater",
            ).pvalue
        )
    except ValueError:
        return 1.0


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def summarize_differences(
    differences: np.ndarray,
    n_bootstrap: int,
) -> Dict:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    mean, low, high = mean_bootstrap_ci(values, n_bootstrap=n_bootstrap)
    nonzero = values[np.abs(values) > 1e-15]
    rank_biserial = (
        float((np.sum(nonzero > 0.0) - np.sum(nonzero < 0.0)) / len(nonzero))
        if len(nonzero)
        else 0.0
    )
    return {
        "n_pairs": int(len(values)),
        "mean_advantage": mean,
        "ci_low": low,
        "ci_high": high,
        "wilcoxon_one_sided_p": safe_wilcoxon_greater(values),
        "rank_biserial": rank_biserial,
        "win_rate": float(np.mean(values > 1e-15)) if len(values) else float("nan"),
        "tie_rate": float(np.mean(np.abs(values) <= 1e-15)) if len(values) else float("nan"),
        "loss_rate": float(np.mean(values < -1e-15)) if len(values) else float("nan"),
    }


def method_comparison(
    frame: pd.DataFrame,
    relation: str,
    context_size: int,
    metric: str,
    method_a: str,
    method_b: str,
    higher_is_better: bool,
    n_bootstrap: int,
) -> Dict:
    subset = frame[
        (frame["relation"] == relation)
        & (frame["target_context_size"] == context_size)
    ]
    keys = ["problem", "dim", "seed", "relation", "target_context_size"]
    first = subset[subset["method"] == method_a][keys + [metric]].rename(
        columns={metric: "a"}
    )
    second = subset[subset["method"] == method_b][keys + [metric]].rename(
        columns={metric: "b"}
    )
    paired = first.merge(second, on=keys, how="inner")
    raw = paired["a"].astype(float).to_numpy() - paired["b"].astype(float).to_numpy()
    oriented = raw if higher_is_better else -raw
    return {
        "relation": relation,
        "metric": metric,
        "method_a": method_a,
        "method_b": method_b,
        **summarize_differences(oriented, n_bootstrap),
    }


def relation_comparison(
    frame: pd.DataFrame,
    method: str,
    context_size: int,
    metric: str,
    relation_a: str,
    relation_b: str,
    higher_is_better: bool,
    n_bootstrap: int,
) -> Dict:
    subset = frame[
        (frame["method"] == method)
        & (frame["target_context_size"] == context_size)
    ].copy()
    subset[metric] = subset[metric].astype(float)
    keys = ["problem", "dim", "seed", "method", "target_context_size"]
    first = subset[subset["relation"] == relation_a][keys + [metric]].rename(
        columns={metric: "a"}
    )
    second = subset[subset["relation"] == relation_b][keys + [metric]].rename(
        columns={metric: "b"}
    )
    paired = first.merge(second, on=keys, how="inner")
    raw = paired["a"].to_numpy() - paired["b"].to_numpy()
    oriented = raw if higher_is_better else -raw
    return {
        "relation": f"{relation_a}_vs_{relation_b}",
        "metric": metric,
        "method_a": f"{method}@{relation_a}",
        "method_b": f"{method}@{relation_b}",
        **summarize_differences(oriented, n_bootstrap),
    }


def run_analysis(input_dir: Path, config_path: Path, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_json(config_path)
    pilot = dict(config["pilot"])
    analysis = dict(config["analysis"])
    primary_context = int(pilot["primary_context_size"])
    n_bootstrap = int(analysis.get("bootstrap_samples", 5000))
    alpha = float(analysis.get("familywise_alpha", 0.05))

    results = pd.read_csv(input_dir / "local_surrogate_transfer_results.csv")
    diagnostics = pd.read_csv(
        input_dir / "local_surrogate_transfer_diagnostics.csv"
    )
    failure_path = input_dir / "local_surrogate_transfer_failures.csv"
    failures = (
        pd.read_csv(failure_path)
        if failure_path.exists() and failure_path.stat().st_size > 0
        else pd.DataFrame()
    )

    primary_rows: List[Dict] = []
    for specification in PRIMARY_HYPOTHESES:
        if specification["kind"] == "method":
            result = method_comparison(
                results,
                relation=specification["relation"],
                context_size=primary_context,
                metric=specification["metric"],
                method_a=specification["method_a"],
                method_b=specification["method_b"],
                higher_is_better=specification["higher_is_better"],
                n_bootstrap=n_bootstrap,
            )
        else:
            result = relation_comparison(
                results,
                method=specification["method"],
                context_size=primary_context,
                metric=specification["metric"],
                relation_a=specification["relation_a"],
                relation_b=specification["relation_b"],
                higher_is_better=specification["higher_is_better"],
                n_bootstrap=n_bootstrap,
            )
        result["hypothesis"] = specification["hypothesis"]
        result["primary_context_size"] = primary_context
        primary_rows.append(result)

    primary = pd.DataFrame(primary_rows)
    primary["holm_adjusted_p"] = holm_adjust(
        primary["wilcoxon_one_sided_p"].to_numpy()
    )
    primary["supported"] = (
        (primary["ci_low"] > 0.0)
        & (primary["holm_adjusted_p"] < alpha)
    )
    primary.to_csv(output_dir / "local_surrogate_transfer_primary_tests.csv", index=False)

    summary = (
        results.groupby(
            ["relation", "target_context_size", "method"],
            as_index=False,
        )
        .agg(
            instances=("seed", "size"),
            standardized_rmse=("standardized_rmse", "mean"),
            ndcg_at_top=("ndcg_at_top", "mean"),
            pairwise_accuracy=("pairwise_accuracy", "mean"),
            normalized_top1_regret=("normalized_top1_regret", "mean"),
            negative_transfer_rate=("negative_transfer", "mean"),
            interval_coverage_95=("interval_coverage_95", "mean"),
        )
    )
    summary.to_csv(output_dir / "local_surrogate_transfer_summary.csv", index=False)

    gated = results[results["method"] == "Gated-Source+Residual"].copy()
    gate_summary = (
        gated.groupby(["relation", "target_context_size"], as_index=False)
        .agg(
            instances=("seed", "size"),
            acceptance_coverage=("gate_accepted", "mean"),
            intention_to_use_negative_transfer_rate=("negative_transfer", "mean"),
            mean_srmse_delta=("srmse_delta_vs_target_only", "mean"),
            mean_context_cv_gain=("cv_relative_rmse_gain", "mean"),
            mean_context_pairwise=("context_pairwise_accuracy", "mean"),
        )
    )
    accepted = gated[gated["gate_accepted"].astype(bool)]
    accepted_risk = (
        accepted.groupby(["relation", "target_context_size"], as_index=False)
        .agg(
            accepted_instances=("seed", "size"),
            accepted_negative_transfer_risk=("negative_transfer", "mean"),
            accepted_mean_srmse_delta=("srmse_delta_vs_target_only", "mean"),
        )
        if not accepted.empty
        else pd.DataFrame(
            columns=[
                "relation",
                "target_context_size",
                "accepted_instances",
                "accepted_negative_transfer_risk",
                "accepted_mean_srmse_delta",
            ]
        )
    )
    gate_summary = gate_summary.merge(
        accepted_risk,
        on=["relation", "target_context_size"],
        how="left",
    )
    gate_summary["accepted_instances"] = gate_summary["accepted_instances"].fillna(0).astype(int)
    gate_summary.to_csv(output_dir / "local_surrogate_transfer_gate_summary.csv", index=False)

    _plot_response_curves(summary, output_dir)
    _plot_gate_summary(gate_summary, output_dir)
    _write_report(
        output_dir / "LOCAL_SURROGATE_TRANSFER_REPORT.md",
        config,
        results,
        diagnostics,
        failures,
        primary,
        summary,
        gate_summary,
    )
    return primary


def _plot_response_curves(summary: pd.DataFrame, output_dir: Path) -> None:
    selected = [
        "Target-Only",
        "Fixed-Source+Residual",
        "Calibrated-Source+Residual",
        "Gated-Source+Residual",
    ]
    for relation in ["matching", "wrong", "reversed"]:
        subset = summary[
            (summary["relation"] == relation)
            & (summary["method"].isin(selected))
        ]
        if subset.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=180)
        for method, method_frame in subset.groupby("method"):
            ordered = method_frame.sort_values("target_context_size")
            axes[0].plot(
                ordered["target_context_size"],
                ordered["standardized_rmse"],
                marker="o",
                label=method,
            )
            axes[1].plot(
                ordered["target_context_size"],
                ordered["ndcg_at_top"],
                marker="o",
                label=method,
            )
        axes[0].set_ylabel("Mean standardized RMSE")
        axes[1].set_ylabel("Mean NDCG@top")
        for axis in axes:
            axis.set_xlabel("Target context size")
            axis.grid(True, linestyle=":", alpha=0.6)
        axes[0].set_title(f"{relation}: prediction")
        axes[1].set_title(f"{relation}: decision ranking")
        axes[1].legend(fontsize=8, frameon=True)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"local_surrogate_transfer_{relation}_curves.png",
            dpi=300,
        )
        plt.close(fig)


def _plot_gate_summary(gate_summary: pd.DataFrame, output_dir: Path) -> None:
    if gate_summary.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), dpi=180)
    for relation, relation_frame in gate_summary.groupby("relation"):
        ordered = relation_frame.sort_values("target_context_size")
        axes[0].plot(
            ordered["target_context_size"],
            ordered["acceptance_coverage"],
            marker="o",
            label=relation,
        )
        axes[1].plot(
            ordered["target_context_size"],
            ordered["intention_to_use_negative_transfer_rate"],
            marker="o",
            label=relation,
        )
    axes[0].set_ylabel("Gate acceptance coverage")
    axes[1].set_ylabel("Intention-to-use negative-transfer rate")
    for axis in axes:
        axis.set_xlabel("Target context size")
        axis.set_ylim(-0.03, 1.03)
        axis.grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_dir / "local_surrogate_transfer_gate.png", dpi=300)
    plt.close(fig)


def _write_report(
    path: Path,
    config: Mapping,
    results: pd.DataFrame,
    diagnostics: pd.DataFrame,
    failures: pd.DataFrame,
    primary: pd.DataFrame,
    summary: pd.DataFrame,
    gate_summary: pd.DataFrame,
) -> None:
    pilot = dict(config["pilot"])
    primary_context = int(pilot["primary_context_size"])
    lines: List[str] = []
    lines.append("# Source Local-Surrogate Transfer Pilot v1 Report\n")
    lines.append(
        "This report is generated from frozen CSV artifacts. The statistical unit is one `(problem, dimension, seed)` task instance; candidate-panel points are not treated as replicates.\n"
    )
    lines.append("## 1. Scope\n")
    lines.append(
        "This is a controlled 2D static held-out model-transfer Pilot under oracle region correspondence and a frozen isotropic local chart. It does not test unknown alignment, online BO, or a general no-harm guarantee.\n"
    )
    lines.append("## 2. Pre-specified primary tests\n")
    lines.append(
        "| Hypothesis | Pairs | Mean oriented advantage [95% bootstrap CI] | Holm p | Rank-biserial | Supported |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, row in primary.iterrows():
        lines.append(
            f"| {row['hypothesis']} | {int(row['n_pairs'])} | "
            f"{row['mean_advantage']:+.4f} [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | "
            f"{row['holm_adjusted_p']:.4g} | {row['rank_biserial']:+.3f} | "
            f"{'yes' if bool(row['supported']) else 'no'} |"
        )

    lines.append(f"\n## 3. Primary-slice model means (context={primary_context})\n")
    primary_summary = summary[
        summary["target_context_size"] == primary_context
    ]
    lines.append(
        "| Relation | Method | sRMSE | NDCG@top | Pairwise accuracy | Top-1 regret | Negative-transfer rate |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for _, row in primary_summary.iterrows():
        lines.append(
            f"| {row['relation']} | {row['method']} | "
            f"{row['standardized_rmse']:.4f} | {row['ndcg_at_top']:.4f} | "
            f"{row['pairwise_accuracy']:.4f} | {row['normalized_top1_regret']:.4f} | "
            f"{row['negative_transfer_rate']:.3f} |"
        )

    lines.append(f"\n## 4. Gate behavior (context={primary_context})\n")
    primary_gate = gate_summary[
        gate_summary["target_context_size"] == primary_context
    ]
    lines.append(
        "| Relation | Acceptance coverage | Accepted instances | Risk among accepted | Intention-to-use risk | Mean sRMSE delta |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, row in primary_gate.iterrows():
        accepted_risk = row["accepted_negative_transfer_risk"]
        risk_text = "NA" if not np.isfinite(accepted_risk) else f"{accepted_risk:.3f}"
        lines.append(
            f"| {row['relation']} | {row['acceptance_coverage']:.3f} | "
            f"{int(row['accepted_instances'])} | {risk_text} | "
            f"{row['intention_to_use_negative_transfer_rate']:.3f} | "
            f"{row['mean_srmse_delta']:+.4f} |"
        )

    lines.append("\n## 5. Source-expert and support diagnostics\n")
    if diagnostics.empty:
        lines.append("No diagnostic rows were produced.\n")
    else:
        lines.append(
            "| Relation | Source held-out NDCG | Source held-out pairwise accuracy | Below-membership-0.05 fraction | Normalized extraction-center error |"
        )
        lines.append("|---|---:|---:|---:|---:|")
        diagnostic_summary = (
            diagnostics.groupby("relation", as_index=False)
            .agg(
                source_ndcg=("source_fidelity_ndcg", "mean"),
                source_pairwise=("source_fidelity_pairwise", "mean"),
                low_membership=("source_membership_below_0_05", "mean"),
                center_error=("normalized_anchor_error", "mean"),
            )
        )
        for _, row in diagnostic_summary.iterrows():
            lines.append(
                f"| {row['relation']} | {row['source_ndcg']:.4f} | "
                f"{row['source_pairwise']:.4f} | {row['low_membership']:.4f} | "
                f"{row['center_error']:.4f} |"
            )
        lines.append("")

    lines.append("## 6. Completeness\n")
    lines.append(
        f"Result rows: {len(results)}; diagnostic rows: {len(diagnostics)}; failures: {len(failures)}. Primary tests use complete paired instances only.\n"
    )

    lines.append("## 7. Data-derived interpretation boundary\n")
    supported = primary[primary["supported"]]
    if supported.empty:
        lines.append(
            "None of the four pre-specified Pilot hypotheses is supported after bootstrap uncertainty and Holm correction. The current local-model transfer mechanism should not advance to online BO without a new, separately frozen revision.\n"
        )
    else:
        lines.append("Supported Pilot hypotheses:\n")
        for hypothesis in supported["hypothesis"]:
            lines.append(f"- {hypothesis}")
        lines.append("")
    lines.append(
        "Any supported matching result applies only to correct oracle region correspondence in this 2D controlled Pilot. Wrong-source behavior and explicit reversal behavior are separate safety stress tests. Gate rejection is not proof that a source is intrinsically non-transferable, and low observed harm is not a universal no-negative-transfer guarantee.\n"
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "results" / "local_surrogate_transfer_pilot_quick",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "local_surrogate_transfer_quick.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "local_surrogate_transfer_pilot_quick" / "analysis",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_analysis(arguments.input, arguments.config, arguments.output)
