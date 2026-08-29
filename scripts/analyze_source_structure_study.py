"""Data-driven analysis for source local-structure extraction studies."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent


PRIMARY_COMPARISONS = [
    {
        "name": "H1_Source_NDCG_Proposed_vs_Geometry",
        "dataset": "validation",
        "metric": "ndcg_at_top",
        "higher_is_better": True,
        "filters": {
            "evaluation_domain": "source",
            "subset": "all",
        },
        "method_a": "Proposed-Local-Structure",
        "method_b": "Geometry-Only",
    },
    {
        "name": "H2_Source_Local_Spearman_Proposed_vs_Geometry",
        "dataset": "validation",
        "metric": "spearman",
        "higher_is_better": True,
        "filters": {
            "evaluation_domain": "source",
            "subset": "local",
        },
        "method_a": "Proposed-Local-Structure",
        "method_b": "Geometry-Only",
    },
    {
        "name": "H3_Source_NDCG_Proposed_vs_Permutation",
        "dataset": "validation",
        "metric": "ndcg_at_top",
        "higher_is_better": True,
        "filters": {
            "evaluation_domain": "source",
            "subset": "all",
        },
        "method_a": "Proposed-Local-Structure",
        "method_b": "Label-Permutation",
    },
    {
        "name": "H4_Matching_Target_NDCG_Proposed_vs_Geometry",
        "dataset": "validation",
        "metric": "ndcg_at_top",
        "higher_is_better": True,
        "filters": {
            "evaluation_domain": "target",
            "subset": "all",
            "source_scenario": "matching",
        },
        "method_a": "Proposed-Local-Structure",
        "method_b": "Geometry-Only",
    },
    {
        "name": "H5_Recovery_Recall_Proposed_vs_TopObservations",
        "dataset": "recovery",
        "metric": "basin_recall",
        "higher_is_better": True,
        "filters": {},
        "method_a": "Proposed-Local-Structure",
        "method_b": "Top-Observations",
    },
    {
        "name": "H6_Recovery_CenterError_Proposed_vs_TopObservations",
        "dataset": "recovery",
        "metric": "normalized_center_error",
        "higher_is_better": False,
        "filters": {},
        "method_a": "Proposed-Local-Structure",
        "method_b": "Top-Observations",
    },
]


def mean_bootstrap_ci(
    values: Iterable[float],
    n_bootstrap: int = 5000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(n_bootstrap, len(array)), replace=True)
    means = np.mean(samples, axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.mean(array)),
        float(np.quantile(means, alpha)),
        float(np.quantile(means, 1.0 - alpha)),
    )


def safe_wilcoxon(differences: Sequence[float]) -> float:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0 or np.all(np.abs(values) < 1e-15):
        return 1.0
    try:
        return float(wilcoxon(values, zero_method="pratt", alternative="greater").pvalue)
    except ValueError:
        return 1.0


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    m = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def paired_comparison(
    frame: pd.DataFrame,
    metric: str,
    method_a: str,
    method_b: str,
    higher_is_better: bool,
    filters: Dict,
    method_column: str,
    instance_columns: Sequence[str],
) -> Dict:
    subset = frame.copy()
    for key, value in filters.items():
        subset = subset[subset[key] == value]

    first = subset[subset[method_column] == method_a]
    second = subset[subset[method_column] == method_b]
    first = first[list(instance_columns) + [metric]].rename(columns={metric: "a"})
    second = second[list(instance_columns) + [metric]].rename(columns={metric: "b"})
    paired = first.merge(second, on=list(instance_columns), how="inner")
    paired = paired[np.isfinite(paired["a"]) & np.isfinite(paired["b"])]

    raw_difference = paired["a"].to_numpy() - paired["b"].to_numpy()
    oriented = raw_difference if higher_is_better else -raw_difference
    mean, low, high = mean_bootstrap_ci(oriented)
    p_value = safe_wilcoxon(oriented)
    nonzero = oriented[np.abs(oriented) > 1e-15]
    rank_biserial = (
        float((np.sum(nonzero > 0) - np.sum(nonzero < 0)) / len(nonzero))
        if len(nonzero)
        else 0.0
    )
    return {
        "method_a": method_a,
        "method_b": method_b,
        "metric": metric,
        "higher_is_better": higher_is_better,
        "n_pairs": int(len(oriented)),
        "mean_advantage": mean,
        "ci_low": low,
        "ci_high": high,
        "wilcoxon_one_sided_p": p_value,
        "rank_biserial": rank_biserial,
        "win_rate": float(np.mean(oriented > 1e-15)) if len(oriented) else float("nan"),
        "tie_rate": float(np.mean(np.abs(oriented) <= 1e-15)) if len(oriented) else float("nan"),
        "loss_rate": float(np.mean(oriented < -1e-15)) if len(oriented) else float("nan"),
    }


def run_analysis(input_root: Path, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    recovery_path = input_root / "recovery" / "source_structure_recovery.csv"
    validation_path = input_root / "validation" / "source_structure_validation.csv"
    diagnostics_path = input_root / "validation" / "source_structure_diagnostics.csv"
    recovery_failures_path = input_root / "recovery" / "source_structure_recovery_failures.csv"
    validation_failures_path = input_root / "validation" / "source_structure_validation_failures.csv"

    recovery = pd.read_csv(recovery_path)
    validation = pd.read_csv(validation_path)
    diagnostics = (
        pd.read_csv(diagnostics_path)
        if diagnostics_path.exists() and diagnostics_path.stat().st_size > 0
        else pd.DataFrame()
    )
    recovery_failures = (
        pd.read_csv(recovery_failures_path)
        if recovery_failures_path.exists() and recovery_failures_path.stat().st_size > 0
        else pd.DataFrame()
    )
    validation_failures = (
        pd.read_csv(validation_failures_path)
        if validation_failures_path.exists() and validation_failures_path.stat().st_size > 0
        else pd.DataFrame()
    )

    comparison_rows: List[Dict] = []
    for specification in PRIMARY_COMPARISONS:
        if specification["dataset"] == "validation":
            result = paired_comparison(
                validation,
                metric=specification["metric"],
                method_a=specification["method_a"],
                method_b=specification["method_b"],
                higher_is_better=specification["higher_is_better"],
                filters=specification["filters"],
                method_column="method",
                instance_columns=[
                    "problem",
                    "dim",
                    "seed",
                    "source_scenario",
                    "source_index",
                    "evaluation_domain",
                    "subset",
                ],
            )
        else:
            result = paired_comparison(
                recovery,
                metric=specification["metric"],
                method_a=specification["method_a"],
                method_b=specification["method_b"],
                higher_is_better=specification["higher_is_better"],
                filters=specification["filters"],
                method_column="method",
                instance_columns=[
                    "dim",
                    "seed",
                    "sample_size",
                    "noise_level",
                ],
            )
        result["hypothesis"] = specification["name"]
        comparison_rows.append(result)

    comparison = pd.DataFrame(comparison_rows)
    comparison["holm_adjusted_p"] = holm_adjust(
        comparison["wilcoxon_one_sided_p"].to_numpy()
    )
    comparison["supported"] = (
        (comparison["ci_low"] > 0.0)
        & (comparison["holm_adjusted_p"] < 0.05)
    )
    comparison.to_csv(output_dir / "source_structure_primary_tests.csv", index=False)

    _plot_recovery(recovery, output_dir)
    _plot_validation(validation, output_dir)
    _write_report(
        output_dir / "SOURCE_STRUCTURE_REPORT.md",
        recovery,
        validation,
        diagnostics,
        comparison,
        recovery_failures,
        validation_failures,
    )
    return comparison


def _plot_recovery(recovery: pd.DataFrame, output_dir: Path) -> None:
    grouped = (
        recovery.groupby(["sample_size", "method"], as_index=False)["basin_recall"]
        .mean()
    )
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=180)
    for method, method_frame in grouped.groupby("method"):
        method_frame = method_frame.sort_values("sample_size")
        ax.plot(
            method_frame["sample_size"],
            method_frame["basin_recall"],
            marker="o",
            label=method,
        )
    ax.set_xlabel("Source training samples")
    ax.set_ylabel("Mean oracle-basin recall")
    ax.set_title("Controlled local-structure recovery")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_dir / "source_structure_recovery.png", dpi=300)
    plt.close(fig)


def _plot_validation(validation: pd.DataFrame, output_dir: Path) -> None:
    selected_methods = [
        "Proposed-Local-Structure",
        "Geometry-Only",
        "Global-Source-GP",
        "Label-Permutation",
    ]
    subset = validation[
        (validation["subset"] == "all")
        & (validation["method"].isin(selected_methods))
    ]
    grouped = (
        subset.groupby(["evaluation_domain", "source_scenario", "method"], as_index=False)[
            "ndcg_at_top"
        ]
        .mean()
    )

    for domain in ["source", "target"]:
        domain_frame = grouped[grouped["evaluation_domain"] == domain]
        if domain_frame.empty:
            continue
        pivot = domain_frame.pivot(
            index="method",
            columns="source_scenario",
            values="ndcg_at_top",
        )
        fig, ax = plt.subplots(figsize=(9, 4.8), dpi=180)
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel("Mean NDCG@top")
        ax.set_xlabel("")
        ax.set_title(f"Held-out {domain} ranking fidelity")
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)
        ax.legend(title="Source scenario", frameon=True)
        fig.tight_layout()
        fig.savefig(output_dir / f"source_structure_{domain}_ndcg.png", dpi=300)
        plt.close(fig)


def _write_report(
    path: Path,
    recovery: pd.DataFrame,
    validation: pd.DataFrame,
    diagnostics: pd.DataFrame,
    comparison: pd.DataFrame,
    recovery_failures: pd.DataFrame,
    validation_failures: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines.append("# Source local-structure extraction and validation report\n")
    lines.append(
        "This report is generated from frozen CSV artifacts. Statistical units are independent source-task instances, not individual candidate points.\n"
    )
    lines.append("## 1. Pre-specified primary tests\n")
    lines.append(
        "| Hypothesis | Pairs | Mean oriented advantage [95% bootstrap CI] | Holm p | Rank-biserial | Supported |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, row in comparison.iterrows():
        lines.append(
            f"| {row['hypothesis']} | {int(row['n_pairs'])} | "
            f"{row['mean_advantage']:+.4f} [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | "
            f"{row['holm_adjusted_p']:.4g} | {row['rank_biserial']:+.3f} | "
            f"{'yes' if bool(row['supported']) else 'no'} |"
        )

    lines.append("\n## 2. Controlled recovery summary\n")
    recovery_summary = (
        recovery.groupby("method")
        .agg(
            basin_recall=("basin_recall", "mean"),
            center_error=("normalized_center_error", "mean"),
            shape_error=("normalized_shape_error", "mean"),
        )
        .reset_index()
    )
    lines.append("| Method | Basin recall | Normalized center error | Shape error |")
    lines.append("|---|---:|---:|---:|")
    for _, row in recovery_summary.iterrows():
        lines.append(
            f"| {row['method']} | {row['basin_recall']:.4f} | "
            f"{row['center_error']:.4f} | {row['shape_error']:.4f} |"
        )

    lines.append("\n## 3. Held-out ranking summary\n")
    ranking_summary = (
        validation[
            (validation["subset"] == "all")
            & (validation["method"].isin([
                "Proposed-Local-Structure",
                "Geometry-Only",
                "Global-Source-GP",
                "Label-Permutation",
            ]))
        ]
        .groupby(["evaluation_domain", "source_scenario", "method"])
        .agg(
            ndcg=("ndcg_at_top", "mean"),
            spearman=("spearman", "mean"),
            precision=("precision_at_top", "mean"),
            regret=("normalized_top1_regret", "mean"),
        )
        .reset_index()
    )
    lines.append(
        "| Domain | Source scenario | Method | NDCG@top | Spearman | Precision@top | Top-1 regret |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for _, row in ranking_summary.iterrows():
        lines.append(
            f"| {row['evaluation_domain']} | {row['source_scenario']} | {row['method']} | "
            f"{row['ndcg']:.4f} | {row['spearman']:.4f} | "
            f"{row['precision']:.4f} | {row['regret']:.4f} |"
        )

    lines.append("\n## 4. Extraction diagnostics\n")
    if diagnostics.empty:
        lines.append("No structure-level diagnostic rows were produced.\n")
    else:
        lines.append(
            f"Structures evaluated: {len(diagnostics)}; mean context size: "
            f"{diagnostics['context_count'].mean():.2f}; mean boundary fraction: "
            f"{diagnostics['boundary_fraction'].mean():.3f}; mean OOF Spearman: "
            f"{diagnostics['oof_spearman'].mean():.3f}.\n"
        )

    supported = comparison[comparison["supported"]]
    lines.append("## 5. Completeness and failures\n")
    lines.append(
        f"Controlled-recovery failures: {len(recovery_failures)}; "
        f"held-out validation failures: {len(validation_failures)}. "
        "Primary tests use complete paired instances only.\n"
    )

    lines.append("## 6. Data-derived interpretation\n")
    if supported.empty:
        lines.append(
            "None of the pre-specified primary claims is supported after bootstrap uncertainty and Holm correction. The extraction method should not yet be described as empirically validated.\n"
        )
    else:
        lines.append("Supported primary claims:\n")
        for hypothesis in supported["hypothesis"]:
            lines.append(f"- {hypothesis}")
        lines.append("")
    lines.append(
        "Source fidelity and target transferability are reported separately. A structure may be extracted faithfully from its source task while remaining non-transferable to a mismatched target; this distinction must be preserved in any paper claim.\n"
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "results" / "source_structure_stage",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "source_structure_stage" / "analysis",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_analysis(args.input, args.output)
