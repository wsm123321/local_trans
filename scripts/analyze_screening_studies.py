"""Analyze source-region screening studies and generate data-driven reports.

No conclusion is hardcoded. All statements are derived from paired differences,
bootstrap confidence intervals, activation rates, and observed drift boundaries.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from region_guided_reranking_study.screening_research import load_json, mean_bootstrap_ci


METHOD_ORDER = [
    "Target-Only",
    "Matching-Fixed-Filter",
    "Matching-Adaptive-Filter",
    "Matching-Soft-Rerank",
    "Random-Adaptive-Filter",
    "Wrong-Adaptive-Filter",
    "Oracle-Fixed-Filter",
]


def _paired_difference(
    frame: pd.DataFrame,
    method: str,
    baseline: str,
    value_column: str,
) -> np.ndarray:
    index_columns = ["problem", "dim", "seed"]
    left = (
        frame[frame["method"] == method]
        .set_index(index_columns)[value_column]
        .rename("method")
    )
    right = (
        frame[frame["method"] == baseline]
        .set_index(index_columns)[value_column]
        .rename("baseline")
    )
    paired = pd.concat([left, right], axis=1, join="inner").dropna()
    # Positive means the named method has lower regret than baseline.
    return (paired["baseline"] - paired["method"]).to_numpy(dtype=float)


def _safe_wilcoxon(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or np.all(np.abs(values) < 1e-14):
        return 1.0
    try:
        return float(wilcoxon(values, zero_method="wilcox").pvalue)
    except ValueError:
        return 1.0


def _format_ci(mean: float, low: float, high: float) -> str:
    return f"{mean:+.4f} [{low:+.4f}, {high:+.4f}]"


def plot_mechanism(frame: pd.DataFrame, output: Path) -> None:
    methods = [method for method in METHOD_ORDER if method in set(frame["method"])]
    data = [
        frame.loc[frame["method"] == method, "normalized_regret"].to_numpy()
        for method in methods
    ]
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=180)
    ax.boxplot(data, labels=methods, showfliers=False)
    ax.set_ylabel("Proposal-set normalized regret (lower is better)")
    ax.set_title("Source-region screening: shared target-proposal decision quality")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(output / "mechanism_normalized_regret.png", dpi=300)
    plt.close(fig)


def plot_sequential(frame: pd.DataFrame, output: Path) -> None:
    methods = [method for method in METHOD_ORDER if method in set(frame["method"])]
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=180)
    for method in methods:
        subset = frame[frame["method"] == method]
        grouped = subset.groupby("step")["normalized_regret"]
        mean = grouped.mean()
        count = grouped.count().clip(lower=1)
        sem = grouped.std().fillna(0.0) / np.sqrt(count)
        steps = mean.index.to_numpy()
        ax.plot(steps, mean.to_numpy(), label=method, linewidth=2)
        ax.fill_between(
            steps,
            (mean - 1.96 * sem).to_numpy(),
            (mean + 1.96 * sem).to_numpy(),
            alpha=0.12,
        )
    ax.set_xlabel("Target evaluation step")
    ax.set_ylabel("Normalized simple regret")
    ax.set_title("Equal-budget closed-loop convergence")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "sequential_normalized_regret.png", dpi=300)
    plt.close(fig)


def plot_drift(frame: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=180)
    for method in sorted(frame["method"].unique()):
        subset = frame[frame["method"] == method]
        grouped = subset.groupby("delta")["regret_reduction"]
        mean = grouped.mean()
        count = grouped.count().clip(lower=1)
        sem = grouped.std().fillna(0.0) / np.sqrt(count)
        delta = mean.index.to_numpy(dtype=float)
        ax.plot(delta, mean.to_numpy(), marker="o", label=method, linewidth=2)
        ax.fill_between(
            delta,
            (mean - 1.96 * sem).to_numpy(),
            (mean + 1.96 * sem).to_numpy(),
            alpha=0.15,
        )
    ax.axhline(0.0, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Source-region center drift")
    ax.set_ylabel("Regret reduction relative to Target-Only")
    ax.set_title("Transferability boundary of source-region filtering")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "drift_transfer_boundary.png", dpi=300)
    plt.close(fig)


def _mechanism_report(frame: pd.DataFrame) -> List[str]:
    lines: List[str] = [
        "## 1. Shared-proposal mechanism study",
        "",
        "All methods used the same target observations, target GP, raw pool, and target-proposed candidates.",
        "",
        "| Method | Mean normalized regret | Top-10% hit rate | Mean retained fraction | Filter activation |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        subset = frame[frame["method"] == method]
        if subset.empty:
            continue
        lines.append(
            f"| {method} | {subset['normalized_regret'].mean():.4f} | "
            f"{100.0 * subset['hit_top10'].mean():.1f}% | "
            f"{subset['retained_fraction'].mean():.3f} | "
            f"{100.0 * subset['filter_active'].mean():.1f}% |"
        )

    lines.extend(["", "### Paired comparisons against Target-Only", ""])
    for method in METHOD_ORDER:
        if method == "Target-Only" or method not in set(frame["method"]):
            continue
        differences = _paired_difference(
            frame,
            method,
            "Target-Only",
            "normalized_regret",
        )
        mean, low, high = mean_bootstrap_ci(differences)
        p_value = _safe_wilcoxon(differences)
        wins = 100.0 * float(np.mean(differences > 1e-12)) if len(differences) else float("nan")
        lines.append(
            f"- **{method}**: regret reduction {_format_ci(mean, low, high)}; "
            f"Wilcoxon p={p_value:.4g}; paired win rate={wins:.1f}%."
        )
    lines.append("")
    return lines


def _sequential_report(summary: pd.DataFrame) -> List[str]:
    lines = [
        "## 2. Equal-budget sequential optimization",
        "",
        "| Method | Final normalized regret | Total improvement | Filter activation | Mean trust |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        subset = summary[summary["method"] == method]
        if subset.empty:
            continue
        lines.append(
            f"| {method} | {subset['final_normalized_regret'].mean():.4f} | "
            f"{subset['total_improvement'].mean():.4f} | "
            f"{100.0 * subset['filter_activation_rate'].mean():.1f}% | "
            f"{subset['mean_compatibility_trust'].mean():.3f} |"
        )

    lines.extend(["", "### Final-regret paired comparisons", ""])
    for method in METHOD_ORDER:
        if method == "Target-Only" or method not in set(summary["method"]):
            continue
        differences = _paired_difference(
            summary,
            method,
            "Target-Only",
            "final_normalized_regret",
        )
        mean, low, high = mean_bootstrap_ci(differences)
        p_value = _safe_wilcoxon(differences)
        lines.append(
            f"- **{method}**: final-regret reduction {_format_ci(mean, low, high)}; "
            f"Wilcoxon p={p_value:.4g}."
        )
    lines.append("")
    return lines


def _drift_report(frame: pd.DataFrame) -> List[str]:
    lines = [
        "## 3. Source-region drift boundary",
        "",
        "Positive regret reduction means the filter improves over Target-Only.",
        "",
        "| Method | Drift | Mean reduction [95% bootstrap CI] | Mean trust | Activation |",
        "|---|---:|---:|---:|---:|",
    ]
    classifications: Dict[str, List[Tuple[float, str]]] = {}
    for method in sorted(frame["method"].unique()):
        classifications[method] = []
        for delta in sorted(frame["delta"].unique()):
            subset = frame[
                (frame["method"] == method) & (frame["delta"] == delta)
            ]
            mean, low, high = mean_bootstrap_ci(subset["regret_reduction"])
            if low > 0.0:
                label = "positive"
            elif high < 0.0:
                label = "negative"
            else:
                label = "uncertain"
            classifications[method].append((float(delta), label))
            lines.append(
                f"| {method} | {delta:g} | {_format_ci(mean, low, high)} | "
                f"{subset['compatibility_trust'].mean():.3f} | "
                f"{100.0 * subset['filter_active'].mean():.1f}% |"
            )

    lines.extend(["", "### Data-derived boundary summary", ""])
    for method, labels in classifications.items():
        positive = [delta for delta, label in labels if label == "positive"]
        negative = [delta for delta, label in labels if label == "negative"]
        uncertain = [delta for delta, label in labels if label == "uncertain"]
        lines.append(
            f"- **{method}**: positive={positive or 'none'}; "
            f"negative={negative or 'none'}; uncertain={uncertain or 'none'}."
        )
    lines.append("")
    return lines


def analyze(config_path: str, output_dir: str | None = None) -> Path:
    config = load_json(config_path)
    root = Path(output_dir or config["output_dir"])
    analysis_dir = root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    mechanism_path = root / "mechanism" / "screening_mechanism_summary.csv"
    sequential_summary_path = root / "sequential" / "screening_sequential_summary.csv"
    sequential_trace_path = root / "sequential" / "screening_sequential_traces.csv"
    drift_path = root / "drift" / "screening_drift_summary.csv"

    missing = [
        str(path)
        for path in [
            mechanism_path,
            sequential_summary_path,
            sequential_trace_path,
            drift_path,
        ]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Run all studies before analysis. Missing:\n" + "\n".join(missing)
        )

    mechanism = pd.read_csv(mechanism_path)
    sequential_summary = pd.read_csv(sequential_summary_path)
    sequential_traces = pd.read_csv(sequential_trace_path)
    drift = pd.read_csv(drift_path)

    plot_mechanism(mechanism, analysis_dir)
    plot_sequential(sequential_traces, analysis_dir)
    plot_drift(drift, analysis_dir)

    lines = [
        "# Target-proposal / source-region-screening research report",
        "",
        "This report is generated from experiment CSV files; conclusions are not hardcoded.",
        "",
    ]
    lines.extend(_mechanism_report(mechanism))
    lines.extend(_sequential_report(sequential_summary))
    lines.extend(_drift_report(drift))
    lines.extend(
        [
            "## 4. Interpretation rule",
            "",
            "The mechanism is supported only when matching-region filtering improves over both Target-Only and structure-matched random/wrong controls under paired tests, and when the sequential result preserves the same direction. Adaptive filtering is considered safer only if its negative-drift loss and wrong-source loss are smaller than fixed filtering.",
            "",
        ]
    )

    report_path = analysis_dir / "SCREENING_STUDY_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved analysis to {analysis_dir}")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "region_screening_full.json"),
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    analyze(args.config, args.output_dir)


if __name__ == "__main__":
    main()
