"""Analyze ARISE-BO identification accuracy and optimization performance."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent


def bootstrap_mean_ci(values, seed=42, n_boot=2000) -> Tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sample = rng.choice(array, size=(n_boot, len(array)), replace=True)
    means = np.mean(sample, axis=1)
    return float(np.mean(array)), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def safe_wilcoxon(values) -> float:
    array = np.asarray(values, dtype=float)
    if len(array) == 0 or np.allclose(array, 0.0):
        return 1.0
    return float(wilcoxon(array).pvalue)


def identification_metrics(group: pd.DataFrame) -> Dict[str, float]:
    y = group["true_useful"].astype(int).to_numpy()
    policy = str(group["policy"].iloc[0]) if len(group) else ""
    if policy == "global_adaptive":
        # The old rule emits one task-level scalar, repeated for every region.
        # This is intentional: its inability to rank heterogeneous local regions
        # should be visible in AUROC/AUPRC and gain correlation.
        score = group["global_compatibility_trust"].to_numpy(dtype=float)
        effect = score
    else:
        score = group["probability_positive"].to_numpy(dtype=float)
        effect = group["posterior_mean"].to_numpy(dtype=float)
    result = {
        "rows": float(len(group)),
        "positive_rate": float(np.mean(y)) if len(y) else float("nan"),
        "brier": float(brier_score_loss(y, score)) if len(np.unique(y)) > 1 else float("nan"),
        "auroc": float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else float("nan"),
        "auprc": float(average_precision_score(y, score)) if len(np.unique(y)) > 1 else float("nan"),
    }
    if (
        np.ptp(effect) > 1e-14
        and group["true_gain"].nunique() > 1
    ):
        rho = spearmanr(effect, group["true_gain"]).statistic
        result["gain_spearman"] = float(rho) if np.isfinite(rho) else float("nan")
    else:
        result["gain_spearman"] = float("nan")

    trusted = group[group["status"] == "trusted"]
    rejected = group[group["status"] == "rejected"]
    result["trusted_rate"] = float(len(trusted) / max(len(group), 1))
    result["trusted_precision"] = float(trusted["true_useful"].mean()) if len(trusted) else float("nan")
    result["rejected_rate"] = float(len(rejected) / max(len(group), 1))
    result["rejected_precision"] = float((1.0 - rejected["true_useful"]).mean()) if len(rejected) else float("nan")
    return result


def paired_policy_difference(summary: pd.DataFrame, scenario: str, policy: str) -> Dict[str, float]:
    keys = ["problem", "dim", "seed", "scenario"]
    target = summary[(summary["scenario"] == scenario) & (summary["policy"] == "target_only")]
    method = summary[(summary["scenario"] == scenario) & (summary["policy"] == policy)]
    merged = target.merge(method, on=keys, suffixes=("_target", "_method"))
    difference = (
        merged["final_normalized_regret_target"]
        - merged["final_normalized_regret_method"]
    ).to_numpy(dtype=float)
    mean, low, high = bootstrap_mean_ci(difference)
    return {
        "n": len(difference),
        "mean_reduction": mean,
        "ci_low": low,
        "ci_high": high,
        "wilcoxon_p": safe_wilcoxon(difference),
        "win_rate": float(np.mean(difference > 1e-12)) if len(difference) else float("nan"),
        "loss_rate": float(np.mean(difference < -1e-12)) if len(difference) else float("nan"),
    }


def plot_convergence(traces: pd.DataFrame, output: Path) -> None:
    policies = ["target_only", "fixed", "global_adaptive", "posterior", "arise"]
    scenarios = list(traces["scenario"].drop_duplicates())
    fig, axes = plt.subplots(1, len(scenarios), figsize=(6 * len(scenarios), 4.5), squeeze=False)
    for ax, scenario in zip(axes[0], scenarios):
        subset = traces[traces["scenario"] == scenario]
        for policy in policies:
            values = subset[subset["policy"] == policy]
            if values.empty:
                continue
            grouped = values.groupby("step")["normalized_regret"]
            mean = grouped.mean()
            se = grouped.std() / np.sqrt(grouped.count().clip(lower=1))
            ax.plot(mean.index, mean.values, label=policy)
            ax.fill_between(mean.index, mean.values - 1.96 * se.values, mean.values + 1.96 * se.values, alpha=0.15)
        ax.set_title(f"Scenario: {scenario}")
        ax.set_xlabel("Target evaluations")
        ax.set_ylabel("Normalized simple regret")
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "arise_convergence.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_identification(identification: pd.DataFrame, output: Path) -> None:
    policies = ["target_only", "global_adaptive", "posterior", "arise"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for policy in policies:
        subset = identification[identification["policy"] == policy]
        if subset.empty or subset["true_useful"].nunique() < 2:
            continue
        fraction, predicted = calibration_curve(
            subset["true_useful"].astype(int),
            subset["probability_positive"],
            n_bins=8,
            strategy="quantile",
        )
        axes[0].plot(predicted, fraction, marker="o", label=policy)
    axes[0].plot([0, 1], [0, 1], linestyle="--")
    axes[0].set_xlabel("Predicted probability of positive transfer")
    axes[0].set_ylabel("Observed useful-region frequency")
    axes[0].set_title("Region-transfer posterior calibration")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend()

    arise = identification[identification["policy"] == "arise"]
    if not arise.empty:
        sample = arise.sample(min(5000, len(arise)), random_state=42)
        axes[1].scatter(sample["posterior_mean"], sample["true_gain"], s=10, alpha=0.35)
    axes[1].axhline(0.0, linestyle="--")
    axes[1].axvline(0.0, linestyle="--")
    axes[1].set_xlabel("Estimated excess-improvement region effect")
    axes[1].set_ylabel("True counterfactual region gain")
    axes[1].set_title("Estimated compatibility vs decision utility")

    fig.tight_layout()
    fig.savefig(output / "arise_identification.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def generate_report(summary: pd.DataFrame, identification: pd.DataFrame, output: Path) -> None:
    lines: List[str] = []
    lines.append("# ARISE-BO: decision-conditional local transferability report\n")
    lines.append("The report is generated from CSV files. Positive regret reduction means improvement over target-only BO.\n")

    lines.append("## 1. Region-identification quality\n")
    rows = []
    for (scenario, policy), group in identification.groupby(["scenario", "policy"]):
        metrics = identification_metrics(group)
        rows.append({"scenario": scenario, "policy": policy, **metrics})
    id_table = pd.DataFrame(rows)
    if not id_table.empty:
        lines.append("| Scenario | Policy | AUROC | AUPRC | Brier | Gain Spearman | Trusted precision | Rejected precision |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for _, row in id_table.iterrows():
            lines.append(
                f"| {row['scenario']} | {row['policy']} | {row['auroc']:.3f} | {row['auprc']:.3f} | "
                f"{row['brier']:.3f} | {row['gain_spearman']:.3f} | {row['trusted_precision']:.3f} | "
                f"{row['rejected_precision']:.3f} |"
            )

    lines.append("\n## 2. Equal-budget optimization\n")
    aggregate = summary.groupby(["scenario", "policy"]).agg(
        final_regret=("final_normalized_regret", "mean"),
        regret_auc=("normalized_regret_auc", "mean"),
        improvement=("total_improvement", "mean"),
        global_steps=("global_steps", "mean"),
        probe_steps=("probe_steps", "mean"),
        exploit_steps=("exploit_steps", "mean"),
    ).reset_index()
    lines.append("| Scenario | Policy | Final normalized regret | Regret AUC | Mean improvement | Global steps | Probe steps | Exploit steps |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for _, row in aggregate.iterrows():
        lines.append(
            f"| {row['scenario']} | {row['policy']} | {row['final_regret']:.4f} | "
            f"{row['regret_auc']:.4f} | {row['improvement']:.4f} | {row['global_steps']:.2f} | "
            f"{row['probe_steps']:.2f} | {row['exploit_steps']:.2f} |"
        )

    lines.append("\n## 3. Paired comparisons against target-only\n")
    for scenario in summary["scenario"].drop_duplicates():
        lines.append(f"### {scenario}\n")
        for policy in ["fixed", "global_adaptive", "posterior", "arise"]:
            if policy not in set(summary["policy"]):
                continue
            result = paired_policy_difference(summary, scenario, policy)
            lines.append(
                f"- **{policy}**: reduction {result['mean_reduction']:+.4f} "
                f"[{result['ci_low']:+.4f}, {result['ci_high']:+.4f}], "
                f"Wilcoxon p={result['wilcoxon_p']:.4g}, win={result['win_rate']:.1%}, "
                f"loss={result['loss_rate']:.1%}."
            )

    lines.append("\n## 4. Interpretation\n")
    lines.append(
        "ARISE is supported only when its region posterior is calibrated, trusted-region precision is high, "
        "and the full method improves over both target-only and fixed guidance in the mixed/wrong-source scenarios. "
        "A useful region is defined by positive counterfactual decision gain, not merely geometric or global task similarity."
    )

    (output / "ARISE_STUDY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    id_table.to_csv(output / "arise_identification_metrics.csv", index=False)
    aggregate.to_csv(output / "arise_aggregate_metrics.csv", index=False)


def analyze(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(input_dir / "arise_optimizer_summary.csv")
    traces = pd.read_csv(input_dir / "arise_optimizer_traces.csv")
    identification = pd.read_csv(input_dir / "arise_region_identification.csv")
    plot_convergence(traces, output_dir)
    plot_identification(identification, output_dir)
    generate_report(summary, identification, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "results" / "arise_stage",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "arise_stage" / "analysis",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyze(args.input, args.output)
