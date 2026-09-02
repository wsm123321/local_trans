"""Approved unified local-guidance analysis.

Only the frozen artifact names and method identities from
``PROTOCOL_UNIFIED_LOCAL_GUIDANCE.md`` are accepted here.  The unit of analysis is
one independent ``(problem, dim, seed)`` instance; candidates and sequential steps
are never replicates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTANCE_KEYS = ("problem", "dim", "seed")
MAIN_METHODS = (
    "Target-Only",
    "Geometry-Only",
    "Local-Rank-No-Reliability",
    "Local-Rank+Reliability",
)
SAFETY_METHOD = "Reversed-Local-Rank"
MECHANISM_METHODS = MAIN_METHODS + (SAFETY_METHOD,)

# The sign is defined by higher_is_better for A-B.  For lower-is-better regret,
# this function reports B-A, so positive always means the named new method A wins.
PRIMARY_CONTRASTS: Tuple[Mapping[str, Any], ...] = (
    {"hypothesis": "H1_mechanism_normalized_regret_LocalReliability_vs_Geometry",
     "dataset": "mechanism", "method_a": "Local-Rank+Reliability",
     "method_b": "Geometry-Only", "metric": "normalized_regret",
     "higher_is_better": False},
    {"hypothesis": "H2_mechanism_top10_hit_LocalReliability_vs_Geometry",
     "dataset": "mechanism", "method_a": "Local-Rank+Reliability",
     "method_b": "Geometry-Only", "metric": "top10_hit",
     "higher_is_better": True},
    {"hypothesis": "H3_sequential_final_normalized_regret_LocalReliability_vs_Geometry",
     "dataset": "sequential", "method_a": "Local-Rank+Reliability",
     "method_b": "Geometry-Only", "metric": "final_normalized_regret",
     "higher_is_better": False},
    {"hypothesis": "H4_sequential_regret_auc_LocalReliability_vs_Geometry",
     "dataset": "sequential", "method_a": "Local-Rank+Reliability",
     "method_b": "Geometry-Only", "metric": "auc_normalized_regret",
     "higher_is_better": False},
    {"hypothesis": "H5_mechanism_reliability_increment_LocalReliability_vs_LocalNoReliability",
     "dataset": "mechanism", "method_a": "Local-Rank+Reliability",
     "method_b": "Local-Rank-No-Reliability", "metric": "normalized_regret",
     "higher_is_better": False},
)


def load_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def paired_bootstrap_ci(values: Iterable[float], n_bootstrap: int = 5000,
                        confidence: float = 0.95, seed: int = 42) -> Tuple[float, float, float]:
    """Percentile CI from resampling paired instance differences."""
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    if int(n_bootstrap) < 1 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap configuration")
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(int(n_bootstrap), len(x)), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(x.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


# Compatibility name used by the other stage analyzers; it is the same estimator.
def mean_bootstrap_ci(values: Iterable[float], n_bootstrap: int = 5000,
                      confidence: float = 0.95, seed: int = 42) -> Tuple[float, float, float]:
    return paired_bootstrap_ci(values, n_bootstrap, confidence, seed)


def wilcoxon_pratt_one_sided(values: Sequence[float]) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0 or np.all(np.abs(x) <= 1e-15):
        return 1.0
    try:
        return float(wilcoxon(x, zero_method="pratt", alternative="greater").pvalue)
    except ValueError:
        return 1.0


def rank_biserial(values: Sequence[float], tie_epsilon: float = 1e-15) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    x = x[np.abs(x) > tie_epsilon]
    return float((np.sum(x > 0) - np.sum(x < 0)) / len(x)) if len(x) else 0.0


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    p = np.asarray(p_values, dtype=float)
    if len(p) == 0:
        return []
    if not np.all(np.isfinite(p)):
        raise ValueError("Holm correction requires finite p-values")
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(len(p), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, float((len(p) - rank) * p[index])))
        adjusted[index] = running
    return adjusted.tolist()


def paired_statistics(differences: Sequence[float], n_bootstrap: int = 5000,
                       seed: int = 42) -> Dict[str, Any]:
    x = np.asarray(differences, dtype=float)
    x = x[np.isfinite(x)]
    mean, low, high = paired_bootstrap_ci(x, n_bootstrap=n_bootstrap, seed=seed)
    wins = int(np.sum(x > 1e-15)); ties = int(np.sum(np.abs(x) <= 1e-15)); losses = int(np.sum(x < -1e-15))
    n = len(x)
    return {
        "n_pairs": int(n), "mean_effect": mean, "ci_low": low, "ci_high": high,
        "wilcoxon_pratt_one_sided_p": wilcoxon_pratt_one_sided(x),
        # Keep an explicit generic alias for downstream tables while retaining
        # the Pratt-specific column used by the approved report/audit.
        "wilcoxon_one_sided_p": wilcoxon_pratt_one_sided(x),
        "rank_biserial": rank_biserial(x), "wins": wins, "ties": ties, "losses": losses,
        "win_rate": wins / n if n else float("nan"),
        "tie_rate": ties / n if n else float("nan"),
        "loss_rate": losses / n if n else float("nan"),
    }


def strict_paired_differences(frame: pd.DataFrame, method_a: str, method_b: str,
                              metric: str, higher_is_better: bool,
                              keys: Sequence[str] = INSTANCE_KEYS) -> np.ndarray:
    """Pair exactly once per instance; duplicate rows are an error."""
    keys = tuple(keys)
    required = set(keys) | {"method", metric}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"missing columns: {missing}")
    left = frame[frame["method"].astype(str) == method_a]
    right = frame[frame["method"].astype(str) == method_b]
    if left.duplicated(list(keys)).any() or right.duplicated(list(keys)).any():
        raise ValueError("duplicate instance/method rows; candidates or steps are not replicates")
    left = left[list(keys) + [metric]].rename(columns={metric: "a"})
    right = right[list(keys) + [metric]].rename(columns={metric: "b"})
    paired = left.merge(right, on=list(keys), how="inner", validate="one_to_one")
    a = pd.to_numeric(paired["a"], errors="coerce").to_numpy(float)
    b = pd.to_numeric(paired["b"], errors="coerce").to_numpy(float)
    keep = np.isfinite(a) & np.isfinite(b)
    raw = a[keep] - b[keep]
    return raw if higher_is_better else -raw


def trace_auc(frame: pd.DataFrame, metric: str = "normalized_regret",
              include_initial: bool = True) -> pd.DataFrame:
    """Reduce each trace to one final/AUC row.

    ``include_initial=False`` matches the runner's approved sequential summary:
    step 0 is an optional initial-state display row, while paid evaluations are
    steps 1..budget and only those paid steps define the regret AUC.
    """
    required = set(INSTANCE_KEYS) | {"method", "step", metric}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"missing trace columns: {missing}")
    rows: List[Dict[str, Any]] = []
    for key, group in frame.groupby(list(INSTANCE_KEYS) + ["method"], sort=False):
        group = group.copy().sort_values("step")
        if group["step"].duplicated().any():
            raise ValueError("duplicate step within an instance/method")
        if not include_initial:
            paid = pd.to_numeric(group["step"], errors="coerce") > 0
            group = group.loc[paid]
        if group.empty:
            raise ValueError("no paid sequential steps for an instance/method")
        steps = pd.to_numeric(group["step"], errors="coerce").to_numpy(float)
        values = pd.to_numeric(group[metric], errors="coerce").to_numpy(float)
        if not (np.isfinite(steps).all() and np.isfinite(values).all()):
            raise ValueError("non-finite trace value")
        row = dict(zip(list(INSTANCE_KEYS) + ["method"], key if isinstance(key, tuple) else (key,)))
        row.update({"final_normalized_regret": float(values[-1]),
                    "auc_normalized_regret": float(np.trapezoid(values, steps) if hasattr(np, "trapezoid") else np.trapz(values, steps)),
                    "trace_points": int(len(values)), "first_step": int(steps[0]), "last_step": int(steps[-1])})
        rows.append(row)
    return pd.DataFrame(rows)


def _top10_hit(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "top10_hit" in out.columns:
        out["top10_hit"] = pd.to_numeric(out["top10_hit"], errors="coerce")
    elif {"true_rank", "candidate_count"}.issubset(out.columns):
        rank = pd.to_numeric(out["true_rank"], errors="coerce")
        count = pd.to_numeric(out["candidate_count"], errors="coerce")
        out["top10_hit"] = (rank < np.ceil(0.10 * count)).astype(float)
    elif "true_rank" in out.columns:
        out["top10_hit"] = (pd.to_numeric(out["true_rank"], errors="coerce") < 10).astype(float)
    else:
        raise KeyError("mechanism artifact needs top10_hit or true_rank/candidate_count")
    return out


def _read_exact_artifacts(root: Path) -> Dict[str, pd.DataFrame]:
    paths = {name: root / name for name in ("mechanism_results.csv", "sequential_summary.csv", "sequential_traces.csv")}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("missing approved artifact filename(s):\n" + "\n".join(missing))
    return {name: pd.read_csv(path) for name, path in paths.items()}


def _read_manifest(root: Path, manifest_path: Path | None = None) -> Dict[str, Any]:
    """Read the approved run manifest; legacy aliases are intentionally rejected."""
    path = manifest_path or (root / "run_manifest.json")
    if not path.exists():
        raise FileNotFoundError(f"missing approved manifest: {path}")
    return load_json(path)


def _config_analysis(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get("analysis", {})
    return value if isinstance(value, Mapping) else {}


def _write_plot_mechanism(frame: pd.DataFrame, output: Path) -> None:
    order = [m for m in MECHANISM_METHODS if m in set(frame["method"])]
    data = [pd.to_numeric(frame.loc[frame.method == m, "normalized_regret"], errors="coerce").dropna() for m in order]
    fig, ax = plt.subplots(figsize=(11, 5), dpi=180)
    # Matplotlib 3.9 renamed ``labels`` to ``tick_labels``; use the new name
    # while keeping the plot independent of candidate-level statistics.
    ax.boxplot(data, tick_labels=order, showfliers=False); ax.set_ylabel("Normalized regret (lower is better)")
    ax.set_title("Unified local-guidance mechanism regret"); ax.tick_params(axis="x", rotation=25); ax.grid(axis="y", linestyle=":", alpha=.6)
    fig.tight_layout(); fig.savefig(output / "mechanism_regret.png", dpi=300); plt.close(fig)


def _write_plot_sequential(summary: pd.DataFrame, output: Path, column: str, name: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5), dpi=180)
    for method in MAIN_METHODS:
        values = pd.to_numeric(summary.loc[summary.method == method, column], errors="coerce").dropna()
        if len(values): ax.bar(method, float(values.mean()))
    ax.set_ylabel(column); ax.set_title(title); ax.tick_params(axis="x", rotation=25); ax.grid(axis="y", linestyle=":", alpha=.6)
    fig.tight_layout(); fig.savefig(output / name, dpi=300); plt.close(fig)


def _write_report(path: Path, primary: pd.DataFrame, summary: pd.DataFrame, secondary: pd.DataFrame, n_bootstrap: int) -> None:
    lines = ["# 统一 local-guidance 统计分析报告", "", "统计单位为独立 `(problem, dim, seed)` instance。候选点不是 replicate；sequential step 只在 instance 内汇总为 final 和 AUC。正 effect 统一表示方法 A（新方法）更好。", "", "## 五项批准的 primary contrasts", "", "| Hypothesis | Dataset | n | Effect [95% CI] | Pratt one-sided p | Holm p | Rank-biserial | W/T/L |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for _, row in primary.iterrows():
        lines.append(f"| {row['hypothesis']} | {row['dataset']} | {int(row['n_pairs'])} | {row['mean_effect']:+.5f} [{row['ci_low']:+.5f}, {row['ci_high']:+.5f}] | {row['wilcoxon_pratt_one_sided_p']:.4g} | {row['holm_adjusted_p']:.4g} | {row['rank_biserial']:+.3f} | {int(row['wins'])}/{int(row['ties'])}/{int(row['losses'])} |")
    lines += ["", "## 方法汇总", "", "| Dataset | Method | Instances | Means |", "|---|---|---:|---:|"]
    for _, row in summary.iterrows():
        means = "; ".join(f"{k[5:]}={v:.6g}" for k, v in row.items() if str(k).startswith("mean_") and pd.notna(v))
        lines.append(f"| {row['dataset']} | {row['method']} | {int(row['n_instances'])} | {means} |")
    lines += ["", "## 次要对比", "", "次要对比仅用于描述，不替代五项 primary contrasts。", ""]
    if secondary.empty:
        lines.append("无可用的完整次要配对。")
    else:
        lines += ["| Dataset | A | B | Metric | n | Effect [95% CI] | p |", "|---|---|---|---|---:|---:|---:|"]
        for _, row in secondary.iterrows():
            lines.append(f"| {row['dataset']} | {row['method_a']} | {row['method_b']} | {row['metric']} | {int(row['n_pairs'])} | {row['mean_effect']:+.5f} [{row['ci_low']:+.5f}, {row['ci_high']:+.5f}] | {row['wilcoxon_pratt_one_sided_p']:.4g} |")
    lines += ["", f"Bootstrap replicates: {n_bootstrap} (from config; full protocol uses 5000).", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_analysis(input_dir: Path, config_path: Path | None = None,
                 output_dir: Path | None = None, manifest_path: Path | None = None) -> pd.DataFrame:
    root = Path(input_dir).resolve()
    artifacts = _read_exact_artifacts(root)
    config_path = config_path or (root / "config.json")
    config = load_json(config_path) if config_path.exists() else {}
    manifest = _read_manifest(root, manifest_path)
    analysis = _config_analysis(config)
    n_bootstrap = int(analysis.get("bootstrap_samples", 5000))
    seed = int(analysis.get("bootstrap_seed", 42))
    out = Path(output_dir).resolve() if output_dir else root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    mechanism = _top10_hit(artifacts["mechanism_results.csv"])
    sequential = artifacts["sequential_summary.csv"]
    traces = artifacts["sequential_traces.csv"]
    for label, frame, allowed in (("mechanism_results.csv", mechanism, set(MECHANISM_METHODS)), ("sequential_summary.csv", sequential, set(MAIN_METHODS)), ("sequential_traces.csv", traces, set(MAIN_METHODS))):
        if "method" not in frame.columns:
            raise KeyError(f"{label} missing method column")
        observed = set(frame["method"].dropna().astype(str))
        if observed != allowed:
            raise ValueError(f"{label} method set mismatch: expected={sorted(allowed)}, observed={sorted(observed)}")
    # Validate/reduce traces even when summary already contains final/AUC.
    trace_summary = trace_auc(traces, include_initial=False)
    if not sequential.empty:
        required_seq = set(INSTANCE_KEYS) | {"method", "final_normalized_regret", "auc_normalized_regret"}
        missing = sorted(required_seq - set(sequential.columns))
        if missing: raise KeyError(f"sequential_summary.csv missing columns: {missing}")
    frames = {"mechanism": mechanism, "sequential": sequential}
    rows: List[Dict[str, Any]] = []
    for index, spec in enumerate(PRIMARY_CONTRASTS):
        dataset = str(spec["dataset"]); frame = frames[dataset]
        result = dict(spec)
        result.update(paired_statistics(strict_paired_differences(frame, str(spec["method_a"]), str(spec["method_b"]), str(spec["metric"]), bool(spec["higher_is_better"])), n_bootstrap=n_bootstrap, seed=seed + index))
        rows.append(result)
    primary = pd.DataFrame(rows)
    primary["holm_adjusted_p"] = holm_adjust(primary["wilcoxon_pratt_one_sided_p"].to_numpy())
    alpha = float(analysis.get("familywise_alpha", 0.05))
    primary["supported"] = (primary["ci_low"] > 0.0) & (primary["holm_adjusted_p"] < alpha)
    primary.to_csv(out / "PRIMARY_TESTS.csv", index=False)

    summary_rows: List[Dict[str, Any]] = []
    for dataset, frame in (("mechanism", mechanism), ("sequential", sequential)):
        numeric = [c for c in ("normalized_regret", "top10_hit", "final_normalized_regret", "auc_normalized_regret", "total_improvement") if c in frame.columns]
        for method, group in frame.groupby("method", sort=False):
            row = {"dataset": dataset, "method": method, "n_instances": int(group[list(INSTANCE_KEYS)].drop_duplicates().shape[0])}
            row.update({f"mean_{c}": float(pd.to_numeric(group[c], errors="coerce").mean()) for c in numeric})
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows); summary.to_csv(out / "METHOD_SUMMARY.csv", index=False)

    secondary_rows: List[Dict[str, Any]] = []
    for dataset, frame, metric, higher in (("mechanism", mechanism, "normalized_regret", False), ("mechanism", mechanism, "top10_hit", True), ("sequential", sequential, "final_normalized_regret", False), ("sequential", sequential, "auc_normalized_regret", False)):
        if metric not in frame.columns: continue
        for method in frame["method"].astype(str).unique():
            if method in {"Target-Only", "Geometry-Only"}: continue
            try: difference = strict_paired_differences(frame, method, "Geometry-Only", metric, higher)
            except (KeyError, ValueError): continue
            row = {"dataset": dataset, "method_a": method, "method_b": "Geometry-Only", "metric": metric}
            row.update(paired_statistics(difference, n_bootstrap=n_bootstrap, seed=seed + 100 + len(secondary_rows)))
            secondary_rows.append(row)
    secondary = pd.DataFrame(secondary_rows); secondary.to_csv(out / "SECONDARY_CONTRASTS.csv", index=False)
    _write_plot_mechanism(mechanism, out)
    _write_plot_sequential(sequential, out, "final_normalized_regret", "sequential_final.png", "Sequential final normalized regret")
    _write_plot_sequential(sequential, out, "auc_normalized_regret", "sequential_auc.png", "Sequential normalized-regret AUC")
    _write_report(out / "UNIFIED_LOCAL_GUIDANCE_REPORT_CN.md", primary, summary, secondary, n_bootstrap)
    return primary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "results" / "unified_local_guidance")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output", "--output-dir", dest="output", type=Path, default=None)
    args = parser.parse_args()
    run_analysis(args.input, args.config, args.output, args.manifest)


if __name__ == "__main__":
    main()
