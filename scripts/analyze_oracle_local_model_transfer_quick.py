"""Descriptive Gate-0 analysis for oracle local-model transfer.

Decision metrics are deliberately limited to pairwise accuracy, NDCG@top and
normalized top-1 regret.  All comparisons are paired at seed×condition×shell;
bootstrap intervals resample seed means, never candidate rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_METHODS = ("target_only", "geometry", "rank", "value", "dual")
METHOD_DISPLAY = {"target_only": "Target-Only", "geometry": "Geometry-Prior+Residual", "rank": "Oracle-Rank+Residual", "value": "Oracle-Value+Residual", "dual": "Oracle-Rank+Value+Residual"}
POSITIVE_RELATIONS = ("identity", "output_affine", "scale_0.7", "scale_1.5")
NEGATIVE_RELATIONS = ("reversal", "independent_expert", "identity_label_permutation")
BOUNDARY_RELATIONS = ("rotate_45", "roughness")
DECISION_METRICS = ("pairwise_accuracy", "ndcg_at_top", "normalized_top1_regret")
FROZEN_DIMENSIONS = frozenset({2})
FROZEN_PANELS = frozenset({"test"})
FROZEN_SEEDS = frozenset({11, 23, 37, 53, 71, 89, 107, 131})
FROZEN_SHELLS = frozenset({0.35, 0.7, 1.0})
FROZEN_CONDITIONS = frozenset(POSITIVE_RELATIONS + NEGATIVE_RELATIONS + BOUNDARY_RELATIONS)
FROZEN_METHODS = frozenset(CANONICAL_METHODS)
FROZEN_RESULT_CELL_COUNT = len(FROZEN_DIMENSIONS) * len(FROZEN_PANELS) * len(FROZEN_SEEDS) * len(FROZEN_CONDITIONS) * len(FROZEN_SHELLS) * len(FROZEN_METHODS)
DECISION_INPUT_KEYS = ("dimension", "panel", "seed", "relation_or_control", "shell", "method")
METRIC_ALIASES = {
    "pairwise_accuracy": ("pairwise_accuracy", "pairwise", "rank_accuracy"),
    "ndcg_at_top": ("ndcg_at_top", "ndcg", "ndcg_top"),
    "normalized_top1_regret": ("normalized_top1_regret", "top1_regret", "regret"),
    "standardized_rmse": ("standardized_rmse", "srmse"),
    "spearman": ("spearman",), "precision_at_top": ("precision_at_top",),
    "mean_negative_log_likelihood": ("mean_negative_log_likelihood", "nll"),
    "interval_coverage_95": ("interval_coverage_95", "coverage"),
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict): raise ValueError(f"{path} must contain a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find(directory: Path, names: Iterable[str], patterns: Iterable[str] = ()) -> Path | None:
    for name in names:
        p = directory / name
        if p.exists(): return p
    for pattern in patterns:
        hits = sorted(directory.glob(pattern))
        if hits: return hits[0]
    return None


def locate(input_dir: Path) -> dict[str, Path]:
    specs = {
        "results": (["results.csv", "oracle_local_model_transfer_results.csv"], ["*results*.csv"]),
        "ledger": (["prediction_ledger.csv", "oracle_local_model_transfer_ledger.csv", "ledger.csv"], ["*ledger*.csv"]),
        "diagnostics": (["source_expert_diagnostics.csv", "oracle_local_model_transfer_diagnostics.csv", "diagnostics.csv"], ["*diagnostic*.csv"]),
        "manifest": (["run_manifest.json", "oracle_local_model_transfer_manifest.json", "manifest.json"], ["*manifest*.json"]),
        "config": (["config.json", "oracle_local_model_transfer_config.json"], ["*config*.json"]),
        "failures": (["failures.csv", "oracle_local_model_transfer_failures.csv"], ["*failure*.csv"]),
    }
    found = {}
    for key, (names, patterns) in specs.items():
        p = _find(input_dir, names, patterns)
        if p is not None: found[key] = p
    if "results" not in found or "ledger" not in found: raise FileNotFoundError("results.csv and prediction_ledger.csv are required")
    return found


def _col(frame: pd.DataFrame, names: Iterable[str], default: Any = np.nan) -> pd.Series:
    for name in names:
        if name in frame.columns: return frame[name]
    return pd.Series(default, index=frame.index)


def canonical_method(value: Any) -> str:
    low = str(value).strip().lower().replace("_", "-")
    if low in {"target-only", "targetonly", "target"}: return "target_only"
    if "rank" in low and "value" in low: return "dual"
    if "geometry" in low or "geometric" in low: return "geometry"
    if "rank" in low: return "rank"
    if "value" in low: return "value"
    return str(value).strip()


def _normalise_conditions(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    x["relation"] = _col(x, ("relation", "scenario", "transfer_relation"), "unknown").fillna("unknown").astype(str)
    x["control"] = _col(x, ("control", "condition"), "none").fillna("none").astype(str)
    if "relation_or_control" in x:
        x["relation_or_control"] = x["relation_or_control"].fillna(x["relation"]).astype(str)
    else:
        permutation = x["control"].str.lower().isin({"identity_label_permutation", "permuted", "permutation"})
        x["relation_or_control"] = x["relation"].where(~((x["relation"] == "identity") & permutation), "identity_label_permutation")
    x["condition"] = x["relation_or_control"]
    x["method"] = _col(x, ("method", "model", "approach", "estimator"), "unknown").map(canonical_method)
    x["seed"] = pd.to_numeric(_col(x, ("seed", "random_seed")), errors="coerce")
    x["shell"] = _col(x, ("shell", "target_shell", "evaluation_shell"), np.nan)
    x["shell"] = pd.to_numeric(x["shell"], errors="coerce")
    if "dimension" not in x: x["dimension"] = _col(x, ("dim",), np.nan)
    x["dimension"] = pd.to_numeric(x["dimension"], errors="coerce")
    # Keep a missing panel as missing.  It is an input-validation failure, not
    # a wildcard that may be silently omitted from decision grouping.
    if "panel" not in x: x["panel"] = np.nan
    x["panel"] = x["panel"].where(x["panel"].notna(), np.nan).astype(object)
    for canonical, aliases in METRIC_ALIASES.items():
        if canonical not in x:
            value = _col(x, aliases)
            if value.notna().any(): x[canonical] = pd.to_numeric(value, errors="coerce")
    return x


def _normalise_results(frame: pd.DataFrame) -> pd.DataFrame:
    return _normalise_conditions(frame)


def _normalise_ledger(frame: pd.DataFrame) -> pd.DataFrame:
    x = _normalise_conditions(frame)
    x["point_index"] = pd.to_numeric(_col(x, ("candidate_index", "point_index", "index", "row_id")), errors="coerce")
    x["truth"] = pd.to_numeric(_col(x, ("truth", "true_y", "target_y", "clean_target_y", "observed_target_y")), errors="coerce")
    x["prediction"] = pd.to_numeric(_col(x, ("predicted_mean", "prediction", "predicted", "prediction_mean", "pred")), errors="coerce")
    x["predicted_std"] = pd.to_numeric(_col(x, ("predicted_std", "prediction_std", "std", "uncertainty")), errors="coerce")
    return x


def _summary(results: pd.DataFrame) -> pd.DataFrame:
    keys = ["dimension", "panel", "seed", "relation", "control", "relation_or_control", "condition", "shell", "method"]
    keys = [key for key in keys if key in results]
    metrics = [key for key in METRIC_ALIASES if key in results]
    agg = {"n_rows": ("method", "size")}
    agg.update({metric: (metric, "mean") for metric in metrics})
    return results.groupby(keys, dropna=False, as_index=False).agg(**agg)


def _oriented_delta(method_value: float, baseline_value: float, metric: str) -> float:
    return float(baseline_value - method_value) if metric in {"normalized_top1_regret", "standardized_rmse"} else float(method_value - baseline_value)


def bootstrap_seed(values: pd.DataFrame, value_col: str = "delta", seed_col: str = "seed", n: int = 2000, random_state: int = 17, minimum_delta: float = 0.0) -> tuple[float, float, float, float, int]:
    """Bootstrap seed means and report the fraction exceeding ``minimum_delta``."""
    seed_values = values.groupby(seed_col, dropna=True)[value_col].mean().dropna().to_numpy(float)
    if len(seed_values) == 0: return np.nan, np.nan, np.nan, np.nan, 0
    rng = np.random.default_rng(random_state)
    draws = rng.choice(seed_values, size=(max(100, int(n)), len(seed_values)), replace=True).mean(axis=1)
    return float(seed_values.mean()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975)), float(np.mean(seed_values > minimum_delta)), int(len(seed_values))


def _validate_decision_input(results: pd.DataFrame) -> dict[str, Any]:
    """Validate every cell in the exact frozen Cartesian result panel."""
    raw_columns = set(results.columns)
    x = _normalise_results(results)
    required_keys = ("dimension", "panel", "seed", "relation_or_control", "shell", "method")
    # Dimension may use the runner's ``dim`` alias; panel has no safe alias.
    missing = [key for key in required_keys if key not in raw_columns and not (key == "dimension" and "dim" in raw_columns)]
    observed_dimensions = set(pd.to_numeric(x["dimension"], errors="coerce").dropna().astype(float)) if "dimension" in x else set()
    observed_panels = set(x["panel"].dropna().astype(str)) if "panel" in x else set()
    observed_seeds = set(pd.to_numeric(x["seed"], errors="coerce").dropna().astype(int)) if "seed" in x else set()
    observed_relations = set(x["relation_or_control"].dropna().astype(str)) if "relation_or_control" in x else set()
    observed_shells = set(pd.to_numeric(x["shell"], errors="coerce").dropna().astype(float)) if "shell" in x else set()
    observed_methods = set(x["method"].dropna().astype(str)) if "method" in x else set()
    failures = list(missing)
    if observed_dimensions != {float(v) for v in FROZEN_DIMENSIONS}:
        failures.append(f"dimensions={sorted(observed_dimensions)}")
    if observed_panels != set(FROZEN_PANELS): failures.append(f"panels={sorted(observed_panels)}")
    if observed_seeds != set(FROZEN_SEEDS): failures.append(f"seeds={sorted(observed_seeds)}")
    if observed_relations != set(FROZEN_CONDITIONS): failures.append(f"relations={sorted(observed_relations)}")
    if observed_shells != set(FROZEN_SHELLS): failures.append(f"shells={sorted(observed_shells)}")
    if observed_methods != set(FROZEN_METHODS): failures.append(f"methods={sorted(observed_methods)}")

    key_columns = list(DECISION_INPUT_KEYS)
    expected_cells = {
        (float(dimension), panel, int(seed), condition, float(shell), method)
        for dimension in FROZEN_DIMENSIONS for panel in FROZEN_PANELS
        for seed in FROZEN_SEEDS for condition in FROZEN_CONDITIONS
        for shell in FROZEN_SHELLS for method in FROZEN_METHODS
    }
    observed_cells = set()
    duplicate_cells: list[list[Any]] = []
    if not missing:
        cell_frame = x[key_columns].copy()
        for column in ("dimension", "seed", "shell"):
            cell_frame[column] = pd.to_numeric(cell_frame[column], errors="coerce")
        cell_frame["method"] = cell_frame["method"].astype(str)
        cell_frame["relation_or_control"] = cell_frame["relation_or_control"].astype(str)
        for values, count in cell_frame.value_counts(dropna=False).items():
            if count > 1:
                duplicate_cells.append(list(values) + [int(count)])
        for values in cell_frame.itertuples(index=False, name=None):
            if all(pd.notna(value) for value in values):
                dimension, panel, seed, condition, shell, method = values
                observed_cells.add((float(dimension), str(panel), int(seed), str(condition), float(shell), str(method)))
    missing_cells = sorted(expected_cells - observed_cells, key=str)
    unexpected_cells = sorted(observed_cells - expected_cells, key=str)
    if len(x) != FROZEN_RESULT_CELL_COUNT:
        failures.append(f"n_rows={len(x)} expected={FROZEN_RESULT_CELL_COUNT}")
    if missing_cells:
        failures.append(f"missing_cells={len(missing_cells)}")
    if duplicate_cells:
        failures.append(f"duplicate_cells={len(duplicate_cells)}")
    payload = {
        "passes": not failures, "missing_keys": missing, "failures": failures,
        "dimensions": sorted(observed_dimensions), "panels": sorted(observed_panels),
        "seeds": sorted(observed_seeds), "relations": sorted(observed_relations),
        "shells": sorted(observed_shells), "methods": sorted(observed_methods),
        "n_rows": int(len(x)), "expected_n_rows": FROZEN_RESULT_CELL_COUNT,
        "n_expected_cells": FROZEN_RESULT_CELL_COUNT,
        "n_observed_cells": int(len(observed_cells)),
        "missing_cells": missing_cells, "unexpected_cells": unexpected_cells,
        "duplicate_cells": duplicate_cells,
    }
    return payload


def _contrasts(results: pd.DataFrame) -> pd.DataFrame:
    # Public callers may pass the raw runner frame; normalize here so the
    # collision-free identity/permutation condition is always present.
    results = _normalise_results(results)
    key = ["dimension", "panel", "seed", "relation", "control", "relation_or_control", "condition", "shell"]
    key = [column for column in key if column in results]
    baselines = [name for name in ("geometry", "target_only") if name in set(results["method"])]
    heads = [name for name in ("geometry", "rank", "value", "dual") if name in set(results["method"])]
    metrics = [name for name in METRIC_ALIASES if name in results]
    rows = []
    for group_key, group in results.groupby(key, dropna=False, sort=True):
        if not isinstance(group_key, tuple): group_key = (group_key,)
        prefix = dict(zip(key, group_key))
        for baseline in baselines:
            base = group[group.method == baseline]
            for method in heads:
                if method == baseline: continue
                current = group[group.method == method]
                if base.empty or current.empty: continue
                for metric in metrics:
                    bv = pd.to_numeric(base[metric], errors="coerce").mean(); mv = pd.to_numeric(current[metric], errors="coerce").mean()
                    if np.isfinite(bv) and np.isfinite(mv):
                        rows.append({**prefix, "method": method, "baseline": baseline, "metric": metric, "baseline_value": float(bv), "method_value": float(mv), "delta": _oriented_delta(float(mv), float(bv), metric)})
    columns = key + ["method", "baseline", "metric", "baseline_value", "method_value", "delta"]
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty: return frame
    ci_rows = []
    for keys, group in frame.groupby(["dimension", "panel", "relation_or_control", "shell", "method", "baseline", "metric"], dropna=False):
        dimension, panel, condition, shell, method, baseline, metric = keys
        est, low, high, win_rate, n_seeds = bootstrap_seed(group, n=2000, random_state=17)
        ci_rows.append({"dimension": dimension, "panel": panel, "relation_or_control": condition, "shell": shell, "method": method, "baseline": baseline, "metric": metric, "delta_mean": est, "ci_low": low, "ci_high": high, "seed_win_rate": win_rate, "n_seed_units": n_seeds, "n_pair_units": int(len(group))})
    return frame.merge(pd.DataFrame(ci_rows), on=["dimension", "panel", "relation_or_control", "shell", "method", "baseline", "metric"], how="left")


def _decision_contrasts(contrasts: pd.DataFrame, baseline: str = "geometry") -> pd.DataFrame:
    if contrasts.empty: return contrasts.copy()
    return contrasts[(contrasts.baseline == baseline) & contrasts.metric.isin(DECISION_METRICS)].copy()


def _head_to_head_pairs(results: pd.DataFrame, challenger: str, baseline: str = "rank") -> pd.DataFrame:
    """Pair challenger and rank on the raw result panels.

    This deliberately does not derive the comparison from the Geometry contrasts:
    a head can look good against Geometry while adding nothing beyond Rank.
    """
    results = _normalise_results(results)
    key = ["dimension", "panel", "seed", "relation_or_control", "shell"]
    positive = results[results["relation_or_control"].isin(POSITIVE_RELATIONS)]
    positive = positive[positive.method.isin({baseline, challenger})]
    rows: list[dict[str, Any]] = []
    if positive.empty:
        return pd.DataFrame(columns=key + ["challenger", "baseline", "metric", "baseline_value", "challenger_value", "delta"])
    grouped = positive.groupby(key + ["method"], dropna=False, sort=True)[list(DECISION_METRICS)].mean().reset_index()
    for values, group in grouped.groupby(key, dropna=False, sort=True):
        if not isinstance(values, tuple): values = (values,)
        prefix = dict(zip(key, values))
        by_method = {str(row.method): row for row in group.itertuples(index=False)}
        if baseline not in by_method or challenger not in by_method:
            continue
        base_row, challenger_row = by_method[baseline], by_method[challenger]
        for metric in DECISION_METRICS:
            bv = float(getattr(base_row, metric)); cv = float(getattr(challenger_row, metric))
            if np.isfinite(bv) and np.isfinite(cv):
                rows.append({**prefix, "challenger": challenger, "baseline": baseline,
                             "metric": metric, "baseline_value": bv,
                             "challenger_value": cv, "delta": _oriented_delta(cv, bv, metric)})
    return pd.DataFrame(rows, columns=key + ["challenger", "baseline", "metric", "baseline_value", "challenger_value", "delta"])


def _head_increment_evaluation(results: pd.DataFrame, challenger: str, threshold: float = 0.0) -> dict[str, Any]:
    """Evaluate whether a head has a positive increment beyond Oracle-Rank."""
    pairs = _head_to_head_pairs(results, challenger, "rank")
    detail: dict[str, Any] = {
        "head": challenger, "baseline": "rank", "decision_metrics": list(DECISION_METRICS),
        "positive_relations": list(POSITIVE_RELATIONS), "threshold": threshold,
        "paired_unit": "dimension × panel × seed × relation_or_control × shell",
        "n_pair_units": int(pairs[["dimension", "panel", "seed", "relation_or_control", "shell"]].drop_duplicates().shape[0]) if len(pairs) else 0,
        "n_seed_units": int(pairs[["dimension", "panel", "seed"]].drop_duplicates().shape[0]) if len(pairs) else 0,
        "metric_detail": {}, "shell_detail": {},
    }
    for metric in DECISION_METRICS:
        m = pairs[pairs.metric == metric]
        seed_level = m.groupby(["dimension", "panel", "seed"], as_index=False, dropna=True).delta.mean() if len(m) else pd.DataFrame(columns=["dimension", "panel", "seed", "delta"])
        estimate, low, high, seed_win_rate, n_seeds = bootstrap_seed(seed_level, n=2000, random_state=101 + DECISION_METRICS.index(metric), minimum_delta=threshold)
        d = pd.to_numeric(m.delta, errors="coerce").dropna()
        wins = int((d > threshold).sum()); losses = int((d < -threshold).sum())
        ties = int(len(d) - wins - losses)
        seed_mean = float(seed_level.delta.mean()) if len(seed_level) else np.nan
        detail["metric_detail"][metric] = {
            "mean_delta": float(d.mean()) if len(d) else np.nan,
            "seed_level_mean_delta": seed_mean,
            "ci_low": low, "ci_high": high,
            "seed_win_rate": seed_win_rate, "n_seed_units": n_seeds, "n_seeds": n_seeds, "n_pair_units": int(len(d)),
            "pair_wins": wins, "pair_ties": ties, "pair_losses": losses,
            "passes": bool(np.isfinite(seed_mean) and seed_mean > threshold and seed_win_rate >= .5),
        }
    detail["metrics_passed"] = int(sum(x["passes"] for x in detail["metric_detail"].values()))
    detail["pair_units"] = pairs.to_dict(orient="records")
    for shell, shell_group in pairs.groupby("shell", dropna=False, sort=True):
        metric_detail: dict[str, Any] = {}
        for metric in DECISION_METRICS:
            m = shell_group[shell_group.metric == metric]
            seed_level = m.groupby("seed", as_index=False, dropna=True).delta.mean() if len(m) else pd.DataFrame(columns=["seed", "delta"])
            seed_mean = float(seed_level.delta.mean()) if len(seed_level) else np.nan
            seed_win_rate = float((seed_level.delta > threshold).mean()) if len(seed_level) else np.nan
            d = pd.to_numeric(m.delta, errors="coerce").dropna()
            wins = int((d > threshold).sum()); losses = int((d < -threshold).sum())
            metric_detail[metric] = {
                "mean_delta": float(d.mean()) if len(d) else np.nan,
                "seed_level_mean_delta": seed_mean, "seed_win_rate": seed_win_rate,
                "n_pair_units": int(len(d)), "n_seeds": int(len(seed_level)),
                "pair_wins": wins, "pair_ties": int(len(d) - wins - losses), "pair_losses": losses,
                "passes": bool(len(seed_level) and seed_mean > threshold and seed_win_rate >= .5),
            }
        passed = int(sum(x["passes"] for x in metric_detail.values()))
        detail["shell_detail"][str(shell)] = {"metric_detail": metric_detail, "metrics_passed": passed, "passes_shell": passed >= 2}
    detail["shells_passed"] = int(sum(x["passes_shell"] for x in detail["shell_detail"].values()))
    detail["passes"] = bool(detail["metrics_passed"] >= 2 and detail["shells_passed"] >= 2)
    return detail


def _relation_class(condition: str) -> str:
    if condition in POSITIVE_RELATIONS: return "positive"
    if condition in NEGATIVE_RELATIONS: return "negative"
    if condition in BOUNDARY_RELATIONS: return "boundary"
    return "other"


def _head_evaluation(contrasts: pd.DataFrame, head: str, threshold: float = 0.0) -> dict[str, Any]:
    x = _decision_contrasts(contrasts)
    x = x[x.method == head].copy()
    x["relation_class"] = x.relation_or_control.map(_relation_class)
    positive = x[x.relation_class == "positive"]
    negative = x[x.relation_class == "negative"]
    detail: dict[str, Any] = {
        "head": head, "decision_metrics": list(DECISION_METRICS), "threshold": threshold,
        "positive_paired_unit": "dimension × panel × seed × positive relation × shell",
        "negative_paired_unit": "dimension × panel × seed × negative condition × shell",
    }

    def seed_units(frame: pd.DataFrame) -> pd.DataFrame:
        # Shell/relation rows are correlated observations.  A seed contributes
        # exactly one unit after all such rows have been averaged.
        if frame.empty:
            return pd.DataFrame(columns=["seed", "delta"])
        return frame.groupby("seed", as_index=False, dropna=True)["delta"].mean()

    metric_detail: dict[str, Any] = {}
    for metric in DECISION_METRICS:
        m = positive[positive.metric == metric]
        seed_level = seed_units(m)
        estimate, low, high, win_rate, n_seed_units = bootstrap_seed(
            seed_level, n=2000, random_state=31 + DECISION_METRICS.index(metric), minimum_delta=threshold
        )
        pair_mean = float(pd.to_numeric(m.delta, errors="coerce").mean()) if len(m) else np.nan
        metric_detail[metric] = {
            "mean_delta": estimate, "pair_mean_delta": pair_mean,
            "seed_level_mean_delta": estimate, "ci_low": low, "ci_high": high,
            "seed_win_rate": win_rate, "n_seed_units": n_seed_units,
            "n_pair_units": int(len(m)),
            "passes": bool(np.isfinite(estimate) and estimate > threshold and win_rate >= .5),
        }
    detail["positive_metric_detail"] = metric_detail
    detail["positive_metrics_passed"] = int(sum(v["passes"] for v in metric_detail.values()))
    detail["positive_n_seed_units"] = int(max((v["n_seed_units"] for v in metric_detail.values()), default=0))
    detail["positive_n_pair_units"] = int(max((v["n_pair_units"] for v in metric_detail.values()), default=0))

    shell_detail: dict[str, Any] = {}
    for shell, group in positive.groupby("shell", dropna=False, sort=True):
        metric_passes: dict[str, bool] = {}
        metric_seed_detail: dict[str, Any] = {}
        for metric in DECISION_METRICS:
            m = group[group.metric == metric]
            seed_level = seed_units(m)
            estimate, low, high, seed_win_rate, n_seed_units = bootstrap_seed(
                seed_level, n=2000, random_state=47 + DECISION_METRICS.index(metric), minimum_delta=threshold
            )
            metric_passes[metric] = bool(np.isfinite(estimate) and estimate > threshold and seed_win_rate >= .5)
            metric_seed_detail[metric] = {
                "mean_delta": estimate, "ci_low": low, "ci_high": high,
                "seed_win_rate": seed_win_rate, "n_seed_units": n_seed_units,
                "n_pair_units": int(len(m)), "passes": metric_passes[metric],
            }
        passed = int(sum(metric_passes.values()))
        shell_detail[str(shell)] = {
            "metric_passes": metric_passes, "metric_seed_detail": metric_seed_detail,
            "metrics_passed": passed, "passes_shell": passed >= 2,
        }
    detail["positive_shell_detail"] = shell_detail
    detail["positive_shells_passed"] = int(sum(v["passes_shell"] for v in shell_detail.values()))

    identity = x[x.relation_or_control == "identity"]
    permuted = x[x.relation_or_control == "identity_label_permutation"]
    identity_detail: dict[str, Any] = {}
    identity_key = ["dimension", "panel", "seed", "shell"]
    for metric in DECISION_METRICS:
        a = identity[identity.metric == metric][identity_key + ["delta"]].rename(columns={"delta": "identity_delta"})
        b = permuted[permuted.metric == metric][identity_key + ["delta"]].rename(columns={"delta": "permuted_delta"})
        paired = a.merge(b, on=identity_key, how="inner", validate="one_to_one")
        if len(paired):
            paired["delta"] = paired["identity_delta"] - paired["permuted_delta"]
            seed_level = paired.groupby("seed", as_index=False, dropna=True)["delta"].mean()
        else:
            paired = pd.DataFrame(columns=identity_key + ["identity_delta", "permuted_delta"])
            seed_level = pd.DataFrame(columns=["seed", "delta"])
        mean_delta, low, high, seed_win_rate, n_seed_units = bootstrap_seed(
            seed_level, n=2000, random_state=61 + DECISION_METRICS.index(metric), minimum_delta=threshold
        )
        identity_detail[metric] = {
            "mean_identity_minus_permuted": mean_delta, "pair_mean_identity_minus_permuted": float(paired["identity_delta"].sub(paired["permuted_delta"]).mean()) if len(paired) else np.nan,
            "ci_low": low, "ci_high": high, "seed_win_rate": seed_win_rate,
            "n_seed_units": n_seed_units, "n_pair_units": int(len(paired)),
            "n_pairs": int(len(paired)),
            "passes": bool(np.isfinite(mean_delta) and mean_delta > threshold and seed_win_rate >= .5),
        }
    detail["identity_observed_vs_permuted"] = identity_detail
    detail["identity_metrics_passed"] = int(sum(v["passes"] for v in identity_detail.values()))

    negative_condition_detail: dict[str, Any] = {}
    for condition in NEGATIVE_RELATIONS:
        condition_metrics: dict[str, Any] = {}
        for metric in DECISION_METRICS:
            m = negative[(negative.relation_or_control == condition) & (negative.metric == metric)]
            seed_level = seed_units(m)
            mean_delta, low, high, seed_win_rate, n_seed_units = bootstrap_seed(
                seed_level, n=2000, random_state=79 + DECISION_METRICS.index(metric), minimum_delta=threshold
            )
            condition_metrics[metric] = {
                "mean_delta": mean_delta,
                "pair_mean_delta": float(pd.to_numeric(m.delta, errors="coerce").mean()) if len(m) else np.nan,
                "ci_low": low, "ci_high": high, "seed_win_rate": seed_win_rate,
                "n_seed_units": n_seed_units, "n_pair_units": int(len(m)),
                "replicates": bool(np.isfinite(mean_delta) and mean_delta > threshold and seed_win_rate >= .5),
            }
        metrics_replicating = int(sum(v["replicates"] for v in condition_metrics.values()))
        negative_condition_detail[condition] = {
            "metric_detail": condition_metrics, "metrics_replicating": metrics_replicating,
            "replicates": metrics_replicating >= 2,
        }
    detail["negative_condition_detail"] = negative_condition_detail
    # Alias retained for consumers of the prior public decision shape; unlike
    # the old metric-pooled result, it is now keyed by negative condition.
    detail["negative_detail"] = negative_condition_detail
    detail["negative_conditions_replicating"] = int(sum(v["replicates"] for v in negative_condition_detail.values()))
    detail["negative_metrics_replicating"] = detail["negative_conditions_replicating"]
    detail["negative_replication"] = detail["negative_conditions_replicating"] >= 2
    detail["positive_rule_pass"] = bool(detail["positive_shells_passed"] >= 2 and detail["positive_metrics_passed"] >= 2)
    detail["identity_rule_pass"] = bool(detail["identity_metrics_passed"] >= 2)
    detail["negative_rule_pass"] = not detail["negative_replication"]
    detail["passes"] = bool(detail["positive_rule_pass"] and detail["identity_rule_pass"] and detail["negative_rule_pass"])
    detail["selection_score"] = [detail["positive_metrics_passed"], detail["positive_shells_passed"], detail["identity_metrics_passed"]]
    return detail


def decide(results: pd.DataFrame, contrasts: pd.DataFrame, config: Mapping[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
    cfg = config or {}; analysis_cfg = cfg.get("analysis", {}) if isinstance(cfg.get("analysis", {}), Mapping) else {}
    threshold = float(analysis_cfg.get("minimum_delta", 0.0))
    input_validation = _validate_decision_input(results)
    # Geometry-relative evidence is still required for each head.  The
    # increment decision below is independently recomputed from raw results.
    evaluations = {head: _head_evaluation(contrasts, head, threshold) for head in ("rank", "value", "dual")}
    head_to_head = {head: _head_increment_evaluation(results, head, threshold) for head in ("value", "dual")}
    rank_passes = bool(evaluations["rank"]["passes"])
    # A challenger replaces a supported Rank head; it cannot qualify when
    # Rank itself fails its Geometry-relative Gate-0 rule.
    challengers_with_increment = ([head for head in ("value", "dual")
                                   if rank_passes and evaluations[head]["passes"] and head_to_head[head]["passes"]]
                                  if input_validation["passes"] else [])

    def challenger_key(head: str) -> tuple[Any, ...]:
        """Apply the documented lexicographic value-vs-dual rule exactly."""
        ev = head_to_head[head]
        means = tuple(
            float(ev["metric_detail"][m]["seed_level_mean_delta"])
            if np.isfinite(ev["metric_detail"][m]["seed_level_mean_delta"]) else -np.inf
            for m in DECISION_METRICS
        )
        pair_wins = tuple(int(ev["metric_detail"][m]["pair_wins"]) for m in DECISION_METRICS)
        # Last component is an explicit stable preference, not an accidental
        # property of dictionary/set iteration: value wins exact ties.
        return (int(ev["metrics_passed"]), int(ev["shells_passed"]), means, pair_wins, int(head == "value"))

    if not input_validation["passes"]:
        label, selected = "no_oracle_headroom_stop_before_alignment", None
    elif challengers_with_increment:
        selected = max(challengers_with_increment, key=challenger_key)
        label = "promising_value_or_dual_head_transfer"
    elif rank_passes:
        selected = "rank"
        label = "promising_rank_transfer"
    else:
        label, selected = "no_oracle_headroom_stop_before_alignment", None
    rules = {
        "decision_input_keys": list(DECISION_INPUT_KEYS),
        "frozen_dimensions": sorted(FROZEN_DIMENSIONS), "frozen_panels": sorted(FROZEN_PANELS),
        "frozen_seeds": sorted(FROZEN_SEEDS), "frozen_shells": sorted(FROZEN_SHELLS),
        "frozen_conditions": sorted(FROZEN_CONDITIONS), "frozen_methods": sorted(FROZEN_METHODS),
        "frozen_result_cell_count": FROZEN_RESULT_CELL_COUNT,
        "positive_relations": list(POSITIVE_RELATIONS), "negative_relations": list(NEGATIVE_RELATIONS), "boundary_relations": list(BOUNDARY_RELATIONS), "canonical_methods": list(CANONICAL_METHODS),
        "baseline_for_decision": "geometry", "secondary_baseline": "target_only", "decision_metrics": list(DECISION_METRICS),
        "minimum_delta": threshold,
        "delta_convention": "positive means challenger is better; regret delta = baseline regret - challenger regret",
        "paired_unit": "dimension × panel × seed × relation_or_control × shell × method × baseline",
        "head_to_head_paired_unit": "dimension × panel × seed × relation_or_control × shell",
        "bootstrap": "descriptive percentile CI and win rate over seed-cluster units; never candidate rows",
        "positive_rule": f"for one head, at least 2/3 positive shells pass at least 2/3 metrics; each metric requires seed-level win rate >= 0.5 against minimum_delta={threshold:g} and seed mean delta > minimum_delta; at least 2 metrics pass overall",
        "head_increment_rule": f"challenger has Rank-beyond increment iff at least 2/3 decision-metric seed means > minimum_delta={threshold:g} with seed win rate >= 0.5 AND at least 2/3 shells have at least 2/3 such metrics",
        "head_increment_source": "raw results.csv, explicitly paired value-vs-rank and dual-vs-rank on every dimension × panel × seed × positive relation × shell",
        "identity_rule": f"after shell rows are averaged within seed, identity observed−permuted must have mean > minimum_delta={threshold:g} and seed win rate >= 0.5 for at least 2/3 metrics; report descriptive seed bootstrap CI",
        "negative_rule": f"for each negative condition, average shell rows within seed; a metric replicates only when mean > minimum_delta={threshold:g} and seed win rate >= 0.5; a condition replicates at least 2/3 metrics, and the overall negative rule fails when at least 2/3 (2 of 3) conditions replicate",
        "shell_rule": "benefit cannot be confined to shell .35; at least 2 of 3 shells must pass",
        "selection_rule": "Rank must pass first. For eligible value/dual challengers, compare metrics_passed, shells_passed, then the tuple of seed-level mean deltas in DECISION_METRICS order, then pair wins in DECISION_METRICS order; on an exact tie explicitly prefer value. Never sum or mix metric values.",
        "srmse_role": "secondary only; never used for decision",
        "quick_informal": True,
    }
    return label, {"label": label, "selected_head": selected, "head_evaluations": evaluations,
                   "head_to_head_evaluations": head_to_head, "input_validation": input_validation,
                   "rules": rules, "quick_informal": True}


def _plot_headroom(contrasts: pd.DataFrame, path: Path) -> None:
    x = _decision_contrasts(contrasts)
    relation_order = list(POSITIVE_RELATIONS + BOUNDARY_RELATIONS + NEGATIVE_RELATIONS)
    shell_order = sorted(FROZEN_SHELLS)
    row_index = pd.MultiIndex.from_product(
        [relation_order, shell_order], names=["relation_or_control", "shell"]
    )
    method_order = ["rank", "value", "dual"]
    matrices: list[pd.DataFrame] = []
    for metric in DECISION_METRICS:
        q = x[x.metric == metric].groupby(
            ["relation_or_control", "shell", "method"], as_index=False
        ).delta.mean()
        matrices.append(
            q.pivot_table(
                index=["relation_or_control", "shell"], columns="method",
                values="delta", aggfunc="mean",
            ).reindex(index=row_index, columns=method_order)
        )
    magnitude = max(
        (float(np.nanmax(np.abs(matrix.to_numpy()))) for matrix in matrices),
        default=1.0,
    )
    magnitude = max(magnitude, 1e-12)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 10.5), dpi=150, sharey=True)
    image = None
    row_labels = [f"{relation}  |  r={shell:g}" for relation, shell in row_index]
    for ax, metric, matrix in zip(axes, DECISION_METRICS, matrices):
        image = ax.imshow(
            matrix.to_numpy(), aspect="auto", cmap="RdBu_r",
            vmin=-magnitude, vmax=magnitude,
        )
        ax.set_title(metric.replace("_", " "))
        ax.set_xticks(range(len(method_order)), ["Rank", "Value", "Dual"])
        ax.set_yticks(range(len(row_labels)), row_labels, fontsize=7.5)
        ax.tick_params(axis="x", labelrotation=0)
        for boundary in (len(POSITIVE_RELATIONS) * len(shell_order) - 0.5,
                         (len(POSITIVE_RELATIONS) + len(BOUNDARY_RELATIONS)) * len(shell_order) - 0.5):
            ax.axhline(boundary, color="black", linewidth=0.8)
    axes[0].set_ylabel("relation and evaluation shell")
    if image is not None:
        colorbar_axis = fig.add_axes([0.935, 0.17, 0.014, 0.68])
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.set_label("oriented delta vs Geometry (positive = better)")
    fig.suptitle("Gate-0 oracle headroom by relation and shell", fontsize=14)
    fig.text(
        0.5, 0.02,
        "Rows: positive relations, boundary relations, then negative controls. "
        "Each cell is the 8-seed descriptive mean; metrics are not combined.",
        ha="center", fontsize=9,
    )
    fig.subplots_adjust(left=0.20, right=0.90, top=0.92, bottom=0.07, wspace=0.18)
    fig.savefig(path, dpi=220); plt.close(fig)


def _plot_heads(contrasts: pd.DataFrame, path: Path) -> None:
    x = _decision_contrasts(contrasts)
    x = x[x.relation_or_control.isin(POSITIVE_RELATIONS)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), dpi=150)
    method_order = ["rank", "value", "dual"]
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    titles = {
        "pairwise_accuracy": "Pairwise accuracy",
        "ndcg_at_top": "NDCG @ top 10%",
        "normalized_top1_regret": "Top-1 regret (direction-adjusted)",
    }
    for ax, metric in zip(axes, DECISION_METRICS):
        q = x[x.metric == metric].groupby("method").delta.mean().reindex(method_order)
        ax.bar(["Rank", "Value", "Dual"], q.to_numpy(), color=colors, width=0.68)
        ax.axhline(0, color="black", lw=.8)
        ax.set_title(titles[metric])
        ax.set_ylabel("oriented delta vs Geometry\npositive = better")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Gate-0 oracle heads versus Geometry (positive relations only)", fontsize=14)
    fig.text(
        0.5, 0.02,
        "Descriptive 8-seed means. Regret is direction-adjusted as Geometry regret − transfer regret; metrics are not combined.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94)); fig.savefig(path, dpi=220); plt.close(fig)


def _plot_identity(results: pd.DataFrame, path: Path) -> None:
    x = results[results.relation_or_control.isin(
        {"identity", "identity_label_permutation"}
    )].copy()
    condition_order = ["identity", "identity_label_permutation"]
    condition_titles = ["Observed identity source", "Permuted source labels"]
    method_order = ["target_only", "geometry", "rank", "value", "dual"]
    labels = ["Target-Only", "Geometry", "Rank", "Value", "Dual"]
    colors = dict(zip(method_order, ["#9D755D", "#BAB0AC", "#4C78A8", "#F58518", "#54A24B"]))
    titles = {
        "pairwise_accuracy": "Pairwise accuracy",
        "ndcg_at_top": "NDCG @ top 10%",
        "normalized_top1_regret": "Normalized Top-1 regret",
    }
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4), dpi=150, sharex=True)
    for row, (condition, condition_title) in enumerate(zip(condition_order, condition_titles)):
        condition_data = x[x.relation_or_control == condition]
        for col, metric in enumerate(DECISION_METRICS):
            ax = axes[row, col]
            q = condition_data.groupby(["shell", "method"])[metric].mean().unstack("method")
            for method, label in zip(method_order, labels):
                if method in q:
                    ax.plot(q.index, q[method], marker="o", linewidth=1.8,
                            color=colors[method], label=label)
            ax.set_title(f"{condition_title}\n{titles[metric]}")
            ax.set_xlabel("evaluation shell")
            ax.grid(alpha=0.2)
            if col == 0:
                ax.set_ylabel("descriptive mean")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=5, frameon=False)
    fig.suptitle("Identity source evidence versus source-label permutation", fontsize=14)
    fig.text(
        0.5, 0.045,
        "Rows explicitly separate observed source responses from the label-permutation null; lower regret is better.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.94)); fig.savefig(path, dpi=220); plt.close(fig)


def _fmt(value: Any) -> str:
    return "NA" if value is None or not np.isfinite(float(value)) else f"{float(value):+.5f}"


def _write_conclusion(path: Path, label: str, decision: Mapping[str, Any], summary: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    rules = decision["rules"]
    lines = ["# Gate-0 Oracle Local-Model Transfer Quick 结论", "", f"**结论标签：`{label}`**", "", "这是 quick、非正式、描述性分析，不是确认性统计结论，也不涉及未知 alignment、候选获取或 online BO。Bootstrap 只按 seed 重采样，candidate 不是独立重复。", "", "## 明确的 Gate-0 规则", "", f"- decision input keys：`{' × '.join(rules['decision_input_keys'])}`；frozen dimensions=`{rules['frozen_dimensions']}`，panels=`{rules['frozen_panels']}`，shells=`{rules['frozen_shells']}`。缺失 key/panel 或 frozen set 不匹配时 decision failure。", f"- canonical methods：`{', '.join(rules['canonical_methods'])}`。", f"- positive：`{', '.join(rules['positive_relations'])}`。", f"- negative：`{', '.join(rules['negative_relations'])}`；boundary：`{', '.join(rules['boundary_relations'])}`。", f"- decision metrics 仅为 `{', '.join(rules['decision_metrics'])}`；正 delta 表示 challenger 更好。", f"- {rules['positive_rule']}。", f"- {rules['head_increment_rule']}。", f"- {rules['identity_rule']}。", f"- {rules['negative_rule']}。", f"- {rules['shell_rule']}。", "- sRMSE 是 secondary，绝不驱动 decision；所有量纲不被混合或取 max。", "", "## 三个 head 相对 Geometry 的 positive 结果", "", "| head | metric | seed mean | 95% seed CI | seed win rate | n_seed_units | n_pair_units | pass |", "|---|---|---:|---|---:|---:|---:|---|"]
    for head in ("rank", "value", "dual"):
        ev = decision["head_evaluations"][head]
        for metric in DECISION_METRICS:
            d = ev["positive_metric_detail"][metric]
            lines.append(f"| {head} | {metric} | {_fmt(d['mean_delta'])} | [{_fmt(d['ci_low'])}, {_fmt(d['ci_high'])}] | {_fmt(d['seed_win_rate'])} | {d['n_seed_units']} | {d['n_pair_units']} | {'yes' if d['passes'] else 'no'} |")
    lines += ["", "## Value/Dual 相对 Rank 的增量（只用原始 results 配对）", "", "| challenger | metric | seed mean | 95% seed CI | seed win rate | n_seed_units | n_pair_units | pair W/T/L | pass |", "|---|---|---:|---|---:|---:|---:|---|---|"]
    for head in ("value", "dual"):
        ev = decision["head_to_head_evaluations"][head]
        for metric in DECISION_METRICS:
            d = ev["metric_detail"][metric]
            wtl = f"{d['pair_wins']}/{d['pair_ties']}/{d['pair_losses']}"
            lines.append(f"| {head} vs rank | {metric} | {_fmt(d['seed_level_mean_delta'])} | [{_fmt(d['ci_low'])}, {_fmt(d['ci_high'])}] | {_fmt(d['seed_win_rate'])} | {d['n_seeds']} | {d['n_pair_units']} | {wtl} | {'yes' if d['passes'] else 'no'} |")
        lines.append(f"| {head} | rule | {ev['metrics_passed']}/3 metrics; {ev['shells_passed']}/3 shells | — | — | {ev['n_seed_units']} | {ev['n_pair_units']} | — | {'yes' if ev['passes'] else 'no'} |")
    lines += ["", "## Negative 与 permutation 结论", ""]
    for head in ("rank", "value", "dual"):
        ev = decision["head_evaluations"][head]
        negative_lines = []
        for condition, condition_detail in ev["negative_condition_detail"].items():
            metric_text = ", ".join(
                f"{metric}: mean={_fmt(metric_detail['mean_delta'])}, seed_win={_fmt(metric_detail['seed_win_rate'])}, n_seed={metric_detail['n_seed_units']}, n_pair={metric_detail['n_pair_units']}, replicate={'yes' if metric_detail['replicates'] else 'no'}"
                for metric, metric_detail in condition_detail["metric_detail"].items()
            )
            negative_lines.append(f"{condition} ({condition_detail['metrics_replicating']}/3 metrics; condition replicate={'yes' if condition_detail['replicates'] else 'no'}): {metric_text}")
        neg = "; ".join(negative_lines)
        perm = "; ".join(
            f"{metric}: observed−permuted={_fmt(identity['mean_identity_minus_permuted'])}, CI=[{_fmt(identity['ci_low'])}, {_fmt(identity['ci_high'])}], seed_win={_fmt(identity['seed_win_rate'])}, n_seed={identity['n_seed_units']}, n_pair={identity['n_pair_units']}, pass={'yes' if identity['passes'] else 'no'}"
            for metric, identity in ev["identity_observed_vs_permuted"].items()
        )
        lines.append(f"- **{head}** negative conditions：{neg}。Permutation：{perm}；{ev['identity_metrics_passed']}/3 identity metrics pass。")
    lines += ["", "Scale_1.5 是 source-GP 外推关系；其结果不能解释为源域观测范围内的直接证据。", "", "结论严格按独立的 Geometry transfer 规则与 Rank-beyond 增量规则给出；不会因为三头 selection score 相同而默认选择 Rank。"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(input_dir: Path, config_path: Path | None = None, output_dir: Path | None = None) -> dict[str, Any]:
    input_dir = Path(input_dir).resolve(); paths = locate(input_dir); manifest = load_json(paths["manifest"]) if "manifest" in paths else {}
    if config_path is not None and Path(config_path).exists():
        config_source = Path(config_path).resolve(); config = load_json(config_source)
    elif "config" in paths:
        config_source = paths["config"].resolve(); config = load_json(config_source)
    else:
        config_source = None; config = manifest.get("config", {})
    results = _normalise_results(pd.read_csv(paths["results"])); ledger = _normalise_ledger(pd.read_csv(paths["ledger"])); diagnostics = pd.read_csv(paths["diagnostics"]) if "diagnostics" in paths else pd.DataFrame()
    output = (Path(output_dir) if output_dir else input_dir / "analysis").resolve(); output.mkdir(parents=True, exist_ok=True)
    summary = _summary(results); contrasts = _contrasts(results); summary.to_csv(output / "summary.csv", index=False); contrasts.to_csv(output / "contrasts.csv", index=False)
    _plot_headroom(contrasts, output / "relation_shell_headroom.png"); _plot_heads(contrasts, output / "rank_value_dual_vs_geometry.png"); _plot_identity(results, output / "identity_observed_vs_permuted.png")
    label, decision = decide(results, contrasts, config); decision["input_rows"] = {"results": int(len(results)), "ledger": int(len(ledger)), "diagnostics": int(len(diagnostics))}
    conclusion = input_dir / "ORACLE_LOCAL_MODEL_TRANSFER_QUICK_CONCLUSION_CN.md"; _write_conclusion(conclusion, label, decision, summary, contrasts); decision["conclusion_sha256"] = _sha256_file(conclusion)
    decision_path = output / "decision.json"
    decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

    # Write a provenance manifest only after every other analysis artifact is
    # finalized.  The manifest intentionally does not hash itself.
    source_files = {"results.csv": paths["results"], "prediction_ledger.csv": paths["ledger"]}
    if "diagnostics" in paths: source_files["source_expert_diagnostics.csv"] = paths["diagnostics"]
    if config_source is not None: source_files["config.json"] = config_source
    if "manifest" in paths: source_files["run_manifest.json"] = paths["manifest"]
    output_files = {
        "summary.csv": output / "summary.csv", "contrasts.csv": output / "contrasts.csv",
        "relation_shell_headroom.png": output / "relation_shell_headroom.png",
        "rank_value_dual_vs_geometry.png": output / "rank_value_dual_vs_geometry.png",
        "identity_observed_vs_permuted.png": output / "identity_observed_vs_permuted.png",
        "decision.json": decision_path,
    }
    analysis_manifest = {
        "analyzer": {"path": str(Path(__file__).resolve()), "sha256": _sha256_file(Path(__file__).resolve())},
        "source_inputs": {name: {"path": str(path.resolve()), "sha256": _sha256_file(path)} for name, path in source_files.items()},
        "outputs": {name: {"path": str(path.resolve()), "sha256": _sha256_file(path)} for name, path in output_files.items()},
        "conclusion": {"path": str(conclusion.resolve()), "sha256": _sha256_file(conclusion)},
    }
    (output / "analysis_manifest.json").write_text(json.dumps(analysis_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"summary": summary, "contrasts": contrasts, "decision": decision, "paths": paths}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(); parser.add_argument("--input", type=Path, default=ROOT / "results" / "oracle_local_model_transfer_quick"); parser.add_argument("--config", type=Path); parser.add_argument("--output", type=Path); return parser.parse_args()


if __name__ == "__main__":
    args = parse_args(); run_analysis(args.input, args.config, args.output)
