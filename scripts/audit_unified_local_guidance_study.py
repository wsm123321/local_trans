"""Semantic audit for the approved unified local-guidance artifacts.

The audit intentionally reconstructs the frozen metrics from the candidate panel
and full sequential traces.  Hashes are an integrity check, not a substitute for
semantic checks.  Every failed check is recorded before ``AUDIT.json`` and
``FULL_RUN_AUDIT.md`` are written and the CLI exits non-zero.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTANCE_KEYS = ("problem", "dim", "seed")
MAIN_METHODS = (
    "Target-Only",
    "Geometry-Only",
    "Local-Rank-No-Reliability",
    "Local-Rank+Reliability",
)
SAFETY_METHOD = "Reversed-Local-Rank"
MECHANISM_METHODS = frozenset(MAIN_METHODS + (SAFETY_METHOD,))
SEQUENTIAL_METHODS = frozenset(MAIN_METHODS)
CANONICAL_ARTIFACTS = (
    "mechanism_results.csv",
    "mechanism_candidate_panel.csv",
    "sequential_summary.csv",
    "sequential_traces.csv",
    "source_structure_diagnostics.csv",
    "failures.csv",
    "config.json",
)
# Kept as a compatibility alias for callers of the old helper.
APPROVED_FILES = CANONICAL_ARTIFACTS + ("run_manifest.json",)


def load_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def expected_instances(config: Mapping[str, Any]) -> List[Tuple[str, int, int]]:
    study = config.get("study", {})
    if not isinstance(study, Mapping):
        raise ValueError("config.study must be an object")
    required = ("problems", "dimensions", "seeds", "budget")
    missing = [key for key in required if key not in study]
    if missing:
        raise KeyError(f"config.study missing {missing}")
    problems = [str(x) for x in study["problems"]]
    dimensions = [int(x) for x in study["dimensions"]]
    seeds = [int(x) for x in study["seeds"]]
    if not problems or not dimensions or not seeds:
        raise ValueError("config study Cartesian axes must be non-empty")
    return [(problem, dim, seed) for dim in dimensions for problem in problems for seed in seeds]


def expected_counts(config: Mapping[str, Any]) -> Dict[str, int]:
    study = config.get("study", {})
    instances = expected_instances(config)
    budget = int(study["budget"])
    if budget < 1:
        raise ValueError("study.budget must be positive")
    proposal = int(study.get("proposal_size", 0))
    if proposal < 1:
        raise ValueError("study.proposal_size must be positive")
    n = len(instances)
    return {
        "instances": n,
        "mechanism_rows": n * len(MECHANISM_METHODS),
        "mechanism_candidate_rows": n * proposal,
        "sequential_summary_rows": n * len(SEQUENTIAL_METHODS),
        "sequential_trace_rows": n * len(SEQUENTIAL_METHODS) * budget,
        "sequential_initial_rows": n * len(SEQUENTIAL_METHODS),
        "sequential_trace_rows_total": n * len(SEQUENTIAL_METHODS) * (budget + 1),
        "budget": budget,
        "proposal_size": proposal,
    }


def validate_method_set(frame: pd.DataFrame, expected: Iterable[str]) -> Tuple[bool, str]:
    if "method" not in frame.columns:
        return False, "method column missing"
    observed = frozenset(frame["method"].dropna().astype(str))
    wanted = frozenset(map(str, expected))
    return observed == wanted, f"expected={sorted(wanted)}, observed={sorted(observed)}"


def _instance_tuples(frame: pd.DataFrame) -> List[Tuple[str, int, int]]:
    if any(k not in frame.columns for k in INSTANCE_KEYS):
        return []
    result = []
    for row in frame[list(INSTANCE_KEYS)].itertuples(index=False, name=None):
        try:
            result.append((str(row[0]), int(row[1]), int(row[2])))
        except (TypeError, ValueError):
            result.append((str(row[0]), -1, -1))
    return result


def _group_key(row: Mapping[str, Any]) -> Tuple[str, int, int]:
    return (str(row["problem"]), int(row["dim"]), int(row["seed"]))


def _expected_group_set(frame: pd.DataFrame, instances: Sequence[Tuple[str, int, int]]) -> Tuple[bool, str]:
    observed = set(_instance_tuples(frame))
    wanted = set(instances)
    missing, extra = sorted(wanted - observed), sorted(observed - wanted)
    return not missing and not extra, f"missing={missing[:5]}, extra={extra[:5]}"


def duplicate_count(frame: pd.DataFrame, keys: Sequence[str]) -> int:
    missing = [key for key in keys if key not in frame.columns]
    if missing:
        return -1
    return int(frame.duplicated(list(keys)).sum())


def shared_hash_violations(frame: pd.DataFrame, hash_columns: Sequence[str],
                           group_keys: Sequence[str] = INSTANCE_KEYS) -> int:
    missing = [key for key in list(group_keys) + list(hash_columns) if key not in frame.columns]
    if missing:
        return -1
    violations = 0
    for _, group in frame.groupby(list(group_keys), sort=False, dropna=False):
        if any(group[column].astype(str).nunique(dropna=False) != 1 for column in hash_columns):
            violations += 1
    return violations


def trace_length_violations(frame: pd.DataFrame, expected_length: int) -> Dict[str, Any]:
    keys = list(INSTANCE_KEYS) + ["method"]
    missing = [key for key in keys + ["step"] if key not in frame.columns]
    if missing:
        return {"missing": missing, "groups": -1, "bad_groups": -1, "lengths": []}
    lengths = frame.groupby(keys, sort=False, dropna=False).size()
    return {"missing": [], "groups": int(len(lengths)),
            "bad_groups": int((lengths != int(expected_length)).sum()),
            "lengths": sorted(set(map(int, lengths.tolist())))}


def _finite(frame: pd.DataFrame, columns: Iterable[str]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for column in columns:
        if column in frame.columns:
            result[column] = int((~np.isfinite(pd.to_numeric(frame[column], errors="coerce"))).sum())
    return result


def _resolve_hash_path(root: Path, value: str) -> Path:
    # Manifests were generated on Windows, but an audit can be run from either
    # Windows or a POSIX-compatible shell.
    normalized = str(value).replace("\\", "/")
    path = Path(normalized)
    if path.is_absolute():
        return path
    for candidate in (root / path, REPO_ROOT / path):
        if candidate.exists():
            return candidate
    return root / path


def _canonical_manifest_artifacts(manifest: Mapping[str, Any]) -> Dict[str, Tuple[str, str]]:
    declared = manifest.get("artifact_sha256")
    if not isinstance(declared, Mapping):
        return {}
    result: Dict[str, Tuple[str, str]] = {}
    for name, sha in declared.items():
        canonical = Path(str(name).replace("\\", "/")).name
        result[canonical] = (str(name), str(sha))
    return result


def _check_artifact_hashes(root: Path, manifest: Mapping[str, Any]) -> Tuple[bool, Any]:
    declared = manifest.get("artifact_sha256")
    if not isinstance(declared, Mapping):
        return False, "manifest.artifact_sha256 missing or not an object"
    failures = []
    for relative, expected in declared.items():
        path = _resolve_hash_path(root, str(relative))
        if not path.exists() or file_sha256(path) != str(expected):
            failures.append(str(relative))
    return not failures, ("all declared hashes match" if not failures else failures)


def target_only_selection_ok(frame: pd.DataFrame) -> Tuple[bool, str]:
    if "method" not in frame.columns:
        return False, "method column missing"
    target = frame[frame["method"].astype(str) == "Target-Only"]
    if target.empty:
        return False, "no Target-Only rows"
    if "acquisition_rank" not in target.columns:
        return False, "acquisition_rank column missing"
    rank = pd.to_numeric(target["acquisition_rank"], errors="coerce")
    ok = bool(rank.notna().all() and (rank == 0).all())
    return ok, f"runner uses zero-based rank; observed={sorted(rank.dropna().astype(int).unique().tolist())}"


def _check_fallback(mechanism: pd.DataFrame) -> Tuple[bool, str]:
    if "selected_index" not in mechanism.columns or "method" not in mechanism.columns:
        return False, "selected_index or method column missing"
    fallback_column = "fallback" if "fallback" in mechanism.columns else "effective_mode" if "effective_mode" in mechanism.columns else None
    if fallback_column is None:
        return False, "fallback marker column missing"
    target = mechanism[mechanism["method"].astype(str) == "Target-Only"].set_index(list(INSTANCE_KEYS))
    if target.empty:
        return False, "Target-Only rows unavailable"
    failures: List[str] = []
    methods = set(mechanism["method"].astype(str)) - {"Target-Only", "Geometry-Only"}
    for method in methods:
        method_rows = mechanism[mechanism["method"].astype(str) == method]
        marker = method_rows[fallback_column].astype(str).str.lower()
        is_fallback = marker.isin({"true", "1", "yes"}) if fallback_column == "fallback" else marker.isin({"target_only", "target-only", "fallback"})
        part = method_rows.loc[is_fallback].set_index(list(INSTANCE_KEYS))
        common = part.index.intersection(target.index)
        if len(common) and not np.array_equal(
            pd.to_numeric(part.loc[common, "selected_index"], errors="coerce").to_numpy(),
            pd.to_numeric(target.loc[common, "selected_index"], errors="coerce").to_numpy(),
        ):
            failures.append(method)
    return not failures, ("all fallback selections equal Target-Only" if not failures else failures)


def _parse_vector(value: Any, dim: int) -> np.ndarray:
    parsed = json.loads(str(value))
    arr = np.asarray(parsed, dtype=float).reshape(-1)
    if len(arr) != int(dim) or not np.isfinite(arr).all():
        raise ValueError(f"x is not a finite vector of dimension {dim}")
    return arr


def _isclose(a: Any, b: Any) -> bool:
    try:
        return bool(np.isclose(float(a), float(b), rtol=1e-9, atol=1e-12, equal_nan=False))
    except (TypeError, ValueError):
        return False


def _value_equal(observed: Any, expected: Any) -> bool:
    if isinstance(expected, (bool, np.bool_)):
        if isinstance(observed, str):
            text = observed.strip().lower()
            if text in {"true", "1", "yes"}:
                observed = True
            elif text in {"false", "0", "no"}:
                observed = False
            else:
                return False
        elif pd.isna(observed):
            return False
        else:
            observed = bool(observed)
        return observed == bool(expected)
    if isinstance(expected, (int, np.integer)) and not isinstance(expected, bool):
        try:
            return int(observed) == int(expected) and float(observed) == float(expected)
        except (TypeError, ValueError):
            return False
    if isinstance(expected, (float, np.floating)):
        return _isclose(observed, expected)
    return str(observed) == str(expected)


def _mechanism_panel_check(mechanism: pd.DataFrame, panel: pd.DataFrame,
                           instances: Sequence[Tuple[str, int, int]], config: Mapping[str, Any]) -> Tuple[bool, str]:
    required_panel = set(INSTANCE_KEYS) | {"candidate_index", "x", "acquisition", "truth", "candidate_count", "raw_pool_hash", "proposal_hash", "truth_hash"}
    required_mech = set(INSTANCE_KEYS) | {"method", "selected_index", "selected_x", "selected_y", "truth_min", "truth_q90", "raw_regret", "normalized_regret", "top10_hit", "true_rank", "acquisition_rank", "source_score", "normalized_source_score", "normalized_target_score", "combined_score", "target_top1_retained", "candidate_count", "raw_pool_hash", "proposal_hash", "truth_hash"}
    missing_panel, missing_mech = sorted(required_panel - set(panel.columns)), sorted(required_mech - set(mechanism.columns))
    if missing_panel or missing_mech:
        return False, f"missing_panel={missing_panel}, missing_mechanism={missing_mech}"
    proposal_size = int(config["study"]["proposal_size"])
    tolerance = float(config["study"].get("ranking_tolerance", 1e-12))
    errors: List[str] = []
    wanted = set(instances)
    panel_groups = {(str(p), int(d), int(s)): g.copy() for (p, d, s), g in panel.groupby(list(INSTANCE_KEYS), sort=False, dropna=False)}
    mech_groups = {(str(p), int(d), int(s)): g.copy() for (p, d, s), g in mechanism.groupby(list(INSTANCE_KEYS), sort=False, dropna=False)}
    if set(panel_groups) != wanted:
        errors.append(f"panel instance set mismatch missing={sorted(wanted-set(panel_groups))[:3]} extra={sorted(set(panel_groups)-wanted)[:3]}")
    if set(mech_groups) != wanted:
        errors.append(f"mechanism instance set mismatch missing={sorted(wanted-set(mech_groups))[:3]} extra={sorted(set(mech_groups)-wanted)[:3]}")
    for key in sorted(wanted):
        pg, mg = panel_groups.get(key), mech_groups.get(key)
        if pg is None or mg is None:
            continue
        if len(pg) != proposal_size:
            errors.append(f"{key}: panel rows={len(pg)} expected={proposal_size}")
            continue
        ci = pd.to_numeric(pg["candidate_index"], errors="coerce")
        ci_values = ci.to_numpy(dtype=float)
        integer_ci = np.isfinite(ci_values).all() and np.equal(ci_values, np.floor(ci_values)).all()
        if ci.isna().any() or not integer_ci or not np.array_equal(np.sort(ci_values.astype(int)), np.arange(proposal_size)) or ci.duplicated().any():
            errors.append(f"{key}: candidate_index is not unique complete 0..P-1")
            continue
        pg = pg.assign(_ci=ci.astype(int)).sort_values("_ci", kind="stable")
        count = pd.to_numeric(pg["candidate_count"], errors="coerce")
        if count.isna().any() or not (count == proposal_size).all():
            errors.append(f"{key}: candidate_count mismatch")
        try:
            x_values = [_parse_vector(v, key[1]) for v in pg["x"]]
        except Exception as exc:
            errors.append(f"{key}: invalid panel x: {exc}")
            continue
        acq = pd.to_numeric(pg["acquisition"], errors="coerce").to_numpy(float)
        truth = pd.to_numeric(pg["truth"], errors="coerce").to_numpy(float)
        if not np.isfinite(acq).all() or not np.isfinite(truth).all():
            errors.append(f"{key}: acquisition/truth not finite")
            continue
        hashes = {}
        for col in ("raw_pool_hash", "proposal_hash", "truth_hash"):
            values = pg[col].astype(str).tolist()
            hashes[col] = values[0] if values and len(set(values)) == 1 else None
            if hashes[col] is None:
                errors.append(f"{key}: {col} is not single-valued in panel")
        min_y, q90 = float(np.min(truth)), float(np.quantile(truth, .90))
        scale = max(q90 - min_y, 1e-12)
        true_order = np.argsort(truth, kind="stable")
        acq_order = np.argsort(-acq, kind="stable")
        true_rank = np.empty(proposal_size, dtype=int); true_rank[true_order] = np.arange(proposal_size)
        acq_rank = np.empty(proposal_size, dtype=int); acq_rank[acq_order] = np.arange(proposal_size)
        top10 = truth <= float(np.quantile(truth, .10)) + tolerance
        # Reconstruct the runner's stable nomination/shortlist/argmax from the
        # common panel.  This catches a changed selected_index even when all
        # scalar regret fields happen to remain plausible.
        guidance = config.get("guidance", {})
        target_norm = np.zeros(proposal_size, dtype=float)
        if proposal_size > 1 and np.ptp(acq) >= 1e-12:
            target_norm = (rankdata(acq, method="average") - 1.0) / (proposal_size - 1.0)
        target_count = max(1, int(np.ceil(proposal_size * float(guidance.get("target_nomination_ratio", .20)))))
        target_nominees = np.argsort(-target_norm, kind="stable")[:target_count]
        mode_score_columns = {
            "Geometry-Only": "geometry_score",
            "Local-Rank-No-Reliability": "local_rank_no_reliability_score",
            "Local-Rank+Reliability": "local_rank_reliability_score",
            "Reversed-Local-Rank": "reversed_local_rank_score",
        }
        methods = set(mg["method"].astype(str))
        if methods != set(MECHANISM_METHODS) or len(mg) != len(MECHANISM_METHODS) or mg["method"].duplicated().any():
            errors.append(f"{key}: mechanism methods/rows mismatch")
        # Reproduce rank_local_structure_candidates using only the panel's
        # published score arrays.  All methods share the same target nominees;
        # source methods union target/source nominees and then stable-argmax the
        # combined normalized score.  Constant/empty source scores are the
        # runner's explicit Target-Only fallback.
        expected_selection: Dict[str, int] = {"Target-Only": int(target_nominees[0])}
        source_weight = float(guidance.get("source_weight", 1.0))
        source_count = max(1, int(np.ceil(proposal_size * float(guidance.get("source_nomination_ratio", .20)))))
        for method, score_column in mode_score_columns.items():
            norm_column = score_column + "_normalized"
            if score_column not in pg.columns or norm_column not in pg.columns:
                errors.append(f"{key}: panel missing {score_column} or {norm_column}")
                continue
            source_raw = pd.to_numeric(pg[score_column], errors="coerce").to_numpy(float)
            source_norm_published = pd.to_numeric(pg[norm_column], errors="coerce").to_numpy(float)
            if not np.isfinite(source_raw).all() or not np.isfinite(source_norm_published).all():
                errors.append(f"{key}: non-finite {score_column}")
                continue
            if source_weight == 0.0 or np.ptp(source_raw) < 1e-12:
                expected_norm = np.zeros(proposal_size, dtype=float)
                fallback = True
            else:
                expected_norm = (rankdata(source_raw, method="average") - 1.0) / (proposal_size - 1.0)
                fallback = False
            if not np.allclose(source_norm_published, expected_norm, rtol=1e-9, atol=1e-12, equal_nan=False):
                errors.append(f"{key}: {norm_column} does not match stable rank normalization")
            if fallback:
                expected_selection[method] = int(target_nominees[0])
            else:
                source_nominees = np.argsort(-expected_norm, kind="stable")[:source_count]
                shortlist = np.asarray(sorted(set(target_nominees.tolist()) | set(source_nominees.tolist())), dtype=int)
                combined = target_norm + source_weight * expected_norm
                expected_selection[method] = int(shortlist[np.argsort(-combined[shortlist], kind="stable")[0]])
        for _, row in mg.iterrows():
            method = str(row["method"])
            if method not in MECHANISM_METHODS:
                continue
            try:
                idx = int(row["selected_index"])
                sx = _parse_vector(row["selected_x"], key[1])
            except Exception as exc:
                errors.append(f"{key}/{method}: invalid selection: {exc}")
                continue
            if not 0 <= idx < proposal_size:
                errors.append(f"{key}/{method}: selected_index outside panel")
                continue
            # Diagnostic fields are also reconstructed from the panel.  The
            # panel is the sole source of truth for these values; no runner code
            # or hidden candidate state is consulted.
            if method == "Target-Only":
                source_score = 0.0
                source_norm = 0.0
                combined_score = target_norm[idx]
                expected_top1_retained = True
            else:
                score_column = mode_score_columns.get(method)
                score_norm_column = score_column + "_normalized" if score_column else None
                source_score_values = pd.to_numeric(pg[score_column], errors="coerce").to_numpy(float) if score_column in pg.columns else np.full(proposal_size, np.nan)
                source_norm_values = pd.to_numeric(pg[score_norm_column], errors="coerce").to_numpy(float) if score_norm_column in pg.columns else np.full(proposal_size, np.nan)
                source_score = source_score_values[idx]
                source_norm = source_norm_values[idx]
                fallback = source_weight == 0.0 or np.ptp(source_score_values) < 1e-12
                if fallback:
                    source_score = 0.0
                    source_norm = 0.0
                    combined_score = target_norm[idx]
                else:
                    combined_score = target_norm[idx] + source_weight * source_norm
                expected_top1_retained = True
            expected_values = {
                "selected_index": idx,
                "selected_y": truth[idx], "truth_min": min_y, "truth_q90": q90,
                "raw_regret": truth[idx] - min_y,
                "normalized_regret": (truth[idx] - min_y) / scale,
                "top10_hit": bool(top10[idx]), "true_rank": int(true_rank[idx]),
                "acquisition_rank": int(acq_rank[idx]), "candidate_count": proposal_size,
                "source_score": source_score, "normalized_source_score": source_norm,
                "normalized_target_score": target_norm[idx], "combined_score": combined_score,
                "target_top1_retained": expected_top1_retained,
                "raw_pool_hash": hashes.get("raw_pool_hash"), "proposal_hash": hashes.get("proposal_hash"),
                "truth_hash": hashes.get("truth_hash"),
            }
            if not np.allclose(sx, x_values[idx], rtol=1e-9, atol=1e-12, equal_nan=False):
                errors.append(f"{key}/{method}: selected_x mismatch")
            for field, expected in expected_values.items():
                if expected is None or field not in row or not _value_equal(row[field], expected):
                    errors.append(f"{key}/{method}: {field} mismatch")
            if method in expected_selection and idx != expected_selection[method]:
                errors.append(f"{key}/{method}: selected_index does not match panel score reconstruction")
            if method == "Target-Only" and idx != int(acq_order[0]):
                errors.append(f"{key}/{method}: not stable acquisition argmax")
    # The same panel hashes must be represented identically on all five mechanism rows.
    for col in ("raw_pool_hash", "proposal_hash", "truth_hash"):
        if col in mechanism.columns and shared_hash_violations(mechanism, [col]) != 0:
            errors.append(f"mechanism {col} is not single-valued per instance")
    return not errors, ("all panel-derived mechanism fields match" if not errors else errors[:30])


def _trace_semantics_check(traces: pd.DataFrame, summary: pd.DataFrame,
                           instances: Sequence[Tuple[str, int, int]], config: Mapping[str, Any]) -> Tuple[bool, str]:
    required_trace = set(INSTANCE_KEYS) | {"method", "step", "best_y", "selected_y", "known_optimum_y", "normalized_regret"}
    required_summary = set(INSTANCE_KEYS) | {"method", "initial_best_y", "final_best_y", "known_optimum_y", "final_normalized_regret", "auc_normalized_regret", "total_improvement", "trace_points"}
    missing = sorted(required_trace - set(traces.columns)) + sorted(required_summary - set(summary.columns))
    if missing:
        return False, f"missing columns={missing}"
    budget = int(config["study"]["budget"])
    wanted = set(instances)
    errors: List[str] = []
    groups = {(str(p), int(d), int(s), str(m)): g.copy() for (p, d, s, m), g in traces.groupby(list(INSTANCE_KEYS)+["method"], sort=False, dropna=False)}
    expected_groups = {(p, d, s, m) for p, d, s in instances for m in SEQUENTIAL_METHODS}
    if set(groups) != expected_groups:
        errors.append(f"trace group set mismatch missing={sorted(expected_groups-set(groups))[:3]} extra={sorted(set(groups)-expected_groups)[:3]}")
    summary_groups = {(str(p), int(d), int(s), str(m)): g for (p, d, s, m), g in summary.groupby(list(INSTANCE_KEYS)+["method"], sort=False, dropna=False)}
    if set(summary_groups) != expected_groups:
        errors.append(f"summary group set mismatch missing={sorted(expected_groups-set(summary_groups))[:3]} extra={sorted(set(summary_groups)-expected_groups)[:3]}")
    for key in sorted(expected_groups):
        group = groups.get(key)
        sg = summary_groups.get(key)
        if group is None or sg is None or len(sg) != 1:
            continue
        steps = pd.to_numeric(group["step"], errors="coerce")
        step_values = steps.to_numpy(dtype=float)
        integer_steps = np.isfinite(step_values).all() and np.equal(step_values, np.floor(step_values)).all()
        if steps.isna().any() or not integer_steps or not np.array_equal(np.sort(step_values.astype(int)), np.arange(budget + 1)) or steps.duplicated().any():
            errors.append(f"{key}: steps are not exact 0..budget")
            continue
        group = group.assign(_step=steps.astype(int)).sort_values("_step", kind="stable")
        best = pd.to_numeric(group["best_y"], errors="coerce").to_numpy(float)
        selected = pd.to_numeric(group["selected_y"], errors="coerce").to_numpy(float)
        optimum = pd.to_numeric(group["known_optimum_y"], errors="coerce").to_numpy(float)
        norm = pd.to_numeric(group["normalized_regret"], errors="coerce").to_numpy(float)
        if not np.isfinite(best).all() or not np.isfinite(optimum).all() or not np.isfinite(norm).all():
            errors.append(f"{key}: non-finite best/optimum/normalized regret")
            continue
        if not np.isnan(selected[0]) or not np.isfinite(selected[1:]).all():
            errors.append(f"{key}: step0 selected_y must be NaN and paid selected_y finite")
        if np.any(np.diff(best) > 0.0):
            errors.append(f"{key}: best_y increases")
        if len(set(optimum.tolist())) != 1:
            errors.append(f"{key}: known optimum is not constant")
        initial, final, known = float(best[0]), float(best[-1]), float(optimum[0])
        scale = max(1e-12, initial - known)
        expected_norm = np.maximum(0.0, best - known) / scale
        if not np.allclose(norm, expected_norm, rtol=1e-9, atol=1e-12, equal_nan=False):
            errors.append(f"{key}: normalized regret is not reconstructed from initial/optimum")
        srow = sg.iloc[0]
        expected_summary = {
            "initial_best_y": initial, "final_best_y": final, "known_optimum_y": known,
            "final_normalized_regret": float(expected_norm[-1]),
            "auc_normalized_regret": float(np.trapezoid(expected_norm, np.arange(budget+1)) if hasattr(np, "trapezoid") else np.trapz(expected_norm, np.arange(budget+1))),
            "total_improvement": initial - final, "trace_points": budget + 1,
        }
        for field, expected in expected_summary.items():
            if not _value_equal(srow[field], expected):
                errors.append(f"{key}: summary {field} mismatch")
    return not errors, ("all trace and summary fields match" if not errors else errors[:30])


def _load_analyzer() -> Any:
    path = REPO_ROOT / "scripts" / "analyze_unified_local_guidance_study.py"
    spec = importlib.util.spec_from_file_location("unified_local_guidance_analyzer_for_audit", path)
    if not spec or not spec.loader:
        raise ImportError(f"cannot import analyzer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _primary_check(mechanism: pd.DataFrame, summary: pd.DataFrame,
                   root: Path, config: Mapping[str, Any]) -> Tuple[bool, str]:
    analysis_dir = root / "analysis"
    primary_path = analysis_dir / "PRIMARY_TESTS.csv"
    if not primary_path.exists():
        return False, f"missing={primary_path}"
    observed = pd.read_csv(primary_path)
    try:
        analyzer = _load_analyzer()
        specs = tuple(analyzer.PRIMARY_CONTRASTS)
        if len(specs) != 5:
            return False, f"analyzer PRIMARY_CONTRASTS length={len(specs)}"
        analysis = config.get("analysis", {}) if isinstance(config.get("analysis", {}), Mapping) else {}
        n_bootstrap = int(analysis.get("bootstrap_samples", 5000)); seed = int(analysis.get("bootstrap_seed", 42))
        alpha = float(analysis.get("familywise_alpha", 0.05))
        frames = {"mechanism": mechanism, "sequential": summary}
        expected_rows: List[Dict[str, Any]] = []
        for index, spec in enumerate(specs):
            spec = dict(spec)
            dataset = str(spec["dataset"])
            differences = analyzer.strict_paired_differences(
                frames[dataset], str(spec["method_a"]), str(spec["method_b"]),
                str(spec["metric"]), bool(spec["higher_is_better"]), keys=INSTANCE_KEYS)
            row = dict(spec)
            row.update(analyzer.paired_statistics(differences, n_bootstrap=n_bootstrap, seed=seed + index))
            expected_rows.append(row)
        adjusted = analyzer.holm_adjust([r["wilcoxon_pratt_one_sided_p"] for r in expected_rows])
        for row, p in zip(expected_rows, adjusted):
            row["holm_adjusted_p"] = p
            row["supported"] = bool(row["ci_low"] > 0.0 and p < alpha)
        if len(observed) != len(expected_rows):
            return False, f"PRIMARY_TESTS rows={len(observed)} expected=5"
        errors: List[str] = []
        for i, expected in enumerate(expected_rows):
            actual = observed.iloc[i]
            for field, value in expected.items():
                if field not in observed.columns or not _value_equal(actual[field], value):
                    errors.append(f"row {i} {field} expected={value!r} observed={actual.get(field)!r}")
        # Every analyzer numeric output, including aliases/rates and W/T/L/n,
        # is covered by the positional field comparison above.
        return not errors, ("five PRIMARY contrasts and Holm/support status match analyzer" if not errors else errors[:30])
    except Exception as exc:
        return False, repr(exc)


def _write_audit_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    lines = ["# Unified local-guidance full-run audit", "",
             f"- Overall status: **{'PASS' if payload['ok'] else 'FAIL'}**",
             f"- Stage: `{payload.get('stage_id')}`", "", "## Checks", "",
             "| Check | Status | Detail |", "|---|---|---|"]
    for item in payload["checks"]:
        detail = str(item["detail"]).replace("|", "\\|")
        lines.append(f"| {item['name']} | {'PASS' if item['passed'] else 'FAIL'} | {detail} |")
    lines += ["", "## Warnings", ""]
    lines += [f"- {warning}" for warning in payload.get("warnings", [])] or ["- None."]
    lines += ["", "## Counts", ""]
    lines += [f"- `{key}`: {value}" for key, value in payload["counts"].items()]
    lines += ["", "## Artifact SHA-256", ""]
    lines += [f"- `{key}`: `{value}`" for key, value in payload["artifact_sha256"].items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(input_dir: Path, config_path: Optional[Path] = None,
              manifest_path: Optional[Path] = None) -> Dict[str, Any]:
    root = Path(input_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "mechanism": root / "mechanism_results.csv",
        "panel": root / "mechanism_candidate_panel.csv",
        "source_diagnostics": root / "source_structure_diagnostics.csv",
        "sequential_summary": root / "sequential_summary.csv",
        "sequential_traces": root / "sequential_traces.csv",
        "failures": root / "failures.csv",
        "config": Path(config_path).resolve() if config_path else root / "config.json",
        "manifest": Path(manifest_path).resolve() if manifest_path else root / "run_manifest.json",
    }
    checks: List[Dict[str, Any]] = []
    warnings: List[str] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": str(detail)})

    def semantic_check(name: str, function, *args) -> None:
        """Turn malformed semantic input into a recorded check failure."""
        try:
            result = function(*args)
            if not isinstance(result, tuple) or len(result) != 2:
                raise TypeError("semantic helper must return (passed, detail)")
            check(name, result[0], result[1])
        except Exception as exc:
            check(name, False, f"semantic audit exception: {type(exc).__name__}: {exc}")

    missing = [str(path) for path in paths.values() if not path.exists()]
    check("approved_artifact_paths", not missing, "all canonical files present" if not missing else missing)
    config: Dict[str, Any] = {}; manifest: Dict[str, Any] = {}
    try:
        config = load_json(paths["config"]); manifest = load_json(paths["manifest"])
        check("json_readable", True, "config and manifest parse as objects")
    except Exception as exc:
        check("json_readable", False, repr(exc))
    frames: Dict[str, pd.DataFrame] = {}
    csv_paths = {k: paths[k] for k in ("mechanism", "panel", "source_diagnostics", "sequential_summary", "sequential_traces", "failures")}
    for key, path in csv_paths.items():
        try:
            frames[key] = pd.read_csv(path)
        except Exception as exc:
            check(f"{key}_readable", False, repr(exc))
    check("csv_readable", len(frames) == len(csv_paths), sorted(frames))
    mechanism = frames.get("mechanism", pd.DataFrame()); panel = frames.get("panel", pd.DataFrame())
    diagnostics = frames.get("source_diagnostics", pd.DataFrame()); summary = frames.get("sequential_summary", pd.DataFrame())
    traces = frames.get("sequential_traces", pd.DataFrame()); failures = frames.get("failures", pd.DataFrame())
    counts: Dict[str, int] = {}
    instances: List[Tuple[str, int, int]] = []
    try:
        counts = expected_counts(config); instances = expected_instances(config)
        check("expected_cartesian_instances_from_config", True, f"instances={len(instances)}")
    except Exception as exc:
        check("expected_cartesian_instances_from_config", False, repr(exc))
    if config and manifest:
        config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        check("config_hash", str(manifest.get("config_sha256")) == config_hash, f"expected={config_hash}, observed={manifest.get('config_sha256')}")
        check("stage_identity", config.get("stage_id") == manifest.get("stage_id"), f"config={config.get('stage_id')}, manifest={manifest.get('stage_id')}")
        protocol_path, protocol_hash = manifest.get("protocol_path"), manifest.get("protocol_sha256")
        if protocol_path and protocol_hash:
            actual = _resolve_hash_path(root, str(protocol_path))
            check("protocol_hash", actual.exists() and file_sha256(actual) == str(protocol_hash), str(actual))
        else:
            check("protocol_hash", False, "manifest protocol path/hash missing")
        runner_path, runner_hash = manifest.get("runner_path"), manifest.get("runner_sha256")
        if runner_path and runner_hash:
            actual = _resolve_hash_path(root, str(runner_path))
            check("runner_hash", actual.exists() and file_sha256(actual) == str(runner_hash), str(actual))
        else:
            check("runner_hash", False, "manifest runner path/hash missing")
        companion_path, companion_hash = manifest.get("companion_path"), manifest.get("companion_sha256")
        if companion_path and companion_hash:
            actual = _resolve_hash_path(root, str(companion_path))
            check("companion_hash", actual.exists() and file_sha256(actual) == str(companion_hash), str(actual))
        else:
            check("companion_hash", False, "manifest companion path/hash missing")
        declared = _canonical_manifest_artifacts(manifest)
        missing_declared = sorted(set(CANONICAL_ARTIFACTS) - set(declared))
        check("canonical_artifact_declarations", not missing_declared, f"missing={missing_declared}")
        hash_ok, detail = _check_artifact_hashes(root, manifest)
        check("artifact_hashes", hash_ok, detail)
    check("mechanism_method_set", *validate_method_set(mechanism, MECHANISM_METHODS))
    check("sequential_method_set", *validate_method_set(summary, SEQUENTIAL_METHODS))
    check("trace_method_set", *validate_method_set(traces, SEQUENTIAL_METHODS))
    semantic_check("mechanism_instance_set", _expected_group_set, mechanism, instances) if instances else check("mechanism_instance_set", False, "no expected instances")
    semantic_check("panel_instance_set", _expected_group_set, panel, instances) if instances else check("panel_instance_set", False, "no expected instances")
    semantic_check("sequential_summary_instance_set", _expected_group_set, summary, instances) if instances else check("sequential_summary_instance_set", False, "no expected instances")
    semantic_check("sequential_trace_instance_set", _expected_group_set, traces, instances) if instances else check("sequential_trace_instance_set", False, "no expected instances")
    mechanism_key = list(INSTANCE_KEYS) + ["method"]; summary_key = mechanism_key; trace_key = list(INSTANCE_KEYS) + ["method", "step"]
    check("mechanism_unique_keys", duplicate_count(mechanism, mechanism_key) == 0, f"duplicates={duplicate_count(mechanism, mechanism_key)}")
    check("sequential_summary_unique_keys", duplicate_count(summary, summary_key) == 0, f"duplicates={duplicate_count(summary, summary_key)}")
    check("sequential_trace_unique_keys", duplicate_count(traces, trace_key) == 0, f"duplicates={duplicate_count(traces, trace_key)}")
    if counts:
        check("mechanism_row_count", len(mechanism) == counts["mechanism_rows"], f"expected={counts['mechanism_rows']}, observed={len(mechanism)}")
        check("mechanism_panel_row_count", len(panel) == counts["mechanism_candidate_rows"], f"expected={counts['mechanism_candidate_rows']}, observed={len(panel)}")
        check("sequential_summary_row_count", len(summary) == counts["sequential_summary_rows"], f"expected={counts['sequential_summary_rows']}, observed={len(summary)}")
        steps = pd.to_numeric(traces["step"], errors="coerce") if "step" in traces.columns else pd.Series(dtype=float)
        paid = int((steps > 0).sum()) if "step" in traces.columns else -1; initial = int((steps == 0).sum()) if "step" in traces.columns else -1
        check("sequential_trace_row_count", paid == counts["sequential_trace_rows"] and initial == counts["sequential_initial_rows"] and len(traces) == counts["sequential_trace_rows_total"], f"expected_paid={counts['sequential_trace_rows']}, observed_paid={paid}, initial_rows={initial}, total={len(traces)}")
    check("finite_mechanism_values", bool(_finite(mechanism, ["selected_y", "truth_min", "truth_q90", "raw_regret", "normalized_regret"])) and all(v == 0 for v in _finite(mechanism, ["selected_y", "truth_min", "truth_q90", "raw_regret", "normalized_regret"]).values()), _finite(mechanism, ["selected_y", "truth_min", "truth_q90", "raw_regret", "normalized_regret"]))
    check("finite_summary_values", bool(_finite(summary, ["initial_best_y", "final_best_y", "known_optimum_y", "final_normalized_regret", "auc_normalized_regret", "total_improvement"])) and all(v == 0 for v in _finite(summary, ["initial_best_y", "final_best_y", "known_optimum_y", "final_normalized_regret", "auc_normalized_regret", "total_improvement"]).values()), _finite(summary, ["initial_best_y", "final_best_y", "known_optimum_y", "final_normalized_regret", "auc_normalized_regret", "total_improvement"]))
    if not failures.empty or failures.shape[1] == 0:
        check("zero_failures", failures.empty, f"failure_rows={len(failures)}")
    else:
        check("zero_failures", True, "failure_rows=0")
    if instances and config:
        semantic_check("mechanism_panel_semantics", _mechanism_panel_check, mechanism, panel, instances, config)
        semantic_check("trace_and_summary_semantics", _trace_semantics_check, traces, summary, instances, config)
    else:
        check("mechanism_panel_semantics", False, "config/panel unavailable")
        check("trace_and_summary_semantics", False, "config/traces unavailable")
    hash_columns = [c for c in ("raw_pool_hash", "proposal_hash", "truth_hash") if c in mechanism.columns]
    check("mechanism_candidate_hashes_shared_across_five_methods", bool(hash_columns) and shared_hash_violations(mechanism, hash_columns) == 0, f"columns={hash_columns}, violations={shared_hash_violations(mechanism, hash_columns)}")
    if instances:
        try:
            diagnostics_ok, diagnostics_detail = _expected_group_set(diagnostics, instances)
            check("source_structure_diagnostics_present", bool(len(diagnostics)) and diagnostics_ok, f"rows={len(diagnostics)}, {diagnostics_detail}")
        except Exception as exc:
            check("source_structure_diagnostics_present", False, f"semantic audit exception: {type(exc).__name__}: {exc}")
    else:
        check("source_structure_diagnostics_present", False, "no expected instances")
    semantic_check("target_only_zero_based_acquisition_rank", target_only_selection_ok, mechanism)
    semantic_check("fallback_consistency", _check_fallback, mechanism)
    required_analysis = [root / "analysis" / name for name in ("PRIMARY_TESTS.csv", "METHOD_SUMMARY.csv", "SECONDARY_CONTRASTS.csv")]
    check("analysis_exists", all(path.exists() for path in required_analysis), [str(path) for path in required_analysis if not path.exists()])
    check("primary_semantics", *_primary_check(mechanism, summary, root, config) if config else (False, "config unavailable"))
    if isinstance(config.get("study"), Mapping) and float(config["study"].get("target_noise_std", 0.0)) == 0.0:
        warnings.append("target_noise_std=0: runner noise stream bug is documented and does not affect this full run")
    warnings.append("known optimum is a declared oracle basin-center approximation, not a separately verified global optimum")
    passed = all(item["passed"] for item in checks)
    hashed: Dict[str, str] = {}
    for path in list(paths.values()) + required_analysis + [root / "analysis" / "mechanism_regret.png", root / "analysis" / "sequential_final.png", root / "analysis" / "sequential_auc.png"]:
        if path.exists() and path.name != "run_manifest.json":
            try:
                hashed[str(path.relative_to(REPO_ROOT))] = file_sha256(path)
            except ValueError:
                hashed[str(path)] = file_sha256(path)
    payload = {"ok": passed, "stage_id": config.get("stage_id", manifest.get("stage_id")), "checks": checks,
               "warnings": warnings,
               "counts": {**counts, "mechanism_rows_observed": len(mechanism), "mechanism_panel_rows_observed": len(panel), "sequential_summary_rows_observed": len(summary), "sequential_trace_rows_observed": len(traces)},
               "artifact_sha256": hashed}
    (root / "AUDIT.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_audit_markdown(root / "FULL_RUN_AUDIT.md", payload)
    if not passed:
        raise RuntimeError(f"Audit failed: {[item['name'] for item in checks if not item['passed']]}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "results" / "unified_local_guidance")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()
    try:
        run_audit(args.input, args.config, args.manifest)
    except Exception as exc:
        print(str(exc))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
