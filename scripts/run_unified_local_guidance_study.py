"""Run the frozen unified local-guidance experiment.

The runner deliberately keeps proposal generation and guidance separate.  The
mechanism phase shares one target-only GP, raw pool, EI proposal panel, and truth
vector across all methods.  Sequential methods have independent target histories
and therefore fit independent target-only GPs at each step.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy
import sklearn

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from region_guided_reranking_study.landscapes import get_task_suite  # noqa: E402
from region_guided_reranking_study.screening_research import known_optimum_value  # noqa: E402
from region_guided_reranking_study.source_local_structure import (  # noqa: E402
    LocalStructureConfig,
    SourceLocalStructureExtractor,
    SourceLocalStructureLibrary,
)
from region_guided_reranking_study.source_structure_research import (  # noqa: E402
    latin_hypercube_sample,
)
from region_guided_reranking_study.surrogate_and_candidates import (  # noqa: E402
    TargetGPSurrogate,
)
from region_guided_reranking_study.local_structure_guidance import (  # noqa: E402
    LocalStructureGuidanceDecision,
    rank_local_structure_candidates,
)

PROTOCOL_PATH = REPO_ROOT / "PROTOCOL_UNIFIED_LOCAL_GUIDANCE.md"
RUNNER_PATH = Path(__file__).resolve()
MAIN_METHODS: Tuple[str, ...] = (
    "Target-Only",
    "Geometry-Only",
    "Local-Rank-No-Reliability",
    "Local-Rank+Reliability",
)
SAFETY_METHOD = "Reversed-Local-Rank"
ALL_MECHANISM_METHODS: Tuple[str, ...] = MAIN_METHODS + (SAFETY_METHOD,)
METHOD_MODES = {
    "Target-Only": "target_only",
    "Geometry-Only": "geometry_only",
    "Local-Rank-No-Reliability": "local_rank_no_reliability",
    "Local-Rank+Reliability": "local_rank_reliability",
    "Reversed-Local-Rank": "reversed_local_rank",
}


def load_config(path: Path | str) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a JSON object")
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    required = {"protocol_id", "stage_id", "output_dir", "study", "extraction", "guidance", "analysis"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"configuration is missing keys: {sorted(missing)}")
    study = config["study"]
    for key in ("problems", "dimensions", "seeds", "source_samples", "raw_pool_size", "proposal_size", "budget", "acquisition", "methods", "safety_diagnostics"):
        if key not in study:
            raise ValueError(f"study is missing {key!r}")
    if str(study["acquisition"]).lower() != "ei":
        raise ValueError("the unified protocol fixes acquisition to EI")
    if list(study["methods"]) != list(MAIN_METHODS):
        raise ValueError("methods must be exactly the four approved principal methods")
    if list(study["safety_diagnostics"]) != [SAFETY_METHOD]:
        raise ValueError("safety_diagnostics must be exactly Reversed-Local-Rank")
    if int(study["proposal_size"]) < 1 or int(study["proposal_size"]) > int(study["raw_pool_size"]):
        raise ValueError("proposal_size must lie within raw_pool_size")
    if str(study.get("n_init_formula", "2d+2")) != "2d+2":
        raise ValueError("n_init is frozen to 2d+2")
    guidance = config["guidance"]
    for key in ("source_weight", "target_nomination_ratio", "source_nomination_ratio", "aggregation"):
        if key not in guidance:
            raise ValueError(f"guidance is missing {key!r}")
    forbidden = {"calibration_ridge", "gate_min_pairwise_accuracy", "gate_min_relative_rmse_gain", "gate_min_target_points", "fixed_weight"}
    present = forbidden.intersection(guidance)
    if present:
        raise ValueError(f"calibration/gate guidance settings are not permitted: {sorted(present)}")
    if guidance["aggregation"] not in {"max", "weighted_sum"}:
        raise ValueError("guidance aggregation must be max or weighted_sum")
    LocalStructureConfig(**dict(config["extraction"]))

    stage = str(config["stage_id"])
    quick = stage.endswith("quick")
    expected = {
        "problems": {"GMM", "Ackley"} if quick else {"GMM", "Rastrigin", "Lunacek", "Ackley"},
        "dimensions": {2} if quick else {2, 5},
        "seeds": 2 if quick else 8,
        "source_samples": 80 if quick else 160,
        "raw_pool_size": 300 if quick else 1500,
        "proposal_size": 40 if quick else 100,
        "budget": 5 if quick else 20,
        "bootstrap_samples": 500 if quick else 5000,
    }
    observed = {
        "problems": set(map(str, study["problems"])),
        "dimensions": set(map(int, study["dimensions"])),
        "seeds": len(study["seeds"]),
        "source_samples": int(study["source_samples"]),
        "raw_pool_size": int(study["raw_pool_size"]),
        "proposal_size": int(study["proposal_size"]),
        "budget": int(study["budget"]),
        "bootstrap_samples": int(config["analysis"]["bootstrap_samples"]),
    }
    for key, value in expected.items():
        if observed[key] != value:
            raise ValueError(f"frozen {stage} setting {key}={observed[key]!r}, expected {value!r}")
    if quick and str(config["output_dir"]) != "results/unified_local_guidance_quick":
        raise ValueError("quick output_dir is frozen")
    if not quick and str(config["output_dir"]) != "results/unified_local_guidance_full":
        raise ValueError("full output_dir is frozen to results/unified_local_guidance_full")


def _stable_problem_code(problem: str) -> int:
    return int.from_bytes(hashlib.sha256(problem.encode("utf-8")).digest()[:4], "little")


def _array_hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(array, dtype="<f8"))
    header = json.dumps({"shape": value.shape, "dtype": str(value.dtype)}, sort_keys=True).encode()
    return hashlib.sha256(header + value.tobytes()).hexdigest()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _identity_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _rounded_keys(points: np.ndarray, decimals: int = 12) -> set[Tuple[float, ...]]:
    return {tuple(row) for row in np.round(np.asarray(points, dtype=float), decimals)}


def assert_unique_nonoverlap(points: np.ndarray, excluded: Sequence[np.ndarray], decimals: int = 12) -> None:
    rounded = np.round(np.asarray(points, dtype=float), decimals)
    if len(np.unique(rounded, axis=0)) != len(rounded):
        raise AssertionError("candidate pool contains duplicate points")
    forbidden: set[Tuple[float, ...]] = set()
    for data in excluded:
        forbidden.update(_rounded_keys(data, decimals))
    if _rounded_keys(rounded, decimals).intersection(forbidden):
        raise AssertionError("candidate pool overlaps target/source points")


def generate_unique_pool(bounds: np.ndarray, size: int, rng: np.random.Generator,
                         excluded: Sequence[np.ndarray]) -> np.ndarray:
    bounds = np.asarray(bounds, dtype=float)
    forbidden: set[Tuple[float, ...]] = set()
    for data in excluded:
        forbidden.update(_rounded_keys(data))
    accepted: List[np.ndarray] = []
    accepted_keys: set[Tuple[float, ...]] = set()
    for _ in range(max(10000, int(size) * 100)):
        if len(accepted) >= int(size):
            break
        point = rng.uniform(bounds[:, 0], bounds[:, 1])
        key = tuple(np.round(point, 12))
        if key in forbidden or key in accepted_keys:
            continue
        accepted.append(point)
        accepted_keys.add(key)
    if len(accepted) != int(size):
        raise RuntimeError("unable to construct a unique non-overlapping candidate pool")
    result = np.asarray(accepted, dtype=float)
    assert_unique_nonoverlap(result, excluded)
    return result


def make_ei_proposal(gp: TargetGPSurrogate, raw_pool: np.ndarray,
                     proposal_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    acquisition = np.asarray(gp.compute_acquisition(raw_pool, acq_type="ei"), dtype=float).reshape(-1)
    if len(acquisition) != len(raw_pool) or not np.all(np.isfinite(acquisition)):
        raise RuntimeError("target GP returned invalid EI scores")
    indices = np.argsort(-acquisition, kind="stable")[:int(proposal_size)]
    return raw_pool[indices].copy(), acquisition[indices].copy(), indices.astype(int)


def make_extraction_config(config: Mapping[str, Any], seed: int) -> LocalStructureConfig:
    values = dict(config["extraction"])
    values["random_state"] = int(seed)
    return LocalStructureConfig(**values)


def _rank_candidates(method: str, candidates: np.ndarray, acquisition: np.ndarray,
                     library: SourceLocalStructureLibrary,
                     guidance: Mapping[str, Any]) -> LocalStructureGuidanceDecision:
    return rank_local_structure_candidates(
        candidates=candidates,
        acquisition_scores=acquisition,
        library=library,
        mode=METHOD_MODES[method],
        source_weight=float(guidance["source_weight"]),
        target_nomination_ratio=float(guidance["target_nomination_ratio"]),
        source_nomination_ratio=float(guidance["source_nomination_ratio"]),
        aggregation=str(guidance["aggregation"]),
        n_points=1,
    )


def _mechanism_rows(instance: Mapping[str, Any], target: Any, init_X: np.ndarray,
                    init_y: np.ndarray, bounds: np.ndarray, source_X: np.ndarray,
                    library: SourceLocalStructureLibrary, config: Mapping[str, Any],
                    rng: np.random.Generator) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build one shared mechanism panel and one result row per method."""
    study, guidance = config["study"], config["guidance"]
    gp = TargetGPSurrogate(dim=bounds.shape[0], noise_level=1e-4,
                           random_state=int(rng.integers(0, 2**31 - 1)))
    gp.fit(init_X, init_y)
    raw_pool = generate_unique_pool(bounds, int(study["raw_pool_size"]), rng, [init_X, source_X])
    proposal, acquisition, raw_indices = make_ei_proposal(gp, raw_pool, int(study["proposal_size"]))
    truth = np.asarray(target(proposal), dtype=float).reshape(-1)
    truth_min = float(np.min(truth))
    truth_q90 = float(np.quantile(truth, 0.90))
    regret_scale = max(truth_q90 - truth_min, 1e-12)
    tol = float(study.get("ranking_tolerance", 1e-12))
    top10_cutoff = float(np.quantile(truth, 0.10)) + tol
    raw_pool_hash, proposal_hash, truth_hash = _array_hash(raw_pool), _array_hash(proposal), _array_hash(truth)

    # Compute every guidance representation once from the same library.  The
    # panel is an offline truth reveal, not training data for any target GP.
    decisions = {
        method: _rank_candidates(method, proposal, acquisition, library, guidance)
        for method in ALL_MECHANISM_METHODS
    }
    panel: List[Dict[str, Any]] = []
    panel_scores = {
        "geometry": decisions["Geometry-Only"],
        "local_rank_no_reliability": decisions["Local-Rank-No-Reliability"],
        "local_rank_reliability": decisions["Local-Rank+Reliability"],
        "reversed_local_rank": decisions[SAFETY_METHOD],
    }
    for i, point in enumerate(proposal):
        record = {
            **instance, "candidate_index": i, "x": json.dumps(point.tolist()),
            "acquisition": float(acquisition[i]), "truth": float(truth[i]),
            "truth_reveal": "offline_mechanism_reveal",
            "raw_pool_index": int(raw_indices[i]), "raw_pool_hash": raw_pool_hash,
            "proposal_hash": proposal_hash, "truth_hash": truth_hash,
            "candidate_count": len(proposal), "pool_duplicate_count": 0,
            "target_source_overlap_count": 0,
        }
        for mode, decision in panel_scores.items():
            # Raw and rank-normalized scores make every method fully
            # reconstructable from the common panel and one shared library.
            record[f"{mode}_score"] = float(decision.source_scores[i])
            record[f"{mode}_score_normalized"] = float(decision.normalized_source_scores[i])
        panel.append(record)

    rows: List[Dict[str, Any]] = []
    for method, decision in decisions.items():
        index = int(decision.selected_index)
        selected_y = float(truth[index])
        metrics = mechanism_metrics(truth, index, tolerance=tol)
        rows.append({
            **instance, "method": method, "mode": METHOD_MODES[method],
            "selected_index": index, "selected_x": json.dumps(proposal[index].tolist()),
            "selected_y": selected_y, **metrics,
            # Ranks are explicitly zero-based: 0 is best.
            "acquisition_rank": int(np.where(np.argsort(-acquisition, kind="stable") == index)[0][0]),
            "source_score": float(decision.source_scores[index]),
            "normalized_source_score": float(decision.normalized_source_scores[index]),
            "normalized_target_score": float(decision.normalized_target_scores[index]),
            "combined_score": float(decision.scores[index]),
            "target_top1_retained": bool(decision.shortlist_mask[int(np.argmax(acquisition))]),
            "best_region_id": decision.best_region_id or "",
            "fallback": bool(decision.fallback), "fallback_reason": decision.fallback_reason or "",
            "raw_pool_size": len(raw_pool), "proposal_size": len(proposal),
            "candidate_count": len(proposal), "pool_duplicate_count": 0,
            "target_source_overlap_count": 0, "raw_pool_hash": raw_pool_hash,
            "proposal_hash": proposal_hash, "truth_hash": truth_hash,
            "source_library_hash": _library_hash(library),
        })
    return rows, panel


def mechanism_metrics(truth: np.ndarray, selected_index: int,
                      tolerance: float = 1e-12) -> Dict[str, Any]:
    """Compute the frozen mechanism regret/rank fields (all ranks are 0-based)."""
    values = np.asarray(truth, dtype=float).reshape(-1)
    if len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("truth must be a non-empty finite vector")
    index = int(selected_index)
    if not 0 <= index < len(values):
        raise IndexError("selected_index is outside truth")
    minimum = float(np.min(values))
    q90 = float(np.quantile(values, 0.90))
    raw_regret = float(values[index] - minimum)
    scale = max(q90 - minimum, 1e-12)
    top10_cutoff = float(np.quantile(values, 0.10)) + float(tolerance)
    return {
        "truth_min": minimum,
        "truth_q90": q90,
        "raw_regret": raw_regret,
        "normalized_regret": raw_regret / scale,
        "top10_hit": bool(values[index] <= top10_cutoff),
        "true_rank": int(np.where(np.argsort(values, kind="stable") == index)[0][0]),
    }


def _library_hash(library: SourceLocalStructureLibrary) -> str:
    return _json_hash(library.records())


def _source_diagnostic_rows(instance: Mapping[str, Any], library: SourceLocalStructureLibrary,
                            source_hash: str) -> List[Dict[str, Any]]:
    rows = []
    for record in library.records():
        rows.append({**instance, "source_data_hash": source_hash, **record,
                     "source_structure_count": len(library.structures),
                     "source_library_hash": _library_hash(library)})
    return rows


def run_study(config: Mapping[str, Any], output_dir: Path | str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    validate_config(config)
    output = Path(output_dir)
    if not output.is_absolute():
        output = (REPO_ROOT / output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    study = config["study"]
    results: List[Dict[str, Any]] = []
    mechanism: List[Dict[str, Any]] = []
    panels: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    source_diagnostics: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for dim in map(int, study["dimensions"]):
        for problem in map(str, study["problems"]):
            for seed in map(int, study["seeds"]):
                instance = {"problem": problem, "dim": dim, "seed": seed}
                try:
                    root = np.random.SeedSequence([seed, dim, _stable_problem_code(problem)])
                    task_ss, matching_ss, init_ss, extractor_ss, mechanism_ss, sequential_ss, noise_ss = root.spawn(7)
                    suite = get_task_suite(dim=dim, rng=np.random.default_rng(task_ss))[problem]
                    target = suite["target"]
                    bounds = np.asarray(suite["bounds"], dtype=float)
                    source = suite["matching_sources"][0]
                    source_rng = np.random.default_rng(matching_ss)
                    source_X = latin_hypercube_sample(bounds, int(study["source_samples"]),
                                                      int(source_rng.integers(0, 2**31 - 1)))
                    source_y = np.asarray(source(source_X), dtype=float).reshape(-1)
                    extractor_rng = np.random.default_rng(extractor_ss)
                    library = SourceLocalStructureExtractor(
                        make_extraction_config(config, int(extractor_rng.integers(0, 2**31 - 1)))
                    ).fit_dataset(source_X, source_y, task_id=f"matching_{problem}_{dim}_{seed}")
                    source_hash = _array_hash(np.column_stack([source_X, source_y]))
                    source_diagnostics.extend(_source_diagnostic_rows(instance, library, source_hash))
                    n_init = 2 * dim + 2
                    init_rng = np.random.default_rng(init_ss)
                    init_X = init_rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_init, dim))
                    init_y = np.asarray(target(init_X), dtype=float).reshape(-1)
                    mech_rows, panel_rows = _mechanism_rows(
                        instance, target, init_X, init_y, bounds, source_X, library,
                        config, np.random.default_rng(mechanism_ss)
                    )
                    mechanism.extend(mech_rows)
                    panels.extend(panel_rows)
                    optimum = float(known_optimum_value(target))
                    for method_index, method in enumerate(MAIN_METHODS):
                        X, y = init_X.copy(), init_y.copy()
                        initial_best = float(np.min(y))
                        method_traces = [{**instance, "method": method, "step": 0,
                                          "best_y": initial_best, "selected_y": np.nan,
                                          "known_optimum_y": optimum,
                                          "normalized_regret": max(0.0, initial_best - optimum) /
                                          max(1e-12, initial_best - optimum)}]
                        # The same preallocated step seed is used by each method;
                        # their different histories may consequently produce
                        # different proposals.  No cross-method candidate hash is
                        # asserted or claimed in this phase.
                        for step in range(1, int(study["budget"]) + 1):
                            step_ss = np.random.SeedSequence([seed, dim, _stable_problem_code(problem), step, 20260901])
                            step_rng = np.random.default_rng(step_ss)
                            gp = TargetGPSurrogate(dim=dim, noise_level=1e-4,
                                                   random_state=int(step_rng.integers(0, 2**31 - 1)))
                            gp.fit(X, y)
                            raw_pool = generate_unique_pool(bounds, int(study["raw_pool_size"]), step_rng, [X, source_X])
                            proposal, acquisition, raw_indices = make_ei_proposal(gp, raw_pool, int(study["proposal_size"]))
                            decision = _rank_candidates(method, proposal, acquisition, library, config["guidance"])
                            selected_index = int(decision.selected_index)
                            selected_X = proposal[selected_index].reshape(1, -1)
                            clean_truth = float(np.asarray(target(selected_X)).reshape(-1)[0])
                            observed_truth = clean_truth
                            if float(study.get("target_noise_std", 0.0)) > 0:
                                observed_truth += float(np.random.default_rng(noise_ss).normal(0.0, float(study["target_noise_std"])))
                            X = np.vstack([X, selected_X])
                            y = np.concatenate([y, [observed_truth]])
                            previous_best = float(np.min(y[:-1]))
                            best_y = min(previous_best, clean_truth)
                            regret_scale = max(1e-12, initial_best - optimum)
                            normalized_regret = max(0.0, best_y - optimum) / regret_scale
                            pool_hash, proposal_hash = _array_hash(raw_pool), _array_hash(proposal)
                            result_row = {**instance, "step": step, "method": method,
                                          "mode": METHOD_MODES[method], "selected_index": selected_index,
                                          "selected_x": json.dumps(selected_X[0].tolist()),
                                          "selected_y": clean_truth, "observed_y": observed_truth,
                                          "best_y": best_y, "initial_best_y": initial_best,
                                          "known_optimum_y": optimum, "raw_regret": best_y - optimum,
                                          "normalized_regret": normalized_regret,
                                          "acquisition": float(acquisition[selected_index]),
                                          "source_score": float(decision.source_scores[selected_index]),
                                          "normalized_source_score": float(decision.normalized_source_scores[selected_index]),
                                          "normalized_target_score": float(decision.normalized_target_scores[selected_index]),
                                          "combined_score": float(decision.scores[selected_index]),
                                          "fallback": bool(decision.fallback), "fallback_reason": decision.fallback_reason or "",
                                          "raw_pool_size": len(raw_pool), "proposal_size": len(proposal),
                                          "candidate_count": len(proposal), "pool_duplicate_count": 0,
                                          "target_source_overlap_count": 0, "raw_pool_hash": pool_hash,
                                          "proposal_hash": proposal_hash, "source_library_hash": _library_hash(library),
                                          "step_seed": int(step_ss.generate_state(1, dtype=np.uint64)[0])}
                            results.append(result_row)
                            method_traces.append({**instance, "method": method, "step": step,
                                                  "best_y": best_y, "selected_y": clean_truth,
                                                  "known_optimum_y": optimum,
                                                  "normalized_regret": normalized_regret,
                                                  "raw_pool_hash": pool_hash, "proposal_hash": proposal_hash})
                        traces.extend(method_traces)
                except Exception as exc:
                    failures.append({**instance, "error_type": type(exc).__name__, "error": str(exc)})

    result_columns = ["problem", "dim", "seed", "step", "method", "mode", "selected_index", "selected_x", "selected_y", "observed_y", "best_y", "initial_best_y", "known_optimum_y", "raw_regret", "normalized_regret", "acquisition", "source_score", "normalized_source_score", "normalized_target_score", "combined_score", "fallback", "fallback_reason", "raw_pool_size", "proposal_size", "candidate_count", "pool_duplicate_count", "target_source_overlap_count", "raw_pool_hash", "proposal_hash", "source_library_hash", "step_seed"]
    mechanism_columns = ["problem", "dim", "seed", "method", "mode", "selected_index", "selected_x", "selected_y", "truth_min", "truth_q90", "raw_regret", "normalized_regret", "top10_hit", "true_rank", "acquisition_rank", "source_score", "normalized_source_score", "normalized_target_score", "combined_score", "target_top1_retained", "best_region_id", "fallback", "fallback_reason", "raw_pool_size", "proposal_size", "candidate_count", "pool_duplicate_count", "target_source_overlap_count", "raw_pool_hash", "proposal_hash", "truth_hash", "source_library_hash"]
    trace_columns = ["problem", "dim", "seed", "method", "step", "best_y", "selected_y", "known_optimum_y", "normalized_regret", "raw_pool_hash", "proposal_hash"]
    panel_columns = ["problem", "dim", "seed", "candidate_index", "x", "acquisition", "truth", "truth_reveal", "raw_pool_index", "raw_pool_hash", "proposal_hash", "truth_hash", "candidate_count", "pool_duplicate_count", "target_source_overlap_count", "geometry_score", "geometry_score_normalized", "local_rank_no_reliability_score", "local_rank_no_reliability_score_normalized", "local_rank_reliability_score", "local_rank_reliability_score_normalized", "reversed_local_rank_score", "reversed_local_rank_score_normalized"]
    source_columns = ["problem", "dim", "seed", "source_data_hash", "source_library_hash", "source_structure_count", "task_id", "region_id", "center", "covariance", "region_quality", "core_count", "context_count", "bic", "model_type", "oof_spearman", "oof_ndcg", "oof_precision_at_top", "geometry_spearman", "geometry_ndcg", "reliability", "boundary_fraction"]
    failure_columns = ["problem", "dim", "seed", "error_type", "error"]
    result_frame = pd.DataFrame(results, columns=result_columns)
    trace_frame = pd.DataFrame(traces, columns=trace_columns)
    panel_frame = pd.DataFrame(panels, columns=panel_columns)
    mechanism_frame = pd.DataFrame(mechanism, columns=mechanism_columns)
    source_frame = pd.DataFrame(source_diagnostics, columns=source_columns)
    failure_frame = pd.DataFrame(failures, columns=failure_columns)

    paths = {
        "mechanism_results": output / "mechanism_results.csv",
        "mechanism_candidate_panel": output / "mechanism_candidate_panel.csv",
        "sequential_summary": output / "sequential_summary.csv",
        "sequential_traces": output / "sequential_traces.csv",
        "source_structure_diagnostics": output / "source_structure_diagnostics.csv",
        "failures": output / "failures.csv",
        "config": output / "config.json",
    }
    mechanism_frame.to_csv(paths["mechanism_results"], index=False)
    panel_frame.to_csv(paths["mechanism_candidate_panel"], index=False)
    summary_rows: List[Dict[str, Any]] = []
    for key, group in trace_frame.groupby(["problem", "dim", "seed", "method"], sort=False):
        # Include step 0 in both the AUC and the explicit trace length.
        group = group.sort_values("step")
        if len(group) != int(study["budget"]) + 1 or int(group.iloc[0]["step"]) != 0:
            raise RuntimeError("sequential trace must contain exactly step 0..budget")
        summary_rows.append({"problem": key[0], "dim": key[1], "seed": key[2], "method": key[3],
                             "initial_best_y": float(group.iloc[0]["best_y"]),
                             "final_best_y": float(group.iloc[-1]["best_y"]), "known_optimum_y": float(group.iloc[-1]["known_optimum_y"]),
                             "final_normalized_regret": float(group.iloc[-1]["normalized_regret"]),
                             "auc_normalized_regret": float(np.trapezoid(group["normalized_regret"], group["step"]) if hasattr(np, "trapezoid") else np.trapz(group["normalized_regret"], group["step"])),
                             "total_improvement": float(group.iloc[0]["best_y"] - group.iloc[-1]["best_y"]),
                             "trace_points": len(group)})
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(paths["sequential_summary"], index=False)
    trace_frame.to_csv(paths["sequential_traces"], index=False)
    source_frame.to_csv(paths["source_structure_diagnostics"], index=False)
    failure_frame.to_csv(paths["failures"], index=False)
    paths["config"].write_text(json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    _write_manifest(output / "run_manifest.json", config, paths,
                    {"instances": len(study["problems"]) * len(study["dimensions"]) * len(study["seeds"]),
                     "mechanism_rows": len(mechanism_frame), "mechanism_candidate_rows": len(panel_frame),
                     "sequential_summary_rows": len(summary_frame), "sequential_trace_rows": len(trace_frame),
                     "source_structure_rows": len(source_frame), "failure_rows": len(failure_frame)})
    return summary_frame, trace_frame


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata() -> Dict[str, Any]:
    def run(args: List[str]) -> str:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    try:
        status = run(["status", "--porcelain"])
        return {"head": run(["rev-parse", "HEAD"]), "status_porcelain": status, "dirty": bool(status)}
    except Exception as exc:
        return {"head": "unavailable", "status_porcelain": str(exc), "dirty": "unknown"}


def _write_manifest(path: Path, config: Mapping[str, Any], artifacts: Mapping[str, Path], counts: Mapping[str, Any]) -> None:
    companion = REPO_ROOT / "src" / "region_guided_reranking_study" / "local_structure_guidance.py"
    payload = {"protocol_id": config["protocol_id"], "stage_id": config["stage_id"], "config": config,
               "config_sha256": _json_hash(config), "protocol_path": _identity_path(PROTOCOL_PATH),
               "protocol_sha256": _file_sha256(PROTOCOL_PATH), "runner_path": _identity_path(RUNNER_PATH),
               "runner_sha256": _file_sha256(RUNNER_PATH), "companion_path": _identity_path(companion),
               "companion_sha256": _file_sha256(companion) if companion.exists() else "not-present",
               "code_identity": {"runner": _identity_path(RUNNER_PATH), "companion": _identity_path(companion), "protocol": _identity_path(PROTOCOL_PATH)},
               "counts": dict(counts), "artifact_sha256": {_identity_path(p): _file_sha256(p) for p in artifacts.values()},
               "git": _git_metadata(), "python": sys.version, "platform": platform.platform(),
               "packages": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__}}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified local-guidance study")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "unified_local_guidance_quick.json")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_study(config, args.output or (REPO_ROOT / config["output_dir"]))


if __name__ == "__main__":
    main()
