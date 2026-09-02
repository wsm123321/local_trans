"""Pilot runner for controlled oracle benchmark local-model transfer.

Evaluates 2D benchmark landscapes (GMM, Rastrigin, Lunacek, Ackley) across
conditions (matching, reversed, label_permutation), context budgets (e.g. 6, 12, 20),
and radial shells with 5 transfer methods (Target-Only, Geometry, Rank, Value, Dual).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy
import sklearn

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from region_guided_reranking_study.local_surrogate_transfer_research import (  # noqa: E402
    evaluate_predictions,
    pairwise_cost_accuracy,
)
from region_guided_reranking_study.oracle_benchmark_transfer import (  # noqa: E402
    BENCHMARK_CONDITIONS,
    BENCHMARK_METHOD_MODES,
    BENCHMARK_METHODS,
    BENCHMARK_PROBLEMS,
    BenchmarkLandscapePair,
    compute_physical_chart_radius,
    create_benchmark_pair,
    derive_benchmark_seed,
    generate_unit_ball_points,
    generate_unit_sphere_directions,
    partition_context_points,
)
from region_guided_reranking_study.oracle_local_model_transfer import (  # noqa: E402
    OracleLocalModelTransfer,
    OracleLocalModelTransferConfig,
    fit_source_oracle_expert,
    geometry_prior_from_chart,
)

Array = np.ndarray

DEFAULT_PILOT_CONFIG = {
    "stage_id": "oracle-benchmark-transfer-pilot-v1",
    "scope": "benchmark_oracle_local_model_transfer_pilot",
    "dimension": 2,
    "seeds": [11, 23, 37, 53, 71, 89, 107, 131],
    "problems": list(BENCHMARK_PROBLEMS),
    "conditions": list(BENCHMARK_CONDITIONS),
    "context_sample_sizes": [6, 12, 20],
    "shells": [0.35, 0.7, 1.0],
    "methods": list(BENCHMARK_METHODS),
    "source_train_samples": 128,
    "target_test_samples": 512,
    "chart_radius_fraction": 0.04,
    "source_expert": {"length_scale": 0.45, "noise": 0.0001},
    "transfer_model": {"gp_length_scale": 0.6, "gp_noise": 0.0001, "calibration_ridge": 1.0, "fixed_prior_scale": 1.0},
    "top_fraction": 0.1,
    "harm_margin_srmse": 0.01,
}


def load_config(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a JSON object")
    return config


def validate_and_normalize_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate and normalize configuration dictionary."""
    normalized = dict(config)

    # Normalize context sample sizes: context_sample_sizes / target_context_samples / context_sizes
    if "context_sample_sizes" in normalized:
        val = normalized["context_sample_sizes"]
        normalized["context_sample_sizes"] = [int(v) for v in val] if isinstance(val, (list, tuple)) else [int(val)]
    elif "target_context_samples" in normalized:
        val = normalized["target_context_samples"]
        normalized["context_sample_sizes"] = [int(v) for v in val] if isinstance(val, (list, tuple)) else [int(val)]
    elif "context_sizes" in normalized:
        val = normalized["context_sizes"]
        normalized["context_sample_sizes"] = [int(v) for v in val] if isinstance(val, (list, tuple)) else [int(val)]
    else:
        normalized["context_sample_sizes"] = [6, 12, 20]

    # Normalize conditions: conditions / relations
    if "conditions" in normalized:
        normalized["conditions"] = list(normalized["conditions"])
    elif "relations" in normalized:
        normalized["conditions"] = list(normalized["relations"])
    else:
        normalized["conditions"] = list(BENCHMARK_CONDITIONS)

    # Normalize source train samples: source_train_samples / source_samples
    if "source_train_samples" in normalized:
        normalized["source_train_samples"] = int(normalized["source_train_samples"])
    elif "source_samples" in normalized:
        normalized["source_train_samples"] = int(normalized["source_samples"])
    else:
        normalized["source_train_samples"] = 128

    # Normalize test samples: target_test_samples / test_samples / target_test_samples_per_shell
    if "target_test_samples" in normalized:
        normalized["target_test_samples"] = int(normalized["target_test_samples"])
    elif "test_samples" in normalized:
        normalized["target_test_samples"] = int(normalized["test_samples"])
    elif "target_test_samples_per_shell" in normalized:
        normalized["target_test_samples_per_shell"] = int(normalized["target_test_samples_per_shell"])
    else:
        normalized["target_test_samples"] = 512

    # Defaults for optional fields
    normalized.setdefault("dimension", 2)
    normalized.setdefault("shells", [0.35, 0.7, 1.0])
    normalized.setdefault("methods", list(BENCHMARK_METHODS))
    normalized.setdefault("problems", list(BENCHMARK_PROBLEMS))
    normalized.setdefault("seeds", [11])
    normalized.setdefault("top_fraction", 0.1)
    normalized.setdefault("harm_margin_srmse", 0.01)
    normalized.setdefault("source_expert", {"length_scale": 0.45, "noise": 0.0001})
    normalized.setdefault("transfer_model", {"gp_length_scale": 0.6, "gp_noise": 0.0001, "calibration_ridge": 1.0})
    normalized.setdefault("stage_id", "oracle-benchmark-transfer-pilot-v1")
    normalized.setdefault("scope", "benchmark_oracle_local_model_transfer_pilot")

    # Validations
    if normalized["dimension"] != 2:
        raise ValueError(f"Dimension must be 2, got {normalized['dimension']}")
    if not normalized["seeds"]:
        raise ValueError("seeds list cannot be empty")
    if not normalized["problems"]:
        raise ValueError("problems list cannot be empty")
    for problem in normalized["problems"]:
        if problem not in BENCHMARK_PROBLEMS:
            raise ValueError(f"Unsupported problem: {problem}. Allowed: {BENCHMARK_PROBLEMS}")
    for condition in normalized["conditions"]:
        if condition not in BENCHMARK_CONDITIONS:
            raise ValueError(f"Unsupported condition: {condition}. Allowed: {BENCHMARK_CONDITIONS}")
    for method in normalized["methods"]:
        if method not in BENCHMARK_METHODS:
            raise ValueError(f"Unsupported method: {method}. Allowed: {BENCHMARK_METHODS}")
    if not normalized["shells"]:
        raise ValueError("shells list cannot be empty")
    for shell in normalized["shells"]:
        if float(shell) <= 0:
            raise ValueError(f"Shell radius must be positive, got {shell}")
    for ctx in normalized["context_sample_sizes"]:
        if int(ctx) < len(normalized["shells"]):
            raise ValueError(
                f"Context sample size ({ctx}) must be >= number of shells ({len(normalized['shells'])})"
            )

    return normalized


def _assert_disjoint(first: Array, second: Array) -> None:
    left = {tuple(row) for row in np.round(np.asarray(first), 12)}
    right = {tuple(row) for row in np.round(np.asarray(second), 12)}
    if left.intersection(right):
        raise AssertionError("Target context and target test panels overlap")


def _array_hash(array: Array) -> str:
    value = np.ascontiguousarray(np.asarray(array, dtype="<f8"))
    header = json.dumps({"shape": value.shape, "dtype": str(value.dtype)}, sort_keys=True).encode()
    return hashlib.sha256(header + value.tobytes()).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _robust_standardize(values: Array) -> Array:
    values = np.asarray(values, dtype=float).reshape(-1)
    center = float(np.median(values))
    scale = 1.4826 * float(np.median(np.abs(values - center)))
    if scale <= 1e-12:
        q25, q75 = np.quantile(values, [0.25, 0.75])
        scale = float((q75 - q25) / 1.349)
    if scale <= 1e-12:
        scale = float(np.std(values, ddof=0))
    return (values - center) / max(scale, 1e-12)


def _oob(points: Array) -> Tuple[int, float]:
    values = np.asarray(points, dtype=float)
    mask = np.any((values < -1.0) | (values > 1.0), axis=1)
    return int(np.sum(mask)), float(np.mean(mask))


def _git_metadata() -> Dict[str, Any]:
    def run(args: Sequence[str]) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    try:
        status = run(["status", "--porcelain"])
        return {
            "head": run(["rev-parse", "HEAD"]),
            "status_porcelain": status,
            "dirty": bool(status),
        }
    except Exception as exc:
        return {"head": "unavailable", "status_porcelain": str(exc), "dirty": "unknown"}


def _resolve(path: Path) -> Path:
    return Path(path).resolve() if Path(path).is_absolute() else (REPO_ROOT / path).resolve()


def _fit_transfer_models(
    context: Array,
    context_y: Array,
    test: Array,
    source_rank_context: Array,
    source_rank_test: Array,
    source_value_context: Array,
    source_value_test: Array,
    config: Mapping[str, Any],
    model_seed: int,
) -> Dict[str, Dict[str, Any]]:
    transfer = dict(config["transfer_model"])
    allowed = {"gp_length_scale", "gp_noise", "calibration_ridge", "random_state"}
    filtered_transfer = {k: v for k, v in transfer.items() if k in allowed}
    filtered_transfer["random_state"] = model_seed
    model_config = OracleLocalModelTransferConfig(**filtered_transfer)
    priors_context = {
        "geometry_prior": geometry_prior_from_chart(context),
        "oracle_rank": source_rank_context,
        "oracle_value": source_value_context,
    }
    priors_test = {
        "geometry_prior": geometry_prior_from_chart(test),
        "oracle_rank": source_rank_test,
        "oracle_value": source_value_test,
    }
    outputs: Dict[str, Dict[str, Any]] = {}
    for method in config["methods"]:
        mode = BENCHMARK_METHOD_MODES[method]
        model = OracleLocalModelTransfer(mode, model_config).fit(
            context, context_y, **priors_context
        )
        mean, std = model.predict(test, return_std=True, **priors_test)
        evidence = model.evidence_
        outputs[method] = {
            "mean": mean,
            "std": std,
            "effective_mode": model.effective_mode_,
            "prior_names": "|".join(evidence.prior_names) if evidence else "none",
            "prior_coefficients": json.dumps(evidence.coefficients) if evidence else "[]",
        }
    return outputs


def run_benchmark_pilot(config: Mapping[str, Any], output_dir: Path) -> pd.DataFrame:
    """Run oracle benchmark local-model transfer pilot study and write all artifacts.

    Refuses non-empty output directory to ensure immutable provenance.
    """
    output_dir = _resolve(output_dir)
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"Output path exists and is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise FileExistsError(f"Refusing non-empty output directory: {output_dir}")
    else:
        output_dir.mkdir(parents=True)

    normalized_config = validate_and_normalize_config(config)

    seeds = [int(v) for v in normalized_config["seeds"]]
    problems = [str(v) for v in normalized_config["problems"]]
    conditions = [str(v) for v in normalized_config["conditions"]]
    context_sample_sizes = [int(v) for v in normalized_config["context_sample_sizes"]]
    shells = [float(v) for v in normalized_config["shells"]]
    methods = [str(v) for v in normalized_config["methods"]]
    n_source = int(normalized_config["source_train_samples"])
    chart_radius = normalized_config.get("chart_radius", None)
    chart_radius_fraction = normalized_config.get("chart_radius_fraction", None)

    # Shell test counts
    K = len(shells)
    if "target_test_samples" in normalized_config:
        total_test = int(normalized_config["target_test_samples"])
        shell_test_counts = [total_test // K + (1 if i < (total_test % K) else 0) for i in range(K)]
    else:
        per_shell = int(normalized_config.get("target_test_samples_per_shell", 128))
        shell_test_counts = [per_shell for _ in range(K)]

    rows: List[Dict[str, Any]] = []
    ledger: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    lineage_records: List[List[int]] = []
    reproducibility: Dict[str, Array] = {
        "schema_version": np.asarray([1], dtype=np.int64),
        "seed_values": np.asarray(seeds, dtype=np.int64),
        "shell_values": np.asarray(shells, dtype=np.float64),
        "problem_names": np.asarray(problems, dtype="U32"),
        "condition_names": np.asarray(conditions, dtype="U32"),
        "context_sample_values": np.asarray(context_sample_sizes, dtype=np.int64),
    }

    max_n_context = max(context_sample_sizes)

    for seed in seeds:
        for problem in problems:
            instance_info = {
                "seed": seed,
                "dimension": 2,
                "problem": problem,
            }
            try:
                landscape_seed = derive_benchmark_seed(seed, f"{problem}_landscape")
                source_design_seed = derive_benchmark_seed(seed, f"{problem}_source_design")
                source_expert_seed = derive_benchmark_seed(seed, f"{problem}_source_expert")
                context_design_seed = derive_benchmark_seed(seed, f"{problem}_context_design")
                permutation_seed = derive_benchmark_seed(seed, f"{problem}_permutation")

                pair = create_benchmark_pair(
                    problem=problem,
                    seed=landscape_seed,
                    dim=2,
                    chart_radius=chart_radius,
                    chart_radius_fraction=chart_radius_fraction,
                )

                # Source points in unit ball via 2D uniform disk Sobol
                source_dirs = generate_unit_ball_points(n_source, seed=source_design_seed, dim=2)
                context_dirs = generate_unit_sphere_directions(max_n_context, seed=context_design_seed, dim=2)

                # Generate per-shell test points
                tests = {}
                test_dirs_list = []
                for i, (shell, n_test_shell) in enumerate(zip(shells, shell_test_counts)):
                    test_design_seed = derive_benchmark_seed(seed, f"{problem}_test_design_shell_{i}")
                    shell_dirs = generate_unit_sphere_directions(n_test_shell, seed=test_design_seed, dim=2)
                    tests[shell] = shell_dirs * shell
                    test_dirs_list.append(shell_dirs)

                test_hashes = {str(shell): _array_hash(tests[shell]) for shell in shells}

                permutation_rng = np.random.default_rng(permutation_seed)
                permutation = permutation_rng.permutation(n_source).astype(np.int64)

                n_conditions = len(conditions)
                n_model_evals = n_conditions * len(context_sample_sizes) * len(shells)
                model_seed_values = [
                    derive_benchmark_seed(seed, f"{problem}_model_{idx}")
                    for idx in range(n_model_evals)
                ]
                lineage_records.append([
                    seed,
                    landscape_seed,
                    source_design_seed,
                    source_expert_seed,
                    context_design_seed,
                    permutation_seed,
                    *model_seed_values,
                ])

                reproducibility[f"seed_{seed}_{problem}_source_dirs"] = np.asarray(source_dirs, dtype=np.float64)
                reproducibility[f"seed_{seed}_{problem}_context_dirs"] = np.asarray(context_dirs, dtype=np.float64)
                reproducibility[f"seed_{seed}_{problem}_permutation"] = permutation
                reproducibility[f"seed_{seed}_{problem}_target_anchor"] = np.asarray(pair.target_anchor, dtype=np.float64)
                reproducibility[f"seed_{seed}_{problem}_source_anchor"] = np.asarray(pair.source_anchor, dtype=np.float64)
                reproducibility[f"seed_{seed}_{problem}_chart_radius"] = np.asarray([pair.chart_radius], dtype=np.float64)

                expert_model_config = OracleLocalModelTransferConfig(
                    gp_length_scale=float(normalized_config["source_expert"]["length_scale"]),
                    gp_noise=float(normalized_config["source_expert"]["noise"]),
                    calibration_ridge=float(normalized_config["transfer_model"].get("calibration_ridge", 1.0)),
                    random_state=source_expert_seed,
                )

                model_seed_index = 0

                for condition in conditions:
                    source_y_raw = pair.evaluate_source(source_dirs, condition="matching")
                    reproducibility[f"seed_{seed}_{problem}_source_y_raw"] = np.asarray(source_y_raw, dtype=np.float64)

                    if condition == "reversed":
                        train_source_y = -source_y_raw
                    elif condition == "label_permutation":
                        train_source_y = source_y_raw[permutation]
                    else:
                        train_source_y = source_y_raw

                    reproducibility[f"seed_{seed}_{problem}_source_y_{condition}"] = np.asarray(train_source_y, dtype=np.float64)

                    source_expert_for_condition = fit_source_oracle_expert(
                        source_dirs, train_source_y, expert_model_config, seed=source_expert_seed,
                    )
                    source_hash = _array_hash(np.column_stack([source_dirs, train_source_y]))

                    for n_context in context_sample_sizes:
                        context = partition_context_points(context_dirs, shells, n_context)
                        context_hash = _array_hash(context)

                        for test in tests.values():
                            _assert_disjoint(context, test)

                        source_context_points = pair.target_to_source_chart(context)
                        source_context_query_hash = _array_hash(source_context_points)

                        target_context_y = pair.evaluate_target(context)
                        reproducibility[f"seed_{seed}_{problem}_context_points_{n_context}"] = np.asarray(context, dtype=np.float64)
                        reproducibility[f"seed_{seed}_{problem}_target_context_truth_{condition}_{n_context}"] = np.asarray(target_context_y, dtype=np.float64)

                        source_rank_context_quality, _ = source_expert_for_condition.predict_rank(source_context_points)
                        rank_context = 1.0 - source_rank_context_quality
                        source_value_context_points, _ = source_expert_for_condition.predict(source_context_points, feature="raw_value")

                        for shell in shells:
                            test = tests[shell]
                            target_test_y = pair.evaluate_target(test)
                            source_test_points = pair.target_to_source_chart(test)
                            source_test_query_hash = _array_hash(source_test_points)

                            reproducibility[f"seed_{seed}_{problem}_target_test_points_{shell}"] = np.asarray(test, dtype=np.float64)
                            reproducibility[f"seed_{seed}_{problem}_target_test_truth_{condition}_{n_context}_{shell}"] = np.asarray(target_test_y, dtype=np.float64)
                            reproducibility[f"seed_{seed}_{problem}_source_test_query_{condition}_{n_context}_{shell}"] = np.asarray(source_test_points, dtype=np.float64)

                            source_rank_test_quality, _ = source_expert_for_condition.predict_rank(source_test_points)
                            rank_test = 1.0 - source_rank_test_quality
                            value_test_standardized, _ = source_expert_for_condition.predict(source_test_points, feature="raw_value")
                            value_test_raw = (
                                source_expert_for_condition.raw_standardizer_.inverse_transform(value_test_standardized)
                                if source_expert_for_condition.raw_standardizer_ is not None
                                else value_test_standardized
                            )
                            value_context = source_value_context_points

                            model_seed = model_seed_values[model_seed_index]
                            model_seed_index += 1

                            outputs = _fit_transfer_models(
                                context,
                                target_context_y,
                                test,
                                rank_context,
                                rank_test,
                                value_context,
                                value_test_standardized,
                                normalized_config,
                                model_seed,
                            )

                            target_srmse: Optional[float] = None
                            metric_rows: List[Dict[str, Any]] = []

                            for method in methods:
                                output = outputs[method]
                                metrics = evaluate_predictions(
                                    target_test_y,
                                    output["mean"],
                                    output["std"],
                                    top_fraction=float(normalized_config["top_fraction"]),
                                )
                                if method == "Target-Only":
                                    target_srmse = metrics.standardized_rmse

                                metric_rows.append({
                                    **instance_info,
                                    "condition": condition,
                                    "relation": condition,
                                    "context_samples": n_context,
                                    "context_size": n_context,
                                    "shell": shell,
                                    "panel": "test",
                                    "method": method,
                                    "model_seed": model_seed,
                                    **metrics.__dict__,
                                    "effective_mode": output["effective_mode"],
                                    "prior_names": output["prior_names"],
                                    "prior_coefficients": output["prior_coefficients"],
                                    "source_data_hash": source_hash,
                                    "source_query_transform": "exact_rotation" if problem == "Rastrigin" else "identity",
                                    "source_context_query_hash": source_context_query_hash,
                                    "source_test_query_hash": source_test_query_hash,
                                    "target_context_design_hash": context_hash,
                                    "target_test_design_hash": test_hashes[str(shell)],
                                    "target_test_truth_hash": _array_hash(target_test_y),
                                })

                                for candidate_index, (point, truth, mean, std) in enumerate(
                                    zip(test, target_test_y, output["mean"], output["std"])
                                ):
                                    ledger.append({
                                        **instance_info,
                                        "condition": condition,
                                        "relation": condition,
                                        "context_samples": n_context,
                                        "context_size": n_context,
                                        "shell": shell,
                                        "panel": "test",
                                        "candidate_index": candidate_index,
                                        "chart_point": json.dumps(point.tolist()),
                                        "truth": float(truth),
                                        "method": method,
                                        "model_seed": model_seed,
                                        "predicted_mean": float(mean),
                                        "predicted_std": float(std),
                                    })

                            if target_srmse is None:
                                raise RuntimeError("Target-Only metrics were not produced")

                            for record in metric_rows:
                                record["srmse_delta_vs_target_only"] = float(
                                    record["standardized_rmse"] - target_srmse
                                )
                                record["negative_transfer"] = bool(
                                    record["srmse_delta_vs_target_only"]
                                    > float(normalized_config["harm_margin_srmse"])
                                )
                                rows.append(record)

                            source_rank_test = 1.0 - source_rank_test_quality
                            target_standardized = _robust_standardize(target_test_y)
                            source_value_standardized_target_rmse = float(
                                np.sqrt(np.mean((value_test_standardized - target_standardized) ** 2))
                            )
                            context_oob_count, context_oob_rate = _oob(source_context_points)
                            test_oob_count, test_oob_rate = _oob(source_test_points)

                            diagnostic = {
                                **instance_info,
                                "condition": condition,
                                "relation": condition,
                                "context_samples": n_context,
                                "context_size": n_context,
                                "shell": shell,
                                "panel": "test",
                                "source_data_hash": source_hash,
                                "source_value_pairwise_accuracy": pairwise_cost_accuracy(value_test_raw, target_test_y),
                                "source_rank_target_agreement": pairwise_cost_accuracy(source_rank_test, target_test_y),
                                "source_value_standardized_target_rmse": source_value_standardized_target_rmse,
                                "source_value_standardized_target_rmse_definition": "source raw-value GP prediction in source standard units versus independently robust-standardized target test truth",
                                "source_query_transform": "exact_rotation" if problem == "Rastrigin" else "identity",
                                "source_context_query_hash": source_context_query_hash,
                                "source_test_query_hash": source_test_query_hash,
                                "source_context_oob_count": context_oob_count,
                                "source_context_oob_rate": context_oob_rate,
                                "source_test_oob_count": test_oob_count,
                                "source_test_oob_rate": test_oob_rate,
                                "target_context_design_hash": context_hash,
                                "target_test_design_hash": test_hashes[str(shell)],
                                "target_test_truth_hash": _array_hash(target_test_y),
                                "source_permutation_hash": _array_hash(permutation),
                                "target_anchor": json.dumps(pair.target_anchor.tolist()),
                                "source_anchor": json.dumps(pair.source_anchor.tolist()),
                                "chart_radius": float(pair.chart_radius),
                                "model_seed": model_seed,
                            }
                            diagnostics.append(diagnostic)

            except Exception as exc:
                failures.append({
                    **instance_info,
                    "condition": "all",
                    "relation": "all",
                    "context_samples": -1,
                    "context_size": -1,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })

    result_frame = pd.DataFrame(rows)
    ledger_frame = pd.DataFrame(ledger)
    diagnostics_frame = pd.DataFrame(diagnostics)
    failures_frame = pd.DataFrame(
        failures,
        columns=["seed", "dimension", "problem", "condition", "relation", "context_samples", "context_size", "error_type", "error"],
    )

    paths = {
        "results": output_dir / "results.csv",
        "prediction_ledger": output_dir / "prediction_ledger.csv",
        "source_expert_diagnostics": output_dir / "source_expert_diagnostics.csv",
        "failures": output_dir / "failures.csv",
        "config": output_dir / "config.json",
        "reproducibility_inputs": output_dir / "reproducibility_inputs.npz",
    }

    result_frame.to_csv(paths["results"], index=False)
    ledger_frame.to_csv(paths["prediction_ledger"], index=False)
    diagnostics_frame.to_csv(paths["source_expert_diagnostics"], index=False)
    failures_frame.to_csv(paths["failures"], index=False)

    config_bytes = json.dumps(normalized_config, indent=2, ensure_ascii=False).encode("utf-8")
    paths["config"].write_bytes(config_bytes)

    if lineage_records:
        reproducibility["seed_lineage"] = np.asarray(lineage_records, dtype=np.int64)

    np.savez_compressed(paths["reproducibility_inputs"], **reproducibility)

    reproducibility_shapes = {
        key: {"shape": list(np.asarray(value).shape), "dtype": str(np.asarray(value).dtype)}
        for key, value in reproducibility.items()
    }
    reproducibility_hashes = {
        key: _array_hash(value)
        for key, value in reproducibility.items()
        if "target_" in key or "source_" in key or key in {"seed_lineage", "seed_values", "shell_values"}
    }

    dependency_paths = {
        "runner": Path(__file__).resolve(),
        "oracle_benchmark_transfer": SRC_DIR / "region_guided_reranking_study" / "oracle_benchmark_transfer.py",
        "oracle_core": SRC_DIR / "region_guided_reranking_study" / "oracle_local_model_transfer.py",
        "local_surrogate_transfer_research": SRC_DIR / "region_guided_reranking_study" / "local_surrogate_transfer_research.py",
        "local_surrogate_transfer": SRC_DIR / "region_guided_reranking_study" / "local_surrogate_transfer.py",
        "landscapes": SRC_DIR / "region_guided_reranking_study" / "landscapes.py",
    }
    dependency_hashes = {
        name: {"path": str(path.relative_to(REPO_ROOT)), "sha256": _file_hash(path)}
        for name, path in dependency_paths.items()
    }

    manifest = {
        "stage_id": normalized_config["stage_id"],
        "scope": normalized_config["scope"],
        "config": normalized_config,
        "config_sha256": _file_hash(paths["config"]),
        "counts": {
            "result_rows": len(result_frame),
            "ledger_rows": len(ledger_frame),
            "diagnostic_rows": len(diagnostics_frame),
            "failure_rows": len(failures_frame),
            "seed_count": len(seeds),
            "problem_count": len(problems),
            "condition_count": len(conditions),
            "context_sample_count": len(context_sample_sizes),
            "shell_count": len(shells),
            "reproducibility_array_count": len(reproducibility),
        },
        "artifacts": {name: _file_hash(path) for name, path in paths.items()},
        "artifact_metadata": {
            "config": {"encoding": "utf-8", "indent": 2},
            "reproducibility_inputs": {
                "schema_version": 1,
                "array_count": len(reproducibility),
                "keys": sorted(reproducibility),
                "shapes": reproducibility_shapes,
                "array_hashes": reproducibility_hashes,
            },
        },
        "dependencies": dependency_hashes,
        "dependency_sha256": {name: item["sha256"] for name, item in dependency_hashes.items()},
        "lineage": {
            "columns": [
                "seed",
                "landscape_seed",
                "source_design_seed",
                "source_expert_seed",
                "context_design_seed",
                "permutation_seed",
            ] + [
                f"model_seed_{i}"
                for i in range(max(0, len(lineage_records[0]) - 6))
            ] if lineage_records else [],
            "values": lineage_records,
        },
        "git": _git_metadata(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
    }

    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return result_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run oracle benchmark local-model transfer pilot")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "oracle_benchmark_transfer_pilot.json")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results" / "oracle_benchmark_transfer_pilot")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg_path = _resolve(args.config)
    cfg = load_config(cfg_path) if cfg_path.exists() else DEFAULT_PILOT_CONFIG
    run_benchmark_pilot(cfg, args.output)
