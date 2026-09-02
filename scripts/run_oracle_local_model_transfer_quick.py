"""Gate-0 quick runner for controlled oracle local-model transfer.

This is a static held-out 2D benchmark.  It intentionally has no optimizer,
acquisition, plotting, or result reuse.  Source/context/test directions are
shared within a seed, target test truth is generated once per relation/shell,
and every random stream has an explicit integer child seed recorded in provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

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
    sobol_chart_design,
)
from region_guided_reranking_study.oracle_local_model_transfer import (  # noqa: E402
    OracleLocalModelTransfer,
    OracleLocalModelTransferConfig,
    fit_source_oracle_expert,
    geometry_prior_from_chart,
    oracle_coordinate_transform,
)

Array = np.ndarray
METHOD_MODES = {
    "Target-Only": "target_only",
    "Geometry-Prior+Residual": "geometry_prior",
    "Oracle-Rank+Residual": "oracle_rank",
    "Oracle-Value+Residual": "oracle_value",
    "Oracle-Rank+Value+Residual": "oracle_rank_value",
}
RELATIONS = (
    "identity", "output_affine", "scale_0.7", "scale_1.5", "rotate_45",
    "roughness", "reversal", "independent_expert",
)
CANONICAL_CONFIG = {
    "stage_id": "gate-0-oracle-local-model-transfer-quick-v0",
    "scope": "gate_0_controlled_2d_static_held_out",
    "dimension": 2,
    "seeds": [11, 23, 37, 53, 71, 89, 107, 131],
    "source_train_samples": 128,
    "target_context_samples": 12,
    "target_test_samples_per_shell": 128,
    "shells": [0.35, 0.7, 1.0],
    "relations": list(RELATIONS),
    "controls": ["identity_label_permutation"],
    "methods": list(METHOD_MODES),
    "source_expert": {"length_scale": 0.45, "noise": 0.0001},
    "transfer_model": {"gp_length_scale": 0.6, "gp_noise": 0.0001, "calibration_ridge": 1.0},
    "top_fraction": 0.1,
    "harm_margin_srmse": 0.01,
}


def load_config(path: Path) -> Dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a JSON object")
    return config


def rotation(theta: float) -> Array:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def local_cost(
    Z: Array,
    theta: float,
    scale: float = 1.0,
    weights: Tuple[float, float] = (1.0, 0.35),
    ripple: float = 0.07,
    frequencies: Tuple[float, float] = (3.0, 2.0),
) -> Array:
    points = np.asarray(Z, dtype=float)
    U = scale * (points @ rotation(theta).T)
    return (
        weights[0] * U[:, 0] ** 2 + weights[1] * U[:, 1] ** 2
        + ripple * (1.0 - np.cos(frequencies[0] * np.pi * U[:, 0]))
        + 0.5 * ripple * (1.0 - np.cos(frequencies[1] * np.pi * U[:, 1]))
    )


def make_relation(
    relation: str,
    theta: float,
    independent_theta: float,
) -> Tuple[Callable[[Array], Array], Callable[[Array], Array]]:
    """Return ``(source_truth, target_truth)`` for a frozen chart relation."""
    source = lambda Z: local_cost(Z, theta=theta)

    def independent(Z: Array) -> Array:
        points = np.asarray(Z, dtype=float)
        U = points @ rotation(independent_theta).T
        phase = 0.7 * independent_theta
        return (
            np.sin(3.3 * np.pi * U[:, 0] + phase)
            + 0.8 * np.cos(4.1 * np.pi * U[:, 1] - phase)
            + 0.35 * np.sin(2.0 * np.pi * (U[:, 0] + U[:, 1]))
        )

    if relation == "identity":
        return source, source
    if relation == "output_affine":
        return source, lambda Z: 4.0 + 2.5 * source(Z)
    if relation == "scale_0.7":
        return source, lambda Z: local_cost(Z, theta=theta, scale=0.7)
    if relation == "scale_1.5":
        return source, lambda Z: local_cost(Z, theta=theta, scale=1.5)
    if relation == "rotate_45":
        return source, lambda Z: local_cost(Z, theta=theta + np.pi / 4.0)
    if relation == "roughness":
        return source, lambda Z: local_cost(
            Z, theta=theta, ripple=0.28, frequencies=(7.0, 5.0)
        )
    if relation == "reversal":
        return source, lambda Z: -source(Z)
    if relation == "independent_expert":
        return independent, source
    raise ValueError(f"Unknown relation: {relation}")


def relation_label(relation: str, control: str) -> str:
    return relation if control == "none" else control


def relation_transform(relation: str) -> Callable[[Array], Array]:
    """Return the frozen target-to-source chart transform for a relation."""
    if relation in {
        "identity",
        "output_affine",
        "roughness",
        "reversal",
        "independent_expert",
    }:
        return lambda points: oracle_coordinate_transform(points, "identity")
    if relation == "scale_0.7":
        return lambda points: oracle_coordinate_transform(points, "scale", scale=0.7)
    if relation == "scale_1.5":
        return lambda points: oracle_coordinate_transform(points, "scale", scale=1.5)
    if relation == "rotate_45":
        return lambda points: oracle_coordinate_transform(
            points, "rotate", angle=np.pi / 4.0
        )
    raise ValueError(f"Unknown relation: {relation}")


def _unit_directions(n: int, seed: int) -> Array:
    raw = sobol_chart_design(2, n, seed=seed, lower=-1.0, upper=1.0)
    norms = np.linalg.norm(raw, axis=1)
    # Sobol points do not contain the origin, but this keeps the helper total.
    norms = np.maximum(norms, 1e-12)
    return raw / norms[:, None]


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


def _derive_seed(seed: int, stream: str) -> int:
    """Derive an explicit, stable 31-bit child seed from integer inputs."""
    payload = f"gate0|{int(seed)}|{stream}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**31 - 1)


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


def _git_metadata() -> Dict[str, str]:
    def run(args: Sequence[str]) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    try:
        return {
            "head": run(["rev-parse", "HEAD"]),
            "status_porcelain": run(["status", "--porcelain"]),
            "dirty": bool(run(["status", "--porcelain"])),
        }
    except Exception as exc:
        return {"head": "unavailable", "status_porcelain": str(exc), "dirty": "unknown"}


def _resolve(path: Path) -> Path:
    return Path(path).resolve() if Path(path).is_absolute() else (REPO_ROOT / path).resolve()


def _canonical_json(value: Mapping) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _validate_config(config: Mapping) -> None:
    # Do not silently accept a partially changed experiment.  Tests that need a
    # reduced panel explicitly monkeypatch this validator, preserving the normal
    # runner's exact checked-in protocol freeze.
    if dict(config) != CANONICAL_CONFIG:
        raise ValueError("configuration does not match the exact frozen Gate-0 canonical values")


def _fit_models(
    context: Array,
    context_y: Array,
    test: Array,
    source_rank_context: Array,
    source_rank_test: Array,
    source_value_context: Array,
    source_value_test: Array,
    config: Mapping,
    model_seed: int,
) -> Dict[str, Dict[str, object]]:
    transfer = dict(config["transfer_model"])
    transfer["random_state"] = model_seed
    model_config = OracleLocalModelTransferConfig(**transfer)
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
    outputs: Dict[str, Dict[str, object]] = {}
    for method, mode in METHOD_MODES.items():
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


def run_quick(config: Mapping, output_dir: Path) -> pd.DataFrame:
    """Run Gate-0 and write all auditable artifacts.

    The output directory must not exist or must be empty.  No existing file is
    removed or overwritten.
    """
    output_dir = _resolve(output_dir)
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"Output path exists and is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise FileExistsError(f"Refusing non-empty output directory: {output_dir}")
    else:
        output_dir.mkdir(parents=True)
    _validate_config(config)

    seeds = [int(value) for value in config["seeds"]]
    shells = [float(value) for value in config["shells"]]
    configured_relations = [str(value) for value in config["relations"]]
    n_source = int(config["source_train_samples"])
    n_context = int(config["target_context_samples"])
    n_test = int(config["target_test_samples_per_shell"])
    rows: List[Dict] = []
    ledger: List[Dict] = []
    diagnostics: List[Dict] = []
    failures: List[Dict] = []
    lineage_records: List[List[int]] = []
    reproducibility: Dict[str, Array] = {
        "schema_version": np.asarray([1], dtype=np.int64),
        "seed_values": np.asarray(seeds, dtype=np.int64),
        "shell_values": np.asarray(shells, dtype=np.float64),
        "relation_names": np.asarray(configured_relations, dtype="U32"),
        "control_names": np.asarray([str(value) for value in config["controls"]], dtype="U64"),
    }

    for seed in seeds:
        instance = {"seed": seed, "dimension": 2}
        try:
            theta_seed = _derive_seed(seed, "theta")
            source_seed = _derive_seed(seed, "source_design")
            source_expert_seed = _derive_seed(seed, "source_expert")
            context_seed = _derive_seed(seed, "target_context_design")
            test_seed = _derive_seed(seed, "target_test_design")
            permutation_seed = _derive_seed(seed, "source_permutation")
            theta_rng = np.random.default_rng(theta_seed)
            theta = float(theta_rng.uniform(0.0, np.pi))
            independent_theta = float(theta + theta_rng.uniform(0.35 * np.pi, 0.85 * np.pi))
            source_dirs = sobol_chart_design(2, n_source, seed=source_seed)
            context_dirs = _unit_directions(n_context, context_seed)
            test_dirs = _unit_directions(n_test, test_seed)
            context = np.vstack([
                context_dirs[i * 4:(i + 1) * 4] * shell
                for i, shell in enumerate(shells)
            ])
            tests = {shell: test_dirs * shell for shell in shells}
            for test in tests.values():
                _assert_disjoint(context, test)
            expert_config = dict(config["transfer_model"])
            expert_config["gp_length_scale"] = float(config["source_expert"]["length_scale"])
            expert_config["gp_noise"] = float(config["source_expert"]["noise"])
            expert_model_config = OracleLocalModelTransferConfig(**expert_config)
            context_hash = _array_hash(context)
            test_hashes = {str(shell): _array_hash(test) for shell, test in tests.items()}
            permutation_rng = np.random.default_rng(permutation_seed)
            permutation = permutation_rng.permutation(n_source).astype(np.int64)
            configured_relations = [str(value) for value in config["relations"]]
            n_conditions = sum(
                2 if relation == "identity" and "identity_label_permutation" in config["controls"] else 1
                for relation in configured_relations
            )
            model_seed_values = [_derive_seed(seed, f"model_{index}") for index in range(n_conditions * len(shells))]
            lineage_records.append([seed, theta_seed, source_seed, source_expert_seed, context_seed, test_seed, permutation_seed, *model_seed_values])
            reproducibility[f"seed_{seed}_source_dirs"] = np.asarray(source_dirs, dtype=np.float64)
            reproducibility[f"seed_{seed}_context_dirs"] = np.asarray(context_dirs, dtype=np.float64)
            reproducibility[f"seed_{seed}_context_points"] = np.asarray(context, dtype=np.float64)
            reproducibility[f"seed_{seed}_test_dirs"] = np.asarray(test_dirs, dtype=np.float64)
            reproducibility[f"seed_{seed}_permutation"] = permutation
            reproducibility[f"seed_{seed}_theta"] = np.asarray([theta, independent_theta], dtype=np.float64)
            reproducibility[f"seed_{seed}_theta_value"] = np.asarray([theta], dtype=np.float64)
            reproducibility[f"seed_{seed}_independent_theta"] = np.asarray([independent_theta], dtype=np.float64)
            reproducibility[f"seed_{seed}_seed_lineage"] = np.asarray(lineage_records[-1], dtype=np.int64)
            reproducibility[f"seed_{seed}_context_shell_index"] = np.repeat(np.arange(len(shells), dtype=np.int64), 4)
            model_seed_index = 0

            for relation in configured_relations:
                configured_controls = ["none"]
                if relation == "identity" and "identity_label_permutation" in config["controls"]:
                    configured_controls.append("identity_label_permutation")
                for control in configured_controls:
                    source_truth, target_truth = make_relation(relation, theta, independent_theta)
                    source_y = np.asarray(source_truth(source_dirs), dtype=float)
                    reproducibility[f"seed_{seed}_source_y_{relation}"] = source_y.astype(np.float64)
                    train_source_y = source_y[permutation] if control != "none" else source_y
                    source_expert_for_condition = fit_source_oracle_expert(
                        source_dirs, train_source_y, expert_model_config, seed=source_expert_seed,
                    )
                    source_hash = _array_hash(np.column_stack([source_dirs, train_source_y]))
                    condition_name = relation_label(relation, control)
                    transform = relation_transform(relation)
                    source_context_points = transform(context)
                    source_context_key = f"seed_{seed}_source_context_query_{relation}"
                    reproducibility[source_context_key] = np.asarray(source_context_points, dtype=np.float64)
                    reproducibility[f"seed_{seed}_target_context_truth_{relation}"] = np.asarray(target_truth(context), dtype=np.float64)
                    source_rank_context_quality, _ = source_expert_for_condition.predict_rank(
                        source_context_points
                    )
                    # The rank GP reports larger-is-better quality.  Transfer
                    # calibration is a minimization model, hence source cost.
                    rank_context = 1.0 - source_rank_context_quality
                    # SourceOracleExpert.predict defaults to its raw-value GP;
                    # retain that standardized value feature for transfer.
                    source_value_context_points, _ = source_expert_for_condition.predict(
                        source_context_points, feature="raw_value"
                    )
                    for shell in shells:
                        test = tests[shell]
                        target_context_y = np.asarray(target_truth(context), dtype=float)
                        target_test_y = np.asarray(target_truth(test), dtype=float)
                        source_test_points = transform(test)
                        reproducibility[f"seed_{seed}_target_test_points_{relation}_{shell}"] = np.asarray(test, dtype=np.float64)
                        reproducibility[f"seed_{seed}_target_test_truth_{relation}_{shell}"] = np.asarray(target_test_y, dtype=np.float64)
                        reproducibility[f"seed_{seed}_source_test_query_{relation}_{shell}"] = np.asarray(source_test_points, dtype=np.float64)
                        source_rank_test_quality, _ = source_expert_for_condition.predict_rank(
                            source_test_points
                        )
                        rank_test = 1.0 - source_rank_test_quality
                        value_test_standardized, _ = source_expert_for_condition.predict(
                            source_test_points, feature="raw_value"
                        )
                        value_test_raw = (source_expert_for_condition.raw_standardizer_.inverse_transform(value_test_standardized)
                                          if source_expert_for_condition.raw_standardizer_ is not None else value_test_standardized)
                        value_context = source_value_context_points
                        source_context_query_hash = _array_hash(source_context_points)
                        source_test_query_hash = _array_hash(source_test_points)
                        outputs = _fit_models(
                            context, target_context_y, test,
                            rank_context, rank_test, value_context, value_test_standardized,
                            config, model_seed_values[model_seed_index],
                        )
                        model_seed = model_seed_values[model_seed_index]
                        model_seed_index += 1
                        target_srmse: Optional[float] = None
                        metric_rows: List[Dict] = []
                        for method in config["methods"]:
                            output = outputs[method]
                            metrics = evaluate_predictions(
                                target_test_y, output["mean"], output["std"],
                                top_fraction=float(config["top_fraction"]),
                            )
                            if method == "Target-Only":
                                target_srmse = metrics.standardized_rmse
                            metric_rows.append({
                                **instance, "relation": relation, "control": control,
                                "relation_or_control": condition_name, "shell": shell,
                                "panel": "test", "method": method, "model_seed": model_seed,
                                **metrics.__dict__,
                                "effective_mode": output["effective_mode"],
                                "prior_names": output["prior_names"],
                                "prior_coefficients": output["prior_coefficients"],
                                "source_data_hash": source_hash,
                                "source_query_transform": relation,
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
                                    **instance, "relation": relation, "control": control,
                                    "relation_or_control": condition_name, "shell": shell,
                                    "panel": "test", "candidate_index": candidate_index,
                                    "chart_point": json.dumps(point.tolist()), "truth": float(truth),
                                    "method": method, "model_seed": model_seed,
                                    "predicted_mean": float(mean), "predicted_std": float(std),
                                })
                        if target_srmse is None:
                            raise RuntimeError("Target-Only metrics were not produced")
                        for record in metric_rows:
                            record["srmse_delta_vs_target_only"] = float(
                                record["standardized_rmse"] - target_srmse
                            )
                            record["negative_transfer"] = bool(
                                record["srmse_delta_vs_target_only"]
                                > float(config["harm_margin_srmse"])
                            )
                            rows.append(record)
                        source_rank_test = 1.0 - source_rank_test_quality
                        target_standardized = _robust_standardize(target_test_y)
                        source_value_standardized_target_rmse = float(np.sqrt(np.mean((value_test_standardized - target_standardized) ** 2)))
                        context_oob_count, context_oob_rate = _oob(source_context_points)
                        test_oob_count, test_oob_rate = _oob(source_test_points)
                        diagnostic = {
                            **instance, "relation": relation, "control": control,
                            "relation_or_control": condition_name, "shell": shell,
                            "panel": "test", "source_data_hash": source_hash,
                            # Pairwise ordering is unit-invariant (including output_affine);
                            # only an RMSE would incorrectly claim common raw units.
                            "source_value_pairwise_accuracy": pairwise_cost_accuracy(value_test_raw, target_test_y),
                            "source_rank_target_agreement": pairwise_cost_accuracy(source_rank_test, target_test_y),
                            "source_value_standardized_target_rmse": source_value_standardized_target_rmse,
                            "source_value_standardized_target_rmse_definition": "source raw-value GP prediction in source standard units versus independently robust-standardized target test truth",
                            "source_query_transform": relation,
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
                            "theta": theta, "independent_theta": independent_theta,
                            "model_seed": model_seed,
                        }
                        diagnostics.append(diagnostic)
        except Exception as exc:
            failures.append({**instance, "error_type": type(exc).__name__, "error": str(exc)})

    result_frame = pd.DataFrame(rows)
    ledger_frame = pd.DataFrame(ledger)
    diagnostics_frame = pd.DataFrame(diagnostics)
    failures_frame = pd.DataFrame(failures, columns=["seed", "dimension", "error_type", "error"])
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
    # Persist exactly the config supplied to the run; this is the bytes hashed
    # by the manifest and is the authoritative provenance artifact.
    config_bytes = json.dumps(config, indent=2, ensure_ascii=False).encode("utf-8")
    paths["config"].write_bytes(config_bytes)
    if lineage_records:
        reproducibility["seed_lineage"] = np.asarray(lineage_records, dtype=np.int64)
    # Convenient stacked aliases make reconstruction independent of key parsing,
    # while the seed/relation-specific keys above retain lossless provenance.
    reproducibility["source_dirs"] = np.stack([reproducibility[f"seed_{seed}_source_dirs"] for seed in seeds])
    reproducibility["target_context_points"] = np.stack([reproducibility[f"seed_{seed}_context_points"] for seed in seeds])
    reproducibility["target_test_points"] = np.stack([reproducibility[f"seed_{seed}_test_dirs"] * np.asarray(shells)[:, None, None] for seed in seeds])
    reproducibility["permutation"] = np.stack([reproducibility[f"seed_{seed}_permutation"] for seed in seeds])
    reproducibility["theta"] = np.stack([reproducibility[f"seed_{seed}_theta"] for seed in seeds])
    np.savez_compressed(paths["reproducibility_inputs"], **reproducibility)
    reproducibility_shapes = {
        key: {"shape": list(np.asarray(value).shape), "dtype": str(np.asarray(value).dtype)}
        for key, value in reproducibility.items()
    }
    reproducibility_hashes = {
        key: _array_hash(value) for key, value in reproducibility.items()
        if "target_" in key or "source_" in key or key in {"seed_lineage", "seed_values", "shell_values"}
    }
    dependency_paths = {
        "runner": Path(__file__).resolve(),
        "oracle_core": SRC_DIR / "region_guided_reranking_study" / "oracle_local_model_transfer.py",
        "local_surrogate_transfer_research": SRC_DIR / "region_guided_reranking_study" / "local_surrogate_transfer_research.py",
        "local_surrogate_transfer": SRC_DIR / "region_guided_reranking_study" / "local_surrogate_transfer.py",
    }
    dependency_hashes = {
        name: {"path": str(path.relative_to(REPO_ROOT)), "sha256": _file_hash(path)}
        for name, path in dependency_paths.items()
    }
    manifest = {
        "stage_id": config["stage_id"], "scope": config["scope"], "config": config,
        "config_sha256": _file_hash(paths["config"]),
        "counts": {"result_rows": len(result_frame), "ledger_rows": len(ledger_frame),
                   "diagnostic_rows": len(diagnostics_frame), "failure_rows": len(failures_frame),
                   "seed_count": len(seeds), "relation_count": len(configured_relations),
                   "shell_count": len(shells), "reproducibility_array_count": len(reproducibility)},
        "artifacts": {name: _file_hash(path) for name, path in paths.items()},
        "artifact_metadata": {
            "config": {"encoding": "utf-8", "indent": 2},
            "reproducibility_inputs": {"schema_version": 1, "array_count": len(reproducibility),
                                        "keys": sorted(reproducibility), "shapes": reproducibility_shapes,
                                        "array_hashes": reproducibility_hashes},
        },
        "dependencies": dependency_hashes,
        "dependency_sha256": {name: item["sha256"] for name, item in dependency_hashes.items()},
        "lineage": {"columns": ["seed", "theta_seed", "source_design_seed", "source_expert_seed", "target_context_design_seed",
                                  "target_test_design_seed", "permutation_seed"] +
                                 [f"model_seed_{i}" for i in range(max(0, len(lineage_records[0]) - 7))]
                    if lineage_records else [],
                    "values": lineage_records},
        "runner_sha256": dependency_hashes["runner"]["sha256"],
        "runner_path": dependency_hashes["runner"]["path"],
        "core_sha256": dependency_hashes["oracle_core"]["sha256"],
        "core_path": dependency_hashes["oracle_core"]["path"],
        "runner_core_sha256": {"runner": dependency_hashes["runner"]["sha256"],
                                "core": dependency_hashes["oracle_core"]["sha256"]},
        "git": _git_metadata(), "python": sys.version,
        "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "sklearn": sklearn.__version__,
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return result_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "oracle_local_model_transfer_quick.json")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results" / "oracle_local_model_transfer_quick")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_quick(load_config(_resolve(args.config)), args.output)
