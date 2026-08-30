"""Run the frozen static held-out local-surrogate transfer Pilot v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from region_guided_reranking_study.landscapes import get_task_suite  # noqa: E402
from region_guided_reranking_study.local_surrogate_transfer import (  # noqa: E402
    LocalExpertResidualRegressor,
    LocalSurrogateTransferConfig,
    fit_affine_source_calibration,
)
from region_guided_reranking_study.local_surrogate_transfer_research import (  # noqa: E402
    AlignedLocalExpert,
    bounded_chart_radius,
    chart_evaluator,
    evaluate_predictions,
    oracle_global_anchor,
    select_structure_near_anchor,
    sobol_chart_design,
    source_support_diagnostics,
)
from region_guided_reranking_study.source_local_structure import (  # noqa: E402
    LocalStructureConfig,
    SourceLocalStructureExtractor,
)
from region_guided_reranking_study.source_structure_research import (  # noqa: E402
    latin_hypercube_sample,
)

PROTOCOL_PATH = REPO_ROOT / "PROTOCOL_LOCAL_SURROGATE_TRANSFER_PILOT.md"
METHODS = [
    "Target-Only",
    "Source-Affine-Only",
    "Fixed-Source+Residual",
    "Calibrated-Source+Residual",
    "Gated-Source+Residual",
]


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Configuration root must be a JSON object.")
    return data


def make_extraction_config(config: Mapping, seed: int) -> LocalStructureConfig:
    values = dict(config["extraction"])
    values["random_state"] = int(seed)
    return LocalStructureConfig(**values)


def make_transfer_config(config: Mapping, seed: int) -> LocalSurrogateTransferConfig:
    values = dict(config["transfer_model"])
    values["random_state"] = int(seed)
    return LocalSurrogateTransferConfig(**values)


def run_pilot(config: Mapping, output_dir: Path) -> pd.DataFrame:
    output_dir = _resolve_repo_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pilot = dict(config["pilot"])
    problems = [str(item) for item in pilot["problems"]]
    dimensions = [int(item) for item in pilot["dimensions"]]
    seeds = [int(item) for item in pilot["seeds"]]
    relations = [str(item) for item in pilot["relations"]]
    context_sizes = sorted(int(item) for item in pilot["target_context_sizes"])
    source_train_samples = int(pilot["source_train_samples"])
    target_test_samples = int(pilot["target_test_samples"])
    chart_fraction = float(pilot["chart_radius_fraction"])
    chart_extent = float(pilot.get("chart_extent", 1.0))
    top_fraction = float(pilot.get("top_fraction", 0.10))
    target_noise = float(pilot.get("target_observation_noise_std", 0.0))
    harm_margin = float(pilot.get("harm_margin_srmse", 0.01))

    if any(relation not in {"matching", "wrong", "reversed"} for relation in relations):
        raise ValueError("relations may contain only matching, wrong, and reversed.")
    if context_sizes[0] < 4:
        raise ValueError("Every target context size must be at least four.")
    if int(pilot["primary_context_size"]) not in context_sizes:
        raise ValueError("primary_context_size must be one configured context size.")

    rows: List[Dict] = []
    diagnostics: List[Dict] = []
    target_ledger_rows: List[Dict] = []
    failures: List[Dict] = []
    total = len(problems) * len(dimensions) * len(seeds)
    completed = 0

    for dim in dimensions:
        for problem in problems:
            for seed in seeds:
                instance = {"problem": problem, "dim": dim, "seed": seed}
                try:
                    instance_sequence = np.random.SeedSequence(
                        [seed, dim, _stable_problem_code(problem)]
                    )
                    seed_children = instance_sequence.spawn(8)
                    task_rng = np.random.default_rng(seed_children[0])
                    matching_train_rng = np.random.default_rng(seed_children[1])
                    wrong_train_rng = np.random.default_rng(seed_children[2])
                    noise_rng = np.random.default_rng(seed_children[3])
                    model_seed_rng = np.random.default_rng(seed_children[4])
                    context_seed_rng = np.random.default_rng(seed_children[5])
                    test_seed_rng = np.random.default_rng(seed_children[6])

                    suite = get_task_suite(dim=dim, rng=task_rng)[problem]
                    target_function = suite["target"]
                    matching_source = suite["matching_sources"][0]
                    wrong_source = suite["mismatched_sources"][0]
                    bounds = np.asarray(suite["bounds"], dtype=float)
                    target_anchor = oracle_global_anchor(target_function)
                    matching_anchor = oracle_global_anchor(matching_source)
                    wrong_anchor = oracle_global_anchor(wrong_source)
                    chart_radius = bounded_chart_radius(bounds, chart_fraction)

                    matching_X = latin_hypercube_sample(
                        bounds,
                        source_train_samples,
                        seed=int(matching_train_rng.integers(0, 2**31 - 1)),
                    )
                    wrong_X = latin_hypercube_sample(
                        bounds,
                        source_train_samples,
                        seed=int(wrong_train_rng.integers(0, 2**31 - 1)),
                    )
                    matching_y = np.asarray(matching_source(matching_X), dtype=float).reshape(-1)
                    wrong_y = np.asarray(wrong_source(wrong_X), dtype=float).reshape(-1)

                    matching_library = SourceLocalStructureExtractor(
                        make_extraction_config(
                            config,
                            int(model_seed_rng.integers(0, 2**31 - 1)),
                        )
                    ).fit_dataset(
                        matching_X,
                        matching_y,
                        task_id=f"matching_{problem}_{dim}_{seed}",
                    )
                    wrong_library = SourceLocalStructureExtractor(
                        make_extraction_config(
                            config,
                            int(model_seed_rng.integers(0, 2**31 - 1)),
                        )
                    ).fit_dataset(
                        wrong_X,
                        wrong_y,
                        task_id=f"wrong_{problem}_{dim}_{seed}",
                    )
                    matching_structure = select_structure_near_anchor(
                        matching_library,
                        matching_anchor,
                    )
                    wrong_structure = select_structure_near_anchor(
                        wrong_library,
                        wrong_anchor,
                    )
                    experts = {
                        "matching": AlignedLocalExpert(
                            matching_structure,
                            matching_structure.center,
                            target_anchor,
                            chart_radius,
                            reverse_quality=False,
                        ),
                        "wrong": AlignedLocalExpert(
                            wrong_structure,
                            wrong_structure.center,
                            target_anchor,
                            chart_radius,
                            reverse_quality=False,
                        ),
                        "reversed": AlignedLocalExpert(
                            matching_structure,
                            matching_structure.center,
                            target_anchor,
                            chart_radius,
                            reverse_quality=True,
                        ),
                    }
                    oracle_source_anchors = {
                        "matching": matching_anchor,
                        "wrong": wrong_anchor,
                        "reversed": matching_anchor,
                    }

                    max_context = max(context_sizes)
                    context_Z = sobol_chart_design(
                        dim,
                        max_context,
                        seed=int(context_seed_rng.integers(0, 2**31 - 1)),
                        lower=-chart_extent,
                        upper=chart_extent,
                    )
                    test_Z = sobol_chart_design(
                        dim,
                        target_test_samples,
                        seed=int(test_seed_rng.integers(0, 2**31 - 1)),
                        lower=-chart_extent,
                        upper=chart_extent,
                    )
                    _assert_disjoint(context_Z, test_Z)
                    target_evaluate = chart_evaluator(
                        target_function,
                        target_anchor,
                        chart_radius,
                    )
                    clean_context_y = target_evaluate(context_Z)
                    if target_noise > 0.0:
                        context_y_all = clean_context_y + noise_rng.normal(
                            0.0,
                            target_noise,
                            size=len(clean_context_y),
                        )
                    else:
                        context_y_all = clean_context_y.copy()
                    test_y = target_evaluate(test_Z)
                    for point_index, (chart_point, clean_y, observed_y) in enumerate(
                        zip(context_Z, clean_context_y, context_y_all)
                    ):
                        target_ledger_rows.append(
                            {
                                **instance,
                                "panel": "context",
                                "point_index": point_index,
                                "chart_point": json.dumps(chart_point.tolist()),
                                "clean_target_y": float(clean_y),
                                "observed_target_y": float(observed_y),
                            }
                        )
                    for point_index, (chart_point, truth) in enumerate(
                        zip(test_Z, test_y)
                    ):
                        target_ledger_rows.append(
                            {
                                **instance,
                                "panel": "test",
                                "point_index": point_index,
                                "chart_point": json.dumps(chart_point.tolist()),
                                "clean_target_y": float(truth),
                                "observed_target_y": float(truth),
                            }
                        )

                    hashes = {
                        "matching_source_data_hash": _array_hash(
                            np.column_stack([matching_X, matching_y])
                        ),
                        "wrong_source_data_hash": _array_hash(
                            np.column_stack([wrong_X, wrong_y])
                        ),
                        "target_context_design_hash": _array_hash(context_Z),
                        "target_test_design_hash": _array_hash(test_Z),
                        "target_test_truth_hash": _array_hash(test_y),
                    }

                    for relation in relations:
                        expert = experts[relation]
                        source_function = (
                            wrong_source if relation == "wrong" else matching_source
                        )
                        source_test_y = chart_evaluator(
                            source_function,
                            expert.source_anchor,
                            chart_radius,
                        )(test_Z)
                        test_quality, _ = expert.predict_quality_from_chart(test_Z)
                        source_fidelity = evaluate_predictions(
                            source_test_y,
                            1.0 - test_quality,
                            predicted_std=None,
                            top_fraction=top_fraction,
                        )
                        support = source_support_diagnostics(
                            expert,
                            [context_Z, test_Z],
                        )

                        for context_size in context_sizes:
                            train_Z = context_Z[:context_size]
                            train_y = context_y_all[:context_size]
                            train_quality, _ = expert.predict_quality_from_chart(train_Z)
                            shared_model_sequence = np.random.SeedSequence(
                                [
                                    seed,
                                    dim,
                                    _stable_problem_code(problem),
                                    context_size,
                                    202603,
                                ]
                            )
                            model_cfg = make_transfer_config(
                                config,
                                int(
                                    np.random.default_rng(
                                        shared_model_sequence
                                    ).integers(0, 2**31 - 1)
                                ),
                            )
                            method_outputs = _fit_and_predict_methods(
                                train_Z,
                                train_y,
                                train_quality,
                                test_Z,
                                test_quality,
                                model_cfg,
                            )
                            target_only_srmse: Optional[float] = None
                            temporary_rows: List[Dict] = []
                            for method in METHODS:
                                output = method_outputs[method]
                                metrics = evaluate_predictions(
                                    test_y,
                                    output["mean"],
                                    output["std"],
                                    top_fraction=top_fraction,
                                )
                                if method == "Target-Only":
                                    target_only_srmse = metrics.standardized_rmse
                                temporary_rows.append(
                                    {
                                        **instance,
                                        "relation": relation,
                                        "target_context_size": context_size,
                                        "method": method,
                                        **metrics.__dict__,
                                        "effective_mode": output["effective_mode"],
                                        "gate_accepted": output["gate_accepted"],
                                        "calibration_slope": output["calibration_slope"],
                                        "calibration_raw_slope": output[
                                            "calibration_raw_slope"
                                        ],
                                        "cv_target_rmse": output["cv_target_rmse"],
                                        "cv_transfer_rmse": output["cv_transfer_rmse"],
                                        "cv_relative_rmse_gain": output[
                                            "cv_relative_rmse_gain"
                                        ],
                                        "context_pairwise_accuracy": output[
                                            "context_pairwise_accuracy"
                                        ],
                                        "gate_rejection_reason": output[
                                            "gate_rejection_reason"
                                        ],
                                        "source_fidelity_ndcg": source_fidelity.ndcg_at_top,
                                        "source_fidelity_spearman": source_fidelity.spearman,
                                        "source_fidelity_pairwise": source_fidelity.pairwise_accuracy,
                                        "n_matching_structures": len(
                                            matching_library.structures
                                        ),
                                        "n_wrong_structures": len(
                                            wrong_library.structures
                                        ),
                                        **support,
                                        **hashes,
                                    }
                                )
                            if target_only_srmse is None:
                                raise RuntimeError("Target-Only metrics were not produced.")
                            for row in temporary_rows:
                                row["srmse_delta_vs_target_only"] = float(
                                    row["standardized_rmse"] - target_only_srmse
                                )
                                row["negative_transfer"] = bool(
                                    row["srmse_delta_vs_target_only"] > harm_margin
                                )
                                rows.append(row)

                        diagnostics.append(
                            {
                                **instance,
                                "relation": relation,
                                "source_task_id": expert.structure.task_id,
                                "source_region_id": expert.structure.region_id,
                                "source_chart_anchor": json.dumps(expert.source_anchor.tolist()),
                                "oracle_source_anchor": json.dumps(
                                    oracle_source_anchors[relation].tolist()
                                ),
                                "target_anchor": json.dumps(target_anchor.tolist()),
                                "extracted_center": json.dumps(
                                    expert.structure.center.tolist()
                                ),
                                "normalized_anchor_error": float(
                                    np.linalg.norm(
                                        expert.structure.center
                                        - oracle_source_anchors[relation]
                                    )
                                    / max(chart_radius, 1e-12)
                                ),
                                "chart_radius": chart_radius,
                                "source_oof_spearman": expert.structure.validation.oof_spearman,
                                "source_oof_ndcg": expert.structure.validation.oof_ndcg,
                                "source_reliability": expert.structure.validation.reliability,
                                "source_context_count": expert.structure.context_count,
                                "source_core_count": expert.structure.core_count,
                                "source_fidelity_ndcg": source_fidelity.ndcg_at_top,
                                "source_fidelity_spearman": source_fidelity.spearman,
                                "source_fidelity_pairwise": source_fidelity.pairwise_accuracy,
                                **support,
                                **hashes,
                            }
                        )
                except Exception as exc:
                    failures.append(
                        {
                            **instance,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                completed += 1
                if completed % max(1, total // 20) == 0:
                    print(f"Local-surrogate Pilot progress: {completed}/{total}")

    frame = pd.DataFrame(rows)
    diagnostics_frame = pd.DataFrame(diagnostics)
    target_ledger_frame = pd.DataFrame(target_ledger_rows)
    failures_frame = pd.DataFrame(
        failures,
        columns=["problem", "dim", "seed", "error_type", "error"],
    )
    result_path = output_dir / "local_surrogate_transfer_results.csv"
    diagnostics_path = output_dir / "local_surrogate_transfer_diagnostics.csv"
    target_ledger_path = output_dir / "local_surrogate_transfer_target_ledger.csv"
    failures_path = output_dir / "local_surrogate_transfer_failures.csv"
    frame.to_csv(result_path, index=False)
    diagnostics_frame.to_csv(diagnostics_path, index=False)
    target_ledger_frame.to_csv(target_ledger_path, index=False)
    failures_frame.to_csv(failures_path, index=False)

    _write_manifest(
        output_dir / "local_surrogate_transfer_manifest.json",
        config,
        {
            "result_rows": len(frame),
            "diagnostic_rows": len(diagnostics_frame),
            "target_ledger_rows": len(target_ledger_frame),
            "failure_rows": len(failures_frame),
            "independent_instances": len(problems) * len(dimensions) * len(seeds),
        },
        [result_path, diagnostics_path, target_ledger_path, failures_path],
    )
    return frame


def _fit_and_predict_methods(
    train_X: np.ndarray,
    train_y: np.ndarray,
    train_quality: np.ndarray,
    test_X: np.ndarray,
    test_quality: np.ndarray,
    config: LocalSurrogateTransferConfig,
) -> Dict[str, Dict]:
    outputs: Dict[str, Dict] = {}
    modes = {
        "Target-Only": "target_only",
        "Fixed-Source+Residual": "fixed",
        "Calibrated-Source+Residual": "calibrated",
        "Gated-Source+Residual": "gated",
    }
    for method, mode in modes.items():
        model = LocalExpertResidualRegressor(mode, config).fit(
            train_X,
            train_y,
            None if mode == "target_only" else train_quality,
        )
        mean, std = model.predict(
            test_X,
            None if model.effective_mode_ == "target_only" else test_quality,
            return_std=True,
        )
        evidence = model.evidence_
        calibration = model.calibration_
        calibration_attempt = model.calibration_attempt_
        recorded_slope = (
            float(evidence.calibration_slope)
            if evidence is not None
            else (
                float(calibration_attempt.slope)
                if calibration_attempt is not None
                else 0.0
            )
        )
        recorded_raw_slope = (
            float(evidence.calibration_raw_slope)
            if evidence is not None
            else (
                float(calibration_attempt.raw_slope)
                if calibration_attempt is not None
                else 0.0
            )
        )
        outputs[method] = {
            "mean": mean,
            "std": std,
            "effective_mode": str(model.effective_mode_),
            "gate_accepted": bool(evidence.accepted) if evidence is not None else False,
            "calibration_slope": recorded_slope,
            "calibration_raw_slope": recorded_raw_slope,
            "cv_target_rmse": (
                float(evidence.cv_target_rmse) if evidence is not None else np.nan
            ),
            "cv_transfer_rmse": (
                float(evidence.cv_transfer_rmse) if evidence is not None else np.nan
            ),
            "cv_relative_rmse_gain": (
                float(evidence.relative_rmse_gain) if evidence is not None else np.nan
            ),
            "context_pairwise_accuracy": (
                float(evidence.pairwise_accuracy) if evidence is not None else np.nan
            ),
            "gate_rejection_reason": (
                evidence.rejection_reason if evidence is not None else "not_applicable"
            ),
        }

    calibration = fit_affine_source_calibration(train_quality, train_y, config)
    if calibration is None or calibration.slope <= 0.0:
        affine_mean = np.full(len(test_X), float(np.mean(train_y)), dtype=float)
        residual = train_y - np.mean(train_y)
        slope = 0.0
        raw_slope = 0.0 if calibration is None else float(calibration.raw_slope)
        effective = "constant_target_mean"
    else:
        affine_mean = calibration.predict(test_quality)
        residual = train_y - calibration.predict(train_quality)
        slope = float(calibration.slope)
        raw_slope = float(calibration.raw_slope)
        effective = "calibrated_source_only"
    affine_std = np.full(
        len(test_X),
        max(float(np.sqrt(np.mean(residual**2))), 1e-6),
        dtype=float,
    )
    outputs["Source-Affine-Only"] = {
        "mean": affine_mean,
        "std": affine_std,
        "effective_mode": effective,
        "gate_accepted": False,
        "calibration_slope": slope,
        "calibration_raw_slope": raw_slope,
        "cv_target_rmse": np.nan,
        "cv_transfer_rmse": np.nan,
        "cv_relative_rmse_gain": np.nan,
        "context_pairwise_accuracy": np.nan,
        "gate_rejection_reason": "not_applicable",
    }
    return outputs


def _assert_disjoint(first: np.ndarray, second: np.ndarray) -> None:
    first_rows = {tuple(row) for row in np.round(first, 12)}
    second_rows = {tuple(row) for row in np.round(second, 12)}
    if first_rows.intersection(second_rows):
        raise AssertionError("Target context and target test designs overlap.")


def _resolve_repo_path(path: Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (REPO_ROOT / candidate).resolve()


def _stable_problem_code(problem: str) -> int:
    digest = hashlib.sha256(problem.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def _array_hash(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(np.asarray(array, dtype="<f8"))
    header = json.dumps(
        {"shape": contiguous.shape, "dtype": str(contiguous.dtype)},
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(header + contiguous.tobytes()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata() -> Dict[str, object]:
    def run(args: List[str]) -> str:
        return subprocess.check_output(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

    try:
        return {
            "head": run(["rev-parse", "HEAD"]),
            "status_porcelain": run(["status", "--porcelain"]),
        }
    except Exception as exc:
        return {"head": "unavailable", "status_porcelain": f"unavailable: {exc}"}


def _write_manifest(
    path: Path,
    config: Mapping,
    counts: Mapping,
    artifact_paths: List[Path],
) -> None:
    encoded = json.dumps(config, sort_keys=True).encode("utf-8")
    payload = {
        "stage_id": config.get("stage_id", "unknown"),
        "config": config,
        "config_sha256": hashlib.sha256(encoded).hexdigest(),
        "protocol_path": str(PROTOCOL_PATH.relative_to(REPO_ROOT)),
        "protocol_sha256": _file_sha256(PROTOCOL_PATH),
        "counts": dict(counts),
        "artifact_sha256": {
            str(item.relative_to(REPO_ROOT)): _file_sha256(item)
            for item in artifact_paths
        },
        "git": _git_metadata(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "local_surrogate_transfer_quick.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "local_surrogate_transfer_pilot_quick",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_pilot(load_config(arguments.config), arguments.output)
