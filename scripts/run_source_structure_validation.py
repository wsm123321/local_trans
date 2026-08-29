"""Held-out source-fidelity and cross-task utility study.

The extractor is fitted only on source training observations.  All reported source
and target metrics are computed on an independent, frozen candidate pool shared by
all methods within an instance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from region_guided_reranking_study.landscapes import get_task_suite  # noqa: E402
from region_guided_reranking_study.source_local_structure import (  # noqa: E402
    LocalStructureConfig,
    SourceLocalStructureExtractor,
)
from region_guided_reranking_study.source_structure_research import (  # noqa: E402
    best_point_distance_score,
    evaluate_ranking,
    fit_global_source_gp,
    latin_hypercube_sample,
    local_subset_mask,
    make_independent_test_pool,
    rank_quality,
)

BASE_COMMIT = "917e38a32c947d9394d7d3ff96cbbe39c6236bd0"


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Configuration root must be a JSON object.")
    return data


def make_extraction_config(config: Mapping, seed: int) -> LocalStructureConfig:
    extraction = dict(config.get("extraction", {}))
    extraction["random_state"] = int(seed)
    return LocalStructureConfig(**extraction)


def run_validation(config: Mapping, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    validation = dict(config["validation"])
    problems = list(validation["problems"])
    dimensions = [int(value) for value in validation["dimensions"]]
    seeds = [int(value) for value in validation["seeds"]]
    scenarios = list(validation.get("source_scenarios", ["matching", "wrong"]))
    train_samples = int(validation["source_train_samples"])
    test_global = int(validation["test_global_samples"])
    test_local = int(validation["test_local_per_structure"])
    tasks_per_scenario = int(validation.get("source_tasks_per_scenario", 1))
    top_fraction = float(validation.get("top_fraction", 0.10))
    local_fraction = float(validation.get("local_subset_fraction", 0.35))
    local_covariance_scale = float(validation.get("local_covariance_scale", 2.0))

    rows: List[Dict] = []
    structure_rows: List[Dict] = []
    failure_rows: List[Dict] = []

    total = len(problems) * len(dimensions) * len(seeds) * len(scenarios)
    completed = 0

    for dim in dimensions:
        for problem in problems:
            for seed in seeds:
                master = np.random.SeedSequence(seed)
                task_ss, train_ss, test_ss, permutation_ss = master.spawn(4)
                task_rng = np.random.default_rng(task_ss)
                train_rng = np.random.default_rng(train_ss)
                test_rng = np.random.default_rng(test_ss)
                permutation_rng = np.random.default_rng(permutation_ss)

                suite = get_task_suite(dim=dim, rng=task_rng)[problem]
                target_function = suite["target"]
                bounds = np.asarray(suite["bounds"], dtype=float)

                for scenario in scenarios:
                    if scenario == "matching":
                        source_functions: Sequence = suite["matching_sources"]
                    elif scenario == "wrong":
                        source_functions = suite["mismatched_sources"]
                    else:
                        raise ValueError(f"Unknown source scenario: {scenario}")

                    selected_sources = list(source_functions)[:tasks_per_scenario]
                    for source_index, source_function in enumerate(selected_sources):
                        instance = {
                            "problem": problem,
                            "dim": dim,
                            "seed": seed,
                            "source_scenario": scenario,
                            "source_index": source_index,
                        }
                        try:
                            sample_seed = int(train_rng.integers(0, 2**31 - 1))
                            train_X = latin_hypercube_sample(
                                bounds,
                                n_samples=train_samples,
                                seed=sample_seed,
                            )
                            train_y = np.asarray(source_function(train_X), dtype=float).reshape(-1)

                            extractor = SourceLocalStructureExtractor(
                                make_extraction_config(config, seed + source_index * 101)
                            )
                            library = extractor.fit_dataset(
                                train_X,
                                train_y,
                                task_id=f"{scenario}_{problem}_{source_index}",
                            )

                            permuted_y = permutation_rng.permutation(train_y)
                            null_library = SourceLocalStructureExtractor(
                                make_extraction_config(
                                    config,
                                    seed + 100000 + source_index * 101,
                                )
                            ).fit_dataset(
                                train_X,
                                permuted_y,
                                task_id=f"permuted_{scenario}_{problem}_{source_index}",
                            )

                            test_X = make_independent_test_pool(
                                bounds,
                                library,
                                rng=test_rng,
                                n_global=test_global,
                                n_local_per_structure=test_local,
                                local_covariance_scale=local_covariance_scale,
                            )
                            source_test_y = np.asarray(source_function(test_X), dtype=float).reshape(-1)
                            target_test_y = np.asarray(target_function(test_X), dtype=float).reshape(-1)

                            global_gp = fit_global_source_gp(
                                train_X,
                                train_y,
                                random_state=seed + source_index,
                            )
                            global_prediction = -np.asarray(
                                global_gp.predict(test_X),
                                dtype=float,
                            ).reshape(-1)

                            geometry = library.geometry_score(test_X)
                            proposed = library.score(
                                test_X,
                                use_reliability=True,
                            )
                            no_reliability = library.score(
                                test_X,
                                use_reliability=False,
                            )
                            permutation_score = null_library.score(test_X)
                            best_distance = best_point_distance_score(
                                train_X,
                                train_y,
                                test_X,
                            )
                            random_score = test_rng.normal(size=len(test_X))

                            methods = {
                                "Proposed-Local-Structure": proposed,
                                "Proposed-No-CV-Weight": no_reliability,
                                "Geometry-Only": geometry,
                                "Global-Source-GP": global_prediction,
                                "Best-Point-Distance": best_distance,
                                "Label-Permutation": permutation_score,
                                "Random-Score": random_score,
                            }
                            local_mask = local_subset_mask(
                                geometry,
                                fraction=local_fraction,
                                minimum_points=max(30, 4 * dim),
                            )

                            for evaluation_domain, true_y in {
                                "source": source_test_y,
                                "target": target_test_y,
                            }.items():
                                for subset_name, subset in {
                                    "all": None,
                                    "local": local_mask,
                                }.items():
                                    for method, score in methods.items():
                                        metrics = evaluate_ranking(
                                            true_y,
                                            score,
                                            top_fraction=top_fraction,
                                            subset_mask=subset,
                                        )
                                        rows.append(
                                            {
                                                **instance,
                                                "evaluation_domain": evaluation_domain,
                                                "subset": subset_name,
                                                "method": method,
                                                **metrics.__dict__,
                                                "n_structures": len(library.structures),
                                                "mean_reliability": float(
                                                    np.mean(
                                                        [
                                                            item.validation.reliability
                                                            for item in library.structures
                                                        ]
                                                    )
                                                ),
                                            }
                                        )

                            source_quality = rank_quality(source_test_y)
                            target_quality = rank_quality(target_test_y)
                            for structure_index, structure in enumerate(library.structures):
                                membership = structure.membership(test_X)
                                component_score = structure.structure_score(test_X)
                                top_count = max(1, int(np.ceil(len(test_X) * top_fraction)))
                                member_top = np.argsort(-membership, kind="stable")[:top_count]
                                structure_top = np.argsort(-component_score, kind="stable")[:top_count]
                                center = structure.center.reshape(1, -1)
                                center_source_y = float(source_function(center)[0])
                                center_target_y = float(target_function(center)[0])
                                structure_rows.append(
                                    {
                                        **instance,
                                        "structure_index": structure_index,
                                        **structure.to_record(),
                                        "center_source_y": center_source_y,
                                        "center_target_y": center_target_y,
                                        "geometry_source_enrichment": float(
                                            np.mean(source_quality[member_top])
                                            - np.mean(source_quality)
                                        ),
                                        "structure_source_enrichment": float(
                                            np.mean(source_quality[structure_top])
                                            - np.mean(source_quality)
                                        ),
                                        "geometry_target_enrichment": float(
                                            np.mean(target_quality[member_top])
                                            - np.mean(target_quality)
                                        ),
                                        "structure_target_enrichment": float(
                                            np.mean(target_quality[structure_top])
                                            - np.mean(target_quality)
                                        ),
                                    }
                                )
                        except Exception as exc:
                            failure_rows.append(
                                {
                                    **instance,
                                    "error_type": type(exc).__name__,
                                    "error": str(exc),
                                }
                            )

                    completed += 1
                    if completed % max(1, total // 20) == 0:
                        print(f"Validation progress: {completed}/{total}")

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "source_structure_validation.csv", index=False)
    pd.DataFrame(structure_rows).to_csv(
        output_dir / "source_structure_diagnostics.csv",
        index=False,
    )
    pd.DataFrame(
        failure_rows,
        columns=[
            "problem", "dim", "seed", "source_scenario",
            "source_index", "error_type", "error",
        ],
    ).to_csv(
        output_dir / "source_structure_validation_failures.csv",
        index=False,
    )

    _write_manifest(
        output_dir / "source_structure_validation_manifest.json",
        config,
        {
            "rows": len(frame),
            "structure_rows": len(structure_rows),
            "failure_rows": len(failure_rows),
        },
    )
    return frame


def _write_manifest(path: Path, config: Mapping, counts: Mapping) -> None:
    encoded = json.dumps(config, sort_keys=True).encode("utf-8")
    payload = {
        "base_commit": BASE_COMMIT,
        "config": config,
        "config_sha256": hashlib.sha256(encoded).hexdigest(),
        "counts": dict(counts),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "source_structure_quick.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "source_structure_stage" / "validation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_validation(load_config(args.config), args.output)
