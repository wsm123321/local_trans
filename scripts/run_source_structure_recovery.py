"""Controlled ground-truth recovery study for source local structures."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from region_guided_reranking_study.source_local_structure import (  # noqa: E402
    LocalStructureConfig,
    SourceLocalStructureExtractor,
)
from region_guided_reranking_study.source_structure_research import (  # noqa: E402
    generate_controlled_landscape,
    latin_hypercube_sample,
    random_center_baseline,
    recovery_metrics,
    recovery_metrics_from_arrays,
    top_observation_baseline,
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


def run_recovery(config: Mapping, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    recovery_cfg = dict(config["recovery"])
    dimensions = [int(value) for value in recovery_cfg["dimensions"]]
    seeds = [int(value) for value in recovery_cfg["seeds"]]
    sample_sizes = [int(value) for value in recovery_cfg["sample_sizes"]]
    noise_levels = [float(value) for value in recovery_cfg["noise_levels"]]
    n_basins = int(recovery_cfg.get("n_basins", 3))
    hit_threshold = float(recovery_cfg.get("hit_threshold", 2.5))

    rows: List[Dict] = []
    diagnostic_rows: List[Dict] = []
    failure_rows: List[Dict] = []

    total = len(dimensions) * len(seeds) * len(sample_sizes) * len(noise_levels)
    completed = 0

    for dim in dimensions:
        for seed in seeds:
            seed_sequence = np.random.SeedSequence(seed)
            landscape_ss, noise_ss, random_ss = seed_sequence.spawn(3)
            landscape_rng = np.random.default_rng(landscape_ss)
            noise_rng = np.random.default_rng(noise_ss)
            random_rng = np.random.default_rng(random_ss)
            landscape = generate_controlled_landscape(
                dim=dim,
                n_basins=n_basins,
                rng=landscape_rng,
                noise_std=0.0,
            )
            oracle = landscape.oracle_structures()
            oracle_centers = [item["center"] for item in oracle]
            oracle_covariances = [item["covariance"] for item in oracle]

            for sample_size in sample_sizes:
                train_X = latin_hypercube_sample(
                    landscape.bounds,
                    n_samples=sample_size,
                    seed=seed + 1009 * sample_size + 37 * dim,
                )
                noiseless_y = landscape(train_X)

                for noise_level in noise_levels:
                    train_y = noiseless_y + noise_rng.normal(
                        0.0,
                        noise_level,
                        size=sample_size,
                    )
                    instance_key = {
                        "dim": dim,
                        "seed": seed,
                        "sample_size": sample_size,
                        "noise_level": noise_level,
                    }
                    try:
                        extractor = SourceLocalStructureExtractor(
                            make_extraction_config(config, seed)
                        )
                        library = extractor.fit_dataset(
                            train_X,
                            train_y,
                            task_id=f"controlled_d{dim}_s{seed}",
                        )

                        proposed = recovery_metrics(
                            library.structures,
                            oracle_centers,
                            oracle_covariances,
                            landscape.bounds,
                            method="Proposed-Local-Structure",
                            hit_threshold=hit_threshold,
                        )
                        rows.append({**instance_key, **proposed.__dict__})

                        default_covariance = np.diag(
                            np.maximum(np.var(train_X, axis=0) * 0.05, 1e-4)
                        )
                        top_centers, top_covariances = top_observation_baseline(
                            train_X,
                            train_y,
                            n_centers=n_basins,
                            default_covariance=default_covariance,
                        )
                        top_metrics = recovery_metrics_from_arrays(
                            top_centers,
                            top_covariances,
                            oracle_centers,
                            oracle_covariances,
                            landscape.bounds,
                            method="Top-Observations",
                            hit_threshold=hit_threshold,
                        )
                        rows.append({**instance_key, **top_metrics.__dict__})

                        random_centers, random_covariances = random_center_baseline(
                            landscape.bounds,
                            n_centers=n_basins,
                            dim=dim,
                            rng=random_rng,
                        )
                        random_metrics = recovery_metrics_from_arrays(
                            random_centers,
                            random_covariances,
                            oracle_centers,
                            oracle_covariances,
                            landscape.bounds,
                            method="Random-Centers",
                            hit_threshold=hit_threshold,
                        )
                        rows.append({**instance_key, **random_metrics.__dict__})

                        for structure in library.structures:
                            diagnostic_rows.append(
                                {
                                    **instance_key,
                                    **structure.to_record(),
                                    "center_true_y": float(
                                        landscape(structure.center.reshape(1, -1))[0]
                                    ),
                                }
                            )
                    except Exception as exc:
                        failure_rows.append(
                            {
                                **instance_key,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                        )

                    completed += 1
                    if completed % max(1, total // 20) == 0:
                        print(f"Recovery progress: {completed}/{total}")

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "source_structure_recovery.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(
        output_dir / "source_structure_recovery_diagnostics.csv",
        index=False,
    )
    pd.DataFrame(
        failure_rows,
        columns=[
            "dim", "seed", "sample_size", "noise_level",
            "error_type", "error",
        ],
    ).to_csv(
        output_dir / "source_structure_recovery_failures.csv",
        index=False,
    )

    _write_manifest(
        output_dir / "source_structure_recovery_manifest.json",
        config,
        {
            "rows": len(frame),
            "diagnostic_rows": len(diagnostic_rows),
            "failure_rows": len(failure_rows),
        },
    )
    return frame


def _write_manifest(
    path: Path,
    config: Mapping,
    counts: Mapping,
) -> None:
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
        default=REPO_ROOT / "results" / "source_structure_stage" / "recovery",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_recovery(load_config(args.config), args.output)
