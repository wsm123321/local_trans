"""Run all candidate-screening studies and generate the final dynamic report."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(script: str, config: str, output_dir: str | None) -> None:
    command = [sys.executable, str(REPO_ROOT / "scripts" / script), "--config", config]
    if output_dir is not None:
        command.extend(["--output-dir", output_dir])
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "region_screening_full.json"),
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    _run("run_screening_mechanism_study.py", args.config, args.output_dir)
    _run("run_screening_sequential_study.py", args.config, args.output_dir)
    _run("run_screening_drift_study.py", args.config, args.output_dir)
    _run("analyze_screening_studies.py", args.config, args.output_dir)


if __name__ == "__main__":
    main()
