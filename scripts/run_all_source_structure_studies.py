"""Run recovery, held-out validation, and data-driven analysis."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent


def run(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True, cwd=REPO_ROOT)


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
        default=REPO_ROOT / "results" / "source_structure_stage",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            str(CURRENT_DIR / "run_source_structure_recovery.py"),
            "--config",
            str(args.config),
            "--output",
            str(args.output / "recovery"),
        ]
    )
    run(
        [
            sys.executable,
            str(CURRENT_DIR / "run_source_structure_validation.py"),
            "--config",
            str(args.config),
            "--output",
            str(args.output / "validation"),
        ]
    )
    run(
        [
            sys.executable,
            str(CURRENT_DIR / "analyze_source_structure_study.py"),
            "--input",
            str(args.output),
            "--output",
            str(args.output / "analysis"),
        ]
    )


if __name__ == "__main__":
    main()
