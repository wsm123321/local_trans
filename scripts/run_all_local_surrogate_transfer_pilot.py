"""Run Pilot v1 experiment, analysis, and audit as separate processes."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent


def run(command: list[str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=REPO_ROOT)


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


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            str(CURRENT_DIR / "run_local_surrogate_transfer_pilot.py"),
            "--config",
            str(args.config),
            "--output",
            str(args.output),
        ]
    )
    run(
        [
            sys.executable,
            str(CURRENT_DIR / "analyze_local_surrogate_transfer_pilot.py"),
            "--input",
            str(args.output),
            "--config",
            str(args.config),
            "--output",
            str(args.output / "analysis"),
        ]
    )
    run(
        [
            sys.executable,
            str(CURRENT_DIR / "audit_local_surrogate_transfer_pilot.py"),
            "--input",
            str(args.output),
            "--config",
            str(args.config),
        ]
    )


if __name__ == "__main__":
    main()
