"""Run ARISE experiments and data-driven analysis."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "arise_quick.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "arise_stage",
    )
    args = parser.parse_args()

    subprocess.run(
        [
            sys.executable,
            str(CURRENT_DIR / "run_arise_similarity_study.py"),
            "--config",
            str(args.config),
            "--output",
            str(args.output),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(CURRENT_DIR / "analyze_arise_study.py"),
            "--input",
            str(args.output),
            "--output",
            str(args.output / "analysis"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
