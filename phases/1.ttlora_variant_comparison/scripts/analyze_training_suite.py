from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    phase_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run suite-level and focused TT-LoRA analysis for this phase.")
    parser.add_argument("--suite-name", required=True)
    parser.add_argument("--phase-root", default=str(phase_root))
    parser.add_argument("--python-bin", default=sys.executable)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    phase_root = Path(args.phase_root).expanduser().resolve()
    runs_root = phase_root / "runs"
    output_root = phase_root / "suites"

    commands = [[
        args.python_bin,
        str(phase_root / "scripts" / "analyze_rank_lr_grid.py"),
        "--suite-name",
        args.suite_name,
        "--runs-root",
        str(runs_root),
        "--output-root",
        str(output_root),
    ]]

    for command in commands:
        print("Running:", " ".join(command))
        completed = subprocess.run(command, text=True)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
