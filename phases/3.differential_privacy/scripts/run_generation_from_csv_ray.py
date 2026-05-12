from __future__ import annotations

import argparse
import csv
import os
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import ray

from run_generation_from_csv import (
    add_compact_default_args,
    build_command,
    expand_row_defaults,
    row_is_enabled,
    run_dir_for_row,
)


def parse_args() -> argparse.Namespace:
    project_root = Path("/home/pkunwar/characterize_ttlora")
    parser = argparse.ArgumentParser(
        description="Submit train_generation.py runs from a CSV file to a Ray cluster."
    )
    parser.add_argument("--csv-path", required=True, help="Path to the input CSV file.")
    parser.add_argument("--ray-address", default="auto", help="Ray cluster address. Use 'auto' for an existing cluster.")
    parser.add_argument("--project-root", default=str(project_root))
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--train-script", default=None)
    parser.add_argument("--cpus-per-run", type=float, default=4.0)
    parser.add_argument("--gpus-per-run", type=float, default=1.0)
    parser.add_argument("--only-run-name", action="append", default=None)
    parser.add_argument(
        "--skip-completed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip rows whose output_dir/run_name/summary.json already exists.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execution-log-name", default="ray_execution_log.csv")
    add_compact_default_args(parser, project_root)
    return parser.parse_args()


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    return rows


def write_execution_log(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@ray.remote
def run_command_on_ray(
    command: list[str],
    *,
    cwd: str,
    run_name: str,
    stdout_path: str,
    stderr_path: str,
) -> dict[str, Any]:
    stdout_file = Path(stdout_path)
    stderr_file = Path(stderr_path)
    stdout_file.parent.mkdir(parents=True, exist_ok=True)
    stderr_file.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    with stdout_file.open("w", encoding="utf-8") as stdout_handle, stderr_file.open("w", encoding="utf-8") as stderr_handle:
        process = subprocess.run(
            command,
            cwd=cwd,
            stdout=stdout_handle,
            stderr=stderr_handle,
            check=False,
            env=os.environ.copy(),
        )
    elapsed = time.time() - start

    try:
        accelerator_ids = ray.get_runtime_context().get_accelerator_ids().get("GPU", [])
    except Exception:
        accelerator_ids = []

    stderr_tail = ""
    try:
        lines = stderr_file.read_text(encoding="utf-8", errors="replace").splitlines()
        stderr_tail = "\n".join(lines[-40:])
    except OSError:
        pass

    return {
        "run_name": run_name,
        "returncode": int(process.returncode),
        "elapsed_seconds": elapsed,
        "hostname": socket.gethostname(),
        "node_ip": ray.util.get_node_ip_address(),
        "gpu_ids": ",".join(str(item) for item in accelerator_ids),
        "stdout_path": str(stdout_file),
        "stderr_path": str(stderr_file),
        "stderr_tail": stderr_tail,
    }


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path).expanduser().resolve()
    project_root = Path(args.project_root).expanduser().resolve()
    train_script = (
        Path(args.train_script).expanduser().resolve()
        if args.train_script is not None
        else project_root / "train_generation.py"
    )
    rows = read_rows(csv_path)
    selected_run_names = set(args.only_run_name or [])

    runnable: list[tuple[str, list[str], Path | None]] = []
    for index, row in enumerate(rows, start=1):
        expanded_row = expand_row_defaults(row, args, project_root)
        run_name = str(expanded_row.get("run_name", "")).strip() or f"row_{index}"
        if not row_is_enabled(expanded_row):
            print(f"[skip-disabled] {run_name}", flush=True)
            continue
        if selected_run_names and run_name not in selected_run_names:
            continue
        run_dir = run_dir_for_row(expanded_row)
        if args.skip_completed and run_dir is not None and (run_dir / "summary.json").exists():
            print(f"[skip-completed] {run_name}", flush=True)
            continue
        command = build_command(
            expanded_row,
            python_executable=args.python_executable,
            train_script=train_script,
        )
        print(f"[run] {run_name}", flush=True)
        print(" ".join(shlex.quote(part) for part in command), flush=True)
        runnable.append((run_name, command, run_dir))

    if args.dry_run:
        print(f"[summary] rows={len(rows)} runnable={len(runnable)} dry_run=true", flush=True)
        return

    logs_dir = csv_path.parent / f"{csv_path.stem}_ray_logs"
    execution_log_path = csv_path.parent / args.execution_log_name
    logs_dir.mkdir(parents=True, exist_ok=True)

    ray.init(address=args.ray_address, ignore_reinit_error=True)
    remote_runner = run_command_on_ray.options(num_cpus=args.cpus_per_run, num_gpus=args.gpus_per_run)

    futures = []
    for run_name, command, _run_dir in runnable:
        stdout_path = logs_dir / f"{run_name}.stdout.log"
        stderr_path = logs_dir / f"{run_name}.stderr.log"
        futures.append(
            remote_runner.remote(
                command,
                cwd=str(project_root),
                run_name=run_name,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
            )
        )

    print(f"[runs] submitting {len(futures)} run(s) to Ray", flush=True)
    results = ray.get(futures)
    write_execution_log(execution_log_path, results)
    print(f"[execution-log] {execution_log_path}", flush=True)

    failed = [result for result in results if int(result["returncode"]) != 0]
    for result in results:
        print(
            f"[done] {result['run_name']} rc={result['returncode']} time={result['elapsed_seconds']:.1f}s "
            f"host={result['hostname']} gpus=[{result['gpu_ids']}]",
            flush=True,
        )
        if int(result["returncode"]) != 0 and result.get("stderr_tail"):
            print("[stderr-tail]", flush=True)
            print(result["stderr_tail"], flush=True)

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
