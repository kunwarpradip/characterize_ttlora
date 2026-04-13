from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "run_specs" not in payload or not isinstance(payload["run_specs"], list):
        raise ValueError(f"Manifest at {path} does not contain a valid run_specs list.")
    return payload


def load_execution_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def sanitize_command_for_ray(command: list[str]) -> list[str]:
    sanitized: list[str] = []
    skip_next = False
    for index, token in enumerate(command):
        if skip_next:
            skip_next = False
            continue
        if token in {"--gpu-id", "--device"}:
            skip_next = True
            continue
        sanitized.append(token)
    sanitized.extend(["--device", "cuda"])
    return sanitized


def should_skip_run(
    run_name: str,
    existing_rows: list[dict[str, Any]],
    resume: bool,
    rerun_failed: bool,
) -> bool:
    if not resume:
        return False
    matching_rows = [row for row in existing_rows if row.get("run_name") == run_name]
    if not matching_rows:
        return False
    latest = matching_rows[-1]
    returncode = str(latest.get("returncode", ""))
    if returncode == "0":
        return True
    return not rerun_failed


def read_summary(spec: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    summary_path = Path(spec["dataset_runs_root"]) / spec["run_name"] / "summary.json"
    if not summary_path.exists():
        return summary_path, {}
    try:
        return summary_path, json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return summary_path, {}


def build_result_row(
    manifest: dict[str, Any],
    spec: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    summary_path, summary = read_summary(spec)
    return {
        "run_name": spec["run_name"],
        "suite_name": manifest["suite_name"],
        "dataset_name": spec.get("dataset_name"),
        "task_type": spec.get("task_type"),
        "ttlora_variant": spec.get("ttlora_variant"),
        "ttlora_rank": spec.get("ttlora_rank"),
        "ttlora_alpha": spec.get("ttlora_alpha"),
        "learning_rate": spec.get("learning_rate"),
        "seed": spec.get("seed"),
        "total_cores": spec.get("total_cores"),
        "input_cores": spec.get("input_cores"),
        "output_cores": spec.get("output_cores"),
        "input_factors": json.dumps(spec.get("input_factors", [])),
        "output_factors": json.dumps(spec.get("output_factors", [])),
        "tt_shape": json.dumps(spec.get("tt_shape", [])),
        "generation_weight_candidates": json.dumps(spec.get("generation_weight_candidates", {})),
        "returncode": result["returncode"],
        "elapsed_seconds": result["elapsed_seconds"],
        "node_ip": result["node_ip"],
        "hostname": result["hostname"],
        "gpu_ids": json.dumps(result["gpu_ids"]),
        "stdout_path": result["stdout_path"],
        "stderr_path": result["stderr_path"],
        "summary_path": str(summary_path) if summary_path.exists() else "",
        "history_path": summary.get("history_path", ""),
        "step_history_path": summary.get("step_history_path", ""),
        "best_checkpoint_dir": summary.get("best_checkpoint_dir", ""),
        "best_epoch": summary.get("best_epoch"),
        "epochs_ran": summary.get("epochs_ran"),
        "best_validation_accuracy": summary.get("best_validation_accuracy"),
        "best_validation_loss": summary.get("best_validation_loss"),
        "best_validation_perplexity": summary.get("best_validation_perplexity"),
        "best_validation_token_accuracy": summary.get("best_validation_token_accuracy"),
        "final_validation_accuracy": summary.get("final_validation_accuracy"),
        "initial_validation_accuracy": summary.get("initial_validation_accuracy"),
        "initial_validation_loss": summary.get("initial_validation_loss"),
        "initial_validation_token_accuracy": summary.get("initial_validation_token_accuracy"),
        "final_validation_loss": summary.get("final_validation_loss"),
        "initial_validation_perplexity": summary.get("initial_validation_perplexity"),
        "final_validation_perplexity": summary.get("final_validation_perplexity"),
        "final_validation_token_accuracy": summary.get("final_validation_token_accuracy"),
        "max_peak_memory_gb": summary.get("max_peak_memory_gb"),
        "avg_epoch_seconds": summary.get("avg_epoch_seconds"),
        "avg_grad_norm": summary.get("avg_grad_norm"),
        "max_grad_norm": summary.get("max_grad_norm"),
        "min_grad_norm": summary.get("min_grad_norm"),
        "avg_clipped_step_fraction": summary.get("avg_clipped_step_fraction"),
        "total_clipped_steps": summary.get("total_clipped_steps"),
        "step_metrics_logged": summary.get("step_metrics_logged"),
        "trainable_parameters": summary.get("trainable_parameters"),
        "frozen_parameters": summary.get("frozen_parameters"),
        "total_parameters": summary.get("total_parameters"),
        "resolved_run_dir": summary.get("resolved_run_dir", ""),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a core-count study suite manifest on a Ray cluster with one GPU per run."
    )
    parser.add_argument("--manifest-path", required=True, help="Path to a suite manifest.json file.")
    parser.add_argument(
        "--ray-address",
        default="auto",
        help="Ray cluster address. Use 'auto' for an already-started cluster.",
    )
    parser.add_argument(
        "--cpus-per-run",
        type=float,
        default=4.0,
        help="CPU resources to reserve for each training run.",
    )
    parser.add_argument(
        "--gpus-per-run",
        type=float,
        default=1.0,
        help="GPU resources to reserve for each training run.",
    )
    parser.add_argument(
        "--log-name",
        default="ray_execution_log.csv",
        help="Name of the Ray execution log written under the suite directory.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip runs already marked successful in the Ray execution log.",
    )
    parser.add_argument(
        "--rerun-failed",
        action="store_true",
        help="With --resume, rerun failed entries instead of skipping them.",
    )
    parser.add_argument(
        "--include-datasets",
        nargs="*",
        default=None,
        help="Optional dataset filter applied to manifest run_specs.",
    )
    parser.add_argument(
        "--include-variants",
        nargs="*",
        default=None,
        help="Optional variant filter applied to manifest run_specs.",
    )
    parser.add_argument(
        "--include-core-counts",
        nargs="*",
        type=int,
        default=None,
        help="Optional total core-count filter applied to manifest run_specs.",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop after the first failed remote run result is returned.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print filtered runs without submitting them to Ray.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    manifest_path = Path(args.manifest_path).expanduser().resolve()
    manifest = load_manifest(manifest_path)
    suite_dir = manifest_path.parent
    logs_dir = suite_dir / "ray_task_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    execution_log_path = suite_dir / args.log_name
    execution_rows = load_execution_log(execution_log_path)

    include_datasets = set(args.include_datasets or [])
    include_variants = set(args.include_variants or [])
    include_core_counts = {int(value) for value in args.include_core_counts or []}

    filtered_specs: list[dict[str, Any]] = []
    for spec in manifest["run_specs"]:
        if include_datasets and spec.get("dataset_name") not in include_datasets:
            continue
        if include_variants and spec.get("ttlora_variant") not in include_variants:
            continue
        if include_core_counts and int(spec.get("total_cores")) not in include_core_counts:
            continue
        if should_skip_run(spec["run_name"], execution_rows, args.resume, args.rerun_failed):
            print(f"[resume] Skipping completed run {spec['run_name']}", flush=True)
            continue
        filtered_specs.append(spec)

    if not filtered_specs:
        print("No run_specs remain after filtering/resume checks.", flush=True)
        return

    print(f"[suite] {manifest['suite_name']}", flush=True)
    print(f"[manifest] {manifest_path}", flush=True)
    print(f"[runs] submitting {len(filtered_specs)} run(s) to Ray", flush=True)

    if args.dry_run:
        for spec in filtered_specs:
            print(f"[dry-run] {spec['run_name']}", flush=True)
            print(" ".join(sanitize_command_for_ray(spec["command"])), flush=True)
        return

    import ray

    ray.init(address=args.ray_address, ignore_reinit_error=True, log_to_driver=True)

    @ray.remote(num_cpus=args.cpus_per_run, num_gpus=args.gpus_per_run)
    def run_one(spec: dict[str, Any], logs_dir_str: str) -> dict[str, Any]:
        command = sanitize_command_for_ray(list(spec["command"]))
        logs_dir = Path(logs_dir_str)
        stdout_path = logs_dir / f"{spec['run_name']}.stdout.log"
        stderr_path = logs_dir / f"{spec['run_name']}.stderr.log"

        env = os.environ.copy()
        spec_env = spec.get("env")
        if isinstance(spec_env, dict):
            env.update({str(key): str(value) for key, value in spec_env.items()})
        env["PYTHONUNBUFFERED"] = "1"

        started_at = time.time()
        with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_handle:
            result = subprocess.run(
                command,
                cwd=spec.get("cwd"),
                env=env,
                stdout=stdout_handle,
                stderr=stderr_handle,
                check=False,
            )
        elapsed_seconds = time.time() - started_at

        try:
            node_ip = ray.util.get_node_ip_address()
        except Exception:
            node_ip = ""

        return {
            "run_name": spec["run_name"],
            "returncode": result.returncode,
            "elapsed_seconds": elapsed_seconds,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "gpu_ids": list(ray.get_gpu_ids()),
            "hostname": socket.gethostname(),
            "node_ip": node_ip,
        }

    pending = {
        run_one.remote(spec, str(logs_dir)): spec
        for spec in filtered_specs
    }

    while pending:
        ready_refs, _ = ray.wait(list(pending.keys()), num_returns=1)
        ready_ref = ready_refs[0]
        spec = pending.pop(ready_ref)
        result = ray.get(ready_ref)

        print(
            f"[done] {spec['run_name']} rc={result['returncode']} "
            f"time={result['elapsed_seconds']:.1f}s host={result['hostname']} gpus={result['gpu_ids']}",
            flush=True,
        )

        execution_rows.append(build_result_row(manifest, spec, result))
        write_execution_log(execution_log_path, execution_rows)

        if result["returncode"] != 0:
            print(f"[run-failed] {spec['run_name']} exited with return code {result['returncode']}", flush=True)
            if args.stop_on_failure:
                for ref in pending:
                    ray.cancel(ref, force=True)
                raise SystemExit(result["returncode"])


if __name__ == "__main__":
    main()
