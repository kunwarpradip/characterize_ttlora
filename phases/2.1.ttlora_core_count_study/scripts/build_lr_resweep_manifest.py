from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "run_specs" not in payload or not isinstance(payload["run_specs"], list):
        raise ValueError(f"Manifest at {path} does not contain a valid run_specs list.")
    return payload


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def format_lr(lr: float) -> str:
    return f"{lr:.0e}".replace("+0", "").replace("+", "")


def parse_selection(spec: str) -> tuple[str, str, set[int]]:
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid --selection '{spec}'. Expected format dataset:variant:core1,core2,..."
        )
    dataset_name = parts[0].strip()
    variant = parts[1].strip()
    core_values = {int(item.strip()) for item in parts[2].split(",") if item.strip()}
    if not dataset_name or not variant or not core_values:
        raise ValueError(
            f"Invalid --selection '{spec}'. Dataset, variant, and at least one core are required."
        )
    return dataset_name, variant, core_values


def matches_any_selection(
    run_spec: dict[str, Any],
    selections: list[tuple[str, str, set[int]]],
) -> bool:
    dataset_name = str(run_spec.get("dataset_name", ""))
    variant = str(run_spec.get("ttlora_variant", ""))
    total_cores = int(run_spec.get("total_cores"))
    for selected_dataset, selected_variant, selected_cores in selections:
        if dataset_name == selected_dataset and variant == selected_variant and total_cores in selected_cores:
            return True
    return False


def replace_arg_value(command: list[str], flag: str, value: str) -> list[str]:
    updated = list(command)
    for index, token in enumerate(updated[:-1]):
        if token == flag:
            updated[index + 1] = value
            return updated
    raise ValueError(f"Command does not contain expected flag {flag}: {' '.join(command)}")


def rewrite_run_name(run_name: str, old_lr: float, new_lr: float) -> str:
    old_fragment = f"lr{format_lr(old_lr)}"
    new_fragment = f"lr{format_lr(new_lr)}"
    if old_fragment in run_name:
        return run_name.replace(old_fragment, new_fragment, 1)
    rewritten, count = re.subn(r"lr[^_]+", new_fragment, run_name, count=1)
    if count:
        return rewritten
    return f"{run_name}_{new_fragment}"


def rewrite_notes(notes: str, old_suite_name: str, new_suite_name: str, source_lr: float, new_lr: float) -> str:
    updated = notes.replace(f"suite={old_suite_name}", f"suite={new_suite_name}")
    updated = re.sub(r"lr=[^ ]+", f"lr={new_lr}", updated)
    suffix = f" resweep_from_lr={source_lr} resweep_to_lr={new_lr}"
    return updated + suffix if suffix.strip() not in updated else updated


def clone_run_spec(
    run_spec: dict[str, Any],
    old_suite_name: str,
    new_suite_name: str,
    new_lr: float,
) -> dict[str, Any]:
    cloned = copy.deepcopy(run_spec)
    source_lr = float(cloned["learning_rate"])
    new_run_name = rewrite_run_name(str(cloned["run_name"]), source_lr, new_lr)

    cloned["suite_name"] = new_suite_name
    cloned["learning_rate"] = new_lr
    cloned["run_name"] = new_run_name
    cloned["command"] = replace_arg_value(list(cloned["command"]), "--run-name", new_run_name)
    cloned["command"] = replace_arg_value(cloned["command"], "--learning-rate", str(new_lr))

    if "--notes" in cloned["command"]:
        notes_index = cloned["command"].index("--notes") + 1
        old_notes = str(cloned["command"][notes_index])
        cloned["command"][notes_index] = rewrite_notes(
            old_notes,
            old_suite_name=old_suite_name,
            new_suite_name=new_suite_name,
            source_lr=source_lr,
            new_lr=new_lr,
        )
    return cloned


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a manifest containing only selected run specs cloned at new learning rates."
    )
    parser.add_argument("--source-manifest", required=True, help="Path to the original manifest.json")
    parser.add_argument("--suite-name", required=True, help="Name of the new suite to write under suites/")
    parser.add_argument(
        "--selection",
        action="append",
        required=True,
        help="Repeatable selection in the form dataset:variant:core1,core2,...",
    )
    parser.add_argument(
        "--learning-rates",
        nargs="+",
        type=float,
        required=True,
        help="One or more replacement learning rates to clone for each selected run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected/cloned run names without writing a manifest.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    source_manifest_path = Path(args.source_manifest).expanduser().resolve()
    source_manifest = load_manifest(source_manifest_path)
    old_suite_name = str(source_manifest["suite_name"])
    new_suite_name = str(args.suite_name)
    selections = [parse_selection(item) for item in args.selection]

    selected_specs = [
        spec for spec in source_manifest["run_specs"] if matches_any_selection(spec, selections)
    ]
    if not selected_specs:
        raise ValueError("No run_specs matched the requested selections.")

    cloned_specs: list[dict[str, Any]] = []
    for spec in selected_specs:
        for learning_rate in args.learning_rates:
            cloned_specs.append(
                clone_run_spec(
                    spec,
                    old_suite_name=old_suite_name,
                    new_suite_name=new_suite_name,
                    new_lr=float(learning_rate),
                )
            )

    cloned_specs.sort(
        key=lambda spec: (
            str(spec.get("dataset_name", "")),
            str(spec.get("ttlora_variant", "")),
            int(spec.get("total_cores", 0)),
            float(spec.get("learning_rate", 0.0)),
            int(spec.get("seed", 0)),
        )
    )

    if args.dry_run:
        print(f"source suite: {old_suite_name}")
        print(f"new suite: {new_suite_name}")
        print(f"selected base runs: {len(selected_specs)}")
        print(f"cloned runs: {len(cloned_specs)}")
        for spec in cloned_specs:
            print(
                f"{spec['run_name']} dataset={spec['dataset_name']} "
                f"variant={spec['ttlora_variant']} cores={spec['total_cores']} lr={spec['learning_rate']}"
            )
        return

    new_manifest = copy.deepcopy(source_manifest)
    new_manifest["suite_name"] = new_suite_name
    new_manifest["learning_rates"] = [float(value) for value in args.learning_rates]
    new_manifest["source_manifest"] = str(source_manifest_path)
    new_manifest["source_suite_name"] = old_suite_name
    new_manifest["num_runs"] = len(cloned_specs)
    new_manifest["run_specs"] = cloned_specs

    output_manifest_path = source_manifest_path.parent.parent / new_suite_name / "manifest.json"
    save_json(output_manifest_path, new_manifest)

    print(f"[source-suite] {old_suite_name}")
    print(f"[new-suite] {new_suite_name}")
    print(f"[selected-base-runs] {len(selected_specs)}")
    print(f"[cloned-runs] {len(cloned_specs)}")
    print(f"[manifest] {output_manifest_path}")


if __name__ == "__main__":
    main()
