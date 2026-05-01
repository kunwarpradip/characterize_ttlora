from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    run_specs = payload.get("run_specs")
    if not isinstance(run_specs, list):
        raise ValueError(f"Manifest {path} does not contain a valid run_specs list.")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple suite manifests into one manifest.")
    parser.add_argument("--manifest-path", action="append", required=True, help="Repeatable input manifest path.")
    parser.add_argument("--output-path", required=True, help="Path to the merged manifest JSON.")
    parser.add_argument("--suite-name", required=True, help="Suite name to write into the merged manifest.")
    args = parser.parse_args()

    manifest_paths = [Path(item).expanduser().resolve() for item in args.manifest_path]
    manifests = [load_manifest(path) for path in manifest_paths]

    merged_run_specs: list[dict] = []
    seen_run_names: set[str] = set()
    for manifest in manifests:
        for spec in manifest["run_specs"]:
            run_name = str(spec.get("run_name", "")).strip()
            if not run_name:
                raise ValueError("Encountered a run_spec without run_name while merging manifests.")
            if run_name in seen_run_names:
                raise ValueError(f"Duplicate run_name encountered while merging manifests: {run_name}")
            seen_run_names.add(run_name)
            merged_run_specs.append(spec)

    base_manifest = dict(manifests[0])
    base_manifest["suite_name"] = args.suite_name
    base_manifest["num_runs"] = len(merged_run_specs)
    base_manifest["run_specs"] = merged_run_specs
    base_manifest["merged_from_manifests"] = [str(path) for path in manifest_paths]
    base_manifest["merged_rank_values"] = sorted(
        {
            spec.get("ttlora_rank")
            for spec in merged_run_specs
            if spec.get("ttlora_rank") is not None
        }
    )

    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(base_manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote merged manifest to {output_path}")
    print(f"Merged {len(manifest_paths)} manifests and {len(merged_run_specs)} run_specs.")


if __name__ == "__main__":
    main()
