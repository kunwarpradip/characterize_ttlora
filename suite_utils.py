from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def read_csv_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [{key: parse_scalar(value) for key, value in row.items()} for row in reader]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class RunSummaryBundle:
    run_dir: Path
    summary: dict[str, Any]
    config: dict[str, Any]
    history: list[dict[str, Any]]
    step_history: list[dict[str, Any]]


def load_run_bundle(run_dir: Path) -> RunSummaryBundle:
    summary = load_json(run_dir / "summary.json")
    config = load_json(run_dir / "config.json")
    history = read_csv_records(run_dir / "history.csv")
    step_path = run_dir / "step_history.csv"
    step_history = read_csv_records(step_path) if step_path.exists() else []
    return RunSummaryBundle(
        run_dir=run_dir,
        summary=summary,
        config=config,
        history=history,
        step_history=step_history,
    )
