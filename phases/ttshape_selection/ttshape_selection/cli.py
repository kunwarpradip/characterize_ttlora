from __future__ import annotations

import argparse
import json
from pathlib import Path

from .generator import generate_catalog, load_weight_specs


def _parse_int_list(values: list[str] | None) -> tuple[int, ...] | None:
    if not values:
        return None
    parsed: list[int] = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                parsed.append(int(item))
    return tuple(parsed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate TT-shape candidates for one or more model weights.")
    parser.add_argument("--weights-json", required=True, help="JSON file containing a 'weights' list.")
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--core-counts", nargs="*", default=None, help="Optional core counts, e.g. 2 3 4 or 2,3,4.")
    parser.add_argument("--split-strategy", default="all", choices=("all", "symmetric", "near-symmetric"))
    parser.add_argument("--allow-one-factors", action="store_true")
    parser.add_argument("--top-k-per-weight", type=int, default=None)
    parser.add_argument("--output-json", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    weights = load_weight_specs(args.weights_json)
    catalog = generate_catalog(
        weights,
        rank=args.rank,
        core_counts=_parse_int_list(args.core_counts),
        split_strategy=args.split_strategy,
        allow_one_factors=args.allow_one_factors,
        top_k_per_weight=args.top_k_per_weight,
    )
    payload = catalog.to_dict()
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json:
        output_path = Path(args.output_json).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
