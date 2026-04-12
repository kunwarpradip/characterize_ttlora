from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class TTShapeCandidate:
    total_cores: int
    input_cores: int
    output_cores: int
    input_factors: tuple[int, ...]
    output_factors: tuple[int, ...]
    tt_shape: tuple[int, ...]
    balance_score: float
    max_factor: int
    min_factor: int
    uses_one_factor: bool


@dataclass(frozen=True)
class TTShapeCount:
    total_cores: int
    input_cores: int
    output_cores: int
    num_input_factorizations: int
    num_output_factorizations: int
    num_input_orderings: int
    num_output_orderings: int
    num_tt_shapes: int


@dataclass(frozen=True)
class MergeLineageCandidate:
    total_cores: int
    input_cores: int
    output_cores: int
    input_factors: tuple[int, ...]
    output_factors: tuple[int, ...]
    tt_shape: tuple[int, ...]


def _safe_log(value: int) -> float:
    return math.log(float(value))


def balance_score(factors: tuple[int, ...]) -> float:
    """Lower is better. Penalize skewed factorizations and factors of 1."""
    logs = [_safe_log(max(1, value)) for value in factors]
    mean = sum(logs) / len(logs)
    variance = sum((value - mean) ** 2 for value in logs) / len(logs)
    one_penalty = sum(1 for value in factors if value == 1) * 10.0
    spread_penalty = math.log(max(factors) / min(factors)) if min(factors) > 0 else 0.0
    return variance + spread_penalty + one_penalty


@lru_cache(maxsize=None)
def ordered_factorizations(n: int, parts: int, min_factor: int) -> tuple[tuple[int, ...], ...]:
    """
    Generate non-decreasing multiplicative factorizations of n into `parts` terms.

    Factors are constrained to be >= min_factor. Use min_factor=1 to allow
    singleton factors.
    """
    if parts == 1:
        if n >= min_factor:
            return ((n,),)
        return tuple()

    results: list[tuple[int, ...]] = []
    upper = int(round(n ** (1 / parts))) + 1
    for factor in range(min_factor, max(min_factor, upper) + 1):
        if n % factor != 0:
            continue
        for suffix in ordered_factorizations(n // factor, parts - 1, factor):
            results.append((factor, *suffix))
    return tuple(results)


@lru_cache(maxsize=None)
def distinct_factor_orderings(factors: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Return all unique orderings of a factor tuple."""
    if len(factors) <= 1:
        return (factors,)

    remaining: dict[int, int] = {}
    for factor in factors:
        remaining[factor] = remaining.get(factor, 0) + 1

    ordered_values = tuple(sorted(remaining))
    current: list[int] = []
    results: list[tuple[int, ...]] = []

    def backtrack() -> None:
        if len(current) == len(factors):
            results.append(tuple(current))
            return
        for value in ordered_values:
            count = remaining.get(value, 0)
            if count == 0:
                continue
            remaining[value] = count - 1
            current.append(value)
            backtrack()
            current.pop()
            remaining[value] = count

    backtrack()
    return tuple(results)


def candidate_splits(total_cores: int, strategy: str) -> list[tuple[int, int]]:
    if total_cores < 2:
        raise ValueError("Total core count must be at least 2.")

    if strategy == "all":
        return [(input_cores, total_cores - input_cores) for input_cores in range(1, total_cores)]

    if strategy == "symmetric":
        if total_cores % 2 != 0:
            raise ValueError(
                "Symmetric split requires an even total core count so input/output cores match."
            )
        return [(total_cores // 2, total_cores // 2)]

    if strategy == "near-symmetric":
        left = total_cores // 2
        right = total_cores - left
        return [(left, right)] if left != right else [(left, right)]

    raise ValueError(f"Unsupported split strategy: {strategy}")


def prime_factors(n: int) -> list[int]:
    factors: list[int] = []
    divisor = 2
    remaining = n
    while divisor * divisor <= remaining:
        while remaining % divisor == 0:
            factors.append(divisor)
            remaining //= divisor
        divisor += 1
    if remaining > 1:
        factors.append(remaining)
    return factors


def merge_smallest_factors_desc(factors: tuple[int, ...]) -> tuple[int, ...]:
    if len(factors) < 2:
        raise ValueError("Need at least two factors to merge.")
    working = list(factors)
    smallest = working.pop()
    second_smallest = working.pop()
    working.append(smallest * second_smallest)
    working.sort(reverse=True)
    return tuple(working)


def build_merge_lineage(n: int) -> list[tuple[int, ...]]:
    """
    Build one canonical factorization lineage for n.

    Start from the full prime-factor decomposition sorted descending, then
    repeatedly merge the two smallest factors and re-sort descending until a
    single factor remains.
    """
    current = tuple(sorted(prime_factors(n), reverse=True))
    lineage = [current]
    while len(current) > 1:
        current = merge_smallest_factors_desc(current)
        lineage.append(current)
    return lineage


def generate_merge_lineage_candidates(
    in_features: int,
    out_features: int,
    split_strategy: str,
) -> list[MergeLineageCandidate]:
    if split_strategy != "symmetric":
        raise ValueError("Merge-lineage mode currently supports symmetric input/output core splits only.")

    input_lineage = {len(factors): factors for factors in build_merge_lineage(in_features)}
    output_lineage = {len(factors): factors for factors in build_merge_lineage(out_features)}

    shared_lengths = sorted(set(input_lineage).intersection(output_lineage), reverse=True)
    candidates: list[MergeLineageCandidate] = []
    for length in shared_lengths:
        input_factors = input_lineage[length]
        output_factors = output_lineage[length]
        candidates.append(
            MergeLineageCandidate(
                total_cores=length * 2,
                input_cores=length,
                output_cores=length,
                input_factors=input_factors,
                output_factors=output_factors,
                tt_shape=(*input_factors, *reversed(output_factors)),
            )
        )
    return candidates


def possible_factor_counts(n: int, allow_one_factors: bool) -> list[int]:
    """
    Return all factorization lengths that are feasible for n.

    By default we disallow factors of 1, so the maximum length is limited by the
    prime-factor count of n. If factors of 1 are allowed, this set is not very
    meaningful because the number of factors can be extended arbitrarily by
    adding ones, so inference of core counts is disabled in that mode.
    """
    if allow_one_factors:
        raise ValueError(
            "Automatic core-count inference is not supported when --allow-one-factors is enabled."
        )

    counts: list[int] = []
    parts = 1
    while True:
        factorizations = ordered_factorizations(n, parts, 2)
        if not factorizations:
            break
        counts.append(parts)
        parts += 1
    return counts


def infer_core_counts(
    in_features: int,
    out_features: int,
    split_strategy: str,
    allow_one_factors: bool,
) -> tuple[int, ...]:
    input_counts = possible_factor_counts(in_features, allow_one_factors)
    output_counts = possible_factor_counts(out_features, allow_one_factors)

    inferred: set[int] = set()
    if split_strategy == "symmetric":
        for count in sorted(set(input_counts).intersection(output_counts)):
            inferred.add(count * 2)
    elif split_strategy == "near-symmetric":
        for input_cores in input_counts:
            for output_cores in output_counts:
                if abs(input_cores - output_cores) <= 1:
                    inferred.add(input_cores + output_cores)
    elif split_strategy == "all":
        for input_cores in input_counts:
            for output_cores in output_counts:
                inferred.add(input_cores + output_cores)
    else:
        raise ValueError(f"Unsupported split strategy: {split_strategy}")

    return tuple(sorted(inferred))


def count_tt_shapes(
    in_features: int,
    out_features: int,
    core_counts: tuple[int, ...],
    split_strategy: str,
    allow_one_factors: bool,
) -> list[TTShapeCount]:
    min_factor = 1 if allow_one_factors else 2
    counts: list[TTShapeCount] = []

    for total_cores in core_counts:
        for input_cores, output_cores in candidate_splits(total_cores, split_strategy):
            input_factorizations = ordered_factorizations(in_features, input_cores, min_factor)
            output_factorizations = ordered_factorizations(out_features, output_cores, min_factor)
            input_orderings = {factors: distinct_factor_orderings(factors) for factors in input_factorizations}
            output_orderings = {factors: distinct_factor_orderings(factors) for factors in output_factorizations}
            num_input_orderings = sum(len(orderings) for orderings in input_orderings.values())
            num_output_orderings = sum(len(orderings) for orderings in output_orderings.values())
            num_tt_shapes = sum(
                len(input_orderings[input_factors]) * len(output_orderings[output_factors])
                for input_factors in input_factorizations
                for output_factors in output_factorizations
            )
            counts.append(
                TTShapeCount(
                    total_cores=total_cores,
                    input_cores=input_cores,
                    output_cores=output_cores,
                    num_input_factorizations=len(input_factorizations),
                    num_output_factorizations=len(output_factorizations),
                    num_input_orderings=num_input_orderings,
                    num_output_orderings=num_output_orderings,
                    num_tt_shapes=num_tt_shapes,
                )
            )

    return counts


def generate_tt_shape_candidates(
    in_features: int,
    out_features: int,
    core_counts: tuple[int, ...],
    split_strategy: str,
    top_k: int,
    allow_one_factors: bool,
) -> list[TTShapeCandidate]:
    min_factor = 1 if allow_one_factors else 2
    candidates: list[TTShapeCandidate] = []

    for total_cores in core_counts:
        for input_cores, output_cores in candidate_splits(total_cores, split_strategy):
            input_factorizations = ordered_factorizations(in_features, input_cores, min_factor)
            output_factorizations = ordered_factorizations(out_features, output_cores, min_factor)

            local_candidates: list[TTShapeCandidate] = []
            for input_factors in input_factorizations:
                for output_factors in output_factorizations:
                    all_factors = (*input_factors, *output_factors)
                    ordering_score = balance_score(all_factors)
                    for input_ordering in distinct_factor_orderings(input_factors):
                        for output_ordering in distinct_factor_orderings(output_factors):
                            tt_shape = (*input_ordering, *reversed(output_ordering))
                            candidate = TTShapeCandidate(
                                total_cores=total_cores,
                                input_cores=input_cores,
                                output_cores=output_cores,
                                input_factors=input_ordering,
                                output_factors=output_ordering,
                                tt_shape=tt_shape,
                                balance_score=ordering_score,
                                max_factor=max(all_factors),
                                min_factor=min(all_factors),
                                uses_one_factor=any(value == 1 for value in all_factors),
                            )
                            local_candidates.append(candidate)

            local_candidates.sort(key=lambda item: (item.balance_score, item.max_factor, item.tt_shape))
            candidates.extend(local_candidates[:top_k])

    return candidates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate valid TT-shape candidates for studying the effect of number of TT cores."
    )
    parser.add_argument("--in-features", type=int, default=768)
    parser.add_argument("--out-features", type=int, default=768)
    parser.add_argument(
        "--core-counts",
        nargs="+",
        type=int,
        default=None,
        help="Total TT core counts to enumerate. If omitted, infer all feasible counts first.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=("symmetric", "near-symmetric", "all"),
        default="symmetric",
        help="How to split the total number of cores between input and output factorizations.",
    )
    parser.add_argument(
        "--family-mode",
        choices=("balanced", "merge-lineage"),
        default="balanced",
        help="Shape family generation mode. 'balanced' enumerates many candidates; "
        "'merge-lineage' builds one canonical shape per core count by merging the two smallest factors.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Keep only the top-k most balanced shapes per core split.",
    )
    parser.add_argument(
        "--allow-one-factors",
        action="store_true",
        help="Allow factors of 1. Disabled by default because they are usually not informative.",
    )
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Only count feasible TT shapes per total core count instead of emitting top-k candidates.",
    )
    parser.add_argument("--output-json", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.family_mode == "merge-lineage" and args.allow_one_factors:
        raise ValueError("Merge-lineage mode does not support --allow-one-factors.")

    core_counts = tuple(args.core_counts) if args.core_counts else infer_core_counts(
        in_features=args.in_features,
        out_features=args.out_features,
        split_strategy=args.split_strategy,
        allow_one_factors=args.allow_one_factors,
    )

    if args.family_mode == "merge-lineage":
        lineage_candidates = generate_merge_lineage_candidates(
            in_features=args.in_features,
            out_features=args.out_features,
            split_strategy=args.split_strategy,
        )
        payload = {
            "in_features": args.in_features,
            "out_features": args.out_features,
            "split_strategy": args.split_strategy,
            "family_mode": args.family_mode,
            "num_candidates": len(lineage_candidates),
            "candidates": [asdict(candidate) for candidate in lineage_candidates],
        }
        if args.output_json:
            output_path = Path(args.output_json).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"Saved merge-lineage TT shapes to {output_path}")

        for candidate in lineage_candidates:
            print(json.dumps(asdict(candidate)))
        return

    shape_counts = count_tt_shapes(
        in_features=args.in_features,
        out_features=args.out_features,
        core_counts=core_counts,
        split_strategy=args.split_strategy,
        allow_one_factors=args.allow_one_factors,
    )

    if args.count_only:
        payload = {
            "in_features": args.in_features,
            "out_features": args.out_features,
            "core_counts": list(core_counts),
            "split_strategy": args.split_strategy,
            "allow_one_factors": args.allow_one_factors,
            "counts": [asdict(item) for item in shape_counts],
        }
        if args.output_json:
            output_path = Path(args.output_json).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"Saved TT-shape counts to {output_path}")

        for item in shape_counts:
            print(json.dumps(asdict(item)))
        return

    candidates = generate_tt_shape_candidates(
        in_features=args.in_features,
        out_features=args.out_features,
        core_counts=core_counts,
        split_strategy=args.split_strategy,
        top_k=args.top_k,
        allow_one_factors=args.allow_one_factors,
    )

    payload = {
        "in_features": args.in_features,
        "out_features": args.out_features,
        "core_counts": list(core_counts),
        "split_strategy": args.split_strategy,
        "family_mode": args.family_mode,
        "top_k": args.top_k,
        "allow_one_factors": args.allow_one_factors,
        "counts": [asdict(item) for item in shape_counts],
        "num_candidates": len(candidates),
        "candidates": [asdict(candidate) for candidate in candidates],
    }

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved {len(candidates)} TT-shape candidates to {output_path}")

    for candidate in candidates:
        print(
            json.dumps(
                {
                    "total_cores": candidate.total_cores,
                    "input_cores": candidate.input_cores,
                    "output_cores": candidate.output_cores,
                    "input_factors": candidate.input_factors,
                    "output_factors": candidate.output_factors,
                    "tt_shape": candidate.tt_shape,
                    "balance_score": round(candidate.balance_score, 6),
                }
            )
        )


if __name__ == "__main__":
    main()
