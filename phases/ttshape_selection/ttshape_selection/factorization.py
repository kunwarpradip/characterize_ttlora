from __future__ import annotations

import math
import statistics
from functools import lru_cache


def _pad_to_equal_length(
    output_factors: tuple[int, ...],
    input_factors: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    length = max(len(output_factors), len(input_factors))
    padded_output = (*output_factors, *([1] * (length - len(output_factors))))
    padded_input = (*input_factors, *([1] * (length - len(input_factors))))
    return padded_output, padded_input


def score_tt_shape(
    output_factors: tuple[int, ...],
    input_factors: tuple[int, ...],
    rank: int,
) -> dict[str, float | int]:
    """
    Score a TT-matrix factorization.

    The scoring formula expects paired output/input factor lists. If the
    candidate split has unequal input/output core counts, the shorter side is
    padded with factors of 1, which the score penalizes through ones_penalty.
    """
    if rank < 1:
        raise ValueError("rank must be at least 1.")
    if any(value < 1 for value in (*output_factors, *input_factors)):
        raise ValueError("TT factors must be positive.")

    m_shape, n_shape = _pad_to_equal_length(output_factors, input_factors)
    k = len(m_shape)
    ranks = [1] + [rank] * (k - 1) + [1]

    factors = m_shape + n_shape
    log_factors = [math.log(value) for value in factors]
    balance_penalty = statistics.pstdev(log_factors)

    core_sizes = [m * n for m, n in zip(m_shape, n_shape)]
    log_core_sizes = [math.log(value) for value in core_sizes]
    core_size_penalty = statistics.pstdev(log_core_sizes)

    ones_penalty = sum(1 for value in factors if value == 1) / len(factors)

    tt_params = sum(
        ranks[idx] * m_shape[idx] * n_shape[idx] * ranks[idx + 1]
        for idx in range(k)
    )

    dense_params = math.prod(m_shape) * math.prod(n_shape)
    param_penalty = tt_params / dense_params

    score = (
        1.0 * balance_penalty
        + 1.0 * core_size_penalty
        + 2.0 * ones_penalty
        + 0.5 * param_penalty
    )

    return {
        "score": score,
        "balance_penalty": balance_penalty,
        "core_size_penalty": core_size_penalty,
        "ones_penalty": ones_penalty,
        "tt_params": tt_params,
        "compression_ratio": dense_params / tt_params,
    }


@lru_cache(maxsize=None)
def ordered_factorizations(n: int, parts: int, min_factor: int = 2) -> tuple[tuple[int, ...], ...]:
    """Return non-decreasing multiplicative factorizations of n into parts terms."""
    if n < 1:
        raise ValueError("n must be positive.")
    if parts < 1:
        raise ValueError("parts must be positive.")
    if parts == 1:
        return ((n,),) if n >= min_factor else tuple()

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


def candidate_splits(total_cores: int, strategy: str) -> tuple[tuple[int, int], ...]:
    if total_cores < 2:
        raise ValueError("total_cores must be at least 2.")

    if strategy == "all":
        return tuple((input_cores, total_cores - input_cores) for input_cores in range(1, total_cores))
    if strategy == "symmetric":
        if total_cores % 2 != 0:
            return tuple()
        return ((total_cores // 2, total_cores // 2),)
    if strategy == "near-symmetric":
        left = total_cores // 2
        right = total_cores - left
        return ((left, right),)
    raise ValueError("split_strategy must be one of: all, symmetric, near-symmetric.")


def infer_core_counts(in_features: int, out_features: int, *, allow_one_factors: bool) -> tuple[int, ...]:
    if allow_one_factors:
        raise ValueError("core_counts must be provided when allow_one_factors=True.")

    def feasible_counts(n: int) -> list[int]:
        counts: list[int] = []
        parts = 1
        while ordered_factorizations(n, parts, 2):
            counts.append(parts)
            parts += 1
        return counts

    inferred = {
        input_count + output_count
        for input_count in feasible_counts(in_features)
        for output_count in feasible_counts(out_features)
    }
    return tuple(sorted(inferred))


def tt_parameter_count(tt_shape: tuple[int, ...], rank: int) -> int:
    if rank < 1:
        raise ValueError("rank must be at least 1.")
    if not tt_shape:
        raise ValueError("tt_shape cannot be empty.")
    tt_rank = (1, *([rank] * (len(tt_shape) - 1)), 1)
    return sum(tt_rank[idx] * dim * tt_rank[idx + 1] for idx, dim in enumerate(tt_shape))
