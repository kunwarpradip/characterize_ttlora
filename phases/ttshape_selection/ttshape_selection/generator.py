from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .factorization import (
    candidate_splits,
    distinct_factor_orderings,
    infer_core_counts,
    ordered_factorizations,
    score_tt_shape,
    tt_parameter_count,
)
from .schema import ShapeCandidate, ShapeCatalog, WeightSpec


def load_weight_specs(path: str | Path) -> tuple[WeightSpec, ...]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    weights = payload.get("weights", payload)
    if not isinstance(weights, list):
        raise ValueError("Input JSON must be a list of weights or contain a 'weights' list.")
    return tuple(WeightSpec.from_mapping(item) for item in weights)


def generate_candidates(
    weight: WeightSpec,
    *,
    rank: int,
    core_counts: Iterable[int] | None = None,
    split_strategy: str = "all",
    allow_one_factors: bool = False,
) -> tuple[ShapeCandidate, ...]:
    min_factor = 1 if allow_one_factors else 2
    selected_core_counts = (
        tuple(int(item) for item in core_counts)
        if core_counts is not None
        else infer_core_counts(weight.in_features, weight.out_features, allow_one_factors=allow_one_factors)
    )

    candidates: list[ShapeCandidate] = []
    for core_count in selected_core_counts:
        for input_cores, output_cores in candidate_splits(core_count, split_strategy):
            input_factorizations = ordered_factorizations(weight.in_features, input_cores, min_factor)
            output_factorizations = ordered_factorizations(weight.out_features, output_cores, min_factor)
            for input_factors in input_factorizations:
                for output_factors in output_factorizations:
                    for input_ordering in distinct_factor_orderings(input_factors):
                        for output_ordering in distinct_factor_orderings(output_factors):
                            tt_shape = (*input_ordering, *reversed(output_ordering))
                            factors = (*input_ordering, *output_ordering)
                            score_payload = score_tt_shape(
                                output_factors=output_ordering,
                                input_factors=input_ordering,
                                rank=rank,
                            )
                            candidates.append(
                                ShapeCandidate(
                                    weight_name=weight.name,
                                    weight_shape=weight.shape,
                                    rank=rank,
                                    core_count=core_count,
                                    input_cores=input_cores,
                                    output_cores=output_cores,
                                    input_factors=input_ordering,
                                    output_factors=output_ordering,
                                    tt_shape=tt_shape,
                                    parameter_count=tt_parameter_count(tt_shape, rank),
                                    score=float(score_payload["score"]),
                                    balance_penalty=float(score_payload["balance_penalty"]),
                                    core_size_penalty=float(score_payload["core_size_penalty"]),
                                    ones_penalty=float(score_payload["ones_penalty"]),
                                    tt_matrix_parameter_count=int(score_payload["tt_params"]),
                                    compression_ratio=float(score_payload["compression_ratio"]),
                                    max_factor=max(factors),
                                    min_factor=min(factors),
                                    uses_one_factor=any(value == 1 for value in factors),
                                )
                            )

    return tuple(
        sorted(
            candidates,
            key=lambda item: (item.core_count, item.score, item.parameter_count, item.tt_shape),
        )
    )


def generate_catalog(
    weights: Iterable[WeightSpec | dict[str, Any]],
    *,
    rank: int,
    core_counts: Iterable[int] | None = None,
    split_strategy: str = "all",
    allow_one_factors: bool = False,
    top_k_per_weight: int | None = None,
) -> ShapeCatalog:
    weight_specs = tuple(
        item if isinstance(item, WeightSpec) else WeightSpec.from_mapping(item)
        for item in weights
    )
    candidates_by_weight: dict[str, tuple[ShapeCandidate, ...]] = {}
    for weight in weight_specs:
        candidates = generate_candidates(
            weight,
            rank=rank,
            core_counts=core_counts,
            split_strategy=split_strategy,
            allow_one_factors=allow_one_factors,
        )
        if top_k_per_weight is not None:
            candidates = candidates[: int(top_k_per_weight)]
        candidates_by_weight[weight.name] = candidates

    return ShapeCatalog(
        rank=rank,
        split_strategy=split_strategy,
        allow_one_factors=allow_one_factors,
        weights=weight_specs,
        candidates_by_weight=candidates_by_weight,
    )
