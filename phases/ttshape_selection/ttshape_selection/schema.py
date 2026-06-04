from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class WeightSpec:
    name: str
    shape: tuple[int, int]

    @property
    def out_features(self) -> int:
        return int(self.shape[0])

    @property
    def in_features(self) -> int:
        return int(self.shape[1])

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "WeightSpec":
        name = str(payload["name"]).strip()
        shape = payload["shape"]
        if not isinstance(shape, (list, tuple)) or len(shape) != 2:
            raise ValueError(f"Weight '{name}' must provide shape as [out_features, in_features].")
        out_features, in_features = (int(shape[0]), int(shape[1]))
        if out_features < 1 or in_features < 1:
            raise ValueError(f"Weight '{name}' dimensions must be positive, got {shape}.")
        return cls(name=name, shape=(out_features, in_features))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "shape": list(self.shape)}


@dataclass(frozen=True, slots=True)
class ShapeCandidate:
    weight_name: str
    weight_shape: tuple[int, int]
    rank: int
    core_count: int
    input_cores: int
    output_cores: int
    input_factors: tuple[int, ...]
    output_factors: tuple[int, ...]
    tt_shape: tuple[int, ...]
    parameter_count: int
    score: float
    balance_penalty: float
    core_size_penalty: float
    ones_penalty: float
    tt_matrix_parameter_count: int
    compression_ratio: float
    max_factor: int
    min_factor: int
    uses_one_factor: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["weight_shape"] = list(self.weight_shape)
        payload["input_factors"] = list(self.input_factors)
        payload["output_factors"] = list(self.output_factors)
        payload["tt_shape"] = list(self.tt_shape)
        return payload


@dataclass(frozen=True, slots=True)
class ShapeCatalog:
    rank: int
    split_strategy: str
    allow_one_factors: bool
    weights: tuple[WeightSpec, ...]
    candidates_by_weight: dict[str, tuple[ShapeCandidate, ...]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "split_strategy": self.split_strategy,
            "allow_one_factors": self.allow_one_factors,
            "weights": [weight.to_dict() for weight in self.weights],
            "candidates_by_weight": {
                name: [candidate.to_dict() for candidate in candidates]
                for name, candidates in self.candidates_by_weight.items()
            },
        }
