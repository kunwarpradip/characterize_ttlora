from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

VALID_VARIANTS = {"contraction", "reconstruction"}


def _as_int_tuple(values: Iterable[int], field_name: str) -> tuple[int, ...]:
    try:
        result = tuple(int(value) for value in values)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an iterable of positive integers.") from exc
    if not result:
        raise ValueError(f"{field_name} must not be empty.")
    if any(value < 1 for value in result):
        raise ValueError(f"{field_name} must contain only positive integers.")
    return result


@dataclass(frozen=True)
class TTLoRATarget:
    """Configuration for adapting one group of matching modules.

    weight_shape is semantic [out_features, in_features]. This is the usual
    nn.Linear weight layout and also the layout users should provide for GPT-2
    Conv1D modules, even though Conv1D stores its tensor as [in_features,
    out_features].
    """

    module_name_pattern: str
    weight_shape: tuple[int, int]
    input_factors: tuple[int, ...]
    output_factors: tuple[int, ...]
    rank: int
    alpha: float = 1.0
    variant: str = "contraction"

    def __post_init__(self) -> None:
        object.__setattr__(self, "weight_shape", _as_int_tuple(self.weight_shape, "weight_shape"))
        object.__setattr__(self, "input_factors", _as_int_tuple(self.input_factors, "input_factors"))
        object.__setattr__(self, "output_factors", _as_int_tuple(self.output_factors, "output_factors"))
        object.__setattr__(self, "rank", int(self.rank))
        object.__setattr__(self, "alpha", float(self.alpha))
        object.__setattr__(self, "variant", str(self.variant).lower())
        self.validate()

    @property
    def tt_shape(self) -> tuple[int, ...]:
        return self.input_factors + tuple(reversed(self.output_factors))

    def validate(self) -> None:
        try:
            re.compile(self.module_name_pattern)
        except re.error as exc:
            raise ValueError(f"Invalid module_name_pattern: {self.module_name_pattern!r}") from exc

        if len(self.weight_shape) != 2:
            raise ValueError("weight_shape must have exactly two entries: [out_features, in_features].")
        if self.rank < 1:
            raise ValueError("rank must be >= 1.")
        if self.alpha < 0:
            raise ValueError("alpha must be >= 0.")
        if self.variant not in VALID_VARIANTS:
            supported = ", ".join(sorted(VALID_VARIANTS))
            raise ValueError(f"Unsupported variant {self.variant!r}. Expected one of: {supported}.")

        out_features, in_features = self.weight_shape
        if math.prod(self.input_factors) != in_features:
            raise ValueError(
                f"input_factors multiply to {math.prod(self.input_factors)}, "
                f"but weight_shape declares in_features={in_features}."
            )
        if math.prod(self.output_factors) != out_features:
            raise ValueError(
                f"output_factors multiply to {math.prod(self.output_factors)}, "
                f"but weight_shape declares out_features={out_features}."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_name_pattern": self.module_name_pattern,
            "weight_shape": list(self.weight_shape),
            "input_factors": list(self.input_factors),
            "output_factors": list(self.output_factors),
            "rank": self.rank,
            "alpha": self.alpha,
            "variant": self.variant,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TTLoRATarget":
        return cls(
            module_name_pattern=str(data["module_name_pattern"]),
            weight_shape=tuple(data["weight_shape"]),
            input_factors=tuple(data["input_factors"]),
            output_factors=tuple(data["output_factors"]),
            rank=int(data["rank"]),
            alpha=float(data.get("alpha", 1.0)),
            variant=str(data.get("variant", "contraction")),
        )


@dataclass(frozen=True)
class TTLoRAConfig:
    targets: tuple[TTLoRATarget, ...] = field(default_factory=tuple)
    freeze_base_model: bool = True
    init_seed: int | None = None
    strict: bool = True

    def __post_init__(self) -> None:
        targets = tuple(
            target if isinstance(target, TTLoRATarget) else TTLoRATarget.from_dict(target)
            for target in self.targets
        )
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "freeze_base_model", bool(self.freeze_base_model))
        if self.init_seed is not None:
            object.__setattr__(self, "init_seed", int(self.init_seed))
        object.__setattr__(self, "strict", bool(self.strict))
        self.validate()

    def validate(self) -> None:
        if not self.targets:
            raise ValueError("TTLoRAConfig requires at least one target.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "freeze_base_model": self.freeze_base_model,
            "init_seed": self.init_seed,
            "strict": self.strict,
            "targets": [target.to_dict() for target in self.targets],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TTLoRAConfig":
        return cls(
            targets=tuple(TTLoRATarget.from_dict(item) for item in data["targets"]),
            freeze_base_model=bool(data.get("freeze_base_model", True)),
            init_seed=data.get("init_seed"),
            strict=bool(data.get("strict", True)),
        )

    def to_json_file(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_json_file(cls, path: str | Path) -> "TTLoRAConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save_pretrained(self, save_directory: str | Path) -> None:
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        self.to_json_file(save_path / "adapter_config.json")

    @classmethod
    def from_pretrained(cls, load_directory: str | Path) -> "TTLoRAConfig":
        return cls.from_json_file(Path(load_directory) / "adapter_config.json")
