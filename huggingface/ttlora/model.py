from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator

import torch.nn as nn

from .config import TTLoRAConfig, TTLoRATarget
from .layers import (
    TTLoRAModule,
    is_supported_module,
    semantic_weight_shape,
    stable_ttlora_init_seed,
)


@dataclass(frozen=True)
class ParameterReport:
    total_parameters: int
    trainable_parameters: int
    frozen_parameters: int
    trainable_percent: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "total_parameters": self.total_parameters,
            "trainable_parameters": self.trainable_parameters,
            "frozen_parameters": self.frozen_parameters,
            "trainable_percent": self.trainable_percent,
        }


def _get_parent_module(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    if not module_name:
        raise ValueError("Cannot replace the root module with a TT-LoRA wrapper.")
    parent = model
    parts = module_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _matches_target(module_name: str, target: TTLoRATarget) -> bool:
    return re.search(target.module_name_pattern, module_name) is not None


def _validate_target_module(module_name: str, module: nn.Module, target: TTLoRATarget) -> None:
    if not is_supported_module(module):
        raise TypeError(
            f"Target pattern {target.module_name_pattern!r} matched {module_name!r}, "
            f"but its type is unsupported: {type(module)}."
        )
    actual_shape = semantic_weight_shape(module)
    if actual_shape != target.weight_shape:
        raise ValueError(
            f"Target pattern {target.module_name_pattern!r} matched {module_name!r}, "
            f"but semantic weight_shape is {actual_shape}; config expected {target.weight_shape}."
        )


def get_ttlora_model(model: nn.Module, config: TTLoRAConfig) -> nn.Module:
    """Wrap target modules in-place with TT-LoRA adapters and return model."""
    if not isinstance(config, TTLoRAConfig):
        config = TTLoRAConfig.from_dict(config)  # type: ignore[arg-type]

    if config.freeze_base_model:
        for parameter in model.parameters():
            parameter.requires_grad = False

    named_modules = list(model.named_modules())
    match_counts = [0 for _ in config.targets]
    adapted_names: set[str] = set()

    for module_name, module in named_modules:
        if not module_name or isinstance(module, TTLoRAModule):
            continue

        matching_indices = [
            idx for idx, target in enumerate(config.targets)
            if _matches_target(module_name, target)
        ]
        if not matching_indices:
            continue
        if len(matching_indices) > 1:
            patterns = [config.targets[idx].module_name_pattern for idx in matching_indices]
            raise ValueError(f"Module {module_name!r} matched multiple TT-LoRA targets: {patterns}.")
        if module_name in adapted_names:
            raise ValueError(f"Module {module_name!r} was selected more than once.")

        target_index = matching_indices[0]
        target = config.targets[target_index]
        _validate_target_module(module_name, module, target)
        init_seed = None
        if config.init_seed is not None:
            init_seed = stable_ttlora_init_seed(config.init_seed, module_name, target_index)

        wrapped = TTLoRAModule(original_layer=module, target=target, init_seed=init_seed)
        parent, child_name = _get_parent_module(model, module_name)
        setattr(parent, child_name, wrapped)
        match_counts[target_index] += 1
        adapted_names.add(module_name)

    missing_targets = [
        target.module_name_pattern
        for target, count in zip(config.targets, match_counts)
        if count == 0
    ]
    if config.strict and missing_targets:
        raise ValueError(f"No modules matched TT-LoRA target pattern(s): {missing_targets}.")

    model.ttlora_config = config  # type: ignore[attr-defined]
    model.ttlora_target_match_counts = tuple(match_counts)  # type: ignore[attr-defined]
    return model


def iter_ttlora_modules(model: nn.Module) -> Iterator[tuple[str, TTLoRAModule]]:
    for name, module in model.named_modules():
        if isinstance(module, TTLoRAModule):
            yield name, module


def mark_only_ttlora_as_trainable(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for _, module in iter_ttlora_modules(model):
        for parameter in module.tt_cores.parameters():
            parameter.requires_grad = True


def get_parameter_report(model: nn.Module) -> ParameterReport:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    frozen = total - trainable
    percent = 100.0 * trainable / total if total else 0.0
    return ParameterReport(
        total_parameters=total,
        trainable_parameters=trainable,
        frozen_parameters=frozen,
        trainable_percent=percent,
    )


def print_trainable_parameters(model: nn.Module) -> None:
    report = get_parameter_report(model)
    print(
        "trainable params: "
        f"{report.trainable_parameters:,} || all params: {report.total_parameters:,} || "
        f"trainable%: {report.trainable_percent:.4f}"
    )
