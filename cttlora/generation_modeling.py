from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from .adapters import (
    freeze_model_parameters,
    generate_tt_cores,
    reconstruct_tt_weight_matrix,
    tensorized_multiplication,
    ttlora_rank_list,
)
from .generation_config import GenerationModelConfig, TTLoRAWeightConfig

GPT2_TARGET_MAP = {
    "c_attn": ("attn", "c_attn"),
    "c_proj": ("attn", "c_proj"),
}


def _resolve_attr(module, path: tuple[str, ...]):
    current = module
    for name in path:
        current = getattr(current, name)
    return current


def _assign_attr(module, path: tuple[str, ...], value) -> None:
    parent = module
    for name in path[:-1]:
        parent = getattr(parent, name)
    setattr(parent, path[-1], value)


def _linear_like_features(module: nn.Module) -> tuple[int, int]:
    if isinstance(module, nn.Linear):
        return module.in_features, module.out_features
    if module.__class__.__name__ == "Conv1D" and hasattr(module, "weight"):
        in_features, out_features = module.weight.shape
        return int(in_features), int(out_features)
    raise TypeError(f"Unsupported GPT-2 target module type: {type(module)}")


def _is_conv1d_like(module: nn.Module) -> bool:
    return module.__class__.__name__ == "Conv1D" and hasattr(module, "weight")


class TTLoRAGenerationWrapper(nn.Module):
    def __init__(
        self,
        original_layer: nn.Module,
        weight_config: TTLoRAWeightConfig,
        rank: int,
        alpha: float,
        mode: str,
    ) -> None:
        super().__init__()
        if mode not in {"contraction", "reconstruction"}:
            raise ValueError(f"Unsupported TT-LoRA mode '{mode}'.")

        self.original = original_layer
        self.weight_config = weight_config
        self.alpha = alpha
        self.mode = mode
        self.input_factors = weight_config.input_factors
        self.output_factors = weight_config.output_factors
        self.tt_shape = weight_config.tt_shape
        self.tt_rank = ttlora_rank_list(rank, self.tt_shape)
        self.tt_cores = generate_tt_cores(self.tt_shape, self.tt_rank)

        in_features, out_features = _linear_like_features(original_layer)
        if math.prod(self.input_factors) != in_features:
            raise ValueError(
                f"Input TT factors {self.input_factors} multiply to {math.prod(self.input_factors)}, "
                f"but {type(original_layer).__name__} expects in_features={in_features}."
            )
        if math.prod(self.output_factors) != out_features:
            raise ValueError(
                f"Output TT factors {self.output_factors} multiply to {math.prod(self.output_factors)}, "
                f"but {type(original_layer).__name__} expects out_features={out_features}."
            )
        if len(self.tt_shape) != len(self.input_factors) + len(self.output_factors):
            raise ValueError("TT shape length must equal len(input_factors) + len(output_factors).")
        if tuple(self.tt_shape[: len(self.input_factors)]) != self.input_factors:
            raise ValueError("TT shape input prefix does not match input_factors.")
        if tuple(self.tt_shape[len(self.input_factors) :]) != self.output_factors[::-1]:
            raise ValueError("TT shape output suffix must equal reversed output_factors.")

        expected_weight_shape = weight_config.weight_shape
        if expected_weight_shape is not None:
            normalized_expected = tuple(int(item) for item in expected_weight_shape)
            if normalized_expected != (out_features, in_features):
                raise ValueError(
                    f"Weight config for {weight_config.weight_name} expects semantic weight shape "
                    f"{normalized_expected}, but the module resolves to {(out_features, in_features)}."
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "contraction":
            update = tensorized_multiplication(
                x=x,
                tt_cores=self.tt_cores,
                input_factors=self.input_factors,
                output_factors=self.output_factors,
            )
            return self.original(x) + update * self.alpha

        dense_update = reconstruct_tt_weight_matrix(
            tt_cores=self.tt_cores,
            input_factors=self.input_factors,
            output_factors=self.output_factors,
        )
        if isinstance(self.original, nn.Linear):
            adapted_weight = self.original.weight + self.alpha * dense_update
            return F.linear(x, adapted_weight, self.original.bias)

        if _is_conv1d_like(self.original):
            adapted_weight = self.original.weight + self.alpha * dense_update.transpose(0, 1)
            size_out = x.size()[:-1] + (adapted_weight.size(1),)
            x_2d = x.reshape(-1, x.size(-1))
            out = torch.addmm(self.original.bias, x_2d, adapted_weight)
            return out.view(size_out)

        raise TypeError(f"Unsupported GPT-2 target module type: {type(self.original)}")


def _should_adapt_layer(layer_idx: int, adapt_layers: tuple[int, ...] | None) -> bool:
    return adapt_layers is None or layer_idx in adapt_layers


def apply_gpt2_ttlora(model, model_config: GenerationModelConfig):
    freeze_model_parameters(model)
    if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
        raise ValueError("GPT-2 TT-LoRA adaptation expects a model with transformer.h blocks.")

    wrapped_count = 0
    for layer_idx, block in enumerate(model.transformer.h):
        if not _should_adapt_layer(layer_idx, model_config.adapt_layers):
            continue
        for weight_config in model_config.weight_configs:
            weight_name = weight_config.weight_name.lower()
            if weight_name not in GPT2_TARGET_MAP:
                supported = ", ".join(sorted(GPT2_TARGET_MAP))
                raise ValueError(
                    f"Unsupported GPT-2 target weight '{weight_config.weight_name}'. Supported: {supported}."
                )
            path = GPT2_TARGET_MAP[weight_name]
            original = _resolve_attr(block, path)
            wrapped = TTLoRAGenerationWrapper(
                original_layer=original,
                weight_config=weight_config,
                rank=model_config.ttlora_rank,
                alpha=model_config.ttlora_alpha,
                mode=model_config.ttlora_variant.lower(),
            )
            _assign_attr(block, path, wrapped)
            wrapped_count += 1

    if wrapped_count == 0:
        raise ValueError("No GPT-2 layers were adapted. Check the selected layers and weight configs.")
    return model


def load_generation_model(model_config: GenerationModelConfig):
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path)
    if model.config.pad_token_id is None:
        eos_token_id = getattr(model.config, "eos_token_id", None)
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0]
        model.config.pad_token_id = eos_token_id
    return apply_gpt2_ttlora(model, model_config)
