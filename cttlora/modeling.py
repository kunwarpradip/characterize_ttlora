from __future__ import annotations

import hashlib

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

from .adapters import (
    LoRALinearWrapper,
    TTLoRALinearWrapperContraction,
    TTLoRALinearWrapperReconstruction,
    freeze_model_parameters,
)
from .config import ModelConfig

HEAD_NAME_HINTS = ("classifier", "score", "qa_outputs", "pre_classifier")
ROBERTA_TARGET_MAP = {
    "query": ("attention", "self", "query"),
    "q": ("attention", "self", "query"),
    "key": ("attention", "self", "key"),
    "k": ("attention", "self", "key"),
    "value": ("attention", "self", "value"),
    "v": ("attention", "self", "value"),
}


def load_sequence_classification_model(model_config: ModelConfig, num_labels: int):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_path,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    if model.config.pad_token_id is None:
        pad_token_id = getattr(model.config, "eos_token_id", None)
        if isinstance(pad_token_id, list):
            pad_token_id = pad_token_id[0]
        model.config.pad_token_id = pad_token_id
    return apply_adaptation(model, model_config)


def apply_adaptation(model, model_config: ModelConfig):
    method = model_config.adaptation_method.lower()
    if method == "full":
        return model
    if method == "classifier-only":
        freeze_model_parameters(model)
        unfrozen = 0
        for name, param in model.named_parameters():
            if any(hint in name for hint in HEAD_NAME_HINTS):
                param.requires_grad = True
                unfrozen += 1
        if unfrozen == 0:
            raise ValueError(
                "No classifier head parameters were detected. "
                "Update HEAD_NAME_HINTS in cttlora/modeling.py for this model family."
            )
        return model
    if method == "lora":
        return apply_lora(model, model_config)
    if method == "ttlora":
        return apply_ttlora(model, model_config)
    raise ValueError(f"Unsupported adaptation method: {model_config.adaptation_method}")


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


def _selected_roberta_paths(target_modules: tuple[str, ...]) -> list[tuple[str, ...]]:
    paths = []
    for target in target_modules:
        key = target.lower()
        if key not in ROBERTA_TARGET_MAP:
            supported = ", ".join(sorted(ROBERTA_TARGET_MAP))
            raise ValueError(
                f"Unsupported target module '{target}' for RoBERTa. Supported targets: {supported}."
            )
        paths.append(ROBERTA_TARGET_MAP[key])
    deduped: list[tuple[str, ...]] = []
    seen = set()
    for path in paths:
        if path not in seen:
            deduped.append(path)
            seen.add(path)
    return deduped


def _should_adapt_layer(layer_idx: int, adapt_layers: tuple[int, ...] | None) -> bool:
    return adapt_layers is None or layer_idx in adapt_layers


def _wrap_roberta_layers(model, model_config: ModelConfig, wrapper_factory):
    if not hasattr(model, "roberta"):
        raise ValueError("Current PEFT adapters are wired for RoBERTa sequence classification models.")
    target_paths = _selected_roberta_paths(model_config.target_modules)
    layer_count = 0
    wrapped_count = 0
    for layer in model.roberta.encoder.layer:
        if not _should_adapt_layer(layer_count, model_config.adapt_layers):
            layer_count += 1
            continue
        for path in target_paths:
            original = _resolve_attr(layer, path)
            if not isinstance(original, nn.Linear):
                raise TypeError(f"Expected nn.Linear at RoBERTa path {'.'.join(path)}, got {type(original)}")
            wrapped = wrapper_factory(original)
            _assign_attr(layer, path, wrapped)
            wrapped_count += 1
        layer_count += 1
    if wrapped_count == 0:
        raise ValueError("No layers were adapted. Check --adapt-layers and --target-modules.")
    return model


def apply_lora(model, model_config: ModelConfig):
    freeze_model_parameters(model)
    return _wrap_roberta_layers(
        model,
        model_config,
        wrapper_factory=lambda original: LoRALinearWrapper(
            original_layer=original,
            rank=model_config.lora_rank,
            alpha=model_config.lora_alpha,
        ),
    )


def apply_ttlora(model, model_config: ModelConfig):
    freeze_model_parameters(model)
    variant = model_config.ttlora_variant.lower()
    if variant not in {"contraction", "reconstruction"}:
        raise ValueError(
            f"Unsupported TT-LoRA variant '{model_config.ttlora_variant}'. "
            "Supported variants: contraction, reconstruction."
        )
    return _wrap_roberta_layers(
        model,
        model_config,
        wrapper_factory=lambda original: (
            TTLoRALinearWrapperContraction(
                original_layer=original,
                tt_shape=model_config.ttlora_shape,
                rank=model_config.ttlora_rank,
                alpha=model_config.ttlora_alpha,
                input_factors=model_config.ttlora_input_factors,
                output_factors=model_config.ttlora_output_factors,
            )
            if variant == "contraction"
            else TTLoRALinearWrapperReconstruction(
                original_layer=original,
                tt_shape=model_config.ttlora_shape,
                rank=model_config.ttlora_rank,
                alpha=model_config.ttlora_alpha,
            )
        ),
    )


def count_parameters(model) -> dict[str, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
    }


def trainable_parameter_names(model) -> list[str]:
    return [name for name, param in model.named_parameters() if param.requires_grad]


def parameter_groups(model, weight_decay: float) -> list[dict]:
    decay_params = []
    nodecay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or "LayerNorm.weight" in name:
            nodecay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]


def classifier_state_dict(model) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if any(hint in name for hint in HEAD_NAME_HINTS):
            state[name] = param.detach().cpu().clone()
    if not state:
        raise ValueError(
            "No classifier head parameters were detected. "
            "Update HEAD_NAME_HINTS in cttlora/modeling.py for this model family."
        )
    return state


def classifier_state_summary(model) -> dict:
    state = classifier_state_dict(model)
    hasher = hashlib.sha256()
    per_tensor = {}

    for name in sorted(state):
        tensor = state[name].contiguous()
        hasher.update(name.encode("utf-8"))
        hasher.update(tensor.numpy().tobytes())
        per_tensor[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "mean": float(tensor.float().mean().item()),
            "std": float(tensor.float().std().item()) if tensor.numel() > 1 else 0.0,
            "sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
        }

    return {
        "parameter_count": sum(t.numel() for t in state.values()),
        "global_sha256": hasher.hexdigest(),
        "tensors": per_tensor,
    }
