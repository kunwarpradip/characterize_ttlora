from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

from .adapters import (
    LoRALinearWrapper,
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

LLAMA_TARGET_MAP = {
    "q_proj": ("self_attn", "q_proj"),
    "k_proj": ("self_attn", "k_proj"),
    "v_proj": ("self_attn", "v_proj"),
    "o_proj": ("self_attn", "o_proj"),
    "wq": ("self_attn", "q_proj"),
    "wk": ("self_attn", "k_proj"),
    "wv": ("self_attn", "v_proj"),
    "wo": ("self_attn", "o_proj"),
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


def _llama_intermediate_size(hidden_size: int, multiple_of: int) -> int:
    hidden_dim = int(2 * (4 * hidden_size) / 3)
    return multiple_of * math.ceil(hidden_dim / multiple_of)


def _is_original_llama_dir(model_path: str) -> bool:
    path = Path(model_path).expanduser()
    candidates = [path, path / "checkpoints"]
    return any((candidate / "params.json").exists() for candidate in candidates)


def _resolve_original_llama_dir(model_path: str) -> Path:
    path = Path(model_path).expanduser()
    for candidate in (path, path / "checkpoints"):
        if (candidate / "params.json").exists():
            return candidate
    raise FileNotFoundError(f"Could not find original LLaMA params.json under {model_path}")


def _load_original_llama_params(model_dir: Path) -> dict:
    import json

    return json.loads((model_dir / "params.json").read_text(encoding="utf-8"))


def _llama_config_from_original(model_dir: Path) -> LlamaConfig:
    params = _load_original_llama_params(model_dir)
    hidden_size = int(params["dim"])
    num_heads = int(params["n_heads"])
    num_layers = int(params["n_layers"])
    multiple_of = int(params.get("multiple_of", 256))
    vocab_size = int(params.get("vocab_size", -1))
    if vocab_size <= 0:
        vocab_size = 32000
    max_position_embeddings = 4096 if "llama2" in str(model_dir).lower() or "llama-2" in str(model_dir).lower() else 2048

    return LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=_llama_intermediate_size(hidden_size, multiple_of),
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        hidden_act="silu",
        max_position_embeddings=max_position_embeddings,
        initializer_range=0.02,
        rms_norm_eps=float(params.get("norm_eps", 1e-5)),
        use_cache=True,
        pad_token_id=2,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )


def _map_original_llama_key(name: str) -> str | None:
    if name == "tok_embeddings.weight":
        return "model.embed_tokens.weight"
    if name == "norm.weight":
        return "model.norm.weight"
    if name == "output.weight":
        return "lm_head.weight"

    replacements = {
        ".attention.wq.weight": ".self_attn.q_proj.weight",
        ".attention.wk.weight": ".self_attn.k_proj.weight",
        ".attention.wv.weight": ".self_attn.v_proj.weight",
        ".attention.wo.weight": ".self_attn.o_proj.weight",
        ".feed_forward.w1.weight": ".mlp.gate_proj.weight",
        ".feed_forward.w2.weight": ".mlp.down_proj.weight",
        ".feed_forward.w3.weight": ".mlp.up_proj.weight",
        ".attention_norm.weight": ".input_layernorm.weight",
        ".ffn_norm.weight": ".post_attention_layernorm.weight",
    }
    if not name.startswith("layers."):
        return None
    for source, target in replacements.items():
        if source in name:
            return "model." + name.replace(source, target)
    return None


def _load_original_llama_state_dict(model_dir: Path) -> dict[str, torch.Tensor]:
    checkpoint_paths = sorted(model_dir.glob("consolidated.*.pth"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No consolidated.*.pth files found in {model_dir}")

    mapped: dict[str, torch.Tensor] = {}
    for checkpoint_path in checkpoint_paths:
        try:
            state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(str(checkpoint_path), map_location="cpu")
        for key, value in state.items():
            mapped_key = _map_original_llama_key(str(key))
            if mapped_key is not None:
                mapped[mapped_key] = value
    return mapped


def load_original_llama_causal_lm(model_path: str) -> LlamaForCausalLM:
    model_dir = _resolve_original_llama_dir(model_path)
    config = _llama_config_from_original(model_dir)
    model = LlamaForCausalLM(config)
    state_dict = _load_original_llama_state_dict(model_dir)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    unexpected = [key for key in unexpected if not key.endswith("rotary_emb.inv_freq")]
    if unexpected:
        raise ValueError(f"Unexpected keys while loading original LLaMA checkpoint: {unexpected[:20]}")
    important_missing = [
        key for key in missing
        if not key.endswith("rotary_emb.inv_freq")
    ]
    if important_missing:
        raise ValueError(f"Missing keys while loading original LLaMA checkpoint: {important_missing[:20]}")
    return model


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


def apply_llama_ttlora(model, model_config: GenerationModelConfig):
    freeze_model_parameters(model)
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("LLaMA TT-LoRA adaptation expects a model with model.layers blocks.")

    wrapped_count = 0
    for layer_idx, block in enumerate(model.model.layers):
        if not _should_adapt_layer(layer_idx, model_config.adapt_layers):
            continue
        for weight_config in model_config.weight_configs:
            weight_name = weight_config.weight_name.lower()
            if weight_name not in LLAMA_TARGET_MAP:
                supported = ", ".join(sorted(LLAMA_TARGET_MAP))
                raise ValueError(
                    f"Unsupported LLaMA target weight '{weight_config.weight_name}'. Supported: {supported}."
                )
            path = LLAMA_TARGET_MAP[weight_name]
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
        raise ValueError("No LLaMA layers were adapted. Check the selected layers and weight configs.")
    return model


def apply_llama_lora(model, model_config: GenerationModelConfig):
    freeze_model_parameters(model)
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("LLaMA LoRA adaptation expects a model with model.layers blocks.")

    wrapped_count = 0
    for layer_idx, block in enumerate(model.model.layers):
        if not _should_adapt_layer(layer_idx, model_config.adapt_layers):
            continue
        for weight_name in model_config.lora_target_weights:
            normalized_weight_name = weight_name.lower()
            if normalized_weight_name not in LLAMA_TARGET_MAP:
                supported = ", ".join(sorted(LLAMA_TARGET_MAP))
                raise ValueError(
                    f"Unsupported LLaMA LoRA target weight '{weight_name}'. Supported: {supported}."
                )
            path = LLAMA_TARGET_MAP[normalized_weight_name]
            original = _resolve_attr(block, path)
            if not isinstance(original, nn.Linear):
                raise TypeError(f"LLaMA LoRA expects nn.Linear targets, got {type(original)} for {weight_name}.")
            wrapped = LoRALinearWrapper(
                original_layer=original,
                rank=model_config.lora_rank,
                alpha=model_config.lora_alpha,
            )
            _assign_attr(block, path, wrapped)
            wrapped_count += 1

    if wrapped_count == 0:
        raise ValueError("No LLaMA LoRA layers were adapted. Check the selected layers and target weights.")
    return model


def apply_generation_ttlora(model, model_config: GenerationModelConfig):
    model_type = getattr(getattr(model, "config", None), "model_type", "").lower()
    if model_type == "llama" or (hasattr(model, "model") and hasattr(model.model, "layers")):
        return apply_llama_ttlora(model, model_config)
    return apply_gpt2_ttlora(model, model_config)


def apply_generation_lora(model, model_config: GenerationModelConfig):
    model_type = getattr(getattr(model, "config", None), "model_type", "").lower()
    if model_type == "llama" or (hasattr(model, "model") and hasattr(model.model, "layers")):
        return apply_llama_lora(model, model_config)
    raise ValueError("Generation LoRA is currently implemented for LLaMA-style nn.Linear attention weights.")


def apply_generation_adaptation(model, model_config: GenerationModelConfig):
    method = model_config.adaptation_method.lower()
    if method == "full":
        return model
    if method == "ttlora":
        return apply_generation_ttlora(model, model_config)
    if method == "lora":
        return apply_generation_lora(model, model_config)
    raise ValueError(f"Unsupported generation adaptation method: {model_config.adaptation_method}")


def load_generation_model(model_config: GenerationModelConfig):
    if _is_original_llama_dir(model_config.model_name_or_path):
        model = load_original_llama_causal_lm(model_config.model_name_or_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path)
    if model.config.pad_token_id is None:
        eos_token_id = getattr(model.config, "eos_token_id", None)
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0]
        model.config.pad_token_id = eos_token_id
    return apply_generation_adaptation(model, model_config)


def load_generation_checkpoint_into_model(model: nn.Module, checkpoint_dir: str | Path) -> tuple[list[str], list[str]]:
    checkpoint_path = Path(checkpoint_dir).expanduser().resolve()
    safetensors_path = checkpoint_path / "model.safetensors"
    pytorch_bin_path = checkpoint_path / "pytorch_model.bin"

    state_dict = None
    if safetensors_path.exists():
        from safetensors.torch import load_file

        state_dict = load_file(str(safetensors_path))
    elif pytorch_bin_path.exists():
        state_dict = torch.load(str(pytorch_bin_path), map_location="cpu")
    else:
        raise FileNotFoundError(
            f"Could not find model.safetensors or pytorch_model.bin under checkpoint directory {checkpoint_path}"
        )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return list(missing), list(unexpected)
