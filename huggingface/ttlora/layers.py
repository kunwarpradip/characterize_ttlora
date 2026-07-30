from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TTLoRATarget

_TT_INIT_SEED_MOD = 2**31 - 1


def ttlora_rank_list(rank: int, tt_shape: tuple[int, ...]) -> tuple[int, ...]:
    if rank < 1:
        raise ValueError("TT-LoRA rank must be >= 1.")
    return (1, *([int(rank)] * (len(tt_shape) - 1)), 1)


def stable_ttlora_init_seed(base_seed: int, module_name: str, target_index: int = 0) -> int:
    value = int(base_seed) % _TT_INIT_SEED_MOD
    value = (value + 1009 * int(target_index)) % _TT_INIT_SEED_MOD
    for char in str(module_name):
        value = (value * 131 + ord(char)) % _TT_INIT_SEED_MOD
    return value


def generate_tt_cores(
    tt_shape: tuple[int, ...],
    rank: int,
    init_seed: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> nn.ParameterList:
    tt_rank = ttlora_rank_list(rank, tt_shape)
    cores = nn.ParameterList()
    init_device = torch.device("cpu") if init_seed is not None else device

    for idx, dim in enumerate(tt_shape):
        core_shape = (tt_rank[idx], int(dim), tt_rank[idx + 1])
        core = torch.empty(core_shape, device=init_device, dtype=dtype)
        if init_seed is None:
            nn.init.kaiming_uniform_(core, a=math.sqrt(8))
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed((int(init_seed) + idx) % _TT_INIT_SEED_MOD)
            nn.init.kaiming_uniform_(core, a=math.sqrt(8), generator=generator)
        core = core / (core.norm() + 1e-6)
        if device is not None and core.device != device:
            core = core.to(device=device)
        cores.append(nn.Parameter(core))

    return cores


def tt_parameter_count(tt_shape: tuple[int, ...], rank: int) -> int:
    tt_rank = ttlora_rank_list(rank, tt_shape)
    return int(sum(tt_rank[idx] * dim * tt_rank[idx + 1] for idx, dim in enumerate(tt_shape)))


def tensorized_multiplication(
    x: torch.Tensor,
    tt_cores: nn.ParameterList,
    input_factors: tuple[int, ...],
    output_factors: tuple[int, ...],
) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"tensorized_multiplication expects a 3D tensor, got shape {tuple(x.shape)}.")

    batch_size = x.size(0)
    seq_len = x.size(1)
    num_input_cores = len(input_factors)
    num_output_cores = len(output_factors)

    tt_state = x.contiguous().view(batch_size, seq_len, *input_factors[::-1]).unsqueeze(1)

    for idx in range(num_input_cores):
        core = tt_cores[idx]
        tt_state = torch.einsum("br...m,rmp->bp...", tt_state, core)

    for idx in range(num_output_cores):
        core = tt_cores[num_input_cores + idx]
        tt_state = torch.einsum("br...,rnp->bp...n", tt_state, core)

    return tt_state.view(batch_size, seq_len, -1)


def reconstruct_tt_tensor(tt_cores: nn.ParameterList) -> torch.Tensor:
    result = tt_cores[0]
    for core in tt_cores[1:]:
        result = torch.tensordot(result, core, dims=([-1], [0]))
    return result.squeeze(0).squeeze(-1)


def reconstruct_tt_weight_matrix(
    tt_cores: nn.ParameterList,
    input_factors: tuple[int, ...],
    output_factors: tuple[int, ...],
) -> torch.Tensor:
    """Return the dense [out_features, in_features] matrix represented by TT cores."""
    tt_tensor = reconstruct_tt_tensor(tt_cores)
    num_input_dims = len(input_factors)
    input_axes = list(range(num_input_dims))
    output_axes = list(range(num_input_dims, num_input_dims + len(output_factors)))
    permuted = tt_tensor.permute(*output_axes, *reversed(input_axes))
    return permuted.reshape(math.prod(output_factors), math.prod(input_factors))


def is_supported_module(module: nn.Module) -> bool:
    return isinstance(module, nn.Linear) or is_transformers_conv1d(module)


def is_transformers_conv1d(module: nn.Module) -> bool:
    return module.__class__.__name__ == "Conv1D" and hasattr(module, "weight")


def module_features(module: nn.Module) -> tuple[int, int]:
    """Return (in_features, out_features) for supported modules."""
    if isinstance(module, nn.Linear):
        return int(module.in_features), int(module.out_features)
    if is_transformers_conv1d(module):
        in_features, out_features = module.weight.shape
        return int(in_features), int(out_features)
    raise TypeError(f"TT-LoRA supports nn.Linear and transformers Conv1D modules, got {type(module)}.")


def semantic_weight_shape(module: nn.Module) -> tuple[int, int]:
    in_features, out_features = module_features(module)
    return out_features, in_features


class TTLoRAModule(nn.Module):
    def __init__(
        self,
        original_layer: nn.Module,
        target: TTLoRATarget,
        init_seed: int | None = None,
    ) -> None:
        super().__init__()
        if not is_supported_module(original_layer):
            raise TypeError(
                f"TT-LoRA supports nn.Linear and transformers Conv1D modules, got {type(original_layer)}."
            )

        self.original = original_layer
        self.target = target
        self.input_factors = target.input_factors
        self.output_factors = target.output_factors
        self.tt_shape = target.tt_shape
        self.rank = target.rank
        self.alpha = target.alpha
        self.variant = target.variant
        self.tt_cores = generate_tt_cores(
            tt_shape=self.tt_shape,
            rank=self.rank,
            init_seed=init_seed,
            device=self.original.weight.device,
            dtype=self.original.weight.dtype,
        )
        self._validate_against_original()

    @property
    def trainable_parameter_count(self) -> int:
        return tt_parameter_count(self.tt_shape, self.rank)

    def _validate_against_original(self) -> None:
        in_features, out_features = module_features(self.original)
        actual_shape = (out_features, in_features)
        if actual_shape != self.target.weight_shape:
            raise ValueError(
                f"Target expected semantic weight_shape={self.target.weight_shape}, "
                f"but module has semantic weight_shape={actual_shape}."
            )
        if math.prod(self.input_factors) != in_features:
            raise ValueError(
                f"input_factors multiply to {math.prod(self.input_factors)}, "
                f"but module in_features={in_features}."
            )
        if math.prod(self.output_factors) != out_features:
            raise ValueError(
                f"output_factors multiply to {math.prod(self.output_factors)}, "
                f"but module out_features={out_features}."
            )

    def _reshape_input(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        in_features, _ = module_features(self.original)
        if x.shape[-1] != in_features:
            raise ValueError(f"Expected last input dimension {in_features}, got {x.shape[-1]}.")
        leading_shape = x.shape[:-1]
        if x.ndim == 2:
            return x.unsqueeze(1), leading_shape
        if x.ndim >= 3:
            flat = x.reshape(-1, x.shape[-2], x.shape[-1])
            return flat, leading_shape
        raise ValueError(f"TT-LoRA expects input with ndim >= 2, got shape {tuple(x.shape)}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.variant == "contraction":
            x_reshaped, leading_shape = self._reshape_input(x)
            update = tensorized_multiplication(
                x=x_reshaped,
                tt_cores=self.tt_cores,
                input_factors=self.input_factors,
                output_factors=self.output_factors,
            )
            if x.ndim == 2:
                update = update.squeeze(1)
            else:
                _, out_features = module_features(self.original)
                update = update.reshape(*leading_shape, out_features)
            return self.original(x) + update * self.alpha

        dense_update = reconstruct_tt_weight_matrix(
            tt_cores=self.tt_cores,
            input_factors=self.input_factors,
            output_factors=self.output_factors,
        )

        if isinstance(self.original, nn.Linear):
            adapted_weight = self.original.weight + self.alpha * dense_update
            return F.linear(x, adapted_weight, self.original.bias)

        if is_transformers_conv1d(self.original):
            adapted_weight = self.original.weight + self.alpha * dense_update.transpose(0, 1)
            size_out = x.size()[:-1] + (adapted_weight.size(1),)
            x_2d = x.reshape(-1, x.size(-1))
            bias = getattr(self.original, "bias", None)
            if bias is None:
                out = x_2d.matmul(adapted_weight)
            else:
                out = torch.addmm(bias, x_2d, adapted_weight)
            return out.view(size_out)

        raise TypeError(f"Unsupported TT-LoRA target module type: {type(self.original)}")

    def extra_repr(self) -> str:
        return (
            f"variant={self.variant}, rank={self.rank}, alpha={self.alpha}, "
            f"tt_shape={self.tt_shape}, trainable_parameters={self.trainable_parameter_count}"
        )
