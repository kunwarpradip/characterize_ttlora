from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def freeze_model_parameters(model) -> None:
    for param in model.parameters():
        param.requires_grad = False


def ttlora_rank_list(rank: int, tt_shape: tuple[int, ...]) -> tuple[int, ...]:
    if rank < 1:
        raise ValueError("TT-LoRA rank must be >= 1.")
    return (1, *([rank] * (len(tt_shape) - 1)), 1)


def generate_tt_cores(tt_shape: tuple[int, ...], tt_rank: tuple[int, ...]) -> nn.ParameterList:
    cores = nn.ParameterList()
    for idx, dim in enumerate(tt_shape):
        core_shape = (tt_rank[idx], dim, tt_rank[idx + 1])
        core = nn.Parameter(torch.empty(core_shape))
        nn.init.kaiming_uniform_(core, a=math.sqrt(8))
        core.data /= core.data.norm() + 1e-6
        cores.append(core)
    return cores


class LoRALinearWrapper(nn.Module):
    def __init__(self, original_layer: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        if not isinstance(original_layer, nn.Linear):
            raise TypeError(f"LoRA wrapper currently supports nn.Linear only, got {type(original_layer)}")
        if rank < 1:
            raise ValueError("LoRA rank must be >= 1.")

        self.original = original_layer
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Linear(original_layer.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, original_layer.out_features, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original(x) + self.lora_B(self.lora_A(x)) * self.scaling


def tensorized_multiplication(
    x: torch.Tensor,
    tt_cores: nn.ParameterList,
    input_factors: tuple[int, ...],
    output_factors: tuple[int, ...],
) -> torch.Tensor:
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


class TTLoRALinearWrapperContraction(nn.Module):
    def __init__(
        self,
        original_layer: nn.Linear,
        tt_shape: tuple[int, ...],
        rank: int,
        alpha: float,
        input_factors: tuple[int, ...],
        output_factors: tuple[int, ...],
    ) -> None:
        super().__init__()
        if not isinstance(original_layer, nn.Linear):
            raise TypeError(f"TT-LoRA wrapper currently supports nn.Linear only, got {type(original_layer)}")

        self.original = original_layer
        self.tt_shape = tt_shape
        self.tt_rank = ttlora_rank_list(rank, tt_shape)
        self.alpha = alpha
        self.input_factors = input_factors
        self.output_factors = output_factors

        if math.prod(input_factors) != original_layer.in_features:
            raise ValueError(
                f"Input TT factors {input_factors} multiply to {math.prod(input_factors)}, "
                f"but layer in_features is {original_layer.in_features}."
            )
        if math.prod(output_factors) != original_layer.out_features:
            raise ValueError(
                f"Output TT factors {output_factors} multiply to {math.prod(output_factors)}, "
                f"but layer out_features is {original_layer.out_features}."
            )
        if math.prod(tt_shape) != original_layer.in_features * original_layer.out_features:
            raise ValueError(
                f"TT shape {tt_shape} multiplies to {math.prod(tt_shape)}, "
                f"but layer matrix size is {original_layer.in_features * original_layer.out_features}."
            )
        if len(tt_shape) != len(input_factors) + len(output_factors):
            raise ValueError(
                "TT shape length must equal len(input_factors) + len(output_factors)."
            )

        self.tt_cores = generate_tt_cores(self.tt_shape, self.tt_rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = tensorized_multiplication(
            x=x,
            tt_cores=self.tt_cores,
            input_factors=self.input_factors,
            output_factors=self.output_factors,
        )
        return self.original(x) + update * self.alpha


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
    """Return the dense [out_features, in_features] matrix equivalent to contraction mode."""
    tt_tensor = reconstruct_tt_tensor(tt_cores)
    num_input_dims = len(input_factors)
    input_axes = list(range(num_input_dims))
    output_axes = list(range(num_input_dims, num_input_dims + len(output_factors)))
    permuted = tt_tensor.permute(*output_axes, *reversed(input_axes))
    return permuted.reshape(math.prod(output_factors), math.prod(input_factors))


class TTLoRALinearWrapper(nn.Module):
    def __init__(
        self,
        original_layer: nn.Linear,
        tt_shape: tuple[int, ...],
        rank: int,
        alpha: float,
        mode: str,
        input_factors: tuple[int, ...],
        output_factors: tuple[int, ...],
    ) -> None:
        super().__init__()
        if not isinstance(original_layer, nn.Linear):
            raise TypeError(f"TT-LoRA wrapper currently supports nn.Linear only, got {type(original_layer)}")
        if mode not in {"contraction", "reconstruction"}:
            raise ValueError(f"Unsupported TT-LoRA mode '{mode}'. Expected 'contraction' or 'reconstruction'.")

        self.original = original_layer
        self.tt_shape = tt_shape
        self.tt_rank = ttlora_rank_list(rank, tt_shape)
        self.alpha = alpha
        self.mode = mode
        self.input_factors = input_factors
        self.output_factors = output_factors

        if math.prod(input_factors) != original_layer.in_features:
            raise ValueError(
                f"Input TT factors {input_factors} multiply to {math.prod(input_factors)}, "
                f"but layer in_features is {original_layer.in_features}."
            )
        if math.prod(output_factors) != original_layer.out_features:
            raise ValueError(
                f"Output TT factors {output_factors} multiply to {math.prod(output_factors)}, "
                f"but layer out_features is {original_layer.out_features}."
            )
        if len(tt_shape) != len(input_factors) + len(output_factors):
            raise ValueError("TT shape length must equal len(input_factors) + len(output_factors).")
        if tuple(tt_shape[: len(input_factors)]) != input_factors:
            raise ValueError(
                f"TT shape input prefix {tt_shape[: len(input_factors)]} does not match input_factors {input_factors}."
            )
        if tuple(tt_shape[len(input_factors) :]) != output_factors[::-1]:
            raise ValueError(
                "TT shape output suffix must equal reversed output_factors so contraction and reconstruction "
                "represent the same linear operator."
            )
        if math.prod(tt_shape) != original_layer.in_features * original_layer.out_features:
            raise ValueError(
                f"TT shape {tt_shape} multiplies to {math.prod(tt_shape)}, "
                f"but layer matrix size is {original_layer.in_features * original_layer.out_features}."
            )
        self.tt_cores = generate_tt_cores(self.tt_shape, self.tt_rank)

    def _reshape_input(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        if x.shape[-1] != self.original.in_features:
            raise ValueError(
                f"Expected last dimension {self.original.in_features}, got {x.shape[-1]} for TT-LoRA input."
            )
        leading_shape = x.shape[:-1]
        if x.ndim == 2:
            return x.unsqueeze(1), leading_shape
        if x.ndim >= 3:
            flat = x.reshape(-1, x.shape[-2], x.shape[-1])
            return flat, leading_shape
        raise ValueError(f"TT-LoRA wrapper expects input with ndim >= 2, got shape {tuple(x.shape)}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "contraction":
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
                update = update.reshape(*leading_shape, self.original.out_features)
            return self.original(x) + update * self.alpha

        dense_update_weight = reconstruct_tt_weight_matrix(
            tt_cores=self.tt_cores,
            input_factors=self.input_factors,
            output_factors=self.output_factors,
        )
        adapted_weight = self.original.weight + self.alpha * dense_update_weight
        return F.linear(x, adapted_weight, self.original.bias)


class TTLoRALinearWrapperContraction(TTLoRALinearWrapper):
    def __init__(
        self,
        original_layer: nn.Linear,
        tt_shape: tuple[int, ...],
        rank: int,
        alpha: float,
        input_factors: tuple[int, ...],
        output_factors: tuple[int, ...],
    ) -> None:
        super().__init__(
            original_layer=original_layer,
            tt_shape=tt_shape,
            rank=rank,
            alpha=alpha,
            mode="contraction",
            input_factors=input_factors,
            output_factors=output_factors,
        )


class TTLoRALinearWrapperReconstruction(TTLoRALinearWrapper):
    def __init__(
        self,
        original_layer: nn.Linear,
        tt_shape: tuple[int, ...],
        rank: int,
        alpha: float,
        input_factors: tuple[int, ...],
        output_factors: tuple[int, ...],
    ) -> None:
        super().__init__(
            original_layer=original_layer,
            tt_shape=tt_shape,
            rank=rank,
            alpha=alpha,
            mode="reconstruction",
            input_factors=input_factors,
            output_factors=output_factors,
        )
