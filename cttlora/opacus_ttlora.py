from __future__ import annotations

import torch
import torch.nn as nn

from .adapters import (
    TTLoRALinearWrapper,
    TTLoRALinearWrapperContraction,
    capture_tt_contraction_activations,
    reconstruct_tt_weight_matrix,
)
from .generation_modeling import TTLoRAGenerationWrapper

try:
    from opacus.grad_sample import register_grad_sampler
    from opacus.grad_sample.grad_sample_module import GradSampleModule
    import opacus.grad_sample.grad_sample_module as grad_sample_module_lib
except ImportError:  # pragma: no cover - optional dependency
    register_grad_sampler = None
    GradSampleModule = None
    grad_sample_module_lib = None

try:  # pragma: no cover - optional dependency
    from opacus.grad_sample.grad_sample_module_fast_gradient_clipping import GradSampleModuleFastGradientClipping
    import opacus.grad_sample.grad_sample_module_fast_gradient_clipping as grad_sample_fast_gc_lib
except ImportError:  # pragma: no cover - optional dependency
    GradSampleModuleFastGradientClipping = None
    grad_sample_fast_gc_lib = None


def _numel_except_first(tensor: torch.Tensor) -> int:
    return int(tensor.numel() // tensor.shape[0])


def _maybe_compute_linear_grad_sample(
    original: nn.Module,
    x_flat: torch.Tensor,
    d_y: torch.Tensor,
) -> dict[nn.Parameter, torch.Tensor]:
    grad_samples: dict[nn.Parameter, torch.Tensor] = {}
    if isinstance(original, nn.Linear):
        if original.weight.requires_grad:
            if x_flat.dim() == 3 and d_y.dim() == 3:
                grad_samples[original.weight] = torch.bmm(d_y.permute(0, 2, 1), x_flat)
            elif x_flat.dim() == 2 and d_y.dim() == 2:
                grad_samples[original.weight] = d_y[:, :, None] * x_flat[:, None, :]
        if original.bias is not None and original.bias.requires_grad:
            grad_samples[original.bias] = d_y.sum(dim=1) if d_y.dim() == 3 else d_y
    return grad_samples


def _tt_contraction_grad_samples(
    module: nn.Module,
    x_flat: torch.Tensor,
    d_y: torch.Tensor,
    cached_activations: tuple[torch.Tensor, ...],
) -> dict[nn.Parameter, torch.Tensor]:
    if x_flat.dim() == 2:
        x_flat = x_flat.unsqueeze(1)
    if d_y.dim() == 2:
        d_y = d_y.unsqueeze(1)

    batch_size = x_flat.size(0)
    num_input_cores = len(module.input_factors)
    num_output_cores = len(module.output_factors)
    tt_cores = module.tt_cores
    grad_samples: dict[nn.Parameter, torch.Tensor] = {}

    grad_samples.update(_maybe_compute_linear_grad_sample(module.original, x_flat, d_y))

    cur = (d_y * module.alpha).reshape(batch_size, x_flat.size(1), *module.output_factors).unsqueeze(1).contiguous()

    for output_idx in range(num_output_cores - 1, -1, -1):
        core = tt_cores[num_input_cores + output_idx]
        rank_in, output_dim, rank_out = core.shape
        state_in = cached_activations[1 + num_input_cores + output_idx]

        z_pre = state_in.numel() // (batch_size * rank_in)
        z_total = _numel_except_first(cur) // rank_out
        if z_total % (z_pre * output_dim) != 0:
            raise ValueError("Inconsistent TT output-core shapes while computing per-sample gradients.")
        z_post = z_total // (z_pre * output_dim)

        cur_reshaped = cur.reshape(batch_size, rank_out, z_pre, output_dim, z_post)
        state_flat = state_in.reshape(batch_size, rank_in, z_pre)
        grad_samples[core] = torch.einsum("b a z n p, b r z -> b r n a", cur_reshaped, state_flat)
        cur = torch.einsum("b a z n p, r n a -> b r z", cur_reshaped, core).reshape(state_in.shape).contiguous()

    for input_idx in range(num_input_cores - 1, -1, -1):
        core = tt_cores[input_idx]
        rank_in, input_dim, rank_out = core.shape
        state_in = cached_activations[1 + input_idx]

        z_width = state_in.numel() // (batch_size * rank_in * input_dim)
        d_state = cur.reshape(batch_size, rank_out, z_width)
        state_flat = state_in.reshape(batch_size, rank_in, z_width, input_dim)
        grad_samples[core] = torch.einsum("b a z, b r z m -> b r m a", d_state, state_flat)
        cur = torch.einsum("b a z, r m a -> b r z m", d_state, core).contiguous()

    return grad_samples


def _dense_tt_update_grad_sample(
    x_flat: torch.Tensor,
    d_y: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    if x_flat.dim() == 2:
        x_flat = x_flat.unsqueeze(1)
    if d_y.dim() == 2:
        d_y = d_y.unsqueeze(1)
    return alpha * torch.bmm(d_y.permute(0, 2, 1), x_flat)


def _tt_reconstruction_grad_samples(
    module: nn.Module,
    x_flat: torch.Tensor,
    d_y: torch.Tensor,
) -> dict[nn.Parameter, torch.Tensor]:
    grad_samples: dict[nn.Parameter, torch.Tensor] = {}
    grad_samples.update(_maybe_compute_linear_grad_sample(module.original, x_flat, d_y))

    grad_dense = _dense_tt_update_grad_sample(x_flat=x_flat, d_y=d_y, alpha=module.alpha).detach()
    tt_cores = list(module.tt_cores)
    per_core_samples: list[list[torch.Tensor]] = [[] for _ in tt_cores]

    with torch.enable_grad():
        dense_update = reconstruct_tt_weight_matrix(
            tt_cores=module.tt_cores,
            input_factors=module.input_factors,
            output_factors=module.output_factors,
        )
        for sample_idx in range(grad_dense.size(0)):
            scalar_objective = (dense_update * grad_dense[sample_idx]).sum()
            core_grads = torch.autograd.grad(
                scalar_objective,
                tt_cores,
                retain_graph=sample_idx + 1 < grad_dense.size(0),
                allow_unused=False,
            )
            for core_idx, core_grad in enumerate(core_grads):
                per_core_samples[core_idx].append(core_grad.detach())

    for core, sample_list in zip(tt_cores, per_core_samples):
        grad_samples[core] = torch.stack(sample_list, dim=0)

    return grad_samples


def _get_or_build_cached_activations(
    module: nn.Module,
    activations,
) -> tuple[torch.Tensor, ...]:
    cached = getattr(module, "_tt_opacus_activations", None)
    if cached is not None:
        return cached

    if isinstance(activations, (list, tuple)) and activations:
        x_flat = activations[0]
    else:
        x_flat = activations
    if not torch.is_tensor(x_flat):
        raise ValueError("Could not recover TT-LoRA activations for Opacus grad sampling.")
    if x_flat.dim() == 2:
        x_flat = x_flat.unsqueeze(1)
    return capture_tt_contraction_activations(
        x=x_flat,
        tt_cores=module.tt_cores,
        input_factors=module.input_factors,
        output_factors=module.output_factors,
    )


if register_grad_sampler is not None:
    _TTLORA_MODULE_TYPES = (TTLoRAGenerationWrapper, TTLoRALinearWrapperContraction, TTLoRALinearWrapper)

    def _ttlora_requires_grad(module: nn.Module) -> bool:
        return any(parameter.requires_grad for _, parameter in _ttlora_trainable_parameters(module))

    def _ttlora_trainable_parameters(module: nn.Module) -> list[tuple[str, nn.Parameter]]:
        params: list[tuple[str, nn.Parameter]] = []
        if hasattr(module, "tt_cores"):
            for idx, parameter in enumerate(module.tt_cores):
                if parameter.requires_grad:
                    params.append((f"tt_cores.{idx}", parameter))
        if hasattr(module, "original"):
            for name, parameter in module.original.named_parameters(recurse=False):
                if parameter.requires_grad:
                    params.append((f"original.{name}", parameter))
        return params

    def _patch_grad_sample_helpers(lib) -> None:
        if lib is None:
            return
        original_requires_grad = lib.requires_grad
        original_trainable_parameters = lib.trainable_parameters
        if not getattr(original_requires_grad, "_ttlora_patched", False):
            def requires_grad_with_ttlora(module: nn.Module):
                if isinstance(module, _TTLORA_MODULE_TYPES):
                    return _ttlora_requires_grad(module)
                return original_requires_grad(module)

            requires_grad_with_ttlora._ttlora_patched = True  # type: ignore[attr-defined]
            lib.requires_grad = requires_grad_with_ttlora
        if not getattr(original_trainable_parameters, "_ttlora_patched", False):
            def trainable_parameters_with_ttlora(module: nn.Module):
                if isinstance(module, _TTLORA_MODULE_TYPES):
                    return _ttlora_trainable_parameters(module)
                return original_trainable_parameters(module)

            trainable_parameters_with_ttlora._ttlora_patched = True  # type: ignore[attr-defined]
            lib.trainable_parameters = trainable_parameters_with_ttlora

    def _patch_iterate_submodules(module_class) -> None:
        if module_class is None:
            return
        original_iterate_submodules = module_class.iterate_submodules
        if getattr(original_iterate_submodules, "_ttlora_patched", False):
            return

        def iterate_submodules_with_ttlora(self, module):
            if isinstance(module, _TTLORA_MODULE_TYPES):
                yield module
                return
            yield from original_iterate_submodules(self, module)

        iterate_submodules_with_ttlora._ttlora_patched = True  # type: ignore[attr-defined]
        module_class.iterate_submodules = iterate_submodules_with_ttlora

    _patch_grad_sample_helpers(grad_sample_module_lib)
    _patch_grad_sample_helpers(grad_sample_fast_gc_lib)
    _patch_iterate_submodules(GradSampleModule)
    _patch_iterate_submodules(GradSampleModuleFastGradientClipping)

    @register_grad_sampler(_TTLORA_MODULE_TYPES)
    def compute_ttlora_grad_sample(module, activations, backprops):
        if isinstance(activations, (list, tuple)) and activations:
            x_flat = activations[0]
        else:
            x_flat = activations
        if not torch.is_tensor(x_flat):
            raise ValueError("Expected tensor activations for TT-LoRA Opacus grad sampler.")
        if getattr(module, "mode", "contraction") == "contraction":
            cached_activations = _get_or_build_cached_activations(module, activations)
            grad_samples = _tt_contraction_grad_samples(
                module,
                x_flat=x_flat,
                d_y=backprops,
                cached_activations=cached_activations,
            )
        else:
            grad_samples = _tt_reconstruction_grad_samples(module, x_flat=x_flat, d_y=backprops)
        if hasattr(module, "_tt_opacus_activations"):
            delattr(module, "_tt_opacus_activations")
        return grad_samples
