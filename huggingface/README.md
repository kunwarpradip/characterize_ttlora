# TTLoRA

TTLoRA is a small PyTorch adapter library for adding Tensor-Train LoRA updates
to selected model weights. It is designed to work with Hugging Face models, but
the core wrapper only depends on PyTorch.

This first version supports non-private training. It does not include Opacus,
Ray launchers, CSV sweeps, or dataset-specific training code.

## Install

From this folder:

```bash
pip install -e .
```

For Hugging Face model examples:

```bash
pip install -e ".[transformers]"
```

## Basic Usage

```python
from transformers import AutoModelForCausalLM

from ttlora import TTLoRAConfig, TTLoRATarget, get_ttlora_model, print_trainable_parameters

model = AutoModelForCausalLM.from_pretrained("gpt2")

config = TTLoRAConfig(
    init_seed=42,
    targets=[
        TTLoRATarget(
            module_name_pattern=r".*attn\.c_attn$",
            weight_shape=(2304, 768),
            input_factors=(32, 4, 6),
            output_factors=(48, 6, 8),
            rank=6,
            alpha=16.0,
            variant="contraction",
        )
    ],
)

model = get_ttlora_model(model, config)
print_trainable_parameters(model)
```

After this, pass `model` to your usual PyTorch loop or Hugging Face `Trainer`.
The base model is frozen by default and only TT-LoRA cores are trainable.

## Target Configuration

Each `TTLoRATarget` describes one set of modules to adapt.

- `module_name_pattern`: Python regular expression matched against
  `model.named_modules()` names.
- `weight_shape`: semantic `[out_features, in_features]` shape.
- `input_factors`: factors whose product equals `in_features`.
- `output_factors`: factors whose product equals `out_features`.
- `rank`: uniform TT rank.
- `alpha`: multiplicative scale applied to the TT update.
- `variant`: either `"contraction"` or `"reconstruction"`.

For GPT-2 `Conv1D`, still provide `weight_shape` as `[out_features,
in_features]`. GPT-2 stores `Conv1D.weight` physically as `[in_features,
out_features]`, but TTLoRA normalizes this for the user.

The internal TT shape is:

```python
tt_shape = input_factors + tuple(reversed(output_factors))
```

This matches the convention used in the characterization training code.

## Multiple Targets

```python
config = TTLoRAConfig(
    targets=[
        TTLoRATarget(
            module_name_pattern=r".*attn\.c_attn$",
            weight_shape=(2304, 768),
            input_factors=(32, 4, 6),
            output_factors=(48, 6, 8),
            rank=6,
            alpha=16.0,
            variant="contraction",
        ),
        TTLoRATarget(
            module_name_pattern=r".*attn\.c_proj$",
            weight_shape=(768, 768),
            input_factors=(32, 4, 6),
            output_factors=(32, 4, 6),
            rank=6,
            alpha=16.0,
            variant="contraction",
        ),
    ]
)
```

Patterns must not overlap. If a target matches no module, `strict=True` raises
an error.

## Save And Load Adapters

```python
from ttlora import save_ttlora_adapters, load_ttlora_adapters

save_ttlora_adapters(model, "my_ttlora_adapter")

base_model = AutoModelForCausalLM.from_pretrained("gpt2")
base_model = load_ttlora_adapters(base_model, "my_ttlora_adapter")
```

The saved directory contains:

- `adapter_config.json`
- `adapter_model.bin`

`load_ttlora_adapters` first applies the saved config to the base model, then
loads only the TT-LoRA core weights.
