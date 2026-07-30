from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .config import TTLoRAConfig
from .model import get_ttlora_model

ADAPTER_WEIGHTS_NAME = "adapter_model.bin"
ADAPTER_CONFIG_NAME = "adapter_config.json"


def ttlora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if ".tt_cores." in key
    }


def save_ttlora_adapters(
    model: nn.Module,
    save_directory: str | Path,
    config: TTLoRAConfig | None = None,
) -> None:
    save_path = Path(save_directory)
    save_path.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = getattr(model, "ttlora_config", None)
    if config is None:
        raise ValueError("No TTLoRAConfig was provided and model.ttlora_config is not set.")

    config.to_json_file(save_path / ADAPTER_CONFIG_NAME)
    torch.save(ttlora_state_dict(model), save_path / ADAPTER_WEIGHTS_NAME)


def load_ttlora_adapters(
    model: nn.Module,
    load_directory: str | Path,
    config: TTLoRAConfig | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> nn.Module:
    load_path = Path(load_directory)
    if config is None:
        config = TTLoRAConfig.from_json_file(load_path / ADAPTER_CONFIG_NAME)

    model = get_ttlora_model(model, config)
    state_dict = torch.load(load_path / ADAPTER_WEIGHTS_NAME, map_location=map_location)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    expected_adapter_keys = set(ttlora_state_dict(model))
    missing_adapter_keys = sorted(key for key in missing if key in expected_adapter_keys)
    if strict and (missing_adapter_keys or unexpected):
        raise ValueError(
            "Could not load TT-LoRA adapter weights cleanly. "
            f"missing_adapter_keys={missing_adapter_keys}, unexpected_keys={unexpected}."
        )
    return model
