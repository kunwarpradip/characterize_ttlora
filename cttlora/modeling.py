from __future__ import annotations

from transformers import AutoModelForSequenceClassification

from .config import ModelConfig


HEAD_NAME_HINTS = ("classifier", "score", "qa_outputs", "pre_classifier")


def load_sequence_classification_model(model_config: ModelConfig, num_labels: int):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_path,
        num_labels=num_labels,
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
        for param in model.parameters():
            param.requires_grad = False
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
    if method == "ttlora":
        raise NotImplementedError(
            "TT-LoRA wrapping is not wired yet in this repo. "
            "Use the apply_adaptation() hook in cttlora/modeling.py for the next phase."
        )
    raise ValueError(f"Unsupported adaptation method: {model_config.adaptation_method}")


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
