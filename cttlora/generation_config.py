from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TTLoRAWeightConfig:
    weight_name: str
    tt_shape: tuple[int, ...]
    input_factors: tuple[int, ...]
    output_factors: tuple[int, ...]
    weight_shape: tuple[int, int] | None = None


@dataclass(slots=True)
class GenerationModelConfig:
    model_name_or_path: str
    tokenizer_name_or_path: str
    ttlora_rank: int
    ttlora_alpha: float
    ttlora_variant: str
    weight_configs: tuple[TTLoRAWeightConfig, ...]
    adaptation_method: str = "ttlora"
    adapt_layers: tuple[int, ...] | None = None
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_target_weights: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


@dataclass(slots=True)
class GenerationDataConfig:
    dataset_name: str
    dataset_root: str
    max_length: int = 1024
    train_split: str = "train"
    validation_split: str = "validation"
    text_column: str = "text"
    training_format: str = "blocks"
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    generation_eval_samples: int | None = None
    generation_eval_max_new_tokens: int = 256


@dataclass(slots=True)
class GenerationTrainingConfig:
    output_dir: str = "runs"
    run_name: str = "generation"
    overwrite_run_dir: bool = False
    resume_from_last_epoch: bool = False
    summary_only: bool = False
    batch_size: int = 8
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    epochs: int = 5
    learning_rate: float = 2e-4
    lr_scheduler: str = "none"
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.0
    num_workers: int = 0
    seed: int = 42
    patience: int = 3
    device: str = "auto"
    log_every_steps: int = 20
    step_metrics_every: int = 1


@dataclass(slots=True)
class GenerationExperimentConfig:
    model: GenerationModelConfig
    data: GenerationDataConfig
    training: GenerationTrainingConfig
    notes: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def run_dir(self) -> Path:
        return Path(self.training.output_dir).expanduser().resolve() / self.training.run_name

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": asdict(self.model),
            "data": asdict(self.data),
            "training": asdict(self.training),
            "notes": self.notes,
            "metadata": self.metadata,
        }
