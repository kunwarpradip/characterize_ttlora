from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ModelConfig:
    model_name_or_path: str
    tokenizer_name_or_path: str
    adaptation_method: str = "full"
    target_modules: tuple[str, ...] = ("query", "value")
    adapt_layers: tuple[int, ...] | None = None
    lora_rank: int = 8
    lora_alpha: float = 16.0
    ttlora_rank: int = 5
    ttlora_alpha: float = 8.0
    ttlora_variant: str = "contraction"
    ttlora_shape: tuple[int, ...] = (64, 4, 3, 3, 4, 64)
    ttlora_input_factors: tuple[int, ...] = (64, 4, 3)
    ttlora_output_factors: tuple[int, ...] = (64, 4, 3)


@dataclass(slots=True)
class DataConfig:
    dataset_name: str
    dataset_root: str
    max_length: int = 512 #token truncation length
    train_split: str = "train"
    validation_split: str = "validation"
    label_column: str = "label"
    text_columns: tuple[str, ...] | None = None
    num_labels: int | None = None
    max_train_samples: int | None = None #for debugging, not passed to the trainer directly
    max_eval_samples: int | None = None #for debugging, not passed to the trainer directly


@dataclass(slots=True)
class TrainingConfig:
    output_dir: str = "runs"
    run_name: str = "phase1"
    overwrite_run_dir: bool = False
    summary_only: bool = False
    batch_size: int = 16
    eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    epochs: int = 5
    learning_rate: float = 2e-5  # Interpreted as base/peak LR depending on scheduler choice.
    lr_scheduler: str = "none"
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.0
    num_workers: int = 4
    seed: int = 42
    patience: int = 3
    device: str = "auto"
    log_every_steps: int = 20
    step_metrics_every: int = 1


@dataclass(slots=True)
class ExperimentConfig:
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
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
