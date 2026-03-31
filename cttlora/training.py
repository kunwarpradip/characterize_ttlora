from __future__ import annotations

import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from .config import ExperimentConfig
from .data import prepare_phase1_data
from .modeling import count_parameters, load_sequence_classification_model, parameter_groups, trainable_parameter_names
from .tasks import get_task_spec


@dataclass(slots=True)
class EpochRecord:
    epoch: int
    train_loss: float
    train_accuracy: float
    validation_loss: float
    validation_accuracy: float
    avg_grad_norm: float
    learning_rate: float
    epoch_seconds: float
    peak_memory_gb: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def compute_grad_norm(model) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        grad_norm = param.grad.detach().data.norm(2)
        total += grad_norm.item() ** 2
    return math.sqrt(total)


def evaluate(model, dataloader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            labels = batch["labels"]
            total_loss += outputs.loss.item() * labels.size(0)
            total_correct += (outputs.logits.argmax(dim=-1) == labels).sum().item()
            total_examples += labels.size(0)
    loss = total_loss / max(1, total_examples)
    accuracy = total_correct / max(1, total_examples)
    return loss, accuracy


def _write_history_csv(path: Path, records: list[EpochRecord]) -> None:
    fieldnames = list(asdict(records[0]).keys()) if records else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def _save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def run_phase1_experiment(config: ExperimentConfig) -> dict:
    seed_everything(config.training.seed)
    run_dir = config.run_dir()
    checkpoints_dir = run_dir / "checkpoints" / "best"
    run_dir.mkdir(parents=True, exist_ok=True)

    task_spec = get_task_spec(
        config.data.dataset_name,
        text_columns_override=config.data.text_columns,
        num_labels_override=config.data.num_labels,
    )
    tokenizer, train_dataset, val_dataset, train_loader, val_loader = prepare_phase1_data(
        data_config=config.data,
        tokenizer_name_or_path=config.model.tokenizer_name_or_path,
        task_spec=task_spec,
        batch_size=config.training.batch_size,
        eval_batch_size=config.training.eval_batch_size,
        num_workers=config.training.num_workers,
    )

    model = load_sequence_classification_model(config.model, task_spec.num_labels)
    parameter_stats = count_parameters(model)
    trainable_names = trainable_parameter_names(model)
    device = resolve_device(config.training.device)
    model.to(device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    optimizer = AdamW(
        parameter_groups(model, config.training.weight_decay),
        lr=config.training.learning_rate,
    )
    steps_per_epoch = math.ceil(len(train_loader) / max(1, config.training.gradient_accumulation_steps))
    total_steps = max(1, steps_per_epoch * config.training.epochs)
    warmup_steps = int(total_steps * config.training.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_accuracy = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[EpochRecord] = []

    _save_json(run_dir / "config.json", config.to_dict())

    for epoch in range(1, config.training.epochs + 1):
        model.train()
        start_time = time.time()
        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        running_correct = 0
        running_examples = 0
        grad_norm_sum = 0.0
        optimizer_steps = 0

        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / max(1, config.training.gradient_accumulation_steps)
            loss.backward()

            running_loss += outputs.loss.item() * batch["labels"].size(0)
            running_correct += (outputs.logits.argmax(dim=-1) == batch["labels"]).sum().item()
            running_examples += batch["labels"].size(0)

            should_step = (
                step % config.training.gradient_accumulation_steps == 0
                or step == len(train_loader)
            )
            if should_step:
                grad_norm = compute_grad_norm(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                grad_norm_sum += grad_norm
                optimizer_steps += 1

                if optimizer_steps % max(1, config.training.log_every_steps) == 0:
                    avg_loss = running_loss / max(1, running_examples)
                    avg_acc = running_correct / max(1, running_examples)
                    print(
                        f"epoch={epoch} step={optimizer_steps}/{steps_per_epoch} "
                        f"train_loss={avg_loss:.4f} train_acc={avg_acc:.4f} "
                        f"grad_norm={grad_norm:.4f} lr={scheduler.get_last_lr()[0]:.2e}"
                    )

        train_loss = running_loss / max(1, running_examples)
        train_accuracy = running_correct / max(1, running_examples)
        val_loss, val_accuracy = evaluate(model, val_loader, device)
        epoch_seconds = time.time() - start_time
        peak_memory_gb = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 3) if device.type == "cuda" else 0.0
        )
        avg_grad_norm = grad_norm_sum / max(1, optimizer_steps)
        learning_rate = scheduler.get_last_lr()[0]

        record = EpochRecord(
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            validation_loss=val_loss,
            validation_accuracy=val_accuracy,
            avg_grad_norm=avg_grad_norm,
            learning_rate=learning_rate,
            epoch_seconds=epoch_seconds,
            peak_memory_gb=peak_memory_gb,
        )
        history.append(record)

        print(
            f"[epoch {epoch}] "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f} "
            f"peak_mem_gb={peak_memory_gb:.2f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            stale_epochs = 0
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(checkpoints_dir)
            tokenizer.save_pretrained(checkpoints_dir)
        else:
            stale_epochs += 1
            if stale_epochs >= config.training.patience:
                print(
                    f"Early stopping triggered after epoch {epoch}. "
                    f"Best validation accuracy was {best_val_accuracy:.4f} at epoch {best_epoch}."
                )
                break

    history_path = run_dir / "history.csv"
    _write_history_csv(history_path, history)

    summary = {
        "dataset_name": config.data.dataset_name,
        "train_examples": len(train_dataset),
        "validation_examples": len(val_dataset),
        "model_name_or_path": config.model.model_name_or_path,
        "adaptation_method": config.model.adaptation_method,
        "target_modules": list(config.model.target_modules),
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "epochs_to_best": best_epoch,
        "best_validation_accuracy": best_val_accuracy,
        "final_validation_accuracy": history[-1].validation_accuracy if history else None,
        "final_validation_loss": history[-1].validation_loss if history else None,
        "max_peak_memory_gb": max((record.peak_memory_gb for record in history), default=0.0),
        "avg_epoch_seconds": (
            sum(record.epoch_seconds for record in history) / len(history) if history else 0.0
        ),
        "avg_grad_norm": (
            sum(record.avg_grad_norm for record in history) / len(history) if history else 0.0
        ),
        "history_path": str(history_path),
        "best_checkpoint_dir": str(checkpoints_dir),
        "trainable_parameter_names": trainable_names,
        **parameter_stats,
        **config.metadata,
    }
    _save_json(run_dir / "summary.json", summary)
    return summary
