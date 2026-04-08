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
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from .config import ExperimentConfig
from .data import prepare_phase1_data
from .modeling import (
    classifier_state_dict,
    classifier_state_summary,
    count_parameters,
    load_sequence_classification_model,
    parameter_groups,
    trainable_parameter_names,
)
from .tasks import get_task_spec


@dataclass(slots=True)
class EpochRecord:
    epoch: int
    train_loss: float
    train_accuracy: float
    validation_loss: float
    validation_accuracy: float
    avg_grad_norm: float
    max_grad_norm: float
    min_grad_norm: float
    clipped_step_fraction: float
    clipped_steps: int
    learning_rate: float
    epoch_seconds: float
    peak_memory_gb: float
    validation_improved: bool
    best_validation_accuracy_so_far: float
    epochs_since_improvement: int


@dataclass(slots=True)
class StepRecord:
    epoch: int
    optimizer_step: int
    optimizer_step_in_epoch: int
    micro_step_end: int
    train_loss: float
    train_accuracy: float
    grad_norm_pre_clip: float
    clipping_triggered: bool
    learning_rate: float
    examples_seen_epoch: int
    elapsed_seconds: float


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


def _write_step_history_csv(path: Path, records: list[StepRecord]) -> None:
    fieldnames = list(asdict(records[0]).keys()) if records else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def _save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def prepare_run_dir(base_run_dir: Path, overwrite: bool) -> Path:
    if overwrite:
        base_run_dir.mkdir(parents=True, exist_ok=True)
        return base_run_dir

    if not base_run_dir.exists() or not any(base_run_dir.iterdir()):
        base_run_dir.mkdir(parents=True, exist_ok=True)
        return base_run_dir

    version = 1
    while True:
        candidate = base_run_dir.parent / f"{base_run_dir.name}_v{version}"
        if not candidate.exists() or not any(candidate.iterdir()):
            candidate.mkdir(parents=True, exist_ok=True)
            print(f"Run directory exists; using versioned run directory: {candidate}")
            return candidate
        version += 1


def save_classifier_init_artifacts(run_dir: Path, model) -> dict:
    classifier_dir = run_dir / "artifacts"
    classifier_dir.mkdir(parents=True, exist_ok=True)
    state = classifier_state_dict(model)
    summary = classifier_state_summary(model)

    state_path = classifier_dir / "classifier_init.pt"
    summary_path = classifier_dir / "classifier_init_summary.json"

    torch.save(state, state_path)
    _save_json(summary_path, summary)

    return {
        "classifier_init_path": str(state_path),
        "classifier_init_summary_path": str(summary_path),
        "classifier_init_sha256": summary["global_sha256"],
        "classifier_parameter_count": summary["parameter_count"],
    }


def build_scheduler(optimizer: AdamW, config: ExperimentConfig, total_steps: int):
    scheduler_name = config.training.lr_scheduler
    if scheduler_name == "none":
        return None
    warmup_steps = int(total_steps * config.training.warmup_ratio)
    if scheduler_name == "linear_with_warmup":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    if scheduler_name == "cosine_with_warmup":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    if scheduler_name == "constant_with_warmup":
        return get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
        )
    raise ValueError(f"Unsupported lr_scheduler: {scheduler_name}")


def run_phase1_experiment(config: ExperimentConfig) -> dict:
    seed_everything(config.training.seed)
    base_run_dir = config.run_dir()
    run_dir = prepare_run_dir(base_run_dir, config.training.overwrite_run_dir)
    checkpoints_dir = run_dir / "checkpoints" / "best"

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
    classifier_init_info = (
        save_classifier_init_artifacts(run_dir, model) if not config.training.summary_only else {}
    )
    parameter_stats = count_parameters(model)
    trainable_names = [] if config.training.summary_only else trainable_parameter_names(model)
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
    scheduler = build_scheduler(optimizer, config, total_steps)

    best_val_accuracy = float("-inf")
    best_epoch = 0
    last_improvement_epoch = 0
    validation_improvement_events = 0
    stale_epochs = 0
    history: list[EpochRecord] = []
    step_history: list[StepRecord] = []
    global_optimizer_step = 0

    if not config.training.summary_only:
        _save_json(run_dir / "config.json", config.to_dict())

    for epoch in range(1, config.training.epochs + 1):
        model.train()
        start_time = time.time()
        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        running_correct = 0
        running_examples = 0
        grad_norm_sum = 0.0
        grad_norm_max = 0.0
        grad_norm_min = float("inf")
        clipped_steps = 0
        optimizer_steps = 0
        micro_loss_sum = 0.0
        micro_correct = 0
        micro_examples = 0

        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / max(1, config.training.gradient_accumulation_steps)
            loss.backward()

            running_loss += outputs.loss.item() * batch["labels"].size(0)
            running_correct += (outputs.logits.argmax(dim=-1) == batch["labels"]).sum().item()
            running_examples += batch["labels"].size(0)
            micro_loss_sum += outputs.loss.item() * batch["labels"].size(0)
            micro_correct += (outputs.logits.argmax(dim=-1) == batch["labels"]).sum().item()
            micro_examples += batch["labels"].size(0)

            should_step = (
                step % config.training.gradient_accumulation_steps == 0
                or step == len(train_loader)
            )
            if should_step:
                grad_norm = compute_grad_norm(model)
                grad_norm_max = max(grad_norm_max, grad_norm)
                grad_norm_min = min(grad_norm_min, grad_norm)
                clipping_triggered = grad_norm > config.training.max_grad_norm
                if clipping_triggered:
                    clipped_steps += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                grad_norm_sum += grad_norm
                optimizer_steps += 1
                global_optimizer_step += 1
                current_lr = (
                    scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]["lr"]
                )

                if optimizer_steps % max(1, config.training.step_metrics_every) == 0:
                    step_history.append(
                        StepRecord(
                            epoch=epoch,
                            optimizer_step=global_optimizer_step,
                            optimizer_step_in_epoch=optimizer_steps,
                            micro_step_end=step,
                            train_loss=micro_loss_sum / max(1, micro_examples),
                            train_accuracy=micro_correct / max(1, micro_examples),
                            grad_norm_pre_clip=grad_norm,
                            clipping_triggered=clipping_triggered,
                            learning_rate=current_lr,
                            examples_seen_epoch=running_examples,
                            elapsed_seconds=time.time() - start_time,
                        )
                    )
                micro_loss_sum = 0.0
                micro_correct = 0
                micro_examples = 0

                if optimizer_steps % max(1, config.training.log_every_steps) == 0:
                    avg_loss = running_loss / max(1, running_examples)
                    avg_acc = running_correct / max(1, running_examples)
                    print(
                        f"epoch={epoch} step={optimizer_steps}/{steps_per_epoch} "
                        f"train_loss={avg_loss:.4f} train_acc={avg_acc:.4f} "
                        f"grad_norm={grad_norm:.4f} clipped={clipping_triggered} lr={current_lr:.2e}"
                    )

        train_loss = running_loss / max(1, running_examples)
        train_accuracy = running_correct / max(1, running_examples)
        val_loss, val_accuracy = evaluate(model, val_loader, device)
        epoch_seconds = time.time() - start_time
        peak_memory_gb = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 3) if device.type == "cuda" else 0.0
        )
        avg_grad_norm = grad_norm_sum / max(1, optimizer_steps)
        learning_rate = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]["lr"]
        validation_improved = val_accuracy > best_val_accuracy

        record = EpochRecord(
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            validation_loss=val_loss,
            validation_accuracy=val_accuracy,
            avg_grad_norm=avg_grad_norm,
            max_grad_norm=grad_norm_max,
            min_grad_norm=0.0 if grad_norm_min == float("inf") else grad_norm_min,
            clipped_step_fraction=clipped_steps / max(1, optimizer_steps),
            clipped_steps=clipped_steps,
            learning_rate=learning_rate,
            epoch_seconds=epoch_seconds,
            peak_memory_gb=peak_memory_gb,
            validation_improved=validation_improved,
            best_validation_accuracy_so_far=max(best_val_accuracy, val_accuracy),
            epochs_since_improvement=0 if validation_improved else stale_epochs + 1,
        )
        history.append(record)

        print(
            f"[epoch {epoch}] "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f} "
            f"peak_mem_gb={peak_memory_gb:.2f}"
        )

        if validation_improved:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            last_improvement_epoch = epoch
            validation_improvement_events += 1
            stale_epochs = 0
            if not config.training.summary_only:
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
    step_history_path = run_dir / "step_history.csv"
    if not config.training.summary_only:
        _write_history_csv(history_path, history)
        _write_step_history_csv(step_history_path, step_history)

    first_epoch = history[0] if history else None
    initial_validation_accuracy = first_epoch.validation_accuracy if first_epoch else None
    initial_validation_loss = first_epoch.validation_loss if first_epoch else None
    first_validation_improvement_epoch = None
    for record in history:
        if record.validation_improved:
            first_validation_improvement_epoch = record.epoch
            break

    summary = {
        "dataset_name": config.data.dataset_name,
        "base_run_dir": str(base_run_dir),
        "resolved_run_dir": str(run_dir),
        "train_examples": len(train_dataset),
        "validation_examples": len(val_dataset),
        "model_name_or_path": config.model.model_name_or_path,
        "adaptation_method": config.model.adaptation_method,
        "ttlora_variant": config.model.ttlora_variant if config.model.adaptation_method == "ttlora" else None,
        "target_modules": list(config.model.target_modules),
        "seed": config.training.seed,
        "learning_rate": config.training.learning_rate,
        "lr_scheduler": config.training.lr_scheduler,
        "batch_size": config.training.batch_size,
        "eval_batch_size": config.training.eval_batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "max_grad_norm_threshold": config.training.max_grad_norm,
        "num_workers": config.training.num_workers,
        "summary_only": config.training.summary_only,
        "ttlora_rank": config.model.ttlora_rank if config.model.adaptation_method == "ttlora" else None,
        "ttlora_alpha": config.model.ttlora_alpha if config.model.adaptation_method == "ttlora" else None,
        "ttlora_shape": list(config.model.ttlora_shape) if config.model.adaptation_method == "ttlora" else None,
        "ttlora_input_factors": list(config.model.ttlora_input_factors)
        if config.model.adaptation_method == "ttlora"
        else None,
        "ttlora_output_factors": list(config.model.ttlora_output_factors)
        if config.model.adaptation_method == "ttlora"
        else None,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "epochs_to_best": best_epoch,
        "last_improvement_epoch": last_improvement_epoch,
        "epochs_since_last_improvement": (len(history) - last_improvement_epoch) if history and last_improvement_epoch else None,
        "validation_improvement_events": validation_improvement_events,
        "first_validation_improvement_epoch": first_validation_improvement_epoch,
        "initial_validation_accuracy": initial_validation_accuracy,
        "initial_validation_loss": initial_validation_loss,
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
        "max_grad_norm": max((record.max_grad_norm for record in history), default=0.0),
        "min_grad_norm": min((record.min_grad_norm for record in history), default=0.0),
        "avg_clipped_step_fraction": (
            sum(record.clipped_step_fraction for record in history) / len(history) if history else 0.0
        ),
        "total_clipped_steps": sum(record.clipped_steps for record in history),
        "step_metrics_logged": len(step_history),
        "history_path": str(history_path) if not config.training.summary_only else None,
        "step_history_path": str(step_history_path) if not config.training.summary_only else None,
        "best_checkpoint_dir": str(checkpoints_dir) if not config.training.summary_only else None,
        "trainable_parameter_names": trainable_names,
        **classifier_init_info,
        **parameter_stats,
        **config.metadata,
    }
    _save_json(run_dir / "summary.json", summary)
    return summary
