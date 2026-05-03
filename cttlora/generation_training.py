from __future__ import annotations

import csv
import json
import logging
import math
import random
import re
import sys
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

from .generation_config import GenerationExperimentConfig
from .generation_data import prepare_generation_data
from .generation_eval import (
    evaluate_cnn_summarization,
    evaluate_gsm8k_exact_match,
    is_cnn_dataset,
    is_gsm8k_dataset,
)
from .generation_modeling import load_generation_checkpoint_into_model, load_generation_model
from .modeling import count_parameters, parameter_groups, trainable_parameter_names
from .training import compute_grad_norm, prepare_run_dir, resolve_device

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


@dataclass(slots=True)
class GenerationEpochRecord:
    epoch: int
    train_loss: float
    train_token_accuracy: float
    validation_loss: float
    validation_perplexity: float
    validation_token_accuracy: float
    avg_grad_norm: float
    max_grad_norm: float
    min_grad_norm: float
    clipped_step_fraction: float
    clipped_steps: int
    learning_rate: float
    epoch_seconds: float
    peak_memory_gb: float
    validation_improved: bool
    best_validation_loss_so_far: float
    epochs_since_improvement: int


@dataclass(slots=True)
class GenerationStepRecord:
    epoch: int
    optimizer_step: int
    optimizer_step_in_epoch: int
    micro_step_end: int
    train_loss: float
    train_token_accuracy: float
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


def _write_history_csv(path: Path, records: list[GenerationEpochRecord]) -> None:
    fieldnames = list(asdict(records[0]).keys()) if records else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def _write_step_history_csv(path: Path, records: list[GenerationStepRecord]) -> None:
    fieldnames = list(asdict(records[0]).keys()) if records else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def _save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _prepare_generation_run_dir(base_run_dir: Path, overwrite: bool, resume: bool) -> Path:
    if resume:
        base_run_dir.mkdir(parents=True, exist_ok=True)
        return base_run_dir
    return prepare_run_dir(base_run_dir, overwrite)


def _setup_run_logger(run_dir: Path) -> tuple[logging.Logger, Path]:
    log_path = run_dir / "training.log"
    logger_name = f"cttlora.generation.{run_dir.resolve()}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger, log_path


def _load_epoch_history_from_csv(path: Path) -> list[GenerationEpochRecord]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    records: list[GenerationEpochRecord] = []
    for row in rows:
        records.append(
            GenerationEpochRecord(
                epoch=int(row["epoch"]),
                train_loss=float(row["train_loss"]),
                train_token_accuracy=float(row["train_token_accuracy"]),
                validation_loss=float(row["validation_loss"]),
                validation_perplexity=float(row["validation_perplexity"]),
                validation_token_accuracy=float(row["validation_token_accuracy"]),
                avg_grad_norm=float(row["avg_grad_norm"]),
                max_grad_norm=float(row["max_grad_norm"]),
                min_grad_norm=float(row["min_grad_norm"]),
                clipped_step_fraction=float(row["clipped_step_fraction"]),
                clipped_steps=int(row["clipped_steps"]),
                learning_rate=float(row["learning_rate"]),
                epoch_seconds=float(row["epoch_seconds"]),
                peak_memory_gb=float(row["peak_memory_gb"]),
                validation_improved=str(row["validation_improved"]).strip().lower() == "true",
                best_validation_loss_so_far=float(row["best_validation_loss_so_far"]),
                epochs_since_improvement=int(row["epochs_since_improvement"]),
            )
        )
    return records


def _load_step_history_from_csv(path: Path) -> list[GenerationStepRecord]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    records: list[GenerationStepRecord] = []
    for row in rows:
        records.append(
            GenerationStepRecord(
                epoch=int(row["epoch"]),
                optimizer_step=int(row["optimizer_step"]),
                optimizer_step_in_epoch=int(row["optimizer_step_in_epoch"]),
                micro_step_end=int(row["micro_step_end"]),
                train_loss=float(row["train_loss"]),
                train_token_accuracy=float(row["train_token_accuracy"]),
                grad_norm_pre_clip=float(row["grad_norm_pre_clip"]),
                clipping_triggered=str(row["clipping_triggered"]).strip().lower() == "true",
                learning_rate=float(row["learning_rate"]),
                examples_seen_epoch=int(row["examples_seen_epoch"]),
                elapsed_seconds=float(row["elapsed_seconds"]),
            )
        )
    return records


_EPOCH_SUMMARY_RE = re.compile(
    r"\[epoch (?P<epoch>\d+)\] "
    r"train_loss=(?P<train_loss>[-+0-9.eE]+) "
    r"train_tok_acc=(?P<train_tok_acc>[-+0-9.eE]+) "
    r"val_loss=(?P<val_loss>[-+0-9.eE]+) "
    r"val_ppl=(?P<val_ppl>[-+0-9.eE]+) "
    r"val_tok_acc=(?P<val_tok_acc>[-+0-9.eE]+) "
    r"peak_mem_gb=(?P<peak_mem_gb>[-+0-9.eE]+) "
    r"improved=(?P<improved>True|False)"
)


def _load_epoch_history_from_log(path: Path) -> list[GenerationEpochRecord]:
    if not path.exists():
        return []
    records: list[GenerationEpochRecord] = []
    best_val_loss = float("inf")
    stale_epochs = 0
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _EPOCH_SUMMARY_RE.search(raw_line)
        if not match:
            continue
        epoch = int(match.group("epoch"))
        val_loss = float(match.group("val_loss"))
        improved = match.group("improved") == "True"
        if improved:
            best_val_loss = val_loss
            stale_epochs = 0
        else:
            stale_epochs += 1
        records.append(
            GenerationEpochRecord(
                epoch=epoch,
                train_loss=float(match.group("train_loss")),
                train_token_accuracy=float(match.group("train_tok_acc")),
                validation_loss=val_loss,
                validation_perplexity=float(match.group("val_ppl")),
                validation_token_accuracy=float(match.group("val_tok_acc")),
                avg_grad_norm=0.0,
                max_grad_norm=0.0,
                min_grad_norm=0.0,
                clipped_step_fraction=0.0,
                clipped_steps=0,
                learning_rate=0.0,
                epoch_seconds=0.0,
                peak_memory_gb=float(match.group("peak_mem_gb")),
                validation_improved=improved,
                best_validation_loss_so_far=best_val_loss,
                epochs_since_improvement=stale_epochs,
            )
        )
    return records


def _find_resume_epoch(run_dir: Path) -> int | None:
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            best_epoch = summary.get("best_epoch")
            if best_epoch is not None:
                return int(best_epoch)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    history_path = run_dir / "history.csv"
    history = _load_epoch_history_from_csv(history_path)
    improved_epochs = [record.epoch for record in history if record.validation_improved]
    if improved_epochs:
        return max(improved_epochs)

    log_path = run_dir / "training.log"
    history_from_log = _load_epoch_history_from_log(log_path)
    improved_epochs = [record.epoch for record in history_from_log if record.validation_improved]
    if improved_epochs:
        return max(improved_epochs)
    return None


def _load_resume_histories(run_dir: Path, resume_epoch: int) -> tuple[list[GenerationEpochRecord], list[GenerationStepRecord]]:
    history_path = run_dir / "history.csv"
    step_history_path = run_dir / "step_history.csv"

    history = _load_epoch_history_from_csv(history_path)
    if not history:
        history = _load_epoch_history_from_log(run_dir / "training.log")
    history = [record for record in history if record.epoch <= resume_epoch]

    step_history = _load_step_history_from_csv(step_history_path)
    step_history = [record for record in step_history if record.epoch <= resume_epoch]
    return history, step_history


def _safe_perplexity(loss_value: float) -> float:
    return float(math.exp(min(loss_value, 20.0)))


def causal_lm_accuracy_counts(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    shifted_logits = logits[..., :-1, :]
    shifted_labels = labels[..., 1:]
    predictions = shifted_logits.argmax(dim=-1)
    valid_mask = shifted_labels != -100
    correct = ((predictions == shifted_labels) & valid_mask).sum().item()
    total = valid_mask.sum().item()
    return int(correct), int(total)


def build_scheduler(optimizer: AdamW, config: GenerationExperimentConfig, total_steps: int):
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


def evaluate_generation(model, dataloader, device: torch.device) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    total_correct_tokens = 0
    total_tokens = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            batch_size = batch["input_ids"].size(0)
            total_loss += outputs.loss.item() * batch_size
            total_examples += batch_size
            correct_tokens, token_total = causal_lm_accuracy_counts(outputs.logits, batch["labels"])
            total_correct_tokens += correct_tokens
            total_tokens += token_total
    loss = total_loss / max(1, total_examples)
    token_accuracy = total_correct_tokens / max(1, total_tokens)
    return loss, _safe_perplexity(loss), token_accuracy


def run_generation_experiment(config: GenerationExperimentConfig) -> dict:
    seed_everything(config.training.seed)
    base_run_dir = config.run_dir()
    run_dir = _prepare_generation_run_dir(
        base_run_dir,
        config.training.overwrite_run_dir,
        config.training.resume_from_last_epoch,
    )
    checkpoints_dir = run_dir / "checkpoints" / "best"
    logger, log_path = _setup_run_logger(run_dir)

    resume_epoch = None
    history: list[GenerationEpochRecord] = []
    step_history: list[GenerationStepRecord] = []
    if config.training.resume_from_last_epoch:
        resume_epoch = _find_resume_epoch(run_dir)
        if resume_epoch is None:
            raise ValueError(
                f"--resume-from-last-epoch was requested, but no saved best epoch could be inferred in {run_dir}."
            )
        if not checkpoints_dir.exists():
            raise FileNotFoundError(
                f"--resume-from-last-epoch was requested, but best checkpoint directory does not exist: {checkpoints_dir}"
            )
        history, step_history = _load_resume_histories(run_dir, resume_epoch)

    logger.info("Starting generation TT-LoRA run")
    logger.info("Run directory: %s", run_dir)
    logger.info("Training log: %s", log_path)
    logger.info("Dataset config: %s", config.data)
    logger.info("Model path: %s", config.model.model_name_or_path)
    logger.info("Tokenizer path: %s", config.model.tokenizer_name_or_path)
    logger.info(
        "Adaptation config: method=%s ttlora_variant=%s ttlora_rank=%s ttlora_alpha=%s "
        "ttlora_weights=%s lora_rank=%s lora_alpha=%s lora_weights=%s adapt_layers=%s",
        config.model.adaptation_method,
        config.model.ttlora_variant,
        config.model.ttlora_rank,
        config.model.ttlora_alpha,
        [weight.weight_name for weight in config.model.weight_configs],
        config.model.lora_rank,
        config.model.lora_alpha,
        list(config.model.lora_target_weights),
        config.model.adapt_layers,
    )
    logger.info("Training config: %s", config.training)

    tokenizer, train_dataset, val_dataset, train_loader, val_loader, data_stats = prepare_generation_data(
        data_config=config.data,
        tokenizer_name_or_path=config.model.tokenizer_name_or_path,
        batch_size=config.training.batch_size,
        eval_batch_size=config.training.eval_batch_size,
        num_workers=config.training.num_workers,
    )
    logger.info(
        "Prepared data: train_rows=%d validation_rows=%d train_blocks=%d validation_blocks=%d "
        "block_size=%d train_batches=%d validation_batches=%d tokenizer=%s",
        data_stats.train_rows,
        data_stats.validation_rows,
        data_stats.train_blocks,
        data_stats.validation_blocks,
        data_stats.block_size,
        len(train_loader),
        len(val_loader),
        type(tokenizer).__name__,
    )
    model = load_generation_model(config.model)
    if config.training.resume_from_last_epoch:
        missing_keys, unexpected_keys = load_generation_checkpoint_into_model(model, checkpoints_dir)
        if missing_keys or unexpected_keys:
            logger.warning(
                "Checkpoint load was non-exact. missing=%s unexpected=%s",
                missing_keys[:20],
                unexpected_keys[:20],
            )
        logger.info(
            "Resuming from best checkpoint: %s (saved_epoch=%d next_epoch=%d)",
            checkpoints_dir,
            resume_epoch,
            resume_epoch + 1,
        )
    parameter_stats = count_parameters(model)
    logger.info("Loaded model: %s", type(model).__name__)
    logger.info("Parameter stats: %s", parameter_stats)

    trainable_names = [] if config.training.summary_only else trainable_parameter_names(model)
    logger.info("Trainable parameter tensors: %d", len(trainable_names))
    device = resolve_device(config.training.device)
    model.to(device)
    logger.info("Resolved device: %s", device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        logger.info("CUDA device name: %s", torch.cuda.get_device_name(device))

    optimizer = AdamW(
        parameter_groups(model, config.training.weight_decay),
        lr=config.training.learning_rate,
    )
    steps_per_epoch = math.ceil(len(train_loader) / max(1, config.training.gradient_accumulation_steps))
    total_steps = max(1, steps_per_epoch * config.training.epochs)
    scheduler = build_scheduler(optimizer, config, total_steps)
    logger.info(
        "Optimization setup: steps_per_epoch=%d total_steps=%d scheduler=%s",
        steps_per_epoch,
        total_steps,
        type(scheduler).__name__ if scheduler is not None else "none",
    )

    if history:
        best_record = min(history, key=lambda record: record.validation_loss)
        best_val_loss = best_record.validation_loss
        best_epoch = best_record.epoch
        last_improvement_epoch = best_epoch
        stale_epochs = history[-1].epochs_since_improvement
    else:
        best_val_loss = float("inf")
        best_epoch = 0
        last_improvement_epoch = 0
        stale_epochs = 0
    global_optimizer_step = step_history[-1].optimizer_step if step_history else 0

    if not config.training.summary_only:
        _save_json(run_dir / "config.json", config.to_dict())
        logger.info("Wrote config.json")

    start_epoch = (resume_epoch + 1) if resume_epoch is not None else 1
    if start_epoch > config.training.epochs:
        logger.info(
            "Resume target is already at or beyond requested epochs: saved_epoch=%d requested_epochs=%d",
            resume_epoch,
            config.training.epochs,
        )
    for epoch in range(start_epoch, config.training.epochs + 1):
        model.train()
        start_time = time.time()
        optimizer.zero_grad(set_to_none=True)
        logger.info("Starting epoch %d/%d", epoch, config.training.epochs)

        running_loss = 0.0
        running_examples = 0
        running_token_correct = 0
        running_token_total = 0
        grad_norm_sum = 0.0
        grad_norm_max = 0.0
        grad_norm_min = float("inf")
        clipped_steps = 0
        optimizer_steps = 0
        micro_loss_sum = 0.0
        micro_examples = 0
        micro_token_correct = 0
        micro_token_total = 0

        train_iterator = enumerate(train_loader, start=1)
        if tqdm is not None:
            train_iterator = enumerate(
                tqdm(
                    train_loader,
                    total=len(train_loader),
                    desc=f"Epoch {epoch}/{config.training.epochs}",
                    unit="batch",
                    leave=False,
                ),
                start=1,
            )

        for step, batch in train_iterator:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / max(1, config.training.gradient_accumulation_steps)
            loss.backward()

            batch_size = batch["input_ids"].size(0)
            correct_tokens, token_total = causal_lm_accuracy_counts(outputs.logits.detach(), batch["labels"])
            running_loss += outputs.loss.item() * batch_size
            running_examples += batch_size
            running_token_correct += correct_tokens
            running_token_total += token_total
            micro_loss_sum += outputs.loss.item() * batch_size
            micro_examples += batch_size
            micro_token_correct += correct_tokens
            micro_token_total += token_total

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
                        GenerationStepRecord(
                            epoch=epoch,
                            optimizer_step=global_optimizer_step,
                            optimizer_step_in_epoch=optimizer_steps,
                            micro_step_end=step,
                            train_loss=micro_loss_sum / max(1, micro_examples),
                            train_token_accuracy=micro_token_correct / max(1, micro_token_total),
                            grad_norm_pre_clip=grad_norm,
                            clipping_triggered=clipping_triggered,
                            learning_rate=current_lr,
                            examples_seen_epoch=running_examples,
                            elapsed_seconds=time.time() - start_time,
                        )
                    )
                micro_loss_sum = 0.0
                micro_examples = 0
                micro_token_correct = 0
                micro_token_total = 0

                if optimizer_steps % max(1, config.training.log_every_steps) == 0:
                    avg_loss = running_loss / max(1, running_examples)
                    avg_token_accuracy = running_token_correct / max(1, running_token_total)
                    logger.info(
                        f"epoch={epoch} step={optimizer_steps}/{steps_per_epoch} "
                        f"train_loss={avg_loss:.4f} train_tok_acc={avg_token_accuracy:.4f} grad_norm={grad_norm:.4f} "
                        f"clipped={clipping_triggered} lr={current_lr:.2e}"
                    )

        train_loss = running_loss / max(1, running_examples)
        train_token_accuracy = running_token_correct / max(1, running_token_total)
        val_loss, val_perplexity, val_token_accuracy = evaluate_generation(model, val_loader, device)
        epoch_seconds = time.time() - start_time
        peak_memory_gb = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 3) if device.type == "cuda" else 0.0
        )
        avg_grad_norm = grad_norm_sum / max(1, optimizer_steps)
        learning_rate = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]["lr"]
        validation_improved = val_loss < best_val_loss

        history.append(
            GenerationEpochRecord(
                epoch=epoch,
                train_loss=train_loss,
                train_token_accuracy=train_token_accuracy,
                validation_loss=val_loss,
                validation_perplexity=val_perplexity,
                validation_token_accuracy=val_token_accuracy,
                avg_grad_norm=avg_grad_norm,
                max_grad_norm=grad_norm_max,
                min_grad_norm=0.0 if grad_norm_min == float("inf") else grad_norm_min,
                clipped_step_fraction=clipped_steps / max(1, optimizer_steps),
                clipped_steps=clipped_steps,
                learning_rate=learning_rate,
                epoch_seconds=epoch_seconds,
                peak_memory_gb=peak_memory_gb,
                validation_improved=validation_improved,
                best_validation_loss_so_far=min(best_val_loss, val_loss),
                epochs_since_improvement=0 if validation_improved else stale_epochs + 1,
            )
        )

        logger.info(
            f"[epoch {epoch}] train_loss={train_loss:.4f} train_tok_acc={train_token_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_ppl={val_perplexity:.4f} val_tok_acc={val_token_accuracy:.4f} "
            f"peak_mem_gb={peak_memory_gb:.2f} improved={validation_improved}"
        )

        if not config.training.summary_only:
            _write_history_csv(run_dir / "history.csv", history)
            _write_step_history_csv(run_dir / "step_history.csv", step_history)

        if validation_improved:
            best_val_loss = val_loss
            best_epoch = epoch
            last_improvement_epoch = epoch
            stale_epochs = 0
            if not config.training.summary_only:
                checkpoints_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(checkpoints_dir)
                tokenizer.save_pretrained(checkpoints_dir)
                logger.info("Saved new best checkpoint: %s", checkpoints_dir)
        else:
            stale_epochs += 1
            if stale_epochs >= config.training.patience:
                logger.info(
                    f"Early stopping triggered after epoch {epoch}. "
                    f"Best validation loss was {best_val_loss:.4f} at epoch {best_epoch}."
                )
                break

    history_path = run_dir / "history.csv"
    step_history_path = run_dir / "step_history.csv"
    if not config.training.summary_only:
        _write_history_csv(history_path, history)
        _write_step_history_csv(step_history_path, step_history)
        logger.info("Wrote history: %s", history_path)
        logger.info("Wrote step history: %s", step_history_path)

    gsm8k_eval_summary = None
    cnn_eval_summary = None
    if is_gsm8k_dataset(config.data.dataset_name):
        gsm8k_predictions_path = None if config.training.summary_only else run_dir / "gsm8k_predictions.csv"
        gsm8k_eval_limit = (
            config.data.generation_eval_samples
            if config.data.generation_eval_samples is not None
            else config.data.max_eval_samples
        )
        logger.info(
            "Starting GSM8K exact-match evaluation: samples=%s max_new_tokens=%d",
            gsm8k_eval_limit if gsm8k_eval_limit is not None else "all",
            config.data.generation_eval_max_new_tokens,
        )
        gsm8k_eval_summary = evaluate_gsm8k_exact_match(
            model=model,
            tokenizer=tokenizer,
            data_config=config.data,
            device=device,
            output_path=gsm8k_predictions_path,
            max_eval_samples=gsm8k_eval_limit,
            batch_size=config.training.eval_batch_size,
            max_new_tokens=config.data.generation_eval_max_new_tokens,
            logger=logger,
        )
        logger.info(
            "GSM8K exact-match evaluation complete: accuracy=%.4f exact_matches=%d/%d predictions=%s",
            gsm8k_eval_summary.exact_match_accuracy,
            gsm8k_eval_summary.exact_matches,
            gsm8k_eval_summary.evaluated_examples,
            gsm8k_eval_summary.predictions_path,
        )
    elif is_cnn_dataset(config.data.dataset_name):
        cnn_predictions_path = None if config.training.summary_only else run_dir / "cnn_predictions.csv"
        cnn_eval_limit = (
            config.data.generation_eval_samples
            if config.data.generation_eval_samples is not None
            else config.data.max_eval_samples
        )
        logger.info(
            "Starting CNN/DailyMail summarization evaluation: samples=%s max_new_tokens=%d",
            cnn_eval_limit if cnn_eval_limit is not None else "all",
            config.data.generation_eval_max_new_tokens,
        )
        cnn_eval_summary = evaluate_cnn_summarization(
            model=model,
            tokenizer=tokenizer,
            data_config=config.data,
            device=device,
            output_path=cnn_predictions_path,
            max_eval_samples=cnn_eval_limit,
            batch_size=config.training.eval_batch_size,
            max_new_tokens=config.data.generation_eval_max_new_tokens,
            logger=logger,
        )
        logger.info(
            "CNN/DailyMail summarization evaluation complete: rouge1=%.4f rouge2=%.4f rougeL=%.4f meteor=%.4f predictions=%s",
            cnn_eval_summary.rouge1,
            cnn_eval_summary.rouge2,
            cnn_eval_summary.rougeL,
            cnn_eval_summary.meteor,
            cnn_eval_summary.predictions_path,
        )

    first_epoch = history[0] if history else None
    best_record = min(history, key=lambda record: record.validation_loss) if history else None
    gsm8k_accuracy = (
        gsm8k_eval_summary.exact_match_accuracy if gsm8k_eval_summary is not None else None
    )
    cnn_rouge1 = cnn_eval_summary.rouge1 if cnn_eval_summary is not None else None
    cnn_rouge2 = cnn_eval_summary.rouge2 if cnn_eval_summary is not None else None
    cnn_rougeL = cnn_eval_summary.rougeL if cnn_eval_summary is not None else None
    cnn_meteor = cnn_eval_summary.meteor if cnn_eval_summary is not None else None
    summary = {
        "dataset_name": config.data.dataset_name,
        "base_run_dir": str(base_run_dir),
        "resolved_run_dir": str(run_dir),
        "train_examples": len(train_dataset),
        "validation_examples": len(val_dataset),
        "train_rows": data_stats.train_rows,
        "validation_rows": data_stats.validation_rows,
        "train_blocks": data_stats.train_blocks,
        "validation_blocks": data_stats.validation_blocks,
        "block_size": data_stats.block_size,
        "model_name_or_path": config.model.model_name_or_path,
        "adaptation_method": config.model.adaptation_method,
        "task_type": "generation",
        "target_weights": (
            [weight.weight_name for weight in config.model.weight_configs]
            if config.model.adaptation_method == "ttlora"
            else list(config.model.lora_target_weights)
            if config.model.adaptation_method == "lora"
            else None
        ),
        "adapt_layers": list(config.model.adapt_layers) if config.model.adapt_layers is not None else None,
        "seed": config.training.seed,
        "learning_rate": config.training.learning_rate,
        "lr_scheduler": config.training.lr_scheduler,
        "batch_size": config.training.batch_size,
        "eval_batch_size": config.training.eval_batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "max_grad_norm_threshold": config.training.max_grad_norm,
        "num_workers": config.training.num_workers,
        "summary_only": config.training.summary_only,
        "resumed_from_last_epoch": config.training.resume_from_last_epoch,
        "resume_checkpoint_epoch": resume_epoch,
        "resume_start_epoch": start_epoch if config.training.resume_from_last_epoch else None,
        "ttlora_variant": config.model.ttlora_variant if config.model.adaptation_method == "ttlora" else None,
        "ttlora_rank": config.model.ttlora_rank if config.model.adaptation_method == "ttlora" else None,
        "ttlora_alpha": config.model.ttlora_alpha if config.model.adaptation_method == "ttlora" else None,
        "lora_rank": config.model.lora_rank if config.model.adaptation_method == "lora" else None,
        "lora_alpha": config.model.lora_alpha if config.model.adaptation_method == "lora" else None,
        "lora_target_weights": (
            list(config.model.lora_target_weights)
            if config.model.adaptation_method == "lora"
            else None
        ),
        "ttlora_weight_configs": [
            {
                "weight_name": weight.weight_name,
                "tt_shape": list(weight.tt_shape),
                "input_factors": list(weight.input_factors),
                "output_factors": list(weight.output_factors),
                "weight_shape": list(weight.weight_shape) if weight.weight_shape is not None else None,
            }
            for weight in config.model.weight_configs
        ],
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "epochs_to_best": best_epoch,
        "last_improvement_epoch": last_improvement_epoch,
        "epochs_since_last_improvement": (len(history) - last_improvement_epoch) if history and last_improvement_epoch else None,
        "initial_validation_loss": first_epoch.validation_loss if first_epoch else None,
        "initial_validation_perplexity": first_epoch.validation_perplexity if first_epoch else None,
        "initial_validation_token_accuracy": first_epoch.validation_token_accuracy if first_epoch else None,
        "best_validation_loss": best_val_loss if history else None,
        "best_validation_perplexity": best_record.validation_perplexity if best_record else None,
        "best_validation_token_accuracy": best_record.validation_token_accuracy if best_record else None,
        "best_validation_accuracy": None,
        "final_validation_accuracy": gsm8k_accuracy,
        "gsm8k_exact_match_accuracy": gsm8k_accuracy,
        "gsm8k_exact_matches": (
            gsm8k_eval_summary.exact_matches if gsm8k_eval_summary is not None else None
        ),
        "gsm8k_evaluated_examples": (
            gsm8k_eval_summary.evaluated_examples if gsm8k_eval_summary is not None else None
        ),
        "gsm8k_predictions_path": (
            gsm8k_eval_summary.predictions_path if gsm8k_eval_summary is not None else None
        ),
        "cnn_rouge1": cnn_rouge1,
        "cnn_rouge2": cnn_rouge2,
        "cnn_rougeL": cnn_rougeL,
        "cnn_meteor": cnn_meteor,
        "cnn_evaluated_examples": (
            cnn_eval_summary.evaluated_examples if cnn_eval_summary is not None else None
        ),
        "cnn_predictions_path": (
            cnn_eval_summary.predictions_path if cnn_eval_summary is not None else None
        ),
        "final_validation_loss": history[-1].validation_loss if history else None,
        "final_validation_perplexity": history[-1].validation_perplexity if history else None,
        "final_validation_token_accuracy": history[-1].validation_token_accuracy if history else None,
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
        "training_log_path": str(log_path),
        "trainable_parameter_names": trainable_names,
        **parameter_stats,
        **config.metadata,
    }
    _save_json(run_dir / "summary.json", summary)
    logger.info("Wrote summary: %s", run_dir / "summary.json")
    logger.info(
        "Run finished: best_epoch=%s best_validation_loss=%s best_validation_perplexity=%s",
        summary["best_epoch"],
        summary["best_validation_loss"],
        summary["best_validation_perplexity"],
    )
    return summary
