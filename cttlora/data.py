from __future__ import annotations

from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .config import DataConfig
from .tasks import TaskSpec, normalize_dataset


class SequenceClassificationCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict]) -> dict:
        labels = [int(feature["label"]) for feature in features]
        inputs = [{k: v for k, v in feature.items() if k != "label"} for feature in features]
        batch = self.tokenizer.pad(inputs, padding=True, return_tensors="pt")
        batch["labels"] = batch["input_ids"].new_tensor(labels)
        return batch


def load_local_dataset(data_config: DataConfig) -> DatasetDict:
    dataset_name = data_config.dataset_name
    dataset_dir = Path(data_config.dataset_root).expanduser() / dataset_name
    if not dataset_dir.exists() and dataset_name == "winogrande":
        dataset_dir = Path(data_config.dataset_root).expanduser() / "winogrande_l"
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}. "
            "Update --dataset-root or --dataset-name to match the local stack_v2 layout."
        )
    parquet_splits = {
        split: str(dataset_dir / f"{split}.parquet")
        for split in ("train", "validation", "test")
        if (dataset_dir / f"{split}.parquet").exists()
    }
    if parquet_splits:
        dataset = load_dataset("parquet", data_files=parquet_splits)
    else:
        dataset = load_dataset(str(dataset_dir))
    return normalize_dataset(dataset_name, dataset)


def _select_subset(dataset: Dataset, limit: int | None) -> Dataset:
    if limit is None or limit >= len(dataset):
        return dataset
    return dataset.select(range(limit))


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    task_spec: TaskSpec,
    data_config: DataConfig,
) -> DatasetDict:
    text_columns = task_spec.text_columns

    def tokenize_batch(batch: dict) -> dict:
        texts = [batch[column] for column in text_columns]
        return tokenizer(
            *texts,
            truncation=True,
            max_length=data_config.max_length,
        )

    tokenized = dataset.map(tokenize_batch, batched=True)
    keep_columns = {"input_ids", "attention_mask", data_config.label_column}

    if "token_type_ids" in tokenized[data_config.train_split].column_names:
        keep_columns.add("token_type_ids")

    for split in (data_config.train_split, data_config.validation_split):
        removable = [c for c in tokenized[split].column_names if c not in keep_columns]
        if removable:
            tokenized[split] = tokenized[split].remove_columns(removable)
        tokenized[split] = _select_subset(
            tokenized[split],
            data_config.max_train_samples if split == data_config.train_split else data_config.max_eval_samples,
        )

    tokenized.set_format("python")
    return tokenized


def prepare_phase1_data(
    data_config: DataConfig,
    tokenizer_name_or_path: str,
    task_spec: TaskSpec,
    batch_size: int,
    eval_batch_size: int,
    num_workers: int,
) -> tuple[PreTrainedTokenizerBase, Dataset, Dataset, DataLoader, DataLoader]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token
        if tokenizer.pad_token is None:
            raise ValueError("Tokenizer has no pad_token, eos_token, or sep_token.")

    dataset = load_local_dataset(data_config)
    tokenized = tokenize_dataset(dataset, tokenizer, task_spec, data_config)
    train_dataset = tokenized[data_config.train_split]
    val_dataset = tokenized[data_config.validation_split]
    collator = SequenceClassificationCollator(tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )
    return tokenizer, train_dataset, val_dataset, train_loader, val_loader
