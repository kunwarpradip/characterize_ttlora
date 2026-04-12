from __future__ import annotations

from itertools import chain
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase, default_data_collator

from .generation_config import GenerationDataConfig


def _select_subset(dataset: Dataset, limit: int | None) -> Dataset:
    if limit is None or limit >= len(dataset):
        return dataset
    return dataset.select(range(limit))


def load_local_generation_dataset(data_config: GenerationDataConfig) -> DatasetDict:
    dataset_dir = Path(data_config.dataset_root).expanduser() / data_config.dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    if (dataset_dir / "dataset_dict.json").exists():
        dataset = load_from_disk(str(dataset_dir))
    else:
        jsonl_train = dataset_dir / "cleaned_short_train_scrubbed.jsonl"
        jsonl_valid = dataset_dir / "cleaned_short_test_scrubbed.jsonl"
        csv_train = dataset_dir / "cleaned_short_train_scrubbed.csv"
        csv_valid = dataset_dir / "cleaned_short_test_scrubbed.csv"

        if jsonl_train.exists() and jsonl_valid.exists():
            dataset = load_dataset(
                "json",
                data_files={"train": str(jsonl_train), "validation": str(jsonl_valid)},
            )
        elif csv_train.exists() and csv_valid.exists():
            dataset = load_dataset(
                "csv",
                data_files={"train": str(csv_train), "validation": str(csv_valid)},
            )
        else:
            raise FileNotFoundError(
                f"Could not infer a local text-generation dataset layout inside {dataset_dir}"
            )

    if "validation" not in dataset and "test" in dataset:
        dataset = DatasetDict(
            {
                "train": dataset[data_config.train_split],
                "validation": dataset["test"],
            }
        )

    sample_columns = dataset[data_config.train_split].column_names
    text_column = data_config.text_column
    if text_column not in sample_columns:
        if data_config.dataset_name == "ptb" and "sentence" in sample_columns:
            text_column = "sentence"
        else:
            text_column = sample_columns[0]

    if text_column != "text":
        dataset = DatasetDict(
            {
                split: dataset[split].rename_column(text_column, "text")
                if text_column in dataset[split].column_names
                else dataset[split]
                for split in dataset.keys()
            }
        )

    return dataset


def prepare_generation_data(
    data_config: GenerationDataConfig,
    tokenizer_name_or_path: str,
    batch_size: int,
    eval_batch_size: int,
    num_workers: int,
) -> tuple[PreTrainedTokenizerBase, Dataset, Dataset, DataLoader, DataLoader]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token is None:
            raise ValueError("Tokenizer has no pad_token or eos_token.")

    dataset = load_local_generation_dataset(data_config)

    def tokenize_batch(batch: dict) -> dict:
        return tokenizer(batch["text"])

    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset[data_config.train_split].column_names,
    )

    block_size = min(data_config.max_length, tokenizer.model_max_length)

    def group_texts(examples: dict) -> dict:
        concatenated = {key: list(chain(*examples[key])) for key in examples.keys()}
        total_length = len(concatenated["input_ids"])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            key: [values[i : i + block_size] for i in range(0, total_length, block_size)]
            for key, values in concatenated.items()
        }
        result["labels"] = list(result["input_ids"])
        return result

    grouped = tokenized.map(group_texts, batched=True)
    train_dataset = _select_subset(grouped[data_config.train_split], data_config.max_train_samples)
    val_dataset = _select_subset(grouped[data_config.validation_split], data_config.max_eval_samples)
    train_dataset.set_format("torch")
    val_dataset.set_format("torch")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=default_data_collator,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=default_data_collator,
        pin_memory=True,
    )
    return tokenizer, train_dataset, val_dataset, train_loader, val_loader
