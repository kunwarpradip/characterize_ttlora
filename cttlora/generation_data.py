from __future__ import annotations

from itertools import chain
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase, default_data_collator

from .generation_config import GenerationDataConfig

try:
    import sentencepiece as spm
except ImportError:
    spm = None


def _select_subset(dataset: Dataset, limit: int | None) -> Dataset:
    if limit is None or limit >= len(dataset):
        return dataset
    return dataset.select(range(limit))


def resolve_local_dataset_dir(data_config: GenerationDataConfig) -> Path:
    dataset_root = Path(data_config.dataset_root).expanduser()
    dataset_name = data_config.dataset_name
    candidates = [dataset_root / dataset_name]

    aliases = {
        "gsm8k": dataset_root / "gsm8k" / "main",
        "gsm8k/main": dataset_root / "gsm8k" / "main",
        "cnn": dataset_root / "cnn" / "3.0.0",
        "cnn_dailymail": dataset_root / "cnn" / "3.0.0",
        "cnn_daily_mail": dataset_root / "cnn" / "3.0.0",
        "cnn/dailymail": dataset_root / "cnn" / "3.0.0",
    }
    if dataset_name in aliases:
        candidates.insert(0, aliases[dataset_name])

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Dataset directory not found for '{dataset_name}'. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def _split_from_file_name(path: Path) -> str | None:
    name = path.name.lower()
    for split in ("train", "validation", "test"):
        if name.startswith(split) or f"/{split}" in str(path).lower():
            return split
    return None


def _parquet_data_files(dataset_dir: Path) -> dict[str, list[str]]:
    data_files: dict[str, list[str]] = {}
    for path in sorted(dataset_dir.glob("*.parquet")):
        split = _split_from_file_name(path)
        if split is not None:
            data_files.setdefault(split, []).append(str(path))
    return data_files


def _format_generation_examples(dataset_name: str, dataset: DatasetDict) -> DatasetDict:
    normalized_name = dataset_name.lower()

    def format_batch(batch: dict[str, list[Any]]) -> dict[str, list[str]]:
        columns = set(batch)
        if normalized_name.startswith("gsm8k") and {"question", "answer"} <= columns:
            return {
                "text": [
                    f"Question:\n{question}\n\nAnswer:\n{answer}"
                    for question, answer in zip(batch["question"], batch["answer"])
                ]
            }
        if (normalized_name.startswith("cnn") or normalized_name in {"cnn_dailymail", "cnn_daily_mail", "cnn/dailymail"}) and {
            "article",
            "highlights",
        } <= columns:
            return {
                "text": [
                    f"Article:\n{article}\n\nSummary:\n{highlights}"
                    for article, highlights in zip(batch["article"], batch["highlights"])
                ]
            }
        if {"instruction", "output"} <= columns:
            inputs = batch.get("input") or [""] * len(batch["instruction"])
            return {
                "text": [
                    f"Instruction:\n{instruction}\n\nInput:\n{input_text}\n\nResponse:\n{output}"
                    if str(input_text).strip()
                    else f"Instruction:\n{instruction}\n\nResponse:\n{output}"
                    for instruction, input_text, output in zip(
                        batch["instruction"], inputs, batch["output"]
                    )
                ]
            }
        if "text" in columns:
            return {"text": [str(item) for item in batch["text"]]}
        raise ValueError(
            f"Could not infer generation text formatting for dataset '{dataset_name}' "
            f"with columns {sorted(columns)}."
        )

    return DatasetDict(
        {
            split: dataset[split].map(
                format_batch,
                batched=True,
                remove_columns=dataset[split].column_names,
            )
            for split in dataset.keys()
        }
    )


def load_local_generation_dataset(data_config: GenerationDataConfig) -> DatasetDict:
    dataset_dir = resolve_local_dataset_dir(data_config)

    if (dataset_dir / "dataset_dict.json").exists():
        dataset = load_from_disk(str(dataset_dir))
    else:
        jsonl_train = dataset_dir / "cleaned_short_train_scrubbed.jsonl"
        jsonl_valid = dataset_dir / "cleaned_short_test_scrubbed.jsonl"
        csv_train = dataset_dir / "cleaned_short_train_scrubbed.csv"
        csv_valid = dataset_dir / "cleaned_short_test_scrubbed.csv"

        parquet_splits = _parquet_data_files(dataset_dir)

        if parquet_splits:
            dataset = load_dataset("parquet", data_files=parquet_splits)
        elif jsonl_train.exists() and jsonl_valid.exists():
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

    if data_config.train_split not in dataset:
        raise ValueError(f"Train split '{data_config.train_split}' not found. Available: {list(dataset.keys())}")
    if data_config.validation_split not in dataset:
        raise ValueError(
            f"Validation split '{data_config.validation_split}' not found. Available: {list(dataset.keys())}"
        )

    sample_columns = dataset[data_config.train_split].column_names
    text_column = data_config.text_column
    if text_column not in sample_columns:
        if data_config.dataset_name == "ptb" and "sentence" in sample_columns:
            text_column = "sentence"
        elif "text" not in sample_columns:
            return _format_generation_examples(data_config.dataset_name, dataset)
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


class SentencePieceTokenizerAdapter:
    def __init__(self, tokenizer_path: Path, model_max_length: int = 2048) -> None:
        if spm is None:
            raise ImportError(
                "sentencepiece is required to load original LLaMA tokenizer.model files. "
                "Install it in the active environment with: pip install sentencepiece"
            )
        self.tokenizer_path = tokenizer_path
        self.processor = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
        self.model_max_length = model_max_length
        self.pad_token_id = self.processor.eos_id()
        self.eos_token_id = self.processor.eos_id()
        self.bos_token_id = self.processor.bos_id()
        self.pad_token = "</s>"
        self.eos_token = "</s>"
        self.bos_token = "<s>"

    def __len__(self) -> int:
        return int(self.processor.get_piece_size())

    def __call__(self, texts, **_: Any) -> dict[str, list[list[int]]]:
        if isinstance(texts, str):
            texts = [texts]
        input_ids = [self.encode(text, add_special_tokens=True) for text in texts]
        return {
            "input_ids": input_ids,
            "attention_mask": [[1] * len(ids) for ids in input_ids],
        }

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        truncation: bool = False,
        **_: Any,
    ) -> list[int]:
        ids = list(self.processor.encode(str(text), out_type=int))
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        if truncation:
            ids = ids[: self.model_max_length]
        return ids

    def save_pretrained(self, save_directory: str | Path) -> None:
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        target = save_directory / "tokenizer.model"
        target.write_bytes(self.tokenizer_path.read_bytes())


def resolve_generation_tokenizer(tokenizer_name_or_path: str):
    tokenizer_path = Path(tokenizer_name_or_path).expanduser()
    candidates = [tokenizer_path, tokenizer_path / "checkpoints"]
    for candidate in candidates:
        if (candidate / "params.json").exists() and (candidate / "tokenizer.model").exists():
            model_max_length = 4096 if "llama2" in str(candidate).lower() or "llama-2" in str(candidate).lower() else 2048
            return SentencePieceTokenizerAdapter(candidate / "tokenizer.model", model_max_length=model_max_length)
    return AutoTokenizer.from_pretrained(tokenizer_name_or_path)


def prepare_generation_data(
    data_config: GenerationDataConfig,
    tokenizer_name_or_path: str,
    batch_size: int,
    eval_batch_size: int,
    num_workers: int,
) -> tuple[PreTrainedTokenizerBase, Dataset, Dataset, DataLoader, DataLoader]:
    tokenizer = resolve_generation_tokenizer(tokenizer_name_or_path)
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
