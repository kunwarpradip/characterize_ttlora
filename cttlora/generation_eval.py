from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset

from .generation_config import GenerationDataConfig
from .generation_data import load_local_generation_dataset_raw

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


@dataclass(slots=True)
class GSM8KEvaluationRecord:
    index: int
    prompt: str
    gold_answer: str
    gold_number: str | None
    generated_text: str
    predicted_number: str | None
    exact_match: bool


@dataclass(slots=True)
class GSM8KEvaluationSummary:
    evaluated_examples: int
    exact_matches: int
    exact_match_accuracy: float
    predictions_path: str | None


_NUMBER_PATTERN = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def is_gsm8k_dataset(dataset_name: str) -> bool:
    normalized = dataset_name.lower().replace("\\", "/")
    return normalized == "gsm8k" or normalized.startswith("gsm8k/")


def extract_gsm8k_number(text: Any) -> str | None:
    if text is None:
        return None
    value = str(text)
    if "####" in value:
        value = value.rsplit("####", maxsplit=1)[-1]
    matches = _NUMBER_PATTERN.findall(value)
    if not matches:
        return None
    return normalize_number(matches[-1])


def normalize_number(value: str) -> str | None:
    cleaned = value.replace(",", "").strip()
    if not cleaned:
        return None
    try:
        number = Decimal(cleaned)
    except InvalidOperation:
        return None
    if number == number.to_integral_value():
        return str(number.quantize(Decimal("1")))
    return format(number.normalize(), "f").rstrip("0").rstrip(".")


def _gsm8k_prompt(question: Any) -> str:
    return f"Question:\n{question}\n\nAnswer:\n"


def _encode_prompts(tokenizer, prompts: list[str], max_prompt_length: int, device: torch.device) -> dict[str, torch.Tensor]:
    encoded_prompts: list[list[int]] = []
    for prompt in prompts:
        ids = tokenizer.encode(str(prompt), add_special_tokens=True, truncation=True)
        encoded_prompts.append([int(item) for item in ids[:max_prompt_length]])

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id or eos_token_id for generation evaluation.")

    max_length = max(len(ids) for ids in encoded_prompts)
    input_ids = []
    attention_mask = []
    for ids in encoded_prompts:
        pad_count = max_length - len(ids)
        input_ids.append(ids + [int(pad_token_id)] * pad_count)
        attention_mask.append([1] * len(ids) + [0] * pad_count)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
    }


def _decode_generated(tokenizer, generated_sequences: torch.Tensor, prompt_lengths: list[int]) -> list[str]:
    decoded: list[str] = []
    for sequence, prompt_length in zip(generated_sequences, prompt_lengths):
        continuation = sequence[prompt_length:].detach().cpu().tolist()
        decoded.append(tokenizer.decode(continuation, skip_special_tokens=True).strip())
    return decoded


def _write_gsm8k_predictions(path: Path, records: list[GSM8KEvaluationRecord]) -> None:
    fieldnames = list(asdict(records[0]).keys()) if records else [
        "index",
        "prompt",
        "gold_answer",
        "gold_number",
        "generated_text",
        "predicted_number",
        "exact_match",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def load_gsm8k_eval_rows(data_config: GenerationDataConfig) -> Dataset:
    dataset = load_local_generation_dataset_raw(data_config)
    split = dataset[data_config.validation_split]
    if not {"question", "answer"} <= set(split.column_names):
        raise ValueError(
            "GSM8K exact-match evaluation requires raw 'question' and 'answer' columns. "
            f"Found columns: {split.column_names}"
        )
    return split


def evaluate_gsm8k_exact_match(
    *,
    model,
    tokenizer,
    data_config: GenerationDataConfig,
    device: torch.device,
    output_path: Path | None,
    max_eval_samples: int | None = None,
    batch_size: int = 1,
    max_new_tokens: int = 256,
    logger=None,
    log_every: int = 25,
) -> GSM8KEvaluationSummary:
    rows = load_gsm8k_eval_rows(data_config)
    if max_eval_samples is not None:
        rows = rows.select(range(min(max_eval_samples, len(rows))))

    model_max_length = int(getattr(tokenizer, "model_max_length", data_config.max_length))
    max_prompt_length = max(1, min(data_config.max_length, model_max_length) - max_new_tokens)
    pad_token_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None)

    records: list[GSM8KEvaluationRecord] = []
    model.eval()
    if logger is not None:
        logger.info(
            "GSM8K exact-match generation loop: examples=%d batch_size=%d max_new_tokens=%d",
            len(rows),
            max(1, batch_size),
            max_new_tokens,
        )
    starts = range(0, len(rows), max(1, batch_size))
    if tqdm is not None:
        starts = tqdm(
            starts,
            total=(len(rows) + max(1, batch_size) - 1) // max(1, batch_size),
            desc="GSM8K exact-match",
            unit="batch",
            leave=False,
        )
    with torch.no_grad():
        for start in starts:
            batch_end = min(start + max(1, batch_size), len(rows))
            batch_rows = rows.select(range(start, batch_end))
            prompts = [_gsm8k_prompt(question) for question in batch_rows["question"]]
            encoded = _encode_prompts(tokenizer, prompts, max_prompt_length, device)
            prompt_lengths = encoded["attention_mask"].sum(dim=1).detach().cpu().tolist()
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
                eos_token_id=getattr(tokenizer, "eos_token_id", None),
            )
            generated_texts = _decode_generated(tokenizer, generated, [int(item) for item in prompt_lengths])

            for offset, generated_text in enumerate(generated_texts):
                row_index = start + offset
                gold_answer = str(batch_rows["answer"][offset])
                gold_number = extract_gsm8k_number(gold_answer)
                predicted_number = extract_gsm8k_number(generated_text)
                records.append(
                    GSM8KEvaluationRecord(
                        index=row_index,
                        prompt=prompts[offset],
                        gold_answer=gold_answer,
                        gold_number=gold_number,
                        generated_text=generated_text,
                        predicted_number=predicted_number,
                        exact_match=gold_number is not None and gold_number == predicted_number,
                    )
                )
            if logger is not None and (
                batch_end == len(rows) or batch_end % max(1, log_every) < max(1, batch_size)
            ):
                running_matches = sum(1 for record in records if record.exact_match)
                logger.info(
                    "GSM8K exact-match progress: %d/%d generated current_accuracy=%.4f",
                    batch_end,
                    len(rows),
                    running_matches / max(1, len(records)),
                )

    exact_matches = sum(1 for record in records if record.exact_match)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_gsm8k_predictions(output_path, records)

    return GSM8KEvaluationSummary(
        evaluated_examples=len(records),
        exact_matches=exact_matches,
        exact_match_accuracy=exact_matches / max(1, len(records)),
        predictions_path=str(output_path) if output_path is not None else None,
    )
