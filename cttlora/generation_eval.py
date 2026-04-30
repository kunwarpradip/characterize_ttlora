from __future__ import annotations

import csv
import math
import re
from collections import Counter
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


@dataclass(slots=True)
class SummarizationEvaluationRecord:
    index: int
    prompt: str
    reference_summary: str
    generated_summary: str
    rouge1: float
    rouge2: float
    rougeL: float
    meteor: float


@dataclass(slots=True)
class SummarizationEvaluationSummary:
    evaluated_examples: int
    rouge1: float
    rouge2: float
    rougeL: float
    meteor: float
    predictions_path: str | None


_NUMBER_PATTERN = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")


def is_gsm8k_dataset(dataset_name: str) -> bool:
    normalized = dataset_name.lower().replace("\\", "/")
    return normalized == "gsm8k" or normalized.startswith("gsm8k/")


def is_cnn_dataset(dataset_name: str) -> bool:
    normalized = dataset_name.lower().replace("\\", "/")
    return normalized.startswith("cnn") or normalized in {"cnn_dailymail", "cnn_daily_mail", "cnn/dailymail"}


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
        return format(number.to_integral_value(), "f")
    return format(number.normalize(), "f").rstrip("0").rstrip(".")


def _gsm8k_prompt(question: Any) -> str:
    return f"Question:\n{question}\n\nAnswer:\n"


def _cnn_prompt(article: Any) -> str:
    return f"Article:\n{article}\n\nSummary:\n"


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
        input_ids.append([int(pad_token_id)] * pad_count + ids)
        attention_mask.append([0] * pad_count + [1] * len(ids))

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
    }


def _decode_generated(tokenizer, generated_sequences: torch.Tensor, prompt_width: int) -> list[str]:
    decoded: list[str] = []
    for sequence in generated_sequences:
        continuation = sequence[prompt_width:].detach().cpu().tolist()
        decoded.append(tokenizer.decode(continuation, skip_special_tokens=True).strip())
    return decoded


def _write_records_csv(path: Path, records: list[object], fallback_fieldnames: list[str]) -> None:
    fieldnames = list(asdict(records[0]).keys()) if records else fallback_fieldnames
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


def load_cnn_eval_rows(data_config: GenerationDataConfig) -> Dataset:
    dataset = load_local_generation_dataset_raw(data_config)
    split = dataset[data_config.validation_split]
    if not {"article", "highlights"} <= set(split.column_names):
        raise ValueError(
            "CNN/DailyMail evaluation requires raw 'article' and 'highlights' columns. "
            f"Found columns: {split.column_names}"
        )
    return split


def _evaluation_starts(total_rows: int, batch_size: int, desc: str):
    starts = range(0, total_rows, max(1, batch_size))
    if tqdm is not None:
        starts = tqdm(
            starts,
            total=(total_rows + max(1, batch_size) - 1) // max(1, batch_size),
            desc=desc,
            unit="batch",
            leave=False,
        )
    return starts


def _prepare_generation_context(tokenizer, data_config: GenerationDataConfig, max_new_tokens: int):
    model_max_length = int(getattr(tokenizer, "model_max_length", data_config.max_length))
    max_prompt_length = max(1, min(data_config.max_length, model_max_length) - max_new_tokens)
    pad_token_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None)
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"
    return max_prompt_length, pad_token_id


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

    max_prompt_length, pad_token_id = _prepare_generation_context(tokenizer, data_config, max_new_tokens)
    records: list[GSM8KEvaluationRecord] = []
    model.eval()
    if logger is not None:
        logger.info(
            "GSM8K exact-match generation loop: examples=%d batch_size=%d max_new_tokens=%d",
            len(rows),
            max(1, batch_size),
            max_new_tokens,
        )
    with torch.no_grad():
        for start in _evaluation_starts(len(rows), batch_size, "GSM8K exact-match"):
            batch_end = min(start + max(1, batch_size), len(rows))
            batch_rows = rows.select(range(start, batch_end))
            prompts = [_gsm8k_prompt(question) for question in batch_rows["question"]]
            encoded = _encode_prompts(tokenizer, prompts, max_prompt_length, device)
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
                eos_token_id=getattr(tokenizer, "eos_token_id", None),
            )
            generated_texts = _decode_generated(tokenizer, generated, encoded["input_ids"].shape[1])

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
        _write_records_csv(
            output_path,
            records,
            [
                "index",
                "prompt",
                "gold_answer",
                "gold_number",
                "generated_text",
                "predicted_number",
                "exact_match",
            ],
        )

    return GSM8KEvaluationSummary(
        evaluated_examples=len(records),
        exact_matches=exact_matches,
        exact_match_accuracy=exact_matches / max(1, len(records)),
        predictions_path=str(output_path) if output_path is not None else None,
    )


def _normalize_text(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(str(text).lower())


def _ngram_counts(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _rouge_n_score(reference_tokens: list[str], candidate_tokens: list[str], n: int) -> float:
    reference_counts = _ngram_counts(reference_tokens, n)
    candidate_counts = _ngram_counts(candidate_tokens, n)
    if not reference_counts or not candidate_counts:
        return 0.0
    overlap = sum((reference_counts & candidate_counts).values())
    precision = overlap / max(1, sum(candidate_counts.values()))
    recall = overlap / max(1, sum(reference_counts.values()))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _lcs_length(reference_tokens: list[str], candidate_tokens: list[str]) -> int:
    if not reference_tokens or not candidate_tokens:
        return 0
    previous = [0] * (len(candidate_tokens) + 1)
    for ref_token in reference_tokens:
        current = [0]
        for idx, cand_token in enumerate(candidate_tokens, start=1):
            if ref_token == cand_token:
                current.append(previous[idx - 1] + 1)
            else:
                current.append(max(previous[idx], current[-1]))
        previous = current
    return previous[-1]


def _rouge_l_score(reference_tokens: list[str], candidate_tokens: list[str]) -> float:
    lcs = _lcs_length(reference_tokens, candidate_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / max(1, len(candidate_tokens))
    recall = lcs / max(1, len(reference_tokens))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _meteor_score(reference_tokens: list[str], candidate_tokens: list[str]) -> float:
    if not reference_tokens or not candidate_tokens:
        return 0.0

    reference_positions: dict[str, list[int]] = {}
    for idx, token in enumerate(reference_tokens):
        reference_positions.setdefault(token, []).append(idx)

    matches: list[tuple[int, int]] = []
    used_reference_positions: set[int] = set()
    for cand_idx, token in enumerate(candidate_tokens):
        for ref_idx in reference_positions.get(token, []):
            if ref_idx not in used_reference_positions:
                used_reference_positions.add(ref_idx)
                matches.append((cand_idx, ref_idx))
                break

    match_count = len(matches)
    if match_count == 0:
        return 0.0

    precision = match_count / len(candidate_tokens)
    recall = match_count / len(reference_tokens)
    f_mean = (10 * precision * recall) / max(1e-12, recall + 9 * precision)

    matches.sort()
    chunks = 1
    for (cand_prev, ref_prev), (cand_curr, ref_curr) in zip(matches, matches[1:]):
        if cand_curr != cand_prev + 1 or ref_curr != ref_prev + 1:
            chunks += 1
    penalty = 0.5 * ((chunks / match_count) ** 3)
    return f_mean * (1 - penalty)


def evaluate_cnn_summarization(
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
) -> SummarizationEvaluationSummary:
    rows = load_cnn_eval_rows(data_config)
    if max_eval_samples is not None:
        rows = rows.select(range(min(max_eval_samples, len(rows))))

    max_prompt_length, pad_token_id = _prepare_generation_context(tokenizer, data_config, max_new_tokens)
    records: list[SummarizationEvaluationRecord] = []
    model.eval()
    if logger is not None:
        logger.info(
            "CNN/DailyMail summarization evaluation: examples=%d batch_size=%d max_new_tokens=%d",
            len(rows),
            max(1, batch_size),
            max_new_tokens,
        )

    with torch.no_grad():
        for start in _evaluation_starts(len(rows), batch_size, "CNN summarization"):
            batch_end = min(start + max(1, batch_size), len(rows))
            batch_rows = rows.select(range(start, batch_end))
            prompts = [_cnn_prompt(article) for article in batch_rows["article"]]
            encoded = _encode_prompts(tokenizer, prompts, max_prompt_length, device)
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
                eos_token_id=getattr(tokenizer, "eos_token_id", None),
            )
            generated_summaries = _decode_generated(tokenizer, generated, encoded["input_ids"].shape[1])

            for offset, generated_summary in enumerate(generated_summaries):
                row_index = start + offset
                reference_summary = str(batch_rows["highlights"][offset]).strip()
                reference_tokens = _normalize_text(reference_summary)
                generated_tokens = _normalize_text(generated_summary)
                records.append(
                    SummarizationEvaluationRecord(
                        index=row_index,
                        prompt=prompts[offset],
                        reference_summary=reference_summary,
                        generated_summary=generated_summary,
                        rouge1=_rouge_n_score(reference_tokens, generated_tokens, 1),
                        rouge2=_rouge_n_score(reference_tokens, generated_tokens, 2),
                        rougeL=_rouge_l_score(reference_tokens, generated_tokens),
                        meteor=_meteor_score(reference_tokens, generated_tokens),
                    )
                )
            if logger is not None and (
                batch_end == len(rows) or batch_end % max(1, log_every) < max(1, batch_size)
            ):
                logger.info("CNN/DailyMail summarization progress: %d/%d generated", batch_end, len(rows))

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_records_csv(
            output_path,
            records,
            [
                "index",
                "prompt",
                "reference_summary",
                "generated_summary",
                "rouge1",
                "rouge2",
                "rougeL",
                "meteor",
            ],
        )

    rouge1 = sum(record.rouge1 for record in records) / max(1, len(records))
    rouge2 = sum(record.rouge2 for record in records) / max(1, len(records))
    rougeL = sum(record.rougeL for record in records) / max(1, len(records))
    meteor = sum(record.meteor for record in records) / max(1, len(records))
    return SummarizationEvaluationSummary(
        evaluated_examples=len(records),
        rouge1=rouge1,
        rouge2=rouge2,
        rougeL=rougeL,
        meteor=meteor,
        predictions_path=str(output_path) if output_path is not None else None,
    )
