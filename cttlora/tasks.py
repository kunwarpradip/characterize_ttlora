from __future__ import annotations

from dataclasses import dataclass

from datasets import DatasetDict


@dataclass(frozen=True, slots=True)
class TaskSpec:
    name: str
    text_columns: tuple[str, ...]
    num_labels: int


TASK_SPECS: dict[str, TaskSpec] = {
    "boolq": TaskSpec(name="boolq", text_columns=("question", "passage"), num_labels=2),
    "cb": TaskSpec(name="cb", text_columns=("premise", "hypothesis"), num_labels=3),
    "cola": TaskSpec(name="cola", text_columns=("sentence",), num_labels=2),
    "mrpc": TaskSpec(name="mrpc", text_columns=("sentence1", "sentence2"), num_labels=2),
    "qnli": TaskSpec(name="qnli", text_columns=("question", "sentence"), num_labels=2),
    "qqp": TaskSpec(name="qqp", text_columns=("question1", "question2"), num_labels=2),
    "rte": TaskSpec(name="rte", text_columns=("sentence1", "sentence2"), num_labels=2),
    "sst2": TaskSpec(name="sst2", text_columns=("sentence",), num_labels=2),
}


def get_task_spec(
    dataset_name: str,
    text_columns_override: tuple[str, ...] | None = None,
    num_labels_override: int | None = None,
) -> TaskSpec:
    if text_columns_override is not None and num_labels_override is None:
        raise ValueError("num_labels must be provided when overriding text_columns.")

    if text_columns_override is not None and num_labels_override is not None:
        return TaskSpec(
            name=dataset_name,
            text_columns=text_columns_override,
            num_labels=num_labels_override,
        )

    if dataset_name not in TASK_SPECS:
        supported = ", ".join(sorted(TASK_SPECS))
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. "
            f"Supported datasets: {supported}. "
            "Use --text-columns and --num-labels to add a new task without editing code."
        )

    spec = TASK_SPECS[dataset_name]
    if num_labels_override is not None:
        return TaskSpec(name=spec.name, text_columns=spec.text_columns, num_labels=num_labels_override)
    return spec


def normalize_dataset(dataset_name: str, dataset: DatasetDict) -> DatasetDict:
    if dataset_name == "boolq" and "answer" in dataset["train"].column_names:
        dataset = dataset.rename_column("answer", "label")

        def cast_bool_label(example: dict) -> dict:
            example["label"] = int(example["label"])
            return example

        dataset = dataset.map(cast_bool_label)

    return dataset
