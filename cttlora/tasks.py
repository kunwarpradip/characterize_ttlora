from __future__ import annotations

from dataclasses import dataclass

from datasets import DatasetDict


@dataclass(frozen=True, slots=True)
class TaskSpec:
    name: str
    text_columns: tuple[str, ...]
    num_labels: int


TASK_SPECS: dict[str, TaskSpec] = {
    "ax": TaskSpec(name="ax", text_columns=("premise", "hypothesis"), num_labels=3),
    "boolq": TaskSpec(name="boolq", text_columns=("question", "passage"), num_labels=2),
    "cb": TaskSpec(name="cb", text_columns=("premise", "hypothesis"), num_labels=3),
    "cola": TaskSpec(name="cola", text_columns=("sentence",), num_labels=2),
    "cosmosqa": TaskSpec(name="cosmosqa", text_columns=("context_question", "answers"), num_labels=4),
    "csqa": TaskSpec(name="csqa", text_columns=("question_with_concept", "choices"), num_labels=5),
    "hellaswag": TaskSpec(name="hellaswag", text_columns=("text", "ending"), num_labels=4),
    "imdb": TaskSpec(name="imdb", text_columns=("text",), num_labels=2),
    "mnli": TaskSpec(name="mnli", text_columns=("premise", "hypothesis"), num_labels=3),
    "mrpc": TaskSpec(name="mrpc", text_columns=("sentence1", "sentence2"), num_labels=2),
    "qnli": TaskSpec(name="qnli", text_columns=("question", "sentence"), num_labels=2),
    "qqp": TaskSpec(name="qqp", text_columns=("question1", "question2"), num_labels=2),
    "rte": TaskSpec(name="rte", text_columns=("sentence1", "sentence2"), num_labels=2),
    "scitail": TaskSpec(name="scitail", text_columns=("sentence1", "sentence2"), num_labels=2),
    "sick": TaskSpec(name="sick", text_columns=("sentence_A", "sentence_B"), num_labels=3),
    "socialiqa": TaskSpec(name="socialiqa", text_columns=("context_question", "answers"), num_labels=3),
    "sst2": TaskSpec(name="sst2", text_columns=("sentence",), num_labels=2),
    "winogrande": TaskSpec(name="winogrande", text_columns=("sentence", "options"), num_labels=2),
    "winogrande_l": TaskSpec(name="winogrande_l", text_columns=("sentence", "options"), num_labels=2),
    "wnli": TaskSpec(name="wnli", text_columns=("sentence1", "sentence2"), num_labels=2),
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
    def parse_int_label(value) -> int:
        text = str(value).strip()
        if text == "":
            return -1
        return int(float(text))

    def remap_labels(example: dict) -> dict:
        label = str(example["label"]).strip()
        if label == "":
            example["label"] = -1
        elif label in {"neutral", "NEUTRAL"}:
            example["label"] = 0
        elif label in {"entailment", "ENTAILMENT"}:
            example["label"] = 1
        elif label in {"contradiction", "CONTRADICTION"}:
            example["label"] = 2
        elif label == "A":
            example["label"] = 0
        elif label == "B":
            example["label"] = 1
        elif label == "C":
            example["label"] = 2
        elif label == "D":
            example["label"] = 3
        elif label == "E":
            example["label"] = 4
        else:
            numeric_label = int(float(label))
            example["label"] = numeric_label - 1 if numeric_label > 0 else numeric_label
        return example

    if dataset_name == "boolq" and "answer" in dataset["train"].column_names:
        dataset = dataset.rename_column("answer", "label")

        def cast_bool_label(example: dict) -> dict:
            example["label"] = int(example["label"])
            return example

        dataset = dataset.map(cast_bool_label)

    if dataset_name in {"cb"}:
        columns_to_remove = ["idx", "template_name", "template", "rendered_input", "rendered_output"]
        available_columns = [column for column in columns_to_remove if column in dataset["train"].column_names]
        if available_columns:
            dataset = dataset.remove_columns(available_columns)

    if dataset_name in {"sick"}:
        columns_to_remove = [
            "pair_ID",
            "relatedness_score",
            "entailment_AB",
            "entailment_BA",
            "sentence_A_original",
            "sentence_B_original",
            "sentence_A_dataset",
            "sentence_B_dataset",
            "SemEval_set",
        ]
        available_columns = [column for column in columns_to_remove if column in dataset["train"].column_names]
        if available_columns:
            dataset = dataset.remove_columns(available_columns)
        if "entailment_label" in dataset["train"].column_names:
            dataset = dataset.rename_column("entailment_label", "label")
        dataset = dataset.map(remap_labels)

    if dataset_name in {"scitail"}:
        columns_to_remove = ["annotator_labels"]
        available_columns = [column for column in columns_to_remove if column in dataset["train"].column_names]
        if available_columns:
            dataset = dataset.remove_columns(available_columns)
        if "gold_label" in dataset["train"].column_names:
            dataset = dataset.rename_column("gold_label", "label")
        dataset = dataset.map(remap_labels)

        def flatten_scitail(example: dict) -> dict:
            sentence1 = f"{example['sentence1_binary_parse']} {example['sentence1_parse']}  {example['sentence1']}"
            sentence2 = f"{example['sentence2_parse']} {example['sentence2']}"
            return {"sentence1": sentence1, "sentence2": sentence2, "label": parse_int_label(example["label"])}

        dataset = DatasetDict({split: dataset[split].map(flatten_scitail) for split in dataset.keys()})
        for split in dataset.keys():
            extra = [
                column
                for column in dataset[split].column_names
                if column not in {"sentence1", "sentence2", "label"}
            ]
            if extra:
                dataset[split] = dataset[split].remove_columns(extra)

    if dataset_name in {"hellaswag"}:
        removable_columns = {"ind", "source_id", "split", "split_type"}

        def flatten_hellaswag(example: dict) -> dict:
            endings = example.get("endings", [])
            if isinstance(endings, dict) and "text" in endings:
                endings = endings["text"]
            context = f"{example['activity_label']}: {example['ctx_a']} {example['ctx_b']} {example['ctx']}"
            ending = " ".join(str(choice) for choice in endings)
            return {
                "text": " ".join(context.split()),
                "ending": " ".join(ending.split()),
                "label": parse_int_label(example["label"]),
            }

        normalized_splits = {}
        for split_name, split_dataset in dataset.items():
            columns_to_remove = [column for column in split_dataset.column_names if column in removable_columns]
            if columns_to_remove:
                split_dataset = split_dataset.remove_columns(columns_to_remove)
            split_dataset = split_dataset.map(flatten_hellaswag)
            extra_columns = [
                column for column in split_dataset.column_names if column not in {"text", "ending", "label"}
            ]
            if extra_columns:
                split_dataset = split_dataset.remove_columns(extra_columns)
            normalized_splits[split_name] = split_dataset
        dataset = DatasetDict(normalized_splits)

    if dataset_name in {"socialiqa"}:
        dataset = dataset.map(remap_labels)

        def flatten_socialiqa(example: dict) -> dict:
            context_question = f"{example['context']} {example['question']}"
            answers = (
                f"option0: {example['answerA']}, "
                f"option1: {example['answerB']}, "
                f"option2: {example['answerC']}"
            )
            return {
                "context_question": context_question,
                "answers": answers,
                "label": parse_int_label(example["label"]),
            }

        dataset = DatasetDict({split: dataset[split].map(flatten_socialiqa) for split in dataset.keys()})
        for split in dataset.keys():
            extra = [
                column
                for column in dataset[split].column_names
                if column not in {"context_question", "answers", "label"}
            ]
            if extra:
                dataset[split] = dataset[split].remove_columns(extra)

    if dataset_name in {"cosmosqa"}:
        def flatten_cosmosqa(example: dict) -> dict:
            context_question = f"{example['context']} {example['question']}"
            answers = (
                f"option0: {example['answer0']}, "
                f"option1: {example['answer1']}, "
                f"option2: {example['answer2']}, "
                f"option3: {example['answer3']}"
            )
            return {
                "context_question": context_question,
                "answers": answers,
                "label": parse_int_label(example["label"]),
            }

        dataset = DatasetDict({split: dataset[split].map(flatten_cosmosqa) for split in dataset.keys()})
        for split in dataset.keys():
            extra = [
                column
                for column in dataset[split].column_names
                if column not in {"context_question", "answers", "label"}
            ]
            if extra:
                dataset[split] = dataset[split].remove_columns(extra)

    if dataset_name in {"csqa"}:
        if "answerKey" in dataset["train"].column_names:
            dataset = dataset.rename_column("answerKey", "label")
        dataset = dataset.map(remap_labels)

        def flatten_csqa(example: dict) -> dict:
            choices = example["choices"]
            if isinstance(choices, dict) and "text" in choices:
                choices_text = choices["text"]
            else:
                choices_text = choices
            question = f"{example['question']}"
            question_concept = f"{example['question_concept']}"
            choices_joined = ", ".join(str(choice) for choice in choices_text)
            return {
                "question_with_concept": f"{question} {question_concept}",
                "choices": choices_joined,
                "label": parse_int_label(example["label"]),
            }

        dataset = DatasetDict({split: dataset[split].map(flatten_csqa) for split in dataset.keys()})
        for split in dataset.keys():
            extra = [
                column
                for column in dataset[split].column_names
                if column not in {"question_with_concept", "choices", "label"}
            ]
            if extra:
                dataset[split] = dataset[split].remove_columns(extra)

    if dataset_name in {"winogrande", "winogrande_l"}:
        if "answer" in dataset["train"].column_names:
            dataset = dataset.rename_column("answer", "label")
        dataset = dataset.map(remap_labels)

        def flatten_winogrande(example: dict) -> dict:
            return {
                "sentence": example["sentence"],
                "options": f"option0: {example['option1']}, option1: {example['option2']}",
                "label": int(example["label"]),
            }

        dataset = DatasetDict({split: dataset[split].map(flatten_winogrande) for split in dataset.keys()})
        for split in dataset.keys():
            extra = [
                column
                for column in dataset[split].column_names
                if column not in {"sentence", "options", "label"}
            ]
            if extra:
                dataset[split] = dataset[split].remove_columns(extra)

    return dataset
