"""Canonical 15-OOD benchmark over the production rejection runtime."""

from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


NI_DATASETS = ("task149", "task476", "task933", "task1622", "task1670")
BBH_DATASETS = (
    "causal_judgement",
    "dyck_languages",
    "logical_deduction_five_objects",
    "multistep_arithmetic_two",
    "tracking_shuffled_objects_five_objects",
)
MMLU_PRO_DATASETS = (
    "biology",
    "chemistry",
    "computer_science",
    "economics",
    "math",
)
GENERATION_DATASETS = {"task933", "task1622", "task1670"}
GROUPS = ("ni", "bbh", "mmlu_pro")
EXPECTED_GROUP_COUNTS = {"ni": 1722, "bbh": 1187, "mmlu_pro": 1250}
BENCHMARK_GENERATION = {
    "ni": {
        "max_input_tokens": 8192,
        "max_new_tokens": 1024,
        "stop_strings": [],
        "reject_prompt_truncation": True,
    },
    "bbh": {
        "max_input_tokens": 8192,
        "max_new_tokens": 1024,
        "stop_strings": ["\nQ:", "\nQuestion:", "\n\n\n"],
        "reject_prompt_truncation": True,
    },
    "mmlu_pro": {
        "max_input_tokens": 8192,
        "max_new_tokens": 1024,
        "stop_strings": ["\nQ:", "\nQuestion:", "\n\n\n"],
        "reject_prompt_truncation": True,
    },
}


class BenchmarkError(RuntimeError):
    """The external benchmark data or generated records violate the contract."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"cannot read benchmark file {path}: {exc}") from exc


def _dataset_from_instance_id(instance_id: str) -> str:
    parts = str(instance_id).split("::")
    if len(parts) < 2 or not parts[1]:
        raise BenchmarkError(f"cannot derive dataset from instance_id {instance_id!r}")
    return parts[1]


def load_benchmark_records(root: str | Path, group: str) -> list[dict[str, Any]]:
    """Load answer-free full prompts using the MoEA 15-OOD directory layout."""

    if group not in GROUPS:
        raise BenchmarkError(f"unknown benchmark group: {group}")
    root = Path(root)
    records: list[dict[str, Any]] = []
    if group == "ni":
        prompts = _read_json(root / "prompts/ood_ni_tasks.json")
        for dataset in NI_DATASETS:
            path = root / (
                "dataset/natural_instructions/data/selected_10_tasks/test_data/"
                f"{dataset}_test.json"
            )
            payload = _read_json(path)
            for index, item in enumerate(payload.get("instances", [])):
                records.append(
                    {
                        "order": len(records),
                        "dataset": dataset,
                        "benchmark": "natural_instructions",
                        "task_type": (
                            "generation" if dataset in GENERATION_DATASETS
                            else "classification"
                        ),
                        "task_description": (
                            payload.get("definition")
                            or prompts[dataset]["description"]
                        ),
                        "sample_id": index,
                        "instance_id": str(item["instance_id"]),
                        "task_input": str(item.get("input", item["full_prompt"])),
                        "prompt": str(item["full_prompt"]),
                        "target": str(item["output"]),
                    }
                )
    else:
        filename = "bbh_test.json" if group == "bbh" else "mmlu_pro_test.json"
        expected = set(BBH_DATASETS if group == "bbh" else MMLU_PRO_DATASETS)
        payload = _read_json(root / "ood/data" / filename)
        for index, item in enumerate(payload.get("instances", [])):
            dataset = _dataset_from_instance_id(item["instance_id"])
            if dataset not in expected:
                continue
            prompt = str(item["full_prompt"])
            records.append(
                {
                    "order": len(records),
                    "dataset": dataset,
                    "benchmark": group,
                    "task_type": "classification",
                    "task_description": prompt.split("\n", 1)[0],
                    "sample_id": index,
                    "instance_id": str(item["instance_id"]),
                    "task_input": prompt,
                    "prompt": prompt,
                    "target": str(item["output"]),
                }
            )
        found = {record["dataset"] for record in records}
        if found != expected:
            raise BenchmarkError(
                f"{group} dataset mismatch: missing={sorted(expected - found)}"
            )
    identifiers = [record["instance_id"] for record in records]
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise BenchmarkError(f"{group} benchmark is empty or has duplicate IDs")
    return records


def _answer_free(record: Mapping[str, Any]) -> None:
    prompt = str(record["prompt"]).rstrip()
    target = str(record["target"]).strip()
    if target and prompt.endswith(target):
        raise BenchmarkError(
            f"benchmark prompt contains its target: {record['instance_id']}"
        )


def run_rejection_benchmark(
    engine,
    *,
    benchmark_root: str | Path,
    output_dir: str | Path,
    groups: Iterable[str] = GROUPS,
    batch_size: int = 1,
    smoke: bool = False,
) -> dict[str, Any]:
    """Generate and score benchmark records through ``ensure_rejection``."""

    if batch_size <= 0:
        raise BenchmarkError("batch_size must be positive")
    selected_groups = tuple(groups)
    if not selected_groups or any(group not in GROUPS for group in selected_groups):
        raise BenchmarkError(f"groups must be selected from {GROUPS}")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    all_results: list[dict[str, Any]] = []
    records_by_group = {
        group: load_benchmark_records(benchmark_root, group)
        for group in selected_groups
    }
    if not smoke:
        for group, records in records_by_group.items():
            expected = EXPECTED_GROUP_COUNTS[group]
            if len(records) != expected:
                raise BenchmarkError(
                    f"{group} requires {expected} records; found {len(records)}"
                )
    identity = engine.ensure_rejection()
    for group in selected_groups:
        records = records_by_group[group]
        if smoke:
            records = records[:1]
        for record in records:
            _answer_free(record)
        pending = list(records)
        if hasattr(engine, "prompt_lengths"):
            lengths = engine.prompt_lengths([record["prompt"] for record in pending])
            if len(lengths) != len(pending):
                raise BenchmarkError("prompt length count differs from record count")
            for record, length in zip(pending, lengths):
                record["prompt_tokens_unpadded"] = int(length)
            pending.sort(key=lambda record: record["prompt_tokens_unpadded"])
        group_results: list[dict[str, Any]] = []
        for start in range(0, len(pending), batch_size):
            chunk = pending[start : start + batch_size]
            outputs = engine.generate(
                [record["prompt"] for record in chunk],
                generation_overrides=BENCHMARK_GENERATION[group],
            )
            if len(outputs) != len(chunk):
                raise BenchmarkError("generation output count differs from prompt count")
            for record, prediction in zip(chunk, outputs):
                group_results.append({**record, "prediction": str(prediction)})
        group_results.sort(key=lambda record: int(record["order"]))
        all_results.extend(group_results)
        _write_json(
            destination / f"{group}_results.json",
            {
                "status": "complete",
                "rejection": identity,
                "group": group,
                "n": len(group_results),
                "results": group_results,
            },
        )
    metrics = evaluate_benchmark(all_results)
    report = {
        "status": "complete",
        "rejection": identity,
        "groups": list(selected_groups),
        "smoke": smoke,
        "n": len(all_results),
        "metrics": metrics,
    }
    _write_json(destination / "metrics.json", report)
    return report


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).strip()


def _truncate(text: str) -> str:
    for stop in ("\nQ:", "\nQuestion:", "\n\nQ:"):
        position = text.find(stop)
        if position != -1:
            text = text[:position]
    return text


def _extract_bbh(text: str) -> str | None:
    match = re.search(r"(?i)answer is\s*(.+?)\s*\.?\s*$", text.strip(), re.M)
    return _normalize(match.group(1).rstrip(".")) if match else None


def _extract_mmlu(text: str) -> str | None:
    for pattern in (r"answer is \(?([A-J])\)?", r"[aA]nswer:\s*\(?([A-J])\)?"):
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    matches = re.findall(r"\b([A-J])\b", text)
    return matches[-1] if matches else None


def _ni_classification_correct(record: Mapping[str, Any]) -> bool:
    labels = {
        "task149": ("valid", "invalid"),
        "task476": ("pos", "neg"),
    }[str(record["dataset"])]
    prediction = _preprocess_ni_prediction(str(record["prediction"]))
    target = _normalize(str(record["target"])).lower()
    found = {
        label for label in labels
        if re.search(r"\b" + re.escape(label) + r"\b", prediction)
    }
    return found == {target}


def _preprocess_ni_prediction(text: str) -> str:
    normalized = _normalize(text).lower()
    normalized = re.sub(
        r"(?i)\b(?:output|answer)\s*:?\s*", "", normalized
    ).strip()
    words = normalized.split()
    window_size = 7
    if len(words) >= window_size * 2:
        last_window = words[-window_size:]
        for index in range(len(words) - window_size * 2, -1, -1):
            if words[index : index + window_size] == last_window:
                normalized = " ".join(words[: index + window_size])
                sentences = [
                    sentence.strip()
                    for sentence in re.split(r"[.!?]", normalized)
                    if sentence.strip()
                ]
                if len(sentences) > 3:
                    normalized = ". ".join(sentences[:3]) + "."
                break
    return normalized


def _classification_correct(record: Mapping[str, Any]) -> tuple[bool, bool]:
    benchmark = str(record["benchmark"])
    prediction = _truncate(str(record["prediction"]))
    if benchmark == "natural_instructions":
        return _ni_classification_correct(record), False
    if benchmark == "bbh":
        extracted = _extract_bbh(prediction)
        gold = _normalize(str(record["target"]))
    else:
        extracted = _extract_mmlu(prediction)
        gold = str(record["target"]).strip()
    return (
        extracted is not None and extracted.strip().lower() == gold.lower(),
        extracted is None,
    )


def _generation_scores(records: list[Mapping[str, Any]]) -> tuple[float, float]:
    try:
        from nltk.tokenize import word_tokenize
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
        from rouge_score import rouge_scorer
    except ImportError as exc:
        raise BenchmarkError(
            "generation scoring requires nltk and rouge-score"
        ) from exc
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    smoothing = SmoothingFunction().method1
    rouge: list[float] = []
    bleu: list[float] = []
    try:
        for record in records:
            prediction = _preprocess_ni_prediction(str(record["prediction"]))
            target = _normalize(str(record["target"])).lower()
            rouge.append(scorer.score(target, prediction)["rougeL"].fmeasure)
            bleu.append(
                sentence_bleu(
                    [word_tokenize(target)],
                    word_tokenize(prediction),
                    smoothing_function=smoothing,
                )
            )
    except LookupError as exc:
        raise BenchmarkError(
            "NLTK tokenizer data is missing; run: python -m nltk.downloader punkt punkt_tab"
        ) from exc
    return statistics.fmean(rouge), statistics.fmean(bleu)


def evaluate_benchmark(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Local deterministic metrics; GPT judgments are intentionally separate."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["dataset"])].append(record)
    datasets: dict[str, dict[str, Any]] = {}
    classification_total = classification_correct = 0
    generation_total = 0
    for dataset, items in sorted(grouped.items()):
        task_type = str(items[0]["task_type"])
        row: dict[str, Any] = {
            "benchmark": items[0]["benchmark"],
            "task_type": task_type,
            "n": len(items),
        }
        if task_type == "classification":
            outcomes = [_classification_correct(item) for item in items]
            correct = sum(int(outcome[0]) for outcome in outcomes)
            row.update(
                {
                    "accuracy": round(correct / len(items), 3),
                    "correct": correct,
                    "unparsed": sum(int(outcome[1]) for outcome in outcomes),
                }
            )
            classification_total += len(items)
            classification_correct += correct
        else:
            rouge, bleu = _generation_scores(items)
            row.update({"rougeL": round(rouge, 3), "bleu": round(bleu, 3)})
            generation_total += len(items)
        datasets[dataset] = row
    generation_rows = [
        row for row in datasets.values() if row["task_type"] == "generation"
    ]
    classification_rows = [
        row for row in datasets.values() if row["task_type"] == "classification"
    ]
    suite_aggregates = {}
    for suite, suite_name in (
        ("ni", "natural_instructions"),
        ("bbh", "bbh"),
        ("mmlu_pro", "mmlu_pro"),
    ):
        rows = [row for row in datasets.values() if row["benchmark"] == suite_name]
        if not rows:
            continue
        classification = [row for row in rows if row["task_type"] == "classification"]
        generation = [row for row in rows if row["task_type"] == "generation"]
        aggregate: dict[str, Any] = {}
        if classification:
            count = sum(row["n"] for row in classification)
            correct = sum(row["correct"] for row in classification)
            aggregate.update(
                {
                    "classification_n": count,
                    "classification_micro_accuracy": round(correct / count, 3),
                    "classification_macro_accuracy": round(
                        statistics.fmean(row["accuracy"] for row in classification), 3
                    ),
                }
            )
        if generation:
            count = sum(row["n"] for row in generation)
            aggregate.update(
                {
                    "generation_n": count,
                    "generation_macro_rougeL": round(
                        statistics.fmean(row["rougeL"] for row in generation), 3
                    ),
                    "generation_macro_bleu": round(
                        statistics.fmean(row["bleu"] for row in generation), 3
                    ),
                    "generation_micro_rougeL": round(
                        sum(row["rougeL"] * row["n"] for row in generation) / count,
                        3,
                    ),
                    "generation_micro_bleu": round(
                        sum(row["bleu"] * row["n"] for row in generation) / count,
                        3,
                    ),
                }
            )
        suite_aggregates[suite] = aggregate
    return {
        "dataset_metrics": datasets,
        "suite_aggregates": suite_aggregates,
        "classification": {
            "n": classification_total,
            "correct": classification_correct,
            "micro_accuracy": (
                round(classification_correct / classification_total, 3)
                if classification_total else None
            ),
            "macro_accuracy": (
                round(
                    statistics.fmean(row["accuracy"] for row in classification_rows),
                    3,
                )
                if classification_rows else None
            ),
        },
        "generation": {
            "n": generation_total,
            "macro_rougeL": (
                round(statistics.fmean(row["rougeL"] for row in generation_rows), 3)
                if generation_rows else None
            ),
            "macro_bleu": (
                round(statistics.fmean(row["bleu"] for row in generation_rows), 3)
                if generation_rows else None
            ),
            "micro_rougeL": (
                round(
                    sum(row["rougeL"] * row["n"] for row in generation_rows)
                    / generation_total,
                    3,
                )
                if generation_rows else None
            ),
            "micro_bleu": (
                round(
                    sum(row["bleu"] * row["n"] for row in generation_rows)
                    / generation_total,
                    3,
                )
                if generation_rows else None
            ),
        },
        "judge": "not_run",
    }
