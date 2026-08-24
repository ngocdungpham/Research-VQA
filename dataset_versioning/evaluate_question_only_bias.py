#!/usr/bin/env python3
"""Evaluate label-frequency baselines that never inspect an image."""

from __future__ import annotations

import argparse
import gzip
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).lower()
    return " ".join(value.split())


def evaluate(
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
    key_name: str,
    key_fn: Callable[[dict[str, Any]], str],
) -> dict[str, Any]:
    distributions: defaultdict[str, Counter[str]] = defaultdict(Counter)
    global_answer = Counter(row["canonical_answer"] for row in train).most_common(1)[0][0]
    for row in train:
        distributions[key_fn(row)][row["canonical_answer"]] += 1

    total = correct = seen = 0
    by_format: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_intent: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for row in test:
        key = key_fn(row)
        prediction = distributions[key].most_common(1)[0][0] if key in distributions else global_answer
        is_correct = prediction == row["canonical_answer"]
        total += 1
        correct += is_correct
        seen += key in distributions
        for bucket, name in ((by_format, row["question_format"]), (by_intent, row["question_intent"])):
            bucket[name]["records"] += 1
            bucket[name]["correct"] += is_correct

    def summarize(groups: defaultdict[str, Counter[str]]) -> dict[str, Any]:
        return {
            key: {
                "records": value["records"],
                "correct": value["correct"],
                "accuracy": value["correct"] / value["records"],
            }
            for key, value in sorted(groups.items())
        }

    return {
        "baseline": f"train-majority-by-{key_name}",
        "uses_image": False,
        "records": total,
        "correct": correct,
        "accuracy": correct / total,
        "test_key_coverage": seen / total,
        "by_format": summarize(by_format),
        "by_intent": summarize(by_intent),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "derived_v4_1" / "vqa")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "vqa_28k_question_only_bias.json")
    args = parser.parse_args()
    train = list(read_jsonl(args.data / "train.jsonl.gz"))
    test = list(read_jsonl(args.data / "test.jsonl.gz"))
    results = {
        "dataset": "derived_v4.1",
        "train_records": len(train),
        "test_records": len(test),
        "metric": "exact canonical-answer accuracy",
        "baselines": [
            evaluate(train, test, "normalized-question", lambda row: normalize_text(row["question"])),
            evaluate(train, test, "question-intent", lambda row: row["question_intent"]),
            evaluate(train, test, "question-format", lambda row: row["question_format"]),
        ],
        "interpretation": (
            "These baselines never inspect images. High accuracy indicates answer/template bias and must not be "
            "interpreted as visual reasoning."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
