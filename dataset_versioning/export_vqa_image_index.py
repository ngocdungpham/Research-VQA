#!/usr/bin/env python3
"""Export readable image-path indexes for derived_v4.1 VQA and golden candidates."""

from __future__ import annotations

import csv
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
FACTS = ROOT / "derived_v4" / "facts_v1.jsonl.gz"
VQA = ROOT / "derived_v4_1" / "accepted_vqa.jsonl.gz"
GOLDEN = ROOT / "derived_v4_1" / "golden_candidates_v1_1" / "golden_candidates_v1_1.jsonl.gz"
ASSIGNMENTS = ROOT / "derived_v3" / "split_assignments.jsonl.gz"
REPORT_DIR = ROOT / "reports"


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def image_row(
    image_id: str,
    fact_record: dict[str, Any],
    assignment: dict[str, Any],
    qa_count: int,
    golden: bool,
) -> dict[str, Any]:
    relative = fact_record["image"]["path"]
    absolute = (ROOT / relative).resolve()
    return {
        "image_id": image_id,
        "split": assignment["split"],
        "patient_id": assignment["patient_id"],
        "procedure_id": assignment["procedure_id"],
        "leakage_group_id": assignment["leakage_group_id"],
        "qa_count": qa_count,
        "is_golden_candidate": str(golden).lower(),
        "relative_image_path": relative,
        "absolute_image_path": str(absolute),
        "image_exists": str(absolute.is_file()).lower(),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "image_id", "split", "patient_id", "procedure_id", "leakage_group_id",
        "qa_count", "is_golden_candidate", "relative_image_path",
        "absolute_image_path", "image_exists",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    facts = {row["image_id"]: row for row in read_jsonl(FACTS)}
    assignments = {row["image_id"]: row for row in read_jsonl(ASSIGNMENTS)}
    qa_counts = Counter(row["image_id"] for row in read_jsonl(VQA))
    golden_records = list(read_jsonl(GOLDEN))
    golden_ids = {row["image_id"] for row in golden_records}

    all_rows = [
        image_row(image_id, facts[image_id], assignments[image_id], qa_count, image_id in golden_ids)
        for image_id, qa_count in sorted(qa_counts.items())
    ]
    golden_rows = [
        image_row(
            row["image_id"], facts[row["image_id"]], assignments[row["image_id"]],
            len(row["qa"]), True,
        )
        for row in golden_records
    ]

    REPORT_DIR.mkdir(exist_ok=True)
    all_path = REPORT_DIR / "vqa_v4_1_image_index.csv"
    golden_path = REPORT_DIR / "golden_candidates_v1_1_image_paths.csv"
    write_csv(all_path, all_rows)
    write_csv(golden_path, golden_rows)

    summary = {
        "all_vqa": {
            "qa_records": sum(qa_counts.values()),
            "unique_images": len(all_rows),
            "images_by_split": dict(Counter(row["split"] for row in all_rows)),
            "missing_image_files": sum(row["image_exists"] != "true" for row in all_rows),
            "output": str(all_path),
        },
        "golden_candidates": {
            "qa_records": sum(len(row["qa"]) for row in golden_records),
            "unique_images": len(golden_rows),
            "missing_image_files": sum(row["image_exists"] != "true" for row in golden_rows),
            "output": str(golden_path),
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
