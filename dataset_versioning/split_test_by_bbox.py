#!/usr/bin/env python3
"""Partition a LLaVA-format test set by the presence of target boxes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BOX_RE = re.compile(
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,"
    r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def target_text(row: dict[str, Any]) -> str:
    return next(turn["value"] for turn in row["conversations"] if turn["from"] == "gpt")


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(rows),
        "unique_qa_ids": len({row["qa_id"] for row in rows}),
        "q_type": dict(sorted(Counter(row["q_type"] for row in rows).items())),
        "question_type": dict(
            sorted(Counter(row["question_type"] for row in rows).items())
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "model_data" / "derived_v4_1_llava_en" / "test.json",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or args.input.parent
    with_bbox_path = output_dir / "test_with_bbox.json"
    without_bbox_path = output_dir / "test_without_bbox.json"
    manifest_path = output_dir / "test_bbox_partitions.manifest.json"
    for path in (with_bbox_path, without_bbox_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing partition: {path}")

    rows = json.loads(args.input.read_text(encoding="utf-8"))
    with_bbox: list[dict[str, Any]] = []
    without_bbox: list[dict[str, Any]] = []
    original_ids = []
    for row in rows:
        original_ids.append(row["qa_id"])
        partition = "with_bbox" if BOX_RE.search(target_text(row)) else "without_bbox"
        output = {**row, "bbox_partition": partition}
        (with_bbox if partition == "with_bbox" else without_bbox).append(output)

    with_ids = {row["qa_id"] for row in with_bbox}
    without_ids = {row["qa_id"] for row in without_bbox}
    if with_ids & without_ids:
        raise ValueError("Partitions are not disjoint")
    if with_ids | without_ids != set(original_ids):
        raise ValueError("Partitions do not reconstruct the source test set")
    if len(original_ids) != len(set(original_ids)):
        raise ValueError("Source test set contains duplicate qa_id values")
    if any(not BOX_RE.search(target_text(row)) for row in with_bbox):
        raise ValueError("with_bbox contains a row without target boxes")
    if any(BOX_RE.search(target_text(row)) for row in without_bbox):
        raise ValueError("without_bbox contains a row with target boxes")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(with_bbox_path, with_bbox)
    write_json(without_bbox_path, without_bbox)
    manifest = {
        "source": str(args.input.relative_to(ROOT)),
        "source_sha256": sha256_file(args.input),
        "partition_rule": "target <location> contains at least one [x1,y1,x2,y2] box",
        "source_records": len(rows),
        "validation": {
            "disjoint": True,
            "complete_union": True,
            "source_qa_ids_unique": True,
            "relative_order_preserved_within_each_partition": True,
        },
        "partitions": {
            "with_bbox": {
                **distribution(with_bbox),
                "path": str(with_bbox_path.relative_to(ROOT)),
                "sha256": sha256_file(with_bbox_path),
            },
            "without_bbox": {
                **distribution(without_bbox),
                "path": str(without_bbox_path.relative_to(ROOT)),
                "sha256": sha256_file(without_bbox_path),
            },
        },
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
