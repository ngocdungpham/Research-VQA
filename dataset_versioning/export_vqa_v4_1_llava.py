#!/usr/bin/env python3
"""Export derived_v4.1 into the conversation schema used by GEMeX/LLaVA.

The versioned source files are never modified. Image paths and evidence boxes
are resolved from canonical_v2.2 and derived_v2.0 into a model-facing export.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
FORMAT_TO_Q_TYPE = {
    "closed-ended": "closed_ended_questions",
    "single-choice": "single_choice_questions",
    "multi-choice": "multi_choice_questions",
    "open-ended": "open_ended_questions",
}
TYPE_PROMPTS = {
    "closed-ended": "Input a closed-ended question, and the assistant will output its answer (yes or no) with a detailed reason and corresponding visual location.",
    "single-choice": "Input a single-choice question, and the assistant will output its answer (an option) with a detailed reason and corresponding visual location.",
    "multi-choice": "Input a multi-choice question, and the assistant will output its answer (some options) with a detailed reason and corresponding visual location.",
    "open-ended": "Input an open-ended question, and the assistant will output its answer with a detailed reason and corresponding visual location.",
}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def polygon_bbox(points: list[list[float]]) -> list[float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def rounded_box(box: list[float]) -> list[int]:
    return [round(value) for value in box]


def option_text(options: dict[str, str]) -> str:
    if not options:
        return ""
    return "\n" + "\n".join(f"{key}. {value}" for key, value in sorted(options.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "derived_v4_1")
    parser.add_argument("--canonical", type=Path, default=ROOT / "canonical_v2_2" / "images.jsonl.gz")
    parser.add_argument("--derivations", type=Path, default=ROOT / "derived_v2" / "region_derivations.jsonl.gz")
    parser.add_argument("--output", type=Path, default=ROOT / "model_data" / "derived_v4_1_llava")
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing export: {args.output}")

    canonical = {}
    for image in read_jsonl(args.canonical):
        regions = {region["region_id"]: region for region in image["regions"]}
        canonical[image["image_id"]] = {**image, "regions_by_id": regions}

    evidence_boxes: dict[tuple[str, str], list[int]] = {}
    for derivation in read_jsonl(args.derivations):
        image_id = derivation["image_id"]
        image = canonical[image_id]
        for region in derivation["regions"]:
            if region.get("training_polygon_override"):
                box = polygon_bbox(region["training_polygon_override"])
            else:
                source_id = region.get("training_geometry_source_region_id")
                if not source_id:
                    source_ids = region.get("source_region_ids", [])
                    source_id = source_ids[0] if source_ids else None
                if source_id is None:
                    continue
                box = image["regions_by_id"][source_id]["bbox_xyxy"]
            evidence_boxes[(image_id, region["canonical_region_id"])] = rounded_box(box)

    args.output.mkdir(parents=True)
    manifest = {
        "format": "gemex_llava_conversation_v1",
        "source_dataset": "derived_v4.1",
        "source_manifest_sha256": sha256_file(args.source / "manifest.json"),
        "canonical_sha256": sha256_file(args.canonical),
        "derivations_sha256": sha256_file(args.derivations),
        "splits": {},
    }

    for split in ("train", "validation", "test"):
        source_path = args.source / "vqa" / f"{split}.jsonl.gz"
        output_path = args.output / f"{split}.json"
        output_rows = []
        for index, row in enumerate(read_jsonl(source_path)):
            image = canonical[row["image_id"]]
            boxes = [
                evidence_boxes[(row["image_id"], region_id)]
                for region_id in row["evidence_region_ids"]
                if (row["image_id"], region_id) in evidence_boxes
            ]
            question = row["question"] + option_text(row.get("options", {}))
            question += f"\nKích thước ảnh gốc: {image['width']}x{image['height']} pixel."
            target = (
                f"<answer> {row['answer_text']} "
                f"<reason> {row['reasoning_text']} "
                f"<location> {json.dumps(boxes, ensure_ascii=False)}"
            )
            fmt = row["question_format"]
            output_rows.append(
                {
                    "conversations": [
                        {"from": "human", "value": f"<image>\n{question}"},
                        {"from": "gpt", "value": target},
                    ],
                    "row_id": index,
                    "qa_id": row["qa_id"],
                    "case_id": row["patient_id"],
                    "procedure_id": row["procedure_id"],
                    "object_id": row["image_id"],
                    "q_type": FORMAT_TO_Q_TYPE[fmt],
                    "question_type": row["question_intent"],
                    "type_prompt": TYPE_PROMPTS[fmt],
                    "image": image["image_path"],
                    "width": image["width"],
                    "height": image["height"],
                    "source_split": row["split"],
                    "source_dataset": "derived_v4.1",
                }
            )
        output_path.write_text(
            json.dumps(output_rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        missing_images = sum(not (ROOT / item["image"]).is_file() for item in output_rows)
        manifest["splits"][split] = {
            "records": len(output_rows),
            "missing_images": missing_images,
            "path": str(output_path.relative_to(ROOT)),
            "sha256": sha256_file(output_path),
        }

    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
