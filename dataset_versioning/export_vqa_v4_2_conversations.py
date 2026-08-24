#!/usr/bin/env python3
"""Export v4.2 into a GEMeX/LLaVA-style conversation schema.

The raw LLM surface response is intentionally sparse. This exporter joins it
with the protected blueprint, resolves evidence regions to boxes, and writes a
model-facing record without asking the LLM to reproduce protected fields.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
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
PROTECTED_FIELDS = ("source_fact_ids", "answer_structured", "answer_text", "options", "evidence_region_ids")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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


def protected_sha(row: dict[str, Any]) -> str:
    payload = {field: row[field] for field in PROTECTED_FIELDS}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def polygon_bbox(points: list[list[float]]) -> list[float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def rounded_box(box: list[float]) -> list[int]:
    return [round(value) for value in box]


def choices_text(options: dict[str, str]) -> str:
    if not options:
        return ""
    values = ", ".join(f"{key}: {value}" for key, value in sorted(options.items()))
    return f" <choices>: [{values}]"


def conversation_answer(row: dict[str, Any]) -> str:
    structured = row.get("answer_structured") or {}
    if row["question_format"] in {"single-choice", "multi-choice"}:
        if "option" in structured:
            return str(structured["option"])
        if "options" in structured and isinstance(structured["options"], list):
            return ", ".join(map(str, structured["options"]))
    return row["answer_text"]


def load_preview_rows(source_dir: Path) -> list[dict[str, Any]]:
    blueprints = {row["qa_id"]: row for row in read_jsonl(source_dir / "qa_blueprints_v1_2.jsonl.gz")}
    candidates: dict[str, dict[str, Any]] = {}
    for candidate in read_jsonl(source_dir / "surface_candidates.jsonl"):
        candidates[candidate["qa_id"]] = candidate
    rows = []
    for qa_id, candidate in candidates.items():
        blueprint = blueprints.get(qa_id)
        if blueprint is None:
            raise ValueError(f"Candidate without blueprint: {qa_id}")
        if candidate.get("protected_sha256") != blueprint["protected_sha256"]:
            raise ValueError(f"Protected seal mismatch: {qa_id}")
        if protected_sha(blueprint) != blueprint["protected_sha256"]:
            raise ValueError(f"Blueprint protected fields changed: {qa_id}")
        rows.append({
            **blueprint,
            "question": candidate["question"],
            "reasoning_text": candidate["reasoning_text"],
            "llm_trace": candidate.get("llm_trace"),
            "validation_status": "surface_validator_passed_preview",
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "derived_v4_2")
    parser.add_argument("--mode", choices=("preview", "accepted"), default="preview")
    parser.add_argument("--canonical", type=Path, default=ROOT / "canonical_v2_2" / "images.jsonl.gz")
    parser.add_argument("--derivations", type=Path, default=ROOT / "derived_v2" / "region_derivations.jsonl.gz")
    parser.add_argument("--output", type=Path, default=ROOT / "model_data" / "derived_v4_2_llava_preview")
    args = parser.parse_args()
    args.source = args.source.resolve()
    args.canonical = args.canonical.resolve()
    args.derivations = args.derivations.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")

    if args.mode == "preview":
        source_rows = load_preview_rows(args.source)
        export_status = "PREVIEW_NOT_A_RELEASE"
    else:
        source_rows = list(read_jsonl(args.source / "accepted_vqa.jsonl.gz"))
        export_status = "ACCEPTED_RELEASE_EXPORT"

    canonical = {}
    for image in read_jsonl(args.canonical):
        canonical[image["image_id"]] = {**image, "regions_by_id": {r["region_id"]: r for r in image["regions"]}}

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

    split_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        image = canonical[row["image_id"]]
        boxes = [
            evidence_boxes[(row["image_id"], region_id)]
            for region_id in row["evidence_region_ids"]
            if (row["image_id"], region_id) in evidence_boxes
        ]
        human = f"<image>\n{row['question']}{choices_text(row.get('options', {}))}"
        gpt = (
            f"<answer> {conversation_answer(row)} "
            f"<reason> {row['reasoning_text']} "
            f"<location> {json.dumps(boxes, ensure_ascii=False)}"
        )
        fmt = row["question_format"]
        split_rows[row["split"]].append({
            "conversations": [
                {"from": "human", "value": human},
                {"from": "gpt", "value": gpt},
            ],
            "row_id": len(split_rows[row["split"]]),
            "qa_id": row["qa_id"],
            "q_type": FORMAT_TO_Q_TYPE[fmt],
            "type_prompt": TYPE_PROMPTS[fmt],
            "image": image["image_path"],
            "case_id": row["patient_id"],
            "procedure_id": row["procedure_id"],
            "object_id": row["image_id"],
            "question_type": row["question_intent"],
            "question_format": fmt,
            "answer_structured": row["answer_structured"],
            "answer_text": row["answer_text"],
            "canonical_answer": row["canonical_answer"],
            "options": row.get("options", {}),
            "source_fact_ids": row["source_fact_ids"],
            "evidence_region_ids": row["evidence_region_ids"],
            "evidence_boxes_xyxy": boxes,
            "protected_sha256": row["protected_sha256"],
            "surface_llm_trace": row.get("llm_trace"),
            "source_split": row["split"],
            "source_dataset": "derived_v4.2",
            "export_status": export_status,
        })

    args.output.mkdir(parents=True)
    manifest = {
        "format": "gemex_llava_conversation_v2_with_provenance",
        "source_dataset": "derived_v4.2", "mode": args.mode, "export_status": export_status,
        "records": sum(map(len, split_rows.values())), "splits": {},
        "canonical_sha256": sha256_file(args.canonical), "derivations_sha256": sha256_file(args.derivations),
    }
    for split in ("train", "validation", "test"):
        path = args.output / f"{split}.json"
        path.write_text(json.dumps(split_rows.get(split, []), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest["splits"][split] = {"records": len(split_rows.get(split, [])), "path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
