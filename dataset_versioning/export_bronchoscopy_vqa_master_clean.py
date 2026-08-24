#!/usr/bin/env python3
"""Build an audit-rich master VQA file and minimal train/val/test exports."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
CLEAN_FIELDS = (
    "qa_id", "image", "conversations", "question_type", "q_type",
    "evidence_boxes_xyxy", "answer_structured",
)
Q_TYPES = {
    "closed-ended": "closed_ended_questions",
    "single-choice": "single_choice_questions",
    "multi-choice": "multi_choice_questions",
    "open-ended": "open_ended_questions",
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


def choices_text(options: dict[str, str]) -> str:
    if not options:
        return ""
    return " <choices>: [" + ", ".join(f"{key}: {value}" for key, value in sorted(options.items())) + "]"


def conversation_answer(row: dict[str, Any]) -> str:
    structured = row.get("answer_structured") or {}
    if row["question_format"] == "single-choice" and "option" in structured:
        return str(structured["option"])
    if row["question_format"] == "multi-choice" and isinstance(structured.get("options"), list):
        return ", ".join(map(str, structured["options"]))
    return row["answer_text"]


def protected_blueprint_conflicts(row: dict[str, Any], request: dict[str, Any]) -> list[str]:
    """Detect protected intent/answer mappings that no paraphrase can repair."""
    fact_ids = set(row.get("source_fact_ids") or [])
    selected = [fact for fact in request.get("facts", []) if fact.get("fact_id") in fact_ids]
    if not selected:
        return []

    def is_explicit_normal(fact: dict[str, Any]) -> bool:
        concept = str(fact.get("concept") or "")
        return fact.get("value") is True and (concept == "image_normal" or concept.endswith("_normal"))

    has_explicit_normal_fact = any(is_explicit_normal(fact) for fact in selected)
    only_normal_facts = all(is_explicit_normal(fact) for fact in selected)
    conflicts = []
    if has_explicit_normal_fact and row.get("question_intent") == "abnormality":
        conflicts.append("abnormality_answer_includes_explicit_normal_fact")
    if (
        only_normal_facts
        and row.get("question_intent") == "abnormality_presence"
        and (row.get("answer_structured") or {}).get("abnormality_present") is True
    ):
        conflicts.append("abnormality_true_contradicts_explicit_normal_facts")
    return conflicts


def validate_clean(row: dict[str, Any]) -> list[str]:
    failures = []
    if tuple(row) != CLEAN_FIELDS:
        failures.append("top_level_fields_not_exact")
    if not isinstance(row.get("qa_id"), str) or not row["qa_id"]:
        failures.append("invalid_qa_id")
    if not isinstance(row.get("image"), str) or not row["image"]:
        failures.append("invalid_image")
    conversations = row.get("conversations")
    if not isinstance(conversations, list) or len(conversations) != 2:
        failures.append("invalid_conversations")
    else:
        if conversations[0].get("from") != "human" or not conversations[0].get("value", "").startswith("<image>\n"):
            failures.append("invalid_human_turn")
        if conversations[1].get("from") != "gpt" or not all(token in conversations[1].get("value", "") for token in ("<answer>", "<reason>", "<visual_evidence>", "<location>")):
            failures.append("invalid_gpt_turn")
    if row.get("q_type") not in set(Q_TYPES.values()):
        failures.append("invalid_q_type")
    if not isinstance(row.get("evidence_boxes_xyxy"), list):
        failures.append("invalid_evidence_boxes")
    elif any(not isinstance(box, list) or len(box) != 4 or any(not isinstance(v, int) for v in box) for box in row["evidence_boxes_xyxy"]):
        failures.append("invalid_bbox")
    if not isinstance(row.get("answer_structured"), dict):
        failures.append("invalid_answer_structured")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--accepted", type=Path, nargs="+", required=True,
        help="One or more accepted JSONL(.gz) files, in cohort order.",
    )
    parser.add_argument(
        "--requests", type=Path, nargs="+", required=True,
        help="One or more request JSONL(.gz) files matching --accepted cohorts.",
    )
    parser.add_argument(
        "--review", type=Path, nargs="*", default=[],
        help="Optional review-required JSONL(.gz) files retained only in Master.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-version", default="derived_v4.2-vision")
    args = parser.parse_args()
    args.accepted = [path.resolve() for path in args.accepted]
    args.requests = [path.resolve() for path in args.requests]
    args.review = [path.resolve() for path in args.review]
    args.output = args.output.resolve()
    if len(args.accepted) != len(args.requests):
        raise ValueError("--accepted and --requests must contain the same number of cohort files")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite export: {args.output}")
    args.output.mkdir(parents=True)

    requests = {}
    for path in args.requests:
        for row in read_jsonl(path):
            image_id = row["image_id"]
            if image_id in requests:
                raise ValueError(f"Duplicate request image_id across cohorts: {image_id}")
            requests[image_id] = row
    accepted = [row for path in args.accepted for row in read_jsonl(path)]
    review_required = [row for path in args.review for row in read_jsonl(path)]
    master_rows = []
    clean_by_split = {"train": [], "validation": [], "test": []}
    failures = []
    qa_ids = set()
    blueprint_conflicts = []
    for row in sorted(accepted, key=lambda x: (x["split"], x["image_id"], x["qa_id"])):
        if row["qa_id"] in qa_ids:
            failures.append({"qa_id": row["qa_id"], "failures": ["duplicate_qa_id"]}); continue
        qa_ids.add(row["qa_id"])
        request = requests[row["image_id"]]
        region_map = {region["evidence_region_id"]: region for region in request["evidence_regions"]}
        boxes = [region_map[rid]["bbox_xyxy_pixels"] for rid in row["evidence_region_ids"] if rid in region_map]
        options = row.get("options") or {}
        conversations = [
            {"from": "human", "value": f"<image>\n{row['question']}{choices_text(options)}"},
            {"from": "gpt", "value": f"<answer> {conversation_answer(row)} <reason> {row['reasoning_text']} <visual_evidence> {row['visual_evidence_summary']} <location> {json.dumps(boxes, ensure_ascii=False)}"},
        ]
        row_conflicts = protected_blueprint_conflicts(row, request)
        master = {
            **row,
            "image": request["image_path"],
            "overlay_image": request["overlay_path"],
            "conversations": conversations,
            "question_type": row["question_intent"],
            "q_type": Q_TYPES[row["question_format"]],
            "evidence_boxes_xyxy": boxes,
            "master_dataset_version": args.dataset_version,
            "master_record_status": "PROTECTED_BLUEPRINT_CONFLICT_NOT_EXPORTED" if row_conflicts else "ACCEPTED_EXPORTED",
        }
        if row_conflicts:
            master["export_review_reasons"] = row_conflicts
            blueprint_conflicts.append({"qa_id": row["qa_id"], "reasons": row_conflicts})
        master_rows.append(master)
        if row_conflicts:
            continue
        clean = {
            "qa_id": row["qa_id"],
            "image": request["image_path"],
            "conversations": conversations,
            "question_type": row["question_intent"],
            "q_type": Q_TYPES[row["question_format"]],
            "evidence_boxes_xyxy": boxes,
            "answer_structured": row["answer_structured"],
        }
        clean_failures = validate_clean(clean)
        if clean_failures:
            failures.append({"qa_id": row["qa_id"], "failures": clean_failures})
        clean_by_split[row["split"]].append(clean)

    # Review-required rows stay audit-visible in Master but never enter clean splits.
    for row in sorted(review_required, key=lambda x: (x["split"], x["image_id"], x["qa_id"])):
        if row["qa_id"] in qa_ids:
            failures.append({"qa_id": row["qa_id"], "failures": ["duplicate_qa_id_across_accepted_and_review"]})
            continue
        qa_ids.add(row["qa_id"])
        request = requests[row["image_id"]]
        region_map = {region["evidence_region_id"]: region for region in request["evidence_regions"]}
        boxes = [region_map[rid]["bbox_xyxy_pixels"] for rid in row["evidence_region_ids"] if rid in region_map]
        master_rows.append({
            **row,
            "image": request["image_path"],
            "overlay_image": request["overlay_path"],
            "question_type": row["question_intent"],
            "q_type": Q_TYPES[row["question_format"]],
            "evidence_boxes_xyxy": boxes,
            "master_dataset_version": args.dataset_version,
            "master_record_status": "REVIEW_REQUIRED_NOT_EXPORTED",
        })

    if failures:
        (args.output / "export_failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise ValueError(f"Clean export validation failed for {len(failures)} rows")

    paths = {"master": args.output / "master_bronchoscopy_vqa.json", "train": args.output / "train.json", "val": args.output / "val.json", "test": args.output / "test.json"}
    paths["master"].write_text(json.dumps(master_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    paths["train"].write_text(json.dumps(clean_by_split["train"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    paths["val"].write_text(json.dumps(clean_by_split["validation"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    paths["test"].write_text(json.dumps(clean_by_split["test"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Read back from disk: clean files must contain only the exact seven fields.
    for split_name in ("train", "val", "test"):
        for index, row in enumerate(json.loads(paths[split_name].read_text(encoding="utf-8"))):
            row_failures = validate_clean(row)
            if row_failures:
                raise ValueError(f"Read-back validation failed: {split_name}[{index}] {row_failures}")
    manifest = {
        "dataset_version": args.dataset_version,
        "master_policy": "retains all audit/provenance/LLM trace fields",
        "clean_policy": {"exact_top_level_fields": list(CLEAN_FIELDS), "derived_automatically_from_master": True},
        "records": len(master_rows),
        "source_accepted_records": len(accepted),
        "review_required_records": len(review_required),
        "protected_blueprint_conflict_records": len(blueprint_conflicts),
        "clean_export_records": sum(len(rows) for rows in clean_by_split.values()),
        "splits": {"train": len(clean_by_split["train"]), "val": len(clean_by_split["validation"]), "test": len(clean_by_split["test"])},
        "source": {
            "cohorts": [
                {
                    "accepted": str(accepted_path.relative_to(ROOT)),
                    "accepted_sha256": sha256_file(accepted_path),
                    "requests": str(request_path.relative_to(ROOT)),
                    "requests_sha256": sha256_file(request_path),
                }
                for accepted_path, request_path in zip(args.accepted, args.requests)
            ],
            "review": [
                {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
                for path in args.review
            ],
        },
        "outputs": {name: {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "bytes": path.stat().st_size} for name, path in paths.items()},
        "validation": {"status": "PASS", "failures": 0},
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
