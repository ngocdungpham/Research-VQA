#!/usr/bin/env python3
"""Create derived_v4.1 by safely rephrasing half of negative normality QA.

The transformation does not flip a label. It replaces:
  "Is this image confirmed normal?" -> "No"
with the logically equivalent, positively oriented question:
  "Is a confirmed abnormality present?" -> "Yes"

Only images with at least one accepted, question-eligible, positive observation
fact can be converted. Selection is deterministic and stratified by split.
The frozen derived_v4 source is opened read-only and is never modified.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "derived_v4"
STAGE3_DIR = ROOT / "derived_v3"
OUTPUT_DIR = ROOT / "derived_v4_1"
DATASET_VERSION = "derived_v4.1"
BLUEPRINT_VERSION = "bronchoscopy_qa_blueprints_v1.1"
TRANSFORMATION_VERSION = "normality_polarity_rebalance_v1.0"
CHUNK = 1024 * 1024


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(handle: Any, value: Any) -> None:
    handle.write(canonical_json(value) + "\n")


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def stable_id(prefix: str, *parts: str, length: int = 24) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def selection_rank(split: str, image_id: str) -> str:
    material = f"{TRANSFORMATION_VERSION}\0{split}\0{image_id}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).lower()
    return " ".join("".join(char if char.isalnum() or char.isspace() else " " for char in value).split())


def join_vi(items: list[str]) -> str:
    unique = list(dict.fromkeys(items))
    if not unique:
        return "bất thường nội soi"
    if len(unique) == 1:
        return unique[0]
    return ", ".join(unique[:-1]) + " và " + unique[-1]


def positive_observations(fact_record: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        [
            fact for fact in fact_record["facts"]
            if fact["fact_kind"] == "observation"
            and fact["polarity"] == "present"
            and fact["question_eligible"]
            and fact["evidence_status"] == "accepted"
        ],
        key=lambda fact: fact["fact_id"],
    )


def convert_qa(row: dict[str, Any], facts: list[dict[str, Any]]) -> dict[str, Any]:
    source_qa_id = row["qa_id"]
    fact_ids = [fact["fact_id"] for fact in facts]
    evidence_ids = sorted({region_id for fact in facts for region_id in fact["evidence_region_ids"]})
    surfaces = sorted({fact["surface_vi"] for fact in facts})
    answer_structured = {"abnormality_present": True}
    question = "Hình ảnh nội soi này có bất thường được xác nhận không?"
    answer = "Có."
    reasoning = f"Các vùng bằng chứng cho thấy {join_vi(surfaces)}."
    qa_id = stable_id(
        "qa", row["image_id"], "abnormality_presence", "closed-ended",
        "abnormality_presence_closed_01", *fact_ids,
    )
    return {
        **row,
        "blueprint_version": BLUEPRINT_VERSION,
        "qa_id": qa_id,
        "question_format": "closed-ended",
        "question_intent": "abnormality_presence",
        "template_id": "abnormality_presence_closed_01",
        "question_draft": question,
        "question": question,
        "answer_structured": answer_structured,
        "answer_text_draft": answer,
        "answer_text": answer,
        "reasoning_text_draft": reasoning,
        "reasoning_text": reasoning,
        "options": {},
        "source_fact_ids": fact_ids,
        "evidence_region_ids": evidence_ids,
        "canonical_answer": canonical_json(answer_structured),
        "required_surface_terms": surfaces,
        "llm_trace": {
            "surface_method": "deterministic_logical_polarity_inversion",
            "source_dataset_version": "derived_v4.0",
            "source_qa_id": source_qa_id,
            "transformation_version": TRANSFORMATION_VERSION,
        },
        "polarity_rebalance_trace": {
            "source_qa_id": source_qa_id,
            "source_question_intent": row["question_intent"],
            "source_answer_text": row["answer_text"],
            "logical_equivalence": "normal=false and positive_observation=true => abnormality_present=true",
        },
    }


def build(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    source_manifest_path = args.source_dir / "manifest.json"
    stage3_manifest_path = args.stage3_dir / "manifest.json"
    if not source_manifest_path.is_file() or not stage3_manifest_path.is_file():
        raise FileNotFoundError("Required source manifest is missing")

    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("dataset_version") != "derived_v4.0":
        raise ValueError("Expected a frozen derived_v4.0 source")

    facts_by_image = {row["image_id"]: row for row in read_jsonl(args.source_dir / "facts_v1.jsonl.gz")}
    source_rows = list(read_jsonl(args.source_dir / "accepted_vqa.jsonl.gz"))

    eligible_by_split: dict[str, list[str]] = defaultdict(list)
    positive_by_image: dict[str, list[dict[str, Any]]] = {}
    negative_normality_by_split = Counter()
    nonconvertible_by_split = Counter()
    for row in source_rows:
        if row["question_intent"] != "image_normality" or row["answer_text"] != "Không.":
            continue
        negative_normality_by_split[row["split"]] += 1
        observations = positive_observations(facts_by_image[row["image_id"]])
        if observations:
            eligible_by_split[row["split"]].append(row["image_id"])
            positive_by_image[row["image_id"]] = observations
        else:
            nonconvertible_by_split[row["split"]] += 1

    selected_by_split: dict[str, set[str]] = {}
    targets = {}
    for split in ("train", "validation", "test"):
        target = negative_normality_by_split[split] // 2
        ranked = sorted(set(eligible_by_split[split]), key=lambda image_id: (selection_rank(split, image_id), image_id))
        if len(ranked) < target:
            raise ValueError(f"Only {len(ranked)} convertible rows for target {target} in {split}")
        selected_by_split[split] = set(ranked[:target])
        targets[split] = target

    selected_images = set().union(*selected_by_split.values())
    converted_rows: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    before_answers = Counter()
    after_answers = Counter()
    before_intents = Counter()
    after_intents = Counter()
    for row in source_rows:
        before_answers[row["answer_text"]] += 1
        before_intents[row["question_intent"]] += 1
        if (
            row["image_id"] in selected_images
            and row["question_intent"] == "image_normality"
            and row["answer_text"] == "Không."
        ):
            converted = convert_qa(row, positive_by_image[row["image_id"]])
            changes.append({
                "image_id": row["image_id"], "split": row["split"],
                "source_qa_id": row["qa_id"], "new_qa_id": converted["qa_id"],
                "source_question": row["question"], "new_question": converted["question"],
                "source_answer": row["answer_text"], "new_answer": converted["answer_text"],
                "source_fact_ids": converted["source_fact_ids"],
                "evidence_region_ids": converted["evidence_region_ids"],
            })
            row = converted
        converted_rows.append(row)
        after_answers[row["answer_text"]] += 1
        after_intents[row["question_intent"]] += 1

    args.output_dir.mkdir(parents=True)
    (args.output_dir / "vqa").mkdir()
    (args.output_dir / "golden_candidates_v1_1").mkdir()

    accepted_path = args.output_dir / "accepted_vqa.jsonl.gz"
    split_paths = {split: args.output_dir / "vqa" / f"{split}.jsonl.gz" for split in ("train", "validation", "test")}
    split_handles = {split: gzip.open(path, "wt", encoding="utf-8", compresslevel=9) for split, path in split_paths.items()}
    with gzip.open(accepted_path, "wt", encoding="utf-8", compresslevel=9) as accepted:
        try:
            for row in sorted(converted_rows, key=lambda item: (item["split"], item["image_id"], item["qa_id"])):
                write_jsonl(accepted, row)
                write_jsonl(split_handles[row["split"]], row)
        finally:
            for handle in split_handles.values():
                handle.close()

    changes_path = args.output_dir / "polarity_rebalance_changes.jsonl.gz"
    with gzip.open(changes_path, "wt", encoding="utf-8", compresslevel=9) as handle:
        for row in sorted(changes, key=lambda item: (item["split"], item["image_id"])):
            write_jsonl(handle, row)

    for filename in ("rejected_vqa.jsonl.gz", "review_queue.jsonl.gz", "dedup_log.jsonl.gz"):
        shutil.copy2(args.source_dir / filename, args.output_dir / filename)

    remap = {row["source_qa_id"]: row["new_qa_id"] for row in changes}
    converted_by_id = {row["qa_id"]: row for row in converted_rows}
    golden_source = args.source_dir / "golden_candidates_v1" / "golden_candidates_v1.jsonl.gz"
    golden_output = args.output_dir / "golden_candidates_v1_1" / "golden_candidates_v1_1.jsonl.gz"
    golden_rows = []
    with gzip.open(golden_output, "wt", encoding="utf-8", compresslevel=9) as handle:
        for candidate in read_jsonl(golden_source):
            qas = []
            for qa in candidate["qa"]:
                new_id = remap.get(qa["qa_id"], qa["qa_id"])
                qas.append(converted_by_id[new_id])
            updated = {**candidate, "candidate_status": "pending_dual_physician_review", "qa": qas}
            golden_rows.append(updated)
            write_jsonl(handle, updated)

    review_path = args.output_dir / "golden_candidates_v1_1" / "dual_physician_review.csv"
    fields = [
        "image_id", "qa_id", "doctor_1_answerable", "doctor_1_answer_correct", "doctor_1_evidence_correct",
        "doctor_1_reasoning_factual", "doctor_1_comment", "doctor_2_answerable", "doctor_2_answer_correct",
        "doctor_2_evidence_correct", "doctor_2_reasoning_factual", "doctor_2_comment", "adjudication", "final_status",
    ]
    with review_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for candidate in golden_rows:
            for qa in candidate["qa"]:
                writer.writerow({"image_id": candidate["image_id"], "qa_id": qa["qa_id"]})

    report = {
        "transformation_version": TRANSFORMATION_VERSION,
        "selection_rule": "50_percent_of_negative_image_normality_per_split_ranked_by_sha256",
        "conversion_guard": "at_least_one_accepted_question_eligible_positive_observation",
        "negative_image_normality_before_by_split": dict(negative_normality_by_split),
        "convertible_by_split": {split: len(set(images)) for split, images in eligible_by_split.items()},
        "nonconvertible_by_split": dict(nonconvertible_by_split),
        "converted_by_split": targets,
        "converted_total": len(changes),
        "answers_before": dict(before_answers),
        "answers_after": dict(after_answers),
        "intents_before": dict(before_intents),
        "intents_after": dict(after_intents),
    }
    write_json(args.output_dir / "rebalance_report.json", report)

    golden_manifest = {
        "version": "golden_candidates_v1.1",
        "status": "pending_dual_physician_review",
        "not_a_golden_test_yet": True,
        "selected_images": len(golden_rows),
        "selected_leakage_groups": len({row["leakage_group_id"] for row in golden_rows}),
        "qa_records": sum(len(row["qa"]) for row in golden_rows),
        "converted_qa_records": sum(
            qa["qa_id"] in set(remap.values()) for row in golden_rows for qa in row["qa"]
        ),
        "outputs": {
            golden_output.name: sha256_file(golden_output),
            review_path.name: sha256_file(review_path),
        },
    }
    write_json(args.output_dir / "golden_candidates_v1_1" / "manifest.json", golden_manifest)

    outputs = []
    for path in sorted(args.output_dir.rglob("*")):
        if path.is_file() and path != args.output_dir / "manifest.json":
            outputs.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    manifest = {
        "dataset_version": DATASET_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": {
            "dataset_version": "derived_v4.0",
            "manifest_path": source_manifest_path.relative_to(ROOT).as_posix(),
            "manifest_sha256": sha256_file(source_manifest_path),
        },
        "split_source": {
            "dataset_version": "derived_v3.0",
            "manifest_path": stage3_manifest_path.relative_to(ROOT).as_posix(),
            "manifest_sha256": sha256_file(stage3_manifest_path),
        },
        "conversion_script_sha256": sha256_file(Path(__file__)),
        "transformation": report,
        "golden_test_status": "not_created_pending_dual_physician_review",
        "outputs": outputs,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({
        "dataset_version": DATASET_VERSION,
        "converted_total": len(changes),
        "converted_by_split": targets,
        "binary_answers_before": {key: before_answers[key] for key in ("Có.", "Không.")},
        "binary_answers_after": {key: after_answers[key] for key in ("Có.", "Không.")},
        "golden_candidates": len(golden_rows),
        "manifest": str(args.output_dir / "manifest.json"),
    }, ensure_ascii=False, indent=2))


def verify(args: argparse.Namespace) -> None:
    manifest_path = args.output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for output in manifest["outputs"]:
        path = ROOT / output["path"]
        if not path.is_file() or sha256_file(path) != output["sha256"]:
            failures.append(f"checksum:{output['path']}")

    facts_by_image = {row["image_id"]: row for row in read_jsonl(args.source_dir / "facts_v1.jsonl.gz")}
    assignments = {row["image_id"]: row for row in read_jsonl(args.stage3_dir / "split_assignments.jsonl.gz")}
    seen_qa_ids = set()
    exact_keys = set()
    split_entities = {key: defaultdict(set) for key in ("patient_id", "procedure_id", "leakage_group_id")}
    counts = Counter()
    for row in read_jsonl(args.output_dir / "accepted_vqa.jsonl.gz"):
        counts["records"] += 1
        counts[f"answer:{row['answer_text']}"] += 1
        counts[f"intent:{row['question_intent']}"] += 1
        if row["qa_id"] in seen_qa_ids:
            failures.append(f"duplicate_qa_id:{row['qa_id']}")
        seen_qa_ids.add(row["qa_id"])
        exact_key = (row["image_id"], normalize_text(row["question"]), normalize_text(row["answer_text"]))
        if exact_key in exact_keys:
            failures.append(f"exact_duplicate:{row['qa_id']}")
        exact_keys.add(exact_key)
        assignment = assignments[row["image_id"]]
        if row["split"] != assignment["split"]:
            failures.append(f"split_mismatch:{row['qa_id']}")
        for key in split_entities:
            split_entities[key][assignment[key]].add(row["split"])
        fact_ids = {fact["fact_id"]: fact for fact in facts_by_image[row["image_id"]]["facts"]}
        if any(fact_id not in fact_ids for fact_id in row["source_fact_ids"]):
            failures.append(f"source_fact_missing:{row['qa_id']}")
        if row["question_intent"] == "abnormality_presence":
            source_facts = [fact_ids[fact_id] for fact_id in row["source_fact_ids"]]
            if row["answer_text"] != "Có." or not source_facts or any(
                fact["fact_kind"] != "observation" or fact["polarity"] != "present"
                or fact["evidence_status"] != "accepted" or not fact["question_eligible"]
                for fact in source_facts
            ):
                failures.append(f"invalid_polarity_conversion:{row['qa_id']}")

    for key, entities in split_entities.items():
        if any(len(splits) > 1 for splits in entities.values()):
            failures.append(f"cross_split_leak:{key}")
    expected = manifest["transformation"]["converted_total"]
    if counts["intent:abnormality_presence"] != expected:
        failures.append("converted_count_mismatch")
    print(json.dumps({"valid": not failures, "failures": failures[:50], "counts": dict(counts)}, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify"))
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--stage3-dir", type=Path, default=STAGE3_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if args.command == "build":
        build(args)
    else:
        verify(args)


if __name__ == "__main__":
    main()
