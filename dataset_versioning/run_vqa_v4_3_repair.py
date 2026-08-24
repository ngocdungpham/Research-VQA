#!/usr/bin/env python3
"""Carry forward verified v4.2 records and repair only changed v4.3 QA."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

from PIL import Image, ImageDraw

import build_vqa_v4_2 as v42
import build_vqa_v4_3 as v43
import run_vqa_v4_2_vision_pilot as vision


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "derived_v4_3_vision_repair_001"
MASTER_V42 = ROOT / "derived_v4_2_vision_cumulative_002_quality_filtered_v2" / "dataset_export" / "master_bronchoscopy_vqa.json"
REQUESTS_V42 = (
    ROOT / "derived_v4_2_vision_pilot" / "pilot_requests.jsonl.gz",
    ROOT / "derived_v4_2_vision_cohort_002" / "pilot_requests.jsonl.gz",
)
BLUEPRINTS_V43 = ROOT / "derived_v4_3" / "qa_blueprints_v1_3.jsonl.gz"
FACTS_PATH = ROOT / "derived_v4" / "facts_v1.jsonl.gz"
MODEL = "GenVQAVer2"
BASE_URL = "http://127.0.0.1:20128/v1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as handle:
        for row in rows:
            handle.write(v43.canonical_json(row) + "\n")


def file_sha(path: Path) -> str:
    return v42.sha256_file(path)


def load_facts() -> dict[str, dict[str, Any]]:
    result = {}
    for image in read_jsonl(FACTS_PATH):
        result.update({fact["fact_id"]: fact for fact in image["facts"]})
    return result


def plan_id(qa_id: str) -> str:
    return v42.stable_id("plan", "vision_fact_repair_v1_3", qa_id)


def carried_candidate_failures(row: dict[str, Any], old: dict[str, Any], facts: dict[str, dict[str, Any]]) -> list[str]:
    failures = []
    for field in v42.PROTECTED_FIELDS + ("question_format", "question_intent"):
        if row[field] != old[field]:
            failures.append(f"carried_contract_changed:{field}")
    if v43.protected_sha(row) != row["protected_sha256"]:
        failures.append("v1_3_protected_seal_mismatch")
    failures.extend(v42.fact_semantic_failures(row, facts))
    for field in ("question", "reasoning_text", "visual_evidence_summary"):
        if not isinstance(old.get(field), str) or not old[field].strip():
            failures.append(f"missing_carried_{field}")
    if isinstance(old.get("question"), str) and v42.normalize_text(old["question"]) == v42.normalize_text(row["question_draft"]):
        failures.append("carried_question_not_paraphrased")
    if isinstance(old.get("reasoning_text"), str) and v42.normalize_text(old["reasoning_text"]) == v42.normalize_text(row["reasoning_text_draft"]):
        failures.append("carried_reason_not_paraphrased")
    reason_norm = v42.normalize_text(str(old.get("reasoning_text") or ""))
    for term in row["required_surface_terms"]:
        if v42.normalize_text(term) not in reason_norm:
            failures.append(f"carried_reason_missing_fact:{term}")
    return sorted(set(failures))


def prepare(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite repair version: {output_dir}")
    (output_dir / "results").mkdir(parents=True)
    (output_dir / "overlays").mkdir()
    master = json.loads(MASTER_V42.read_text(encoding="utf-8"))
    blueprints = {row["source_blueprint_qa_id"]: row for row in read_jsonl(BLUEPRINTS_V43)}
    facts = load_facts()
    requests = {}
    for path in REQUESTS_V42:
        for row in read_jsonl(path):
            if row["image_id"] in requests:
                raise ValueError(f"Duplicate source request image: {row['image_id']}")
            requests[row["image_id"]] = row

    carried = []
    repair_source = []
    failures = []
    for old in master:
        blueprint = blueprints[old["qa_id"]]
        can_carry = (
            old["master_record_status"] == "ACCEPTED_EXPORTED"
            and blueprint["migration_status"] == "carried_forward_unchanged_contract"
        )
        if can_carry:
            row_failures = carried_candidate_failures(blueprint, old, facts)
            if row_failures:
                failures.append({"qa_id": old["qa_id"], "failures": row_failures})
                continue
            carried.append({
                **blueprint,
                "question": old["question"],
                "reasoning_text": old["reasoning_text"],
                "visual_evidence_summary": old["visual_evidence_summary"],
                "status": "accepted",
                "review_reasons": [],
                "llm_trace": old["llm_trace"],
                "carry_forward_trace": {
                    "status": "carried_forward_verified",
                    "source_dataset_version": old.get("master_dataset_version"),
                    "source_qa_id": old["qa_id"],
                    "source_protected_sha256": old["protected_sha256"],
                    "verification": [
                        "unchanged v1.2 core protected payload",
                        "v1.3 intent/format-inclusive protected seal",
                        "fact and evidence provenance",
                        "question/reason paraphrase and required terms",
                    ],
                },
            })
        else:
            repair_source.append({"old": old, "blueprint": blueprint})
    if failures:
        write_json(output_dir / "carry_forward_failures.json", failures)
        raise ValueError(f"Carry-forward validation failed for {len(failures)} QA")

    grouped = defaultdict(list)
    for item in repair_source:
        grouped[item["old"]["image_id"]].append(item)
    repair_requests = []
    for image_id, items in sorted(grouped.items()):
        source_request = requests[image_id]
        blueprints_for_image = [item["blueprint"] for item in items]
        selected_fact_ids = sorted({fid for row in blueprints_for_image for fid in row["source_fact_ids"]})
        request_fact_map = {fact["fact_id"]: fact for fact in source_request["facts"]}
        selected_facts = [request_fact_map[fid] for fid in selected_fact_ids]
        selected_region_ids = sorted({rid for row in blueprints_for_image for rid in row["evidence_region_ids"]})
        request_region_map = {region["evidence_region_id"]: region for region in source_request["evidence_regions"]}
        selected_regions = [request_region_map[rid] for rid in selected_region_ids]

        image_path = ROOT / source_request["image_path"]
        with Image.open(image_path) as source_image:
            overlay = source_image.convert("RGB")
            draw = ImageDraw.Draw(overlay)
            for index, region in enumerate(selected_regions, 1):
                box = region["bbox_xyxy_pixels"]
                draw.rectangle(box, outline=(255, 0, 0), width=max(2, round(min(overlay.size) / 160)))
                draw.rectangle([box[0], box[1], box[0] + 28, box[1] + 20], fill=(255, 0, 0))
                draw.text((box[0] + 5, box[1] + 3), str(index), fill=(255, 255, 255))
                region["region_index"] = index
            overlay_path = output_dir / "overlays" / f"{image_id}.jpg"
            overlay.save(overlay_path, format="JPEG", quality=92)

        old_by_source = {item["blueprint"]["source_blueprint_qa_id"]: item["old"] for item in items}
        plans = []
        for blueprint in sorted(blueprints_for_image, key=lambda row: row["qa_id"]):
            old = old_by_source[blueprint["source_blueprint_qa_id"]]
            selected = [facts[fid] for fid in blueprint["source_fact_ids"]]
            plans.append({
                "qa_plan_id": plan_id(blueprint["qa_id"]),
                "qa_id": blueprint["qa_id"],
                "protected_sha256": blueprint["protected_sha256"],
                "question_format": blueprint["question_format"],
                "question_intent": blueprint["question_intent"],
                "question_draft": blueprint["question_draft"],
                "reasoning_text_draft": blueprint["reasoning_text_draft"],
                "answer_structured_locked": blueprint["answer_structured"],
                "answer_text_locked": blueprint["answer_text"],
                "options_locked": blueprint["options"],
                "source_fact_ids": blueprint["source_fact_ids"],
                "evidence_region_ids": blueprint["evidence_region_ids"],
                "required_question_anchors": blueprint.get("question_anchor_groups") or [],
                "required_reason_terms": blueprint["required_surface_terms"],
                "disease_policy": vision.disease_policy(selected),
                "repair_trace": {
                    "source_qa_id": old["qa_id"],
                    "source_master_status": old["master_record_status"],
                    "source_review_reasons": old.get("review_reasons") or [],
                    "migration_status": blueprint["migration_status"],
                },
            })
        repair_requests.append({
            "pilot_version": "vision_fact_grounded_repair_v1_3",
            "image_id": image_id,
            "image_path": source_request["image_path"],
            "overlay_path": str(overlay_path.relative_to(ROOT)),
            "image_metadata": source_request["image_metadata"],
            "split": source_request["split"],
            "policy": source_request["policy"],
            "evidence_regions": selected_regions,
            "facts": selected_facts,
            "qa_plans": plans,
        })

    write_jsonl_gz(output_dir / "carried_forward_v1_3.jsonl.gz", sorted(carried, key=lambda row: (row["split"], row["image_id"], row["qa_id"])))
    write_jsonl_gz(output_dir / "pilot_requests.jsonl.gz", repair_requests)
    manifest = {
        "repair_version": "vision_fact_grounded_repair_v1_3",
        "created_at_utc": now(),
        "processed_v1_2_records": len(master),
        "carried_forward_verified": len(carried),
        "repair_plans": sum(len(row["qa_plans"]) for row in repair_requests),
        "repair_images": len(repair_requests),
        "repair_reason_counts": dict(Counter(
            item["blueprint"]["migration_status"] if item["blueprint"]["migration_status"] != "carried_forward_unchanged_contract" else "unsafe_or_prior_review"
            for item in repair_source
        )),
        "source_checksums": {
            "v1_3_blueprints": file_sha(BLUEPRINTS_V43),
            "v1_2_master": file_sha(MASTER_V42),
            "facts": file_sha(FACTS_PATH),
            **{str(path.relative_to(ROOT)): file_sha(path) for path in REQUESTS_V42},
        },
        "outputs": {
            "carried_forward": {"path": "carried_forward_v1_3.jsonl.gz", "sha256": file_sha(output_dir / "carried_forward_v1_3.jsonl.gz")},
            "repair_requests": {"path": "pilot_requests.jsonl.gz", "sha256": file_sha(output_dir / "pilot_requests.jsonl.gz")},
        },
        "status": "READY_FOR_TARGETED_LLM_REPAIR",
    }
    write_json(output_dir / "repair_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def configure_v1_3_validator() -> None:
    vision.BLUEPRINTS = BLUEPRINTS_V43
    v42.protected_sha = v43.protected_sha


def run(args: argparse.Namespace) -> None:
    configure_v1_3_validator()
    vision.run(args)


def status(args: argparse.Namespace) -> None:
    vision.status(args)


def candidate_failures(blueprint: dict[str, Any], item: dict[str, Any], facts: dict[str, dict[str, Any]]) -> list[str]:
    configure_v1_3_validator()
    candidate = {
        "protected_sha256": blueprint["protected_sha256"],
        "question": item.get("question"),
        "reasoning_text": item.get("reasoning_text"),
    }
    return v42.candidate_failures(blueprint, candidate, facts)


def audit(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    manifest = json.loads((output_dir / "repair_manifest.json").read_text(encoding="utf-8"))
    requests = list(read_jsonl(output_dir / "pilot_requests.jsonl.gz"))
    request_by_image = {row["image_id"]: row for row in requests}
    blueprints = {row["qa_id"]: row for row in read_jsonl(BLUEPRINTS_V43)}
    facts = load_facts()
    carried = list(read_jsonl(output_dir / "carried_forward_v1_3.jsonl.gz"))
    result_paths = sorted((output_dir / "results").glob("*.json"))
    accepted_repair = []
    review = []
    failures = Counter()
    failures["repair_images_incomplete"] = len(requests) - len(result_paths)
    failures["repair_request_checksum_changed"] = int(file_sha(output_dir / "pilot_requests.jsonl.gz") != manifest["outputs"]["repair_requests"]["sha256"])
    failures["carried_checksum_changed"] = int(file_sha(output_dir / "carried_forward_v1_3.jsonl.gz") != manifest["outputs"]["carried_forward"]["sha256"])
    question_keys = defaultdict(list)
    response_ids = Counter()
    token_usage = Counter()

    for row in carried:
        question_keys[(row["image_id"], v42.normalize_text(row["question"]))].append(row["qa_id"])
    for path in result_paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        request = request_by_image[result["image_id"]]
        plans = {plan["qa_plan_id"]: plan for plan in request["qa_plans"]}
        trace = result.get("llm_trace") or {}
        response_ids[trace.get("response_id")] += 1
        token_usage.update(trace.get("usage") or {})
        if len(result["items"]) != len(plans):
            failures["result_plan_count_mismatch"] += 1
        for item in result["items"]:
            plan = plans[item["qa_plan_id"]]
            blueprint = blueprints[plan["qa_id"]]
            for field, plan_field in (
                ("question_format", "question_format"), ("question_intent", "question_intent"),
                ("answer_structured", "answer_structured_locked"), ("answer_text", "answer_text_locked"),
                ("options", "options_locked"), ("source_fact_ids", "source_fact_ids"),
                ("evidence_region_ids", "evidence_region_ids"), ("protected_sha256", "protected_sha256"),
            ):
                if blueprint[field] != plan[plan_field]:
                    failures[f"repair_plan_changed_protected_field:{field}"] += 1
            materialized = {
                **blueprint,
                **item,
                "llm_trace": trace,
                "repair_image_id": result["image_id"],
                "repair_trace": plan.get("repair_trace") or {},
            }
            if item["status"] == "accepted":
                for failure in candidate_failures(blueprint, item, facts):
                    failures[f"accepted_semantic:{failure}"] += 1
                visual = item.get("visual_evidence_summary")
                if not isinstance(visual, str) or not visual.strip():
                    failures["accepted_missing_visual_evidence_summary"] += 1
                accepted_repair.append(materialized)
                question_keys[(blueprint["image_id"], v42.normalize_text(item["question"]))].append(blueprint["qa_id"])
            else:
                review.append(materialized)

    failures["duplicate_response_id"] = sum(value - 1 for key, value in response_ids.items() if key and value > 1)
    failures["image_normalized_question_collision"] = sum(len(ids) for ids in question_keys.values() if len(ids) > 1)
    expected_qa_ids = {
        row["qa_id"] for row in carried
    } | {
        plan["qa_id"] for request in requests for plan in request["qa_plans"]
    }
    actual_qa_ids = {row["qa_id"] for row in carried + accepted_repair + review}
    failures["processed_record_coverage_missing"] = len(expected_qa_ids - actual_qa_ids)
    failures["processed_record_coverage_unexpected"] = len(actual_qa_ids - expected_qa_ids)
    failures["processed_record_count_not_920"] = abs(len(actual_qa_ids) - 920)
    failures = Counter({key: value for key, value in failures.items() if value})

    accepted = sorted(carried + accepted_repair, key=lambda row: (row["split"], row["image_id"], row["qa_id"]))
    review = sorted(review, key=lambda row: (row["split"], row["image_id"], row["qa_id"]))
    write_jsonl_gz(output_dir / "accepted_v4_3.jsonl.gz", accepted)
    write_jsonl_gz(output_dir / "review_v4_3.jsonl.gz", review)
    source_requests = sorted(
        [row for path in REQUESTS_V42 for row in read_jsonl(path)],
        key=lambda row: row["image_id"],
    )
    if len(source_requests) != 200 or len({row["image_id"] for row in source_requests}) != 200:
        failures["source_request_index_not_200_unique_images"] += 1
    write_jsonl_gz(output_dir / "source_requests_200.jsonl.gz", source_requests)
    report = {
        "repair_version": "vision_fact_grounded_repair_v1_3",
        "created_at_utc": now(),
        "gate_status": "STOP_AUTOMATED_THRESHOLDS_FAILED" if failures else ("STOP_MANUAL_REVIEW_REQUIRED" if review else "PASS_READY_FOR_EXPORT"),
        "automatic_thresholds_passed": not failures,
        "processed_records": len(actual_qa_ids),
        "carried_forward_verified": len(carried),
        "repair_accepted": len(accepted_repair),
        "repair_review_required": len(review),
        "accepted_total": len(accepted),
        "failure_counts": dict(sorted(failures.items())),
        "response_models": dict(Counter((json.loads(path.read_text(encoding="utf-8")).get("llm_trace") or {}).get("response_model") for path in result_paths)),
        "token_usage_repair": dict(token_usage),
        "outputs": {
            "accepted": {"path": "accepted_v4_3.jsonl.gz", "sha256": file_sha(output_dir / "accepted_v4_3.jsonl.gz")},
            "review": {"path": "review_v4_3.jsonl.gz", "sha256": file_sha(output_dir / "review_v4_3.jsonl.gz")},
            "source_requests": {"path": "source_requests_200.jsonl.gz", "sha256": file_sha(output_dir / "source_requests_200.jsonl.gz")},
        },
    }
    write_json(output_dir / "repair_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    command = sub.add_parser("prepare")
    command.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    command.set_defaults(func=prepare)
    for name, func in (("run", run), ("status", status), ("audit", audit)):
        command = sub.add_parser(name)
        command.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
        if name == "run":
            command.add_argument("--base-url", default=BASE_URL)
            command.add_argument("--model", default=MODEL)
            command.add_argument("--api-key", default=os.environ.get("VQA_LLM_API_KEY", "local-vqa-v4-3"))
            command.add_argument("--temperature", type=float, default=0.25)
            command.add_argument("--timeout", type=int, default=300)
            command.add_argument("--retries", type=int, default=2)
            command.add_argument("--limit-images", type=int, default=0)
        command.set_defaults(func=func)
    return result


def main() -> None:
    args = parser().parse_args()
    if hasattr(args, "output_dir"):
        args.output_dir = args.output_dir.resolve()
    args.func(args)


if __name__ == "__main__":
    main()
