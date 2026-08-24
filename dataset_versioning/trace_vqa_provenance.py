#!/usr/bin/env python3
"""Trace one released QA back to its source annotation and intermediate artifacts.

This utility is intentionally read-only and uses only the Python standard library.
It prints record positions and paths, so no jq installation is required.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "derived_v4_3_vision_production"
EXPORT = PRODUCTION / "final_release" / "dataset_export"


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                yield line_number, json.loads(line)


def find_jsonl(
    path: Path, predicate: Callable[[dict[str, Any]], bool]
) -> tuple[int | None, dict[str, Any] | None]:
    if not path.is_file():
        return None, None
    for line_number, record in iter_jsonl(path):
        if predicate(record):
            return line_number, record
    return None, None


def find_array(
    path: Path, predicate: Callable[[dict[str, Any]], bool]
) -> tuple[int | None, dict[str, Any] | None]:
    if not path.is_file():
        return None, None
    with path.open(encoding="utf-8") as stream:
        records = json.load(stream)
    for index, record in enumerate(records):
        if predicate(record):
            return index, record
    return None, None


def shown_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def find_value_paths(value: Any, target: str, prefix: str = "$") -> list[str]:
    """Return JSONPath-like locations containing an exact scalar target."""
    paths: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            paths.extend(find_value_paths(child, target, f"{prefix}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(find_value_paths(child, target, f"{prefix}[{index}]"))
    elif value == target:
        paths.append(prefix)
    return paths


def located(path: Path | None, position: int | None, position_name: str) -> dict[str, Any]:
    return {
        "path": shown_path(path),
        "exists": bool(path and path.is_file()),
        position_name: position,
    }


def infer_cohort(master: dict[str, Any]) -> Path | None:
    overlay = master.get("overlay_image")
    if overlay:
        overlay_path = ROOT / overlay
        if overlay_path.parent.name == "overlays":
            return overlay_path.parent.parent
    image_id = master["image_id"]
    cohorts = PRODUCTION / "cohorts"
    for cohort in sorted(cohorts.glob("cohort_*")):
        request_path = cohort / "pilot_requests.jsonl.gz"
        _, request = find_jsonl(request_path, lambda row: row.get("image_id") == image_id)
        if request:
            return cohort
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-id", required=True, help="Released qa_id to trace")
    args = parser.parse_args()

    qa_id = args.qa_id
    master_path = EXPORT / "master_bronchoscopy_vqa.json"
    master_index, master = find_array(master_path, lambda row: row.get("qa_id") == qa_id)
    if master is None:
        raise SystemExit(f"qa_id not found in Master: {qa_id}")

    image_id = master["image_id"]
    source_fact_ids = set(master.get("source_fact_ids", []))
    evidence_region_ids = set(master.get("evidence_region_ids", []))

    canonical_path = ROOT / "canonical_v2_2" / "images.jsonl.gz"
    canonical_line, canonical = find_jsonl(
        canonical_path, lambda row: row.get("image_id") == image_id
    )
    facts_path = ROOT / "derived_v4" / "facts_v1.jsonl.gz"
    facts_line, facts_record = find_jsonl(
        facts_path, lambda row: row.get("image_id") == image_id
    )
    blueprint_path = ROOT / "derived_v4_3" / "qa_blueprints_v1_3.jsonl.gz"
    blueprint_line, blueprint = find_jsonl(
        blueprint_path, lambda row: row.get("qa_id") == qa_id
    )

    facts = (facts_record or {}).get("facts", [])
    selected_facts = [fact for fact in facts if fact.get("fact_id") in source_fact_ids]
    fact_evidence = {
        region_id
        for fact in selected_facts
        for region_id in fact.get("evidence_region_ids", [])
    }

    source_annotation = None
    if canonical:
        relative = canonical.get("provenance", {}).get("source_annotation_path")
        if relative:
            source_annotation = ROOT / relative
    source_image = ROOT / master["image"] if master.get("image") else None

    cohort = infer_cohort(master)
    request_path = cohort / "pilot_requests.jsonl.gz" if cohort else None
    request_line, request = (
        find_jsonl(request_path, lambda row: row.get("image_id") == image_id)
        if request_path
        else (None, None)
    )
    system_prompt = cohort / "system_prompt.txt" if cohort else None
    overlay = cohort / "overlays" / f"{image_id}.jpg" if cohort else None
    result_path = cohort / "results" / f"{image_id}.json" if cohort else None

    accepted_path = None
    accepted_line = None
    accepted = None
    if cohort:
        for name in (
            "accepted_after_quarantine.jsonl.gz",
            "accepted_pilot.jsonl.gz",
            "accepted.jsonl.gz",
        ):
            candidate = cohort / name
            line, record = find_jsonl(candidate, lambda row: row.get("qa_id") == qa_id)
            if record:
                accepted_path, accepted_line, accepted = candidate, line, record
                break

    split = master.get("split")
    clean_path = EXPORT / f"{split}.json" if split in {"train", "val", "test"} else None
    clean_index, clean = (
        find_array(clean_path, lambda row: row.get("qa_id") == qa_id)
        if clean_path
        else (None, None)
    )

    source_annotation_ids = sorted(
        {
            annotation_id
            for fact in selected_facts
            for annotation_id in fact.get("source_annotation_ids", [])
        }
    )
    annotation_jsonpaths: dict[str, list[str]] = {}
    if source_annotation and source_annotation.is_file():
        with source_annotation.open(encoding="utf-8") as stream:
            source_annotation_payload = json.load(stream)
        annotation_jsonpaths = {
            annotation_id: find_value_paths(source_annotation_payload, annotation_id)
            for annotation_id in source_annotation_ids
        }
    report = {
        "query": {"qa_id": qa_id, "image_id": image_id},
        "artifact_trace": {
            "source_image": located(source_image, None, "record"),
            "source_annotation": located(source_annotation, None, "record"),
            "canonical_record": located(canonical_path, canonical_line, "jsonl_line"),
            "facts_record": located(facts_path, facts_line, "jsonl_line"),
            "protected_blueprint": located(blueprint_path, blueprint_line, "jsonl_line"),
            "llm_system_prompt": located(system_prompt, None, "record"),
            "llm_request": located(request_path, request_line, "jsonl_line"),
            "request_overlay": located(overlay, None, "record"),
            "raw_llm_result": located(result_path, None, "record"),
            "accepted_candidate": located(accepted_path, accepted_line, "jsonl_line"),
            "master_record": located(master_path, master_index, "json_array_index_0_based"),
            "clean_split_record": located(clean_path, clean_index, "json_array_index_0_based"),
            "release_manifest": located(EXPORT / "manifest.json", None, "record"),
        },
        "provenance_keys": {
            "source_annotation_ids": source_annotation_ids,
            "source_annotation_jsonpaths": annotation_jsonpaths,
            "source_fact_ids": sorted(source_fact_ids),
            "evidence_region_ids": sorted(evidence_region_ids),
            "protected_sha256": master.get("protected_sha256"),
        },
        "selected_facts": selected_facts,
        "released_fields": {
            "split": split,
            "question_type": master.get("question_intent"),
            "question": master.get("question"),
            "answer_structured": master.get("answer_structured"),
            "answer_text": master.get("answer_text"),
            "evidence_boxes_xyxy": (clean or {}).get("evidence_boxes_xyxy"),
            "validation_status": master.get("validation_status"),
        },
        "cross_checks": {
            "blueprint_found": blueprint is not None,
            "request_found": request is not None,
            "accepted_candidate_found": accepted is not None,
            "clean_record_found": clean is not None,
            "source_facts_found_exactly": {
                fact.get("fact_id") for fact in selected_facts
            }
            == source_fact_ids,
            "evidence_equals_selected_fact_evidence": fact_evidence == evidence_region_ids,
            "blueprint_protected_sha_matches_master": bool(
                blueprint
                and blueprint.get("protected_sha256") == master.get("protected_sha256")
            ),
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
