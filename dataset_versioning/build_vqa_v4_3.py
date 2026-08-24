#!/usr/bin/env python3
"""Build and audit immutable v1.3 VQA blueprints from v1.2.

The migration removes non-pathological visual facts from abnormality QA. Rows
supported only by normal findings are converted to region-grounded normality
QA. Changed semantic contracts receive new QA IDs. No v1.2 artifact is edited.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import build_vqa_v4_2 as v42


ROOT = Path(__file__).resolve().parents[1]
SOURCE_BLUEPRINTS = ROOT / "derived_v4_2" / "qa_blueprints_v1_2.jsonl.gz"
FACTS_PATH = ROOT / "derived_v4" / "facts_v1.jsonl.gz"
SPLITS_PATH = ROOT / "derived_v3" / "split_assignments.jsonl.gz"
OUTPUT_DIR = ROOT / "derived_v4_3"
OUTPUT_BLUEPRINTS = "qa_blueprints_v1_3.jsonl.gz"
DATASET_VERSION = "derived_v4.3"
BLUEPRINT_VERSION = "bronchoscopy_qa_blueprints_v1.3"
PROTECTED_FIELDS = (
    "source_fact_ids", "answer_structured", "answer_text", "options",
    "evidence_region_ids", "question_format", "question_intent",
)
NON_PATHOLOGICAL_VISUAL_CONCEPTS = {
    "image_normal",
    "smooth_mucosa",
    "vocal_cords_normal",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def protected_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row[field] for field in PROTECTED_FIELDS}


def protected_sha(row: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(protected_payload(row)).encode("utf-8")).hexdigest()


def is_non_pathological(fact: dict[str, Any]) -> bool:
    return (
        fact.get("concept") in NON_PATHOLOGICAL_VISUAL_CONCEPTS
        and fact.get("value") is True
        and fact.get("polarity") == "present"
    )


def unique_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return list({fact["fact_id"]: fact for fact in facts}.values())


def terms_for(facts: list[dict[str, Any]]) -> list[str]:
    return list(dict.fromkeys(v42.surface(fact["surface_vi"]) for fact in facts))


def sentence(terms: list[str]) -> str:
    text = v42.join_vi(terms)
    return (text[:1].upper() + text[1:] + ".") if text else ""


def observations(facts: list[dict[str, Any]]) -> dict[str, Any]:
    values = sorted({(fact["concept"], fact["polarity"]) for fact in facts})
    return {"observations": [{"concept": concept, "polarity": polarity} for concept, polarity in values]}


def evidence_union(facts: list[dict[str, Any]]) -> list[str]:
    return sorted({region_id for fact in facts for region_id in fact["evidence_region_ids"]})


def location_text(facts: list[dict[str, Any]]) -> str:
    return v42.location_text(v42.location_key(facts))


def normal_open_contract(facts: list[dict[str, Any]]) -> dict[str, Any]:
    concepts = {fact["concept"] for fact in facts}
    loc = location_text(facts)
    if concepts == {"vocal_cords_normal"}:
        question = "Hình thái dây thanh được ghi nhận như thế nào?"
        intent = "vocal_cords_normality"
        template = "vocal_cords_normality_open_01"
        anchors = [["hình thái", "đặc điểm"], ["dây thanh"]]
    elif concepts == {"smooth_mucosa"}:
        question = f"Đặc điểm bề mặt niêm mạc{loc} được ghi nhận như thế nào?"
        intent = "mucosal_surface"
        template = "smooth_mucosa_open_01"
        anchors = [["bề mặt", "đặc điểm"], ["niêm mạc"]]
    else:
        question = f"Những đặc điểm hình thái bình thường nào được ghi nhận{loc}?"
        intent = "normal_findings"
        template = "normal_findings_open_01"
        anchors = [["bình thường", "không bệnh lý"], ["đặc điểm", "hình thái"]]
    terms = terms_for(facts)
    return {
        "question_format": "open-ended",
        "question_intent": intent,
        "template_id": template,
        "question_draft": question,
        "question_anchor_groups": anchors,
        "answer_structured": observations(facts),
        "answer_text": sentence(terms),
        "answer_text_draft": sentence(terms),
        "reasoning_text_draft": f"Vùng bằng chứng{loc} cho thấy {v42.join_vi(terms)}.",
        "required_surface_terms": terms,
        "options": {},
    }


def normal_closed_contract(facts: list[dict[str, Any]]) -> dict[str, Any]:
    concepts = {fact["concept"] for fact in facts}
    loc = location_text(facts)
    if concepts == {"vocal_cords_normal"}:
        question = "Dây thanh trên hình ảnh nội soi có hình thái bình thường không?"
        intent = "vocal_cords_normality"
        template = "vocal_cords_normality_closed_01"
        anchors = [["dây thanh"], ["bình thường", "hình thái"]]
    elif concepts == {"smooth_mucosa"}:
        question = f"Bề mặt niêm mạc{loc} có nhẵn không?"
        intent = "mucosal_surface"
        template = "smooth_mucosa_closed_01"
        anchors = [["bề mặt", "niêm mạc"], ["nhẵn"]]
    else:
        question = f"Các đặc điểm hình thái bình thường{loc} có được ghi nhận không?"
        intent = "normal_findings"
        template = "normal_findings_closed_01"
        anchors = [["bình thường", "không bệnh lý"]]
    terms = terms_for(facts)
    return {
        "question_format": "closed-ended",
        "question_intent": intent,
        "template_id": template,
        "question_draft": question,
        "question_anchor_groups": anchors,
        "answer_structured": observations(facts),
        "answer_text": "Có.",
        "answer_text_draft": "Có.",
        "reasoning_text_draft": f"Vùng bằng chứng{loc} cho thấy {v42.join_vi(terms)}.",
        "required_surface_terms": terms,
        "options": {},
    }


def migrate_row(row: dict[str, Any], facts_by_id: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], str]:
    old = dict(row)
    selected = unique_facts([facts_by_id[fid] for fid in row["source_fact_ids"]])
    normal = [fact for fact in selected if is_non_pathological(fact)]
    pathological = [fact for fact in selected if not is_non_pathological(fact)]
    affected_intent = row["question_intent"] in {"abnormality", "abnormality_presence"}
    out = dict(row)
    action = "carried_forward_unchanged_contract"

    if affected_intent and normal:
        if pathological:
            out["source_fact_ids"] = sorted(fact["fact_id"] for fact in pathological)
            out["evidence_region_ids"] = evidence_union(pathological)
            out["required_surface_terms"] = terms_for(pathological)
            out["reasoning_text_draft"] = (
                f"Các vùng bằng chứng{location_text(pathological)} cho thấy "
                f"{v42.join_vi(terms_for(pathological))}."
            )
            if row["question_intent"] == "abnormality":
                out["answer_structured"] = observations(pathological)
                out["answer_text"] = sentence(terms_for(pathological))
                out["answer_text_draft"] = out["answer_text"]
                out["canonical_answer"] = canonical_json(out["answer_structured"])
            action = "repaired_remove_normal_facts_from_abnormality"
        else:
            contract = normal_open_contract(normal) if row["question_intent"] == "abnormality" else normal_closed_contract(normal)
            out.update(contract)
            out["canonical_answer"] = canonical_json(out["answer_structured"])
            action = "repaired_convert_abnormality_to_normal_finding"

        out["qa_id"] = v42.stable_id(
            "qa", DATASET_VERSION, old["qa_id"], action,
            canonical_json({field: out[field] for field in PROTECTED_FIELDS}),
        )

    out.update({
        "blueprint_version": BLUEPRINT_VERSION,
        "protected_fields_version": "v1.3",
        "source_blueprint_qa_id": old["qa_id"],
        "source_blueprint_version": row.get("blueprint_version", "bronchoscopy_qa_blueprints_v1.2"),
        "origin_protected_sha256": row["protected_sha256"],
        "migration_status": action,
    })
    out["protected_sha256"] = protected_sha(out)
    return out, action


def load_facts() -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    facts = {}
    images = {}
    for image in read_jsonl(FACTS_PATH):
        for fact in image["facts"]:
            facts[fact["fact_id"]] = fact
            images[fact["fact_id"]] = image["image_id"]
    return facts, images


def build(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite version: {output_dir}")
    output_dir.mkdir(parents=True)
    facts, _ = load_facts()
    rows = []
    counts = Counter()
    migration_rows = []
    for old in read_jsonl(SOURCE_BLUEPRINTS):
        new, action = migrate_row(old, facts)
        rows.append(new)
        counts[action] += 1
        if action != "carried_forward_unchanged_contract":
            migration_rows.append({
                "source_qa_id": old["qa_id"], "new_qa_id": new["qa_id"],
                "image_id": new["image_id"], "split": new["split"], "action": action,
                "old_intent": old["question_intent"], "new_intent": new["question_intent"],
                "old_source_fact_ids": old["source_fact_ids"], "new_source_fact_ids": new["source_fact_ids"],
                "old_answer_structured": old["answer_structured"], "new_answer_structured": new["answer_structured"],
                "old_protected_sha256": old["protected_sha256"], "new_protected_sha256": new["protected_sha256"],
            })

    output_path = output_dir / OUTPUT_BLUEPRINTS
    with gzip.open(output_path, "wt", encoding="utf-8", compresslevel=9) as handle:
        for row in sorted(rows, key=lambda item: (item["split"], item["image_id"], item["qa_id"])):
            handle.write(canonical_json(row) + "\n")
    with gzip.open(output_dir / "blueprint_migrations.jsonl.gz", "wt", encoding="utf-8", compresslevel=9) as handle:
        for row in sorted(migration_rows, key=lambda item: (item["split"], item["image_id"], item["new_qa_id"])):
            handle.write(canonical_json(row) + "\n")

    manifest = {
        "dataset_version": DATASET_VERSION,
        "blueprint_version": BLUEPRINT_VERSION,
        "created_at_utc": now(),
        "source": {
            "blueprints": str(SOURCE_BLUEPRINTS.relative_to(ROOT)),
            "blueprints_sha256": v42.sha256_file(SOURCE_BLUEPRINTS),
            "facts": str(FACTS_PATH.relative_to(ROOT)),
            "facts_sha256": v42.sha256_file(FACTS_PATH),
            "splits": str(SPLITS_PATH.relative_to(ROOT)),
            "splits_sha256": v42.sha256_file(SPLITS_PATH),
        },
        "normal_visual_concepts": sorted(NON_PATHOLOGICAL_VISUAL_CONCEPTS),
        "protected_fields": list(PROTECTED_FIELDS),
        "counts": {"blueprints": len(rows), **dict(sorted(counts.items()))},
        "output": {"path": str(output_path.relative_to(ROOT)), "sha256": v42.sha256_file(output_path)},
        "migrations": {
            "path": str((output_dir / "blueprint_migrations.jsonl.gz").relative_to(ROOT)),
            "rows": len(migration_rows),
            "sha256": v42.sha256_file(output_dir / "blueprint_migrations.jsonl.gz"),
        },
        "status": "BUILT_AWAITING_AUDIT",
    }
    write_json(output_dir / "blueprint_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def audit(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    path = output_dir / OUTPUT_BLUEPRINTS
    rows = list(read_jsonl(path))
    old_rows = {row["qa_id"]: row for row in read_jsonl(SOURCE_BLUEPRINTS)}
    facts, fact_images = load_facts()
    split_map = {row["image_id"]: row["split"] for row in read_jsonl(SPLITS_PATH)}
    failures = Counter()
    qa_ids = Counter(row["qa_id"] for row in rows)
    source_ids = Counter(row.get("source_blueprint_qa_id") for row in rows)
    question_keys = defaultdict(list)
    actions = Counter()

    failures["blueprint_count_changed"] = abs(len(rows) - len(old_rows))
    failures["duplicate_qa_id"] = sum(value - 1 for value in qa_ids.values() if value > 1)
    failures["missing_v1_2_blueprint_coverage"] = len(set(old_rows) - set(source_ids))
    failures["unexpected_v1_2_blueprint_id"] = len(set(source_ids) - set(old_rows))
    failures["v1_2_blueprint_mapped_more_than_once"] = sum(value - 1 for value in source_ids.values() if value > 1)

    for row in rows:
        actions[row.get("migration_status")] += 1
        if row["split"] != split_map.get(row["image_id"]):
            failures["split_differs_from_derived_v3"] += 1
        if row.get("blueprint_version") != BLUEPRINT_VERSION:
            failures["wrong_blueprint_version"] += 1
        if row.get("protected_fields_version") != "v1.3":
            failures["wrong_protected_fields_version"] += 1
        if protected_sha(row) != row.get("protected_sha256"):
            failures["protected_field_seal_mismatch"] += 1
        if any(fid not in facts for fid in row["source_fact_ids"]):
            failures["source_fact_missing"] += 1
            continue
        selected = unique_facts([facts[fid] for fid in row["source_fact_ids"]])
        if any(fact_images[fact["fact_id"]] != row["image_id"] for fact in selected):
            failures["source_fact_wrong_image"] += 1
        if evidence_union(selected) != sorted(row["evidence_region_ids"]):
            failures["evidence_not_exact_fact_union"] += 1
        for failure in v42.fact_semantic_failures(row, facts):
            failures[f"base_semantic:{failure}"] += 1
        has_normal = any(is_non_pathological(fact) for fact in selected)
        if row["question_intent"] in {"abnormality", "abnormality_presence"} and has_normal:
            failures["normal_fact_remaining_in_abnormality_contract"] += 1
        if row["question_intent"] in {"normal_findings", "vocal_cords_normality", "mucosal_surface"}:
            if not selected or any(not is_non_pathological(fact) for fact in selected):
                failures["pathological_fact_in_normal_finding_contract"] += 1
        if row["migration_status"] == "carried_forward_unchanged_contract":
            old = old_rows[row["source_blueprint_qa_id"]]
            for field in v42.PROTECTED_FIELDS + ("question_format", "question_intent"):
                if row[field] != old[field]:
                    failures[f"carried_contract_changed:{field}"] += 1
            if row["qa_id"] != old["qa_id"]:
                failures["carried_qa_id_changed"] += 1
        else:
            if row["qa_id"] == row["source_blueprint_qa_id"]:
                failures["migrated_qa_id_not_changed"] += 1
            if row["protected_sha256"] == row["origin_protected_sha256"]:
                failures["migrated_protected_sha_not_changed"] += 1
        question_keys[(row["image_id"], v42.normalize_text(row["question_draft"]))].append(row["qa_id"])

    failures["image_normalized_question_collision"] = sum(len(ids) for ids in question_keys.values() if len(ids) > 1)
    failures = Counter({key: value for key, value in failures.items() if value})
    report = {
        "dataset_version": DATASET_VERSION,
        "blueprint_version": BLUEPRINT_VERSION,
        "created_at_utc": now(),
        "blueprints": len(rows),
        "migration_counts": dict(sorted(actions.items())),
        "semantic_failure_count": sum(failures.values()),
        "failure_counts": dict(sorted(failures.items())),
        "status": "PASS" if not failures else "FAIL",
        "checks": [
            "one-to-one v1.2 blueprint coverage",
            "derived_v3 split preservation",
            "v1.3 protected SHA-256 seal",
            "source fact, answer and evidence agreement",
            "no non-pathological fact in abnormality QA",
            "no pathological fact in normal-finding QA",
            "changed QA ID and protected seal for migrated contracts",
            "unique (image_id, normalize(question_draft))",
        ],
    }
    write_json(output_dir / "blueprint_audit.json", report)
    release = {
        "dataset_version": DATASET_VERSION,
        "blueprint_version": BLUEPRINT_VERSION,
        "created_at_utc": now(),
        "status": "RELEASED_FOR_LLM_REALIZATION" if not failures else "BLOCKED",
        "semantic_failure_count": sum(failures.values()),
        "blueprints": {
            "path": str(path.relative_to(ROOT)),
            "sha256": v42.sha256_file(path),
            "rows": len(rows),
        },
        "audit": {
            "path": str((output_dir / "blueprint_audit.json").relative_to(ROOT)),
            "sha256": v42.sha256_file(output_dir / "blueprint_audit.json"),
        },
        "migration_table": {
            "path": str((output_dir / "blueprint_migrations.jsonl.gz").relative_to(ROOT)),
            "sha256": v42.sha256_file(output_dir / "blueprint_migrations.jsonl.gz"),
        },
    }
    write_json(output_dir / "release_manifest.json", release)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    for name, func in (("build", build), ("audit", audit)):
        command = sub.add_parser(name)
        command.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
        command.set_defaults(func=func)
    return result


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
