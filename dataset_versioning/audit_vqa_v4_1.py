#!/usr/bin/env python3
"""Independent integrity audit for the fact-grounded VQA v4.1 release.

This audit follows each QA through source facts, derived evidence regions,
canonical labels/source annotations, polygons, split assignments, and the
model-facing LLaVA export. It also checks hard contradictions, duplicates,
and patient/procedure/image/near-duplicate leakage.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]

OBSERVATION_PATHS = {
    "mucosal_infiltration": ("mucosal_findings", "infiltration"),
    "mucosal_erythema": ("mucosal_findings", "erythema"),
    "carinal_edema": ("mucosal_findings", "carinal_edema"),
    "anthracotic_pigmentation": ("mucosal_findings", "anthracotic_pigmentation"),
    "pseudomembrane": ("mucosal_findings", "pseudomembrane"),
    "mucosal_ulceration": ("mucosal_findings", "ulceration"),
    "smooth_mucosa": ("mucosal_findings", "smooth_mucosa"),
    "mucosal_atrophy": ("mucosal_findings", "mucosal_atrophy"),
    "visible_orifice": ("mucosal_findings", "visible_orifice"),
    "other_mucosal_finding": ("mucosal_findings", "other"),
    "hypervascularity": ("vascular_findings", "hypervascularity"),
    "tracheomalacia": ("airway_wall_findings", "tracheomalacia"),
    "stenosis": ("stenosis", "presence"),
    "bronchial_tumor": ("tumor", "presence"),
    "secretion": ("secretion", "presence"),
    "vocal_cords_normal": ("vocal_cords", "normal"),
    "vocal_cord_paralysis": ("vocal_cords", "paralysis"),
}

ATTRIBUTE_PATHS = {
    "stenosis_severity": ("stenosis", "severity"),
    "stenosis_cause": ("stenosis", "cause"),
    "tumor_morphology": ("tumor", "morphology"),
    "secretion_type": ("secretion", "type"),
    "secretion_color": ("secretion", "color"),
    "secretion_consistency": ("secretion", "consistency"),
}

SEVERITY_OPTIONS = {
    "0_25_percent": ("A", "Hẹp 0–25%"),
    "26_50_percent": ("B", "Hẹp 26–50%"),
    "51_75_percent": ("C", "Hẹp 51–75%"),
    "76_90_percent": ("D", "Hẹp 76–90%"),
    "over_90_percent": ("E", "Hẹp trên 90%"),
}

NON_PATHOLOGICAL_PRESENT_OBSERVATIONS = {
    "vocal_cords_normal",
    "smooth_mucosa",
    "visible_orifice",
}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).lower().replace("–", "-").replace("—", "-")
    return " ".join("".join(char if char.isalnum() or char.isspace() else " " for char in text).split())


def polygon_bbox(points: list[list[float]]) -> list[float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def polygon_area(points: list[list[float]]) -> float:
    if len(points) < 3:
        return 0.0
    return abs(sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * points[index][1]
        for index in range(len(points))
    )) / 2.0


def format_vi(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def nested(labels: dict[str, Any], path: tuple[str, str]) -> Any:
    return labels.get(path[0], {}).get(path[1])


def expected_polarity(value: Any) -> str:
    return "present" if value is True else "absent"


def fact_matches_label(fact: dict[str, Any], region: dict[str, Any] | None, image_normal: Any) -> bool:
    if fact["concept"] == "image_normal":
        return (
            fact["fact_kind"] == "image_state"
            and fact["value"] is image_normal
            and fact["polarity"] == expected_polarity(image_normal)
        )
    if region is None:
        return False
    labels = region["labels"]
    if fact["fact_kind"] == "anatomy":
        return (
            fact["concept"] in labels.get("anatomy", [])
            and fact["value"] == fact["concept"]
            and fact["polarity"] == "present"
        )
    if fact["fact_kind"] == "observation":
        if fact["concept"] in OBSERVATION_PATHS:
            value = nested(labels, OBSERVATION_PATHS[fact["concept"]])
            return value is not None and fact["value"] is value and fact["polarity"] == expected_polarity(value)
        return (
            fact["concept"] in labels.get("other_findings", [])
            and fact["value"] == fact["concept"]
            and fact["polarity"] == "present"
        )
    if fact["fact_kind"] == "attribute" and fact["concept"] in ATTRIBUTE_PATHS:
        value = nested(labels, ATTRIBUTE_PATHS[fact["concept"]])
        return value is not None and fact["value"] == value and fact["polarity"] == "present"
    return False


def qa_semantic_failures(row: dict[str, Any], facts: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    template = row["template_id"]
    structured = row["answer_structured"]
    if row["canonical_answer"] != canonical_json(structured):
        failures.append("canonical_answer_mismatch")
    if row["answer_text"] != row["answer_text_draft"]:
        failures.append("answer_changed_from_draft")
    for term in row["required_surface_terms"]:
        if normalize_text(term) not in normalize_text(row["reasoning_text"]):
            failures.append("reason_missing_required_term")
            break

    if template == "abnormality_open_01":
        concepts = sorted({fact["concept"] for fact in facts})
        expected = sorted(item["concept"] for item in structured.get("observations", []))
        if concepts != expected or any(fact["fact_kind"] != "observation" or fact["polarity"] != "present" for fact in facts):
            failures.append("invalid_abnormality_answer")
    elif template == "explicit_negative_closed_01":
        if (
            row["answer_text"] != "Không."
            or structured.get("presence") is not False
            or not facts
            or any(fact["fact_kind"] != "observation" or fact["polarity"] != "absent" for fact in facts)
            or any(fact["concept"] != structured.get("concept") for fact in facts)
        ):
            failures.append("invalid_explicit_negative")
    elif template == "attribute_open_01":
        if (
            not facts
            or any(fact["fact_kind"] != "attribute" or fact["polarity"] != "present" for fact in facts)
            or any(fact["concept"] != structured.get("concept") or fact["value"] != structured.get("value") for fact in facts)
        ):
            failures.append("invalid_attribute_answer")
    elif template == "stenosis_severity_single_01":
        value = structured.get("value")
        option = SEVERITY_OPTIONS.get(value)
        if (
            option is None
            or structured.get("concept") != "stenosis_severity"
            or structured.get("option") != option[0]
            or row["answer_text"] != f"{option[0]}. {option[1]}."
            or any(fact["concept"] != "stenosis_severity" or fact["value"] != value for fact in facts)
        ):
            failures.append("invalid_stenosis_choice")
    elif template == "normal_closed_01":
        expected = structured.get("normal")
        if (
            len(facts) != 1
            or facts[0]["concept"] != "image_normal"
            or facts[0]["value"] is not expected
            or row["answer_text"] != ("Có." if expected else "Không.")
        ):
            failures.append("invalid_normality_answer")
    elif template == "abnormality_presence_closed_01":
        if (
            structured.get("abnormality_present") is not True
            or row["answer_text"] != "Có."
            or not facts
            or any(fact["fact_kind"] != "observation" or fact["polarity"] != "present" for fact in facts)
        ):
            failures.append("invalid_rebalanced_abnormality")
    else:
        failures.append("unknown_template")
    return failures


def load_model_export(path: Path) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    if not path.is_dir():
        return by_id
    for split in ("train", "validation", "test"):
        source = path / f"{split}.json"
        if not source.is_file():
            continue
        for row in json.loads(source.read_text(encoding="utf-8")):
            by_id[row["qa_id"]] = row
    return by_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=ROOT / "canonical_v2_2" / "images.jsonl.gz")
    parser.add_argument("--derivations", type=Path, default=ROOT / "derived_v2" / "region_derivations.jsonl.gz")
    parser.add_argument("--assignments", type=Path, default=ROOT / "derived_v3" / "split_assignments.jsonl.gz")
    parser.add_argument("--hashes", type=Path, default=ROOT / "derived_v3" / "perceptual_hashes.jsonl.gz")
    parser.add_argument("--edges", type=Path, default=ROOT / "derived_v3" / "leakage_group_edges.jsonl.gz")
    parser.add_argument("--groups", type=Path, default=ROOT / "derived_v3" / "leakage_groups.jsonl.gz")
    parser.add_argument("--facts", type=Path, default=ROOT / "derived_v4" / "facts_v1.jsonl.gz")
    parser.add_argument("--vqa", type=Path, default=ROOT / "derived_v4_1" / "accepted_vqa.jsonl.gz")
    parser.add_argument("--model-export", type=Path, default=ROOT / "model_data" / "derived_v4_1_llava")
    parser.add_argument("--json-report", type=Path, default=ROOT / "reports" / "vqa_28k_integrity_audit.json")
    parser.add_argument("--md-report", type=Path, default=ROOT / "reports" / "VQA_28K_INTEGRITY_AUDIT.md")
    args = parser.parse_args()

    canonical: dict[str, dict[str, Any]] = {}
    for image in read_jsonl(args.canonical):
        canonical[image["image_id"]] = {
            **image,
            "regions_by_id": {region["region_id"]: region for region in image["regions"]},
        }

    evidence: dict[tuple[str, str], dict[str, Any]] = {}
    for derivation in read_jsonl(args.derivations):
        image = canonical[derivation["image_id"]]
        for region in derivation["regions"]:
            source_regions = [image["regions_by_id"][source_id] for source_id in region["source_region_ids"]]
            geometry_source = image["regions_by_id"][region["training_geometry_source_region_id"]]
            evidence[(derivation["image_id"], region["canonical_region_id"])] = {
                "labels": region["labels_override"] or source_regions[0]["labels"],
                "polygon": region["training_polygon_override"] or geometry_source["polygon"],
                "source_annotation_ids": sorted({
                    annotation_id
                    for source_region in source_regions
                    for annotation_id in source_region["source_annotation_ids"]
                }),
                "training_status": region["training_status"],
            }

    assignments = {row["image_id"]: row for row in read_jsonl(args.assignments)}
    fact_records = {row["image_id"]: row for row in read_jsonl(args.facts)}
    model_export = load_model_export(args.model_export)
    counters: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    examples: defaultdict[str, list[str]] = defaultdict(list)

    def fail(kind: str, identifier: str) -> None:
        failures[kind] += 1
        if len(examples[kind]) < 10:
            examples[kind].append(identifier)

    # Independently check that every fact matches canonical/derived labels and provenance.
    fact_index: dict[str, dict[str, Any]] = {}
    for image_id, record in fact_records.items():
        image = canonical.get(image_id)
        assignment = assignments.get(image_id)
        if image is None or assignment is None:
            fail("fact_image_or_assignment_missing", image_id)
            continue
        if (record["patient_id"], record["procedure_id"], record["split"]) != (
            image["patient_id"], image["procedure_id"], assignment["split"]
        ):
            fail("fact_identity_or_split_mismatch", image_id)
        normal = image["image_labels"]["normal"]
        for fact in record["facts"]:
            counters["facts"] += 1
            if fact["fact_id"] in fact_index:
                fail("duplicate_fact_id", fact["fact_id"])
            fact_index[fact["fact_id"]] = fact
            region = evidence.get((image_id, fact["subject_region_id"])) if fact["subject_region_id"] else None
            if not fact_matches_label(fact, region, normal):
                fail("fact_label_mismatch", fact["fact_id"])
            if fact["scope"] == "region":
                if region is None:
                    fail("fact_region_missing", fact["fact_id"])
                else:
                    if fact["source_annotation_ids"] != region["source_annotation_ids"]:
                        fail("fact_source_annotation_mismatch", fact["fact_id"])
                    if fact["evidence_region_ids"] != [fact["subject_region_id"]]:
                        fail("fact_evidence_mismatch", fact["fact_id"])
            if fact["question_eligible"] and fact["evidence_status"] != "accepted":
                fail("eligible_fact_has_unaccepted_evidence", fact["fact_id"])

    rows = list(read_jsonl(args.vqa))
    seen_ids: set[str] = set()
    exact_seen: set[tuple[str, str, str]] = set()
    question_answers: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    question_rows: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    fact_answers: defaultdict[tuple[str, str, tuple[str, ...]], set[str]] = defaultdict(set)
    entity_splits = {key: defaultdict(set) for key in ("patient_id", "procedure_id", "leakage_group_id", "image_id")}
    image_splits: dict[str, str] = {}

    for row in rows:
        qa_id = row["qa_id"]
        counters["qa"] += 1
        counters[f"qa_split:{row['split']}"] += 1
        counters[f"qa_format:{row['question_format']}"] += 1
        counters["question_equals_draft"] += row["question"] == row["question_draft"]
        counters["answer_equals_draft"] += row["answer_text"] == row["answer_text_draft"]
        counters["reason_equals_draft"] += row["reasoning_text"] == row["reasoning_text_draft"]
        if qa_id in seen_ids:
            fail("duplicate_qa_id", qa_id)
        seen_ids.add(qa_id)
        exact_key = (row["image_id"], normalize_text(row["question"]), normalize_text(row["answer_text"]))
        if exact_key in exact_seen:
            fail("exact_qa_duplicate", qa_id)
        exact_seen.add(exact_key)

        assignment = assignments.get(row["image_id"])
        record = fact_records.get(row["image_id"])
        if assignment is None or record is None:
            fail("qa_image_assignment_or_facts_missing", qa_id)
            continue
        expected_identity = (assignment["patient_id"], assignment["procedure_id"], assignment["split"])
        if (row["patient_id"], row["procedure_id"], row["split"]) != expected_identity:
            fail("qa_identity_or_split_mismatch", qa_id)
        image_splits[row["image_id"]] = row["split"]
        for key in entity_splits:
            value = assignment[key] if key in assignment else row[key]
            entity_splits[key][value].add(row["split"])

        facts_by_id = {fact["fact_id"]: fact for fact in record["facts"]}
        source_facts = []
        for fact_id in row["source_fact_ids"]:
            fact = facts_by_id.get(fact_id)
            if fact is None:
                fail("qa_source_fact_missing", qa_id)
            else:
                source_facts.append(fact)
        if not source_facts:
            fail("qa_without_source_facts", qa_id)
        expected_evidence = sorted({region_id for fact in source_facts for region_id in fact["evidence_region_ids"]})
        if row["evidence_region_ids"] != expected_evidence:
            fail("qa_evidence_not_equal_fact_evidence", qa_id)
        for region_id in row["evidence_region_ids"]:
            region = evidence.get((row["image_id"], region_id))
            if region is None:
                fail("qa_evidence_region_missing", qa_id)
                continue
            if region["training_status"] != "accepted":
                fail("qa_evidence_region_not_accepted", qa_id)
            polygon = region["polygon"]
            if (
                len({tuple(point) for point in polygon}) < 3
                or polygon_area(polygon) <= 0
                or any(not math.isfinite(value) for point in polygon for value in point)
            ):
                fail("qa_evidence_polygon_invalid", qa_id)

        semantic = qa_semantic_failures(row, source_facts)
        for kind in semantic:
            fail(kind, qa_id)
        question_answers[(row["image_id"], normalize_text(row["question"]))].add(row["canonical_answer"])
        question_rows[(row["image_id"], normalize_text(row["question"]))].append(row)
        fact_answers[(row["image_id"], row["question_intent"], tuple(row["source_fact_ids"]))].add(row["canonical_answer"])

        export = model_export.get(qa_id)
        if model_export:
            if export is None:
                fail("qa_missing_from_model_export", qa_id)
            else:
                target = export["conversations"][1]["value"]
                match = re.search(r"<location>\s*(\[.*\])\s*$", target)
                expected_boxes = [
                    [round(value) for value in polygon_bbox(evidence[(row["image_id"], region_id)]["polygon"])]
                    for region_id in row["evidence_region_ids"]
                ]
                if not match or json.loads(match.group(1)) != expected_boxes:
                    fail("model_export_bbox_mismatch", qa_id)

    counters["unique_normalized_question_strings"] = len({normalize_text(row["question"]) for row in rows})
    counters["qa_with_evidence"] = sum(bool(row["evidence_region_ids"]) for row in rows)
    counters["qa_without_evidence"] = len(rows) - counters["qa_with_evidence"]
    for key, answers in question_answers.items():
        if len(answers) > 1:
            rows_in_conflict = question_rows[key]
            fail("same_image_question_answer_conflict", key[0])
            counters["qa_records_in_question_answer_conflicts"] += len(rows_in_conflict)
            intent_key = "+".join(sorted({row["question_intent"] for row in rows_in_conflict}))
            counters[f"question_answer_conflict_intents:{intent_key}"] += 1
    for key, answers in fact_answers.items():
        if len(answers) > 1:
            fail("same_fact_signature_contradiction", key[0])

    # Hard fact contradictions are scoped to the same region and concept.
    region_concepts: defaultdict[tuple[str, str | None, str], set[str]] = defaultdict(set)
    for image_id, record in fact_records.items():
        positive_abnormal = False
        normal_true = False
        by_id = {fact["fact_id"]: fact for fact in record["facts"]}
        for fact in record["facts"]:
            region_concepts[(image_id, fact["subject_region_id"], fact["concept"])].add(fact["polarity"])
            positive_abnormal |= (
                fact["fact_kind"] == "observation"
                and fact["polarity"] == "present"
                and fact["question_eligible"]
                and fact["concept"] not in NON_PATHOLOGICAL_PRESENT_OBSERVATIONS
            )
            normal_true |= fact["concept"] == "image_normal" and fact["value"] is True and fact["question_eligible"]
            if fact["fact_kind"] == "attribute" and fact["parent_fact_id"]:
                parent = by_id.get(fact["parent_fact_id"])
                if parent is None or parent["polarity"] != "present":
                    fail("attribute_parent_contradiction", fact["fact_id"])
        if normal_true and positive_abnormal:
            fail("normal_positive_observation_contradiction", image_id)
    for key, polarities in region_concepts.items():
        if "present" in polarities and "absent" in polarities:
            fail("same_region_concept_polarity_contradiction", key[0])

    # Leakage checks over QA-bearing entities and source image hashes.
    for key, values in entity_splits.items():
        counters[f"unique_{key}"] = len(values)
        for entity, splits in values.items():
            if len(splits) > 1:
                fail(f"cross_split_{key}_leakage", entity)

    sha_splits: defaultdict[str, set[str]] = defaultdict(set)
    for row in read_jsonl(args.hashes):
        if row["image_id"] in assignments and row.get("image_sha256"):
            sha_splits[row["image_sha256"]].add(assignments[row["image_id"]]["split"])
    for sha, splits in sha_splits.items():
        if len(splits) > 1:
            fail("cross_split_exact_image_sha_leakage", sha)

    for edge in read_jsonl(args.edges):
        splits = {assignments[image_id]["split"] for image_id in edge["image_ids"]}
        if len(splits) > 1:
            fail("cross_split_duplicate_edge_leakage", "|".join(edge["image_ids"]))
    group_splits: defaultdict[str, set[str]] = defaultdict(set)
    for assignment in assignments.values():
        group_splits[assignment["leakage_group_id"]].add(assignment["split"])
    for group in read_jsonl(args.groups):
        assigned = group_splits[group["leakage_group_id"]]
        if assigned != {group["split"]}:
            fail("leakage_group_assignment_mismatch", group["leakage_group_id"])

    status = "pass" if not failures else "fail"
    report = {
        "audit_version": "vqa_v4_1_independent_integrity_audit_v1",
        "status": status,
        "scope": [
            "QA-to-fact semantic agreement",
            "fact-to-canonical/derived label and source-annotation agreement",
            "evidence-region and polygon integrity",
            "model-export bbox agreement",
            "exact/fact contradictions and duplicates",
            "patient/procedure/image/exact-near-duplicate split leakage",
        ],
        "counts": dict(sorted(counters.items())),
        "failures": dict(sorted(failures.items())),
        "failure_examples": dict(sorted(examples.items())),
        "interpretation": {
            "automated_integrity": "Passed" if not failures else "Failed; inspect failure categories",
            "clinical_validity": "Not established by this audit; requires independent physician review.",
            "dataset_difficulty": "Not established by integrity checks; requires controlled baselines and ablations.",
            "training_utility": "Not established by integrity checks; requires zero-shot versus fine-tuned and transfer experiments.",
        },
    }
    args.json_report.parent.mkdir(parents=True, exist_ok=True)
    args.json_report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md = f"""# Audit tính toàn vẹn Bronchoscopy-VQA-28K (`derived_v4.1`)

**Kết quả tự động:** `{status.upper()}`  
**Số QA:** {format_vi(counters['qa'])}  
**Số fact:** {format_vi(counters['facts'])}  
**QA có region evidence:** {format_vi(counters['qa_with_evidence'])}  
**QA không có region evidence:** {format_vi(counters['qa_without_evidence'])}

## Kết quả kiểm tra

| Hạng mục | Kết quả |
|---|---:|
| QA ID/exact duplicate | {failures['duplicate_qa_id'] + failures['exact_qa_duplicate']} lỗi |
| QA–fact–answer/reason inconsistency | {sum(value for key, value in failures.items() if key.startswith('invalid_') or key in {'canonical_answer_mismatch', 'reason_missing_required_term', 'answer_changed_from_draft'})} lỗi |
| Fact–canonical label mismatch | {failures['fact_label_mismatch']} lỗi |
| Fact–source annotation mismatch | {failures['fact_source_annotation_mismatch']} lỗi |
| QA–evidence region/polygon mismatch | {sum(value for key, value in failures.items() if 'evidence' in key or 'polygon' in key)} lỗi |
| Bounding box model export mismatch | {failures['model_export_bbox_mismatch']} lỗi |
| Cùng ảnh + cùng câu hỏi nhưng khác đáp án tham chiếu | {failures['same_image_question_answer_conflict']} khóa câu hỏi |
| Hard label/fact contradiction | {sum(value for key, value in failures.items() if 'contradiction' in key)} lỗi |
| Patient/procedure/image/leakage-group leakage | {sum(value for key, value in failures.items() if 'leakage' in key)} lỗi |

## Kiểm tra vai trò của LLM

- `question == question_draft`: {format_vi(counters['question_equals_draft'])}/{format_vi(counters['qa'])} QA.
- `answer_text == answer_text_draft`: {format_vi(counters['answer_equals_draft'])}/{format_vi(counters['qa'])} QA.
- `reasoning_text == reasoning_text_draft`: {format_vi(counters['reason_equals_draft'])}/{format_vi(counters['qa'])} QA.
- Số chuỗi câu hỏi khác nhau sau chuẩn hóa: {format_vi(counters['unique_normalized_question_strings'])}.

Các số trên xác nhận release hiện tại được materialize trực tiếp từ blueprint cố định. LLM chỉ audit các họ template đại diện; output diễn đạt của LLM không được đưa vào 28.026 QA cuối.

## Xung đột câu hỏi--đáp án cần sửa

- Số khóa `(image_id, normalized question)` có nhiều đáp án: {failures['same_image_question_answer_conflict']}.
- Số QA bị ảnh hưởng: {format_vi(counters['qa_records_in_question_answer_conflicts'])}.
- `color` so với `type`: {counters['question_answer_conflict_intents:color+type']} khóa.
- `consistency` so với `type`: {counters['question_answer_conflict_intents:consistency+type']} khóa.
- Nhiều giá trị cùng intent `type`: {counters['question_answer_conflict_intents:type']} khóa.

Nguyên nhân là template chung “Đặc điểm được xác nhận của dịch tiết bất thường ... là gì?” được dùng đồng thời cho loại dịch, màu và độ đặc. Với cùng một ảnh, câu hỏi giống hệt nhau nhưng reference answer khác nhau; đây là ambiguity ở thiết kế câu hỏi, không phải sai liên kết polygon/label.

## Giới hạn diễn giải

Audit này chứng minh tính toàn vẹn logic và provenance của artifact, không chứng minh độ đúng lâm sàng, độ khó hay lợi ích huấn luyện. Ba kết luận đó lần lượt cần dual-physician review, benchmark có question-only/image-shuffle baselines, và thí nghiệm zero-shot–fine-tuned/transfer trên cùng test set.
"""
    args.md_report.write_text(md, encoding="utf-8")
    print(json.dumps({"status": status, "failures": dict(failures), "json_report": str(args.json_report), "md_report": str(args.md_report)}, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
