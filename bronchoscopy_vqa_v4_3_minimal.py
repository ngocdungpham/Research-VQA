"""Minimal, public reference for the Bronchoscopy-VQA v4.0–v4.3 design.

This single-file example demonstrates the scientific contract of the private
production pipeline without exposing clinical data or operational code:

  source polygon/labels -> canonical region -> grounded facts
  -> protected QA blueprint -> constrained LLM realization
  -> semantic validation -> clean conversation export -> basic statistics

The ``demo`` command uses a synthetic annotation and a mock LLM response. The
``stats`` command summarizes clean JSON arrays. Python 3.10+ and no third-party
packages are required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


PIPELINE_VERSION = "bronchoscopy_vqa_reference_v4.3"
BLUEPRINT_VERSION = "bronchoscopy_qa_blueprints_v1.3"
PROTECTED_KEYS = (
    "image_id",
    "qa_id",
    "question_intent",
    "question_format",
    "answer_structured",
    "answer_text",
    "options",
    "source_fact_ids",
    "evidence_region_ids",
    "evidence_boxes_xyxy",
)

ANATOMY_VI = {
    "left_main_bronchus": "phế quản gốc trái",
    "right_main_bronchus": "phế quản gốc phải",
    "trachea": "khí quản",
    "carina": "carina",
    "vocal_cords": "dây thanh",
}

VALUE_VI = {
    "blood": "Máu.",
    "clotted": "Dạng cục.",
    "mucus": "Dịch nhầy.",
    "purulent": "Dịch mủ.",
    "non_pedunculated": "Không cuống.",
    True: "Có.",
    False: "Không.",
}

# Attribute-specific rules are the central v4.2/v4.3 correction: one question
# targets one answerable property instead of using an ambiguous generic prompt.
QUESTION_RULES = {
    "secretion_type": {
        "question": "Loại dịch tiết nào được ghi nhận tại {anatomy}?",
        "reason": "Vùng bằng chứng tại {anatomy} ghi nhận {value_lower}.",
        "q_type": "open_ended_questions",
        "anchors": [["loại"], ["dịch tiết"]],
    },
    "secretion_consistency": {
        "question": "Dịch tiết tại {anatomy} có tính chất hoặc hình thái như thế nào?",
        "reason": "Dịch tiết tại {anatomy} có hình thái {value_lower}.",
        "q_type": "open_ended_questions",
        "anchors": [["dịch tiết"], ["tính chất", "hình thái", "độ đặc"]],
    },
    "tumor_morphology": {
        "question": "Tổn thương dạng khối tại {anatomy} có hình thái nào?",
        "reason": "Vùng bằng chứng tại {anatomy} cho thấy tổn thương {value_lower}.",
        "q_type": "open_ended_questions",
        "anchors": [["khối", "tổn thương"], ["hình thái", "dạng"]],
    },
}

FORBIDDEN_DEEP_DIAGNOSES = {
    "ung thư",
    "carcinoma",
    "ác tính",
    "di căn",
    "mô bệnh học",
    "tế bào học",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_id(prefix: str, *parts: Any, length: int = 24) -> str:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:length]}"


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def bbox_from_polygon(points: list[dict[str, float]]) -> list[int]:
    if len(points) < 3:
        raise ValueError("A polygon requires at least three points")
    xs = [point["x"] for point in points]
    ys = [point["y"] for point in points]
    return [round(min(xs)), round(min(ys)), round(max(xs)), round(max(ys))]


# v4.0: normalize source annotation and retain provenance.
def canonicalize(source: dict[str, Any]) -> dict[str, Any]:
    annotation = source["annotation"]
    region_id = stable_id("canonical_region", source["image_id"], annotation["annotation_id"])
    return {
        "image_id": source["image_id"],
        "image": source["image"],
        "split": source["split"],
        "regions": [
            {
                "region_id": region_id,
                "bbox_xyxy": bbox_from_polygon(annotation["polygon"]),
                "anatomy": annotation["anatomy"],
                "labels": annotation["labels"],
                "source_annotation_ids": [annotation["annotation_id"]],
            }
        ],
    }


# v4.0/v4.1: facts decide what is true and where the evidence is located.
def build_fact_package(canonical: dict[str, Any]) -> dict[str, Any]:
    facts: list[dict[str, Any]] = []
    for region in canonical["regions"]:
        labels = region["labels"]
        parent_id = None
        if labels.get("secretion") is not None:
            parent_id = stable_id("fact", canonical["image_id"], region["region_id"], "secretion")
            facts.append(
                {
                    "fact_id": parent_id,
                    "fact_kind": "observation",
                    "concept": "secretion",
                    "value": labels["secretion"],
                    "polarity": "present" if labels["secretion"] else "absent",
                    "anatomical_locations": [region["anatomy"]],
                    "source_annotation_ids": region["source_annotation_ids"],
                    "evidence_region_ids": [region["region_id"]],
                }
            )
        for concept in ("secretion_type", "secretion_consistency", "tumor_morphology"):
            value = labels.get(concept)
            if value is None:
                continue
            facts.append(
                {
                    "fact_id": stable_id("fact", canonical["image_id"], region["region_id"], concept, value),
                    "fact_kind": "attribute",
                    "concept": concept,
                    "value": value,
                    "polarity": "present",
                    "parent_fact_id": parent_id,
                    "anatomical_locations": [region["anatomy"]],
                    "source_annotation_ids": region["source_annotation_ids"],
                    "evidence_region_ids": [region["region_id"]],
                }
            )
    return {
        "image_id": canonical["image_id"],
        "image": canonical["image"],
        "split": canonical["split"],
        "regions": canonical["regions"],
        "facts": facts,
    }


def answer_text(value: Any) -> str:
    return VALUE_VI.get(value, f"{str(value).replace('_', ' ').capitalize()}.")


# v4.2: lock answer, facts, evidence, intent and format before calling an LLM.
def build_blueprints(package: dict[str, Any]) -> list[dict[str, Any]]:
    blueprints: list[dict[str, Any]] = []
    region_by_id = {region["region_id"]: region for region in package["regions"]}
    for fact in package["facts"]:
        rule = QUESTION_RULES.get(fact["concept"])
        if rule is None:
            continue
        anatomy = fact["anatomical_locations"][0]
        anatomy_vi = ANATOMY_VI.get(anatomy, anatomy.replace("_", " "))
        value_vi = answer_text(fact["value"]).rstrip(".")
        evidence_ids = fact["evidence_region_ids"]
        evidence_boxes = [region_by_id[region_id]["bbox_xyxy"] for region_id in evidence_ids]
        qa_id = stable_id("qa", package["image_id"], fact["fact_id"], fact["concept"])
        protected = {
            "image_id": package["image_id"],
            "qa_id": qa_id,
            "question_intent": fact["concept"],
            "question_format": rule["q_type"],
            "answer_structured": {"concept": fact["concept"], "value": fact["value"]},
            "answer_text": answer_text(fact["value"]),
            "options": {},
            "source_fact_ids": [fact["fact_id"]],
            "evidence_region_ids": evidence_ids,
            "evidence_boxes_xyxy": evidence_boxes,
        }
        blueprints.append(
            {
                "blueprint_version": BLUEPRINT_VERSION,
                "qa_plan_id": stable_id("plan", qa_id, BLUEPRINT_VERSION),
                **protected,
                "question_draft": rule["question"].format(anatomy=anatomy_vi),
                "reasoning_text_draft": rule["reason"].format(
                    anatomy=anatomy_vi, value_lower=value_vi.lower()
                ),
                "required_question_anchors": rule["anchors"],
                "required_reason_terms": [value_vi.lower()],
                "protected_sha256": hashlib.sha256(canonical_json(protected).encode("utf-8")).hexdigest(),
            }
        )
    return blueprints


# v4.3: the LLM sees whole-image context, grounded facts and fixed QA plans.
def build_llm_request(package: dict[str, Any], blueprints: list[dict[str, Any]]) -> dict[str, Any]:
    fact_by_id = {fact["fact_id"]: fact for fact in package["facts"]}
    return {
        "system_contract": {
            "role": "Vietnamese bronchoscopy VQA language realizer",
            "may_write": ["question", "reasoning_text", "visual_evidence_summary"],
            "must_not_change": [
                "answer_structured",
                "answer_text",
                "options",
                "source_fact_ids",
                "evidence_region_ids",
                "question_intent",
                "question_format",
            ],
            "clinical_rule": "Describe visible abnormalities; do not infer deep pathology.",
            "image_policy": "Use the full image and numbered bbox overlay; do not use ROI crops.",
        },
        "image_id": package["image_id"],
        "image": package["image"],
        "visual_evidence": [
            {
                "region_id": region["region_id"],
                "bbox_xyxy": region["bbox_xyxy"],
                "anatomy": region["anatomy"],
            }
            for region in package["regions"]
        ],
        "qa_plans": [
            {
                "qa_plan_id": blueprint["qa_plan_id"],
                "qa_id": blueprint["qa_id"],
                "question_draft": blueprint["question_draft"],
                "reasoning_text_draft": blueprint["reasoning_text_draft"],
                "selected_facts": [fact_by_id[fact_id] for fact_id in blueprint["source_fact_ids"]],
                "locked_answer": blueprint["answer_structured"],
            }
            for blueprint in blueprints
        ],
        "output_fields": [
            "qa_plan_id",
            "qa_id",
            "status",
            "question",
            "reasoning_text",
            "visual_evidence_summary",
        ],
    }


def mock_llm_realization(blueprint: dict[str, Any]) -> dict[str, Any]:
    """Synthetic stand-in. Replace only this function with an API call."""
    concept = blueprint["question_intent"]
    if concept == "secretion_type":
        question = "Tại phế quản gốc trái, dịch tiết quan sát được thuộc loại nào?"
        reason = "Dịch tiết nhìn thấy tại phế quản gốc trái có màu và hình thái phù hợp với máu."
        visual = "Lòng phế quản gốc trái chứa chất tiết màu đỏ sẫm."
    elif concept == "secretion_consistency":
        question = "Dịch tiết tại phế quản gốc trái có hình thái như thế nào?"
        reason = "Vùng quan sát cho thấy dịch tiết kết tụ thành dạng cục."
        visual = "Trong lòng phế quản có chất tiết không đồng nhất và kết tụ."
    else:
        question = blueprint["question_draft"]
        reason = blueprint["reasoning_text_draft"]
        visual = "Dấu hiệu được mô tả nằm trong vùng bằng chứng đã định vị."
    return {
        "qa_plan_id": blueprint["qa_plan_id"],
        "qa_id": blueprint["qa_id"],
        "status": "accepted",
        "question": question,
        "reasoning_text": reason,
        "visual_evidence_summary": visual,
    }


def validate_realization(blueprint: dict[str, Any], response: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    protected = {key: blueprint[key] for key in PROTECTED_KEYS}
    expected_seal = hashlib.sha256(canonical_json(protected).encode("utf-8")).hexdigest()
    if expected_seal != blueprint.get("protected_sha256"):
        failures.append("invalid_blueprint_seal")
    if response.get("qa_plan_id") != blueprint["qa_plan_id"]:
        failures.append("qa_plan_id_mismatch")
    if response.get("qa_id") != blueprint["qa_id"]:
        failures.append("qa_id_mismatch")
    if response.get("status") != "accepted":
        failures.append("not_accepted")
    for field in ("question", "reasoning_text", "visual_evidence_summary"):
        if not isinstance(response.get(field), str) or not response[field].strip():
            failures.append(f"missing_{field}")

    question = normalize_text(response.get("question", ""))
    reason = normalize_text(response.get("reasoning_text", ""))
    visual = normalize_text(response.get("visual_evidence_summary", ""))
    for alternatives in blueprint["required_question_anchors"]:
        if not any(normalize_text(term) in question for term in alternatives):
            failures.append("missing_question_anchor:" + "|".join(alternatives))
    for term in blueprint["required_reason_terms"]:
        if normalize_text(term) not in reason:
            failures.append("missing_reason_fact_term:" + term)
    answer = normalize_text(blueprint["answer_text"])
    if answer and answer in question:
        failures.append("answer_leakage_in_question")
    if any(term in f"{question} {reason} {visual}" for term in FORBIDDEN_DEEP_DIAGNOSES):
        failures.append("unsafe_clinical_inference")

    # Protected content is always reattached from the blueprint. Identical echoes
    # are ignored; mutations are rejected. An echoed SHA is transport metadata,
    # so even an incorrect echo is ignored and the system recomputes the seal.
    for field in PROTECTED_KEYS:
        if field in response and response[field] != blueprint[field]:
            failures.append("protected_field_mutation:" + field)
    return failures


def materialize_clean_record(
    package: dict[str, Any], blueprint: dict[str, Any], response: dict[str, Any]
) -> dict[str, Any]:
    failures = validate_realization(blueprint, response)
    if failures:
        raise ValueError("Semantic validation failed: " + ", ".join(failures))
    locations = json.dumps(blueprint["evidence_boxes_xyxy"], ensure_ascii=False)
    assistant = (
        f"<answer> {blueprint['answer_text']} "
        f"<reason> {response['reasoning_text']} "
        f"<visual_evidence> {response['visual_evidence_summary']} "
        f"<location> {locations}"
    )
    return {
        "qa_id": blueprint["qa_id"],
        "image": package["image"],
        "conversations": [
            {"from": "human", "value": f"<image>\n{response['question']}"},
            {"from": "gpt", "value": assistant},
        ],
        "question_type": blueprint["question_intent"],
        "q_type": blueprint["question_format"],
        "evidence_boxes_xyxy": blueprint["evidence_boxes_xyxy"],
        "answer_structured": blueprint["answer_structured"],
    }


def contradiction_audit(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[tuple[str, str], str] = {}
    failures: list[dict[str, Any]] = []
    for record in records:
        question = record["conversations"][0]["value"].replace("<image>", "", 1)
        key = (record["image"], normalize_text(question))
        answer = canonical_json(record["answer_structured"])
        if key in seen and seen[key] != answer:
            failures.append({"key": key, "answer_a": seen[key], "answer_b": answer})
        seen[key] = answer
    return failures


def summarize_clean_records(paths: list[Path]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON array: {path}")
        records.extend(payload)
    q_types = Counter(record.get("q_type", "missing") for record in records)
    question_types = Counter(record.get("question_type", "missing") for record in records)
    return {
        "records": len(records),
        "unique_images": len({record.get("image") for record in records}),
        "bbox_instances": sum(len(record.get("evidence_boxes_xyxy", [])) for record in records),
        "q_type": dict(q_types.most_common()),
        "question_type": dict(question_types.most_common()),
        "contradiction_count": len(contradiction_audit(records)),
    }


def synthetic_source() -> dict[str, Any]:
    return {
        "image_id": "image_demo_001",
        "image": "images/demo_bronchoscopy.png",
        "split": "train",
        "annotation": {
            "annotation_id": "annotation_demo_001",
            "polygon": [
                {"x": 58.2, "y": 21.9},
                {"x": 276.7, "y": 21.9},
                {"x": 276.7, "y": 199.1},
                {"x": 58.2, "y": 199.1},
            ],
            "anatomy": "left_main_bronchus",
            "labels": {
                "secretion": True,
                "secretion_type": "blood",
                "secretion_consistency": "clotted",
            },
        },
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_demo(output_dir: Path) -> None:
    canonical = canonicalize(synthetic_source())
    facts = build_fact_package(canonical)
    blueprints = build_blueprints(facts)
    request = build_llm_request(facts, blueprints)
    responses = [mock_llm_realization(blueprint) for blueprint in blueprints]
    clean = [
        materialize_clean_record(facts, blueprint, response)
        for blueprint, response in zip(blueprints, responses)
    ]
    audit = {
        "pipeline_version": PIPELINE_VERSION,
        "semantic_failure_count": 0,
        "contradiction_count": len(contradiction_audit(clean)),
        "accepted_records": len(clean),
        "status": "PASS",
    }
    write_json(output_dir / "01_canonical.json", canonical)
    write_json(output_dir / "02_facts.json", facts)
    write_json(output_dir / "03_protected_blueprints.json", blueprints)
    write_json(output_dir / "04_llm_request.json", request)
    write_json(output_dir / "05_llm_response.json", responses)
    write_json(output_dir / "06_clean_records.json", clean)
    write_json(output_dir / "07_audit.json", audit)
    print(json.dumps({"output_dir": str(output_dir), **audit}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run the complete synthetic example")
    demo.add_argument("--output-dir", type=Path, default=Path("demo_output"))
    stats = commands.add_parser("stats", help="Summarize clean JSON arrays")
    stats.add_argument("inputs", nargs="+", type=Path)
    stats.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.command == "demo":
        run_demo(args.output_dir)
    else:
        summary = summarize_clean_records(args.inputs)
        if args.output:
            write_json(args.output, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
