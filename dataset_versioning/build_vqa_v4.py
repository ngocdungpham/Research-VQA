#!/usr/bin/env python3
"""Build fact-grounded bronchoscopy VQA v4 in reproducible stages.

Stages:
  facts -> blueprints -> realize -> validate/deduplicate -> golden-candidates

Canonical labels and Stage-2 evidence are protected fields. The LLM can only
rewrite the question and short evidence statement; it cannot choose facts,
answers, options, polarity, or evidence IDs.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
import re
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import requests
from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DIR = ROOT / "canonical_v2_2"
STAGE2_DIR = ROOT / "derived_v2"
STAGE3_DIR = ROOT / "derived_v3"
OUTPUT_DIR = ROOT / "derived_v4"
FACT_VERSION = "bronchoscopy_facts_v1.0"
BLUEPRINT_VERSION = "bronchoscopy_qa_blueprints_v1.0"
DATASET_VERSION = "derived_v4.0"
DEFAULT_BASE_URL = "http://127.0.0.1:20128/v1"
DEFAULT_MODEL = "gen-VQA"
SEMANTIC_THRESHOLD = 0.92
CHUNK = 1024 * 1024


FACT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["fact_table_version", "image_id", "patient_id", "procedure_id", "split", "image", "regions", "facts"],
    "properties": {
        "fact_table_version": {"const": FACT_VERSION},
        "image_id": {"type": "string"}, "patient_id": {"type": "string"},
        "procedure_id": {"type": "string"}, "split": {"enum": ["train", "validation", "test"]},
        "image": {"type": "object"}, "regions": {"type": "array"},
        "facts": {"type": "array", "items": {"$ref": "#/$defs/fact"}},
    },
    "additionalProperties": False,
    "$defs": {
        "fact": {
            "type": "object", "additionalProperties": False,
            "required": [
                "fact_id", "scope", "subject_region_id", "fact_kind", "concept", "concept_family",
                "predicate", "value", "surface_vi", "polarity", "certainty", "anatomical_locations",
                "evidence_region_ids", "source_annotation_ids", "parent_fact_id", "origin",
                "evidence_status", "question_eligible"
            ],
            "properties": {
                "fact_id": {"type": "string"}, "scope": {"enum": ["image", "region"]},
                "subject_region_id": {"type": ["string", "null"]},
                "fact_kind": {"enum": ["image_state", "anatomy", "observation", "attribute"]},
                "concept": {"type": "string"}, "concept_family": {"type": "string"},
                "predicate": {"type": "string"}, "value": {}, "surface_vi": {"type": "string"},
                "polarity": {"enum": ["present", "absent", "unknown"]},
                "certainty": {"enum": ["confirmed", "uncertain"]},
                "anatomical_locations": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
                "evidence_region_ids": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
                "source_annotation_ids": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
                "parent_fact_id": {"type": ["string", "null"]}, "origin": {"type": "string"},
                "evidence_status": {"enum": ["accepted", "review_required"]},
                "question_eligible": {"type": "boolean"},
            },
        }
    },
}


BLUEPRINT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object", "additionalProperties": False,
    "required": [
        "blueprint_version", "qa_id", "image_id", "patient_id", "procedure_id", "split",
        "question_format", "question_intent", "template_id", "question_draft", "answer_structured",
        "answer_text_draft", "reasoning_text_draft", "options", "source_fact_ids",
        "evidence_region_ids", "canonical_answer", "required_surface_terms"
    ],
    "properties": {
        "blueprint_version": {"const": BLUEPRINT_VERSION}, "qa_id": {"type": "string"},
        "image_id": {"type": "string"}, "patient_id": {"type": "string"},
        "procedure_id": {"type": "string"}, "split": {"enum": ["train", "validation", "test"]},
        "question_format": {"enum": ["open-ended", "closed-ended", "single-choice", "multi-choice"]},
        "question_intent": {"type": "string"}, "template_id": {"type": "string"},
        "question_draft": {"type": "string"}, "answer_structured": {},
        "answer_text_draft": {"type": "string"}, "reasoning_text_draft": {"type": "string"},
        "options": {"type": "object"}, "source_fact_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1, "uniqueItems": True},
        "evidence_region_ids": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
        "canonical_answer": {"type": "string"},
        "required_surface_terms": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
    },
}


OBSERVATION_SPECS = [
    ("mucosal_findings", "infiltration", "mucosal_infiltration", "mucosa", "niêm mạc thâm nhiễm"),
    ("mucosal_findings", "erythema", "mucosal_erythema", "mucosa", "niêm mạc xung huyết"),
    ("mucosal_findings", "carinal_edema", "carinal_edema", "mucosa", "phù nề carina"),
    ("mucosal_findings", "anthracotic_pigmentation", "anthracotic_pigmentation", "mucosa", "nhiễm sắc tố anthracotic"),
    ("mucosal_findings", "pseudomembrane", "pseudomembrane", "mucosa", "giả mạc"),
    ("mucosal_findings", "ulceration", "mucosal_ulceration", "mucosa", "loét niêm mạc"),
    ("mucosal_findings", "smooth_mucosa", "smooth_mucosa", "mucosa", "niêm mạc nhẵn"),
    ("mucosal_findings", "mucosal_atrophy", "mucosal_atrophy", "mucosa", "teo niêm mạc"),
    ("mucosal_findings", "visible_orifice", "visible_orifice", "mucosa", "thấy lỗ phế quản"),
    ("mucosal_findings", "other", "other_mucosal_finding", "mucosa", "bất thường niêm mạc khác"),
    ("vascular_findings", "hypervascularity", "hypervascularity", "vascular", "tăng sinh mạch"),
    ("airway_wall_findings", "tracheomalacia", "tracheomalacia", "airway_wall", "nhuyễn khí quản"),
]

ANATOMY_VI = {
    "trachea": "khí quản", "carina": "carina", "right_main_bronchus": "phế quản gốc phải",
    "left_main_bronchus": "phế quản gốc trái", "right_upper_lobe_bronchus": "phế quản thùy trên phải",
    "right_middle_lobe_bronchus": "phế quản thùy giữa phải", "right_lower_lobe_bronchus": "phế quản thùy dưới phải",
    "left_upper_lobe_bronchus": "phế quản thùy trên trái", "left_lower_lobe_bronchus": "phế quản thùy dưới trái",
    "left_lingula_segment_bronchus": "phế quản phân thùy lưỡi trái", "intermediate_bronchus": "phế quản trung gian",
    "vocal_cords": "dây thanh", "left_apical_segment_bronchus": "phế quản phân thùy đỉnh trái",
}

VALUE_VI = {
    "0_25_percent": "hẹp 0–25%", "26_50_percent": "hẹp 26–50%", "51_75_percent": "hẹp 51–75%",
    "76_90_percent": "hẹp 76–90%", "over_90_percent": "hẹp trên 90%",
    "scarring": "xơ sẹo", "tumor": "khối u", "external_compression": "đè ép ngoài",
    "endoluminal_lesion": "tổn thương trong lòng phế quản", "torsion": "xoắn", "mixed": "hỗn hợp",
    "pedunculated": "có cuống", "non_pedunculated": "không cuống",
    "blood": "máu", "purulent": "dịch mủ", "bright_red": "đỏ tươi", "dark_red": "đỏ sẫm", "clotted": "dạng cục",
}

FORBIDDEN_TEXT = [
    "ung thư", "ác tính", "lao", "viêm phổi", "mô bệnh học", "tiên lượng", "điều trị",
    "theo báo cáo", "annotation", "report", "dữ liệu được cung cấp", "chẩn đoán xác định",
]


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_id(prefix: str, *parts: str, length: int = 24) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


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


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().replace("–", "-").replace("—", "-")
    text = re.sub(r"[^\w%]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def join_vi(items: list[str]) -> str:
    items = list(dict.fromkeys(items))
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " và " + items[-1]


def verify_manifest(directory: Path, expected_version: str) -> dict[str, Any]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["dataset_version"] != expected_version:
        raise ValueError(f"Expected {expected_version}, got {manifest['dataset_version']}")
    for row in manifest["outputs"]:
        path = ROOT / row["path"]
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise ValueError(f"Checksum mismatch: {path}")
    return manifest


def polygon_area(points: list[list[float]]) -> float:
    if len(points) < 3:
        return 0.0
    return abs(sum(points[i][0] * points[(i + 1) % len(points)][1] - points[(i + 1) % len(points)][0] * points[i][1] for i in range(len(points))) / 2)


def size_bin(area_fraction: float | None) -> str | None:
    if area_fraction is None:
        return None
    if area_fraction < 0.01:
        return "small"
    if area_fraction < 0.10:
        return "medium"
    return "large"


def add_fact(
    facts: list[dict[str, Any]], image_id: str, region_id: str | None, fact_kind: str,
    concept: str, family: str, predicate: str, value: Any, surface: str, polarity: str,
    anatomy: list[str], evidence: list[str], source_ids: list[str], parent: str | None,
    origin: str, evidence_status: str, eligible: bool,
) -> str:
    fact_id = stable_id("fact", image_id, region_id or "image", fact_kind, concept, canonical_json(value), polarity)
    facts.append({
        "fact_id": fact_id, "scope": "region" if region_id else "image", "subject_region_id": region_id,
        "fact_kind": fact_kind, "concept": concept, "concept_family": family, "predicate": predicate,
        "value": value, "surface_vi": surface, "polarity": polarity, "certainty": "confirmed",
        "anatomical_locations": sorted(set(anatomy)), "evidence_region_ids": sorted(set(evidence)),
        "source_annotation_ids": sorted(set(source_ids)), "parent_fact_id": parent, "origin": origin,
        "evidence_status": evidence_status, "question_eligible": bool(eligible),
    })
    return fact_id


def build_facts(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    verify_manifest(args.canonical_dir, "canonical_v2.2")
    verify_manifest(args.stage2_dir, "derived_v2.0")
    verify_manifest(args.stage3_dir, "derived_v3.0")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "schemas").mkdir()
    write_json(args.output_dir / "schemas" / "facts_v1.schema.json", FACT_SCHEMA)
    write_json(args.output_dir / "schemas" / "qa_blueprints_v1.schema.json", BLUEPRINT_SCHEMA)

    split_map = {row["image_id"]: row for row in read_jsonl(args.stage3_dir / "split_assignments.jsonl.gz")}
    derivations = {row["image_id"]: row for row in read_jsonl(args.stage2_dir / "region_derivations.jsonl.gz")}
    validator = Draft202012Validator(FACT_SCHEMA)
    output = args.output_dir / "facts_v1.jsonl.gz"
    counts: Counter[str] = Counter()
    validation_errors = []
    with gzip.open(args.canonical_dir / "images.jsonl.gz", "rt", encoding="utf-8") as source, gzip.open(output, "wt", encoding="utf-8", compresslevel=9) as out:
        for line in source:
            image = json.loads(line)
            split = split_map[image["image_id"]]["split"]
            derived = derivations[image["image_id"]]
            canonical_by_id = {region["region_id"]: region for region in image["regions"]}
            facts: list[dict[str, Any]] = []
            regions_out: list[dict[str, Any]] = []

            for dregion in derived["regions"]:
                source_regions = [canonical_by_id[key] for key in dregion["source_region_ids"]]
                geometry_source = canonical_by_id[dregion["training_geometry_source_region_id"]]
                labels = dregion["labels_override"] or source_regions[0]["labels"]
                polygon = dregion["training_polygon_override"] or geometry_source["polygon"]
                source_ids = sorted({sid for row in source_regions for sid in row["source_annotation_ids"]})
                anatomy = labels["anatomy"]
                area_fraction = None
                if image["width"] and image["height"] and polygon:
                    area_fraction = polygon_area(polygon) / (image["width"] * image["height"])
                status = dregion["training_status"]
                eligible = status == "accepted" and image["image_path"] is not None
                regions_out.append({
                    "canonical_region_id": dregion["canonical_region_id"],
                    "source_region_ids": dregion["source_region_ids"], "anatomy": anatomy,
                    "pathology_groups": labels["pathology_group"], "evidence_status": status,
                    "area_fraction": area_fraction, "size_bin": size_bin(area_fraction),
                })
                for anatomy_id in anatomy:
                    add_fact(facts, image["image_id"], dregion["canonical_region_id"], "anatomy", anatomy_id, "anatomy",
                             "located_at", anatomy_id, ANATOMY_VI.get(anatomy_id, anatomy_id), "present", anatomy,
                             [dregion["canonical_region_id"]], source_ids, None, "canonical_region_label", status, False)
                for section, field, concept, family, surface in OBSERVATION_SPECS:
                    state = labels[section][field]
                    if state is None:
                        continue
                    add_fact(facts, image["image_id"], dregion["canonical_region_id"], "observation", concept, family,
                             "has_observation", state, surface, "present" if state else "absent", anatomy,
                             [dregion["canonical_region_id"]], source_ids, None, "canonical_region_label", status, eligible)
                parent_ids: dict[str, str] = {}
                for key, concept, family, surface in [
                    ("stenosis", "stenosis", "stenosis", "hẹp lòng phế quản"),
                    ("tumor", "bronchial_tumor", "tumor", "khối u khí phế quản"),
                    ("secretion", "secretion", "secretion", "dịch tiết bất thường"),
                ]:
                    state = labels[key]["presence"]
                    if state is not None:
                        parent_ids[key] = add_fact(
                            facts, image["image_id"], dregion["canonical_region_id"], "observation", concept, family,
                            "has_observation", state, surface, "present" if state else "absent", anatomy,
                            [dregion["canonical_region_id"]], source_ids, None, "canonical_region_label", status, eligible
                        )
                attributes = [
                    ("stenosis", "severity", "stenosis_severity", "severity"),
                    ("stenosis", "cause", "stenosis_cause", "cause"),
                    ("tumor", "morphology", "tumor_morphology", "morphology"),
                    ("secretion", "type", "secretion_type", "type"),
                    ("secretion", "color", "secretion_color", "color"),
                    ("secretion", "consistency", "secretion_consistency", "consistency"),
                ]
                for section, field, concept, family in attributes:
                    value = labels[section][field]
                    if value is None:
                        continue
                    add_fact(facts, image["image_id"], dregion["canonical_region_id"], "attribute", concept, family,
                             "has_attribute", value, VALUE_VI.get(value, value), "present", anatomy,
                             [dregion["canonical_region_id"]], source_ids, parent_ids.get(section),
                             "canonical_region_label", status, eligible)
                for field, concept, surface in [("normal", "vocal_cords_normal", "dây thanh bình thường"), ("paralysis", "vocal_cord_paralysis", "liệt dây thanh")]:
                    state = labels["vocal_cords"][field]
                    if state is not None:
                        add_fact(facts, image["image_id"], dregion["canonical_region_id"], "observation", concept, "vocal_cords",
                                 "has_observation", state, surface, "present" if state else "absent", anatomy,
                                 [dregion["canonical_region_id"]], source_ids, None, "canonical_region_label", status, eligible)
                for value in labels["other_findings"]:
                    add_fact(facts, image["image_id"], dregion["canonical_region_id"], "observation", value, "other",
                             "has_observation", value, value.replace("_", " "), "present", anatomy,
                             [dregion["canonical_region_id"]], source_ids, None, "canonical_region_label", status, eligible)

            normal = image["image_labels"]["normal"]
            if normal is not None:
                evidence = sorted({r["canonical_region_id"] for r in regions_out if r["pathology_groups"]})
                add_fact(facts, image["image_id"], None, "image_state", "image_normal", "image_state", "image_state",
                         normal, "hình ảnh nội soi bình thường", "present" if normal else "absent", [], evidence, [], None,
                         "canonical_image_label", derived["image_training_status"],
                         derived["image_training_status"] == "accepted" and image["image_path"] is not None)

            record = {
                "fact_table_version": FACT_VERSION, "image_id": image["image_id"], "patient_id": image["patient_id"],
                "procedure_id": image["procedure_id"], "split": split,
                "image": {"path": image["image_path"], "width": image["width"], "height": image["height"],
                          "normal": normal, "training_status": derived["image_training_status"]},
                "regions": regions_out, "facts": sorted(facts, key=lambda row: row["fact_id"]),
            }
            errors = list(validator.iter_errors(record))
            if errors:
                validation_errors.append({"image_id": image["image_id"], "errors": [e.message for e in errors[:5]]})
            write_jsonl(out, record)
            counts["images"] += 1; counts["facts"] += len(facts)
            counts.update(f"fact_kind:{fact['fact_kind']}" for fact in facts)
            counts.update(f"polarity:{fact['polarity']}" for fact in facts)
            counts["eligible_facts"] += sum(f["question_eligible"] for f in facts)
    if validation_errors:
        write_json(args.output_dir / "facts_validation_errors.json", validation_errors)
        raise ValueError(f"Fact schema failed for {len(validation_errors)} images")
    meta = {"stage": "facts", "version": FACT_VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "script_sha256": sha256_file(Path(__file__)), "output_sha256": sha256_file(output), "counts": dict(sorted(counts.items()))}
    write_json(args.output_dir / "facts_manifest.json", meta)
    print(json.dumps(meta, indent=2))


def make_blueprint(record: dict[str, Any], intent: str, fmt: str, template: str, question: str,
                   answer_structured: Any, answer_text: str, reason: str, facts: list[dict[str, Any]],
                   options: dict[str, str] | None = None) -> dict[str, Any]:
    fact_ids = sorted(f["fact_id"] for f in facts)
    evidence = sorted({region for fact in facts for region in fact["evidence_region_ids"]})
    qa_id = stable_id("qa", record["image_id"], intent, fmt, template, *fact_ids)
    required = sorted({f["surface_vi"] for f in facts if f["fact_kind"] != "image_state"})
    return {
        "blueprint_version": BLUEPRINT_VERSION, "qa_id": qa_id, "image_id": record["image_id"],
        "patient_id": record["patient_id"], "procedure_id": record["procedure_id"], "split": record["split"],
        "question_format": fmt, "question_intent": intent, "template_id": template,
        "question_draft": question, "answer_structured": answer_structured, "answer_text_draft": answer_text,
        "reasoning_text_draft": reason, "options": options or {}, "source_fact_ids": fact_ids,
        "evidence_region_ids": evidence, "canonical_answer": canonical_json(answer_structured),
        "required_surface_terms": required,
    }


def build_blueprints(args: argparse.Namespace) -> None:
    facts_path = args.output_dir / "facts_v1.jsonl.gz"
    if not facts_path.is_file():
        raise FileNotFoundError(facts_path)
    output = args.output_dir / "qa_blueprints_v1.jsonl.gz"
    if output.exists():
        raise FileExistsError(output)
    validator = Draft202012Validator(BLUEPRINT_SCHEMA)
    counts: Counter[str] = Counter(); errors = []
    severity_options = {"A": "Hẹp 0–25%", "B": "Hẹp 26–50%", "C": "Hẹp 51–75%", "D": "Hẹp 76–90%", "E": "Hẹp trên 90%"}
    severity_answer = {"0_25_percent": "A", "26_50_percent": "B", "51_75_percent": "C", "76_90_percent": "D", "over_90_percent": "E"}
    with gzip.open(output, "wt", encoding="utf-8", compresslevel=9) as out:
        for record in read_jsonl(facts_path):
            blueprints: list[dict[str, Any]] = []
            eligible = [f for f in record["facts"] if f["question_eligible"]]
            by_region: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for fact in eligible:
                if fact["scope"] == "region":
                    by_region[fact["subject_region_id"]].append(fact)
            positive = [f for f in eligible if f["fact_kind"] == "observation" and f["polarity"] == "present"]
            if positive:
                surfaces = sorted({f["surface_vi"] for f in positive})
                concepts = sorted({f["concept"] for f in positive})
                answer = {"observations": [{"concept": concept, "polarity": "present"} for concept in concepts]}
                blueprints.append(make_blueprint(
                    record, "abnormality", "open-ended", "abnormality_open_01",
                    "Những bất thường nào được ghi nhận trên hình ảnh nội soi?", answer,
                    join_vi(surfaces).capitalize() + ".",
                    f"Các vùng bằng chứng cho thấy {join_vi(surfaces)}.", positive
                ))

            negative_by_concept: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for fact in eligible:
                if fact["fact_kind"] == "observation" and fact["polarity"] == "absent":
                    negative_by_concept[fact["concept"]].append(fact)
            for concept, negative in sorted(negative_by_concept.items()):
                fact = negative[0]
                blueprints.append(make_blueprint(
                    record, fact["concept_family"], "closed-ended", "explicit_negative_closed_01",
                    f"Có ghi nhận {fact['surface_vi']} trên hình ảnh nội soi không?",
                    {"concept": concept, "presence": False}, "Không.",
                    f"Các vùng bằng chứng được xác nhận không ghi nhận {fact['surface_vi']}.", negative
                ))

            attributes: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
            for fact in eligible:
                if fact["fact_kind"] == "attribute" and fact["polarity"] == "present":
                    attributes[(fact["concept"], canonical_json(fact["value"]))].append(fact)
            values_per_concept: Counter[str] = Counter(concept for concept, _ in attributes)
            for (concept, _), attribute_facts in sorted(attributes.items()):
                fact = attribute_facts[0]
                locations = sorted({ANATOMY_VI.get(loc, loc) for row in attribute_facts for loc in row["anatomical_locations"]})
                if values_per_concept[concept] > 1 and not locations:
                    counts["ambiguous_attribute_blueprints_skipped"] += 1
                    continue
                location_text = f" tại {join_vi(locations)}" if locations else ""
                if fact["concept"] == "stenosis_severity":
                    letter = severity_answer[fact["value"]]
                    blueprints.append(make_blueprint(
                        record, "stenosis_degree", "single-choice", "stenosis_severity_single_01",
                        f"Mức độ hẹp lòng phế quản{location_text} được xác nhận thuộc khoảng nào?",
                        {"concept": "stenosis_severity", "value": fact["value"], "option": letter},
                        f"{letter}. {severity_options[letter]}.",
                        f"Vùng bằng chứng{location_text} cho thấy {fact['surface_vi']}.", attribute_facts, severity_options
                    ))
                else:
                    parent_ids = {row["parent_fact_id"] for row in attribute_facts if row["parent_fact_id"]}
                    parents = [f for f in eligible if f["fact_id"] in parent_ids]
                    parent_text = parents[0]["surface_vi"] if parents else "bất thường"
                    blueprints.append(make_blueprint(
                        record, fact["concept_family"], "open-ended", "attribute_open_01",
                        f"Đặc điểm được xác nhận của {parent_text}{location_text} là gì?",
                        {"concept": fact["concept"], "value": fact["value"]},
                        fact["surface_vi"].capitalize() + ".",
                        f"Vùng bằng chứng{location_text} cho thấy đặc điểm {fact['surface_vi']}.", attribute_facts
                    ))
            normal_fact = next((f for f in eligible if f["concept"] == "image_normal"), None)
            if normal_fact:
                present = normal_fact["polarity"] == "present"
                blueprints.append(make_blueprint(record, "image_normality", "closed-ended", "normal_closed_01",
                    "Hình ảnh nội soi này có được xác nhận là bình thường không?",
                    {"normal": present}, "Có." if present else "Không.",
                    "Hình ảnh được xác nhận là bình thường." if present else "Hình ảnh được xác nhận là không bình thường.", [normal_fact]))
            seen = set()
            for bp in blueprints:
                signature = (bp["question_intent"], tuple(bp["source_fact_ids"]), bp["canonical_answer"])
                if signature in seen:
                    counts["fact_duplicates_prevented"] += 1
                    continue
                seen.add(signature)
                validation = list(validator.iter_errors(bp))
                if validation:
                    errors.append({"qa_id": bp["qa_id"], "errors": [e.message for e in validation[:5]]})
                    continue
                write_jsonl(out, bp); counts["blueprints"] += 1
                counts[f"split:{bp['split']}"] += 1; counts[f"format:{bp['question_format']}"] += 1
    if errors:
        write_json(args.output_dir / "blueprint_validation_errors.json", errors)
        raise ValueError(f"Blueprint schema failed for {len(errors)} rows")
    meta = {"stage": "blueprints", "version": BLUEPRINT_VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "script_sha256": sha256_file(Path(__file__)), "output_sha256": sha256_file(output), "counts": dict(sorted(counts.items()))}
    write_json(args.output_dir / "blueprints_manifest.json", meta); print(json.dumps(meta, indent=2))


SURFACE_SYSTEM_PROMPT = """Bạn chỉ diễn đạt lại các QA blueprint y khoa đã được khóa.

Quy tắc bắt buộc:
1. Không thêm, xóa hoặc thay đổi bất kỳ finding, mức độ, vị trí hay trạng thái phủ định nào.
2. answer_text phải được sao chép CHÍNH XÁC từ answer_text_draft.
3. question có thể diễn đạt tự nhiên hơn nhưng phải giữ nguyên intent và không chứa đáp án.
4. reasoning_text chỉ gồm một câu ngắn mô tả trực tiếp evidence; phải giữ tất cả required_surface_terms.
5. Không thêm chẩn đoán, nguyên nhân, tiên lượng, điều trị hay clinical implication.
6. Không dùng các từ report, annotation, dữ liệu được cung cấp, theo báo cáo.
7. Không thay đổi qa_id. Không xuất Markdown.

Trả về đúng JSON: {"items":[{"qa_id":"...","question":"...","answer_text":"...","reasoning_text":"..."}]}.
"""


def extract_json(text: str) -> dict[str, Any]:
    content = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", content, flags=re.DOTALL)
    if fenced:
        content = fenced.group(1)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", content):
        try:
            value, _ = decoder.raw_decode(content[match.start():])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    raise ValueError("No JSON object in model response")


def call_surface_llm(base_url: str, model: str, items: list[dict[str, Any]], retries: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    compact = [{
        "qa_id": x["qa_id"], "question_format": x["question_format"], "question_intent": x["question_intent"],
        "question_draft": x["question_draft"], "answer_text_draft": x["answer_text_draft"],
        "reasoning_text_draft": x["reasoning_text_draft"], "required_surface_terms": x["required_surface_terms"]
    } for x in items]
    payload = {"model": model, "messages": [{"role": "system", "content": SURFACE_SYSTEM_PROMPT},
               {"role": "user", "content": canonical_json({"items": compact})}],
               "temperature": 0.1, "max_completion_tokens": max(1500, len(items) * 180), "stream": False}
    url = base_url.rstrip("/") + "/chat/completions"
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(url, headers={"Authorization": "Bearer local-vqa-v4", "Content-Type": "application/json"},
                                     json=payload, timeout=240)
            response.raise_for_status()
            body = response.json(); message = body["choices"][0]["message"]
            parsed = extract_json(message.get("content") or "")
            output = parsed.get("items")
            if not isinstance(output, list):
                raise ValueError("Model JSON does not contain items list")
            trace = {"requested_model": model, "response_model": body.get("model"), "usage": body.get("usage"),
                     "response_id": body.get("id")}
            return output, trace
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(str(last_error))


def realize(args: argparse.Namespace) -> None:
    blueprint_path = args.output_dir / "qa_blueprints_v1.jsonl.gz"
    suffix = "_pilot" if args.pilot else ""
    output = args.output_dir / f"surface_candidates{suffix}.jsonl.gz"
    error_path = args.output_dir / f"surface_errors{suffix}.jsonl.gz"
    if output.exists() or error_path.exists():
        raise FileExistsError(output)
    blueprints = list(read_jsonl(blueprint_path))
    if args.pilot:
        # Deterministic validation pilot balanced across formats and splits.
        blueprints.sort(key=lambda x: (x["split"] != "validation", x["question_format"], x["qa_id"]))
        blueprints = blueprints[:args.limit]
    elif args.limit:
        blueprints = blueprints[:args.limit]
    counts = Counter(); prompt_hash = hashlib.sha256(SURFACE_SYSTEM_PROMPT.encode()).hexdigest()
    with gzip.open(output, "wt", encoding="utf-8", compresslevel=9) as out, gzip.open(error_path, "wt", encoding="utf-8", compresslevel=9) as errors:
        for start in range(0, len(blueprints), args.batch_size):
            batch = blueprints[start:start + args.batch_size]
            try:
                items, trace = call_surface_llm(args.base_url, args.model, batch, args.retries)
                returned = {str(item.get("qa_id")): item for item in items if isinstance(item, dict)}
                for bp in batch:
                    item = returned.get(bp["qa_id"])
                    if not item:
                        write_jsonl(errors, {"qa_id": bp["qa_id"], "error": "missing_from_model_output"}); counts["errors"] += 1
                        continue
                    write_jsonl(out, {"qa_id": bp["qa_id"], "question": item.get("question"),
                                     "answer_text": item.get("answer_text"), "reasoning_text": item.get("reasoning_text"),
                                     "llm_trace": {**trace, "prompt_sha256": prompt_hash}}); counts["candidates"] += 1
            except Exception as exc:
                for bp in batch:
                    write_jsonl(errors, {"qa_id": bp["qa_id"], "error": str(exc)}); counts["errors"] += 1
            print(f"surface {min(start + len(batch), len(blueprints))}/{len(blueprints)}", flush=True)
    meta = {"stage": "surface_realization", "pilot": args.pilot, "model": args.model, "prompt_sha256": prompt_hash,
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "output_sha256": sha256_file(output),
            "errors_sha256": sha256_file(error_path), "counts": dict(counts)}
    write_json(args.output_dir / f"surface_manifest{suffix}.json", meta); print(json.dumps(meta, indent=2))


def realize_catalog(args: argparse.Namespace) -> None:
    output = args.output_dir / "surface_template_catalog.json"
    if output.exists():
        raise FileExistsError(output)
    representatives: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in read_jsonl(args.output_dir / "qa_blueprints_v1.jsonl.gz"):
        key = (row["template_id"], row["question_intent"], row["question_format"])
        representatives.setdefault(key, row)
    rows = list(representatives.values())
    items, trace = call_surface_llm(args.base_url, args.model, rows, args.retries)
    returned = {item.get("qa_id"): item for item in items if isinstance(item, dict)}
    catalog = []
    for key, bp in sorted(representatives.items()):
        item = returned.get(bp["qa_id"], {})
        failures = validate_candidate(bp, item, set(bp["source_fact_ids"]))
        catalog.append({
            "catalog_key": list(key), "representative_qa_id": bp["qa_id"],
            "draft": {"question": bp["question_draft"], "answer_text": bp["answer_text_draft"],
                      "reasoning_text": bp["reasoning_text_draft"]},
            "llm_surface": {"question": item.get("question"), "answer_text": item.get("answer_text"),
                            "reasoning_text": item.get("reasoning_text")},
            "validation_status": "approved" if not failures else "rejected",
            "validation_failures": failures,
        })
    if any(row["validation_status"] != "approved" for row in catalog):
        write_json(output, {"status": "rejected", "trace": trace, "entries": catalog})
        raise ValueError("One or more surface template representatives failed validation")
    payload = {"status": "approved", "surface_policy": "LLM-audited template families; deterministic protected-slot materialization",
               "trace": trace, "entries": catalog}
    write_json(output, payload)
    write_json(args.output_dir / "surface_template_catalog_manifest.json", {
        "stage": "llm_surface_catalog", "entries": len(catalog), "status": "approved",
        "output_sha256": sha256_file(output), "model": args.model,
        "prompt_sha256": hashlib.sha256(SURFACE_SYSTEM_PROMPT.encode()).hexdigest(),
    })
    print(json.dumps({"entries": len(catalog), "status": "approved", "trace": trace}, indent=2))


def materialize_surface(args: argparse.Namespace) -> None:
    output = args.output_dir / "surface_candidates.jsonl.gz"
    errors = args.output_dir / "surface_errors.jsonl.gz"
    if output.exists() or errors.exists():
        raise FileExistsError(output)
    catalog_path = args.output_dir / "surface_template_catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if catalog.get("status") != "approved":
        raise ValueError("Surface template catalog is not approved")
    catalog_keys = {tuple(row["catalog_key"]) for row in catalog["entries"] if row["validation_status"] == "approved"}
    catalog_sha = sha256_file(catalog_path); count = 0
    with gzip.open(output, "wt", encoding="utf-8", compresslevel=9) as out:
        for bp in read_jsonl(args.output_dir / "qa_blueprints_v1.jsonl.gz"):
            key = (bp["template_id"], bp["question_intent"], bp["question_format"])
            if key not in catalog_keys:
                raise ValueError(f"Blueprint uses unaudited surface family: {key}")
            write_jsonl(out, {
                "qa_id": bp["qa_id"], "question": bp["question_draft"],
                "answer_text": bp["answer_text_draft"], "reasoning_text": bp["reasoning_text_draft"],
                "llm_trace": {"surface_method": "deterministic_instantiation_of_llm_audited_template_family",
                              "catalog_sha256": catalog_sha, "requested_model": args.model,
                              "catalog_key": list(key)}
            }); count += 1
    with gzip.open(errors, "wt", encoding="utf-8", compresslevel=9):
        pass
    meta = {"stage": "surface_materialization", "surface_method": "deterministic protected-slot instantiation",
            "llm_catalog_sha256": catalog_sha, "candidates": count, "output_sha256": sha256_file(output),
            "errors_sha256": sha256_file(errors)}
    write_json(args.output_dir / "surface_manifest.json", meta); print(json.dumps(meta, indent=2))


def validate_candidate(bp: dict[str, Any], candidate: dict[str, Any], fact_ids: set[str]) -> list[str]:
    failures = []
    for field in ("question", "answer_text", "reasoning_text"):
        if not isinstance(candidate.get(field), str) or not candidate[field].strip():
            failures.append(f"missing_{field}")
    if failures:
        return failures
    if candidate["answer_text"].strip() != bp["answer_text_draft"].strip():
        failures.append("protected_answer_text_changed")
    combined = normalize_text(candidate["question"] + " " + candidate["answer_text"] + " " + candidate["reasoning_text"])
    for term in FORBIDDEN_TEXT:
        if normalize_text(term) in combined:
            failures.append(f"forbidden_term:{term}")
    reason_norm = normalize_text(candidate["reasoning_text"])
    for term in bp["required_surface_terms"]:
        if normalize_text(term) not in reason_norm:
            failures.append(f"reason_missing_fact:{term}")
    if any(fid not in fact_ids for fid in bp["source_fact_ids"]):
        failures.append("source_fact_missing")
    if bp["answer_text_draft"] not in ("Có.", "Không.") and normalize_text(bp["answer_text_draft"].rstrip(".")) in normalize_text(candidate["question"]):
        failures.append("answer_leakage_in_question")
    if bp["question_format"] == "closed-ended" and bp["answer_text_draft"] == "Không.":
        # No-answer must originate from an explicit absent fact or explicit normal=false.
        if not (bp["answer_structured"].get("presence") is False or bp["answer_structured"].get("normal") is False):
            failures.append("unsupported_negative_answer")
    return sorted(set(failures))


class TransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch; self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval()
        self.model_name = model_name

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        vectors = []
        with self.torch.no_grad():
            for start in range(0, len(texts), batch_size):
                encoded = self.tokenizer(["query: " + x for x in texts[start:start + batch_size]], padding=True, truncation=True,
                                         max_length=256, return_tensors="pt")
                output = self.model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (output * mask).sum(1) / mask.sum(1).clamp(min=1)
                pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.append(pooled.cpu().numpy())
        return np.concatenate(vectors, axis=0)


def validate_and_dedupe(args: argparse.Namespace) -> None:
    suffix = "_pilot" if args.pilot else ""
    candidate_path = args.output_dir / f"surface_candidates{suffix}.jsonl.gz"
    output_prefix = "pilot_" if args.pilot else ""
    accepted_path = args.output_dir / f"{output_prefix}accepted_vqa.jsonl.gz"
    rejected_path = args.output_dir / f"{output_prefix}rejected_vqa.jsonl.gz"
    review_path = args.output_dir / f"{output_prefix}review_queue.jsonl.gz"
    dedup_path = args.output_dir / f"{output_prefix}dedup_log.jsonl.gz"
    for path in (accepted_path, rejected_path, review_path, dedup_path):
        if path.exists():
            raise FileExistsError(path)
    blueprints = {row["qa_id"]: row for row in read_jsonl(args.output_dir / "qa_blueprints_v1.jsonl.gz")}
    facts_by_image = {row["image_id"]: {f["fact_id"] for f in row["facts"]} for row in read_jsonl(args.output_dir / "facts_v1.jsonl.gz")}
    candidates = list(read_jsonl(candidate_path)); valid_rows = []; rejected = []; counts = Counter()
    for candidate in candidates:
        bp = blueprints.get(candidate["qa_id"])
        if not bp:
            rejected.append({**candidate, "rejection_reasons": ["blueprint_missing"]}); continue
        failures = validate_candidate(bp, candidate, facts_by_image[bp["image_id"]])
        row = {**bp, "question": candidate.get("question"), "answer_text": candidate.get("answer_text"),
               "reasoning_text": candidate.get("reasoning_text"), "llm_trace": candidate.get("llm_trace"),
               "validation_status": "accepted" if not failures else "rejected"}
        if failures:
            row["rejection_reasons"] = failures; rejected.append(row); counts["factual_rejected"] += 1
        else:
            valid_rows.append(row)

    kept: list[dict[str, Any]] = []; dedup_events = []; exact_seen = {}
    for row in sorted(valid_rows, key=lambda x: x["qa_id"]):
        exact_key = (row["image_id"], normalize_text(row["question"]), normalize_text(row["answer_text"]))
        if exact_key in exact_seen:
            dedup_events.append({"type": "exact_duplicate", "kept_qa_id": exact_seen[exact_key], "removed_qa_id": row["qa_id"]})
            rejected.append({**row, "validation_status": "rejected", "rejection_reasons": ["exact_duplicate"]}); counts["exact_duplicates"] += 1
        else:
            exact_seen[exact_key] = row["qa_id"]; kept.append(row)
    fact_seen: dict[tuple[Any, ...], list[str]] = defaultdict(list); after_fact = []
    for row in kept:
        key = (row["image_id"], row["question_intent"], tuple(row["source_fact_ids"]), row["canonical_answer"])
        cap = 2 if row["split"] == "train" else 1
        if len(fact_seen[key]) >= cap:
            dedup_events.append({"type": "fact_duplicate", "kept_qa_ids": fact_seen[key], "removed_qa_id": row["qa_id"]})
            rejected.append({**row, "validation_status": "rejected", "rejection_reasons": ["fact_duplicate"]}); counts["fact_duplicates"] += 1
        else:
            fact_seen[key].append(row["qa_id"]); after_fact.append(row)

    review_rows = []; final_rows = []
    semantic_groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in after_fact:
        semantic_groups[(row["image_id"], row["question_intent"], row["question_format"], row["canonical_answer"])].append(row)
    comparison_groups = [rows for rows in semantic_groups.values() if len(rows) > 1]
    if args.skip_semantic:
        final_rows = after_fact
        semantic_model = "skipped_by_request"
    elif not comparison_groups:
        final_rows = after_fact
        semantic_model = "not_loaded:no_eligible_comparison_groups"
    else:
        embedder = TransformerEmbedder(args.embedding_model); semantic_model = args.embedding_model
        for group_rows in semantic_groups.values():
            if len(group_rows) == 1:
                final_rows.extend(group_rows); continue
            vectors = embedder.encode([r["question"] for r in group_rows])
            removed = set()
            for i in range(len(group_rows)):
                if i in removed: continue
                final_rows.append(group_rows[i])
                for j in range(i + 1, len(group_rows)):
                    if j in removed: continue
                    similarity = float(np.dot(vectors[i], vectors[j]))
                    if similarity >= args.semantic_threshold:
                        removed.add(j)
                        event = {"type": "semantic_duplicate", "kept_qa_id": group_rows[i]["qa_id"],
                                 "removed_qa_id": group_rows[j]["qa_id"], "cosine_similarity": similarity,
                                 "threshold": args.semantic_threshold}
                        dedup_events.append(event)
                        review_rows.append({**group_rows[j], "validation_status": "review_required",
                                            "review_reasons": ["semantic_duplicate"], "semantic_duplicate": event})
                        counts["semantic_duplicates"] += 1

    with gzip.open(accepted_path, "wt", encoding="utf-8", compresslevel=9) as out:
        for row in sorted(final_rows, key=lambda x: (x["split"], x["image_id"], x["qa_id"])):
            write_jsonl(out, row); counts["accepted"] += 1; counts[f"accepted:{row['split']}"] += 1
    with gzip.open(rejected_path, "wt", encoding="utf-8", compresslevel=9) as out:
        for row in rejected: write_jsonl(out, row)
    with gzip.open(review_path, "wt", encoding="utf-8", compresslevel=9) as out:
        for row in review_rows: write_jsonl(out, row)
    with gzip.open(dedup_path, "wt", encoding="utf-8", compresslevel=9) as out:
        for row in dedup_events: write_jsonl(out, row)
    counts["rejected"] = len(rejected); counts["review"] = len(review_rows)
    meta = {"stage": "validate_and_dedupe", "pilot": args.pilot, "semantic_model": semantic_model,
            "semantic_threshold": args.semantic_threshold, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "semantic_comparison_groups": len(comparison_groups),
            "counts": dict(sorted(counts.items())), "outputs": {p.name: sha256_file(p) for p in (accepted_path, rejected_path, review_path, dedup_path)}}
    write_json(args.output_dir / f"{output_prefix}validation_manifest.json", meta); print(json.dumps(meta, indent=2))


def export_splits(args: argparse.Namespace) -> None:
    accepted = args.output_dir / "accepted_vqa.jsonl.gz"
    split_dir = args.output_dir / "vqa"
    if split_dir.exists():
        raise FileExistsError(split_dir)
    split_dir.mkdir()
    paths = {split: split_dir / f"{split}.jsonl.gz" for split in ("train", "validation", "test")}
    handles = {split: gzip.open(path, "wt", encoding="utf-8", compresslevel=9) for split, path in paths.items()}
    counts = Counter()
    try:
        for row in read_jsonl(accepted):
            write_jsonl(handles[row["split"]], row); counts[row["split"]] += 1
    finally:
        for handle in handles.values(): handle.close()
    meta = {"stage": "export_splits", "counts": dict(counts), "outputs": {split: sha256_file(path) for split, path in paths.items()}}
    write_json(args.output_dir / "vqa_split_manifest.json", meta); print(json.dumps(meta, indent=2))


def golden_candidates(args: argparse.Namespace) -> None:
    output_dir = args.output_dir / "golden_candidates_v1"
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir()
    facts = {row["image_id"]: row for row in read_jsonl(args.output_dir / "facts_v1.jsonl.gz") if row["split"] == "test"}
    qa_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.output_dir / "vqa" / "test.jsonl.gz"):
        qa_by_image[row["image_id"]].append(row)
    assignment = {row["image_id"]: row for row in read_jsonl(args.stage3_dir / "split_assignments.jsonl.gz") if row["split"] == "test"}
    profiles = {}
    stratum_frequency = Counter()
    for image_id, rows in qa_by_image.items():
        fact_record = facts[image_id]
        strata = set()
        for fact in fact_record["facts"]:
            if fact["polarity"] == "present" and fact["fact_kind"] == "observation":
                strata.add("family:" + fact["concept_family"])
            for loc in fact["anatomical_locations"]: strata.add("anatomy:" + loc)
        for region in fact_record["regions"]:
            if region["size_bin"]: strata.add("size:" + region["size_bin"])
        for qa in rows:
            strata.add("intent:" + qa["question_intent"]); strata.add("format:" + qa["question_format"])
        profiles[image_id] = strata; stratum_frequency.update(strata)
    selected = []; selected_groups = set(); selected_counts = Counter(); remaining = set(profiles)
    while remaining and len(selected) < args.golden_images:
        best = None
        for image_id in remaining:
            group_id = assignment[image_id]["leakage_group_id"]
            if group_id in selected_groups: continue
            score = sum((1 / math.sqrt(max(stratum_frequency[s], 1))) / (1 + selected_counts[s]) for s in profiles[image_id])
            candidate = (score, image_id)
            if best is None or candidate > best: best = candidate
        if best is None: break
        image_id = best[1]; selected.append(image_id); remaining.remove(image_id)
        selected_groups.add(assignment[image_id]["leakage_group_id"]); selected_counts.update(profiles[image_id])
    candidate_path = output_dir / "golden_candidates_v1.jsonl.gz"
    review_csv = output_dir / "dual_physician_review.csv"
    with gzip.open(candidate_path, "wt", encoding="utf-8", compresslevel=9) as out:
        for image_id in selected:
            write_jsonl(out, {"candidate_status": "pending_dual_physician_review", "image_id": image_id,
                              "leakage_group_id": assignment[image_id]["leakage_group_id"],
                              "strata": sorted(profiles[image_id]), "qa": qa_by_image[image_id]})
    fields = ["image_id", "qa_id", "doctor_1_answerable", "doctor_1_answer_correct", "doctor_1_evidence_correct",
              "doctor_1_reasoning_factual", "doctor_1_comment", "doctor_2_answerable", "doctor_2_answer_correct",
              "doctor_2_evidence_correct", "doctor_2_reasoning_factual", "doctor_2_comment", "adjudication", "final_status"]
    with review_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for image_id in selected:
            for qa in qa_by_image[image_id]: writer.writerow({"image_id": image_id, "qa_id": qa["qa_id"]})
    manifest = {"version": "golden_candidates_v1.0", "status": "pending_dual_physician_review",
                "not_a_golden_test_yet": True, "selected_images": len(selected), "selected_leakage_groups": len(selected_groups),
                "qa_records": sum(len(qa_by_image[i]) for i in selected), "stratum_counts": dict(sorted(selected_counts.items())),
                "outputs": {candidate_path.name: sha256_file(candidate_path), review_csv.name: sha256_file(review_csv)}}
    write_json(output_dir / "manifest.json", manifest); print(json.dumps(manifest, indent=2))


def finalize_manifest(args: argparse.Namespace) -> None:
    required = [
        args.output_dir / "facts_v1.jsonl.gz", args.output_dir / "qa_blueprints_v1.jsonl.gz",
        args.output_dir / "surface_candidates.jsonl.gz", args.output_dir / "accepted_vqa.jsonl.gz",
        args.output_dir / "rejected_vqa.jsonl.gz", args.output_dir / "review_queue.jsonl.gz",
        args.output_dir / "vqa" / "train.jsonl.gz", args.output_dir / "vqa" / "validation.jsonl.gz",
        args.output_dir / "vqa" / "test.jsonl.gz", args.output_dir / "golden_candidates_v1" / "manifest.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing: raise FileNotFoundError(missing)
    inputs = {
        "canonical_v2.2": sha256_file(args.canonical_dir / "manifest.json"),
        "derived_v2.0": sha256_file(args.stage2_dir / "manifest.json"),
        "derived_v3.0": sha256_file(args.stage3_dir / "manifest.json"),
    }
    outputs = []
    for path in sorted(args.output_dir.rglob("*")):
        if path.is_file() and path != args.output_dir / "manifest.json":
            outputs.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    manifest = {"dataset_version": DATASET_VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "input_manifests": inputs, "conversion_script_sha256": sha256_file(Path(__file__)),
                "pipeline": ["facts", "deterministic_blueprints", "llm_surface", "schema_factual_evidence_validation",
                             "exact_fact_semantic_dedup", "split_export", "golden_candidates_pending_doctors"],
                "golden_test_status": "not_created_pending_dual_physician_review", "outputs": outputs}
    write_json(args.output_dir / "manifest.json", manifest); print(json.dumps({"manifest": str(args.output_dir / "manifest.json"), "outputs": len(outputs)}, indent=2))


def verify(args: argparse.Namespace) -> None:
    manifest = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8")); failures = []
    for row in manifest["outputs"]:
        path = ROOT / row["path"]
        if not path.is_file() or sha256_file(path) != row["sha256"]: failures.append(row["path"])
    split_sets = {}
    for split in ("train", "validation", "test"):
        rows = list(read_jsonl(args.output_dir / "vqa" / f"{split}.jsonl.gz"))
        split_sets[split] = {row["image_id"] for row in rows}
        if any(row["split"] != split for row in rows): failures.append(f"wrong split in {split}")
    if split_sets["train"] & split_sets["validation"] or split_sets["train"] & split_sets["test"] or split_sets["validation"] & split_sets["test"]:
        failures.append("VQA image overlap between splits")
    print(json.dumps({"valid": not failures, "failures": failures}, indent=2))
    if failures: raise SystemExit(1)


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--canonical-dir", type=Path, default=CANONICAL_DIR)
    parser.add_argument("--stage2-dir", type=Path, default=STAGE2_DIR)
    parser.add_argument("--stage3-dir", type=Path, default=STAGE3_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(required=True)
    p = sub.add_parser("facts"); add_common(p); p.set_defaults(func=build_facts)
    p = sub.add_parser("blueprints"); add_common(p); p.set_defaults(func=build_blueprints)
    p = sub.add_parser("realize"); add_common(p); p.add_argument("--base-url", default=DEFAULT_BASE_URL); p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--batch-size", type=int, default=20); p.add_argument("--retries", type=int, default=3)
    p.add_argument("--limit", type=int); p.add_argument("--pilot", action="store_true"); p.set_defaults(func=realize)
    p = sub.add_parser("realize-catalog"); add_common(p); p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--model", default=DEFAULT_MODEL); p.add_argument("--retries", type=int, default=3); p.set_defaults(func=realize_catalog)
    p = sub.add_parser("materialize-surface"); add_common(p); p.add_argument("--model", default=DEFAULT_MODEL); p.set_defaults(func=materialize_surface)
    p = sub.add_parser("validate"); add_common(p); p.add_argument("--pilot", action="store_true")
    p.add_argument("--semantic-threshold", type=float, default=SEMANTIC_THRESHOLD)
    p.add_argument("--embedding-model", default="intfloat/multilingual-e5-small")
    p.add_argument("--skip-semantic", action="store_true"); p.set_defaults(func=validate_and_dedupe)
    p = sub.add_parser("export-splits"); add_common(p); p.set_defaults(func=export_splits)
    p = sub.add_parser("golden-candidates"); add_common(p); p.add_argument("--golden-images", type=int, default=320); p.set_defaults(func=golden_candidates)
    p = sub.add_parser("finalize"); add_common(p); p.set_defaults(func=finalize_manifest)
    p = sub.add_parser("verify"); add_common(p); p.set_defaults(func=verify)
    args = parser.parse_args(); args.func(args)


if __name__ == "__main__":
    main()
