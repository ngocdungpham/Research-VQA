#!/usr/bin/env python3
"""Build derived_v4.2 with fact-locked, per-item LLM surface realization.

This script deliberately does not modify derived_v4_1.  It has four stages:

  blueprints -> realize (resumable) -> validate -> audit

Only ``question`` and ``reasoning_text`` may be returned by the LLM.  All
semantic/provenance fields are copied from a protected blueprint after its
SHA-256 seal has been checked.  A release is not complete until every
blueprint has an LLM candidate and the audit reports zero failures.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "derived_v4_1"
FACTS_PATH = ROOT / "derived_v4" / "facts_v1.jsonl.gz"
SPLITS_PATH = ROOT / "derived_v3" / "split_assignments.jsonl.gz"
OUTPUT_DIR = ROOT / "derived_v4_2"
DATASET_VERSION = "derived_v4.2"
BLUEPRINT_VERSION = "bronchoscopy_qa_blueprints_v1.2"
DEFAULT_BASE_URL = "http://127.0.0.1:20128/v1"
DEFAULT_MODEL = "gen-VQA"
PROTECTED_FIELDS = (
    "source_fact_ids", "answer_structured", "answer_text", "options",
    "evidence_region_ids",
)

ANATOMY_VI = {
    "trachea": "khí quản", "carina": "carina", "right_main_bronchus": "phế quản gốc phải",
    "left_main_bronchus": "phế quản gốc trái", "right_upper_lobe_bronchus": "phế quản thùy trên phải",
    "right_middle_lobe_bronchus": "phế quản thùy giữa phải", "right_lower_lobe_bronchus": "phế quản thùy dưới phải",
    "left_upper_lobe_bronchus": "phế quản thùy trên trái", "left_lower_lobe_bronchus": "phế quản thùy dưới trái",
    "left_lingula_segment_bronchus": "phế quản phân thùy lưỡi trái", "intermediate_bronchus": "phế quản trung gian",
    "vocal_cords": "dây thanh", "left_apical_segment_bronchus": "phế quản phân thùy đỉnh trái",
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
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


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


ATTRIBUTE_PROMPTS: dict[str, tuple[str, list[list[str]]]] = {
    "secretion_type": (
        "Loại dịch tiết nào được ghi nhận{location}?",
        [["loại"], ["dịch tiết"]],
    ),
    "secretion_color": (
        "Màu của dịch tiết được ghi nhận{location} là gì?",
        [["màu"], ["dịch tiết"]],
    ),
    "secretion_consistency": (
        "Độ đặc hoặc hình thái của dịch tiết{location} là gì?",
        [["độ đặc", "hình thái", "tính chất"], ["dịch tiết"]],
    ),
    "tumor_morphology": (
        "Hình thái của khối u khí phế quản{location} là gì?",
        [["hình thái", "dạng"], ["khối u"]],
    ),
    "stenosis_cause": (
        "Nguyên nhân hình thái của hẹp lòng phế quản{location} được gắn nhãn là gì?",
        [["nguyên nhân", "cơ chế", "hình thái"], ["hẹp"]],
    ),
}


SYSTEM_PROMPT = """Bạn diễn đạt lại từng mẫu VQA nội soi phế quản đã được khóa bằng fact.

Ràng buộc bắt buộc:
1. Chỉ viết lại question_draft và reasoning_text_draft của CHÍNH TỪNG item; không dùng một câu chung cho cả lô.
2. Giữ nguyên intent, thuộc tính được hỏi, vị trí, cực tính và mức độ chắc chắn.
3. Câu hỏi không được tiết lộ đáp án. Reason chỉ nêu bằng chứng quan sát trực tiếp, không suy diễn.
4. Không thêm chẩn đoán, nguyên nhân bệnh học, tiên lượng, điều trị hoặc thông tin ngoài input.
5. Giữ nguyên qa_id và protected_sha256. Không xuất answer, options, fact IDs hay evidence IDs.
6. Question và reasoning phải thực sự được diễn đạt lại, không sao chép nguyên văn draft.
7. Không dùng Markdown.

Trả về đúng JSON:
{"items":[{"qa_id":"...","protected_sha256":"...","question":"...","reasoning_text":"..."}]}
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(handle: Any, value: Any) -> None:
    handle.write(canonical_json(value) + "\n")


def open_jsonl_append(path: Path) -> Any:
    return path.open("a", encoding="utf-8")


def surface(value: str) -> str:
    return value.strip().rstrip(".")


def location_key(facts: list[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(sorted({loc for fact in facts for loc in fact["anatomical_locations"]}))


def location_text(key: tuple[str, ...]) -> str:
    names = [ANATOMY_VI.get(loc, loc) for loc in key]
    return f" tại {join_vi(names)}" if names else ""


def protected_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row[field] for field in PROTECTED_FIELDS}


def protected_sha(row: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(protected_payload(row)).encode("utf-8")).hexdigest()


def question_anchor_groups(row: dict[str, Any]) -> list[list[str]]:
    structured = row["answer_structured"]
    concept = structured.get("concept") if isinstance(structured, dict) else None
    if concept in ATTRIBUTE_PROMPTS:
        return ATTRIBUTE_PROMPTS[concept][1]
    template = row.get("template_id")
    if template == "abnormality_open_01" or template == "abnormality_presence_closed_01":
        return [["bất thường"]]
    if template == "normal_closed_01":
        return [["bình thường"]]
    if template == "stenosis_severity_single_01":
        return [["mức độ", "tỷ lệ", "khoảng"], ["hẹp"]]
    if template == "explicit_negative_closed_01":
        return [list(row.get("required_surface_terms", []))]
    return []


def source_manifests() -> dict[str, Any]:
    return {
        "derived_v4_1_manifest_sha256": sha256_file(SOURCE_DIR / "manifest.json"),
        "derived_v4_facts_sha256": sha256_file(FACTS_PATH),
        "derived_v3_split_assignments_sha256": sha256_file(SPLITS_PATH),
    }


def build_blueprints(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite version directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    split_map = {r["image_id"]: r["split"] for r in read_jsonl(SPLITS_PATH)}
    facts_by_id: dict[str, dict[str, Any]] = {}
    fact_image: dict[str, str] = {}
    for image in read_jsonl(FACTS_PATH):
        for fact in image["facts"]:
            facts_by_id[fact["fact_id"]] = fact
            fact_image[fact["fact_id"]] = image["image_id"]

    source_rows = list(read_jsonl(SOURCE_DIR / "accepted_vqa.jsonl.gz"))
    ordinary: list[dict[str, Any]] = []
    attributes: dict[tuple[str, str, tuple[str, ...]], list[dict[str, Any]]] = defaultdict(list)
    failures: list[dict[str, Any]] = []

    for row in source_rows:
        if split_map.get(row["image_id"]) != row["split"]:
            failures.append({"qa_id": row["qa_id"], "failure": "split_differs_from_derived_v3"})
            continue
        source_facts = [facts_by_id.get(fid) for fid in row["source_fact_ids"]]
        if any(f is None for f in source_facts):
            failures.append({"qa_id": row["qa_id"], "failure": "source_fact_missing"})
            continue
        if any(fact_image[fid] != row["image_id"] for fid in row["source_fact_ids"]):
            failures.append({"qa_id": row["qa_id"], "failure": "source_fact_wrong_image"})
            continue
        facts = [f for f in source_facts if f is not None]
        if row["template_id"] == "attribute_open_01":
            concepts = {f["concept"] for f in facts}
            if len(concepts) != 1 or next(iter(concepts)) not in ATTRIBUTE_PROMPTS:
                failures.append({"qa_id": row["qa_id"], "failure": "unsupported_attribute_concept", "concepts": sorted(concepts)})
                continue
            concept = next(iter(concepts))
            attributes[(row["image_id"], concept, location_key(facts))].append(row)
        else:
            ordinary.append(row)

    if failures:
        write_json(args.output_dir / "blueprint_build_failures.json", failures)
        raise ValueError(f"Blueprint construction failed for {len(failures)} source rows")

    output_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for row in ordinary:
        out = dict(row)
        out.update({
            "blueprint_version": BLUEPRINT_VERSION,
            "source_dataset": "derived_v4.1",
            "source_qa_ids": [row["qa_id"]],
            "answer_text": row["answer_text"],
            "question_anchor_groups": [],
        })
        for field in ("question", "reasoning_text", "llm_trace", "validation_status"):
            out.pop(field, None)
        out["protected_sha256"] = protected_sha(out)
        output_rows.append(out)

    for (image_id, concept, loc_key), rows in sorted(attributes.items()):
        template, anchors = ATTRIBUTE_PROMPTS[concept]
        facts = [facts_by_id[fid] for row in rows for fid in row["source_fact_ids"]]
        unique_facts = {f["fact_id"]: f for f in facts}
        values = list(dict.fromkeys(f["value"] for f in unique_facts.values()))
        terms = list(dict.fromkeys(surface(f["surface_vi"]) for f in unique_facts.values()))
        loc_text = location_text(loc_key)
        base = dict(sorted(rows, key=lambda x: x["qa_id"])[0])
        merged = len(values) > 1
        if merged:
            base["qa_id"] = stable_id("qa", DATASET_VERSION, image_id, concept, canonical_json(values), canonical_json(loc_key))
            base["answer_structured"] = {"concept": concept, "values": values}
            base["answer_text"] = join_vi([t.capitalize() for t in terms]) + "."
            base["answer_text_draft"] = base["answer_text"]
            base["canonical_answer"] = canonical_json(base["answer_structured"])
            base["source_fact_ids"] = sorted(unique_facts)
            base["evidence_region_ids"] = sorted({rid for f in unique_facts.values() for rid in f["evidence_region_ids"]})
            base["required_surface_terms"] = terms
            counts["multi_value_groups_merged"] += 1
            counts["source_rows_merged"] += len(rows)
        base.update({
            "blueprint_version": BLUEPRINT_VERSION,
            "template_id": f"{concept}_open_01",
            "question_intent": concept,
            "question_draft": template.format(location=loc_text),
            "reasoning_text_draft": f"Vùng bằng chứng{loc_text} cho thấy {join_vi(terms)}.",
            "source_dataset": "derived_v4.1",
            "source_qa_ids": sorted(r["qa_id"] for r in rows),
            "question_anchor_groups": anchors,
        })
        for field in ("question", "reasoning_text", "llm_trace", "validation_status"):
            base.pop(field, None)
        base["protected_sha256"] = protected_sha(base)
        output_rows.append(base)
        counts[f"attribute:{concept}"] += 1

    # The draft itself must already be a deterministic, unambiguous semantic layer.
    collision: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in output_rows:
        collision[(row["image_id"], normalize_text(row["question_draft"]))].append(row)
    bad = []
    for (image_id, question), rows in collision.items():
        answers = {r["canonical_answer"] for r in rows}
        if len(answers) > 1:
            bad.append({"image_id": image_id, "normalized_question": question, "qa_ids": [r["qa_id"] for r in rows], "answers": sorted(answers)})
    if bad:
        write_json(args.output_dir / "blueprint_question_contradictions.json", bad)
        raise ValueError(f"Still found {len(bad)} ambiguous blueprint question keys")

    path = args.output_dir / "qa_blueprints_v1_2.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as out:
        for row in sorted(output_rows, key=lambda r: (r["split"], r["image_id"], r["qa_id"])):
            write_jsonl(out, row)
            counts["blueprints"] += 1
            counts[f"split:{row['split']}"] += 1

    manifest = {
        "stage": "blueprints", "dataset_version": DATASET_VERSION,
        "blueprint_version": BLUEPRINT_VERSION, "status": "awaiting_llm_surface_realization",
        "created_at_utc": now(), "source_rows": len(source_rows),
        "counts": dict(sorted(counts.items())), "source_checksums": source_manifests(),
        "output": {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)},
        "protected_fields": list(PROTECTED_FIELDS),
        "uniqueness_key": ["image_id", "normalize(question)"],
    }
    write_json(args.output_dir / "blueprint_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


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
            continue
    raise ValueError("No JSON object in model response")


def call_llm(args: argparse.Namespace, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    items = [{
        "qa_id": r["qa_id"], "protected_sha256": r["protected_sha256"],
        "question_format": r["question_format"], "question_intent": r["question_intent"],
        "question_draft": r["question_draft"], "reasoning_text_draft": r["reasoning_text_draft"],
        "question_anchor_groups": question_anchor_groups(r),
        "required_reason_terms": r["required_surface_terms"],
    } for r in rows]
    payload = {
        "model": args.model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": canonical_json({"items": items})}],
        "temperature": args.temperature,
        "max_completion_tokens": max(1200, 220 * len(items)),
        "stream": False,
    }
    last: Exception | None = None
    for attempt in range(args.retries + 1):
        try:
            request = urllib.request.Request(
                args.base_url.rstrip("/") + "/chat/completions",
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            parsed = extract_json(body["choices"][0]["message"].get("content") or "")
            if not isinstance(parsed.get("items"), list):
                raise ValueError("Model output has no items list")
            trace = {
                "requested_model": args.model, "response_model": body.get("model"),
                "response_id": body.get("id"), "usage": body.get("usage"),
                "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
            }
            return parsed["items"], trace
        except Exception as exc:
            last = exc
            if attempt < args.retries:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(str(last))


def realize(args: argparse.Namespace) -> None:
    blueprints = list(read_jsonl(args.output_dir / "qa_blueprints_v1_2.jsonl.gz"))
    facts: dict[str, dict[str, Any]] = {}
    for image in read_jsonl(FACTS_PATH):
        facts.update({f["fact_id"]: f for f in image["facts"]})
    candidate_path = args.output_dir / "surface_candidates.jsonl"
    error_path = args.output_dir / "surface_errors.jsonl"
    done = set()
    if candidate_path.exists():
        done = {r["qa_id"] for r in read_jsonl(candidate_path)}
    pending = [r for r in blueprints if r["qa_id"] not in done]
    if args.limit:
        pending = pending[:args.limit]
    counts = Counter(existing=len(done))
    with open_jsonl_append(candidate_path) as out, open_jsonl_append(error_path) as errors:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start:start + args.batch_size]
            try:
                returned, trace = call_llm(args, batch)
                by_id = {str(x.get("qa_id")): x for x in returned if isinstance(x, dict)}
                for bp in batch:
                    item = by_id.get(bp["qa_id"])
                    if item is None:
                        write_jsonl(errors, {"at": now(), "qa_id": bp["qa_id"], "error": "missing_from_model_output"})
                        counts["errors"] += 1
                        continue
                    candidate = {
                        "qa_id": bp["qa_id"], "protected_sha256": item.get("protected_sha256"),
                        "question": item.get("question"), "reasoning_text": item.get("reasoning_text"),
                        "llm_trace": trace,
                    }
                    failures = candidate_failures(bp, candidate, facts)
                    if failures:
                        write_jsonl(errors, {"at": now(), "qa_id": bp["qa_id"], "error": "candidate_validator_failed", "failure_reasons": failures})
                        errors.flush()
                        counts["validator_rejected"] += 1
                        continue
                    write_jsonl(out, candidate)
                    out.flush()
                    counts["new_candidates"] += 1
            except Exception as exc:
                for bp in batch:
                    write_jsonl(errors, {"at": now(), "qa_id": bp["qa_id"], "error": str(exc)})
                errors.flush()
                counts["errors"] += len(batch)
            print(f"LLM {min(start + len(batch), len(pending))}/{len(pending)}", flush=True)
    unique_candidates = {r["qa_id"] for r in read_jsonl(candidate_path)} if candidate_path.exists() else set()
    manifest = {
        "stage": "per_item_llm_surface_realization", "created_at_utc": now(),
        "model": args.model, "temperature": args.temperature,
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "blueprints": len(blueprints), "unique_candidates": len(unique_candidates),
        "pending": len(blueprints) - len(unique_candidates), "run_counts": dict(counts),
    }
    write_json(args.output_dir / "surface_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def contains_any(text: str, terms: list[str]) -> bool:
    norm = normalize_text(text)
    return any(normalize_text(term) in norm for term in terms)


def fact_semantic_failures(row: dict[str, Any], facts: dict[str, dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    selected = [facts.get(fid) for fid in row["source_fact_ids"]]
    if any(f is None for f in selected):
        return ["source_fact_missing"]
    selected = [f for f in selected if f is not None]
    if sorted({rid for f in selected for rid in f["evidence_region_ids"]}) != sorted(row["evidence_region_ids"]):
        failures.append("evidence_region_ids_not_fact_union")
    structured = row["answer_structured"]
    if isinstance(structured, dict) and "concept" in structured:
        if {f["concept"] for f in selected} != {structured["concept"]}:
            failures.append("answer_concept_contradicts_fact")
        fact_values = {canonical_json(f["value"]) for f in selected}
        if "value" in structured and canonical_json(structured["value"]) not in fact_values:
            failures.append("answer_value_contradicts_fact")
        if "values" in structured and {canonical_json(v) for v in structured["values"]} != fact_values:
            failures.append("answer_values_contradict_facts")
        if "value" in structured or "values" in structured:
            for fact in selected:
                if normalize_text(surface(fact["surface_vi"])) not in normalize_text(row["answer_text"]):
                    failures.append("answer_text_contradicts_fact_surface")
        if "presence" in structured:
            expected = "present" if structured["presence"] else "absent"
            if any(f["polarity"] != expected for f in selected):
                failures.append("answer_presence_contradicts_fact_polarity")
        if "option" in structured:
            option = structured["option"]
            expected_answer = f"{option}. {row['options'].get(option, '')}."
            if option not in row["options"] or row["answer_text"] != expected_answer:
                failures.append("answer_option_inconsistent")
    if isinstance(structured, dict) and "normal" in structured:
        expected_value = structured["normal"]
        if any(f["concept"] != "image_normal" or f["value"] is not expected_value for f in selected):
            failures.append("answer_normality_contradicts_fact")
    if isinstance(structured, dict) and "observations" in structured:
        expected = {(x["concept"], x["polarity"]) for x in structured["observations"]}
        observed = {(f["concept"], f["polarity"]) for f in selected}
        if expected != observed:
            failures.append("answer_observations_contradict_facts")
    return failures


def candidate_failures(bp: dict[str, Any], candidate: dict[str, Any], facts: dict[str, dict[str, Any]]) -> list[str]:
    failures = fact_semantic_failures(bp, facts)
    if candidate.get("protected_sha256") != bp["protected_sha256"] or protected_sha(bp) != bp["protected_sha256"]:
        failures.append("protected_field_seal_mismatch")
    question = candidate.get("question")
    reason = candidate.get("reasoning_text")
    if not isinstance(question, str) or not question.strip():
        failures.append("missing_question")
    if not isinstance(reason, str) or not reason.strip():
        failures.append("missing_reasoning_text")
    if failures and (not isinstance(question, str) or not isinstance(reason, str)):
        return sorted(set(failures))
    if normalize_text(question) == normalize_text(bp["question_draft"]):
        failures.append("question_not_paraphrased")
    if normalize_text(reason) == normalize_text(bp["reasoning_text_draft"]):
        failures.append("reason_not_paraphrased")
    combined = normalize_text(question + " " + reason)
    for forbidden in FORBIDDEN_TEXT:
        if normalize_text(forbidden) in combined:
            failures.append(f"forbidden_term:{forbidden}")
    for group in question_anchor_groups(bp):
        if not contains_any(question, group):
            failures.append("question_lost_attribute_anchor:" + "|".join(group))
    # If the protected draft localizes the finding, the paraphrase must retain
    # every named anatomical location. Dropping it can recreate ambiguous QA.
    if " tại " in bp["question_draft"].lower():
        selected = [facts[fid] for fid in bp["source_fact_ids"] if fid in facts]
        for loc in sorted({loc for fact in selected for loc in fact["anatomical_locations"]}):
            loc_vi = ANATOMY_VI.get(loc, loc)
            if not contains_any(question, [loc_vi]):
                failures.append(f"question_lost_location:{loc}")
    for term in bp["required_surface_terms"]:
        if not contains_any(reason, [term]):
            failures.append(f"reason_missing_fact:{term}")
    answer = bp["answer_text"].rstrip(".")
    if answer not in ("Có", "Không") and normalize_text(answer) in normalize_text(question):
        failures.append("answer_leakage_in_question")
    # Simple polarity contradiction screen; semantic fields are checked separately above.
    structured = bp["answer_structured"]
    if isinstance(structured, dict) and structured.get("presence") is False and re.search(r"\b(có|hiện diện)\b", normalize_text(reason)) and "không" not in normalize_text(reason):
        failures.append("reason_polarity_contradiction")
    return sorted(set(failures))


def validate(args: argparse.Namespace) -> None:
    paths = [args.output_dir / name for name in ("accepted_vqa.jsonl.gz", "rejected_vqa.jsonl.gz", "review_queue.jsonl.gz", "validation_manifest.json")]
    if any(p.exists() for p in paths):
        raise FileExistsError("Validation outputs already exist; version outputs are immutable")
    blueprints = {r["qa_id"]: r for r in read_jsonl(args.output_dir / "qa_blueprints_v1_2.jsonl.gz")}
    candidates: dict[str, dict[str, Any]] = {}
    candidate_dupes = set()
    candidate_path = args.output_dir / "surface_candidates.jsonl"
    if candidate_path.exists():
        for row in read_jsonl(candidate_path):
            if row["qa_id"] in candidates:
                candidate_dupes.add(row["qa_id"])
            candidates[row["qa_id"]] = row
    facts: dict[str, dict[str, Any]] = {}
    for image in read_jsonl(FACTS_PATH):
        facts.update({f["fact_id"]: f for f in image["facts"]})

    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for qa_id, bp in blueprints.items():
        candidate = candidates.get(qa_id)
        if candidate is None:
            rejected.append({**bp, "validation_status": "rejected", "rejection_reasons": ["missing_llm_candidate"]})
            continue
        failures = candidate_failures(bp, candidate, facts)
        if qa_id in candidate_dupes:
            failures.append("duplicate_candidate_qa_id")
        row = {
            **bp, "question": candidate.get("question"), "reasoning_text": candidate.get("reasoning_text"),
            "llm_trace": candidate.get("llm_trace"),
        }
        # Protected fields come only from bp; candidate never supplies them.
        if failures:
            rejected.append({**row, "validation_status": "rejected", "rejection_reasons": sorted(set(failures))})
        else:
            valid.append({**row, "validation_status": "accepted"})

    # Lock (image_id, normalize(question)). Any collision is withheld from release.
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        groups[(row["image_id"], normalize_text(row["question"]))].append(row)
    review_ids = {row["qa_id"] for rows in groups.values() if len(rows) > 1 for row in rows}
    review = []
    accepted = []
    for row in valid:
        if row["qa_id"] in review_ids:
            review.append({**row, "validation_status": "review_required", "review_reasons": ["image_question_key_collision"]})
        else:
            accepted.append(row)

    for path, rows in ((paths[0], accepted), (paths[1], rejected), (paths[2], review)):
        with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as out:
            for row in sorted(rows, key=lambda r: (r["split"], r["image_id"], r["qa_id"])):
                write_jsonl(out, row)
    reason_counts = Counter(reason for row in rejected for reason in row["rejection_reasons"])
    manifest = {
        "stage": "protected_validation_and_contradiction_screen", "created_at_utc": now(),
        "blueprints": len(blueprints), "candidates": len(candidates), "accepted": len(accepted),
        "rejected": len(rejected), "review": len(review), "failure_counts": dict(sorted(reason_counts.items())),
        "outputs": {p.name: sha256_file(p) for p in paths[:3]},
    }
    write_json(paths[3], manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def audit(args: argparse.Namespace) -> None:
    accepted_path = args.output_dir / "accepted_vqa.jsonl.gz"
    if not accepted_path.exists():
        raise FileNotFoundError("Run validate first")
    split_map = {r["image_id"]: r["split"] for r in read_jsonl(SPLITS_PATH)}
    facts: dict[str, tuple[str, dict[str, Any]]] = {}
    for image in read_jsonl(FACTS_PATH):
        for fact in image["facts"]:
            facts[fact["fact_id"]] = (image["image_id"], fact)
    rows = list(read_jsonl(accepted_path))
    failures: Counter[str] = Counter()
    question_keys: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    qa_ids = Counter(r["qa_id"] for r in rows)
    failures["duplicate_qa_id"] = sum(v - 1 for v in qa_ids.values() if v > 1)
    for row in rows:
        if row["split"] != split_map.get(row["image_id"]):
            failures["split_differs_from_derived_v3"] += 1
        if protected_sha(row) != row["protected_sha256"]:
            failures["protected_field_seal_mismatch"] += 1
        for fid in row["source_fact_ids"]:
            if fid not in facts:
                failures["source_fact_missing"] += 1
            elif facts[fid][0] != row["image_id"]:
                failures["source_fact_wrong_image"] += 1
        for failure in fact_semantic_failures(row, {fid: fact for fid, (_, fact) in facts.items()}):
            failures[failure] += 1
        question_keys[(row["image_id"], normalize_text(row["question"]))].append(row)
    failures["image_normalized_question_collision"] = sum(len(v) for v in question_keys.values() if len(v) > 1)
    failures = Counter({k: v for k, v in failures.items() if v})
    report = {
        "dataset_version": DATASET_VERSION, "created_at_utc": now(), "accepted_rows": len(rows),
        "status": "PASS" if not failures else "FAIL", "failure_counts": dict(sorted(failures.items())),
        "invariants": {
            "split_preserved_from": "derived_v3.0", "protected_fields": list(PROTECTED_FIELDS),
            "unique_key": "(image_id, normalize(question))", "per_item_llm_required": True,
        },
    }
    write_json(args.output_dir / "audit_v4_2.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


def audit_blueprints(args: argparse.Namespace) -> None:
    """Audit the deterministic semantic layer before any LLM is allowed."""
    rows = list(read_jsonl(args.output_dir / "qa_blueprints_v1_2.jsonl.gz"))
    split_map = {r["image_id"]: r["split"] for r in read_jsonl(SPLITS_PATH)}
    facts: dict[str, dict[str, Any]] = {}
    for image in read_jsonl(FACTS_PATH):
        facts.update({f["fact_id"]: f for f in image["facts"]})
    source_ids = Counter(source_id for row in rows for source_id in row["source_qa_ids"])
    expected_source_ids = {r["qa_id"] for r in read_jsonl(SOURCE_DIR / "accepted_vqa.jsonl.gz")}
    qa_ids = Counter(r["qa_id"] for r in rows)
    keys: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    failures: Counter[str] = Counter()
    failures["duplicate_qa_id"] = sum(v - 1 for v in qa_ids.values() if v > 1)
    failures["missing_source_qa_coverage"] = len(expected_source_ids - set(source_ids))
    failures["unexpected_source_qa_id"] = len(set(source_ids) - expected_source_ids)
    failures["source_qa_mapped_more_than_once"] = sum(v - 1 for v in source_ids.values() if v > 1)
    for row in rows:
        if row["split"] != split_map.get(row["image_id"]):
            failures["split_differs_from_derived_v3"] += 1
        if protected_sha(row) != row["protected_sha256"]:
            failures["protected_field_seal_mismatch"] += 1
        for failure in fact_semantic_failures(row, facts):
            failures[failure] += 1
        if row["template_id"] == "attribute_open_01":
            failures["generic_attribute_template_remaining"] += 1
        keys[(row["image_id"], normalize_text(row["question_draft"]))].append(row)
    failures["image_normalized_question_collision"] = sum(len(v) for v in keys.values() if len(v) > 1)
    failures = Counter({key: value for key, value in failures.items() if value})
    report = {
        "dataset_version": DATASET_VERSION, "layer": "protected_blueprints",
        "created_at_utc": now(), "blueprints": len(rows), "source_qa_rows": len(expected_source_ids),
        "source_qa_coverage": len(source_ids), "status": "PASS" if not failures else "FAIL",
        "failure_counts": dict(sorted(failures.items())),
        "checks": [
            "source fact and answer semantic agreement", "protected-field SHA-256 seals",
            "split identity with derived_v3", "complete one-time source QA coverage",
            "no generic attribute template", "unique (image_id, normalize(question_draft))",
        ],
    }
    write_json(args.output_dir / "blueprint_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name, func in (("blueprints", build_blueprints), ("audit-blueprints", audit_blueprints), ("validate", validate), ("audit", audit)):
        q = sub.add_parser(name)
        q.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
        q.set_defaults(func=func)
    q = sub.add_parser("realize")
    q.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    q.add_argument("--base-url", default=DEFAULT_BASE_URL)
    q.add_argument("--model", default=DEFAULT_MODEL)
    q.add_argument("--api-key", default=os.environ.get("VQA_LLM_API_KEY", "local-vqa-v4-2"))
    q.add_argument("--batch-size", type=int, default=8)
    q.add_argument("--temperature", type=float, default=0.35)
    q.add_argument("--retries", type=int, default=2)
    q.add_argument("--timeout", type=int, default=240)
    q.add_argument("--limit", type=int, default=0)
    q.set_defaults(func=realize)
    return p


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
