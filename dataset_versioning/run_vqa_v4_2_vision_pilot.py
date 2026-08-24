#!/usr/bin/env python3
"""Run the 100-image multimodal, fact-grounded VQA v4.2 pilot.

The pilot is isolated from the text-only v4.2 surface run. One request contains
one original image, one numbered bbox overlay, all locked QA plans for that
image, and the exact supporting facts. Results are checkpointed per image.
"""

from __future__ import annotations

import argparse
import base64
import csv
import gzip
import hashlib
import json
import math
import mimetypes
import os
import re
import time
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw

import build_vqa_v4_2 as base


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "derived_v4_2_vision_pilot"
BLUEPRINTS = ROOT / "derived_v4_2" / "qa_blueprints_v1_2.jsonl.gz"
FACTS = ROOT / "derived_v4" / "facts_v1.jsonl.gz"
CANONICAL = ROOT / "canonical_v2_2" / "images.jsonl.gz"
DERIVATIONS = ROOT / "derived_v2" / "region_derivations.jsonl.gz"
PROMPT_DOC = ROOT / "Paper" / "PROMPT_VQA_V4_2_VISION_FACT_GROUNDED_DRAFT.md"
MODEL = "GenVQAVer2"
BASE_URL = "http://127.0.0.1:20128/v1"
PILOT_SPLIT_QUOTAS = {"train": 70, "validation": 15, "test": 15}
PILOT_IMAGES = 100
MIN_AUTOMATIC_ACCEPTANCE = 0.95
ALLOWED_REVIEW_REASONS = {
    "image_unclear", "evidence_not_visible", "evidence_region_mismatch",
    "fact_pixel_conflict", "ambiguous_location", "unsafe_clinical_inference",
}
UNSAFE_DEEP_DISEASE_TERMS = {
    "ung thư", "carcinoma", "cancer", "sarcoma", "lymphoma", "di căn",
    "metastasis", "lao", "tuberculosis", "mô bệnh học", "tế bào học",
}

# Disease QA is enabled by policy, but only concepts in this explicit whitelist
# may produce it. The current facts_v1 table has no certified static-image
# disease fact, so this pilot correctly creates zero disease QA.
VISIBLE_DISEASE_CONCEPTS: dict[str, str] = {}
MORPHOLOGY_NOT_DISEASE = {
    "bronchial_tumor", "tumor_morphology", "mucosal_ulceration",
    "pseudomembrane", "mucosal_infiltration", "anthracotic_pigmentation",
    "tracheomalacia", "vocal_cord_paralysis",
}


SYSTEM_PROMPT = """Bạn là trợ lý AI tạo dữ liệu Visual Question Answering cho ảnh nội soi phế quản.
Bạn nhận một ảnh gốc, một ảnh overlay có bounding box được đánh số, fact package và các QA plan đã khóa.

MỤC TIÊU
Với từng QA plan, quan sát ảnh và đúng vùng evidence được chỉ định, rồi diễn đạt lại question_draft và reasoning_text_draft thành tiếng Việt tự nhiên. Viết visual_evidence_summary mô tả ngắn dấu hiệu thực sự nhìn thấy trong vùng evidence. Không tạo thêm hoặc bỏ QA plan.

NGUỒN SỰ THẬT VÀ AN TOÀN
1. Fact package và protected fields là ground truth khóa. Pixel dùng để mô tả biểu hiện trực quan và đánh giá evidence có đủ rõ hay không.
2. Nếu ảnh không đủ rõ, evidence không phù hợp, hoặc pixel có vẻ mâu thuẫn fact, trả review_required; không tự sửa answer/evidence.
3. Chỉ được gọi tên bệnh khi qa_plan có disease_policy=allowed_visible_disease. Nếu không, chỉ mô tả abnormality nhìn thấy.
4. Tuyệt đối không suy ra ung thư, carcinoma, lao, mô bệnh học, tế bào học, di căn, tiên lượng hoặc điều trị từ morphology đại thể.

RÀNG BUỘC
1. Giữ nguyên image_id, qa_plan_id và qa_id. protected_sha256 do hệ thống quản lý, không xuất trường này.
2. Không đổi question format, intent, answer, options, source facts hoặc evidence.
3. Question giữ concept, attribute và vị trí giải phẫu; không chứa answer.
4. Reason và visual_evidence_summary chỉ mô tả bằng chứng trực tiếp, đúng polarity, không nhắc annotation/fact/label/bbox/report/dữ liệu được cung cấp.
5. Question của từng item PHẢI được viết lại với cấu trúc khác question_draft; sau khi bỏ hoa/thường và dấu câu, hai chuỗi không được giống nhau.
6. Reasoning_text của từng item PHẢI được viết lại với cấu trúc khác reasoning_text_draft; sau khi bỏ hoa/thường và dấu câu, hai chuỗi không được giống nhau.
7. Mọi chuỗi trong required_reason_terms phải xuất hiện NGUYÊN VĂN trong reasoning_text. Có thể thêm mô tả trực quan nhưng không được thay required term bằng từ đồng nghĩa.
8. Chỉ xuất JSON hợp lệ, không Markdown.

OUTPUT
{"image_id":"...","items":[{"qa_plan_id":"...","qa_id":"...","status":"accepted|review_required","review_reasons":[],"question":"...","reasoning_text":"...","visual_evidence_summary":"..."}]}

Nếu status=review_required thì question, reasoning_text, visual_evidence_summary phải null và review_reasons chỉ dùng: image_unclear, evidence_not_visible, evidence_region_mismatch, fact_pixel_conflict, ambiguous_location, unsafe_clinical_inference.
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(base.canonical_json(value) + "\n")


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def file_sha(path: Path) -> str:
    return base.sha256_file(path)


def script_sha() -> str:
    return file_sha(Path(__file__).resolve())


def plan_id(qa_id: str) -> str:
    return base.stable_id("plan", "vision_fact_pilot_v1", qa_id)


def choose_images(rows_by_image: dict[str, list[dict[str, Any]]], quota: int) -> list[str]:
    candidates = sorted(rows_by_image)
    global_intents = Counter(row["question_intent"] for rows in rows_by_image.values() for row in rows)
    chosen: list[str] = []
    selected_intents: Counter[str] = Counter()
    while candidates and len(chosen) < quota:
        def score(image_id: str) -> tuple[float, str]:
            intents = {row["question_intent"] for row in rows_by_image[image_id]}
            diversity = sum(1.0 / ((1 + selected_intents[i]) * math.sqrt(global_intents[i])) for i in intents)
            tie = hashlib.sha256(("vision-pilot-v1\0" + image_id).encode()).hexdigest()
            return diversity, tie
        selected = max(candidates, key=score)
        candidates.remove(selected)
        chosen.append(selected)
        selected_intents.update({row["question_intent"] for row in rows_by_image[selected]})
    if len(chosen) != quota:
        raise ValueError(f"Could only select {len(chosen)} of {quota} requested images")
    return chosen


def rounded_bbox(box: list[float]) -> list[int]:
    return [round(x) for x in box]


def normalized_bbox(box: list[int], width: int, height: int) -> list[int]:
    return [
        round(1000 * box[0] / width), round(1000 * box[1] / height),
        round(1000 * box[2] / width), round(1000 * box[3] / height),
    ]


def polygon_bbox(points: list[list[float]]) -> list[float]:
    xs = [p[0] for p in points]; ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def build_evidence_map(canonical: dict[str, dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    evidence = {}
    for derivation in read_jsonl(DERIVATIONS):
        image_id = derivation["image_id"]
        if image_id not in canonical:
            continue
        image = canonical[image_id]
        regions = image["regions_by_id"]
        for region in derivation["regions"]:
            source_id = region.get("training_geometry_source_region_id")
            if not source_id:
                source_ids = region.get("source_region_ids", [])
                source_id = source_ids[0] if source_ids else None
            source = regions.get(source_id) if source_id else None
            if region.get("training_polygon_override"):
                bbox = polygon_bbox(region["training_polygon_override"])
            elif source:
                bbox = source["bbox_xyxy"]
            else:
                continue
            source_annotation_ids = sorted({
                annotation_id
                for sid in region.get("source_region_ids", [])
                for annotation_id in regions.get(sid, {}).get("source_annotation_ids", [])
            })
            evidence[(image_id, region["canonical_region_id"])] = {
                "bbox_xyxy_pixels": rounded_bbox(bbox),
                "source_region_ids": region.get("source_region_ids", []),
                "source_annotation_ids": source_annotation_ids,
                "training_status": region.get("training_status"),
            }
    return evidence


def disease_policy(facts: list[dict[str, Any]]) -> dict[str, Any]:
    disease_concepts = sorted({f["concept"] for f in facts if f["concept"] in VISIBLE_DISEASE_CONCEPTS})
    blocked_morphology = sorted({f["concept"] for f in facts if f["concept"] in MORPHOLOGY_NOT_DISEASE})
    return {
        "mode": "allowed_visible_disease" if disease_concepts else "abnormality_only",
        "allowed_disease_concepts": disease_concepts,
        "blocked_from_disease_inference": blocked_morphology,
    }


def prepare(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite pilot: {args.output_dir}")
    (args.output_dir / "overlays").mkdir(parents=True)
    (args.output_dir / "results").mkdir()

    blueprints = list(read_jsonl(BLUEPRINTS))
    split_by_image = {row["image_id"]: row["split"] for row in blueprints}
    excluded_images = set()
    exclusion_checksums = {}
    for exclusion_path in args.exclude_requests:
        exclusion_path = exclusion_path.resolve()
        excluded_images.update(row["image_id"] for row in read_jsonl(exclusion_path))
        exclusion_checksums[str(exclusion_path.relative_to(ROOT))] = file_sha(exclusion_path)
    by_split_image: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in blueprints:
        if row["image_id"] in excluded_images:
            continue
        by_split_image[row["split"]][row["image_id"]].append(row)
    cohort_size = getattr(args, "cohort_size", PILOT_IMAGES)
    available = {split: len(rows) for split, rows in by_split_image.items()}
    target_images = min(cohort_size, sum(available.values()))
    if target_images <= 0:
        raise ValueError("No unprocessed images remain")
    weights = {split: quota / PILOT_IMAGES for split, quota in PILOT_SPLIT_QUOTAS.items()}
    split_quotas = {
        split: min(available.get(split, 0), int(target_images * weights[split]))
        for split in PILOT_SPLIT_QUOTAS
    }
    while sum(split_quotas.values()) < target_images:
        candidates = [split for split in PILOT_SPLIT_QUOTAS if split_quotas[split] < available.get(split, 0)]
        if not candidates:
            break
        split = max(
            candidates,
            key=lambda name: (target_images * weights[name] - split_quotas[name], available[name] - split_quotas[name], name),
        )
        split_quotas[split] += 1
    selected = []
    for split, quota in split_quotas.items():
        selected.extend(choose_images(by_split_image[split], quota))
    if len(selected) != target_images or len(set(selected)) != target_images:
        raise ValueError(f"Cohort selection is not exactly {target_images} unique images")

    canonical = {}
    for image in read_jsonl(CANONICAL):
        if image["image_id"] in set(selected):
            canonical[image["image_id"]] = {**image, "regions_by_id": {r["region_id"]: r for r in image["regions"]}}
    evidence_map = build_evidence_map(canonical)
    facts_by_id = {}
    for image in read_jsonl(FACTS):
        if image["image_id"] in set(selected):
            facts_by_id.update({f["fact_id"]: f for f in image["facts"]})

    requests_path = args.output_dir / "pilot_requests.jsonl.gz"
    counts = Counter()
    with gzip.open(requests_path, "wt", encoding="utf-8", compresslevel=9) as out:
        for image_id in selected:
            rows = sorted(by_split_image[split_by_image[image_id]][image_id], key=lambda x: x["qa_id"])
            split = rows[0]["split"]
            image = canonical[image_id]
            image_path = ROOT / image["image_path"]
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            fact_ids = sorted({fid for row in rows for fid in row["source_fact_ids"]})
            facts = [facts_by_id[fid] for fid in fact_ids]
            evidence_ids = sorted({rid for row in rows for rid in row["evidence_region_ids"]})
            regions = []
            with Image.open(image_path) as source_image:
                overlay = source_image.convert("RGB")
                draw = ImageDraw.Draw(overlay)
                for index, region_id in enumerate(evidence_ids, 1):
                    info = evidence_map.get((image_id, region_id))
                    if not info:
                        raise ValueError(f"Missing bbox for {image_id}/{region_id}")
                    bbox = info["bbox_xyxy_pixels"]
                    draw.rectangle(bbox, outline=(255, 0, 0), width=max(2, round(min(overlay.size) / 160)))
                    draw.rectangle([bbox[0], bbox[1], bbox[0] + 28, bbox[1] + 20], fill=(255, 0, 0))
                    draw.text((bbox[0] + 5, bbox[1] + 3), str(index), fill=(255, 255, 255))
                    linked_facts = [f for f in facts if region_id in f["evidence_region_ids"]]
                    regions.append({
                        "region_index": index, "evidence_region_id": region_id,
                        "bbox_xyxy_pixels": bbox,
                        "bbox_xyxy_normalized_1000": normalized_bbox(bbox, image["width"], image["height"]),
                        "anatomical_locations": sorted({loc for f in linked_facts for loc in f["anatomical_locations"]}),
                        "source_annotation_ids": info["source_annotation_ids"],
                    })
                overlay_path = args.output_dir / "overlays" / f"{image_id}.jpg"
                overlay.save(overlay_path, format="JPEG", quality=92)

            fact_payload = [{
                key: fact[key] for key in (
                    "fact_id", "scope", "fact_kind", "concept", "concept_family", "predicate", "value",
                    "surface_vi", "polarity", "certainty", "anatomical_locations", "evidence_region_ids",
                    "source_annotation_ids", "parent_fact_id",
                )
            } for fact in facts]
            plans = []
            for row in rows:
                selected_facts = [facts_by_id[fid] for fid in row["source_fact_ids"]]
                policy = disease_policy(selected_facts)
                plans.append({
                    "qa_plan_id": plan_id(row["qa_id"]), "qa_id": row["qa_id"],
                    "protected_sha256": row["protected_sha256"],
                    "question_format": row["question_format"], "question_intent": row["question_intent"],
                    "question_draft": row["question_draft"], "reasoning_text_draft": row["reasoning_text_draft"],
                    "answer_structured_locked": row["answer_structured"], "answer_text_locked": row["answer_text"],
                    "options_locked": row["options"], "source_fact_ids": row["source_fact_ids"],
                    "evidence_region_ids": row["evidence_region_ids"],
                    "required_question_anchors": row.get("question_anchor_groups") or base.question_anchor_groups(row),
                    "required_reason_terms": row["required_surface_terms"],
                    "disease_policy": policy,
                })
                counts[f"intent:{row['question_intent']}"] += 1
                counts[f"format:{row['question_format']}"] += 1
                if policy["mode"] == "allowed_visible_disease": counts["disease_plans"] += 1
            record = {
                "pilot_version": "vision_fact_grounded_pilot_v1", "image_id": image_id,
                "split": split, "image_path": image["image_path"],
                "overlay_path": str(overlay_path.relative_to(ROOT)),
                "image_metadata": {"width": image["width"], "height": image["height"]},
                "evidence_regions": regions, "facts": fact_payload, "qa_plans": plans,
                "policy": {"roi_crops": False, "region_extent_fact": False, "visual_evidence_summary": True},
            }
            out.write(base.canonical_json(record) + "\n")
            counts["images"] += 1; counts[f"split:{split}"] += 1; counts["qa_plans"] += len(plans)

    prompt_snapshot = args.output_dir / "system_prompt.txt"
    prompt_snapshot.write_text(SYSTEM_PROMPT + "\n", encoding="utf-8")
    manifest = {
        "pilot_version": "vision_fact_grounded_pilot_v1", "cohort_label": args.cohort_label, "status": "PREPARED_NOT_RUN",
        "created_at_utc": now(), "selection": {"images": target_images, "split_quotas": split_quotas, "algorithm": "deterministic rarity-aware intent stratification"},
        "model": MODEL, "base_url": BASE_URL, "prompt_sha256": file_sha(prompt_snapshot),
        "script_sha256": script_sha(), "source_checksums": {
            "blueprints": file_sha(BLUEPRINTS), "facts": file_sha(FACTS),
            "canonical": file_sha(CANONICAL), "derivations": file_sha(DERIVATIONS),
            "reviewed_prompt_draft": file_sha(PROMPT_DOC),
        },
        "decisions": {
            "visual_evidence_summary": "enabled", "roi_crops": "disabled",
            "region_extent_fact": "disabled", "disease_qa": "enabled_by_explicit_visible-disease_whitelist",
            "current_visible_disease_whitelist": VISIBLE_DISEASE_CONCEPTS,
            "deep_diagnosis_policy": "blocked_or_rewritten_as_abnormality",
        },
        "stop_gate": {"cohort_images": target_images, "minimum_automatic_acceptance": MIN_AUTOMATIC_ACCEPTANCE, "protected_or_contradiction_failures": 0, "question_collisions": 0, "manual_review_required": True},
        "counts": dict(sorted(counts.items())), "requests_sha256": file_sha(requests_path),
        "excluded_images": len(excluded_images), "exclusion_request_checksums": exclusion_checksums,
    }
    write_json(args.output_dir / "pilot_manifest.json", manifest)
    (args.output_dir / "RUNBOOK.md").write_text(
        "# Vision fact-grounded pilot runbook\n\n"
        "This directory is an isolated 100-image pilot, not a dataset release.\n\n"
        "- Inputs: original image, numbered bbox overlay, facts, protected QA plans.\n"
        "- No ROI crops. No region-extent QA. Visual evidence summary enabled.\n"
        "- Disease names require an explicit visible-disease whitelist fact; deep pathology is blocked.\n"
        "- One request per image. Results checkpoint atomically in `results/`.\n"
        "- Full generation is blocked until automatic audit and manual review are complete.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def extract_json(text: str) -> dict[str, Any]:
    content = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", content, flags=re.DOTALL)
    if fenced: content = fenced.group(1)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", content):
        try:
            value, _ = decoder.raw_decode(content[match.start():])
            if isinstance(value, dict): return value
        except json.JSONDecodeError: pass
    raise ValueError("No JSON object in model response")


def call_model(args: argparse.Namespace, request_row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str]:
    payload_text = base.canonical_json({
        key: request_row[key] for key in (
            "pilot_version", "image_id", "image_metadata", "evidence_regions", "facts", "qa_plans", "policy",
        )
    })
    content = [
        {"type": "text", "text": "Ảnh nội soi gốc:"},
        {"type": "image_url", "image_url": {"url": data_url(ROOT / request_row["image_path"])}},
        {"type": "text", "text": "Ảnh overlay; số trên box khớp region_index trong payload:"},
        {"type": "image_url", "image_url": {"url": data_url(ROOT / request_row["overlay_path"])}},
        {"type": "text", "text": "FACT_GROUNDED_PAYLOAD:\n" + payload_text},
    ]
    body = {
        "model": args.model, "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}],
        "temperature": args.temperature, "max_completion_tokens": max(1800, 650 * len(request_row["qa_plans"])), "stream": False,
    }
    req = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=args.timeout) as response:
        response_body = json.loads(response.read().decode("utf-8"))
    raw = response_body["choices"][0]["message"].get("content") or ""
    parsed = extract_json(raw)
    trace = {"requested_model": args.model, "response_model": response_body.get("model"), "response_id": response_body.get("id"), "usage": response_body.get("usage"), "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()}
    return parsed, trace, raw


def output_failures(request_row: dict[str, Any], parsed: dict[str, Any], facts: dict[str, dict[str, Any]], blueprints: dict[str, dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    failures = []
    if parsed.get("image_id") != request_row["image_id"]: failures.append("image_id_mismatch")
    items = parsed.get("items")
    if not isinstance(items, list): return failures + ["items_missing"], []
    plans = {p["qa_plan_id"]: p for p in request_row["qa_plans"]}
    if len(items) != len(plans): failures.append("item_count_mismatch")
    if len({i.get("qa_plan_id") for i in items if isinstance(i, dict)}) != len(items): failures.append("duplicate_qa_plan_id")
    normalized = []
    for item in items:
        if not isinstance(item, dict): failures.append("item_not_object"); continue
        pid = item.get("qa_plan_id"); plan = plans.get(pid)
        if not plan: failures.append("unknown_qa_plan_id"); continue
        bp = blueprints[plan["qa_id"]]
        prefix = plan["qa_id"] + ":"
        if item.get("qa_id") != plan["qa_id"]: failures.append(prefix + "qa_id_mismatch")
        echoed_sha = item.get("protected_sha256")
        transport_normalizations = []
        if echoed_sha != plan["protected_sha256"]:
            transport_normalizations.append({
                "field": "protected_sha256",
                "action": "injected_from_protected_plan" if echoed_sha is None else "replaced_invalid_llm_echo",
                "llm_echo": echoed_sha,
                "system_value": plan["protected_sha256"],
            })
        status = item.get("status")
        if status == "review_required":
            reasons = item.get("review_reasons")
            if not isinstance(reasons, list) or not reasons or any(x not in ALLOWED_REVIEW_REASONS for x in reasons): failures.append(prefix + "invalid_review_reasons")
            if any(item.get(k) is not None for k in ("question", "reasoning_text", "visual_evidence_summary")): failures.append(prefix + "review_text_must_be_null")
            normalized.append({
                "qa_plan_id": pid, "qa_id": plan["qa_id"], "status": status,
                "review_reasons": reasons, "transport_normalizations": transport_normalizations,
            })
            continue
        if status != "accepted": failures.append(prefix + "invalid_status"); continue
        visual = item.get("visual_evidence_summary")
        if not isinstance(visual, str) or not visual.strip(): failures.append(prefix + "missing_visual_evidence_summary")
        candidate = {
            "protected_sha256": plan["protected_sha256"],
            "question": item.get("question"), "reasoning_text": item.get("reasoning_text"),
        }
        for failure in base.candidate_failures(bp, candidate, facts): failures.append(prefix + failure)
        combined = base.normalize_text(" ".join(str(item.get(k) or "") for k in ("question", "reasoning_text", "visual_evidence_summary")))
        if plan["disease_policy"]["mode"] != "allowed_visible_disease":
            for term in UNSAFE_DEEP_DISEASE_TERMS:
                if base.normalize_text(term) in combined: failures.append(prefix + "unsafe_disease_inference:" + term)
        normalized.append({
            "qa_plan_id": pid, "qa_id": plan["qa_id"], "status": "accepted", "review_reasons": [],
            "question": item.get("question"), "reasoning_text": item.get("reasoning_text"),
            "visual_evidence_summary": visual, "transport_normalizations": transport_normalizations,
        })
    missing = set(plans) - {x.get("qa_plan_id") for x in items if isinstance(x, dict)}
    if missing: failures.append("missing_qa_plans:" + ",".join(sorted(missing)))
    return sorted(set(failures)), normalized


def progress(output_dir: Path, total: int) -> dict[str, Any]:
    files = list((output_dir / "results").glob("*.json"))
    accepted = review = plans = 0
    for path in files:
        result = json.loads(path.read_text(encoding="utf-8")); plans += len(result["items"])
        accepted += sum(x["status"] == "accepted" for x in result["items"])
        review += sum(x["status"] == "review_required" for x in result["items"])
    return {"completed_images": len(files), "total_images": total, "pending_images": total - len(files), "completed_plans": plans, "accepted_items": accepted, "review_items": review}


def run(args: argparse.Namespace) -> None:
    prompt_path = args.output_dir / "system_prompt.txt"
    old_prompt = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""
    if old_prompt.strip() != SYSTEM_PROMPT.strip():
        old_sha = hashlib.sha256(old_prompt.encode()).hexdigest() if old_prompt else None
        prompt_path.write_text(SYSTEM_PROMPT + "\n", encoding="utf-8")
        append_jsonl(args.output_dir / "prompt_revisions.jsonl", {
            "at": now(), "old_prompt_sha256": old_sha,
            "new_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
            "reason": "Transport contract: protected_sha256 is injected and verified by the system rather than copied by the LLM",
        })
    request_rows = list(read_jsonl(args.output_dir / "pilot_requests.jsonl.gz"))
    blueprints = {r["qa_id"]: r for r in read_jsonl(BLUEPRINTS)}
    facts = {}
    for image in read_jsonl(FACTS): facts.update({f["fact_id"]: f for f in image["facts"]})
    completed = {p.stem for p in (args.output_dir / "results").glob("*.json")}
    pending = [r for r in request_rows if r["image_id"] not in completed]
    if args.limit_images: pending = pending[:args.limit_images]
    for index, row in enumerate(pending, 1):
        started = time.monotonic(); last_error = None
        for attempt in range(args.retries + 1):
            try:
                parsed, trace, raw = call_model(args, row)
                append_jsonl(args.output_dir / "raw_responses.jsonl", {"at": now(), "image_id": row["image_id"], "attempt": attempt + 1, "trace": trace, "parsed": parsed, "raw_content": raw})
                failures, items = output_failures(row, parsed, facts, blueprints)
                if failures: raise ValueError(";".join(failures))
                result = {"image_id": row["image_id"], "split": row["split"], "items": items, "llm_trace": trace, "completed_at_utc": now(), "elapsed_seconds": round(time.monotonic() - started, 3)}
                target = args.output_dir / "results" / f"{row['image_id']}.json"
                temp = target.with_suffix(".json.tmp"); write_json(temp, result); temp.replace(target)
                last_error = None; break
            except Exception as exc:
                last_error = str(exc)
                append_jsonl(args.output_dir / "attempt_errors.jsonl", {"at": now(), "image_id": row["image_id"], "attempt": attempt + 1, "error": last_error})
                if attempt < args.retries: time.sleep(2 * (attempt + 1))
        state = progress(args.output_dir, len(request_rows)); state.update({"updated_at_utc": now(), "model": args.model})
        write_json(args.output_dir / "progress.json", state)
        append_jsonl(args.output_dir / "run_events.jsonl", {"at": now(), "image_id": row["image_id"], "success": last_error is None, "error": last_error, "progress": state})
        print(json.dumps({"run_image": f"{index}/{len(pending)}", "image_id": row["image_id"], "success": last_error is None, **state}, ensure_ascii=False), flush=True)
    print(json.dumps(progress(args.output_dir, len(request_rows)), ensure_ascii=False, indent=2))


def status(args: argparse.Namespace) -> None:
    requests = list(read_jsonl(args.output_dir / "pilot_requests.jsonl.gz"))
    state = progress(args.output_dir, len(requests))
    errors = list(read_jsonl(args.output_dir / "attempt_errors.jsonl")) if (args.output_dir / "attempt_errors.jsonl").exists() else []
    state["attempt_errors"] = len(errors); state["latest_errors"] = errors[-5:]
    print(json.dumps(state, ensure_ascii=False, indent=2))


def audit(args: argparse.Namespace) -> None:
    requests = list(read_jsonl(args.output_dir / "pilot_requests.jsonl.gz"))
    request_by_image = {r["image_id"]: r for r in requests}
    blueprints = {r["qa_id"]: r for r in read_jsonl(BLUEPRINTS)}
    facts = {}
    for image in read_jsonl(FACTS): facts.update({f["fact_id"]: f for f in image["facts"]})
    prepared_manifest = json.loads((args.output_dir / "pilot_manifest.json").read_text(encoding="utf-8"))
    result_paths = list((args.output_dir / "results").glob("*.json"))
    failures = Counter(); accepted_rows = []; review_rows = []
    failures["pilot_images_incomplete"] = len(requests) - len(result_paths)
    failures["request_file_checksum_changed"] = int(file_sha(args.output_dir / "pilot_requests.jsonl.gz") != prepared_manifest["requests_sha256"])
    for key, path in (("blueprints", BLUEPRINTS), ("facts", FACTS), ("canonical", CANONICAL), ("derivations", DERIVATIONS)):
        failures[f"source_checksum_changed:{key}"] = int(file_sha(path) != prepared_manifest["source_checksums"][key])
    keys = defaultdict(list)
    response_ids = Counter(); prompt_hashes = Counter(); token_usage = Counter(); split_images = Counter(); transport_normalizations = Counter()
    for path in result_paths:
        result = json.loads(path.read_text(encoding="utf-8")); request = request_by_image[result["image_id"]]
        split_images[result["split"]] += 1
        trace = result.get("llm_trace") or {}
        response_id = trace.get("response_id")
        if response_id:
            response_ids[response_id] += 1
        prompt_hashes[trace.get("prompt_sha256") or "system_quarantine_no_prompt"] += 1
        token_usage.update(trace.get("usage") or {})
        plans = {p["qa_plan_id"]: p for p in request["qa_plans"]}
        if len(result["items"]) != len(plans): failures["result_plan_count_mismatch"] += 1
        for item in result["items"]:
            plan = plans[item["qa_plan_id"]]; bp = blueprints[plan["qa_id"]]
            for normalization in item.get("transport_normalizations") or []:
                action = normalization.get("action")
                transport_normalizations[action] += 1
                if (
                    normalization.get("field") != "protected_sha256"
                    or action not in {"injected_from_protected_plan", "replaced_invalid_llm_echo"}
                    or normalization.get("system_value") != bp["protected_sha256"]
                ):
                    failures["invalid_transport_normalization"] += 1
            locked_comparisons = {
                "question_format": plan["question_format"] == bp["question_format"],
                "question_intent": plan["question_intent"] == bp["question_intent"],
                "answer_structured": plan["answer_structured_locked"] == bp["answer_structured"],
                "answer_text": plan["answer_text_locked"] == bp["answer_text"],
                "options": plan["options_locked"] == bp["options"],
                "source_fact_ids": plan["source_fact_ids"] == bp["source_fact_ids"],
                "evidence_region_ids": plan["evidence_region_ids"] == bp["evidence_region_ids"],
                "protected_sha256": plan["protected_sha256"] == bp["protected_sha256"],
            }
            for field, okay in locked_comparisons.items():
                if not okay: failures[f"qa_plan_changed_protected_field:{field}"] += 1
            materialized = {**bp, **item, "llm_trace": result["llm_trace"], "pilot_image_id": result["image_id"]}
            if item["status"] == "accepted":
                candidate = {"protected_sha256": bp["protected_sha256"], "question": item["question"], "reasoning_text": item["reasoning_text"]}
                for failure in base.candidate_failures(bp, candidate, facts): failures[f"accepted_semantic:{failure}"] += 1
                visual = item.get("visual_evidence_summary")
                if not isinstance(visual, str) or not visual.strip(): failures["accepted_missing_visual_evidence_summary"] += 1
                combined = base.normalize_text(" ".join([item["question"], item["reasoning_text"], visual or ""]))
                if plan["disease_policy"]["mode"] != "allowed_visible_disease":
                    for term in UNSAFE_DEEP_DISEASE_TERMS:
                        if base.normalize_text(term) in combined: failures[f"accepted_unsafe_disease_inference:{term}"] += 1
                accepted_rows.append(materialized); keys[(bp["image_id"], base.normalize_text(item["question"]))].append(item["qa_id"])
            else: review_rows.append(materialized)
    failures["duplicate_response_id"] = sum(v - 1 for key, v in response_ids.items() if key and v > 1)
    expected_split_quotas = prepared_manifest["selection"]["split_quotas"]
    for split, expected in expected_split_quotas.items(): failures[f"split_image_count:{split}"] = abs(split_images[split] - expected)
    failures["image_normalized_question_collision"] = sum(len(v) for v in keys.values() if len(v) > 1)
    total_plans = len(accepted_rows) + len(review_rows)
    acceptance = len(accepted_rows) / total_plans if total_plans else 0.0
    if acceptance < MIN_AUTOMATIC_ACCEPTANCE: failures["automatic_acceptance_below_95_percent"] = 1
    failures = Counter({k: v for k, v in failures.items() if v})
    for name, rows in (("accepted_pilot.jsonl.gz", accepted_rows), ("review_pilot.jsonl.gz", review_rows)):
        with gzip.open(args.output_dir / name, "wt", encoding="utf-8", compresslevel=9) as out:
            for row in sorted(rows, key=lambda x: (x["split"], x["image_id"], x["qa_id"])): out.write(base.canonical_json(row) + "\n")
    review_csv = args.output_dir / "manual_review.csv"
    with review_csv.open("w", encoding="utf-8", newline="") as handle:
        fields = ["qa_id", "image_id", "split", "image_path", "overlay_path", "question_intent", "question_format", "question", "answer_locked", "reasoning_text", "visual_evidence_summary", "evidence_region_ids", "answer_correct", "evidence_correct", "question_natural", "reason_factual", "reviewer_notes"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in accepted_rows:
            req = request_by_image[row["image_id"]]
            writer.writerow({"qa_id": row["qa_id"], "image_id": row["image_id"], "split": row["split"], "image_path": req["image_path"], "overlay_path": req["overlay_path"], "question_intent": row["question_intent"], "question_format": row["question_format"], "question": row["question"], "answer_locked": row["answer_text"], "reasoning_text": row["reasoning_text"], "visual_evidence_summary": row["visual_evidence_summary"], "evidence_region_ids": "|".join(row["evidence_region_ids"]), "answer_correct": "", "evidence_correct": "", "question_natural": "", "reason_factual": "", "reviewer_notes": ""})
    q_types = {"closed-ended": "closed_ended_questions", "single-choice": "single_choice_questions", "multi-choice": "multi_choice_questions", "open-ended": "open_ended_questions"}
    conversation_rows = []
    for index, row in enumerate(sorted(accepted_rows, key=lambda x: (x["split"], x["image_id"], x["qa_id"]))):
        req = request_by_image[row["image_id"]]
        region_map = {r["evidence_region_id"]: r for r in req["evidence_regions"]}
        boxes = [region_map[rid]["bbox_xyxy_pixels"] for rid in row["evidence_region_ids"] if rid in region_map]
        options = row.get("options") or {}
        choices = "" if not options else " <choices>: [" + ", ".join(f"{key}: {value}" for key, value in sorted(options.items())) + "]"
        structured = row.get("answer_structured") or {}
        answer = str(structured["option"]) if row["question_format"] == "single-choice" and "option" in structured else row["answer_text"]
        conversation_rows.append({
            "conversations": [
                {"from": "human", "value": f"<image>\n{row['question']}{choices}"},
                {"from": "gpt", "value": f"<answer> {answer} <reason> {row['reasoning_text']} <visual_evidence> {row['visual_evidence_summary']} <location> {json.dumps(boxes, ensure_ascii=False)}"},
            ],
            "row_id": index, "qa_id": row["qa_id"], "q_type": q_types[row["question_format"]],
            "question_type": row["question_intent"], "image": req["image_path"], "overlay_image": req["overlay_path"],
            "image_id": row["image_id"], "patient_id": row["patient_id"], "procedure_id": row["procedure_id"], "split": row["split"],
            "answer_structured": row["answer_structured"], "answer_text": row["answer_text"], "options": options,
            "source_fact_ids": row["source_fact_ids"], "evidence_region_ids": row["evidence_region_ids"], "evidence_boxes_xyxy": boxes,
            "protected_sha256": row["protected_sha256"], "llm_trace": row["llm_trace"],
            "pilot_status": "AUTOMATED_PASS_MANUAL_REVIEW_PENDING",
        })
    conversation_path = args.output_dir / "conversations_pilot.json"
    conversation_path.write_text(json.dumps(conversation_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    automated_pass = not failures
    attempt_errors = list(read_jsonl(args.output_dir / "attempt_errors.jsonl")) if (args.output_dir / "attempt_errors.jsonl").exists() else []
    images_with_retries = len({row["image_id"] for row in attempt_errors})
    report = {
        "pilot_version": "vision_fact_grounded_pilot_v1", "created_at_utc": now(),
        "gate_status": "STOP_MANUAL_REVIEW_REQUIRED" if automated_pass else "STOP_AUTOMATED_THRESHOLDS_FAILED",
        "automatic_thresholds_passed": automated_pass, "manual_review_completed": False,
        "images": len(result_paths), "planned_images": len(requests), "total_plans": total_plans,
        "accepted": len(accepted_rows), "review_required": len(review_rows), "automatic_acceptance_rate": acceptance,
        "failure_counts": dict(sorted(failures.items())), "thresholds": {"images": prepared_manifest["selection"]["images"], "minimum_automatic_acceptance": MIN_AUTOMATIC_ACCEPTANCE, "question_collisions": 0, "manual_review_required": True},
        "policy_results": {"disease_qa_created": sum(r["question_intent"] == "disease" for r in accepted_rows), "region_extent_qa_created": 0, "roi_crops_used": False, "visual_evidence_summary_used": True},
        "execution": {"response_models": dict(Counter((json.loads(p.read_text(encoding="utf-8")).get("llm_trace") or {}).get("response_model") or "system_quarantine_no_model_response" for p in result_paths)), "prompt_hashes": dict(prompt_hashes), "attempt_errors_logged": len(attempt_errors), "images_retried": images_with_retries, "token_usage_completed_results": dict(token_usage), "transport_normalizations": dict(transport_normalizations)},
        "artifacts": {"accepted": "accepted_pilot.jsonl.gz", "review": "review_pilot.jsonl.gz", "manual_review": "manual_review.csv", "conversations": "conversations_pilot.json"},
    }
    write_json(args.output_dir / "pilot_audit.json", report)
    execution_manifest = {
        "created_at_utc": now(), "final_script_sha256": script_sha(),
        "active_system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "prepared_manifest_sha256": file_sha(args.output_dir / "pilot_manifest.json"),
        "requests_sha256": file_sha(args.output_dir / "pilot_requests.jsonl.gz"),
        "raw_responses_sha256": file_sha(args.output_dir / "raw_responses.jsonl") if (args.output_dir / "raw_responses.jsonl").is_file() else None,
        "raw_responses_present": (args.output_dir / "raw_responses.jsonl").is_file(),
        "attempt_errors_sha256": file_sha(args.output_dir / "attempt_errors.jsonl") if (args.output_dir / "attempt_errors.jsonl").is_file() else None,
        "attempt_errors_present": (args.output_dir / "attempt_errors.jsonl").is_file(),
        "run_events_sha256": file_sha(args.output_dir / "run_events.jsonl") if (args.output_dir / "run_events.jsonl").is_file() else None,
        "run_events_present": (args.output_dir / "run_events.jsonl").is_file(),
        "result_files": len(result_paths), "result_set_sha256": hashlib.sha256("\n".join(sorted(file_sha(p) for p in result_paths)).encode()).hexdigest(),
        "audit_sha256": file_sha(args.output_dir / "pilot_audit.json"),
        "accepted_pilot_sha256": file_sha(args.output_dir / "accepted_pilot.jsonl.gz"),
        "manual_review_sha256": file_sha(args.output_dir / "manual_review.csv"),
        "conversations_pilot_sha256": file_sha(args.output_dir / "conversations_pilot.json"),
    }
    write_json(args.output_dir / "execution_manifest.json", execution_manifest)
    (args.output_dir / "METHOD_LOG.md").write_text(
        "# Method log — vision fact-grounded VQA pilot\n\n"
        f"- Cohort: {len(requests)} images, {total_plans} protected QA plans; split {dict(split_images)}.\n"
        "- Model alias: GenVQAVer2; response backend: gpt-5.6-luna.\n"
        "- Input per request: one original image, one numbered bbox overlay, evidence boxes, fact package and protected QA plans.\n"
        "- Enabled: visual_evidence_summary. Disabled: ROI crops and region_extent facts/questions.\n"
        "- Disease policy: only an explicit static-image visible-disease whitelist; whitelist was empty for current facts, so morphology remained abnormality and deep pathology terms were blocked.\n"
        "- Transport: synchronous OpenAI-compatible chat completion, one image per request; atomic result checkpoint and resume by image_id.\n"
        "- Smoke test revision: prompt was tightened after the first response copied drafts and replaced required fact terms with synonyms; the failed attempt remains logged.\n"
        f"- Automated output: {len(accepted_rows)} accepted, {len(review_rows)} model-review, acceptance {acceptance:.4%}; failures {dict(failures)}.\n"
        f"- Retry history: {len(attempt_errors)} failed attempts across {images_with_retries} images; none were materialized.\n"
        f"- Transport normalization: protected SHA echoes are not trusted; system-owned seals were injected {sum(transport_normalizations.values())} times and logged.\n"
        f"- Gate: {report['gate_status']}. Full dataset generation is prohibited until manual_review.csv is completed and adjudicated.\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--exclude-requests", type=Path, action="append", default=[])
    p.add_argument("--cohort-size", type=int, default=PILOT_IMAGES)
    p.add_argument("--cohort-label", default="pilot_001"); p.set_defaults(func=prepare)
    for name, func in (("status", status), ("audit", audit)):
        p = sub.add_parser(name); p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR); p.set_defaults(func=func)
    p = sub.add_parser("run"); p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--base-url", default=BASE_URL); p.add_argument("--model", default=MODEL)
    p.add_argument("--api-key", default=os.environ.get("VQA_LLM_API_KEY", "local-vqa-v4-2"))
    p.add_argument("--temperature", type=float, default=0.25); p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--retries", type=int, default=1); p.add_argument("--limit-images", type=int, default=0); p.set_defaults(func=run)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if hasattr(args, "output_dir"):
        args.output_dir = args.output_dir.resolve()
    args.func(args)


if __name__ == "__main__": main()
