#!/usr/bin/env python3
"""Generate publication statistics and figures for derived_v4.3 VQA.

The clean Train/Validation/Test exports define the publication cohort. The
audit-rich Master is used to recover patient/procedure and provenance fields.
Anatomy is derived from canonical region provenance in facts_v1 rather than
inferred from generated language.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import math
import platform
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from wordcloud import WordCloud


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT = ROOT / "derived_v4_3_vision_production/final_release/dataset_export"
DEFAULT_MASTER = DEFAULT_EXPORT / "master_bronchoscopy_vqa.json"
DEFAULT_FACTS = ROOT / "derived_v4/facts_v1.jsonl.gz"
DEFAULT_OUTPUT = ROOT / "derived_v4_3_vision_production/publication_statistics"
SPLITS = ("train", "validation", "test")
SPLIT_FILE_NAMES = {"train": "train.json", "validation": "val.json", "test": "test.json"}
SPLIT_LABELS = {"train": "Train", "validation": "Validation", "test": "Test"}

PALETTE = ["#2C7BB6", "#00A6CA", "#00CCBC", "#90EB9D", "#F9D057", "#F29E2E", "#D7191C"]
CONTENT_COLOR = "#2878B5"
ANATOMY_COLOR = "#3BA272"
FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")

TOKEN_RE = re.compile(r"[^\W_]+(?:[-–][^\W_]+)*|\d+(?:[.,]\d+)*", flags=re.UNICODE)
TAG_PATTERNS = {
    "answer": re.compile(r"<answer>\s*(.*?)(?=\s*<reason>|\s*<visual_evidence>|\s*<location>|$)", re.S),
    "reason": re.compile(r"<reason>\s*(.*?)(?=\s*<visual_evidence>|\s*<location>|$)", re.S),
    "visual_evidence": re.compile(r"<visual_evidence>\s*(.*?)(?=\s*<location>|$)", re.S),
}

VI_STOPWORDS = {
    "a", "ai", "bị", "bởi", "các", "cái", "cho", "có", "còn", "của", "cũng", "đã",
    "đang", "đây", "đến", "được", "gì", "hay", "hiện", "hình", "hỏi", "khi", "không",
    "là", "lại", "lên", "mà", "mô", "một", "nào", "này", "như", "những", "nội", "ở",
    "qua", "ra", "rằng", "rõ", "sát", "sẽ", "so", "sự", "tại", "thấy", "thể", "thì",
    "theo", "trên", "trong", "từ", "và", "vào", "về", "với", "vùng", "xác", "nhận",
    "ảnh", "soi", "quan", "được", "ghi", "thế", "nên", "nhiều", "ít", "nơi", "phần",
}

MEDICAL_PHRASES = (
    "phế quản phân thùy lưỡi trái", "phế quản thùy trên phải", "phế quản thùy giữa phải",
    "phế quản thùy dưới phải", "phế quản thùy trên trái", "phế quản thùy dưới trái",
    "phế quản gốc phải", "phế quản gốc trái", "phế quản trung gian", "dịch tiết bất thường",
    "hẹp lòng phế quản", "mức độ hẹp", "dây thanh", "khối u", "bình thường", "bất thường",
    "không cuống", "có cuống", "phù nề", "xung huyết", "thâm nhiễm", "dễ chảy máu",
    "niêm mạc", "dịch tiết", "lòng phế quản", "bề mặt", "màu sắc", "tổn thương",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--facts", type=Path, default=DEFAULT_FACTS)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def lexical_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def clean_question(value: str, include_choices: bool = False) -> str:
    text = re.sub(r"^\s*<image>\s*", "", value, flags=re.I)
    if not include_choices:
        text = re.split(r"\s*<choices>\s*:", text, maxsplit=1, flags=re.I)[0]
    return " ".join(text.split())


def extract_tag(text: str, tag: str) -> str:
    match = TAG_PATTERNS[tag].search(text)
    return " ".join(match.group(1).split()) if match else ""


def conversation_parts(record: dict[str, Any]) -> tuple[str, str, str, str, str]:
    messages = record.get("conversations") or []
    human = next((m.get("value", "") for m in messages if m.get("from") == "human"), "")
    assistant = next((m.get("value", "") for m in messages if m.get("from") == "gpt"), "")
    return (
        clean_question(human, include_choices=False),
        clean_question(human, include_choices=True),
        extract_tag(assistant, "answer"),
        extract_tag(assistant, "reason"),
        extract_tag(assistant, "visual_evidence"),
    )


def percentile(values: list[int], q: float) -> float:
    return float(np.percentile(np.asarray(values), q)) if values else math.nan


def describe(values: list[int]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": round(mean(values), 3) if values else math.nan,
        "sd": round(float(np.std(values, ddof=1)), 3) if len(values) > 1 else 0.0,
        "min": min(values) if values else math.nan,
        "p25": round(percentile(values, 25), 3),
        "median": round(median(values), 3) if values else math.nan,
        "p75": round(percentile(values, 75), 3),
        "p95": round(percentile(values, 95), 3),
        "max": max(values) if values else math.nan,
    }


def normalize_phrase_text(text: str) -> list[str]:
    lowered = text.lower()
    phrase_tokens: list[str] = []
    for phrase in MEDICAL_PHRASES:
        count = len(re.findall(rf"(?<!\w){re.escape(phrase)}(?!\w)", lowered))
        if count:
            phrase_tokens.extend([phrase] * count)
            lowered = re.sub(rf"(?<!\w){re.escape(phrase)}(?!\w)", " ", lowered)
    singles = [token.lower() for token in lexical_tokens(lowered)]
    phrase_tokens.extend(token for token in singles if len(token) > 1 and token not in VI_STOPWORDS and not token.isdigit())
    return phrase_tokens


def load_fact_provenance(path: Path) -> dict[str, Any]:
    region_anatomy: dict[str, tuple[str, ...]] = {}
    fact_anatomy: dict[str, tuple[str, ...]] = {}
    label_votes: dict[str, Counter[str]] = defaultdict(Counter)
    fact_count = 0
    image_count = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            image_count += 1
            for region in row.get("regions") or []:
                region_anatomy[region["canonical_region_id"]] = tuple(region.get("anatomy") or [])
            for fact in row.get("facts") or []:
                fact_count += 1
                fact_anatomy[fact["fact_id"]] = tuple(fact.get("anatomical_locations") or [])
                if fact.get("concept_family") == "anatomy" and fact.get("value") and fact.get("surface_vi"):
                    label_votes[str(fact["value"])][str(fact["surface_vi"])] += 1
    labels = {code: votes.most_common(1)[0][0] for code, votes in label_votes.items()}
    return {
        "region_anatomy": region_anatomy,
        "fact_anatomy": fact_anatomy,
        "labels_vi": labels,
        "source_images": image_count,
        "source_facts": fact_count,
    }


def validate_and_join(master: list[dict[str, Any]], export_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    master_by_id: dict[str, dict[str, Any]] = {}
    duplicate_master_ids: list[str] = []
    for record in master:
        qa_id = record.get("qa_id")
        if qa_id in master_by_id:
            duplicate_master_ids.append(str(qa_id))
        master_by_id[qa_id] = record

    clean_records: list[dict[str, Any]] = []
    errors: list[str] = []
    split_ids: dict[str, set[str]] = {}
    for split in SPLITS:
        path = export_dir / SPLIT_FILE_NAMES[split]
        records = read_json(path)
        ids: set[str] = set()
        for clean in records:
            qa_id = clean.get("qa_id")
            if qa_id in ids:
                errors.append(f"duplicate qa_id in {split}: {qa_id}")
                continue
            ids.add(qa_id)
            source = master_by_id.get(qa_id)
            if source is None:
                errors.append(f"qa_id absent from Master: {qa_id}")
                continue
            if source.get("split") != split:
                errors.append(f"split mismatch for {qa_id}: {source.get('split')} != {split}")
            for protected_key in ("image", "conversations", "question_type", "q_type", "evidence_boxes_xyxy", "answer_structured"):
                if source.get(protected_key) != clean.get(protected_key):
                    errors.append(f"Master/clean mismatch for {qa_id}: {protected_key}")
            if source.get("master_record_status") != "ACCEPTED_EXPORTED":
                errors.append(f"non-exportable Master status for {qa_id}: {source.get('master_record_status')}")
            clean_records.append(source)
        split_ids[split] = ids

    pair_overlaps = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        pair_overlaps[f"{left}__{right}"] = len(split_ids[left] & split_ids[right])
    expected_master_ids = {
        record["qa_id"] for record in master if record.get("master_record_status") == "ACCEPTED_EXPORTED"
    }
    exported_ids = set().union(*split_ids.values())
    if expected_master_ids != exported_ids:
        errors.append(
            f"exportable Master IDs differ from clean IDs: master_only={len(expected_master_ids-exported_ids)}, "
            f"clean_only={len(exported_ids-expected_master_ids)}"
        )
    if duplicate_master_ids:
        errors.append(f"duplicate Master qa_id count: {len(duplicate_master_ids)}")
    if errors:
        raise ValueError("Input validation failed:\n- " + "\n- ".join(errors[:30]))
    return clean_records, {
        "status": "PASS",
        "records": len(clean_records),
        "qa_id_split_overlaps": pair_overlaps,
        "master_exportable_records": len(expected_master_ids),
    }


def build_statistics(records: list[dict[str, Any]], provenance: dict[str, Any]) -> dict[str, Any]:
    split_acc: dict[str, dict[str, Any]] = {
        split: {
            "qa_pairs": 0,
            "images": set(),
            "patients": set(),
            "procedures": set(),
            "bbox_instances": 0,
            "qa_with_bbox": 0,
            "qa_without_bbox": 0,
            "invalid_bbox_instances": 0,
            "unique_boxes": set(),
            "evidence_regions": set(),
        }
        for split in SPLITS
    }
    q_type = Counter()
    q_type_split: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    question_type = Counter()
    question_type_split: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    normality = Counter()
    normality_split: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    normality_images: dict[str, set[str]] = defaultdict(set)
    lengths: dict[str, list[int]] = defaultdict(list)
    question_word_frequency = Counter()
    explanation_word_frequency = Counter()
    anatomy_qa = Counter()
    anatomy_unique_regions: dict[str, set[str]] = defaultdict(set)
    anatomy_text_mentions = Counter()
    anatomy_source = Counter()
    missing = Counter()

    region_anatomy = provenance["region_anatomy"]
    fact_anatomy = provenance["fact_anatomy"]
    labels_vi = provenance["labels_vi"]

    for record in records:
        split = record["split"]
        acc = split_acc[split]
        acc["qa_pairs"] += 1
        acc["images"].add(record.get("image_id") or record["image"])
        if record.get("patient_id"):
            acc["patients"].add(record["patient_id"])
        else:
            missing["patient_id"] += 1
        if record.get("procedure_id"):
            acc["procedures"].add(record["procedure_id"])
        else:
            missing["procedure_id"] += 1

        boxes = record.get("evidence_boxes_xyxy") or []
        acc["bbox_instances"] += len(boxes)
        if boxes:
            acc["qa_with_bbox"] += 1
        else:
            acc["qa_without_bbox"] += 1
        image_key = record.get("image_id") or record["image"]
        for box in boxes:
            valid_box = (
                isinstance(box, list)
                and len(box) == 4
                and all(isinstance(value, (int, float)) for value in box)
                and box[2] > box[0]
                and box[3] > box[1]
            )
            if not valid_box:
                acc["invalid_bbox_instances"] += 1
                continue
            acc["unique_boxes"].add((image_key, tuple(box)))
        for region_id in record.get("evidence_region_ids") or []:
            acc["evidence_regions"].add((image_key, region_id))

        q_value = record.get("q_type") or "missing"
        c_value = record.get("question_type") or "missing"
        q_type[q_value] += 1
        q_type_split[split][q_value] += 1
        question_type[c_value] += 1
        question_type_split[split][c_value] += 1

        question, question_input, _answer, reason, visual = conversation_parts(record)
        if not question:
            missing["question"] += 1
        if not reason:
            missing["reason"] += 1
        if not visual:
            missing["visual_evidence"] += 1
        length_values = {
            "question_stem_words": len(lexical_tokens(question)),
            "question_input_words": len(lexical_tokens(question_input)),
            "reason_words": len(lexical_tokens(reason)),
            "visual_evidence_words": len(lexical_tokens(visual)),
            "explanation_words": len(lexical_tokens(f"{reason} {visual}")),
        }
        for key, value in length_values.items():
            lengths[key].append(value)
        question_word_frequency.update(normalize_phrase_text(question))
        explanation_word_frequency.update(normalize_phrase_text(f"{reason} {visual}"))

        if c_value == "image_normality":
            normal = (record.get("answer_structured") or {}).get("normal")
            if normal is True:
                normal_key = "normal_true"
            elif normal is False:
                normal_key = "normal_false"
            else:
                normal_key = "missing_or_invalid"
            normality[normal_key] += 1
            normality_split[split][normal_key] += 1
            normality_images[normal_key].add(image_key)

        anatomy_codes: set[str] = set()
        evidence_regions = record.get("evidence_region_ids") or []
        for region_id in evidence_regions:
            for code in region_anatomy.get(region_id, ()):  # primary provenance route
                anatomy_codes.add(code)
                anatomy_unique_regions[code].add(region_id)
        if anatomy_codes:
            anatomy_source["canonical_region"] += 1
        else:
            for fact_id in record.get("source_fact_ids") or []:
                anatomy_codes.update(fact_anatomy.get(fact_id, ()))
            if anatomy_codes:
                anatomy_source["source_fact_fallback"] += 1
            else:
                anatomy_source["unspecified"] += 1
        for code in anatomy_codes:
            anatomy_qa[code] += 1

        question_lower = question.lower()
        for code, label in labels_vi.items():
            if re.search(rf"(?<!\w){re.escape(label.lower())}(?!\w)", question_lower):
                anatomy_text_mentions[code] += 1

    # Leakage checks are based on the clean publication cohort.
    overlap: dict[str, dict[str, int]] = {}
    for key in ("images", "patients", "procedures"):
        overlap[key] = {}
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            overlap[key][f"{left}__{right}"] = len(split_acc[left][key] & split_acc[right][key])

    split_summary: dict[str, dict[str, int]] = {}
    for split, acc in split_acc.items():
        split_summary[split] = {
            "unique_images": len(acc["images"]),
            "unique_patients": len(acc["patients"]),
            "unique_procedures": len(acc["procedures"]),
            "qa_pairs": acc["qa_pairs"],
            "bbox_instances": acc["bbox_instances"],
            "qa_with_bbox": acc["qa_with_bbox"],
            "qa_without_bbox": acc["qa_without_bbox"],
            "invalid_bbox_instances": acc["invalid_bbox_instances"],
            "unique_image_box_tuples": len(acc["unique_boxes"]),
            "unique_evidence_regions": len(acc["evidence_regions"]),
        }
    total_summary = {
        key: sum(split_summary[split][key] for split in SPLITS)
        for key in next(iter(split_summary.values()))
    }

    return {
        "analysis_scope": {
            "cohort": "clean Train/Validation/Test records only",
            "records": len(records),
            "bbox_instances_definition": "sum of len(evidence_boxes_xyxy) across QA records",
            "unique_box_definition": "unique (image_id, bbox_xyxy) tuples; avoids counting reused boxes repeatedly",
            "normality_unit": "QA pairs with question_type=image_normality",
            "anatomy_primary_source": "evidence_region_ids joined to canonical regions in facts_v1",
            "word_unit": "Unicode lexical tokens; Vietnamese syllables separated by spaces count separately",
            "question_length": "question stem after removing <image> and <choices>",
            "explanation_length": "reason plus visual_evidence; reason and visual_evidence are also reported separately",
        },
        "split_statistics": {"by_split": split_summary, "total": total_summary},
        "split_leakage": overlap,
        "q_type": {
            "counts": dict(q_type.most_common()),
            "by_split": {split: dict(q_type_split[split]) for split in SPLITS},
        },
        "question_type": {
            "counts": dict(question_type.most_common()),
            "by_split": {split: dict(question_type_split[split]) for split in SPLITS},
        },
        "normality": {
            "counts": dict(normality),
            "unique_images": {key: len(value) for key, value in normality_images.items()},
            "by_split": {split: dict(normality_split[split]) for split in SPLITS},
        },
        "lengths": {key: {"summary": describe(values), "values": values} for key, values in lengths.items()},
        "word_frequency": {
            "question": dict(question_word_frequency.most_common()),
            "explanation": dict(explanation_word_frequency.most_common()),
        },
        "anatomy": {
            "qa_grounded_counts": dict(anatomy_qa.most_common()),
            "unique_region_counts": {code: len(regions) for code, regions in anatomy_unique_regions.items()},
            "question_text_mentions": dict(anatomy_text_mentions.most_common()),
            "labels_vi": labels_vi,
            "qa_provenance_route": dict(anatomy_source),
        },
        "missing_fields": dict(missing),
    }


def pct(n: int, denominator: int) -> float:
    return 100.0 * n / denominator if denominator else 0.0


def output_tables(stats: dict[str, Any], output_dir: Path) -> None:
    split_stats = stats["split_statistics"]
    metrics = [
        ("Unique images", "unique_images"),
        ("Unique patients", "unique_patients"),
        ("Unique procedures", "unique_procedures"),
        ("QA pairs", "qa_pairs"),
        ("BBox instances", "bbox_instances"),
        ("QA pairs with $\\geq$1 bbox", "qa_with_bbox"),
        ("QA pairs without bbox", "qa_without_bbox"),
        ("Unique image–box tuples", "unique_image_box_tuples"),
    ]
    table_rows = []
    for label, key in metrics:
        total = split_stats["total"][key]
        row: dict[str, Any] = {"Metric": label}
        for split in SPLITS:
            value = split_stats["by_split"][split][key]
            row[SPLIT_LABELS[split]] = f"{value:,} ({pct(value, total):.2f}%)"
        row["Total"] = f"{total:,} (100.00%)"
        table_rows.append(row)
    fields = ["Metric", "Train", "Validation", "Test", "Total"]
    write_csv(output_dir / "table_1_dataset_split_statistics.csv", fields, table_rows)

    md = [
        "| Metric | Train | Validation | Test | Total |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in table_rows:
        md.append("| " + " | ".join(str(row[field]) for field in fields) + " |")
    (output_dir / "table_1_dataset_split_statistics.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Statistics of the clean bronchoscopy VQA dataset. Percentages are column-metric shares across splits.}",
        r"\label{tab:dataset_statistics}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Metric & Train & Validation & Test & Total \\",
        r"\midrule",
    ]
    for row in table_rows:
        label = row["Metric"].replace("–", "--")
        cells = [str(row[key]).replace("%", r"\%") for key in ("Train", "Validation", "Test", "Total")]
        latex_row = f"{label} & " + " & ".join(cells) + " "
        tex_lines.append(latex_row + r"\\")
    tex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (output_dir / "table_1_dataset_split_statistics.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

    total_qa = split_stats["total"]["qa_pairs"]
    q_rows = []
    for value, count in stats["q_type"]["counts"].items():
        row = {"q_type": value, "count": count, "percentage": round(pct(count, total_qa), 4)}
        row.update({split: stats["q_type"]["by_split"][split].get(value, 0) for split in SPLITS})
        q_rows.append(row)
    write_csv(output_dir / "q_type_distribution.csv", ["q_type", "count", "percentage", *SPLITS], q_rows)

    c_rows = []
    for value, count in stats["question_type"]["counts"].items():
        row = {"question_type": value, "count": count, "percentage": round(pct(count, total_qa), 4)}
        row.update({split: stats["question_type"]["by_split"][split].get(value, 0) for split in SPLITS})
        c_rows.append(row)
    write_csv(
        output_dir / "question_type_distribution.csv",
        ["question_type", "count", "percentage", *SPLITS],
        c_rows,
    )

    normal_total = sum(stats["normality"]["counts"].values())
    n_rows = []
    for value in ("normal_true", "normal_false", "missing_or_invalid"):
        count = stats["normality"]["counts"].get(value, 0)
        if count or value != "missing_or_invalid":
            row = {
                "normality": value,
                "qa_count": count,
                "qa_percentage": round(pct(count, normal_total), 4),
                "unique_images": stats["normality"]["unique_images"].get(value, 0),
            }
            row.update({split: stats["normality"]["by_split"][split].get(value, 0) for split in SPLITS})
            n_rows.append(row)
    write_csv(
        output_dir / "normality_distribution.csv",
        ["normality", "qa_count", "qa_percentage", "unique_images", *SPLITS],
        n_rows,
    )

    length_rows = []
    for field, payload in stats["lengths"].items():
        length_rows.append({"field": field, **payload["summary"]})
    write_csv(
        output_dir / "language_length_summary.csv",
        ["field", "n", "mean", "sd", "min", "p25", "median", "p75", "p95", "max"],
        length_rows,
    )

    anatomy = stats["anatomy"]
    all_codes = set(anatomy["qa_grounded_counts"]) | set(anatomy["question_text_mentions"])
    anatomy_rows = []
    for code in sorted(all_codes, key=lambda x: (-anatomy["qa_grounded_counts"].get(x, 0), x)):
        anatomy_rows.append(
            {
                "anatomy_code": code,
                "label_vi": anatomy["labels_vi"].get(code, ""),
                "grounded_qa_count": anatomy["qa_grounded_counts"].get(code, 0),
                "grounded_qa_percentage": round(
                    pct(anatomy["qa_grounded_counts"].get(code, 0), total_qa), 4
                ),
                "unique_evidence_regions": anatomy["unique_region_counts"].get(code, 0),
                "question_text_mentions": anatomy["question_text_mentions"].get(code, 0),
            }
        )
    write_csv(
        output_dir / "anatomical_region_distribution.csv",
        [
            "anatomy_code", "label_vi", "grounded_qa_count", "grounded_qa_percentage",
            "unique_evidence_regions", "question_text_mentions",
        ],
        anatomy_rows,
    )

    for name in ("question", "explanation"):
        rows = [
            {"rank": rank, "term": term, "count": count}
            for rank, (term, count) in enumerate(stats["word_frequency"][name].items(), start=1)
        ]
        write_csv(output_dir / f"word_frequency_{name}.csv", ["rank", "term", "count"], rows)


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)


def add_bar_labels(ax: plt.Axes, bars: Any, fontsize: int = 8) -> None:
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height() / 2, f" {int(width):,}", va="center", fontsize=fontsize)


def plot_figure_3(stats: dict[str, Any], output_dir: Path, dpi: int) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)

    q_counts = stats["q_type"]["counts"]
    q_labels = [key.replace("_questions", "").replace("_", "-").title() for key in q_counts]
    q_values = list(q_counts.values())
    axes[0, 0].pie(
        q_values,
        labels=q_labels,
        colors=PALETTE[: len(q_values)],
        startangle=90,
        counterclock=False,
        autopct=lambda p: f"{p:.1f}%",
        pctdistance=0.78,
        wedgeprops={"width": 0.42, "edgecolor": "white"},
        textprops={"fontsize": 9},
    )
    axes[0, 0].text(0, 0, f"n={sum(q_values):,}", ha="center", va="center", fontweight="bold")
    axes[0, 0].set_title("(a) Question format distribution", loc="left", fontweight="bold")

    content_items = list(stats["question_type"]["counts"].items())
    content_items.reverse()
    labels = [key.replace("_", " ") for key, _ in content_items]
    values = [value for _, value in content_items]
    bars = axes[0, 1].barh(labels, values, color=CONTENT_COLOR)
    add_bar_labels(axes[0, 1], bars)
    axes[0, 1].set_xlabel("Number of QA pairs")
    axes[0, 1].set_title("(b) Clinical question content", loc="left", fontweight="bold")
    style_axes(axes[0, 1])

    anatomy_items = list(stats["anatomy"]["qa_grounded_counts"].items())
    anatomy_items.reverse()
    labels = [key.replace("_", " ") for key, _ in anatomy_items]
    values = [value for _, value in anatomy_items]
    bars = axes[1, 0].barh(labels, values, color=ANATOMY_COLOR)
    add_bar_labels(axes[1, 0], bars)
    axes[1, 0].set_xlabel("Grounded QA–anatomy associations")
    axes[1, 0].set_title("(c) Canonical anatomical regions", loc="left", fontweight="bold")
    style_axes(axes[1, 0])

    normal = stats["normality"]["counts"]
    n_values = [normal.get("normal_true", 0), normal.get("normal_false", 0)]
    n_labels = ["Normal", "Abnormal"]
    axes[1, 1].pie(
        n_values,
        labels=n_labels,
        colors=["#67B7DC", "#E76F51"],
        startangle=90,
        counterclock=False,
        autopct=lambda p: f"{p:.1f}%",
        pctdistance=0.78,
        wedgeprops={"width": 0.42, "edgecolor": "white"},
    )
    axes[1, 1].text(0, 0, f"n={sum(n_values):,}", ha="center", va="center", fontweight="bold")
    axes[1, 1].set_title("(d) Image-normality QA distribution", loc="left", fontweight="bold")

    fig.suptitle("Bronchoscopy VQA dataset composition", fontsize=16, fontweight="bold")
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"figure_3_dataset_composition.{suffix}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_wordcloud(frequencies: dict[str, int], color_map: str) -> WordCloud:
    font_path = str(FONT_PATH) if FONT_PATH.exists() else None
    return WordCloud(
        width=1800,
        height=1000,
        background_color="white",
        font_path=font_path,
        colormap=color_map,
        max_words=120,
        prefer_horizontal=0.9,
        relative_scaling=0.45,
        random_state=42,
        collocations=False,
        margin=4,
    ).generate_from_frequencies(dict(list(frequencies.items())[:1000]))


def plot_figure_4(stats: dict[str, Any], output_dir: Path, dpi: int) -> None:
    question_freq = stats["word_frequency"]["question"]
    explanation_freq = stats["word_frequency"]["explanation"]
    question_cloud = make_wordcloud(question_freq, "Blues")
    explanation_cloud = make_wordcloud(explanation_freq, "YlOrRd")

    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    grid = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.2])
    ax_hist = fig.add_subplot(grid[0, :])
    ax_q = fig.add_subplot(grid[1, 0])
    ax_e = fig.add_subplot(grid[1, 1])

    series = {
        "Question stem": stats["lengths"]["question_stem_words"]["values"],
        "Reason": stats["lengths"]["reason_words"]["values"],
        "Visual evidence": stats["lengths"]["visual_evidence_words"]["values"],
        "Reason + visual evidence": stats["lengths"]["explanation_words"]["values"],
    }
    max_words = max(max(values) for values in series.values())
    bin_width = 2
    bins = np.arange(0, max_words + bin_width + 1, bin_width)
    for (label, values), color in zip(series.items(), ["#2C7BB6", "#00A676", "#F29E2E", "#D7191C"]):
        ax_hist.hist(values, bins=bins, alpha=0.45, label=label, color=color, edgecolor="none")
    ax_hist.set_xlabel("Length (Unicode lexical tokens)")
    ax_hist.set_ylabel("Number of QA pairs")
    ax_hist.set_title("(a) Question and explanation length distributions", loc="left", fontweight="bold")
    ax_hist.legend(frameon=False, ncol=2)
    style_axes(ax_hist)

    ax_q.imshow(question_cloud, interpolation="bilinear")
    ax_q.axis("off")
    ax_q.set_title("(b) Question stems", loc="left", fontweight="bold")
    ax_e.imshow(explanation_cloud, interpolation="bilinear")
    ax_e.axis("off")
    ax_e.set_title("(c) Reason + visual evidence", loc="left", fontweight="bold")
    fig.suptitle("Language and explainability analysis", fontsize=16, fontweight="bold")
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"figure_4_language_analysis.{suffix}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    for name, cloud in (("questions", question_cloud), ("explanations", explanation_cloud)):
        fig, ax = plt.subplots(figsize=(12, 6.7))
        ax.imshow(cloud, interpolation="bilinear")
        ax.axis("off")
        fig.savefig(output_dir / f"wordcloud_{name}.png", dpi=dpi, bbox_inches="tight", pad_inches=0)
        plt.close(fig)


def build_report(stats: dict[str, Any], validation: dict[str, Any], output_dir: Path) -> None:
    split = stats["split_statistics"]
    total = split["total"]
    q_type = stats["q_type"]["counts"]
    question_type = stats["question_type"]["counts"]
    normal = stats["normality"]["counts"]
    normal_total = sum(normal.values())
    anatomy_route = stats["anatomy"]["qa_provenance_route"]
    grounded = anatomy_route.get("canonical_region", 0) + anatomy_route.get("source_fact_fallback", 0)
    content_top = list(question_type.items())[:5]

    report = f"""# Báo cáo thống kê bộ dữ liệu Bronchoscopy VQA derived_v4.3

## Phạm vi và tính toàn vẹn

- Cohort phân tích: **27.570 QA sạch** trong `train.json`, `val.json`, `test.json`.
- Đối chiếu Master–clean: **{validation['status']}**; không có QA trùng giữa các split.
- Giao nhau theo ảnh, bệnh nhân và quy trình giữa mọi cặp split: **0**.
- Không đưa 455 record `REVIEW_REQUIRED` vào các thống kê huấn luyện/phát hành.

## Quy mô dữ liệu

- **{total['unique_images']:,}** ảnh độc nhất; **{total['unique_patients']:,}** bệnh nhân; **{total['unique_procedures']:,}** quy trình.
- **{total['qa_pairs']:,}** cặp QA và **{total['bbox_instances']:,}** bbox instances.
- Sau khi khử việc cùng bbox được dùng lại cho nhiều QA: **{total['unique_image_box_tuples']:,}** unique image–box tuples.
- **{total['qa_with_bbox']:,} QA ({pct(total['qa_with_bbox'], total['qa_pairs']):.2f}%)** có ít nhất một bbox; **{total['qa_without_bbox']:,} QA ({pct(total['qa_without_bbox'], total['qa_pairs']):.2f}%)** không có bbox. Toàn bộ nhóm không có bbox là câu hỏi `image_normality` cấp toàn ảnh.
- Số bbox có tọa độ không hợp lệ: **{total['invalid_bbox_instances']:,}**.
- Phân bố QA Train/Validation/Test: **{pct(split['by_split']['train']['qa_pairs'], total['qa_pairs']):.2f}% / {pct(split['by_split']['validation']['qa_pairs'], total['qa_pairs']):.2f}% / {pct(split['by_split']['test']['qa_pairs'], total['qa_pairs']):.2f}%**.

## Định dạng và nội dung câu hỏi

"""
    for key, count in q_type.items():
        report += f"- `{key}`: **{count:,} ({pct(count, total['qa_pairs']):.2f}%)**.\n"
    absent_formats = [
        key for key in (
            "closed_ended_questions", "open_ended_questions", "single_choice_questions", "multi_choice_questions"
        ) if key not in q_type
    ]
    if absent_formats:
        report += "- Không xuất hiện trong bản sạch: " + ", ".join(f"`{key}`" for key in absent_formats) + ".\n"
    report += "\nNăm nhóm nội dung lớn nhất:\n\n"
    for key, count in content_top:
        report += f"- `{key}`: **{count:,} ({pct(count, total['qa_pairs']):.2f}%)**.\n"

    report += f"""

## Bình thường và bất thường

Trong **{normal_total:,}** QA thuộc `image_normality`:

- `normal=true`: **{normal.get('normal_true', 0):,} ({pct(normal.get('normal_true', 0), normal_total):.2f}%)**.
- `normal=false`: **{normal.get('normal_false', 0):,} ({pct(normal.get('normal_false', 0), normal_total):.2f}%)**.

Đây là phân bố **QA/image-normality**, không phải prevalence ở cấp bệnh nhân. Tỷ lệ này tự nó không chứng minh “cân bằng công bằng”; cần đánh giá thêm sensitivity/specificity và calibration theo bệnh nhân.

## Độ dài ngôn ngữ

| Thành phần | Mean | Median | P25–P75 | P95 | Max |
|---|---:|---:|---:|---:|---:|
"""
    for key in ("question_stem_words", "reason_words", "visual_evidence_words", "explanation_words"):
        summary = stats["lengths"][key]["summary"]
        report += (
            f"| `{key}` | {summary['mean']:.2f} | {summary['median']:.2f} | "
            f"{summary['p25']:.2f}–{summary['p75']:.2f} | {summary['p95']:.2f} | {summary['max']} |\n"
        )

    report += f"""

## Vùng giải phẫu

- **{grounded:,}/{total['qa_pairs']:,} QA ({pct(grounded, total['qa_pairs']):.2f}%)** có vùng giải phẫu xác định được bằng provenance canonical.
- **{anatomy_route.get('unspecified', 0):,} QA** có bbox/evidence nhưng canonical region chưa mang nhãn giải phẫu; các QA này không bị ép gán vùng bằng dò từ khóa.
- Phân bố trong Figure 3(c) là số liên kết QA–anatomy; một QA có thể liên kết nhiều vùng nên tổng cột có thể lớn hơn số QA grounded.
- `question_text_mentions` được xuất riêng trong CSV như một phân tích ngôn ngữ thứ cấp, không thay thế provenance.

## Lưu ý diễn giải khoa học

1. `BBox instances` đếm bbox theo từng QA và có thể đếm lại cùng một bbox. Khi so sánh với dataset khác, nên báo cáo song song `unique image–box tuples`.
2. Word cloud mang tính mô tả khám phá; không phải bằng chứng độc lập về độ sâu lâm sàng hoặc chất lượng reasoning.
3. Độ dài không đồng nghĩa với chất lượng. Chất lượng explainability cần được kiểm tra thêm bằng fact consistency, visual grounding và đánh giá chuyên gia.
4. Các tỷ lệ của bộ dữ liệu này phải được trình bày theo số liệu thực tế, không nên khẳng định tuân theo 8:1:1.

## Tệp đầu ra

- `table_1_dataset_split_statistics.csv/.md/.tex`
- `figure_3_dataset_composition.png/.pdf`
- `figure_4_language_analysis.png/.pdf`
- Các bảng phân bố CSV, word-frequency CSV, word clouds riêng và `statistics.json`.
"""
    (output_dir / "REPORT_VI.md").write_text(report, encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.master = args.master.resolve()
    args.facts = args.facts.resolve()
    args.export_dir = args.export_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in (args.master, args.facts, args.export_dir / "train.json", args.export_dir / "val.json", args.export_dir / "test.json"):
        if not path.is_file():
            raise FileNotFoundError(path)

    master = read_json(args.master)
    records, validation = validate_and_join(master, args.export_dir)
    provenance = load_fact_provenance(args.facts)
    stats = build_statistics(records, provenance)
    output_tables(stats, args.output_dir)
    plot_figure_3(stats, args.output_dir, args.dpi)
    plot_figure_4(stats, args.output_dir, args.dpi)

    compact_stats = json.loads(json.dumps(stats))
    for payload in compact_stats["lengths"].values():
        payload.pop("values", None)
    # Full word lists live in CSV; keep the JSON summary readable.
    compact_stats["word_frequency"] = {
        key: dict(list(value.items())[:100]) for key, value in compact_stats["word_frequency"].items()
    }
    compact_stats["input_validation"] = validation
    write_json(args.output_dir / "statistics.json", compact_stats)
    build_report(stats, validation, args.output_dir)

    input_files = [
        args.master,
        args.facts,
        args.export_dir / "train.json",
        args.export_dir / "val.json",
        args.export_dir / "test.json",
    ]
    output_files = sorted(
        path for path in args.output_dir.iterdir()
        if path.is_file() and path.name != "analysis_manifest.json"
    )
    manifest = {
        "analysis": "bronchoscopy_vqa_publication_statistics_v1",
        "script": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": sha256_file(Path(__file__)),
        "environment": {
            "python": platform.python_version(),
            "matplotlib": importlib.metadata.version("matplotlib"),
            "numpy": importlib.metadata.version("numpy"),
            "wordcloud": importlib.metadata.version("wordcloud"),
        },
        "inputs": [
            {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in input_files
        ],
        "outputs": [
            {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in output_files
        ],
        "validation": validation,
    }
    write_json(args.output_dir / "analysis_manifest.json", manifest)
    print(json.dumps({
        "status": "PASS",
        "output_dir": str(args.output_dir),
        "qa_pairs": stats["split_statistics"]["total"]["qa_pairs"],
        "unique_images": stats["split_statistics"]["total"]["unique_images"],
        "outputs": len(output_files) + 1,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
