#!/usr/bin/env python3
"""Create an English, model-facing export of bronchoscopy VQA v4.1.

The Vietnamese source export is never modified.  English questions, answers,
and evidence statements are generated from protected structured labels rather
than translated independently, so clinical concepts and answer polarity stay
aligned with the source data.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]

CONCEPT_EN = {
    "anthracotic_pigmentation": "anthracotic pigmentation",
    "bronchial_tumor": "tracheobronchial tumor",
    "carinal_edema": "carinal edema",
    "hypervascularity": "hypervascularity",
    "mucosal_atrophy": "mucosal atrophy",
    "mucosal_erythema": "mucosal erythema",
    "mucosal_infiltration": "mucosal infiltration",
    "mucosal_ulceration": "mucosal ulceration",
    "other_mucosal_finding": "other mucosal abnormality",
    "pseudomembrane": "pseudomembrane",
    "secretion": "abnormal secretions",
    "smooth_mucosa": "smooth mucosa",
    "stenosis": "bronchial lumen stenosis",
    "tracheomalacia": "tracheomalacia",
    "vocal_cord_other": "other vocal cord abnormality",
    "vocal_cords_normal": "normal vocal cords",
}

NEGATIVE_CONCEPT_EN = {
    "hypervascularity": "hypervascularity",
    "secretion": "abnormal secretions",
    "vocal_cord_paralysis": "vocal cord paralysis",
}

VALUE_EN = {
    "0_25_percent": "0–25% stenosis",
    "26_50_percent": "26–50% stenosis",
    "51_75_percent": "51–75% stenosis",
    "76_90_percent": "76–90% stenosis",
    "over_90_percent": "greater than 90% stenosis",
    "scarring": "scarring",
    "tumor": "tumor",
    "external_compression": "external compression",
    "endoluminal_lesion": "endoluminal lesion",
    "torsion": "torsion",
    "mixed": "mixed features",
    "pedunculated": "pedunculated",
    "non_pedunculated": "non-pedunculated",
    "blood": "blood",
    "purulent": "purulent secretions",
    "bright_red": "bright red",
    "dark_red": "dark red",
    "clotted": "clotted secretions",
}

ANATOMY_EN = {
    "trachea": "the trachea",
    "carina": "the carina",
    "right_main_bronchus": "the right main bronchus",
    "left_main_bronchus": "the left main bronchus",
    "right_upper_lobe_bronchus": "the right upper lobe bronchus",
    "right_middle_lobe_bronchus": "the right middle lobe bronchus",
    "right_lower_lobe_bronchus": "the right lower lobe bronchus",
    "left_upper_lobe_bronchus": "the left upper lobe bronchus",
    "left_lower_lobe_bronchus": "the left lower lobe bronchus",
    "left_lingula_segment_bronchus": "the left lingular segmental bronchus",
    "intermediate_bronchus": "the bronchus intermedius",
    "vocal_cords": "the vocal cords",
    "left_apical_segment_bronchus": "the left apical segmental bronchus",
}

STENOSIS_OPTIONS = {
    "A": "0–25% stenosis",
    "B": "26–50% stenosis",
    "C": "51–75% stenosis",
    "D": "76–90% stenosis",
    "E": "Greater than 90% stenosis",
}

TYPE_PROMPTS = {
    "closed-ended": "Input a closed-ended question, and the assistant will output its answer (yes or no) with a detailed reason and corresponding visual location.",
    "single-choice": "Input a single-choice question, and the assistant will output its answer (an option) with a detailed reason and corresponding visual location.",
    "multi-choice": "Input a multi-choice question, and the assistant will output its answer (some options) with a detailed reason and corresponding visual location.",
    "open-ended": "Input an open-ended question, and the assistant will output its answer with a detailed reason and corresponding visual location.",
}

VIETNAMESE_RE = re.compile(
    r"[ăâđêôơư"
    r"àáảãạầấẩẫậằắẳẵặ"
    r"èéẻẽẹềếểễệ"
    r"ìíỉĩị"
    r"òóỏõọồốổỗộờớởỡợ"
    r"ùúủũụừứửữự"
    r"ỳýỷỹỵ]",
    re.IGNORECASE,
)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def join_english(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def sentence(text: str) -> str:
    return text[:1].upper() + text[1:] + "."


def locations_for(row: dict[str, Any], facts: dict[str, dict[str, Any]]) -> list[str]:
    codes = {
        location
        for fact_id in row["source_fact_ids"]
        for location in facts[fact_id].get("anatomical_locations", [])
    }
    unknown = codes - ANATOMY_EN.keys()
    if unknown:
        raise KeyError(f"Unmapped anatomical locations for {row['qa_id']}: {sorted(unknown)}")
    return [ANATOMY_EN[code] for code in sorted(codes)]


def observations_for(row: dict[str, Any], facts: dict[str, dict[str, Any]]) -> list[str]:
    structured = row["answer_structured"]
    if "observations" in structured:
        concepts = [item["concept"] for item in structured["observations"]]
    else:
        concepts = sorted(
            {
                facts[fact_id]["concept"]
                for fact_id in row["source_fact_ids"]
                if facts[fact_id].get("fact_kind") == "observation"
                and facts[fact_id].get("polarity") == "present"
            }
        )
    unknown = set(concepts) - CONCEPT_EN.keys()
    if unknown:
        raise KeyError(f"Unmapped observation concepts for {row['qa_id']}: {sorted(unknown)}")
    return [CONCEPT_EN[concept] for concept in concepts]


def english_qa(
    row: dict[str, Any], facts: dict[str, dict[str, Any]]
) -> tuple[str, str, str, dict[str, str]]:
    template = row["template_id"]
    structured = row["answer_structured"]
    locations = locations_for(row, facts)
    location_phrase = f" at {join_english(locations)}" if locations else ""

    if template == "abnormality_open_01":
        findings = observations_for(row, facts)
        finding_text = join_english(findings)
        return (
            "What abnormalities are identified in this bronchoscopic image?",
            sentence(finding_text),
            sentence(f"The evidence regions show {finding_text}"),
            {},
        )

    if template == "normal_closed_01":
        normal = structured["normal"]
        return (
            "Is this bronchoscopic image confirmed to be normal?",
            "Yes." if normal else "No.",
            "The image is confirmed to be normal."
            if normal
            else "The image is confirmed to be abnormal.",
            {},
        )

    if template == "abnormality_presence_closed_01":
        findings = observations_for(row, facts)
        finding_text = join_english(findings)
        return (
            "Does this bronchoscopic image show any confirmed abnormalities?",
            "Yes.",
            sentence(f"The evidence regions show {finding_text}"),
            {},
        )

    if template == "explicit_negative_closed_01":
        concept = structured["concept"]
        if concept not in NEGATIVE_CONCEPT_EN:
            raise KeyError(f"Unmapped negative concept for {row['qa_id']}: {concept}")
        label = NEGATIVE_CONCEPT_EN[concept]
        plural = concept == "secretion"
        return (
            f"{'Are' if plural else 'Is'} {label} identified in this bronchoscopic image?",
            "No.",
            f"The evidence regions confirm that no {label} "
            f"{'are' if plural else 'is'} present.",
            {},
        )

    if template == "stenosis_severity_single_01":
        letter = structured["option"]
        value = VALUE_EN[structured["value"]]
        return (
            f"What is the confirmed degree of bronchial stenosis{location_phrase}?",
            f"{letter}. {STENOSIS_OPTIONS[letter]}.",
            sentence(f"The evidence regions{location_phrase} show {value}"),
            STENOSIS_OPTIONS,
        )

    if template == "attribute_open_01":
        concept = structured["concept"]
        value = VALUE_EN[structured["value"]]
        subjects = {
            "stenosis_cause": ("cause", "bronchial stenosis"),
            "secretion_type": ("type", "abnormal secretions"),
            "secretion_color": ("color", "abnormal secretions"),
            "secretion_consistency": ("consistency", "abnormal secretions"),
            "tumor_morphology": ("morphology", "tracheobronchial tumor"),
        }
        if concept not in subjects:
            raise KeyError(f"Unmapped attribute concept for {row['qa_id']}: {concept}")
        attribute, subject = subjects[concept]
        return (
            f"What is the confirmed {attribute} of the {subject}{location_phrase}?",
            sentence(value),
            sentence(f"The evidence regions{location_phrase} show {value}"),
            {},
        )

    raise KeyError(f"Unmapped template for {row['qa_id']}: {template}")


def extract_location(target: str) -> str:
    match = re.fullmatch(r"<answer> .*? <reason> .*? <location> (.*)", target, re.DOTALL)
    if not match:
        raise ValueError(f"Invalid source target: {target!r}")
    # Parse and re-serialize to ensure that only a JSON array is retained.
    boxes = json.loads(match.group(1))
    return json.dumps(boxes, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-export",
        type=Path,
        default=ROOT / "model_data" / "derived_v4_1_llava",
    )
    parser.add_argument(
        "--source-vqa", type=Path, default=ROOT / "derived_v4_1" / "vqa"
    )
    parser.add_argument(
        "--facts", type=Path, default=ROOT / "derived_v4" / "facts_v1.jsonl.gz"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "model_data" / "derived_v4_1_llava_en",
    )
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing export: {args.output}")

    facts: dict[str, dict[str, Any]] = {}
    for image_record in read_jsonl(args.facts):
        for fact in image_record["facts"]:
            facts[fact["fact_id"]] = fact

    args.output.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "format": "gemex_llava_conversation_v1",
        "dataset_version": "derived_v4.1-en",
        "language": "en",
        "translation_method": "deterministic_generation_from_structured_labels_v1",
        "translation_script_sha256": sha256_file(Path(__file__)),
        "source_export_manifest_sha256": sha256_file(args.source_export / "manifest.json"),
        "facts_sha256": sha256_file(args.facts),
        "validation": {
            "source_records_preserved": True,
            "source_order_preserved": True,
            "identifiers_preserved": True,
            "locations_preserved": True,
            "vietnamese_diacritics_in_conversations": 0,
        },
        "splits": {},
    }

    for split in ("train", "validation", "test"):
        source_model_path = args.source_export / f"{split}.json"
        source_vqa_path = args.source_vqa / f"{split}.jsonl.gz"
        output_path = args.output / f"{split}.json"
        source_model_rows = json.loads(source_model_path.read_text(encoding="utf-8"))
        source_vqa_rows = list(read_jsonl(source_vqa_path))
        if len(source_model_rows) != len(source_vqa_rows):
            raise ValueError(f"Record count mismatch in {split}")

        output_rows = []
        for source_model, source_vqa in zip(source_model_rows, source_vqa_rows):
            if source_model["qa_id"] != source_vqa["qa_id"]:
                raise ValueError(
                    f"qa_id/order mismatch: {source_model['qa_id']} != {source_vqa['qa_id']}"
                )
            question, answer, reason, options = english_qa(source_vqa, facts)
            if options:
                question += "\n" + "\n".join(
                    f"{key}. {value}" for key, value in options.items()
                )
            question += (
                f"\nOriginal image size: {source_model['width']}x"
                f"{source_model['height']} pixels."
            )
            location = extract_location(source_model["conversations"][1]["value"])

            output = copy.deepcopy(source_model)
            output["conversations"] = [
                {"from": "human", "value": f"<image>\n{question}"},
                {
                    "from": "gpt",
                    "value": (
                        f"<answer> {answer} <reason> {reason} <location> {location}"
                    ),
                },
            ]
            output["type_prompt"] = TYPE_PROMPTS[source_vqa["question_format"]]
            output["language"] = "en"
            output_rows.append(output)

        translated_text = "\n".join(
            turn["value"]
            for row in output_rows
            for turn in row["conversations"]
        )
        vietnamese_hits = len(VIETNAMESE_RE.findall(translated_text))
        if vietnamese_hits:
            raise ValueError(
                f"Found {vietnamese_hits} Vietnamese diacritic characters in {split}"
            )

        output_path.write_text(
            json.dumps(output_rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest["splits"][split] = {
            "records": len(output_rows),
            "path": str(output_path.relative_to(ROOT)),
            "sha256": sha256_file(output_path),
            "source_sha256": sha256_file(source_model_path),
        }

    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
