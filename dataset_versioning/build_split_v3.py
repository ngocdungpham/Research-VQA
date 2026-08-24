#!/usr/bin/env python3
"""Build leakage-safe patient/procedure/near-duplicate splits, then VQA.

Inputs are immutable canonical_v2.2 and the reversible Stage-2 region status.
No VQA record is created until the final group assignment is fixed.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DIR = ROOT / "canonical_v2_2"
STAGE2_DIR = ROOT / "derived_v2"
OUTPUT_DIR = ROOT / "derived_v3"
VERSION = "derived_v3.0"
TARGET_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
RATIO_BOUNDS = {
    "train": (0.70, 0.80),
    "validation": (0.10, 0.15),
    "test": (0.10, 0.15),
}
STRATIFY_LABELS = ("mucosal", "tumor", "stenosis", "secretion", "normal")
PHASH_DISTANCE = 4
DHASH_DISTANCE = 4
MIN_TEST_INSTANCES_FOR_AP = 20
CHUNK = 1024 * 1024


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_id(prefix: str, *parts: str, length: int = 20) -> str:
    digest = hashlib.sha256("\0".join(parts).encode()).hexdigest()[:length]
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


def verify_input(directory: Path, expected_version: str) -> dict[str, Any]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["dataset_version"] != expected_version:
        raise ValueError(f"Expected {expected_version}, got {manifest['dataset_version']}")
    for row in manifest["outputs"]:
        path = ROOT / row["path"]
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise ValueError(f"Input checksum mismatch: {path}")
    return manifest


class UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> bool:
        left, right = self.find(left), self.find(right)
        if left == right:
            return False
        if right < left:
            left, right = right, left
        self.parent[right] = left
        return True


class BKTree:
    """Small exact Hamming-radius index for 64-bit perceptual hashes."""

    def __init__(self) -> None:
        self.root: dict[str, Any] | None = None

    @staticmethod
    def distance(left: int, right: int) -> int:
        return (left ^ right).bit_count()

    def add(self, value: int, index: int) -> None:
        if self.root is None:
            self.root = {"value": value, "indices": [index], "children": {}}
            return
        node = self.root
        while True:
            distance = self.distance(value, node["value"])
            if distance == 0:
                node["indices"].append(index)
                return
            child = node["children"].get(distance)
            if child is None:
                node["children"][distance] = {"value": value, "indices": [index], "children": {}}
                return
            node = child

    def query(self, value: int, radius: int) -> Iterable[tuple[int, int]]:
        if self.root is None:
            return
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = self.distance(value, node["value"])
            if distance <= radius:
                for index in node["indices"]:
                    yield index, distance
            low, high = distance - radius, distance + radius
            stack.extend(child for edge, child in node["children"].items() if low <= edge <= high)


def perceptual_hashes(path: Path) -> tuple[int, int]:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("L")
        phash_image = image.resize((32, 32), Image.Resampling.LANCZOS)
        values = np.asarray(phash_image, dtype=np.float32)
        low = dctn(values, type=2, norm="ortho")[:8, :8]
        median = float(np.median(low.flatten()[1:]))
        phash = 0
        for value in low.flatten():
            phash = (phash << 1) | int(value > median)
        dhash_values = np.asarray(image.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
        dhash = 0
        for value in (dhash_values[:, :-1] > dhash_values[:, 1:]).flatten():
            dhash = (dhash << 1) | int(value)
    return phash, dhash


def image_major_labels(record: dict[str, Any]) -> set[str]:
    labels: set[str] = set()
    groups = {group for region in record["regions"] for group in region["labels"]["pathology_group"]}
    if "mucosal_lesion" in groups:
        labels.add("mucosal")
    for name in ("tumor", "stenosis", "secretion"):
        if name in groups:
            labels.add(name)
    if record["image_labels"]["normal"] is True:
        labels.add("normal")
    return labels


def positive_paths(value: Any, prefix: str = "") -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield from positive_paths(child, f"{prefix}.{key}" if prefix else key)
    elif isinstance(value, list):
        for child in value:
            yield f"{prefix}:{child}"
    elif value is True:
        yield prefix
    elif isinstance(value, str):
        yield f"{prefix}:{value}"


def assignment_objective(
    totals: dict[str, Counter[str]], global_counts: Counter[str], assignment: dict[str, str],
    groups: dict[str, dict[str, Any]], enforce_bounds: bool = True,
) -> float:
    total_images = global_counts["images"]
    score = 0.0
    for split, ratio in TARGET_RATIOS.items():
        image_delta = (totals[split]["images"] - total_images * ratio) / max(total_images * ratio, 1)
        score += 8.0 * image_delta * image_delta
        for label in STRATIFY_LABELS:
            target = global_counts[label] * ratio
            delta = (totals[split][label] - target) / max(target, 1)
            score += delta * delta
        if enforce_bounds:
            actual = totals[split]["images"] / total_images
            low, high = RATIO_BOUNDS[split]
            if actual < low:
                score += 10000 * (low - actual) ** 2
            if actual > high:
                score += 10000 * (actual - high) ** 2
    return score


def assign_groups(groups: dict[str, dict[str, Any]], seed: int) -> tuple[dict[str, str], float]:
    global_counts: Counter[str] = Counter()
    for group in groups.values():
        global_counts.update(group["counts"])
    totals = {split: Counter() for split in TARGET_RATIOS}
    assignment: dict[str, str] = {}
    rng = random.Random(seed)
    tie = {key: rng.random() for key in groups}

    def rarity(key: str) -> tuple[float, int, float]:
        group = groups[key]
        rarity_score = sum(group["counts"][label] / max(global_counts[label], 1) for label in STRATIFY_LABELS)
        return rarity_score, group["counts"]["images"], tie[key]

    for key in sorted(groups, key=rarity, reverse=True):
        best: tuple[float, str] | None = None
        for split in TARGET_RATIOS:
            totals[split].update(groups[key]["counts"])
            assignment[key] = split
            score = assignment_objective(totals, global_counts, assignment, groups)
            totals[split].subtract(groups[key]["counts"])
            del assignment[key]
            candidate = (score, split)
            if best is None or candidate < best:
                best = candidate
        chosen = best[1]
        assignment[key] = chosen
        totals[chosen].update(groups[key]["counts"])

    keys = list(groups)
    current = assignment_objective(totals, global_counts, assignment, groups)
    # Deterministic hill-climbing moves/swaps refine frame ratios and label balance.
    for _ in range(30000):
        if rng.random() < 0.55:
            key = rng.choice(keys)
            old = assignment[key]
            new = rng.choice([split for split in TARGET_RATIOS if split != old])
            totals[old].subtract(groups[key]["counts"])
            totals[new].update(groups[key]["counts"])
            assignment[key] = new
            candidate = assignment_objective(totals, global_counts, assignment, groups)
            if candidate + 1e-12 < current:
                current = candidate
            else:
                totals[new].subtract(groups[key]["counts"])
                totals[old].update(groups[key]["counts"])
                assignment[key] = old
        else:
            left, right = rng.sample(keys, 2)
            split_left, split_right = assignment[left], assignment[right]
            if split_left == split_right:
                continue
            totals[split_left].subtract(groups[left]["counts"])
            totals[split_right].subtract(groups[right]["counts"])
            totals[split_left].update(groups[right]["counts"])
            totals[split_right].update(groups[left]["counts"])
            assignment[left], assignment[right] = split_right, split_left
            candidate = assignment_objective(totals, global_counts, assignment, groups)
            if candidate + 1e-12 < current:
                current = candidate
            else:
                totals[split_left].subtract(groups[right]["counts"])
                totals[split_right].subtract(groups[left]["counts"])
                totals[split_left].update(groups[left]["counts"])
                totals[split_right].update(groups[right]["counts"])
                assignment[left], assignment[right] = split_left, split_right
    return assignment, current


def bbox(points: list[list[float]], width: int, height: int) -> list[float]:
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return [min(xs) / width, min(ys) / height, max(xs) / width, max(ys) / height]


def build(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    canonical_manifest = verify_input(args.canonical_dir, "canonical_v2.2")
    stage2_manifest = verify_input(args.stage2_dir, "derived_v2.0")
    canonical_path = args.canonical_dir / "images.jsonl.gz"

    procedures: dict[str, dict[str, Any]] = {}
    image_rows: list[dict[str, Any]] = []
    exact_sha_first: dict[str, tuple[str, str]] = {}
    exact_edges: list[dict[str, Any]] = []
    patient_procedures: dict[str, set[str]] = defaultdict(set)
    class_totals: Counter[str] = Counter()

    with gzip.open(canonical_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            procedure = record["procedure_id"]
            patient_procedures[record["patient_id"]].add(procedure)
            proc = procedures.setdefault(procedure, {
                "patient_ids": set(), "image_ids": [], "counts": Counter(), "source_groups": set()
            })
            proc["patient_ids"].add(record["patient_id"])
            proc["image_ids"].append(record["image_id"])
            proc["counts"]["images"] += 1
            proc["source_groups"].add(record["provenance"]["source_group"])
            major = image_major_labels(record)
            proc["counts"].update(major)
            for region in record["regions"]:
                class_totals.update(set(positive_paths(region["labels"])))
            image_rows.append({
                "image_id": record["image_id"], "patient_id": record["patient_id"],
                "procedure_id": procedure, "image_path": record["image_path"],
                "image_sha256": record["image_sha256"], "width": record["width"], "height": record["height"],
                "major_labels": sorted(major),
            })

    uf = UnionFind(procedures)
    union_edges: list[dict[str, Any]] = []
    for patient, proc_ids in patient_procedures.items():
        proc_ids = sorted(proc_ids)
        for proc in proc_ids[1:]:
            if uf.union(proc_ids[0], proc):
                union_edges.append({"type": "same_patient", "patient_id": patient, "procedures": [proc_ids[0], proc]})
    for row in image_rows:
        digest = row["image_sha256"]
        if digest is None:
            continue
        previous = exact_sha_first.get(digest)
        if previous and previous[0] != row["procedure_id"]:
            if uf.union(previous[0], row["procedure_id"]):
                edge = {
                    "type": "exact_image_sha256", "procedures": [previous[0], row["procedure_id"]],
                    "image_ids": [previous[1], row["image_id"]], "image_sha256": digest
                }
                union_edges.append(edge); exact_edges.append(edge)
        else:
            exact_sha_first[digest] = (row["procedure_id"], row["image_id"])

    args.output_dir.mkdir(parents=True)
    (args.output_dir / "vqa").mkdir()
    hashes_path = args.output_dir / "perceptual_hashes.jsonl.gz"
    hash_rows: list[dict[str, Any]] = []
    hash_errors: list[dict[str, str]] = []
    with gzip.open(hashes_path, "wt", encoding="utf-8", compresslevel=9) as out:
        for index, row in enumerate(image_rows, start=1):
            if row["image_path"] is None:
                hashed = {**row, "phash64": None, "dhash64": None, "hash_status": "missing_asset"}
            else:
                try:
                    phash, dhash = perceptual_hashes(ROOT / row["image_path"])
                    hashed = {**row, "phash64": f"{phash:016x}", "dhash64": f"{dhash:016x}", "hash_status": "ok"}
                except Exception as exc:
                    hashed = {**row, "phash64": None, "dhash64": None, "hash_status": "error"}
                    hash_errors.append({"image_id": row["image_id"], "error": str(exc)})
            hash_rows.append(hashed)
            write_jsonl(out, hashed)
            if index % 5000 == 0:
                print(f"hashed {index}/{len(image_rows)} images", flush=True)

    tree = BKTree()
    indexed: list[dict[str, Any]] = []
    near_candidate_pairs = 0
    near_union_edges = 0
    for row in hash_rows:
        if row["hash_status"] != "ok":
            continue
        phash, dhash = int(row["phash64"], 16), int(row["dhash64"], 16)
        for candidate_index, phash_distance in tree.query(phash, args.phash_distance):
            other = indexed[candidate_index]
            if other["procedure_id"] == row["procedure_id"]:
                continue
            if other["width"] and row["width"] and other["height"] and row["height"]:
                aspect_left = other["width"] / other["height"]
                aspect_right = row["width"] / row["height"]
                if abs(aspect_left - aspect_right) / max(aspect_left, aspect_right) > 0.01:
                    continue
            dhash_distance = (int(other["dhash64"], 16) ^ dhash).bit_count()
            if dhash_distance > args.dhash_distance:
                continue
            near_candidate_pairs += 1
            if uf.union(other["procedure_id"], row["procedure_id"]):
                near_union_edges += 1
                union_edges.append({
                    "type": "near_duplicate_perceptual_hash",
                    "procedures": [other["procedure_id"], row["procedure_id"]],
                    "image_ids": [other["image_id"], row["image_id"]],
                    "phash_distance": phash_distance, "dhash_distance": dhash_distance
                })
        tree.add(phash, len(indexed))
        indexed.append(row)

    component_procedures: dict[str, list[str]] = defaultdict(list)
    for procedure in procedures:
        component_procedures[uf.find(procedure)].append(procedure)
    groups: dict[str, dict[str, Any]] = {}
    procedure_to_group: dict[str, str] = {}
    for proc_ids in component_procedures.values():
        proc_ids = sorted(proc_ids)
        group_id = stable_id("leakage_group", *proc_ids)
        group = {"procedure_ids": proc_ids, "patient_ids": set(), "counts": Counter(), "source_groups": set()}
        for procedure in proc_ids:
            procedure_to_group[procedure] = group_id
            group["patient_ids"].update(procedures[procedure]["patient_ids"])
            group["counts"].update(procedures[procedure]["counts"])
            group["source_groups"].update(procedures[procedure]["source_groups"])
        groups[group_id] = group

    assignment, objective = assign_groups(groups, args.seed)
    procedure_split = {procedure: assignment[group_id] for procedure, group_id in procedure_to_group.items()}

    assignment_path = args.output_dir / "split_assignments.jsonl.gz"
    with gzip.open(assignment_path, "wt", encoding="utf-8", compresslevel=9) as out:
        for row in image_rows:
            write_jsonl(out, {
                "image_id": row["image_id"], "patient_id": row["patient_id"],
                "procedure_id": row["procedure_id"], "leakage_group_id": procedure_to_group[row["procedure_id"]],
                "split": procedure_split[row["procedure_id"]]
            })

    procedure_path = args.output_dir / "procedure_assignments.csv"
    with procedure_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["patient_id", "procedure_id", "leakage_group_id", "split", "frames", *STRATIFY_LABELS])
        for procedure, data in sorted(procedures.items()):
            patient = sorted(data["patient_ids"])[0]
            group_id = procedure_to_group[procedure]
            writer.writerow([patient, procedure, group_id, assignment[group_id], data["counts"]["images"], *[data["counts"][x] for x in STRATIFY_LABELS]])

    edge_path = args.output_dir / "leakage_group_edges.jsonl.gz"
    with gzip.open(edge_path, "wt", encoding="utf-8", compresslevel=9) as out:
        for edge in union_edges:
            write_jsonl(out, edge)

    group_path = args.output_dir / "leakage_groups.jsonl.gz"
    with gzip.open(group_path, "wt", encoding="utf-8", compresslevel=9) as out:
        for group_id, data in sorted(groups.items()):
            if len(data["procedure_ids"]) > 1:
                write_jsonl(out, {
                    "leakage_group_id": group_id, "procedure_ids": data["procedure_ids"],
                    "patient_ids": sorted(data["patient_ids"]), "frames": data["counts"]["images"],
                    "split": assignment[group_id]
                })

    derivations: dict[str, dict[str, Any]] = {}
    with gzip.open(args.stage2_dir / "region_derivations.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            derivations[row["image_id"]] = row

    vqa_paths = {split: args.output_dir / "vqa" / f"{split}.jsonl.gz" for split in TARGET_RATIOS}
    vqa_handles = {split: gzip.open(path, "wt", encoding="utf-8", compresslevel=9) for split, path in vqa_paths.items()}
    vqa_counts: Counter[str] = Counter()
    instance_counts: dict[str, Counter[str]] = {split: Counter() for split in TARGET_RATIOS}
    image_label_counts: dict[str, Counter[str]] = {split: Counter() for split in TARGET_RATIOS}
    split_image_counts: Counter[str] = Counter()
    split_procedures: dict[str, set[str]] = defaultdict(set)
    split_patients: dict[str, set[str]] = defaultdict(set)
    split_groups: dict[str, set[str]] = defaultdict(set)
    question_index = 0
    try:
        with gzip.open(canonical_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                split = procedure_split[record["procedure_id"]]
                split_image_counts[split] += 1
                split_procedures[split].add(record["procedure_id"])
                split_patients[split].add(record["patient_id"])
                split_groups[split].add(procedure_to_group[record["procedure_id"]])
                image_label_counts[split].update(image_major_labels(record))
                derived = derivations[record["image_id"]]
                canonical_by_id = {region["region_id"]: region for region in record["regions"]}
                resolved_regions = []
                for derived_region in derived["regions"]:
                    source_region = canonical_by_id[derived_region["training_geometry_source_region_id"]]
                    resolved_regions.append({
                        "derived": derived_region,
                        "labels": derived_region["labels_override"] or source_region["labels"],
                        "polygon": derived_region["training_polygon_override"] or source_region["polygon"],
                    })
                normal = record["image_labels"]["normal"]
                if normal is not None and derived["image_training_status"] == "accepted" and record["image_path"]:
                    question_index += 1; vqa_counts[split] += 1
                    evidence = [
                        item["derived"]["canonical_region_id"]
                        for item in resolved_regions if item["labels"]["pathology_group"]
                    ]
                    write_jsonl(vqa_handles[split], {
                        "question_id": f"vqa3_{question_index:08d}", "split": split,
                        "image_id": record["image_id"], "patient_id": record["patient_id"],
                        "procedure_id": record["procedure_id"], "image_path": record["image_path"],
                        "question": "Hình ảnh nội soi này có được xác nhận là bình thường không?",
                        "answer": {"normal": normal}, "evidence_region_ids": evidence,
                        "provenance": "canonical_v2.2 + derived_v2.0 + split_v3"
                    })
                for item in resolved_regions:
                    derived_region, labels, polygon = item["derived"], item["labels"], item["polygon"]
                    if derived_region["training_status"] == "accepted":
                        for label in set(positive_paths(labels)):
                            instance_counts[split][label] += 1
                    if (
                        derived_region["training_status"] != "accepted"
                        or not labels["pathology_group"]
                        or not record["image_path"] or not record["width"] or not record["height"]
                    ):
                        continue
                    question_index += 1; vqa_counts[split] += 1
                    write_jsonl(vqa_handles[split], {
                        "question_id": f"vqa3_{question_index:08d}", "split": split,
                        "image_id": record["image_id"], "patient_id": record["patient_id"],
                        "procedure_id": record["procedure_id"], "image_path": record["image_path"],
                        "question": "Vùng được chỉ định có những phát hiện bệnh lý nào?",
                        "answer": labels,
                        "evidence_region_ids": [derived_region["canonical_region_id"]],
                        "bbox_xyxy_normalized": bbox(polygon, record["width"], record["height"]),
                        "provenance": "canonical_v2.2 + derived_v2.0 + split_v3"
                    })
    finally:
        for handle in vqa_handles.values():
            handle.close()

    policy_path = args.output_dir / "class_evaluation_policy.csv"
    all_class_names = sorted(set().union(*(counter.keys() for counter in instance_counts.values())))
    with policy_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "train_instances", "validation_instances", "test_instances", "total_instances", "evaluation_policy"])
        for label in all_class_names:
            values = [instance_counts[split][label] for split in TARGET_RATIOS]
            total = sum(values)
            policy = "report_ap" if values[2] >= args.min_test_instances and total >= 50 else "parent_only_or_exploratory"
            writer.writerow([label, *values, total, policy])

    distribution_path = args.output_dir / "stratification_distribution.csv"
    with distribution_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "images", "image_ratio", *STRATIFY_LABELS])
        for split in TARGET_RATIOS:
            writer.writerow([
                split, split_image_counts[split], split_image_counts[split] / len(image_rows),
                *[image_label_counts[split][label] for label in STRATIFY_LABELS]
            ])

    patient_overlap = {}
    procedure_overlap = {}
    group_overlap = {}
    names = list(TARGET_RATIOS)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            key = f"{left}__{right}"
            patient_overlap[key] = len(split_patients[left] & split_patients[right])
            procedure_overlap[key] = len(split_procedures[left] & split_procedures[right])
            group_overlap[key] = len(split_groups[left] & split_groups[right])
    report = {
        "dataset_version": VERSION,
        "source_versions": [canonical_manifest["dataset_version"], stage2_manifest["dataset_version"]],
        "seed": args.seed,
        "grouping_hierarchy": ["patient_id", "procedure_id", "frame/image_id"],
        "patient_id_audit": {
            "unique_patients": len(patient_procedures), "unique_procedures": len(procedures),
            "patients_with_multiple_procedures": sum(len(v) > 1 for v in patient_procedures.values()),
            "interpretation": "patient_id is one-to-one with procedure_id in this release; procedure is the effective minimum leakage group"
        },
        "near_duplicate_policy": {
            "algorithm": "64-bit DCT pHash AND 64-bit dHash, cross-procedure only",
            "phash_max_hamming": args.phash_distance, "dhash_max_hamming": args.dhash_distance,
            "aspect_ratio_relative_difference_max": 0.01,
            "hashed_images": sum(r["hash_status"] == "ok" for r in hash_rows),
            "missing_or_error_images": sum(r["hash_status"] != "ok" for r in hash_rows),
            "candidate_pairs": near_candidate_pairs, "union_edges": near_union_edges,
            "exact_sha_union_edges": len(exact_edges), "hash_errors": hash_errors
        },
        "leakage_groups": {
            "total": len(groups),
            "multi_procedure": sum(len(g["procedure_ids"]) > 1 for g in groups.values()),
            "largest_procedure_count": max(len(g["procedure_ids"]) for g in groups.values()),
            "largest_frame_count": max(g["counts"]["images"] for g in groups.values())
        },
        "split_policy": {
            "targets": TARGET_RATIOS, "allowed_bounds": RATIO_BOUNDS,
            "stratification_unit": "positive image counts within leakage groups",
            "stratification_labels": list(STRATIFY_LABELS), "optimization_objective": objective
        },
        "splits": {
            split: {
                "images": split_image_counts[split], "image_ratio": split_image_counts[split] / len(image_rows),
                "patients": len(split_patients[split]), "procedures": len(split_procedures[split]),
                "leakage_groups": len(split_groups[split]), "major_label_positive_images": dict(image_label_counts[split]),
                "vqa_records": vqa_counts[split]
            }
            for split in TARGET_RATIOS
        },
        "leakage_checks": {
            "patient_overlap": patient_overlap, "procedure_overlap": procedure_overlap,
            "near_duplicate_group_overlap": group_overlap,
            "passed": not any(patient_overlap.values()) and not any(procedure_overlap.values()) and not any(group_overlap.values())
        },
        "rare_class_policy": {
            "minimum_test_instances_for_ap": args.min_test_instances,
            "rule": "Classes below the threshold are parent-only or exploratory and must not receive a standalone reported AP"
        }
    }
    report_path = args.output_dir / "split_report.json"
    write_json(report_path, report)

    readme_path = args.output_dir / "README.md"
    readme_path.write_text(
        "# derived_v3\n\nLeakage-safe group split is frozen before VQA generation. Grouping is patient → procedure → frame, with cross-procedure exact/perceptual duplicate unions. Multi-label stratification uses only mucosal, tumor, stenosis, secretion, and confirmed normal.\n\nVQA files are generated after assignment and stored separately by split. Rare subclasses are governed by `class_evaluation_policy.csv`.\n",
        encoding="utf-8"
    )
    output_files = [
        hashes_path, assignment_path, procedure_path, edge_path, group_path, policy_path,
        distribution_path, report_path, readme_path, *vqa_paths.values()
    ]
    manifest = {
        "dataset_version": VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_manifest_sha256": "sha256:" + sha256_file(args.canonical_dir / "manifest.json"),
        "stage2_manifest_sha256": "sha256:" + sha256_file(args.stage2_dir / "manifest.json"),
        "conversion_script_sha256": "sha256:" + sha256_file(Path(__file__)),
        "split_frozen_before_vqa": True,
        "outputs": [
            {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in output_files
        ]
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({"output": str(args.output_dir), "leakage_passed": report["leakage_checks"]["passed"], "splits": report["splits"]}, indent=2))


def verify(args: argparse.Namespace) -> None:
    manifest = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
    failures = []
    for row in manifest["outputs"]:
        path = ROOT / row["path"]
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            failures.append(row["path"])
    report = json.loads((args.output_dir / "split_report.json").read_text(encoding="utf-8"))
    if not report["leakage_checks"]["passed"]:
        failures.append("split_report leakage check")
    print(json.dumps({"valid": not failures, "failures": failures}, indent=2))
    if failures:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    create = sub.add_parser("create")
    create.add_argument("--canonical-dir", type=Path, default=CANONICAL_DIR)
    create.add_argument("--stage2-dir", type=Path, default=STAGE2_DIR)
    create.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    create.add_argument("--seed", type=int, default=42)
    create.add_argument("--phash-distance", type=int, default=PHASH_DISTANCE)
    create.add_argument("--dhash-distance", type=int, default=DHASH_DISTANCE)
    create.add_argument("--min-test-instances", type=int, default=MIN_TEST_INSTANCES_FOR_AP)
    create.set_defaults(func=build)
    check = sub.add_parser("verify")
    check.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    check.set_defaults(func=verify)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
