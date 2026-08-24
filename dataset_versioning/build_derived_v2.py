#!/usr/bin/env python3
"""Build reversible Stage-2 annotations from canonical_v2.2 only.

The immutable doctor annotations and canonical polygons are never modified. Any
safe technical normalization is stored separately as ``training_polygon``.
COCO contains both bbox and segmentation so the same file supports detection
and instance-segmentation training without duplicating a large JSON payload.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageChops, ImageDraw
from shapely.geometry import Polygon, box


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL_DIR = ROOT / "canonical_v2_2"
DEFAULT_OUTPUT_DIR = ROOT / "derived_v2"
DERIVED_VERSION = "derived_v2.0"
EXPECTED_CANONICAL_VERSION = "canonical_v2.2"
IOU_DUPLICATE = 0.98
IOU_REVIEW = 0.80
BOUNDARY_EPSILON_PX = 0.5
MAX_CLIP_AREA_FRACTION = 0.001
RARE_LABEL_THRESHOLD = 50
CHUNK_SIZE = 1024 * 1024


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(prefix: str, *parts: str, length: int = 24) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def patient_split(patient_id: str) -> str:
    bucket = int(hashlib.sha256(patient_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "train" if bucket < 70 else "val" if bucket < 85 else "test"


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(handle: Any, value: Any) -> None:
    handle.write(canonical_json(value) + "\n")


def semantic_signature(labels: dict[str, Any]) -> str:
    return canonical_json(labels)


def merge_values(left: Any, right: Any, path: str = "") -> tuple[Any, list[str]]:
    """Positive-only union while surfacing contradictory known scalar values."""
    if isinstance(left, dict) and isinstance(right, dict):
        result: dict[str, Any] = {}
        conflicts: list[str] = []
        for key in sorted(set(left) | set(right)):
            merged, sub = merge_values(left.get(key), right.get(key), f"{path}.{key}" if path else key)
            result[key] = merged
            conflicts.extend(sub)
        return result, conflicts
    if isinstance(left, list) and isinstance(right, list):
        return sorted({canonical_json(v): v for v in left + right}.values(), key=canonical_json), []
    if left is None:
        return right, []
    if right is None or left == right:
        return left, []
    return left, [path]


def normalize_training_polygon(
    points: list[list[float]], width: int | None, height: int | None
) -> tuple[list[list[float]] | None, str, list[str], dict[str, Any]]:
    """Return a separate training polygon; never mutate the canonical polygon."""
    flags: list[str] = []
    clean: list[list[float]] = []
    for point in points:
        if len(point) != 2 or not all(math.isfinite(float(v)) for v in point):
            return None, "review_required", ["non_finite_or_malformed_point"], {}
        item = [float(point[0]), float(point[1])]
        if not clean or item != clean[-1]:
            clean.append(item)
        else:
            flags.append("removed_consecutive_duplicate_vertex")
    if len(clean) > 1 and clean[0] == clean[-1]:
        clean.pop()
        flags.append("removed_duplicate_closing_vertex")
    if len({tuple(p) for p in clean}) < 3:
        return None, "review_required", sorted(set(flags + ["fewer_than_three_unique_vertices"])), {}
    try:
        polygon = Polygon(clean)
    except Exception as exc:
        return None, "review_required", sorted(set(flags + ["polygon_constructor_error"])), {"error": str(exc)}
    if polygon.is_empty or polygon.area <= 0:
        return None, "review_required", sorted(set(flags + ["zero_area_polygon"])), {}
    if not polygon.is_valid:
        return None, "review_required", sorted(set(flags + ["self_intersection_or_invalid_topology"])), {}
    if width is None or height is None:
        return None, "review_required", sorted(set(flags + ["missing_image_dimensions"])), {}
    minx, miny, maxx, maxy = polygon.bounds
    overflow = max(0.0, -minx, -miny, maxx - width, maxy - height)
    if overflow > BOUNDARY_EPSILON_PX:
        return None, "review_required", sorted(set(flags + ["polygon_out_of_bounds"])), {"overflow_px": overflow}
    if overflow > 0:
        clipped = polygon.intersection(box(0, 0, width, height))
        if clipped.geom_type != "Polygon" or clipped.is_empty:
            return None, "review_required", sorted(set(flags + ["boundary_clip_not_single_polygon"])), {}
        fraction = abs(polygon.area - clipped.area) / polygon.area
        if fraction > MAX_CLIP_AREA_FRACTION:
            return None, "review_required", sorted(set(flags + ["boundary_clip_area_change_too_large"])), {
                "area_change_fraction": fraction
            }
        clean = [[float(x), float(y)] for x, y in list(clipped.exterior.coords)[:-1]]
        flags.append("clipped_subpixel_boundary_overflow")
        return clean, "accepted", sorted(set(flags)), {"area_change_fraction": fraction}
    return clean, "accepted", sorted(set(flags)), {}


def polygon_bounds(points: list[list[float]]) -> list[float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def bbox_intersects(a: list[float], b: list[float]) -> bool:
    return min(a[2], b[2]) >= max(a[0], b[0]) and min(a[3], b[3]) >= max(a[1], b[1])


def mask_iou(a: list[list[float]], b: list[list[float]], width: int, height: int) -> float:
    bounds_a, bounds_b = polygon_bounds(a), polygon_bounds(b)
    left = max(0, math.floor(min(bounds_a[0], bounds_b[0])))
    top = max(0, math.floor(min(bounds_a[1], bounds_b[1])))
    right = min(width - 1, math.ceil(max(bounds_a[2], bounds_b[2])))
    bottom = min(height - 1, math.ceil(max(bounds_a[3], bounds_b[3])))
    if right < left or bottom < top:
        return 0.0
    size = (right - left + 1, bottom - top + 1)
    masks = []
    for points in (a, b):
        image = Image.new("1", size, 0)
        ImageDraw.Draw(image).polygon([(x - left, y - top) for x, y in points], fill=1)
        masks.append(image)
    intersection = ImageChops.logical_and(masks[0], masks[1]).histogram()[255]
    union = ImageChops.logical_or(masks[0], masks[1]).histogram()[255]
    return intersection / union if union else 0.0


def positive_label_paths(value: Any, prefix: str = "") -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else key
            yield from positive_label_paths(child, next_prefix)
    elif isinstance(value, list):
        for child in value:
            yield f"{prefix}:{child}"
    elif value is True:
        yield prefix
    elif isinstance(value, str):
        yield f"{prefix}:{value}"


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> int:
        a, b = self.find(a), self.find(b)
        if a != b:
            if b < a:
                a, b = b, a
            self.parent[b] = a
        return a


def verify_canonical(canonical_dir: Path) -> tuple[dict[str, Any], Path]:
    manifest_path = canonical_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["dataset_version"] != EXPECTED_CANONICAL_VERSION:
        raise ValueError(f"Expected {EXPECTED_CANONICAL_VERSION}, got {manifest['dataset_version']}")
    records = canonical_dir / "images.jsonl.gz"
    expected = next(row["sha256"] for row in manifest["outputs"] if row["path"].endswith("images.jsonl.gz"))
    if sha256_file(records) != expected:
        raise ValueError("Canonical images.jsonl.gz checksum mismatch")
    return manifest, records


def build(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    canonical_manifest, records_path = verify_canonical(args.canonical_dir)
    if shutil_free := os.statvfs(ROOT):
        free_bytes = shutil_free.f_bavail * shutil_free.f_frsize
        if free_bytes < 50 * 1024 * 1024:
            raise OSError(f"Only {free_bytes} bytes free; refusing to start derived build")

    args.output_dir.mkdir(parents=True)
    (args.output_dir / "coco").mkdir()
    (args.output_dir / "vqa").mkdir()
    clean_path = args.output_dir / "region_derivations.jsonl.gz"
    log_path = args.output_dir / "transformation_log.jsonl.gz"
    review_path = args.output_dir / "review_queue.jsonl.gz"
    vqa_path = args.output_dir / "vqa" / "grounded_vqa.jsonl.gz"
    coco_path = args.output_dir / "coco" / "instances_detection_and_segmentation.json.gz"

    counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    split_images: Counter[str] = Counter()
    split_patients: dict[str, set[str]] = defaultdict(set)
    review_region_ids: set[str] = set()
    review_source_annotations: set[str] = set()
    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    review_csv_rows: list[dict[str, Any]] = []
    event_seq = 0
    vqa_seq = 0
    coco_ann_id = 0

    with (
        gzip.open(records_path, "rt", encoding="utf-8") as source,
        gzip.open(clean_path, "wt", encoding="utf-8", compresslevel=9) as clean_out,
        gzip.open(log_path, "wt", encoding="utf-8", compresslevel=9) as log_out,
        gzip.open(review_path, "wt", encoding="utf-8", compresslevel=9) as review_out,
        gzip.open(vqa_path, "wt", encoding="utf-8", compresslevel=9) as vqa_out,
    ):
        for image_index, line in enumerate(source, start=1):
            image = json.loads(line)
            counts["images_total"] += 1
            normal = image["image_labels"]["normal"]
            counts[f"images_normal_{'true' if normal is True else 'false' if normal is False else 'null'}"] += 1
            split = patient_split(image["patient_id"])
            split_images[split] += 1
            split_patients[split].add(image["patient_id"])
            base_regions: list[dict[str, Any]] = []
            image_review: set[str] = set()

            for region in image["regions"]:
                counts["source_polygons_initial"] += len(region["source_evidence"])
                counts["canonical_regions_initial"] += 1
                canonical_region_id = stable_id("canonical_region", image["image_id"], region["geometry_hash"])
                evidence_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for evidence in region["source_evidence"]:
                    evidence_groups[semantic_signature(evidence["canonical_labels"])].append(evidence)
                for evidence_rows in evidence_groups.values():
                    if len(evidence_rows) > 1:
                        kept = sorted(e["source_annotation_id"] for e in evidence_rows)[0]
                        for removed in sorted(e["source_annotation_id"] for e in evidence_rows)[1:]:
                            counts["duplicates_same_label_removed"] += 1
                            event_seq += 1
                            append_jsonl(log_out, {
                                "event_id": f"event_{event_seq:08d}", "type": "A_same_geometry_same_label",
                                "image_id": image["image_id"], "canonical_region_id": canonical_region_id,
                                "kept_source_annotation_id": kept, "removed_source_annotation_id": removed,
                                "mask_iou": 1.0, "reversible": True,
                                "rollback": "Re-read all source_evidence from canonical_v2.2"
                            })
                if len(evidence_groups) > 1:
                    counts["regions_merged_multilabel"] += 1
                    event_seq += 1
                    append_jsonl(log_out, {
                        "event_id": f"event_{event_seq:08d}", "type": "B_same_geometry_different_label",
                        "image_id": image["image_id"], "canonical_region_id": canonical_region_id,
                        "source_annotation_ids": region["source_annotation_ids"], "mask_iou": 1.0,
                        "merged_label_signatures": len(evidence_groups), "reversible": True,
                        "rollback": "Use per-annotation canonical_labels in canonical_v2.2 source_evidence"
                    })

                training_polygon, status, geometry_flags, detail = normalize_training_polygon(
                    region["polygon"], image["width"], image["height"]
                )
                if geometry_flags:
                    event_seq += 1
                    append_jsonl(log_out, {
                        "event_id": f"event_{event_seq:08d}", "type": "technical_geometry_check",
                        "image_id": image["image_id"], "canonical_region_id": canonical_region_id,
                        "canonical_geometry_hash": region["geometry_hash"],
                        "training_polygon_created": training_polygon is not None,
                        "quality_flags": geometry_flags, "detail": detail, "reversible": True,
                        "rollback": "Discard training_polygon; canonical polygon is unchanged"
                    })
                item = {
                    "canonical_region_id": canonical_region_id,
                    "source_region_ids": [region["region_id"]],
                    "canonical_geometry_hashes": [region["geometry_hash"]],
                    "canonical_polygon": region["polygon"],
                    "training_polygon": training_polygon,
                    "labels": region["labels"],
                    "source_annotation_ids": region["source_annotation_ids"],
                    "source_label_ids": region["source_label_ids"],
                    "training_status": status,
                    "quality_flags": sorted(set(region["quality_flags"] + geometry_flags)),
                }
                base_regions.append(item)
                if status != "accepted":
                    counts["invalid_canonical_regions"] += 1
                    counts["invalid_source_polygons"] += len(region["source_evidence"])
                    image_review.add(canonical_region_id)
                    review_region_ids.add(canonical_region_id)
                    review_source_annotations.update(region["source_annotation_ids"])
                    review = {
                        "review_id": stable_id("review", image["image_id"], canonical_region_id, "geometry"),
                        "reason": "invalid_or_uncertain_geometry", "image_id": image["image_id"],
                        "image_path": image["image_path"], "patient_id": image["patient_id"],
                        "procedure_id": image["procedure_id"], "canonical_region_ids": [canonical_region_id],
                        "source_annotation_ids": region["source_annotation_ids"], "mask_iou": None,
                        "quality_flags": geometry_flags, "bbox_xyxy": region["bbox_xyxy"],
                        "labels": region["labels"]
                    }
                    append_jsonl(review_out, review)
                    review_csv_rows.append({k: canonical_json(v) if isinstance(v, (list, dict)) else v for k, v in review.items() if k != "labels"})

            valid_indices = [i for i, r in enumerate(base_regions) if r["training_status"] == "accepted"]
            edges: list[tuple[float, int, int]] = []
            for offset, i in enumerate(valid_indices):
                for j in valid_indices[offset + 1:]:
                    if not bbox_intersects(polygon_bounds(base_regions[i]["training_polygon"]), polygon_bounds(base_regions[j]["training_polygon"])):
                        continue
                    iou = mask_iou(base_regions[i]["training_polygon"], base_regions[j]["training_polygon"], image["width"], image["height"])
                    if iou >= IOU_REVIEW:
                        edges.append((iou, i, j))

            uf = UnionFind(len(base_regions))
            component_labels = {i: base_regions[i]["labels"] for i in range(len(base_regions))}
            for iou, i, j in sorted(edges, key=lambda row: (-row[0], row[1], row[2])):
                left, right = uf.find(i), uf.find(j)
                ids = [base_regions[i]["canonical_region_id"], base_regions[j]["canonical_region_id"]]
                if iou < IOU_DUPLICATE:
                    counts["overlap_pairs_for_review"] += 1
                    image_review.update(ids)
                    review_region_ids.update(ids)
                    for idx in (i, j):
                        review_source_annotations.update(base_regions[idx]["source_annotation_ids"])
                    review = {
                        "review_id": stable_id("review", image["image_id"], *sorted(ids), f"{iou:.8f}"),
                        "reason": "C_partial_overlap_0.80_to_0.98", "image_id": image["image_id"],
                        "image_path": image["image_path"], "patient_id": image["patient_id"],
                        "procedure_id": image["procedure_id"], "canonical_region_ids": sorted(ids),
                        "source_annotation_ids": sorted(set(base_regions[i]["source_annotation_ids"] + base_regions[j]["source_annotation_ids"])),
                        "mask_iou": round(iou, 8), "quality_flags": [],
                        "bbox_xyxy": [polygon_bounds(base_regions[i]["training_polygon"]), polygon_bounds(base_regions[j]["training_polygon"])]
                    }
                    append_jsonl(review_out, review)
                    review_csv_rows.append({k: canonical_json(v) if isinstance(v, (list, dict)) else v for k, v in review.items()})
                    event_seq += 1
                    append_jsonl(log_out, {
                        "event_id": f"event_{event_seq:08d}", "type": "C_partial_overlap_review",
                        "image_id": image["image_id"], "canonical_region_ids": sorted(ids),
                        "mask_iou": round(iou, 8), "action": "no_merge", "reversible": True
                    })
                    continue
                if left == right:
                    continue
                merged, conflicts = merge_values(component_labels[left], component_labels[right])
                if conflicts:
                    counts["label_conflict_pairs_for_review"] += 1
                    image_review.update(ids)
                    review_region_ids.update(ids)
                    for idx in (i, j):
                        review_source_annotations.update(base_regions[idx]["source_annotation_ids"])
                    review = {
                        "review_id": stable_id("review", image["image_id"], *sorted(ids), "label_conflict"),
                        "reason": "label_conflict_prevents_auto_merge", "image_id": image["image_id"],
                        "image_path": image["image_path"], "patient_id": image["patient_id"],
                        "procedure_id": image["procedure_id"], "canonical_region_ids": sorted(ids),
                        "source_annotation_ids": sorted(set(base_regions[i]["source_annotation_ids"] + base_regions[j]["source_annotation_ids"])),
                        "mask_iou": round(iou, 8), "quality_flags": conflicts,
                        "bbox_xyxy": [polygon_bounds(base_regions[i]["training_polygon"]), polygon_bounds(base_regions[j]["training_polygon"])]
                    }
                    append_jsonl(review_out, review)
                    review_csv_rows.append({k: canonical_json(v) if isinstance(v, (list, dict)) else v for k, v in review.items()})
                    continue
                event_type = "A_near_geometry_same_label" if semantic_signature(component_labels[left]) == semantic_signature(component_labels[right]) else "B_near_geometry_different_label"
                counts["duplicates_same_label_removed" if event_type.startswith("A") else "regions_merged_multilabel"] += 1
                root = uf.union(left, right)
                component_labels[root] = merged
                event_seq += 1
                append_jsonl(log_out, {
                    "event_id": f"event_{event_seq:08d}", "type": event_type,
                    "image_id": image["image_id"], "canonical_region_ids": sorted(ids),
                    "mask_iou": round(iou, 8), "action": "merge_for_training", "reversible": True,
                    "rollback": "Use source_region_ids and canonical polygons"
                })

            components: dict[int, list[int]] = defaultdict(list)
            for i in range(len(base_regions)):
                components[uf.find(i)].append(i)
            cleaned_regions: list[dict[str, Any]] = []
            for root, members in sorted(components.items()):
                representative = max(members, key=lambda i: (Polygon(base_regions[i]["training_polygon"]).area if base_regions[i]["training_polygon"] else -1, base_regions[i]["canonical_region_id"]))
                rows = [base_regions[i] for i in members]
                merged_id = rows[0]["canonical_region_id"] if len(rows) == 1 else stable_id(
                    "canonical_region", image["image_id"], *sorted(r["canonical_region_id"] for r in rows)
                )
                status = (
                    "review_required"
                    if any(r["canonical_region_id"] in image_review for r in rows)
                    else base_regions[representative]["training_status"]
                )
                cleaned = {
                    "canonical_region_id": merged_id,
                    "source_region_ids": sorted({z for r in rows for z in r["source_region_ids"]}),
                    "canonical_geometry_hashes": sorted({z for r in rows for z in r["canonical_geometry_hashes"]}),
                    "canonical_polygons": [r["canonical_polygon"] for r in rows],
                    "training_polygon": base_regions[representative]["training_polygon"],
                    "labels": component_labels[uf.find(root)],
                    "source_annotation_ids": sorted({z for r in rows for z in r["source_annotation_ids"]}),
                    "source_label_ids": sorted({z for r in rows for z in r["source_label_ids"]}),
                    "training_status": status,
                    "quality_flags": sorted({z for r in rows for z in r["quality_flags"]}),
                    "_training_geometry_source_region_id": base_regions[representative]["source_region_ids"][0],
                    "_training_polygon_override": (
                        base_regions[representative]["training_polygon"]
                        if base_regions[representative]["training_polygon"]
                        != base_regions[representative]["canonical_polygon"]
                        else None
                    ),
                    "_labels_override": component_labels[uf.find(root)] if len(rows) > 1 else None,
                }
                cleaned_regions.append(cleaned)
                for group in cleaned["labels"]["pathology_group"]:
                    group_counts[group] += 1
                for label in positive_label_paths(cleaned["labels"]):
                    label_counts[label] += 1
            counts["canonical_regions_after_cleaning"] += len(cleaned_regions)

            pathology_regions = [r for r in cleaned_regions if r["labels"]["pathology_group"]]
            if not pathology_regions:
                counts["images_without_pathology_regions"] += 1
                if normal is None:
                    counts["unknown_images_without_pathology_regions"] += 1
            if normal is True:
                counts["confirmed_negative_images"] += 1
            if image_review:
                counts["images_requiring_review"] += 1

            region_derivations = [
                {
                    "canonical_region_id": r["canonical_region_id"],
                    "source_region_ids": r["source_region_ids"],
                    "training_geometry_source_region_id": r["_training_geometry_source_region_id"],
                    "training_polygon_override": r["_training_polygon_override"],
                    "labels_override": r["_labels_override"],
                    "training_status": r["training_status"],
                    "quality_flags": r["quality_flags"],
                }
                for r in cleaned_regions
            ]
            clean_record = {
                "dataset_version": DERIVED_VERSION, "source_canonical_version": EXPECTED_CANONICAL_VERSION,
                "image_id": image["image_id"], "patient_id": image["patient_id"],
                "procedure_id": image["procedure_id"], "split": split, "normal": normal,
                "regions": region_derivations,
                "image_training_status": "review_required" if image_review else "accepted"
            }
            append_jsonl(clean_out, clean_record)

            eligible = not image_review and image["image_path"] is not None and image["width"] is not None and image["height"] is not None
            accepted = [r for r in cleaned_regions if r["training_status"] == "accepted" and (r["labels"]["pathology_group"] or r["labels"]["anatomy"])]
            include_coco = eligible and (bool(accepted) or normal is True)
            if include_coco:
                coco_image_id = image_index
                coco_images.append({
                    "id": coco_image_id, "file_name": image["image_path"], "width": image["width"],
                    "height": image["height"], "canonical_image_id": image["image_id"],
                    "patient_id": image["patient_id"], "procedure_id": image["procedure_id"],
                    "split": split, "confirmed_normal": normal is True
                })
                counts["coco_images"] += 1
                if normal is True and not accepted:
                    counts["coco_confirmed_negative_images"] += 1
                for region in accepted:
                    poly = region["training_polygon"]
                    bbox = polygon_bounds(poly)
                    coco_ann_id += 1
                    category_id = 2 if region["labels"]["pathology_group"] else 1
                    coco_annotations.append({
                        "id": coco_ann_id, "image_id": coco_image_id, "category_id": category_id,
                        "bbox": [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                        "area": float(Polygon(poly).area), "segmentation": [[v for point in poly for v in point]],
                        "iscrowd": 0, "canonical_region_id": region["canonical_region_id"],
                        "canonical_labels": region["labels"], "source_annotation_ids": region["source_annotation_ids"]
                    })
                    counts["coco_annotations"] += 1

            if normal is not None and not image_review and image["image_path"] is not None:
                vqa_seq += 1
                append_jsonl(vqa_out, {
                    "question_id": f"vqa_{vqa_seq:08d}", "image_id": image["image_id"],
                    "image_path": image["image_path"], "split": split,
                    "question": "Hình ảnh nội soi này có được xác nhận là bình thường không?",
                    "answer": {"normal": normal},
                    "evidence_region_ids": [
                        r["canonical_region_id"] for r in cleaned_regions
                        if r["labels"]["pathology_group"]
                    ],
                    "provenance": "canonical_v2.2"
                })
            for region in cleaned_regions:
                if region["training_status"] != "accepted" or not region["labels"]["pathology_group"] or image["image_path"] is None:
                    continue
                vqa_seq += 1
                bbox = polygon_bounds(region["training_polygon"])
                normalized_bbox = [bbox[0] / image["width"], bbox[1] / image["height"], bbox[2] / image["width"], bbox[3] / image["height"]]
                append_jsonl(vqa_out, {
                    "question_id": f"vqa_{vqa_seq:08d}", "image_id": image["image_id"],
                    "image_path": image["image_path"], "split": split,
                    "question": "Vùng được chỉ định có những phát hiện bệnh lý nào?",
                    "answer": region["labels"], "evidence_region_ids": [region["canonical_region_id"]],
                    "bbox_xyxy_normalized": normalized_bbox, "provenance": "canonical_v2.2"
                })
            counts["vqa_records"] = vqa_seq

    coco = {
        "info": {
            "description": "Bronchoscopy regions; one multi-label ROI per annotation. Supports bbox detection and instance segmentation.",
            "version": DERIVED_VERSION, "source": EXPECTED_CANONICAL_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat()
        },
        "licenses": [],
        "categories": [
            {"id": 1, "name": "anatomy_region", "supercategory": "bronchoscopy_region"},
            {"id": 2, "name": "pathology_region", "supercategory": "bronchoscopy_region"}
        ],
        "images": coco_images, "annotations": coco_annotations
    }
    with gzip.open(coco_path, "wt", encoding="utf-8", compresslevel=9) as handle:
        handle.write(canonical_json(coco))

    csv_path = args.output_dir / "review_queue.csv"
    csv_fields = [
        "review_id", "reason", "image_id", "image_path", "patient_id", "procedure_id",
        "canonical_region_ids", "source_annotation_ids", "mask_iou", "quality_flags", "bbox_xyxy"
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(review_csv_rows)

    counts["regions_requiring_review"] = len(review_region_ids)
    counts["source_polygons_requiring_review"] = len(review_source_annotations)
    report = {
        "dataset_version": DERIVED_VERSION,
        "source_canonical_version": EXPECTED_CANONICAL_VERSION,
        "thresholds": {
            "duplicate_mask_iou_gte": IOU_DUPLICATE, "review_mask_iou_gte": IOU_REVIEW,
            "review_mask_iou_lt": IOU_DUPLICATE, "boundary_epsilon_px": BOUNDARY_EPSILON_PX,
            "max_clip_area_fraction": MAX_CLIP_AREA_FRACTION, "rare_label_occurrences_lt": args.rare_threshold
        },
        "counts": dict(sorted(counts.items())),
        "regions_by_pathology_group": dict(sorted(group_counts.items())),
        "splits": {
            key: {"images": split_images[key], "patients": len(split_patients[key])}
            for key in ("train", "val", "test")
        },
        "negative_policy": {
            "confirmed_negative": "image_labels.normal is true",
            "unknown": "image_labels.normal is null; never exported as a negative",
            "coco": "Only confirmed-normal images can be empty-image negatives; any image with a review item is excluded."
        },
        "notes": [
            "Canonical polygons remain unchanged; training_polygon is a separate derived field.",
            "COCO segmentation and bbox are in one standard instances file to avoid duplicating data.",
            "VQA is template-derived solely from canonical labels and evidence; legacy raw VQA is not imported."
        ]
    }
    write_json(args.output_dir / "cleaning_report.json", report)

    with (args.output_dir / "pathology_group_counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["pathology_group", "region_count"])
        writer.writerows(sorted(group_counts.items()))
    rare = sorted((label, count) for label, count in label_counts.items() if count < args.rare_threshold)
    with (args.output_dir / "rare_labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["label", "region_count", "rare_threshold_lt"])
        writer.writerows((label, count, args.rare_threshold) for label, count in rare)
    summary_keys = [
        "source_polygons_initial", "duplicates_same_label_removed", "regions_merged_multilabel",
        "invalid_source_polygons", "invalid_canonical_regions", "regions_requiring_review",
        "source_polygons_requiring_review",
        "confirmed_negative_images", "images_without_pathology_regions", "unknown_images_without_pathology_regions"
    ]
    with (args.output_dir / "cleaning_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["metric", "count"])
        writer.writerows((key, counts[key]) for key in summary_keys)

    output_files = [
        clean_path, log_path, review_path, csv_path, coco_path, vqa_path,
        args.output_dir / "cleaning_report.json", args.output_dir / "cleaning_summary.csv",
        args.output_dir / "pathology_group_counts.csv", args.output_dir / "rare_labels.csv"
    ]
    manifest = {
        "dataset_version": DERIVED_VERSION,
        "source_dataset_version": canonical_manifest["dataset_version"],
        "source_hash": "sha256:" + sha256_file(records_path),
        "canonical_manifest_sha256": "sha256:" + sha256_file(args.canonical_dir / "manifest.json"),
        "conversion_script_sha256": "sha256:" + sha256_file(Path(__file__)),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "transformation_policy": "reversible_delta; canonical polygons never overwritten",
        "outputs": [
            {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in output_files
        ]
    }
    write_json(args.output_dir / "manifest.json", manifest)
    readme = """# derived_v2\n\nGenerated only from immutable `canonical_v2.2`. Doctor-source polygons are never edited.\n\n- `region_derivations.jsonl.gz`: sparse deltas referencing canonical regions; polygon/label overrides exist only when required.\n- `transformation_log.jsonl.gz`: reversible A/B/C and geometry events.\n- `review_queue.*`: ambiguous or invalid items for physician review.\n- `coco/instances_detection_and_segmentation.json.gz`: one COCO file with bbox and segmentation.\n- `vqa/grounded_vqa.jsonl.gz`: deterministic canonical-only grounded VQA.\n\nOnly `normal=true` images are eligible as empty-image negatives; `null` is never converted to `false`.\n"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "counts": dict(sorted(counts.items()))}, indent=2))


def verify(args: argparse.Namespace) -> None:
    manifest = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
    failures = []
    for row in manifest["outputs"]:
        path = ROOT / row["path"]
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            failures.append(row["path"])
    for path in args.output_dir.rglob("*.gz"):
        try:
            with gzip.open(path, "rb") as handle:
                while handle.read(CHUNK_SIZE):
                    pass
        except Exception as exc:
            failures.append(f"{path}: {exc}")
    print(json.dumps({"valid": not failures, "failures": failures}, indent=2))
    if failures:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    create = sub.add_parser("create")
    create.add_argument("--canonical-dir", type=Path, default=DEFAULT_CANONICAL_DIR)
    create.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    create.add_argument("--rare-threshold", type=int, default=RARE_LABEL_THRESHOLD)
    create.set_defaults(func=build)
    check = sub.add_parser("verify")
    check.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    check.set_defaults(func=verify)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
