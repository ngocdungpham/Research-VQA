#!/usr/bin/env python3
"""Build canonical_v2 from the immutable raw_v1 bronchoscopy snapshot.

Core invariants:
- one image record per globally unique canonical image ID;
- one region per image and normalized polygon geometry;
- multiple source labels are merged onto the same region;
- true, false, and null retain distinct meanings;
- missing annotations remain null and are never converted to false;
- source label and annotation IDs remain attached as evidence.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import re
import subprocess
import tarfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from PIL import Image
# from jsonschema import Draft202012Validator
from jsonschema import Draft202012Validator
# Or alternatively:
from jsonschema.validators import Draft202012Validator



ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_DIR = ROOT / "raw_v1"
DEFAULT_OUTPUT_DIR = ROOT / "canonical_v2"
DEFAULT_SCHEMA = SCRIPT_DIR / "canonical_schema_v2.json"
DEFAULT_TAXONOMY = SCRIPT_DIR / "bronchoscopy_taxonomy_v2.json"
DATASET_VERSION = "canonical_v2.1"
SCHEMA_VERSION = "bronchoscopy_canonical_schema_v2.0"
CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(prefix: str, *parts: str, length: int) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def git_commit() -> str:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={ROOT}", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable:no_commits"


def verify_raw_snapshot(raw_dir: Path) -> dict[str, Any]:
    manifest = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))
    for bundle in manifest["bundles"]:
        path = ROOT / bundle["path"]
        actual = sha256_file(path)
        if actual != bundle["sha256"]:
            raise ValueError(f"Raw bundle checksum mismatch: {path}")
    files_manifest = ROOT / manifest["files_manifest"]
    if sha256_file(files_manifest) != manifest["files_manifest_sha256"]:
        raise ValueError("raw_v1/files.sha256 checksum mismatch")
    return manifest


def source_sha_index(raw_manifest: dict[str, Any]) -> dict[str, str]:
    path = ROOT / raw_manifest["files_manifest"]
    index: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        index[relative] = digest
    return index


def bundle_path(raw_manifest: dict[str, Any], group: str) -> Path:
    for bundle in raw_manifest["bundles"]:
        if bundle["group"] == group:
            return ROOT / bundle["path"]
    raise KeyError(f"Raw bundle group not found: {group}")


def iter_bundle_json(path: Path) -> Iterator[tuple[str, Any]]:
    with tarfile.open(path, mode="r:gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            yield member.name, json.loads(extracted.read().decode("utf-8-sig"))


def walk_label_rows(payload: Any) -> Iterator[dict[str, Any]]:
    if isinstance(payload, dict):
        if payload.get("id") and payload.get("name"):
            yield payload
        for value in payload.values():
            yield from walk_label_rows(value)
    elif isinstance(payload, list):
        for value in payload:
            yield from walk_label_rows(value)


def load_label_catalogs(raw_manifest: dict[str, Any]) -> dict[str, dict[str, str]]:
    catalogs: dict[str, dict[str, str]] = {
        "labels_v1": {},
        "labels_v2": {},
        "labels_final": {},
        "supplemental": {},
    }
    metadata_bundle = bundle_path(raw_manifest, "metadata_and_labels")
    for path, payload in iter_bundle_json(metadata_bundle):
        target = None
        if path == "Metadata/labels_ver1.json":
            target = "labels_v1"
        elif path == "Metadata/labels_ver2.json":
            target = "labels_v2"
        elif path == "labels_final.json":
            target = "labels_final"
        if target:
            for row in walk_label_rows(payload):
                catalogs[target][str(row["id"])] = str(row["name"])

    supplemental_bundle = bundle_path(raw_manifest, "supplemental_annotations")
    for path, payload in iter_bundle_json(supplemental_bundle):
        if path == "nhom_benh_bosung_6/labels.json":
            for row in walk_label_rows(payload):
                catalogs["supplemental"][str(row["id"])] = str(row["name"])
    return catalogs


def load_supplemental_patient_map(raw_manifest: dict[str, Any]) -> dict[str, str]:
    supplemental_bundle = bundle_path(raw_manifest, "supplemental_annotations")
    for path, payload in iter_bundle_json(supplemental_bundle):
        if path == "nhom_benh_bosung_6/metadata.json":
            return {
                str(study["object_id"]): str(study.get("code") or study["object_id"])
                for study in payload.get("studies", [])
                if isinstance(study, dict) and study.get("object_id")
            }
    return {}


def annotation_members(raw_manifest: dict[str, Any]) -> Iterator[tuple[str, str, Any]]:
    patterns = {
        "case": re.compile(r"^Cabenh/[^/]+/[^/]+/(?:annotation|KC_BRONCHOSCO_\d+)\.json$"),
        "control": re.compile(r"^Cachung/[^/]+/(?:annotation|KC_BRONCHOSCO_\d+)\.json$"),
        "supplemental": re.compile(r"^nhom_benh_bosung_6/annotation/KC_BRONCHOSCO_\d+\.json$"),
    }
    groups = {
        "case": "case_annotations",
        "control": "control_annotations",
        "supplemental": "supplemental_annotations",
    }
    for source_group in ("case", "control", "supplemental"):
        for path, payload in iter_bundle_json(bundle_path(raw_manifest, groups[source_group])):
            if patterns[source_group].match(path):
                yield source_group, path, payload


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold())


def label_tail(name: str) -> str:
    normalized = normalize_text(name)
    prefix = re.match(r"^(mgpkv|ttkv_[a-z]+|vtos)\s*-\s*(.*)$", normalized)
    return prefix.group(2) if prefix else normalized


def resolve_label(
    label_id: str,
    primary_catalog: str,
    catalogs: dict[str, dict[str, str]],
) -> tuple[str | None, list[str]]:
    if label_id in catalogs[primary_catalog]:
        return catalogs[primary_catalog][label_id], []
    # labels_final is an English renaming of the v1 polygon concepts and
    # intentionally reuses their UUIDs. Prefer it when a v2 document still
    # contains a v1 UUID; different display names do not imply different
    # clinical concepts. Primary-catalog matches above always win, which is
    # essential for UUIDs that were genuinely reused in labels_v2.
    fallback_order = ("labels_final", "labels_v1", "labels_v2", "supplemental")
    for fallback in fallback_order:
        if label_id in catalogs[fallback]:
            return catalogs[fallback][label_id], [f"label_resolved_from_{fallback}"]
    return None, ["unknown_source_label_id"]


def primary_catalog(source_group: str, source_path: str) -> str:
    if source_group == "supplemental":
        return "supplemental"
    if source_path.endswith("/annotation.json"):
        return "labels_v1"
    return "labels_v2"


def source_identity(
    source_group: str,
    source_path: str,
    supplemental_patient_map: dict[str, str],
) -> tuple[str, str]:
    parts = Path(source_path).parts
    filename = parts[-1]
    if source_group == "case":
        source_patient_key = parts[-2]
    elif source_group == "control":
        source_patient_key = parts[-2]
    else:
        raw_procedure = Path(filename).stem
        source_patient_key = supplemental_patient_map.get(raw_procedure, raw_procedure)
        return source_patient_key, raw_procedure
    raw_procedure = source_patient_key if filename == "annotation.json" else Path(filename).stem
    return source_patient_key, raw_procedure


def points_from_annotation(annotation: dict[str, Any]) -> list[list[float]]:
    points: list[list[float]] = []
    for point in annotation.get("data", []) or []:
        if not isinstance(point, dict):
            continue
        try:
            points.append([float(point["x"]), float(point["y"])])
        except (KeyError, TypeError, ValueError):
            continue
    return points


def normalized_ring(points: list[list[float]]) -> tuple[tuple[float, float], ...]:
    cleaned: list[tuple[float, float]] = []
    for x, y in points:
        point = (round(float(x), 6), round(float(y), 6))
        if not cleaned or point != cleaned[-1]:
            cleaned.append(point)
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    if not cleaned:
        return ()

    def minimum_rotation(sequence: list[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
        return min(tuple(sequence[index:] + sequence[:index]) for index in range(len(sequence)))

    forward = minimum_rotation(cleaned)
    reverse = minimum_rotation(list(reversed(cleaned)))
    return min(forward, reverse)


def geometry_hash(points: list[list[float]]) -> str:
    ring = normalized_ring(points)
    payload = json.dumps(ring, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def polygon_bbox(points: list[list[float]]) -> list[float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def polygon_area(points: list[list[float]]) -> float:
    ring = normalized_ring(points)
    if len(ring) < 3:
        return 0.0
    total = 0.0
    for index, (x1, y1) in enumerate(ring):
        x2, y2 = ring[(index + 1) % len(ring)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def empty_region_labels(taxonomy: dict[str, Any]) -> dict[str, Any]:
    return {
        "anatomy": [],
        "pathology_group": [],
        "mucosal_findings": {name: None for name in taxonomy["mucosal_findings"]},
        "vascular_findings": {"hypervascularity": None},
        "airway_wall_findings": {"tracheomalacia": None},
        "stenosis": {"presence": None, "severity": None, "cause": None},
        "tumor": {"presence": None, "morphology": None},
        "secretion": {"presence": None, "type": None, "color": None, "consistency": None},
        "vocal_cords": {"normal": None, "paralysis": None},
        "other_findings": [],
    }


def add_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def set_value(container: dict[str, Any], key: str, value: Any, flags: set[str], path: str) -> None:
    conflict_flag = f"conflicting_{path}"
    if conflict_flag in flags:
        return
    current = container.get(key)
    if current is None or current == value:
        container[key] = value
    else:
        container[key] = None
        flags.add(conflict_flag)


def add_pathology_group(labels: dict[str, Any], group: str) -> None:
    add_unique(labels["pathology_group"], group)


def apply_region_label(
    labels: dict[str, Any],
    source_name: str,
    taxonomy: dict[str, Any],
    flags: set[str],
) -> bool:
    full = normalize_text(source_name)
    tail = label_tail(source_name)
    anatomy = taxonomy["source_aliases"]["anatomy"].get(tail)
    if anatomy and (full.startswith("mgpkv") or full in taxonomy["source_aliases"]["anatomy"]):
        add_unique(labels["anatomy"], anatomy)
        return True

    mucosal = taxonomy["source_aliases"]["mucosal_findings"].get(tail)
    if mucosal and not full.startswith("ttkv_dt"):
        set_value(labels["mucosal_findings"], mucosal, True, flags, f"mucosal_{mucosal}")
        add_pathology_group(labels, "mucosal_lesion")
        return True

    if tail in {"tăng sinh mạch", "vascular growth"}:
        set_value(labels["vascular_findings"], "hypervascularity", True, flags, "hypervascularity")
        add_pathology_group(labels, "vascular_abnormality")
        return True
    if tail == "không tăng sinh mạch":
        set_value(labels["vascular_findings"], "hypervascularity", False, flags, "hypervascularity")
        return True

    if tail in {"hẹp lòng phế quản", "stenosis"}:
        set_value(labels["stenosis"], "presence", True, flags, "stenosis_presence")
        add_pathology_group(labels, "stenosis")
        return True
    severity = {
        "hẹp dưới 25%": "0_25_percent",
        "hẹp từ 26 đến 50%": "26_50_percent",
        "hẹp từ 51 đến 75%": "51_75_percent",
        "hẹp từ 76 đến 90%": "76_90_percent",
        "hẹp trên 90%": "over_90_percent",
    }.get(tail)
    if severity:
        set_value(labels["stenosis"], "presence", True, flags, "stenosis_presence")
        set_value(labels["stenosis"], "severity", severity, flags, "stenosis_severity")
        add_pathology_group(labels, "stenosis")
        return True
    cause = {
        "hẹp do xơ sẹo": "scarring",
        "sẹo hẹp": "scarring",
        "hẹp do xoắn vặn": "torsion",
        "đè ép từ bên ngoài": "external_compression",
        "tổn thương trong lòng khí phế quản": "endoluminal_lesion",
        "hỗn hợp": "mixed",
    }.get(tail)
    if cause:
        set_value(labels["stenosis"], "presence", True, flags, "stenosis_presence")
        set_value(labels["stenosis"], "cause", cause, flags, "stenosis_cause")
        add_pathology_group(labels, "stenosis")
        return True

    if tail in {"khối u khí phế quản", "tumor"}:
        set_value(labels["tumor"], "presence", True, flags, "tumor_presence")
        add_pathology_group(labels, "tumor")
        return True
    morphology = {
        "khối u có cuống": "pedunculated",
        "khối u không có cuống": "non_pedunculated",
    }.get(tail)
    if morphology:
        set_value(labels["tumor"], "presence", True, flags, "tumor_presence")
        set_value(labels["tumor"], "morphology", morphology, flags, "tumor_morphology")
        add_pathology_group(labels, "tumor")
        return True

    if full.startswith("ttkv_ct"):
        if tail == "bình thường":
            set_value(labels["secretion"], "presence", False, flags, "secretion_presence")
            return True
        secretion = {
            "dịch máu": ("blood", None, None),
            "dịch máu đỏ tươi": ("blood", "bright_red", None),
            "dịch máu đỏ sẫm": ("blood", "dark_red", None),
            "máu đông": ("blood", None, "clotted"),
            "dịch mủ": ("purulent", None, None),
        }.get(tail)
        if secretion:
            secretion_type, color, consistency = secretion
            set_value(labels["secretion"], "presence", True, flags, "secretion_presence")
            set_value(labels["secretion"], "type", secretion_type, flags, "secretion_type")
            if color:
                set_value(labels["secretion"], "color", color, flags, "secretion_color")
            if consistency:
                set_value(labels["secretion"], "consistency", consistency, flags, "secretion_consistency")
            add_pathology_group(labels, "secretion")
            return True

    if tail == "dây thanh bình thường":
        set_value(labels["vocal_cords"], "normal", True, flags, "vocal_cords_normal")
        set_value(labels["vocal_cords"], "paralysis", False, flags, "vocal_cords_paralysis")
        add_unique(labels["anatomy"], "vocal_cords")
        return True
    if tail in {"liệt đây thanh", "liệt dây thanh"}:
        set_value(labels["vocal_cords"], "normal", False, flags, "vocal_cords_normal")
        set_value(labels["vocal_cords"], "paralysis", True, flags, "vocal_cords_paralysis")
        add_unique(labels["anatomy"], "vocal_cords")
        add_pathology_group(labels, "vocal_cord_abnormality")
        return True
    if full.startswith("ttkv_dt") and tail == "tổn thương khác":
        add_unique(labels["other_findings"], "vocal_cord_other")
        add_pathology_group(labels, "other")
        return True
    if tail == "nhuyễn sụn khí quản":
        set_value(labels["airway_wall_findings"], "tracheomalacia", True, flags, "tracheomalacia")
        add_pathology_group(labels, "airway_wall_abnormality")
        return True
    return False


def empty_image_labels() -> dict[str, Any]:
    return {
        "normal": None,
        "image_quality": None,
        "anatomical_location": [],
        "evidence_region_ids": [],
        "source_label_ids": [],
    }


def empty_procedure_labels() -> dict[str, Any]:
    return {
        "lung_cancer": None,
        "lesion_site_count": None,
        "lesion_extent": None,
        "source_label_ids": [],
    }


def apply_impression_label(
    image_labels: dict[str, Any] | None,
    procedure_labels: dict[str, Any],
    source_name: str,
    source_label_id: str,
    taxonomy: dict[str, Any],
    flags: set[str],
) -> bool:
    normalized = normalize_text(source_name)
    if normalized.startswith("vtos"):
        anatomy = taxonomy["source_aliases"]["anatomy"].get(label_tail(source_name))
        if anatomy and image_labels is not None:
            add_unique(image_labels["anatomical_location"], anatomy)
            add_unique(image_labels["source_label_ids"], source_label_id)
            return True
    state = taxonomy["source_aliases"]["image_state"].get(normalized)
    if state and image_labels is not None:
        set_value(image_labels, state["field"], state["value"], flags, f"image_{state['field']}")
        add_unique(image_labels["source_label_ids"], source_label_id)
        return True
    procedure = taxonomy["source_aliases"]["procedure_labels"].get(normalized)
    if procedure:
        set_value(
            procedure_labels,
            procedure["field"],
            procedure["value"],
            flags,
            f"procedure_{procedure['field']}",
        )
        add_unique(procedure_labels["source_label_ids"], source_label_id)
        return True
    return normalized == "nhãn bỏ"


def resolve_image_path(source_group: str, source_path: str, raw_procedure: str, source_image_id: str) -> Path | None:
    if source_group == "supplemental":
        candidates = sorted((ROOT / "nhom_benh_bosung_6" / "data" / raw_procedure).glob(f"{source_image_id}.*"))
    else:
        folder = ROOT / Path(source_path).parent / "imgs"
        candidates = sorted(folder.glob(f"{source_image_id}.*")) + sorted(folder.glob(f"{source_image_id}_*"))
    candidates = [path for path in candidates if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    return candidates[0] if candidates else None


def discover_source_image_ids(
    source_group: str,
    source_path: str,
    raw_procedure: str,
    payload: dict[str, Any],
) -> list[str]:
    ids = {str(slide["object_id"]) for slide in payload.get("slides", []) if isinstance(slide, dict) and slide.get("object_id")}
    for section in ("finding", "findings"):
        ids.update(
            str(row["object_id"])
            for row in payload.get(section, []) or []
            if isinstance(row, dict) and row.get("object_id")
        )
    if source_path.endswith("/annotation.json"):
        folder = ROOT / Path(source_path).parent / "imgs"
        for image_path in folder.glob("*"):
            if image_path.is_file() and image_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                ids.add(image_path.stem)
    return sorted(ids)


def image_metadata(path: Path | None) -> tuple[int | None, int | None, str | None, str | None]:
    if path is None or not path.is_file():
        return None, None, None, None
    with Image.open(path) as image:
        width, height = image.size
    return width, height, path.relative_to(ROOT).as_posix(), "sha256:" + sha256_file(path)


def annotation_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("finding_id") or "")


def impression_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("impression_id") or "")


def source_rows(payload: dict[str, Any], singular: str, plural: str) -> list[dict[str, Any]]:
    rows = payload.get(plural) if plural in payload else payload.get(singular, [])
    return [row for row in (rows or []) if isinstance(row, dict)]


def make_regions(
    findings: list[dict[str, Any]],
    source_image_id: str,
    source_group: str,
    source_path: str,
    width: int | None,
    height: int | None,
    catalogs: dict[str, dict[str, str]],
    taxonomy: dict[str, Any],
    registry: dict[str, Any],
    counters: Counter[str],
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    catalog_name = primary_catalog(source_group, source_path)
    for finding in findings:
        if str(finding.get("object_id")) != source_image_id:
            continue
        points = points_from_annotation(finding)
        if len(points) < 2:
            counters["findings_without_valid_geometry"] += 1
            continue
        key = geometry_hash(points)
        group = grouped.setdefault(
            key,
            {
                "points": points,
                "source_label_ids": set(),
                "source_annotation_ids": set(),
                "label_evidence": [],
            },
        )
        source_annotation_id = annotation_id(finding)
        if source_annotation_id:
            group["source_annotation_ids"].add(source_annotation_id)
        for label_id in finding.get("label_ids", []) or []:
            label_id = str(label_id)
            group["source_label_ids"].add(label_id)
            name, resolution_flags = resolve_label(label_id, catalog_name, catalogs)
            group["label_evidence"].append((label_id, name, resolution_flags))
            registry_key = f"{catalog_name}:{label_id}"
            registry.setdefault(
                registry_key,
                {
                    "catalog": catalog_name,
                    "source_label_id": label_id,
                    "source_name": name,
                    "occurrences": 0,
                    "resolution_flags": resolution_flags,
                },
            )["occurrences"] += 1

    regions: list[dict[str, Any]] = []
    for index, key in enumerate(sorted(grouped), start=1):
        group = grouped[key]
        points = group["points"]
        labels = empty_region_labels(taxonomy)
        flags: set[str] = set()
        unmapped: list[dict[str, Any]] = []
        for label_id, source_name, resolution_flags in group["label_evidence"]:
            flags.update(resolution_flags)
            if source_name is None or not apply_region_label(labels, source_name, taxonomy, flags):
                candidate = {"id": label_id, "name": source_name}
                if candidate not in unmapped:
                    unmapped.append(candidate)
                counters["unmapped_region_label_occurrences"] += 1
        bbox = polygon_bbox(points)
        if len(normalized_ring(points)) < 3:
            flags.add("polygon_has_fewer_than_three_unique_points")
        if width is not None and height is not None:
            if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > width or bbox[3] > height:
                flags.add("polygon_out_of_image_bounds")
                counters["regions_out_of_bounds"] += 1
        region = {
            "region_id": f"region_{index:04d}",
            "geometry_hash": key,
            "polygon": points,
            "bbox_xyxy": bbox,
            "area": polygon_area(points),
            "labels": labels,
            "source_label_ids": sorted(group["source_label_ids"]),
            "source_annotation_ids": sorted(group["source_annotation_ids"]),
            "unmapped_source_labels": sorted(unmapped, key=lambda row: (row["id"], row["name"] or "")),
            "quality_flags": sorted(flags),
        }
        regions.append(region)
    counters["source_polygon_annotations"] += sum(
        1 for row in findings if str(row.get("object_id")) == source_image_id and len(points_from_annotation(row)) >= 2
    )
    counters["canonical_regions"] += len(regions)
    counters["merged_polygon_annotations"] += max(
        0,
        sum(1 for row in findings if str(row.get("object_id")) == source_image_id and len(points_from_annotation(row)) >= 2)
        - len(regions),
    )
    return regions


def build(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite {args.output_dir}. Use a new canonical version directory."
        )
    raw_manifest = verify_raw_snapshot(args.raw_dir)
    source_hashes = source_sha_index(raw_manifest)
    taxonomy = json.loads(args.taxonomy.read_text(encoding="utf-8"))
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    catalogs = load_label_catalogs(raw_manifest)
    supplemental_patient_map = load_supplemental_patient_map(raw_manifest)

    args.output_dir.mkdir(parents=True)
    records_path = args.output_dir / "images.jsonl.gz"
    registry: dict[str, Any] = {}
    counters: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    image_asset_digest = hashlib.sha256()
    seen_image_ids: set[str] = set()
    seen_procedure_keys: dict[str, set[str]] = defaultdict(set)
    validation_errors: list[dict[str, Any]] = []

    with records_path.open("wb") as raw_output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_output, mtime=0) as compressed:
            for source_group, source_path, payload in annotation_members(raw_manifest):
                if not isinstance(payload, dict):
                    counters["invalid_annotation_documents"] += 1
                    continue
                source_patient_key, raw_procedure = source_identity(
                    source_group, source_path, supplemental_patient_map
                )
                patient_id = stable_id(
                    "patient", "bronchoscopy_patient_v2", source_group, source_patient_key, length=16
                )
                procedure_id = stable_id(
                    "procedure", "bronchoscopy_procedure_v2", source_group, source_patient_key, raw_procedure, length=20
                )
                seen_procedure_keys[raw_procedure].add(procedure_id)
                image_ids = discover_source_image_ids(source_group, source_path, raw_procedure, payload)
                findings = source_rows(payload, "finding", "findings")
                impressions = source_rows(payload, "impression", "impressions")
                procedure_labels = empty_procedure_labels()
                procedure_image_labels = empty_image_labels()
                procedure_flags: set[str] = set()
                image_impressions: dict[str, list[dict[str, Any]]] = defaultdict(list)
                procedure_impressions: list[dict[str, Any]] = []
                image_id_set = set(image_ids)
                for impression in impressions:
                    target = str(impression.get("object_id") or "")
                    if target in image_id_set:
                        image_impressions[target].append(impression)
                    else:
                        procedure_impressions.append(impression)

                catalog_name = primary_catalog(source_group, source_path)
                procedure_unmapped: list[dict[str, Any]] = []
                for impression in procedure_impressions:
                    for label_id in impression.get("label_ids", []) or []:
                        label_id = str(label_id)
                        name, resolution_flags = resolve_label(label_id, catalog_name, catalogs)
                        procedure_flags.update(resolution_flags)
                        registry_key = f"{catalog_name}:{label_id}"
                        registry.setdefault(
                            registry_key,
                            {
                                "catalog": catalog_name,
                                "source_label_id": label_id,
                                "source_name": name,
                                "occurrences": 0,
                                "resolution_flags": resolution_flags,
                            },
                        )["occurrences"] += 1
                        if name is None or not apply_impression_label(
                            procedure_image_labels,
                            procedure_labels,
                            name,
                            label_id,
                            taxonomy,
                            procedure_flags,
                        ):
                            candidate = {"id": label_id, "name": name}
                            if candidate not in procedure_unmapped:
                                procedure_unmapped.append(candidate)

                # Diagnosis/extent tags are procedure-level even when a source
                # export attaches them to a particular slide. Resolve them
                # before emitting any image so every image gets one consistent
                # procedure label state.
                for scoped_impressions in image_impressions.values():
                    for impression in scoped_impressions:
                        for label_id in impression.get("label_ids", []) or []:
                            label_id = str(label_id)
                            name, resolution_flags = resolve_label(label_id, catalog_name, catalogs)
                            procedure_flags.update(resolution_flags)
                            if name is not None:
                                normalized = normalize_text(name)
                                if normalized in taxonomy["source_aliases"]["procedure_labels"]:
                                    apply_impression_label(
                                        None,
                                        procedure_labels,
                                        name,
                                        label_id,
                                        taxonomy,
                                        procedure_flags,
                                    )

                for source_image_id in image_ids:
                    canonical_image_id = stable_id(
                        "image", "bronchoscopy_image_v2", procedure_id, source_image_id, length=20
                    )
                    if canonical_image_id in seen_image_ids:
                        raise ValueError(f"Canonical image ID collision: {canonical_image_id}")
                    seen_image_ids.add(canonical_image_id)
                    image_path = resolve_image_path(source_group, source_path, raw_procedure, source_image_id)
                    width, height, relative_image_path, image_sha = image_metadata(image_path)
                    quality_flags = set(procedure_flags)
                    if image_path is None:
                        quality_flags.add("missing_image_asset")
                        counters["missing_image_assets"] += 1
                    else:
                        image_asset_digest.update(canonical_image_id.encode("ascii"))
                        image_asset_digest.update(b"\0")
                        image_asset_digest.update(image_sha.encode("ascii"))
                        image_asset_digest.update(b"\n")

                    image_labels = copy.deepcopy(procedure_image_labels)
                    image_unmapped = list(procedure_unmapped)
                    source_impression_ids = [
                        impression_id(row) for row in procedure_impressions if impression_id(row)
                    ]
                    for impression in image_impressions.get(source_image_id, []):
                        iid = impression_id(impression)
                        if iid:
                            source_impression_ids.append(iid)
                        for label_id in impression.get("label_ids", []) or []:
                            label_id = str(label_id)
                            name, resolution_flags = resolve_label(label_id, catalog_name, catalogs)
                            quality_flags.update(resolution_flags)
                            registry_key = f"{catalog_name}:{label_id}"
                            registry.setdefault(
                                registry_key,
                                {
                                    "catalog": catalog_name,
                                    "source_label_id": label_id,
                                    "source_name": name,
                                    "occurrences": 0,
                                    "resolution_flags": resolution_flags,
                                },
                            )["occurrences"] += 1
                            if name is None or not apply_impression_label(
                                image_labels,
                                procedure_labels,
                                name,
                                label_id,
                                taxonomy,
                                quality_flags,
                            ):
                                candidate = {"id": label_id, "name": name}
                                if candidate not in image_unmapped:
                                    image_unmapped.append(candidate)

                    regions = make_regions(
                        findings,
                        source_image_id,
                        source_group,
                        source_path,
                        width,
                        height,
                        catalogs,
                        taxonomy,
                        registry,
                        counters,
                    )
                    for region in regions:
                        for anatomy in region["labels"]["anatomy"]:
                            add_unique(image_labels["anatomical_location"], anatomy)
                            add_unique(image_labels["evidence_region_ids"], region["region_id"])
                        if region["labels"]["pathology_group"]:
                            add_unique(image_labels["evidence_region_ids"], region["region_id"])
                            set_value(
                                image_labels,
                                "normal",
                                False,
                                quality_flags,
                                "image_normal",
                            )

                    for key in ("anatomical_location", "evidence_region_ids", "source_label_ids"):
                        image_labels[key] = sorted(image_labels[key])
                    procedure_labels["source_label_ids"] = sorted(procedure_labels["source_label_ids"])
                    record = {
                        "schema_version": SCHEMA_VERSION,
                        "dataset_version": DATASET_VERSION,
                        "image_id": canonical_image_id,
                        "patient_id": patient_id,
                        "procedure_id": procedure_id,
                        "width": width,
                        "height": height,
                        "image_path": relative_image_path,
                        "image_sha256": image_sha,
                        "image_labels": image_labels,
                        "procedure_labels": copy.deepcopy(procedure_labels),
                        "regions": regions,
                        "unmapped_source_labels": sorted(
                            image_unmapped, key=lambda row: (row["id"], row["name"] or "")
                        ),
                        "provenance": {
                            "source_group": source_group,
                            "source_annotation_path": source_path,
                            "source_annotation_sha256": "sha256:" + source_hashes[source_path],
                            "source_procedure_id": raw_procedure,
                            "source_image_id": source_image_id,
                            "source_impression_ids": sorted(set(source_impression_ids)),
                        },
                        "quality_flags": sorted(quality_flags),
                    }
                    errors = list(validator.iter_errors(record))
                    if errors:
                        validation_errors.append(
                            {
                                "image_id": canonical_image_id,
                                "errors": [error.message for error in errors[:10]],
                            }
                        )
                    compressed.write(canonical_bytes(record))
                    counters["canonical_images"] += 1
                    counters["images_with_regions"] += bool(regions)
                    counters["images_normal_true"] += image_labels["normal"] is True
                    counters["images_normal_false"] += image_labels["normal"] is False
                    counters["images_normal_null"] += image_labels["normal"] is None
                    source_counts[source_group] += 1
                counters["source_annotation_documents"] += 1

    if validation_errors:
        write_json(args.output_dir / "validation_errors.json", validation_errors)
        raise ValueError(f"Schema validation failed for {len(validation_errors)} records")

    registry_rows = sorted(registry.values(), key=lambda row: (row["catalog"], row["source_label_id"]))
    write_json(args.output_dir / "source_label_registry.json", registry_rows)
    raw_procedure_collisions = {
        key: sorted(values) for key, values in seen_procedure_keys.items() if len(values) > 1
    }
    quality_report = {
        "dataset_version": DATASET_VERSION,
        "counts": dict(sorted(counters.items())),
        "images_by_source_group": dict(sorted(source_counts.items())),
        "raw_procedure_id_collision_count": len(raw_procedure_collisions),
        "raw_procedure_id_collisions": raw_procedure_collisions,
        "schema_validation_errors": 0,
        "notes": [
            "A raw procedure ID may identify different studies across source projects; canonical procedure/image IDs are namespaced hashes.",
            "Out-of-bounds polygons are retained and flagged, never clipped in canonical data.",
            "Positive-only arrays must not be interpreted as exhaustive negative labels."
        ],
    }
    write_json(args.output_dir / "quality_report.json", quality_report)

    records_sha = sha256_file(records_path)
    registry_sha = sha256_file(args.output_dir / "source_label_registry.json")
    report_sha = sha256_file(args.output_dir / "quality_report.json")
    manifest = {
        "dataset_version": DATASET_VERSION,
        "schema_version": SCHEMA_VERSION,
        "source_dataset_version": raw_manifest["dataset_version"],
        "source_hash": raw_manifest["source_hash"],
        "image_asset_hash": "sha256:" + image_asset_digest.hexdigest(),
        "conversion_commit": git_commit(),
        "conversion_script_sha256": "sha256:" + sha256_file(Path(__file__)),
        "taxonomy_version": taxonomy["taxonomy_version"],
        "taxonomy_sha256": "sha256:" + sha256_file(args.taxonomy),
        "json_schema_sha256": "sha256:" + sha256_file(args.schema),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "region_identity_rule": "Same image + winding/rotation-invariant polygon coordinates rounded to 6 decimals",
        "three_state_policy": taxonomy["three_state_semantics"],
        "patient_id_policy": "Stable pseudonym from source group and source patient/study key; no cross-project patient linkage inferred",
        "outputs": [
            {"path": records_path.relative_to(ROOT).as_posix(), "sha256": records_sha, "records": counters["canonical_images"]},
            {"path": (args.output_dir / "source_label_registry.json").relative_to(ROOT).as_posix(), "sha256": registry_sha},
            {"path": (args.output_dir / "quality_report.json").relative_to(ROOT).as_posix(), "sha256": report_sha}
        ]
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({
        "manifest": str(args.output_dir / "manifest.json"),
        "images": counters["canonical_images"],
        "regions": counters["canonical_regions"],
        "merged_polygon_annotations": counters["merged_polygon_annotations"],
        "raw_procedure_id_collisions": len(raw_procedure_collisions),
    }, indent=2))


def verify(args: argparse.Namespace) -> None:
    manifest = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
    failures = []
    for output in manifest["outputs"]:
        path = ROOT / output["path"]
        if not path.is_file() or sha256_file(path) != output["sha256"]:
            failures.append(output["path"])
    print(json.dumps({"valid": not failures, "failures": failures}, indent=2))
    if failures:
        raise SystemExit(1)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    create.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    create.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    create.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    create.set_defaults(func=build)
    check = subparsers.add_parser("verify")
    check.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    check.set_defaults(func=verify)
    return result


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
