#!/usr/bin/env python3
"""Resumable production orchestrator for all remaining v4.3 vision VQA."""

from __future__ import annotations

import argparse
import fcntl
import gzip
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

import build_vqa_v4_2 as v42
import build_vqa_v4_3 as v43
import run_vqa_v4_2_vision_pilot as vision


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_ROOT = ROOT / "derived_v4_3_vision_production"
BLUEPRINTS = ROOT / "derived_v4_3" / "qa_blueprints_v1_3.jsonl.gz"
RELEASE_MANIFEST = ROOT / "derived_v4_3" / "release_manifest.json"
SEED_ACCEPTED = ROOT / "derived_v4_3_vision_repair_001" / "accepted_v4_3.jsonl.gz"
SEED_REVIEW = ROOT / "derived_v4_3_vision_repair_001" / "review_v4_3.jsonl.gz"
SEED_REQUESTS = ROOT / "derived_v4_3_vision_repair_001" / "source_requests_200.jsonl.gz"
EXPORT_SCRIPT = ROOT / "dataset_versioning" / "export_bronchoscopy_vqa_master_clean.py"
TOTAL_IMAGES = 13238
SEED_IMAGES = 200
DEFAULT_CONFIG = {
    "production_version": "derived_v4.3-vision-production-v1",
    "base_url": "http://127.0.0.1:20128/v1",
    "model": "GenVQAVer2",
    "cohort_size": 100,
    "minimum_free_gib": 10.0,
    "minimum_acceptance_rate": 0.95,
    "maximum_review_rate": 0.05,
    "manual_gate_every_new_images": 1000,
    "maximum_consecutive_image_failures": 3,
    "request_retries": 2,
    "request_timeout_seconds": 300,
    "restart_backoff_seconds": 120,
    "hard_stop_review_reasons": [
        "fact_pixel_conflict",
        "evidence_region_mismatch",
        "unsafe_clinical_inference",
    ],
}

STOP_REQUESTED = False


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def append_event(root: Path, event: str, **payload: Any) -> None:
    row = {"at": now(), "event": event, **payload}
    with (root / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(v43.canonical_json(row) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def quarantine_records(root: Path) -> list[dict[str, Any]]:
    path = root / "quarantine.json"
    if not path.is_file():
        return []
    value = load_json(path)
    records = value.get("records")
    if not isinstance(records, list):
        raise ValueError(f"Invalid quarantine registry: {path}")
    return records


def cohort_quarantine_ids(root: Path, cohort_name: str) -> set[str]:
    return {
        row["image_id"] for row in quarantine_records(root)
        if row.get("cohort") == cohort_name and isinstance(row.get("image_id"), str)
    }


def append_quarantine_records(root: Path, additions: list[dict[str, Any]]) -> int:
    if not additions:
        return 0
    path = root / "quarantine.json"
    registry = load_json(path) if path.is_file() else {
        "created_at_utc": now(),
        "policy": "Exclude every QA belonging to quarantined images from clean exports while retaining provenance for audit.",
        "records": [],
    }
    records = registry.setdefault("records", [])
    existing = {row.get("image_id") for row in records}
    fresh = [row for row in additions if row.get("image_id") not in existing]
    records.extend(fresh)
    registry["updated_at_utc"] = now()
    atomic_json(path, registry)
    return len(fresh)


def auto_quarantine_review_images(root: Path, cohort_dir: Path) -> int:
    """Quarantine whole images for every model/system review row in a completed cohort."""
    review_path = cohort_dir / "review_pilot.jsonl.gz"
    if not review_path.is_file():
        return 0
    review_rows = list(vision.read_jsonl(review_path))
    if not review_rows:
        return 0
    reviews_by_image: dict[str, list[dict[str, Any]]] = {}
    for row in review_rows:
        reviews_by_image.setdefault(row["image_id"], []).append(row)
    requests = {row["image_id"]: row for row in vision.read_jsonl(cohort_dir / "pilot_requests.jsonl.gz")}
    additions = []
    for image_id, rows in sorted(reviews_by_image.items()):
        request = requests[image_id]
        additions.append({
            "cohort": cohort_dir.name,
            "image_id": image_id,
            "image_path": request.get("image_path"),
            "overlay_path": request.get("overlay_path"),
            "source_annotation_ids": sorted({
                annotation_id
                for fact in request.get("facts", [])
                for annotation_id in fact.get("source_annotation_ids", [])
            }),
            "qa_ids": [plan["qa_id"] for plan in request.get("qa_plans", [])],
            "trigger_review_qa_ids": [row["qa_id"] for row in rows],
            "evidence_boxes_xyxy": [region["bbox_xyxy_pixels"] for region in request.get("evidence_regions", [])],
            "issues": sorted({reason for row in rows for reason in row.get("review_reasons", [])}),
            "operator_decision": "auto_quarantined_for_deferred_review",
            "auto_quarantine": True,
        })
    added = append_quarantine_records(root, additions)
    if added:
        append_event(root, "review_images_auto_quarantined", cohort=cohort_dir.name, images=added)
    return added


def checkpoint_failed_image_for_quarantine(root: Path, cohort_dir: Path, config: dict[str, Any]) -> str | None:
    """Create an explicit non-training checkpoint after repeated generation/API failure."""
    completed = {path.stem for path in (cohort_dir / "results").glob("*.json")}
    pending = [row for row in vision.read_jsonl(cohort_dir / "pilot_requests.jsonl.gz") if row["image_id"] not in completed]
    if not pending:
        return None
    request = pending[0]
    items = [{
        "qa_plan_id": plan["qa_plan_id"],
        "qa_id": plan["qa_id"],
        "status": "review_required",
        "review_reasons": ["generation_failure_after_retries"],
        "transport_normalizations": [],
    } for plan in request["qa_plans"]]
    result = {
        "image_id": request["image_id"],
        "split": request["split"],
        "items": items,
        "llm_trace": {
            "requested_model": config["model"],
            "response_model": None,
            "response_id": None,
            "usage": {},
            "prompt_sha256": None,
            "surface_method": "system_quarantine_after_repeated_generation_failure",
        },
        "completed_at_utc": now(),
        "elapsed_seconds": 0.0,
        "system_quarantine_checkpoint": True,
    }
    target = cohort_dir / "results" / f"{request['image_id']}.json"
    temp = target.with_suffix(".json.tmp")
    vision.write_json(temp, result)
    temp.replace(target)
    append_event(root, "generation_failure_auto_quarantined", cohort=cohort_dir.name, image_id=request["image_id"], qa_plans=len(items))
    return request["image_id"]


def repeated_failure_is_infrastructure(cohort_dir: Path) -> tuple[bool, str | None]:
    error_path = cohort_dir / "attempt_errors.jsonl"
    if not error_path.is_file():
        return False, None
    completed = {path.stem for path in (cohort_dir / "results").glob("*.json")}
    pending = [row for row in vision.read_jsonl(cohort_dir / "pilot_requests.jsonl.gz") if row["image_id"] not in completed]
    if not pending:
        return False, None
    image_id = pending[0]["image_id"]
    errors = [row.get("error", "") for row in vision.read_jsonl(error_path) if row.get("image_id") == image_id]
    if not errors:
        return False, None
    recent = errors[-9:]
    infrastructure_markers = (
        "HTTP Error 429", "HTTP Error 500", "HTTP Error 502", "HTTP Error 503", "HTTP Error 504",
        "timed out", "timeout", "Connection refused", "Connection reset", "Remote end closed",
        "Temporary failure", "Name or service not known",
    )
    infrastructure = all(any(marker.lower() in error.lower() for marker in infrastructure_markers) for error in recent)
    return infrastructure, recent[-1]


def write_jsonl_gz_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temp, "wt", encoding="utf-8", compresslevel=9) as handle:
        for row in rows:
            handle.write(v43.canonical_json(row) + "\n")
    temp.replace(path)


def effective_cohort_inputs(root: Path, cohort_dir: Path) -> tuple[Path, Path, dict[str, int]]:
    """Materialize operator quarantine without mutating original model outputs."""
    accepted_source = cohort_dir / "accepted_pilot.jsonl.gz"
    review_source = cohort_dir / "review_pilot.jsonl.gz"
    quarantine_ids = cohort_quarantine_ids(root, cohort_dir.name)
    if not quarantine_ids:
        accepted_count = sum(1 for _ in vision.read_jsonl(accepted_source))
        review_count = sum(1 for _ in vision.read_jsonl(review_source)) if review_source.exists() else 0
        return accepted_source, review_source, {"accepted": accepted_count, "review": review_count, "quarantined_images": 0, "quarantined_qa": 0}

    accepted_rows = list(vision.read_jsonl(accepted_source))
    review_rows = list(vision.read_jsonl(review_source)) if review_source.exists() else []
    retained_accepted = [row for row in accepted_rows if row.get("image_id") not in quarantine_ids]
    moved_to_quarantine = []
    for row in accepted_rows:
        if row.get("image_id") in quarantine_ids:
            moved = dict(row)
            moved["status"] = "review_required"
            moved["model_validation_status_before_operator_quarantine"] = row.get("status")
            moved["review_reasons"] = ["operator_quarantined_suspect_source_annotation"]
            moved["operator_quarantine"] = True
            moved_to_quarantine.append(moved)
    retained_review = []
    for row in review_rows:
        item = dict(row)
        if item.get("image_id") in quarantine_ids:
            item["model_review_reasons"] = item.get("review_reasons") or []
            item["review_reasons"] = ["operator_quarantined_suspect_source_annotation"]
            item["operator_quarantine"] = True
        retained_review.append(item)

    accepted_effective = cohort_dir / "accepted_after_quarantine.jsonl.gz"
    review_effective = cohort_dir / "review_after_quarantine.jsonl.gz"
    write_jsonl_gz_atomic(accepted_effective, retained_accepted)
    write_jsonl_gz_atomic(review_effective, retained_review + moved_to_quarantine)
    quarantined_qa = sum(row.get("image_id") in quarantine_ids for row in accepted_rows + review_rows)
    return accepted_effective, review_effective, {
        "accepted": len(retained_accepted),
        "review": len(retained_review) + len(moved_to_quarantine),
        "quarantined_images": len(quarantine_ids),
        "quarantined_qa": quarantined_qa,
    }


def configure_validator() -> None:
    vision.BLUEPRINTS = BLUEPRINTS
    v42.protected_sha = v43.protected_sha


def initial_state() -> dict[str, Any]:
    return {
        "production_version": DEFAULT_CONFIG["production_version"],
        "status": "INITIALIZED",
        "created_at_utc": now(),
        "updated_at_utc": now(),
        "total_images": TOTAL_IMAGES,
        "seed_completed_images": SEED_IMAGES,
        "completed_new_images": 0,
        "completed_images": SEED_IMAGES,
        "completion_percent": round(100 * SEED_IMAGES / TOTAL_IMAGES, 4),
        "completed_cohorts": [],
        "current_cohort": None,
        "next_cohort_number": 3,
        "images_since_manual_gate": 0,
        "accepted_qa_seed": 920,
        "accepted_qa_production": 0,
        "review_qa_production": 0,
        "block_reason": None,
        "last_heartbeat_utc": None,
    }


def ensure_initialized(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root.mkdir(parents=True, exist_ok=True)
    (root / "cohorts").mkdir(exist_ok=True)
    (root / "final_release").mkdir(exist_ok=True)
    config_path = root / "config.json"
    state_path = root / "state.json"
    if not config_path.exists():
        atomic_json(config_path, DEFAULT_CONFIG)
    if not state_path.exists():
        atomic_json(state_path, initial_state())
        append_event(root, "initialized", seed_images=SEED_IMAGES, total_images=TOTAL_IMAGES)
    return load_json(config_path), load_json(state_path)


def save_state(root: Path, state: dict[str, Any]) -> None:
    state["updated_at_utc"] = now()
    state["completed_images"] = state["seed_completed_images"] + state["completed_new_images"]
    state["completion_percent"] = round(100 * state["completed_images"] / state["total_images"], 4)
    atomic_json(root / "state.json", state)


def disk_free_gib(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024 ** 3)


def tcp_health(base_url: str, timeout: float = 5.0) -> tuple[bool, str]:
    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, f"tcp://{host}:{port} reachable"
    except OSError as exc:
        return False, f"tcp://{host}:{port} unavailable: {exc}"


def blueprint_image_count() -> int:
    return len({row["image_id"] for row in v43.read_jsonl(BLUEPRINTS)})


def preflight_checks(root: Path, config: dict[str, Any], check_api: bool = True) -> dict[str, Any]:
    checks = {}
    release = load_json(RELEASE_MANIFEST)
    checks["blueprint_release"] = release.get("status") == "RELEASED_FOR_LLM_REALIZATION"
    checks["blueprint_semantic_failure_zero"] = release.get("semantic_failure_count") == 0
    checks["blueprint_checksum"] = v42.sha256_file(BLUEPRINTS) == release["blueprints"]["sha256"]
    checks["blueprint_unique_images"] = blueprint_image_count() == TOTAL_IMAGES
    checks["seed_artifacts"] = all(path.is_file() for path in (SEED_ACCEPTED, SEED_REVIEW, SEED_REQUESTS))
    free_gib = disk_free_gib(root)
    checks["disk_free_gib"] = round(free_gib, 3)
    checks["disk_guard_pass"] = free_gib >= float(config["minimum_free_gib"])
    api_message = "skipped"
    if check_api:
        api_ok, api_message = tcp_health(config["base_url"])
        checks["api_tcp_reachable"] = api_ok
    checks["api_health_detail"] = api_message
    boolean_checks = [value for key, value in checks.items() if isinstance(value, bool)]
    checks["status"] = "PASS" if all(boolean_checks) else "FAIL"
    checks["checked_at_utc"] = now()
    atomic_json(root / "preflight.json", checks)
    return checks


def exclusion_paths(root: Path, state: dict[str, Any]) -> list[Path]:
    paths = [SEED_REQUESTS]
    for cohort in state["completed_cohorts"]:
        path = root / "cohorts" / cohort / "pilot_requests.jsonl.gz"
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(path)
    return paths


def prepare_next_cohort(root: Path, state: dict[str, Any], config: dict[str, Any]) -> Path:
    cohort_name = f"cohort_{state['next_cohort_number']:04d}"
    cohort_dir = root / "cohorts" / cohort_name
    configure_validator()
    args = SimpleNamespace(
        output_dir=cohort_dir,
        exclude_requests=exclusion_paths(root, state),
        cohort_label=cohort_name,
        cohort_size=int(config["cohort_size"]),
    )
    vision.prepare(args)
    manifest = load_json(cohort_dir / "pilot_manifest.json")
    state["current_cohort"] = cohort_name
    state["status"] = "RUNNING"
    state["block_reason"] = None
    save_state(root, state)
    append_event(root, "cohort_prepared", cohort=cohort_name, images=manifest["selection"]["images"], plans=manifest["counts"]["qa_plans"])
    return cohort_dir


def result_progress(cohort_dir: Path) -> dict[str, Any]:
    requests = list(vision.read_jsonl(cohort_dir / "pilot_requests.jsonl.gz"))
    return vision.progress(cohort_dir, len(requests))


def run_one_image(cohort_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    configure_validator()
    args = SimpleNamespace(
        output_dir=cohort_dir,
        base_url=config["base_url"],
        model=config["model"],
        api_key=os.environ.get("VQA_LLM_API_KEY", "local-vqa-v4-3"),
        temperature=0.25,
        timeout=int(config["request_timeout_seconds"]),
        retries=int(config["request_retries"]),
        limit_images=1,
    )
    vision.run(args)
    return result_progress(cohort_dir)


def audit_cohort(cohort_dir: Path) -> dict[str, Any]:
    configure_validator()
    vision.audit(SimpleNamespace(output_dir=cohort_dir))
    return load_json(cohort_dir / "pilot_audit.json")


def export_cohort(root: Path, cohort_dir: Path, cohort_name: str, accepted: Path, review: Path, quarantined_images: int) -> None:
    output = cohort_dir / ("dataset_export_quarantine_filtered" if quarantined_images else "dataset_export")
    if output.exists():
        return
    command = [
        sys.executable, str(EXPORT_SCRIPT),
        "--accepted", str(accepted),
        "--requests", str(cohort_dir / "pilot_requests.jsonl.gz"),
        "--review", str(review),
        "--output", str(output),
        "--dataset-version", f"derived_v4.3-vision-{cohort_name}",
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def evaluate_audit(report: dict[str, Any], review_path: Path, effective_counts: dict[str, int], config: dict[str, Any]) -> tuple[bool, str | None]:
    failures = dict(report.get("failure_counts") or {})
    failures.pop("automatic_acceptance_below_95_percent", None)
    if failures:
        return False, "automatic_audit_failed"
    hard_reasons = set(config["hard_stop_review_reasons"])
    observed = set()
    if review_path.exists():
        for row in vision.read_jsonl(review_path):
            observed.update(row.get("review_reasons") or [])
    blocked = sorted(observed & hard_reasons)
    if blocked:
        return False, "hard_review_reason:" + ",".join(blocked)
    return True, None


def complete_cohort(root: Path, state: dict[str, Any], config: dict[str, Any], cohort_dir: Path) -> bool:
    cohort_name = cohort_dir.name
    report = audit_cohort(cohort_dir)
    auto_quarantine_review_images(root, cohort_dir)
    accepted_path, review_path, effective_counts = effective_cohort_inputs(root, cohort_dir)
    export_cohort(root, cohort_dir, cohort_name, accepted_path, review_path, effective_counts["quarantined_images"])
    passed, reason = evaluate_audit(report, review_path, effective_counts, config)
    if not passed:
        state["status"] = "BLOCKED_QUALITY_GATE"
        state["block_reason"] = reason
        save_state(root, state)
        append_event(root, "quality_gate_blocked", cohort=cohort_name, reason=reason, report=report)
        return False
    images = int(report["images"])
    state["completed_cohorts"].append(cohort_name)
    state["completed_new_images"] += images
    state["images_since_manual_gate"] += images
    state["accepted_qa_production"] += effective_counts["accepted"]
    state["review_qa_production"] += effective_counts["review"]
    state.setdefault("quarantined_images", [])
    state["quarantined_images"] = sorted(set(state["quarantined_images"]) | cohort_quarantine_ids(root, cohort_name))
    state["current_cohort"] = None
    state["next_cohort_number"] += 1
    state["status"] = "RUNNING"
    save_state(root, state)
    append_event(root, "cohort_completed", cohort=cohort_name, images=images, accepted=effective_counts["accepted"], review=effective_counts["review"], quarantined_images=effective_counts["quarantined_images"], quarantined_qa=effective_counts["quarantined_qa"])
    return True


def manual_gate(root: Path, state: dict[str, Any], config: dict[str, Any]) -> bool:
    threshold = int(config["manual_gate_every_new_images"])
    if threshold <= 0 or state["images_since_manual_gate"] < threshold:
        return False
    gate = {
        "created_at_utc": now(),
        "completed_images": state["completed_images"],
        "completion_percent": state["completion_percent"],
        "completed_cohorts": state["completed_cohorts"],
        "instruction": "Review cohort manual_review.csv files, then run the approve command.",
    }
    atomic_json(root / "MANUAL_REVIEW_REQUIRED.json", gate)
    state["status"] = "PAUSED_MANUAL_REVIEW"
    state["block_reason"] = "periodic_manual_review_gate"
    save_state(root, state)
    append_event(root, "manual_gate", **gate)
    return True


def finalize(root: Path, state: dict[str, Any]) -> Path:
    if state["completed_images"] != state["total_images"]:
        raise ValueError(f"Cannot finalize incomplete run: {state['completed_images']}/{state['total_images']}")
    output = root / "final_release" / "dataset_export"
    if output.exists():
        return output
    accepted = [SEED_ACCEPTED]
    requests = [SEED_REQUESTS]
    reviews = [SEED_REVIEW]
    for cohort in state["completed_cohorts"]:
        cohort_dir = root / "cohorts" / cohort
        accepted_path, review_path, _ = effective_cohort_inputs(root, cohort_dir)
        accepted.append(accepted_path)
        requests.append(cohort_dir / "pilot_requests.jsonl.gz")
        reviews.append(review_path)
    command = [sys.executable, str(EXPORT_SCRIPT), "--accepted", *map(str, accepted), "--requests", *map(str, requests), "--review", *map(str, reviews), "--output", str(output), "--dataset-version", "derived_v4.3-vision-full"]
    subprocess.run(command, cwd=ROOT, check=True)
    append_event(root, "finalized", output=str(output.relative_to(ROOT)))
    return output


def handle_signal(signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def command_init(args: argparse.Namespace) -> None:
    config, state = ensure_initialized(args.root.resolve())
    print(json.dumps({"config": config, "state": state}, ensure_ascii=False, indent=2))


def command_preflight(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    config, _ = ensure_initialized(root)
    report = preflight_checks(root, config, check_api=not args.skip_api)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


def command_status(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    config, state = ensure_initialized(root)
    current = None
    if state.get("current_cohort"):
        cohort_dir = root / "cohorts" / state["current_cohort"]
        if cohort_dir.exists():
            current = result_progress(cohort_dir)
    print(json.dumps({"state": state, "current_progress": current, "disk_free_gib": round(disk_free_gib(root), 3), "config": config}, ensure_ascii=False, indent=2))


def command_pause(args: argparse.Namespace) -> None:
    root = args.root.resolve(); ensure_initialized(root)
    atomic_json(root / "PAUSE_REQUESTED.json", {"at": now(), "reason": args.reason})
    append_event(root, "pause_requested", reason=args.reason)
    print("Pause requested; runner will stop before the next image.")


def command_approve(args: argparse.Namespace) -> None:
    root = args.root.resolve(); _, state = ensure_initialized(root)
    gate = root / "MANUAL_REVIEW_REQUIRED.json"
    if gate.exists():
        gate.unlink()
    pause = root / "PAUSE_REQUESTED.json"
    if pause.exists():
        pause.unlink()
    state["images_since_manual_gate"] = 0
    state["status"] = "READY_TO_RESUME"
    state["block_reason"] = None
    save_state(root, state)
    append_event(root, "manual_gate_approved", note=args.note)
    print("Manual gate approved. Start the user service to resume.")


def command_finalize(args: argparse.Namespace) -> None:
    root = args.root.resolve(); _, state = ensure_initialized(root)
    print(finalize(root, state))


def command_run(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    config, state = ensure_initialized(root)
    lock_handle = (root / ".runner.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another production runner is already active")
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    preflight = preflight_checks(root, config, check_api=True)
    if not preflight["disk_guard_pass"]:
        state["status"] = "BLOCKED_DISK_GUARD"
        state["block_reason"] = f"free={preflight['disk_free_gib']}GiB required={config['minimum_free_gib']}GiB"
        save_state(root, state); append_event(root, "disk_guard_blocked", reason=state["block_reason"])
        return
    if preflight["status"] != "PASS":
        state["status"] = "BLOCKED_PREFLIGHT"
        state["block_reason"] = "preflight_failed"
        save_state(root, state); append_event(root, "preflight_blocked", report=preflight)
        raise SystemExit(75)

    state["status"] = "RUNNING"; state["block_reason"] = None; save_state(root, state)
    consecutive_failures = 0
    while state["completed_images"] < state["total_images"]:
        if STOP_REQUESTED or (root / "PAUSE_REQUESTED.json").exists():
            state["status"] = "PAUSED_BY_REQUEST"; state["block_reason"] = "operator_pause_or_signal"; save_state(root, state)
            append_event(root, "paused", reason=state["block_reason"]); return
        if disk_free_gib(root) < float(config["minimum_free_gib"]):
            state["status"] = "BLOCKED_DISK_GUARD"; state["block_reason"] = "disk_fell_below_threshold"; save_state(root, state)
            append_event(root, "disk_guard_blocked", reason=state["block_reason"]); return
        if manual_gate(root, state, config):
            return
        if not state.get("current_cohort"):
            cohort_dir = prepare_next_cohort(root, state, config)
        else:
            cohort_dir = root / "cohorts" / state["current_cohort"]
        progress_before = result_progress(cohort_dir)
        if progress_before["pending_images"] == 0:
            if not complete_cohort(root, state, config, cohort_dir):
                return
            continue
        progress_after = run_one_image(cohort_dir, config)
        state["last_heartbeat_utc"] = now()
        save_state(root, state)
        if progress_after["completed_images"] <= progress_before["completed_images"]:
            consecutive_failures += 1
            append_event(root, "image_cycle_failed", cohort=cohort_dir.name, consecutive_failures=consecutive_failures)
            if consecutive_failures >= int(config["maximum_consecutive_image_failures"]):
                infrastructure_failure, latest_error = repeated_failure_is_infrastructure(cohort_dir)
                if infrastructure_failure:
                    backoff = int(config["restart_backoff_seconds"])
                    append_event(root, "infrastructure_backoff", cohort=cohort_dir.name, seconds=backoff, error=latest_error)
                    state["status"] = "WAITING_FOR_API"
                    state["block_reason"] = latest_error
                    save_state(root, state)
                    time.sleep(backoff)
                    state["status"] = "RUNNING"
                    state["block_reason"] = None
                    save_state(root, state)
                    consecutive_failures = 0
                    continue
                quarantined_image = checkpoint_failed_image_for_quarantine(root, cohort_dir, config)
                if quarantined_image is None:
                    state["status"] = "BLOCKED_REPEATED_IMAGE_FAILURE"; state["block_reason"] = f"{consecutive_failures} consecutive cycles"; save_state(root, state)
                    return
                consecutive_failures = 0
                state["last_heartbeat_utc"] = now()
                save_state(root, state)
                append_event(root, "image_checkpoint", cohort=cohort_dir.name, progress=result_progress(cohort_dir), quarantined_image=quarantined_image)
        else:
            consecutive_failures = 0
            append_event(root, "image_checkpoint", cohort=cohort_dir.name, progress=progress_after)

    output = finalize(root, state)
    state["status"] = "COMPLETE"; state["block_reason"] = None; save_state(root, state)
    append_event(root, "complete", output=str(output.relative_to(ROOT)))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    for name, func in (("init", command_init), ("status", command_status), ("run", command_run), ("finalize", command_finalize)):
        command = sub.add_parser(name); command.add_argument("--root", type=Path, default=PRODUCTION_ROOT); command.set_defaults(func=func)
    command = sub.add_parser("preflight"); command.add_argument("--root", type=Path, default=PRODUCTION_ROOT); command.add_argument("--skip-api", action="store_true"); command.set_defaults(func=command_preflight)
    command = sub.add_parser("pause"); command.add_argument("--root", type=Path, default=PRODUCTION_ROOT); command.add_argument("--reason", default="operator_request"); command.set_defaults(func=command_pause)
    command = sub.add_parser("approve"); command.add_argument("--root", type=Path, default=PRODUCTION_ROOT); command.add_argument("--note", default="manual_review_completed"); command.set_defaults(func=command_approve)
    return result


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
