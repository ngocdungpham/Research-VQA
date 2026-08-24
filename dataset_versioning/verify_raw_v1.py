#!/usr/bin/env python3
"""Verify the immutable raw_v1 snapshot without reading or modifying live sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHUNK = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def read_file_entries(path: Path) -> list[tuple[str, str]]:
    entries = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            checksum, relative = line.split(maxsplit=1)
        except ValueError as error:
            raise ValueError(f"Malformed files.sha256 line {line_number}") from error
        entries.append((relative.strip(), checksum.strip()))
    return entries


def aggregate_source_hash(entries: list[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for relative, checksum in sorted(entries):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(checksum.encode("ascii"))
        digest.update(b"\n")
    return "sha256:" + digest.hexdigest()


def verify(raw_dir: Path) -> dict[str, object]:
    manifest_path = raw_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []

    files_manifest = ROOT / manifest["files_manifest"]
    if not files_manifest.is_file():
        failures.append(f"missing:{manifest['files_manifest']}")
        entries = []
    else:
        if sha256_file(files_manifest) != manifest["files_manifest_sha256"]:
            failures.append("files_manifest_sha256_mismatch")
        entries = read_file_entries(files_manifest)
        if len(entries) != manifest["file_count"]:
            failures.append("file_count_mismatch")
        if aggregate_source_hash(entries) != manifest["source_hash"]:
            failures.append("aggregate_source_hash_mismatch")

    bundle_files = 0
    for bundle in manifest["bundles"]:
        path = ROOT / bundle["path"]
        if not path.is_file():
            failures.append(f"missing:{bundle['path']}")
            continue
        if sha256_file(path) != bundle["sha256"]:
            failures.append(f"bundle_sha256_mismatch:{bundle['path']}")
        if path.stat().st_size != bundle["size_bytes"]:
            failures.append(f"bundle_size_mismatch:{bundle['path']}")
        try:
            with tarfile.open(path, "r:gz") as archive:
                members = [member for member in archive.getmembers() if member.isfile()]
            if len(members) != bundle["files"]:
                failures.append(f"bundle_file_count_mismatch:{bundle['path']}")
            bundle_files += len(members)
        except (tarfile.TarError, OSError) as error:
            failures.append(f"bundle_unreadable:{bundle['path']}:{error}")

    if bundle_files and bundle_files != manifest["file_count"]:
        failures.append("bundle_total_file_count_mismatch")
    return {
        "valid": not failures,
        "failures": failures,
        "dataset_version": manifest.get("dataset_version"),
        "files": len(entries),
        "bundles": len(manifest.get("bundles", [])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "raw_v1")
    args = parser.parse_args()
    result = verify(args.raw_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
