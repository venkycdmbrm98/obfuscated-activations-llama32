"""Validation helpers for completed OAT run artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def sha256_artifact(path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file()) if path.is_dir() else [path]
    relative_root = path if path.is_dir() else path.parent
    for file_path in files:
        digest.update(str(file_path.relative_to(relative_root)).encode("utf-8"))
        with file_path.open("rb") as artifact_file:
            for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def load_oat_run_manifest(info_path, *, require_v2=False, verify_checksums=True):
    info_path = Path(info_path).resolve()
    with info_path.open() as info_file:
        info = json.load(info_file)

    version = info.get("artifact_format_version", 1)
    if require_v2 and version != 2:
        raise ValueError(f"Expected OAT artifact format 2, found {version}")
    if version < 2:
        return info
    if version != 2:
        raise ValueError(f"Unsupported OAT artifact format {version}")

    artifacts = info.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("Completed format-v2 run has no artifact manifest")

    output_dir = info_path.parent
    resolved_artifacts = {}
    for name, item in artifacts.items():
        relative_path = item.get("path") if isinstance(item, dict) else None
        expected_checksum = item.get("sha256") if isinstance(item, dict) else None
        if not relative_path or not expected_checksum:
            raise ValueError(f"Invalid manifest entry for artifact {name!r}")
        artifact_path = (output_dir / relative_path).resolve()
        if os.path.commonpath((output_dir, artifact_path)) != str(output_dir):
            raise ValueError(f"Artifact {name!r} escapes the run output directory")
        if not artifact_path.exists():
            raise FileNotFoundError(f"Missing OAT artifact {name!r}: {artifact_path}")
        if verify_checksums:
            actual_checksum = sha256_artifact(artifact_path)
            if actual_checksum != expected_checksum:
                raise ValueError(
                    f"Checksum mismatch for OAT artifact {name!r}: "
                    f"expected {expected_checksum}, got {actual_checksum}"
                )
        resolved_artifacts[name] = str(artifact_path)

    info["resolved_artifacts"] = resolved_artifacts
    return info
