#!/usr/bin/env python3
"""Validate the frozen PI0.5 checkpoint against a pre-outcome attestation."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit(
            "usage: validate_stage_r_checkpoint_attestation.py "
            "CHECKPOINT TREE_SHA ATTESTATION ATTESTATION_SHA"
        )
    root = Path(sys.argv[1])
    expected_tree = sys.argv[2]
    attestation_path = Path(sys.argv[3])
    expected_attestation = sys.argv[4]
    attestation_bytes = attestation_path.read_bytes()
    if hashlib.sha256(attestation_bytes).hexdigest() != expected_attestation:
        raise SystemExit("checkpoint attestation SHA mismatch")
    attestation = json.loads(attestation_bytes)
    expected_header = {
        "schema_version": 1,
        "marker_type": "full_content_checkpoint_attestation",
        "checkpoint": str(root),
        "tree_sha256": expected_tree,
        "file_count": 16,
        "bytes": 12439085481,
        "probe_scheme": "sha256_first_and_last_1MiB_plus_full_file_metadata",
        "uid": 2254,
        "gid": 2254,
    }
    if any(attestation.get(key) != value for key, value in expected_header.items()):
        raise SystemExit("checkpoint attestation header mismatch")
    rows = attestation.get("files")
    if not isinstance(rows, list) or len(rows) != 16:
        raise SystemExit("checkpoint attestation file inventory mismatch")
    expected_paths = [str(row.get("path", "")) for row in rows]
    actual_paths = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise SystemExit(f"checkpoint contains symlink: {path}")
        if path.is_file():
            actual_paths.append(path.relative_to(root).as_posix())
    if actual_paths != expected_paths:
        raise SystemExit("checkpoint file inventory drifted from full-content attestation")
    probe_bytes = 1024 * 1024
    for row in rows:
        path = root / row["path"]
        metadata = path.stat()
        observed_metadata = (metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns)
        expected_metadata = (row.get("size"), row.get("mtime_ns"), row.get("ctime_ns"))
        if observed_metadata != expected_metadata:
            raise SystemExit(f"checkpoint metadata drifted: {path}")
        with path.open("rb") as handle:
            head = handle.read(probe_bytes)
            if metadata.st_size > probe_bytes:
                handle.seek(max(0, metadata.st_size - probe_bytes))
                tail = handle.read(probe_bytes)
            else:
                tail = head
        if (
            hashlib.sha256(head).hexdigest() != row.get("head_sha256")
            or hashlib.sha256(tail).hexdigest() != row.get("tail_sha256")
        ):
            raise SystemExit(f"checkpoint probe digest drifted: {path}")
        if metadata.st_size <= 2 * probe_bytes:
            if hashlib.sha256(path.read_bytes()).hexdigest() != row.get("sha256"):
                raise SystemExit(f"checkpoint small-file full digest drifted: {path}")
    print(
        json.dumps(
            {
                "valid": True,
                "validation_mode": (
                    "frozen_full_content_attestation_plus_exact_metadata_and_content_probes"
                ),
                "file_count": len(rows),
                "bytes": attestation["bytes"],
                "tree_sha256": expected_tree,
                "attestation_sha256": expected_attestation,
                "uid": os.getuid(),
                "gid": os.getgid(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
