#!/usr/bin/env python3
"""Validate frozen calibration or authoritative Phase-0R prerequisites."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            value.update(chunk)
    return value.hexdigest()


def check_sums(path: Path, base: Path) -> list[str]:
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise SystemExit(f"unsafe digest line: {line!r}")
        digest, name = match.groups()
        rel = PurePosixPath(name)
        if rel.is_absolute() or ".." in rel.parts or "\\" in name or not name:
            raise SystemExit(f"unsafe digest path: {name!r}")
        target = base.joinpath(*rel.parts)
        if target.resolve() != target or not target.is_file() or target.is_symlink():
            raise SystemExit(f"missing/noncanonical file: {target}")
        if sha(target) != digest:
            raise SystemExit(f"digest mismatch: {target}")
        names.append(name)
    if len(names) != len(set(names)):
        raise SystemExit(f"duplicate path in {path}")
    return names


def calibration(calibration_file: Path, seal: Path) -> None:
    root = calibration_file.parent
    check_sums(seal, root)
    payload = json.loads(calibration_file.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != "r142-stage-r-phase1r-human-override-v1":
        raise SystemExit("calibration protocol mismatch")
    if int(payload.get("shuffles", 0)) < 1000:
        raise SystemExit("calibration shuffles below frozen minimum")
    if (
        payload.get("unpermuted_curve_present") is not False
        or payload.get("natural_curve_present") is not False
    ):
        raise SystemExit("calibration is not blinded")
    if any(key in payload for key in ("curve", "curves", "natural")):
        raise SystemExit("calibration contains unblinded data")
    marker = json.loads((root / "COMPLETED_CALIBRATION.json").read_text(encoding="utf-8"))
    if marker.get("protocol_id") != payload["protocol_id"] or marker.get("owner") != "2254:2254":
        raise SystemExit("calibration completion marker drifted")
    print(json.dumps({"valid": True, "kind": "calibration"}, sort_keys=True))


def phase0(merge_root: Path, raw_root: Path, expected_sha: str) -> None:
    authority = merge_root / "AUTHORITY_MANIFEST.json"
    if sha(authority) != expected_sha:
        raise SystemExit("authority manifest SHA mismatch")
    payload = json.loads(authority.read_text(encoding="utf-8"))
    if (
        payload.get("protocol_id") != "r142-stage-r-phase0r-v1"
        or payload.get("outcome_selection_permitted") is not False
    ):
        raise SystemExit("authority protocol/outcome contract drifted")
    if payload.get("authority_rule") != "parent[0:32]+shard_A[32:36]+shard_B[36:40]":
        raise SystemExit("authority range rule drifted")
    records = payload.get("records")
    if not isinstance(records, list) or [row.get("index") for row in records] != list(range(40)):
        raise SystemExit("authority records do not cover indices 0..39")
    if any(row.get("source") != "parent" for row in records[:32]):
        raise SystemExit("parent authority source drift")
    if any(row.get("source") != "shard_a" for row in records[32:36]):
        raise SystemExit("shard A authority source drift")
    if any(row.get("source") != "shard_b" for row in records[36:]):
        raise SystemExit("shard B authority source drift")
    raw_marker = json.loads((merge_root / "COMPLETED_PHASE0R_RAW.json").read_text(encoding="utf-8"))
    if (
        raw_marker.get("protocol_id") != "r142-stage-r-phase0r-v1"
        or raw_marker.get("task_count") != 40
        or raw_marker.get("rollout_count") != 20480
    ):
        raise SystemExit("raw completion marker cardinality/protocol drift")
    if (
        raw_marker.get("outcomes_unblinded") is not False
        or raw_marker.get("authority_manifest_sha256") != expected_sha
    ):
        raise SystemExit("raw completion marker unblinding/authority drift")
    names = check_sums(merge_root / "MERGE_SHA256SUMS", merge_root)
    npz = sorted(path.name for path in raw_root.glob("*.npz"))
    metadata = sorted(path.name for path in raw_root.glob("*.json"))
    if len(npz) != 40 or len(metadata) != 40:
        raise SystemExit("raw input must contain exactly 40 NPZ and 40 metadata files")
    expected_raw = sorted("raw/" + name for name in npz + metadata)
    if sorted(name for name in names if name.startswith("raw/")) != expected_raw:
        raise SystemExit("merge seal raw file set drifted")
    print(
        json.dumps(
            {
                "valid": True,
                "authority_records": 40,
                "raw_tasks": 40,
                "raw_rollouts": 20480,
                "raw_files": 80,
            },
            sort_keys=True,
        )
    )


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("missing prerequisite validation kind")
    if sys.argv[1] == "calibration" and len(sys.argv) == 4:
        calibration(Path(sys.argv[2]), Path(sys.argv[3]))
    elif sys.argv[1] == "phase0" and len(sys.argv) == 5:
        phase0(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4])
    else:
        raise SystemExit("invalid prerequisite validation arguments")


if __name__ == "__main__":
    main()
