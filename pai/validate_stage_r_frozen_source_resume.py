#!/usr/bin/env python3
"""Seal or quickly revalidate a frozen git-archive source tree."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path, PurePosixPath


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def inventory(root: Path, sums: Path) -> tuple[int, int, str, int]:
    listed: list[str] = []
    rows: list[str] = []
    total = 0
    for line in sums.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise SystemExit("unsafe frozen-source digest line")
        name = match.group(2)
        rel = PurePosixPath(name)
        parts = rel.parts[1:] if rel.parts and rel.parts[0] == "." else rel.parts
        if not parts or rel.is_absolute() or ".." in parts or "\\" in name:
            raise SystemExit("unsafe frozen-source digest path")
        normalized = PurePosixPath(*parts).as_posix()
        path = root.joinpath(*parts)
        if path.resolve() != path or not path.is_file() or path.is_symlink():
            raise SystemExit(f"missing/noncanonical frozen-source file: {path}")
        stat = path.stat()
        listed.append(normalized)
        total += stat.st_size
        rows.append(
            f"{normalized}\0{stat.st_size}\0{stat.st_mtime_ns}\0{stat.st_ctime_ns}\n"
        )
    actual: list[str] = []
    writable = int(bool(root.stat().st_mode & 0o222))
    for path in root.rglob("*"):
        writable += int(bool(path.stat().st_mode & 0o222))
        if path.is_symlink():
            raise SystemExit(f"frozen source contains symlink: {path}")
        if path.is_file():
            actual.append(path.relative_to(root).as_posix())
    actual.sort()
    if listed != actual:
        raise SystemExit("frozen-source file inventory drifted")
    metadata = hashlib.sha256("".join(rows).encode("utf-8")).hexdigest()
    return len(actual), total, metadata, writable


def main() -> None:
    if len(sys.argv) != 7 or sys.argv[1] not in {"seal", "validate", "validate-fast"}:
        raise SystemExit(
            "usage: validate_stage_r_frozen_source_resume.py "
            "seal|validate|validate-fast ROOT SUMS MARKER SOURCE_COMMIT SOURCE_TREE"
        )
    mode = sys.argv[1]
    root, sums, marker = map(Path, sys.argv[2:5])
    source_commit, source_tree = sys.argv[5:7]
    if mode == "validate-fast":
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if (
            marker.is_symlink()
            or payload.get("marker_type") != "frozen_source_resume_attestation"
            or payload.get("source_commit") != source_commit
            or payload.get("source_tree") != source_tree
            or payload.get("sha256sums_sha256") != digest(sums)
            or payload.get("owner") != f"{os.getuid()}:{os.getgid()}"
            or payload.get("write_locked") is not True
            or root.stat().st_mode & 0o222
        ):
            raise SystemExit("fast frozen-source resume attestation drifted")
        print(json.dumps({"valid": True, "validation_mode": "locked_preoutcome_attestation"}, sort_keys=True))
        return
    file_count, total_bytes, metadata_sha256, writable_count = inventory(root, sums)
    payload = {
        "schema_version": 1,
        "marker_type": "frozen_source_resume_attestation",
        "source_commit": source_commit,
        "source_tree": source_tree,
        "sha256sums_sha256": digest(sums),
        "file_count": file_count,
        "bytes": total_bytes,
        "metadata_sha256": metadata_sha256,
        "owner": f"{os.getuid()}:{os.getgid()}",
        "write_locked": writable_count == 0,
    }
    if not payload["write_locked"]:
        raise SystemExit("frozen source is not write-locked")
    if mode == "seal":
        temporary = marker.with_suffix(marker.suffix + f".tmp.{os.getpid()}")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, marker)
    else:
        if json.loads(marker.read_text(encoding="utf-8")) != payload:
            raise SystemExit("frozen-source resume attestation drifted")
    print(json.dumps({"valid": True, **payload}, sort_keys=True))


if __name__ == "__main__":
    main()
