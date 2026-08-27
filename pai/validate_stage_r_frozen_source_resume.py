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


def inventory(root: Path, sums: Path) -> tuple[int, int, str]:
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
    for path in root.rglob("*"):
        if path.is_symlink():
            raise SystemExit(f"frozen source contains symlink: {path}")
        if path.is_file():
            actual.append(path.relative_to(root).as_posix())
    actual.sort()
    if listed != actual:
        raise SystemExit("frozen-source file inventory drifted")
    metadata = hashlib.sha256("".join(rows).encode("utf-8")).hexdigest()
    return len(actual), total, metadata


def main() -> None:
    if len(sys.argv) != 7 or sys.argv[1] not in {"seal", "validate"}:
        raise SystemExit(
            "usage: validate_stage_r_frozen_source_resume.py "
            "seal|validate ROOT SUMS MARKER SOURCE_COMMIT SOURCE_TREE"
        )
    mode = sys.argv[1]
    root, sums, marker = map(Path, sys.argv[2:5])
    source_commit, source_tree = sys.argv[5:7]
    file_count, total_bytes, metadata_sha256 = inventory(root, sums)
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
    }
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
