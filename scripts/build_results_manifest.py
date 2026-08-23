#!/usr/bin/env python3
"""Build a deterministic manifest for repository-archived result artifacts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def digest_stream(handle) -> str:
    value = hashlib.sha256()
    while chunk := handle.read(1024 * 1024):
        value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    files = []
    for path in sorted(item for item in args.results.rglob("*") if item.is_file()):
        if path.resolve() == output or path.name == "RESULTS_MANIFEST.json":
            continue
        with path.open("rb") as handle:
            sha256 = digest_stream(handle)
        row = {
            "path": path.relative_to(args.results).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256,
        }
        if path.suffix == ".gz":
            with gzip.open(path, "rb") as handle:
                row["decompressed_sha256"] = digest_stream(handle)
        files.append(row)
    value = {
        "schema_version": 1,
        "artifact_root": "results",
        "file_count_excluding_this_manifest": len(files),
        "total_bytes_excluding_this_manifest": sum(row["bytes"] for row in files),
        "files": files,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"file_count": len(files), "total_bytes": value["total_bytes_excluding_this_manifest"]}))


if __name__ == "__main__":
    main()
