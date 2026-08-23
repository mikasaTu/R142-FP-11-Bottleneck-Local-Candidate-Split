#!/usr/bin/env python3
"""Apply the frozen, descendant-blind natural snapshot selection rule."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def rank(row: dict) -> str:
    return hashlib.sha256(
        f"stage2a-snapshot-selection-v1:{row['snapshot_id']}".encode()
    ).hexdigest()


def stratum(row: dict) -> str:
    if not row["episode_success"]:
        return "hard"
    if row["max_progress_so_far"] >= 0.75:
        return "easy"
    return "ambiguous"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=Path, required=True)
    parser.add_argument("--target-per-stratum", type=int, default=8)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.frame.read_text().splitlines() if line.strip()]
    groups = {name: [] for name in ("easy", "ambiguous", "hard")}
    for row in rows:
        row["stratum"] = stratum(row)
        row["selection_rank_sha256"] = rank(row)
        groups[row["stratum"]].append(row)
    selected = set()
    shortfall = {}
    for name, values in groups.items():
        values.sort(key=lambda row: row["selection_rank_sha256"])
        chosen = values[: args.target_per_stratum]
        selected.update(row["snapshot_id"] for row in chosen)
        shortfall[name] = max(0, args.target_per_stratum - len(chosen))
    for row in rows:
        row["selected"] = row["snapshot_id"] in selected
    tmp = args.frame.with_suffix(args.frame.suffix + ".tmp")
    with tmp.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(tmp, args.frame)
    manifest = {
        "schema_version": 1,
        "rule": "stratify on frozen baseline outcomes only, deterministic SHA-256 rank, no descendant outcomes",
        "target_per_stratum": args.target_per_stratum,
        "candidate_count": len(rows),
        "selected_count": len(selected),
        "counts": {name: len(values) for name, values in groups.items()},
        "shortfall": shortfall,
        "selected_snapshot_ids": sorted(selected),
    }
    output = args.frame.parent / "snapshot_selection_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
