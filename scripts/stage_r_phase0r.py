#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from r142_stage_r.phase0 import Phase0Runtime
from r142_stage_r.protocol import PROTOCOL_ID, atomic_json, sha256_file


TASKS = [(suite, task_id) for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10") for task_id in range(10)]


def aggregate(output: Path, world_size: int) -> None:
    artifacts = []
    for suite, task_id in TASKS:
        stem = f"{suite}_task{task_id:02d}"
        metadata_path = output / f"{stem}.json"
        npz_path = output / f"{stem}.npz"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        actual = sha256_file(npz_path)
        if actual != metadata["data_sha256"]:
            raise RuntimeError(f"SHA mismatch for {npz_path}")
        artifacts.extend(
            [
                {"path": npz_path.name, "sha256": actual},
                {"path": metadata_path.name, "sha256": sha256_file(metadata_path)},
            ]
        )
    for rank in range(int(world_size)):
        marker = output / f"COMPLETE.rank{rank}.json"
        artifacts.append({"path": marker.name, "sha256": sha256_file(marker)})
    atomic_json(
        output / "COMPLETED_PHASE0R_RAW.json",
        {
            "protocol_id": PROTOCOL_ID,
            "task_count": 40,
            "rollout_count": 40 * 16 * 32,
            "artifacts": artifacts,
            "outcomes_unblinded": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--qpilots-root")
    parser.add_argument("--libero-root")
    parser.add_argument("--checkpoint")
    parser.add_argument("--microbatch", type=int, default=8)
    parser.add_argument("--rank", type=int, default=int(os.environ.get("RANK", "0")))
    parser.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", "1")))
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        aggregate(output, args.world_size)
        return
    for required in (args.qpilots_root, args.libero_root, args.checkpoint):
        if not required:
            parser.error("runtime source paths are required outside --aggregate-only")
    runtime = Phase0Runtime(
        qpilots_root=args.qpilots_root,
        libero_root=args.libero_root,
        checkpoint=args.checkpoint,
        microbatch=args.microbatch,
    )
    completed = []
    for index, (suite, task_id) in enumerate(TASKS):
        if index % int(args.world_size) != int(args.rank):
            continue
        stem = f"{suite}_task{task_id:02d}"
        metadata_path = output / f"{stem}.json"
        npz_path = output / f"{stem}.npz"
        if metadata_path.exists() and npz_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("data_sha256") == sha256_file(npz_path):
                completed.append(stem)
                continue
        metadata = runtime.run_task(suite, task_id, output)
        completed.append(stem)
        if not (output / f"FIRST_WORK.rank{args.rank}.json").exists():
            atomic_json(
                output / f"FIRST_WORK.rank{args.rank}.json",
                {"protocol_id": PROTOCOL_ID, "rank": args.rank, "first_complete_task": stem, "metadata_sha256": sha256_file(metadata_path)},
            )
    atomic_json(
        output / f"COMPLETE.rank{args.rank}.json",
        {"protocol_id": PROTOCOL_ID, "rank": args.rank, "completed_tasks": completed},
    )


if __name__ == "__main__":
    main()
