#!/usr/bin/env python3
"""Write or validate the frozen Phase-1R execution-shard task mapping."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


RANKS = {
    "A0": list(range(0, 4)),
    "A1": list(range(4, 8)),
    "B0": list(range(8, 12)),
    "B1": list(range(12, 16)),
}
COUNTS = {"A0": 7, "A1": 10, "B0": 11, "B1": 12}


def expected_text(
    shards_path: Path,
    execution_path: Path,
    execution_shard: str,
    selection_root: Path,
) -> str:
    shards = json.loads(shards_path.read_text(encoding="utf-8"))
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    execution_entry = execution["execution_shards"].get(execution_shard)
    if not isinstance(execution_entry, dict):
        raise SystemExit(f"missing execution shard {execution_shard}")
    logical_shard = execution_entry.get("logical_shard")
    payload = shards["shards"].get(logical_shard)
    if not isinstance(payload, dict):
        raise SystemExit(f"missing shard {logical_shard}")
    global_ranks = [int(value) for value in execution_entry.get("global_ranks", [])]
    if global_ranks != RANKS[execution_shard]:
        raise SystemExit(f"shard global ranks drifted: {global_ranks}")
    rank_tasks = payload.get("rank_tasks")
    if not isinstance(rank_tasks, dict):
        raise SystemExit("rank_tasks missing")
    lines = ["local_rank\tglobal_rank\ttask_name\tsuite\ttask_id\tselection_path"]
    seen = set()
    for local_rank, global_rank in enumerate(global_ranks):
        names = rank_tasks.get(str(global_rank))
        if not isinstance(names, list) or not names:
            raise SystemExit(f"rank {global_rank} has no fixed tasks")
        for task_name in names:
            match = re.fullmatch(r"(libero_(?:spatial|object|goal|10))_task(\d{2})", str(task_name))
            if match is None or task_name in seen:
                raise SystemExit(f"invalid/duplicate task {task_name!r}")
            seen.add(task_name)
            suite, task_text = match.groups()
            task_id = int(task_text)
            selection = selection_root / f"{task_name}.json"
            if not selection.is_file() or selection.is_symlink():
                raise SystemExit(f"missing selection {selection}")
            lines.append(
                f"{local_rank}\t{global_rank}\t{task_name}\t{suite}\t{task_id}\t{selection}"
            )
    if len(seen) != COUNTS[execution_shard]:
        raise SystemExit(f"shard task count drifted: {len(seen)}")
    return "\n".join(lines) + "\n"


def main() -> None:
    if len(sys.argv) != 7 or sys.argv[1] not in {"write", "validate"}:
        raise SystemExit(
            "usage: stage_r_phase1r_task_mapping.py write|validate "
            "SHARDS EXECUTION EXECUTION_SHARD SELECTION_ROOT OUTPUT"
        )
    mode = sys.argv[1]
    shards_path, execution_path = Path(sys.argv[2]), Path(sys.argv[3])
    execution_shard, selection_root, output = sys.argv[4], Path(sys.argv[5]), Path(sys.argv[6])
    if execution_shard not in RANKS:
        raise SystemExit("unknown execution shard")
    expected = expected_text(shards_path, execution_path, execution_shard, selection_root)
    if mode == "write":
        output.write_text(expected, encoding="utf-8")
    elif output.is_symlink() or not output.is_file() or output.read_text(encoding="utf-8") != expected:
        raise SystemExit("task mapping drifted")
    print(json.dumps({"valid": True, "execution_shard": execution_shard}, sort_keys=True))


if __name__ == "__main__":
    main()
