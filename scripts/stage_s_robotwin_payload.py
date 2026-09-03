#!/usr/bin/env python3
"""Build (but never submit) the PAI payload for Stage-S substrate A.

The payload is deliberately a plain JSON template.  Submission, scheduler
polling, and job cancellation belong to the parent orchestration layer.  The
runtime command embeds the same fail-closed time guard and exact replay gate
used by :mod:`stage_s_robotwin_main`.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
from zoneinfo import ZoneInfo

from r142_stage_s.robotwin import CapabilityError, RoboTwinPins, select_published_tasks

from scripts.stage_s_robotwin_main import assert_outside_blackout


BEIJING = ZoneInfo("Asia/Shanghai")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def build_payload(
    *,
    run_id: str,
    output_root: Path,
    robotwin_root: Path,
    evo_root: Path,
    checkpoint_dir: Path,
    server_url: str,
    rank: int = 0,
    world_size: int = 1,
    seed_base: int = 14211,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Return a validated robot-idle PAI template for one rank."""
    if not _SAFE_ID.fullmatch(run_id):
        raise CapabilityError("run-id must be a short filesystem-safe identifier")
    if rank < 0 or world_size <= 0 or rank >= world_size:
        raise CapabilityError("rank/world-size must satisfy 0 <= rank < world-size")
    # The check is intentionally performed while constructing a payload too:
    # a caller cannot accidentally queue a job in either forbidden window.
    assert_outside_blackout(now)
    pins = RoboTwinPins()
    tasks = list(select_published_tasks())
    command = [
        "python3",
        "scripts/stage_s_robotwin_main.py",
        "--phase",
        "main",
        "--robotwin-root",
        str(robotwin_root),
        "--evo-root",
        str(evo_root),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--output-root",
        str(output_root),
        "--server-url",
        server_url,
        "--rank",
        str(rank),
        "--world-size",
        str(world_size),
        "--families-per-task",
        "16",
        "--candidates",
        "32",
        "--seed-base",
        str(seed_base),
    ]
    return {
        "template_version": 1,
        "submission": {
            "mode": "template_only",
            "submit": False,
            "reason": "parent orchestrator owns PAI submission and lifecycle",
        },
        "run_id": run_id,
        "protocol": "R142-FP-11 Stage-S substrate A",
        "pins": pins.as_dict(),
        "tasks": tasks,
        "families_per_task": 16,
        "candidates_per_family": 32,
        "shard": {
            "rank": rank,
            "world_size": world_size,
            "assignment": "flat_task_family_index % world_size == rank",
            "same_output_root": str(output_root),
        },
        "resources": {
            "pool": "robot",
            "resource_mode": "idle",
            "preemptible": True,
            "gpu_type": "A800",
            "gpu_count": 8,
            "cpu_cores": 88,
            "memory_gib": 1400,
        },
        "resume": {
            "policy": "same_directory_idempotent",
            "completed_marker": "COMPLETED_FAMILY.json",
            "integrity_manifest": "SHA256SUMS",
            "valid_marker_action": "skip_family",
            "invalid_marker_action": "fail_closed",
            "partial_family_action": "resume_same_family_path",
        },
        "scheduler_guard": {
            "timezone": "Asia/Shanghai",
            "forbidden_windows": ["09:30-09:40", "19:30-19:40"],
            "action": "fail_closed_do_not_start_or_resume",
        },
        "runtime": {
            "command": command,
            "synthetic_rollouts": False,
            "expert_trajectory": False,
            "termination": "official eval_success or step_lim",
            "replay_gate": "restore -> same action -> next-state <= 1e-9",
            "rank_completion": f"COMPLETED_A_RANK-{rank:04d}.json",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--evo-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=14211)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--now",
        help="optional ISO timestamp for guard testing (interpreted in Asia/Shanghai)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        now = datetime.fromisoformat(args.now) if args.now else None
        if now is not None and now.tzinfo is None:
            now = now.replace(tzinfo=BEIJING)
        payload = build_payload(
            run_id=args.run_id,
            output_root=args.output_root,
            robotwin_root=args.robotwin_root,
            evo_root=args.evo_root,
            checkpoint_dir=args.checkpoint_dir,
            server_url=args.server_url,
            rank=args.rank,
            world_size=args.world_size,
            seed_base=args.seed_base,
            now=now,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except (CapabilityError, ValueError) as exc:
        print(json.dumps({"status": "BLOCKED_CAPABILITY", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
