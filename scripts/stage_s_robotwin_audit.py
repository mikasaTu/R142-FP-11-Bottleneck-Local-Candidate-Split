#!/usr/bin/env python3
"""Audit exact RoboTwin/Evo-1 assets without downloading the checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from r142_stage_s.robotwin import (
    PUBLISHED_CLEAN_SUCCESS,
    PUBLISHED_EVAL_URL,
    RoboTwinPins,
    select_published_tasks,
)


def _head(path: Path) -> Optional[str]:
    if not (path / ".git").exists():
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(
    *,
    robotwin_root: Path,
    evo_root: Path,
    checkpoint_dir: Path,
    pins: RoboTwinPins = RoboTwinPins(),
) -> Dict[str, Any]:
    selected = select_published_tasks()
    inventory = {
        "robotwin_root": str(robotwin_root),
        "robotwin_head": _head(robotwin_root),
        "evo_root": str(evo_root),
        "evo_head": _head(evo_root),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_files": {
            name: _sha(checkpoint_dir / name)
            for name in ("config.json", "norm_stats.json", "mp_rank_00_model_states.pt")
        },
    }
    tasks = [
        {
            "task": task,
            "published_clean_success": PUBLISHED_CLEAN_SUCCESS[task],
            "task_module_present": (robotwin_root / "envs" / f"{task}.py").is_file(),
            "instruction_present": (
                robotwin_root / "description" / "task_instruction" / f"{task}.json"
            ).is_file(),
        }
        for task in selected
    ]
    missing = []
    if inventory["robotwin_head"] != pins.robotwin_revision:
        missing.append(f"RoboTwin checkout {pins.robotwin_revision}")
    if inventory["evo_head"] != pins.evo_revision:
        missing.append(f"Evo-1 checkout {pins.evo_revision}")
    if any(value is None for value in inventory["checkpoint_files"].values()):
        missing.append(f"checkpoint files at HF revision {pins.checkpoint_revision}")
    if not all(row["task_module_present"] and row["instruction_present"] for row in tasks):
        missing.append("all ten task modules and instruction files")
    result: Dict[str, Any] = {
        "status": "READY_FOR_REAL_RUNTIME_PREFLIGHT" if not missing else "BLOCKED_CAPABILITY",
        "capability_error": "; ".join(missing) if missing else None,
        "pins": pins.as_dict(),
        "published_eval_source": PUBLISHED_EVAL_URL,
        "source_inventory": inventory,
        "selected_tasks": tasks,
        "synthetic_rollouts": False,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--evo-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        robotwin_root=args.robotwin_root,
        evo_root=args.evo_root,
        checkpoint_dir=args.checkpoint_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"].startswith("READY") else 2


if __name__ == "__main__":
    raise SystemExit(main())
