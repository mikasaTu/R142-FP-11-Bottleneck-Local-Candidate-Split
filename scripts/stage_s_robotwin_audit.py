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


def _checkpoint_revision_evidence(
    checkpoint_dir: Path, expected_revision: str
) -> Dict[str, Any]:
    """Resolve explicit HF revision evidence without assuming a cache layout."""
    if expected_revision in checkpoint_dir.name:
        return {"source": "directory_name", "revision": expected_revision, "matches": True}
    for filename in (
        "revision.txt",
        "checkpoint_revision.txt",
        "REVISION",
        "revision.json",
        "checkpoint_revision.json",
    ):
        path = checkpoint_dir / filename
        if not path.is_file():
            continue
        try:
            raw = path.read_text(encoding="utf-8").strip()
            if path.suffix == ".json":
                value = json.loads(raw)
                revision = value.get("revision") if isinstance(value, dict) else value
            else:
                revision = raw.splitlines()[0] if raw else None
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            revision = None
        return {
            "source": str(path),
            "revision": revision,
            "matches": revision == expected_revision,
        }
    return {"source": None, "revision": None, "matches": False}


def _checkpoint_hash_expectations(checkpoint_dir: Path) -> Dict[str, str]:
    """Read optional expected file hashes from explicit revision metadata."""
    for filename in ("revision.json", "checkpoint_revision.json"):
        path = checkpoint_dir / filename
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        files = value.get("files") if isinstance(value, dict) else None
        if isinstance(files, dict):
            return {str(name): str(sha) for name, sha in files.items()}
    return {}


def audit(
    *,
    robotwin_root: Path,
    evo_root: Path,
    checkpoint_dir: Path,
    runtime_wrapper: Optional[Path] = None,
    pins: RoboTwinPins = RoboTwinPins(),
) -> Dict[str, Any]:
    selected = select_published_tasks()
    revision_evidence = _checkpoint_revision_evidence(
        checkpoint_dir, pins.checkpoint_revision
    )
    expected_hashes = _checkpoint_hash_expectations(checkpoint_dir)
    checkpoint_files = {
        name: _sha(checkpoint_dir / name)
        for name in ("config.json", "norm_stats.json", "mp_rank_00_model_states.pt")
    }
    hash_mismatches = {
        name: {"expected": expected, "actual": checkpoint_files.get(name)}
        for name, expected in expected_hashes.items()
        if checkpoint_files.get(name) != expected
    }
    inventory = {
        "robotwin_root": str(robotwin_root),
        "robotwin_head": _head(robotwin_root),
        "evo_root": str(evo_root),
        "evo_head": _head(evo_root),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_files": checkpoint_files,
        "checkpoint_revision_evidence": revision_evidence,
        "checkpoint_hash_mismatches": hash_mismatches,
        "runtime_wrapper": str(runtime_wrapper) if runtime_wrapper else None,
    }
    wrapper_text = ""
    if runtime_wrapper is not None and runtime_wrapper.is_file():
        wrapper_text = runtime_wrapper.read_text(encoding="utf-8", errors="replace")
    concrete_wrapper_verified = all(
        symbol in wrapper_text
        for symbol in ("ConcreteRoboTwinRuntime", "EvoProxyStateAdapter")
    )
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
    if not revision_evidence["matches"]:
        missing.append(
            f"explicit checkpoint revision evidence for HF commit {pins.checkpoint_revision}"
        )
    if hash_mismatches:
        missing.append("checkpoint file hashes matching revision metadata")
    if not all(row["task_module_present"] and row["instruction_present"] for row in tasks):
        missing.append("all ten task modules and instruction files")
    if not concrete_wrapper_verified:
        missing.append(
            "concrete wrapper exporting ConcreteRoboTwinRuntime and EvoProxyStateAdapter"
        )
    result: Dict[str, Any] = {
        "status": "READY_FOR_REAL_RUNTIME_PREFLIGHT" if not missing else "BLOCKED_CAPABILITY",
        "capability_error": "; ".join(missing) if missing else None,
        "pins": pins.as_dict(),
        "published_eval_source": PUBLISHED_EVAL_URL,
        "source_inventory": inventory,
        "concrete_wrapper_verified": concrete_wrapper_verified,
        "selected_tasks": tasks,
        "synthetic_rollouts": False,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--evo-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--runtime-wrapper", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        robotwin_root=args.robotwin_root,
        evo_root=args.evo_root,
        checkpoint_dir=args.checkpoint_dir,
        runtime_wrapper=args.runtime_wrapper,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"].startswith("READY") else 2


if __name__ == "__main__":
    raise SystemExit(main())
