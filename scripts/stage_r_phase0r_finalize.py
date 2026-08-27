#!/usr/bin/env python3
"""Seal frozen Phase-0R analysis at Checkpoint 1 without authorizing Phase 1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


PROTOCOL_ID = "r142-stage-r-phase0r-v1"
ALLOWED_DECISIONS = {
    "PIPELINE_INVALID",
    "NO_STAGE_R_PRECONDITION_ON_PINNED_PI05_LIBERO",
    "CHECKPOINT1_TASKS_RETAINED",
}
TASK_KEYS = [
    (suite, task_id)
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    for task_id in range(10)
]


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def verify_manifest(root: Path, name: str) -> None:
    for line in (root / name).read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError(f"unsafe checksum entry: {relative}")
        path = root / relative_path
        if path.is_symlink() or not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"checksum mismatch: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--uid", type=int, default=2254)
    parser.add_argument("--gid", type=int, default=2254)
    args = parser.parse_args()
    root = args.root
    if (os.getuid(), os.getgid()) != (args.uid, args.gid):
        raise RuntimeError("finalizer must run as the frozen numeric owner")
    verify_manifest(root, "SHA256SUMS")
    authority_path = root / "AUTHORITY_MANIFEST.json"
    raw_complete_path = root / "COMPLETED_PHASE0R_RAW.json"
    summary_path = root / "analysis/phase0r_summary.json"
    analysis_complete_path = root / "analysis/COMPLETED_PHASE0R.json"
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    raw_complete = json.loads(raw_complete_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    analysis_complete = json.loads(analysis_complete_path.read_text(encoding="utf-8"))
    if authority["protocol_id"] != PROTOCOL_ID or authority["outcome_selection_permitted"] is not False:
        raise RuntimeError("authority manifest mismatch")
    if raw_complete["protocol_id"] != PROTOCOL_ID or raw_complete["task_count"] != 40 or raw_complete["rollout_count"] != 20480 or raw_complete["outcomes_unblinded"] is not False:
        raise RuntimeError("raw completion mismatch")
    if summary["protocol_id"] != PROTOCOL_ID or summary["decision"] not in ALLOWED_DECISIONS:
        raise RuntimeError("analysis decision mismatch")
    if summary["checkpoint"] != "CHECKPOINT_1_STOP" or summary["phase1_authorized"] is not False:
        raise RuntimeError("analysis escaped Checkpoint 1")
    rows = summary["candidate_rows"]
    if [(row["suite"], row["task_id"]) for row in rows[:40]] != TASK_KEYS:
        raise RuntimeError("candidate table order mismatch")
    if rows[40].get("suite") != "robotwin" or rows[40].get("source_status") != "SOURCE_LIMITATION_UNVERIFIABLE":
        raise RuntimeError("RoboTwin limitation row mismatch")
    retained = [{"suite": row["suite"], "task_id": row["task_id"]} for row in rows[:40] if row.get("retained")]
    if retained != summary["retained_tasks"]:
        raise RuntimeError("retained-task table mismatch")
    summary_sha = sha256(summary_path)
    if analysis_complete != {
        "checkpoint": "CHECKPOINT_1_STOP",
        "decision": summary["decision"],
        "protocol_id": PROTOCOL_ID,
        "summary": str(summary_path),
        "summary_sha256": summary_sha,
    }:
        raise RuntimeError("analysis completion marker mismatch")
    merge_manifest = root / "MERGE_SHA256SUMS"
    if merge_manifest.exists():
        if merge_manifest.read_bytes() != (root / "SHA256SUMS").read_bytes():
            raise RuntimeError("existing merge checksum snapshot mismatch")
    else:
        shutil.copyfile(root / "SHA256SUMS", merge_manifest)
    first = {
        "schema_version": 1,
        "milestone": "all_authoritative_raw_complete_before_frozen_analysis",
        "protocol_id": PROTOCOL_ID,
        "authority_manifest_sha256": sha256(authority_path),
        "raw_task_count": 40,
        "raw_rollout_count": 20480,
        "uid": os.getuid(),
        "gid": os.getgid(),
    }
    completed = {
        "schema_version": 1,
        "success_gate": "persisted_complete_phase0r_evaluation",
        "protocol_id": PROTOCOL_ID,
        "decision": summary["decision"],
        "retained_tasks": summary["retained_tasks"],
        "positive_control_pass": summary["positive_control_pass"],
        "checkpoint": "CHECKPOINT_1_STOP",
        "phase1_authorized": False,
        "authority_manifest_sha256": sha256(authority_path),
        "raw_completion_sha256": sha256(raw_complete_path),
        "analysis_summary_sha256": summary_sha,
        "analysis_completion_sha256": sha256(analysis_complete_path),
        "uid": os.getuid(),
        "gid": os.getgid(),
    }
    atomic_json(root / "FIRST_WORK.json", first)
    atomic_json(root / "COMPLETED_EVALUATION_RESULT.json", completed)
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    temporary = root / ".SHA256SUMS.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        for path in paths:
            handle.write(f"{sha256(path)}  {path.relative_to(root)}\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, root / "SHA256SUMS")
    verify_manifest(root, "SHA256SUMS")
    print(f"PHASE0R_FINALIZED decision={summary['decision']} checkpoint=CHECKPOINT_1_STOP")


if __name__ == "__main__":
    main()
