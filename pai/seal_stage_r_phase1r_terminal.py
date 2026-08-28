#!/usr/bin/env python3
"""Seal one terminally Succeeded Phase-1R execution shard fail-closed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Any


EXPECTED_OWNER = (2254, 2254)
SOURCE_COMMIT = "10308c471a846f8636cf05e5a40a2dad64f4d8ec"
RESOURCE_ID = "quotaewyznuc7b9l"
EXECUTION_SHARDS = {"A0": "A", "A1": "A", "B0": "B", "B1": "B"}


def fail(message: str) -> None:
    raise SystemExit(message)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        fail(f"missing regular JSON file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"invalid JSON {path}: {exc}")
    if not isinstance(payload, dict):
        fail(f"JSON object required: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_owner(path: Path) -> None:
    observed = (path.stat().st_uid, path.stat().st_gid)
    if observed != EXPECTED_OWNER:
        fail(f"owner mismatch for {path}: {observed} != {EXPECTED_OWNER}")


def regular_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            fail(f"symlink forbidden in terminal bundle: {path}")
        if path.is_file():
            require_owner(path)
            files.append(path)
        elif not path.is_dir():
            fail(f"unsupported filesystem entry: {path}")
    return files


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)
    require_owner(path)


def write_exhaustive_sums(root: Path) -> None:
    lines = []
    for path in regular_files(root):
        if path.name == "SHA256SUMS" or path.name.endswith(".tmp"):
            continue
        relative = path.relative_to(root).as_posix()
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or "\\" in relative:
            fail(f"unsafe relative path in terminal bundle: {relative}")
        lines.append(f"{sha256_file(path)}  {relative}\n")
    temporary = root / ".SHA256SUMS.tmp"
    temporary.write_text("".join(lines), encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(root / "SHA256SUMS")
    require_owner(root / "SHA256SUMS")


def validate_job(job: dict[str, Any], execution_shard: str, run_id: str) -> str:
    job_id = job.get("JobId")
    if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}", job_id):
        fail("invalid exact PAI JobId in terminal readback")
    if job.get("Status") != "Succeeded":
        fail(f"PAI job is not terminal Succeeded: {job.get('Status')!r}")
    if job.get("ResourceId") != RESOURCE_ID:
        fail("terminal PAI ResourceId mismatch")
    specs = job.get("JobSpecs")
    if not isinstance(specs, list) or len(specs) != 1 or specs[0].get("Type") != "Worker" or specs[0].get("PodCount") != 1:
        fail("terminal PAI worker topology mismatch")
    resources = specs[0].get("ResourceConfig")
    expected_resources = {"GPU": "4", "CPU": "46", "Memory": "800Gi", "SharedMemory": "800Gi"}
    if not isinstance(resources, dict) or any(resources.get(key) != value for key, value in expected_resources.items()):
        fail(f"terminal PAI resources mismatch: {resources}")
    settings = job.get("Settings")
    if not isinstance(settings, dict) or settings.get("OversoldType") != "AcceptQuotaOverSold":
        fail("terminal PAI idle placement contract mismatch")
    tags = settings.get("Tags")
    expected_tags = {
        "run_id": run_id,
        "experiment_role": f"phase1r-natural-execution-shard-{execution_shard.lower()}",
        "hardware": "4xa800-idle",
        "resource_pool": "idle-a800",
    }
    if not isinstance(tags, dict) or any(tags.get(key) != value for key, value in expected_tags.items()):
        fail(f"terminal PAI tags mismatch: {tags}")
    return job_id


def seal(args: argparse.Namespace) -> dict[str, Any]:
    if (os.getuid(), os.getgid()) != EXPECTED_OWNER:
        fail(f"terminal seal must run as UID:GID 2254:2254, got {os.getuid()}:{os.getgid()}")
    if args.source_commit != SOURCE_COMMIT:
        fail("terminal source commit mismatch")
    execution_shard = args.execution_shard.upper()
    logical_shard = EXECUTION_SHARDS.get(execution_shard)
    if logical_shard is None:
        fail("execution shard must be A0, A1, B0, or B1")
    root = args.run_root.resolve()
    if not root.is_dir() or args.run_root.is_symlink():
        fail(f"missing exact run directory: {root}")
    require_owner(root)
    if not root.name.startswith(f"r142-stage-r-phase1r-shard-{execution_shard.lower()}-"):
        fail(f"run id does not match execution shard {execution_shard}: {root.name}")
    completion_path = root / "COMPLETED_EVALUATION_RESULT.json"
    completion = read_json(completion_path)
    expected_completion = {
        "success_gate": "persisted_completed_evaluation_result",
        "decision": "NATURAL_EXECUTION_SHARD_COMPLETE_NO_UNBLINDING",
        "checkpoint": "CHECKPOINT_1_PENDING_GLOBAL_MERGE",
        "source_commit": SOURCE_COMMIT,
        "shard": logical_shard,
        "execution_shard": execution_shard,
        "phase1_authorized": False,
        "analysis_performed": False,
    }
    if any(completion.get(key) != value for key, value in expected_completion.items()):
        fail("execution shard completion marker mismatch")
    readback_path = args.job_readback.resolve()
    if readback_path.parent != root / "provenance" or readback_path.name != "PAI_JOB_TERMINAL_READBACK.json":
        fail("terminal job readback must be provenance/PAI_JOB_TERMINAL_READBACK.json inside the run")
    job = read_json(readback_path)
    require_owner(readback_path)
    job_id = validate_job(job, execution_shard, root.name)
    sidecar = {
        "schema_version": 1,
        "marker_type": "pai_terminal_completion",
        "shard": logical_shard,
        "execution_shard": execution_shard,
        "run_id": root.name,
        "job_id": job_id,
        "terminal_status": "Succeeded",
        "completion_marker": "COMPLETED_EVALUATION_RESULT.json",
        "completion_marker_sha256": sha256_file(completion_path),
        "sha256sums": "SHA256SUMS",
        "job_readback": "provenance/PAI_JOB_TERMINAL_READBACK.json",
        "job_readback_sha256": sha256_file(readback_path),
        "uid": os.getuid(),
        "gid": os.getgid(),
        "source_commit": SOURCE_COMMIT,
    }
    write_json_atomic(root / "PAI_TERMINAL_COMPLETION.json", sidecar)
    write_exhaustive_sums(root)
    print(json.dumps(sidecar, sort_keys=True))
    return sidecar


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--execution-shard", required=True)
    parser.add_argument("--job-readback", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    return parser


if __name__ == "__main__":
    seal(build_parser().parse_args())
