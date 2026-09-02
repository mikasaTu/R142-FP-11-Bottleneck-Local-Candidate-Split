#!/usr/bin/env python3
"""Fail-closed aggregation of the eight Stage-S substrate-A rank shards.

This command is run only after all eight client processes and their paired
Evo servers have exited.  It accepts no partial or synthetic evidence.  It
verifies every rank completion marker, every family marker and family
``SHA256SUMS`` before writing the single top-level
``COMPLETED_EVALUATION_RESULT.json`` and a recursive ``SHA256SUMS`` manifest.
The result is idempotent: a previously complete directory is re-verified and
never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from r142_stage_s.frozen_protocol import (
    DEFAULT_PROTOCOL_PATH,
    FrozenProtocolError,
    load_frozen_protocol,
)
from r142_stage_s.robotwin import RoboTwinPins, select_published_tasks


WORLD_SIZE = 8
FAMILIES_PER_TASK = 16
CANDIDATES_PER_FAMILY = 32
EXPECTED_CANDIDATES = 10 * FAMILIES_PER_TASK * CANDIDATES_PER_FAMILY


class EvaluationBundleError(RuntimeError):
    """Incomplete, inconsistent, or unverifiable persisted evaluation data."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise EvaluationBundleError(f"required JSON file is missing or symlinked: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise EvaluationBundleError(f"invalid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise EvaluationBundleError(f"JSON root must be an object: {path}")
    return value


def _verify_manifest_file(root: Path, manifest: Path, *, allow_self: bool = False) -> None:
    if not manifest.is_file() or manifest.is_symlink():
        raise EvaluationBundleError(f"SHA256SUMS is missing or symlinked: {manifest}")
    seen: set[str] = set()
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        parts = raw.split(None, 1)
        if len(parts) != 2:
            raise EvaluationBundleError(f"malformed checksum line in {manifest}: {raw!r}")
        expected, rel = parts
        rel = rel.lstrip(" *")
        if not rel or Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise EvaluationBundleError(f"unsafe checksum path in {manifest}: {rel!r}")
        if rel in seen:
            raise EvaluationBundleError(f"duplicate checksum path in {manifest}: {rel}")
        seen.add(rel)
        path = manifest.parent / rel
        if not path.is_file() or path.is_symlink():
            raise EvaluationBundleError(f"checksum target is missing or symlinked: {path}")
        if _sha256(path) != expected:
            raise EvaluationBundleError(f"checksum mismatch: {path}")
    if not seen:
        raise EvaluationBundleError(f"empty checksum manifest: {manifest}")
    if not allow_self and manifest.name in seen:
        raise EvaluationBundleError(f"nested checksum manifest self-hash is forbidden: {manifest}")


def _verify_family(root: Path, *, task: str, family_index: int, rank: int) -> Mapping[str, Any]:
    family_id = f"family-{family_index:04d}"
    directory = root / f"rank-{rank:04d}" / task / family_id
    marker_path = directory / "COMPLETED_FAMILY.json"
    marker = _read_json(marker_path)
    if marker.get("family_id") != family_id:
        raise EvaluationBundleError(f"family marker id mismatch: {marker_path}")
    if int(marker.get("candidate_count", -1)) != CANDIDATES_PER_FAMILY:
        raise EvaluationBundleError(f"family candidate count mismatch: {marker_path}")
    files = marker.get("files")
    if not isinstance(files, Mapping) or not files:
        raise EvaluationBundleError(f"family marker has no immutable file map: {marker_path}")
    for name, expected in files.items():
        path = directory / str(name)
        if not path.is_file() or path.is_symlink() or _sha256(path) != str(expected):
            raise EvaluationBundleError(f"family file hash mismatch: {path}")
    _verify_manifest_file(directory, directory / "SHA256SUMS")
    family_payload = _read_json(directory / "family.json")
    if family_payload.get("family_id") != family_id:
        raise EvaluationBundleError(f"family payload id mismatch: {directory / 'family.json'}")
    metadata = family_payload.get("metadata")
    candidates = family_payload.get("candidates")
    if not isinstance(metadata, Mapping) or not isinstance(candidates, list):
        raise EvaluationBundleError(f"family payload schema mismatch: {directory / 'family.json'}")
    if int(metadata.get("candidate_count", -1)) != CANDIDATES_PER_FAMILY:
        raise EvaluationBundleError(f"family metadata candidate count mismatch: {directory}")
    if metadata.get("termination") != "official eval_success or step_lim":
        raise EvaluationBundleError(f"family termination is not the frozen official rule: {directory}")
    required = {
        "candidate_id", "parent_id", "generation_step", "action_prefix",
        "final_success", "task_name", "family_id", "initial_state_id", "seed",
        "seed_sequence", "seed_genealogy", "policy_history", "action_queue", "rng_state",
    }
    ids: set[str] = set()
    successes = 0
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping) or not required.issubset(candidate):
            raise EvaluationBundleError(f"candidate genealogy schema mismatch: {directory}")
        expected_id = f"{family_id}/candidate-{index:04d}"
        if candidate.get("candidate_id") != expected_id:
            raise EvaluationBundleError(f"candidate id/order mismatch: {directory}")
        if candidate.get("family_id") != family_id or candidate.get("task_name") != task:
            raise EvaluationBundleError(f"candidate family/task mismatch: {directory}")
        if not isinstance(candidate.get("final_success"), bool):
            raise EvaluationBundleError(f"candidate success is not a boolean: {directory}")
        if not isinstance(candidate.get("action_prefix"), list):
            raise EvaluationBundleError(f"candidate action prefix is not persisted: {directory}")
        if not isinstance(candidate.get("seed_genealogy"), Mapping):
            raise EvaluationBundleError(f"candidate seed genealogy is not persisted: {directory}")
        if candidate.get("candidate_id") in ids:
            raise EvaluationBundleError(f"duplicate candidate id: {directory}")
        ids.add(str(candidate.get("candidate_id")))
        successes += int(candidate.get("final_success"))
    if len(candidates) != CANDIDATES_PER_FAMILY:
        raise EvaluationBundleError(f"family candidate list has wrong length: {directory}")
    genealogy = directory / "genealogy.jsonl"
    lines = [line for line in genealogy.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != CANDIDATES_PER_FAMILY:
        raise EvaluationBundleError(f"genealogy line count mismatch: {genealogy}")
    for index, line in enumerate(lines):
        try:
            genealogy_record = json.loads(line)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise EvaluationBundleError(f"invalid genealogy JSON: {genealogy}") from exc
        if not isinstance(genealogy_record, Mapping) or genealogy_record.get("candidate_id") != f"{family_id}/candidate-{index:04d}":
            raise EvaluationBundleError(f"genealogy candidate order mismatch: {genealogy}")
    snapshot = _read_json(directory / "SNAPSHOT.json")
    if set(snapshot) != {"simulator", "policy_history", "action_queue", "rng_streams"}:
        raise EvaluationBundleError(f"snapshot does not contain full replay state: {directory}")
    return {
        "family_id": f"{task}/{family_id}",
        "task": task,
        "rank": rank,
        "candidate_count": len(candidates),
        "success_count": successes,
        "path": str(directory),
        "completion_sha256": _sha256(marker_path),
    }


def _verify_rank(
    root: Path,
    *,
    rank: int,
    tasks: Sequence[str],
    run_id: str | None,
    frozen_protocol: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    run_manifest_path = root / f"RUN_MANIFEST_RANK-{rank:04d}.json"
    run_manifest = _read_json(run_manifest_path)
    if run_manifest.get("protocol") != "R142-FP-11 Stage-S substrate A":
        raise EvaluationBundleError(f"rank run manifest protocol mismatch: {run_manifest_path}")
    if int(run_manifest.get("rank", -1)) != rank or int(run_manifest.get("world_size", -1)) != WORLD_SIZE:
        raise EvaluationBundleError(f"rank run manifest rank/world_size mismatch: {run_manifest_path}")
    if run_manifest.get("selected_tasks") != list(tasks):
        raise EvaluationBundleError(f"rank run manifest task selection drifted: {run_manifest_path}")
    if int(run_manifest.get("families_per_task", -1)) != FAMILIES_PER_TASK or int(run_manifest.get("candidate_budget", -1)) != CANDIDATES_PER_FAMILY:
        raise EvaluationBundleError(f"rank run manifest budget drifted: {run_manifest_path}")
    if run_manifest.get("synthetic_rollouts") is not False or run_manifest.get("expert_trajectory") is not False:
        raise EvaluationBundleError(f"rank run manifest permits synthetic/expert evidence: {run_manifest_path}")
    if run_manifest.get("termination") != "official eval_success or step_lim":
        raise EvaluationBundleError(f"rank run manifest termination drifted: {run_manifest_path}")
    if run_manifest.get("pins") != RoboTwinPins().as_dict():
        raise EvaluationBundleError(f"rank run manifest source pins drifted: {run_manifest_path}")
    if run_manifest.get("frozen_protocol") != dict(frozen_protocol):
        raise EvaluationBundleError(f"rank run manifest frozen protocol drifted: {run_manifest_path}")
    if run_manifest.get("protocol_git_commit") != frozen_protocol["protocol_git_commit"]:
        raise EvaluationBundleError(f"rank run manifest protocol commit drifted: {run_manifest_path}")
    if run_manifest.get("protocol_json_sha256") != frozen_protocol["protocol_json_sha256"]:
        raise EvaluationBundleError(f"rank run manifest protocol JSON hash drifted: {run_manifest_path}")
    if run_manifest.get("protocol_md_sha256") != frozen_protocol["protocol_md_sha256"]:
        raise EvaluationBundleError(f"rank run manifest PROTOCOL.md hash drifted: {run_manifest_path}")
    expected_report_shas = {
        name: item["sha256"] for name, item in frozen_protocol["calibration_reports"].items()
    }
    if run_manifest.get("calibration_report_sha256") != expected_report_shas:
        raise EvaluationBundleError(f"rank run manifest calibration report hashes drifted: {run_manifest_path}")
    audit = run_manifest.get("asset_audit")
    if not isinstance(audit, Mapping) or not str(audit.get("status", "")).startswith("READY") or audit.get("server_control_deployed") is not True:
        raise EvaluationBundleError(f"rank run manifest does not prove audited server dispatch: {run_manifest_path}")
    marker_path = root / f"COMPLETED_A_RANK-{rank:04d}.json"
    marker = _read_json(marker_path)
    if marker.get("status") != "COMPLETED":
        raise EvaluationBundleError(f"rank marker is not completed: {marker_path}")
    if int(marker.get("rank", -1)) != rank or int(marker.get("world_size", -1)) != WORLD_SIZE:
        raise EvaluationBundleError(f"rank/world_size mismatch: {marker_path}")
    if marker.get("frozen_protocol") != dict(frozen_protocol):
        raise EvaluationBundleError(f"rank completion frozen protocol drifted: {marker_path}")
    if marker.get("protocol_git_commit") != frozen_protocol["protocol_git_commit"]:
        raise EvaluationBundleError(f"rank completion protocol commit drifted: {marker_path}")
    if marker.get("protocol_json_sha256") != frozen_protocol["protocol_json_sha256"]:
        raise EvaluationBundleError(f"rank completion protocol JSON hash drifted: {marker_path}")
    if marker.get("protocol_md_sha256") != frozen_protocol["protocol_md_sha256"]:
        raise EvaluationBundleError(f"rank completion PROTOCOL.md hash drifted: {marker_path}")
    if marker.get("calibration_report_sha256") != expected_report_shas:
        raise EvaluationBundleError(f"rank completion calibration report hashes drifted: {marker_path}")
    if run_id is not None and marker.get("run_id") not in (None, run_id):
        raise EvaluationBundleError(f"rank marker run id mismatch: {marker_path}")
    sums_path = root / f"SHA256SUMS_A_RANK-{rank:04d}"
    _verify_manifest_file(root, sums_path)
    listed = marker.get("families")
    if not isinstance(listed, list) or len(listed) != 20:
        raise EvaluationBundleError(f"rank marker must list exactly 20 families: {marker_path}")
    expected: list[tuple[str, int]] = []
    for task_index, task in enumerate(tasks):
        for family_index in range(FAMILIES_PER_TASK):
            flat = task_index * FAMILIES_PER_TASK + family_index
            if flat % WORLD_SIZE == rank:
                expected.append((task, family_index))
    if len(expected) != 20:
        raise EvaluationBundleError(f"internal shard assignment error for rank {rank}")
    outputs = []
    expected_paths = {
        str((root / f"rank-{rank:04d}" / task / f"family-{index:04d}").resolve())
        for task, index in expected
    }
    listed_paths = {
        str(Path(str(item.get("path"))).resolve())
        for item in listed
        if isinstance(item, Mapping) and item.get("path")
    }
    # AtomicFamilyWriter's immutable marker intentionally stores only the
    # local family id.  The task-qualified directory path is the unambiguous
    # identity at this aggregate layer.
    if listed_paths != expected_paths:
        raise EvaluationBundleError(f"rank family assignment mismatch: {marker_path}")
    for task, family_index in expected:
        outputs.append(_verify_family(root, task=task, family_index=family_index, rank=rank))
    return outputs


def _all_files(root: Path) -> list[Path]:
    paths = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        if path.name == "SHA256SUMS" and path.parent == root:
            continue
        if path.name.endswith(".tmp") or path.name.startswith("."):
            raise EvaluationBundleError(f"temporary/hidden file remains in completed bundle: {path}")
        paths.append(path)
    return paths


def _write_atomic(path: Path, data: bytes) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_top_level_manifest(root: Path) -> None:
    lines = []
    for path in _all_files(root):
        lines.append(f"{_sha256(path)}  {path.relative_to(root).as_posix()}")
    if not lines:
        raise EvaluationBundleError("cannot write an empty top-level SHA256SUMS")
    _write_atomic(root / "SHA256SUMS", ("\n".join(lines) + "\n").encode())


def verify_completed_bundle(
    root: Path,
    *,
    frozen_protocol: Mapping[str, Any] | None = None,
    frozen_protocol_path: Path = DEFAULT_PROTOCOL_PATH,
) -> Mapping[str, Any]:
    """Verify an already completed aggregate without mutating it."""

    if frozen_protocol is None:
        try:
            frozen_protocol = load_frozen_protocol(frozen_protocol_path)
        except FrozenProtocolError as exc:
            raise EvaluationBundleError(f"frozen protocol gate: {exc}") from exc
    result = _read_json(root / "COMPLETED_EVALUATION_RESULT.json")
    if result.get("status") != "COMPLETED" or result.get("marker_type") != "completed_stage_s_a_evaluation":
        raise EvaluationBundleError("top-level completion marker has an unexpected schema")
    if result.get("frozen_protocol") != dict(frozen_protocol):
        raise EvaluationBundleError("top-level completion frozen protocol drifted")
    if result.get("protocol_git_commit") != frozen_protocol["protocol_git_commit"]:
        raise EvaluationBundleError("top-level completion protocol commit drifted")
    if result.get("protocol_json_sha256") != frozen_protocol["protocol_json_sha256"]:
        raise EvaluationBundleError("top-level completion protocol JSON hash drifted")
    if result.get("protocol_md_sha256") != frozen_protocol["protocol_md_sha256"]:
        raise EvaluationBundleError("top-level completion PROTOCOL.md hash drifted")
    expected_report_shas = {
        name: item["sha256"] for name, item in frozen_protocol["calibration_reports"].items()
    }
    if result.get("calibration_report_sha256") != expected_report_shas:
        raise EvaluationBundleError("top-level completion calibration report hashes drifted")
    _verify_manifest_file(root, root / "SHA256SUMS")
    return result


def finalize(
    root: Path,
    *,
    run_id: str | None = None,
    job_id: str | None = None,
    source_commit: str | None = None,
    frozen_protocol_path: Path = DEFAULT_PROTOCOL_PATH,
) -> Mapping[str, Any]:
    root = root.resolve()
    try:
        frozen_protocol = load_frozen_protocol(frozen_protocol_path)
    except FrozenProtocolError as exc:
        raise EvaluationBundleError(f"frozen protocol gate: {exc}") from exc
    existing_result = root / "COMPLETED_EVALUATION_RESULT.json"
    if existing_result.exists() or (root / "SHA256SUMS").exists():
        if not existing_result.exists() or not (root / "SHA256SUMS").exists():
            raise EvaluationBundleError("partial top-level completion bundle is present")
        return verify_completed_bundle(root, frozen_protocol=frozen_protocol)
    for forbidden in ("FAILED_A_MAIN.json", "REFUSED_WINDOW.txt"):
        if (root / forbidden).exists():
            raise EvaluationBundleError(f"failed/refused run cannot be finalized: {root / forbidden}")
    tasks = tuple(select_published_tasks())
    pins = RoboTwinPins()
    rank_outputs: list[Mapping[str, Any]] = []
    for rank in range(WORLD_SIZE):
        rank_outputs.extend(
            _verify_rank(
                root,
                rank=rank,
                tasks=tasks,
                run_id=run_id,
                frozen_protocol=frozen_protocol,
            )
        )
    if len(rank_outputs) != 10 * FAMILIES_PER_TASK:
        raise EvaluationBundleError(f"expected 160 families, got {len(rank_outputs)}")
    total = sum(int(item["candidate_count"]) for item in rank_outputs)
    successes = sum(int(item["success_count"]) for item in rank_outputs)
    if total != EXPECTED_CANDIDATES:
        raise EvaluationBundleError(f"expected {EXPECTED_CANDIDATES} terminal candidates, got {total}")
    result = {
        "status": "COMPLETED",
        "marker_type": "completed_stage_s_a_evaluation",
        "protocol": "R142-FP-11 Stage-S substrate A",
        "run_id": run_id,
        "job_id": job_id,
        "source_commit": source_commit,
        "pins": pins.as_dict(),
        "frozen_protocol": dict(frozen_protocol),
        "protocol_git_commit": frozen_protocol["protocol_git_commit"],
        "protocol_json_sha256": frozen_protocol["protocol_json_sha256"],
        "protocol_md_sha256": frozen_protocol["protocol_md_sha256"],
        "calibration_report_sha256": {
            name: item["sha256"]
            for name, item in frozen_protocol["calibration_reports"].items()
        },
        "tasks": list(tasks),
        "world_size": WORLD_SIZE,
        "families_per_task": FAMILIES_PER_TASK,
        "candidate_budget": CANDIDATES_PER_FAMILY,
        "family_count": len(rank_outputs),
        "terminal_candidate_count": total,
        "successful_candidate_count": successes,
        "success_rate": successes / total,
        "server_client_ownership": "one_server_one_client_one_gpu_one_port",
        "synthetic_rollouts": False,
        "expert_trajectory": False,
        "termination": "official eval_success or step_lim",
        "replay_gate": "restore -> same action -> next-state <= 1e-9",
        "ranks": [
            {
                "rank": rank,
                "completion": str(root / f"COMPLETED_A_RANK-{rank:04d}.json"),
                "sha256sums": str(root / f"SHA256SUMS_A_RANK-{rank:04d}"),
                "family_count": sum(1 for row in rank_outputs if int(row["rank"]) == rank),
            }
            for rank in range(WORLD_SIZE)
        ],
    }
    data = (json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    _write_atomic(existing_result, data)
    _write_top_level_manifest(root)
    verify_completed_bundle(root, frozen_protocol=frozen_protocol)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--job-id")
    parser.add_argument("--source-commit")
    parser.add_argument(
        "--frozen-protocol",
        type=Path,
        default=DEFAULT_PROTOCOL_PATH,
        help="stable CPFS Stage-S protocol authority",
    )
    return parser


if __name__ == "__main__":
    try:
        parsed = build_parser().parse_args()
        result = finalize(
            parsed.output_root,
            run_id=parsed.run_id or os.environ.get("PAI_STAGE_S_RUN_ID"),
            job_id=parsed.job_id or os.environ.get("PAI_TASK_JOB_ID"),
            source_commit=parsed.source_commit or os.environ.get("STAGE_S_SOURCE_COMMIT"),
            frozen_protocol_path=parsed.frozen_protocol,
        )
    except EvaluationBundleError as exc:
        print(json.dumps({"status": "BLOCKED_INCOMPLETE_EVALUATION", "error": str(exc)}))
        raise SystemExit(2)
    print(json.dumps(result, indent=2, ensure_ascii=False))
