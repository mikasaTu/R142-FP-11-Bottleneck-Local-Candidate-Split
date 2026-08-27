#!/usr/bin/env python3
"""Outcome-blind, fail-closed merge of the two Phase-1R natural shards.

This utility is deliberately independent of the natural-run collector.  It
only consumes terminally sealed shard directories, the frozen rank mapping,
and the pre-registered selection bundle.  It never reads candidate outcomes
to choose a task: the mapping in ``stage_r_phase1r_shards.json`` is the sole
authority for the output order and membership.

The PAI launcher writes a shard completion bundle before the PAI job returns.
The controller (or a post-terminal readback step) must then write
``PAI_TERMINAL_COMPLETION.json`` and regenerate the shard's ``SHA256SUMS``.
The terminal sidecar intentionally contains no hash of the enclosing
``SHA256SUMS`` file: including that hash would create a self-referential
digest because the sidecar itself is part of the exhaustive seal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


PROTOCOL_ID = "r142-stage-r-phase1r-human-override-v1"
PROTOCOL_SHA256 = "7e4de68cba5c0fdb288ee25d81f30b72d65483753973f06f44b395e3db0b9cb4"
SHARDS_SHA256 = "f0e4b137d5f5b39737f671dc273428fdd5b646327863f03a542c2c7c5e2977d6"
SELECTION_MANIFEST_SHA256 = "082f1f28f6ed8bddb1ed2ef87a3b848ac3daccec5c333f7f2cf1c4ef5d988231"
AUTHORITY_MANIFEST_SHA256 = "3d5a37ec8a7e2c0dfd0c808ad59553c43a13c846b90f99c1afaa3529a072469c"
CALIBRATION_SHA256 = "f8a6486a96b9fc02071c391c4971ac2251d5c5e89dbb405f9d51dcd44fbfad6a"
SHARD_A_RANKS = tuple(range(0, 8))
SHARD_B_RANKS = tuple(range(8, 16))
SHARD_A_NAME = "A"
SHARD_B_NAME = "B"
EXPECTED_OWNER = (2254, 2254)
STREAMS = ("calibration", "heldout")
TASK_RE = re.compile(r"^(libero_(?:spatial|object|goal|10)_task(?:0[0-9]))$")
SHA_LINE_RE = re.compile(r"^([0-9a-f]{64})  (.+)$")
NPZ_MEMBERS = {
    "lengths.npy",
    "offsets.npy",
    "actions.npy",
    "states.npy",
    "progress.npy",
    "success.npy",
    "branch_id.npy",
    "generation_step.npy",
    "branch_seed.npy",
    "policy_forwards.npy",
    "policy_batches.npy",
    "environment_steps.npy",
}


class ContractError(RuntimeError):
    """Raised for any malformed, partial, or non-authoritative input."""


def fail(message: str) -> "NoReturn":
    raise ContractError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        fail(f"missing regular JSON file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - message is evidence
        fail(f"invalid JSON {path}: {type(exc).__name__}: {exc}")
    if not isinstance(value, dict):
        fail(f"JSON object required: {path}")
    return value


def owner(path: Path) -> tuple[int, int]:
    info = path.stat()
    return int(info.st_uid), int(info.st_gid)


def require_owner(path: Path, expected: tuple[int, int] = EXPECTED_OWNER) -> None:
    if owner(path) != expected:
        fail(f"owner mismatch for {path}: {owner(path)} != {expected}")


def require_sha(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        fail(f"SHA mismatch for {path}: {actual} != {expected}")


def regular_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink() and not path.name.endswith(".tmp")
    )


def safe_relative(name: str) -> PurePosixPath:
    if not name or "\\" in name:
        fail(f"unsafe checksum path: {name!r}")
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        fail(f"unsafe checksum path: {name!r}")
    return relative


def validate_exhaustive_sums(
    root: Path,
    sums_name: str = "SHA256SUMS",
    *,
    expected_owner: tuple[int, int] = EXPECTED_OWNER,
) -> dict[str, str]:
    """Validate every digest and require a complete regular-file inventory."""

    sums_path = root / sums_name
    if not sums_path.is_file() or sums_path.is_symlink():
        fail(f"missing checksum seal: {sums_path}")
    require_owner(sums_path, expected_owner)
    observed: dict[str, str] = {}
    for line_number, line in enumerate(sums_path.read_text(encoding="utf-8").splitlines(), 1):
        match = SHA_LINE_RE.fullmatch(line)
        if match is None:
            fail(f"malformed {sums_name} line {line_number}")
        digest, name = match.groups()
        relative = safe_relative(name)
        key = relative.as_posix()
        if key == sums_name or key in observed:
            fail(f"duplicate/self checksum entry in {sums_path}: {key}")
        target = root.joinpath(*relative.parts)
        if not target.is_file() or target.is_symlink():
            fail(f"checksum target missing or symlink: {target}")
        require_owner(target, expected_owner)
        actual = sha256_file(target)
        if actual != digest:
            fail(f"checksum mismatch: {target}: {actual} != {digest}")
        observed[key] = digest
    actual_names = {
        path.relative_to(root).as_posix()
        for path in regular_files(root)
        if path.name != sums_name
    }
    if set(observed) != actual_names:
        missing = sorted(actual_names - set(observed))
        extra = sorted(set(observed) - actual_names)
        fail(f"{sums_name} is not exhaustive; missing={missing[:5]} extra={extra[:5]}")
    return observed


def require_safe_dir(path: Path, label: str) -> Path:
    if path.exists() and path.is_symlink():
        fail(f"{label} is a symlink: {path}")
    if not path.is_dir():
        fail(f"{label} is not a directory: {path}")
    resolved = path.resolve()
    if resolved != path.absolute().resolve():
        fail(f"{label} resolves unexpectedly: {path}")
    return resolved


def parse_task_name(name: str) -> tuple[str, int]:
    match = TASK_RE.fullmatch(name)
    if match is None:
        fail(f"invalid frozen task name: {name}")
    suite, task_number = name.rsplit("_task", 1)
    return suite, int(task_number)


def validate_config(shards_path: Path) -> dict[str, Any]:
    require_sha(shards_path, SHARDS_SHA256)
    config = read_json(shards_path)
    if config.get("schema_version") != 1 or config.get("protocol_id") != PROTOCOL_ID:
        fail("shard config schema/protocol mismatch")
    if config.get("global_world_size") != 16 or config.get("active_job_limit") != 2:
        fail("shard config world-size/active-job contract mismatch")
    if config.get("gpu_per_job") != 8:
        fail("shard config GPU contract mismatch")
    if config.get("streams") != list(STREAMS):
        fail("shard config stream mismatch")
    if config.get("phase0r_authority_manifest_sha256") != AUTHORITY_MANIFEST_SHA256:
        fail("shard config authority SHA mismatch")
    shards = config.get("shards")
    if not isinstance(shards, dict) or set(shards) != {SHARD_A_NAME, SHARD_B_NAME}:
        fail("shard config must contain exactly A and B")
    for shard, expected_ranks in ((SHARD_A_NAME, SHARD_A_RANKS), (SHARD_B_NAME, SHARD_B_RANKS)):
        entry = shards[shard]
        if entry.get("global_ranks") != list(expected_ranks):
            fail(f"{shard} global rank range mismatch")
        rank_tasks = entry.get("rank_tasks")
        if not isinstance(rank_tasks, dict) or {int(key) for key in rank_tasks} != set(expected_ranks):
            fail(f"{shard} rank task keys mismatch")
        for rank in expected_ranks:
            names = rank_tasks.get(str(rank))
            if not isinstance(names, list) or not names:
                fail(f"{shard} rank {rank} has no task mapping")
            for task_name in names:
                parse_task_name(str(task_name))
    return config


def validate_selection(selection_root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest_path = selection_root / "SELECTION_MANIFEST.json"
    require_sha(manifest_path, SELECTION_MANIFEST_SHA256)
    manifest = read_json(manifest_path)
    if manifest.get("marker_type") != "selection" or manifest.get("protocol_id") != PROTOCOL_ID:
        fail("selection manifest schema/protocol mismatch")
    if manifest.get("task_count") != 40 or manifest.get("total_selected") != 480:
        fail("selection manifest count mismatch")
    validate_exhaustive_sums(selection_root, "SELECTION_SHA256SUMS")
    rows = manifest.get("tasks")
    if not isinstance(rows, list) or len(rows) != 40:
        fail("selection manifest task rows mismatch")
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            fail("malformed selection manifest row")
        path_name = row.get("path")
        suite = row.get("suite")
        task_id = row.get("task_id")
        if not isinstance(path_name, str) or not isinstance(suite, str) or not isinstance(task_id, int):
            fail("selection manifest identity fields missing")
        expected_name = f"{suite}_task{task_id:02d}.json"
        if path_name != expected_name or expected_name in indexed:
            fail(f"selection manifest duplicate/wrong path: {path_name}")
        selection_path = selection_root / path_name
        require_sha(selection_path, str(row.get("sha256", "")))
        selection = read_json(selection_path)
        if selection.get("protocol_id") != PROTOCOL_ID:
            fail(f"selection protocol mismatch: {selection_path}")
        selected = selection.get("selected")
        if not isinstance(selected, list) or len(selected) != 12:
            fail(f"selection count mismatch: {selection_path}")
        indices: dict[int, str] = {}
        for item in selected:
            if not isinstance(item, dict) or not isinstance(item.get("selection_index"), int):
                fail(f"malformed selected identity: {selection_path}")
            index = int(item["selection_index"])
            episode = item.get("episode")
            if index in indices or not isinstance(episode, str) or not episode:
                fail(f"duplicate/empty selected episode identity: {selection_path}")
            indices[index] = episode
        if set(indices) != set(range(12)):
            fail(f"selection indices must be 0..11: {selection_path}")
        indexed[expected_name] = {
            "path": selection_path,
            "suite": suite,
            "task_id": task_id,
            "episodes": indices,
            "sha256": sha256_file(selection_path),
        }
    return manifest, indexed


def validate_terminal_sidecar(
    run_root: Path,
    shard: str,
    source_commit: str,
    expected_job_id: str | None,
) -> dict[str, Any]:
    path = run_root / "PAI_TERMINAL_COMPLETION.json"
    terminal = read_json(path)
    required = {
        "schema_version",
        "marker_type",
        "shard",
        "run_id",
        "job_id",
        "terminal_status",
        "completion_marker",
        "completion_marker_sha256",
        "sha256sums",
        "uid",
        "gid",
        "source_commit",
    }
    if not required.issubset(terminal):
        fail(f"terminal completion fields missing: {path}")
    if terminal.get("schema_version") != 1 or terminal.get("marker_type") != "pai_terminal_completion":
        fail(f"terminal completion schema mismatch: {path}")
    if terminal.get("shard") != shard or terminal.get("run_id") != run_root.name:
        fail(f"terminal completion identity mismatch: {path}")
    job_id = terminal.get("job_id")
    if not isinstance(job_id, str) or not job_id or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}", job_id):
        fail(f"invalid PAI job id: {path}")
    if expected_job_id is not None and job_id != expected_job_id:
        fail(f"unexpected PAI job id for {shard}: {job_id} != {expected_job_id}")
    if terminal.get("terminal_status") != "Succeeded":
        fail(f"PAI terminal status is not Succeeded for {shard}: {terminal.get('terminal_status')!r}")
    if terminal.get("completion_marker") != "COMPLETED_EVALUATION_RESULT.json" or terminal.get("sha256sums") != "SHA256SUMS":
        fail(f"terminal completion file names mismatch: {path}")
    if terminal.get("source_commit") != source_commit:
        fail(f"terminal source commit mismatch: {path}")
    if (terminal.get("uid"), terminal.get("gid")) != EXPECTED_OWNER:
        fail(f"terminal owner mismatch: {path}")
    completion_path = run_root / "COMPLETED_EVALUATION_RESULT.json"
    if terminal.get("completion_marker_sha256") != sha256_file(completion_path):
        fail(f"terminal completion marker SHA mismatch: {path}")
    require_owner(path)
    return {
        "shard": shard,
        "run_id": run_root.name,
        "job_id": job_id,
        "terminal_status": "Succeeded",
        "completion_marker_sha256": sha256_file(completion_path),
        "sha256sums_sha256": sha256_file(run_root / "SHA256SUMS"),
        "terminal_sidecar_sha256": sha256_file(path),
        "uid": 2254,
        "gid": 2254,
    }


def expected_cells(selection: dict[str, Any]) -> list[tuple[int, str, int, str]]:
    result: list[tuple[int, str, int, str]] = []
    for index, parent in sorted(selection["episodes"].items()):
        for location in range(10):
            for stream in STREAMS:
                result.append((index, parent, location, stream))
    return result


def validate_npz(path: Path) -> None:
    try:
        with zipfile.ZipFile(path, "r") as archive:
            members = archive.namelist()
            names = set(members)
    except Exception as exc:
        fail(f"invalid NPZ {path}: {type(exc).__name__}: {exc}")
    if len(members) != len(names):
        fail(f"NPZ has duplicate members {path}")
    if not NPZ_MEMBERS.issubset(names):
        fail(f"NPZ missing arrays {path}: {sorted(NPZ_MEMBERS - names)}")
    unsafe = [name for name in names if not name.endswith(".npy") or "/" in name or "\\" in name or name.startswith(".")]
    if unsafe:
        fail(f"NPZ has unexpected members {path}: {unsafe[:5]}")


def marker_digest(cell_markers: Iterable[tuple[str, Path]]) -> str:
    digest = hashlib.sha256()
    for relative, marker_path in sorted(cell_markers, key=lambda item: item[0]):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(marker_path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def validate_cell(
    cell_root: Path,
    *,
    suite: str,
    task_id: int,
    parent_id: str,
    selection_index: int,
    location: int,
    stream: str,
) -> Path:
    if not cell_root.is_dir() or cell_root.is_symlink():
        fail(f"missing natural cell: {cell_root}")
    required = [cell_root / name for name in ("cell.npz", "metadata.json", "COMPLETED_CELL.json", "SHA256SUMS")]
    if any(not path.is_file() or path.is_symlink() for path in required):
        fail(f"incomplete natural cell: {cell_root}")
    for path in required:
        require_owner(path)
    metadata = read_json(cell_root / "metadata.json")
    marker = read_json(cell_root / "COMPLETED_CELL.json")
    if metadata.get("schema_version") != 1 or metadata.get("protocol_id") != PROTOCOL_ID:
        fail(f"cell metadata protocol/schema mismatch: {cell_root}")
    identity = {
        "suite": suite,
        "task_id": task_id,
        "parent_id": parent_id,
        "selection_index": selection_index,
        "location_index": location,
        "stream": stream,
    }
    for key, value in identity.items():
        if metadata.get(key) != value or marker.get("cell", {}).get(key) != value:
            fail(f"cell identity mismatch {key}: {cell_root}")
    if metadata.get("descendant_count") != 16 or metadata.get("data_file") != "cell.npz":
        fail(f"cell descendant/data contract mismatch: {cell_root}")
    npz_sha = sha256_file(cell_root / "cell.npz")
    metadata_sha = sha256_file(cell_root / "metadata.json")
    if metadata.get("data_sha256") != npz_sha:
        fail(f"cell metadata data SHA mismatch: {cell_root}")
    if marker.get("npz") != {"path": "cell.npz", "sha256": npz_sha}:
        fail(f"cell NPZ marker mismatch: {cell_root}")
    if marker.get("metadata") != {"path": "metadata.json", "sha256": metadata_sha}:
        fail(f"cell metadata marker mismatch: {cell_root}")
    if (metadata.get("owner_uid"), metadata.get("owner_gid")) != EXPECTED_OWNER:
        fail(f"cell metadata owner mismatch: {cell_root}")
    validate_npz(cell_root / "cell.npz")
    expected_lines = [f"{npz_sha}  cell.npz", f"{metadata_sha}  metadata.json"]
    observed_lines = (cell_root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    if len(observed_lines) != 2 or sorted(observed_lines) != sorted(expected_lines):
        fail(f"cell checksum mismatch: {cell_root}")
    return cell_root / "COMPLETED_CELL.json"


def validate_task(
    run_root: Path,
    task_name: str,
    selection: dict[str, Any],
    selection_sha: str,
) -> dict[str, Any]:
    suite, task_id = parse_task_name(task_name)
    task_root = run_root / "natural" / suite / f"task{task_id:02d}"
    if not task_root.is_dir() or task_root.is_symlink():
        fail(f"missing mapped task directory: {task_root}")
    marker_path = task_root / "COMPLETED_TASK.json"
    require_owner(marker_path)
    marker = read_json(marker_path)
    if marker.get("schema_version") != 1 or marker.get("protocol_id") != PROTOCOL_ID or marker.get("marker_type") != "task":
        fail(f"task marker protocol/type mismatch: {marker_path}")
    expected = {
        "suite": suite,
        "task_id": task_id,
        "selection_manifest": selection["path"].name,
        "selection_manifest_sha256": selection_sha,
        "streams": list(STREAMS),
        "selected_episode_count": 12,
        "locations_per_episode": 10,
        "descendants_per_cell": 16,
        "completed_cells": 240,
        "owner_required": [2254, 2254],
        "checkpoint": "TASK_COMPLETE",
    }
    for key, value in expected.items():
        if marker.get(key) != value:
            fail(f"task marker mismatch {key}: {marker_path}")
    cell_markers: list[tuple[str, Path]] = []
    for index, parent, location, stream in expected_cells(selection):
        cell = task_root / f"episode{index:02d}" / f"location{location:02d}" / stream
        marker_file = validate_cell(
            cell,
            suite=suite,
            task_id=task_id,
            parent_id=parent,
            selection_index=index,
            location=location,
            stream=stream,
        )
        cell_markers.append((f"episode{index:02d}/location{location:02d}/{stream}", marker_file))
    if marker.get("cell_marker_sha256") != marker_digest(cell_markers):
        fail(f"task cell-marker digest mismatch: {marker_path}")
    return {
        "task_name": task_name,
        "suite": suite,
        "task_id": task_id,
        "selection": selection["path"].name,
        "selection_sha256": selection_sha,
        "completed_task_sha256": sha256_file(marker_path),
        "natural_relative_path": f"natural/{suite}/task{task_id:02d}",
        "completed_cells": 240,
    }


def validate_shard(
    run_root: Path,
    shard: str,
    mapping_names: list[str],
    selection_index: dict[str, dict[str, Any]],
    source_commit: str,
    expected_job_id: str | None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, str]]:
    run_root = require_safe_dir(run_root, f"shard {shard} run")
    prefix = f"r142-stage-r-phase1r-shard-{shard.lower()}-"
    if not run_root.name.startswith(prefix):
        fail(f"shard {shard} run id is not frozen-prefix: {run_root.name}")
    if owner(run_root) != EXPECTED_OWNER:
        fail(f"shard {shard} run owner mismatch: {run_root}")
    sums = validate_exhaustive_sums(run_root)
    terminal = validate_terminal_sidecar(run_root, shard, source_commit, expected_job_id)
    complete_path = run_root / "COMPLETED_EVALUATION_RESULT.json"
    complete = read_json(complete_path)
    if complete.get("success_gate") != "persisted_completed_evaluation_result":
        fail(f"shard {shard} completion gate mismatch")
    for key, value in {
        "protocol_id": PROTOCOL_ID,
        "source_commit": source_commit,
        "protocol_sha256": PROTOCOL_SHA256,
        "shards_sha256": SHARDS_SHA256,
        "selection_manifest_sha256": SELECTION_MANIFEST_SHA256,
        "calibration_sha256": CALIBRATION_SHA256,
        "authority_manifest_sha256": AUTHORITY_MANIFEST_SHA256,
        "shard": shard,
        "phase1_authorized": False,
        "analysis_performed": False,
        "checkpoint": "CHECKPOINT_1_PENDING_GLOBAL_MERGE",
        "decision": "NATURAL_SHARD_COMPLETE_NO_UNBLINDING",
        "streams": list(STREAMS),
    }.items():
        if complete.get(key) != value:
            fail(f"shard {shard} completion mismatch {key}")
    if (complete.get("uid"), complete.get("gid")) != EXPECTED_OWNER:
        fail(f"shard {shard} completion owner mismatch")
    shard_marker_path = run_root / "SHARD_NATURAL_COMPLETE.json"
    shard_marker = read_json(shard_marker_path)
    expected_task_count = len(mapping_names)
    if shard_marker.get("marker_type") != "natural_phase1r_shard" or shard_marker.get("protocol_id") != PROTOCOL_ID:
        fail(f"shard {shard} marker type/protocol mismatch")
    for key, value in {
        "source_commit": source_commit,
        "protocol_sha256": PROTOCOL_SHA256,
        "shards_sha256": SHARDS_SHA256,
        "selection_manifest_sha256": SELECTION_MANIFEST_SHA256,
        "calibration_sha256": CALIBRATION_SHA256,
        "authority_manifest_sha256": AUTHORITY_MANIFEST_SHA256,
        "shard": shard,
        "task_names": mapping_names,
        "task_count": expected_task_count,
        "natural_cells": expected_task_count * 240,
        "natural_descendants_per_stream": expected_task_count * 12 * 10 * 16,
        "streams": list(STREAMS),
        "checkpoint": "SHARD_NATURAL_COMPLETE",
    }.items():
        if shard_marker.get(key) != value:
            fail(f"shard {shard} marker mismatch {key}")
    if (shard_marker.get("uid"), shard_marker.get("gid")) != EXPECTED_OWNER:
        fail(f"shard {shard} marker owner mismatch")
    tasks: dict[str, dict[str, Any]] = {}
    for task_name in mapping_names:
        selection_name = f"{task_name}.json"
        selection = selection_index.get(selection_name)
        if selection is None:
            fail(f"mapped task has no frozen selection: {task_name}")
        tasks[task_name] = validate_task(run_root, task_name, selection, selection["sha256"])
    # The top-level completion seal was checked before task validation; repeat
    # its inventory check after all reads to catch any file appearing mid-run.
    sums_after = validate_exhaustive_sums(run_root)
    if sums_after != sums:
        fail(f"shard {shard} checksum inventory changed during validation")
    return terminal, tasks, sums_after


def link_or_copy_tree(source: Path, destination: Path) -> None:
    """Copy a mapped task tree into an inode-independent snapshot."""

    if destination.exists() or destination.is_symlink():
        fail(f"destination already exists: {destination}")
    destination.mkdir(parents=True, mode=0o750)
    os.chmod(destination, stat.S_IMODE(source.stat().st_mode))
    require_owner(source)
    for entry in sorted(source.iterdir(), key=lambda path: path.name):
        target = destination / entry.name
        if entry.is_symlink():
            fail(f"mapped task contains symlink: {entry}")
        if entry.is_dir():
            link_or_copy_tree(entry, target)
            continue
        if not entry.is_file():
            fail(f"mapped task contains unsupported entry: {entry}")
        shutil.copy2(entry, target)
        require_sha(target, sha256_file(entry))
        require_owner(target)


def copy_provenance(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, mode=0o750)
    for name in (
        "SHARD_NATURAL_COMPLETE.json",
        "COMPLETED_EVALUATION_RESULT.json",
        "PAI_TERMINAL_COMPLETION.json",
        "SHA256SUMS",
    ):
        source_path = source / name
        target = destination / name
        shutil.copy2(source_path, target)
        require_sha(target, sha256_file(source_path))
        require_owner(target)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(path, 0o640)
    require_owner(path)


def write_exhaustive_sums(root: Path) -> None:
    lines = []
    for path in regular_files(root):
        if path.name == "SHA256SUMS":
            continue
        lines.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n")
    temporary = root / ".SHA256SUMS.tmp"
    temporary.write_text("".join(lines), encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(root / "SHA256SUMS")
    require_owner(root / "SHA256SUMS")


def merge(args: argparse.Namespace) -> dict[str, Any]:
    if os.getuid() != EXPECTED_OWNER[0] or os.getgid() != EXPECTED_OWNER[1]:
        fail(f"merge must run as UID:GID 2254:2254, got {os.getuid()}:{os.getgid()}")
    source_commit = args.source_commit
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit) or not set(source_commit) - {"0"}:
        fail("--source-commit must be one exact nonzero 40-hex commit")
    shards_path = args.shards_config.resolve()
    selection_root = require_safe_dir(args.selection_root, "selection root")
    manifest, selection_index = validate_selection(selection_root)
    config = validate_config(shards_path)
    mapping: dict[str, list[str]] = {}
    global_rank_for_task: dict[str, int] = {}
    for shard in (SHARD_A_NAME, SHARD_B_NAME):
        rank_tasks = config["shards"][shard]["rank_tasks"]
        names = []
        for rank in config["shards"][shard]["global_ranks"]:
            for name in rank_tasks[str(rank)]:
                names.append(name)
                global_rank_for_task[name] = int(rank)
        if len(names) != len(set(names)):
            fail(f"duplicate task inside shard {shard}")
        mapping[shard] = names
    if set(mapping[SHARD_A_NAME]) & set(mapping[SHARD_B_NAME]):
        fail("A/B shard task overlap")
    if len(set(mapping[SHARD_A_NAME] + mapping[SHARD_B_NAME])) != 40:
        fail("frozen shard mapping does not cover exactly 40 tasks")
    for name in mapping[SHARD_A_NAME] + mapping[SHARD_B_NAME]:
        if f"{name}.json" not in selection_index:
            fail(f"frozen mapping task lacks selection: {name}")
    output = args.output
    if output.exists() or output.is_symlink():
        fail(f"output must be a fresh non-existing path: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o750)
    require_owner(output)
    terminal_a, tasks_a, _ = validate_shard(args.shard_a, SHARD_A_NAME, mapping[SHARD_A_NAME], selection_index, source_commit, args.expected_job_id_a)
    terminal_b, tasks_b, _ = validate_shard(args.shard_b, SHARD_B_NAME, mapping[SHARD_B_NAME], selection_index, source_commit, args.expected_job_id_b)
    tasks = {**tasks_a, **tasks_b}
    if len(tasks) != 40:
        fail(f"validated task count is not 40: {len(tasks)}")
    for shard, source, names in (
        (SHARD_A_NAME, args.shard_a, mapping[SHARD_A_NAME]),
        (SHARD_B_NAME, args.shard_b, mapping[SHARD_B_NAME]),
    ):
        for name in names:
            suite, task_id = parse_task_name(name)
            link_or_copy_tree(
                source / "natural" / suite / f"task{task_id:02d}",
                output / "natural" / suite / f"task{task_id:02d}",
            )
        copy_provenance(source, output / "provenance" / f"shard_{shard.lower()}")
    authority = {
        "schema_version": 1,
        "marker_type": "phase1r_natural_authority",
        "protocol_id": PROTOCOL_ID,
        "source_commit": source_commit,
        "protocol_sha256": PROTOCOL_SHA256,
        "shards_sha256": SHARDS_SHA256,
        "selection_manifest_sha256": SELECTION_MANIFEST_SHA256,
        "selection_total_selected": manifest["total_selected"],
        "calibration_sha256": CALIBRATION_SHA256,
        "phase0r_authority_manifest_sha256": AUTHORITY_MANIFEST_SHA256,
        "outcome_blind": True,
        "selection_by_outcome": False,
        "authority_rule": {"A": "global_ranks_0_7", "B": "global_ranks_8_15"},
        "terminal_completions": [terminal_a, terminal_b],
        "tasks": [
            tasks[name]
            | {
                "frozen_index": index,
                "shard": "A" if index < len(mapping["A"]) else "B",
                "global_rank": global_rank_for_task[name],
            }
            for index, name in enumerate(mapping["A"] + mapping["B"])
        ],
        "task_count": 40,
        "natural_cells": 40 * 240,
        "natural_descendants_per_stream": 40 * 12 * 10 * 16,
        "checkpoint": "CHECKPOINT_1_PENDING_GLOBAL_ANALYSIS",
        "phase1_authorized": False,
        "analysis_performed": False,
    }
    write_json(output / "AUTHORITY_MANIFEST.json", authority)
    completion = {
        "schema_version": 1,
        "marker_type": "natural_phase1r_authority_merge",
        "success_gate": "persisted_completed_natural_merge",
        "decision": "NATURAL_SHARDS_MERGED_NO_UNBLINDING",
        "checkpoint": "CHECKPOINT_1_PENDING_GLOBAL_ANALYSIS",
        "phase1_authorized": False,
        "analysis_performed": False,
        "protocol_id": PROTOCOL_ID,
        "source_commit": source_commit,
        "protocol_sha256": PROTOCOL_SHA256,
        "shards_sha256": SHARDS_SHA256,
        "selection_manifest_sha256": SELECTION_MANIFEST_SHA256,
        "calibration_sha256": CALIBRATION_SHA256,
        "phase0r_authority_manifest_sha256": AUTHORITY_MANIFEST_SHA256,
        "authority_manifest_sha256": sha256_file(output / "AUTHORITY_MANIFEST.json"),
        "outcome_blind": True,
        "task_count": 40,
        "natural_cells": 40 * 240,
        "natural_descendants_per_stream": 40 * 12 * 10 * 16,
        "streams": list(STREAMS),
        "shards": [terminal_a, terminal_b],
        "owner_required": [2254, 2254],
    }
    write_json(output / "COMPLETED_NATURAL_MERGE.json", completion)
    write_exhaustive_sums(output)
    validate_exhaustive_sums(output)
    completion["sha256sums_sha256"] = sha256_file(output / "SHA256SUMS")
    # Adding the SHA to the completion marker changes the seal, so rewrite and
    # reseal once.  The marker is deliberately not referenced by its own hash.
    write_json(output / "COMPLETED_NATURAL_MERGE.json", completion)
    write_exhaustive_sums(output)
    validate_exhaustive_sums(output)
    if read_json(output / "COMPLETED_NATURAL_MERGE.json").get("authority_manifest_sha256") != sha256_file(output / "AUTHORITY_MANIFEST.json"):
        fail("merged completion authority SHA mismatch")
    print("STAGE_R_PHASE1R_NATURAL_MERGE_COMPLETE_VALIDATED")
    return completion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-a", type=Path, required=True, help="exact completed shard A run directory")
    parser.add_argument("--shard-b", type=Path, required=True, help="exact completed shard B run directory")
    parser.add_argument("--shards-config", type=Path, required=True, help="frozen stage_r_phase1r_shards.json")
    parser.add_argument("--selection-root", type=Path, required=True, help="frozen selection directory")
    parser.add_argument("--source-commit", required=True, help="exact source commit used by both natural shards")
    parser.add_argument("--output", type=Path, required=True, help="fresh merged output directory")
    parser.add_argument("--expected-job-id-a", required=True, help="exact PAI job id for shard A")
    parser.add_argument("--expected-job-id-b", required=True, help="exact PAI job id for shard B")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        merge(args)
    except ContractError as exc:
        print(f"STAGE_R_PHASE1R_NATURAL_MERGE_FAILED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
