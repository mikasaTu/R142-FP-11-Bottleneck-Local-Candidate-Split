"""Fail-closed total Stage-S analysis and publication.

This module is the single boundary used after A/B/C terminal screens.  It does
not infer a gate from an upstream summary: it verifies completion markers,
SHA256 manifests, terminal candidate records, genealogy, snapshots and
compute accounting before passing raw rows to the pure gate functions.  Gate
calculation is deliberately independent for all three arms, so a failed S1
or malformed *other* arm never short-circuits the remaining analysis.  A
malformed artifact produces a ``PIPELINE_INVALID`` decision while the
successfully verified arms still receive a complete S1--S5 result.

The public entry point is :func:`analyze_stage_s`.  ``arms`` accepts either a
mapping with ``records``/``probes``/``extended_rollouts`` (useful for a
verified in-memory handoff) or paths to the persisted main, S4 and S5
bundles.  In-memory handoffs must carry explicit ``artifact_verification``
flags; this prevents a convenient list of booleans from becoming scientific
evidence without the same artifact audit as a disk bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .analysis import (
    DECISION_CODES,
    S3_SUBSTRATE_WORKSPACE_SCALES,
    compute_s1,
    compute_s2,
    compute_s3_production,
    compute_s4_from_protocol,
    compute_s5,
    decide_stage_s,
)
from .integrity import verify_completion_bundle, write_completion
from .s45_runtime import PROTOCOL_ID, ProtocolAuthority, S45Error, load_s4_probes


TOTAL_ANALYSIS_SCHEMA = "r142-stage-s-total-analysis-v1"
TOTAL_RESULT_FILE = "STAGE_S_TOTAL_ANALYSIS.json"
TOTAL_COMPLETION_FILE = "COMPLETED_EVALUATION_RESULT.json"
EXPECTED_SUBSTRATES = ("A", "B", "C")
EXPECTED_WORLD_SIZE = 8
EXPECTED_FAMILY_COUNT = 160
EXPECTED_CANDIDATES = 32
EXPECTED_S4_GRID_COUNT = 9
EXPECTED_S4_HELDOUT_COUNT = 8
EXPECTED_S4_SEARCH_COUNT = 4
EXPECTED_S4_BOOTSTRAP_SEED = 14211
EXPECTED_S4_BOOTSTRAP_REPLICATES = 10_000
PROTOCOL_KEYS = (
    "protocol_id",
    "protocol_authority_path",
    "protocol_authority_sha256",
    "protocol_git_commit",
)


class TotalAnalysisError(RuntimeError):
    """The total Stage-S analysis input is incomplete or inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise TotalAnalysisError(f"cannot hash artifact: {path}") from exc
    return digest.hexdigest()


def _strict_path(value: str | Path, *, label: str, directory: bool | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_symlink():
        raise TotalAnalysisError(f"{label} is symlinked: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise TotalAnalysisError(f"{label} is missing: {path}") from exc
    if resolved.is_symlink():
        raise TotalAnalysisError(f"{label} resolves through a symlink: {resolved}")
    if directory is True and not resolved.is_dir():
        raise TotalAnalysisError(f"{label} is not a directory: {resolved}")
    if directory is False and not resolved.is_file():
        raise TotalAnalysisError(f"{label} is not a regular file: {resolved}")
    return resolved


def _read_json(path: str | Path, *, label: str) -> Mapping[str, Any]:
    target = _strict_path(path, label=label, directory=False)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise TotalAnalysisError(f"{label} is invalid JSON: {target}") from exc
    if not isinstance(value, Mapping):
        raise TotalAnalysisError(f"{label} must be a JSON object: {target}")
    return value


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_bytes(path, (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))


def _safe_relative(root: Path, value: str) -> Path:
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or str(relative) in ("", "."):
        raise TotalAnalysisError(f"unsafe checksum path {value!r} under {root}")
    return root.joinpath(*relative.parts)


def _verify_manifest(root: Path, *, required: bool = True) -> dict[str, str]:
    manifest = root / "SHA256SUMS"
    if not manifest.is_file() or manifest.is_symlink():
        if required:
            raise TotalAnalysisError(f"SHA256SUMS is missing or symlinked: {manifest}")
        return {}
    entries: dict[str, str] = {}
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise TotalAnalysisError(f"cannot read SHA256SUMS: {manifest}") from exc
    for line_number, line in enumerate(lines, 1):
        fields = line.split()
        if len(fields) != 2 or not all(character in "0123456789abcdefABCDEF" for character in fields[0]) or len(fields[0]) != 64:
            raise TotalAnalysisError(f"malformed SHA256SUMS line {line_number}: {manifest}")
        relative = PurePosixPath(fields[1])
        if relative.is_absolute() or ".." in relative.parts or fields[1] in entries:
            raise TotalAnalysisError(f"unsafe or duplicate SHA256SUMS entry: {fields[1]}")
        if fields[1] == "SHA256SUMS":
            raise TotalAnalysisError(f"SHA256SUMS self-entry is forbidden: {manifest}")
        target = _safe_relative(root, fields[1])
        if target.is_symlink() or not target.is_file() or _sha256(target) != fields[0].lower():
            raise TotalAnalysisError(f"SHA256SUMS digest mismatch: {target}")
        entries[fields[1]] = fields[0].lower()
    if required and not entries:
        raise TotalAnalysisError(f"SHA256SUMS is empty: {manifest}")
    return entries


def _verify_completion(root: Path, *, marker_name: str = TOTAL_COMPLETION_FILE) -> Mapping[str, Any]:
    marker = _read_json(root / marker_name, label="evaluation completion marker")
    if marker.get("status") not in ("COMPLETED", None):
        raise TotalAnalysisError(f"evaluation completion marker is not terminal: {root / marker_name}")
    bundle = verify_completion_bundle(root, completion_name=marker_name)
    if not bundle["valid"]:
        raise TotalAnalysisError(f"evaluation completion/SHA bundle is invalid: {root}: {bundle}")
    return marker


def _verify_family_manifest(directory: Path, marker: Mapping[str, Any], *, marker_name: str | None = None) -> None:
    files = marker.get("files")
    if not isinstance(files, Mapping) or not files:
        raise TotalAnalysisError(f"family marker lacks files: {directory}")
    for name, expected in files.items():
        if not isinstance(name, str) or not isinstance(expected, str) or len(expected) != 64:
            raise TotalAnalysisError(f"family marker has invalid file hash: {directory}")
        target = directory / name
        if target.is_symlink() or not target.is_file() or _sha256(target) != expected.lower():
            raise TotalAnalysisError(f"family marker hash mismatch: {target}")
    # A marker may list a payload subset, but every listed manifest item must
    # be represented by the immutable marker as well.
    manifest = _verify_manifest(directory, required=True)
    if marker_name is None:
        marker_candidates = sorted(path.name for path in directory.glob("COMPLETED_*.json"))
        marker_name = marker_candidates[0] if len(marker_candidates) == 1 else None
    for name, digest in manifest.items():
        # write_atomic_bundle includes the terminal marker in SHA256SUMS but
        # intentionally keeps it out of marker.files to avoid self-reference.
        if marker_name is not None and name == marker_name:
            continue
        if name not in files or str(files[name]).lower() != digest:
            raise TotalAnalysisError(f"family marker and SHA256SUMS disagree: {directory}/{name}")

def _verify_protocol_identity(payload: Mapping[str, Any], protocol: ProtocolAuthority, *, label: str) -> None:
    expected = {
        "protocol_id": protocol.protocol_id,
        "protocol_authority_sha256": protocol.sha256,
        "protocol_git_commit": protocol.git_commit,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise TotalAnalysisError(f"{label} {key} disagrees with frozen authority")


def _finite_matrix(value: Any, *, label: str, width: int | None = None) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TotalAnalysisError(f"{label} is not numeric") from exc
    if array.ndim != 2 or (width is not None and array.shape[1] != width) or array.shape[0] < 1:
        raise TotalAnalysisError(f"{label} has invalid shape {array.shape}; expected [time,{width or 'd'}]")
    if not np.all(np.isfinite(array)):
        raise TotalAnalysisError(f"{label} contains non-finite values")
    return array


def _verify_snapshot(value: Any, *, label: str, per_candidate: bool = False) -> None:
    if not isinstance(value, Mapping):
        raise TotalAnalysisError(f"{label} is not a snapshot object")
    aliases = (("simulator", ("simulator", "environment", "env_state")), ("history", ("observation_history", "policy_history")), ("queue", ("action_queue", "policy_action_queue")), ("rng", ("rng_state", "rng_streams", "python_rng_state")))
    missing = [name for name, keys in aliases if not any(key in value for key in keys)]
    if missing:
        raise TotalAnalysisError(f"{label} lacks full replay components: {', '.join(missing)}")
    if any(value[next(key for key in keys if key in value)] is None for _, keys in aliases):
        raise TotalAnalysisError(f"{label} contains a null replay component")
    # LIBERO persists each stream separately, while Robotwin stores an
    # aggregate rng_streams object.  Either is acceptable only when all four
    # policy/environment streams are represented.
    aggregate_rng = any(key in value for key in ("rng_streams", "rng_state", "all_rng"))
    if aggregate_rng:
        snapshot = value
        aggregate = next(item for key, item in snapshot.items() if key in ("rng_streams", "rng_state", "all_rng"))
        if not isinstance(aggregate, Mapping):
            raise TotalAnalysisError(f"{label} aggregate RNG state is not an object")
        if {"environment", "policy"}.issubset(aggregate):
            if aggregate.get("environment") is None or aggregate.get("policy") is None:
                raise TotalAnalysisError(f"{label} aggregate RNG environment/policy state is incomplete")
        elif not {"python", "numpy", "torch"}.issubset(aggregate):
            raise TotalAnalysisError(f"{label} aggregate RNG state lacks Python/Numpy/Torch streams")
    for stream in ("python_rng_state", "numpy_rng_state", "policy_rng_state", "torch_rng_state"):
        if stream not in value and not aggregate_rng:
            raise TotalAnalysisError(f"{label} lacks replay RNG stream {stream}")
        if stream in value and value[stream] is None:
            raise TotalAnalysisError(f"{label} has null replay RNG stream {stream}")


def _row_value(row: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return None


def _normalise_row(row: Mapping[str, Any], *, family: Mapping[str, Any], index: int, substrate: str, genealogy: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise TotalAnalysisError(f"candidate {index} is not an object")
    source = {**dict(row), **(dict(genealogy) if genealogy is not None else {})}
    candidate_index = _row_value(row, "candidate_index", "index")
    if candidate_index is None:
        candidate_index = index
    try:
        candidate_index = int(candidate_index)
    except (TypeError, ValueError) as exc:
        raise TotalAnalysisError(f"candidate {index} has invalid candidate_index") from exc
    if candidate_index != index:
        raise TotalAnalysisError(f"candidate index/order mismatch in family {family.get('family_id')}")
    family_id = str(_row_value(row, "family_id") or family.get("family_id") or "")
    if not family_id:
        raise TotalAnalysisError("candidate lacks family_id")
    task_id = _row_value(row, "task_id", "task")
    if task_id is None:
        task_id = family.get("task_id")
    init_state = _row_value(row, "init_state", "initial_state", "init_state_id")
    if init_state is None:
        init_state = family.get("init_state", family.get("initial_state_id"))
    candidate_id = _row_value(row, "candidate_id", "id")
    if candidate_id is None:
        raise TotalAnalysisError(f"candidate {family_id}/{index} lacks candidate_id")
    seed = _row_value(row, "candidate_seed", "seed", "rollout_seed")
    if seed is None:
        raise TotalAnalysisError(f"candidate {family_id}/{index} lacks candidate seed")
    try:
        seed = int(seed)
    except (TypeError, ValueError) as exc:
        raise TotalAnalysisError(f"candidate {family_id}/{index} seed is invalid") from exc
    success = _row_value(row, "success", "final_success", "eventual_success")
    if not isinstance(success, (bool, np.bool_, int, np.integer)) or int(success) not in (0, 1):
        raise TotalAnalysisError(f"candidate {family_id}/{index} success label is invalid")
    terminated = _row_value(row, "terminated")
    if terminated is not True and not isinstance(terminated, (np.bool_,)):
        raise TotalAnalysisError(f"candidate {family_id}/{index} is not terminal")
    termination = _row_value(row, "termination", "termination_reason")
    if not isinstance(termination, (str, Mapping)) or not termination:
        raise TotalAnalysisError(f"candidate {family_id}/{index} lacks termination evidence")
    actions = _row_value(row, "actions", "action_prefix", "action_trajectory")
    if actions is None:
        raise TotalAnalysisError(f"candidate {family_id}/{index} lacks actions")
    try:
        action_count = len(actions)
    except TypeError as exc:
        raise TotalAnalysisError(f"candidate {family_id}/{index} actions are not a sequence") from exc
    trajectory = _row_value(row, "trajectory", "poses", "pose_trajectory", "eef_trajectory", "workspace_poses")
    expected_width = 14 if substrate == "A" else 6
    trajectory_array = _finite_matrix(trajectory, label=f"candidate {family_id}/{index} trajectory", width=expected_width)
    env_steps = _row_value(row, "env_steps", "environment_steps", "terminal_step")
    if env_steps is None:
        env_steps = action_count
    try:
        env_steps = int(env_steps)
    except (TypeError, ValueError) as exc:
        raise TotalAnalysisError(f"candidate {family_id}/{index} env_steps is invalid") from exc
    if env_steps < 0 or env_steps != action_count or trajectory_array.shape[0] not in (env_steps, env_steps + 1):
        raise TotalAnalysisError(f"candidate {family_id}/{index} action/trajectory/terminal lengths disagree")
    forwards = _row_value(row, "policy_forwards", "policy_forward_passes", "forward_count")
    if forwards is None:
        raise TotalAnalysisError(f"candidate {family_id}/{index} lacks policy forward count")
    try:
        forwards = int(forwards)
    except (TypeError, ValueError) as exc:
        raise TotalAnalysisError(f"candidate {family_id}/{index} policy_forwards is invalid") from exc
    if forwards < 0:
        raise TotalAnalysisError(f"candidate {family_id}/{index} policy_forwards is negative")
    compute = _row_value(row, "compute")
    if not isinstance(compute, Mapping):
        raise TotalAnalysisError(f"candidate {family_id}/{index} lacks compute record")
    for key, expected in (("policy_forwards", forwards), ("environment_steps", env_steps), ("env_steps", env_steps)):
        try:
            observed = int(compute.get(key, -1))
        except (TypeError, ValueError) as exc:
            raise TotalAnalysisError(f"candidate {family_id}/{index} compute.{key} is invalid") from exc
        if observed != expected:
            raise TotalAnalysisError(f"candidate {family_id}/{index} compute.{key} disagrees with raw count")
    if compute.get("primary_unit") != "policy_forward_pass" or compute.get("secondary_unit") != "environment_step":
        raise TotalAnalysisError(f"candidate {family_id}/{index} compute units drifted")
    genealogy_payload = _row_value(row, "genealogy")
    if genealogy_payload is None:
        genealogy_payload = source
    if not isinstance(genealogy_payload, Mapping):
        raise TotalAnalysisError(f"candidate {family_id}/{index} genealogy is not an object")
    for key in ("parent_id", "generation_step", "action_prefix", "final_success"):
        if key not in genealogy_payload:
            raise TotalAnalysisError(f"candidate {family_id}/{index} genealogy lacks {key}")
    if bool(genealogy_payload["final_success"]) != bool(success):
        raise TotalAnalysisError(f"candidate {family_id}/{index} genealogy success disagrees")
    result = {
        "family_id": family_id,
        "task_id": task_id,
        "init_state": init_state,
        "candidate_id": str(candidate_id),
        "candidate_index": candidate_index,
        "candidate_seed": seed,
        "success": bool(success),
        "terminated": True,
        "termination": termination,
        "actions": actions,
        "trajectory": trajectory_array,
        "poses": trajectory_array,
        "policy_forwards": forwards,
        "env_steps": env_steps,
        "compute": dict(compute),
        "genealogy": dict(genealogy_payload),
    }
    if "seed" not in result:
        result["seed"] = seed
    return result


def _verify_records(
    records: Sequence[Any],
    *,
    substrate: str,
    verification: Mapping[str, Any] | None = None,
    expected_family_count: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    required_flags = ("terminal_markers", "sha256", "genealogy", "compute")
    if verification is None or not all(verification.get(key) is True for key in required_flags):
        raise TotalAnalysisError(f"in-memory handoff lacks complete artifact verification: {required_flags}")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)) or not records:
        raise TotalAnalysisError(f"{substrate} main records are empty")
    groups: dict[str, list[dict[str, Any]]] = {}
    for index, row in enumerate(records):
        if not isinstance(row, Mapping):
            raise TotalAnalysisError(f"{substrate} main row {index} is not an object")
        family_id = str(row.get("family_id", ""))
        family = {"family_id": family_id, "task_id": row.get("task_id"), "init_state": row.get("init_state")}
        normalised = _normalise_row(row, family=family, index=int(row.get("candidate_index", index)), substrate=substrate)
        groups.setdefault(normalised["family_id"], []).append(normalised)
    for family_id, group in groups.items():
        if len(group) != EXPECTED_CANDIDATES:
            raise TotalAnalysisError(f"{substrate} family {family_id} has {len(group)} candidates")
        if [row["candidate_index"] for row in group] != list(range(EXPECTED_CANDIDATES)):
            raise TotalAnalysisError(f"{substrate} family {family_id} candidate order is not 0..31")
        if len({row["candidate_id"] for row in group}) != EXPECTED_CANDIDATES:
            raise TotalAnalysisError(f"{substrate} family {family_id} candidate IDs are not unique")
        if len({row["candidate_seed"] for row in group}) != EXPECTED_CANDIDATES:
            raise TotalAnalysisError(f"{substrate} family {family_id} candidate seeds are not unique")
    if not groups:
        raise TotalAnalysisError(f"{substrate} has no complete families")
    if expected_family_count is not None and len(groups) != int(expected_family_count):
        raise TotalAnalysisError(
            f"{substrate} family count {len(groups)} != frozen {int(expected_family_count)}"
        )
    return [row for family_id in sorted(groups) for row in groups[family_id]], {
        "family_count": len(groups),
        "candidate_count": len(records),
        "terminal_markers": True,
        "sha256": verification is None or verification.get("sha256") is True,
        "genealogy": True,
        "compute": True,
    }


def _family_rows_from_npz(directory: Path, *, substrate: str, protocol: ProtocolAuthority | None = None) -> list[dict[str, Any]]:
    marker = _read_json(directory / "COMPLETED_FAMILY.json", label="family completion marker")
    _verify_family_manifest(directory, marker, marker_name="COMPLETED_FAMILY.json")
    metadata = _read_json(directory / "metadata.json", label="family metadata")
    if marker.get("protocol_id") != PROTOCOL_ID or metadata.get("protocol_id") not in (None, PROTOCOL_ID):
        raise TotalAnalysisError(f"family protocol identity drifted: {directory}")
    family_id = str(metadata.get("family_id", marker.get("family_id", "")))
    if not family_id or str(marker.get("family_id")) != family_id:
        raise TotalAnalysisError(f"family identity drifted: {directory}")
    if int(marker.get("candidate_count", -1)) != EXPECTED_CANDIDATES:
        raise TotalAnalysisError(f"family candidate count marker drifted: {directory}")
    if metadata.get("substrate") not in (None, substrate) or marker.get("substrate") not in (None, substrate):
        raise TotalAnalysisError(f"family substrate identity drifted: {directory}")
    if protocol is not None:
        _verify_protocol_identity(marker, protocol, label=f"{directory} marker")
        _verify_protocol_identity(metadata, protocol, label=f"{directory} metadata")
    genealogy_path = directory / "genealogy.json"
    try:
        genealogy = json.loads(genealogy_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise TotalAnalysisError(f"invalid genealogy: {genealogy_path}") from exc
    if not isinstance(genealogy, list) or len(genealogy) != EXPECTED_CANDIDATES:
        raise TotalAnalysisError(f"genealogy count drifted: {directory}")
    snapshots = directory / "snapshots.pkl"
    try:
        with snapshots.open("rb") as stream:
            snapshot_bundle = pickle.load(stream)
    except Exception as exc:  # noqa: BLE001
        raise TotalAnalysisError(f"invalid snapshots bundle: {snapshots}") from exc
    if not isinstance(snapshot_bundle, Mapping) or snapshot_bundle.get("schema_version") != 1 or not isinstance(snapshot_bundle.get("candidates"), Mapping):
        raise TotalAnalysisError(f"snapshot bundle schema drifted: {snapshots}")
    if len(snapshot_bundle["candidates"]) != EXPECTED_CANDIDATES:
        raise TotalAnalysisError(f"snapshot candidate count drifted: {snapshots}")
    try:
        with np.load(directory / "rollouts.npz", allow_pickle=False) as data:
            required = {"lengths", "offsets", "actions", "poses", "success", "candidate_index", "candidate_seed", "terminated", "terminal_step", "policy_forwards", "environment_steps"}
            if not required.issubset(set(data.files)):
                raise TotalAnalysisError(f"rollouts.npz lacks fields: {directory}")
            lengths = np.asarray(data["lengths"]); offsets = np.asarray(data["offsets"])
            if lengths.shape != (EXPECTED_CANDIDATES,) or offsets.shape != (EXPECTED_CANDIDATES + 1,):
                raise TotalAnalysisError(f"rollouts lengths/offsets shape drifted: {directory}")
            rows: list[dict[str, Any]] = []
            candidate_ids = (
                np.asarray(data["candidate_id"])
                if "candidate_id" in data.files
                else np.arange(EXPECTED_CANDIDATES)
            )
            if candidate_ids.shape != (EXPECTED_CANDIDATES,):
                raise TotalAnalysisError(f"rollouts candidate_id shape drifted: {directory}")
            for index in range(EXPECTED_CANDIDATES):
                start, stop = int(offsets[index]), int(offsets[index + 1])
                if stop - start != int(lengths[index]):
                    raise TotalAnalysisError(f"rollout offsets disagree: {directory}/{index}")
                row = dict(genealogy[index])
                row.update({
                    "family_id": metadata.get("family_id"),
                    "task_id": metadata.get("task_id"),
                    "init_state": metadata.get("init_state"),
                    "candidate_index": int(data["candidate_index"][index]),
                    "candidate_id": str(candidate_ids[index]),
                    "candidate_seed": int(data["candidate_seed"][index]),
                    "success": bool(data["success"][index]),
                    "final_success": bool(data["success"][index]),
                    "terminated": bool(data["terminated"][index]),
                    "termination": row.get("termination", metadata.get("termination", "official_step_limit")),
                    "actions": np.asarray(data["actions"])[start:stop].tolist(),
                    "poses": np.asarray(data["poses"])[start:stop].tolist(),
                    "env_steps": int(data["environment_steps"][index]),
                    "policy_forwards": int(data["policy_forwards"][index]),
                    "compute": {
                        "policy_forwards": int(data["policy_forwards"][index]),
                        "environment_steps": int(data["environment_steps"][index]),
                        "env_steps": int(data["environment_steps"][index]),
                        "primary_unit": "policy_forward_pass",
                        "secondary_unit": "environment_step",
                    },
                })
                _verify_snapshot(snapshot_bundle["candidates"].get(str(row["candidate_id"])), label=f"{directory} candidate {index}", per_candidate=True)
                rows.append(row)
    except TotalAnalysisError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise TotalAnalysisError(f"invalid rollouts.npz: {directory}") from exc
    return rows


def _family_rows_from_robotwin(directory: Path, *, substrate: str, protocol: ProtocolAuthority | None = None) -> list[dict[str, Any]]:
    marker = _read_json(directory / "COMPLETED_FAMILY.json", label="family completion marker")
    _verify_family_manifest(directory, marker, marker_name="COMPLETED_FAMILY.json")
    payload = _read_json(directory / "family.json", label="family result")
    candidates = payload.get("candidates")
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
    if marker.get("protocol_id") != PROTOCOL_ID:
        raise TotalAnalysisError(f"Robotwin family protocol identity drifted: {directory}")
    family_id = str(payload.get("family_id", marker.get("family_id", "")))
    if not family_id or str(marker.get("family_id")) != family_id:
        raise TotalAnalysisError(f"Robotwin family identity drifted: {directory}")
    if int(marker.get("candidate_count", -1)) != EXPECTED_CANDIDATES:
        raise TotalAnalysisError(f"Robotwin family candidate count marker drifted: {directory}")
    if not isinstance(candidates, list) or len(candidates) != EXPECTED_CANDIDATES:
        raise TotalAnalysisError(f"Robotwin family candidate count drifted: {directory}")
    if metadata.get("substrate", marker.get("substrate")) not in (None, substrate):
        raise TotalAnalysisError(f"Robotwin family substrate drifted: {directory}")
    if protocol is not None:
        _verify_protocol_identity(marker, protocol, label=f"{directory} marker")
        _verify_protocol_identity(metadata, protocol, label=f"{directory} metadata")
    snapshot_path = directory / "SNAPSHOT.json"
    if snapshot_path.is_file():
        _verify_snapshot(_read_json(snapshot_path, label="Robotwin snapshot"), label=str(snapshot_path))
    else:
        inline_snapshot = payload.get("snapshot")
        if not isinstance(inline_snapshot, Mapping):
            raise TotalAnalysisError(f"Robotwin family lacks full snapshot: {directory}")
        _verify_snapshot(inline_snapshot, label=f"{directory} inline snapshot")
    genealogy_lines = [line for line in (directory / "genealogy.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(genealogy_lines) != EXPECTED_CANDIDATES:
        raise TotalAnalysisError(f"Robotwin genealogy count drifted: {directory}")
    genealogy: list[Mapping[str, Any]] = []
    for line in genealogy_lines:
        try:
            item = json.loads(line)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise TotalAnalysisError(f"invalid Robotwin genealogy: {directory}") from exc
        if not isinstance(item, Mapping):
            raise TotalAnalysisError(f"Robotwin genealogy row is not an object: {directory}")
        genealogy.append(item)
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise TotalAnalysisError(f"Robotwin candidate is not an object: {directory}/{index}")
        task_id = metadata.get("task_id", metadata.get("task_name", candidate.get("task_id", candidate.get("task_name"))))
        init_state = metadata.get("init_state", metadata.get("initial_state_id", candidate.get("init_state", candidate.get("initial_state_id"))))
        row = {
            **dict(candidate),
            "family_id": payload.get("family_id", marker.get("family_id")),
            "task_id": task_id,
            "init_state": init_state,
            "genealogy": genealogy[index],
        }
        if "compute" not in row:
            try:
                forwards = int(row["policy_forwards"])
                env_steps = int(row["env_steps"])
            except (KeyError, TypeError, ValueError) as exc:
                raise TotalAnalysisError(f"Robotwin candidate {index} lacks raw compute counters: {directory}") from exc
            row["compute"] = {
                "policy_forwards": forwards,
                "environment_steps": env_steps,
                "env_steps": env_steps,
                "primary_unit": "policy_forward_pass",
                "secondary_unit": "environment_step",
            }
        rows.append(row)
    # The robotwin payload itself is the raw row source; normalization below
    # performs the candidate/compute/termination checks.
    return rows


def _load_main_path(value: str | Path, *, substrate: str, protocol: ProtocolAuthority | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = _strict_path(value, label=f"{substrate} main root")
    if source.is_file():
        payload = _read_json(source, label=f"{substrate} main result")
        rows = payload.get("records", payload.get("rollouts", payload.get("rows")))
        if not isinstance(rows, list):
            raise TotalAnalysisError(f"{substrate} main JSON lacks raw records: {source}")
        verified_rows, summary = _verify_records(rows, substrate=substrate, verification=payload.get("artifact_verification"))
        summary["pipeline_identity"] = payload.get("pipeline_identity", payload.get("pipeline_commit", payload.get("source_commit")))
        return verified_rows, summary
    completion = source / "COMPLETED_EVALUATION_RESULT.json"
    pipeline_identity = None
    if completion.is_file():
        marker = _read_json(completion, label=f"{substrate} main completion")
        if marker.get("status") not in ("COMPLETED", None):
            raise TotalAnalysisError(f"{substrate} main completion is not terminal: {completion}")
        expected_marker = f"completed_stage_s_{substrate.lower()}_main_evaluation"
        if substrate == "A":
            expected_marker = "completed_stage_s_a_evaluation"
        if marker.get("marker_type") != expected_marker:
            raise TotalAnalysisError(f"{substrate} main completion marker schema drifted: {completion}")
        pipeline_identity = marker.get("pipeline_identity", marker.get("pipeline_commit", marker.get("source_commit")))
        manifest_entries = _verify_manifest(source, required=True)
        if completion.name not in manifest_entries:
            raise TotalAnalysisError(f"{substrate} main completion is not bound by SHA256SUMS: {completion}")
    family_jsons = sorted(source.rglob("family.json"))
    if family_jsons:
        raw: list[dict[str, Any]] = []
        for path in family_jsons:
            raw.extend(_family_rows_from_robotwin(path.parent, substrate=substrate, protocol=protocol))
        verified_rows, summary = _verify_records(
            raw,
            substrate=substrate,
            verification={"terminal_markers": True, "sha256": True, "genealogy": True, "compute": True},
            expected_family_count=EXPECTED_FAMILY_COUNT,
        )
        summary["pipeline_identity"] = pipeline_identity
        return verified_rows, summary
    npzs = sorted(source.rglob("rollouts.npz"))
    if npzs:
        raw = []
        for path in npzs:
            raw.extend(_family_rows_from_npz(path.parent, substrate=substrate, protocol=protocol))
        verified_rows, summary = _verify_records(
            raw,
            substrate=substrate,
            verification={"terminal_markers": True, "sha256": True, "genealogy": True, "compute": True},
            expected_family_count=EXPECTED_FAMILY_COUNT,
        )
        summary["pipeline_identity"] = pipeline_identity
        return verified_rows, summary
    raise TotalAnalysisError(f"{substrate} main root contains no family artifacts: {source}")



def _near_family_ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    counts: dict[str, int] = {}
    for row in rows:
        family = str(row["family_id"])
        counts[family] = counts.get(family, 0) + int(bool(row["success"]))
    return sorted(family for family, count in counts.items() if count <= 1)


def _verify_s5_compute(row: Mapping[str, Any], *, label: str) -> None:
    # Robotwin's legacy base rows store counters as fields instead of nesting
    # them; derive the same frozen record only after checking both raw counts.
    compute = row.get("compute")
    if not isinstance(compute, Mapping):
        try:
            forwards = int(row["policy_forwards"])
            env_steps = int(row["env_steps"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TotalAnalysisError(f"{label} lacks compute counters") from exc
        compute = {
            "policy_forwards": forwards,
            "environment_steps": env_steps,
            "env_steps": env_steps,
            "primary_unit": "policy_forward_pass",
            "secondary_unit": "environment_step",
        }
    try:
        forwards = int(row["policy_forwards"])
        env_steps = int(row["env_steps"])
        observed_forwards = int(compute.get("policy_forwards", -1))
        observed_steps = int(compute.get("environment_steps", -1))
        observed_env_steps = int(compute.get("env_steps", -1))
    except (KeyError, TypeError, ValueError) as exc:
        raise TotalAnalysisError(f"{label} has invalid compute counts") from exc
    if (observed_forwards, observed_steps, observed_env_steps) != (forwards, env_steps, env_steps):
        raise TotalAnalysisError(f"{label} compute disagrees with raw counters")
    if compute.get("primary_unit") != "policy_forward_pass" or compute.get("secondary_unit") != "environment_step":
        raise TotalAnalysisError(f"{label} compute units drifted")


def _load_s4_path(root: Path, *, protocol: ProtocolAuthority, expected_family_ids: Sequence[str]) -> list[Mapping[str, Any]]:
    if not expected_family_ids:
        return []
    try:
        return list(load_s4_probes(root, protocol=protocol, expected_family_ids=expected_family_ids))
    except S45Error as exc:
        raise TotalAnalysisError(f"S4 persisted probe verification failed: {root}: {exc}") from exc


def _load_s5_path(root: Path, *, protocol: ProtocolAuthority, expected_family_ids: Sequence[str]) -> dict[str, list[Mapping[str, Any]]]:
    payload_paths = sorted(root.rglob("S5_FAMILY.json"))
    if not payload_paths:
        raise TotalAnalysisError(f"S5 root contains no S5_FAMILY.json artifacts: {root}")
    expected = {str(value) for value in expected_family_ids}
    found: dict[str, list[Mapping[str, Any]]] = {}
    for payload_path in payload_paths:
        directory = payload_path.parent
        marker = _read_json(directory / "COMPLETED_S5_FAMILY.json", label="S5 family completion marker")
        _verify_family_manifest(directory, marker, marker_name="COMPLETED_S5_FAMILY.json")
        payload = _read_json(payload_path, label="S5 family result")
        if payload.get("schema") != "r142-stage-s-s5-family-v1" or marker.get("schema") != "r142-stage-s-s5-family-v1":
            raise TotalAnalysisError(f"S5 family schema drifted: {directory}")
        family_id = str(payload.get("family_id", marker.get("family_id", "")))
        if not family_id or family_id in found or str(marker.get("family_id")) != family_id:
            raise TotalAnalysisError(f"S5 family identity is missing or duplicated: {directory}")
        for source in (payload, marker):
            for key, expected_value in protocol.identity().items():
                if source.get(key) != expected_value:
                    raise TotalAnalysisError(f"S5 {family_id} protocol identity drifted: {key}")
        if int(payload.get("base_candidate_count", -1)) != EXPECTED_CANDIDATES or int(payload.get("extended_candidate_count", -1)) != 64:
            raise TotalAnalysisError(f"S5 {family_id} candidate budget drifted")
        if list(payload.get("fresh_candidate_indices", ())) != list(range(32, 64)):
            raise TotalAnalysisError(f"S5 {family_id} fresh indices drifted")
        if int(marker.get("base_candidate_count", -1)) != EXPECTED_CANDIDATES or int(marker.get("extended_candidate_count", -1)) != 64 or list(marker.get("fresh_candidate_indices", ())) != list(range(32, 64)):
            raise TotalAnalysisError(f"S5 {family_id} completion budget drifted")
        base_rows = payload.get("base_rows")
        fresh_rows = payload.get("fresh_rows")
        extended_rows = payload.get("extended_rows")
        if not isinstance(base_rows, list) or len(base_rows) != EXPECTED_CANDIDATES or not isinstance(fresh_rows, list) or len(fresh_rows) != EXPECTED_CANDIDATES or not isinstance(extended_rows, list) or len(extended_rows) != 64:
            raise TotalAnalysisError(f"S5 {family_id} base/fresh/extended row counts drifted")
        if _canonical_digest(base_rows) != str(payload.get("base_digest")):
            raise TotalAnalysisError(f"S5 {family_id} base digest does not bind source rows")
        try:
            extended_indices = [int(row.get("candidate_index", -1)) for row in extended_rows]
            fresh_indices = [int(row.get("candidate_index", -1)) for row in fresh_rows]
        except (AttributeError, TypeError, ValueError) as exc:
            raise TotalAnalysisError(f"S5 {family_id} candidate index is invalid") from exc
        if extended_indices != list(range(64)) or fresh_indices != list(range(32, 64)):
            raise TotalAnalysisError(f"S5 {family_id} extended/fresh candidate indices are not frozen")
        ids: list[str] = []
        for index, row in enumerate(extended_rows):
            if not isinstance(row, Mapping):
                raise TotalAnalysisError(f"S5 {family_id} candidate {index} is not an object")
            if row.get("family_id") not in (None, family_id):
                raise TotalAnalysisError(f"S5 {family_id} candidate family identity drifted")
            if row.get("candidate_id") is None:
                raise TotalAnalysisError(f"S5 {family_id} candidate {index} lacks candidate_id")
            ids.append(str(row["candidate_id"]))
            _verify_s5_compute(row, label=f"S5 {family_id}/{index}")
        if len(set(ids)) != 64:
            raise TotalAnalysisError(f"S5 {family_id} candidate IDs are not unique")
        genealogy_path = directory / "S5_GENEALOGY.jsonl"
        if not genealogy_path.is_file() or genealogy_path.is_symlink():
            raise TotalAnalysisError(f"S5 {family_id} genealogy artifact is missing")
        lines = [line for line in genealogy_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(lines) != 64:
            raise TotalAnalysisError(f"S5 {family_id} genealogy count drifted")
        for index, line in enumerate(lines):
            try:
                genealogy = json.loads(line)
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                raise TotalAnalysisError(f"S5 {family_id} genealogy is invalid") from exc
            if not isinstance(genealogy, Mapping) or str(genealogy.get("candidate_id")) != ids[index]:
                raise TotalAnalysisError(f"S5 {family_id} genealogy identity drifted at {index}")
        found[family_id] = extended_rows
    if set(found) != expected:
        raise TotalAnalysisError(f"S5 output family coverage differs from main screen: expected {len(expected)}, got {len(found)}")
    return found


def _load_value(value: Any, *, label: str) -> Any:
    if isinstance(value, (str, Path)):
        path = _strict_path(value, label=label)
        if path.is_dir():
            for name in ("S45_RESULT.json", "S4_RESULT.json", "S5_RESULT.json", "RESULT.json"):
                candidate = path / name
                if candidate.is_file():
                    return _read_json(candidate, label=label)
            raise TotalAnalysisError(f"{label} directory has no JSON result")
        return _read_json(path, label=label)
    return value


def _extract_arm_inputs(name: str, arm: Any, *, protocol: ProtocolAuthority) -> tuple[list[dict[str, Any]], Any, Any, str | None, dict[str, Any]]:
    if not isinstance(arm, Mapping):
        raise TotalAnalysisError(f"arm {name} must be an object")
    substrate = str(arm.get("substrate", name))
    if substrate not in EXPECTED_SUBSTRATES:
        raise TotalAnalysisError(f"unsupported Stage-S arm {name!r}")
    records = arm.get("records", arm.get("rollouts"))
    verification = arm.get("artifact_verification")
    if records is None:
        main = arm.get("main_root", arm.get("main"))
        if main is None:
            raise TotalAnalysisError(f"arm {name} lacks main_root/records")
        records, verification_from_path = _load_main_path(main, substrate=substrate, protocol=protocol)
        verification = {**verification_from_path, **(dict(verification) if isinstance(verification, Mapping) else {})}
    rows, verification_summary = _verify_records(records, substrate=substrate, verification=verification)
    probes = arm.get("probes", arm.get("s4_probes"))
    if probes is None and arm.get("s4_root") is not None:
        s4_source = _strict_path(arm["s4_root"], label=f"{name} S4 root")
        if s4_source.is_dir():
            probes = _load_s4_path(s4_source, protocol=protocol, expected_family_ids=_near_family_ids(rows))
        else:
            payload = _load_value(s4_source, label=f"{name} S4 result")
            probes = payload.get("probes", payload.get("families", payload.get("S4_probes"))) if isinstance(payload, Mapping) else payload
    extended = arm.get("extended_rollouts", arm.get("s5_rollouts"))
    if extended is None and arm.get("s5_root") is not None:
        s5_source = _strict_path(arm["s5_root"], label=f"{name} S5 root")
        if s5_source.is_dir():
            extended = _load_s5_path(s5_source, protocol=protocol, expected_family_ids=sorted({str(row["family_id"]) for row in rows}))
        else:
            payload = _load_value(s5_source, label=f"{name} S5 result")
            if isinstance(payload, Mapping):
                extended = payload.get("extended_rollouts", payload.get("extended", payload.get("families")))
            else:
                extended = payload
    pipeline = arm.get("pipeline_identity", arm.get("pipeline_commit"))
    if pipeline is None:
        pipeline = arm.get("source_commit")
    if pipeline is None:
        pipeline = verification_summary.get("pipeline_identity")
    return rows, probes, extended, None if pipeline is None else str(pipeline), {**verification_summary, "arm": name}


def _load_controls(value: Any) -> dict[str, Any]:
    if isinstance(value, (str, Path)):
        root = _strict_path(value, label="controls root", directory=True)
        _verify_completion(root, marker_name="COMPLETED_CONTROLS.json")
        payload = _read_json(root / "CONTROLS_REPORT.json", label="controls report")
    elif isinstance(value, Mapping):
        payload = value.get("aggregate", value)
    else:
        raise TotalAnalysisError("controls must be a path or object")
    verdict = payload.get("overall_verdict", payload.get("verdict"))
    positive = payload.get("positive", {})
    null = payload.get("null", {})
    positive_verdict = payload.get("positive_verdict")
    null_verdict = payload.get("null_verdict")
    if verdict != "CONTROLS_PASS":
        raise TotalAnalysisError(f"controls did not pass: {verdict!r}")
    if positive_verdict != "POSITIVE_CONTROL_PASS":
        raise TotalAnalysisError(f"positive control did not pass: {positive_verdict!r}")
    if null_verdict != "NO_FAMILY_COLLAPSE":
        raise TotalAnalysisError(f"null control did not pass: {null_verdict!r}")
    pipeline = payload.get("pipeline_commit")
    if pipeline is None:
        pipeline = positive.get("pipeline_commit") if isinstance(positive, Mapping) else None
    if pipeline is None and isinstance(null, Mapping):
        pipeline = null.get("pipeline_commit")
    if pipeline is None:
        raise TotalAnalysisError("controls lack pipeline identity")
    return {"overall_verdict": verdict, "pipeline_commit": str(pipeline), "report": payload}


def _invalid_gate(error: str) -> dict[str, Any]:
    return {"pass": False, "status": "INVALID_INPUT", "error": str(error)}


def _compute_arm(rows: list[dict[str, Any]], probes: Any, extended: Any, *, substrate: str, protocol: ProtocolAuthority) -> dict[str, Any]:
    result: dict[str, Any] = {}
    # These calls are intentionally not nested under a previous gate. Every
    # arm gets all five computations, including S4/S5 when S2 has no near
    # families (those gates return an explicit non-applicable failure).
    result["S1"] = compute_s1(rows)
    result["S2"] = compute_s2(rows)
    try:
        result["S3"] = compute_s3_production(rows, substrate=substrate, successful_episodes=[row for row in rows if row["success"]])
    except (ValueError, TypeError, TotalAnalysisError) as exc:
        result["S3"] = _invalid_gate(str(exc))
    near_count = int(result["S2"].get("near_all_fail_count", 0))
    if probes is None:
        probes = []
    if not isinstance(probes, Sequence) or isinstance(probes, (str, bytes)):
        result["S4"] = _invalid_gate("S4 probes are not a sequence")
    elif near_count == 0:
        result["S4"] = {"probed_family_count": 0, "oracle_recovered_count": 0, "random_recovered_count": 0, "equal_branch_counts": True, "bootstrap": {"replicates": EXPECTED_S4_BOOTSTRAP_REPLICATES, "seed": EXPECTED_S4_BOOTSTRAP_SEED}, "pass": False, "status": "NO_NEAR_ALL_FAIL_FAMILIES"}
    else:
        try:
            if len(probes) != near_count:
                raise ValueError(
                    f"S4 probe coverage is incomplete: expected {near_count}, got {len(probes)}"
                )
            result["S4"] = compute_s4_from_protocol(list(probes), protocol)
        except (ValueError, TypeError, TotalAnalysisError) as exc:
            result["S4"] = _invalid_gate(str(exc))
    if extended is None:
        extended = {}
    try:
        base_groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            base_groups.setdefault(str(row["family_id"]), []).append(row)
        result["S5"] = compute_s5(base_groups, extended)
    except (ValueError, TypeError, TotalAnalysisError) as exc:
        result["S5"] = _invalid_gate(str(exc))
    # Lowercase aliases are useful to downstream report tooling, but the
    # single decision code lives only at the total-result level.
    result["s1"], result["s2"], result["s3"], result["s4"], result["s5"] = (result[key] for key in ("S1", "S2", "S3", "S4", "S5"))
    result["pass"] = bool(all(bool(result[key].get("pass", False)) for key in ("S1", "S2", "S3", "S4", "S5")))
    return result


def _artifact_identity(protocol: ProtocolAuthority, controls: Mapping[str, Any], arms: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    identities = {name: arm.get("pipeline_identity") for name, arm in arms.items()}
    expected = str(controls["pipeline_commit"])
    mismatches = {name: value for name, value in identities.items() if value is None or str(value) != expected}
    return {"expected": expected, "arms": {name: None if value is None else str(value) for name, value in identities.items()}, "equivalent": not mismatches, "mismatches": mismatches}


def analyze_stage_s(*, protocol_path: str | Path, controls: Any, arms: Mapping[str, Any], output_root: str | Path | None = None) -> dict[str, Any]:
    """Verify and analyze every Stage-S arm, then optionally publish a bundle."""

    if set(arms) != set(EXPECTED_SUBSTRATES):
        raise TotalAnalysisError(f"Stage-S total analysis requires exactly A/B/C arms, got {sorted(arms)}")
    protocol = ProtocolAuthority.load(protocol_path)
    control_error: str | None = None
    try:
        controls_payload = _load_controls(controls)
    except (TotalAnalysisError, OSError, ValueError, TypeError) as exc:
        # Controls are a hard scientific validity gate, but their failure must
        # not suppress independent A/B/C artifact and gate evaluation.
        control_error = str(exc)
        controls_payload = {"overall_verdict": "CONTROLS_INVALID", "pipeline_commit": None, "report": {}}
    arm_rows: dict[str, list[dict[str, Any]]] = {}
    arm_probes: dict[str, Any] = {}
    arm_extended: dict[str, Any] = {}
    arm_identity: dict[str, str | None] = {}
    arm_verification: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    for name in EXPECTED_SUBSTRATES:
        try:
            rows, probes, extended, identity, verification = _extract_arm_inputs(name, arms[name], protocol=protocol)
            arm_rows[name], arm_probes[name], arm_extended[name], arm_identity[name], arm_verification[name] = rows, probes, extended, identity, verification
        except (TotalAnalysisError, OSError, ValueError, TypeError) as exc:
            errors[name] = str(exc)
            arm_rows[name] = []
            arm_probes[name] = []
            arm_extended[name] = {}
            arm_identity[name] = None
            arm_verification[name] = {"arm": name, "valid": False}
    arm_results: dict[str, dict[str, Any]] = {}
    for name in EXPECTED_SUBSTRATES:
        if errors.get(name):
            # Keep an explicit five-gate record even when disk validation for
            # this arm failed. Other arms are still computed below.
            arm_results[name] = {key: _invalid_gate(errors[name]) for key in ("S1", "S2", "S3", "S4", "S5")}
            arm_results[name].update({key.lower(): arm_results[name][key] for key in ("S1", "S2", "S3", "S4", "S5")})
            arm_results[name]["pass"] = False
            continue
        try:
            arm_results[name] = _compute_arm(arm_rows[name], arm_probes[name], arm_extended[name], substrate=name, protocol=protocol)
        except (TotalAnalysisError, ValueError, TypeError) as exc:
            # A gate implementation failure is evidence invalidity, not a
            # reason to suppress the other arms' gate calculations.
            arm_results[name] = {key: _invalid_gate(str(exc)) for key in ("S1", "S2", "S3", "S4", "S5")}
            arm_results[name].update({key.lower(): arm_results[name][key] for key in ("S1", "S2", "S3", "S4", "S5")})
            arm_results[name]["pass"] = False
            errors[name] = str(exc)
    expected_pipeline = controls_payload.get("pipeline_commit")
    equivalence = {"expected": expected_pipeline, "arms": arm_identity, "equivalent": control_error is None and expected_pipeline is not None and all(value is not None and str(value) == str(expected_pipeline) for value in arm_identity.values()), "mismatches": {name: value for name, value in arm_identity.items() if expected_pipeline is None or value is None or str(value) != str(expected_pipeline)}}
    if control_error is not None:
        errors["controls"] = control_error
    decision = "PIPELINE_INVALID" if errors or not equivalence["equivalent"] else decide_stage_s(arm_results, positive_control_pass=True)
    if decision not in DECISION_CODES:
        raise TotalAnalysisError(f"decision code is not frozen: {decision}")
    report: dict[str, Any] = {
        "schema": TOTAL_ANALYSIS_SCHEMA,
        "status": "COMPLETED",
        "protocol": protocol.identity(),
        "controls": controls_payload,
        "pipeline_equivalence": equivalence,
        "arms": arm_results,
        "arm_verification": arm_verification,
        "decision_code": decision,
        "decision_code_count": 1,
        "verification_errors": errors,
        "all_arms_evaluated": all(name in arm_results and all(key in arm_results[name] for key in ("S1", "S2", "S3", "S4", "S5")) for name in EXPECTED_SUBSTRATES),
        "s4_protocol": {"grid_points": EXPECTED_S4_GRID_COUNT, "search_branches": EXPECTED_S4_SEARCH_COUNT, "oracle_heldout_branches": EXPECTED_S4_HELDOUT_COUNT, "random_heldout_branches": EXPECTED_S4_HELDOUT_COUNT, "bootstrap_replicates": EXPECTED_S4_BOOTSTRAP_REPLICATES, "bootstrap_seed": EXPECTED_S4_BOOTSTRAP_SEED},
        "s3_protocol": {"scale_by_substrate": {name: list(S3_SUBSTRATE_WORKSPACE_SCALES[name]) for name in EXPECTED_SUBSTRATES}, "tau": "successful_same_task_matched_time_95th", "scalar_override": False},
        "weak_substrate_rule": "C is WEAK_SUBSTRATE only and never headline SUBSTRATE_QUALIFIED",
    }
    if output_root is not None:
        output = _strict_path(output_root, label="total analysis output", directory=True) if Path(output_root).exists() else Path(output_root).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(output / TOTAL_RESULT_FILE, report)
        completion = write_completion(output, decision, artifacts=[TOTAL_RESULT_FILE], completion_name=TOTAL_COMPLETION_FILE, metadata={"schema": TOTAL_ANALYSIS_SCHEMA, "decision_code": decision, "decision_code_count": 1, "all_arms_evaluated": report["all_arms_evaluated"]})
        bundle = verify_completion_bundle(output, completion_name=completion.name)
        if not bundle["valid"]:
            raise TotalAnalysisError(f"total analysis completion bundle failed verification: {bundle}")
        report["output_root"] = str(output)
        report["completion"] = str(completion)
        report["bundle"] = bundle
    return report


# Names used by different launcher generations are intentionally aliases of
# the same fail-closed implementation.
finalize_stage_s = analyze_stage_s
finalise_stage_s = analyze_stage_s
analyze_all_stage_s = analyze_stage_s


__all__ = [
    "EXPECTED_S4_BOOTSTRAP_REPLICATES",
    "EXPECTED_S4_BOOTSTRAP_SEED",
    "EXPECTED_S4_GRID_COUNT",
    "EXPECTED_S4_HELDOUT_COUNT",
    "EXPECTED_S4_SEARCH_COUNT",
    "EXPECTED_SUBSTRATES",
    "TOTAL_ANALYSIS_SCHEMA",
    "TOTAL_COMPLETION_FILE",
    "TOTAL_RESULT_FILE",
    "TotalAnalysisError",
    "analyze_all_stage_s",
    "analyze_stage_s",
    "finalise_stage_s",
    "finalize_stage_s",
]
