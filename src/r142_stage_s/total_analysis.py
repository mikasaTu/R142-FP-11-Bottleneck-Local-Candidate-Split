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
import re
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
from .frozen_protocol import FrozenProtocolError, load_frozen_protocol
from .main_protocol import A_TASKS, FrozenProtocolError as MainFrozenProtocolError, read_frozen_protocol
from .s45_runtime import (
    PROTOCOL_ID,
    N32Family,
    ProtocolAuthority,
    S45Error,
    discover_n32_families,
    load_s4_probes,
    load_s5_extended,
)


TOTAL_ANALYSIS_SCHEMA = "r142-stage-s-total-analysis-v1"
TOTAL_RESULT_FILE = "STAGE_S_TOTAL_ANALYSIS.json"
TOTAL_COMPLETION_FILE = "COMPLETED_EVALUATION_RESULT.json"
EXPECTED_SUBSTRATES = ("A", "B", "C")
EXPECTED_WORLD_SIZE = 8
EXPECTED_FAMILY_COUNT = 160
EXPECTED_CANDIDATES = 32
EXPECTED_TASK_IDS = tuple(range(10))
EXPECTED_INIT_STATE_IDS = tuple(range(16))
EXPECTED_TOTAL_CANDIDATES = EXPECTED_FAMILY_COUNT * EXPECTED_CANDIDATES
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


_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA64 = re.compile(r"^[0-9a-f]{64}$")


def _require_sha(value: Any, *, length: int, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{%d}" % length, value) is None:
        raise TotalAnalysisError(f"{label} must be a lowercase full SHA-{length * 4}")
    return value


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


def _verify_completion(
    root: Path,
    *,
    marker_name: str = TOTAL_COMPLETION_FILE,
    require_status: bool = True,
) -> Mapping[str, Any]:
    marker = _read_json(root / marker_name, label="evaluation completion marker")
    if require_status and marker.get("status") != "COMPLETED":
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


def _load_production_protocol(path: str | Path) -> tuple[ProtocolAuthority, dict[str, Any]]:
    """Load the two independent frozen-protocol validators for production.

    ``ProtocolAuthority`` is the S4/S5 runtime validator, while
    ``load_frozen_protocol`` and ``read_frozen_protocol`` validate the signed
    Stage-S acceptance envelope, adjacent ``PROTOCOL.md`` and the B/C
    calibration report hashes.  Keeping both checks here prevents the total
    analyzer from accepting a convenient JSON with only the statistical fields
    populated.
    """

    source = _strict_path(path, label="production frozen protocol authority", directory=False)
    if source.name != "FROZEN_PROTOCOL.json" or source.parent.name != "protocol":
        raise TotalAnalysisError(f"production protocol path must be protocol/FROZEN_PROTOCOL.json: {source}")
    try:
        frozen = load_frozen_protocol(source)
    except (FrozenProtocolError, OSError, ValueError, TypeError) as exc:
        raise TotalAnalysisError(f"frozen_protocol validator rejected authority: {source}") from exc
    if frozen.get("status") != "FROZEN":
        raise TotalAnalysisError("frozen protocol status must be FROZEN")
    try:
        protocol = ProtocolAuthority.load(source)
    except (S45Error, OSError, ValueError, TypeError) as exc:
        raise TotalAnalysisError(f"S4/S5 protocol validator rejected authority: {source}") from exc
    if protocol.protocol_id != PROTOCOL_ID:
        raise TotalAnalysisError(f"production protocol id drifted: {protocol.protocol_id}")
    if protocol.path != source:
        raise TotalAnalysisError("protocol authority path was not canonicalized")
    if protocol.sha256 != frozen.get("protocol_json_sha256"):
        raise TotalAnalysisError("ProtocolAuthority and frozen_protocol JSON SHA-256 disagree")
    if protocol.git_commit != frozen.get("protocol_git_commit"):
        raise TotalAnalysisError("ProtocolAuthority and frozen_protocol git commit disagree")

    calibration = frozen.get("calibration_reports")
    if not isinstance(calibration, Mapping) or set(calibration) != {"B", "C"}:
        raise TotalAnalysisError("frozen protocol must carry exactly B/C calibration hashes")
    accepted: dict[str, Mapping[str, Any]] = {}
    for substrate in ("B", "C"):
        entry = calibration.get(substrate)
        if not isinstance(entry, Mapping):
            raise TotalAnalysisError(f"frozen protocol calibration entry missing: {substrate}")
        report_path = _strict_path(entry.get("path"), label=f"{substrate} calibration report", directory=False)
        declared_sha = _require_sha(entry.get("sha256"), length=64, label=f"{substrate} calibration report hash")
        if _sha256(report_path) != declared_sha:
            raise TotalAnalysisError(f"{substrate} calibration report SHA-256 mismatch")
        try:
            acceptance = read_frozen_protocol(
                source,
                substrate=substrate,
                calibration_report=report_path,
            )
        except (MainFrozenProtocolError, OSError, ValueError, TypeError) as exc:
            raise TotalAnalysisError(f"main_protocol validator rejected {substrate} acceptance") from exc
        if acceptance.get("status") != "FROZEN":
            raise TotalAnalysisError(f"main_protocol {substrate} acceptance is not FROZEN")
        if acceptance.get("protocol_acceptance_sha256") != frozen.get("protocol_json_sha256"):
            raise TotalAnalysisError(f"main_protocol {substrate} acceptance hash disagrees with frozen protocol")
        if acceptance.get("protocol_git_commit") != frozen.get("protocol_git_commit"):
            raise TotalAnalysisError(f"main_protocol {substrate} protocol commit disagrees")
        if acceptance.get("protocol_md_path") != frozen.get("protocol_md_path") or acceptance.get("protocol_md_sha256") != frozen.get("protocol_md_sha256"):
            raise TotalAnalysisError(f"main_protocol {substrate} PROTOCOL.md binding disagrees")
        accepted[substrate] = acceptance

    summary = frozen.get("frozen_summary")
    if not isinstance(summary, Mapping):
        raise TotalAnalysisError("frozen protocol lacks frozen_summary")
    budget = summary.get("budget")
    expected_budget = {
        "task_count": 10,
        "families_per_task": 16,
        "candidates_per_family": 32,
        "terminal_episode_count": EXPECTED_TOTAL_CANDIDATES,
        "world_size": EXPECTED_WORLD_SIZE,
    }
    if budget != expected_budget:
        raise TotalAnalysisError(f"frozen protocol main budget drifted: {budget!r}")
    thresholds = summary.get("thresholds")
    if not isinstance(thresholds, Mapping) or not thresholds:
        raise TotalAnalysisError("frozen protocol lacks gate thresholds")
    return protocol, {
        "status": "FROZEN",
        "protocol": protocol.identity(),
        "frozen": frozen,
        "acceptance": {name: dict(value) for name, value in accepted.items()},
    }


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


def _verify_rng_state(value: Any, *, label: str) -> None:
    """Require Python/NumPy/Torch CPU/CUDA plus policy/environment RNGs."""

    if not isinstance(value, Mapping):
        raise TotalAnalysisError(f"{label} RNG state is not an object")
    required = {"python", "numpy", "torch", "torch_cuda"}
    if not required.issubset(value):
        raise TotalAnalysisError(f"{label} RNG state lacks Python/NumPy/Torch CPU/CUDA streams")
    if any(value.get(key) is None for key in required):
        raise TotalAnalysisError(f"{label} RNG state contains a null stream")


def _verify_snapshot(value: Any, *, label: str, per_candidate: bool = False) -> None:
    del per_candidate
    if not isinstance(value, Mapping):
        raise TotalAnalysisError(f"{label} is not a snapshot object")
    aliases = (
        ("simulator", ("simulator", "simulator_state", "environment", "env_state")),
        ("history", ("observation_history", "policy_observation_history", "policy_history", "history")),
        ("queue", ("action_queue", "policy_action_queue", "queue")),
    )
    missing = [name for name, keys in aliases if not any(key in value for key in keys)]
    if missing:
        raise TotalAnalysisError(f"{label} lacks full replay components: {', '.join(missing)}")
    if any(value[next(key for key in keys if key in value)] is None for _, keys in aliases):
        raise TotalAnalysisError(f"{label} contains a null replay component")

    aggregate_key = next((key for key in ("rng_streams", "rng_state", "all_rng") if key in value), None)
    if aggregate_key is not None:
        aggregate = value.get(aggregate_key)
        if not isinstance(aggregate, Mapping):
            raise TotalAnalysisError(f"{label} aggregate RNG state is not an object")
        if {"environment", "policy"}.issubset(aggregate):
            _verify_rng_state(aggregate["environment"], label=f"{label}.environment")
            _verify_rng_state(aggregate["policy"], label=f"{label}.policy")
        elif {"runtime", "policy"}.issubset(aggregate):
            # RoboTwin's ConcreteReplaySnapshot names the simulator-side
            # process stream ``runtime`` and keeps the policy stream separate.
            _verify_rng_state(aggregate["runtime"], label=f"{label}.runtime")
            _verify_rng_state(aggregate["policy"], label=f"{label}.policy")
        else:
            _verify_rng_state(aggregate, label=f"{label}.aggregate")
    else:
        individual = {
            "python": value.get("python_rng_state"),
            "numpy": value.get("numpy_rng_state"),
            "policy": value.get("policy_rng_state"),
            "torch": value.get("torch_rng_state"),
        }
        if any(item is None for item in individual.values()):
            raise TotalAnalysisError(f"{label} lacks replay Python/NumPy/Torch/policy streams")
        torch_state = individual["torch"]
        if not isinstance(torch_state, Mapping) or "cpu" not in torch_state or "cuda" not in torch_state:
            raise TotalAnalysisError(f"{label} Torch RNG state lacks CPU/CUDA streams")


def _verify_replay_check(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TotalAnalysisError(f"{label} is not a replay-check object")
    if value.get("same_action") is not True or value.get("passed") is not True:
        raise TotalAnalysisError(f"{label} lacks a passing restore->same-action check")
    raw_error = value.get(
        "max_abs_error",
        value.get("same_action_next_state_max_abs_error", value.get("next_state_error")),
    )
    try:
        error = float(raw_error)
    except (TypeError, ValueError) as exc:
        raise TotalAnalysisError(f"{label} has no numeric same-action error") from exc
    if not np.isfinite(error) or error > 1e-9:
        raise TotalAnalysisError(f"{label} same-action next-state error exceeds 1e-9")
    return dict(value)


def _jsonable(value: Any) -> Any:
    """Convert NumPy containers for exact provenance comparisons."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _strict_genealogy(
    row: Mapping[str, Any],
    genealogy: Mapping[str, Any],
    *,
    family_id: str,
    substrate: str,
    index: int,
    candidate_id: str,
    seed: int,
    actions: Any,
    success: bool,
) -> None:
    """Bind raw genealogy fields to the terminal candidate row byte-for-byte."""

    expected_id = f"{family_id}/candidate-{index:04d}" if substrate == "A" else str(index)
    observed_genealogy_id = genealogy.get("candidate_id")
    if observed_genealogy_id is None or str(observed_genealogy_id) != expected_id:
        raise TotalAnalysisError(f"candidate {family_id}/{index} genealogy candidate_id is not canonical")
    if str(candidate_id) != expected_id:
        raise TotalAnalysisError(f"candidate {family_id}/{index} candidate_id is not canonical")
    try:
        if int(genealogy.get("candidate_index", -1)) != index:
            raise TotalAnalysisError(f"candidate {family_id}/{index} genealogy order is not frozen")
    except (TypeError, ValueError) as exc:
        raise TotalAnalysisError(f"candidate {family_id}/{index} genealogy candidate_index is invalid") from exc
    if genealogy.get("parent_id") is not None:
        raise TotalAnalysisError(f"candidate {family_id}/{index} parent_id is not a root candidate")
    try:
        if int(genealogy.get("generation_step", -1)) != 0:
            raise TotalAnalysisError(f"candidate {family_id}/{index} generation_step is not zero")
    except (TypeError, ValueError) as exc:
        raise TotalAnalysisError(f"candidate {family_id}/{index} generation_step is invalid") from exc
    root = genealogy.get("root_family_id", genealogy.get("root_id", genealogy.get("root")))
    if root is None:
        # ``family_id`` is the explicit root alias used by the A genealogy
        # schema; a missing root binding is not silently inferred from the
        # surrounding row in strict mode.
        root = genealogy.get("family_id")
    if root is None:
        raise TotalAnalysisError(f"candidate {family_id}/{index} genealogy lacks root binding")
    if str(root) != family_id:
        raise TotalAnalysisError(f"candidate {family_id}/{index} root binding drifted")
    if "candidate_seed" in genealogy:
        try:
            if int(genealogy["candidate_seed"]) != seed:
                raise TotalAnalysisError(f"candidate {family_id}/{index} genealogy seed disagrees")
        except (TypeError, ValueError) as exc:
            raise TotalAnalysisError(f"candidate {family_id}/{index} genealogy seed is invalid") from exc
    prefix = genealogy.get("action_prefix")
    if prefix is None or _canonical(_jsonable(prefix)) != _canonical(_jsonable(actions)):
        raise TotalAnalysisError(f"candidate {family_id}/{index} action_prefix is not bound to raw actions")
    if bool(genealogy.get("final_success")) != bool(success):
        raise TotalAnalysisError(f"candidate {family_id}/{index} genealogy success disagrees")


def _row_value(row: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return None


def _normalise_row(
    row: Mapping[str, Any],
    *,
    family: Mapping[str, Any],
    index: int,
    substrate: str,
    genealogy: Mapping[str, Any] | None = None,
    snapshot: Mapping[str, Any] | None = None,
    strict: bool = False,
) -> dict[str, Any]:
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
    if strict and task_id is None:
        raise TotalAnalysisError(f"candidate {family_id}/{index} lacks task_id")
    init_state = _row_value(row, "init_state", "initial_state", "init_state_id")
    if init_state is None:
        init_state = family.get("init_state", family.get("initial_state_id"))
    if strict and init_state is None:
        raise TotalAnalysisError(f"candidate {family_id}/{index} lacks init_state")
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
    if strict and not isinstance(success, (bool, np.bool_)):
        raise TotalAnalysisError(f"candidate {family_id}/{index} success label is not boolean")
    terminated = _row_value(row, "terminated")
    if not isinstance(terminated, (bool, np.bool_)) or not bool(terminated):
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
    if genealogy_payload is None and genealogy is not None:
        genealogy_payload = genealogy
    if genealogy_payload is None:
        genealogy_payload = source
    if not isinstance(genealogy_payload, Mapping):
        raise TotalAnalysisError(f"candidate {family_id}/{index} genealogy is not an object")
    for key in ("parent_id", "generation_step", "action_prefix", "final_success"):
        if key not in genealogy_payload:
            raise TotalAnalysisError(f"candidate {family_id}/{index} genealogy lacks {key}")
    if bool(genealogy_payload["final_success"]) != bool(success):
        raise TotalAnalysisError(f"candidate {family_id}/{index} genealogy success disagrees")
    replay = _row_value(row, "snapshot_restore_check", "replay_check")
    if replay is None and isinstance(snapshot, Mapping):
        replay = _row_value(snapshot, "snapshot_restore_check", "replay_check")
    if strict and replay is None:
        raise TotalAnalysisError(f"candidate {family_id}/{index} lacks snapshot_restore_check")
    replay_checked = _verify_replay_check(replay, label=f"candidate {family_id}/{index}.snapshot_restore_check") if replay is not None else None
    if strict:
        _strict_genealogy(
            row,
            genealogy_payload,
            family_id=family_id,
            substrate=substrate,
            index=index,
            candidate_id=str(candidate_id),
            seed=seed,
            actions=actions,
            success=bool(success),
        )
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
    if replay_checked is not None:
        result["snapshot_restore_check"] = replay_checked
    for key in ("rank", "world_size", "substrate_annotation"):
        value = _row_value(row, key)
        if value is None:
            value = family.get(key)
        if value is not None:
            result[key] = value
    if "seed" not in result:
        result["seed"] = seed
    return result


def _verify_records(
    records: Sequence[Any],
    *,
    substrate: str,
    verification: Mapping[str, Any] | None = None,
    expected_family_count: int | None = None,
    strict: bool = False,
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
        candidate_index = row.get("candidate_index", index)
        try:
            candidate_index = int(candidate_index)
        except (TypeError, ValueError) as exc:
            raise TotalAnalysisError(f"{substrate} main row {index} has an invalid candidate_index") from exc
        normalised = _normalise_row(
            row,
            family=family,
            index=candidate_index,
            substrate=substrate,
            strict=strict,
        )
        groups.setdefault(normalised["family_id"], []).append(normalised)
    expected_pairs: set[tuple[int, int]] = set()
    observed_pairs: set[tuple[int, int]] = set()
    observed_ranks: set[int] = set()
    for family_id, group in groups.items():
        if len(group) != EXPECTED_CANDIDATES:
            raise TotalAnalysisError(f"{substrate} family {family_id} has {len(group)} candidates")
        if [row["candidate_index"] for row in group] != list(range(EXPECTED_CANDIDATES)):
            raise TotalAnalysisError(f"{substrate} family {family_id} candidate order is not 0..31")
        if len({row["candidate_id"] for row in group}) != EXPECTED_CANDIDATES:
            raise TotalAnalysisError(f"{substrate} family {family_id} candidate IDs are not unique")
        if len({row["candidate_seed"] for row in group}) != EXPECTED_CANDIDATES:
            raise TotalAnalysisError(f"{substrate} family {family_id} candidate seeds are not unique")
        if strict:
            try:
                task_id = int(group[0]["task_id"])
                init_state = int(group[0]["init_state"])
            except (TypeError, ValueError) as exc:
                raise TotalAnalysisError(f"{substrate} family {family_id} task_id/init_state is not an integer") from exc
            if task_id not in EXPECTED_TASK_IDS or init_state not in EXPECTED_INIT_STATE_IDS:
                raise TotalAnalysisError(f"{substrate} family {family_id} task/state is outside 10x16 frozen grid")
            if any(int(row["task_id"]) != task_id or int(row["init_state"]) != init_state for row in group):
                raise TotalAnalysisError(f"{substrate} family {family_id} mixes task/state identities")
            pair = (task_id, init_state)
            if pair in observed_pairs:
                raise TotalAnalysisError(f"{substrate} duplicates task/state family {pair}")
            observed_pairs.add(pair)
            expected_rank = (task_id * len(EXPECTED_INIT_STATE_IDS) + init_state) % EXPECTED_WORLD_SIZE
            ranks = {int(row.get("rank", -1)) for row in group}
            world_sizes = {int(row.get("world_size", -1)) for row in group}
            if ranks != {expected_rank} or world_sizes != {EXPECTED_WORLD_SIZE}:
                raise TotalAnalysisError(f"{substrate} family {family_id} rank/world-size assignment drifted")
            observed_ranks.add(expected_rank)
    if not groups:
        raise TotalAnalysisError(f"{substrate} has no complete families")
    if expected_family_count is not None and len(groups) != int(expected_family_count):
        raise TotalAnalysisError(
            f"{substrate} family count {len(groups)} != frozen {int(expected_family_count)}"
        )
    if strict:
        expected_pairs = {
            (task_id, init_state)
            for task_id in EXPECTED_TASK_IDS
            for init_state in EXPECTED_INIT_STATE_IDS
        }
        if observed_pairs != expected_pairs:
            missing = sorted(expected_pairs - observed_pairs)
            extra = sorted(observed_pairs - expected_pairs)
            raise TotalAnalysisError(f"{substrate} task/state coverage is not exactly 10x16: missing={missing[:4]} extra={extra[:4]}")
        if observed_ranks != set(range(EXPECTED_WORLD_SIZE)):
            raise TotalAnalysisError(f"{substrate} rank coverage is not 0..{EXPECTED_WORLD_SIZE - 1}")
        if len(records) != EXPECTED_TOTAL_CANDIDATES:
            raise TotalAnalysisError(f"{substrate} terminal candidate total {len(records)} != {EXPECTED_TOTAL_CANDIDATES}")
    return [row for family_id in sorted(groups) for row in groups[family_id]], {
        "family_count": len(groups),
        "candidate_count": len(records),
        "terminal_markers": True,
        "sha256": verification is None or verification.get("sha256") is True,
        "genealogy": True,
        "compute": True,
    }


def _family_rows_from_npz(
    directory: Path,
    *,
    substrate: str,
    protocol: ProtocolAuthority | None = None,
    strict: bool = False,
) -> list[dict[str, Any]]:
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
    if strict:
        annotations = [
            value
            for value in (
                metadata.get("substrate_annotation"),
                marker.get("substrate_annotation"),
            )
            if value is not None
        ]
        if substrate == "C" and any(value != "WEAK_SUBSTRATE" for value in annotations):
            raise TotalAnalysisError(f"C family lacks WEAK_SUBSTRATE annotation: {directory}")
        if substrate == "C" and not annotations:
            raise TotalAnalysisError(f"C family lacks WEAK_SUBSTRATE annotation: {directory}")
        if substrate in {"A", "B"} and any(value not in (None, "") for value in annotations):
            raise TotalAnalysisError(f"{substrate} family has an unexpected weak-substrate annotation: {directory}")
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
            if strict and "candidate_id" not in data.files:
                raise TotalAnalysisError(f"rollouts.npz lacks immutable candidate_id array: {directory}")
            candidate_ids = np.asarray(data["candidate_id"]) if "candidate_id" in data.files else np.arange(EXPECTED_CANDIDATES)
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
                snapshot_value = snapshot_bundle["candidates"].get(str(row["candidate_id"]))
                _verify_snapshot(snapshot_value, label=f"{directory} candidate {index}", per_candidate=True)
                replay = snapshot_value.get("snapshot_restore_check") if isinstance(snapshot_value, Mapping) else None
                if replay is not None:
                    row["snapshot_restore_check"] = replay
                if strict:
                    row = _normalise_row(
                        row,
                        family=metadata,
                        index=index,
                        substrate=substrate,
                        genealogy=genealogy[index],
                        snapshot=snapshot_value,
                        strict=True,
                    )
                rows.append(row)
    except TotalAnalysisError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise TotalAnalysisError(f"invalid rollouts.npz: {directory}") from exc
    return rows


def _family_rows_from_robotwin(
    directory: Path,
    *,
    substrate: str,
    protocol: ProtocolAuthority | None = None,
    strict: bool = False,
) -> list[dict[str, Any]]:
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
    if strict:
        annotations = [
            value
            for value in (
                metadata.get("substrate_annotation"),
                marker.get("substrate_annotation"),
            )
            if value is not None
        ]
        if substrate == "C" and any(value != "WEAK_SUBSTRATE" for value in annotations):
            raise TotalAnalysisError(f"C family lacks WEAK_SUBSTRATE annotation: {directory}")
        if substrate == "C" and not annotations:
            raise TotalAnalysisError(f"C family lacks WEAK_SUBSTRATE annotation: {directory}")
        if substrate in {"A", "B"} and any(value not in (None, "") for value in annotations):
            raise TotalAnalysisError(f"{substrate} family has an unexpected weak-substrate annotation: {directory}")
    if protocol is not None:
        _verify_protocol_identity(marker, protocol, label=f"{directory} marker")
        _verify_protocol_identity(metadata, protocol, label=f"{directory} metadata")
    snapshot_path = directory / "SNAPSHOT.json"
    family_snapshot: Mapping[str, Any] | None = None
    if snapshot_path.is_file():
        family_snapshot = _read_json(snapshot_path, label="Robotwin snapshot")
        _verify_snapshot(family_snapshot, label=str(snapshot_path))
    else:
        inline_snapshot = payload.get("snapshot")
        if not isinstance(inline_snapshot, Mapping):
            raise TotalAnalysisError(f"Robotwin family lacks full snapshot: {directory}")
        family_snapshot = inline_snapshot
        _verify_snapshot(family_snapshot, label=f"{directory} inline snapshot")
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
        if strict:
            task_name = metadata.get("task_name", candidate.get("task_name"))
            if task_name not in A_TASKS:
                raise TotalAnalysisError(f"Robotwin family task_id/task_name is not one of the frozen tasks: {directory}")
            task_id = A_TASKS.index(str(task_name))
            state_match = re.search(r"(?:state|family)-0*(\d+)", str(init_state))
            if state_match is None:
                state_match = re.search(r"(?:state|family)-0*(\d+)", str(family_id))
            if state_match is None:
                raise TotalAnalysisError(f"Robotwin family lacks a canonical init_state: {directory}")
            init_state = int(state_match.group(1))
            rank_match = re.search(r"(?:^|/)rank-0*(\d+)(?:/|$)", str(directory))
            if rank_match is None:
                raise TotalAnalysisError(f"Robotwin family path lacks rank identity: {directory}")
            rank = int(rank_match.group(1))
        else:
            rank = metadata.get("rank")
        candidate_snapshot: Mapping[str, Any] | None = None
        if isinstance(candidate.get("snapshot"), Mapping):
            candidate_snapshot = candidate["snapshot"]
        elif isinstance(payload.get("snapshots"), Mapping):
            candidate_snapshot = payload["snapshots"].get(str(candidate.get("candidate_id")))
        if strict and candidate_snapshot is None:
            raise TotalAnalysisError(f"Robotwin candidate {family_id}/{index} lacks per-candidate snapshot")
        if candidate_snapshot is not None:
            _verify_snapshot(candidate_snapshot, label=f"{directory} candidate {index}", per_candidate=True)
        row = {
            **dict(candidate),
            "family_id": payload.get("family_id", marker.get("family_id")),
            "task_id": task_id,
            "init_state": init_state,
            "genealogy": genealogy[index],
        }
        if rank is not None:
            row["rank"] = rank
            row["world_size"] = metadata.get("world_size", EXPECTED_WORLD_SIZE)
        replay = candidate.get("snapshot_restore_check")
        if replay is None and isinstance(candidate_snapshot, Mapping):
            replay = candidate_snapshot.get("snapshot_restore_check")
        if replay is not None:
            row["snapshot_restore_check"] = replay
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
        if strict:
            row = _normalise_row(
                row,
                family={**metadata, "family_id": family_id, "task_id": task_id, "init_state": init_state, "rank": rank, "world_size": EXPECTED_WORLD_SIZE},
                index=index,
                substrate=substrate,
                genealogy=genealogy[index],
                snapshot=candidate_snapshot,
                strict=True,
            )
        rows.append(row)
    # The robotwin payload itself is the raw row source; normalization below
    # performs the candidate/compute/termination checks.
    return rows


def _expected_main_marker(substrate: str) -> str:
    return "completed_stage_s_a_evaluation" if substrate == "A" else f"completed_stage_s_{substrate.lower()}_main_evaluation"


def _verify_main_root(root: Path, *, substrate: str, protocol: ProtocolAuthority) -> tuple[Mapping[str, Any], dict[str, str]]:
    """Verify the immutable top-level main-screen completion boundary."""

    marker = _read_json(root / TOTAL_COMPLETION_FILE, label=f"{substrate} main completion")
    if marker.get("status") != "COMPLETED" or marker.get("marker_type") != _expected_main_marker(substrate):
        raise TotalAnalysisError(f"{substrate} main root completion is not a terminal audited marker: {root}")
    if marker.get("protocol_id") not in (None, PROTOCOL_ID):
        raise TotalAnalysisError(f"{substrate} main completion protocol_id drifted")
    if marker.get("substrate") not in (None, substrate):
        raise TotalAnalysisError(f"{substrate} main completion substrate identity drifted")
    manifest = _verify_manifest(root, required=True)
    if TOTAL_COMPLETION_FILE not in manifest:
        raise TotalAnalysisError(f"{substrate} main completion marker is not bound by root SHA256SUMS: {root}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink() and path.name != "SHA256SUMS"
    }
    if actual != set(manifest):
        missing = sorted(actual - set(manifest))
        extra = sorted(set(manifest) - actual)
        raise TotalAnalysisError(f"{substrate} root SHA256SUMS does not cover exactly the audited files: missing={missing[:3]} extra={extra[:3]}")

    def integer_field(*names: str) -> int:
        for name in names:
            if name in marker:
                try:
                    return int(marker[name])
                except (TypeError, ValueError) as exc:
                    raise TotalAnalysisError(f"{substrate} main completion field {name} is not an integer") from exc
        raise TotalAnalysisError(f"{substrate} main completion lacks one of {names}")

    if integer_field("world_size") != EXPECTED_WORLD_SIZE:
        raise TotalAnalysisError(f"{substrate} main world_size is not {EXPECTED_WORLD_SIZE}")
    if integer_field("family_count") != EXPECTED_FAMILY_COUNT:
        raise TotalAnalysisError(f"{substrate} main family_count is not {EXPECTED_FAMILY_COUNT}")
    if integer_field("candidate_budget") != EXPECTED_CANDIDATES:
        raise TotalAnalysisError(f"{substrate} main candidate_budget is not {EXPECTED_CANDIDATES}")
    if integer_field("terminal_candidate_count") != EXPECTED_TOTAL_CANDIDATES:
        raise TotalAnalysisError(f"{substrate} main terminal_candidate_count is not {EXPECTED_TOTAL_CANDIDATES}")
    if "task_count" in marker and integer_field("task_count") != 10:
        raise TotalAnalysisError(f"{substrate} main task_count is not 10")
    if "initial_state_count" in marker and integer_field("initial_state_count") != 16:
        raise TotalAnalysisError(f"{substrate} main initial_state_count is not 16")
    if "families_per_task" in marker and integer_field("families_per_task") != 16:
        raise TotalAnalysisError(f"{substrate} main families_per_task is not 16")
    if substrate == "A":
        tasks = marker.get("tasks")
        if tasks is not None and list(tasks) != list(A_TASKS):
            raise TotalAnalysisError("A main task list disagrees with the frozen ten-task order")
    if substrate == "C" and marker.get("substrate_annotation") != "WEAK_SUBSTRATE":
        raise TotalAnalysisError("C main completion lacks WEAK_SUBSTRATE annotation")
    if substrate in {"A", "B"} and marker.get("substrate_annotation") not in (None, ""):
        raise TotalAnalysisError(f"{substrate} main completion has an unexpected weak-substrate annotation")
    declared_commit = marker.get("protocol_git_commit")
    if declared_commit != protocol.git_commit:
        raise TotalAnalysisError(f"{substrate} main completion protocol git commit drifted")
    declared_authority_sha = marker.get("protocol_authority_sha256", marker.get("protocol_json_sha256", marker.get("protocol_acceptance_sha256")))
    if declared_authority_sha != protocol.sha256:
        raise TotalAnalysisError(f"{substrate} main completion protocol authority SHA-256 drifted")
    declared_pipeline = marker.get("pipeline_identity", marker.get("pipeline_commit", marker.get("source_commit")))
    _require_sha(declared_pipeline, length=40, label=f"{substrate} main pipeline identity")
    ranks = marker.get("ranks")
    if not isinstance(ranks, list) or len(ranks) != EXPECTED_WORLD_SIZE:
        raise TotalAnalysisError(f"{substrate} main completion must list all eight ranks")
    rank_values: list[int] = []
    for rank in ranks:
        if not isinstance(rank, Mapping):
            raise TotalAnalysisError(f"{substrate} main rank summary is not an object")
        try:
            rank_values.append(int(rank.get("rank", -1)))
        except (TypeError, ValueError) as exc:
            raise TotalAnalysisError(f"{substrate} main rank identity is invalid") from exc
    if sorted(rank_values) != list(range(EXPECTED_WORLD_SIZE)):
        raise TotalAnalysisError(f"{substrate} main completion rank coverage is not 0..7")
    return marker, manifest


def _family_n32(
    directory: Path,
    *,
    marker: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    family_file: Path,
) -> N32Family:
    return N32Family(
        family_id=str(marker.get("family_id", metadata.get("family_id", rows[0].get("family_id", "")))),
        directory=directory,
        marker=dict(marker),
        metadata=dict(metadata),
        candidates=tuple(_jsonable(row) for row in rows),
        source_marker_sha256=_sha256(directory / "COMPLETED_FAMILY.json"),
        source_bundle_sha256=_sha256(directory / "SHA256SUMS"),
        source_family_file_sha256=_sha256(family_file),
    )


def _load_audited_main_directory(
    source: Path,
    *,
    substrate: str,
    protocol: ProtocolAuthority,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    marker, manifest = _verify_main_root(source, substrate=substrate, protocol=protocol)
    family_jsons = sorted(path for path in source.rglob("family.json") if path.is_file() and not path.is_symlink())
    npzs = sorted(path for path in source.rglob("rollouts.npz") if path.is_file() and not path.is_symlink())
    if substrate == "A":
        if not family_jsons or npzs:
            raise TotalAnalysisError("A main audited root must contain only family.json terminal artifacts")
        family_paths = family_jsons
    else:
        if not npzs or family_jsons:
            raise TotalAnalysisError(f"{substrate} main audited root must contain only rollouts.npz terminal artifacts")
        family_paths = npzs
    if len(family_paths) != EXPECTED_FAMILY_COUNT:
        raise TotalAnalysisError(f"{substrate} main contains {len(family_paths)} families, expected {EXPECTED_FAMILY_COUNT}")
    try:
        # Keep the exact source candidate mappings used for S5's immutable
        # base digest.  The analyzer's normalized rows intentionally add
        # derived fields (task/state/compute aliases), so rebuilding an
        # N32Family from those rows would make an otherwise valid S5 base
        # appear rewritten.
        discovered = {
            family.family_id: family
            for family in discover_n32_families(source, protocol=protocol)
        }
    except S45Error as exc:
        raise TotalAnalysisError(f"{substrate} main N32 family discovery failed: {source}: {exc}") from exc
    if len(discovered) != EXPECTED_FAMILY_COUNT:
        raise TotalAnalysisError(f"{substrate} discovered {len(discovered)} families, expected {EXPECTED_FAMILY_COUNT}")
    raw: list[dict[str, Any]] = []
    families: list[N32Family] = []
    seen_ids: set[str] = set()
    for family_path in family_paths:
        directory = family_path.parent
        family_marker = _read_json(directory / "COMPLETED_FAMILY.json", label=f"{substrate} family completion")
        _verify_family_manifest(directory, family_marker, marker_name="COMPLETED_FAMILY.json")
        if substrate == "A":
            rows = _family_rows_from_robotwin(directory, substrate=substrate, protocol=protocol, strict=True)
            payload = _read_json(family_path, label="A family result")
            metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
        else:
            rows = _family_rows_from_npz(directory, substrate=substrate, protocol=protocol, strict=True)
            metadata = _read_json(directory / "metadata.json", label=f"{substrate} family metadata")
        if not rows:
            raise TotalAnalysisError(f"{substrate} family has no candidate rows: {directory}")
        family_id = str(rows[0]["family_id"])
        if family_id in seen_ids or str(family_marker.get("family_id")) != family_id:
            raise TotalAnalysisError(f"{substrate} family identity is duplicated or inconsistent: {directory}")
        seen_ids.add(family_id)
        source_family = discovered.get(family_id)
        if source_family is None or source_family.directory.resolve() != directory.resolve():
            raise TotalAnalysisError(f"{substrate} family source discovery/path mismatch: {directory}")
        if substrate in {"B", "C"}:
            try:
                task_id = int(metadata["task_id"])
                init_state = int(metadata.get("init_state", metadata["initial_state_id"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise TotalAnalysisError(f"{substrate} family metadata lacks task_id/init_state: {directory}") from exc
            expected_dir = source / substrate / f"task{task_id:02d}" / f"init{init_state:03d}"
            if directory.resolve() != expected_dir.resolve() or family_id != f"task{task_id:02d}_init{init_state:03d}":
                raise TotalAnalysisError(f"{substrate} family path/id is not canonical: {directory}")
            family_rank = metadata.get("rank")
            family_world = metadata.get("world_size")
            if family_rank is None or family_world is None:
                raise TotalAnalysisError(f"{substrate} family lacks rank/world-size metadata: {directory}")
        else:
            task_name = metadata.get("task_name")
            if task_name not in A_TASKS:
                raise TotalAnalysisError(f"A family task_name is not in the frozen task order: {directory}")
            try:
                init_state = int(rows[0]["init_state"])
                family_rank = int(rows[0]["rank"])
                family_world = int(rows[0]["world_size"])
            except (KeyError, TypeError, ValueError) as exc:
                raise TotalAnalysisError(f"A family lacks canonical task/state/rank metadata: {directory}") from exc
            expected_dir = source / f"rank-{family_rank:04d}" / str(task_name) / f"family-{init_state:04d}"
            expected_id = f"{task_name}/family-{init_state:04d}"
            if directory.resolve() != expected_dir.resolve() or family_id != expected_id:
                raise TotalAnalysisError(f"A family path/id is not canonical: {directory}")
            if family_world != EXPECTED_WORLD_SIZE:
                raise TotalAnalysisError(f"A family world_size is not {EXPECTED_WORLD_SIZE}: {directory}")
        families.append(source_family)
        raw.extend(rows)
    verified_rows, summary = _verify_records(
        raw,
        substrate=substrate,
        verification={"terminal_markers": True, "sha256": True, "genealogy": True, "compute": True},
        expected_family_count=EXPECTED_FAMILY_COUNT,
        strict=True,
    )
    summary["pipeline_identity"] = marker.get("pipeline_identity", marker.get("pipeline_commit", marker.get("source_commit")))
    summary["root_manifest_sha256"] = _sha256(source / "SHA256SUMS")
    summary["_families"] = tuple(families)
    summary["_completion"] = dict(marker)
    return verified_rows, summary


def _load_main_path(
    value: str | Path,
    *,
    substrate: str,
    protocol: ProtocolAuthority | None = None,
    strict: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = _strict_path(value, label=f"{substrate} main root")
    if strict:
        if protocol is None:
            raise TotalAnalysisError(f"{substrate} strict main loading requires a frozen protocol")
        if not source.is_dir():
            raise TotalAnalysisError(f"{substrate} production main input must be an audited directory root")
        return _load_audited_main_directory(source, substrate=substrate, protocol=protocol)
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


def _load_s4_path(
    root: Path,
    *,
    protocol: ProtocolAuthority,
    expected_family_ids: Sequence[str],
    expected_families: Sequence[N32Family] | None = None,
) -> list[Mapping[str, Any]]:
    if not expected_family_ids:
        return []
    try:
        return list(
            load_s4_probes(
                root,
                protocol=protocol,
                expected_family_ids=expected_family_ids,
                expected_families=expected_families,
            )
        )
    except S45Error as exc:
        raise TotalAnalysisError(f"S4 persisted probe verification failed: {root}: {exc}") from exc


def _load_s5_path_legacy(
    root: Path,
    *,
    protocol: ProtocolAuthority,
    expected_family_ids: Sequence[str],
) -> dict[str, list[Mapping[str, Any]]]:
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


def _load_s5_path(
    root: Path,
    *,
    protocol: ProtocolAuthority,
    expected_family_ids: Sequence[str],
    expected_families: Sequence[N32Family] | None = None,
    strict: bool = False,
) -> dict[str, list[Mapping[str, Any]]]:
    """Load S5 through the canonical runtime validator in production.

    The legacy parser remains available for the non-production in-memory
    compatibility API.  Production must reuse ``load_s5_extended`` so that
    immutable base rows, source family SHAs, the complete 64-candidate
    extension and the fresh-candidate replay/termination contract are checked
    by the same validator as the S4/S5 runtime.
    """

    if not strict:
        return _load_s5_path_legacy(
            root,
            protocol=protocol,
            expected_family_ids=expected_family_ids,
        )
    if expected_families is None:
        raise TotalAnalysisError("strict S5 loading requires the audited main N32 families")
    if {str(family.family_id) for family in expected_families} != {str(value) for value in expected_family_ids}:
        raise TotalAnalysisError("strict S5 expected family IDs and audited families disagree")
    try:
        _base_rows, extended_rows = load_s5_extended(
            root,
            protocol=protocol,
            families=list(expected_families),
        )
    except S45Error as exc:
        raise TotalAnalysisError(f"S5 persisted extension verification failed: {root}: {exc}") from exc
    return {str(family_id): list(rows) for family_id, rows in extended_rows.items()}


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


def _extract_arm_inputs(
    name: str,
    arm: Any,
    *,
    protocol: ProtocolAuthority,
    strict: bool = False,
) -> tuple[list[dict[str, Any]], Any, Any, str | None, dict[str, Any]]:
    if not isinstance(arm, Mapping):
        raise TotalAnalysisError(f"arm {name} must be an object")
    substrate = str(arm.get("substrate", name))
    if substrate not in EXPECTED_SUBSTRATES:
        raise TotalAnalysisError(f"unsupported Stage-S arm {name!r}")
    if strict:
        # Production accepts only path references to the audited terminal
        # bundles.  In particular, a caller cannot smuggle an in-memory list
        # or a hand-written verification flag alongside an otherwise valid
        # path and have it override the audited rows.
        allowed = {
            "substrate",
            "main_root",
            "main",
            "s4_root",
            "s5_root",
            "pipeline_identity",
            "pipeline_commit",
            "source_commit",
        }
        unexpected = sorted(str(key) for key in arm if key not in allowed)
        if unexpected:
            raise TotalAnalysisError(f"production {name} arm contains unsupported/in-memory fields: {unexpected}")
        forbidden = {
            "records",
            "rollouts",
            "rows",
            "probes",
            "s4_probes",
            "extended_rollouts",
            "s5_rollouts",
            "artifact_verification",
        }
        present = sorted(key for key in forbidden if key in arm)
        if present:
            raise TotalAnalysisError(f"production {name} arm contains in-memory/override fields: {present}")
        if arm.get("main_root", arm.get("main")) is None:
            raise TotalAnalysisError(f"production {name} arm lacks audited main_root")
        if arm.get("s4_root") is None or arm.get("s5_root") is None:
            raise TotalAnalysisError(f"production {name} arm requires audited s4_root and s5_root")
    records = arm.get("records", arm.get("rollouts"))
    verification = arm.get("artifact_verification")
    families: tuple[N32Family, ...] | None = None
    if records is None:
        main = arm.get("main_root", arm.get("main"))
        if main is None:
            raise TotalAnalysisError(f"arm {name} lacks main_root/records")
        records, verification_from_path = _load_main_path(
            main,
            substrate=substrate,
            protocol=protocol,
            strict=strict,
        )
        family_value = verification_from_path.pop("_families", None)
        verification_from_path.pop("_completion", None)
        if family_value is not None:
            families = tuple(family_value)
        verification = {**verification_from_path, **(dict(verification) if isinstance(verification, Mapping) else {})}
    if strict and families is None:
        raise TotalAnalysisError(f"production {name} arm did not produce audited N32 family objects")
    rows, verification_summary = _verify_records(
        records,
        substrate=substrate,
        verification=verification,
        expected_family_count=EXPECTED_FAMILY_COUNT if strict else None,
        strict=strict,
    )
    probes = arm.get("probes", arm.get("s4_probes"))
    if probes is None and arm.get("s4_root") is not None:
        s4_source = _strict_path(arm["s4_root"], label=f"{name} S4 root")
        if s4_source.is_dir():
            near_ids = _near_family_ids(rows)
            near_families = None
            if strict:
                assert families is not None
                near_families = tuple(family for family in families if family.family_id in set(near_ids))
            probes = _load_s4_path(
                s4_source,
                protocol=protocol,
                expected_family_ids=near_ids,
                expected_families=near_families,
            )
        else:
            if strict:
                raise TotalAnalysisError(f"production {name} S4 input must be an audited directory root")
            payload = _load_value(s4_source, label=f"{name} S4 result")
            probes = payload.get("probes", payload.get("families", payload.get("S4_probes"))) if isinstance(payload, Mapping) else payload
    extended = arm.get("extended_rollouts", arm.get("s5_rollouts"))
    if extended is None and arm.get("s5_root") is not None:
        s5_source = _strict_path(arm["s5_root"], label=f"{name} S5 root")
        if s5_source.is_dir():
            family_ids = sorted({str(row["family_id"]) for row in rows})
            extended = _load_s5_path(
                s5_source,
                protocol=protocol,
                expected_family_ids=family_ids,
                expected_families=families,
                strict=strict,
            )
        else:
            if strict:
                raise TotalAnalysisError(f"production {name} S5 input must be an audited directory root")
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
    if strict:
        audited_pipeline = verification_summary.get("pipeline_identity")
        if audited_pipeline is None:
            raise TotalAnalysisError(f"production {name} main root lacks pipeline identity")
        if arm.get("pipeline_identity", arm.get("pipeline_commit", arm.get("source_commit"))) not in (None, audited_pipeline):
            raise TotalAnalysisError(f"production {name} pipeline identity override disagrees with audited marker")
        pipeline = str(audited_pipeline)
        _require_sha(pipeline, length=40, label=f"production {name} pipeline identity")
    return rows, probes, extended, None if pipeline is None else str(pipeline), {**verification_summary, "arm": name}


def _load_controls(value: Any, *, strict: bool = False) -> dict[str, Any]:
    if strict and isinstance(value, Mapping):
        raise TotalAnalysisError("production controls must be an audited directory, not an in-memory mapping")
    if isinstance(value, (str, Path)):
        root = _strict_path(value, label="controls root", directory=True)
        _verify_completion(root, marker_name="COMPLETED_CONTROLS.json", require_status=False)
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
    if strict:
        _require_sha(pipeline, length=40, label="controls pipeline identity")
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


def analyze_stage_s(
    *,
    protocol_path: str | Path,
    controls: Any,
    arms: Mapping[str, Any],
    output_root: str | Path | None = None,
    production: bool = False,
) -> dict[str, Any]:
    """Verify and analyze every Stage-S arm, then optionally publish a bundle.

    ``production=True`` is the audited-directory boundary used by the CLI and
    final publication.  The default keeps the small verified in-memory API
    used by unit tests and legacy callers; it does not weaken the production
    path because the CLI always opts into the strict mode.
    """

    if set(arms) != set(EXPECTED_SUBSTRATES):
        raise TotalAnalysisError(f"Stage-S total analysis requires exactly A/B/C arms, got {sorted(arms)}")
    protocol_validation: dict[str, Any] | None = None
    if production:
        protocol, protocol_validation = _load_production_protocol(protocol_path)
    else:
        protocol = ProtocolAuthority.load(protocol_path)
    control_error: str | None = None
    try:
        controls_payload = _load_controls(controls, strict=production)
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
            rows, probes, extended, identity, verification = _extract_arm_inputs(
                name,
                arms[name],
                protocol=protocol,
                strict=production,
            )
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
    if protocol_validation is not None:
        report["protocol_validation"] = protocol_validation
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
