#!/usr/bin/env python3
"""Fail-closed aggregation for the Stage-S B/C LIBERO main screen.

The eight foreground ranks share one run directory.  A rank marker is written
only after its assigned families are complete; this verifier then checks every
family, genealogy, trajectory/accounting array, full replay snapshot, and
rank assignment before publishing the top-level evaluation result and SHA
manifest.  It never turns a partial or synthetic screen into completion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from r142_stage_s.libero import (
    MAIN_CANDIDATE_COUNT,
    MAIN_INITIAL_STATE_COUNT,
    STAGE_S_PROTOCOL_ID,
    TASK_COUNT,
    family_is_complete,
)


WORLD_SIZE = 8
FAMILY_COUNT = TASK_COUNT * MAIN_INITIAL_STATE_COUNT
SUBSTRATES = {"B", "C"}


class MainEvaluationError(RuntimeError):
    """The main screen is incomplete, inconsistent, or not replay-audited."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise MainEvaluationError(f"required JSON is missing or symlinked: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise MainEvaluationError(f"invalid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise MainEvaluationError(f"JSON root must be an object: {path}")
    return value


def _verify_manifest(root: Path, manifest: Path, *, allow_self: bool = False) -> None:
    if manifest.is_symlink() or not manifest.is_file():
        raise MainEvaluationError(f"checksum manifest is missing or symlinked: {manifest}")
    seen: set[str] = set()
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        pieces = raw.split(None, 1)
        if len(pieces) != 2:
            raise MainEvaluationError(f"malformed checksum line: {manifest}")
        expected, relative = pieces
        relative_path = Path(relative.lstrip(" *"))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise MainEvaluationError(f"unsafe checksum path in {manifest}: {relative}")
        key = relative_path.as_posix()
        if key in seen:
            raise MainEvaluationError(f"duplicate checksum path in {manifest}: {key}")
        seen.add(key)
        target = manifest.parent / relative_path
        if target.is_symlink() or not target.is_file() or _sha256(target) != expected:
            raise MainEvaluationError(f"checksum mismatch: {target}")
    if not seen:
        raise MainEvaluationError(f"empty checksum manifest: {manifest}")
    if not allow_self and manifest.name in seen:
        raise MainEvaluationError(f"manifest self-hash is forbidden: {manifest}")


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _expected_pairs(rank: int) -> tuple[tuple[int, int], ...]:
    pairs = tuple((task, state) for task in range(TASK_COUNT) for state in range(MAIN_INITIAL_STATE_COUNT))
    return pairs[rank::WORLD_SIZE]


def _verify_snapshot(path: Path, *, expected_candidates: int) -> None:
    if path.is_symlink() or not path.is_file():
        raise MainEvaluationError(f"snapshot bundle is missing or symlinked: {path}")
    try:
        with path.open("rb") as stream:
            payload = pickle.load(stream)
    except Exception as exc:  # noqa: BLE001 - fail closed on any untrusted artifact
        raise MainEvaluationError(f"invalid snapshot bundle: {path}") from exc
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise MainEvaluationError(f"snapshot schema drifted: {path}")
    candidates = payload.get("candidates")
    if not isinstance(candidates, Mapping) or len(candidates) != expected_candidates:
        raise MainEvaluationError(f"snapshot candidate count drifted: {path}")
    for candidate_id, snapshot in candidates.items():
        if not isinstance(snapshot, Mapping):
            raise MainEvaluationError(f"snapshot entry is not an object: {path} candidate={candidate_id}")
        if snapshot.get("environment") is None or snapshot.get("observation_history") is None:
            raise MainEvaluationError(f"snapshot lacks simulator/history state: {path} candidate={candidate_id}")
        if "action_queue" not in snapshot or "python_rng_state" not in snapshot or "numpy_rng_state" not in snapshot:
            raise MainEvaluationError(f"snapshot lacks queue/Python/NumPy RNG state: {path} candidate={candidate_id}")
        if snapshot.get("policy_rng_state") is None:
            raise MainEvaluationError(f"snapshot lacks policy RNG state: {path} candidate={candidate_id}")
        torch_state = snapshot.get("torch_rng_state")
        if not isinstance(torch_state, Mapping) or "cpu" not in torch_state or "cuda" not in torch_state:
            raise MainEvaluationError(f"snapshot lacks Torch CPU/CUDA RNG state: {path} candidate={candidate_id}")


def _verify_family(
    root: Path,
    *,
    substrate: str,
    task: int,
    state: int,
    rank: int,
    source_commit: str,
    calibration_report: str,
) -> Mapping[str, Any]:
    directory = root / substrate / f"task{task:02d}" / f"init{state:03d}"
    if not family_is_complete(directory, expected_candidates=MAIN_CANDIDATE_COUNT):
        raise MainEvaluationError(f"family is incomplete or hash-invalid: {directory}")
    metadata = _read_json(directory / "metadata.json")
    expected = {
        "protocol_id": STAGE_S_PROTOCOL_ID,
        "substrate": substrate,
        "task_id": task,
        "init_state": state,
        "candidate_count": MAIN_CANDIDATE_COUNT,
        "rank": rank,
        "world_size": WORLD_SIZE,
        "source_commit": source_commit,
        "termination": "official eval_success or step_limit",
        "replay_gate": "restore -> same action -> next-state <= 1e-9",
        "policy_rng_streams": "python_numpy_torch_cpu_torch_cuda_policy",
        "calibration_report": calibration_report,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise MainEvaluationError(f"family metadata drifted at {directory}: {key}")
    if substrate == "C" and metadata.get("substrate_annotation") != "WEAK_SUBSTRATE":
        raise MainEvaluationError(f"C family missing WEAK_SUBSTRATE annotation: {directory}")
    if substrate == "B" and "substrate_annotation" in metadata:
        raise MainEvaluationError(f"B family has an unexpected substrate annotation: {directory}")

    with np_load(directory / "rollouts.npz") as data:
        success = data["success"]
        if len(success) != MAIN_CANDIDATE_COUNT:
            raise MainEvaluationError(f"rollout candidate count drifted: {directory}")
        successes = int(success.astype("int64").sum())
        policy_forwards = int(data["policy_forwards"].astype("int64").sum())
        environment_steps = int(data["environment_steps"].astype("int64").sum())
    try:
        genealogy = json.loads((directory / "genealogy.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise MainEvaluationError(f"invalid genealogy: {directory}") from exc
    if not isinstance(genealogy, list) or len(genealogy) != MAIN_CANDIDATE_COUNT:
        raise MainEvaluationError(f"genealogy candidate count drifted: {directory}")
    for index, item in enumerate(genealogy):
        if not isinstance(item, Mapping) or item.get("candidate_id") != index or "action_prefix" not in item or "final_success" not in item:
            raise MainEvaluationError(f"genealogy schema drifted: {directory} candidate={index}")
    _verify_snapshot(directory / "snapshots.pkl", expected_candidates=MAIN_CANDIDATE_COUNT)
    return {
        "task": task,
        "init_state": state,
        "rank": rank,
        "path": str(directory.resolve()),
        "successes": successes,
        "candidate_count": MAIN_CANDIDATE_COUNT,
        "policy_forwards": policy_forwards,
        "environment_steps": environment_steps,
        "completion_sha256": _sha256(directory / "COMPLETED_FAMILY.json"),
    }


class np_load:
    """Tiny context manager keeping NumPy optional at module import time."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.value: Any = None

    def __enter__(self) -> Any:
        try:
            import numpy as np
            self.value = np.load(self.path, allow_pickle=False)
            return self.value
        except Exception as exc:  # noqa: BLE001
            raise MainEvaluationError(f"invalid rollouts NPZ: {self.path}") from exc

    def __exit__(self, *_args: object) -> None:
        if self.value is not None:
            self.value.close()


def verify_completed_bundle(root: Path, *, substrate: str) -> Mapping[str, Any]:
    result = _read_json(root / "COMPLETED_EVALUATION_RESULT.json")
    expected_marker = f"completed_stage_s_{substrate.lower()}_main_evaluation"
    if result.get("status") != "COMPLETED" or result.get("marker_type") != expected_marker:
        raise MainEvaluationError("top-level completion marker schema drifted")
    _verify_manifest(root, root / "SHA256SUMS")
    return result


def finalize(
    root: Path,
    *,
    substrate: str,
    run_id: str | None,
    source_commit: str,
    calibration_report: str,
) -> Mapping[str, Any]:
    if substrate not in SUBSTRATES:
        raise MainEvaluationError(f"unsupported substrate: {substrate}")
    root = root.resolve()
    if not root.is_dir() or root.is_symlink():
        raise MainEvaluationError(f"output root is missing or symlinked: {root}")
    completed = root / "COMPLETED_EVALUATION_RESULT.json"
    sums = root / "SHA256SUMS"
    if completed.exists() or sums.exists():
        if not completed.is_file() or not sums.is_file():
            raise MainEvaluationError("partial top-level completion bundle is present")
        return verify_completed_bundle(root, substrate=substrate)
    for forbidden in (f"FAILED_{substrate}_MAIN.json", "REFUSED_DAILY_NO_JOB_WINDOW.json"):
        if (root / forbidden).exists():
            raise MainEvaluationError(f"failed/refused output cannot be finalized: {root / forbidden}")
    rank_outputs: list[Mapping[str, Any]] = []
    for rank in range(WORLD_SIZE):
        marker_path = root / f"COMPLETED_{substrate}_MAIN_RANK-{rank:04d}.json"
        marker = _read_json(marker_path)
        if marker.get("status") != "COMPLETED" or marker.get("marker_type") != "completed_stage_s_main_rank":
            raise MainEvaluationError(f"rank marker is not completed: {marker_path}")
        if marker.get("substrate") != substrate or int(marker.get("rank", -1)) != rank or int(marker.get("world_size", -1)) != WORLD_SIZE:
            raise MainEvaluationError(f"rank marker identity drifted: {marker_path}")
        if marker.get("source_commit") != source_commit or marker.get("calibration_report") != calibration_report:
            raise MainEvaluationError(f"rank marker provenance drifted: {marker_path}")
        if substrate == "C" and marker.get("substrate_annotation") != "WEAK_SUBSTRATE":
            raise MainEvaluationError(f"C rank marker lacks WEAK_SUBSTRATE: {marker_path}")
        pairs = _expected_pairs(rank)
        if int(marker.get("family_count", -1)) != len(pairs) or int(marker.get("candidate_budget", -1)) != MAIN_CANDIDATE_COUNT:
            raise MainEvaluationError(f"rank marker budget drifted: {marker_path}")
        listed = marker.get("families")
        expected_paths = {
            str((root / substrate / f"task{task:02d}" / f"init{state:03d}").resolve())
            for task, state in pairs
        }
        if not isinstance(listed, list) or {str(Path(str(value)).resolve()) for value in listed} != expected_paths:
            raise MainEvaluationError(f"rank family assignment drifted: {marker_path}")
        summary = root / str(marker.get("summary", ""))
        summary_payload = _read_json(summary)
        if summary_payload.get("rank") != rank or summary_payload.get("world_size") != WORLD_SIZE:
            raise MainEvaluationError(f"rank summary identity drifted: {summary}")
        for task, state in pairs:
            rank_outputs.append(
                _verify_family(
                    root,
                    substrate=substrate,
                    task=task,
                    state=state,
                    rank=rank,
                    source_commit=source_commit,
                    calibration_report=calibration_report,
                )
            )
    if len(rank_outputs) != FAMILY_COUNT:
        raise MainEvaluationError(f"expected {FAMILY_COUNT} families, got {len(rank_outputs)}")
    total = sum(int(row["candidate_count"]) for row in rank_outputs)
    successes = sum(int(row["successes"]) for row in rank_outputs)
    result = {
        "status": "COMPLETED",
        "marker_type": f"completed_stage_s_{substrate.lower()}_main_evaluation",
        "protocol_id": STAGE_S_PROTOCOL_ID,
        "run_id": run_id,
        "source_commit": source_commit,
        "substrate": substrate,
        "substrate_annotation": "WEAK_SUBSTRATE" if substrate == "C" else None,
        "calibration_report": calibration_report,
        "calibration_gate": "pooled-only calibration completed and frozen before main screen",
        "no_s2_s5_peeking": True,
        "task_count": TASK_COUNT,
        "initial_state_count": MAIN_INITIAL_STATE_COUNT,
        "world_size": WORLD_SIZE,
        "family_count": len(rank_outputs),
        "candidate_budget": MAIN_CANDIDATE_COUNT,
        "terminal_candidate_count": total,
        "successful_candidate_count": successes,
        "success_rate": successes / total,
        "policy_forwards": sum(int(row["policy_forwards"]) for row in rank_outputs),
        "environment_steps": sum(int(row["environment_steps"]) for row in rank_outputs),
        "termination": "official eval_success or step_limit",
        "replay_gate": "restore -> same action -> next-state <= 1e-9; full simulator/history/queue/Python/NumPy/Torch CPU/CUDA/policy RNG",
        "genealogy": "per-candidate parent_id/generation_step/action_prefix/final_success persisted",
        "trajectory": "per-candidate actions and poses persisted in rollouts.npz",
        "accounting": "policy_forwards and environment_steps per candidate/family/rank/top",
        "ranks": [
            {
                "rank": rank,
                "completion": str(root / f"COMPLETED_{substrate}_MAIN_RANK-{rank:04d}.json"),
                "summary": str(root / f"{substrate}_MAIN_SUMMARY_RANK-{rank:04d}.json"),
                "family_count": len(_expected_pairs(rank)),
            }
            for rank in range(WORLD_SIZE)
        ],
    }
    _write_atomic(completed, (json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode())
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink() or path == sums:
            continue
        if path.name.endswith(".tmp") or path.name.startswith("."):
            raise MainEvaluationError(f"temporary file remains in completed output: {path}")
        files.append(path)
    if not files:
        raise MainEvaluationError("cannot publish an empty main-screen bundle")
    lines = [f"{_sha256(path)}  {path.relative_to(root).as_posix()}" for path in files]
    _write_atomic(sums, ("\n".join(lines) + "\n").encode())
    return verify_completed_bundle(root, substrate=substrate)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--substrate", choices=sorted(SUBSTRATES), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--calibration-report", type=Path, required=True)
    return parser


if __name__ == "__main__":
    try:
        args = build_parser().parse_args()
        report = str(args.calibration_report.resolve())
        value = finalize(
            args.output_root,
            substrate=args.substrate,
            run_id=args.run_id,
            source_commit=args.source_commit,
            calibration_report=report,
        )
    except MainEvaluationError as exc:
        print(json.dumps({"status": "BLOCKED_INCOMPLETE_EVALUATION", "error": str(exc)}))
        raise SystemExit(2)
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))
