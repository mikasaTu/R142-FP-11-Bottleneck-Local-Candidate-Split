from __future__ import annotations

"""Blinded calibration and unblinded Phase-1R analysis."""

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .phase1 import (
    DECILES,
    DESCENDANTS_PER_CELL,
    LOCATIONS_PER_EPISODE,
    PHASE1_PROTOCOL_ID,
    STREAMS,
    TASKS,
    _atomic_text,
    cell_path,
    validate_cell,
)
from .protocol import atomic_json, sha256_file


def _load_success_matrix(
    root: str | Path,
    *,
    suite: str,
    task_id: int,
    episodes: int = 12,
    stream: str = "heldout",
    require_owner: tuple[int, int] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = np.empty((int(episodes), LOCATIONS_PER_EPISODE, DESCENDANTS_PER_CELL), dtype=np.float64)
    metadata_rows: list[dict[str, Any]] = []
    for episode in range(int(episodes)):
        for location in range(LOCATIONS_PER_EPISODE):
            directory = cell_path(root, suite, task_id, episode, location, stream)
            ok, errors, metadata = validate_cell(
                directory,
                suite=suite,
                task_id=task_id,
                location_index=location,
                stream=stream,
                require_owner=require_owner,
            )
            if not ok or metadata is None:
                raise RuntimeError(f"invalid Phase-1R cell {directory}: {'; '.join(errors)}")
            with np.load(directory / "cell.npz", allow_pickle=False) as data:
                matrix[episode, location] = np.asarray(data["success"], dtype=np.float64)
            metadata_rows.append(metadata)
    return matrix, {
        "cell_count": len(metadata_rows),
        "policy_forwards": int(sum(int(row.get("policy_forwards", 0)) for row in metadata_rows)),
        "environment_steps": int(sum(int(row.get("environment_steps", 0)) for row in metadata_rows)),
        "branch_count": len(metadata_rows) * DESCENDANTS_PER_CELL,
        "metadata": metadata_rows,
    }


def _load_control_matrix(
    root: str | Path,
    kind: str,
    *,
    episodes: int = 12,
    stream: str = "calibration",
    require_owner: tuple[int, int] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = np.empty((int(episodes), LOCATIONS_PER_EPISODE, DESCENDANTS_PER_CELL), dtype=np.float64)
    metadata_rows: list[dict[str, Any]] = []
    for episode in range(int(episodes)):
        for location in range(LOCATIONS_PER_EPISODE):
            directory = Path(root) / kind / f"episode{episode:02d}" / f"location{location:02d}" / stream
            ok, errors, metadata = validate_cell(
                directory,
                suite=f"control_{kind}",
                task_id=0,
                parent_id=f"{kind}_episode{episode:02d}",
                location_index=location,
                stream=stream,
                require_owner=require_owner,
            )
            if not ok or metadata is None:
                raise RuntimeError(f"invalid control cell {directory}: {'; '.join(errors)}")
            with np.load(directory / "cell.npz", allow_pickle=False) as data:
                matrix[episode, location] = np.asarray(data["success"], dtype=np.float64)
            metadata_rows.append(metadata)
    return matrix, {
        "cell_count": len(metadata_rows),
        "policy_forwards": int(sum(int(row.get("policy_forwards", 0)) for row in metadata_rows)),
        "environment_steps": int(sum(int(row.get("environment_steps", 0)) for row in metadata_rows)),
        "branch_count": len(metadata_rows) * DESCENDANTS_PER_CELL,
        "metadata": metadata_rows,
    }


def location_curve(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (LOCATIONS_PER_EPISODE, DESCENDANTS_PER_CELL):
        raise ValueError(f"expected (episodes,10,16) matrix, got {values.shape}")
    return np.mean(values, axis=(0, 2))


def location_summary(curve: np.ndarray) -> dict[str, Any]:
    curve = np.asarray(curve, dtype=np.float64)
    if curve.shape != (LOCATIONS_PER_EPISODE,):
        raise ValueError("curve must have ten locations")
    sensitivity = float(np.max(curve) - np.min(curve))
    cliffs = np.abs(np.diff(curve))
    largest_index = int(np.argmax(cliffs)) if len(cliffs) else None
    midpoint = float((np.max(curve) + np.min(curve)) / 2.0)
    commit = np.flatnonzero(curve < midpoint)
    return {
        "location_sensitivity": sensitivity,
        "largest_adjacent_cliff": None if largest_index is None else float(cliffs[largest_index]),
        "largest_adjacent_cliff_between": None if largest_index is None else [largest_index, largest_index + 1],
        "commit_point": None if len(commit) == 0 else int(commit[0]),
        "commit_point_decile": None if len(commit) == 0 else float(DECILES[int(commit[0])]),
        "r_min": float(np.min(curve)),
        "r_max": float(np.max(curve)),
    }


def paired_episode_bootstrap(matrix_by_task: Iterable[np.ndarray], *, seed: int, replicates: int = 10000) -> dict[str, list[float]]:
    matrices = [np.asarray(value, dtype=np.float64) for value in matrix_by_task]
    if not matrices:
        raise ValueError("at least one task matrix is required")
    if any(value.ndim != 3 or value.shape[1:] != (LOCATIONS_PER_EPISODE, DESCENDANTS_PER_CELL) for value in matrices):
        raise ValueError("invalid task matrix")
    rng = np.random.default_rng(int(seed))
    draws = np.empty((int(replicates), LOCATIONS_PER_EPISODE), dtype=np.float64)
    for replicate in range(int(replicates)):
        task_curves = []
        for matrix in matrices:
            indices = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
            task_curves.append(location_curve(matrix[indices]))
        draws[replicate] = np.mean(np.asarray(task_curves), axis=0)
    low, high = np.quantile(draws, [0.025, 0.975], axis=0)
    return {"low": [float(value) for value in low], "high": [float(value) for value in high]}


def _matrix_hash(matrix: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(matrix, dtype=np.float64).tobytes()).hexdigest()


def calibrate_phase1r(
    controls_root: str | Path,
    output_file: str | Path,
    *,
    shuffles: int = 1000,
    seed: int = 0x142F011,
    require_owner: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Build the pre-unblinding threshold artifact.

    The artifact contains only permutation distributions and hashes.  It does
    not contain an unpermuted R(t) curve, which keeps natural headline results
    blinded until the calibrator has been committed.
    """

    if int(shuffles) < 1000:
        raise ValueError("at least 1000 permutation shuffles are required")
    matrices = {}
    accounting = {}
    for kind in ("positive", "null"):
        matrix, meta = _load_control_matrix(controls_root, kind, stream="calibration", require_owner=require_owner)
        matrices[kind] = matrix
        accounting[kind] = {key: value for key, value in meta.items() if key != "metadata"}
    rng = np.random.default_rng(int(seed))
    null_distributions: dict[str, list[float]] = {}
    for kind, matrix in matrices.items():
        samples = np.empty(int(shuffles), dtype=np.float64)
        for index in range(int(shuffles)):
            shuffled = np.empty_like(matrix)
            for episode in range(matrix.shape[0]):
                shuffled[episode] = matrix[episode, rng.permutation(matrix.shape[1]), :]
            curve = location_curve(shuffled)
            samples[index] = float(np.max(curve) - np.min(curve))
        null_distributions[kind] = [float(value) for value in samples]
    thresholds = {kind: float(np.quantile(values, 0.95)) for kind, values in null_distributions.items()}
    threshold = float(max(thresholds.values()))
    payload = {
        "protocol_id": PHASE1_PROTOCOL_ID,
        "artifact": "BLINDED_PHASE1R_CALIBRATION",
        "shuffles": int(shuffles),
        "seed": int(seed),
        "quantile": 0.95,
        "permutation_unit": "within_episode_branch_location_labels",
        "thresholds_by_control": thresholds,
        "location_sensitivity_threshold": threshold,
        "matrix_shapes": {kind: list(matrix.shape) for kind, matrix in matrices.items()},
        "matrix_hashes": {kind: _matrix_hash(matrix) for kind, matrix in matrices.items()},
        "accounting": accounting,
        "permutation_samples": null_distributions,
        "unpermuted_curve_present": False,
        "natural_curve_present": False,
    }
    atomic_json(output_file, payload)
    return payload


def _validate_calibration(payload: dict[str, Any]) -> None:
    if payload.get("protocol_id") != PHASE1_PROTOCOL_ID:
        raise RuntimeError("calibration protocol mismatch")
    if payload.get("unpermuted_curve_present") or payload.get("natural_curve_present"):
        raise RuntimeError("calibration artifact is not blinded")
    if "curve" in payload or "curves" in payload or "natural" in payload:
        raise RuntimeError("calibration artifact contains unblinded curve data")
    if int(payload.get("shuffles", 0)) < 1000:
        raise RuntimeError("calibration shuffles below frozen minimum")
    if "location_sensitivity_threshold" not in payload:
        raise RuntimeError("missing frozen threshold")


def analyze_phase1r(
    natural_root: str | Path,
    controls_root: str | Path,
    calibration_file: str | Path,
    output_dir: str | Path,
    *,
    bootstrap_replicates: int = 10000,
    require_owner: tuple[int, int] | None = None,
) -> dict[str, Any]:
    calibration_path = Path(calibration_file)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    _validate_calibration(calibration)
    natural_matrices: dict[tuple[str, int], np.ndarray] = {}
    task_accounting: dict[str, Any] = {}
    for suite, task_id in TASKS:
        matrix, meta = _load_success_matrix(natural_root, suite=suite, task_id=task_id, stream="heldout", require_owner=require_owner)
        natural_matrices[(suite, task_id)] = matrix
        task_accounting[f"{suite}_task{task_id:02d}"] = {key: value for key, value in meta.items() if key != "metadata"}
    matrices = list(natural_matrices.values())
    aggregate = np.concatenate(matrices, axis=0)
    pooled_curve = location_curve(aggregate)
    task_rows = []
    for (suite, task_id), matrix in natural_matrices.items():
        curve = location_curve(matrix)
        row = {"suite": suite, "task_id": int(task_id), "curve": [float(value) for value in curve]}
        row.update(location_summary(curve))
        row["threshold"] = float(calibration["location_sensitivity_threshold"])
        row["nonflat_above_threshold"] = bool(row["location_sensitivity"] > row["threshold"])
        task_rows.append(row)
    bootstrap = paired_episode_bootstrap(matrices, seed=0x142F011, replicates=int(bootstrap_replicates))
    controls = {}
    control_accounting = {}
    for kind in ("positive", "null"):
        matrix, meta = _load_control_matrix(controls_root, kind, stream="heldout", require_owner=require_owner)
        curve = location_curve(matrix)
        controls[kind] = {"curve": [float(value) for value in curve], **location_summary(curve)}
        controls[kind]["threshold"] = float(calibration["thresholds_by_control"][kind])
        controls[kind]["nonflat_above_threshold"] = bool(controls[kind]["location_sensitivity"] > controls[kind]["threshold"])
        control_accounting[kind] = {key: value for key, value in meta.items() if key != "metadata"}
    summary = {
        "protocol_id": PHASE1_PROTOCOL_ID,
        "decision_label": "PHASE1R_OBSERVED_NONFLATNESS_ONLY",
        "calibration_file": str(calibration_path),
        "calibration_sha256": sha256_file(calibration_path),
        "task_count": len(task_rows),
        "selected_episodes_per_task": 12,
        "descendants_per_location": DESCENDANTS_PER_CELL,
        "location_count": LOCATIONS_PER_EPISODE,
        "pooled_curve": [float(value) for value in pooled_curve],
        "pooled_location_summary": location_summary(pooled_curve),
        "pooled_threshold": float(calibration["location_sensitivity_threshold"]),
        "pooled_nonflat_above_threshold": bool(location_summary(pooled_curve)["location_sensitivity"] > calibration["location_sensitivity_threshold"]),
        "bootstrap_replicates": int(bootstrap_replicates),
        "pooled_bootstrap_ci95": bootstrap,
        "task_rows": task_rows,
        "controls": controls,
        "compute": {
            "natural": task_accounting,
            "controls": control_accounting,
            "natural_policy_forwards": int(sum(row["policy_forwards"] for row in task_accounting.values())),
            "natural_environment_steps": int(sum(row["environment_steps"] for row in task_accounting.values())),
            "natural_branch_count": int(sum(row["branch_count"] for row in task_accounting.values())),
            "budget_slack": 0,
        },
        "evidence_boundary": "Phase-1R tests branch-location recoverability only; it does not compare bottleneck-local split with candidate-sampling baselines.",
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "phase1r_summary.json"
    atomic_json(summary_path, summary)
    completed = {
        "protocol_id": PHASE1_PROTOCOL_ID,
        "summary": summary_path.name,
        "summary_sha256": sha256_file(summary_path),
        "calibration_sha256": sha256_file(calibration_path),
        "task_count": len(task_rows),
        "natural_cells": len(task_rows) * 12 * 10,
        "control_cells": 2 * 12 * 10,
        "bootstrap_replicates": int(bootstrap_replicates),
        "unblinded": True,
        "checkpoint": "CHECKPOINT_2_STOP",
    }
    atomic_json(output / "COMPLETED_PHASE1R.json", completed)
    manifest_path = output / "SHA256SUMS"
    _atomic_text(
        manifest_path,
        "".join(
            f"{sha256_file(path)}  {path.name}\n"
            for path in (summary_path, output / "COMPLETED_PHASE1R.json")
        ),
    )
    return summary


def validate_phase1_analysis(output_dir: str | Path) -> tuple[bool, list[str]]:
    root = Path(output_dir)
    errors = []
    summary_path = root / "phase1r_summary.json"
    marker_path = root / "COMPLETED_PHASE1R.json"
    sums_path = root / "SHA256SUMS"
    if not summary_path.is_file() or not marker_path.is_file() or not sums_path.is_file():
        return False, ["missing phase1 completion artifacts"]
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if summary.get("protocol_id") != PHASE1_PROTOCOL_ID:
            errors.append("summary protocol mismatch")
        if marker.get("summary_sha256") != sha256_file(summary_path):
            errors.append("summary SHA mismatch")
        if marker.get("task_count") != 40:
            errors.append("task count mismatch")
        if marker.get("natural_cells") != 40 * 12 * 10:
            errors.append("natural cell count mismatch")
        if marker.get("checkpoint") != "CHECKPOINT_2_STOP":
            errors.append("missing CHECKPOINT_2 marker")
        expected_sums = {
            summary_path.name: sha256_file(summary_path),
            marker_path.name: sha256_file(marker_path),
        }
        observed_sums: dict[str, str] = {}
        for line in sums_path.read_text(encoding="utf-8").splitlines():
            digest, relative = line.split("  ", 1)
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                errors.append(f"unsafe analysis checksum entry: {relative}")
            else:
                observed_sums[str(relative_path)] = digest
        if observed_sums != expected_sums:
            errors.append("analysis SHA256SUMS mismatch")
    except Exception as exc:
        errors.append(f"parse error: {type(exc).__name__}: {exc}")
    return not errors, errors
