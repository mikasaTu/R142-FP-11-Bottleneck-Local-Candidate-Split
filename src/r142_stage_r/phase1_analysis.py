from __future__ import annotations

"""Blinded calibration and unblinded Phase-1R analysis."""

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

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


ANALYSIS_DECISIONS = {
    "PIPELINE_INVALID",
    "NO_TRAJECTORY_BOTTLENECK_ON_PINNED_PI05_LIBERO",
    "TRAJECTORY_NONFLATNESS_DETECTED_CHECKPOINT_2",
}
PERMUTATION_SEED = 0x142F011
REQUIRED_BOOTSTRAP_REPLICATES = 10000


def _stable_seed(*parts: object) -> int:
    """Derive a portable, task-specific RNG seed from the frozen protocol."""

    encoded = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big", signed=False)


def _float64_digest(values: np.ndarray | Iterable[float]) -> str:
    array = np.asarray(values, dtype=np.float64)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _accounting(metadata_rows: Iterable[Mapping[str, Any]], *, branch_count: int) -> dict[str, int]:
    rows = list(metadata_rows)
    logical_forwards = int(sum(int(row.get("policy_forwards", 0)) for row in rows))
    policy_batches = int(sum(int(row.get("policy_batches", 0)) for row in rows))
    physical_batches = int(
        sum(int(row.get("physical_policy_batches", row.get("policy_batches", 0))) for row in rows)
    )
    return {
        "cell_count": len(rows),
        "policy_forwards": logical_forwards,
        "logical_policy_forwards": logical_forwards,
        "policy_batches": policy_batches,
        "physical_policy_batches": physical_batches,
        "environment_steps": int(sum(int(row.get("environment_steps", 0)) for row in rows)),
        "branch_count": int(branch_count),
    }


def _sum_accounting(*accountings: Mapping[str, Any]) -> dict[str, int]:
    keys = (
        "cell_count",
        "policy_forwards",
        "logical_policy_forwards",
        "policy_batches",
        "physical_policy_batches",
        "environment_steps",
        "branch_count",
    )
    return {key: int(sum(int(accounting.get(key, 0)) for accounting in accountings)) for key in keys}

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
    accounting = _accounting(
        metadata_rows,
        branch_count=len(metadata_rows) * DESCENDANTS_PER_CELL,
    )
    accounting["metadata"] = metadata_rows
    return matrix, accounting


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
    accounting = _accounting(
        metadata_rows,
        branch_count=len(metadata_rows) * DESCENDANTS_PER_CELL,
    )
    accounting["metadata"] = metadata_rows
    return matrix, accounting


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



def task_permutation_statistics(
    matrix: np.ndarray,
    suite: str,
    task_id: int,
    *,
    shuffles: int = 1000,
    seed: int = PERMUTATION_SEED,
) -> dict[str, Any]:
    """Compute the frozen, task-local within-episode location-label null.

    A permutation independently shuffles the ten location labels for each
    selected episode, while keeping the sixteen descendant outcomes tied to
    that episode.  The generator is task-keyed and deterministic; its result
    is descriptive audit evidence and never feeds threshold calibration.
    """

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (LOCATIONS_PER_EPISODE, DESCENDANTS_PER_CELL):
        raise ValueError(f"expected (episodes,10,16) matrix, got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("permutation matrix contains non-finite values")
    if int(shuffles) < 1000:
        raise ValueError("at least 1000 task permutations are required")
    observed_curve = location_curve(values)
    observed_sensitivity = float(np.ptp(observed_curve))
    permutation_seed = _stable_seed(
        PHASE1_PROTOCOL_ID,
        "natural",
        "branch-location-permutation",
        suite,
        int(task_id),
        int(seed),
    )
    rng = np.random.default_rng(permutation_seed)
    samples = np.empty(int(shuffles), dtype=np.float64)
    for index in range(int(shuffles)):
        shuffled = np.empty_like(values)
        for episode in range(values.shape[0]):
            shuffled[episode] = values[episode, rng.permutation(values.shape[1]), :]
        samples[index] = float(np.ptp(location_curve(shuffled)))
    p95 = float(np.quantile(samples, 0.95))
    empirical_p = float((np.count_nonzero(samples >= observed_sensitivity) + 1) / (len(samples) + 1))
    digest = _float64_digest(samples)
    return {
        "suite": str(suite),
        "task_id": int(task_id),
        "shuffles": int(shuffles),
        "seed": int(seed),
        "permutation_seed": int(permutation_seed),
        "permutation_unit": "within_episode_branch_location_labels",
        "observed_sensitivity": observed_sensitivity,
        "null_p95": p95,
        "p95": p95,
        "empirical_p_value": empirical_p,
        "null_distribution_sha256": digest,
        "null_distribution": [float(value) for value in samples],
    }


def phase1r_decision_label(positive_control_pass: bool, pooled_sensitivity: float, threshold: float) -> str:
    """Apply the frozen Phase-1R decision tree with an inclusive threshold."""

    if not bool(positive_control_pass):
        return "PIPELINE_INVALID"
    if float(pooled_sensitivity) < float(threshold):
        return "NO_TRAJECTORY_BOTTLENECK_ON_PINNED_PI05_LIBERO"
    return "TRAJECTORY_NONFLATNESS_DETECTED_CHECKPOINT_2"

def paired_episode_bootstrap(matrix_by_task: Iterable[np.ndarray], *, seed: int, replicates: int = 10000) -> dict[str, list[float]]:
    if int(replicates) != REQUIRED_BOOTSTRAP_REPLICATES:
        raise ValueError("Phase-1R requires exactly 10000 paired episode bootstrap replicates")
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



def _public_accounting(accounting: Mapping[str, Any]) -> dict[str, int]:
    """Return stable integer accounting without retaining metadata rows."""

    return {
        str(key): int(value)
        for key, value in accounting.items()
        if key != "metadata"
    }


def _curve_values(curve: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(curve, dtype=np.float64)]


def _control_metrics(matrix: np.ndarray, threshold: float, *, pass_rule: str) -> dict[str, Any]:
    curve = location_curve(matrix)
    summary = location_summary(curve)
    sensitivity = float(summary["location_sensitivity"])
    at_or_above = bool(sensitivity >= float(threshold))
    if pass_rule == "location_sensitivity >= frozen_global_threshold" or pass_rule == "location_sensitivity >= frozen_positive_control_threshold":
        passed = at_or_above
    elif pass_rule == "location_sensitivity <= frozen_null_threshold":
        passed = bool(sensitivity <= float(threshold))
    else:
        raise ValueError(f"unknown control pass rule {pass_rule!r}")
    return {
        "curve": _curve_values(curve),
        "location_summary": summary,
        **summary,
        "threshold": float(threshold),
        "nonflat_above_threshold": at_or_above,
        "nonflat_at_or_above_threshold": at_or_above,
        "pass_rule": pass_rule,
        "pass": bool(passed),
    }


def _permutation_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: payload[key]
        for key in (
            "suite",
            "task_id",
            "shuffles",
            "seed",
            "permutation_seed",
            "permutation_unit",
            "observed_sensitivity",
            "null_p95",
            "p95",
            "empirical_p_value",
            "null_distribution_sha256",
        )
    }


def analyze_phase1r(
    natural_root: str | Path,
    controls_root: str | Path,
    calibration_file: str | Path,
    output_dir: str | Path,
    *,
    bootstrap_replicates: int = REQUIRED_BOOTSTRAP_REPLICATES,
    require_owner: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Analyze unblinded natural Phase-1R cells under frozen calibration.

    Natural calibration and heldout streams are loaded independently. The
    task-local location-label nulls are descriptive audit evidence only and
    never feed threshold calibration or the decision.
    """

    if int(bootstrap_replicates) != REQUIRED_BOOTSTRAP_REPLICATES:
        raise ValueError("Phase-1R requires exactly 10000 paired episode bootstrap replicates")
    calibration_path = Path(calibration_file)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    _validate_calibration(calibration)
    threshold = float(calibration["location_sensitivity_threshold"])
    thresholds_by_control = calibration.get("thresholds_by_control", {})
    if not isinstance(thresholds_by_control, Mapping):
        raise RuntimeError("calibration thresholds_by_control is malformed")
    if "positive" not in thresholds_by_control or "null" not in thresholds_by_control:
        raise RuntimeError("calibration is missing positive/null control thresholds")

    natural_matrices: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    task_accounting: dict[str, dict[str, Any]] = {}
    task_rows: list[dict[str, Any]] = []
    permutation_rows: list[dict[str, Any]] = []
    for suite, task_id in TASKS:
        calibration_matrix, calibration_meta = _load_success_matrix(
            natural_root,
            suite=suite,
            task_id=task_id,
            stream="calibration",
            require_owner=require_owner,
        )
        heldout_matrix, heldout_meta = _load_success_matrix(
            natural_root,
            suite=suite,
            task_id=task_id,
            stream="heldout",
            require_owner=require_owner,
        )
        key = f"{suite}_task{int(task_id):02d}"
        natural_matrices[(suite, int(task_id))] = {
            "calibration": calibration_matrix,
            "heldout": heldout_matrix,
        }
        cal_accounting = _public_accounting(calibration_meta)
        held_accounting = _public_accounting(heldout_meta)
        task_accounting[key] = {
            "suite": suite,
            "task_id": int(task_id),
            "calibration": cal_accounting,
            "heldout": held_accounting,
            "both_streams": _sum_accounting(cal_accounting, held_accounting),
        }
        cal_curve = location_curve(calibration_matrix)
        held_curve = location_curve(heldout_matrix)
        held_summary = location_summary(held_curve)
        permutation = task_permutation_statistics(
            heldout_matrix,
            suite,
            int(task_id),
            shuffles=1000,
            seed=PERMUTATION_SEED,
        )
        permutation_rows.append(permutation)
        sensitivity = float(held_summary["location_sensitivity"])
        at_or_above = bool(sensitivity >= threshold)
        task_rows.append(
            {
                "suite": suite,
                "task_id": int(task_id),
                "task_key": key,
                "analysis_stream": "heldout",
                "calibration_curve": _curve_values(cal_curve),
                "heldout_curve": _curve_values(held_curve),
                "analysis_curve": _curve_values(held_curve),
                "curve": _curve_values(held_curve),
                "location_summary": held_summary,
                **held_summary,
                "threshold": threshold,
                "nonflat_above_threshold": at_or_above,
                "nonflat_at_or_above_threshold": at_or_above,
                "permutation": _permutation_summary(permutation),
            }
        )

    calibration_matrices = [item["calibration"] for item in natural_matrices.values()]
    heldout_matrices = [item["heldout"] for item in natural_matrices.values()]
    pooled_calibration = np.concatenate(calibration_matrices, axis=0)
    pooled_heldout = np.concatenate(heldout_matrices, axis=0)
    pooled_calibration_curve = location_curve(pooled_calibration)
    pooled_heldout_curve = location_curve(pooled_heldout)
    pooled_summary = location_summary(pooled_heldout_curve)
    pooled_sensitivity = float(pooled_summary["location_sensitivity"])
    pooled_at_or_above = bool(pooled_sensitivity >= threshold)
    bootstrap = paired_episode_bootstrap(
        heldout_matrices,
        seed=PERMUTATION_SEED,
        replicates=REQUIRED_BOOTSTRAP_REPLICATES,
    )

    controls: dict[str, dict[str, Any]] = {}
    control_accounting: dict[str, dict[str, Any]] = {}
    for kind, pass_rule in (
        ("positive", "location_sensitivity >= frozen_positive_control_threshold"),
        ("null", "location_sensitivity <= frozen_null_threshold"),
    ):
        cal_matrix, cal_meta = _load_control_matrix(
            controls_root,
            kind,
            stream="calibration",
            require_owner=require_owner,
        )
        held_matrix, held_meta = _load_control_matrix(
            controls_root,
            kind,
            stream="heldout",
            require_owner=require_owner,
        )
        control_threshold = float(thresholds_by_control[kind])
        cal_metrics = _control_metrics(cal_matrix, control_threshold, pass_rule=pass_rule)
        held_metrics = _control_metrics(held_matrix, control_threshold, pass_rule=pass_rule)
        cal_accounting = _public_accounting(cal_meta)
        held_accounting = _public_accounting(held_meta)
        both_accounting = _sum_accounting(cal_accounting, held_accounting)
        control_accounting[kind] = {
            "calibration": cal_accounting,
            "heldout": held_accounting,
            "both_streams": both_accounting,
        }
        controls[kind] = {
            **held_metrics,
            "calibration": cal_metrics,
            "heldout": held_metrics,
            "both_streams": both_accounting,
            "heldout_pass": bool(held_metrics["pass"]),
            "pass": bool(held_metrics["pass"]),
        }

    positive_control_pass = bool(controls["positive"]["heldout_pass"])
    null_control_pass = bool(controls["null"]["heldout_pass"])
    decision_label = phase1r_decision_label(
        positive_control_pass=positive_control_pass,
        pooled_sensitivity=pooled_sensitivity,
        threshold=threshold,
    )

    natural_calibration_accounting = _sum_accounting(
        *(task_accounting[key]["calibration"] for key in task_accounting)
    )
    natural_heldout_accounting = _sum_accounting(
        *(task_accounting[key]["heldout"] for key in task_accounting)
    )
    natural_both_accounting = _sum_accounting(
        natural_calibration_accounting,
        natural_heldout_accounting,
    )
    natural_calibration_cells = natural_calibration_accounting["cell_count"]
    natural_heldout_cells = natural_heldout_accounting["cell_count"]
    natural_cells_total = natural_both_accounting["cell_count"]
    control_calibration_cells = sum(
        value["calibration"]["cell_count"] for value in control_accounting.values()
    )
    control_heldout_cells = sum(
        value["heldout"]["cell_count"] for value in control_accounting.values()
    )
    control_cells_total = sum(
        value["both_streams"]["cell_count"] for value in control_accounting.values()
    )

    task_curves = {
        "protocol_id": PHASE1_PROTOCOL_ID,
        "artifact": "PHASE1R_NATURAL_TASK_CURVES",
        "analysis_stream": "heldout",
        "task_count": len(task_rows),
        "location_count": LOCATIONS_PER_EPISODE,
        "descendants_per_location": DESCENDANTS_PER_CELL,
        "tasks": [
            {
                "suite": row["suite"],
                "task_id": row["task_id"],
                "task_key": row["task_key"],
                "calibration_curve": row["calibration_curve"],
                "heldout_curve": row["heldout_curve"],
                "analysis_curve": row["analysis_curve"],
                "location_summary": row["location_summary"],
            }
            for row in task_rows
        ],
        "pooled_calibration_curve": _curve_values(pooled_calibration_curve),
        "pooled_heldout_curve": _curve_values(pooled_heldout_curve),
        "pooled_curve": _curve_values(pooled_heldout_curve),
        "pooled_location_summary": pooled_summary,
    }
    task_nulls = {
        "protocol_id": PHASE1_PROTOCOL_ID,
        "artifact": "PHASE1R_NATURAL_TASK_PERMUTATION_NULLS",
        "analysis_stream": "heldout",
        "permutation_unit": "within_episode_branch_location_labels",
        "seed": PERMUTATION_SEED,
        "shuffles": 1000,
        "task_count": len(permutation_rows),
        "tasks": permutation_rows,
    }
    summary = {
        "protocol_id": PHASE1_PROTOCOL_ID,
        "artifact": "UNBLINDED_PHASE1R_ANALYSIS",
        "decision_label": decision_label,
        "decision": decision_label,
        "checkpoint": "CHECKPOINT_2_STOP",
        "phase2r": "NOT_DEFINED_IN_SUPPLIED_PLAN",
        "phase2r_authorized": False,
        "calibration_file": str(calibration_path),
        "calibration_sha256": sha256_file(calibration_path),
        "task_count": len(task_rows),
        "selected_episodes_per_task": 12,
        "descendants_per_location": DESCENDANTS_PER_CELL,
        "location_count": LOCATIONS_PER_EPISODE,
        "pooled_calibration_curve": _curve_values(pooled_calibration_curve),
        "pooled_heldout_curve": _curve_values(pooled_heldout_curve),
        "pooled_curve": _curve_values(pooled_heldout_curve),
        "pooled_location_summary": pooled_summary,
        "pooled_threshold": threshold,
        "pooled_nonflat_above_threshold": pooled_at_or_above,
        "pooled_nonflat_at_or_above_threshold": pooled_at_or_above,
        "bootstrap_replicates": REQUIRED_BOOTSTRAP_REPLICATES,
        "pooled_bootstrap_ci95": bootstrap,
        "task_rows": task_rows,
        "controls": controls,
        "positive_control_pass": positive_control_pass,
        "null_control_pass": null_control_pass,
        "null_control_status": "PASS" if null_control_pass else "FAIL",
        "controls_pass": bool(positive_control_pass and null_control_pass),
        "natural_cells_total": natural_cells_total,
        "natural_cells": natural_cells_total,
        "natural_heldout_cells": natural_heldout_cells,
        "natural_calibration_cells": natural_calibration_cells,
        "control_cells": control_cells_total,
        "control_heldout_cells": control_heldout_cells,
        "control_calibration_cells": control_calibration_cells,
        "compute": {
            "natural": {
                "task_count": len(task_rows),
                "analysis_stream": "heldout",
                "tasks": task_accounting,
                "calibration": natural_calibration_accounting,
                "heldout": natural_heldout_accounting,
                "both_streams": natural_both_accounting,
                "cells_total": natural_cells_total,
                "analysis_cells": natural_heldout_cells,
            },
            "controls": {
                **control_accounting,
                "cells_total": control_cells_total,
                "calibration_cells": control_calibration_cells,
                "heldout_cells": control_heldout_cells,
            },
            "natural_policy_forwards": natural_both_accounting["policy_forwards"],
            "natural_logical_policy_forwards": natural_both_accounting["logical_policy_forwards"],
            "natural_policy_batches": natural_both_accounting["policy_batches"],
            "natural_physical_policy_batches": natural_both_accounting["physical_policy_batches"],
            "natural_environment_steps": natural_both_accounting["environment_steps"],
            "natural_branch_count": natural_both_accounting["branch_count"],
            "budget_slack": 0,
            "zero_budget_slack": True,
        },
        "evidence_boundary": (
            "Phase-1R tests branch-location recoverability only; it does not "
            "compare bottleneck-local split with candidate-sampling baselines."
        ),
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "phase1r_summary.json"
    curves_path = output / "phase1r_task_curves.json"
    nulls_path = output / "phase1r_task_nulls.json"
    marker_path = output / "COMPLETED_PHASE1R.json"
    sums_path = output / "SHA256SUMS"
    atomic_json(summary_path, summary)
    atomic_json(curves_path, task_curves)
    atomic_json(nulls_path, task_nulls)
    completed = {
        "protocol_id": PHASE1_PROTOCOL_ID,
        "artifact": "PHASE1R_COMPLETE",
        "decision_label": decision_label,
        "summary": summary_path.name,
        "summary_sha256": sha256_file(summary_path),
        "task_curves": curves_path.name,
        "task_curves_sha256": sha256_file(curves_path),
        "task_nulls": nulls_path.name,
        "task_nulls_sha256": sha256_file(nulls_path),
        "calibration_sha256": sha256_file(calibration_path),
        "task_count": len(task_rows),
        "natural_cells": natural_cells_total,
        "natural_heldout_cells": natural_heldout_cells,
        "natural_calibration_cells": natural_calibration_cells,
        "control_cells": control_cells_total,
        "control_heldout_cells": control_heldout_cells,
        "control_calibration_cells": control_calibration_cells,
        "bootstrap_replicates": REQUIRED_BOOTSTRAP_REPLICATES,
        "unblinded": True,
        "positive_control_pass": positive_control_pass,
        "null_control_pass": null_control_pass,
        "checkpoint": "CHECKPOINT_2_STOP",
        "phase2r_authorized": False,
    }
    atomic_json(marker_path, completed)
    _atomic_text(
        sums_path,
        "".join(
            f"{sha256_file(path)}  {path.name}\n"
            for path in (summary_path, curves_path, nulls_path, marker_path)
        ),
    )
    return summary


def validate_phase1_analysis(
    output_dir: str | Path,
    *,
    require_owner: tuple[int, int] | None = None,
) -> tuple[bool, list[str]]:
    """Fail-closed validation of all deterministic Phase-1R artifacts."""

    root = Path(output_dir)
    errors: list[str] = []
    expected_regular = (
        "phase1r_summary.json",
        "phase1r_task_curves.json",
        "phase1r_task_nulls.json",
        "COMPLETED_PHASE1R.json",
    )
    expected_all = set(expected_regular) | {"SHA256SUMS"}
    if not root.is_dir():
        return False, ["missing phase1 analysis directory"]
    missing = [name for name in expected_all if not (root / name).is_file()]
    if missing:
        errors.append(f"missing phase1 completion artifacts: {sorted(missing)}")
        return False, errors
    for name in expected_all:
        path = root / name
        if path.is_symlink():
            errors.append(f"symlink analysis artifact: {name}")
    try:
        summary = json.loads((root / "phase1r_summary.json").read_text(encoding="utf-8"))
        curves = json.loads((root / "phase1r_task_curves.json").read_text(encoding="utf-8"))
        nulls = json.loads((root / "phase1r_task_nulls.json").read_text(encoding="utf-8"))
        marker = json.loads((root / "COMPLETED_PHASE1R.json").read_text(encoding="utf-8"))
        if summary.get("protocol_id") != PHASE1_PROTOCOL_ID:
            errors.append("summary protocol mismatch")
        if summary.get("artifact") != "UNBLINDED_PHASE1R_ANALYSIS":
            errors.append("summary artifact mismatch")
        if summary.get("decision_label") not in ANALYSIS_DECISIONS:
            errors.append("invalid decision label")
        if summary.get("decision") != summary.get("decision_label"):
            errors.append("decision alias mismatch")
        if summary.get("checkpoint") != "CHECKPOINT_2_STOP":
            errors.append("missing CHECKPOINT_2_STOP")
        if summary.get("phase2r_authorized") is not False:
            errors.append("Phase-2R must remain unauthorized")
        if summary.get("task_count") != 40:
            errors.append("task count mismatch")
        exact_counts = {
            "natural_cells_total": 9600,
            "natural_cells": 9600,
            "natural_heldout_cells": 4800,
            "natural_calibration_cells": 4800,
            "control_cells": 480,
            "control_heldout_cells": 240,
            "control_calibration_cells": 240,
            "bootstrap_replicates": REQUIRED_BOOTSTRAP_REPLICATES,
        }
        for key, expected in exact_counts.items():
            if summary.get(key) != expected:
                errors.append(f"{key} mismatch")
        if marker.get("protocol_id") != PHASE1_PROTOCOL_ID:
            errors.append("marker protocol mismatch")
        if marker.get("artifact") != "PHASE1R_COMPLETE":
            errors.append("marker artifact mismatch")
        if marker.get("decision_label") != summary.get("decision_label"):
            errors.append("marker decision mismatch")
        marker_counts = {
            "task_count": 40,
            "natural_cells": 9600,
            "natural_heldout_cells": 4800,
            "natural_calibration_cells": 4800,
            "control_cells": 480,
            "control_heldout_cells": 240,
            "control_calibration_cells": 240,
            "bootstrap_replicates": REQUIRED_BOOTSTRAP_REPLICATES,
        }
        for key, expected in marker_counts.items():
            if marker.get(key) != expected:
                errors.append(f"marker {key} mismatch")
        if marker.get("checkpoint") != "CHECKPOINT_2_STOP":
            errors.append("marker checkpoint mismatch")
        if marker.get("phase2r_authorized") is not False:
            errors.append("marker Phase-2R authorization mismatch")
        if not isinstance(summary.get("positive_control_pass"), bool):
            errors.append("positive control result is not explicit")
        if not isinstance(summary.get("null_control_pass"), bool):
            errors.append("null control result is not explicit")
        if summary.get("null_control_status") not in {"PASS", "FAIL"}:
            errors.append("null control status is not explicit")
        if summary.get("controls_pass") != (
            summary.get("positive_control_pass") and summary.get("null_control_pass")
        ):
            errors.append("controls pass aggregation mismatch")

        expected_task_keys = {f"{suite}_task{int(task_id):02d}" for suite, task_id in TASKS}
        summary_rows = summary.get("task_rows", [])
        curve_rows = curves.get("tasks", [])
        null_rows = nulls.get("tasks", [])
        if not isinstance(summary_rows, list) or len(summary_rows) != 40:
            errors.append("summary task rows mismatch")
            summary_rows = []
        if (
            curves.get("protocol_id") != PHASE1_PROTOCOL_ID
            or curves.get("artifact") != "PHASE1R_NATURAL_TASK_CURVES"
            or curves.get("task_count") != 40
            or curves.get("location_count") != LOCATIONS_PER_EPISODE
            or curves.get("descendants_per_location") != DESCENDANTS_PER_CELL
            or not isinstance(curve_rows, list)
            or len(curve_rows) != 40
        ):
            errors.append("task curves artifact contract mismatch")
            curve_rows = []
        if (
            nulls.get("protocol_id") != PHASE1_PROTOCOL_ID
            or nulls.get("artifact") != "PHASE1R_NATURAL_TASK_PERMUTATION_NULLS"
            or nulls.get("analysis_stream") != "heldout"
            or nulls.get("permutation_unit") != "within_episode_branch_location_labels"
            or nulls.get("shuffles") != 1000
            or nulls.get("task_count") != 40
            or not isinstance(null_rows, list)
            or len(null_rows) != 40
        ):
            errors.append("task null artifact contract mismatch")
            null_rows = []
        summary_map: dict[str, Mapping[str, Any]] = {}
        curve_map: dict[str, Mapping[str, Any]] = {}
        null_map: dict[str, Mapping[str, Any]] = {}
        for row, target, label in (
            *((row, summary_map, "summary") for row in summary_rows),
            *((row, curve_map, "curves") for row in curve_rows),
            *((row, null_map, "nulls") for row in null_rows),
        ):
            if not isinstance(row, Mapping):
                errors.append(f"{label} task row is malformed")
                continue
            try:
                key = str(row.get("task_key", f"{row.get('suite')}_task{int(row.get('task_id', -1)):02d}"))
            except (TypeError, ValueError):
                errors.append(f"{label} task key malformed")
                continue
            if key in target:
                errors.append(f"duplicate {label} task key: {key}")
            target[key] = row
        if set(summary_map) != expected_task_keys:
            errors.append("summary task keys mismatch")
        if set(curve_map) != expected_task_keys:
            errors.append("curve task keys mismatch")
        if set(null_map) != expected_task_keys:
            errors.append("null task keys mismatch")
        for key in sorted(expected_task_keys):
            summary_row = summary_map.get(key)
            curve_row = curve_map.get(key)
            null_row = null_map.get(key)
            if summary_row is None or curve_row is None or null_row is None:
                continue
            for field in ("calibration_curve", "heldout_curve", "analysis_curve"):
                values = curve_row.get(field)
                if not isinstance(values, list) or len(values) != LOCATIONS_PER_EPISODE:
                    errors.append(f"{key} curve {field} malformed")
            summary_curve = summary_row.get("analysis_curve", summary_row.get("curve"))
            curve_analysis = curve_row.get("analysis_curve")
            if not isinstance(summary_curve, list) or not isinstance(curve_analysis, list):
                errors.append(f"{key} analysis curve missing")
            elif not np.allclose(summary_curve, curve_analysis, rtol=0.0, atol=1e-15):
                errors.append(f"{key} analysis curve mismatch")
            permutation = summary_row.get("permutation")
            if not isinstance(permutation, Mapping):
                errors.append(f"{key} permutation summary missing")
                continue
            if permutation.get("null_distribution_sha256") != null_row.get("null_distribution_sha256"):
                errors.append(f"{key} permutation digest mismatch")
            try:
                shuffles = int(null_row["shuffles"])
                observed = float(null_row["observed_sensitivity"])
                p95 = float(null_row["p95"])
                empirical = float(null_row["empirical_p_value"])
            except (KeyError, TypeError, ValueError):
                errors.append(f"{key} permutation statistics malformed")
                continue
            samples = np.asarray(null_row.get("null_distribution", []), dtype=np.float64)
            if shuffles < 1000 or len(samples) != shuffles:
                errors.append(f"{key} permutation count mismatch")
            if not np.all(np.isfinite(samples)):
                errors.append(f"{key} permutation samples nonfinite")
            if _float64_digest(samples) != null_row.get("null_distribution_sha256"):
                errors.append(f"{key} permutation SHA mismatch")
            if len(samples):
                expected_p95 = float(np.quantile(samples, 0.95))
                expected_p = float((np.count_nonzero(samples >= observed) + 1) / (len(samples) + 1))
                if not np.isclose(p95, expected_p95, rtol=1e-12, atol=1e-12):
                    errors.append(f"{key} permutation p95 mismatch")
                if not np.isclose(empirical, expected_p, rtol=1e-12, atol=1e-12):
                    errors.append(f"{key} permutation empirical p mismatch")
            for field in ("p95", "empirical_p_value", "observed_sensitivity"):
                if field in permutation and field in null_row:
                    if not np.isclose(float(permutation[field]), float(null_row[field]), rtol=1e-12, atol=1e-12):
                        errors.append(f"{key} permutation summary {field} mismatch")
            if permutation.get("shuffles") != shuffles:
                errors.append(f"{key} permutation summary count mismatch")

        pooled_summary = summary.get("pooled_location_summary", {})
        try:
            pooled_sensitivity = float(pooled_summary["location_sensitivity"])
            pooled_threshold = float(summary["pooled_threshold"])
            expected_nonflat = bool(pooled_sensitivity >= pooled_threshold)
            if summary.get("pooled_nonflat_above_threshold") != expected_nonflat:
                errors.append("pooled inclusive threshold mismatch")
            if summary.get("pooled_nonflat_at_or_above_threshold") != expected_nonflat:
                errors.append("pooled inclusive threshold alias mismatch")
            expected_decision = phase1r_decision_label(
                positive_control_pass=bool(summary["positive_control_pass"]),
                pooled_sensitivity=pooled_sensitivity,
                threshold=pooled_threshold,
            )
            if summary.get("decision_label") != expected_decision:
                errors.append("decision tree mismatch")
        except (KeyError, TypeError, ValueError):
            errors.append("pooled decision fields malformed")

        controls = summary.get("controls")
        if not isinstance(controls, Mapping):
            errors.append("controls missing")
            controls = {}
        for kind in ("positive", "null"):
            control = controls.get(kind)
            if not isinstance(control, Mapping):
                errors.append(f"{kind} control missing")
                continue
            for field in ("heldout_pass", "pass"):
                if not isinstance(control.get(field), bool):
                    errors.append(f"{kind} control {field} missing")
            if not isinstance(control.get("pass_rule"), str) or not control.get("pass_rule"):
                errors.append(f"{kind} control pass rule missing")
            for stream in ("calibration", "heldout"):
                stream_metrics = control.get(stream)
                if not isinstance(stream_metrics, Mapping):
                    errors.append(f"{kind} control {stream} metrics missing")
                elif not isinstance(stream_metrics.get("pass"), bool):
                    errors.append(f"{kind} control {stream} pass missing")
            both = control.get("both_streams")
            if not isinstance(both, Mapping) or both.get("cell_count") != 240:
                errors.append(f"{kind} control accounting mismatch")

        natural_compute = summary.get("compute", {}).get("natural", {})
        if not isinstance(natural_compute, Mapping):
            errors.append("natural compute accounting missing")
        else:
            for stream, expected_cells in (("calibration", 4800), ("heldout", 4800), ("both_streams", 9600)):
                stream_accounting = natural_compute.get(stream)
                if not isinstance(stream_accounting, Mapping) or stream_accounting.get("cell_count") != expected_cells:
                    errors.append(f"natural {stream} accounting mismatch")
            for field in (
                "natural_policy_forwards",
                "natural_logical_policy_forwards",
                "natural_policy_batches",
                "natural_physical_policy_batches",
                "natural_environment_steps",
                "natural_branch_count",
                "budget_slack",
            ):
                if field not in summary.get("compute", {}):
                    errors.append(f"compute field missing: {field}")
            if summary.get("compute", {}).get("budget_slack") != 0:
                errors.append("nonzero budget slack")
            if summary.get("compute", {}).get("zero_budget_slack") is not True:
                errors.append("zero budget slack marker missing")
        controls_compute = summary.get("compute", {}).get("controls", {})
        if not isinstance(controls_compute, Mapping):
            errors.append("control compute accounting missing")
        else:
            if controls_compute.get("cells_total") != 480:
                errors.append("control compute total mismatch")
            if controls_compute.get("calibration_cells") != 240 or controls_compute.get("heldout_cells") != 240:
                errors.append("control compute stream totals mismatch")

        boundary = summary.get("evidence_boundary", "")
        if "branch-location recoverability only" not in boundary or "candidate-sampling baselines" not in boundary:
            errors.append("evidence boundary missing Phase-1R-only claim")

        if marker.get("summary_sha256") != sha256_file(root / "phase1r_summary.json"):
            errors.append("summary SHA mismatch")
        if marker.get("task_curves_sha256") != sha256_file(root / "phase1r_task_curves.json"):
            errors.append("task curves SHA mismatch")
        if marker.get("task_nulls_sha256") != sha256_file(root / "phase1r_task_nulls.json"):
            errors.append("task nulls SHA mismatch")
        if marker.get("calibration_sha256") != summary.get("calibration_sha256"):
            errors.append("calibration SHA mismatch")

        observed_sums: dict[str, str] = {}
        for line_number, line in enumerate((root / "SHA256SUMS").read_text(encoding="utf-8").splitlines(), 1):
            if not line or line.count("  ") != 1:
                errors.append(f"malformed SHA256SUMS line {line_number}")
                continue
            digest, relative = line.split("  ", 1)
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest.lower()):
                errors.append(f"malformed SHA256SUMS digest {line_number}")
                continue
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                errors.append(f"unsafe analysis checksum entry: {relative}")
                continue
            if relative in observed_sums:
                errors.append(f"duplicate analysis checksum entry: {relative}")
            observed_sums[relative] = digest
        expected_sums = {name: sha256_file(root / name) for name in expected_regular}
        if observed_sums != expected_sums:
            errors.append("analysis SHA256SUMS mismatch")
        actual_regular = {
            path.name
            for path in root.iterdir()
            if path.is_file() and path.name != "SHA256SUMS"
        }
        if actual_regular != set(expected_regular):
            errors.append("unlisted analysis artifact present")
        if require_owner is not None:
            owner = tuple(require_owner)
            for name in (*expected_regular, "SHA256SUMS"):
                stat = (root / name).stat()
                if (int(stat.st_uid), int(stat.st_gid)) != owner:
                    errors.append(f"analysis owner mismatch: {name}")
    except Exception as exc:
        errors.append(f"parse error: {type(exc).__name__}: {exc}")
    return not errors, errors
