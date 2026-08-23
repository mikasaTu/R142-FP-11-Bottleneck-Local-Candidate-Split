"""Pre-registered branchability and fixed-NFE accounting utilities."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

import numpy as np


def pairwise_rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return 0.0
    diffs = values[:, None, ...] - values[None, :, ...]
    distances = np.sqrt(np.mean(diffs * diffs, axis=tuple(range(2, diffs.ndim))))
    return float(distances[np.triu_indices(len(values), 1)].mean())


def branchability_vector(progress: Iterable[float], actions: np.ndarray, control: Iterable[float]) -> dict:
    progress = np.asarray(list(progress), dtype=float)
    control = np.asarray(list(control), dtype=float)
    q10, q90 = np.quantile(progress, [0.1, 0.9])
    baseline = float(np.mean(control)) if len(control) else float("nan")
    return {
        "progress_q10": float(q10),
        "progress_q90": float(q90),
        "progress_spread": float(q90 - q10),
        "rescue_probability": float(np.mean(progress >= 0.95)),
        "support_diversity": pairwise_rms(np.asarray(actions)),
        "best_descendant_gain": float(np.max(progress) - baseline),
        "mean_progress": float(np.mean(progress)),
    }


def actual_nfe(root_count: int, total_steps: int, branch_index: int, suffix_count: int) -> int:
    return root_count * total_steps + root_count * suffix_count * (total_steps - branch_index)


def fixed_nfe_suffix_count(
    budget: int, root_count: int, total_steps: int, branch_index: int
) -> tuple[int, int]:
    remaining = total_steps - branch_index
    fixed_root_cost = root_count * total_steps
    if budget < fixed_root_cost or remaining <= 0:
        return 0, budget - fixed_root_cost
    count = (budget - fixed_root_cost) // (root_count * remaining)
    used = fixed_root_cost + count * root_count * remaining
    return int(count), int(budget - used)


def grouped_curves(records: Iterable[dict]) -> dict[str, list[dict]]:
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in records:
        groups[(row["snapshot_id"], int(row["checkpoint_index"]))].append(row)
    curves: dict[str, list[dict]] = defaultdict(list)
    for (snapshot_id, checkpoint), rows in sorted(groups.items()):
        progress = [r["final_progress"] for r in rows if r["stream"] == "heldout"]
        action_values = np.asarray([r["actions_normalized"] for r in rows if r["stream"] == "heldout"])
        control = [r["final_progress"] for r in rows if r["stream"] == "control"]
        if not progress:
            continue
        vector = branchability_vector(progress, action_values, control)
        vector.update(
            {
                "checkpoint_index": checkpoint,
                "remaining_nfe": int(rows[0]["remaining_nfe"]),
                "benefit_per_nfe": vector["best_descendant_gain"]
                / max(1, int(rows[0]["remaining_nfe"])),
            }
        )
        curves[snapshot_id].append(vector)
    return dict(curves)


def classify_curve(curve: list[dict], min_drop: float = 0.10) -> list[str]:
    values = np.asarray([row["progress_spread"] for row in curve], dtype=float)
    labels: list[str] = []
    if len(values) == 0:
        return ["missing"]
    if float(np.ptp(values)) < min_drop:
        labels.append("no-bottleneck")
    diffs = np.diff(values)
    if len(diffs) and np.all(diffs <= min_drop / 4):
        labels.append("smooth-decay")
    large = np.where(-diffs >= min_drop)[0]
    if len(large) > 1:
        labels.append("multiple-cliff")
    elif len(large) == 1:
        labels.append("single-cliff-candidate")
    return labels or ["non-monotone-no-preregistered-cliff"]


def normalized_gain(lhs: float, rhs: float) -> float:
    if not math.isfinite(lhs) or not math.isfinite(rhs):
        return float("nan")
    return float(lhs - rhs)
