from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform


def overdispersion(success: np.ndarray, initial_state: np.ndarray, n: int = 32) -> dict[str, float | None]:
    groups = []
    for value in sorted(set(int(x) for x in initial_state)):
        values = success[initial_state == value]
        if len(values) != int(n):
            raise ValueError(f"initial state {value} has {len(values)} descendants, expected {n}")
        groups.append(float(np.mean(values)))
    p_e = np.asarray(groups, dtype=np.float64)
    p_bar = float(np.mean(p_e))
    denominator = p_bar * (1.0 - p_bar) / float(n)
    rho = None if denominator <= 0.0 else float(np.var(p_e, ddof=1) / denominator)
    return {
        "p_bar": p_bar,
        "rho": rho,
        "low_p_fraction": float(np.mean(p_e <= (1.0 / float(n)))),
    }


def mean_pairwise_rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] < 2:
        return float("nan")
    distances = pdist(values / np.sqrt(max(1, values.shape[1])), metric="euclidean")
    return float(np.mean(distances))


def divergence_curve(trajectories: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if len(trajectories) < 2:
        return np.empty(0), np.empty(0, dtype=np.int64)
    horizon = max(len(value) for value in trajectories)
    curve = []
    at_risk = []
    for step in range(horizon):
        alive = [value[step] for value in trajectories if step < len(value)]
        at_risk.append(len(alive))
        curve.append(mean_pairwise_rms(np.asarray(alive)) if len(alive) >= 2 else np.nan)
    return np.asarray(curve, dtype=np.float64), np.asarray(at_risk, dtype=np.int64)


def first_crossing(curve: np.ndarray, threshold: float) -> int | None:
    indices = np.flatnonzero(np.isfinite(curve) & (curve > float(threshold)))
    return None if not len(indices) else int(indices[0])


def silhouette(features: np.ndarray, labels: np.ndarray) -> float:
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    if len(set(labels.tolist())) < 2 or len(features) < 4:
        return float("-inf")
    matrix = squareform(pdist(features, metric="euclidean"))
    values = []
    for index, label in enumerate(labels):
        same = labels == label
        same[index] = False
        if not np.any(same):
            return float("-inf")
        a = float(np.mean(matrix[index, same]))
        other_means = [float(np.mean(matrix[index, labels == other])) for other in set(labels.tolist()) if other != label]
        b = min(other_means)
        values.append((b - a) / max(a, b, 1e-12))
    return float(np.mean(values))


def hierarchical_modes(features: np.ndarray, threshold: float, max_modes: int = 4) -> dict[str, Any]:
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] < 4:
        return {"mode_count": 0, "silhouette": None, "labels": []}
    scale = np.std(features, axis=0)
    standardized = (features - np.mean(features, axis=0)) / np.where(scale > 1e-9, scale, 1.0)
    tree = linkage(standardized, method="average", metric="euclidean")
    best = (float("-inf"), 0, np.zeros(len(features), dtype=np.int64))
    for modes in range(2, min(int(max_modes), len(features) - 1) + 1):
        labels = fcluster(tree, modes, criterion="maxclust")
        if len(set(labels.tolist())) != modes or min(np.bincount(labels)[1:]) < 2:
            continue
        score = silhouette(standardized, labels)
        if score > best[0]:
            best = (score, modes, labels)
    if not np.isfinite(best[0]) or best[0] <= float(threshold):
        return {"mode_count": 1, "silhouette": None if not np.isfinite(best[0]) else float(best[0]), "labels": [1] * len(features)}
    return {"mode_count": int(best[1]), "silhouette": float(best[0]), "labels": best[2].astype(int).tolist()}


def midtrajectory_features(actions: list[np.ndarray], poses: list[np.ndarray]) -> np.ndarray:
    if len(actions) != len(poses):
        raise ValueError("actions and poses must have the same number of trajectories")
    output = []
    for action, pose in zip(actions, poses):
        length = min(len(action), len(pose))
        if length == 0:
            output.append(np.zeros(action.shape[1] * 2 + pose.shape[1], dtype=np.float64))
            continue
        lo, hi = length // 3, max(length // 3 + 1, (2 * length) // 3)
        middle_action = np.asarray(action[lo:hi], dtype=np.float64)
        middle_pose = np.asarray(pose[lo:hi], dtype=np.float64)
        output.append(np.concatenate([np.mean(middle_action, axis=0), np.std(middle_action, axis=0), middle_pose[len(middle_pose) // 2]]))
    return np.asarray(output, dtype=np.float64)


def action_split_step(actions: list[np.ndarray], threshold: float) -> int | None:
    if len(actions) < 4:
        return None
    horizon = min(len(value) for value in actions)
    for step in range(1, horizon + 1):
        features = np.asarray([value[:step].reshape(-1) for value in actions], dtype=np.float64)
        result = hierarchical_modes(features, threshold, max_modes=2)
        if result["mode_count"] >= 2:
            return step - 1
    return None


def task_metrics(rollouts: list[dict[str, Any]], thresholds: dict[str, float], *, n: int = 32) -> dict[str, Any]:
    success = np.asarray([bool(row["success"]) for row in rollouts], dtype=np.bool_)
    initial_state = np.asarray([int(row["init_state"]) for row in rollouts], dtype=np.int64)
    base = overdispersion(success, initial_state, n=n)
    t_div_values: list[int] = []
    t_div_fractions: list[float] = []
    action_split_values: list[int] = []
    all_fail_count = 0
    for state in sorted(set(initial_state.tolist())):
        group = [row for row in rollouts if int(row["init_state"]) == state]
        if any(bool(row["success"]) for row in group):
            continue
        all_fail_count += 1
        poses = [np.asarray(row["poses"], dtype=np.float64) for row in group]
        actions = [np.asarray(row["actions"], dtype=np.float64) for row in group]
        curve, _ = divergence_curve(poses)
        crossing = first_crossing(curve, thresholds["divergence_rms"])
        if crossing is not None:
            t_div_values.append(crossing)
            t_div_fractions.append(crossing / max(1, int(np.median([len(value) for value in poses]))))
        split = action_split_step(actions, thresholds["action_split_silhouette"])
        if split is not None:
            action_split_values.append(split)
    successful = [row for row in rollouts if bool(row["success"])]
    if successful:
        features = midtrajectory_features(
            [np.asarray(row["actions"]) for row in successful],
            [np.asarray(row["poses"]) for row in successful],
        )
        modes = hierarchical_modes(features, thresholds["mode_silhouette"])
    else:
        modes = {"mode_count": 0, "silhouette": None, "labels": []}
    progress = np.asarray([float(row["final_progress"]) for row in rollouts], dtype=np.float64)
    q25, median, q75 = np.quantile(progress, [0.25, 0.5, 0.75])
    ceiling = bool(median == 1.0 and q25 == 1.0 and q75 == 1.0)
    e6 = bool(0.25 <= float(base["p_bar"]) <= 0.75 and not ceiling)
    median_fraction = None if not t_div_fractions else float(np.median(t_div_fractions))
    retained = bool(
        base["rho"] is not None
        and float(base["rho"]) >= 3.0
        and float(base["low_p_fraction"]) >= 0.25
        and median_fraction is not None
        and median_fraction >= 0.10
        and int(modes["mode_count"]) >= 2
        and e6
    )
    return {
        **base,
        "all_fail_initial_states": all_fail_count,
        "median_t_div": None if not t_div_values else float(np.median(t_div_values)),
        "median_t_div_episode_fraction": median_fraction,
        "median_action_split_step": None if not action_split_values else float(np.median(action_split_values)),
        "stable_modes": int(modes["mode_count"]),
        "mode_silhouette": modes["silhouette"],
        "final_progress_q25": float(q25),
        "final_progress_median": float(median),
        "final_progress_q75": float(q75),
        "progress_ceiling_pile": ceiling,
        "e6": e6,
        "retained": retained,
    }


def group_controls(traces: Iterable[Any]) -> dict[int, list[Any]]:
    result: dict[int, list[Any]] = defaultdict(list)
    for trace in traces:
        result[int(trace.initial_state)].append(trace)
    return result
