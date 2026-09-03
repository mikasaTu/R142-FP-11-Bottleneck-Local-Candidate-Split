"""Pure Stage-S gate metrics.

The functions consume completed rollout records; they do not generate or alter
trajectories.  All literal thresholds in this file mirror the frozen
stage-s/PROTOCOL.md contract so a caller cannot silently retune a gate.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Iterable

import numpy as np

# Literal preregistered Stage-S numbers.
BASE_CANDIDATE_COUNT = 32
EXTENDED_CANDIDATE_COUNT = 64
NEAR_ALL_FAIL_MAX_SUCCESS = 1
S1_MIN_SUCCESS = 0.30
S1_MAX_SUCCESS = 0.60
S2_MIN_NEAR_ALL_FAIL_FRACTION = 0.10
S2_MIN_RHO = 3.0
S2_BINOMIAL_EXPECTATION_MULTIPLIER = 20.0
S3_MIN_MEDIAN_DIVERGENCE_FRACTION = 0.10
S3_MAX_ORIGIN_FRACTION = 0.25
TAU_QUANTILE = 0.95
S4_MIN_ORACLE_RECOVERY_FRACTION = 0.30
S4_BOOTSTRAP_REPLICATES = 10000
S4_BOOTSTRAP_CI_LEVEL = 0.95
S5_MAX_RESCUE_FRACTION = 0.05
DEFAULT_BOOTSTRAP_SEED = 14211
S4_BOOTSTRAP_SEED = 14211

# The production S3 metric is defined in the workspace pose contract.  Keep
# these tuples immutable so a caller cannot accidentally use a qpos or scalar
# normalization when analyzing a real substrate.
S3_SUBSTRATE_WORKSPACE_SCALES: dict[str, tuple[float, ...]] = {
    "A": (1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0),
    "B": (1.0, 1.0, 1.0, math.pi, math.pi, math.pi),
    "C": (1.0, 1.0, 1.0, math.pi, math.pi, math.pi),
}


DECISION_CODES = frozenset(
    {
        "SUBSTRATE_QUALIFIED",
        "NO_SUBSTRATE_AT_TARGET_DIFFICULTY",
        "NO_FAMILY_COLLAPSE",
        "COLLAPSE_AT_ORIGIN",
        "UNRECOVERABLE_FAILURES",
        "BUDGET_SUFFICES",
        "WEAK_SUBSTRATE_ONLY",
        "PIPELINE_INVALID",
    }
)


def _success(row: object) -> bool:
    if isinstance(row, Mapping):
        if "success" not in row:
            raise ValueError("rollout row lacks eventual success")
        return bool(row["success"])
    if hasattr(row, "success"):
        return bool(getattr(row, "success"))
    if isinstance(row, (bool, np.bool_)):
        return bool(row)
    if isinstance(row, (int, np.integer, float, np.floating)) and float(row) in (0.0, 1.0):
        return bool(row)
    raise TypeError("rollout success must be present as a boolean")


def _family_id(row: object, fallback: object) -> object:
    if isinstance(row, Mapping):
        for key in ("family_id", "family", "initial_state", "init_state"):
            if key in row:
                return row[key]
    for key in ("family_id", "family", "initial_state", "init_state"):
        if hasattr(row, key):
            return getattr(row, key)
    if fallback is not None:
        return fallback
    raise ValueError("rollout row lacks family_id/initial_state")


def group_families(data: Any) -> dict[object, list[object]]:
    """Normalize row lists or family mappings without dropping any row."""

    if isinstance(data, Mapping):
        groups: dict[object, list[object]] = {}
        for key, value in data.items():
            if isinstance(value, Mapping) and "rollouts" in value:
                value = value["rollouts"]
            if not isinstance(value, Iterable) or isinstance(value, (str, bytes, Mapping)):
                raise TypeError(f"family {key!r} must contain rollout rows")
            groups[key] = list(value)
        return groups
    if isinstance(data, np.ndarray):
        data = data.tolist()
    if isinstance(data, (str, bytes)) or not isinstance(data, Iterable):
        raise TypeError("rollouts must be an iterable or family mapping")
    groups = defaultdict(list)
    for index, row in enumerate(data):
        groups[_family_id(row, index if not isinstance(row, Mapping) else None)].append(row)
    return dict(groups)


def _json_key(value: object) -> str:
    return str(value)


def _require_count(group: Sequence[object], expected: int, family: object) -> None:
    if len(group) != int(expected):
        raise ValueError(f"family {family!r} has {len(group)} candidates, expected {expected}")


def pooled_success_rate(data: Any) -> float:
    rows: list[object] = []
    if isinstance(data, Mapping):
        groups = group_families(data)
        for group in groups.values():
            rows.extend(group)
    else:
        rows = list(data)
    if not rows:
        raise ValueError("rollout set cannot be empty")
    return float(np.mean(np.asarray([_success(row) for row in rows], dtype=np.float64)))


def compute_s1(data: Any) -> dict[str, Any]:
    """Evaluate S1 against pooled eventual episode success."""

    pooled = pooled_success_rate(data)
    return {
        "pooled_success": pooled,
        "lower": S1_MIN_SUCCESS,
        "upper": S1_MAX_SUCCESS,
        "pass": bool(S1_MIN_SUCCESS <= pooled <= S1_MAX_SUCCESS),
    }


def _binomial_near_all_fail_probability(p: float, n: int) -> float:
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0 if n > NEAR_ALL_FAIL_MAX_SUCCESS else 1.0
    # P[X <= 1] for X ~ Binomial(n, p), fixed by the protocol.
    return float((1.0 - p) ** n + n * p * (1.0 - p) ** (n - 1))


def family_collapse_metrics(data: Any, *, n: int = BASE_CANDIDATE_COUNT) -> dict[str, Any]:
    """Return S2 family statistics, including the exact binomial expectation."""

    groups = group_families(data)
    if not groups:
        raise ValueError("at least one family is required")
    counts: dict[str, int] = {}
    rates: list[float] = []
    for family, group in sorted(groups.items(), key=lambda item: str(item[0])):
        _require_count(group, n, family)
        success_count = int(sum(_success(row) for row in group))
        counts[_json_key(family)] = success_count
        rates.append(success_count / float(n))
    rates_array = np.asarray(rates, dtype=np.float64)
    pooled = float(np.mean(rates_array))
    observed_variance = float(np.var(rates_array, ddof=1)) if len(rates_array) > 1 else 0.0
    binomial_variance = float(pooled * (1.0 - pooled) / float(n))
    rho = None if binomial_variance <= 0.0 else float(observed_variance / binomial_variance)
    near_count = int(np.sum(rates_array <= NEAR_ALL_FAIL_MAX_SUCCESS / float(n)))
    strict_zero_count = int(np.sum(rates_array == 0.0))
    near_fraction = float(near_count / len(rates_array))
    probability = _binomial_near_all_fail_probability(pooled, int(n))
    expected_near_count = float(len(rates_array) * probability)
    if expected_near_count == 0.0:
        observed_to_expected = float("inf") if near_count > 0 else 0.0
    else:
        observed_to_expected = float(near_count / expected_near_count)
    pass_gate = bool(
        near_fraction >= S2_MIN_NEAR_ALL_FAIL_FRACTION
        and rho is not None
        and rho >= S2_MIN_RHO
        and near_count > S2_BINOMIAL_EXPECTATION_MULTIPLIER * expected_near_count
    )
    return {
        "family_count": len(rates),
        "candidate_count_per_family": int(n),
        "pooled_success": pooled,
        "success_counts": counts,
        "family_success_rates": rates,
        "near_all_fail_definition": f"successes <= {NEAR_ALL_FAIL_MAX_SUCCESS}/{int(n)}",
        "near_all_fail_count": near_count,
        "near_all_fail_fraction": near_fraction,
        "strict_zero_count": strict_zero_count,
        "strict_zero_fraction": float(strict_zero_count / len(rates)),
        "variance_observed": observed_variance,
        "variance_binomial": binomial_variance,
        "rho": rho,
        "binomial_near_all_fail_probability": probability,
        "binomial_expected_near_all_fail_count": expected_near_count,
        "observed_to_binomial_expected": observed_to_expected,
        "binomial_multiplier_threshold": S2_BINOMIAL_EXPECTATION_MULTIPLIER,
        "pass": pass_gate,
    }


def compute_s2(data: Any, *, n: int = BASE_CANDIDATE_COUNT) -> dict[str, Any]:
    return family_collapse_metrics(data, n=n)


def _trajectory(row: object) -> np.ndarray:
    if isinstance(row, Mapping):
        for key in ("workspace_poses", "poses", "positions", "trajectory"):
            if key in row:
                value = row[key]
                break
        else:
            raise ValueError("rollout row lacks workspace pose trajectory")
    else:
        for key in ("workspace_poses", "poses", "positions", "trajectory"):
            if hasattr(row, key):
                value = getattr(row, key)
                break
        else:
            value = row
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"trajectory must have shape [time,workspace_dim], got {array.shape}")
    if array.shape[1] == 0:
        raise ValueError("trajectory workspace dimension cannot be zero")
    if not np.all(np.isfinite(array)):
        raise ValueError("trajectory contains non-finite pose values")
    return array


def _workspace_scale(
    dimension: int,
    *,
    workspace_bounds: Any | None = None,
    workspace_scale: Any | None = None,
) -> np.ndarray:
    if workspace_bounds is not None and workspace_scale is not None:
        raise ValueError("pass either workspace_bounds or workspace_scale, not both")
    if workspace_bounds is None and workspace_scale is None:
        scale = np.ones(dimension, dtype=np.float64)
    elif workspace_scale is not None:
        scale = np.asarray(workspace_scale, dtype=np.float64)
        if scale.ndim == 0:
            scale = np.repeat(scale, dimension)
        if scale.shape != (dimension,):
            raise ValueError("workspace_scale must be scalar or one value per pose dimension")
    else:
        bounds = np.asarray(workspace_bounds, dtype=np.float64)
        if bounds.shape == (2, dimension):
            scale = np.abs(bounds[1] - bounds[0])
        elif bounds.shape == (dimension, 2):
            scale = np.abs(bounds[:, 1] - bounds[:, 0])
        else:
            raise ValueError("workspace_bounds must have shape [2,d] or [d,2]")
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("workspace normalization extents must be finite and positive")
    return scale


def _pairwise_rms(values: np.ndarray, scale: np.ndarray) -> np.ndarray:
    if values.shape[0] < 2:
        return np.empty(0, dtype=np.float64)
    differences = (values[:, None, :] - values[None, :, :]) / scale[None, None, :]
    upper = differences[np.triu_indices(values.shape[0], 1)]
    return np.sqrt(np.mean(np.square(upper), axis=1))


def normalized_workspace_pose_rms(
    trajectories: Sequence[Any],
    *,
    workspace_bounds: Any | None = None,
    workspace_scale: Any | None = None,
) -> np.ndarray:
    """Compute D(t), the mean pairwise normalized workspace-pose RMS."""

    arrays = [_trajectory(value) for value in trajectories]
    if not arrays:
        return np.empty(0, dtype=np.float64)
    dimension = arrays[0].shape[1]
    if any(array.shape[1] != dimension for array in arrays):
        raise ValueError("all trajectories must have the same workspace dimension")
    scale = _workspace_scale(
        dimension, workspace_bounds=workspace_bounds, workspace_scale=workspace_scale
    )
    horizon = max(len(array) for array in arrays)
    curve = np.full(horizon, np.nan, dtype=np.float64)
    for step in range(horizon):
        at_risk = [array[step] for array in arrays if step < len(array)]
        if len(at_risk) >= 2:
            pairwise = _pairwise_rms(np.asarray(at_risk, dtype=np.float64), scale)
            curve[step] = float(np.mean(pairwise))
    return curve


def compute_divergence_curve(
    trajectories: Sequence[Any],
    *,
    workspace_bounds: Any | None = None,
    workspace_scale: Any | None = None,
) -> np.ndarray:
    return normalized_workspace_pose_rms(
        trajectories, workspace_bounds=workspace_bounds, workspace_scale=workspace_scale
    )


def _rows_with_trajectories(data: Any, *, successful_only: bool = False) -> list[object]:
    if isinstance(data, Mapping):
        rows: list[object] = []
        for value in data.values():
            if isinstance(value, Mapping) and "rollouts" in value:
                value = value["rollouts"]
            if isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
                rows.extend(list(value))
            else:
                rows.append(value)
    else:
        rows = list(data)
    if successful_only:
        rows = [row for row in rows if _success(row)]
    return rows


def _task_id(row: object) -> object:
    if isinstance(row, Mapping):
        return row.get("task_id", "__all_tasks__")
    return getattr(row, "task_id", "__all_tasks__")


def _pairwise_values_by_task(data: Any) -> list[float]:
    rows = _rows_with_trajectories(data, successful_only=True)
    grouped: dict[object, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        grouped[_task_id(row)].append(_trajectory(row))
    values: list[float] = []
    for trajectories in grouped.values():
        if len(trajectories) < 2:
            continue
        dimension = trajectories[0].shape[1]
        if any(array.shape[1] != dimension for array in trajectories):
            raise ValueError("successful same-task trajectories have inconsistent dimensions")
        scale = _workspace_scale(dimension)
        horizon = max(len(array) for array in trajectories)
        for step in range(horizon):
            at_risk = [array[step] for array in trajectories if step < len(array)]
            if len(at_risk) >= 2:
                values.extend(_pairwise_rms(np.asarray(at_risk), scale).tolist())
    return values


def matched_time_tau(
    successful_episodes: Any,
    *,
    workspace_bounds: Any | None = None,
    workspace_scale: Any | None = None,
    quantile: float = TAU_QUANTILE,
) -> float | None:
    """Derive tau from successful same-task episodes at matched time indices."""

    if not 0.0 < float(quantile) < 1.0:
        raise ValueError("tau quantile must lie strictly between 0 and 1")
    rows = _rows_with_trajectories(successful_episodes, successful_only=True)
    if not rows:
        return None
    grouped: dict[object, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        grouped[_task_id(row)].append(_trajectory(row))
    values: list[float] = []
    for trajectories in grouped.values():
        if len(trajectories) < 2:
            continue
        dimension = trajectories[0].shape[1]
        if any(array.shape[1] != dimension for array in trajectories):
            raise ValueError("successful same-task trajectories have inconsistent dimensions")
        scale = _workspace_scale(
            dimension, workspace_bounds=workspace_bounds, workspace_scale=workspace_scale
        )
        horizon = max(len(array) for array in trajectories)
        for step in range(horizon):
            at_risk = [array[step] for array in trajectories if step < len(array)]
            if len(at_risk) >= 2:
                values.extend(_pairwise_rms(np.asarray(at_risk), scale).tolist())
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), float(quantile)))


def _first_crossing(curve: np.ndarray, tau: float | Sequence[float]) -> int | None:
    threshold = np.asarray(tau, dtype=np.float64)
    if threshold.ndim == 0:
        threshold = np.full(curve.shape, float(threshold), dtype=np.float64)
    elif threshold.shape != curve.shape:
        raise ValueError("tau curve must have the same length as D(t)")
    valid = np.isfinite(curve) & np.isfinite(threshold) & (curve > threshold)
    indices = np.flatnonzero(valid)
    return None if len(indices) == 0 else int(indices[0])


def divergence_onset(
    trajectories: Sequence[Any],
    tau: float,
    *,
    workspace_bounds: Any | None = None,
    workspace_scale: Any | None = None,
) -> dict[str, Any]:
    curve = normalized_workspace_pose_rms(
        trajectories, workspace_bounds=workspace_bounds, workspace_scale=workspace_scale
    )
    if curve.size == 0:
        return {"t_div": None, "episode_length": 0, "fraction": None, "censored": True, "curve": curve}
    threshold = np.asarray(tau, dtype=np.float64)
    if threshold.ndim == 0:
        threshold = np.full(curve.shape, float(threshold), dtype=np.float64)
    elif threshold.shape != curve.shape:
        raise ValueError("tau curve must have the same length as D(t)")
    comparable = np.isfinite(curve) & np.isfinite(threshold)
    comparable_steps = np.flatnonzero(comparable)
    if comparable_steps.size == 0:
        return {
            "t_div": None,
            "episode_length": int(curve.size),
            "fraction": None,
            "censored": True,
            "comparable_step_count": 0,
            "last_comparable_step": None,
            "curve": curve,
        }
    crossing_steps = np.flatnonzero(comparable & (curve > threshold))
    crossing = None if crossing_steps.size == 0 else int(crossing_steps[0])
    # A task-matched tau(t) can end before a long failed episode when fewer
    # than two successful reference trajectories remain at risk.  Right-
    # censor at the last genuinely comparable control step; using the family
    # horizon would manufacture a late divergence onset from missing tau.
    t_div = int(comparable_steps[-1]) if crossing is None else crossing
    return {
        "t_div": t_div,
        "episode_length": int(curve.size),
        "fraction": float(t_div / max(1, curve.size)),
        "censored": crossing is None,
        "comparable_step_count": int(comparable_steps.size),
        "last_comparable_step": int(comparable_steps[-1]),
        "curve": curve,
    }


def compute_s3(
    data: Any,
    *,
    tau: float | None = None,
    successful_episodes: Any | None = None,
    workspace_bounds: Any | None = None,
    workspace_scale: Any | None = None,
    n: int = BASE_CANDIDATE_COUNT,
) -> dict[str, Any]:
    """Evaluate the frozen prefix gate among near-all-fail families."""

    groups = group_families(data)
    collapse = family_collapse_metrics(groups, n=n)
    near_groups = {
        family: group
        for family, group in groups.items()
        if sum(_success(row) for row in group) <= NEAR_ALL_FAIL_MAX_SUCCESS
    }
    if successful_episodes is None:
        successful_episodes = [
            row for group in groups.values() for row in group if _success(row)
        ]
    if tau is None:
        tau = matched_time_tau(
            successful_episodes,
            workspace_bounds=workspace_bounds,
            workspace_scale=workspace_scale,
        )
    records: list[dict[str, Any]] = []
    if tau is not None and np.isfinite(float(tau)):
        for family, group in sorted(near_groups.items(), key=lambda item: str(item[0])):
            onset = divergence_onset(
                [_trajectory(row) for row in group],
                float(tau),
                workspace_bounds=workspace_bounds,
                workspace_scale=workspace_scale,
            )
            records.append({"family_id": _json_key(family), **onset})
    fractions = [float(row["fraction"]) for row in records if row["fraction"] is not None]
    origins = [row for row in records if row["t_div"] == 0]
    origin_fraction = float(len(origins) / len(records)) if records else None
    median_fraction = float(np.median(np.asarray(fractions))) if fractions else None
    missing_tau_family_count = len(near_groups) - len(records)
    pass_gate = bool(
        records
        and missing_tau_family_count == 0
        and median_fraction is not None
        and median_fraction >= S3_MIN_MEDIAN_DIVERGENCE_FRACTION
        and origin_fraction is not None
        and origin_fraction < S3_MAX_ORIGIN_FRACTION
    )
    return {
        "tau": None if tau is None else float(tau),
        "tau_quantile": TAU_QUANTILE,
        "near_all_fail_family_count": len(near_groups),
        "evaluable_family_count": len(records),
        "missing_tau_family_count": missing_tau_family_count,
        "t_div_records": records,
        "median_t_div_fraction": median_fraction,
        "origin_t_div_count": len(origins),
        "origin_t_div_fraction": origin_fraction,
        "origin_dominant": bool(
            origin_fraction is not None and origin_fraction >= S3_MAX_ORIGIN_FRACTION
        ),
        "pass": pass_gate,
    }


def _read_field(row: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return None


def _branch_recovery(branches: Iterable[Any]) -> tuple[bool, int]:
    branch_list = list(branches)
    recovered = False
    for branch in branch_list:
        if not isinstance(branch, Mapping):
            raise TypeError("branch record must be a mapping")
        if not bool(_read_field(branch, "prefix_preserving", "prefix_preserved")):
            continue
        split_step = _read_field(branch, "split_step", "t", "branch_step")
        episode_length = _read_field(branch, "episode_length", "horizon", "length")
        if split_step is None or episode_length is None:
            raise ValueError("branch record must include split step and episode length")
        if not (0 < int(split_step) < int(episode_length) - 1):
            # Boundary branches are valid controls but cannot contribute to
            # the recoverability gate, which is explicitly interior-only.
            continue
        if bool(_read_field(branch, "success", "eventual_success")):
            recovered = True
    return recovered, len(branch_list)


def _probe_outcomes(probe: Mapping[str, Any]) -> tuple[bool, bool, int | None]:
    oracle = _read_field(probe, "oracle_recovered", "oracle_success")
    random = _read_field(probe, "random_recovered", "random_success")
    oracle_count = _read_field(probe, "oracle_branch_count", "branch_count")
    random_count = _read_field(probe, "random_branch_count")
    oracle_branches = _read_field(probe, "oracle_branches", "branches")
    random_branches = _read_field(probe, "random_branches", "random_probe_branches")
    if oracle is None:
        if oracle_branches is None:
            raise ValueError("probe lacks oracle outcome or oracle branches")
        oracle, inferred = _branch_recovery(oracle_branches)
        if oracle_count is None:
            oracle_count = inferred
    if random is None:
        if random_branches is None:
            raise ValueError("probe lacks random outcome or random branches")
        random, inferred = _branch_recovery(random_branches)
        if random_count is None:
            random_count = inferred
    if oracle_count is not None and random_count is not None and int(oracle_count) != int(random_count):
        raise ValueError("oracle and random probes must use equal branch counts")
    if oracle_count is not None and int(oracle_count) <= 0:
        raise ValueError("branch count must be positive")
    return bool(oracle), bool(random), None if oracle_count is None else int(oracle_count)


def paired_bootstrap_recovery(
    oracle_recovered: Sequence[bool],
    random_recovered: Sequence[bool],
    *,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    replicates: int = S4_BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Paired bootstrap with exactly the preregistered 10,000 replicates."""

    if int(replicates) != S4_BOOTSTRAP_REPLICATES:
        raise ValueError("Stage-S S4 bootstrap is frozen at exactly 10000 replicates")
    oracle = np.asarray(oracle_recovered, dtype=np.float64)
    random = np.asarray(random_recovered, dtype=np.float64)
    if oracle.shape != random.shape or oracle.ndim != 1 or oracle.size == 0:
        raise ValueError("paired bootstrap arrays must be non-empty and equal length")
    differences = oracle - random
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, differences.size, size=(S4_BOOTSTRAP_REPLICATES, differences.size))
    bootstrap_means = np.mean(differences[indices], axis=1)
    alpha = (1.0 - S4_BOOTSTRAP_CI_LEVEL) / 2.0
    ci = np.quantile(bootstrap_means, [alpha, 1.0 - alpha])
    return {
        "replicates": S4_BOOTSTRAP_REPLICATES,
        "seed": int(seed),
        "oracle_rate": float(np.mean(oracle)),
        "random_rate": float(np.mean(random)),
        "paired_rate_difference": float(np.mean(differences)),
        "ci_lower": float(ci[0]),
        "ci_upper": float(ci[1]),
    }


def compute_s4(
    probes: Sequence[Mapping[str, Any]],
    *,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    replicates: int = S4_BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    if not probes:
        raise ValueError("at least one near-all-fail family must be probed")
    oracle_results: list[bool] = []
    random_results: list[bool] = []
    branch_counts: list[int | None] = []
    for probe in probes:
        if not isinstance(probe, Mapping):
            raise TypeError("S4 probes must be mappings")
        oracle, random, branch_count = _probe_outcomes(probe)
        oracle_results.append(oracle)
        random_results.append(random)
        branch_counts.append(branch_count)
    bootstrap = paired_bootstrap_recovery(
        oracle_results, random_results, seed=seed, replicates=replicates
    )
    oracle_rate = bootstrap["oracle_rate"]
    pass_gate = bool(
        oracle_rate >= S4_MIN_ORACLE_RECOVERY_FRACTION and bootstrap["ci_lower"] > 0.0
    )
    return {
        "probed_family_count": len(probes),
        "oracle_recovered_count": int(sum(oracle_results)),
        "random_recovered_count": int(sum(random_results)),
        "oracle_recovery_fraction": oracle_rate,
        "random_recovery_fraction": bootstrap["random_rate"],
        "equal_branch_counts": len(set(branch_counts)) <= 1
        if all(value is not None for value in branch_counts)
        else None,
        "oracle_upper_bound": True,
        "bootstrap": bootstrap,
        "pass": pass_gate,
    }


def _summary_groups(data: Any, expected: int, aliases: Sequence[str]) -> dict[object, tuple[int, list[object] | None]]:
    """Read raw rows or explicit success-count summaries without inference."""

    if isinstance(data, Mapping):
        source = list(data.items())
    else:
        rows = list(data)
        if rows and all(isinstance(row, Mapping) and any(key in row for key in aliases) for row in rows):
            source = [(_read_field(row, "family_id", "family", "initial_state", "init_state"), row) for row in rows]
        else:
            grouped = group_families(rows)
            source = list(grouped.items())
    output: dict[object, tuple[int, list[object] | None]] = {}
    for family, value in source:
        summary: Mapping[str, Any] | None = value if isinstance(value, Mapping) else None
        if summary is not None and any(alias in summary for alias in aliases):
            count_value = _read_field(summary, *aliases)
            count = int(count_value)
            group_rows: list[object] | None = None
            if "rollouts" in summary:
                group_rows = list(summary["rollouts"])
            elif "candidates" in summary and isinstance(summary["candidates"], Iterable):
                group_rows = list(summary["candidates"])
        else:
            if isinstance(value, Mapping) and "rollouts" in value:
                value = value["rollouts"]
            if not isinstance(value, Iterable) or isinstance(value, (str, bytes, Mapping)):
                raise TypeError(f"family {family!r} must contain raw rows or a count summary")
            group_rows = list(value)
            _require_count(group_rows, expected, family)
            count = int(sum(_success(row) for row in group_rows))
        output[family] = (count, group_rows)
    return output


def compute_s5(
    base_data: Any,
    extended_data: Any,
    *,
    n_base: int = BASE_CANDIDATE_COUNT,
    n_extended: int = EXTENDED_CANDIDATE_COUNT,
) -> dict[str, Any]:
    """Evaluate fresh-seed N32->64 rescue, fail-closed on missing freshness proof."""

    if int(n_base) != BASE_CANDIDATE_COUNT or int(n_extended) != EXTENDED_CANDIDATE_COUNT:
        raise ValueError("Stage-S S5 is frozen at N=32 versus N=64")
    base = _summary_groups(
        base_data,
        n_base,
        aliases=("success_count_32", "base_success_count", "successes_32"),
    )
    extended = _summary_groups(
        extended_data,
        n_extended,
        aliases=("success_count_64", "extended_success_count", "successes_64"),
    )
    near_families = [family for family, (count, _) in base.items() if count <= NEAR_ALL_FAIL_MAX_SUCCESS]
    if not near_families:
        return {
            "near_all_fail_family_count": 0,
            "rescued_family_count": 0,
            "rescue_fraction": None,
            "fresh_seed_verified": False,
            "pass": False,
        }
    rescued = 0
    freshness_values: list[bool] = []
    for family in near_families:
        if family not in extended:
            raise ValueError(f"extended N=64 data lacks base family {family!r}")
        extended_count, extended_rows = extended[family]
        if extended_count > NEAR_ALL_FAIL_MAX_SUCCESS:
            rescued += 1
        base_rows = base[family][1]
        if base_rows is None or extended_rows is None:
            freshness_values.append(False)
            continue
        base_ids = []
        for row in base_rows:
            if isinstance(row, Mapping):
                identifier = _read_field(row, "seed", "candidate_seed", "rollout_seed", "candidate_id")
            else:
                identifier = None
            if identifier is not None:
                base_ids.append(identifier)
        extended_ids = []
        explicit_flags = []
        for row in extended_rows:
            if isinstance(row, Mapping):
                identifier = _read_field(row, "seed", "candidate_seed", "rollout_seed", "candidate_id")
                flag = _read_field(row, "fresh_seed", "fresh")
                if flag is not None:
                    explicit_flags.append(bool(flag))
            else:
                identifier = None
            if identifier is not None:
                extended_ids.append(identifier)
        # If row-level identifiers are present, all extended candidates must be
        # new relative to N=32. Otherwise the run cannot prove fresh-seed S5.
        freshness_values.append(
            bool(
                extended_ids
                and len(set(extended_ids)) == len(extended_ids)
                and not set(base_ids).intersection(extended_ids)
                and (not explicit_flags or all(explicit_flags))
            )
        )
    fresh_verified = bool(freshness_values) and all(freshness_values)
    rescue_fraction = float(rescued / len(near_families))
    return {
        "near_all_fail_family_count": len(near_families),
        "rescued_family_count": rescued,
        "rescue_fraction": rescue_fraction,
        "fresh_seed_verified": fresh_verified,
        "pass": bool(fresh_verified and rescue_fraction < S5_MAX_RESCUE_FRACTION),
    }


def evaluate_substrate(
    rollouts: Any,
    *,
    probes: Sequence[Mapping[str, Any]],
    extended_rollouts: Any,
    substrate: str | None = None,
    tau: float | None = None,
    successful_episodes: Any | None = None,
    workspace_bounds: Any | None = None,
    workspace_scale: Any | None = None,
) -> dict[str, Any]:
    """Compute all five gates from one completed substrate screen."""

    s1 = compute_s1(rollouts)
    s2 = compute_s2(rollouts)
    s3 = compute_s3(
        rollouts,
        tau=tau,
        successful_episodes=successful_episodes,
        workspace_bounds=workspace_bounds,
        workspace_scale=workspace_scale,
    )
    s4 = compute_s4(probes)
    s5 = compute_s5(rollouts, extended_rollouts)
    gates = {"S1": s1, "S2": s2, "S3": s3, "S4": s4, "S5": s5}
    return {
        **gates,
        "s1": s1,
        "s2": s2,
        "s3": s3,
        "s4": s4,
        "s5": s5,
        "pass": bool(all(bool(value["pass"]) for value in gates.values())),
    }


def _gate_value(item: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    aliases = (name, name.lower())
    for alias in aliases:
        value = item.get(alias)
        if isinstance(value, Mapping):
            return value
    gates = item.get("gates")
    if isinstance(gates, Mapping):
        for alias in aliases:
            value = gates.get(alias)
            if isinstance(value, Mapping):
                return value
    raise ValueError(f"substrate result lacks {name}")


def _full_pass(item: Mapping[str, Any]) -> bool:
    return all(bool(_gate_value(item, name).get("pass", False)) for name in ("S1", "S2", "S3", "S4", "S5"))


def decide_stage_s(
    substrates: Mapping[str, Mapping[str, Any]],
    *,
    positive_control_pass: bool | Mapping[str, Any],
) -> str:
    """Emit one and only one Stage-S decision code.

    A/B are headline substrates. C is mechanism-isolation only and can produce
    WEAK_SUBSTRATE_ONLY, never SUBSTRATE_QUALIFIED.
    """

    required = {"A", "B", "C"}
    missing = required - set(substrates)
    if missing:
        raise ValueError(f"substrate results missing {sorted(missing)}")
    if isinstance(positive_control_pass, Mapping):
        positive_ok = bool(positive_control_pass.get("pass", False))
    else:
        positive_ok = bool(positive_control_pass)
    if not positive_ok:
        return "PIPELINE_INVALID"
    if _full_pass(substrates["A"]) or _full_pass(substrates["B"]):
        return "SUBSTRATE_QUALIFIED"
    if _full_pass(substrates["C"]):
        return "WEAK_SUBSTRATE_ONLY"
    # The literal "all three" target-difficulty rule is evaluated before
    # later failure explanations.
    if all(not bool(_gate_value(substrates[name], "S1").get("pass", False)) for name in required):
        return "NO_SUBSTRATE_AT_TARGET_DIFFICULTY"
    primary = (substrates["A"], substrates["B"])
    if all(not bool(_gate_value(item, "S2").get("pass", False)) for item in primary):
        return "NO_FAMILY_COLLAPSE"
    origin_failure = any(
        not bool(_gate_value(item, "S3").get("pass", False))
        and bool(_gate_value(item, "S3").get("origin_dominant", False))
        for item in primary
    )
    if origin_failure:
        return "COLLAPSE_AT_ORIGIN"
    if any(not bool(_gate_value(item, "S4").get("pass", False)) for item in primary):
        return "UNRECOVERABLE_FAILURES"
    if any(not bool(_gate_value(item, "S5").get("pass", False)) for item in primary):
        return "BUDGET_SUFFICES"
    if any(not bool(_gate_value(item, "S3").get("pass", False)) for item in primary):
        return "UNRECOVERABLE_FAILURES"
    # No unclassified gate is allowed to silently look like a pass.
    return "UNRECOVERABLE_FAILURES"


# Friendly aliases used by analysis/report scripts.
compute_tau = matched_time_tau
divergence_curve = normalized_workspace_pose_rms
s4_oracle_vs_random = compute_s4
s5_budget_rescue = compute_s5
decide = decide_stage_s


__all__ = [
    "BASE_CANDIDATE_COUNT",
    "DEFAULT_BOOTSTRAP_SEED",
    "DECISION_CODES",
    "EXTENDED_CANDIDATE_COUNT",
    "NEAR_ALL_FAIL_MAX_SUCCESS",
    "S1_MAX_SUCCESS",
    "S1_MIN_SUCCESS",
    "S2_BINOMIAL_EXPECTATION_MULTIPLIER",
    "S2_MIN_NEAR_ALL_FAIL_FRACTION",
    "S2_MIN_RHO",
    "S3_MAX_ORIGIN_FRACTION",
    "S3_MIN_MEDIAN_DIVERGENCE_FRACTION",
    "S4_BOOTSTRAP_CI_LEVEL",
    "S4_BOOTSTRAP_REPLICATES",
    "S4_MIN_ORACLE_RECOVERY_FRACTION",
    "S5_MAX_RESCUE_FRACTION",
    "TAU_QUANTILE",
    "compute_divergence_curve",
    "compute_s1",
    "compute_s2",
    "compute_s3",
    "compute_s4",
    "compute_s5",
    "compute_tau",
    "decide",
    "decide_stage_s",
    "divergence_curve",
    "divergence_onset",
    "evaluate_substrate",
    "family_collapse_metrics",
    "group_families",
    "matched_time_tau",
    "normalized_workspace_pose_rms",
    "paired_bootstrap_recovery",
    "pooled_success_rate",
    "s4_oracle_vs_random",
    "s5_budget_rescue",
    "S4_BOOTSTRAP_SEED",
    "S3_SUBSTRATE_WORKSPACE_SCALES",
    "compute_s3_production",
    "compute_s4_from_protocol",
]

def matched_time_tau_curve(
    successful_episodes: Any,
    *,
    workspace_bounds: Any | None = None,
    workspace_scale: Any | None = None,
    quantile: float = TAU_QUANTILE,
) -> dict[object, np.ndarray]:
    """Derive a separate tau(t) curve for every task.

    Distances are computed only between successful episodes with the same
    task_id and the same control step t. The quantile is taken across matched
    episode pairs at each t; values from different tasks or times are never
    pooled into a scalar threshold.
    """

    if not 0.0 < float(quantile) < 1.0:
        raise ValueError("tau quantile must lie strictly between 0 and 1")
    rows = _rows_with_trajectories(successful_episodes, successful_only=True)
    grouped: dict[object, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        grouped[_task_id(row)].append(_trajectory(row))
    curves: dict[object, np.ndarray] = {}
    for task, trajectories in grouped.items():
        if len(trajectories) < 2:
            continue
        dimension = trajectories[0].shape[1]
        if any(array.shape[1] != dimension for array in trajectories):
            raise ValueError("successful same-task trajectories have inconsistent dimensions")
        scale = _workspace_scale(
            dimension, workspace_bounds=workspace_bounds, workspace_scale=workspace_scale
        )
        horizon = max(len(array) for array in trajectories)
        curve = np.full(horizon, np.nan, dtype=np.float64)
        for step in range(horizon):
            at_risk = [array[step] for array in trajectories if step < len(array)]
            if len(at_risk) >= 2:
                values = _pairwise_rms(np.asarray(at_risk), scale)
                curve[step] = float(np.quantile(values, float(quantile)))
        curves[task] = curve
    return curves


def _task_tau_curve(
    tau_curves: Mapping[object, Any] | None,
    task: object,
    horizon: int,
    *,
    fallback: float | None = None,
) -> np.ndarray | None:
    if tau_curves is None:
        if fallback is None:
            return None
        return np.repeat(float(fallback), horizon).astype(np.float64)
    value = tau_curves.get(task)
    if value is None:
        value = tau_curves.get(_json_key(task))
    if value is None:
        value = tau_curves.get("__all_tasks__")
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return np.repeat(float(array), horizon).astype(np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("task tau curve must be a non-empty one-dimensional array")
    if array.size >= horizon:
        return array[:horizon]
    # Successful episodes may terminate earlier than a failed family.  Missing
    # matched-time reference support is represented explicitly as NaN so the
    # onset is right-censored at the final comparable control step.
    padded = np.full(horizon, np.nan, dtype=np.float64)
    padded[: array.size] = array
    return padded




# Protocol corrections kept at module tail so older report imports remain
# source-compatible while all public names below resolve to the corrected
# implementations.

def compute_s3(
    data: Any,
    *,
    tau: float | None = None,
    successful_episodes: Any | None = None,
    tau_curves: Mapping[object, Any] | None = None,
    workspace_bounds: Any | None = None,
    workspace_scale: Any | None = None,
    substrate: str | None = None,
    n: int = BASE_CANDIDATE_COUNT,
) -> dict[str, Any]:
    """Evaluate S3 with task-matched tau(t).

    Passing ``substrate`` selects the strict production contract: the
    substrate-specific pose scale is mandatory, tau is derived from successful
    same-task matched-time references, and scalar/externally supplied tau
    curves are rejected.  The legacy scalar arguments remain available only
    when ``substrate`` is omitted for backwards-compatible unit diagnostics.
    """

    if substrate is not None:
        if substrate not in S3_SUBSTRATE_WORKSPACE_SCALES:
            raise ValueError(f"unsupported S3 substrate: {substrate!r}")
        if tau is not None:
            raise ValueError("production S3 forbids a scalar tau override")
        if tau_curves is not None:
            raise ValueError("production S3 derives tau from successful same-task matched-time references")
        if workspace_bounds is not None:
            raise ValueError("production S3 forbids workspace bounds; use the frozen substrate scale")
        expected_scale = S3_SUBSTRATE_WORKSPACE_SCALES[substrate]
        if workspace_scale is None:
            workspace_scale = expected_scale
        elif tuple(float(value) for value in np.asarray(workspace_scale).reshape(-1)) != expected_scale:
            raise ValueError(f"production S3 workspace scale drift for substrate {substrate}")

    groups = group_families(data)
    family_collapse_metrics(groups, n=n)
    near_groups = {
        family: group
        for family, group in groups.items()
        if sum(_success(row) for row in group) <= NEAR_ALL_FAIL_MAX_SUCCESS
    }
    if successful_episodes is None:
        successful_episodes = [
            row for group in groups.values() for row in group if _success(row)
        ]
    tau_source = "explicit_scalar"
    if tau_curves is None:
        if tau is None:
            tau_curves = matched_time_tau_curve(
                successful_episodes,
                workspace_bounds=workspace_bounds,
                workspace_scale=workspace_scale,
            )
            tau_source = "successful_same_task_matched_time"
        else:
            tau_curves = {"__all_tasks__": float(tau)}
    records: list[dict[str, Any]] = []
    tau_by_task: dict[str, list[float]] = {}
    for family, group in sorted(near_groups.items(), key=lambda item: str(item[0])):
        if not group:
            continue
        trajectories = [_trajectory(row) for row in group]
        task = _task_id(group[0])
        task_tau = _task_tau_curve(
            tau_curves, task, max(len(array) for array in trajectories)
        )
        if task_tau is None:
            continue
        onset = divergence_onset(
            trajectories,
            task_tau,
            workspace_bounds=workspace_bounds,
            workspace_scale=workspace_scale,
        )
        tau_by_task[_json_key(task)] = task_tau.tolist()
        records.append(
            {
                "family_id": _json_key(family),
                "task_id": _json_key(task),
                "tau_curve": task_tau.tolist(),
                **onset,
            }
        )
    fractions = [float(row["fraction"]) for row in records if row["fraction"] is not None]
    origins = [row for row in records if row["t_div"] == 0]
    origin_fraction = float(len(origins) / len(records)) if records else None
    median_fraction = float(np.median(np.asarray(fractions))) if fractions else None
    missing_tau_family_count = len(near_groups) - len(records)
    pass_gate = bool(
        records
        and missing_tau_family_count == 0
        and median_fraction is not None
        and median_fraction >= S3_MIN_MEDIAN_DIVERGENCE_FRACTION
        and origin_fraction is not None
        and origin_fraction < S3_MAX_ORIGIN_FRACTION
    )
    return {
        "tau": None if tau is None else float(tau),
        "tau_by_task": tau_by_task,
        "substrate": substrate,
        "workspace_scale": None if workspace_scale is None else [float(value) for value in np.asarray(workspace_scale).reshape(-1)],
        "tau_override_forbidden": substrate is not None,
        "tau_source": tau_source,
        "tau_quantile": TAU_QUANTILE,
        "near_all_fail_family_count": len(near_groups),
        "evaluable_family_count": len(records),
        "missing_tau_family_count": missing_tau_family_count,
        "t_div_records": records,
        "median_t_div_fraction": median_fraction,
        "origin_t_div_count": len(origins),
        "origin_t_div_fraction": origin_fraction,
        "origin_dominant": bool(
            origin_fraction is not None and origin_fraction >= S3_MAX_ORIGIN_FRACTION
        ),
        "pass": pass_gate,
    }


def _seed_identifier(row: object) -> object | None:
    if not isinstance(row, Mapping):
        return None
    return _read_field(row, "seed", "candidate_seed", "rollout_seed", "candidate_id")


def _extension_freshness(
    base_rows: Sequence[object] | None,
    extended_rows: Sequence[object] | None,
) -> tuple[bool, int]:
    """Verify a 32+32 extension without charging an accidental 96 candidates."""

    if base_rows is None or extended_rows is None:
        return False, 0
    if len(base_rows) != BASE_CANDIDATE_COUNT or len(extended_rows) != EXTENDED_CANDIDATE_COUNT:
        return False, 0
    base_ids = [_seed_identifier(row) for row in base_rows]
    extended_ids = [_seed_identifier(row) for row in extended_rows]
    if any(value is None for value in base_ids + extended_ids):
        return False, 0
    if len(set(base_ids)) != BASE_CANDIDATE_COUNT:
        return False, 0
    if len(set(extended_ids)) != EXTENDED_CANDIDATE_COUNT:
        return False, 0
    base_set = set(base_ids)
    extended_set = set(extended_ids)
    # The total-N=64 run must contain the original N=32 exactly once plus
    # exactly 32 new seeds. It must not be a second disjoint N=64 run.
    if not base_set.issubset(extended_set):
        return False, 0
    added_ids = extended_set - base_set
    if len(added_ids) != BASE_CANDIDATE_COUNT:
        return False, 0
    base_success = {
        identifier: _success(row) for identifier, row in zip(base_ids, base_rows)
    }
    extended_success = {
        identifier: _success(row) for identifier, row in zip(extended_ids, extended_rows)
    }
    if any(base_success[identifier] != extended_success[identifier] for identifier in base_set):
        return False, 0
    # If a producer emits a fresh flag, it is an auditable assertion for only
    # the added candidates. No flag is required when disjoint seed identity
    # proves freshness directly.
    added_flags = [
        _read_field(row, "fresh_seed", "fresh")
        for identifier, row in zip(extended_ids, extended_rows)
        if identifier in added_ids
    ]
    if any(flag is not None and not bool(flag) for flag in added_flags):
        return False, 0
    return True, len(added_ids)


def compute_s5(
    base_data: Any,
    extended_data: Any,
    *,
    n_base: int = BASE_CANDIDATE_COUNT,
    n_extended: int = EXTENDED_CANDIDATE_COUNT,
) -> dict[str, Any]:
    """Evaluate fresh-seed N32->64 rescue using a 32+32 total."""

    if int(n_base) != BASE_CANDIDATE_COUNT or int(n_extended) != EXTENDED_CANDIDATE_COUNT:
        raise ValueError("Stage-S S5 is frozen at N=32 versus N=64")
    base = _summary_groups(
        base_data,
        n_base,
        aliases=("success_count_32", "base_success_count", "successes_32"),
    )
    extended = _summary_groups(
        extended_data,
        n_extended,
        aliases=("success_count_64", "extended_success_count", "successes_64"),
    )
    near_families = [
        family for family, (count, _) in base.items()
        if count <= NEAR_ALL_FAIL_MAX_SUCCESS
    ]
    if not near_families:
        return {
            "near_all_fail_family_count": 0,
            "rescued_family_count": 0,
            "rescue_fraction": None,
            "fresh_seed_verified": False,
            "total_candidate_count": EXTENDED_CANDIDATE_COUNT,
            "added_fresh_candidate_count": BASE_CANDIDATE_COUNT,
            "pass": False,
        }
    rescued = 0
    freshness_values: list[bool] = []
    added_counts: list[int] = []
    for family in near_families:
        if family not in extended:
            raise ValueError(f"extended N=64 data lacks base family {family!r}")
        extended_count, extended_rows = extended[family]
        if extended_count > NEAR_ALL_FAIL_MAX_SUCCESS:
            rescued += 1
        fresh_ok, added_count = _extension_freshness(base[family][1], extended_rows)
        freshness_values.append(fresh_ok)
        added_counts.append(added_count)
    fresh_verified = bool(freshness_values) and all(freshness_values)
    rescue_fraction = float(rescued / len(near_families))
    return {
        "near_all_fail_family_count": len(near_families),
        "rescued_family_count": rescued,
        "rescue_fraction": rescue_fraction,
        "fresh_seed_verified": fresh_verified,
        "total_candidate_count": EXTENDED_CANDIDATE_COUNT,
        "added_fresh_candidate_count": sorted(set(added_counts))[0] if added_counts else 0,
        "pass": bool(fresh_verified and rescue_fraction < S5_MAX_RESCUE_FRACTION),
    }


def evaluate_substrate(
    rollouts: Any,
    *,
    probes: Sequence[Mapping[str, Any]],
    extended_rollouts: Any,
    substrate: str | None = None,
    tau: float | None = None,
    tau_curves: Mapping[object, Any] | None = None,
    successful_episodes: Any | None = None,
    workspace_bounds: Any | None = None,
    workspace_scale: Any | None = None,
) -> dict[str, Any]:
    """Compute all five gates from one completed substrate screen."""

    s1 = compute_s1(rollouts)
    s2 = compute_s2(rollouts)
    s3 = compute_s3(
        rollouts,
        tau=tau,
        tau_curves=tau_curves,
        successful_episodes=successful_episodes,
        workspace_bounds=workspace_bounds,
        workspace_scale=workspace_scale,
        substrate=substrate,
    )
    s4 = compute_s4(probes)
    s5 = compute_s5(rollouts, extended_rollouts)
    gates = {"S1": s1, "S2": s2, "S3": s3, "S4": s4, "S5": s5}
    return {
        **gates,
        "s1": s1,
        "s2": s2,
        "s3": s3,
        "s4": s4,
        "s5": s5,
        "pass": bool(all(bool(value["pass"]) for value in gates.values())),
    }


def decide_stage_s(
    substrates: Mapping[str, Mapping[str, Any]],
    *,
    positive_control_pass: bool | Mapping[str, Any],
) -> str:
    """Return exactly one protocol decision after depth-wise survivor pruning.

    The total analyzer evaluates every arm/gate first. This function only
    applies the frozen precedence: A/B are headline arms, C is weak-only, and
    a failed gate removes only arms that failed that gate before the next
    depth. It never stops collection of later gates.
    """

    required = {"A", "B", "C"}
    missing = required - set(substrates)
    if missing:
        raise ValueError(f"substrate results missing {sorted(missing)}")
    positive_ok = bool(positive_control_pass.get("pass", False)) if isinstance(positive_control_pass, Mapping) else bool(positive_control_pass)
    if not positive_ok:
        return "PIPELINE_INVALID"
    if _full_pass(substrates["A"]) or _full_pass(substrates["B"]):
        return "SUBSTRATE_QUALIFIED"
    if _full_pass(substrates["C"]):
        return "WEAK_SUBSTRATE_ONLY"
    if all(not bool(_gate_value(substrates[name], "S1").get("pass", False)) for name in required):
        return "NO_SUBSTRATE_AT_TARGET_DIFFICULTY"

    active = [substrates[name] for name in ("A", "B") if bool(_gate_value(substrates[name], "S1").get("pass", False))]
    if not active:
        active = [substrates["C"]]
    for gate in ("S2", "S3", "S4", "S5"):
        survivors = [item for item in active if bool(_gate_value(item, gate).get("pass", False))]
        if survivors:
            active = survivors
            continue
        if gate == "S2":
            return "NO_FAMILY_COLLAPSE"
        if gate == "S3":
            return "COLLAPSE_AT_ORIGIN" if any(bool(_gate_value(item, "S3").get("origin_dominant", False)) for item in active) else "UNRECOVERABLE_FAILURES"
        if gate == "S4":
            return "UNRECOVERABLE_FAILURES"
        if gate == "S5":
            return "BUDGET_SUFFICES"
    return "SUBSTRATE_QUALIFIED"


# Rebind aliases after the corrected definitions.
compute_tau_curves = matched_time_tau_curve
s4_oracle_vs_random = compute_s4
s5_budget_rescue = compute_s5
decide = decide_stage_s

if "matched_time_tau_curve" not in __all__:
    __all__.append("matched_time_tau_curve")


def _extension_freshness(
    base_rows: Sequence[object] | None,
    extended_rows: Sequence[object] | None,
) -> tuple[bool, int]:
    """Verify a 32+32 seed extension; success labels are measured anew at N=64."""

    if base_rows is None or extended_rows is None:
        return False, 0
    if len(base_rows) != BASE_CANDIDATE_COUNT or len(extended_rows) != EXTENDED_CANDIDATE_COUNT:
        return False, 0
    base_ids = [_seed_identifier(row) for row in base_rows]
    extended_ids = [_seed_identifier(row) for row in extended_rows]
    if any(value is None for value in base_ids + extended_ids):
        return False, 0
    if len(set(base_ids)) != BASE_CANDIDATE_COUNT:
        return False, 0
    if len(set(extended_ids)) != EXTENDED_CANDIDATE_COUNT:
        return False, 0
    base_set = set(base_ids)
    extended_set = set(extended_ids)
    # N=64 consists of the original N=32 identities plus exactly 32 new,
    # disjoint identities. This is the only freshness requirement.
    if not base_set.issubset(extended_set):
        return False, 0
    added_ids = extended_set - base_set
    if len(added_ids) != BASE_CANDIDATE_COUNT:
        return False, 0
    base_success = {
        identifier: _success(row) for identifier, row in zip(base_ids, base_rows)
    }
    extended_success = {
        identifier: _success(row) for identifier, row in zip(extended_ids, extended_rows)
    }
    if any(base_success[identifier] != extended_success[identifier] for identifier in base_set):
        return False, 0
    added_flags = [
        _read_field(row, "fresh_seed", "fresh")
        for identifier, row in zip(extended_ids, extended_rows)
        if identifier in added_ids
    ]
    if any(flag is not None and not bool(flag) for flag in added_flags):
        return False, 0
    return True, len(added_ids)


if "matched_time_tau_curve" not in __all__:
    __all__.append("matched_time_tau_curve")


# Strict protocol-bound production entry points.  These wrappers deliberately
# keep the permissive legacy helpers above available for historical fixtures,
# while making it impossible for a production caller to pass scalar tau or a
# non-substrate pose scale by accident.
def compute_s3_production(
    data: Any,
    *,
    substrate: str,
    successful_episodes: Any | None = None,
) -> dict[str, Any]:
    return compute_s3(
        data,
        substrate=substrate,
        successful_episodes=successful_episodes,
        tau=None,
        tau_curves=None,
        workspace_bounds=None,
        workspace_scale=S3_SUBSTRATE_WORKSPACE_SCALES.get(substrate),
    )


def compute_s4_from_protocol(
    probes: Sequence[Mapping[str, Any]],
    protocol: Any,
) -> dict[str, Any]:
    """Run S4 only with protocol-owned seed, nine-point grid and 8/8 pairs."""

    s4 = getattr(protocol, "s4", None)
    if s4 is None and isinstance(protocol, Mapping):
        s4 = protocol.get("s4", protocol.get("S4"))
        if not isinstance(s4, Mapping):
            summary = protocol.get("frozen_summary", {})
            s4 = summary.get("s4", summary.get("S4", {})) if isinstance(summary, Mapping) else {}
    if not isinstance(s4, Mapping):
        raise ValueError("S4 protocol object is missing")
    if isinstance(s4.get("paired_bootstrap_seed"), bool) or not isinstance(s4.get("paired_bootstrap_seed"), (int, np.integer)):
        raise ValueError("S4 protocol paired_bootstrap_seed must be an integer")
    if isinstance(s4.get("paired_bootstrap_replicates"), bool) or not isinstance(s4.get("paired_bootstrap_replicates"), (int, np.integer)):
        raise ValueError("S4 protocol paired_bootstrap_replicates must be an integer")
    try:
        seed = int(s4["paired_bootstrap_seed"])
        replicates = int(s4["paired_bootstrap_replicates"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("S4 protocol must carry paired_bootstrap_seed and paired_bootstrap_replicates") from exc
    if seed != S4_BOOTSTRAP_SEED:
        raise ValueError("S4 bootstrap seed is frozen at 14211")
    if replicates != S4_BOOTSTRAP_REPLICATES:
        raise ValueError("S4 bootstrap replicates are frozen at 10000")
    grid = s4.get("search_t_grid", s4.get("oracle_search_grid", s4.get("oracle_t_grid")))
    if isinstance(grid, Mapping):
        grids = list(grid.values())
    else:
        grids = [grid]
    if not grids or any(
        not isinstance(item, (list, tuple))
        or len(item) != 9
        or any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in item)
        or len(set(int(value) for value in item)) != 9
        for item in grids
    ):
        raise ValueError("S4 protocol search grid must contain exactly nine distinct integer points")
    for probe in probes:
        if not isinstance(probe, Mapping):
            raise TypeError("S4 probes must be mappings")
        oracle_rows = probe.get("oracle_branches")
        random_rows = probe.get("random_branches")
        declared_oracle = probe.get("oracle_branch_count")
        declared_random = probe.get("random_branch_count")
        if declared_oracle is None and isinstance(oracle_rows, Sequence):
            declared_oracle = len(oracle_rows)
        if declared_random is None and isinstance(random_rows, Sequence):
            declared_random = len(random_rows)
        if int(declared_oracle or -1) != 8 or int(declared_random or -1) != 8:
            raise ValueError("S4 requires equal held-out oracle/random branch counts of exactly 8")
        if isinstance(oracle_rows, Sequence) and not isinstance(oracle_rows, (str, bytes)) and len(oracle_rows) != 8:
            raise ValueError("S4 oracle branch list must contain exactly 8 held-out branches")
        if isinstance(random_rows, Sequence) and not isinstance(random_rows, (str, bytes)) and len(random_rows) != 8:
            raise ValueError("S4 random branch list must contain exactly 8 held-out branches")
        probe_seed = probe.get("paired_bootstrap_seed")
        if probe_seed is not None and int(probe_seed) != seed:
            raise ValueError("S4 probe bootstrap seed disagrees with protocol")
        probe_grid = probe.get("search_grid")
        if probe_grid is not None:
            family_id = str(probe.get("family_id", ""))
            expected_grid = grid.get(family_id, grid.get(str(family_id))) if isinstance(grid, Mapping) else grid
            if not isinstance(probe_grid, (list, tuple)) or not isinstance(expected_grid, (list, tuple)):
                raise ValueError("S4 probe search grid must be an explicit list")
            if len(probe_grid) != 9 or len(set(int(value) for value in probe_grid)) != 9 or list(map(int, probe_grid)) != list(map(int, expected_grid)):
                raise ValueError("S4 probe search grid disagrees with the nine-point protocol grid")
            if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in probe_grid):
                raise ValueError("S4 probe search grid must contain integer points")
    result = compute_s4(probes, seed=seed, replicates=replicates)
    result["protocol_bootstrap_seed"] = seed
    result["protocol_grid_point_count"] = 9
    result["protocol_branch_count_oracle"] = 8
    result["protocol_branch_count_random"] = 8
    result["equal_branch_counts"] = True
    return result


# Explicit names make the strict boundary discoverable to launchers and tests.
production_s3 = compute_s3_production
production_s4 = compute_s4_from_protocol
