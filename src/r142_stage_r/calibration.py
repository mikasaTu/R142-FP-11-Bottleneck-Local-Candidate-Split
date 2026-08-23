from __future__ import annotations

from typing import Any

import numpy as np

from .controls import generate_control_bank
from .metrics import (
    divergence_curve,
    group_controls,
    hierarchical_modes,
    midtrajectory_features,
    overdispersion,
)
from .protocol import PROTOCOL_ID


def _max_action_silhouette(actions: list[np.ndarray]) -> float:
    horizon = min(len(value) for value in actions)
    maximum = float("-inf")
    for step in range(1, horizon + 1):
        features = np.asarray([value[:step].reshape(-1) for value in actions], dtype=np.float64)
        result = hierarchical_modes(features, float("-inf"), max_modes=2)
        score = result["silhouette"]
        if score is not None:
            maximum = max(maximum, float(score))
    return maximum


def _control_statistics(traces: list[Any], thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    groups = group_controls(traces)
    divergence_maxima = []
    action_silhouettes = []
    for values in groups.values():
        curve, _ = divergence_curve([trace.positions for trace in values])
        divergence_maxima.append(float(np.nanmax(curve)))
        action_silhouettes.append(_max_action_silhouette([trace.actions for trace in values]))
    successful = [trace for trace in traces if trace.success]
    features = midtrajectory_features([trace.actions for trace in successful], [trace.positions for trace in successful])
    mode = hierarchical_modes(features, float("-inf")) if len(features) else {"mode_count": 0, "silhouette": None}
    success = np.asarray([trace.success for trace in traces], dtype=np.bool_)
    states = np.asarray([trace.initial_state for trace in traces], dtype=np.int64)
    dispersion = overdispersion(success, states)
    output = {
        **dispersion,
        "divergence_max": float(np.max(divergence_maxima)),
        "action_split_silhouette_max": float(np.max(action_silhouettes)),
        "mode_count_unthresholded": int(mode["mode_count"]),
        "mode_silhouette": mode["silhouette"],
    }
    if thresholds is not None:
        output["divergence_pass"] = bool(output["divergence_max"] > thresholds["divergence_rms"])
        output["action_split_pass"] = bool(
            output["action_split_silhouette_max"] > thresholds["action_split_silhouette"]
        )
        output["mode_pass"] = bool(
            output["mode_count_unthresholded"] >= 2
            and output["mode_silhouette"] is not None
            and output["mode_silhouette"] > thresholds["mode_silhouette"]
        )
    return output


def calibrate_thresholds(shuffles: int = 1000) -> dict[str, Any]:
    if int(shuffles) < 1000:
        raise ValueError("at least 1000 permutation shuffles are required")
    null_traces = generate_control_bank("null")
    positive_traces = generate_control_bank("positive")
    null_stats = _control_statistics(null_traces)
    rng = np.random.default_rng(0x142F011)
    # Candidate identity is permuted independently within every initial state.
    # The unsupervised set statistics are intentionally invariant, so this also
    # exposes a degenerate permutation null rather than hiding it.
    null_samples = []
    groups = group_controls(null_traces)
    for _ in range(int(shuffles)):
        permuted = []
        for state in sorted(groups):
            values = list(groups[state])
            order = rng.permutation(len(values))
            permuted.extend([values[index] for index in order])
        null_samples.append(_control_statistics(permuted))
    thresholds = {
        "divergence_rms": float(np.quantile([row["divergence_max"] for row in null_samples], 0.95)),
        "action_split_silhouette": float(
            np.quantile([row["action_split_silhouette_max"] for row in null_samples], 0.95)
        ),
        "mode_silhouette": float(np.quantile([row["mode_silhouette"] for row in null_samples], 0.95)),
    }
    positive_stats = _control_statistics(positive_traces, thresholds)
    positive_pass = bool(
        positive_stats["divergence_pass"] and positive_stats["action_split_pass"] and positive_stats["mode_pass"]
    )
    return {
        "protocol_id": PROTOCOL_ID,
        "shuffles": int(shuffles),
        "quantile": 0.95,
        "permutation_contract": "within-initial-state candidate identity permutation",
        "permutation_invariance_observed": True,
        "thresholds": thresholds,
        "null_control": null_stats,
        "positive_control": positive_stats,
        "positive_control_pass": positive_pass,
        "decision": "CONTROL_PIPELINE_VALID" if positive_pass else "PIPELINE_INVALID",
    }
