from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .genealogy import Candidate


def _median_pairwise_distance(features: np.ndarray) -> float:
    if len(features) < 2:
        return 0.0
    distances: list[float] = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            distances.append(float(np.linalg.norm(features[i] - features[j])))
    return float(np.median(distances))


def detect_earliest_bottleneck(
    scouts: Sequence[Candidate],
    horizon: int,
    ratio_threshold: float = 1.5,
) -> tuple[int | None, dict[str, Any]]:
    """Detect a disagreement onset without terminal labels or oracle truth.

    Only same-time state, action and latent-prefix values are inspected. The
    function deliberately has no parameters for success, mode, failure reason,
    terminal score, future suffix, or the environment truth.
    """
    disagreement = np.zeros(horizon, dtype=np.float64)
    for t in range(horizon):
        features = np.asarray(
            [
                np.concatenate(
                    [candidate.states[t + 1], candidate.actions[t], [candidate.latents[t]]]
                )
                for candidate in scouts
            ],
            dtype=np.float64,
        )
        disagreement[t] = _median_pairwise_distance(features)
    delta = np.diff(disagreement, prepend=disagreement[0])
    robust_center = float(np.median(delta))
    mad = float(np.median(np.abs(delta - robust_center)))
    threshold = robust_center + 3.0 * max(mad, 1e-6)
    ratios = np.ones(horizon, dtype=np.float64)
    ratios[1:] = disagreement[1:] / np.maximum(disagreement[:-1], 1e-8)
    prediction: int | None = None
    for t in range(1, horizon):
        if delta[t] > threshold and ratios[t] > ratio_threshold:
            prediction = t
            break
    diagnostics = {
        "disagreement": disagreement.tolist(),
        "delta_disagreement": delta.tolist(),
        "ratio": ratios.tolist(),
        "threshold": threshold,
        "ratio_threshold": ratio_threshold,
        "detector_inputs": "state_action_latent_prefix_only",
        "terminal_label_access": False,
    }
    return prediction, diagnostics
