from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .detector import detect_earliest_bottleneck
from .genealogy import CandidateSet


def _entropy(labels: list[str]) -> float:
    if not labels:
        return 0.0
    counts = np.asarray(list(Counter(labels).values()), dtype=np.float64)
    probs = counts / counts.sum()
    return float(-(probs * np.log(np.maximum(probs, 1e-12))).sum())


def candidate_set_metrics(candidate_set: CandidateSet, true_bottleneck_step: int) -> dict[str, Any]:
    candidates = candidate_set.candidates
    selected = candidate_set.selected()
    successful = [candidate for candidate in candidates if candidate.final_success]
    successful_modes = sorted(candidate_set.successful_modes())
    failure_counts = Counter(
        candidate.failure_reason for candidate in candidates if candidate.failure_reason is not None
    )
    mode_labels = [candidate.final_mode or "failure" for candidate in candidates]
    predicted = candidate_set.predicted_bottleneck_step
    return {
        "episode_id": candidate_set.episode_id,
        "policy": candidate_set.policy,
        "candidate_count": len(candidates),
        "success_at_n": int(selected.final_success),
        "any_success_at_n": int(bool(successful)),
        "candidate_success_rate": len(successful) / len(candidates),
        "mode_discovery_rate": len(successful_modes) / 2.0,
        "successful_modes_per_sample": len(successful_modes) / len(candidates),
        "both_modes_discovered": int(len(successful_modes) == 2),
        "mode_entropy": _entropy(mode_labels),
        "upper_success_count": sum(c.final_success and c.final_mode == "upper" for c in candidates),
        "lower_success_count": sum(c.final_success and c.final_mode == "lower" for c in candidates),
        "predicted_bottleneck_step": predicted,
        "true_bottleneck_step": true_bottleneck_step,
        "bottleneck_localization_error": None if predicted is None else abs(predicted - true_bottleneck_step),
        "failure_counts": dict(sorted(failure_counts.items())),
        "split_steps": candidate_set.split_steps,
        "detector_diagnostics": candidate_set.detector_diagnostics,
    }


def collapse_diagnostics(candidate_set: CandidateSet, true_bottleneck_step: int, scout_count: int) -> dict[str, Any]:
    scouts = candidate_set.candidates[:scout_count]
    _, detector = detect_earliest_bottleneck(scouts, scouts[0].actions.shape[0])
    disagreement = detector["disagreement"]
    prior = float(disagreement[max(0, true_bottleneck_step - 1)])
    at = float(disagreement[true_bottleneck_step])
    ratio = prior / max(at, 1e-12)
    labels: list[str] = []
    for candidate in candidate_set.candidates:
        if candidate.final_success and candidate.final_mode is not None:
            labels.append(candidate.final_mode)
        else:
            labels.append("failure")
    purity = max(Counter(labels).values()) / len(labels)
    return {
        "prefix_to_bottleneck_disagreement_ratio": ratio,
        "candidate_family_purity": purity,
        "collapse_valid": bool(ratio <= 0.35 and purity >= 0.80),
        "detector_disagreement": disagreement,
    }


def paired_bootstrap(
    proposed: np.ndarray,
    baseline: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    if proposed.shape != baseline.shape:
        raise ValueError("paired arrays must have identical shape")
    differences = proposed.astype(np.float64) - baseline.astype(np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(replicates, len(differences)))
    bootstrap_means = differences[indices].mean(axis=1)
    return {
        "mean_difference": float(differences.mean()),
        "ci95_low": float(np.quantile(bootstrap_means, 0.025)),
        "ci95_high": float(np.quantile(bootstrap_means, 0.975)),
    }
