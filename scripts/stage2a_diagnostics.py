#!/usr/bin/env python3
"""Independent descriptive diagnostics for the completed Stage-2A falsification run."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


STRATEGIES = (
    "always_early",
    "random",
    "uniform_three_quantiles",
    "oracle_local_crossfit",
)


def load_json(path: Path):
    return json.loads(path.read_text())


def corr(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    # Freeze insignificant BLAS/NumPy reduction drift across the local and
    # pinned PAI/dev14 runtimes so the generated JSON is byte-reproducible.
    return round(float(np.corrcoef(x, y)[0, 1]), 12)


def describe(values) -> dict:
    values = np.asarray(list(values), dtype=float)
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
    }


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [float("nan"), float("nan")]
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [float(center - radius), float(center + radius)]


def checkpoint_row(item: dict, checkpoint: int) -> dict:
    return next(row for row in item["curve"] if row["checkpoint_index"] == checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal", type=Path, required=True)
    parser.add_argument("--continuation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pilot = args.formal / "pilot"
    curves = [load_json(path) for path in sorted((pilot / "branchability_curves").glob("*.json"))]
    fixed = [load_json(path) for path in sorted((pilot / "fixed_nfe").glob("*.json"))]
    continuation = load_json(args.continuation / "continuation" / "eventual_continuation_summary.json")
    if len(curves) != 24 or len(fixed) != 24:
        raise RuntimeError(f"incomplete input: curves={len(curves)} fixed={len(fixed)}")

    spread_ranges = {}
    by_stratum = defaultdict(list)
    pooled_z, pooled_spread, pooled_support = [], [], []
    calibration_spread, heldout_spread = [], []
    exact_oracle_agreement = 0
    oracle_distances = []
    heldout_regrets = []
    for item in curves:
        heldout = np.asarray([row["heldout"]["progress_spread"] for row in item["curve"]])
        z = np.asarray([row["z_disagreement"] for row in item["root_disagreement"]])
        support = np.asarray([row["heldout"]["support_diversity"] for row in item["curve"]])
        value = float(np.ptp(heldout))
        spread_ranges[item["snapshot_id"]] = value
        by_stratum[item["stratum"]].append(value)
        pooled_z.extend(z)
        pooled_spread.extend(heldout)
        pooled_support.extend(support)
        calibration_spread.extend(row["calibration"]["progress_spread"] for row in item["curve"])
        heldout_spread.extend(heldout)

        calibration_oracle = max(
            item["curve"],
            key=lambda row: (
                row["calibration"]["best_descendant_gain"],
                row["calibration"]["progress_spread"],
                -row["checkpoint_index"],
            ),
        )["checkpoint_index"]
        heldout_oracle = max(
            item["curve"],
            key=lambda row: (
                row["heldout"]["best_descendant_gain"],
                row["heldout"]["progress_spread"],
                -row["checkpoint_index"],
            ),
        )["checkpoint_index"]
        exact_oracle_agreement += int(calibration_oracle == heldout_oracle)
        oracle_distances.append(abs(calibration_oracle - heldout_oracle))
        selected_gain = checkpoint_row(item, calibration_oracle)["heldout"]["best_descendant_gain"]
        best_gain = checkpoint_row(item, heldout_oracle)["heldout"]["best_descendant_gain"]
        heldout_regrets.append(best_gain - selected_gain)

    checkpoint_profiles = {}
    for checkpoint in (0, 53, 99):
        checkpoint_profiles[str(checkpoint)] = {
            "z_disagreement": describe(
                next(row["z_disagreement"] for row in item["root_disagreement"] if row["checkpoint_index"] == checkpoint)
                for item in curves
            ),
            "heldout_support_diversity": describe(
                checkpoint_row(item, checkpoint)["heldout"]["support_diversity"] for item in curves
            ),
            "heldout_progress_spread": describe(
                checkpoint_row(item, checkpoint)["heldout"]["progress_spread"] for item in curves
            ),
        }

    strategy_summary = {}
    for name in STRATEGIES:
        strategy_summary[name] = {
            "gain_over_no_branch": describe(row["strategies"][name]["gain_over_no_branch"] for row in fixed),
            "actual_sample_nfe": describe(row["strategies"][name]["actual_sample_nfe"] for row in fixed),
            "budget_slack": describe(row["strategies"][name]["budget_slack"] for row in fixed),
            "generated_suffixes": describe(row["strategies"][name]["generated_suffixes"] for row in fixed),
            "wall_clock_seconds": describe(row["strategies"][name]["wall_clock_seconds"] for row in fixed),
        }

    continuation_intervals = {}
    for checkpoint, row in continuation["by_checkpoint"].items():
        successes = round(row["eventual_success_rate"] * row["n"])
        continuation_intervals[checkpoint] = {
            **row,
            "successes": successes,
            "wilson_ci95": wilson(successes, row["n"]),
        }

    diagnostics = {
        "schema_version": 1,
        "snapshot_count": len(curves),
        "spread_range_over_checkpoints": {
            "all": describe(spread_ranges.values()),
            "by_stratum": {name: describe(values) for name, values in sorted(by_stratum.items())},
        },
        "crossfit_stability": {
            "calibration_vs_heldout_spread_pooled_correlation": corr(calibration_spread, heldout_spread),
            "calibration_vs_heldout_oracle_exact_agreement": exact_oracle_agreement,
            "calibration_vs_heldout_oracle_exact_agreement_rate": exact_oracle_agreement / len(curves),
            "oracle_checkpoint_absolute_distance": describe(oracle_distances),
            "heldout_regret_from_calibration_oracle": describe(heldout_regrets),
        },
        "mechanism_correlations": {
            "pooled_z_disagreement_vs_heldout_progress_spread": corr(pooled_z, pooled_spread),
            "pooled_action_support_diversity_vs_heldout_progress_spread": corr(pooled_support, pooled_spread),
            "pooled_z_disagreement_vs_action_support_diversity": corr(pooled_z, pooled_support),
        },
        "checkpoint_profiles": checkpoint_profiles,
        "fixed_nfe_strategy_summary": strategy_summary,
        "eventual_continuation_descriptive": {
            "selection_boundary": continuation["selection_boundary"],
            "snapshot_count": continuation["snapshot_count"],
            "case_count": continuation["case_count"],
            "by_checkpoint_with_wilson_ci95": continuation_intervals,
            "scope_warning": "Small descendant-blind representative subset; descriptive only and not the preregistered fixed-NFE gate.",
        },
        "scientific_boundary": {
            "decision": "R142_FP11_CORE_HYPOTHESIS_WEAKENED",
            "reason": "No natural snapshot met the preregistered location-sensitivity threshold, no hard snapshot had a recoverability cliff, and oracle-local branching failed to beat early/uniform/random by the required fixed-NFE margin.",
            "not_tested": "Other learned policy families, other tasks, larger natural-state samples, and VLA-scale behavior are not inferred.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(diagnostics, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "decision": diagnostics["scientific_boundary"]["decision"]}))


if __name__ == "__main__":
    main()
