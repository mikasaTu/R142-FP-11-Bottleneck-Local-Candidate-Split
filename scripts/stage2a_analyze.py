#!/usr/bin/env python3
"""Aggregate Stage-2A curves, negative controls, fixed-NFE tests and decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


BASELINES = ("always_early", "uniform_three_quantiles", "random")


def write_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def paired_bootstrap(values, seed=142150000, repetitions=10000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(repetitions, len(values)))
    means = values[draws].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "n": int(len(values)),
    }


def cliff_indices(curve):
    values = np.asarray([row["heldout"]["progress_spread"] for row in curve])
    result = []
    for i in range(len(values) - 2):
        drop = values[i] - values[i + 1]
        if drop >= 0.10 and values[i + 1] >= values[i + 2] - 0.02:
            result.append(i)
    return result


def classify_negative(curve, disagreement):
    values = np.asarray([row["heldout"]["progress_spread"] for row in curve])
    z = np.asarray([row["z_disagreement"] for row in disagreement])
    labels = []
    if np.ptp(values) < 0.10:
        labels.append("no-bottleneck")
    if len(values) > 2 and np.all(np.diff(values) <= 0.025) and values[0] > values[-1]:
        labels.append("smooth-decay")
    if np.corrcoef(z, values)[0, 1] < 0.1 if np.std(z) and np.std(values) else True:
        labels.append("disagreement-outcome-decoupled")
    if z[np.argmax(values)] <= np.quantile(z, 0.25) and np.ptp(values) >= 0.10:
        labels.append("silent-bottleneck")
    if z[np.argmin(values)] >= np.quantile(z, 0.75) and np.ptp(values) < 0.10:
        labels.append("fake-disagreement")
    if len(cliff_indices(curve)) > 1:
        labels.append("multiple-cliff")
    return labels or ["no-preregistered-negative-control-label"]


def markdown_report(summary):
    lines = [
        "# R142-FP-11 Stage-2A Pilot Report",
        "",
        f"Scientific decision: `{summary['scientific_decision']}`",
        "",
        "This report evaluates the official learned LeRobot Diffusion Policy on unmodified standard PushT. It does not use ForkPush2D, a VLA, a learned detector, or oracle labels at deployment time.",
        "",
        "## Evidence boundary",
        "",
        f"- Baseline rollouts: {summary['baseline']['episodes']} fixed seeds; success rate {summary['baseline']['success_rate']:.3f}; mean max progress {summary['baseline']['mean_max_progress']:.3f}.",
        f"- Natural snapshots: {summary['snapshot_count']} selected before descendant outcomes.",
        f"- A-E real-source equivalence gates: {'PASS' if summary['all_source_gates_pass'] else 'FAIL'}.",
        f"- Discovery tree: K=8 roots, 16 real DDPM checkpoints, M=8 calibration and M=8 held-out suffixes per root/checkpoint.",
        f"- Fixed sample-NFE budget: {summary['fixed_nfe_budget']} per snapshot for every branching strategy, with slack reported.",
        "",
        "## Q1-Q6 quantitative answer",
        "",
        f"1. Location-sensitive branchability prevalence over all snapshots: {summary['location_sensitive_fraction']:.3f} ({summary['location_sensitive_count']}/{summary['snapshot_count']}).",
        f"2. Natural recoverability-cliff prevalence in hard/failing snapshots: {summary['hard_cliff_fraction']:.3f} ({summary['hard_cliff_count']}/{summary['hard_snapshot_count']}).",
        f"3. Oracle checkpoint indices: {summary['oracle_checkpoint_histogram']}.",
        f"4. Always-early equivalent-to-oracle flag: {summary['always_early_equivalent_to_oracle']}.",
        f"5. Raw disagreement/branchability correlation (snapshot mean): {summary['mean_disagreement_branchability_correlation']:.3f}.",
        "6. Raw, benefit/NFE and fixed-NFE results are all retained; the decision uses only held-out fixed-NFE results.",
        "",
        "## Fixed-NFE held-out comparisons",
        "",
        "| Comparison | Mean oracle gain | Paired 95% CI | Required |",
        "|---|---:|---:|---:|",
    ]
    for name in BASELINES:
        value = summary["fixed_nfe_comparisons"][name]
        lines.append(
            f"| oracle-local - {name} | {value['mean']:.4f} | [{value['ci95'][0]:.4f}, {value['ci95'][1]:.4f}] | >=0.10 and lower>0 |"
        )
    lines += [
        "",
        "## Negative controls",
        "",
    ]
    for label, ids in sorted(summary["negative_controls"].items()):
        lines.append(f"- `{label}`: {len(ids)} snapshots")
    lines += [
        "",
        "## Mechanism reverse explanation (no new idea)",
        "",
        "### observed_fact",
        "",
        summary["mechanism"]["observed_fact"],
        "",
        "### controlled_intervention",
        "",
        summary["mechanism"]["controlled_intervention"],
        "",
        "### interpretation",
        "",
        summary["mechanism"]["interpretation"],
        "",
        "### untested_hypothesis",
        "",
        summary["mechanism"]["untested_hypothesis"],
        "",
        "## Failure cases",
        "",
        "All negative-control snapshot IDs and complete raw genealogies are retained. Flat curves, early-is-best curves, disagreement/outcome decoupling and any multiple-cliff cases are included rather than filtered out.",
        "",
        "Pilot completion does not authorize Stage-2B or VLA expansion.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-output", type=Path, required=True)
    parser.add_argument("--pilot-output", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads((args.baseline_output / "baseline_summary.json").read_text())
    gates = json.loads((args.baseline_output / "resume_equivalence_tests.json").read_text())
    curves = [json.loads(path.read_text()) for path in sorted((args.pilot_output / "branchability_curves").glob("*.json"))]
    fixed = [json.loads(path.read_text()) for path in sorted((args.pilot_output / "fixed_nfe").glob("*.json"))]
    if len(curves) < 15 or len(fixed) != len(curves):
        raise RuntimeError(f"pilot incomplete: curves={len(curves)} fixed={len(fixed)}")
    fixed_by_id = {row["snapshot_id"]: row for row in fixed}
    negative = {}
    correlations = []
    location_sensitive = []
    cliffs = []
    hard = []
    oracle_hist = {}
    for row in curves:
        snapshot_id = row["snapshot_id"]
        values = np.asarray([x["heldout"]["progress_spread"] for x in row["curve"]])
        z = np.asarray([x["z_disagreement"] for x in row["root_disagreement"]])
        correlations.append(float(np.corrcoef(z, values)[0, 1]) if np.std(z) and np.std(values) else 0.0)
        if np.ptp(values) >= 0.10:
            location_sensitive.append(snapshot_id)
        this_cliffs = cliff_indices(row["curve"])
        if this_cliffs:
            cliffs.append(snapshot_id)
        if row["stratum"] == "hard":
            hard.append(snapshot_id)
        for label in classify_negative(row["curve"], row["root_disagreement"]):
            negative.setdefault(label, []).append(snapshot_id)
        oracle = max(
            row["curve"],
            key=lambda x: (x["calibration"]["best_descendant_gain"], x["calibration"]["progress_spread"]),
        )["checkpoint_index"]
        oracle_hist[str(oracle)] = oracle_hist.get(str(oracle), 0) + 1
    comparisons = {}
    for baseline_name in BASELINES:
        diffs = []
        for snapshot_id, row in fixed_by_id.items():
            strategies = row["strategies"]
            diffs.append(
                strategies["oracle_local_crossfit"]["score_mean_best_progress"]
                - strategies[baseline_name]["score_mean_best_progress"]
            )
        comparisons[baseline_name] = paired_bootstrap(diffs)
    hard_cliffs = sorted(set(hard) & set(cliffs))
    hard_fraction = len(hard_cliffs) / len(hard) if hard else 0.0
    location_fraction = len(location_sensitive) / len(curves)
    fixed_gate = all(
        value["mean"] >= 0.10 and value["ci95"][0] > 0 for value in comparisons.values()
    )
    always_early_equivalent = not (
        comparisons["always_early"]["mean"] >= 0.10
        and comparisons["always_early"]["ci95"][0] > 0
    )
    support = (
        gates["all_pass"]
        and len(hard_cliffs) >= 6
        and hard_fraction >= 0.30
        and fixed_gate
        and not always_early_equivalent
        and bool(cliffs)
    )
    decision = "STAGE2A_SUPPORTS_STAGE2B_ONLY" if support else "R142_FP11_CORE_HYPOTHESIS_WEAKENED"
    if support:
        interpretation = "Location, not merely the suffix operator, changed held-out progress under equal sample-NFE; cross-fitting prevents using the same descendants to choose and score the location."
    else:
        interpretation = "The proposed local mechanism is weakened because natural location effects did not jointly satisfy prevalence, cliff, always-early and fixed-NFE superiority gates. Any isolated gain is insufficient evidence for a deployable bottleneck-local split."
    summary = {
        "schema_version": 1,
        "scientific_decision": decision,
        "baseline": baseline,
        "all_source_gates_pass": gates["all_pass"],
        "snapshot_count": len(curves),
        "location_sensitive_count": len(location_sensitive),
        "location_sensitive_fraction": location_fraction,
        "hard_snapshot_count": len(hard),
        "hard_cliff_count": len(hard_cliffs),
        "hard_cliff_fraction": hard_fraction,
        "cliff_snapshot_ids": cliffs,
        "oracle_checkpoint_histogram": oracle_hist,
        "always_early_equivalent_to_oracle": always_early_equivalent,
        "mean_disagreement_branchability_correlation": float(np.mean(correlations)),
        "fixed_nfe_budget": fixed[0]["sample_nfe_budget"],
        "fixed_nfe_comparisons": comparisons,
        "negative_controls": negative,
        "mechanism": {
            "observed_fact": f"Across {len(curves)} preregistered natural snapshots, {len(location_sensitive)} had a held-out progress-spread range of at least 0.10; {len(hard_cliffs)} hard snapshots met the non-oracle cliff screen.",
            "controlled_intervention": "Every comparison copied the same real z_s, varied only the DDPM suffix RNG, and replayed the same simulator seed/action prefix. Calibration selected oracle-local locations and disjoint held-out suffixes scored them.",
            "interpretation": interpretation,
            "untested_hypothesis": "Whether another learned policy family, task, or substantially larger natural-state sample has a different branchability structure remains untested and is not inferred here.",
        },
    }
    write_json(args.pilot_output / "stage2a_summary.json", summary)
    write_json(args.pilot_output / "negative_controls.json", negative)
    write_json(args.pilot_output / "cost_aware_utility.json", {"fixed_nfe_comparisons": comparisons, "budget": fixed[0]["sample_nfe_budget"]})
    (args.pilot_output / "STAGE2A_PILOT_REPORT.md").write_text(markdown_report(summary))
    print(json.dumps({"scientific_decision": decision, "snapshot_count": len(curves)}, sort_keys=True))


if __name__ == "__main__":
    main()
