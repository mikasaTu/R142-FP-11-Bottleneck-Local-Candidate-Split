#!/usr/bin/env python3
"""Render reproducible Stage-2A figures and compact quantitative tables."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"easy": "#009E73", "ambiguous": "#E69F00", "hard": "#D55E00"}
STRATEGIES = (
    "always_early",
    "random",
    "uniform_three_quantiles",
    "oracle_local_crossfit",
)
STRATEGY_LABELS = ("Always early", "Random", "Uniform-3", "Oracle local\n(cross-fit)")


def load_json(path: Path):
    return json.loads(path.read_text())


def save(fig, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.png", dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(output / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "figure.dpi": 120,
        }
    )


def render_branchability(curves: list[dict], output: Path) -> None:
    checkpoints = [row["checkpoint_index"] for row in curves[0]["curve"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    all_spread = []
    all_utility = []
    for item in curves:
        spread = np.asarray([row["heldout"]["progress_spread"] for row in item["curve"]])
        utility = np.asarray([row["heldout_best_gain_per_nfe"] for row in item["curve"]])
        all_spread.append(spread)
        all_utility.append(utility)
        color = COLORS[item["stratum"]]
        axes[0].plot(checkpoints, spread, color=color, alpha=0.24, linewidth=1)
        axes[1].plot(checkpoints, utility, color=color, alpha=0.24, linewidth=1)
    axes[0].plot(checkpoints, np.median(all_spread, axis=0), color="black", linewidth=2, label="median")
    axes[1].plot(checkpoints, np.median(all_utility, axis=0), color="black", linewidth=2, label="median")
    axes[0].set(title="Held-out branchability", xlabel="DDPM checkpoint index", ylabel="Descendant progress spread")
    axes[1].set(title="Cost-aware branch utility", xlabel="DDPM checkpoint index", ylabel="Best progress gain / remaining NFE")
    for ax in axes:
        ax.legend(frameon=False)
        ax.set_xticks([0, 20, 40, 60, 80, 99])
    save(fig, output, "branchability_and_cost_aware_curves")


def render_fixed_nfe(fixed: list[dict], output: Path) -> None:
    values = [[row["strategies"][name]["gain_over_no_branch"] for row in fixed] for name in STRATEGIES]
    wall = [[row["strategies"][name]["wall_clock_seconds"] for row in fixed] for name in STRATEGIES]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    boxes = axes[0].boxplot(values, tick_labels=STRATEGY_LABELS, patch_artist=True, showmeans=True)
    for patch, color in zip(boxes["boxes"], ["#56B4E9", "#CC79A7", "#E69F00", "#009E73"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set(title="Fixed 7200 sample-NFE outcome", ylabel="Gain over no-branch mean root progress")
    positions = np.arange(len(STRATEGIES))
    means = np.asarray([np.mean(v) for v in wall])
    stds = np.asarray([np.std(v, ddof=1) for v in wall])
    axes[1].bar(positions, means, color=["#56B4E9", "#CC79A7", "#E69F00", "#009E73"], alpha=0.75)
    axes[1].errorbar(positions, means, yerr=stds, fmt="none", ecolor="black", capsize=3)
    axes[1].set_xticks(positions, STRATEGY_LABELS)
    axes[1].set(title="Observed execution cost (mean ± SD)", ylabel="Wall-clock seconds / snapshot")
    save(fig, output, "fixed_nfe_comparison")


def render_disagreement(curves: list[dict], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.8), constrained_layout=True)
    xs, ys = [], []
    for item in curves:
        z = [row["z_disagreement"] for row in item["root_disagreement"]]
        spread = [row["heldout"]["progress_spread"] for row in item["curve"]]
        xs.extend(z)
        ys.extend(spread)
        ax.scatter(z, spread, s=18, alpha=0.45, color=COLORS[item["stratum"]], edgecolors="none")
    corr = float(np.corrcoef(xs, ys)[0, 1]) if np.std(xs) and np.std(ys) else 0.0
    ax.set(title=f"Raw disagreement vs held-out branchability (r={corr:.3f})", xlabel="Root latent pairwise RMS", ylabel="Descendant progress spread")
    save(fig, output, "disagreement_vs_branchability")


def load_genealogy(pilot: Path, snapshot_id: str) -> list[dict]:
    rows = []
    paths = sorted(pilot.glob("descendant_genealogy.rank*.jsonl"))
    paths.extend(sorted(pilot.glob("descendant_genealogy.rank*.jsonl.gz")))
    for path in paths:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("snapshot_id") == snapshot_id and row.get("stream") == "heldout":
                    rows.append(row)
    return rows


def render_genealogy(curves: list[dict], pilot: Path, output: Path) -> None:
    chosen = max(curves, key=lambda item: np.ptp([row["heldout"]["progress_spread"] for row in item["curve"]]))
    rows = load_genealogy(pilot, chosen["snapshot_id"])
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    for root in range(8):
        selected = [row for row in rows if row["root_index"] == root]
        ax.scatter(
            [row["checkpoint_index"] for row in selected],
            [row["final_progress"] for row in selected],
            s=10,
            alpha=0.42,
            color=cmap(root),
            label=f"root {root}",
            rasterized=True,
        )
    ax.set(title=f"Actual held-out descendant genealogy\n{chosen['snapshot_id'][:12]}… ({chosen['stratum']})", xlabel="Branch checkpoint index", ylabel="Immediate descendant progress")
    ax.set_xticks([0, 20, 40, 60, 80, 99])
    ax.legend(ncols=4, frameon=False)
    save(fig, output, "candidate_genealogy_example")


def render_cases(curves: list[dict], output: Path) -> None:
    ranked = sorted(
        curves,
        key=lambda item: np.ptp([row["heldout"]["progress_spread"] for row in item["curve"]]),
        reverse=True,
    )
    chosen = ranked[:3] + ranked[-3:]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True, sharex=True, sharey=True)
    for index, (ax, item) in enumerate(zip(axes.ravel(), chosen)):
        x = [row["checkpoint_index"] for row in item["curve"]]
        y = [row["heldout"]["progress_spread"] for row in item["curve"]]
        ax.plot(x, y, marker="o", markersize=3, color=COLORS[item["stratum"]])
        ax.set_title(("Largest variation" if index < 3 else "Flattest") + f" | {item['stratum']}\n{item['snapshot_id'][:10]}…")
        ax.set_xlabel("Checkpoint")
        ax.set_ylabel("Progress spread")
    save(fig, output, "positive_and_negative_cases")


def write_tables(curves: list[dict], fixed: list[dict], output: Path) -> None:
    with (output / "branchability_snapshot_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["snapshot_id", "stratum", "spread_range", "heldout_best_checkpoint", "heldout_best_gain", "wall_clock_seconds", "peak_vram_bytes"],
            lineterminator="\n",
        )
        writer.writeheader()
        for item in curves:
            spread = [row["heldout"]["progress_spread"] for row in item["curve"]]
            best = max(item["curve"], key=lambda row: row["heldout"]["best_descendant_gain"])
            writer.writerow({"snapshot_id": item["snapshot_id"], "stratum": item["stratum"], "spread_range": float(np.ptp(spread)), "heldout_best_checkpoint": best["checkpoint_index"], "heldout_best_gain": best["heldout"]["best_descendant_gain"], "wall_clock_seconds": item["wall_clock_seconds"], "peak_vram_bytes": item["peak_vram_bytes"]})
    with (output / "fixed_nfe_per_snapshot.csv").open("w", newline="") as handle:
        fields = ["snapshot_id", "stratum", "strategy", "actual_sample_nfe", "budget_slack", "generated_suffixes", "gain_over_no_branch", "score_mean_best_progress", "wall_clock_seconds"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for item in fixed:
            for name in STRATEGIES:
                row = item["strategies"][name]
                writer.writerow({"snapshot_id": item["snapshot_id"], "stratum": item["stratum"], "strategy": name, **{field: row[field] for field in fields[3:]}})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    curves = [load_json(path) for path in sorted((args.pilot_output / "branchability_curves").glob("*.json"))]
    fixed = [load_json(path) for path in sorted((args.pilot_output / "fixed_nfe").glob("*.json"))]
    if len(curves) != 24 or len(fixed) != 24:
        raise RuntimeError(f"formal result incomplete: curves={len(curves)}, fixed={len(fixed)}")
    configure_style()
    render_branchability(curves, args.output)
    render_fixed_nfe(fixed, args.output)
    render_disagreement(curves, args.output)
    render_genealogy(curves, args.pilot_output, args.output)
    render_cases(curves, args.output)
    write_tables(curves, fixed, args.output)


if __name__ == "__main__":
    main()
