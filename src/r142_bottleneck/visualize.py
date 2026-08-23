from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .benchmark import BenchmarkConfig, ForkPush2D
from .methods import generate_policy_set


def render_example(output_dir: str | Path, episode_seed: int = 14211) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    env = ForkPush2D(BenchmarkConfig())
    spec = env.make_episode(0, episode_seed)
    b0 = generate_policy_set(env, spec, "B0_best_of_n")
    proposed = generate_policy_set(env, spec, "proposed_bottleneck_local")
    paths: list[Path] = []

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    for axis, candidate_set, title in zip(
        axes, (b0, proposed), ("B0 best-of-N", "Bottleneck-local split")
    ):
        for candidate in candidate_set.candidates:
            color = "#2a9d8f" if candidate.final_mode == "upper" else "#457b9d" if candidate.final_mode == "lower" else "#d62828"
            axis.plot(candidate.states[:, 0], candidate.states[:, 1], color=color, alpha=0.30, linewidth=0.9)
        axis.fill_between(
            [spec.obstacle_x_min, spec.obstacle_x_max],
            [-spec.gate_clearance] * 2,
            [spec.gate_clearance] * 2,
            color="black",
            alpha=0.2,
        )
        axis.scatter([0, 1.1], [0, 0], c=["black", "gold"], s=45)
        axis.set_title(title)
        axis.set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.suptitle(f"ForkPush2D candidate trajectories (true t*={spec.true_bottleneck_step})")
    fig.tight_layout()
    path = output / "candidate_trajectories.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    diagnostics = proposed.detector_diagnostics
    steps = np.arange(env.config.horizon)
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.plot(steps, diagnostics["disagreement"], marker="o", label="disagreement D(t)")
    axis.axvline(spec.true_bottleneck_step, color="black", linestyle="--", label="true t*")
    if proposed.predicted_bottleneck_step is not None:
        axis.axvline(proposed.predicted_bottleneck_step, color="#e76f51", linestyle=":", label="predicted t")
    axis.set_xlabel("action step")
    axis.set_ylabel("median pairwise disagreement")
    axis.legend()
    axis.set_title("Earliest meaningful disagreement spike")
    fig.tight_layout()
    path = output / "bottleneck_detection.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    fig, axis = plt.subplots(figsize=(8, 4))
    positions = {candidate.candidate_id: (0, idx) for idx, candidate in enumerate(proposed.candidates[: env.config.scout_count])}
    children = proposed.candidates[env.config.scout_count :]
    for idx, candidate in enumerate(children):
        positions[candidate.candidate_id] = (1, idx)
        parent = positions[candidate.parent_id]
        child = positions[candidate.candidate_id]
        axis.plot([parent[0], child[0]], [parent[1], child[1]], color="0.75", linewidth=0.6)
    for candidate in proposed.candidates:
        x, y = positions[candidate.candidate_id]
        color = "#2a9d8f" if candidate.final_success else "#d62828"
        axis.scatter(x, y, color=color, s=18, zorder=2)
    axis.set_xticks([0, 1], ["scout parents", f"children split at t={proposed.predicted_bottleneck_step}"])
    axis.set_yticks([])
    axis.set_title("Candidate genealogy (green=success, red=failure)")
    fig.tight_layout()
    path = output / "candidate_genealogy.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)
    return paths


def render_quantitative_table(csv_path: str | Path, output_path: str | Path) -> Path:
    with Path(csv_path).open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    policies = [row["policy"] for row in rows]
    success = [float(row["success_at_n"]) for row in rows]
    mode = [float(row["mode_discovery_rate"]) for row in rows]
    x = np.arange(len(rows))
    fig, axis = plt.subplots(figsize=(12, 5))
    width = 0.38
    axis.bar(x - width / 2, success, width, label="success@N")
    axis.bar(x + width / 2, mode, width, label="mode discovery rate")
    axis.set_xticks(x, policies, rotation=45, ha="right")
    axis.set_ylim(0, 1.05)
    axis.legend()
    axis.set_title("Fixed-budget methods and ablations")
    fig.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)
    return destination
