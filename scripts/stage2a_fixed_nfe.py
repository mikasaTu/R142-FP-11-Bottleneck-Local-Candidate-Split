#!/usr/bin/env python3
"""Run genuinely fixed-sample-NFE branching upper-bound comparisons."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from r142_stage2a.analysis import actual_nfe, fixed_nfe_suffix_count
from r142_stage2a.tracing import repeat_condition, repeat_roots, resume_suffix

from stage2a_pilot import (
    CHECKPOINTS,
    K,
    generate_roots,
    run_chunks_same_snapshot,
    seed_for,
    unnormalize_chunks,
)
from stage2a_validate import conditioning_from_history, load_policy


BUDGET = actual_nfe(root_count=K, total_steps=100, branch_index=0, suffix_count=8)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def allocations(checkpoint: int, budget: int = BUDGET):
    count, slack = fixed_nfe_suffix_count(budget, K, 100, checkpoint)
    return {"checkpoint_index": checkpoint, "suffixes_per_root": count, "budget_slack": slack}


def strategy_allocations(curve: list[dict], snapshot_id: str):
    oracle = max(
        curve,
        key=lambda row: (
            row["calibration"]["best_descendant_gain"],
            row["calibration"]["progress_spread"],
            -row["checkpoint_index"],
        ),
    )["checkpoint_index"]
    rng = random.Random(142140000 + int(snapshot_id[:8], 16))
    random_checkpoint = rng.choice(CHECKPOINTS)
    uniform_checkpoints = [20, 53, 86]
    uniform_budget = K * 100 + (BUDGET - K * 100) // 3
    return {
        "always_early": [allocations(0)],
        "random": [allocations(random_checkpoint)],
        "oracle_local_crossfit": [allocations(oracle)],
        "uniform_three_quantiles": [
            {
                **allocations(checkpoint, uniform_budget),
                "sub_budget": uniform_budget,
            }
            for checkpoint in uniform_checkpoints
        ],
    }


@torch.no_grad()
def evaluate_allocation(
    policy,
    global_cond,
    root_states,
    snapshot_index,
    episode_seed,
    prefix,
    allocation,
    strategy_index,
    device,
):
    checkpoint = allocation["checkpoint_index"]
    count = allocation["suffixes_per_root"]
    progress_per_root = [[] for _ in range(K)]
    generated = 0
    generator_device = torch.device(device).type
    model_batch_suffixes = max(1, min(count, 16))
    for suffix_offset in range(0, count, model_batch_suffixes):
        local_m = min(model_batch_suffixes, count - suffix_offset)
        z_batch = repeat_roots(root_states[checkpoint], local_m)
        cond_batch = repeat_condition(global_cond.repeat(K, 1), local_m)
        generators = []
        for root in range(K):
            for suffix in range(suffix_offset, suffix_offset + local_m):
                seed = seed_for(
                    142160000 + strategy_index * 100_000_000,
                    snapshot_index,
                    root,
                    checkpoint,
                    suffix,
                )
                generators.append(torch.Generator(device=generator_device).manual_seed(seed))
        final, _ = resume_suffix(
            policy.diffusion,
            z_batch,
            checkpoint,
            cond_batch,
            generators,
            capture=False,
        )
        _, chunks = unnormalize_chunks(policy, final)
        outcomes = run_chunks_same_snapshot(episode_seed, prefix, chunks)
        for candidate, outcome in enumerate(outcomes):
            root = candidate // local_m
            progress_per_root[root].append(outcome["final_progress"])
        generated += K * local_m
    return progress_per_root, generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pilot-output", type=Path, required=True)
    parser.add_argument("--baseline-output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    args = parser.parse_args()
    frame = [
        json.loads(line)
        for line in (args.baseline_output / "snapshot_sampling_frame.jsonl").read_text().splitlines()
        if line.strip()
    ]
    selected = sorted((row for row in frame if row["selected"]), key=lambda row: row["snapshot_id"])
    policy = load_policy(args.checkpoint, args.device)
    completed = []
    for snapshot_index, row in enumerate(selected):
        if snapshot_index % args.world_size != args.rank:
            continue
        destination = args.pilot_output / "fixed_nfe" / f"{row['snapshot_id']}.json"
        if destination.exists():
            completed.append(row["snapshot_id"])
            continue
        data = np.load(args.baseline_output / row["snapshot_file"])
        previous = {"pixels": data["previous_pixels"], "agent_pos": data["previous_agent_pos"]}
        current = {"pixels": data["current_pixels"], "agent_pos": data["current_agent_pos"]}
        prefix = np.asarray(data["actions"], dtype=np.float32)
        global_cond = conditioning_from_history(policy, [previous, current], args.device)
        root_final, root_states = generate_roots(policy.diffusion, global_cond, snapshot_index, args.device)
        _, root_chunks = unnormalize_chunks(policy, root_final)
        root_outcomes = run_chunks_same_snapshot(row["episode_seed"], prefix, root_chunks)
        curve = json.loads(
            (args.pilot_output / "branchability_curves" / f"{row['snapshot_id']}.json").read_text()
        )["curve"]
        strategies = strategy_allocations(curve, row["snapshot_id"])
        result = {
            "snapshot_id": row["snapshot_id"],
            "stratum": row["stratum"],
            "sample_nfe_budget": BUDGET,
            "no_branch": {
                "sample_nfe": K * 100,
                "score_mean_root_progress": float(
                    np.mean([outcome["final_progress"] for outcome in root_outcomes])
                ),
            },
            "strategies": {},
        }
        for strategy_index, (name, strategy) in enumerate(strategies.items()):
            started = time.perf_counter()
            all_progress = [[] for _ in range(K)]
            generated = 0
            used_nfe = K * 100
            slack = BUDGET - K * 100
            for allocation in strategy:
                values, count = evaluate_allocation(
                    policy,
                    global_cond,
                    root_states,
                    snapshot_index,
                    row["episode_seed"],
                    prefix,
                    allocation,
                    strategy_index,
                    args.device,
                )
                for root in range(K):
                    all_progress[root].extend(values[root])
                generated += count
                allocation_budget = allocation.get("sub_budget", BUDGET)
                used = actual_nfe(K, 100, allocation["checkpoint_index"], allocation["suffixes_per_root"])
                used_nfe += used - K * 100
                slack -= used - K * 100
                if used > allocation_budget:
                    raise RuntimeError("fixed-NFE allocation exceeded its preregistered sub-budget")
            scores = [max(values) if values else float("nan") for values in all_progress]
            result["strategies"][name] = {
                "allocations": strategy,
                "generated_suffixes": generated,
                "actual_sample_nfe": used_nfe,
                "budget_slack": slack,
                "score_best_progress_per_root": scores,
                "score_mean_best_progress": float(np.mean(scores)),
                "gain_over_no_branch": float(
                    np.mean(scores) - result["no_branch"]["score_mean_root_progress"]
                ),
                "wall_clock_seconds": time.perf_counter() - started,
            }
            if used_nfe > BUDGET:
                raise RuntimeError(f"{name} exceeded fixed sample-NFE budget")
        write_json(destination, result)
        completed.append(row["snapshot_id"])
        write_json(
            args.pilot_output / f"FIXED_NFE_FIRST_WORK.rank{args.rank}.json",
            {"rank": args.rank, "completed_snapshot_ids": completed},
        )
        print(json.dumps({"rank": args.rank, "fixed_nfe_snapshot": row["snapshot_id"]}), flush=True)
    write_json(
        args.pilot_output / f"FIXED_NFE_COMPLETE.rank{args.rank}.json",
        {"rank": args.rank, "world_size": args.world_size, "completed_snapshot_ids": completed},
    )


if __name__ == "__main__":
    main()
