#!/usr/bin/env python3
"""Run a descendant-blind representative eventual-success continuation subset."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from r142_stage2a.snapshot import native_push_t_state
from stage2a_pilot import unnormalize_chunks
from stage2a_validate import conditioning_from_history, load_policy, make_env


SUBSET_PER_STRATUM = 2
ROOT_INDICES = (0, 7)
CHECKPOINT_INDICES = (0, 53, 99)
SUFFIX_INDICES = (0,)
STREAM = "heldout"


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def load_selected_rows(baseline_output: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in (baseline_output / "snapshot_sampling_frame.jsonl").read_text().splitlines()
        if line.strip()
    ]
    return sorted((row for row in rows if row["selected"]), key=lambda row: row["snapshot_id"])


def representative_snapshots(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["stratum"]].append(row)
    expected = {"easy", "ambiguous", "hard"}
    if set(grouped) != expected:
        raise RuntimeError(f"unexpected selected strata: {sorted(grouped)}")
    result = []
    for stratum in sorted(expected):
        candidates = sorted(grouped[stratum], key=lambda row: row["snapshot_id"])
        if len(candidates) < SUBSET_PER_STRATUM:
            raise RuntimeError(f"insufficient {stratum} snapshots")
        result.extend(candidates[:SUBSET_PER_STRATUM])
    return sorted(result, key=lambda row: row["snapshot_id"])


def load_action_index(pilot_output: Path, snapshot_ids: set[str]) -> dict[tuple, dict]:
    wanted = {
        (snapshot_id, root, checkpoint, suffix)
        for snapshot_id in snapshot_ids
        for root in ROOT_INDICES
        for checkpoint in CHECKPOINT_INDICES
        for suffix in SUFFIX_INDICES
    }
    index = {}
    paths = sorted(pilot_output.glob("descendant_genealogy.rank*.jsonl"))
    paths.extend(sorted(pilot_output.glob("descendant_genealogy.rank*.jsonl.gz")))
    for path in paths:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("record_type") != "descendant" or row.get("stream") != STREAM:
                    continue
                key = (
                    row["snapshot_id"],
                    row["root_index"],
                    row["checkpoint_index"],
                    row["suffix_index"],
                )
                if key in wanted:
                    index[key] = {**row, "genealogy_file": path.name}
    missing = sorted(wanted - set(index))
    if missing:
        raise RuntimeError(f"missing representative descendant records: {missing[:5]}")
    return index


def build_manifest(baseline_output: Path, pilot_output: Path) -> dict:
    selected = representative_snapshots(load_selected_rows(baseline_output))
    by_id = {row["snapshot_id"]: row for row in selected}
    action_index = load_action_index(pilot_output, set(by_id))
    cases = []
    for key in sorted(action_index):
        snapshot_id, root, checkpoint, suffix = key
        source = action_index[key]
        candidate_id = source["candidate_id"]
        case_id = hashlib.sha256(candidate_id.encode()).hexdigest()
        cases.append(
            {
                "case_id": case_id,
                "candidate_id": candidate_id,
                "snapshot_id": snapshot_id,
                "stratum": by_id[snapshot_id]["stratum"],
                "episode_seed": by_id[snapshot_id]["episode_seed"],
                "control_step": by_id[snapshot_id]["control_step"],
                "snapshot_file": by_id[snapshot_id]["snapshot_file"],
                "root_index": root,
                "checkpoint_index": checkpoint,
                "suffix_index": suffix,
                "stream": STREAM,
                "actions_unnormalized": source["actions_unnormalized"],
                "immediate_progress": source["final_progress"],
                "immediate_success": source["success"],
                "genealogy_file": source["genealogy_file"],
            }
        )
    return {
        "schema_version": 1,
        "selection_boundary": "descendant outcomes are not used: take the two lexicographically first frozen snapshot IDs per stratum and fixed root/checkpoint/suffix indices",
        "snapshot_rule": "two lexicographically first selected IDs in each of easy, ambiguous, hard",
        "candidate_rule": {
            "stream": STREAM,
            "root_indices": list(ROOT_INDICES),
            "checkpoint_indices": list(CHECKPOINT_INDICES),
            "suffix_indices": list(SUFFIX_INDICES),
        },
        "snapshot_count": len(selected),
        "case_count": len(cases),
        "cases": cases,
    }


@torch.no_grad()
def sample_chunk(policy, observations: list[dict], device: str, seed: int) -> np.ndarray:
    global_cond = conditioning_from_history(policy, observations[-2:], device)
    model = policy.diffusion
    generator = torch.Generator(device=torch.device(device).type).manual_seed(seed)
    shape = (1, model.config.horizon, model.config.action_feature.shape[0])
    sample = torch.randn(shape, dtype=global_cond.dtype, device=device, generator=generator)
    model.noise_scheduler.set_timesteps(model.num_inference_steps)
    for timestep in model.noise_scheduler.timesteps:
        prediction = model.unet(
            sample,
            torch.full(sample.shape[:1], timestep, dtype=torch.long, device=sample.device),
            global_cond=global_cond,
        )
        sample = model.noise_scheduler.step(
            prediction, timestep, sample, generator=generator
        ).prev_sample
    _, chunks = unnormalize_chunks(policy, sample)
    return chunks[0]


def run_case(policy, baseline_output: Path, case: dict, device: str) -> dict:
    data = np.load(baseline_output / case["snapshot_file"])
    prefix = np.asarray(data["actions"], dtype=np.float32)
    branch_chunk = np.asarray(case["actions_unnormalized"], dtype=np.float32)
    env = make_env()
    current, _ = env.reset(seed=case["episode_seed"])
    previous = current
    for action in prefix:
        previous = current
        current, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            raise RuntimeError("frozen snapshot prefix terminated before branch")

    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    done = False
    success = False
    branch_rewards = []
    continuation_rewards = []
    control_actions = 0
    generated_chunks = 0
    for action in branch_chunk:
        previous = current
        current, reward, terminated, truncated, info = env.step(action)
        branch_rewards.append(float(reward))
        control_actions += 1
        success = success or bool(terminated) or bool(info.get("is_success", False))
        done = bool(terminated or truncated)
        if done:
            break

    continuation_index = 0
    while not done:
        continuation_seed = (
            142170000
            + int(case["snapshot_id"][:8], 16)
            + case["root_index"] * 100_000
            + case["checkpoint_index"] * 1_000
            + continuation_index
        )
        chunk = sample_chunk(policy, [previous, current], device, continuation_seed)
        generated_chunks += 1
        continuation_index += 1
        for action in chunk:
            previous = current
            current, reward, terminated, truncated, info = env.step(action)
            continuation_rewards.append(float(reward))
            control_actions += 1
            success = success or bool(terminated) or bool(info.get("is_success", False))
            done = bool(terminated or truncated)
            if done:
                break

    final_state = native_push_t_state(env)
    env.close()
    rewards = branch_rewards + continuation_rewards
    return {
        **{key: value for key, value in case.items() if key != "actions_unnormalized"},
        "eventual_success": success,
        "eventual_max_progress": float(max(rewards, default=0.0)),
        "branch_max_progress": float(max(branch_rewards, default=0.0)),
        "branch_reward_trace": branch_rewards,
        "continuation_reward_trace": continuation_rewards,
        "continuation_generated_chunks": generated_chunks,
        "continuation_model_nfe": generated_chunks * int(policy.diffusion.num_inference_steps),
        "control_actions_after_snapshot": control_actions,
        "wall_clock_seconds": time.perf_counter() - started,
        "peak_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        "final_native_state": {
            key: value for key, value in final_state.items() if key != "np_random_state"
        },
    }


def aggregate(output: Path) -> None:
    manifest = json.loads((output / "representative_subset_manifest.json").read_text())
    rows = [json.loads(path.read_text()) for path in sorted((output / "cases").glob("*.json"))]
    if len(rows) != manifest["case_count"]:
        raise RuntimeError(f"continuation incomplete: {len(rows)} != {manifest['case_count']}")
    by_checkpoint = {}
    for checkpoint in CHECKPOINT_INDICES:
        selected = [row for row in rows if row["checkpoint_index"] == checkpoint]
        by_checkpoint[str(checkpoint)] = {
            "n": len(selected),
            "eventual_success_rate": float(np.mean([row["eventual_success"] for row in selected])),
            "mean_eventual_max_progress": float(
                np.mean([row["eventual_max_progress"] for row in selected])
            ),
        }
    by_stratum = {}
    for stratum in ("easy", "ambiguous", "hard"):
        selected = [row for row in rows if row["stratum"] == stratum]
        by_stratum[stratum] = {
            "n": len(selected),
            "eventual_success_rate": float(np.mean([row["eventual_success"] for row in selected])),
            "mean_eventual_max_progress": float(
                np.mean([row["eventual_max_progress"] for row in selected])
            ),
        }
    summary = {
        "schema_version": 1,
        "case_count": len(rows),
        "snapshot_count": manifest["snapshot_count"],
        "selection_boundary": manifest["selection_boundary"],
        "overall_eventual_success_rate": float(np.mean([row["eventual_success"] for row in rows])),
        "overall_mean_eventual_max_progress": float(
            np.mean([row["eventual_max_progress"] for row in rows])
        ),
        "by_checkpoint": by_checkpoint,
        "by_stratum": by_stratum,
        "total_continuation_model_nfe": int(sum(row["continuation_model_nfe"] for row in rows)),
        "total_wall_clock_seconds": float(sum(row["wall_clock_seconds"] for row in rows)),
        "peak_vram_bytes": int(max(row["peak_vram_bytes"] for row in rows)),
    }
    write_json(output / "eventual_continuation_summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--baseline-output", type=Path, required=True)
    parser.add_argument("--pilot-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args.baseline_output, args.pilot_output)
    manifest_path = args.output / "representative_subset_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError("representative subset manifest drifted")
    else:
        write_json(manifest_path, manifest)
    if args.manifest_only:
        return
    if args.aggregate:
        aggregate(args.output)
        return
    if args.checkpoint is None:
        parser.error("--checkpoint is required unless --aggregate is set")
    policy = load_policy(args.checkpoint, args.device)
    completed = []
    for index, case in enumerate(manifest["cases"]):
        if index % args.world_size != args.rank:
            continue
        destination = args.output / "cases" / f"{case['case_id']}.json"
        if not destination.exists():
            write_json(destination, run_case(policy, args.baseline_output, case, args.device))
        completed.append(case["case_id"])
        write_json(
            args.output / f"FIRST_WORK.rank{args.rank}.json",
            {"rank": args.rank, "completed_case_ids": completed},
        )
        print(json.dumps({"rank": args.rank, "completed_case": case["case_id"]}), flush=True)
    write_json(
        args.output / f"COMPLETE.rank{args.rank}.json",
        {"rank": args.rank, "world_size": args.world_size, "completed_case_ids": completed},
    )


if __name__ == "__main__":
    main()
