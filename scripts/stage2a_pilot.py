#!/usr/bin/env python3
"""Generate the preregistered real DDPM descendant tree on natural PushT states."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from r142_stage2a.analysis import branchability_vector, pairwise_rms
from r142_stage2a.snapshot import native_push_t_state, observation_sha256
from r142_stage2a.tracing import repeat_condition, repeat_roots, resume_suffix, tensor_sha256

from stage2a_validate import conditioning_from_history, load_policy, make_env


CHECKPOINTS = [0, 7, 13, 20, 26, 33, 40, 46, 53, 59, 66, 73, 79, 86, 92, 99]
K = 8
M = 8


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def make_vector_env(count: int):
    return gym.vector.SyncVectorEnv([make_env for _ in range(count)])


def seed_for(base: int, snapshot_index: int, root: int, checkpoint: int, suffix: int) -> int:
    return base + snapshot_index * 1_000_000 + root * 10_000 + checkpoint * 100 + suffix


@torch.no_grad()
def generate_roots(model, global_cond, snapshot_index: int, device: str):
    generator_device = torch.device(device).type
    generators = [
        torch.Generator(device=generator_device).manual_seed(
            seed_for(142110000, snapshot_index, root, 0, 0)
        )
        for root in range(K)
    ]
    shape = (model.config.horizon, model.config.action_feature.shape[0])
    sample = torch.stack(
        [torch.randn(shape, dtype=global_cond.dtype, device=device, generator=g) for g in generators]
    )
    model.noise_scheduler.set_timesteps(model.num_inference_steps)
    states = {}
    for index, timestep in enumerate(model.noise_scheduler.timesteps):
        if index in CHECKPOINTS:
            states[index] = sample.detach().clone()
        prediction = model.unet(
            sample,
            torch.full(sample.shape[:1], timestep, dtype=torch.long, device=sample.device),
            global_cond=global_cond.repeat(K, 1),
        )
        sample = model.noise_scheduler.step(
            prediction, timestep, sample, generator=generators
        ).prev_sample
    return sample, states


def unnormalize_chunks(policy, final_samples):
    start = policy.config.n_obs_steps - 1
    end = start + policy.config.n_action_steps
    normalized = final_samples[:, start:end]
    actions = policy.unnormalize_outputs({"action": normalized})["action"]
    return normalized.detach().cpu().numpy(), actions.detach().cpu().numpy()


def run_chunks_same_snapshot(seed: int, prefix: np.ndarray, chunks: np.ndarray, batch_size: int = 64):
    """Replay the exact prefix in vector batches, then execute descendant chunks."""

    all_results = []
    for offset in range(0, len(chunks), batch_size):
        part = chunks[offset : offset + batch_size]
        env = make_vector_env(len(part))
        env.reset(seed=[seed] * len(part))
        for action in prefix:
            repeated = np.repeat(action[None, :], len(part), axis=0)
            env.step(repeated)
        start_states = [native_push_t_state(single) for single in env.envs]
        max_progress = np.zeros(len(part), dtype=float)
        success = np.zeros(len(part), dtype=bool)
        done = np.zeros(len(part), dtype=bool)
        contacts = np.zeros(len(part), dtype=int)
        terminal_block_pose = [None] * len(part)
        for action_index in range(part.shape[1]):
            _, reward, terminated, truncated, info = env.step(part[:, action_index])
            active = ~done
            max_progress[active] = np.maximum(max_progress[active], reward[active])
            if "n_contacts" in info:
                contacts[active] += np.asarray(info["n_contacts"], dtype=int)[active]
            if "final_info" in info:
                for i, final_info in enumerate(info["final_info"]):
                    if final_info is not None and active[i]:
                        success[i] |= bool(final_info.get("is_success", False))
                        if "block_pose" in final_info:
                            terminal_block_pose[i] = np.asarray(final_info["block_pose"], dtype=float)
            success |= terminated & ~truncated & active
            done |= terminated | truncated
        final_states = [native_push_t_state(single) for single in env.envs]
        for i in range(len(part)):
            before = start_states[i]
            after = final_states[i]
            if terminal_block_pose[i] is None:
                final_pose = [
                    after["block_position"][0],
                    after["block_position"][1],
                    after["block_angle"],
                ]
            else:
                final_pose = terminal_block_pose[i].tolist()
            all_results.append(
                {
                    "final_progress": float(max_progress[i]),
                    "success": bool(success[i]),
                    "contacts": int(contacts[i]),
                    "block_delta": [
                        float(final_pose[0] - before["block_position"][0]),
                        float(final_pose[1] - before["block_position"][1]),
                        float(final_pose[2] - before["block_angle"]),
                    ],
                    "final_native_state": {
                        key: value for key, value in after.items() if key != "np_random_state"
                    },
                }
            )
        env.close()
    return all_results


def process_snapshot(policy, row: dict, snapshot_index: int, output: Path, device: str):
    data = np.load(output / row["snapshot_file"])
    previous = {"pixels": data["previous_pixels"], "agent_pos": data["previous_agent_pos"]}
    current = {"pixels": data["current_pixels"], "agent_pos": data["current_agent_pos"]}
    prefix = np.asarray(data["actions"], dtype=np.float32)
    global_cond = conditioning_from_history(policy, [previous, current], device)
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    root_final, root_states = generate_roots(policy.diffusion, global_cond, snapshot_index, device)
    root_norm, root_chunks = unnormalize_chunks(policy, root_final)
    control_outcomes = run_chunks_same_snapshot(row["episode_seed"], prefix, root_chunks)
    genealogy_path = output / f"descendant_genealogy.rank{snapshot_index % 2}.jsonl"
    curve = []
    disagreement = []
    for root in range(K):
        append_jsonl(
            genealogy_path,
            {
                "record_type": "root",
                "snapshot_id": row["snapshot_id"],
                "candidate_id": f"{row['snapshot_id']}:root:{root}",
                "parent_id": row["snapshot_id"],
                "root_index": root,
                "seed": seed_for(142110000, snapshot_index, root, 0, 0),
                "normalized_action_sha256": hashlib.sha256(root_norm[root].tobytes()).hexdigest(),
                "actions_normalized": root_norm[root].tolist(),
                "actions_unnormalized": root_chunks[root].tolist(),
                **control_outcomes[root],
                "sample_nfe": 100,
            },
        )
    generator_device = torch.device(device).type
    for checkpoint_index in CHECKPOINTS:
        z_roots = root_states[checkpoint_index]
        disagreement.append(
            {
                "checkpoint_index": checkpoint_index,
                "timestep": int(policy.diffusion.noise_scheduler.timesteps[checkpoint_index].item()),
                "z_disagreement": pairwise_rms(z_roots.detach().cpu().numpy()),
            }
        )
        stream_vectors = {}
        for stream, seed_base in (
            ("calibration", 142120000),
            ("heldout", 142130000),
        ):
            z_batch = repeat_roots(z_roots, M)
            cond_batch = repeat_condition(global_cond.repeat(K, 1), M)
            generators = []
            seeds = []
            for root in range(K):
                for suffix in range(M):
                    seed = seed_for(seed_base, snapshot_index, root, checkpoint_index, suffix)
                    seeds.append(seed)
                    generators.append(torch.Generator(device=generator_device).manual_seed(seed))
            suffix_started = time.perf_counter()
            final, _ = resume_suffix(
                policy.diffusion,
                z_batch,
                checkpoint_index,
                cond_batch,
                generators,
                capture=False,
            )
            norm, chunks = unnormalize_chunks(policy, final)
            outcomes = run_chunks_same_snapshot(row["episode_seed"], prefix, chunks)
            progress = []
            for candidate, outcome in enumerate(outcomes):
                root = candidate // M
                suffix = candidate % M
                progress.append(outcome["final_progress"])
                append_jsonl(
                    genealogy_path,
                    {
                        "record_type": "descendant",
                        "stream": stream,
                        "snapshot_id": row["snapshot_id"],
                        "candidate_id": f"{row['snapshot_id']}:{stream}:{root}:{checkpoint_index}:{suffix}",
                        "parent_id": f"{row['snapshot_id']}:root:{root}",
                        "root_index": root,
                        "checkpoint_index": checkpoint_index,
                        "timestep": int(policy.diffusion.noise_scheduler.timesteps[checkpoint_index].item()),
                        "suffix_index": suffix,
                        "suffix_seed": seeds[candidate],
                        "z_s_sha256": tensor_sha256(z_roots[root]),
                        "actions_normalized": norm[candidate].tolist(),
                        "actions_unnormalized": chunks[candidate].tolist(),
                        "remaining_nfe": 100 - checkpoint_index,
                        "sample_nfe": 100 - checkpoint_index,
                        "suffix_wall_clock_batch_seconds": time.perf_counter() - suffix_started,
                        **outcome,
                    },
                )
            stream_vectors[stream] = branchability_vector(
                progress,
                norm,
                [item["final_progress"] for item in control_outcomes],
            )
        curve.append(
            {
                "checkpoint_index": checkpoint_index,
                "timestep": int(policy.diffusion.noise_scheduler.timesteps[checkpoint_index].item()),
                "remaining_nfe": 100 - checkpoint_index,
                "calibration": stream_vectors["calibration"],
                "heldout": stream_vectors["heldout"],
                "heldout_best_gain_per_nfe": stream_vectors["heldout"]["best_descendant_gain"]
                / max(1, 100 - checkpoint_index),
            }
        )
    result = {
        "snapshot_id": row["snapshot_id"],
        "episode_seed": row["episode_seed"],
        "control_step": row["control_step"],
        "stratum": row["stratum"],
        "curve": curve,
        "root_disagreement": disagreement,
        "wall_clock_seconds": time.perf_counter() - started,
        "peak_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        "compute": {
            "root_sample_nfe": K * 100,
            "discovery_suffix_sample_nfe_per_stream": sum(K * M * (100 - c) for c in CHECKPOINTS),
            "root_count": K,
            "suffix_count_per_root_checkpoint_stream": M,
        },
    }
    write_json(output / "branchability_curves" / f"{row['snapshot_id']}.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [
        json.loads(line)
        for line in (args.baseline_output / "snapshot_sampling_frame.jsonl").read_text().splitlines()
        if line.strip()
    ]
    selected = [row for row in rows if row["selected"]]
    selected.sort(key=lambda row: row["snapshot_id"])
    policy = load_policy(args.checkpoint, args.device)
    completed = []
    for index, row in enumerate(selected):
        if index % args.world_size != args.rank:
            continue
        row = dict(row)
        source = args.baseline_output / row["snapshot_file"]
        destination = args.output / row["snapshot_file"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.write_bytes(source.read_bytes())
        result_path = args.output / "branchability_curves" / f"{row['snapshot_id']}.json"
        if result_path.exists():
            completed.append(row["snapshot_id"])
            continue
        result = process_snapshot(policy, row, index, args.output, args.device)
        completed.append(result["snapshot_id"])
        write_json(
            args.output / f"FIRST_WORK.rank{args.rank}.json",
            {"rank": args.rank, "completed_snapshot_ids": completed, "last_snapshot": result},
        )
        print(json.dumps({"rank": args.rank, "completed_snapshot": result["snapshot_id"]}), flush=True)
    write_json(
        args.output / f"COMPLETE.rank{args.rank}.json",
        {"rank": args.rank, "world_size": args.world_size, "completed_snapshot_ids": completed},
    )


if __name__ == "__main__":
    main()
