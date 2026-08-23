#!/usr/bin/env python3
"""Run real-source A-E gates and frozen PushT baseline rollouts.

This entry point imports the pinned LeRobot checkout through PYTHONPATH and
loads a local, SHA-verified checkpoint directory. It never downloads a floating
revision and never changes the environment geometry or dynamics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from lerobot.common.envs.utils import preprocess_observation
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from r142_stage2a.snapshot import (
    ReplaySnapshot,
    execute_chunk,
    max_abs_state_error,
    native_push_t_state,
    observation_sha256,
    restore_by_replay,
)
from r142_stage2a.tracing import PassiveTrace, resume_suffix, tensor_sha256


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def append_jsonl(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def make_env():
    import gym_pusht  # noqa: F401 - import registers gym_pusht/PushT-v0

    return gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        visualization_width=384,
        visualization_height=384,
        max_episode_steps=300,
        disable_env_checker=True,
    )


def make_vector_env(count: int):
    return gym.vector.SyncVectorEnv([make_env for _ in range(count)])


def batched_observation(observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    pixels = observation.get("pixels")
    if pixels is not None and pixels.ndim == 4:
        return observation
    return {key: np.expand_dims(value, axis=0) for key, value in observation.items()}


def policy_batch(observation: dict[str, np.ndarray], device: str) -> dict[str, torch.Tensor]:
    batch = preprocess_observation(batched_observation(observation))
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def rng_state() -> dict:
    result = {
        "python": repr(random.getstate()),
        "numpy": repr(np.random.get_state()),
        "torch_cpu": torch.random.get_rng_state().cpu().tolist(),
    }
    if torch.cuda.is_available():
        result["torch_cuda"] = [state.cpu().tolist() for state in torch.cuda.get_rng_state_all()]
    return result


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_policy(checkpoint: Path, device: str) -> DiffusionPolicy:
    expected = "995d14d35db57d95c35ad9704c3d79c8612b7bc45f3877e5c46c2cdc516856a8"
    actual = sha256_file(checkpoint / "model.safetensors")
    if actual != expected:
        raise RuntimeError(f"checkpoint weights hash mismatch: {actual} != {expected}")
    # The published checkpoint includes two later CLI-only fields (`device`,
    # `use_amp`) that are not dataclass members at the model-card's pinned
    # training commit. Build the exact old-commit config explicitly and record
    # the compatibility exclusion in the manifest; weights remain untouched.
    raw = json.loads((checkpoint / "config.json").read_text())
    for key in ("type", "device", "use_amp"):
        raw.pop(key, None)
    raw["input_features"] = {
        key: PolicyFeature(type=FeatureType(value["type"]), shape=tuple(value["shape"]))
        for key, value in raw["input_features"].items()
    }
    raw["output_features"] = {
        key: PolicyFeature(type=FeatureType(value["type"]), shape=tuple(value["shape"]))
        for key, value in raw["output_features"].items()
    }
    raw["normalization_mapping"] = {
        key: NormalizationMode(value) for key, value in raw["normalization_mapping"].items()
    }
    config = DiffusionConfig(**raw)
    policy = DiffusionPolicy.from_pretrained(
        checkpoint, config=config, map_location=device, local_files_only=True, strict=True
    )
    policy.eval()
    return policy


def _snapshot_torch_rng():
    return (
        torch.random.get_rng_state().clone(),
        [x.clone() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    )


def _restore_torch_rng(state):
    torch.random.set_rng_state(state[0])
    if state[1]:
        torch.cuda.set_rng_state_all(state[1])


def conditioning_from_history(policy, observations, device):
    normalized = []
    for observation in observations:
        normalized.append(policy.normalize_inputs(policy_batch(observation, device)))
    batch = {
        "observation.state": torch.stack([item["observation.state"] for item in normalized], dim=1)
    }
    if policy.config.image_features:
        per_step = []
        for item in normalized:
            per_step.append(
                torch.stack([item[key] for key in policy.config.image_features], dim=-4)
            )
        batch["observation.images"] = torch.stack(per_step, dim=1)
    if policy.config.env_state_feature:
        batch["observation.environment_state"] = torch.stack(
            [item["observation.environment_state"] for item in normalized], dim=1
        )
    return policy.diffusion._prepare_global_conditioning(batch)


def run_gates(policy, device: str, output: Path) -> dict:
    env = make_env()
    observation, _ = env.reset(seed=14211)
    batch = policy_batch(observation, device)

    # A: the disabled wrapper is delegation; two frozen-original calls must match.
    set_seed(1421101)
    saved = _snapshot_torch_rng()
    policy.reset()
    original = policy.select_action(batch)
    policy.reset()
    _restore_torch_rng(saved)
    delegated = policy.select_action(batch)
    a_equal = torch.equal(original, delegated)

    # B: call the same original function under passive hooks only.
    policy.reset()
    _restore_torch_rng(saved)
    trace = PassiveTrace(policy.diffusion, capture_tensors=True)
    with trace.installed():
        observed = policy.select_action(batch)
    b_equal = torch.equal(original, observed)

    # C/D use the same real model/scheduler with an explicit, auditable generator.
    global_cond = conditioning_from_history(policy, [observation, observation], device)
    generator_device = torch.device(device).type
    root_generator = torch.Generator(device=generator_device).manual_seed(1421102)
    root_trace = PassiveTrace(policy.diffusion, generator=root_generator, capture_tensors=True)
    root_final = root_trace.run(1, global_cond)
    checkpoint_index = 53
    record = root_trace.steps[checkpoint_index]
    if record.rng_before_step is None:
        raise RuntimeError("explicit generator state was not captured")
    same_generator = torch.Generator(device=generator_device)
    same_generator.set_state(record.rng_before_step)
    same_final, _ = resume_suffix(
        policy.diffusion,
        record.z_before,
        checkpoint_index,
        global_cond,
        same_generator,
        capture=True,
    )
    c_equal = torch.equal(root_final, same_final)
    different_generator = torch.Generator(device=generator_device).manual_seed(1421103)
    different_final, _ = resume_suffix(
        policy.diffusion,
        record.z_before,
        checkpoint_index,
        global_cond,
        different_generator,
        capture=False,
    )
    d_max_abs = float((root_final - different_final).abs().max().cpu().item())

    # E: real mid-episode state restored by seed+complete action-prefix replay.
    actions = []
    observations = [observation]
    policy.reset()
    set_seed(1421104)
    for _ in range(12):
        action = policy.select_action(policy_batch(observations[-1], device))[0].detach().cpu().numpy()
        observations.append(env.step(action)[0])
        actions.append(tuple(float(x) for x in action))
    snapshot = ReplaySnapshot(
        episode_seed=14211,
        control_step=12,
        action_prefix=tuple(actions),
        native_state=native_push_t_state(env),
        observation_sha256=observation_sha256(observations[-1]),
    )
    restore1, obs1, _ = restore_by_replay(make_env, snapshot)
    restore2, obs2, _ = restore_by_replay(make_env, snapshot)
    restored_state1 = native_push_t_state(restore1)
    restored_state2 = native_push_t_state(restore2)
    state_error = max(
        max_abs_state_error(snapshot.native_state, restored_state1),
        max_abs_state_error(snapshot.native_state, restored_state2),
    )
    chunk = np.asarray(actions[-8:], dtype=np.float32)
    result1 = execute_chunk(restore1, chunk)
    result2 = execute_chunk(restore2, chunk)
    e_equal = (
        observation_sha256(obs1) == observation_sha256(obs2)
        and result1["observation_sha256"] == result2["observation_sha256"]
        and result1["rewards"] == result2["rewards"]
        and state_error == 0.0
    )
    env.close()
    restore1.close()
    restore2.close()

    result = {
        "schema_version": 1,
        "device": device,
        "tests": {
            "A_tracing_disabled_delegates_exactly": {"pass": a_equal},
            "B_passive_tracing_is_noop": {
                "pass": b_equal,
                "captured_steps": len(trace.steps),
            },
            "C_same_z_same_rng_reproduces_suffix": {
                "pass": c_equal,
                "checkpoint_index": checkpoint_index,
                "root_final_sha256": tensor_sha256(root_final),
                "resumed_final_sha256": tensor_sha256(same_final),
            },
            "D_same_z_new_rng_changes_suffix": {
                "pass": d_max_abs > 0,
                "max_abs_difference": d_max_abs,
                "different_final_sha256": tensor_sha256(different_final),
            },
            "E_replay_snapshot_equivalence": {
                "pass": e_equal,
                "snapshot_sha256": snapshot.snapshot_sha256,
                "max_abs_native_state_error": state_error,
                "restored_state_sha256": hashlib.sha256(
                    json.dumps(restored_state1, sort_keys=True).encode()
                ).hexdigest(),
            },
        },
    }
    result["all_pass"] = all(test["pass"] for test in result["tests"].values())
    write_json(output / "resume_equivalence_tests.json", result)
    write_json(output / "simulator_snapshot_tests.json", result["tests"]["E_replay_snapshot_equivalence"])
    return result


def run_baseline(policy, device: str, output: Path, seed_start: int, episodes: int) -> dict:
    rollouts_path = output / "baseline_reproduction.jsonl"
    if rollouts_path.exists():
        rollouts_path.unlink()
    candidate_path = output / "snapshot_sampling_frame.jsonl"
    if candidate_path.exists():
        candidate_path.unlink()
    seeds = list(range(seed_start, seed_start + episodes))
    env = make_vector_env(episodes)
    observation, _ = env.reset(seed=seeds)
    previous_observation = {key: value.copy() for key, value in observation.items()}
    policy.reset()
    policy_seed = 142200000 + seed_start
    set_seed(policy_seed)
    start = time.perf_counter()
    done = np.zeros(episodes, dtype=bool)
    successes = np.zeros(episodes, dtype=bool)
    trajectories = [[] for _ in seeds]
    all_actions = [[] for _ in seeds]
    all_rewards = [[] for _ in seeds]
    candidates = [[] for _ in seeds]
    generated_chunks = 0
    snapshots_dir = output / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    for step in range(300):
        queue_was_empty = len(policy._queues["action"]) == 0
        action_batch = policy.select_action(policy_batch(observation, device)).detach().cpu().numpy()
        if queue_was_empty:
            generated_chunks += 1
        next_observation, reward, terminated, truncated, info = env.step(action_batch)
        just_done = (terminated | truncated) & ~done
        if "final_info" in info:
            for i, final_info in enumerate(info["final_info"]):
                if final_info is not None:
                    successes[i] = successes[i] or bool(final_info.get("is_success", False))
        for i, seed in enumerate(seeds):
            if done[i]:
                continue
            action = tuple(float(x) for x in action_batch[i])
            all_actions[i].append(action)
            all_rewards[i].append(float(reward[i]))
            state = native_push_t_state(env.envs[i]) if not just_done[i] else {}
            single_next = {key: value[i] for key, value in next_observation.items()}
            trajectories[i].append(
                {
                    "control_step": step,
                    "action": action,
                    "progress": float(reward[i]),
                    "is_success": bool(successes[i]),
                    "native_state": {k: v for k, v in state.items() if k != "np_random_state"},
                    "observation_sha256": observation_sha256(single_next),
                }
            )
            if step + 1 in {50, 100, 150, 200, 250} and not just_done[i]:
                snapshot = ReplaySnapshot(
                    episode_seed=seed,
                    control_step=step + 1,
                    action_prefix=tuple(all_actions[i]),
                    native_state=state,
                    observation_sha256=observation_sha256(single_next),
                )
                np.savez_compressed(
                    snapshots_dir / f"{snapshot.snapshot_sha256}.npz",
                    previous_pixels=previous_observation["pixels"][i],
                    previous_agent_pos=previous_observation["agent_pos"][i],
                    current_pixels=next_observation["pixels"][i],
                    current_agent_pos=next_observation["agent_pos"][i],
                    actions=np.asarray(all_actions[i], dtype=np.float32),
                )
                candidates[i].append(
                    {
                        "snapshot_id": snapshot.snapshot_sha256,
                        "episode_seed": seed,
                        "control_step": step + 1,
                        "progress": float(reward[i]),
                        "max_progress_so_far": float(max(all_rewards[i])),
                        "native_state": state,
                        "snapshot_file": f"snapshots/{snapshot.snapshot_sha256}.npz",
                        "selected": False,
                        "stratum": None,
                    }
                )
        done |= terminated | truncated
        previous_observation = {key: value.copy() for key, value in observation.items()}
        observation = next_observation
        if np.all(done):
            break
    elapsed = time.perf_counter() - start
    summaries = []
    for episode_index, seed in enumerate(seeds):
        final_max = float(max(all_rewards[episode_index], default=0.0))
        row = {
            "episode_index": episode_index,
            "episode_seed": seed,
            "policy_seed": policy_seed,
            "success": bool(successes[episode_index]),
            "max_progress": final_max,
            "sum_reward": float(sum(all_rewards[episode_index])),
            "control_steps": len(all_actions[episode_index]),
            "generated_action_chunks_shared_batch": generated_chunks,
            "actual_model_nfe_shared_batch": generated_chunks * policy.diffusion.num_inference_steps,
            "num_inference_steps": policy.diffusion.num_inference_steps,
            "scheduler_type": policy.config.noise_scheduler_type,
            "horizon": policy.config.horizon,
            "n_obs_steps": policy.config.n_obs_steps,
            "n_action_steps": policy.config.n_action_steps,
            "batch_wall_clock_seconds": elapsed,
            "trajectory": trajectories[episode_index],
        }
        append_jsonl(rollouts_path, row)
        summaries.append(row)
        for candidate in candidates[episode_index]:
            candidate["episode_success"] = bool(successes[episode_index])
            candidate["episode_max_progress"] = final_max
            append_jsonl(candidate_path, candidate)
        print(json.dumps({k: row[k] for k in ("episode_seed", "success", "max_progress", "control_steps")}), flush=True)
    env.close()
    result = {
        "episodes": episodes,
        "seed_start": seed_start,
        "success_rate": float(np.mean([r["success"] for r in summaries])),
        "mean_max_progress": float(np.mean([r["max_progress"] for r in summaries])),
        "shared_batch_model_nfe": int(generated_chunks * policy.diffusion.num_inference_steps),
        "batch_wall_clock_seconds": elapsed,
    }
    write_json(output / "baseline_summary.json", result)
    return result


def environment_manifest(checkpoint: Path, device: str) -> dict:
    import diffusers
    import gym_pusht
    import pymunk
    import torchvision

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "diffusers": diffusers.__version__,
        "gymnasium": gym.__version__,
        "gym_pusht": getattr(gym_pusht, "__version__", "0.1.5-package-metadata"),
        "pymunk": getattr(pymunk, "version", getattr(pymunk, "__version__", "unknown")),
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "checkpoint_files": {
            path.name: sha256_file(path) for path in sorted(checkpoint.iterdir()) if path.is_file()
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=("gates", "baseline", "all"), default="all")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "pinned_environment_manifest.json", environment_manifest(args.checkpoint, args.device))
    policy = load_policy(args.checkpoint, args.device)
    if args.mode in ("gates", "all"):
        gates = run_gates(policy, args.device, args.output)
        print(json.dumps({"gates": gates["all_pass"]}), flush=True)
        if not gates["all_pass"] and args.mode == "all":
            raise SystemExit("A-E gate failed; baseline/descendant work was not started")
    if args.mode in ("baseline", "all"):
        print(json.dumps(run_baseline(policy, args.device, args.output, args.seed_start, args.episodes)), flush=True)


if __name__ == "__main__":
    main()
