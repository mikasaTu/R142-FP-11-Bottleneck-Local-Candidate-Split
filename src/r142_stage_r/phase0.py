from __future__ import annotations

import collections
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from .protocol import PROTOCOL_ID, SUITE_MAX_STEPS, atomic_json, ranked_initial_states, sha256_file
from .runtime import (
    configure_external_sources,
    infer_microbatched,
    make_rollout_seeds,
    policy_noise,
    pose_vector,
    shared_environment_seed,
    stable_pose_keys,
    task_config,
)


def _atomic_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _pack_rollouts(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    lengths = np.asarray([len(row["actions"]) for row in rows], dtype=np.int32)
    offsets = np.concatenate([np.asarray([0], dtype=np.int64), np.cumsum(lengths, dtype=np.int64)])
    return {
        "lengths": lengths,
        "offsets": offsets,
        "actions": np.concatenate([row["actions"] for row in rows], axis=0).astype(np.float32),
        "eef": np.concatenate([row["eef"] for row in rows], axis=0).astype(np.float64),
        "objects": np.concatenate([row["objects"] for row in rows], axis=0).astype(np.float64),
        "progress": np.concatenate([row["progress"] for row in rows], axis=0).astype(np.float32),
        "success": np.asarray([row["success"] for row in rows], dtype=np.bool_),
        "init_state": np.asarray([row["init_state"] for row in rows], dtype=np.int16),
        "candidate_id": np.asarray([row["candidate_id"] for row in rows], dtype=np.int16),
        "rollout_seed": np.asarray([row["rollout_seed"] for row in rows], dtype=np.uint64),
        "policy_forwards": np.asarray([row["policy_forwards"] for row in rows], dtype=np.int32),
    }


def load_task_rollouts(npz_path: str | Path) -> list[dict[str, Any]]:
    with np.load(npz_path, allow_pickle=False) as data:
        result = []
        for index, (start, stop) in enumerate(zip(data["offsets"][:-1], data["offsets"][1:])):
            pose = np.concatenate([data["eef"][start:stop], data["objects"][start:stop]], axis=1)
            result.append(
                {
                    "actions": data["actions"][start:stop].copy(),
                    "poses": pose,
                    "progress": data["progress"][start:stop].copy(),
                    "final_progress": float(data["progress"][stop - 1]) if stop > start else 0.0,
                    "success": bool(data["success"][index]),
                    "init_state": int(data["init_state"][index]),
                    "candidate_id": int(data["candidate_id"][index]),
                    "rollout_seed": int(data["rollout_seed"][index]),
                    "policy_forwards": int(data["policy_forwards"][index]),
                }
            )
        return result


class Phase0Runtime:
    def __init__(self, *, qpilots_root: str, libero_root: str, checkpoint: str, microbatch: int):
        configure_external_sources(qpilots_root, libero_root)
        from qpilots_libero.policy import CleanPi05LiberoPolicy

        self.policy_class = CleanPi05LiberoPolicy
        self.checkpoint = checkpoint
        self.microbatch = int(microbatch)
        self.policy = None

    def _ensure_policy(self, prompt: str):
        if self.policy is None:
            self.policy = self.policy_class(self.checkpoint, default_prompt=prompt)
        return self.policy

    def run_task(self, suite_name: str, task_id: int, output_dir: Path, *, candidates: int = 32) -> dict[str, Any]:
        from libero.libero import benchmark
        from qpilots_libero.environment import Task64Environment

        suite = benchmark.get_benchmark_dict()[suite_name]()
        task = suite.get_task(int(task_id))
        prompt = str(task.language)
        max_steps = SUITE_MAX_STEPS.get(suite_name, 400)
        config = task_config(suite_name, task_id, prompt, max_steps)
        policy = self._ensure_policy(prompt)
        envs = [Task64Environment(config, seed=0) for _ in range(int(candidates))]
        rows: list[dict[str, Any]] = []
        pose_keys: tuple[str, ...] | None = None
        candidate_equivalent_forwards = 0
        physical_batched_calls = 0
        started = time.time()
        try:
            for init_state in ranked_initial_states(suite_name, task_id):
                common_seed = shared_environment_seed(suite_name, task_id, init_state)
                seeds = make_rollout_seeds(suite_name, task_id, init_state)[: int(candidates)]
                queues = [collections.deque() for _ in range(int(candidates))]
                traces = [
                    {"actions": [], "eef": [], "objects": [], "progress": [], "success": False, "forwards": 0}
                    for _ in range(int(candidates))
                ]
                done = np.zeros(int(candidates), dtype=np.bool_)
                counters = np.zeros(int(candidates), dtype=np.int32)
                for env in envs:
                    env.environment.seed(int(common_seed))
                    env.evaluation_seed = int(common_seed)
                    env.reset(int(init_state))
                observed_keys = stable_pose_keys(envs[0]._observation)
                if pose_keys is None:
                    pose_keys = observed_keys
                elif pose_keys != observed_keys:
                    raise RuntimeError(f"object pose keys drifted: {pose_keys} != {observed_keys}")

                while not bool(np.all(done)):
                    needs_plan = [index for index in range(int(candidates)) if not done[index] and not queues[index]]
                    if needs_plan:
                        observations = [envs[index].raw_observation() for index in needs_plan]
                        noises = [policy_noise(policy, seeds[index], int(counters[index])) for index in needs_plan]
                        chunks = infer_microbatched(policy, observations, noises, microbatch=self.microbatch)
                        candidate_equivalent_forwards += len(needs_plan)
                        physical_batched_calls += (len(needs_plan) + self.microbatch - 1) // self.microbatch
                        for local, index in enumerate(needs_plan):
                            queues[index].extend(np.asarray(chunks[local, :5], dtype=np.float32))
                            counters[index] += 1
                            traces[index]["forwards"] += 1

                    for index, env in enumerate(envs):
                        if done[index]:
                            continue
                        action = np.asarray(queues[index].popleft(), dtype=np.float32)
                        step = env.execute_actions(action[None])
                        eef, objects = pose_vector(env._observation, pose_keys or ())
                        predicates = env.official_predicates()
                        traces[index]["actions"].append(action.copy())
                        traces[index]["eef"].append(eef)
                        traces[index]["objects"].append(objects)
                        traces[index]["progress"].append(float(predicates["fraction"]))
                        traces[index]["success"] = bool(step["success"])
                        done[index] = bool(step["done"])

                for index, trace in enumerate(traces):
                    rows.append(
                        {
                            "actions": np.asarray(trace["actions"], dtype=np.float32),
                            "eef": np.asarray(trace["eef"], dtype=np.float64),
                            "objects": np.asarray(trace["objects"], dtype=np.float64),
                            "progress": np.asarray(trace["progress"], dtype=np.float32),
                            "success": bool(trace["success"]),
                            "init_state": int(init_state),
                            "candidate_id": int(index),
                            "rollout_seed": int(seeds[index]),
                            "policy_forwards": int(trace["forwards"]),
                        }
                    )
        finally:
            for env in envs:
                env.close()

        stem = f"{suite_name}_task{int(task_id):02d}"
        npz_path = output_dir / f"{stem}.npz"
        metadata_path = output_dir / f"{stem}.json"
        _atomic_npz(npz_path, _pack_rollouts(rows))
        metadata = {
            "protocol_id": PROTOCOL_ID,
            "suite": suite_name,
            "task_id": int(task_id),
            "prompt": prompt,
            "pose_keys": list(pose_keys or ()),
            "rollout_count": len(rows),
            "candidate_equivalent_policy_forwards": int(candidate_equivalent_forwards),
            "physical_batched_policy_calls": int(physical_batched_calls),
            "environment_steps": int(sum(len(row["actions"]) for row in rows)),
            "branch_count": 0,
            "budget_slack": 0,
            "elapsed_seconds": time.time() - started,
            "data_file": npz_path.name,
            "data_sha256": sha256_file(npz_path),
        }
        atomic_json(metadata_path, metadata)
        return metadata
