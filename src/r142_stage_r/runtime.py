from __future__ import annotations

import collections
import copy
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .protocol import environment_seed, rollout_seed


def configure_external_sources(qpilots_root: str, libero_root: str) -> None:
    import sys

    for value in (str(Path(qpilots_root).resolve()), str(Path(libero_root).resolve())):
        if value not in sys.path:
            sys.path.insert(0, value)
    os.environ["QPILOTS_LIBERO_SITE"] = str(Path(libero_root).resolve())


def task_config(suite: str, task_id: int, prompt: str, max_steps: int) -> dict[str, Any]:
    return {
        "task": {
            "suite": suite,
            "task_id": int(task_id),
            "prompt": prompt,
            "init_state_count": 50,
            "num_steps_wait": 10,
            "replan_steps": 5,
            "max_steps": int(max_steps),
        }
    }


def stable_pose_keys(observation: dict[str, Any]) -> tuple[str, ...]:
    keys = []
    for key, value in observation.items():
        if key.startswith("robot0_") or "_to_" in key:
            continue
        if not (key.endswith("_pos") or key.endswith("_quat")):
            continue
        array = np.asarray(value)
        if array.shape in {(3,), (4,)}:
            keys.append(key)
    return tuple(sorted(keys))


def pose_vector(observation: dict[str, Any], keys: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    eef = np.concatenate(
        [
            np.asarray(observation["robot0_eef_pos"], dtype=np.float64),
            np.asarray(observation["robot0_eef_quat"], dtype=np.float64),
        ]
    )
    objects = np.concatenate([np.asarray(observation[key], dtype=np.float64) for key in keys]) if keys else np.empty(0)
    return eef, objects


def policy_noise(policy: Any, seed: int, counter: int) -> np.ndarray:
    seed = int(seed)
    key = policy.jax.random.PRNGKey(seed & 0xFFFFFFFF)
    key = policy.jax.random.fold_in(key, (seed >> 32) & 0xFFFFFFFF)
    key = policy.jax.random.fold_in(key, int(counter))
    return np.asarray(
        policy.jax.random.normal(key, (int(policy.model.action_horizon), int(policy.model.action_dim))),
        dtype=np.float32,
    )


def infer_physical_many(policy: Any, observations: list[dict[str, Any]], noises: np.ndarray) -> np.ndarray:
    model_actions = np.asarray(
        policy.sample_model_actions_official(observations, policy.jnp.asarray(noises)),
        dtype=np.float32,
    )
    # The pinned LiberoOutputs transform is unbatched: applying it to the full
    # (B,10,32) tensor would slice the horizon axis. Apply the exact official
    # transform per sample so E2 can demand bit identity.
    physical = []
    for observation, actions in zip(observations, model_actions, strict=True):
        transformed = policy.input_transform(dict(observation))
        output = policy.output_transform(
            {"state": np.asarray(transformed["state"]), "actions": np.asarray(actions)}
        )
        physical.append(np.asarray(output["actions"], dtype=np.float32))
    result = np.asarray(physical, dtype=np.float32)
    if result.shape != (len(observations), 10, 7):
        raise RuntimeError(f"official output transform returned {result.shape}, expected {(len(observations), 10, 7)}")
    return result


def infer_microbatched(
    policy: Any,
    observations: list[dict[str, Any]],
    noises: list[np.ndarray],
    *,
    microbatch: int,
) -> np.ndarray:
    if not observations:
        return np.empty((0, 10, 7), dtype=np.float32)
    result = []
    for start in range(0, len(observations), int(microbatch)):
        chunk_obs = observations[start : start + int(microbatch)]
        chunk_noise = noises[start : start + int(microbatch)]
        real_count = len(chunk_obs)
        while len(chunk_obs) < int(microbatch):
            chunk_obs.append(copy.deepcopy(chunk_obs[-1]))
            chunk_noise.append(np.asarray(chunk_noise[-1]).copy())
        result.append(infer_physical_many(policy, chunk_obs, np.asarray(chunk_noise))[:real_count])
    return np.concatenate(result, axis=0)


@dataclass
class RunnerSnapshot:
    simulator: Any
    observation_history: list[dict[str, Any]]
    action_queue: list[np.ndarray]
    python_rng_state: object
    numpy_rng_state: object
    noise_seed: int
    noise_counter: int


class TrajectoryRunner:
    def __init__(self, environment: Any, *, noise_seed: int):
        self.environment = environment
        self.observation_history: collections.deque[dict[str, Any]] = collections.deque(maxlen=1)
        self.action_queue: collections.deque[np.ndarray] = collections.deque()
        self.noise_seed = int(noise_seed)
        self.noise_counter = 0

    def reset(self, init_state_index: int) -> dict[str, Any]:
        observation = self.environment.reset(int(init_state_index))
        self.observation_history.clear()
        self.observation_history.append(copy.deepcopy(observation))
        self.action_queue.clear()
        self.noise_counter = 0
        return observation

    def snapshot(self) -> RunnerSnapshot:
        return RunnerSnapshot(
            simulator=self.environment.capture_snapshot(),
            observation_history=copy.deepcopy(list(self.observation_history)),
            action_queue=[np.asarray(value).copy() for value in self.action_queue],
            python_rng_state=copy.deepcopy(random.getstate()),
            numpy_rng_state=copy.deepcopy(np.random.get_state()),
            noise_seed=int(self.noise_seed),
            noise_counter=int(self.noise_counter),
        )

    def restore(self, snapshot: RunnerSnapshot, *, omit: str | None = None) -> None:
        if omit != "simulator":
            self.environment.restore_snapshot(snapshot.simulator)
        if omit != "history":
            self.observation_history = collections.deque(copy.deepcopy(snapshot.observation_history), maxlen=1)
        if omit != "queue":
            self.action_queue = collections.deque([np.asarray(value).copy() for value in snapshot.action_queue])
        if omit != "rng":
            random.setstate(copy.deepcopy(snapshot.python_rng_state))
            np.random.set_state(copy.deepcopy(snapshot.numpy_rng_state))
            self.noise_seed = int(snapshot.noise_seed)
            self.noise_counter = int(snapshot.noise_counter)


def make_rollout_seeds(suite: str, task_id: int, init_state: int) -> list[int]:
    return [rollout_seed(suite, task_id, init_state, candidate) for candidate in range(32)]


def shared_environment_seed(suite: str, task_id: int, init_state: int) -> int:
    return environment_seed(suite, task_id, init_state)
