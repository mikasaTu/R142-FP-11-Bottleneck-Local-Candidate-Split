"""Replay-backed PushT snapshots and stable evidence hashes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _xy(body: Any, name: str) -> list[float]:
    value = getattr(body, name)
    return [float(value.x), float(value.y)]


def native_push_t_state(env: Any) -> dict[str, Any]:
    """Capture native numerical state without claiming collision-cache restore."""

    base = env.unwrapped
    result = {
        "agent_position": _xy(base.agent, "position"),
        "agent_velocity": _xy(base.agent, "velocity"),
        "agent_angle": float(base.agent.angle),
        "agent_angular_velocity": float(base.agent.angular_velocity),
        "block_position": _xy(base.block, "position"),
        "block_velocity": _xy(base.block, "velocity"),
        "block_angle": float(base.block.angle),
        "block_angular_velocity": float(base.block.angular_velocity),
    }
    if hasattr(base, "goal_pose"):
        result["goal_pose"] = np.asarray(base.goal_pose).astype(float).tolist()
    if hasattr(base, "np_random"):
        result["np_random_state"] = base.np_random.bit_generator.state
    elapsed = getattr(env, "_elapsed_steps", None)
    if elapsed is not None:
        result["elapsed_steps"] = int(elapsed)
    return result


def max_abs_state_error(lhs: dict[str, Any], rhs: dict[str, Any]) -> float:
    errors: list[float] = []
    for key in (
        "agent_position",
        "agent_velocity",
        "agent_angle",
        "agent_angular_velocity",
        "block_position",
        "block_velocity",
        "block_angle",
        "block_angular_velocity",
    ):
        errors.append(float(np.max(np.abs(np.asarray(lhs[key]) - np.asarray(rhs[key])))))
    return max(errors)


@dataclass(frozen=True)
class ReplaySnapshot:
    episode_seed: int
    control_step: int
    action_prefix: tuple[tuple[float, float], ...]
    native_state: dict[str, Any]
    observation_sha256: str

    @property
    def snapshot_sha256(self) -> str:
        return canonical_sha256(
            {
                "episode_seed": self.episode_seed,
                "control_step": self.control_step,
                "action_prefix": self.action_prefix,
                "native_state": self.native_state,
                "observation_sha256": self.observation_sha256,
            }
        )


def observation_sha256(observation: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(observation):
        value = np.ascontiguousarray(observation[key])
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def restore_by_replay(
    env_factory: Callable[[], Any], snapshot: ReplaySnapshot
) -> tuple[Any, dict[str, np.ndarray], dict[str, Any]]:
    """Create/reset an env and replay the exact recorded prefix."""

    env = env_factory()
    observation, info = env.reset(seed=snapshot.episode_seed)
    for action in snapshot.action_prefix:
        observation, _, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
        if terminated or truncated:
            raise RuntimeError("snapshot action prefix terminated before its declared control step")
    return env, observation, info


def execute_chunk(env: Any, actions: Iterable[Iterable[float]]) -> dict[str, Any]:
    rewards: list[float] = []
    infos: list[dict[str, Any]] = []
    observations: list[str] = []
    terminated = truncated = False
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
        rewards.append(float(reward))
        infos.append(_jsonable_info(info))
        observations.append(observation_sha256(obs))
        if terminated or truncated:
            break
    return {
        "rewards": rewards,
        "max_progress": max(rewards, default=float("nan")),
        "final_progress": rewards[-1] if rewards else float("nan"),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": infos,
        "observation_sha256": observations,
        "final_native_state": native_push_t_state(env),
    }


def _jsonable_info(info: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in info.items():
        if isinstance(value, np.ndarray):
            result[key] = value.astype(float).tolist()
        elif isinstance(value, np.generic):
            result[key] = value.item()
        elif isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
    return result
