"""Concrete Stage-S S4/S5 adapters for the pinned LIBERO and RoboTwin APIs.

The S4/S5 executor in :mod:`r142_stage_s.s45_runtime` is deliberately
substrate agnostic.  This module binds it to the two real runtimes already
used by Stage-S:

* :class:`LiberoS45Adapter` delegates environment construction, action-chunk
  inference, replay and full-state snapshots to ``libero.py``.
* :class:`RoboTwinS45Adapter` delegates simulator/policy state capture and
  restore to ``robotwin.py``'s ``ConcreteRoboTwinRuntime``.

No class below creates a fake environment or a synthetic success label.  A
missing official hook, policy RNG stream, observation history, action queue,
simulator snapshot, or terminal signal raises ``S45AdapterError`` before a
completion marker can be written.  The adapter factories require explicit
runtime factories and paths; they never invent a checkpoint, seed formula,
grid, or branch location.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import libero as libero_runtime
from . import robotwin as robotwin_runtime
from .s45_runtime import (
    ProtocolAuthority,
    S45Adapter,
    S45CapabilityError,
    S45ProvenanceError,
    SNAPSHOT_REPLAY_TOLERANCE,
    _candidate_success,
    canonical_json,
)


class S45AdapterError(S45CapabilityError):
    """A concrete substrate adapter cannot prove an execution contract."""


def _copy(value: Any) -> Any:
    """Copy state while retaining official process-local simulator handles."""

    if isinstance(value, Mapping):
        return {key: (item if key in {"object", "scene"} else _copy(item)) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy(item) for item in value)
    if isinstance(value, np.ndarray):
        return value.copy()
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if hasattr(value, "detach") and callable(value.detach):
        return value.detach().cpu().tolist()
    if hasattr(value, "p") and hasattr(value, "q"):
        return {"p": _jsonable(value.p), "q": _jsonable(value.q)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items() if key not in {"object", "scene"}}
    if isinstance(value, (tuple, list)):
        return [_jsonable(child) for child in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _invoke_factory(factory: Callable[..., Any], **kwargs: Any) -> Any:
    """Call a user-supplied factory without masking errors from its body."""

    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory(**kwargs)
    parameters = signature.parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return factory(**kwargs)
    accepted = {key: value for key, value in kwargs.items() if key in parameters}
    required = [
        name
        for name, parameter in parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and name not in accepted
    ]
    if required:
        raise S45AdapterError(f"factory {factory!r} requires unsupported arguments {required}")
    return factory(**accepted)


def _metadata_value(family: Mapping[str, Any], *keys: str) -> Any:
    metadata = family.get("metadata")
    for source in (family, metadata if isinstance(metadata, Mapping) else {}):
        for key in keys:
            if key in source and source[key] is not None:
                return source[key]
    return None


def _family_ids(family: Mapping[str, Any]) -> tuple[int, int]:
    task = _metadata_value(family, "task_id")
    init = _metadata_value(family, "init_state_id", "init_state", "initial_state_id")
    if task is None or init is None:
        raise S45ProvenanceError(f"family {family.get('family_id')} lacks task_id/init_state identity")
    try:
        return int(task), int(init)
    except (TypeError, ValueError) as exc:
        raise S45ProvenanceError("task_id and init_state_id must be integer identities") from exc


def _row_actions(row: Mapping[str, Any]) -> list[Any]:
    value = row.get("actions", row.get("action_prefix"))
    if value is None:
        raise S45ProvenanceError(f"candidate {row.get('candidate_id')} lacks persisted actions")
    if isinstance(value, Mapping) and "data" in value:
        value = value["data"]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise S45ProvenanceError(f"candidate {row.get('candidate_id')} actions are not a sequence")
    return _jsonable(value)


def _row_trajectory(row: Mapping[str, Any]) -> list[Any]:
    value = row.get("trajectory", row.get("poses", row.get("pose_trajectory")))
    if value is None:
        raise S45ProvenanceError(f"candidate {row.get('candidate_id')} lacks persisted trajectory")
    if isinstance(value, Mapping) and "data" in value:
        value = value["data"]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise S45ProvenanceError(f"candidate {row.get('candidate_id')} trajectory is not a sequence")
    return _jsonable(value)


def _row_seed(row: Mapping[str, Any]) -> int:
    value = row.get("candidate_seed", row.get("seed"))
    if value is None:
        raise S45ProvenanceError(f"candidate {row.get('candidate_id')} lacks candidate seed")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise S45ProvenanceError(f"candidate {row.get('candidate_id')} has invalid candidate seed") from exc


def _action_equal(left: Any, right: Any) -> bool:
    try:
        return bool(np.array_equal(np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)))
    except (TypeError, ValueError):
        return canonical_json(left) == canonical_json(right)


def _require_finite_int(value: Any, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise S45AdapterError(f"{label} must be an integer") from exc
    if result < 0:
        raise S45AdapterError(f"{label} must be non-negative")
    return result


def _frozen_seed(
    formula: Any,
    explicit: Any,
    context: Mapping[str, Any],
    *,
    label: str,
    explicit_keys: Sequence[str] = (),
) -> int:
    """Evaluate only a literal protocol seed formula or its explicit table.

    The protocol may publish a table (useful when every family has a
    different episode length), or a restricted hash expression such as
    ``sha256(protocol_id|s4|family_id|mode|branch_index)->first_8_bytes_big_endian``.
    Unknown expressions are rejected; this function never chooses a new salt,
    endian order, or field ordering.
    """

    if isinstance(explicit, Mapping):
        for key in explicit_keys:
            if key in explicit:
                return _require_finite_int(explicit[key], label)
    if not isinstance(formula, str) or not formula.strip():
        raise S45AdapterError(f"{label} has no frozen formula")
    match = re.fullmatch(
        r"sha256\(([^)]*)\)\s*[-=]?>\s*first_8_bytes_(big|little)_endian",
        formula.strip(),
        flags=re.IGNORECASE,
    )
    if match is None:
        raise S45AdapterError(f"unsupported frozen {label} formula; publish a canonical hash expression or explicit seed table")
    tokens = [token.strip() for token in match.group(1).split("|")]
    values: list[str] = []
    for token in tokens:
        if token in context:
            values.append(str(context[token]))
        elif token.startswith("'") and token.endswith("'"):
            values.append(token[1:-1])
        elif token.startswith('"') and token.endswith('"'):
            values.append(token[1:-1])
        elif token:
            # A bare literal is permitted only when it is a protocol-owned
            # namespace token (for example s4); runtime-specific constants are
            # intentionally not accepted here.
            if token not in {"s4", "s5", "branch", "extension", "candidate"}:
                raise S45AdapterError(f"unknown token {token!r} in frozen {label} formula")
            values.append(token)
        else:
            raise S45AdapterError(f"empty token in frozen {label} formula")
    digest = hashlib.sha256("|".join(values).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], match.group(2).lower(), signed=False)


def _termination_result(result: Any) -> Mapping[str, Any]:
    if result is None:
        return {}
    if isinstance(result, Mapping):
        return result
    if isinstance(result, tuple):
        if len(result) >= 5:
            info = result[4] if isinstance(result[4], Mapping) else {}
            return {"done": bool(result[2] or result[3]), "success": bool(info.get("success", False)), "info": info}
        if len(result) >= 4:
            return {"done": bool(result[2]), "success": False, "info": result[3]}
    return {"done": bool(getattr(result, "done", False))}


@dataclass
class _LiberoSnapshot:
    stage_r: libero_runtime.StageRSnapshot
    environment_rng: Any
    policy_rng: Any
    policy_history: Any
    policy_queue: Any


class _LiberoS45Base(S45Adapter):
    """Shared implementation for B (variant) and C (under-trained) LIBERO."""

    def __init__(
        self,
        environment_factory: Callable[..., Any],
        policy_factory: Callable[..., Any],
        *,
        protocol: ProtocolAuthority,
        substrate: str,
        max_steps: int,
        require_torch: bool = True,
    ) -> None:
        if substrate not in {"B", "C"}:
            raise S45AdapterError("LIBERO S45 adapter substrate must be B or C")
        if not callable(environment_factory) or not callable(policy_factory):
            raise S45AdapterError("LIBERO environment_factory and policy_factory are required callables")
        if int(max_steps) <= 0:
            raise S45AdapterError("LIBERO max_steps must be positive")
        self.environment_factory = environment_factory
        self.policy_factory = policy_factory
        self.protocol = protocol
        self.substrate = substrate
        self.max_steps = int(max_steps)
        self.require_torch = bool(require_torch)
        self._handles: list[tuple[Any, Any]] = []

    def _new(self, family: Mapping[str, Any], candidate_index: int, candidate_seed: int) -> tuple[Any, Any]:
        task_id, init_state = _family_ids(family)
        env = _invoke_factory(
            self.environment_factory,
            task_id=task_id,
            init_state=init_state,
            candidate_id=int(candidate_index),
            candidate_index=int(candidate_index),
            seed=int(candidate_seed),
            variant=None,
            family=family,
        )
        if env is None:
            raise S45AdapterError("LIBERO environment_factory returned None")
        prompt = libero_runtime.task_spec(task_id).prompt
        policy = _invoke_factory(
            self.policy_factory,
            prompt_override=prompt,
            task_id=task_id,
            init_state=init_state,
            candidate_id=int(candidate_index),
            candidate_index=int(candidate_index),
            seed=int(candidate_seed),
            family=family,
        )
        if policy is None:
            raise S45AdapterError("LIBERO policy_factory returned None")
        # The Stage-R environment seed is shared by all candidates in one
        # family. It is persisted by the N32 writer and is never recomputed
        # here from the observed outcomes.
        environment_seed = _metadata_value(family, "initial_seed", "environment_seed")
        if environment_seed is None:
            raise S45ProvenanceError(f"family {family.get('family_id')} lacks persisted initial/environment seed")
        try:
            libero_runtime.seeded_reset(env, init_state, int(environment_seed))
        except Exception as exc:  # noqa: BLE001 - preserve official runtime cause
            raise S45AdapterError(f"LIBERO official reset failed for {family.get('family_id')}: {exc}") from exc
        if not callable(getattr(env, "capture_snapshot", None)) or not callable(getattr(env, "restore_snapshot", None)):
            raise S45AdapterError("LIBERO environment lacks capture_snapshot/restore_snapshot")
        if not any(callable(getattr(env, name, None)) for name in ("capture_rng_state", "snapshot_rng", "get_rng_state")):
            raise S45AdapterError("LIBERO environment lacks an owner-specific RNG capture hook")
        if not any(callable(getattr(env, name, None)) for name in ("restore_rng_state", "restore_rng", "set_rng_state")):
            raise S45AdapterError("LIBERO environment lacks an owner-specific RNG restore hook")
        # Stage-R uses env._observation as the policy's observation-history
        # buffer. If that maintained contract disappears, stop rather than
        # treating an empty history as an equivalent state.
        if not hasattr(env, "_observation") and not hasattr(env, "observation_history"):
            raise S45AdapterError("LIBERO policy observation-history buffer is not exposed by the official environment")
        self._handles.append((env, policy))
        return env, policy

    @staticmethod
    def _close_pair(pair: tuple[Any, Any]) -> None:
        env, policy = pair
        for owner in (policy, env):
            close = getattr(owner, "close", None)
            if callable(close):
                close()

    def close(self) -> None:
        while self._handles:
            self._close_pair(self._handles.pop())

    @staticmethod
    def _capture_policy_history(policy: Any, env: Any) -> Any:
        for owner in (policy, env):
            for name in ("capture_observation_history", "snapshot_observation_history", "get_observation_history_state"):
                fn = getattr(owner, name, None)
                if callable(fn):
                    return _copy(fn())
        # Task64Environment's maintained Stage-R contract stores the policy
        # observation history in ``_observation`` rather than exposing a
        # second policy-side accessor. Preserve that exact buffer instead of
        # treating an absent accessor as an empty history.
        if hasattr(env, "_observation"):
            return _copy(getattr(env, "_observation"))
        if hasattr(env, "observation_history"):
            return _copy(getattr(env, "observation_history"))
        raise S45AdapterError("LIBERO policy/environment exposes no observation history capture hook")

    @staticmethod
    def _restore_policy_history(policy: Any, env: Any, value: Any) -> None:
        for owner in (policy, env):
            for name in ("restore_observation_history", "restore_history", "set_observation_history_state"):
                fn = getattr(owner, name, None)
                if callable(fn):
                    fn(_copy(value))
                    return
        if hasattr(env, "_observation"):
            env._observation = _copy(value)
            return
        if hasattr(env, "observation_history"):
            env.observation_history = _copy(value)
            return
        raise S45AdapterError("LIBERO policy/environment exposes no observation history restore hook")

    @staticmethod
    def _capture_policy_queue(policy: Any) -> Any:
        for name in ("capture_action_queue", "snapshot_action_queue", "get_action_queue_state"):
            fn = getattr(policy, name, None)
            if callable(fn):
                return _copy(fn())
        # The maintained LIBERO runner owns this queue in its collector. An
        # adapter-local queue is therefore persisted by _LiberoSnapshot.
        return None

    @staticmethod
    def _restore_policy_queue(policy: Any, value: Any) -> None:
        if value is None:
            return
        for name in ("restore_action_queue", "restore_queue", "set_action_queue_state"):
            fn = getattr(policy, name, None)
            if callable(fn):
                fn(_copy(value))
                return
        raise S45AdapterError("LIBERO policy action queue was captured but cannot be restored")

    @staticmethod
    def _capture_owner_rng(owner: Any, label: str) -> Any:
        for name in ("capture_rng_state", "snapshot_rng", "get_rng_state"):
            fn = getattr(owner, name, None)
            if callable(fn):
                return _copy(fn())
        raise S45AdapterError(f"{label} owner-specific RNG capture hook is missing")

    @staticmethod
    def _restore_owner_rng(owner: Any, state: Any, label: str) -> None:
        for name in ("restore_rng_state", "restore_rng", "set_rng_state"):
            fn = getattr(owner, name, None)
            if callable(fn):
                fn(_copy(state))
                return
        raise S45AdapterError(f"{label} owner-specific RNG restore hook is missing")

    def _capture(self, env: Any, policy: Any, queue: Sequence[Any], seed: int, counter: int, step: int) -> _LiberoSnapshot:
        stage = libero_runtime.capture_stage_r_snapshot(
            env,
            [np.asarray(value, dtype=np.float32) for value in queue],
            int(seed),
            int(counter),
            int(step),
            policy=policy,
            require_full_rng=self.require_torch,
        )
        return _LiberoSnapshot(
            stage_r=stage,
            environment_rng=self._capture_owner_rng(env, "LIBERO environment"),
            policy_rng=self._capture_owner_rng(policy, "LIBERO policy"),
            policy_history=self._capture_policy_history(policy, env),
            policy_queue=self._capture_policy_queue(policy),
        )

    def _restore(self, env: Any, policy: Any, snapshot: _LiberoSnapshot) -> list[np.ndarray]:
        queue = libero_runtime.restore_stage_r_snapshot(env, snapshot.stage_r, policy=policy)
        self._restore_owner_rng(env, snapshot.environment_rng, "LIBERO environment")
        self._restore_owner_rng(policy, snapshot.policy_rng, "LIBERO policy")
        self._restore_policy_history(policy, env, snapshot.policy_history)
        self._restore_policy_queue(policy, snapshot.policy_queue)
        return queue

    @staticmethod
    def _snapshot_payload(snapshot: _LiberoSnapshot) -> dict[str, Any]:
        stage = snapshot.stage_r
        return {
            "simulator_state": _jsonable(stage.environment),
            "environment": _jsonable(stage.environment),
            "observation_history": _jsonable(snapshot.policy_history),
            "action_queue": _jsonable(stage.action_queue),
            "rng_state": {
                "python": _jsonable(stage.python_rng_state),
                "numpy": _jsonable(stage.numpy_rng_state),
                "torch": _jsonable(stage.torch_rng_state),
                "environment_owner": _jsonable(snapshot.environment_rng),
                "policy_owner": _jsonable(snapshot.policy_rng),
            },
            "step": int(stage.step),
            "baseline_noise_seed": int(stage.baseline_noise_seed),
            "baseline_noise_counter": int(stage.baseline_noise_counter),
        }

    def _state_vector(self, env: Any) -> np.ndarray:
        return libero_runtime._state_vector(env, libero_runtime._observation(env))

    @staticmethod
    def _workspace_pose(env: Any, observation: Any) -> list[float]:
        pose = np.asarray(libero_runtime._pose_vector(env, observation), dtype=np.float64).reshape(-1)
        if pose.shape != (6,) or not np.all(np.isfinite(pose)):
            raise S45AdapterError(
                f"LIBERO Stage-S workspace pose must be finite 6D, got {pose.shape}"
            )
        return pose.tolist()

    def _same_action_check(self, env: Any, policy: Any, snapshot: _LiberoSnapshot, action: Any) -> dict[str, Any]:
        action_array = np.asarray(action, dtype=np.float32)
        self._restore(env, policy, snapshot)
        libero_runtime._execute_one(env, action_array)
        first = self._state_vector(env)
        self._restore(env, policy, snapshot)
        libero_runtime._execute_one(env, action_array)
        second = self._state_vector(env)
        if first.shape != second.shape:
            error = float("inf")
        else:
            error = float(np.max(np.abs(first - second))) if first.size else 0.0
        if error > SNAPSHOT_REPLAY_TOLERANCE:
            raise S45AdapterError(f"LIBERO restore same-action next-state error {error} > 1e-9")
        return {"passed": True, "same_action": True, "max_abs_error": error, "tolerance": SNAPSHOT_REPLAY_TOLERANCE}

    def _next_chunk(self, policy: Any, observation: Any, seed: int, counter: int) -> np.ndarray:
        try:
            return libero_runtime._sample_chunk(policy, observation, int(seed), int(counter))
        except Exception as exc:  # noqa: BLE001 - preserve official inference cause
            raise S45AdapterError(f"LIBERO official policy action-chunk inference failed: {exc}") from exc

    def select_anchor(self, family: Mapping[str, Any], *, protocol: ProtocolAuthority) -> Mapping[str, Any]:
        # Anchor identity is supplied by the frozen protocol rule. Do not infer
        # it from a convenient source ordering when the authority asks for the
        # unsuccessful candidate rule.
        rule = str(protocol.s4["anchor_rule"]).strip()
        if rule not in {"candidate_index=0", "candidate_id=0", "first_candidate_in_source_order", "lowest numeric candidate index among unsuccessful base-N32 candidates"}:
            raise S45AdapterError(f"LIBERO adapter does not implement frozen anchor_rule: {rule}")
        candidates = family.get("candidates")
        if not isinstance(candidates, Sequence) or len(candidates) != 32:
            raise S45ProvenanceError("LIBERO family anchor selection requires all 32 source candidates")
        if rule == "lowest numeric candidate index among unsuccessful base-N32 candidates":
            unsuccessful = [row for row in candidates if not _candidate_success(row)]
            if not unsuccessful:
                raise S45ProvenanceError("LIBERO family has no unsuccessful N32 candidate for the frozen anchor rule")
            try:
                return min(unsuccessful, key=lambda row: int(row.get("candidate_index", -1)))
            except (TypeError, ValueError) as exc:
                raise S45ProvenanceError("LIBERO unsuccessful anchor candidate index is invalid") from exc
        return candidates[0]

    def replay_prefix(self, family: Mapping[str, Any], anchor: Mapping[str, Any], split_step: int, *, protocol: ProtocolAuthority) -> Mapping[str, Any]:
        split = _require_finite_int(split_step, "split_step")
        actions = _row_actions(anchor)
        if not (0 < split < len(actions) - 1):
            raise S45ProvenanceError("LIBERO prefix split must be an interior action step")
        candidate_seed = _row_seed(anchor)
        env, policy = self._new(family, int(anchor.get("candidate_index", 0)), candidate_seed)
        queue: list[np.ndarray] = []
        observation = libero_runtime._observation(env)
        forwards = 0
        trajectory: list[Any] = []
        for step in range(split):
            if not queue:
                queue.extend(self._next_chunk(policy, observation, candidate_seed, forwards))
                forwards += 1
            action = np.asarray(queue.pop(0), dtype=np.float32)
            if not _action_equal(action, actions[step]):
                raise S45ProvenanceError(f"LIBERO anchor policy replay diverged at prefix step {step}")
            result = libero_runtime._execute_one(env, action)
            observation = libero_runtime._observation(env)
            trajectory.append(self._workspace_pose(env, observation))
            if bool(result.get("done", False)) and step + 1 < split:
                raise S45ProvenanceError("LIBERO anchor terminated before frozen split step")
        snapshot = self._capture(env, policy, queue, candidate_seed, forwards, split)
        self._same_action_check(env, policy, snapshot, actions[split] if split < len(actions) else actions[0])
        return {
            "snapshot_handle": snapshot,
            "environment_handle": env,
            "policy_handle": policy,
            "actions": _jsonable(actions[:split]),
            "trajectory": _jsonable(trajectory),
            "snapshot": self._snapshot_payload(snapshot),
            "policy_forwards": int(forwards),
            "env_steps": int(split),
            "candidate_seed": int(candidate_seed),
            "observation_history_source": "official_environment._observation_or_observation_history",
        }

    def _run_suffix(self, env: Any, policy: Any, prefix_actions: list[Any], prefix_trajectory: list[Any], *, seed: int, forwards: int, max_steps: int, snapshot: _LiberoSnapshot, branch: bool) -> dict[str, Any]:
        self._restore(env, policy, snapshot)
        # A split replaces the suffix, including any action chunk remaining at
        # the split. The original queue remains in the persisted snapshot;
        # clearing it here is an explicit suffix-only operation.
        queue: list[np.ndarray] = []
        observation = libero_runtime._observation(env)
        actions = [copy.deepcopy(value) for value in prefix_actions]
        trajectory = [copy.deepcopy(value) for value in prefix_trajectory]
        success = False
        done = False
        local_forwards = int(forwards)
        first_action: Any = None
        for _ in range(int(max_steps)):
            if not queue:
                queue.extend(self._next_chunk(policy, observation, seed, local_forwards))
                local_forwards += 1
            action = np.asarray(queue.pop(0), dtype=np.float32)
            if first_action is None:
                first_action = action.copy()
            result = libero_runtime._execute_one(env, action)
            observation = libero_runtime._observation(env)
            actions.append(action.tolist())
            trajectory.append(self._workspace_pose(env, observation))
            success = bool(result.get("success", False))
            done = bool(result.get("done", False))
            if done:
                break
        if not done:
            raise S45AdapterError("LIBERO branch did not reach official termination within max_steps")
        if first_action is None:
            raise S45AdapterError("LIBERO branch produced no suffix action")
        replay = self._same_action_check(env, policy, snapshot, first_action)
        return {
            "actions": actions,
            "trajectory": trajectory,
            "terminated": True,
            "success": bool(success),
            "termination": "official_eval_success_or_step_limit",
            "policy_forwards": int(local_forwards),
            "env_steps": int(len(actions)),
            "snapshot_restore_check": replay,
            "suffix_queue_replaced": bool(branch),
        }

    def branch_seed(self, family: Mapping[str, Any], anchor: Mapping[str, Any], split_step: int, branch_index: int, mode: str, *, protocol: ProtocolAuthority) -> int:
        # Seed derivation is owned by the frozen protocol. A table is useful
        # for per-family grids; the restricted hash grammar is useful for a
        # compact formula. Neither path introduces an adapter-side salt.
        family_id = str(family.get("family_id"))
        key_options = (
            f"{family_id}|{mode}|{int(branch_index)}",
            f"{mode}:{int(branch_index)}",
            str(int(branch_index)),
        )
        task_id, init_state = _family_ids(family)
        context = {
            "protocol_id": protocol.protocol_id,
            "substrate": self.substrate,
            "family_id": family_id,
            "task_id": task_id,
            "init_state_id": init_state,
            "anchor_candidate_id": str(anchor.get("candidate_id")),
            "anchor_candidate_seed": _row_seed(anchor),
            "split_step": int(split_step),
            "branch_index": int(branch_index),
            "mode": str(mode),
        }
        key_options = (
            f"{family_id}|{mode}|{int(split_step)}|{int(branch_index)}",
            f"{family_id}|{int(split_step)}|{mode}|{int(branch_index)}",
            *key_options,
        )
        return _frozen_seed(protocol.s4["branch_seed_formula"], protocol.s4.get("branch_seeds"), context, label="branch seed", explicit_keys=key_options)

    def run_branch(self, family: Mapping[str, Any], anchor: Mapping[str, Any], prefix: Mapping[str, Any], split_step: int, branch_seed: int, branch_index: int, mode: str, *, protocol: ProtocolAuthority) -> Mapping[str, Any]:
        snapshot = prefix.get("snapshot_handle")
        env = prefix.get("environment_handle")
        policy = prefix.get("policy_handle")
        if not isinstance(snapshot, _LiberoSnapshot) or env is None or policy is None:
            raise S45AdapterError("LIBERO branch requires live replay snapshot/environment/policy handles")
        actions = _row_actions(anchor)
        split = _require_finite_int(split_step, "split_step")
        if split >= len(actions) - 1:
            raise S45ProvenanceError("LIBERO branch split is not interior")
        prefix_actions = _row_actions({"actions": prefix.get("actions")})
        prefix_trajectory = _row_trajectory({"trajectory": prefix.get("trajectory")})
        return self._run_suffix(
            env,
            policy,
            prefix_actions,
            prefix_trajectory,
            seed=_require_finite_int(branch_seed, "branch_seed"),
            forwards=int(prefix.get("policy_forwards", snapshot.stage_r.baseline_noise_counter)),
            max_steps=self.max_steps,
            snapshot=snapshot,
            branch=True,
        )

    def extension_seed(self, family: Mapping[str, Any], candidate_index: int, *, protocol: ProtocolAuthority) -> int:
        family_id = str(family.get("family_id"))
        task_id, init_state = _family_ids(family)
        context = {
            "protocol_id": protocol.protocol_id,
            "substrate": self.substrate,
            "family_id": family_id,
            "task_id": task_id,
            "init_state_id": init_state,
            "candidate_index": int(candidate_index),
            "base_candidate_count": 32,
        }
        return _frozen_seed(protocol.s5["extension_seed_formula"], protocol.s5.get("extension_seeds"), context, label="extension seed", explicit_keys=(f"{family_id}|{int(candidate_index)}", str(int(candidate_index))))

    def run_fresh_candidate(self, family: Mapping[str, Any], candidate_index: int, candidate_seed: int, *, protocol: ProtocolAuthority) -> Mapping[str, Any]:
        index = _require_finite_int(candidate_index, "candidate_index")
        if not 32 <= index <= 63:
            raise S45ProvenanceError("LIBERO S5 fresh candidate index must be 32..63")
        env, policy = self._new(family, index, int(candidate_seed))
        snapshot = self._capture(env, policy, [], int(candidate_seed), 0, 0)
        observation = libero_runtime._observation(env)
        chunk = self._next_chunk(policy, observation, int(candidate_seed), 0)
        first_action = np.asarray(chunk[0], dtype=np.float32)
        replay = self._same_action_check(env, policy, snapshot, first_action)
        # Restore after the preflight so the recorded rollout starts exactly at
        # the official initial state.
        self._restore(env, policy, snapshot)
        return self._run_suffix(
            env,
            policy,
            [],
            [],
            seed=int(candidate_seed),
            forwards=0,
            max_steps=self.max_steps,
            snapshot=snapshot,
            branch=False,
        ) | {"snapshot_restore_check": replay}


class LiberoS45Adapter(_LiberoS45Base):
    """Named LIBERO adapter for B/C (variant selection is in the factory)."""


@dataclass
class _RoboTwinHandle:
    task_env: Any
    policy: Any
    runtime: robotwin_runtime.ConcreteRoboTwinRuntime


class RoboTwinS45Adapter(S45Adapter):
    """Bind S4/S5 to official RoboTwin SAPIEN plus Evo policy objects."""

    def __init__(
        self,
        environment_factory: Callable[..., Any],
        policy_factory: Callable[..., Any],
        reset_factory: Callable[..., Any],
        *,
        protocol: ProtocolAuthority,
        max_steps: int,
        require_torch: bool = True,
    ) -> None:
        if not callable(environment_factory) or not callable(policy_factory) or not callable(reset_factory):
            raise S45AdapterError("RoboTwin environment/policy/reset factories are all required callables")
        if int(max_steps) <= 0:
            raise S45AdapterError("RoboTwin max_steps must be positive")
        self.environment_factory = environment_factory
        self.policy_factory = policy_factory
        self.reset_factory = reset_factory
        self.protocol = protocol
        self.max_steps = int(max_steps)
        self.require_torch = bool(require_torch)
        self._handles: list[_RoboTwinHandle] = []

    def _new(self, family: Mapping[str, Any], candidate_index: int, candidate_seed: int) -> _RoboTwinHandle:
        task_id, init_state = _family_ids(family)
        task_name = _metadata_value(family, "task_name", "task")
        if task_name is None:
            raise S45ProvenanceError(f"RoboTwin family {family.get('family_id')} lacks task name")
        env = _invoke_factory(self.environment_factory, task_id=task_id, init_state=init_state, candidate_id=int(candidate_index), candidate_index=int(candidate_index), seed=int(candidate_seed), task_name=str(task_name), family=family)
        policy = _invoke_factory(self.policy_factory, task_id=task_id, init_state=init_state, candidate_id=int(candidate_index), candidate_index=int(candidate_index), seed=int(candidate_seed), task_name=str(task_name), family=family)
        if env is None or policy is None:
            raise S45AdapterError("RoboTwin official environment/policy factory returned None")
        reset = _invoke_factory(self.reset_factory, environment=env, task_env=env, task_id=task_id, init_state=init_state, seed=int(_metadata_value(family, "initial_seed", "environment_seed") or candidate_seed), task_name=str(task_name), family=family)
        # A reset factory may return the initial observation, or perform the
        # reset in-place and return None. Both are official callback results.
        del reset
        runtime = robotwin_runtime.ConcreteRoboTwinRuntime(env, policy, require_torch=self.require_torch)
        if not callable(getattr(env, "get_obs", None)) or not callable(getattr(env, "take_action", None)):
            raise S45AdapterError("RoboTwin official env must expose get_obs/take_action")
        if not hasattr(env, "eval_success") or not hasattr(env, "step_lim") or not hasattr(env, "take_action_cnt"):
            raise S45AdapterError("RoboTwin official env lacks eval_success/step_lim/take_action_cnt termination state")
        # ConcreteRoboTwinRuntime.capture_snapshot itself checks scene actors,
        # policy history/queue and all RNG streams. Probe it before execution.
        runtime.capture_snapshot()
        handle = _RoboTwinHandle(env, policy, runtime)
        self._handles.append(handle)
        return handle

    @staticmethod
    def _close_handle(handle: _RoboTwinHandle) -> None:
        for owner in (handle.policy, handle.task_env):
            close = getattr(owner, "close", None)
            if callable(close):
                close()

    def close(self) -> None:
        while self._handles:
            self._close_handle(self._handles.pop())

    @staticmethod
    def _set_policy_seed(policy: Any, seed: int) -> None:
        for name in ("set_seed", "seed"):
            fn = getattr(policy, name, None)
            if callable(fn):
                fn(int(seed))
                return
        fn = getattr(policy, "set_rng", None)
        if callable(fn):
            fn(np.random.default_rng(int(seed)))
            return
        raise S45AdapterError("RoboTwin policy lacks explicit seed/RNG setter")

    @staticmethod
    def _policy_action(policy: Any, observation: Any) -> Any:
        fn = getattr(policy, "act", None)
        if not callable(fn):
            raise S45AdapterError("RoboTwin official policy must expose act(observation)")
        return fn(observation)

    @staticmethod
    def _pose(observation: Any) -> Any:
        pose = np.asarray(robotwin_runtime._pose(observation), dtype=np.float64).reshape(-1)
        if pose.shape != (14,) or not np.all(np.isfinite(pose)):
            raise S45AdapterError(
                f"RoboTwin Stage-S workspace pose must be finite 14D, got {pose.shape}"
            )
        return _jsonable(pose)

    @staticmethod
    def _snapshot_payload(snapshot: Any) -> Mapping[str, Any]:
        if not isinstance(snapshot, robotwin_runtime.ConcreteReplaySnapshot):
            raise S45AdapterError("RoboTwin snapshot has an unexpected type")
        return {
            "simulator_state": _jsonable(snapshot.simulator),
            "observation_history": _jsonable(snapshot.policy_history),
            "action_queue": _jsonable(snapshot.action_queue),
            "rng_state": _jsonable(snapshot.rng_streams),
        }

    def _same_action_check(self, handle: _RoboTwinHandle, snapshot: Any, action: Any) -> Mapping[str, Any]:
        handle.runtime.restore_snapshot(snapshot)
        result = handle.runtime.verify_restore(action=_copy(action))
        if float(result.get("next_state_error", float("inf"))) > SNAPSHOT_REPLAY_TOLERANCE:
            raise S45AdapterError("RoboTwin restore same-action next-state error exceeds 1e-9")
        return {
            "passed": True,
            "same_action": True,
            "max_abs_error": float(result.get("next_state_error", 0.0)),
            "action_error": float(result.get("action_error", 0.0)),
            "tolerance": SNAPSHOT_REPLAY_TOLERANCE,
        }

    def select_anchor(self, family: Mapping[str, Any], *, protocol: ProtocolAuthority) -> Mapping[str, Any]:
        rule = str(protocol.s4["anchor_rule"]).strip()
        if rule not in {"candidate_index=0", "candidate_id=0", "first_candidate_in_source_order", "lowest numeric candidate index among unsuccessful base-N32 candidates"}:
            raise S45AdapterError(f"RoboTwin adapter does not implement frozen anchor_rule: {rule}")
        candidates = family.get("candidates")
        if not isinstance(candidates, Sequence) or len(candidates) != 32:
            raise S45ProvenanceError("RoboTwin family anchor selection requires all 32 source candidates")
        if rule == "lowest numeric candidate index among unsuccessful base-N32 candidates":
            unsuccessful = [row for row in candidates if not _candidate_success(row)]
            if not unsuccessful:
                raise S45ProvenanceError("RoboTwin family has no unsuccessful N32 candidate for the frozen anchor rule")
            try:
                return min(unsuccessful, key=lambda row: int(row.get("candidate_index", -1)))
            except (TypeError, ValueError) as exc:
                raise S45ProvenanceError("RoboTwin unsuccessful anchor candidate index is invalid") from exc
        return candidates[0]

    def replay_prefix(self, family: Mapping[str, Any], anchor: Mapping[str, Any], split_step: int, *, protocol: ProtocolAuthority) -> Mapping[str, Any]:
        split = _require_finite_int(split_step, "split_step")
        actions = _row_actions(anchor)
        if not (0 < split < len(actions) - 1):
            raise S45ProvenanceError("RoboTwin prefix split must be interior")
        seed = _row_seed(anchor)
        handle = self._new(family, int(anchor.get("candidate_index", 0)), seed)
        self._set_policy_seed(handle.policy, seed)
        # Capture after seeding so the anchor snapshot represents the same
        # candidate RNG stream that produced the accepted N32 actions.
        snapshot = handle.runtime.capture_snapshot()
        if actions:
            self._same_action_check(handle, snapshot, actions[0])
        observation = handle.task_env.get_obs()
        recorded_actions: list[Any] = []
        trajectory: list[Any] = []
        for step in range(split):
            action = self._policy_action(handle.policy, observation)
            if not _action_equal(action, actions[step]):
                raise S45ProvenanceError(f"RoboTwin anchor policy replay diverged at prefix step {step}")
            handle.task_env.take_action(action)
            observation = handle.task_env.get_obs()
            recorded_actions.append(_jsonable(action))
            trajectory.append(self._pose(observation))
            if bool(getattr(handle.task_env, "eval_success", False)) and step + 1 < split:
                raise S45ProvenanceError("RoboTwin anchor terminated before frozen split step")
        snapshot = handle.runtime.capture_snapshot()
        return {
            "snapshot_handle": snapshot,
            "environment_handle": handle,
            "actions": recorded_actions,
            "trajectory": trajectory,
            "snapshot": self._snapshot_payload(snapshot),
            "policy_forwards": int(split),
            "env_steps": int(split),
            "candidate_seed": int(seed),
        }

    def branch_seed(self, family: Mapping[str, Any], anchor: Mapping[str, Any], split_step: int, branch_index: int, mode: str, *, protocol: ProtocolAuthority) -> int:
        family_id = str(family.get("family_id"))
        task_id, init_state = _family_ids(family)
        context = {
            "protocol_id": protocol.protocol_id,
            "substrate": "A",
            "family_id": family_id,
            "task_id": task_id,
            "init_state_id": init_state,
            "anchor_candidate_id": str(anchor.get("candidate_id")),
            "anchor_candidate_seed": _row_seed(anchor),
            "split_step": int(split_step),
            "branch_index": int(branch_index),
            "mode": str(mode),
        }
        key_options = (
            f"{family_id}|{mode}|{int(split_step)}|{int(branch_index)}",
            f"{family_id}|{int(split_step)}|{mode}|{int(branch_index)}",
            f"{family_id}|{mode}|{int(branch_index)}",
            f"{mode}:{int(branch_index)}",
            str(int(branch_index)),
        )
        return _frozen_seed(protocol.s4["branch_seed_formula"], protocol.s4.get("branch_seeds"), context, label="branch seed", explicit_keys=key_options)

    def _run_rollout(self, handle: _RoboTwinHandle, *, seed: int, prefix_actions: Sequence[Any], prefix_trajectory: Sequence[Any], snapshot: Any, max_steps: int, branch: bool, forwards: int = 0) -> Mapping[str, Any]:
        handle.runtime.restore_snapshot(snapshot)
        if branch:
            # The restored queue is retained in the snapshot evidence, then
            # explicitly replaced because the split controls the suffix.
            restore_queue = getattr(handle.policy, "restore_action_queue", None)
            if not callable(restore_queue):
                raise S45AdapterError("RoboTwin branch cannot replace suffix: policy action queue restore hook missing")
            restore_queue([])
            self._set_policy_seed(handle.policy, seed)
        else:
            self._set_policy_seed(handle.policy, seed)
        observation = handle.task_env.get_obs()
        actions = [copy.deepcopy(value) for value in prefix_actions]
        trajectory = [copy.deepcopy(value) for value in prefix_trajectory]
        success = False
        done = bool(getattr(handle.task_env, "eval_success", False))
        try:
            forwards = int(forwards)
        except (TypeError, ValueError) as exc:
            raise S45ProvenanceError("RoboTwin policy forward count is invalid") from exc
        if forwards < 0:
            raise S45ProvenanceError("RoboTwin policy forward count is negative")
        first_action: Any = None
        for _ in range(int(max_steps)):
            if done:
                break
            action = self._policy_action(handle.policy, observation)
            if first_action is None:
                first_action = _copy(action)
            handle.task_env.take_action(action)
            observation = handle.task_env.get_obs()
            actions.append(_jsonable(action))
            trajectory.append(self._pose(observation))
            forwards += 1
            success = bool(getattr(handle.task_env, "eval_success", False))
            count = int(getattr(handle.task_env, "take_action_cnt"))
            limit = int(getattr(handle.task_env, "step_lim"))
            done = success or count >= limit
        if not done:
            raise S45AdapterError("RoboTwin branch did not reach official termination within max_steps")
        if first_action is None:
            raise S45AdapterError("RoboTwin branch produced no suffix action")
        replay = self._same_action_check(handle, snapshot, first_action)
        return {
            "actions": actions,
            "trajectory": trajectory,
            "terminated": True,
            "success": bool(success),
            "termination": "official_eval_success_or_step_limit",
            "policy_forwards": int(forwards),
            "env_steps": int(len(actions)),
            "snapshot_restore_check": replay,
            "suffix_queue_replaced": bool(branch),
        }

    def run_branch(self, family: Mapping[str, Any], anchor: Mapping[str, Any], prefix: Mapping[str, Any], split_step: int, branch_seed: int, branch_index: int, mode: str, *, protocol: ProtocolAuthority) -> Mapping[str, Any]:
        handle = prefix.get("environment_handle")
        snapshot = prefix.get("snapshot_handle")
        if not isinstance(handle, _RoboTwinHandle) or not isinstance(snapshot, robotwin_runtime.ConcreteReplaySnapshot):
            raise S45AdapterError("RoboTwin branch requires live runtime and replay snapshot handles")
        actions = _row_actions(anchor)
        split = _require_finite_int(split_step, "split_step")
        if split >= len(actions) - 1:
            raise S45ProvenanceError("RoboTwin branch split is not interior")
        return self._run_rollout(
            handle,
            seed=_require_finite_int(branch_seed, "branch_seed"),
            prefix_actions=_row_actions({"actions": prefix.get("actions")}),
            prefix_trajectory=_row_trajectory({"trajectory": prefix.get("trajectory")}),
            snapshot=snapshot,
            max_steps=self.max_steps,
            branch=True,
            forwards=int(prefix.get("policy_forwards", split)),
        )

    def extension_seed(self, family: Mapping[str, Any], candidate_index: int, *, protocol: ProtocolAuthority) -> int:
        family_id = str(family.get("family_id"))
        task_id, init_state = _family_ids(family)
        context = {
            "protocol_id": protocol.protocol_id,
            "substrate": "A",
            "family_id": family_id,
            "task_id": task_id,
            "init_state_id": init_state,
            "candidate_index": int(candidate_index),
            "base_candidate_count": 32,
        }
        return _frozen_seed(protocol.s5["extension_seed_formula"], protocol.s5.get("extension_seeds"), context, label="extension seed", explicit_keys=(f"{family_id}|{int(candidate_index)}", str(int(candidate_index))))

    def run_fresh_candidate(self, family: Mapping[str, Any], candidate_index: int, candidate_seed: int, *, protocol: ProtocolAuthority) -> Mapping[str, Any]:
        index = _require_finite_int(candidate_index, "candidate_index")
        if not 32 <= index <= 63:
            raise S45ProvenanceError("RoboTwin S5 fresh candidate index must be 32..63")
        handle = self._new(family, index, int(candidate_seed))
        self._set_policy_seed(handle.policy, int(candidate_seed))
        snapshot = handle.runtime.capture_snapshot()
        observation = handle.task_env.get_obs()
        first_action = self._policy_action(handle.policy, observation)
        replay = self._same_action_check(handle, snapshot, first_action)
        result = self._run_rollout(handle, seed=int(candidate_seed), prefix_actions=[], prefix_trajectory=[], snapshot=snapshot, max_steps=self.max_steps, branch=False)
        return {**dict(result), "snapshot_restore_check": replay}


def make_libero_s45_adapter(
    environment_factory: Callable[..., Any],
    policy_factory: Callable[..., Any],
    *,
    protocol: ProtocolAuthority,
    substrate: str,
    max_steps: int,
    require_torch: bool = True,
) -> LiberoS45Adapter:
    """Build a real B/C adapter from explicit maintained Stage-R factories."""

    return LiberoS45Adapter(
        environment_factory,
        policy_factory,
        protocol=protocol,
        substrate=substrate,
        max_steps=max_steps,
        require_torch=require_torch,
    )


def make_robotwin_s45_adapter(
    environment_factory: Callable[..., Any],
    policy_factory: Callable[..., Any],
    reset_factory: Callable[..., Any],
    *,
    protocol: ProtocolAuthority,
    max_steps: int,
    require_torch: bool = True,
) -> RoboTwinS45Adapter:
    """Build a real A adapter from explicit official RoboTwin callbacks."""

    return RoboTwinS45Adapter(
        environment_factory,
        policy_factory,
        reset_factory,
        protocol=protocol,
        max_steps=max_steps,
        require_torch=require_torch,
    )


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise S45AdapterError(f"required adapter configuration environment variable is missing: {name}")
    return value


def _load_callable(spec: str, *, label: str) -> Callable[..., Any]:
    if ":" not in spec:
        raise S45AdapterError(f"{label} must use module:callable form")
    module_name, name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    value = getattr(module, name, None)
    if not callable(value):
        raise S45AdapterError(f"{label} is not callable: {spec}")
    return value


def build_adapter(*, protocol: ProtocolAuthority, substrate: str) -> S45Adapter:
    """Build a production adapter only from explicit environment settings.

    This entry point is suitable for the S4/S5 CLI.  B/C use the existing
    Stage-R LIBERO factories.  A requires three explicit module:callable
    RoboTwin callbacks because the official eval script's scene setup is
    deployment-specific; absence is a hard failure rather than a fallback.
    """

    max_steps = int(_required_env("R142_STAGE_S_MAX_STEPS"))
    if substrate in {"B", "C"}:
        qpilots_root = _required_env("R142_STAGE_S_QPILOTS_ROOT")
        libero_root = _required_env("R142_STAGE_S_LIBERO_ROOT")
        checkpoint = _required_env("R142_STAGE_S_CHECKPOINT")
        variant_root = os.environ.get("R142_STAGE_S_VARIANT_ROOT") if substrate == "B" else None
        config_root = os.environ.get("R142_STAGE_S_LIBERO_CONFIG_ROOT") if substrate == "C" else None
        if substrate == "B" and not variant_root:
            raise S45AdapterError("B adapter requires R142_STAGE_S_VARIANT_ROOT")
        if substrate == "C" and not config_root:
            raise S45AdapterError("C adapter requires R142_STAGE_S_LIBERO_CONFIG_ROOT")
        env_factory = libero_runtime.make_stage_r_task64_factory(
            qpilots_root,
            libero_root,
            checkpoint=checkpoint,
            variant_root=variant_root,
            libero_config_root=config_root,
            max_steps=max_steps,
            init_state_count=16,
        )
        policy_factory = libero_runtime.make_stage_r_policy_factory(qpilots_root, checkpoint)
        return make_libero_s45_adapter(env_factory, policy_factory, protocol=protocol, substrate=substrate, max_steps=max_steps, require_torch=True)
    if substrate == "A":
        env_factory = _load_callable(_required_env("R142_STAGE_S_ROBOTWIN_ENV_FACTORY"), label="RoboTwin environment factory")
        policy_factory = _load_callable(_required_env("R142_STAGE_S_ROBOTWIN_POLICY_FACTORY"), label="RoboTwin policy factory")
        reset_factory = _load_callable(_required_env("R142_STAGE_S_ROBOTWIN_RESET_FACTORY"), label="RoboTwin reset factory")
        return make_robotwin_s45_adapter(env_factory, policy_factory, reset_factory, protocol=protocol, max_steps=max_steps, require_torch=True)
    raise S45AdapterError(f"unsupported Stage-S substrate: {substrate}")


__all__ = [
    "LiberoS45Adapter",
    "RoboTwinS45Adapter",
    "S45AdapterError",
    "build_adapter",
    "make_libero_s45_adapter",
    "make_robotwin_s45_adapter",
]
