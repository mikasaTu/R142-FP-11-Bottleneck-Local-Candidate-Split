from __future__ import annotations

"""Stage-R Phase-1R collection primitives.

The collector is deliberately adapter based: the frozen LIBERO adapter uses the
existing ``Task64Environment`` and ``CleanPi05LiberoPolicy`` objects, while the
tests and the control diagnostic use small in-process adapters.  All persistent
units are one complete ``(task, episode, location, stream)`` cell; a cell is
reused only after both data and metadata hashes validate.
"""

import copy
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .phase0 import load_task_rollouts
from .protocol import atomic_json, sha256_file
from .runtime import (
    configure_external_sources,
    infer_microbatched,
    infer_physical_many,
    policy_noise,
    shared_environment_seed,
    task_config,
)


PHASE1_PROTOCOL_ID = "r142-stage-r-phase1r-human-override-v1"
PHASE1_SCHEMA_VERSION = 1
STREAMS = ("calibration", "heldout")
SUITE_ORDER = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
TASKS = tuple((suite, task_id) for suite in SUITE_ORDER for task_id in range(10))
DECILES = tuple(i / 10.0 for i in range(1, 11))
DESCENDANTS_PER_CELL = 16
LOCATIONS_PER_EPISODE = 10
BRANCH_ACTIONS = 5
ACTION_REPLAY_RTOL = 1e-5
ACTION_REPLAY_ATOL = 1e-5


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def sha256_int(text: str, *, bytes_count: int = 8) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:bytes_count], "big", signed=False)


def episode_key(suite: str, task_id: int, init_state: int, candidate_id: int) -> str:
    return f"{suite}_task{int(task_id):02d}_init{int(init_state):03d}_candidate{int(candidate_id):02d}"


def selection_rank(
    suite: str,
    task_id: int,
    init_state: int,
    candidate_id: int,
    rollout_seed: int,
) -> str:
    """Return the frozen selection key, including the Phase-0 rollout seed."""

    # This is a deliberately literal seven-field contract.  In particular,
    # do not insert the human-readable ``episode_key`` as an extra field: the
    # frozen protocol is
    # ``protocol_id|episode|suite|task_id|init_state|candidate_id|rollout_seed``.
    key = "|".join(
        (
            PHASE1_PROTOCOL_ID,
            "episode",
            suite,
            str(int(task_id)),
            str(int(init_state)),
            str(int(candidate_id)),
            str(int(rollout_seed)),
        )
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def classify_stratum(row: Mapping[str, Any]) -> str:
    success = bool(row["success"])
    progress = float(row.get("final_progress", 0.0))
    if success:
        return "successful"
    if progress == 0.0:
        return "failed"
    if progress > 0.0:
        return "marginal"
    raise ValueError(f"invalid final_progress={progress!r}")


def select_phase1_episodes(
    raw_dir: str | Path,
    suite: str,
    task_id: int,
    *,
    requested_per_stratum: int = 4,
    count: int = 12,
) -> dict[str, Any]:
    """Select 12 unique Phase-0 rollouts without inspecting descendants.

    The first pass takes at most four hash-ranked episodes per requested
    stratum.  Any shortfall is filled from all remaining rollouts under the
    same hash order.  The result includes the observed shortfall so that a
    ceiling task cannot be mistaken for a balanced sample.
    """

    if count != 12 or requested_per_stratum != 4:
        raise ValueError("the human-override protocol fixes count=12 and 4/4/4 requests")
    # Selection needs only immutable outcome/identity arrays.  Avoid loading
    # the full action and pose tensors (the 40-task archive is hundreds of
    # MiB) so selection remains a cheap, outcome-blind pre-run operation.
    npz_path = Path(raw_dir) / f"{suite}_task{int(task_id):02d}.npz"
    with np.load(npz_path, allow_pickle=False) as data:
        lengths = np.asarray(data["lengths"])
        offsets = np.asarray(data["offsets"])
        rollouts = [
            {
                "success": bool(data["success"][index]),
                "final_progress": float(data["progress"][int(offsets[index + 1]) - 1]) if int(lengths[index]) else 0.0,
                "init_state": int(data["init_state"][index]),
                "candidate_id": int(data["candidate_id"][index]),
                "rollout_seed": int(data["rollout_seed"][index]),
            }
            for index in range(len(lengths))
        ]
    if len(rollouts) < count:
        raise ValueError(f"{suite} task {task_id} contains only {len(rollouts)} rollouts")
    decorated: list[dict[str, Any]] = []
    for row in rollouts:
        init_state = int(row["init_state"])
        candidate_id = int(row["candidate_id"])
        seed = int(row["rollout_seed"])
        decorated.append(
            {
                "suite": suite,
                "task_id": int(task_id),
                "init_state": init_state,
                "candidate_id": candidate_id,
                "rollout_seed": seed,
                "episode": episode_key(suite, task_id, init_state, candidate_id),
                "baseline_success": bool(row["success"]),
                "baseline_final_progress": float(row.get("final_progress", 0.0)),
                "stratum": classify_stratum(row),
                "selection_sha256": selection_rank(suite, task_id, init_state, candidate_id, seed),
            }
        )
    by_stratum: dict[str, list[dict[str, Any]]] = {name: [] for name in ("failed", "marginal", "successful")}
    for row in decorated:
        by_stratum[row["stratum"]].append(row)
    for values in by_stratum.values():
        values.sort(key=lambda row: (str(row["selection_sha256"]), str(row["episode"])))
    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    shortfalls: dict[str, int] = {}
    requested_counts: dict[str, int] = {}
    observed_counts: dict[str, int] = {}
    for name in ("failed", "marginal", "successful"):
        requested_counts[name] = requested_per_stratum
        observed_counts[name] = len(by_stratum[name])
        take = min(requested_per_stratum, len(by_stratum[name]))
        shortfalls[name] = requested_per_stratum - take
        for row in by_stratum[name][:take]:
            selected.append(dict(row, requested_stratum=name, selection_reason="requested_stratum"))
            selected_keys.add(row["episode"])
    remaining = sorted(
        (row for row in decorated if row["episode"] not in selected_keys),
        key=lambda row: (str(row["selection_sha256"]), str(row["episode"])),
    )
    fallback_count = count - len(selected)
    for row in remaining[:fallback_count]:
        selected.append(dict(row, requested_stratum=None, selection_reason="shortfall_sha_rank_fill"))
        selected_keys.add(row["episode"])
    if len(selected) != count or len(selected_keys) != count:
        raise RuntimeError("selection failed to produce 12 unique rollouts")
    # Selection order is stable by requested stratum followed by fallback; the
    # branch runner uses this order only for output naming, never for outcomes.
    for index, row in enumerate(selected):
        row["selection_index"] = int(index)
    return {
        "protocol_id": PHASE1_PROTOCOL_ID,
        "suite": suite,
        "task_id": int(task_id),
        "unit": "phase0r_rollout",
        "count": count,
        "requested_per_stratum": requested_counts,
        "observed_per_stratum": observed_counts,
        "shortfall_per_stratum": shortfalls,
        "selected": selected,
    }


def select_all_phase1_episodes(raw_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tasks = []
    for suite, task_id in TASKS:
        manifest = select_phase1_episodes(raw_dir, suite, task_id)
        manifest_path = output / f"{suite}_task{task_id:02d}.json"
        atomic_json(manifest_path, manifest)
        tasks.append((manifest, manifest_path))
    summary = {
        "protocol_id": PHASE1_PROTOCOL_ID,
        "marker_type": "selection",
        "task_count": len(tasks),
        "total_selected": sum(len(row["selected"]) for row, _ in tasks),
        "tasks": [
            {
                "suite": row["suite"],
                "task_id": row["task_id"],
                "path": path.name,
                "sha256": sha256_file(path),
            }
            for row, path in tasks
        ],
    }
    summary_path = output / "SELECTION_MANIFEST.json"
    atomic_json(summary_path, summary)
    completion_path = output / "COMPLETED_SELECTION.json"
    atomic_json(
        completion_path,
        {
            "schema_version": PHASE1_SCHEMA_VERSION,
            "protocol_id": PHASE1_PROTOCOL_ID,
            "marker_type": "selection",
            "summary": summary_path.name,
            "summary_sha256": sha256_file(summary_path),
            "task_count": len(tasks),
            "total_selected": summary["total_selected"],
            "checkpoint": "SELECTION_COMPLETE",
        },
    )
    _atomic_text(
        output / "SELECTION_SHA256SUMS",
        "".join(
            f"{sha256_file(path)}  {path.name}\n"
            for path in [path for _, path in tasks] + [summary_path, completion_path]
        ),
    )
    return summary


def decile_step(length: int, decile: float | int) -> int:
    """Map q in {0.1,...,1.0} to the frozen zero-based control step."""

    if int(length) <= 0:
        raise ValueError("length must be positive")
    q = float(decile)
    if not 0.0 < q <= 1.0:
        raise ValueError("decile must be in (0,1]")
    return max(0, min(int(length) - 1, int(math.ceil(q * int(length))) - 1))


def decile_steps(length: int) -> list[int]:
    return [decile_step(length, decile) for decile in DECILES]


@dataclass
class BranchSnapshot:
    environment: Any
    observation_history: Any
    action_queue: list[np.ndarray]
    python_rng_state: object
    numpy_rng_state: object
    baseline_noise_seed: int
    baseline_noise_counter: int
    step: int


@dataclass
class BranchTrace:
    branch_id: int
    parent_id: str
    generation_step: int
    branch_seed: int
    actions: np.ndarray
    progress: np.ndarray
    states: np.ndarray
    success: bool
    policy_forwards: int
    policy_batches: int
    environment_steps: int


def _observation(policy_or_env: Any, environment: Any) -> Any:
    if hasattr(environment, "raw_observation"):
        return environment.raw_observation()
    if hasattr(environment, "observation"):
        value = environment.observation
        return value() if callable(value) else value
    return None


def _state_vector(environment: Any, observation: Any = None) -> np.ndarray:
    for name in ("state_vector", "get_state_vector", "current_state_vector"):
        if hasattr(environment, name):
            value = getattr(environment, name)
            result = value() if callable(value) else value
            return np.asarray(result, dtype=np.float64).reshape(-1)
    if isinstance(observation, Mapping):
        pieces = []
        for key in sorted(observation):
            value = np.asarray(observation[key])
            if np.issubdtype(value.dtype, np.number):
                pieces.append(value.astype(np.float64).reshape(-1))
        if pieces:
            return np.concatenate(pieces)
    if observation is not None:
        return np.asarray(observation, dtype=np.float64).reshape(-1)
    return np.zeros(1, dtype=np.float64)


def _execute(environment: Any, action: np.ndarray) -> dict[str, Any]:
    """Call either Task64Environment's batched API or a test adapter."""

    try:
        result = environment.execute_actions(np.asarray(action, dtype=np.float32)[None, ...])
    except (TypeError, ValueError, IndexError):
        result = environment.execute_actions(np.asarray(action, dtype=np.float32))
    if result is None:
        result = {}
    if not isinstance(result, Mapping):
        result = {"done": bool(getattr(result, "done", False))}
    output = dict(result)
    if "done" not in output:
        output["done"] = bool(output.get("success", False))
    if "success" not in output:
        output["success"] = False
    if "progress" not in output:
        predicates = environment.official_predicates() if hasattr(environment, "official_predicates") else {}
        output["progress"] = float(predicates.get("fraction", 0.0)) if isinstance(predicates, Mapping) else 0.0
    return output


def _sample_chunk(policy: Any, observation: Any, seed: int, counter: int, *, action_count: int = BRANCH_ACTIONS) -> np.ndarray:
    if hasattr(policy, "sample_action_chunk"):
        chunk = policy.sample_action_chunk(observation, seed=int(seed), counter=int(counter))
        array = np.asarray(chunk, dtype=np.float32)
    elif hasattr(policy, "sample_action"):
        array = np.asarray(
            [policy.sample_action(observation, seed=int(seed), counter=int(counter)) for _ in range(action_count)],
            dtype=np.float32,
        )
    else:
        # The pinned CleanPi05 adapter exposes only the official model sampler.
        noise = policy_noise(policy, int(seed), int(counter))
        array = infer_physical_many(policy, [observation], np.asarray(noise)[None, ...])[0]
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[0] < action_count:
        raise ValueError(f"policy returned action chunk with shape {array.shape}")
    return np.asarray(array[:action_count], dtype=np.float32)


def _sample_chunks(
    policy: Any,
    observations: Sequence[Any],
    seeds: Sequence[int],
    counters: Sequence[int],
    *,
    action_count: int = BRANCH_ACTIONS,
    microbatch: int = 1,
) -> tuple[list[np.ndarray], int]:
    """Sample chunks for several branch states in one physical model call.

    The pinned CleanPi05 adapter exposes ``sample_model_actions_official``;
    ``infer_microbatched`` preserves its exact per-sample transform while
    padding only the final microbatch.  Small test/control policies continue
    through the explicit per-branch sampler.  The second return value is the
    number of physical policy batches, separate from logical forwards.
    """

    if not observations:
        return [], 0
    if hasattr(policy, "sample_model_actions_official"):
        noises = [policy_noise(policy, int(seed), int(counter)) for seed, counter in zip(seeds, counters, strict=True)]
        arrays = infer_microbatched(
            policy,
            list(observations),
            noises,
            microbatch=max(1, min(int(microbatch), len(observations))),
        )
        return [np.asarray(value[:action_count], dtype=np.float32) for value in arrays], (len(observations) + max(1, int(microbatch)) - 1) // max(1, int(microbatch))
    return [
        _sample_chunk(policy, observation, int(seed), int(counter), action_count=action_count)
        for observation, seed, counter in zip(observations, seeds, counters, strict=True)
    ], len(observations)


def _reset_seeded(environment: Any, suite: str, task_id: int, init_state: int) -> Any:
    common_seed = shared_environment_seed(suite, task_id, init_state)
    if hasattr(environment, "environment") and hasattr(environment.environment, "seed"):
        environment.environment.seed(int(common_seed))
    elif hasattr(environment, "seed"):
        environment.seed(int(common_seed))
    if hasattr(environment, "evaluation_seed"):
        environment.evaluation_seed = int(common_seed)
    return environment.reset(int(init_state))


def _capture_snapshot(environment: Any, queue: Sequence[np.ndarray], seed: int, counter: int, step: int) -> BranchSnapshot:
    if not hasattr(environment, "capture_snapshot"):
        raise RuntimeError("environment adapter must expose capture_snapshot for Phase-1R")
    observation_history = copy.deepcopy(getattr(environment, "_observation", None))
    return BranchSnapshot(
        environment=copy.deepcopy(environment.capture_snapshot()),
        observation_history=observation_history,
        action_queue=[np.asarray(value).copy() for value in queue],
        python_rng_state=copy.deepcopy(random.getstate()),
        numpy_rng_state=copy.deepcopy(np.random.get_state()),
        baseline_noise_seed=int(seed),
        baseline_noise_counter=int(counter),
        step=int(step),
    )


def _action_stream_sha256(actions: Sequence[np.ndarray]) -> str:
    """Hash the exact replayed action tensor with shape/dtype framing."""

    array = np.asarray(actions, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(repr(tuple(int(value) for value in array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _restore_snapshot(environment: Any, snapshot: BranchSnapshot) -> None:
    # A factory-created LIBERO environment starts with seed=0; restore the
    # snapshot seed first so the adapter seed contract remains exact.
    snapshot_seed = getattr(snapshot.environment, "evaluation_seed", None)
    if snapshot_seed is not None and hasattr(environment, "evaluation_seed"):
        environment.evaluation_seed = int(snapshot_seed)
    environment.restore_snapshot(copy.deepcopy(snapshot.environment))
    if snapshot.observation_history is not None and hasattr(environment, "_observation"):
        environment._observation = copy.deepcopy(snapshot.observation_history)
    random.setstate(copy.deepcopy(snapshot.python_rng_state))
    np.random.set_state(copy.deepcopy(snapshot.numpy_rng_state))


def replay_baseline_snapshot(
    environment: Any,
    policy: Any,
    suite: str,
    task_id: int,
    row: Mapping[str, Any],
    target_step: int,
) -> tuple[BranchSnapshot, dict[str, Any]]:
    """Replay the baseline and capture the complete snapshot before ``target_step``."""

    init_state = int(row["init_state"])
    rollout_seed = int(row["rollout_seed"])
    target = int(target_step)
    if target < 0:
        raise ValueError("target_step must be nonnegative")
    raw_actions = np.asarray(row.get("actions", []), dtype=np.float32)
    expected_length = int(len(raw_actions))
    if expected_length <= 0:
        raise RuntimeError("baseline rollout has no actions")
    if target >= expected_length:
        raise RuntimeError(f"target_step={target} is outside baseline length={expected_length}")
    observation = _reset_seeded(environment, suite, task_id, init_state)
    queue: list[np.ndarray] = []
    counter = 0
    step = 0
    done = False
    captured: BranchSnapshot | None = None
    replay_actions: list[np.ndarray] = []
    final_result: Mapping[str, Any] | None = None
    # Replay the complete baseline, rather than stopping immediately after the
    # requested snapshot.  This proves that the pinned environment/policy
    # lineage reproduces the Phase-0 source action stream and terminal label.
    while not done:
        if step >= expected_length:
            raise RuntimeError(
                f"baseline terminated after {step} actions but raw rollout has {expected_length}"
            )
        if not queue:
            queue.extend(_sample_chunk(policy, observation, rollout_seed, counter))
            counter += 1
        if step == target:
            captured = _capture_snapshot(environment, queue, rollout_seed, counter, step)
        action = np.asarray(queue.pop(0), dtype=np.float32)
        expected_action = np.asarray(raw_actions[step], dtype=np.float32)
        if action.shape != expected_action.shape or not np.allclose(
            action, expected_action, rtol=ACTION_REPLAY_RTOL, atol=ACTION_REPLAY_ATOL
        ):
            max_error = float(np.max(np.abs(action - expected_action))) if action.shape == expected_action.shape else float("inf")
            raise RuntimeError(
                f"baseline action mismatch at step={step}: shape={action.shape}/{expected_action.shape}, max_error={max_error}"
            )
        result = _execute(environment, action)
        replay_actions.append(action.copy())
        observation = _observation(policy, environment)
        final_result = result
        done = bool(result.get("done", False))
        step += 1
    if len(replay_actions) != expected_length:
        raise RuntimeError(f"baseline replay length={len(replay_actions)} != raw length={expected_length}")
    if captured is None:
        raise RuntimeError(f"failed to capture baseline snapshot at target_step={target}")
    if final_result is None:
        raise RuntimeError("baseline produced no terminal result")
    expected_success = bool(row.get("success", False))
    observed_success = bool(final_result.get("success", False))
    if observed_success != expected_success:
        raise RuntimeError(
            f"baseline terminal success mismatch: replay={observed_success} raw={expected_success}"
        )
    return captured, {
        "replay_steps": step,
        "replay_actions": len(replay_actions),
        "baseline_length": expected_length,
        "baseline_success": observed_success,
        "replay_action_sha256": _action_stream_sha256(replay_actions),
        "replay_action_shape": list(np.asarray(replay_actions, dtype=np.float32).shape),
        "action_max_abs_error": 0.0,
    }


def _branch_seed(stream: str, suite: str, task_id: int, parent_id: str, location_index: int, branch_id: int) -> int:
    return sha256_int(
        "|".join(
            (
                PHASE1_PROTOCOL_ID,
                "branch",
                stream,
                suite,
                str(int(task_id)),
                parent_id,
                str(int(location_index)),
                str(int(branch_id)),
            )
        )
    )


def replay_baseline_snapshots(
    environment: Any,
    policy: Any,
    suite: str,
    task_id: int,
    row: Mapping[str, Any],
    target_steps: Sequence[int],
) -> tuple[dict[int, BranchSnapshot], dict[str, Any]]:
    """Replay one baseline and capture all requested pre-action snapshots.

    A selected natural episode has ten frozen decile locations.  The source
    baseline is replayed once, while a snapshot is captured whenever the
    replay reaches a requested location.  This keeps baseline accounting
    exact and removes an otherwise silent ten-fold replay multiplier.
    """

    init_state = int(row["init_state"])
    rollout_seed = int(row["rollout_seed"])
    requested_steps = [int(value) for value in target_steps]
    if not requested_steps:
        raise ValueError("target_steps must be nonempty")
    if any(value < 0 for value in requested_steps):
        raise ValueError("target_steps must be nonnegative")
    raw_actions = np.asarray(row.get("actions", []), dtype=np.float32)
    expected_length = int(len(raw_actions))
    if expected_length <= 0:
        raise RuntimeError("baseline rollout has no actions")
    if any(value >= expected_length for value in requested_steps):
        raise RuntimeError(
            f"target_steps={requested_steps} contains a step outside baseline length={expected_length}"
        )
    target_set = set(requested_steps)
    observation = _reset_seeded(environment, suite, task_id, init_state)
    queue: list[np.ndarray] = []
    counter = 0
    step = 0
    done = False
    captured: dict[int, BranchSnapshot] = {}
    replay_actions: list[np.ndarray] = []
    final_result: Mapping[str, Any] | None = None
    action_max_abs_error = 0.0
    # Replay the complete baseline once.  Capturing multiple snapshots during
    # this pass preserves the exact source action stream and terminal label.
    while not done:
        if step >= expected_length:
            raise RuntimeError(
                f"baseline terminated after {step} actions but raw rollout has {expected_length}"
            )
        if not queue:
            queue.extend(_sample_chunk(policy, observation, rollout_seed, counter))
            counter += 1
        if step in target_set and step not in captured:
            captured[step] = _capture_snapshot(environment, queue, rollout_seed, counter, step)
        action = np.asarray(queue.pop(0), dtype=np.float32)
        expected_action = np.asarray(raw_actions[step], dtype=np.float32)
        if action.shape != expected_action.shape or not np.allclose(
            action, expected_action, rtol=ACTION_REPLAY_RTOL, atol=ACTION_REPLAY_ATOL
        ):
            max_error = float(np.max(np.abs(action - expected_action))) if action.shape == expected_action.shape else float("inf")
            raise RuntimeError(
                f"baseline action mismatch at step={step}: shape={action.shape}/{expected_action.shape}, max_error={max_error}"
            )
        if action.shape == expected_action.shape:
            action_max_abs_error = max(action_max_abs_error, float(np.max(np.abs(action - expected_action))))
        result = _execute(environment, action)
        replay_actions.append(action.copy())
        observation = _observation(policy, environment)
        final_result = result
        done = bool(result.get("done", False))
        step += 1
    if len(replay_actions) != expected_length:
        raise RuntimeError(f"baseline replay length={len(replay_actions)} != raw length={expected_length}")
    if set(captured) != target_set:
        raise RuntimeError(
            f"failed to capture baseline snapshots: captured={sorted(captured)} requested={sorted(target_set)}"
        )
    if final_result is None:
        raise RuntimeError("baseline produced no terminal result")
    expected_success = bool(row.get("success", False))
    observed_success = bool(final_result.get("success", False))
    if observed_success != expected_success:
        raise RuntimeError(
            f"baseline terminal success mismatch: replay={observed_success} raw={expected_success}"
        )
    return captured, {
        "replay_steps": step,
        "replay_actions": len(replay_actions),
        "baseline_length": expected_length,
        "baseline_success": observed_success,
        "expected_success": expected_success,
        "policy_forwards": counter,
        "policy_batches": counter,
        "target_steps": requested_steps,
        "captured_steps": sorted(captured),
        "replay_action_sha256": _action_stream_sha256(replay_actions),
        "replay_action_shape": list(np.asarray(replay_actions, dtype=np.float32).shape),
        "action_max_abs_error": action_max_abs_error,
    }
class BranchEnvironmentPool:
    """Factory-created simulator pool reused across natural cells.

    The pool is intentionally bounded by the configured model microbatch.  A
    cell with more descendants than the pool capacity is evaluated in waves;
    no live MuJoCo handle is ever deep-copied or shared by two descendants.
    """

    def __init__(
        self,
        environment_factory: Callable[[str, int, str], Any],
        suite: str,
        task_id: int,
        *,
        capacity: int,
        kind: str = "natural",
    ) -> None:
        if int(capacity) <= 0:
            raise ValueError("branch environment pool capacity must be positive")
        self.environment_factory = environment_factory
        self.suite = str(suite)
        self.task_id = int(task_id)
        self.kind = str(kind)
        self.capacity = int(capacity)
        self._environments: list[Any] = []
        self._closed = False

    def acquire(self, count: int) -> list[Any]:
        if self._closed:
            raise RuntimeError("branch environment pool is closed")
        requested = int(count)
        if not 0 < requested <= self.capacity:
            raise ValueError(f"requested {requested} environments exceeds pool capacity {self.capacity}")
        while len(self._environments) < requested:
            self._environments.append(
                self.environment_factory(self.suite, self.task_id, self.kind)
            )
        return self._environments[:requested]

    @property
    def created_count(self) -> int:
        return len(self._environments)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for environment in self._environments:
            if hasattr(environment, "close"):
                environment.close()
        self._environments.clear()

def _collect_branch_cell_batched(
    environment: Any,
    policy: Any,
    *,
    snapshot: BranchSnapshot,
    suite: str,
    task_id: int,
    parent_id: str,
    generation_step: int,
    stream: str,
    location_index: int,
    descendants: int,
    max_steps: int,
    microbatch: int,
    environment_pool: BranchEnvironmentPool | None = None,
) -> list[BranchTrace] | None:
    """Compatibility wrapper for the factory-backed batched collector."""

    if environment_pool is None:
        return None
    return _collect_branch_cell_batched_pool(
        environment,
        policy,
        snapshot=snapshot,
        suite=suite,
        task_id=task_id,
        parent_id=parent_id,
        generation_step=generation_step,
        stream=stream,
        location_index=location_index,
        descendants=descendants,
        max_steps=max_steps,
        microbatch=microbatch,
        environment_pool=environment_pool,
    )


def _collect_branch_wave(
    policy: Any,
    branch_envs: Sequence[Any],
    branch_ids: Sequence[int],
    *,
    suite: str,
    task_id: int,
    parent_id: str,
    generation_step: int,
    stream: str,
    location_index: int,
    max_steps: int,
    microbatch: int,
) -> tuple[list[BranchTrace], int]:
    seeds = [
        _branch_seed(stream, suite, task_id, parent_id, location_index, branch_id)
        for branch_id in branch_ids
    ]
    count = len(branch_ids)
    queues: list[list[np.ndarray]] = [[] for _ in range(count)]
    counters = [0 for _ in range(count)]
    observations = [_observation(policy, environment) for environment in branch_envs]
    actions: list[list[np.ndarray]] = [[] for _ in range(count)]
    progresses: list[list[float]] = [[] for _ in range(count)]
    states: list[list[np.ndarray]] = [[] for _ in range(count)]
    successes = [False for _ in range(count)]
    done = [False for _ in range(count)]
    forwards = [0 for _ in range(count)]
    policy_batches = 0
    for _ in range(int(max_steps)):
        if all(done):
            break
        needs_plan = [index for index in range(count) if not done[index] and not queues[index]]
        if needs_plan:
            chunks, batch_count = _sample_chunks(
                policy,
                [observations[index] for index in needs_plan],
                [seeds[index] for index in needs_plan],
                [counters[index] for index in needs_plan],
                microbatch=max(1, int(microbatch)),
            )
            policy_batches += int(batch_count)
            for local, index in enumerate(needs_plan):
                queues[index].extend(np.asarray(chunks[local], dtype=np.float32))
                counters[index] += 1
                forwards[index] += 1
        for index, environment in enumerate(branch_envs):
            if done[index]:
                continue
            if not queues[index]:
                raise RuntimeError(f"empty action queue for branch {branch_ids[index]}")
            action = np.asarray(queues[index].pop(0), dtype=np.float32)
            result = _execute(environment, action)
            observations[index] = _observation(policy, environment)
            actions[index].append(action.copy())
            progresses[index].append(float(result.get("progress", 0.0)))
            states[index].append(_state_vector(environment, observations[index]))
            successes[index] = bool(result.get("success", False))
            done[index] = bool(result.get("done", False))
    if not all(done):
        raise RuntimeError(f"branch exceeded max_steps={max_steps}")
    traces = [
        BranchTrace(
            branch_id=int(branch_id),
            parent_id=str(parent_id),
            generation_step=int(generation_step),
            branch_seed=int(seeds[index]),
            actions=np.asarray(actions[index], dtype=np.float32),
            progress=np.asarray(progresses[index], dtype=np.float32),
            states=np.asarray(states[index], dtype=np.float64),
            success=bool(successes[index]),
            policy_forwards=int(forwards[index]),
            policy_batches=int(forwards[index]),
            environment_steps=len(actions[index]),
        )
        for index, branch_id in enumerate(branch_ids)
    ]
    return traces, int(policy_batches)


def _collect_branch_cell_batched_pool(
    environment: Any,
    policy: Any,
    *,
    snapshot: BranchSnapshot,
    suite: str,
    task_id: int,
    parent_id: str,
    generation_step: int,
    stream: str,
    location_index: int,
    descendants: int,
    max_steps: int,
    microbatch: int,
    environment_pool: BranchEnvironmentPool,
) -> list[BranchTrace] | None:
    """Collect descendants in bounded waves using factory-created environments."""

    del environment  # retained for compatibility with the public API
    if not hasattr(policy, "sample_model_actions_official"):
        return None
    traces: list[BranchTrace] = []
    physical_batches = 0
    for start in range(0, int(descendants), int(environment_pool.capacity)):
        branch_ids = list(
            range(start, min(start + int(environment_pool.capacity), int(descendants)))
        )
        branch_envs = environment_pool.acquire(len(branch_ids))
        for branch_environment in branch_envs:
            _restore_snapshot(branch_environment, snapshot)
            if hasattr(branch_environment, "begin_branch"):
                branch_environment.begin_branch()
        wave, wave_batches = _collect_branch_wave(
            policy,
            branch_envs,
            branch_ids,
            suite=suite,
            task_id=task_id,
            parent_id=parent_id,
            generation_step=generation_step,
            stream=stream,
            location_index=location_index,
            max_steps=max_steps,
            microbatch=microbatch,
        )
        traces.extend(wave)
        physical_batches += int(wave_batches)
    if traces:
        setattr(traces[0], "physical_policy_batches", int(physical_batches))
    return traces
def collect_branch_cell(
    environment: Any,
    policy: Any,
    *,
    snapshot: BranchSnapshot,
    suite: str,
    task_id: int,
    parent_id: str,
    generation_step: int,
    stream: str,
    location_index: int,
    descendants: int = DESCENDANTS_PER_CELL,
    max_steps: int = 1000,
    microbatch: int = 1,
    environment_pool: BranchEnvironmentPool | None = None,
) -> list[BranchTrace]:
    if stream not in STREAMS:
        raise ValueError(f"unknown stream {stream!r}")
    if int(microbatch) > 1 and hasattr(policy, "sample_model_actions_official") and environment_pool is not None:
        batched = _collect_branch_cell_batched_pool(
            environment,
            policy,
            snapshot=snapshot,
            suite=suite,
            task_id=task_id,
            parent_id=parent_id,
            generation_step=generation_step,
            stream=stream,
            location_index=location_index,
            descendants=descendants,
            max_steps=max_steps,
            microbatch=int(microbatch),
            environment_pool=environment_pool,
        )
        if batched is not None:
            return batched
    traces = []
    for branch_id in range(int(descendants)):
        branch_seed = _branch_seed(stream, suite, task_id, parent_id, location_index, branch_id)
        _restore_snapshot(environment, snapshot)
        if hasattr(environment, "begin_branch"):
            environment.begin_branch()
        # The inherited action queue is intentionally discarded at the
        # intervention point. Every descendant starts a fresh independent
        # branch RNG stream and then follows the frozen policy.
        queue: list[np.ndarray] = []
        counter = 0
        observation = _observation(policy, environment)
        actions: list[np.ndarray] = []
        progress: list[float] = []
        states: list[np.ndarray] = []
        success = False
        done = False
        forwards = 0
        for _ in range(int(max_steps)):
            if not queue:
                queue.extend(_sample_chunk(policy, observation, branch_seed, counter))
                counter += 1
                forwards += 1
            action = np.asarray(queue.pop(0), dtype=np.float32)
            result = _execute(environment, action)
            observation = _observation(policy, environment)
            actions.append(action.copy())
            progress.append(float(result.get("progress", 0.0)))
            states.append(_state_vector(environment, observation))
            success = bool(result.get("success", False))
            done = bool(result.get("done", False))
            if done:
                break
        if not done:
            raise RuntimeError(f"branch exceeded max_steps={max_steps}")
        traces.append(
            BranchTrace(
                branch_id=int(branch_id),
                parent_id=str(parent_id),
                generation_step=int(generation_step),
                branch_seed=int(branch_seed),
                actions=np.asarray(actions, dtype=np.float32),
                progress=np.asarray(progress, dtype=np.float32),
                states=np.asarray(states, dtype=np.float64),
                success=bool(success),
                policy_forwards=int(forwards),
                policy_batches=int(forwards),
                environment_steps=len(actions),
            )
        )
    return traces


def _atomic_npz(path: Path, payload: Mapping[str, np.ndarray]) -> None:
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


def _pack_traces(traces: Sequence[BranchTrace]) -> dict[str, np.ndarray]:
    if not traces:
        raise ValueError("cannot pack empty cell")
    lengths = np.asarray([len(trace.actions) for trace in traces], dtype=np.int32)
    offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    return {
        "lengths": lengths,
        "offsets": offsets,
        "actions": np.concatenate([trace.actions for trace in traces], axis=0).astype(np.float32),
        "states": np.concatenate([trace.states for trace in traces], axis=0).astype(np.float64),
        "progress": np.concatenate([trace.progress for trace in traces], axis=0).astype(np.float32),
        "success": np.asarray([trace.success for trace in traces], dtype=np.bool_),
        "branch_id": np.asarray([trace.branch_id for trace in traces], dtype=np.int16),
        "generation_step": np.asarray([trace.generation_step for trace in traces], dtype=np.int32),
        "branch_seed": np.asarray([trace.branch_seed for trace in traces], dtype=np.uint64),
        "policy_forwards": np.asarray([trace.policy_forwards for trace in traces], dtype=np.int32),
        "policy_batches": np.asarray([trace.policy_batches for trace in traces], dtype=np.int32),
        "environment_steps": np.asarray([trace.environment_steps for trace in traces], dtype=np.int32),
    }


def write_cell(
    directory: str | Path,
    traces: Sequence[BranchTrace],
    *,
    suite: str,
    task_id: int,
    parent_id: str,
    selection_index: int,
    location_index: int,
    decile: float,
    generation_step: int,
    stream: str,
    baseline_length: int,
    owner_uid: int | None = None,
    owner_gid: int | None = None,
    metadata_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    if len(traces) != DESCENDANTS_PER_CELL:
        raise ValueError(f"expected {DESCENDANTS_PER_CELL} descendants, got {len(traces)}")
    npz_path = root / "cell.npz"
    metadata_path = root / "metadata.json"
    marker_path = root / "COMPLETED_CELL.json"
    _atomic_npz(npz_path, _pack_traces(traces))
    stat = npz_path.stat()
    payload = {
        "schema_version": PHASE1_SCHEMA_VERSION,
        "protocol_id": PHASE1_PROTOCOL_ID,
        "suite": suite,
        "task_id": int(task_id),
        "parent_id": str(parent_id),
        "selection_index": int(selection_index),
        "location_index": int(location_index),
        "decile": float(decile),
        "generation_step": int(generation_step),
        "baseline_length": int(baseline_length),
        "stream": stream,
        "descendant_count": len(traces),
        "data_file": npz_path.name,
        "data_sha256": sha256_file(npz_path),
        "owner_uid": int(stat.st_uid if owner_uid is None else owner_uid),
        "owner_gid": int(stat.st_gid if owner_gid is None else owner_gid),
        "branch_ids": [int(trace.branch_id) for trace in traces],
        "parent_ids": sorted({str(trace.parent_id) for trace in traces}),
        "policy_forwards": int(sum(trace.policy_forwards for trace in traces)),
        "policy_batches": int(sum(trace.policy_batches for trace in traces)),
        "physical_policy_batches": int(getattr(traces[0], "physical_policy_batches", sum(trace.policy_batches for trace in traces))),
        "environment_steps": int(sum(trace.environment_steps for trace in traces)),
        "written_at_unix": time.time(),
    }
    if metadata_extra:
        payload.update(dict(metadata_extra))
    atomic_json(metadata_path, payload)
    marker = {
        "protocol_id": PHASE1_PROTOCOL_ID,
        "cell": {
            "suite": suite,
            "task_id": int(task_id),
            "parent_id": str(parent_id),
            "selection_index": int(selection_index),
            "location_index": int(location_index),
            "stream": stream,
        },
        "npz": {"path": npz_path.name, "sha256": payload["data_sha256"]},
        "metadata": {"path": metadata_path.name, "sha256": sha256_file(metadata_path)},
    }
    atomic_json(marker_path, marker)
    sums = root / "SHA256SUMS"
    _atomic_text(
        sums,
        f"{payload['data_sha256']}  {npz_path.name}\n{marker['metadata']['sha256']}  {metadata_path.name}\n",
    )
    return payload


def validate_cell(
    directory: str | Path,
    *,
    suite: str | None = None,
    task_id: int | None = None,
    parent_id: str | None = None,
    location_index: int | None = None,
    stream: str | None = None,
    descendants: int = DESCENDANTS_PER_CELL,
    require_owner: tuple[int, int] | None = (2254, 2254),
) -> tuple[bool, list[str], dict[str, Any] | None]:
    root = Path(directory)
    errors: list[str] = []
    npz_path = root / "cell.npz"
    metadata_path = root / "metadata.json"
    marker_path = root / "COMPLETED_CELL.json"
    sums_path = root / "SHA256SUMS"
    if not all(path.is_file() for path in (npz_path, metadata_path, marker_path, sums_path)):
        return False, ["missing cell artifact"], None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if metadata.get("protocol_id") != PHASE1_PROTOCOL_ID:
            errors.append("protocol mismatch")
        if metadata.get("schema_version") != PHASE1_SCHEMA_VERSION:
            errors.append("schema mismatch")
        if suite is not None and metadata.get("suite") != suite:
            errors.append("suite mismatch")
        if task_id is not None and int(metadata.get("task_id", -1)) != int(task_id):
            errors.append("task mismatch")
        if parent_id is not None and metadata.get("parent_id") != parent_id:
            errors.append("parent mismatch")
        if location_index is not None and int(metadata.get("location_index", -1)) != int(location_index):
            errors.append("location mismatch")
        if stream is not None and metadata.get("stream") != stream:
            errors.append("stream mismatch")
        if marker.get("protocol_id") != PHASE1_PROTOCOL_ID:
            errors.append("marker protocol mismatch")
        marker_cell = marker.get("cell", {})
        for name in ("suite", "task_id", "parent_id", "selection_index", "location_index", "stream"):
            if name in metadata and marker_cell.get(name) != metadata.get(name):
                errors.append(f"marker identity mismatch: {name}")
        if marker.get("npz", {}).get("path") != npz_path.name or marker.get("metadata", {}).get("path") != metadata_path.name:
            errors.append("marker file-name mismatch")
        actual_sha = sha256_file(npz_path)
        if metadata.get("data_sha256") != actual_sha:
            errors.append("data SHA mismatch")
        if marker.get("npz", {}).get("sha256") != actual_sha:
            errors.append("marker data SHA mismatch")
        if marker.get("metadata", {}).get("sha256") != sha256_file(metadata_path):
            errors.append("marker metadata SHA mismatch")
        if require_owner is not None:
            actual_owner = (int(metadata.get("owner_uid", -1)), int(metadata.get("owner_gid", -1)))
            if actual_owner != tuple(require_owner):
                errors.append(f"owner mismatch: {actual_owner} != {tuple(require_owner)}")
            for path in (npz_path, metadata_path, marker_path, sums_path):
                stat = path.stat()
                if (int(stat.st_uid), int(stat.st_gid)) != tuple(require_owner):
                    errors.append(f"filesystem owner mismatch: {path} {(stat.st_uid, stat.st_gid)} != {tuple(require_owner)}")
        with np.load(npz_path, allow_pickle=False) as data:
            required = {"lengths", "offsets", "actions", "states", "progress", "success", "branch_id", "generation_step", "branch_seed", "policy_forwards", "policy_batches", "environment_steps"}
            missing = required.difference(data.files)
            if missing:
                errors.append(f"missing arrays: {sorted(missing)}")
            else:
                lengths = np.asarray(data["lengths"])
                offsets = np.asarray(data["offsets"])
                count = len(lengths)
                total = int(offsets[-1]) if len(offsets) else -1
                if count != int(descendants):
                    errors.append(f"descendant count {count} != {descendants}")
                if offsets.shape != (count + 1,) or int(offsets[0]) != 0 or not np.array_equal(np.diff(offsets), lengths):
                    errors.append("invalid offsets")
                for name in ("actions", "states", "progress"):
                    if len(data[name]) != total:
                        errors.append(f"invalid {name} total length")
                for name in ("success", "branch_id", "generation_step", "branch_seed", "policy_forwards", "policy_batches", "environment_steps"):
                    if len(data[name]) != count:
                        errors.append(f"invalid {name} length")
                if len(np.unique(data["branch_id"])) != count:
                    errors.append("branch IDs are not unique")
                if set(int(value) for value in data["branch_id"]) != set(range(count)):
                    errors.append("branch IDs are not 0..M-1")
                for name in ("actions", "states", "progress"):
                    if not np.all(np.isfinite(data[name])):
                        errors.append(f"nonfinite {name}")
        sums = sums_path.read_text(encoding="utf-8").splitlines()
        expected_lines = {f"{actual_sha}  cell.npz", f"{sha256_file(metadata_path)}  metadata.json"}
        if set(sums) != expected_lines:
            errors.append("SHA256SUMS mismatch")
    except Exception as exc:  # fail closed on malformed or partial cells
        errors.append(f"parse error: {type(exc).__name__}: {exc}")
        metadata = None
    return not errors, errors, metadata


def record_cell_failure(directory: str | Path, errors: Sequence[str], *, reason: str) -> Path:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    destination = root / f"FAILURE.{stamp}.{os.getpid()}.json"
    atomic_json(destination, {"protocol_id": PHASE1_PROTOCOL_ID, "reason": reason, "errors": list(errors)})
    return destination


def cell_path(output_root: str | Path, suite: str, task_id: int, selection_index: int, location_index: int, stream: str) -> Path:
    return Path(output_root) / suite / f"task{int(task_id):02d}" / f"episode{int(selection_index):02d}" / f"location{int(location_index):02d}" / stream


def _cell_marker_digest(entries: Sequence[tuple[str, Path]]) -> str:
    """Hash a stable relative-cell/marker list without absolute paths."""

    digest = hashlib.sha256()
    for relative, marker_path in sorted(entries, key=lambda value: value[0]):
        digest.update(str(relative).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(marker_path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def baseline_replay_path(
    output_root: str | Path,
    suite: str,
    task_id: int,
    selection_index: int,
) -> Path:
    """Return the portable location of one episode baseline evidence record."""

    return (
        Path(output_root)
        / suite
        / f"task{int(task_id):02d}"
        / f"episode{int(selection_index):02d}"
        / "BASELINE_REPLAY.json"
    )


def _baseline_replay_sums_path(path: str | Path) -> Path:
    return Path(path).with_name("BASELINE_REPLAY_SHA256SUMS")


def write_baseline_replay_evidence(
    output_root: str | Path,
    *,
    suite: str,
    task_id: int,
    selection_index: int,
    parent_id: str,
    row: Mapping[str, Any],
    source_raw_file: str,
    source_raw_sha256: str,
    replay: Mapping[str, Any],
    owner_uid: int | None = None,
    owner_gid: int | None = None,
) -> dict[str, Any]:
    """Atomically persist one descendant-blind baseline replay accounting record."""

    path = baseline_replay_path(output_root, suite, task_id, selection_index)
    payload = {
        "schema_version": PHASE1_SCHEMA_VERSION,
        "protocol_id": PHASE1_PROTOCOL_ID,
        "marker_type": "baseline_replay",
        "suite": str(suite),
        "task_id": int(task_id),
        "selection_index": int(selection_index),
        "parent_id": str(parent_id),
        "source_raw_file": str(source_raw_file),
        "source_raw_sha256": str(source_raw_sha256),
        "source_raw_key": str(parent_id),
        "init_state": int(row["init_state"]),
        "candidate_id": int(row["candidate_id"]),
        "rollout_seed": int(row["rollout_seed"]),
        "baseline_length": int(replay["baseline_length"]),
        "replay_steps": int(replay["replay_steps"]),
        "replay_actions": int(replay["replay_actions"]),
        "baseline_success": bool(replay["baseline_success"]),
        "expected_success": bool(row.get("success", False)),
        "replay_action_sha256": str(replay["replay_action_sha256"]),
        "replay_action_shape": [int(value) for value in replay["replay_action_shape"]],
        "policy_forwards": int(replay["policy_forwards"]),
        "policy_batches": int(replay["policy_batches"]),
        "target_steps": [int(value) for value in replay["target_steps"]],
        "captured_steps": [int(value) for value in replay["captured_steps"]],
        "action_max_abs_error": float(replay["action_max_abs_error"]),
        "owner_uid": int(os.getuid() if owner_uid is None else owner_uid),
        "owner_gid": int(os.getgid() if owner_gid is None else owner_gid),
    }
    atomic_json(path, payload)
    digest = sha256_file(path)
    _atomic_text(
        _baseline_replay_sums_path(path),
        f"{digest}  {path.name}\n",
    )
    return {**payload, "record_sha256": digest}


def validate_baseline_replay_evidence(
    path: str | Path,
    *,
    suite: str | None = None,
    task_id: int | None = None,
    selection_index: int | None = None,
    parent_id: str | None = None,
    source_raw_file: str | None = None,
    source_raw_sha256: str | None = None,
    expected_success: bool | None = None,
    require_owner: tuple[int, int] | None = (2254, 2254),
) -> tuple[bool, list[str], dict[str, Any] | None]:
    """Fail-closed validation of one baseline replay accounting record."""

    destination = Path(path)
    sums_path = _baseline_replay_sums_path(destination)
    errors: list[str] = []
    if not destination.is_file() or not sums_path.is_file():
        return False, ["missing baseline replay evidence"], None
    try:
        payload = json.loads(destination.read_text(encoding="utf-8"))
        if payload.get("protocol_id") != PHASE1_PROTOCOL_ID or payload.get("marker_type") != "baseline_replay":
            errors.append("baseline replay protocol/type mismatch")
        if suite is not None and payload.get("suite") != suite:
            errors.append("baseline replay suite mismatch")
        if task_id is not None and int(payload.get("task_id", -1)) != int(task_id):
            errors.append("baseline replay task mismatch")
        if selection_index is not None and int(payload.get("selection_index", -1)) != int(selection_index):
            errors.append("baseline replay selection mismatch")
        if parent_id is not None and payload.get("parent_id") != parent_id:
            errors.append("baseline replay parent mismatch")
        if source_raw_file is not None and payload.get("source_raw_file") != source_raw_file:
            errors.append("baseline replay source file mismatch")
        if source_raw_sha256 is not None and payload.get("source_raw_sha256") != source_raw_sha256:
            errors.append("baseline replay source SHA mismatch")
        if expected_success is not None and bool(payload.get("expected_success")) != bool(expected_success):
            errors.append("baseline replay expected outcome mismatch")
        target_steps = [int(value) for value in payload.get("target_steps", [])]
        captured_steps = [int(value) for value in payload.get("captured_steps", [])]
        if len(target_steps) != LOCATIONS_PER_EPISODE:
            errors.append("baseline replay decile count mismatch")
        if len(captured_steps) != len(set(target_steps)):
            errors.append("baseline replay captured-step count mismatch")
        if sorted(captured_steps) != sorted(set(target_steps)):
            errors.append("baseline replay captured-step identity mismatch")
        baseline_length = int(payload.get("baseline_length", -1))
        if int(payload.get("replay_steps", -1)) != baseline_length or int(payload.get("replay_actions", -1)) != baseline_length:
            errors.append("baseline replay length accounting mismatch")
        if bool(payload.get("baseline_success")) != bool(payload.get("expected_success")):
            errors.append("baseline replay terminal label mismatch")
        action_sha = str(payload.get("replay_action_sha256", ""))
        if len(action_sha) != 64 or any(value not in "0123456789abcdef" for value in action_sha):
            errors.append("baseline replay action SHA missing or malformed")
        action_shape = payload.get("replay_action_shape")
        if (
            not isinstance(action_shape, list)
            or not action_shape
            or int(action_shape[0]) != baseline_length
            or any(int(value) <= 0 for value in action_shape)
        ):
            errors.append("baseline replay action shape mismatch")
        if int(payload.get("policy_forwards", -1)) < 1 or int(payload.get("policy_batches", -1)) < 1:
            errors.append("baseline replay policy accounting missing")
        if not np.isfinite(float(payload.get("action_max_abs_error", float("nan")))):
            errors.append("baseline replay action error is nonfinite")
        actual_sha = sha256_file(destination)
        sums = sums_path.read_text(encoding="utf-8").splitlines()
        if sums != [f"{actual_sha}  {destination.name}"]:
            errors.append("baseline replay SHA256SUMS mismatch")
        if require_owner is not None:
            required_owner = tuple(int(value) for value in require_owner)
            observed_owner = (int(payload.get("owner_uid", -1)), int(payload.get("owner_gid", -1)))
            if observed_owner != required_owner:
                errors.append(f"baseline replay owner mismatch: {observed_owner} != {required_owner}")
            for candidate in (destination, sums_path):
                stat = candidate.stat()
                if (int(stat.st_uid), int(stat.st_gid)) != required_owner:
                    errors.append(f"baseline replay filesystem owner mismatch: {candidate}")
        return not errors, errors, {**payload, "record_sha256": actual_sha}
    except Exception as exc:
        return False, [f"baseline replay parse error: {type(exc).__name__}: {exc}"], None

def write_task_completion_marker(
    output_root: str | Path,
    suite: str,
    task_id: int,
    selection_manifest: str | Path,
    *,
    streams: Iterable[str] = STREAMS,
    require_owner: tuple[int, int] | None = (2254, 2254),
) -> dict[str, Any]:
    """Validate and atomically seal one complete natural task.

    This marker is intentionally written only after all selected
    episode/location/stream cells pass their own SHA/schema checks.  A later
    resume can therefore trust it only after re-validating the recorded cell
    marker digest.
    """

    selected_streams = tuple(streams)
    if not selected_streams or any(stream not in STREAMS for stream in selected_streams):
        raise ValueError(f"streams must be a nonempty subset of {STREAMS}")
    selection_path = Path(selection_manifest)
    selection_ok, selection_errors = validate_selection_manifest(selection_path)
    if not selection_ok:
        raise RuntimeError("invalid selection manifest: " + "; ".join(selection_errors))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    entries: list[tuple[str, Path]] = []
    baseline_entries: list[tuple[str, Path]] = []
    cell_count = 0
    for selected in selection["selected"]:
        episode = int(selected["selection_index"])
        parent = str(selected["episode"])
        baseline_path = baseline_replay_path(output_root, suite, task_id, episode)
        baseline_valid, baseline_reasons, _ = validate_baseline_replay_evidence(
            baseline_path,
            suite=suite,
            task_id=task_id,
            selection_index=episode,
            parent_id=parent,
            expected_success=(
                bool(selected["baseline_success"])
                if "baseline_success" in selected
                else None
            ),
            require_owner=require_owner,
        )
        if not baseline_valid:
            raise RuntimeError(
                f"cannot seal baseline replay {baseline_path}: {'; '.join(baseline_reasons)}"
            )
        baseline_entries.append((f"episode{episode:02d}", baseline_path))
        for location in range(LOCATIONS_PER_EPISODE):
            for stream in selected_streams:
                directory = cell_path(output_root, suite, task_id, episode, location, stream)
                valid, reasons, _ = validate_cell(
                    directory,
                    suite=suite,
                    task_id=task_id,
                    parent_id=parent,
                    location_index=location,
                    stream=stream,
                    require_owner=require_owner,
                )
                if not valid:
                    raise RuntimeError(f"cannot seal task cell {directory}: {'; '.join(reasons)}")
                entries.append((f"episode{episode:02d}/location{location:02d}/{stream}", directory / "COMPLETED_CELL.json"))
                cell_count += 1
    expected_cells = len(selection["selected"]) * LOCATIONS_PER_EPISODE * len(selected_streams)
    if cell_count != expected_cells:
        raise RuntimeError(f"task cell count={cell_count} != expected={expected_cells}")
    task_root = Path(output_root) / suite / f"task{int(task_id):02d}"
    marker = {
        "schema_version": PHASE1_SCHEMA_VERSION,
        "protocol_id": PHASE1_PROTOCOL_ID,
        "marker_type": "task",
        "suite": suite,
        "task_id": int(task_id),
        "selection_manifest": selection_path.name,
        "selection_manifest_sha256": sha256_file(selection_path),
        "streams": list(selected_streams),
        "selected_episode_count": len(selection["selected"]),
        "locations_per_episode": LOCATIONS_PER_EPISODE,
        "descendants_per_cell": DESCENDANTS_PER_CELL,
        "completed_cells": cell_count,
        "cell_marker_sha256": _cell_marker_digest(entries),
        "baseline_replay_count": len(baseline_entries),
        "baseline_replay_sha256": _cell_marker_digest(baseline_entries),
        "owner_required": None if require_owner is None else [int(require_owner[0]), int(require_owner[1])],
        "checkpoint": "TASK_COMPLETE",
    }
    atomic_json(task_root / "COMPLETED_TASK.json", marker)
    return marker


def validate_task_completion_marker(
    output_root: str | Path,
    suite: str,
    task_id: int,
    selection_manifest: str | Path,
    *,
    streams: Iterable[str] = STREAMS,
    require_owner: tuple[int, int] | None = (2254, 2254),
) -> tuple[bool, list[str], dict[str, Any] | None]:
    """Fail-closed validation of the task-level completion marker."""

    errors: list[str] = []
    selected_streams = tuple(streams)
    marker_path = Path(output_root) / suite / f"task{int(task_id):02d}" / "COMPLETED_TASK.json"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        selection = json.loads(Path(selection_manifest).read_text(encoding="utf-8"))
        if marker.get("protocol_id") != PHASE1_PROTOCOL_ID or marker.get("marker_type") != "task":
            errors.append("task marker protocol/type mismatch")
        if marker.get("suite") != suite or int(marker.get("task_id", -1)) != int(task_id):
            errors.append("task marker identity mismatch")
        if marker.get("selection_manifest") != Path(selection_manifest).name:
            errors.append("selection manifest name mismatch")
        if marker.get("selection_manifest_sha256") != sha256_file(Path(selection_manifest)):
            errors.append("selection manifest SHA mismatch")
        expected_cells = len(selection.get("selected", [])) * LOCATIONS_PER_EPISODE * len(selected_streams)
        if marker.get("completed_cells") != expected_cells:
            errors.append(f"task marker cell count mismatch: {marker.get('completed_cells')} != {expected_cells}")
        if marker.get("streams") != list(selected_streams):
            errors.append("task marker stream mismatch")
        entries: list[tuple[str, Path]] = []
        baseline_entries: list[tuple[str, Path]] = []
        for selected in selection.get("selected", []):
            episode = int(selected["selection_index"])
            parent = str(selected.get("episode", ""))
            baseline_path = baseline_replay_path(output_root, suite, task_id, episode)
            baseline_valid, baseline_reasons, _ = validate_baseline_replay_evidence(
                baseline_path,
                suite=suite,
                task_id=task_id,
                selection_index=episode,
                parent_id=parent,
                expected_success=(
                    bool(selected["baseline_success"])
                    if "baseline_success" in selected
                    else None
                ),
                require_owner=require_owner,
            )
            if not baseline_valid:
                errors.append(
                    f"invalid baseline replay: {baseline_path}: {'; '.join(baseline_reasons)}"
                )
            else:
                baseline_entries.append((f"episode{episode:02d}", baseline_path))
            for location in range(LOCATIONS_PER_EPISODE):
                for stream in selected_streams:
                    directory = cell_path(output_root, suite, task_id, episode, location, stream)
                    valid, reasons, _ = validate_cell(
                        directory,
                        suite=suite,
                        task_id=task_id,
                        parent_id=parent,
                        location_index=location,
                        stream=stream,
                        require_owner=require_owner,
                    )
                    if not valid:
                        errors.append(f"invalid cell: {directory}: {'; '.join(reasons)}")
                    cell_marker = directory / "COMPLETED_CELL.json"
                    if not cell_marker.is_file():
                        errors.append(f"missing cell marker: {cell_marker}")
                    else:
                        entries.append((f"episode{episode:02d}/location{location:02d}/{stream}", cell_marker))
        if entries and marker.get("cell_marker_sha256") != _cell_marker_digest(entries):
            errors.append("task cell-marker digest mismatch")
        expected_baseline_count = len(selection.get("selected", []))
        if marker.get("baseline_replay_count") != expected_baseline_count:
            errors.append(
                f"task marker baseline replay count mismatch: {marker.get('baseline_replay_count')} != {expected_baseline_count}"
            )
        if baseline_entries and marker.get("baseline_replay_sha256") != _cell_marker_digest(baseline_entries):
            errors.append("task baseline-replay digest mismatch")
        if marker.get("owner_required") != (None if require_owner is None else [int(require_owner[0]), int(require_owner[1])]):
            errors.append("task owner contract mismatch")
        return not errors, errors, marker
    except Exception as exc:
        return False, [f"task marker parse error: {type(exc).__name__}: {exc}"], None


def validate_selection_manifest(path: str | Path, *, expected_count: int = 12) -> tuple[bool, list[str]]:
    errors: list[str] = []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("protocol_id") != PHASE1_PROTOCOL_ID:
        errors.append("protocol mismatch")
    selected = payload.get("selected", [])
    if len(selected) != expected_count:
        errors.append(f"selection count {len(selected)} != {expected_count}")
    episodes = [str(row.get("episode")) for row in selected]
    if len(set(episodes)) != len(episodes):
        errors.append("selected episodes are not unique")
    for row in selected:
        expected = selection_rank(row["suite"], int(row["task_id"]), int(row["init_state"]), int(row["candidate_id"]), int(row["rollout_seed"]))
        if row.get("selection_sha256") != expected:
            errors.append(f"selection rank mismatch for {row.get('episode')}")
        if "shortfall" in str(row.get("selection_reason", "")) and row.get("requested_stratum") is not None:
            errors.append("fallback row incorrectly retains requested stratum")
    return not errors, errors


def validate_all_selection_manifest(root: str | Path) -> dict[str, Any]:
    """Validate all 40 task selections plus their top-level SHA seal."""

    base = Path(root)
    errors: list[dict[str, Any] | str] = []
    summary_path = base / "SELECTION_MANIFEST.json"
    completion_path = base / "COMPLETED_SELECTION.json"
    sums_path = base / "SELECTION_SHA256SUMS"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if summary.get("protocol_id") != PHASE1_PROTOCOL_ID or summary.get("marker_type") != "selection":
            errors.append("selection summary protocol/type mismatch")
        if int(summary.get("task_count", -1)) != len(TASKS):
            errors.append("selection summary task count mismatch")
        if completion.get("summary_sha256") != sha256_file(summary_path):
            errors.append("selection completion summary SHA mismatch")
        if completion.get("task_count") != len(TASKS):
            errors.append("selection completion task count mismatch")
        lines = sums_path.read_text(encoding="utf-8").splitlines()
        expected_names = {f"{suite}_task{task_id:02d}.json" for suite, task_id in TASKS} | {summary_path.name, completion_path.name}
        observed_names: set[str] = set()
        for line in lines:
            expected, relative = line.split("  ", 1)
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts or relative_path.name in observed_names:
                errors.append(f"unsafe/duplicate selection checksum entry: {relative}")
                continue
            target = base / relative_path
            observed_names.add(relative_path.name)
            if not target.is_file() or sha256_file(target) != expected:
                errors.append(f"selection checksum mismatch: {relative}")
        if observed_names != expected_names:
            errors.append(f"selection checksum names mismatch: {sorted(observed_names)}")
        indexed = {(str(row.get("suite")), int(row.get("task_id", -1))): row for row in summary.get("tasks", [])}
        if len(indexed) != len(TASKS):
            errors.append("selection summary task entries are incomplete")
        for suite, task_id in TASKS:
            path = base / f"{suite}_task{task_id:02d}.json"
            valid, reasons = validate_selection_manifest(path)
            if not valid:
                errors.append({"suite": suite, "task_id": task_id, "errors": reasons})
            row = indexed.get((suite, task_id))
            if row is None or row.get("path") != path.name or row.get("sha256") != sha256_file(path):
                errors.append({"suite": suite, "task_id": task_id, "errors": ["top-level selection SHA/index mismatch"]})
    except Exception as exc:
        errors.append(f"selection completion parse error: {type(exc).__name__}: {exc}")
    return {"protocol_id": PHASE1_PROTOCOL_ID, "task_count": len(TASKS), "valid": not errors, "errors": errors}


def validate_phase1_config(protocol_path: str | Path, shards_path: str | Path | None = None) -> dict[str, Any]:
    """Validate the frozen Phase-1R protocol and optional rank assignment.

    This is intentionally a structural validator: it checks that the runner
    cannot silently drift from the registered constants, while leaving the
    scientific decision to the frozen analysis artifact.
    """

    errors: list[str] = []
    try:
        protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
        if protocol.get("protocol_id") != PHASE1_PROTOCOL_ID:
            errors.append("protocol ID mismatch")
        scope = protocol.get("task_scope", {})
        if int(scope.get("natural_task_count", -1)) != len(TASKS):
            errors.append("natural task count mismatch")
        if tuple(scope.get("suite_order", ())) != SUITE_ORDER or scope.get("task_ids") != list(range(10)):
            errors.append("task ordering mismatch")
        selection = protocol.get("episode_selection", {})
        if selection.get("unit") != "phase0r_rollout" or int(selection.get("count_per_task", -1)) != 12:
            errors.append("selection contract mismatch")
        if selection.get("requested_per_stratum") != {"failed": 4, "marginal": 4, "successful": 4}:
            errors.append("selection stratum contract mismatch")
        branching = protocol.get("branching", {})
        if int(branching.get("locations", -1)) != LOCATIONS_PER_EPISODE or int(branching.get("descendants_per_cell", -1)) != DESCENDANTS_PER_CELL:
            errors.append("branch grid mismatch")
        if tuple(branching.get("streams", ())) != STREAMS:
            errors.append("stream contract mismatch")
        expected_deciles = [float(value) for value in DECILES]
        if [float(value) for value in branching.get("deciles", [])] != expected_deciles:
            errors.append("decile contract mismatch")
        controls = protocol.get("controls", {})
        if controls.get("positive") != "GeometricCommit2D-Phase1R-v1" or controls.get("null") != "OpenPlane2D-Phase1R-v1":
            errors.append("control contract mismatch")
        statistics = protocol.get("statistics", {})
        if int(statistics.get("permutation_shuffles", -1)) < 1000 or int(statistics.get("bootstrap_replicates", -1)) < 10000:
            errors.append("statistics minimum mismatch")
        compute = protocol.get("compute", {})
        if int(compute.get("max_active_gpus_global", -1)) > 16 or int(compute.get("max_gpu_per_job", -1)) > 8 or int(compute.get("max_cpu_per_job", -1)) > 88 or int(compute.get("max_memory_gib_per_job", -1)) > 1525:
            errors.append("resource ceiling exceeded")
        if shards_path is not None:
            shards = json.loads(Path(shards_path).read_text(encoding="utf-8"))
            if shards.get("protocol_id") != PHASE1_PROTOCOL_ID:
                errors.append("shard protocol ID mismatch")
            if int(shards.get("global_world_size", -1)) != 16 or int(shards.get("gpu_per_job", -1)) > 8:
                errors.append("shard resource contract mismatch")
            assignments: dict[int, str] = {}
            expected_task_names = {f"{suite}_task{task_id:02d}" for suite, task_id in TASKS}
            for shard_name, shard in shards.get("shards", {}).items():
                for rank_text, names in shard.get("rank_tasks", {}).items():
                    rank = int(rank_text)
                    if rank in assignments:
                        errors.append(f"duplicate rank assignment: {rank}")
                    assignments[rank] = shard_name
                    for name in names:
                        if name not in expected_task_names:
                            errors.append(f"unknown shard task: {name}")
                        else:
                            # Store ownership separately below to detect
                            # duplicates without relying on rank order.
                            pass
            task_owners: dict[str, int] = {}
            for shard in shards.get("shards", {}).values():
                for rank_text, names in shard.get("rank_tasks", {}).items():
                    for name in names:
                        rank = int(rank_text)
                        if name in task_owners:
                            errors.append(f"task assigned more than once: {name}")
                        task_owners[name] = rank
            if set(task_owners) != expected_task_names:
                errors.append(f"shard task coverage mismatch: {len(task_owners)} != {len(expected_task_names)}")
            if set(assignments) != set(range(16)):
                errors.append("rank coverage mismatch")
    except Exception as exc:
        errors.append(f"config parse error: {type(exc).__name__}: {exc}")
    return {"protocol_id": PHASE1_PROTOCOL_ID, "valid": not errors, "errors": errors}


def validate_natural_bundle(
    selection_root: str | Path,
    natural_root: str | Path,
    *,
    require_owner: tuple[int, int] | None = (2254, 2254),
) -> dict[str, Any]:
    """Strongly validate every expected natural task/episode/location/stream cell."""

    selection_base = Path(selection_root)
    errors: list[dict[str, Any]] = []
    checked = 0
    for suite, task_id in TASKS:
        manifest_path = selection_base / f"{suite}_task{task_id:02d}.json"
        try:
            valid_selection, selection_errors = validate_selection_manifest(manifest_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            valid_selection, selection_errors, manifest = False, [f"selection parse error: {exc}"], {"selected": []}
        if not valid_selection:
            errors.append({"path": str(manifest_path), "errors": selection_errors})
        for selected in manifest.get("selected", []):
            episode = int(selected.get("selection_index", -1))
            parent = str(selected.get("episode", ""))
            for location in range(LOCATIONS_PER_EPISODE):
                for stream in STREAMS:
                    directory = cell_path(natural_root, suite, task_id, episode, location, stream)
                    valid, reasons, _ = validate_cell(directory, suite=suite, task_id=task_id, parent_id=parent, location_index=location, stream=stream, require_owner=require_owner)
                    checked += 1
                    if not valid:
                        errors.append({"path": str(directory), "errors": reasons})
        task_ok, task_errors, _ = validate_task_completion_marker(
            natural_root,
            suite,
            task_id,
            manifest_path,
            streams=STREAMS,
            require_owner=require_owner,
        )
        if not task_ok:
            errors.append({"path": str(Path(natural_root) / suite / f"task{task_id:02d}" / "COMPLETED_TASK.json"), "errors": task_errors})
    return {
        "protocol_id": PHASE1_PROTOCOL_ID,
        "checked_cells": checked,
        "expected_cells": len(TASKS) * 12 * LOCATIONS_PER_EPISODE * len(STREAMS),
        "valid": not errors and checked == len(TASKS) * 12 * LOCATIONS_PER_EPISODE * len(STREAMS),
        "errors": errors,
    }


class Phase1Collector:
    """Collect natural/control cells using supplied environment and policy factories."""

    def __init__(
        self,
        environment_factory: Callable[[str, int, str], Any],
        policy_factory: Callable[[str, int, str], Any],
        *,
        max_steps: int = 1000,
        microbatch: int = 1,
        require_owner: tuple[int, int] | None = (2254, 2254),
    ):
        self.environment_factory = environment_factory
        self.policy_factory = policy_factory
        self.max_steps = int(max_steps)
        self.microbatch = max(1, int(microbatch))
        self.require_owner = require_owner

    def collect_task(
        self,
        raw_dir: str | Path,
        selection_manifest: str | Path,
        output_root: str | Path,
        suite: str,
        task_id: int,
        *,
        streams: Iterable[str] = STREAMS,
    ) -> dict[str, Any]:
        selection = json.loads(Path(selection_manifest).read_text(encoding="utf-8"))
        ok, errors = validate_selection_manifest(selection_manifest)
        if not ok:
            raise RuntimeError("invalid selection manifest: " + "; ".join(errors))
        selected_streams = tuple(streams)
        if not selected_streams or any(stream not in STREAMS for stream in selected_streams):
            raise ValueError(f"streams must be a nonempty subset of {STREAMS}")
        raw_path = Path(raw_dir) / f"{suite}_task{int(task_id):02d}.npz"
        raw_source_sha = sha256_file(raw_path)
        raw_rows = load_task_rollouts(raw_path)
        raw_by_key = {
            episode_key(suite, task_id, int(row["init_state"]), int(row["candidate_id"])): row for row in raw_rows
        }
        policy = self.policy_factory(suite, int(task_id), str(selection.get("prompt", "")))
        environment = self.environment_factory(suite, int(task_id), "natural")
        branch_pool: BranchEnvironmentPool | None = None
        completed = 0
        failures = []
        try:
            for selected in selection["selected"]:
                parent = str(selected["episode"])
                row = raw_by_key.get(parent)
                if row is None:
                    raise RuntimeError(f"selection row not found in raw archive: {parent}")
                length = len(np.asarray(row["actions"]))
                target_steps = decile_steps(length)
                selection_index = int(selected["selection_index"])
                baseline_path = baseline_replay_path(
                    output_root, suite, task_id, selection_index
                )
                baseline_valid, baseline_reasons, _ = validate_baseline_replay_evidence(
                    baseline_path,
                    suite=suite,
                    task_id=task_id,
                    selection_index=selection_index,
                    parent_id=parent,
                    source_raw_file=raw_path.name,
                    source_raw_sha256=raw_source_sha,
                    expected_success=bool(row["success"]),
                    require_owner=self.require_owner,
                )
                cell_validity: dict[tuple[int, str], tuple[bool, list[str]]] = {}
                all_cells_valid = baseline_valid
                for location_index in range(LOCATIONS_PER_EPISODE):
                    for stream in selected_streams:
                        directory = cell_path(
                            output_root,
                            suite,
                            task_id,
                            selection_index,
                            location_index,
                            stream,
                        )
                        valid, reasons, _ = validate_cell(
                            directory,
                            suite=suite,
                            task_id=task_id,
                            parent_id=parent,
                            location_index=location_index,
                            stream=stream,
                            require_owner=self.require_owner,
                        )
                        cell_validity[(location_index, stream)] = (valid, reasons)
                        all_cells_valid = all_cells_valid and valid
                if all_cells_valid:
                    completed += LOCATIONS_PER_EPISODE * len(selected_streams)
                    continue
                if not baseline_valid:
                    record_cell_failure(
                        baseline_path.parent,
                        baseline_reasons,
                        reason="invalid_or_partial_before_recompute_baseline_replay",
                    )
                snapshots, replay = replay_baseline_snapshots(
                    environment,
                    policy,
                    suite,
                    task_id,
                    row,
                    target_steps,
                )
                baseline_record = write_baseline_replay_evidence(
                    output_root,
                    suite=suite,
                    task_id=task_id,
                    selection_index=selection_index,
                    parent_id=parent,
                    row=row,
                    source_raw_file=raw_path.name,
                    source_raw_sha256=raw_source_sha,
                    replay=replay,
                )
                if (
                    branch_pool is None
                    and self.microbatch > 1
                    and hasattr(policy, "sample_model_actions_official")
                ):
                    branch_pool = BranchEnvironmentPool(
                        self.environment_factory,
                        suite,
                        task_id,
                        capacity=min(DESCENDANTS_PER_CELL, self.microbatch),
                        kind="natural",
                    )
                for location_index, step in enumerate(target_steps):
                    snapshot = snapshots[int(step)]
                    for stream in selected_streams:
                        directory = cell_path(
                            output_root,
                            suite,
                            task_id,
                            selection_index,
                            location_index,
                            stream,
                        )
                        valid, reasons = cell_validity[(location_index, stream)]
                        if valid:
                            completed += 1
                            continue
                        if (directory / "cell.npz").exists() or (directory / "metadata.json").exists():
                            record_cell_failure(
                                directory,
                                reasons,
                                reason="invalid_or_partial_before_recompute",
                            )
                        traces = collect_branch_cell(
                            environment,
                            policy,
                            snapshot=snapshot,
                            suite=suite,
                            task_id=task_id,
                            parent_id=parent,
                            generation_step=step,
                            stream=stream,
                            location_index=location_index,
                            max_steps=self.max_steps,
                            microbatch=self.microbatch,
                            environment_pool=branch_pool,
                        )
                        write_cell(
                            directory,
                            traces,
                            suite=suite,
                            task_id=task_id,
                            parent_id=parent,
                            selection_index=selection_index,
                            location_index=location_index,
                            decile=(location_index + 1) / 10.0,
                            generation_step=step,
                            stream=stream,
                            baseline_length=length,
                            metadata_extra={
                                "baseline_replay_file": baseline_path.name,
                                "baseline_replay_sha256": baseline_record["record_sha256"],
                                "baseline_replay_steps": baseline_record["replay_steps"],
                                "baseline_policy_forwards": baseline_record["policy_forwards"],
                                "baseline_policy_batches": baseline_record["policy_batches"],
                            },
                        )
                        completed += 1
        except Exception as exc:
            failures.append(f"{type(exc).__name__}: {exc}")
            raise
        finally:
            try:
                if branch_pool is not None:
                    branch_pool.close()
            finally:
                if hasattr(environment, "close"):
                    environment.close()
        marker = write_task_completion_marker(
            output_root,
            suite,
            task_id,
            selection_manifest,
            streams=selected_streams,
            require_owner=self.require_owner,
        )
        return {"protocol_id": PHASE1_PROTOCOL_ID, "suite": suite, "task_id": int(task_id), "completed_cells": completed, "failures": failures, "task_completion": marker}


class NaturalPhase1Runtime:
    """Production adapter around the frozen CleanPi05/Task64Environment pair."""

    def __init__(self, *, qpilots_root: str, libero_root: str, checkpoint: str, microbatch: int = 8):
        configure_external_sources(qpilots_root, libero_root)
        from qpilots_libero.policy import CleanPi05LiberoPolicy
        from qpilots_libero.environment import Task64Environment
        from libero.libero import benchmark

        self.policy_class = CleanPi05LiberoPolicy
        self.environment_class = Task64Environment
        self.benchmark = benchmark
        self.checkpoint = checkpoint
        self.microbatch = int(microbatch)
        self._policies: dict[tuple[str, int], Any] = {}

    def policy_factory(self, suite: str, task_id: int, prompt: str) -> Any:
        key = (suite, int(task_id))
        if key not in self._policies:
            # Selection manifests intentionally contain no language prompt.
            # Resolve it from the pinned benchmark task so the production
            # collector cannot silently construct a blank-prompt policy.
            if not prompt:
                task_suite = self.benchmark.get_benchmark_dict()[suite]()
                prompt = str(task_suite.get_task(int(task_id)).language)
            self._policies[key] = self.policy_class(self.checkpoint, default_prompt=prompt)
        return self._policies[key]

    def environment_factory(self, suite: str, task_id: int, prompt: str) -> Any:
        task_suite = self.benchmark.get_benchmark_dict()[suite]()
        task = task_suite.get_task(int(task_id))
        config = task_config(suite, int(task_id), str(task.language), 520 if suite == "libero_10" else {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300}.get(suite, 400))
        return self.environment_class(config, seed=0)

    def collector(self, *, max_steps: int = 1000) -> Phase1Collector:
        return Phase1Collector(self.environment_factory, self.policy_factory, max_steps=max_steps, microbatch=self.microbatch, require_owner=(2254, 2254))
