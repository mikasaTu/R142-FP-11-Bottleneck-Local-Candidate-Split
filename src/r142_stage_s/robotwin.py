"""Fail-closed real RoboTwin adapter for the Stage-S substrate screen.

The released Evo-1 RoboTwin policy is server-backed. This module does not
simulate RoboTwin, copy a state from observations, or invent outcomes. A real
candidate family is accepted only when explicit hooks expose simulator state,
policy observation history, action queue, and every RNG stream.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


class CapabilityError(RuntimeError):
    """Missing/non-deterministic real-runtime capability, not an episode fail."""


@dataclass(frozen=True)
class RoboTwinPins:
    checkpoint_repo: str = "MINT-SJTU/Evo1_RoboTwin2_clean"
    checkpoint_revision: str = "ce8c583724706fbf7a03c17237761c65bf6813a7"
    evo_repo: str = "https://github.com/MINT-SJTU/Evo-1.git"
    evo_revision: str = "5fd14b015013c4fd0aacf5f8f48f868ca9b870a2"
    robotwin_repo: str = "https://github.com/RoboTwin-Platform/RoboTwin.git"
    robotwin_revision: str = "13c3c47ff4312dd62484bcd51be034af55c062d1"
    robotwin_ref: str = "stable_2.0"
    curobo_repo: str = "https://github.com/NVlabs/curobo.git"
    curobo_revision: str = "d64c4b005459db10c5dd867d8b30a87d5bda9bdb"

    def as_dict(self) -> Dict[str, str]:
        return {
            "checkpoint_repo": self.checkpoint_repo,
            "checkpoint_revision": self.checkpoint_revision,
            "evo_repo": self.evo_repo,
            "evo_revision": self.evo_revision,
            "robotwin_repo": self.robotwin_repo,
            "robotwin_revision": self.robotwin_revision,
            "robotwin_ref": self.robotwin_ref,
            "curobo_repo": self.curobo_repo,
            "curobo_revision": self.curobo_revision,
        }


# Published clean-policy values from the pinned Evo-1 README. These are used
# only for the pre-registered, lexical task selection rule.
PUBLISHED_CLEAN_SUCCESS: Mapping[str, float] = {
    "blocks_ranking_size": 0.58,
    "place_a2b_left": 0.48,
    "place_a2b_right": 0.38,
    "place_bread_basket": 0.63,
    "place_bread_skillet": 0.63,
    "place_can_basket": 0.50,
    "place_fan": 0.34,
    "place_object_scale": 0.49,
    "place_shoe": 0.33,
    "put_object_cabinet": 0.39,
    "rotate_qrcode": 0.32,
    "scan_object": 0.32,
    "stamp_seal": 0.28,
    "turn_switch": 0.28,
}

PUBLISHED_EVAL_URL = (
    "https://github.com/MINT-SJTU/Evo-1/blob/evo1-flash/"
    "RoboTwin_evaluation/README.md"
)


def select_published_tasks(
    published_rates: Mapping[str, float] = PUBLISHED_CLEAN_SUCCESS,
    *,
    lower: float = 0.25,
    upper: float = 0.65,
    count: int = 10,
) -> Tuple[str, ...]:
    """Return the first ten eligible names in lexical order, exactly."""

    if count <= 0 or lower > upper:
        raise ValueError("invalid task-selection bounds")
    eligible = sorted(
        str(task)
        for task, rate in published_rates.items()
        if lower <= float(rate) <= upper
    )
    if len(eligible) < count:
        raise ValueError(
            f"only {len(eligible)} tasks satisfy [{lower}, {upper}], need {count}"
        )
    return tuple(eligible[:count])


def _copy(value: Any) -> Any:
    return copy.deepcopy(value)


def _hook(obj: Any, names: Sequence[str], label: str) -> Callable[..., Any]:
    for name in names:
        fn = getattr(obj, name, None)
        if callable(fn):
            return fn
    raise CapabilityError(
        "RoboTwin exact replay unavailable: "
        f"{label} hook missing; implement one of {', '.join(names)}"
    )


def _invoke(obj: Any, names: Sequence[str], label: str, *args: Any) -> Any:
    fn = _hook(obj, names, label)
    try:
        return fn(*args)
    except CapabilityError:
        raise
    except Exception as exc:
        raise CapabilityError(
            f"RoboTwin exact replay unavailable: {label} hook failed: {exc}"
        ) from exc


def _max_error(a: Any, b: Any, path: str = "state") -> float:
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        if set(a) != set(b):
            raise CapabilityError(f"restore verification schema mismatch at {path}")
        return max((_max_error(a[k], b[k], f"{path}.{k}") for k in a), default=0.0)
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        if len(a) != len(b):
            raise CapabilityError(f"restore verification length mismatch at {path}")
        return max(
            (_max_error(x, y, f"{path}[{i}]") for i, (x, y) in enumerate(zip(a, b))),
            default=0.0,
        )
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        aa, bb = np.asarray(a), np.asarray(b)
        if aa.shape != bb.shape:
            raise CapabilityError(f"restore verification shape mismatch at {path}")
        if np.issubdtype(aa.dtype, np.number) and np.issubdtype(bb.dtype, np.number):
            return float(np.max(np.abs(aa.astype(float) - bb.astype(float)))) if aa.size else 0.0
        return 0.0 if np.array_equal(aa, bb) else float("inf")
    if isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)):
        return abs(float(a) - float(b))
    return 0.0 if a == b else float("inf")


@dataclass(frozen=True)
class ReplaySnapshot:
    simulator: Any
    policy_history: Any
    action_queue: Any
    rng_streams: Any


class ExactReplayVerifier:
    """Verify the frozen restore -> same action -> next-state contract."""

    def __init__(self, env: Any, policy: Any, *, tolerance: float = 1e-9):
        self.env = env
        self.policy = policy
        self.tolerance = float(tolerance)

    def capture(self) -> ReplaySnapshot:
        sim = _invoke(
            self.env,
            ("capture_simulator_state", "snapshot_simulator", "get_simulator_state"),
            "simulator state",
        )
        history = _invoke(
            self.policy,
            (
                "capture_observation_history",
                "snapshot_observation_history",
                "get_observation_history_state",
            ),
            "policy observation history",
        )
        queue = _invoke(
            self.policy,
            ("capture_action_queue", "snapshot_action_queue", "get_action_queue_state"),
            "policy action queue",
        )
        env_rng = _invoke(
            self.env,
            ("capture_rng_state", "snapshot_rng", "get_rng_state"),
            "environment RNG streams",
        )
        policy_rng = _invoke(
            self.policy,
            ("capture_rng_state", "snapshot_rng", "get_rng_state"),
            "policy RNG streams",
        )
        return ReplaySnapshot(
            _copy(sim),
            _copy(history),
            _copy(queue),
            {"environment": _copy(env_rng), "policy": _copy(policy_rng)},
        )

    def restore(self, snapshot: ReplaySnapshot) -> None:
        _hook(
            self.env,
            ("restore_simulator_state", "restore_simulator", "set_simulator_state"),
            "simulator state restore",
        )(_copy(snapshot.simulator))
        _hook(
            self.policy,
            (
                "restore_observation_history",
                "restore_history",
                "set_observation_history_state",
            ),
            "policy observation history restore",
        )(_copy(snapshot.policy_history))
        _hook(
            self.policy,
            ("restore_action_queue", "restore_queue", "set_action_queue_state"),
            "policy action queue restore",
        )(_copy(snapshot.action_queue))
        _hook(
            self.env,
            ("restore_rng_state", "restore_rng", "set_rng_state"),
            "environment RNG streams restore",
        )(_copy(snapshot.rng_streams["environment"]))
        _hook(
            self.policy,
            ("restore_rng_state", "restore_rng", "set_rng_state"),
            "policy RNG streams restore",
        )(_copy(snapshot.rng_streams["policy"]))

    def verify_restore(self, action: Optional[Any] = None) -> Dict[str, Any]:
        observe = getattr(self.env, "state_for_verification", None)
        if not callable(observe):
            observe = getattr(self.env, "get_obs", None)
        if not callable(observe):
            raise CapabilityError(
                "restore verification requires state_for_verification() or get_obs()"
            )
        act_fn = getattr(self.policy, "act", None)
        if action is None and not callable(act_fn):
            raise CapabilityError("restore verification requires policy.act(observation)")
        snap = self.capture()
        self.restore(snap)
        obs_a = observe()
        act_a = _copy(action) if action is not None else act_fn(obs_a)
        self.env.take_action(act_a)
        state_a = observe()
        self.restore(snap)
        obs_b = observe()
        act_b = _copy(action) if action is not None else act_fn(obs_b)
        self.env.take_action(act_b)
        state_b = observe()
        action_error = _max_error(act_a, act_b, "action")
        next_state_error = _max_error(state_a, state_b, "next_state")
        if max(action_error, next_state_error) > self.tolerance:
            raise CapabilityError(
                "exact replay verification failed: "
                f"action_error={action_error:.3g}, "
                f"next_state_error={next_state_error:.3g}, "
                f"tolerance={self.tolerance:.3g}"
            )
        return {
            "passed": True,
            "tolerance": self.tolerance,
            "action_error": action_error,
            "next_state_error": next_state_error,
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape), "data": value.tolist()}
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


@dataclass
class CandidateRecord:
    candidate_id: str
    parent_id: Optional[str]
    generation_step: int
    action_prefix: list = field(default_factory=list)
    pose_trajectory: list = field(default_factory=list)
    final_success: bool = False
    task_name: str = ""
    family_id: str = ""
    initial_state_id: str = ""
    seed: int = 0
    policy_forwards: int = 0
    env_steps: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return _jsonable(self.__dict__)


class AtomicFamilyWriter:
    """Write raw outcomes and genealogy before an immutable completion marker."""

    def __init__(self, root: os.PathLike | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _atomic(path: Path, data: bytes) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        return hashlib.sha256(data).hexdigest()

    def write(
        self,
        family_id: str,
        records: Iterable[CandidateRecord],
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        records = list(records)
        directory = self.root / family_id
        payload = {
            "family_id": family_id,
            "metadata": _jsonable(dict(metadata or {})),
            "candidates": [r.as_dict() for r in records],
        }
        result_data = (
            json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
        ).encode()
        result_sha = self._atomic(directory / "family.json", result_data)
        genealogy_data = (
            "\n".join(
                json.dumps(r.as_dict(), sort_keys=True, ensure_ascii=False) for r in records
            )
            + "\n"
        ).encode()
        genealogy_sha = self._atomic(directory / "genealogy.jsonl", genealogy_data)
        marker = {
            "family_id": family_id,
            "candidate_count": len(records),
            "files": {"family.json": result_sha, "genealogy.jsonl": genealogy_sha},
        }
        marker_data = (
            json.dumps(marker, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
        ).encode()
        marker_sha = self._atomic(directory / "COMPLETED_FAMILY.json", marker_data)
        return {**marker, "completion_sha256": marker_sha, "path": str(directory)}


def _pose(observation: Any) -> Any:
    if isinstance(observation, Mapping):
        for key in ("pose", "endpose", "ee_pose", "joint_action"):
            if key in observation:
                return observation[key]
        if "observation" in observation:
            return _pose(observation["observation"])
    return None


class FamilyRolloutRunner:
    """Run independent candidates from one exact real initial state."""

    def __init__(
        self,
        env_factory: Callable[[], Any],
        policy_factory: Callable[[int], Any],
        writer: AtomicFamilyWriter,
    ):
        self.env_factory = env_factory
        self.policy_factory = policy_factory
        self.writer = writer

    def run_family(
        self,
        *,
        task_name: str,
        family_id: str,
        initial_state_id: str,
        initial_seed: int,
        candidate_count: int = 32,
    ) -> Dict[str, Any]:
        if candidate_count <= 0:
            raise ValueError("candidate_count must be positive")
        env = self.env_factory()
        reset = _hook(env, ("reset", "reset_episode", "setup_demo"), "episode reset")
        try:
            reset(seed=int(initial_seed), task_name=task_name)
        except TypeError:
            try:
                reset(seed=int(initial_seed))
            except TypeError:
                reset(int(initial_seed))
        first = self.policy_factory(int(initial_seed))
        base = ExactReplayVerifier(env, first).capture()
        records = []
        for index in range(candidate_count):
            seed = int(
                np.random.SeedSequence([int(initial_seed), index]).generate_state(1)[0]
            )
            policy = first if index == 0 else self.policy_factory(seed)
            ExactReplayVerifier(env, policy).restore(base)
            self._seed_policy(policy, seed)
            record = CandidateRecord(
                candidate_id=f"{family_id}/candidate-{index:04d}",
                parent_id=None,
                generation_step=0,
                task_name=task_name,
                family_id=family_id,
                initial_state_id=initial_state_id,
                seed=seed,
            )
            self._rollout(env, policy, record)
            records.append(record)
        return self.writer.write(
            family_id,
            records,
            metadata={
                "task_name": task_name,
                "initial_state_id": initial_state_id,
                "initial_seed": int(initial_seed),
                "candidate_count": candidate_count,
                "candidate_rng": "SeedSequence([initial_seed, candidate_index])",
                "termination": "official eval_success or step_lim",
            },
        )

    @staticmethod
    def _seed_policy(policy: Any, seed: int) -> None:
        fn = getattr(policy, "set_rng", None)
        if callable(fn):
            fn(np.random.default_rng(seed))
            return
        fn = getattr(policy, "seed", None)
        if callable(fn):
            fn(seed)
            return
        raise CapabilityError(
            "independent candidate RNG unavailable: "
            "policy must implement set_rng(rng) or seed(seed)"
        )

    @staticmethod
    def _act(policy: Any, observation: Any, rng: np.random.Generator) -> Any:
        fn = getattr(policy, "act", None)
        if not callable(fn):
            raise CapabilityError("real RoboTwin policy must implement act(observation)")
        try:
            return fn(observation, rng=rng)
        except TypeError:
            return fn(observation)

    def _rollout(self, env: Any, policy: Any, record: CandidateRecord) -> None:
        rng = np.random.default_rng(record.seed)
        initial_forward = getattr(policy, "forward_count", None)
        while not bool(getattr(env, "eval_success", False)):
            limit = getattr(env, "step_lim", None)
            count = getattr(env, "take_action_cnt", None)
            if limit is not None and count is not None and int(count) >= int(limit):
                break
            observation = env.get_obs()
            action = self._act(policy, observation, rng)
            record.action_prefix.append(_jsonable(action))
            record.pose_trajectory.append(_jsonable(_pose(observation)))
            env.take_action(action)
            record.env_steps += 1
            if initial_forward is None:
                record.policy_forwards += 1
            else:
                record.policy_forwards = int(
                    getattr(policy, "forward_count", initial_forward)
                ) - int(initial_forward)
        record.final_success = bool(getattr(env, "eval_success", False))
        check = getattr(env, "check_success", None)
        if not record.final_success and callable(check):
            record.final_success = bool(check())
