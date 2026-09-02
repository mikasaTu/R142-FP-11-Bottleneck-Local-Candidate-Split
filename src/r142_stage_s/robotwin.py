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
    "pick_diverse_bottles": 0.49,
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
    # SAPIEN actor/articulation handles are process-local opaque objects and
    # cannot be deep-copied without replacing the live simulator object.  The
    # concrete snapshot stores those handles under these explicit keys; copy
    # every numeric/state payload while preserving the handles themselves.
    if isinstance(value, Mapping):
        return {
            key: (item if key in {"object", "scene"} else _copy(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy(item) for item in value)
    if isinstance(value, np.ndarray):
        return value.copy()
    try:
        return copy.deepcopy(value)
    except Exception:
        # Opaque policy/server handles are only safe to retain by identity;
        # all values that participate in exact-state comparison are arrays or
        # plain Python containers handled above.
        return value


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


def _capture_rng_state(
    owner: Any,
    label: str,
    *,
    require_owner: bool = False,
    require_torch: bool = True,
) -> Any:
    """Capture process and owner RNG state; fail if torch is unavailable."""
    import random

    try:
        import torch
    except ImportError as exc:
        if require_torch:
            raise CapabilityError(
                f"{label}: torch is required to capture Torch/CUDA RNG streams"
            ) from exc
        torch = None
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    if torch is not None:
        state["torch"] = torch.get_rng_state().clone()
        state["torch_cuda"] = (
            [x.clone() for x in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        )
    hook = getattr(owner, "capture_rng_state", None)
    if callable(hook):
        state["owner"] = _copy(hook())
    elif require_owner:
        raise CapabilityError(
            f"{label}: owner-specific RNG hook missing; "
            "remote policy RNG cannot be assumed deterministic"
        )
    return state


def _restore_rng_state(
    owner: Any,
    state: Mapping[str, Any],
    label: str,
    *,
    require_torch: bool = True,
) -> None:
    import random

    try:
        import torch
    except ImportError as exc:
        if require_torch:
            raise CapabilityError(
                f"{label}: torch is required to restore Torch/CUDA RNG streams"
            ) from exc
        torch = None
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    if torch is not None and "torch" in state:
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and state.get("torch_cuda"):
            torch.cuda.set_rng_state_all(state["torch_cuda"])
    hook = getattr(owner, "restore_rng_state", None)
    if not callable(hook) and "owner" in state:
        raise CapabilityError(f"{label}: owner-specific RNG restore hook missing")
    if callable(hook) and "owner" in state:
        hook(_copy(state["owner"]))


@dataclass(frozen=True)
class ConcreteReplaySnapshot:
    """In-process snapshot of official RoboTwin task plus Evo policy state."""

    simulator: Mapping[str, Any]
    policy_history: Any
    action_queue: Any
    rng_streams: Mapping[str, Any]


class EvoProxyStateAdapter:
    """Stateful wrapper around the released Evo1Proxy plugin.

    The public plugin only exposes infer/close and has a no-op reset_model.
    Therefore a plain Evo1Proxy is deliberately rejected by
    ConcreteRoboTwinRuntime. This wrapper records history and action chunks,
    while requiring an explicit server-side RNG hook for exact replay.
    """

    def __init__(self, proxy: Any):
        self.proxy = proxy
        self.observation_history = []
        self.action_queue = []
        self.rng = np.random.default_rng()

    def infer(self, *args: Any, **kwargs: Any) -> Any:
        response = self.proxy.infer(*args, **kwargs)
        self.observation_history.append(
            {"state": _copy(args[3]) if len(args) > 3 else None,
             "prompt": _copy(args[4]) if len(args) > 4 else None}
        )
        self.action_queue = _copy(response)
        return response

    def consume_action(self, action: Any) -> None:
        if isinstance(self.action_queue, list) and self.action_queue:
            self.action_queue.pop(0)

    def capture_observation_history(self) -> Any:
        return _copy(self.observation_history)

    def restore_observation_history(self, value: Any) -> None:
        self.observation_history = _copy(value)

    def capture_action_queue(self) -> Any:
        return _copy(self.action_queue)

    def restore_action_queue(self, value: Any) -> None:
        self.action_queue = _copy(value)

    def capture_rng_state(self) -> Any:
        hook = getattr(self.proxy, "capture_rng_state", None)
        if not callable(hook):
            raise CapabilityError(
                "Evo proxy exact replay unavailable: the released WebSocket "
                "server exposes no Torch/CUDA RNG snapshot hook"
            )
        return {"local": _copy(self.rng.bit_generator.state), "server": _copy(hook())}

    def restore_rng_state(self, value: Any) -> None:
        hook = getattr(self.proxy, "restore_rng_state", None)
        if not callable(hook):
            raise CapabilityError(
                "Evo proxy exact replay unavailable: server RNG restore hook missing"
            )
        self.rng.bit_generator.state = _copy(value["local"])
        hook(_copy(value["server"]))

    def set_rng(self, rng: np.random.Generator) -> None:
        self.rng = rng
        # Local RNG alone cannot control a server-backed policy.  A concrete
        # deploy wrapper must expose a seed hook so candidate sampling is
        # genuinely independent; the public unpatched proxy is therefore
        # rejected here (and also lacks the snapshot hooks required by the
        # replay gate).
        hook = getattr(self.proxy, "set_rng", None)
        if not callable(hook):
            hook = getattr(self.proxy, "seed", None)
        if not callable(hook):
            raise CapabilityError(
                "Evo proxy independent sampling unavailable: server policy "
                "must expose set_rng(rng) or seed(seed)"
            )
        try:
            hook(rng)
        except TypeError:
            try:
                hook(_copy(rng.bit_generator.state))
            except TypeError:
                hook(int(rng.integers(0, 2**32, dtype=np.uint32)))


class ConcreteRoboTwinRuntime:
    """Snapshot adapter for a real stable_2.0 SAPIEN task and Evo wrapper.

    The task environment is the official RoboTwin object returned by
    script/eval_policy.py. No synthetic state is accepted. Actor/articulation
    object references remain process-local, which is sufficient for branch
    replay; persisted family records contain only outcomes and trajectories.
    """

    _COUNTERS = (
        "take_action_cnt", "step_lim", "eval_success", "plan_success",
        "stage_success_tag", "left_cnt", "right_cnt", "FRAME_IDX",
        "scene_step", "step_count", "physics_step",
    )

    def __init__(self, task_env: Any, policy: Any, *, require_torch: bool = True):
        self.task_env = task_env
        self.policy = policy
        self.require_torch = bool(require_torch)

    @staticmethod
    def _pose_state(obj: Any) -> Any:
        fn = getattr(obj, "get_pose", None)
        return _copy(fn()) if callable(fn) else None

    def capture_simulator_state(self) -> Dict[str, Any]:
        scene = getattr(self.task_env, "scene", None)
        if scene is None:
            raise CapabilityError("RoboTwin scene missing: expected task_env.scene")
        actors_fn = getattr(scene, "get_all_actors", None)
        arts_fn = getattr(scene, "get_all_articulations", None)
        if not callable(actors_fn) or not callable(arts_fn):
            raise CapabilityError(
                "stable_2.0 SAPIEN scene must expose get_all_actors() and "
                "get_all_articulations()"
            )
        actors = []
        for index, actor in enumerate(actors_fn()):
            pose = self._pose_state(actor)
            if pose is None or not callable(getattr(actor, "set_pose", None)):
                raise CapabilityError(
                    f"rigid actor {index} lacks get_pose/set_pose; cannot restore"
                )
            item = {
                "object": actor,
                "index": index,
                "name": getattr(actor, "get_name", lambda: "")(),
                "pose": pose,
            }
            for getter, setter, key in (
                ("get_velocity", "set_velocity", "velocity"),
                ("get_angular_velocity", "set_angular_velocity", "angular_velocity"),
            ):
                get_fn, set_fn = getattr(actor, getter, None), getattr(actor, setter, None)
                if callable(get_fn) and callable(set_fn):
                    item[key] = _copy(get_fn())
            actors.append(item)
        articulations = []
        for index, articulation in enumerate(arts_fn()):
            item = {"object": articulation, "index": index}
            for getter, setter, key in (
                ("get_root_pose", "set_root_pose", "root_pose"),
                ("get_qpos", "set_qpos", "qpos"),
                ("get_qvel", "set_qvel", "qvel"),
                ("get_qacc", "set_qacc", "qacc"),
            ):
                get_fn, set_fn = getattr(articulation, getter, None), getattr(articulation, setter, None)
                if callable(get_fn) and callable(set_fn):
                    item[key] = _copy(get_fn())
                elif callable(get_fn) != callable(set_fn):
                    raise CapabilityError(
                        f"articulation {index} has {getter} without matching {setter}"
                    )
            articulations.append(item)
        state = {
            "scene": scene,
            "actors": actors,
            "articulations": articulations,
            "counters": {key: _copy(getattr(self.task_env, key))
                        for key in self._COUNTERS if hasattr(self.task_env, key)},
            "now_obs": _copy(getattr(self.task_env, "now_obs", None)),
        }
        # Some SAPIEN builds expose a simulation clock.  Preserve it when the
        # paired setter exists; accepting only a getter would make replay
        # silently drift in integrator state, so that case is rejected.
        for getter, setter in (
            ("get_time", "set_time"),
            ("get_sim_time", "set_sim_time"),
            ("get_simulation_time", "set_simulation_time"),
        ):
            get_fn, set_fn = getattr(scene, getter, None), getattr(scene, setter, None)
            if callable(get_fn) and callable(set_fn):
                state["scene_clock"] = {
                    "getter": getter,
                    "setter": setter,
                    "value": _copy(get_fn()),
                }
                break
            if callable(get_fn) != callable(set_fn):
                raise CapabilityError(
                    f"SAPIEN scene has {getter} without matching {setter}; "
                    "cannot restore simulation clock"
                )
        return state

    def restore_simulator_state(self, state: Mapping[str, Any]) -> None:
        for item in state["actors"]:
            actor = item["object"]
            actor.set_pose(_copy(item["pose"]))
            for key, setter in (
                ("velocity", "set_velocity"),
                ("angular_velocity", "set_angular_velocity"),
            ):
                if key in item:
                    getattr(actor, setter)(_copy(item[key]))
        for item in state["articulations"]:
            articulation = item["object"]
            for key, setter in (
                ("root_pose", "set_root_pose"),
                ("qpos", "set_qpos"),
                ("qvel", "set_qvel"),
                ("qacc", "set_qacc"),
            ):
                if key in item:
                    getattr(articulation, setter)(_copy(item[key]))
        for key, value in state["counters"].items():
            setattr(self.task_env, key, _copy(value))
        clock = state.get("scene_clock")
        if clock is not None:
            setter = getattr(self.task_env.scene, clock["setter"], None)
            if not callable(setter):
                raise CapabilityError("SAPIEN scene simulation clock setter disappeared")
            setter(_copy(clock["value"]))
        if "now_obs" in state:
            self.task_env.now_obs = _copy(state["now_obs"])

    def capture_observation_history(self) -> Any:
        return _copy(_hook(
            self.policy,
            ("capture_observation_history", "snapshot_observation_history"),
            "policy observation history",
        )())

    def restore_observation_history(self, value: Any) -> None:
        _hook(
            self.policy,
            ("restore_observation_history", "restore_history"),
            "policy observation history restore",
        )(_copy(value))

    def capture_action_queue(self) -> Any:
        return _copy(_hook(
            self.policy,
            ("capture_action_queue", "snapshot_action_queue"),
            "policy action queue",
        )())

    def restore_action_queue(self, value: Any) -> None:
        _hook(
            self.policy,
            ("restore_action_queue", "restore_queue"),
            "policy action queue restore",
        )(_copy(value))

    def capture_rng_state(self) -> Any:
        # RoboTwin stable_2.0 uses Python/NumPy global randomness in task
        # setup; Torch and CUDA are included by this process-level snapshot.
        # A task-specific hook is included when supplied by the concrete
        # wrapper, but is not guessed from an observation.
        return _capture_rng_state(
            self.task_env,
            "RoboTwin runtime",
            require_torch=self.require_torch,
        )

    def restore_rng_state(self, value: Mapping[str, Any]) -> None:
        _restore_rng_state(
            self.task_env,
            value,
            "RoboTwin runtime",
            require_torch=self.require_torch,
        )

    def capture_snapshot(self) -> ConcreteReplaySnapshot:
        return ConcreteReplaySnapshot(
            simulator=self.capture_simulator_state(),
            policy_history=self.capture_observation_history(),
            action_queue=self.capture_action_queue(),
            rng_streams={
                "runtime": self.capture_rng_state(),
                "policy": _copy(_hook(
                    self.policy,
                    ("capture_rng_state", "snapshot_rng"),
                    "policy RNG streams",
                )()),
            },
        )

    def restore_snapshot(self, snapshot: ConcreteReplaySnapshot) -> None:
        self.restore_simulator_state(snapshot.simulator)
        self.restore_observation_history(snapshot.policy_history)
        self.restore_action_queue(snapshot.action_queue)
        self.restore_rng_state(snapshot.rng_streams["runtime"])
        _hook(
            self.policy,
            ("restore_rng_state", "restore_rng"),
            "policy RNG streams restore",
        )(_copy(snapshot.rng_streams["policy"]))

    def verify_restore(self, action: Optional[Any] = None) -> Dict[str, Any]:
        observe = getattr(self.task_env, "state_for_verification", None)
        if not callable(observe):
            observe = getattr(self.task_env, "get_obs", None)
        if not callable(observe):
            raise CapabilityError(
                "RoboTwin restore verification requires state_for_verification() or get_obs()"
            )
        act_fn = getattr(self.policy, "act", None)
        if action is None and not callable(act_fn):
            raise CapabilityError("Evo wrapper restore verification requires policy.act()")
        snapshot = self.capture_snapshot()
        self.restore_snapshot(snapshot)
        obs_a = observe()
        action_a = _copy(action) if action is not None else act_fn(obs_a)
        self.task_env.take_action(action_a)
        state_a = observe()
        self.restore_snapshot(snapshot)
        obs_b = observe()
        action_b = _copy(action) if action is not None else act_fn(obs_b)
        self.task_env.take_action(action_b)
        state_b = observe()
        action_error = _max_error(action_a, action_b, "action")
        next_state_error = _max_error(state_a, state_b, "next_state")
        if max(action_error, next_state_error) > 1e-9:
            raise CapabilityError(
                "RoboTwin exact replay failed: "
                f"action_error={action_error:.3g}, next_state_error={next_state_error:.3g}"
            )
        return {
            "passed": True,
            "tolerance": 1e-9,
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
    # ``pose_trajectory`` is retained for compatibility with the initial
    # audit.  These explicit fields make the Stage-S persisted contract
    # unambiguous for downstream analysis.
    eef_trajectory: list = field(default_factory=list)
    object_trajectories: Dict[str, list] = field(default_factory=dict)
    final_success: bool = False
    task_name: str = ""
    family_id: str = ""
    initial_state_id: str = ""
    seed: int = 0
    policy_forwards: int = 0
    env_steps: int = 0
    seed_sequence: list = field(default_factory=list)
    seed_genealogy: Dict[str, Any] = field(default_factory=dict)

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

    def _read_completed(self, family_id: str) -> Optional[Dict[str, Any]]:
        """Return a verified immutable marker, or ``None`` if unfinished."""
        directory = self.root / family_id
        marker_path = directory / "COMPLETED_FAMILY.json"
        if not marker_path.is_file():
            return None
        try:
            existing = json.loads(marker_path.read_text(encoding="utf-8"))
            files = existing.get("files", {})
            if not files:
                raise ValueError("completion marker has no file hashes")
            for name, expected_sha in files.items():
                file_path = directory / str(name)
                if not file_path.is_file():
                    raise ValueError(f"missing completion file: {name}")
                actual_sha = hashlib.sha256(file_path.read_bytes()).hexdigest()
                if actual_sha != str(expected_sha):
                    raise ValueError(f"completion file hash mismatch: {name}")
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise CapabilityError(
                f"immutable family completion marker is invalid: {marker_path}"
            ) from exc
        return {
            **existing,
            "completion_sha256": hashlib.sha256(marker_path.read_bytes()).hexdigest(),
            "path": str(directory),
            "skipped_existing": True,
        }

    def completed(self, family_id: str) -> Optional[Dict[str, Any]]:
        """Verify and return a completed family for idempotent resume."""
        return self._read_completed(family_id)

    def write(
        self,
        family_id: str,
        records: Iterable[CandidateRecord],
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        records = list(records)
        directory = self.root / family_id
        existing = self._read_completed(family_id)
        if existing is not None:
            return existing
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
        sums_data = (
            f"{result_sha}  family.json\n{genealogy_sha}  genealogy.jsonl\n"
        ).encode()
        sums_sha = self._atomic(directory / "SHA256SUMS", sums_data)
        marker = {
            "family_id": family_id,
            "candidate_count": len(records),
            "files": {
                "family.json": result_sha,
                "genealogy.jsonl": genealogy_sha,
                "SHA256SUMS": sums_sha,
            },
        }
        marker_data = (
            json.dumps(marker, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
        ).encode()
        marker_sha = self._atomic(directory / "COMPLETED_FAMILY.json", marker_data)
        return {**marker, "completion_sha256": marker_sha, "path": str(directory)}


def _pose(observation: Any) -> Any:
    if isinstance(observation, Mapping):
        for key in ("endpose", "ee_pose", "pose", "joint_action"):
            if key in observation:
                return observation[key]
        if "observation" in observation:
            return _pose(observation["observation"])
    return None


def _object_poses(env: Any) -> Dict[str, Any]:
    """Read real rigid-actor poses from the official scene, if exposed.

    Object trajectories are never inferred from actions or pixels.  A missing
    scene hook is represented as an empty mapping for lightweight unit tests;
    the real runtime wrapper still fails closed on missing simulator snapshot
    capability before any rollout is accepted.
    """
    source = getattr(env, "task_env", env)
    hook = getattr(source, "get_object_poses", None)
    if callable(hook):
        value = hook()
        if not isinstance(value, Mapping):
            raise CapabilityError("get_object_poses() must return a mapping")
        return {str(k): _jsonable(v) for k, v in value.items()}
    scene = getattr(source, "scene", None)
    actors_fn = getattr(scene, "get_all_actors", None)
    if not callable(actors_fn):
        return {}
    result: Dict[str, Any] = {}
    for index, actor in enumerate(actors_fn()):
        get_pose = getattr(actor, "get_pose", None)
        if not callable(get_pose):
            raise CapabilityError(
                f"rigid actor {index} lacks get_pose; cannot record object trajectory"
            )
        get_name = getattr(actor, "get_name", None)
        name = str(get_name()) if callable(get_name) else f"actor-{index:04d}"
        if name in result:
            name = f"{name}#{index:04d}"
        result[name] = _jsonable(get_pose())
    return result


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
        existing = self.writer.completed(family_id)
        if existing is not None:
            return existing
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
        bind_policy = getattr(env, "bind_policy", None)
        if callable(bind_policy):
            bind_policy(first)
        replay = ExactReplayVerifier(env, first)
        base = replay.capture()
        # Mandatory fail-closed preflight.  It runs before candidate 0 and
        # before any completion marker can be written.  The original initial
        # state is restored after the two replay probes.
        replay_gate = replay.verify_restore()
        replay.restore(base)
        records = []
        for index in range(candidate_count):
            seed_sequence = np.random.SeedSequence([int(initial_seed), index])
            seed = int(
                seed_sequence.generate_state(1)[0]
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
                seed_sequence=[int(initial_seed), int(index)],
                seed_genealogy={
                    "root_seed": int(initial_seed),
                    "candidate_index": int(index),
                    "spawn_key": list(seed_sequence.spawn_key),
                },
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
                "replay_capability_gate": replay_gate,
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
        if not hasattr(env, "eval_success"):
            raise CapabilityError("official RoboTwin env must expose eval_success")
        if not callable(getattr(env, "get_obs", None)) or not callable(
            getattr(env, "take_action", None)
        ):
            raise CapabilityError("official RoboTwin env must expose get_obs/take_action")
        if getattr(env, "step_lim", None) is None or getattr(env, "take_action_cnt", None) is None:
            raise CapabilityError(
                "official RoboTwin env must expose step_lim/take_action_cnt termination counters"
            )
        rng = np.random.default_rng(record.seed)
        initial_forward = getattr(policy, "forward_count", None)

        def record_observation(observation: Any) -> None:
            eef_pose = _jsonable(_pose(observation))
            record.pose_trajectory.append(eef_pose)
            record.eef_trajectory.append(eef_pose)
            for name, pose in _object_poses(env).items():
                record.object_trajectories.setdefault(name, []).append(pose)

        while not bool(getattr(env, "eval_success", False)):
            limit = getattr(env, "step_lim", None)
            count = getattr(env, "take_action_cnt", None)
            if limit is not None and count is not None and int(count) >= int(limit):
                break
            observation = env.get_obs()
            action = self._act(policy, observation, rng)
            record.action_prefix.append(_jsonable(action))
            record_observation(observation)
            env.take_action(action)
            record.env_steps += 1
            if initial_forward is None:
                record.policy_forwards += 1
            else:
                record.policy_forwards = int(
                    getattr(policy, "forward_count", initial_forward)
                ) - int(initial_forward)
        # Persist the terminal observation/object pose as well as the
        # pre-action trajectory, so a successful terminal transition is not
        # omitted from the raw evidence.
        terminal_observation = env.get_obs()
        record_observation(terminal_observation)
        record.final_success = bool(getattr(env, "eval_success", False))
        check = getattr(env, "check_success", None)
        if not record.final_success and callable(check):
            record.final_success = bool(check())
