"""Fail-closed real RoboTwin adapter for the Stage-S substrate screen.

The released Evo-1 RoboTwin policy is server-backed. This module does not
simulate RoboTwin, copy a state from observations, or invent outcomes. A real
candidate family is accepted only when explicit hooks expose simulator state,
policy observation history, action queue, and every RNG stream.
"""

from __future__ import annotations

import copy
import re
import base64
import hashlib
import itertools
import json
import os
import pickle
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


class CapabilityError(RuntimeError):
    """Missing/non-deterministic real-runtime capability, not an episode fail."""


# The released Evo-1 WebSocket payload is intentionally left unchanged.  These
# control messages are opt-in messages on the same connection and are consumed
# only by the Stage-S server shim below.  Keeping the protocol version and
# request id explicit prevents an unpatched/public server from being mistaken
# for an exact-replay server.
EVO_EXACT_REPLAY_PROTOCOL = "r142-evo-exact-replay/v1"
STAGE_S_PROTOCOL_ID = "r142-stage-s-v1"
EVO_CONTROL_KEY = "r142_control"
ROBOTWIN_WORKSPACE_POSE_DIMENSION = 14
ROBOTWIN_WORKSPACE_POSE_SCALE = (
    1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0,
    1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0,
)


def _b64_encode(value: bytes) -> str:
    return base64.b64encode(bytes(value)).decode("ascii")


def _b64_decode(value: Any, label: str) -> bytes:
    if not isinstance(value, str):
        raise CapabilityError(f"{label} must be a base64 string")
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except (ValueError, TypeError) as exc:
        raise CapabilityError(f"{label} is not valid base64") from exc


def _pickle_encode(value: Any) -> str:
    # The control channel is local/trusted and carries Python/NumPy state that
    # cannot be represented as JSON without losing exact dtype/tuple details.
    # It is never used to load a checkpoint or execute user-supplied objects.
    return _b64_encode(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


def _pickle_decode(value: Any, label: str) -> Any:
    try:
        return pickle.loads(_b64_decode(value, label))
    except (pickle.PickleError, EOFError, ValueError, TypeError, AttributeError) as exc:
        raise CapabilityError(f"{label} is not a valid serialized RNG state") from exc


class EvoExactReplayServerControl:
    """Server-side RNG controller for the pinned Evo-1 inference process.

    The public ``Evo_1/scripts/Evo1_server.py`` only dispatches every JSON
    message to ``infer_from_json_dict``.  A real deployment must call
    :meth:`handle_message` before that function and send its non-``None``
    response directly to the same WebSocket.  Ordinary inference messages
    return ``None`` and therefore retain the released model/inference code
    byte-for-byte.  This class changes no model weights, normalization, or
    flow-matching algorithm; it only snapshots/restores the process RNGs.

    ``require_torch=True`` is mandatory for the real server because Evo-1's
    flow-matching head samples its initial action from Torch (and CUDA on the
    A800).  Tests may explicitly disable it when Torch is unavailable.
    """

    def __init__(self, *, require_torch: bool = True):
        self.require_torch = bool(require_torch)

    @staticmethod
    def _torch_or_error(require_torch: bool):
        try:
            import torch
        except ImportError as exc:
            if require_torch:
                raise CapabilityError(
                    "Evo server exact replay requires Torch/CUDA RNG support"
                ) from exc
            return None
        return torch

    @staticmethod
    def _encode_torch_bytes(tensor: Any) -> Dict[str, Any]:
        import numpy as _np

        raw = tensor.detach().to(device="cpu", dtype=tensor.dtype).contiguous()
        # RNG states are ByteTensors.  Rejecting another dtype avoids a silent
        # numeric conversion that would make replay look successful.
        if str(raw.dtype) != "torch.uint8":
            raise CapabilityError(f"Torch RNG state has unexpected dtype {raw.dtype}")
        data = raw.numpy()
        return {
            "dtype": "uint8",
            "shape": list(raw.shape),
            "data": _b64_encode(_np.asarray(data, dtype=_np.uint8).tobytes()),
        }

    @staticmethod
    def _decode_torch_bytes(value: Any, torch: Any, label: str) -> Any:
        import numpy as _np

        if not isinstance(value, Mapping):
            raise CapabilityError(f"{label} must be a mapping")
        if value.get("dtype") != "uint8":
            raise CapabilityError(f"{label} has unsupported dtype")
        shape = value.get("shape")
        if not isinstance(shape, list) or not all(isinstance(x, int) and x >= 0 for x in shape):
            raise CapabilityError(f"{label} has invalid shape")
        raw = _b64_decode(value.get("data"), label)
        expected = int(np.prod(shape, dtype=np.int64))
        if len(raw) != expected:
            raise CapabilityError(
                f"{label} byte length mismatch: expected {expected}, got {len(raw)}"
            )
        array = _np.frombuffer(raw, dtype=_np.uint8).copy().reshape(tuple(shape))
        return torch.from_numpy(array).clone()

    def _capture(self) -> Dict[str, Any]:
        import random

        torch = self._torch_or_error(self.require_torch)
        state: Dict[str, Any] = {
            "protocol": EVO_EXACT_REPLAY_PROTOCOL,
            "python": _pickle_encode(random.getstate()),
            "numpy": _pickle_encode(np.random.get_state()),
        }
        if torch is None:
            state["torch"] = None
            state["torch_cuda"] = []
            state["torch_cuda_available"] = False
        else:
            state["torch"] = self._encode_torch_bytes(torch.get_rng_state())
            cuda_available = bool(torch.cuda.is_available())
            state["torch_cuda_available"] = cuda_available
            state["torch_cuda"] = (
                [self._encode_torch_bytes(item) for item in torch.cuda.get_rng_state_all()]
                if cuda_available
                else []
            )
        return state

    def _restore(self, state: Mapping[str, Any]) -> None:
        import random

        if not isinstance(state, Mapping):
            raise CapabilityError("Evo server RNG restore state must be a mapping")
        if state.get("protocol") != EVO_EXACT_REPLAY_PROTOCOL:
            raise CapabilityError(
                "Evo server RNG restore protocol mismatch; refusing approximate replay"
            )
        random_state = _pickle_decode(state.get("python"), "python RNG state")
        numpy_state = _pickle_decode(state.get("numpy"), "NumPy RNG state")
        torch = self._torch_or_error(self.require_torch)
        torch_state = None
        cuda_states = []
        current_cuda = False
        if torch is not None:
            # Decode and validate every stream before mutating any global RNG;
            # a malformed CUDA state must not leave Python/NumPy half-restored.
            torch_state = self._decode_torch_bytes(
                state.get("torch"), torch, "Torch CPU RNG state"
            )
            captured_cuda = bool(state.get("torch_cuda_available"))
            current_cuda = bool(torch.cuda.is_available())
            if captured_cuda != current_cuda:
                raise CapabilityError(
                    "Torch CUDA availability changed across replay snapshot "
                    f"(captured={captured_cuda}, current={current_cuda})"
                )
            cuda_states_raw = state.get("torch_cuda", [])
            if not isinstance(cuda_states_raw, list):
                raise CapabilityError("Torch CUDA RNG state must be a list")
            if current_cuda:
                current_count = int(torch.cuda.device_count())
                if len(cuda_states_raw) != current_count:
                    raise CapabilityError(
                        "Torch CUDA device count changed across replay snapshot "
                        f"(captured={len(cuda_states_raw)}, current={current_count})"
                    )
                cuda_states = [
                    self._decode_torch_bytes(item, torch, f"Torch CUDA RNG state {i}")
                    for i, item in enumerate(cuda_states_raw)
                ]
            elif cuda_states_raw:
                raise CapabilityError(
                    "Torch CUDA RNG state is present while CUDA is unavailable"
                )
        elif state.get("torch") is not None or state.get("torch_cuda"):
            raise CapabilityError("Torch RNG state cannot be restored without Torch")

        try:
            random.setstate(random_state)
        except (TypeError, ValueError) as exc:
            raise CapabilityError("Python RNG state is not restorable") from exc
        try:
            np.random.set_state(numpy_state)
        except (TypeError, ValueError) as exc:
            raise CapabilityError("NumPy RNG state is not restorable") from exc
        if torch is None:
            return
        try:
            torch.set_rng_state(torch_state)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise CapabilityError("Torch CPU RNG state is not restorable") from exc
        if current_cuda:
            try:
                torch.cuda.set_rng_state_all(cuda_states)
            except (RuntimeError, TypeError, ValueError) as exc:
                raise CapabilityError("Torch CUDA RNG state is not restorable") from exc

    def _set_seed(self, seed: Any) -> Dict[str, Any]:
        import random

        try:
            integer_seed = int(seed)
        except (TypeError, ValueError, OverflowError) as exc:
            raise CapabilityError("Evo server seed must be an integer") from exc
        if integer_seed < 0:
            raise CapabilityError("Evo server seed must be non-negative")
        torch = self._torch_or_error(self.require_torch)
        random.seed(integer_seed)
        np.random.seed(integer_seed % (2**32))
        if torch is not None:
            torch.manual_seed(integer_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(integer_seed)
        return {"protocol": EVO_EXACT_REPLAY_PROTOCOL, "seed": integer_seed}

    def handle_message(self, message: Any) -> Optional[Dict[str, Any]]:
        """Handle one control message, or return ``None`` for inference JSON."""

        if not isinstance(message, Mapping) or EVO_CONTROL_KEY not in message:
            return None
        request_id = message.get("request_id")
        operation = message.get(EVO_CONTROL_KEY)
        response: Dict[str, Any] = {
            EVO_CONTROL_KEY: "ok",
            "protocol": EVO_EXACT_REPLAY_PROTOCOL,
            "request_id": request_id,
            "operation": operation,
        }
        try:
            if message.get("protocol") != EVO_EXACT_REPLAY_PROTOCOL:
                raise CapabilityError("Evo exact-replay control protocol mismatch")
            if not isinstance(request_id, str) or not request_id:
                raise CapabilityError("Evo exact-replay control request_id is required")
            if operation == "set_seed":
                response["state"] = self._set_seed(message.get("seed"))
            elif operation == "capture_rng":
                response["state"] = self._capture()
            elif operation == "restore_rng":
                self._restore(message.get("state"))
            else:
                raise CapabilityError(f"unsupported Evo exact-replay operation: {operation!r}")
        except (CapabilityError, RuntimeError, TypeError, ValueError) as exc:
            response[EVO_CONTROL_KEY] = "error"
            response["error_type"] = type(exc).__name__
            response["error"] = str(exc)
        return response


class EvoExactReplayClient:
    """Synchronous control client layered over the released Evo1Proxy.

    ``Evo1Proxy.infer`` already owns an event loop and WebSocket.  Reusing
    those exact objects avoids changing the inference payload or introducing a
    second connection whose server RNG would not correspond to inference.
    """

    def __init__(self, proxy: Any):
        self.proxy = proxy
        self._request_ids = itertools.count()

    def _request(self, operation: str, **fields: Any) -> Mapping[str, Any]:
        ws = getattr(self.proxy, "ws", None)
        loop = getattr(self.proxy, "loop", None)
        if ws is None or loop is None or not callable(getattr(loop, "run_until_complete", None)):
            raise CapabilityError(
                "Evo exact-replay control requires the proxy's live WebSocket and event loop"
            )
        if bool(getattr(loop, "is_running", lambda: False)()):
            raise CapabilityError(
                "Evo exact-replay control cannot run on an already-running proxy event loop"
            )
        request_id = f"r142-{next(self._request_ids):08d}"
        payload = {
            EVO_CONTROL_KEY: operation,
            "protocol": EVO_EXACT_REPLAY_PROTOCOL,
            "request_id": request_id,
            **fields,
        }

        async def exchange() -> Mapping[str, Any]:
            await ws.send(json.dumps(payload, sort_keys=True, separators=(",", ":")))
            raw = await ws.recv()
            try:
                result = json.loads(raw)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise CapabilityError("Evo server returned non-JSON control response") from exc
            if not isinstance(result, Mapping):
                raise CapabilityError("Evo server control response is not a mapping")
            return result

        try:
            result = loop.run_until_complete(exchange())
        except CapabilityError:
            raise
        except Exception as exc:
            raise CapabilityError(
                "Evo exact-replay control exchange failed; the pinned public server "
                "may be unpatched"
            ) from exc
        if result.get("protocol") != EVO_EXACT_REPLAY_PROTOCOL:
            raise CapabilityError("Evo server did not acknowledge exact-replay protocol")
        if result.get("request_id") != request_id:
            raise CapabilityError("Evo server control response request_id mismatch")
        if result.get(EVO_CONTROL_KEY) != "ok":
            raise CapabilityError(
                "Evo server rejected exact-replay control: "
                f"{result.get('error', 'unknown error')}"
            )
        return result

    def set_seed(self, seed: int) -> Mapping[str, Any]:
        return self._request("set_seed", seed=int(seed))

    def capture_rng_state(self) -> Any:
        return _copy(self._request("capture_rng").get("state"))

    def restore_rng_state(self, state: Mapping[str, Any]) -> None:
        self._request("restore_rng", state=_copy(state))


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



def _numeric_state_leaves(value: Any, path: str = "state") -> Dict[str, np.ndarray]:
    """Extract all numeric leaves from an official simulator snapshot.

    The replay gate must not reduce a simulator snapshot to an observation or
    end-effector pose.  Process-local SAPIEN handles are deliberately skipped,
    while dataclasses/mappings/sequences, poses, arrays, and scalar leaves are
    traversed recursively.  The returned paths are part of the schema check,
    so a missing hidden state also fails closed.
    """
    if value is None or isinstance(value, (str, bytes)):
        return {}
    if isinstance(value, Mapping):
        result: Dict[str, np.ndarray] = {}
        for key, item in value.items():
            if str(key) in {"object", "scene"}:
                continue
            result.update(_numeric_state_leaves(item, f"{path}.{key}"))
        return result
    if hasattr(value, "__dataclass_fields__"):
        result = {}
        for key, item in vars(value).items():
            result.update(_numeric_state_leaves(item, f"{path}.{key}"))
        return result
    if hasattr(value, "p") and hasattr(value, "q"):
        result = {}
        result.update(_numeric_state_leaves(value.p, f"{path}.p"))
        result.update(_numeric_state_leaves(value.q, f"{path}.q"))
        return result
    if hasattr(value, "detach") and callable(value.detach):
        try:
            value = value.detach().cpu().numpy()
        except Exception:
            return {}
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number) or np.issubdtype(value.dtype, np.bool_):
            return {path: np.array(value, copy=True)}
        if value.dtype == object:
            result = {}
            for index, item in enumerate(value.tolist()):
                result.update(_numeric_state_leaves(item, f"{path}[{index}]"))
            return result
        return {}
    if isinstance(value, np.generic):
        return _numeric_state_leaves(np.asarray(value), path)
    if isinstance(value, (tuple, list)):
        result = {}
        for index, item in enumerate(value):
            result.update(_numeric_state_leaves(item, f"{path}[{index}]"))
        return result
    if isinstance(value, (bool, int, float, complex, np.number)):
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_):
            return {path: np.array(array, copy=True)}
    return {}


def _complete_numeric_state_error(first: Any, second: Any, *, tolerance: float = 1e-9) -> float:
    """Compare complete simulator snapshots and reject missing/schema drift."""
    first_leaves = _numeric_state_leaves(first)
    second_leaves = _numeric_state_leaves(second)
    del tolerance
    if not first_leaves or not second_leaves:
        raise CapabilityError(
            "exact replay requires complete simulator snapshots with numeric leaves"
        )
    if set(first_leaves) != set(second_leaves):
        raise CapabilityError(
            "exact replay simulator snapshot schema changed between replays"
        )
    maximum = 0.0
    for path in sorted(first_leaves):
        left = first_leaves[path]
        right = second_leaves[path]
        if left.shape != right.shape:
            raise CapabilityError(
                f"exact replay simulator snapshot shape mismatch at {path}"
            )
        left_integer = np.issubdtype(left.dtype, np.integer) or np.issubdtype(
            left.dtype, np.bool_
        )
        right_integer = np.issubdtype(right.dtype, np.integer) or np.issubdtype(
            right.dtype, np.bool_
        )
        if left_integer or right_integer:
            error = 0.0 if np.array_equal(left, right) else float("inf")
        else:
            if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
                raise CapabilityError(
                    f"exact replay simulator snapshot contains non-finite leaf at {path}"
                )
            error = (
                float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
                if left.size
                else 0.0
            )
        maximum = max(maximum, error)
    return maximum

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
        sim_a = _invoke(self.env, ("capture_simulator_state", "snapshot_simulator", "get_simulator_state"), "simulator next-state")
        self.restore(snap)
        obs_b = observe()
        act_b = _copy(action) if action is not None else act_fn(obs_b)
        self.env.take_action(act_b)
        state_b = observe()
        sim_b = _invoke(self.env, ("capture_simulator_state", "snapshot_simulator", "get_simulator_state"), "simulator next-state")
        action_error = _max_error(act_a, act_b, "action")
        next_state_error = _max_error(state_a, state_b, "next_state")
        simulator_state_error = _complete_numeric_state_error(sim_a, sim_b)
        if max(action_error, next_state_error, simulator_state_error) > self.tolerance:
            raise CapabilityError(
                "exact replay verification failed: "
                f"simulator_state_error={simulator_state_error:.3g}, "
                f"action_error={action_error:.3g}, "
                f"next_state_error={next_state_error:.3g}, "
                f"tolerance={self.tolerance:.3g}"
            )
        return {
            "same_action": True,
            "passed": True,
            "tolerance": self.tolerance,
            "action_error": action_error,
            "simulator_state_error": simulator_state_error,
            "next_state_error": next_state_error,
            # Keep the canonical Stage-S name alongside the historical
            # ``next_state_error`` field.  Finalizers and total analysis bind
            # this value to the frozen <=1e-9 contract.
            "same_action_next_state_max_abs_error": next_state_error,
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

    def __init__(self, proxy: Any, protocol: Optional[EvoExactReplayClient] = None):
        self.proxy = proxy
        self.observation_history = []
        self.action_queue = []
        self.rng = np.random.default_rng()
        # The released proxy has no RNG methods, but it exposes the live
        # ``ws``/``loop`` pair used by infer().  Attach the exact control
        # protocol to that same connection; never create a second socket.
        self.protocol = protocol
        if self.protocol is None and getattr(proxy, "ws", None) is not None:
            self.protocol = EvoExactReplayClient(proxy)

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
        if callable(hook):
            server_state = hook()
        elif self.protocol is not None:
            server_state = self.protocol.capture_rng_state()
        else:
            raise CapabilityError(
                "Evo proxy exact replay unavailable: the released WebSocket "
                "server exposes no Torch/CUDA RNG snapshot hook"
            )
        return {
            "protocol": EVO_EXACT_REPLAY_PROTOCOL if self.protocol is not None else "hook",
            "local": _copy(self.rng.bit_generator.state),
            "server": _copy(server_state),
        }

    def restore_rng_state(self, value: Any) -> None:
        hook = getattr(self.proxy, "restore_rng_state", None)
        if not isinstance(value, Mapping) or "local" not in value or "server" not in value:
            raise CapabilityError("Evo proxy RNG restore state has an invalid schema")
        if callable(hook):
            hook(_copy(value["server"]))
        elif self.protocol is not None:
            self.protocol.restore_rng_state(_copy(value["server"]))
        else:
            raise CapabilityError(
                "Evo proxy exact replay unavailable: server RNG restore hook missing"
            )
        self.rng.bit_generator.state = _copy(value["local"])

    @staticmethod
    def _seed_from_rng(rng: np.random.Generator) -> int:
        """Derive a stable server seed without consuming the caller RNG."""

        state = _jsonable(rng.bit_generator.state)
        encoded = json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
        return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little")

    def set_seed(self, seed: int) -> None:
        try:
            integer_seed = int(seed)
        except (TypeError, ValueError, OverflowError) as exc:
            raise CapabilityError("Evo proxy seed must be an integer") from exc
        if integer_seed < 0:
            raise CapabilityError("Evo proxy seed must be non-negative")
        self.rng = np.random.default_rng(integer_seed)
        hook = getattr(self.proxy, "set_seed", None)
        if not callable(hook):
            hook = getattr(self.proxy, "seed", None)
        if callable(hook):
            try:
                hook(integer_seed)
            except TypeError as exc:
                raise CapabilityError("Evo proxy seed hook must accept an integer") from exc
            return
        if self.protocol is not None:
            self.protocol.set_seed(integer_seed)
            return
        raise CapabilityError(
            "Evo proxy independent sampling unavailable: server policy must expose "
            "set_seed/seed or the Stage-S exact-replay WebSocket protocol"
        )

    def set_rng(self, rng: np.random.Generator) -> None:
        self.rng = rng
        # Local RNG alone cannot control a server-backed policy.  A concrete
        # deploy wrapper must expose a seed hook so candidate sampling is
        # genuinely independent; the public unpatched proxy is therefore
        # rejected here (and also lacks the snapshot hooks required by the
        # replay gate).
        hook = getattr(self.proxy, "set_rng", None)
        if callable(hook):
            try:
                hook(rng)
            except TypeError:
                try:
                    hook(_copy(rng.bit_generator.state))
                except TypeError as exc:
                    raise CapabilityError("Evo proxy set_rng hook rejected Generator/state") from exc
            return
        # Legacy wrappers sometimes expose only seed(seed).  The derivation is
        # stable and does not advance the candidate-local generator.  The real
        # runner prefers set_seed(seed), preserving the registered genealogy
        # seed exactly; this branch is compatibility for generic wrappers.
        self.set_seed(self._seed_from_rng(rng))


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
        sim_a = self.capture_simulator_state()
        self.restore_snapshot(snapshot)
        obs_b = observe()
        action_b = _copy(action) if action is not None else act_fn(obs_b)
        self.task_env.take_action(action_b)
        state_b = observe()
        sim_b = self.capture_simulator_state()
        action_error = _max_error(action_a, action_b, "action")
        next_state_error = _max_error(state_a, state_b, "next_state")
        simulator_state_error = _complete_numeric_state_error(sim_a, sim_b)
        if max(action_error, next_state_error, simulator_state_error) > 1e-9:
            raise CapabilityError(
                "RoboTwin exact replay failed: "
                f"action_error={action_error:.3g}, next_state_error={next_state_error:.3g}"
                f", simulator_state_error={simulator_state_error:.3g}"
            )
        return {
            "same_action": True,
            "passed": True,
            "tolerance": 1e-9,
            "action_error": action_error,
            "next_state_error": next_state_error,
            "simulator_state_error": simulator_state_error,
            "same_action_next_state_max_abs_error": next_state_error,
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape), "data": value.tolist()}
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    # Torch RNG states and model tensors are part of the replay evidence.  A
    # list conversion preserves their exact integer/float payload instead of
    # the non-replayable ``repr(Tensor(...))`` fallback.
    if hasattr(value, "detach") and callable(value.detach):
        tensor = value.detach().cpu()
        return {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "data": tensor.tolist(),
        }
    # SAPIEN Pose exposes p/q arrays but is not JSON serializable itself.
    if hasattr(value, "p") and hasattr(value, "q"):
        return {"p": _jsonable(value.p), "q": _jsonable(value.q)}
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _numeric_jsonable(value: Any) -> Any:
    """Encode action/pose arrays as ordinary numeric JSON arrays.

    Replay snapshots intentionally retain dtype/shape wrappers in ``_jsonable``
    because those fields carry exact RNG/tensor provenance.  Rollout actions
    and trajectories are different: downstream S45 analysis consumes them as
    numeric sequences, so a NumPy ``{dtype, shape, data}`` wrapper would be a
    schema change rather than raw trajectory data.
    """
    if isinstance(value, np.ndarray):
        return [_numeric_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Mapping):
        return {str(key): _numeric_jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_numeric_jsonable(item) for item in value]
    return value


def _numeric_trajectory_value(value: Any, label: str = "trajectory") -> Any:
    """Convert a trajectory leaf into JSON-native numeric nested lists."""
    if hasattr(value, "p") and hasattr(value, "q"):
        value = {"p": getattr(value, "p"), "q": getattr(value, "q")}
    if isinstance(value, Mapping):
        if "p" in value and "q" in value:
            try:
                position = np.asarray(value["p"], dtype=np.float64).reshape(-1)
                quaternion = np.asarray(value["q"], dtype=np.float64).reshape(-1)
            except (TypeError, ValueError) as exc:
                raise CapabilityError(f"{label} pose is not numeric") from exc
            value = np.concatenate((position, quaternion))
        elif "data" in value and set(value).issubset({"dtype", "shape", "data"}):
            return _numeric_trajectory_value(value["data"], label)
        else:
            return {
                str(key): _numeric_trajectory_value(item, f"{label}.{key}")
                for key, item in value.items()
            }
    if hasattr(value, "detach") and callable(value.detach):
        try:
            value = value.detach().cpu().numpy()
        except Exception as exc:
            raise CapabilityError(f"{label} tensor cannot be converted to numeric data") from exc
    if isinstance(value, np.ndarray):
        if not np.issubdtype(value.dtype, np.number) or not np.all(np.isfinite(value)):
            raise CapabilityError(f"{label} must be finite numeric data")
        return _numeric_jsonable(value)
    if isinstance(value, (tuple, list)):
        return [_numeric_trajectory_value(item, f"{label}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, (bool, int, float, np.number)):
        number = np.asarray(value)
        if not np.issubdtype(number.dtype, np.number) or not np.all(np.isfinite(number)):
            raise CapabilityError(f"{label} must be finite numeric data")
        return _numeric_jsonable(number)
    raise CapabilityError(f"{label} has no numeric trajectory representation")


def _snapshot_jsonable(value: Any) -> Any:
    """Serialize a replay snapshot while dropping process-local handles."""
    if hasattr(value, "__dataclass_fields__"):
        return {
            str(key): _snapshot_jsonable(item)
            for key, item in vars(value).items()
        }
    if isinstance(value, Mapping):
        return {
            str(key): _snapshot_jsonable(item)
            for key, item in value.items()
            if key not in {"object", "scene"}
        }
    if isinstance(value, (tuple, list)):
        return [_snapshot_jsonable(item) for item in value]
    return _jsonable(value)


@dataclass
class CandidateRecord:
    candidate_id: str
    parent_id: Optional[str]
    generation_step: int
    action_prefix: list = field(default_factory=list)
    pose_trajectory: list = field(default_factory=list)
    candidate_index: Optional[int] = None
    # ``pose_trajectory`` is retained for compatibility with the initial
    # audit.  These explicit fields make the Stage-S persisted contract
    # unambiguous for downstream analysis.
    eef_trajectory: list = field(default_factory=list)
    object_trajectories: Dict[str, list] = field(default_factory=dict)
    final_success: bool = False
    terminated: bool = False
    termination_reason: Optional[str] = None
    terminal_step: Optional[int] = None
    # ``termination`` is the S45 loader's canonical alias.  Keep
    # ``termination_reason`` as the human-readable producer field too.
    termination: Optional[str] = None
    task_name: str = ""
    family_id: str = ""
    # Explicit root binding is persisted in addition to ``family_id`` so a
    # genealogy row cannot accidentally inherit its root from a directory
    # path during downstream analysis.
    root_family_id: str = ""
    initial_state_id: str = ""
    seed: int = 0
    candidate_seed: Optional[int] = None
    policy_forwards: int = 0
    env_steps: int = 0
    seed_sequence: list = field(default_factory=list)
    seed_genealogy: Dict[str, Any] = field(default_factory=dict)
    # Post-termination policy/runtime state is persisted for every candidate;
    # this is separate from the family-level initial replay snapshot.
    policy_history: Any = None
    action_queue: Any = None
    rng_state: Any = None
    # The snapshot/check are candidate-local evidence.  The family-level
    # SNAPSHOT.json is retained for compatibility, but total analysis must
    # also be able to audit the exact state from which each candidate began.
    snapshot: Any = None
    snapshot_restore_check: Any = None

    def as_dict(self) -> Dict[str, Any]:
        payload = {}
        numeric_fields = {
            "action_prefix",
            "pose_trajectory",
            "eef_trajectory",
            "object_trajectories",
        }
        for key, value in self.__dict__.items():
            if key == "snapshot":
                encoded = _snapshot_jsonable(value)
                if isinstance(encoded, Mapping) and self.snapshot_restore_check is not None:
                    encoded = dict(encoded)
                    encoded["snapshot_restore_check"] = _jsonable(
                        self.snapshot_restore_check
                    )
                payload[key] = encoded
            else:
                payload[key] = (
                    _numeric_jsonable(value) if key in numeric_fields else _jsonable(value)
                )
        return payload


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
            # The manifest is an artifact, not a self-referential line in
            # itself. New markers list payload files in ``files`` and bind
            # the complete SHA256SUMS file separately.
            if "SHA256SUMS" not in files:
                manifest_path = directory / "SHA256SUMS"
                if not manifest_path.is_file():
                    raise ValueError("missing completion manifest: SHA256SUMS")
                manifest_entries = {}
                for line in manifest_path.read_text(encoding="utf-8").splitlines():
                    digest, separator, name = line.partition("  ")
                    if not separator or not name or not re.fullmatch(r"[0-9a-f]{64}", digest):
                        raise ValueError("invalid completion manifest entry")
                    manifest_entries[name] = digest
                expected_entries = {str(name): str(digest) for name, digest in files.items()}
                if manifest_entries != expected_entries:
                    raise ValueError("completion manifest does not match marker files")
                declared_manifest_sha = existing.get("sha256sums_sha256")
                if declared_manifest_sha is not None and hashlib.sha256(
                    manifest_path.read_bytes()
                ).hexdigest() != str(declared_manifest_sha):
                    raise ValueError("completion manifest hash mismatch")
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
        snapshot: Optional[Any] = None,
        logical_family_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        records = list(records)
        for index, record in enumerate(records):
            if record.candidate_index is None:
                # Older local callers did not carry this field.  Materialise
                # the index at the persistence boundary so every producer
                # artifact has an explicit, auditable index.
                record.candidate_index = index
            if int(record.candidate_index) != index:
                raise ValueError(
                    "candidate_index must be contiguous in source order: "
                    f"expected {index}, got {record.candidate_index}"
                )
            if record.termination is None and record.termination_reason:
                record.termination = record.termination_reason
            if (
                not bool(record.terminated)
                or not record.termination_reason
                or record.terminal_step is None
                or int(record.terminal_step) < 0
                or int(record.terminal_step) != int(record.env_steps)
            ):
                raise CapabilityError(
                    "candidate completion requires official termination evidence "
                    "(terminated, termination_reason, terminal_step == env_steps)"
                )
        logical_id = str(logical_family_id or family_id)
        for record in records:
            # Older call sites supplied only ``family_id``/``seed``.  Resolve
            # those aliases at the immutable persistence boundary so every
            # accepted row carries an explicit root and candidate seed.
            if not record.root_family_id:
                record.root_family_id = logical_id
            if record.root_family_id != logical_id:
                raise CapabilityError("candidate root_family_id disagrees with the family root")
            if record.candidate_seed is None:
                record.candidate_seed = int(record.seed)
            if int(record.candidate_seed) != int(record.seed):
                raise CapabilityError("candidate_seed disagrees with seed")
        metadata_payload = dict(metadata or {})
        metadata_payload.setdefault("family_id", logical_id)
        metadata_payload.setdefault("source_family_id", str(family_id))
        directory = self.root / family_id
        existing = self._read_completed(family_id)
        if existing is not None:
            return existing
        payload = {
            "family_id": logical_id,
            "source_family_id": str(family_id),
            "metadata": _jsonable(metadata_payload),
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
        files = {"family.json": result_sha, "genealogy.jsonl": genealogy_sha}
        if snapshot is not None:
            snapshot_data = (
                json.dumps(
                    _snapshot_jsonable(snapshot),
                    sort_keys=True,
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n"
            ).encode()
            files["SNAPSHOT.json"] = self._atomic(
                directory / "SNAPSHOT.json", snapshot_data
            )
        sums_data = ("\n".join(f"{sha}  {name}" for name, sha in files.items()) + "\n").encode()
        sums_sha = self._atomic(directory / "SHA256SUMS", sums_data)
        marker = {
            "family_id": logical_id,
            "source_family_id": str(family_id),
            "candidate_count": len(records),
            "files": files,
            "sha256sums_sha256": sums_sha,
        }
        # Protocol identity and producer shape are duplicated in the
        # immutable marker so S45 can reject a mixed-protocol tree before
        # trusting family.json.
        for key in (
            "protocol_id",
            "protocol_authority_path",
            "protocol_authority_sha256",
            "protocol_git_commit",
            "substrate",
            "pose_dimension",
        ):
            if key in metadata_payload:
                marker[key] = copy.deepcopy(metadata_payload[key])
        marker_data = (
            json.dumps(marker, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
        ).encode()
        marker_sha = self._atomic(directory / "COMPLETED_FAMILY.json", marker_data)
        return {**marker, "completion_sha256": marker_sha, "path": str(directory)}


def _canonical_pose7(value: Any, label: str) -> np.ndarray:
    """Return XYZ + a sign-canonical unit WXYZ quaternion."""

    pose = np.asarray(value, dtype=np.float64).reshape(-1)
    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        raise CapabilityError(f"{label} must be a finite XYZ+WXYZ 7-vector")
    quaternion = pose[3:].copy()
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm <= 0.0:
        raise CapabilityError(f"{label} quaternion must have positive finite norm")
    quaternion /= norm
    # q and -q denote the same orientation.  A deterministic hemisphere
    # prevents an arbitrary sign flip from being counted as trajectory
    # divergence.  Break the w==0 tie by the first non-zero component.
    first_nonzero = next((float(v) for v in quaternion if abs(float(v)) > 1e-15), 0.0)
    if first_nonzero < 0.0:
        quaternion *= -1.0
    return np.concatenate((pose[:3], quaternion))


def _pose(observation: Any) -> Any:
    """Extract real bimanual EEF workspace pose, never joint/qpos state."""

    if isinstance(observation, Mapping):
        if "endpose" in observation:
            endpose = observation["endpose"]
            if isinstance(endpose, Mapping):
                if "left_endpose" not in endpose or "right_endpose" not in endpose:
                    raise CapabilityError(
                        "RoboTwin endpose must contain left_endpose and right_endpose"
                    )
                left = _canonical_pose7(endpose["left_endpose"], "left_endpose")
                right = _canonical_pose7(endpose["right_endpose"], "right_endpose")
                return np.concatenate((left, right))
            # Lightweight unit adapters may expose a direct numeric endpose;
            # the real pinned RoboTwin path above is deliberately stricter.
            direct = np.asarray(endpose, dtype=np.float64).reshape(-1)
            if direct.size == 0 or not np.all(np.isfinite(direct)):
                raise CapabilityError("direct endpose must be a finite non-empty vector")
            return direct
        for key in ("ee_pose", "pose"):
            if key in observation:
                direct = np.asarray(observation[key], dtype=np.float64).reshape(-1)
                if direct.size == 0 or not np.all(np.isfinite(direct)):
                    raise CapabilityError(f"{key} must be a finite non-empty vector")
                return direct
        if "observation" in observation:
            return _pose(observation["observation"])
        if "joint_action" in observation:
            raise CapabilityError(
                "RoboTwin observation exposes joint_action but no EEF workspace pose"
            )
    raise CapabilityError("RoboTwin observation lacks an EEF workspace pose")


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
        return {str(k): _numeric_trajectory_value(v, f"object[{k}]") for k, v in value.items()}
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
        result[name] = _numeric_trajectory_value(get_pose(), f"object[{name}]")
    return result


def _require_pose_dimension(value: Any, expected_dimension: int) -> np.ndarray:
    """Validate the fixed A EEF pose width before persisting a trajectory."""
    try:
        pose = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise CapabilityError("RoboTwin EEF pose is not numeric") from exc
    if pose.size != int(expected_dimension):
        raise CapabilityError(
            f"RoboTwin EEF pose must be {expected_dimension}D, got {pose.size}D"
        )
    if not np.all(np.isfinite(pose)):
        raise CapabilityError("RoboTwin EEF pose contains non-finite values")
    return pose


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
        local_family_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if candidate_count <= 0:
            raise ValueError("candidate_count must be positive")
        storage_family_id = str(local_family_id or family_id)
        existing = self.writer.completed(storage_family_id)
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
        replay_gate = dict(replay.verify_restore())
        replay_gate.setdefault("same_action", True)
        replay_gate.setdefault(
            "same_action_next_state_max_abs_error",
            replay_gate.get("next_state_error"),
        )
        replay.restore(base)
        records = []
        for index in range(candidate_count):
            seed_sequence = np.random.SeedSequence([int(initial_seed), index])
            seed = int(
                seed_sequence.generate_state(1)[0]
            )
            policy = first if index == 0 else self.policy_factory(seed)
            # ``base`` contains the initial simulator and first-policy state.
            # Restore it through the candidate's verifier as well: restoring
            # only through ``replay`` would leave a newly-created candidate
            # with factory-default history/queue/RNG rather than the actual
            # family initial state.
            candidate_replay = ExactReplayVerifier(env, policy)
            candidate_replay.restore(base)
            self._seed_policy(policy, seed)
            # Capture after the candidate seed is installed so the persisted
            # snapshot binds its independent policy RNG to this genealogy row.
            candidate_snapshot = candidate_replay.capture()
            record = CandidateRecord(
                candidate_id=f"{family_id}/candidate-{index:04d}",
                candidate_index=index,
                parent_id=None,
                generation_step=0,
                task_name=task_name,
                family_id=family_id,
                root_family_id=family_id,
                initial_state_id=initial_state_id,
                seed=seed,
                candidate_seed=seed,
                seed_sequence=[int(initial_seed), int(index)],
                seed_genealogy={
                    "root_seed": int(initial_seed),
                    "candidate_index": int(index),
                    "spawn_key": list(seed_sequence.spawn_key),
                },
            )
            self._rollout(env, policy, record)
            if not record.action_prefix:
                raise CapabilityError("candidate produced no action for snapshot replay evidence")
            # Replay the exact first action from the candidate's actual
            # trajectory.  The snapshot is restored before this check, so the
            # check is attached to the same simulator/history/action queue/RNG
            # state that produced the persisted prefix.
            candidate_replay.restore(candidate_snapshot)
            replay_check = dict(
                candidate_replay.verify_restore(action=record.action_prefix[0])
            )
            replay_check.setdefault("same_action", True)
            replay_check.setdefault(
                "same_action_next_state_max_abs_error",
                replay_check.get("next_state_error"),
            )
            record.snapshot = candidate_snapshot
            record.snapshot_restore_check = replay_check
            records.append(record)
        family_metadata = {
            "task_name": task_name,
            "initial_state_id": initial_state_id,
            "initial_seed": int(initial_seed),
            "candidate_count": candidate_count,
            "candidate_rng": "SeedSequence([initial_seed, candidate_index])",
            "termination": "official eval_success or step_lim",
            "replay_capability_gate": replay_gate,
            "genealogy_root": family_id,
            "candidate_snapshot_contract": "per-candidate simulator/history/action_queue/python_numpy_torch_cpu_cuda/environment_policy_rng",
        }
        if metadata is not None:
            for key, value in metadata.items():
                if key in family_metadata and family_metadata[key] != value:
                    raise ValueError(f"family metadata collision for {key!r}")
                family_metadata[key] = copy.deepcopy(value)
        return self.writer.write(
            storage_family_id,
            records,
            snapshot=base,
            metadata=family_metadata,
            logical_family_id=family_id,
        )

    @staticmethod
    def _seed_policy(policy: Any, seed: int) -> None:
        # Prefer an explicit integer seed so the server-side Torch/CUDA stream
        # is tied exactly to the persisted SeedSequence genealogy.  The
        # Generator path remains for local/fake policy adapters only.
        fn = getattr(policy, "seed", None)
        if callable(fn):
            fn(int(seed))
            return
        fn = getattr(policy, "set_rng", None)
        if callable(fn):
            fn(np.random.default_rng(seed))
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
            eef_pose = _numeric_jsonable(_require_pose_dimension(_pose(observation), 14))
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
            record.action_prefix.append(_numeric_jsonable(action))
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
        if record.final_success:
            record.termination_reason = "official_eval_success"
        elif int(getattr(env, "take_action_cnt")) >= int(getattr(env, "step_lim")):
            record.termination_reason = "official_step_limit"
        else:
            raise CapabilityError(
                "RoboTwin rollout exited without official success or step-limit termination"
            )
        record.terminated = True
        record.termination = record.termination_reason
        record.terminal_step = int(record.env_steps)
        # Preserve the exact policy history, queued action suffix, and both
        # runtime/policy RNG streams after termination.  These are raw replay
        # evidence, not derived metrics.
        record.policy_history = _snapshot_jsonable(
            _invoke(
                policy,
                ("capture_observation_history", "snapshot_observation_history"),
                "policy observation history",
            )
        )
        record.action_queue = _snapshot_jsonable(
            _invoke(
                policy,
                ("capture_action_queue", "snapshot_action_queue"),
                "policy action queue",
            )
        )
        record.rng_state = {
            "environment": _snapshot_jsonable(
                _invoke(
                    env,
                    ("capture_rng_state", "snapshot_rng", "get_rng_state"),
                    "environment RNG streams",
                )
            ),
            "policy": _snapshot_jsonable(
                _invoke(
                    policy,
                    ("capture_rng_state", "snapshot_rng", "get_rng_state"),
                    "policy RNG streams",
                )
            ),
        }
