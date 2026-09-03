"""Fail-closed replay comparisons for Stage-S substrate adapters.

The S4/S5 replay gate is stronger than comparing an exposed observation or
end-effector pose. This module keeps the comparison independent of either
simulator API and checks nested structure, every numeric leaf, observation
history, queued actions, and all RNG owners. Process-local simulator handles
are omitted only when they are carried under the explicitly documented
``object``/``scene`` keys; an unknown opaque value is retained in the
structure signature so a schema drift cannot pass silently.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np


class ReplayIntegrityError(ValueError):
    """A replay component is absent, structurally different, or non-exact."""


_OPAQUE_SIMULATOR_KEYS = frozenset({"object", "scene"})


def _is_tensor(value: Any) -> bool:
    return hasattr(value, "detach") and callable(value.detach)


def _tensor_array(value: Any) -> np.ndarray:
    try:
        return np.asarray(value.detach().cpu().numpy())
    except Exception as exc:  # noqa: BLE001 - preserve the replay cause
        raise ReplayIntegrityError("replay component contains an unreadable tensor") from exc


def _is_numeric_dtype(dtype: np.dtype[Any]) -> bool:
    return bool(np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_))


def _map_key(key: Any) -> tuple[str, str]:
    """Make mapping-key structure deterministic without conflating key types."""

    return (type(key).__qualname__, str(key))


def _structural_signature(value: Any, path: str, *, skip_keys: frozenset[str]) -> Any:
    """Return a deterministic schema/value signature for nested replay data.

    Numeric values are represented by dtype/shape only and are compared by
    :func:`_numeric_leaves`. Nonnumeric values are included verbatim. Opaque
    simulator handles retain their type but not process-local identity.
    """

    # ``path`` is currently retained only to make recursive diagnostics
    # extensible; numeric leaf paths are assigned in ``_numeric_leaves``.
    if _is_tensor(value):
        value = _tensor_array(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = []
        for field in dataclasses.fields(value):
            if field.name in skip_keys:
                continue
            fields.append((field.name, _structural_signature(getattr(value, field.name), field.name, skip_keys=skip_keys)))
        return ("dataclass", type(value).__module__, type(value).__qualname__, tuple(fields))
    if isinstance(value, Mapping):
        fields = []
        for key in sorted(value, key=lambda item: _map_key(item)):
            if str(key) in skip_keys:
                continue
            fields.append((_map_key(key), _structural_signature(value[key], str(key), skip_keys=skip_keys)))
        return ("mapping", tuple(fields))
    if isinstance(value, np.ndarray):
        if _is_numeric_dtype(value.dtype):
            return ("array", str(value.dtype), tuple(int(dim) for dim in value.shape))
        if value.dtype == object:
            children = tuple(
                _structural_signature(item, f"[{index}]", skip_keys=skip_keys)
                for index, item in enumerate(value.tolist())
            )
            return ("object_array", str(value.dtype), tuple(int(dim) for dim in value.shape), children)
        return ("array", str(value.dtype), tuple(int(dim) for dim in value.shape))
    if isinstance(value, np.generic):
        return _structural_signature(np.asarray(value), path, skip_keys=skip_keys)
    if isinstance(value, (list, tuple)):
        return (
            type(value).__name__,
            tuple(_structural_signature(item, f"[{index}]", skip_keys=skip_keys) for index, item in enumerate(value)),
        )
    if isinstance(value, (bool, int, float, complex)):
        array = np.asarray(value)
        if _is_numeric_dtype(array.dtype):
            return ("scalar", str(array.dtype))
    if value is None:
        return ("none",)
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, bytes):
        return ("bytes", value.hex())
    # A process-local object is not numerically comparable, but its type is a
    # structural part of the snapshot. Documented handles are skipped above.
    return ("opaque", type(value).__module__, type(value).__qualname__)


def _numeric_leaves(value: Any, path: str, *, skip_keys: frozenset[str]) -> dict[str, np.ndarray]:
    """Collect all numeric leaves with stable paths and copied values."""

    if value is None or isinstance(value, (str, bytes)):
        return {}
    if _is_tensor(value):
        value = _tensor_array(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        result: dict[str, np.ndarray] = {}
        for field in dataclasses.fields(value):
            if field.name in skip_keys:
                continue
            result.update(_numeric_leaves(getattr(value, field.name), f"{path}.{field.name}", skip_keys=skip_keys))
        return result
    if isinstance(value, Mapping):
        result = {}
        for key in sorted(value, key=lambda item: _map_key(item)):
            if str(key) in skip_keys:
                continue
            result.update(_numeric_leaves(value[key], f"{path}.{key}", skip_keys=skip_keys))
        return result
    if isinstance(value, np.ndarray):
        if _is_numeric_dtype(value.dtype):
            return {path: np.array(value, copy=True)}
        if value.dtype == object:
            result = {}
            for index, item in enumerate(value.tolist()):
                result.update(_numeric_leaves(item, f"{path}[{index}]", skip_keys=skip_keys))
            return result
        return {}
    if isinstance(value, np.generic):
        return _numeric_leaves(np.asarray(value), path, skip_keys=skip_keys)
    if isinstance(value, (list, tuple)):
        result = {}
        for index, item in enumerate(value):
            result.update(_numeric_leaves(item, f"{path}[{index}]", skip_keys=skip_keys))
        return result
    if isinstance(value, (bool, int, float, complex)):
        array = np.asarray(value)
        if _is_numeric_dtype(array.dtype):
            return {path: np.array(array, copy=True)}
    return {}


def _schema_digest(signature: Any) -> str:
    payload = json.dumps(signature, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _numeric_digest(leaves: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for path in sorted(leaves):
        array = np.ascontiguousarray(leaves[path])
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(repr(tuple(int(dim) for dim in array.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _leaf_error(left: np.ndarray, right: np.ndarray, path: str) -> float:
    if left.shape != right.shape:
        raise ReplayIntegrityError(f"replay component shape mismatch at {path}: {left.shape} != {right.shape}")
    left_integer = np.issubdtype(left.dtype, np.integer) or np.issubdtype(left.dtype, np.bool_)
    right_integer = np.issubdtype(right.dtype, np.integer) or np.issubdtype(right.dtype, np.bool_)
    if left_integer or right_integer:
        return 0.0 if np.array_equal(left, right) else float("inf")
    if not _is_numeric_dtype(left.dtype) or not _is_numeric_dtype(right.dtype):
        return 0.0 if np.array_equal(left, right) else float("inf")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ReplayIntegrityError(f"replay component contains non-finite numeric leaf at {path}")
    if left.size == 0:
        return 0.0
    if np.issubdtype(left.dtype, np.complexfloating) or np.issubdtype(right.dtype, np.complexfloating):
        return float(np.max(np.abs(left.astype(np.complex128) - right.astype(np.complex128))))
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def compare_replay_component(
    first: Any,
    second: Any,
    label: str,
    *,
    tolerance: float = 1e-9,
    skip_keys: frozenset[str] = frozenset(),
    require_numeric: bool = False,
) -> dict[str, Any]:
    """Compare one replay component and return persisted, detailed evidence.

    Structural differences raise immediately (fail closed), while numeric
    differences are summarized by maximum error and changed paths. A simulator
    snapshot must expose at least one numeric leaf; empty histories and queues
    are valid components and may have zero leaves.
    """

    first_signature = _structural_signature(first, label, skip_keys=skip_keys)
    second_signature = _structural_signature(second, label, skip_keys=skip_keys)
    if first_signature != second_signature:
        raise ReplayIntegrityError(f"{label} structural schema changed between replays")
    first_leaves = _numeric_leaves(first, label, skip_keys=skip_keys)
    second_leaves = _numeric_leaves(second, label, skip_keys=skip_keys)
    if require_numeric and (not first_leaves or not second_leaves):
        raise ReplayIntegrityError(f"{label} requires complete numeric leaves")
    if set(first_leaves) != set(second_leaves):
        raise ReplayIntegrityError(f"{label} numeric leaf paths changed between replays")
    maximum = 0.0
    changed_paths: list[str] = []
    for path in sorted(first_leaves):
        error = _leaf_error(first_leaves[path], second_leaves[path], path)
        maximum = max(maximum, error)
        if error > 0.0:
            changed_paths.append(path)
    if maximum > float(tolerance):
        raise ReplayIntegrityError(f"{label} max numeric error {maximum} exceeds {tolerance}")
    return {
        "passed": True,
        "max_abs_error": float(maximum),
        "tolerance": float(tolerance),
        "numeric_leaf_count": int(len(first_leaves)),
        "numeric_leaf_paths": sorted(first_leaves),
        "changed_numeric_paths": changed_paths,
        "schema_sha256": _schema_digest(first_signature),
        "first_numeric_sha256": _numeric_digest(first_leaves),
        "second_numeric_sha256": _numeric_digest(second_leaves),
    }


def compare_replay_components(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    tolerance: float = 1e-9,
    simulator_key: str = "simulator_snapshot",
) -> dict[str, Any]:
    """Compare the complete LIBERO replay state and aggregate evidence."""

    required = (
        simulator_key,
        "observation_history",
        "action_queue",
        "python_rng",
        "numpy_rng",
        "torch_rng",
        "environment_owner_rng",
        "policy_owner_rng",
    )
    missing = [key for key in required if key not in first or key not in second]
    if missing:
        raise ReplayIntegrityError(f"replay evidence is missing required components: {', '.join(missing)}")
    evidence: dict[str, Any] = {}
    for key in required:
        evidence[key] = compare_replay_component(
            first[key],
            second[key],
            key,
            tolerance=tolerance,
            skip_keys={"object", "scene"} if key == simulator_key else frozenset(),
            require_numeric=key == simulator_key,
        )
    evidence["max_abs_error"] = float(max(item["max_abs_error"] for item in evidence.values()))
    evidence["passed"] = bool(evidence["max_abs_error"] <= float(tolerance))
    evidence["tolerance"] = float(tolerance)
    return evidence


__all__ = ["ReplayIntegrityError", "compare_replay_component", "compare_replay_components"]
