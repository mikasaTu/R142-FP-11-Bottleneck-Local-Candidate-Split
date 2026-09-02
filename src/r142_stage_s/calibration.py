"""Leakage-resistant Step-0 calibration records.

Stage-S Step 0 is allowed to select a substrate setting using pooled
eventual success only. This module makes that restriction executable: any
field whose name can carry an S2--S5 statistic is rejected before a record can
be persisted or selected.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

CALIBRATION_SCHEMA = "r142-stage-s-calibration-v1"
CALIBRATION_TARGET = 0.45

# These are deliberately broad. A false rejection invalidates a calibration
# attempt and is visible; silently allowing a diagnostic field would leak the
# outcome under test.
_FORBIDDEN_TOKENS = (
    "all_fail",
    "near_all_fail",
    "strict_zero",
    "zero_fail",
    "rho",
    "overdisp",
    "binomial",
    "divergence",
    "t_div",
    "tau",
    "recover",
    "oracle",
    "random_probe",
    "bootstrap",
    "rescue",
    "branch",
    "mode_discovery",
    "mode_rate",
    "s2",
    "s3",
    "s4",
    "s5",
)

_ALLOWED_ROW_FIELDS = {
    "success",
    "setting",
    "setting_id",
    "candidate_id",
    "task_id",
    "init_state",
    "initial_state",
    "checkpoint_step",
    "magnitude",
}


def _normalise_key(key: object) -> str:
    return str(key).strip().lower().replace("-", "_").replace(" ", "_")


def _reject_forbidden(value: object, path: str = "payload") -> None:
    """Reject keys that could encode any prohibited calibration statistic."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            normalised = _normalise_key(key)
            if any(token in normalised for token in _FORBIDDEN_TOKENS):
                raise ValueError(f"calibration payload contains prohibited field {path}.{key}")
            _reject_forbidden(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_forbidden(child, f"{path}[{index}]")


def _success_values(values: Iterable[object]) -> list[bool]:
    result: list[bool] = []
    for index, value in enumerate(values):
        if isinstance(value, Mapping):
            unknown = {_normalise_key(key) for key in value} - _ALLOWED_ROW_FIELDS
            if unknown:
                raise ValueError(f"calibration row {index} has non-success fields: {sorted(unknown)}")
            if "success" not in value:
                raise ValueError(f"calibration row {index} lacks success")
            value = value["success"]
        if isinstance(value, bool):
            result.append(bool(value))
        elif isinstance(value, (int, float)) and value in (0, 1):
            result.append(bool(value))
        else:
            raise TypeError(f"calibration success at index {index} must be bool or 0/1")
    if not result:
        raise ValueError("calibration pilot cannot be empty")
    return result


def make_calibration_record(
    setting_id: object,
    successes: Iterable[object],
    *,
    target: float = CALIBRATION_TARGET,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a Step-0 record whose only outcome statistic is pooled success.

    The context may identify the task/checkpoint/magnitude, but it is checked
    recursively and cannot contain any S2--S5 field. The returned record is
    safe to write as JSON and is intentionally not accepted by the main gate
    functions as a substitute for rollout data.
    """

    if not 0.0 <= float(target) <= 1.0:
        raise ValueError("calibration target must lie in [0, 1]")
    if context is not None:
        _reject_forbidden(context, "context")
        unknown = {_normalise_key(key) for key in context} - _ALLOWED_ROW_FIELDS
        if unknown:
            raise ValueError(f"calibration context has unsupported fields: {sorted(unknown)}")
    values = _success_values(successes)
    success_count = int(sum(values))
    record: dict[str, Any] = {
        "schema": CALIBRATION_SCHEMA,
        "setting_id": str(setting_id),
        "sample_count": len(values),
        "success_count": success_count,
        "pooled_success": float(success_count / len(values)),
        "selection_target": float(target),
        "selection_statistic": "pooled_success_only",
    }
    if context:
        record["context"] = dict(context)
    _reject_forbidden(record)
    return record


def validate_calibration_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a persisted record and return a JSON-safe copy."""

    _reject_forbidden(record)
    if record.get("schema") != CALIBRATION_SCHEMA:
        raise ValueError("unsupported calibration schema")
    required = {"schema", "setting_id", "sample_count", "success_count", "pooled_success"}
    missing = required - set(record)
    if missing:
        raise ValueError(f"calibration record missing fields: {sorted(missing)}")
    sample_count = int(record["sample_count"])
    success_count = int(record["success_count"])
    pooled_success = float(record["pooled_success"])
    if sample_count <= 0 or not 0 <= success_count <= sample_count:
        raise ValueError("invalid calibration counts")
    if abs(pooled_success - success_count / sample_count) > 1e-12:
        raise ValueError("pooled_success does not match counts")
    if not 0.0 <= pooled_success <= 1.0:
        raise ValueError("pooled_success must lie in [0, 1]")
    allowed_top = required | {"selection_target", "selection_statistic", "context"}
    unknown = set(record) - allowed_top
    if unknown:
        raise ValueError(f"unsupported calibration fields: {sorted(unknown)}")
    if "context" in record:
        context = record["context"]
        if not isinstance(context, Mapping):
            raise TypeError("calibration context must be a mapping")
        _reject_forbidden(context, "context")
    return json.loads(json.dumps(dict(record), ensure_ascii=False))


def select_calibration_setting(
    records: Sequence[Mapping[str, Any]], *, target: float = CALIBRATION_TARGET
) -> dict[str, Any]:
    """Select the closest pooled-success setting with a deterministic tie-break."""

    if not records:
        raise ValueError("at least one calibration record is required")
    validated = [validate_calibration_record(record) for record in records]
    target = float(target)
    if not 0.0 <= target <= 1.0:
        raise ValueError("calibration target must lie in [0, 1]")
    selected = min(
        validated,
        key=lambda row: (abs(float(row["pooled_success"]) - target), str(row["setting_id"])),
    )
    output = dict(selected)
    output["selection_target"] = target
    output["selection_statistic"] = "pooled_success_only"
    output["selected"] = True
    output["selection_distance"] = abs(float(output["pooled_success"]) - target)
    # Verify the derived output itself remains leakage-free.
    _reject_forbidden({key: value for key, value in output.items() if key != "selected"})
    return output


def persist_calibration_report(
    path: str | Path,
    records: Sequence[Mapping[str, Any]],
    *,
    target: float = CALIBRATION_TARGET,
) -> dict[str, Any]:
    """Atomically persist a JSON report containing pooled-success records only."""

    validated = [validate_calibration_record(row) for row in records]
    selected = select_calibration_setting(validated, target=target)
    payload = {
        "schema": CALIBRATION_SCHEMA,
        "selection_statistic": "pooled_success_only",
        "selection_target": float(target),
        "records": validated,
        "selected": selected,
    }
    _reject_forbidden(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)
    directory_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return payload


__all__ = [
    "CALIBRATION_SCHEMA",
    "CALIBRATION_TARGET",
    "make_calibration_record",
    "persist_calibration_report",
    "select_calibration_setting",
    "validate_calibration_record",
]
