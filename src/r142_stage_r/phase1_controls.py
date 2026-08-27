from __future__ import annotations

"""The registered Phase-1R positive and null control diagnostics.

These controls are intentionally tiny and deterministic.  They exercise the
same snapshot, branching, cell, SHA, and analysis path as natural LIBERO
collection, but are never presented as evidence about the learned policy.
"""

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .phase1 import (
    BRANCH_ACTIONS,
    DESCENDANTS_PER_CELL,
    LOCATIONS_PER_EPISODE,
    PHASE1_PROTOCOL_ID,
    STREAMS,
    collect_branch_cell,
    decile_steps,
    replay_baseline_snapshot,
    sha256_int,
    write_cell,
    _atomic_text,
)
from .protocol import atomic_json, sha256_file


CONTROL_NAMES = {
    "positive": "GeometricCommit2D-Phase1R-v1",
    "null": "OpenPlane2D-Phase1R-v1",
}

# A single, pre-registered baseline seed lets the positive-control policy make
# a deterministic *wrong* baseline choice.  Fresh descendant seeds are all
# different from this sentinel and use the fixed 3/4-versus-1/4 lane split.
BASELINE_SENTINEL_SEED = sha256_int("R142-FP-11|Phase1R|GeometricCommit2D|baseline-sentinel")
POSITIVE_COMMIT_STEP = 32


@dataclass
class _GeoState:
    position: np.ndarray
    velocity: np.ndarray
    committed: int
    pending_lane: int
    initial_state: int
    step: int


class Phase1GeometricEnvironment:
    """A snapshot-capable two-dimensional control environment."""

    horizon = 80
    dt = 0.04

    def __init__(self, kind: str):
        if kind not in CONTROL_NAMES:
            raise ValueError(kind)
        self.kind = kind
        self._state: _GeoState | None = None
        self._rng = np.random.default_rng(0)
        self._branch_mode = False

    def seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(int(seed))

    def reset(self, initial_state: int) -> dict[str, np.ndarray]:
        state = int(initial_state)
        self._branch_mode = False
        self._state = _GeoState(
            position=np.asarray([-1.0, 0.015 * (state - 5.5)], dtype=np.float64),
            velocity=np.zeros(2, dtype=np.float64),
            committed=0,
            pending_lane=0,
            initial_state=state,
            step=0,
        )
        return self.raw_observation()

    def _require_state(self) -> _GeoState:
        if self._state is None:
            raise RuntimeError("environment has not been reset")
        return self._state

    def raw_observation(self) -> dict[str, np.ndarray]:
        state = self._require_state()
        return {
            "position": state.position.copy(),
            "velocity": state.velocity.copy(),
            "committed": np.asarray([state.committed], dtype=np.float64),
            "branch_mode": np.asarray([int(self._branch_mode)], dtype=np.float64),
            # The pinned control task exposes the lane affordance explicitly;
            # this keeps the baseline deterministic enough to traverse the
            # correct lane while branch seeds still control the deliberate
            # pre-commit ambiguity below.
            "allowed_lane": np.asarray([self._allowed_lane()], dtype=np.float64),
        }

    def state_vector(self) -> np.ndarray:
        state = self._require_state()
        return np.asarray(
            [state.position[0], state.position[1], state.velocity[0], state.velocity[1], state.committed, state.step],
            dtype=np.float64,
        )

    def capture_snapshot(self) -> dict[str, Any]:
        state = self._require_state()
        return {
            "position": state.position.copy(),
            "velocity": state.velocity.copy(),
            "committed": int(state.committed),
            "pending_lane": int(state.pending_lane),
            "initial_state": int(state.initial_state),
            "step": int(state.step),
            "branch_mode": bool(self._branch_mode),
            "rng_state": copy.deepcopy(self._rng.bit_generator.state),
        }

    def restore_snapshot(self, snapshot: dict[str, Any]) -> None:
        self._state = _GeoState(
            position=np.asarray(snapshot["position"], dtype=np.float64).copy(),
            velocity=np.asarray(snapshot["velocity"], dtype=np.float64).copy(),
            committed=int(snapshot["committed"]),
            pending_lane=int(snapshot.get("pending_lane", 0)),
            initial_state=int(snapshot["initial_state"]),
            step=int(snapshot["step"]),
        )
        self._branch_mode = bool(snapshot.get("branch_mode", False))
        self._rng = np.random.default_rng()
        self._rng.bit_generator.state = copy.deepcopy(snapshot["rng_state"])

    def begin_branch(self) -> None:
        """Mark a restored snapshot as a fresh descendant rollout.

        The positive control deliberately gives the frozen baseline an oracle
        lane affordance while forcing descendants to resample the lane at the
        intervention point.  This hook is optional and is ignored by natural
        environment adapters.
        """

        self._branch_mode = True

    def _allowed_lane(self) -> int:
        state = self._require_state()
        return 1 if int(state.initial_state) % 4 in {0, 1} else -1

    def official_predicates(self) -> dict[str, float]:
        state = self._require_state()
        if self.kind == "positive":
            progress_x = np.clip((float(state.position[0]) + 1.0) / 2.0, 0.0, 1.0)
            lane_progress = 1.0 if state.committed else 0.0
            return {"fraction": float(0.7 * progress_x + 0.3 * lane_progress)}
        progress_x = np.clip((float(state.position[0]) + 1.0) / 2.0, 0.0, 1.0)
        return {"fraction": float(progress_x)}

    def execute_actions(self, action_batch: np.ndarray) -> dict[str, Any]:
        state = self._require_state()
        action = np.asarray(action_batch, dtype=np.float64)
        if action.ndim == 2:
            action = action[0]
        if action.ndim != 1 or action.shape[0] < 2:
            raise ValueError(f"expected action shape (>=2,), got {action.shape}")
        action = action[:2]
        state.velocity = 0.68 * state.velocity + 0.32 * action
        proposed = state.position + self.dt * state.velocity
        if self.kind == "positive":
            # Registered positive control: the middle decision is an explicit
            # irreversible state transition.  Before COMMIT_STEP the latest
            # signed lateral chunk is only pending; on that step it is frozen
            # into ``committed``.  A wrong lane can never be repaired later.
            attempted_lane = 1 if action[1] >= 0.0 else -1
            if state.step < POSITIVE_COMMIT_STEP:
                state.pending_lane = attempted_lane
                if state.step + 1 >= POSITIVE_COMMIT_STEP:
                    state.committed = attempted_lane
            if state.committed:
                proposed[1] = float(state.committed) * max(0.18, abs(float(proposed[1])))
                proposed[0] = max(proposed[0], 0.101)
        state.position = proposed
        state.step += 1
        if self.kind == "positive":
            # Success is intentionally a pure terminal predicate of the
            # committed lane, independent of how far the position moved.
            success = bool(state.committed == self._allowed_lane())
        else:
            success = bool(np.linalg.norm(state.position - np.asarray([1.0, 0.0])) <= 0.45)
        done = bool(success or state.step >= self.horizon)
        return {"success": success, "done": done, "progress": self.official_predicates()["fraction"]}

    def close(self) -> None:
        return None


class Phase1GeometricPolicy:
    """A fixed, seed-addressable policy used only by control diagnostics."""

    def __init__(self, kind: str):
        if kind not in CONTROL_NAMES:
            raise ValueError(kind)
        self.kind = kind

    def sample_action_chunk(self, observation: Any, *, seed: int, counter: int) -> np.ndarray:
        rng = np.random.default_rng(int(seed) ^ (int(counter) * 0x9E3779B1))
        position = np.asarray(observation["position"], dtype=np.float64) if isinstance(observation, dict) else np.zeros(2)
        if self.kind == "positive":
            # Positive control: the pinned task provides the correct lane,
            # but every freshly sampled pre-commit chunk has a substantial
            # independent lateral perturbation.  Thus a branch can discover
            # the correct lane before the irreversible barrier, whereas the
            # baseline still has a high-probability, reproducible crossing.
            allowed = int(np.asarray(observation.get("allowed_lane", [1])).reshape(-1)[0]) if isinstance(observation, dict) else 1
            committed = int(np.asarray(observation.get("committed", [0])).reshape(-1)[0]) if isinstance(observation, dict) else 0
            if committed:
                # Once the one-way lane has been committed, steer to the
                # target instead of continuing to accumulate lateral drift.
                # This makes the positive control a genuine irreversible
                # mechanism rather than an artefact of an unreachable target.
                target_y = 0.48 * committed
                lateral = float(np.clip(1.35 * (target_y - position[1]), -0.55, 0.55))
            else:
                # The baseline is pinned to one wrong lane.  Descendants use
                # a deterministic 75% correct / 25% wrong split keyed only
                # by their fresh branch seed; this is the registered control
                # mechanism rather than a post-hoc fit to observed outcomes.
                if int(seed) == BASELINE_SENTINEL_SEED:
                    sampled_lane = -allowed
                else:
                    sampled_lane = allowed if sha256_int(f"{int(seed)}|positive-lane") % 4 != 0 else -allowed
                lateral = float(sampled_lane * 0.62)
            actions = np.tile(np.asarray([0.68, lateral], dtype=np.float32), (BRANCH_ACTIONS, 1))
        else:
            lateral = float(-0.55 * position[1] + rng.normal(0.0, 0.035))
            actions = np.tile(np.asarray([0.68, lateral], dtype=np.float32), (BRANCH_ACTIONS, 1))
        return actions


def _control_rows(kind: str, episodes: int) -> list[dict[str, Any]]:
    environment = Phase1GeometricEnvironment(kind)
    policy = Phase1GeometricPolicy(kind)
    rows = []
    try:
        for episode in range(int(episodes)):
            init_state = int(episode % 12)
            seed = (
                BASELINE_SENTINEL_SEED
                if kind == "positive"
                else sha256_int(f"{PHASE1_PROTOCOL_ID}|control|{kind}|episode|{episode}")
            )
            environment.seed(seed)
            observation = environment.reset(init_state)
            queue: list[np.ndarray] = []
            counter = 0
            actions: list[np.ndarray] = []
            success = False
            done = False
            while not done and len(actions) < Phase1GeometricEnvironment.horizon:
                if not queue:
                    queue.extend(policy.sample_action_chunk(observation, seed=seed, counter=counter))
                    counter += 1
                action = np.asarray(queue.pop(0), dtype=np.float32)
                result = environment.execute_actions(action)
                actions.append(action.copy())
                success = bool(result["success"])
                done = bool(result["done"])
                observation = environment.raw_observation()
            if not done:
                raise RuntimeError("control baseline exceeded horizon")
            rows.append(
                {
                    "init_state": init_state,
                    "candidate_id": episode,
                    "rollout_seed": seed,
                    "actions": np.asarray(actions, dtype=np.float32),
                    "success": success,
                }
            )
    finally:
        environment.close()
    return rows


def collect_control_bundle(
    kind: str,
    output_root: str | Path,
    *,
    episodes: int = 12,
    locations: int = LOCATIONS_PER_EPISODE,
    descendants: int = DESCENDANTS_PER_CELL,
    streams: Iterable[str] = STREAMS,
) -> dict[str, Any]:
    """Generate a complete 12x10x16 control bundle for one control kind."""

    if kind not in CONTROL_NAMES:
        raise ValueError(kind)
    if int(episodes) != 12 or int(locations) != 10 or int(descendants) != 16:
        raise ValueError("registered control bundle is fixed at 12x10x16")
    selected_streams = tuple(streams)
    if set(selected_streams) != set(STREAMS):
        raise ValueError("both calibration and heldout streams are required")
    root = Path(output_root)
    environment = Phase1GeometricEnvironment(kind)
    policy = Phase1GeometricPolicy(kind)
    completed = []
    try:
        for episode, row in enumerate(_control_rows(kind, episodes)):
            parent = f"{kind}_episode{episode:02d}"
            baseline_length = len(np.asarray(row["actions"]))
            for location, step in enumerate(decile_steps(baseline_length)):
                snapshot, _ = replay_baseline_snapshot(environment, policy, f"control_{kind}", 0, row, step)
                for stream in selected_streams:
                    directory = root / kind / f"episode{episode:02d}" / f"location{location:02d}" / stream
                    traces = collect_branch_cell(
                        environment,
                        policy,
                        snapshot=snapshot,
                        suite=f"control_{kind}",
                        task_id=0,
                        parent_id=parent,
                        generation_step=step,
                        stream=stream,
                        location_index=location,
                        descendants=descendants,
                        max_steps=Phase1GeometricEnvironment.horizon,
                    )
                    metadata = write_cell(
                        directory,
                        traces,
                        suite=f"control_{kind}",
                        task_id=0,
                        parent_id=parent,
                        selection_index=episode,
                        location_index=location,
                        decile=(location + 1) / 10.0,
                        generation_step=step,
                        stream=stream,
                        baseline_length=baseline_length,
                        metadata_extra={
                            "control_name": CONTROL_NAMES[kind],
                            "control_kind": kind,
                            "baseline_sentinel_seed": int(BASELINE_SENTINEL_SEED) if kind == "positive" else None,
                            "commit_step": POSITIVE_COMMIT_STEP if kind == "positive" else None,
                            "positive_lane_split": "3/4_correct_1/4_wrong" if kind == "positive" else None,
                            "success_rule": "committed_lane_equals_allowed_lane" if kind == "positive" else "open_plane_target",
                        },
                    )
                    # Keep the bundle manifest portable across CPFS mount
                    # points; absolute paths are not part of the protocol.
                    completed.append(str(directory.relative_to(root)))
    finally:
        environment.close()
    summary = {
        "protocol_id": PHASE1_PROTOCOL_ID,
        "control_kind": kind,
        "control_name": CONTROL_NAMES[kind],
        "episodes": int(episodes),
        "locations": int(locations),
        "descendants": int(descendants),
        "streams": list(selected_streams),
        "cell_count": len(completed),
        "cells": completed,
        "baseline_sentinel_seed": int(BASELINE_SENTINEL_SEED) if kind == "positive" else None,
        "commit_step": POSITIVE_COMMIT_STEP if kind == "positive" else None,
        "positive_lane_split": "3/4_correct_1/4_wrong" if kind == "positive" else None,
    }
    summary_path = root / kind / "CONTROL_BUNDLE.json"
    atomic_json(summary_path, summary)
    atomic_json(
        root / kind / "COMPLETED_CONTROL.json",
        {
            "schema_version": 1,
            "protocol_id": PHASE1_PROTOCOL_ID,
            "marker_type": "control",
            "control_kind": kind,
            "control_name": CONTROL_NAMES[kind],
            "summary": summary_path.name,
            "summary_sha256": sha256_file(summary_path),
            "cell_count": len(completed),
            "expected_cell_count": 12 * 10 * 2,
            "checkpoint": "CONTROL_COMPLETE",
        },
    )
    manifest_path = root / kind / "SHA256SUMS"
    files = sorted(path for path in (root / kind).rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    _atomic_text(
        manifest_path,
        "".join(f"{sha256_file(path)}  {path.relative_to(root / kind)}\n" for path in files),
    )
    return summary


def validate_control_bundle(
    root: str | Path,
    kind: str,
    *,
    require_owner: tuple[int, int] | None = None,
) -> dict[str, Any]:
    from .phase1 import cell_path, validate_cell

    errors = []
    checked = 0
    base = Path(root)
    summary_path = base / kind / "CONTROL_BUNDLE.json"
    marker_path = base / kind / "COMPLETED_CONTROL.json"
    manifest_path = base / kind / "SHA256SUMS"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if summary.get("protocol_id") != PHASE1_PROTOCOL_ID:
            errors.append("control summary protocol mismatch")
        if summary.get("control_kind") != kind or summary.get("cell_count") != 12 * 10 * 2:
            errors.append("control summary count mismatch")
        if marker.get("summary_sha256") != sha256_file(summary_path):
            errors.append("control summary SHA mismatch")
        if marker.get("expected_cell_count") != 12 * 10 * 2 or marker.get("cell_count") != 12 * 10 * 2:
            errors.append("control completion count mismatch")
        manifest_lines = manifest_path.read_text(encoding="utf-8").splitlines()
        if not manifest_lines:
            errors.append("empty control SHA256SUMS")
        for line in manifest_lines:
            expected, relative = line.split("  ", 1)
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                errors.append(f"unsafe control checksum entry: {relative}")
                continue
            path = base / kind / relative_path
            if path.name == "SHA256SUMS" or not path.is_file() or sha256_file(path) != expected:
                errors.append(f"control checksum mismatch: {relative}")
    except Exception as exc:
        errors.append(f"control completion marker parse error: {type(exc).__name__}: {exc}")
    for episode in range(12):
        for location in range(10):
            for stream in STREAMS:
                directory = base / kind / f"episode{episode:02d}" / f"location{location:02d}" / stream
                ok, reasons, _ = validate_cell(
                    directory,
                    suite=f"control_{kind}",
                    task_id=0,
                    parent_id=f"{kind}_episode{episode:02d}",
                    location_index=location,
                    stream=stream,
                    require_owner=require_owner,
                )
                checked += 1
                if not ok:
                    errors.append({"directory": str(directory), "errors": reasons})
    return {"protocol_id": PHASE1_PROTOCOL_ID, "control_kind": kind, "checked_cells": checked, "valid": not errors and checked == 12 * 10 * 2, "errors": errors}
