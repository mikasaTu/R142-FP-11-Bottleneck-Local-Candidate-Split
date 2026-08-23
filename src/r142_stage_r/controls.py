from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CONTROL_PROTOCOL_ID = "r142-stage-r-controls-v8"


@dataclass(frozen=True)
class ControlTrace:
    control_name: str
    initial_state: int
    candidate_id: int
    positions: np.ndarray
    actions: np.ndarray
    success: bool
    mode: int


class GeometricControl2D:
    """Actual forward simulator used only to validate the Phase-0R metrics.

    The positive environment has collision geometry and a one-way fork; the
    null environment is the identical integrator on an open plane. Neither is
    used as scientific evidence about the learned policy.
    """

    horizon = 80
    dt = 0.04

    def __init__(self, kind: str):
        if kind not in {"positive", "null"}:
            raise ValueError(kind)
        self.kind = kind

    def rollout(self, initial_state: int, candidate_id: int, seed: int) -> ControlTrace:
        rng = np.random.default_rng(int(seed))
        position = np.asarray([-1.0, 0.015 * (int(initial_state) - 7.5)], dtype=np.float64)
        velocity = np.zeros(2, dtype=np.float64)
        positions: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        committed = 0
        lane_choice = 0
        state_group = int(initial_state) // 4
        allowed_lane = 1 if state_group in {0, 2} else -1
        for step in range(self.horizon):
            x, y = position
            if self.kind == "positive":
                # The policy shares a low-variance approach and becomes
                # stochastic only near the physical fork. Initial-state bias
                # creates real between-state overdispersion.
                fork_gain = 1.0 / (1.0 + np.exp(-18.0 * (x + 0.25)))
                bias = 0.15 * (int(initial_state) - 7.5)
                if lane_choice == 0 and x >= -0.30:
                    lane_choice = 1 if bias + rng.normal(0.0, 0.28) >= 0.0 else -1
                lateral = fork_gain * 2.0 * (0.48 * lane_choice - y) if lane_choice else 0.0
                action = np.asarray([0.68, lateral], dtype=np.float64)
            else:
                action = np.asarray([0.68, -0.55 * y + rng.normal(0.0, 0.035)], dtype=np.float64)

            velocity = 0.68 * velocity + 0.32 * action
            proposed = position + self.dt * velocity
            if self.kind == "positive":
                # Barrier occupies the central strip. Crossing through either
                # geometric aperture creates a one-way committed side.
                in_barrier_x = -0.08 <= proposed[0] <= 0.10
                attempted_lane = 1 if proposed[1] > 0 else -1
                blocked_aperture = in_barrier_x and abs(proposed[1]) >= 0.22 and attempted_lane != allowed_lane
                if (in_barrier_x and abs(proposed[1]) < 0.22) or blocked_aperture:
                    proposed[0] = min(proposed[0], -0.081)
                    velocity[0] = min(velocity[0], 0.0)
                if position[0] <= 0.10 < proposed[0] and abs(proposed[1]) >= 0.22 and attempted_lane == allowed_lane:
                    committed = 1 if proposed[1] > 0 else -1
                if committed and proposed[0] > 0.10:
                    proposed[1] = committed * max(0.18, committed * proposed[1])
                    proposed[0] = max(proposed[0], 0.101)
            position = proposed
            positions.append(position.copy())
            actions.append(action.copy())

        if self.kind == "positive":
            target = np.asarray([1.0, 0.48 if committed >= 0 else -0.48])
            success = bool(committed and np.linalg.norm(position - target) <= 0.42)
            mode = int(committed)
        else:
            success = bool(np.linalg.norm(position - np.asarray([1.0, 0.0])) <= 0.45)
            mode = 0
        return ControlTrace(
            control_name="GeometricCommit2D-v1" if self.kind == "positive" else "OpenPlane2D-v1",
            initial_state=int(initial_state),
            candidate_id=int(candidate_id),
            positions=np.asarray(positions, dtype=np.float64),
            actions=np.asarray(actions, dtype=np.float64),
            success=success,
            mode=mode,
        )


def generate_control_bank(kind: str, *, initial_states: int = 16, candidates: int = 32) -> list[ControlTrace]:
    environment = GeometricControl2D(kind)
    traces = []
    namespace = 17_000_000 if kind == "positive" else 19_000_000
    for initial_state in range(int(initial_states)):
        for candidate_id in range(int(candidates)):
            traces.append(
                environment.rollout(
                    initial_state,
                    candidate_id,
                    namespace + initial_state * 1000 + candidate_id,
                )
            )
    return traces
