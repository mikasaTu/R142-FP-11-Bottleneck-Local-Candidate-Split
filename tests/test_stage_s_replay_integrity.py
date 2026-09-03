from __future__ import annotations

import copy

import numpy as np
import pytest

from r142_stage_s.replay_integrity import (
    ReplayIntegrityError,
    compare_replay_components,
)
from r142_stage_s.s45_adapters import LiberoS45Adapter, S45AdapterError


def _components() -> dict[str, object]:
    return {
        "simulator_snapshot": {
            "visible": np.asarray([0.0, 1.0], dtype=np.float64),
            "hidden_integrator": {"value": np.asarray([7.0], dtype=np.float64)},
        },
        "observation_history": {"frames": [np.asarray([0.0], dtype=np.float32)]},
        "action_queue": {"runner": [np.asarray([0.25, 0.0], dtype=np.float32)], "policy": []},
        "python_rng": {"version": 3, "state": [1, 2, 3]},
        "numpy_rng": {"bit_generator": "MT19937", "state": [4, 5, 6]},
        "torch_rng": {"cpu": np.asarray([8, 9], dtype=np.uint8), "cuda": [np.asarray([10], dtype=np.uint8)]},
        "environment_owner_rng": {"state": np.asarray([11], dtype=np.uint64)},
        "policy_owner_rng": {"state": np.asarray([12], dtype=np.uint64)},
    }


@pytest.mark.parametrize(
    ("component", "mutator"),
    [
        (
            "simulator_snapshot",
            lambda value: value["hidden_integrator"].update(value=np.asarray([8.0], dtype=np.float64)),
        ),
        (
            "observation_history",
            lambda value: value["frames"].append(np.asarray([99.0], dtype=np.float32)),
        ),
        (
            "action_queue",
            lambda value: value["runner"].__setitem__(0, np.asarray([0.5, 0.0], dtype=np.float32)),
        ),
        (
            "python_rng",
            lambda value: value["state"].__setitem__(1, 999),
        ),
        (
            "numpy_rng",
            lambda value: value["state"].__setitem__(2, 999),
        ),
        (
            "torch_rng",
            lambda value: value["cpu"].__setitem__(0, 99),
        ),
        (
            "environment_owner_rng",
            lambda value: value["state"].__setitem__(0, 99),
        ),
        (
            "policy_owner_rng",
            lambda value: value["state"].__setitem__(0, 99),
        ),
    ],
)
def test_same_observation_hidden_or_replay_component_drift_fails_closed(component, mutator) -> None:
    first = _components()
    second = copy.deepcopy(first)
    mutator(second[component])
    # The visible observation is intentionally unchanged. The complete
    # simulator/history/queue/RNG contract must still reject the replay.
    with pytest.raises(ReplayIntegrityError, match=component):
        compare_replay_components(first, second)


def test_complete_replay_evidence_contains_nested_schema_and_numeric_hashes() -> None:
    first = _components()
    evidence = compare_replay_components(first, copy.deepcopy(first))
    assert evidence["passed"] is True
    assert evidence["max_abs_error"] == 0.0
    for component in (
        "simulator_snapshot",
        "observation_history",
        "action_queue",
        "python_rng",
        "numpy_rng",
        "torch_rng",
        "environment_owner_rng",
        "policy_owner_rng",
    ):
        item = evidence[component]
        assert item["passed"] is True
        assert item["numeric_leaf_count"] >= 0
        assert len(item["schema_sha256"]) == 64
        assert item["first_numeric_sha256"] == item["second_numeric_sha256"]


def test_nested_snapshot_schema_drift_fails_even_without_numeric_value_drift() -> None:
    first = _components()
    second = copy.deepcopy(first)
    second["simulator_snapshot"]["hidden_integrator"]["new_component"] = 1
    with pytest.raises(ReplayIntegrityError, match="simulator_snapshot"):
        compare_replay_components(first, second)


def test_simulator_without_numeric_leaves_is_not_accepted() -> None:
    first = _components()
    second = copy.deepcopy(first)
    first["simulator_snapshot"] = {"visible": "same"}
    second["simulator_snapshot"] = {"visible": "same"}
    with pytest.raises(ReplayIntegrityError, match="numeric leaves"):
        compare_replay_components(first, second)


class _HiddenDriftEnv:
    """Visible state is deterministic while a hidden snapshot leaf drifts."""

    def __init__(self) -> None:
        self.state = 0.0
        self.steps = 0
        self.capture_calls = 0
        self._observation = {"observation/state": np.zeros(8, dtype=np.float64)}
        self.owner_rng = {"counter": 0}

    def reset(self, init_state: int) -> None:
        self.state = float(init_state)
        self.steps = 0
        self._observation = {"observation/state": np.zeros(8, dtype=np.float64)}

    def capture_snapshot(self) -> dict[str, object]:
        self.capture_calls += 1
        return {
            "visible_state": np.asarray([self.state, self.steps], dtype=np.float64),
            "hidden_integrator": np.asarray([1.0 if self.capture_calls >= 3 else 0.0], dtype=np.float64),
        }

    def restore_snapshot(self, value: dict[str, object]) -> None:
        visible = np.asarray(value["visible_state"], dtype=np.float64)
        self.state = float(visible[0])
        self.steps = int(visible[1])
        self._observation = {"observation/state": np.zeros(8, dtype=np.float64)}

    def capture_rng_state(self) -> dict[str, int]:
        return copy.deepcopy(self.owner_rng)

    def restore_rng_state(self, value: dict[str, int]) -> None:
        self.owner_rng = copy.deepcopy(value)

    def execute_actions(self, action: np.ndarray) -> dict[str, object]:
        self.state += float(np.asarray(action).reshape(-1)[0])
        self.steps += 1
        # Keep observation/state unchanged so the old visible-state-only gate
        # would incorrectly accept the replay.
        return {"done": False, "success": False}

    def state_vector(self) -> np.ndarray:
        return np.asarray([self.state, self.steps], dtype=np.float64)


class _ReplayPolicy:
    def __init__(self) -> None:
        self.owner_rng = {"counter": 0}

    def capture_rng_state(self) -> dict[str, int]:
        return copy.deepcopy(self.owner_rng)

    def restore_rng_state(self, value: dict[str, int]) -> None:
        self.owner_rng = copy.deepcopy(value)


def test_libero_same_action_gate_rejects_hidden_snapshot_drift_with_same_state() -> None:
    environment = _HiddenDriftEnv()
    environment.reset(0)
    policy = _ReplayPolicy()
    adapter = LiberoS45Adapter(
        lambda **_: environment,
        lambda **_: policy,
        protocol=None,  # type: ignore[arg-type]
        substrate="B",
        max_steps=8,
        require_torch=False,
    )
    snapshot = adapter._capture(environment, policy, [], 7, 0, 0)
    with pytest.raises(S45AdapterError, match="full replay integrity.*simulator_snapshot"):
        adapter._same_action_check(environment, policy, snapshot, [0.5, 0.0])
