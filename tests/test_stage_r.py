from __future__ import annotations

import json

import numpy as np

from r142_stage_r.controls import GeometricControl2D
from r142_stage_r.metrics import divergence_curve, overdispersion
from r142_stage_r.protocol import atomic_json, ranked_initial_states, rollout_seed


def test_deterministic_selection_and_seed() -> None:
    values = ranked_initial_states("libero_spatial", 0)
    assert len(values) == 16
    assert len(set(values)) == 16
    assert values == ranked_initial_states("libero_spatial", 0)
    assert rollout_seed("libero_spatial", 0, values[0], 0) == rollout_seed(
        "libero_spatial", 0, values[0], 0
    )
    assert rollout_seed("libero_spatial", 0, values[0], 0) != rollout_seed(
        "libero_spatial", 0, values[0], 1
    )


def test_atomic_json(tmp_path) -> None:
    target = tmp_path / "nested" / "record.json"
    atomic_json(target, {"b": 2, "a": 1})
    assert json.loads(target.read_text()) == {"a": 1, "b": 2}


def test_control_is_real_forward_trace() -> None:
    trace = GeometricControl2D("positive").rollout(0, 0, 123)
    assert trace.positions.shape == (80, 2)
    assert trace.actions.shape == (80, 2)
    assert np.all(np.isfinite(trace.positions))
    assert not np.array_equal(trace.positions[0], trace.positions[-1])


def test_metrics_shapes() -> None:
    trajectories = [np.arange(12, dtype=np.float64).reshape(6, 2), np.zeros((4, 2))]
    curve, at_risk = divergence_curve(trajectories)
    assert curve.shape == (6,)
    assert at_risk.tolist() == [2, 2, 2, 2, 1, 1]
    success = np.asarray([0, 1, 0, 1, 0, 0, 0, 0], dtype=np.bool_)
    states = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    result = overdispersion(success, states, n=4)
    assert result["p_bar"] == 0.25
    assert result["rho"] is not None
