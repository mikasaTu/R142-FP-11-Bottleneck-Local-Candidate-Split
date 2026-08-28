from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from r142_stage_r.phase1 import (
    BranchEnvironmentPool,
    PHASE1_PROTOCOL_ID,
    Phase1Collector,
    collect_branch_cell,
    decile_step,
    replay_baseline_snapshot,
    selection_rank,
    validate_cell,
    validate_task_completion_marker,
    write_cell,
)
from r142_stage_r.phase1_analysis import (
    _load_control_matrix,
    calibrate_phase1r,
    location_curve,
)
from r142_stage_r.phase1_controls import collect_control_bundle, validate_control_bundle


class MockEnvironment:
    """CPU-only snapshot adapter with deterministic terminal success."""

    def __init__(self) -> None:
        self.step = 0
        self.initial_state = 0
        self.reset_calls = 0
        self.close_calls = 0
        self.begin_branch_calls = 0

    def __deepcopy__(self, memo: object) -> object:
        del memo
        raise AssertionError("live environment deepcopy is forbidden")

    def reset(self, initial_state: int) -> dict[str, np.ndarray]:
        self.reset_calls += 1
        self.initial_state = int(initial_state)
        self.step = 0
        return self.raw_observation()

    def raw_observation(self) -> dict[str, np.ndarray]:
        return {"step": np.asarray([self.step], dtype=np.float32)}

    def state_vector(self) -> np.ndarray:
        return np.asarray([self.initial_state, self.step], dtype=np.float64)

    def capture_snapshot(self) -> dict[str, int]:
        return {"initial_state": self.initial_state, "step": self.step}

    def restore_snapshot(self, snapshot: dict[str, int]) -> None:
        self.initial_state = int(snapshot["initial_state"])
        self.step = int(snapshot["step"])

    def execute_actions(self, action_batch: np.ndarray) -> dict[str, object]:
        action = np.asarray(action_batch)
        if action.ndim == 2:
            action = action[0]
        assert action.ndim == 1 and action.shape[0] >= 2
        self.step += 1
        success = self.step >= 8
        return {"success": success, "done": success, "progress": self.step / 8.0}

    def begin_branch(self) -> None:
        self.begin_branch_calls += 1

    def close(self) -> None:
        self.close_calls += 1


class MockPolicy:
    def __init__(self) -> None:
        self.chunk_calls = 0

    def sample_action_chunk(self, observation: object, *, seed: int, counter: int) -> np.ndarray:
        del observation, seed, counter
        self.chunk_calls += 1
        return np.tile(np.asarray([1.0, 0.0], dtype=np.float32), (5, 1))


class BatchedMockPolicy(MockPolicy):
    """Official-path test policy with deterministic physical 7D actions."""

    class _Random:
        @staticmethod
        def PRNGKey(seed: int) -> int:
            return int(seed)

        @staticmethod
        def fold_in(key: int, value: int) -> int:
            return int(key) + int(value)

        @staticmethod
        def normal(key: int, shape: tuple[int, int]) -> np.ndarray:
            del key
            return np.zeros(shape, dtype=np.float32)

    def __init__(self) -> None:
        super().__init__()
        self.jax = SimpleNamespace(random=self._Random())
        self.jnp = np
        self.model = SimpleNamespace(action_horizon=5, action_dim=2)
        self.official_calls = 0

    @staticmethod
    def _physical_chunk() -> np.ndarray:
        action = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return np.tile(action, (5, 1))

    def sample_action_chunk(self, observation: object, *, seed: int, counter: int) -> np.ndarray:
        del observation, seed, counter
        self.chunk_calls += 1
        return self._physical_chunk()

    def sample_model_actions_official(
        self,
        observations: list[object],
        noises: np.ndarray,
    ) -> np.ndarray:
        del noises
        self.official_calls += 1
        return np.zeros((len(observations), 5, 2), dtype=np.float32)

    @staticmethod
    def input_transform(observation: object) -> dict[str, np.ndarray]:
        del observation
        return {"state": np.zeros(1, dtype=np.float32)}

    @staticmethod
    def output_transform(payload: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        del payload
        action = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return {"actions": np.tile(action, (10, 1))}


class NeverReplayPolicy:
    """Fails if baseline reconstruction performs any policy inference."""

    def sample_action_chunk(self, *args: object, **kwargs: object) -> np.ndarray:
        del args, kwargs
        raise AssertionError("frozen baseline replay must not call the policy")


class PoolEnvironment(MockEnvironment):
    pass


def _raw_archive(root: Path, action_dim: int = 2) -> None:
    rows = 12
    length = 8
    offsets = np.arange(0, rows * length + 1, length, dtype=np.int64)
    action = np.zeros(action_dim, dtype=np.float32)
    action[0] = 1.0
    np.savez_compressed(
        root / "libero_spatial_task00.npz",
        lengths=np.full(rows, length, dtype=np.int32),
        offsets=offsets,
        actions=np.tile(action[None, :], (rows * length, 1)),
        eef=np.zeros((rows * length, 1), dtype=np.float64),
        objects=np.zeros((rows * length, 1), dtype=np.float64),
        progress=np.ones(rows * length, dtype=np.float32),
        success=np.ones(rows, dtype=np.bool_),
        init_state=np.arange(rows, dtype=np.int16),
        candidate_id=np.zeros(rows, dtype=np.int16),
        rollout_seed=np.arange(1000, 1000 + rows, dtype=np.uint64),
        policy_forwards=np.full(rows, 2, dtype=np.int32),
    )


def test_selection_hash_contract_and_decile() -> None:
    value = selection_rank("libero_spatial", 0, 2, 3, 987)
    expected = hashlib.sha256(
        f"{PHASE1_PROTOCOL_ID}|episode|libero_spatial|0|2|3|987".encode("utf-8")
    ).hexdigest()
    assert value == expected
    assert decile_step(10, 0.1) == 0
    assert decile_step(10, 1.0) == 9


def test_natural_collector_resume_sha_and_task_marker(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _raw_archive(raw)
    from r142_stage_r.phase1 import select_phase1_episodes

    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(select_phase1_episodes(raw, "libero_spatial", 0)), encoding="utf-8")
    output = tmp_path / "natural"
    environments: list[MockEnvironment] = []
    policies: list[MockPolicy] = []

    def environment_factory(suite: str, task_id: int, kind: str) -> MockEnvironment:
        del suite, task_id, kind
        environment = MockEnvironment()
        environments.append(environment)
        return environment

    def policy_factory(suite: str, task_id: int, prompt: str) -> MockPolicy:
        del suite, task_id, prompt
        policy = MockPolicy()
        policies.append(policy)
        return policy

    collector = Phase1Collector(
        environment_factory,
        policy_factory,
        max_steps=16,
        microbatch=2,
        require_owner=None,
    )
    first = collector.collect_task(raw, selection_path, output, "libero_spatial", 0)
    assert first["completed_cells"] == 240
    assert len(environments) == 1
    assert environments[0].reset_calls == 12
    assert environments[0].close_calls == 1
    assert len(policies) == 1
    assert policies[0].chunk_calls >= 24

    raw_path = raw / "libero_spatial_task00.npz"
    raw_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    baseline_paths = sorted(output.glob("libero_spatial/task00/episode*/BASELINE_REPLAY.json"))
    assert len(baseline_paths) == 12
    expected_steps = [decile_step(8, decile) for decile in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)]
    for baseline_path in baseline_paths:
        payload = json.loads(baseline_path.read_text())
        assert payload["source_raw_file"] == raw_path.name
        assert payload["source_raw_sha256"] == raw_sha
        assert payload["source_raw_key"] == payload["parent_id"]
        assert payload["target_steps"] == expected_steps
        assert payload["captured_steps"] == sorted(set(expected_steps))
        assert payload["replay_steps"] == payload["baseline_length"] == 8
        assert payload["replay_actions"] == 8
        assert payload["baseline_success"] is True
        assert payload["expected_success"] is True
        assert payload["replay_source"] == "frozen_phase0_action_stream"
        assert payload["policy_reexecuted"] is False
        assert payload["policy_forwards"] == payload["policy_batches"] == 0
        assert payload["source_policy_forwards"] == payload["source_action_chunks"] == 2
        assert len(payload["replay_action_sha256"]) == 64
        assert payload["source_action_sha256"] == payload["replay_action_sha256"]
        assert payload["replay_action_shape"] == [8, 2]
        assert baseline_path.with_name("BASELINE_REPLAY_SHA256SUMS").is_file()
    valid, errors, marker = validate_task_completion_marker(
        output, "libero_spatial", 0, selection_path, require_owner=None
    )
    assert valid, errors
    assert marker is not None
    assert marker["baseline_replay_count"] == 12

    damaged = output / "libero_spatial/task00/episode00/location00/calibration/cell.npz"
    damaged.write_bytes(damaged.read_bytes() + b"corrupt")
    second = collector.collect_task(raw, selection_path, output, "libero_spatial", 0)
    assert second["completed_cells"] == 240
    assert list((damaged.parent).glob("FAILURE.*.json"))
    assert len(environments) == 2
    assert environments[1].reset_calls == 1
    assert environments[1].close_calls == 1
    valid, errors, _ = validate_cell(
        damaged.parent,
        suite="libero_spatial",
        task_id=0,
        parent_id=json.loads(selection_path.read_text())["selected"][0]["episode"],
        location_index=0,
        stream="calibration",
        require_owner=None,
    )
    assert valid, errors


def test_natural_collector_uses_factory_pool_and_single_replay(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _raw_archive(raw, action_dim=7)
    from r142_stage_r.phase1 import select_phase1_episodes

    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(select_phase1_episodes(raw, "libero_spatial", 0)), encoding="utf-8")
    environments: list[PoolEnvironment] = []
    policies: list[BatchedMockPolicy] = []

    def environment_factory(suite: str, task_id: int, kind: str) -> PoolEnvironment:
        del suite, task_id, kind
        environment = PoolEnvironment()
        environments.append(environment)
        return environment

    def policy_factory(suite: str, task_id: int, prompt: str) -> BatchedMockPolicy:
        del suite, task_id, prompt
        policy = BatchedMockPolicy()
        policies.append(policy)
        return policy

    collector = Phase1Collector(
        environment_factory,
        policy_factory,
        max_steps=16,
        microbatch=4,
        require_owner=None,
    )
    result = collector.collect_task(
        raw,
        selection_path,
        tmp_path / "natural",
        "libero_spatial",
        0,
        streams=("heldout",),
    )
    assert result["completed_cells"] == 120
    # One main environment plus exactly microbatch-sized reusable pool.
    assert len(environments) == 1 + 4
    assert environments[0].reset_calls == 12
    assert environments[0].begin_branch_calls == 0
    assert all(environment.begin_branch_calls > 0 for environment in environments[1:])
    assert all(environment.close_calls == 1 for environment in environments)
    assert len(policies) == 1
    assert policies[0].official_calls > 0
    baseline_paths = sorted(
        (tmp_path / "natural").glob("libero_spatial/task00/episode*/BASELINE_REPLAY.json")
    )
    assert len(baseline_paths) == 12


def test_baseline_replay_uses_exact_frozen_phase0_action_stream() -> None:
    action = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    row = {
        "init_state": 3,
        "candidate_id": 4,
        "rollout_seed": 12345,
        "actions": np.tile(action, (8, 1)),
        "success": True,
    }
    environment = MockEnvironment()
    policy = NeverReplayPolicy()
    snapshot, replay = replay_baseline_snapshot(
        environment,
        policy,
        "libero_spatial",
        0,
        row,
        3,
    )
    assert snapshot.step == 3
    assert replay["action_max_abs_error"] == 0.0
    assert replay["replay_source"] == "frozen_phase0_action_stream"
    assert replay["policy_reexecuted"] is False
    assert replay["policy_forwards"] == replay["policy_batches"] == 0
    assert replay["source_action_chunks"] == 2
    assert replay["source_action_sha256"] == replay["replay_action_sha256"]


def test_factory_pool_matches_sequential_reference(tmp_path: Path) -> None:
    action = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    row = {
        "init_state": 3,
        "candidate_id": 4,
        "rollout_seed": 12345,
        "actions": np.tile(action, (8, 1)),
        "success": True,
    }
    baseline_environment = MockEnvironment()
    baseline_policy = BatchedMockPolicy()
    snapshot, _ = replay_baseline_snapshot(
        baseline_environment,
        baseline_policy,
        "libero_spatial",
        0,
        row,
        3,
    )
    sequential_environment = PoolEnvironment()
    sequential_policy = BatchedMockPolicy()
    sequential = collect_branch_cell(
        sequential_environment,
        sequential_policy,
        snapshot=snapshot,
        suite="libero_spatial",
        task_id=0,
        parent_id="parent",
        generation_step=3,
        stream="heldout",
        location_index=0,
        descendants=4,
        max_steps=16,
        microbatch=1,
    )
    created: list[PoolEnvironment] = []

    def factory(suite: str, task_id: int, kind: str) -> PoolEnvironment:
        del suite, task_id, kind
        environment = PoolEnvironment()
        created.append(environment)
        return environment

    pool = BranchEnvironmentPool(
        factory,
        "libero_spatial",
        0,
        capacity=2,
    )
    pooled_environment = PoolEnvironment()
    pooled_policy = BatchedMockPolicy()
    pooled = collect_branch_cell(
        pooled_environment,
        pooled_policy,
        snapshot=snapshot,
        suite="libero_spatial",
        task_id=0,
        parent_id="parent",
        generation_step=3,
        stream="heldout",
        location_index=0,
        descendants=4,
        max_steps=16,
        microbatch=2,
        environment_pool=pool,
    )
    assert len(created) == 2
    assert all(environment.begin_branch_calls == 2 for environment in created)
    assert [trace.branch_seed for trace in sequential] == [trace.branch_seed for trace in pooled]
    for expected, observed in zip(sequential, pooled, strict=True):
        assert expected.branch_id == observed.branch_id
        assert expected.parent_id == observed.parent_id
        assert expected.generation_step == observed.generation_step
        assert expected.policy_forwards == observed.policy_forwards
        assert expected.policy_batches == observed.policy_batches
        assert expected.environment_steps == observed.environment_steps
        assert expected.success == observed.success
        np.testing.assert_array_equal(expected.actions, observed.actions)
        np.testing.assert_array_equal(expected.progress, observed.progress)
        np.testing.assert_array_equal(expected.states, observed.states)
    pool.close()
    pool.close()
    assert all(environment.close_calls == 1 for environment in created)


def test_control_bundle_completion_and_resume(tmp_path: Path) -> None:
    for kind in ("positive", "null"):
        collect_control_bundle(kind, tmp_path)
        result = validate_control_bundle(tmp_path, kind, require_owner=None)
        assert result["valid"], result["errors"]
        marker = tmp_path / kind / "COMPLETED_CONTROL.json"
        assert json.loads(marker.read_text())["cell_count"] == 240

    # The positive control must be a real engineering control: branches
    # before the frozen commit step can discover the correct lane, while
    # branches after that point inherit an irreversible wrong commitment.
    calibration = calibrate_phase1r(
        tmp_path,
        tmp_path / "calibration.json",
        shuffles=1000,
        require_owner=None,
    )
    positive, _ = _load_control_matrix(tmp_path, "positive", stream="heldout", require_owner=None)
    null, _ = _load_control_matrix(tmp_path, "null", stream="heldout", require_owner=None)
    positive_curve = location_curve(positive)
    null_curve = location_curve(null)
    threshold = float(calibration["location_sensitivity_threshold"])
    assert float(positive_curve[3] - positive_curve[4]) > 0.5
    assert float(np.ptp(positive_curve)) > threshold
    assert float(np.ptp(null_curve)) <= threshold


def test_manual_cell_schema_sha(tmp_path: Path) -> None:
    from r142_stage_r.phase1 import BranchTrace

    traces = [
        BranchTrace(
            branch_id=index,
            parent_id="parent",
            generation_step=3,
            branch_seed=index,
            actions=np.ones((1, 2), dtype=np.float32),
            progress=np.ones(1, dtype=np.float32),
            states=np.ones((1, 2), dtype=np.float64),
            success=True,
            policy_forwards=1,
            policy_batches=1,
            environment_steps=1,
        )
        for index in range(16)
    ]
    directory = tmp_path / "cell"
    write_cell(
        directory,
        traces,
        suite="libero_spatial",
        task_id=0,
        parent_id="parent",
        selection_index=0,
        location_index=0,
        decile=0.1,
        generation_step=3,
        stream="heldout",
        baseline_length=10,
    )
    valid, errors, _ = validate_cell(directory, require_owner=None)
    assert valid, errors
