from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from r142_stage_r.phase1 import (
    PHASE1_PROTOCOL_ID,
    Phase1Collector,
    decile_step,
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

    def reset(self, initial_state: int) -> dict[str, np.ndarray]:
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
        assert action.shape == (2,)
        self.step += 1
        success = self.step >= 8
        return {"success": success, "done": success, "progress": self.step / 8.0}

    def close(self) -> None:
        return None


class MockPolicy:
    def sample_action_chunk(self, observation: object, *, seed: int, counter: int) -> np.ndarray:
        del observation, seed, counter
        return np.tile(np.asarray([1.0, 0.0], dtype=np.float32), (5, 1))


def _raw_archive(root: Path) -> None:
    rows = 12
    length = 8
    offsets = np.arange(0, rows * length + 1, length, dtype=np.int64)
    np.savez_compressed(
        root / "libero_spatial_task00.npz",
        lengths=np.full(rows, length, dtype=np.int32),
        offsets=offsets,
        actions=np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (rows * length, 1)),
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
    collector = Phase1Collector(
        lambda suite, task_id, prompt: MockEnvironment(),
        lambda suite, task_id, prompt: MockPolicy(),
        max_steps=16,
        microbatch=2,
        require_owner=None,
    )
    first = collector.collect_task(raw, selection_path, output, "libero_spatial", 0)
    assert first["completed_cells"] == 240
    valid, errors, marker = validate_task_completion_marker(
        output, "libero_spatial", 0, selection_path, require_owner=None
    )
    assert valid, errors
    assert marker is not None

    damaged = output / "libero_spatial/task00/episode00/location00/calibration/cell.npz"
    damaged.write_bytes(damaged.read_bytes() + b"corrupt")
    second = collector.collect_task(raw, selection_path, output, "libero_spatial", 0)
    assert second["completed_cells"] == 240
    assert list((damaged.parent).glob("FAILURE.*.json"))
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
