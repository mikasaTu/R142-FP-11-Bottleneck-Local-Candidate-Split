from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from r142_stage_s.libero import (
    CALIBRATION_CANDIDATE_COUNT,
    CALIBRATION_INITIAL_STATES,
    CALIBRATION_TASK_IDS,
    MAIN_CANDIDATE_COUNT,
    PROXIMITY_MAGNITUDES,
    CheckpointQualificationError,
    MissingRegeneratedInitialStates,
    StageRSnapshot,
    audit_undertrained_checkpoint_set,
    build_b_variant_suite,
    build_c_training_launcher_contract,
    capture_stage_r_snapshot,
    collect_family,
    family_is_complete,
    generate_b_variant,
    run_pooled_calibration,
    stable_seed,
    validate_restore_same_action,
    write_family_atomic,
    write_pooled_calibration,
)


LIBERO_ROOT = Path(
    "/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/QPILOTS-r16p15-stage1-task64-20260812/third_party/openpi/third_party/libero/libero/libero"
)


class FakeSnapshotEnvironment:
    def __init__(self) -> None:
        self.state = 0.0
        self.step = 0
        self.evaluation_seed = 0
        self._observation = {"state": np.asarray([0.0], dtype=np.float64)}
        self.seed_calls: list[int] = []
        self.close_calls = 0

    def seed(self, seed: int) -> None:
        self.evaluation_seed = int(seed)
        self.seed_calls.append(int(seed))

    def reset(self, init_state: int) -> dict[str, np.ndarray]:
        self.state = float(init_state)
        self.step = 0
        self._observation = {"state": np.asarray([self.state], dtype=np.float64)}
        return self._observation

    def raw_observation(self) -> dict[str, np.ndarray]:
        return {"state": np.asarray([self.state], dtype=np.float64)}

    def state_vector(self) -> np.ndarray:
        return np.asarray([self.state, self.step], dtype=np.float64)

    def capture_snapshot(self) -> dict[str, object]:
        return {"state": self.state, "step": self.step, "seed": self.evaluation_seed}

    def restore_snapshot(self, snapshot: dict[str, object]) -> None:
        self.state = float(snapshot["state"])
        self.step = int(snapshot["step"])
        self.evaluation_seed = int(snapshot["seed"])
        self._observation = {"state": np.asarray([self.state], dtype=np.float64)}

    def execute_actions(self, actions: np.ndarray) -> dict[str, object]:
        action = np.asarray(actions, dtype=np.float64)
        if action.ndim == 2:
            action = action[0]
        self.state += float(action[0])
        self.step += 1
        self._observation = {"state": np.asarray([self.state], dtype=np.float64)}
        success = self.step >= 2
        return {"success": success, "done": success}

    def close(self) -> None:
        self.close_calls += 1


class FakePolicy:
    def sample_action_chunk(self, observation: object, *, seed: int, counter: int) -> np.ndarray:
        del observation
        del seed, counter
        return np.tile(np.asarray([1.0, 0.0], dtype=np.float32), (5, 1))


def test_bddl_variants_have_one_same_type_non_goal_duplicate() -> None:
    source_root = LIBERO_ROOT / "bddl_files" / "libero_10"
    for task_id, task in enumerate(__import__("r142_stage_s.libero", fromlist=["LIBERO_TASK_SPECS"]).LIBERO_TASK_SPECS):
        variant = generate_b_variant(source_root / f"{task.name}.bddl", task_id, PROXIMITY_MAGNITUDES[0])
        assert variant.prompt_unchanged is True
        assert variant.bddl_text.count(variant.distractor_object) == 2  # declaration + On
        assert variant.bddl_text.count(variant.distractor_region) == 2  # region + On
        goal = variant.bddl_text[variant.bddl_text.index("(:goal") :]
        assert variant.distractor_object not in goal
        assert f"- {variant.target_type}" in variant.bddl_text


def test_b_build_requires_regenerated_qpos_and_accepts_manifest(tmp_path: Path) -> None:
    source_bddl = LIBERO_ROOT / "bddl_files" / "libero_10"
    source_init = LIBERO_ROOT / "init_files" / "libero_10"
    with pytest.raises(MissingRegeneratedInitialStates):
        build_b_variant_suite(
            source_bddl,
            tmp_path / "variant",
            PROXIMITY_MAGNITUDES[0],
            regenerated_initial_states_root=None,
            source_init_root=source_init,
        )
    regenerated = tmp_path / "regenerated"
    regenerated.mkdir()
    for task in __import__("r142_stage_s.libero", fromlist=["LIBERO_TASK_SPECS"]).LIBERO_TASK_SPECS:
        (regenerated / f"{task.name}.pruned_init").write_bytes(f"new-qpos-{task.task_id}".encode())
    (regenerated / "REGENERATED_INIT_STATES.json").write_text(
        json.dumps({"regenerated": True, "old_init_reused": False}), encoding="utf-8"
    )
    result = build_b_variant_suite(
        source_bddl,
        tmp_path / "variant",
        PROXIMITY_MAGNITUDES[1],
        regenerated_initial_states_root=regenerated,
        source_init_root=source_init,
    )
    assert len(result) == 10
    assert (tmp_path / "variant" / "config.yaml").is_file()
    assert all(Path(row["bddl_path"]).is_file() for row in result)


def test_calibration_persists_aggregate_only() -> None:
    calls: list[tuple[object, int, int, int, int]] = []

    def evaluator(setting: object, task_id: int, init_state: int, candidate_id: int, seed: int) -> bool:
        calls.append((setting, task_id, init_state, candidate_id, seed))
        return int(setting) == 1 and candidate_id == 0

    result = run_pooled_calibration(evaluator, [0, 1, 2, 3])
    assert len(calls) == len(CALIBRATION_TASK_IDS) * len(CALIBRATION_INITIAL_STATES) * CALIBRATION_CANDIDATE_COUNT * 4
    assert set(result) == {"protocol_id", "target_pooled_success", "rows", "selected_setting"}
    assert all(set(row) == {"setting", "successes", "total", "pooled_success"} for row in result["rows"])
    assert result["selected_setting"] == 1
    with pytest.raises(ValueError):
        write_pooled_calibration(
            Path("/tmp/never-written-stage-s.json"),
            {**result, "family": {"success": True}},
        )


def test_stage_r_snapshot_same_action_is_exact() -> None:
    environment = FakeSnapshotEnvironment()
    environment.seed(11)
    environment.reset(3)
    snapshot = capture_stage_r_snapshot(
        environment,
        [np.asarray([0.25, 0.0], dtype=np.float32)],
        stable_seed("snapshot"),
        2,
        4,
    )
    result = validate_restore_same_action(environment, snapshot, [0.5, 0.0])
    assert result["passed"] is True
    assert result["max_abs_error"] <= 1e-9


def test_family_artifacts_are_atomic_and_resumable(tmp_path: Path) -> None:
    def factory(**kwargs: object) -> FakeSnapshotEnvironment:
        del kwargs
        return FakeSnapshotEnvironment()

    family = collect_family(
        factory,
        FakePolicy(),
        task_id=0,
        init_state=0,
        candidate_count=4,
        max_steps=4,
    )
    assert family["candidate_count"] == 4
    assert family["policy_forwards"] == 4
    target = tmp_path / "A" / "task00" / "init000"
    marker = write_family_atomic(target, family)
    assert marker["checkpoint"] == "FAMILY_COMPLETE"
    assert family_is_complete(target, expected_candidates=4)
    (target / "rollouts.npz").write_bytes((target / "rollouts.npz").read_bytes() + b"corrupt")
    assert family_is_complete(target, expected_candidates=4) is False


def test_c_requires_four_exact_real_undertrained_checkpoints(tmp_path: Path) -> None:
    paths = []
    for index in range(4):
        path = tmp_path / f"pi05_libero_step_{1000 * (index + 1):05d}"
        path.mkdir()
        (path / "model.safetensors").write_bytes(b"real-weight")
        (path / "CHECKPOINT_PROVENANCE.json").write_text(
            json.dumps({"repository": "openpi/pi05_libero", "global_step": 1000 * (index + 1)}),
            encoding="utf-8",
        )
        paths.append(path)
    audit = audit_undertrained_checkpoint_set(paths, expected_steps=[1000, 2000, 3000, 4000])
    assert audit["valid"] is True
    contract = build_c_training_launcher_contract(
        qpilots_root=tmp_path / "qpilots",
        output_root=tmp_path / "out",
        checkpoint_paths=paths,
        expected_steps=[1000, 2000, 3000, 4000],
    )
    assert contract["label"] == "WEAK_SUBSTRATE"
    assert contract["launcher"]["no_pai_submit_performed"] is True
    assert "train_pytorch.py" in contract["launcher"]["shell"]
    command = contract["launcher"]["command"]
    assert command[command.index("--save-interval") + 1] == "1000"
    assert command[command.index("--num-train-steps") + 1] == "10001"
    too_late = paths[0] / "CHECKPOINT_PROVENANCE.json"
    too_late.write_text(json.dumps({"repository": "openpi/pi05_libero", "global_step": 30000}))
    assert audit_undertrained_checkpoint_set(paths)["valid"] is False
    bad = audit_undertrained_checkpoint_set(paths[:3])
    assert bad["valid"] is False
