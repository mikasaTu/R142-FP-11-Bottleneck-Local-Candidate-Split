from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pytest

import r142_stage_s.libero as libero
from r142_stage_s.libero import (
    B_INIT_STATE_COUNT,
    C_RETAIN_STEPS,
    CALIBRATION_CANDIDATE_COUNT,
    CALIBRATION_INITIAL_STATES,
    CALIBRATION_TASK_IDS,
    MAIN_CANDIDATE_COUNT,
    PROXIMITY_MAGNITUDES,
    MissingRegeneratedInitialStates,
    audit_undertrained_checkpoint_set,
    audit_c_checkpoint_schedule,
    build_b_variant_matrix,
    build_pai_stage_s_payload,
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


def _fake_bddl(task: object) -> str:
    task_id = int(getattr(task, "task_id"))
    target = str(getattr(task, "target_object"))
    region = str(getattr(task, "target_region"))
    target_type = target.removesuffix("_1")
    fixture = "kitchen_table" if task_id % 2 else "living_room_table"
    return f"""(define (problem fake_{task_id})
  (:domain robosuite)
  (:language {getattr(task, 'prompt')})
  (:regions
    ({region}
      (:target {fixture})
      (:ranges ((0.0 0.0 0.4 0.4)))
      (:yaw_rotation ((0.0 0.0)))
    )
  )
  (:objects
    {fixture} - {fixture}
    {target} - {target_type}
  )
  (:init
    (On {target} {fixture}_{region})
  )
  (:goal
    (And (On {target} {fixture}))
  )
)\n"""


def _fake_source_roots(tmp_path: Path) -> tuple[Path, Path]:
    bddl_root = tmp_path / "source_bddl"
    init_root = tmp_path / "source_init"
    bddl_root.mkdir()
    init_root.mkdir()
    for task in libero.LIBERO_TASK_SPECS:
        (bddl_root / f"{task.name}.bddl").write_text(_fake_bddl(task), encoding="utf-8")
        with (init_root / f"{task.name}.pruned_init").open("wb") as handle:
            np.save(handle, np.arange(4, dtype=np.float64)[None, :])
    return bddl_root, init_root


def test_bddl_variants_have_one_same_type_non_goal_duplicate(tmp_path: Path) -> None:
    for task_id, task in enumerate(libero.LIBERO_TASK_SPECS):
        variant = generate_b_variant(_fake_bddl_path(tmp_path, task), task_id, PROXIMITY_MAGNITUDES[0])
        assert variant.prompt_unchanged is True
        assert variant.bddl_text.count(variant.distractor_object) == 2  # declaration + On
        assert variant.bddl_text.count(variant.distractor_region) == 2  # region + On
        goal = variant.bddl_text[variant.bddl_text.index("(:goal") :]
        assert variant.distractor_object not in goal
        assert f"- {variant.target_type}" in variant.bddl_text


def _fake_bddl_path(root: Path, task: object) -> Path:
    path = root / f"stage_s_fake_{getattr(task, 'task_id')}.bddl"
    path.write_text(_fake_bddl(task), encoding="utf-8")
    return path


def test_b_build_requires_regenerated_qpos_and_accepts_manifest(tmp_path: Path) -> None:
    source_bddl, source_init = _fake_source_roots(tmp_path)
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
    for task in libero.LIBERO_TASK_SPECS:
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


class FakeQposSimulator:
    def __init__(self, seed: int) -> None:
        self.sim = types.SimpleNamespace(
            data=types.SimpleNamespace(qpos=np.arange(10, dtype=np.float64) + float(seed % 100000) / 100000.0)
        )
        self.seed_value = int(seed)

    def seed(self, seed: int) -> None:
        self.seed_value = int(seed)

    def reset(self) -> None:
        return None

    def close(self) -> None:
        return None


def test_b_matrix_uses_fixed_seed_real_simulator_callback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_bddl, source_init = _fake_source_roots(tmp_path)
    def factory(*, task_id: int, bddl_path: str, seed: int) -> FakeQposSimulator:
        assert Path(bddl_path).is_file()
        return FakeQposSimulator(seed + int(task_id))

    def writer(path: str | Path, array: np.ndarray) -> None:
        with Path(path).open("wb") as handle:
            np.save(handle, array)

    monkeypatch.setattr(libero, "_write_torch_qpos", writer)
    result = build_b_variant_matrix(source_bddl, source_init, tmp_path / "matrix", simulator_factory=factory)
    assert len(result) == 4
    assert all(len(row["tasks"]) == 10 for row in result)
    assert all(
        json.loads(Path(row["regenerated_init_states"]).read_text(encoding="utf-8"))["init_state_count"]
        == B_INIT_STATE_COUNT
        for row in result
    )


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
        path = tmp_path / f"pi05_libero_step_{C_RETAIN_STEPS[index]:05d}"
        path.mkdir()
        (path / "model.safetensors").write_bytes(b"real-weight")
        (path / "CHECKPOINT_PROVENANCE.json").write_text(
            json.dumps({"repository": "openpi/pi05_libero", "global_step": C_RETAIN_STEPS[index]}),
            encoding="utf-8",
        )
        paths.append(path)
    audit = audit_undertrained_checkpoint_set(paths, expected_steps=list(C_RETAIN_STEPS))
    assert audit["valid"] is True
    contract = build_c_training_launcher_contract(
        qpilots_root=tmp_path / "qpilots",
        output_root=tmp_path / "out",
        checkpoint_paths=paths,
        expected_steps=list(C_RETAIN_STEPS),
    )
    assert contract["label"] == "WEAK_SUBSTRATE"
    assert contract["launcher"]["no_pai_submit_performed"] is True
    assert "train_pytorch.py" in contract["launcher"]["shell"]
    command = contract["launcher"]["command"]
    assert command[command.index("--save_interval") + 1] == "1000"
    assert command[command.index("--num_train_steps") + 1] == "10001"
    assert command[command.index("--seed") + 1] == "42"
    assert "--nproc_per_node=8" in command
    too_late = paths[0] / "CHECKPOINT_PROVENANCE.json"
    too_late.write_text(json.dumps({"repository": "openpi/pi05_libero", "global_step": 30000}))
    assert audit_undertrained_checkpoint_set(paths)["valid"] is False
    bad = audit_undertrained_checkpoint_set(paths[:3])
    assert bad["valid"] is False


def test_c_schedule_and_idle_payload_are_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "schedule"
    for step in C_RETAIN_STEPS:
        path = root / str(step)
        path.mkdir(parents=True)
        (path / "model.safetensors").write_bytes(b"weight")
        (path / "optimizer.pt").write_bytes(b"optimizer")
        (path / "metadata.pt").write_bytes(b"metadata")
        (path / "metadata.json").write_text(
            json.dumps({"repository": "openpi/pi05_libero", "global_step": step}), encoding="utf-8"
        )
    schedule = audit_c_checkpoint_schedule(root, require_training_state=False)
    assert schedule["valid"] is True
    payload = build_pai_stage_s_payload(
        run_id="r142-stage-s-c-seed42",
        output_root="/mnt/cpfs/zbl-cpfs-new/CKPT/leon/r142_stage_s",
        log_root="/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_s",
        command=("torchrun", "scripts/train_pytorch.py", "pi05_libero"),
        working_directory="/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/QPILOTS",
    )
    assert payload["resource"]["gpu"] == 8
    assert payload["resource"]["cpu"] == 88
    assert payload["resource"]["memory_gib"] == 1525
    assert payload["resource"]["resource_alias"] == "idle-a800-robot-stage-s-graphics-8gpu"
    assert payload["resource"]["resource_id"] == "quota1ssrabud0bh"
    assert payload["resource"]["quota_name"] == "exp-robot"
    assert payload["daily_no_job_windows"][0]["timezone"] == "Asia/Shanghai"
    assert payload["checkpoint_contract"]["retain_and_audit_steps"] == list(C_RETAIN_STEPS)
