from __future__ import annotations

import json
import pickle
import types
from pathlib import Path

import numpy as np
import pytest

import r142_stage_s.libero as libero
from scripts.stage_s_libero_main_finalize import MainEvaluationError, _verify_snapshot
from r142_stage_s.libero import (
    B_INIT_STATE_COUNT,
    C_RETAIN_STEPS,
    CALIBRATION_CANDIDATE_COUNT,
    CALIBRATION_INITIAL_STATES,
    CALIBRATION_SEED,
    CALIBRATION_TASK_IDS,
    MAIN_CANDIDATE_COUNT,
    PROXIMITY_MAGNITUDES,
    MissingRegeneratedInitialStates,
    VariantGenerationError,
    audit_undertrained_checkpoint_set,
    audit_c_checkpoint_schedule,
    aggregate_calibration_shards,
    build_b_variant_matrix,
    build_pai_stage_s_payload,
    build_b_variant_suite,
    build_c_training_launcher_contract,
    capture_stage_r_snapshot,
    collect_family,
    family_is_complete,
    generate_b_variant,
    run_pooled_calibration,
    run_calibration_shard,
    run_stage_s_calibration_episode,
    stable_seed,
    verify_calibration_aggregate,
    verify_calibration_shard,
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
        return np.asarray([self.state, self.step, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

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


class ReplayFakePolicy(FakePolicy):
    """Small policy exposing a real mutable RNG owner for strict snapshots."""

    def __init__(self) -> None:
        self._rng_state = {"seed": 17}

    def get_rng_state(self) -> dict[str, int]:
        return dict(self._rng_state)

    def set_rng_state(self, value: dict[str, int]) -> None:
        self._rng_state = dict(value)


def test_task64_pose_vector_uses_only_eef_workspace_state():
    from r142_stage_s.libero import _pose_vector

    observation = {
        "observation/state": np.asarray(
            [0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 9.0, 8.0], dtype=np.float32
        )
    }
    pose = _pose_vector(object(), observation)
    np.testing.assert_allclose(pose, observation["observation/state"][:6])
    assert pose.shape == (6,)


def test_task64_pose_vector_rejects_schema_drift():
    from r142_stage_s.libero import StageSError, _pose_vector

    with pytest.raises(StageSError, match="exactly"):
        _pose_vector(object(), {"observation/state": np.zeros(7, dtype=np.float32)})


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
        class FakeState:
            def __init__(self, value: np.ndarray) -> None:
                self.value = value

            def flatten(self) -> np.ndarray:
                return self.value

        self.sim = types.SimpleNamespace(
            data=types.SimpleNamespace(qpos=np.arange(10, dtype=np.float64) + float(seed % 100000) / 100000.0),
            get_state=lambda: FakeState(
                np.arange(21, dtype=np.float64) + float(seed % 100000) / 100000.0
            ),
        )
        self.seed_value = int(seed)
        self.last_state: np.ndarray | None = None

    def seed(self, seed: int) -> None:
        self.seed_value = int(seed)

    def reset(self) -> None:
        return None

    def get_sim_state(self) -> np.ndarray:
        return self.sim.get_state().flatten()

    def set_init_state(self, state: np.ndarray) -> dict[str, np.ndarray]:
        self.last_state = np.asarray(state, dtype=np.float64).copy()
        return {"state": self.last_state.copy()}

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
    assert all(
        json.loads(Path(row["regenerated_init_states"]).read_text(encoding="utf-8"))["tasks"][0]["state_dim"] > 4
        for row in result
    )
    probe = FakeQposSimulator(3)
    assert np.array_equal(probe.get_sim_state(), probe.sim.get_state().flatten())


def test_b_generator_rejects_qpos_only_simulator_state(tmp_path: Path) -> None:
    source_bddl, source_init = _fake_source_roots(tmp_path)

    class QposOnlySimulator:
        def __init__(self, seed: int) -> None:
            del seed
            self.sim = types.SimpleNamespace(
                data=types.SimpleNamespace(qpos=np.arange(10, dtype=np.float64))
            )

        def reset(self) -> None:
            return None

        def close(self) -> None:
            return None

    with pytest.raises(VariantGenerationError, match="sim.data.qpos alone"):
        libero.generate_variant_initial_qpos(
            lambda **kwargs: QposOnlySimulator(int(kwargs["seed"])),
            tmp_path / "generated.pruned_init",
            task_id=0,
            bddl_path=source_bddl / f"{libero.LIBERO_TASK_SPECS[0].name}.bddl",
            source_init_path=source_init / f"{libero.LIBERO_TASK_SPECS[0].name}.pruned_init",
            count=B_INIT_STATE_COUNT,
            seeds=tuple(range(B_INIT_STATE_COUNT)),
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


def test_real_stage_s_calibration_shards_are_idempotent_and_aggregate_only(tmp_path: Path) -> None:
    settings = ["s0", "s1", "s2", "s3"]
    root = tmp_path / "calibration"
    calls: list[tuple[int, str, int, int, int, int]] = []

    def evaluator(
        setting_index: int,
        setting: str,
        task_id: int,
        init_state: int,
        candidate_id: int,
        trial_seed: int,
    ) -> bool:
        calls.append((setting_index, setting, task_id, init_state, candidate_id, trial_seed))
        return (trial_seed % 7) == 0

    rank0 = run_calibration_shard(
        evaluator,
        settings,
        root,
        calibration_seed=CALIBRATION_SEED,
        world_size=2,
        rank=0,
        substrate="B",
        sources=["variant0", "variant1", "variant2", "variant3"],
    )
    assert rank0["status"] == "completed"
    assert len(calls) == 512
    assert all(set(row) == {"setting", "successes", "total", "pooled_success"} for row in rank0["payload"]["rows"])
    assert [row["total"] for row in rank0["payload"]["rows"]] == [128] * 4

    # Replaying an already marked rank is a read-only verification and must
    # not execute any evaluator calls again.
    repeated = run_calibration_shard(
        evaluator,
        settings,
        root,
        calibration_seed=CALIBRATION_SEED,
        world_size=2,
        rank=0,
        substrate="B",
        sources=["variant0", "variant1", "variant2", "variant3"],
    )
    assert repeated["status"] == "already_complete"
    assert len(calls) == 512

    rank1 = run_calibration_shard(
        evaluator,
        settings,
        root,
        calibration_seed=CALIBRATION_SEED,
        world_size=2,
        rank=1,
        substrate="B",
        sources=["variant0", "variant1", "variant2", "variant3"],
    )
    assert rank1["status"] == "completed"
    assert len(calls) == 1024
    assert verify_calibration_shard(
        root / "shards" / "rank-00000",
        settings,
        calibration_seed=CALIBRATION_SEED,
        world_size=2,
        rank=0,
    )["rows"]
    assert verify_calibration_shard(
        root / "shards" / "rank-00001",
        settings,
        calibration_seed=CALIBRATION_SEED,
        world_size=2,
        rank=1,
    )["rows"]
    aggregate = aggregate_calibration_shards(
        root,
        settings,
        calibration_seed=CALIBRATION_SEED,
        world_size=2,
    )
    assert [row["total"] for row in aggregate["rows"]] == [256] * 4
    report = root / "CALIBRATION_RESULT.json"
    checked = verify_calibration_aggregate(
        report,
        settings,
        calibration_seed=CALIBRATION_SEED,
        world_size=2,
    )
    assert checked["rows"] == aggregate["rows"]
    assert (root / "SHA256SUMS").is_file()
    assert (root / "COMPLETED_CALIBRATION.json").is_file()
    assert (root / "shards" / "rank-00000" / "COMPLETED_SHARD.json").is_file()
    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            result = {str(key).lower() for key in value}
            for child in value.values():
                result.update(keys(child))
            return result
        if isinstance(value, list):
            result: set[str] = set()
            for child in value:
                result.update(keys(child))
            return result
        return set()

    persisted_keys = keys(json.loads(report.read_text(encoding="utf-8")))
    assert not persisted_keys.intersection({"genealogy", "trajectory", "actions", "poses", "s2", "s3", "s4", "s5"})


def test_stage_s_calibration_episode_runs_until_real_done() -> None:
    environments: list[FakeSnapshotEnvironment] = []

    def factory(**kwargs: object) -> FakeSnapshotEnvironment:
        del kwargs
        environment = FakeSnapshotEnvironment()
        environments.append(environment)
        return environment

    success = run_stage_s_calibration_episode(
        factory,
        FakePolicy(),
        setting_index=0,
        task_id=0,
        init_state=0,
        candidate_id=0,
        calibration_seed=CALIBRATION_SEED,
        max_steps=4,
    )
    assert success is True
    assert len(environments) == 1
    assert environments[0].step == 2
    assert environments[0].close_calls == 1


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

def test_full_snapshot_replay_rejects_hidden_drift_with_same_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class HiddenSnapshotEnvironment(FakeSnapshotEnvironment):
        def __init__(self) -> None:
            super().__init__()
            self.capture_count = 0

        def capture_snapshot(self) -> dict[str, object]:
            self.capture_count += 1
            snapshot = super().capture_snapshot()
            snapshot["hidden_integrator"] = np.asarray(
                [self.capture_count], dtype=np.int64
            )
            return snapshot

    environment = HiddenSnapshotEnvironment()
    environment.seed(11)
    environment.reset(3)
    monkeypatch.setattr(libero, "_require_full_torch_rng", lambda state: None)
    snapshot = capture_stage_r_snapshot(
        environment,
        [],
        stable_seed("hidden-snapshot"),
        0,
        0,
    )
    with pytest.raises(libero.SnapshotReplayError, match="same-action"):
        validate_restore_same_action(
            environment,
            snapshot,
            [0.5, 0.0],
            require_full_rng=True,
        )


def test_family_artifacts_are_atomic_and_resumable(tmp_path: Path) -> None:
    def factory(**kwargs: object) -> FakeSnapshotEnvironment:
        del kwargs
        return FakeSnapshotEnvironment()

    family = collect_family(
        factory,
        FakePolicy(),
        variant=types.SimpleNamespace(substrate="C"),
        task_id=0,
        init_state=0,
        candidate_count=4,
        max_steps=4,
    )
    assert family["candidate_count"] == 4
    assert family["policy_forwards"] == 4
    family["metadata_extra"] = {
        "protocol_authority_path": "/cpfs/stage_s/protocol/FROZEN_PROTOCOL.json",
        "protocol_authority_sha256": "a" * 64,
        "protocol_git_commit": "b" * 40,
        "substrate_annotation": "WEAK_SUBSTRATE",
    }
    target = tmp_path / "C" / "task00" / "init000"
    marker = write_family_atomic(target, family)
    assert marker["checkpoint"] == "FAMILY_COMPLETE"
    assert family["substrate"] == "C"
    with np.load(target / "rollouts.npz", allow_pickle=False) as rollouts:
        assert set(["candidate_index", "terminated", "terminal_step"]).issubset(
            rollouts.files
        )
        assert rollouts["poses"].shape[1] == 6
        assert np.array_equal(rollouts["candidate_index"], np.arange(4))
        assert np.all(rollouts["terminated"])
    assert marker["protocol_id"] == libero.STAGE_S_PROTOCOL_ID
    assert marker["protocol_authority_path"] == family["metadata_extra"]["protocol_authority_path"]
    assert marker["protocol_authority_sha256"] == family["metadata_extra"]["protocol_authority_sha256"]
    assert marker["protocol_git_commit"] == family["metadata_extra"]["protocol_git_commit"]
    assert marker["substrate"] == "C"
    assert marker["pose_dimension"] == 6
    assert marker["substrate_annotation"] == "WEAK_SUBSTRATE"
    assert family_is_complete(target, expected_candidates=4)
    (target / "rollouts.npz").write_bytes((target / "rollouts.npz").read_bytes() + b"corrupt")
    assert family_is_complete(target, expected_candidates=4) is False


def test_strict_bc_family_persists_candidate_replay_and_rejects_missing_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def factory(**kwargs: object) -> FakeSnapshotEnvironment:
        del kwargs
        return FakeSnapshotEnvironment()

    # The dev14 test interpreter is intentionally CPU/minimal and may not
    # ship Torch.  Keep the producer contract test deterministic by supplying
    # a shape-faithful stand-in; the production path still fails closed when
    # these streams cannot be captured.
    monkeypatch.setattr(
        libero,
        "_torch_rng_state",
        lambda: {"cpu": [1], "cuda": []},
    )
    monkeypatch.setattr(libero, "_require_full_torch_rng", lambda state: None)
    monkeypatch.setattr(libero, "_restore_torch_rng", lambda state: None)
    family = collect_family(
        factory,
        ReplayFakePolicy(),
        variant=types.SimpleNamespace(substrate="C"),
        task_id=0,
        init_state=0,
        candidate_count=4,
        max_steps=4,
        validate_snapshots=True,
    )
    family["metadata_extra"] = {
        "substrate_annotation": "WEAK_SUBSTRATE",
        "rank": 0,
        "world_size": 8,
    }
    target = tmp_path / "C" / "task00" / "init000"
    write_family_atomic(target, family)
    assert family_is_complete(target, expected_candidates=4, strict=True)
    with (target / "snapshots.pkl").open("rb") as stream:
        payload = pickle.load(stream)
    payload["candidates"]["0"].pop("snapshot_restore_check")
    bad = tmp_path / "bad-snapshots.pkl"
    with bad.open("wb") as stream:
        pickle.dump(payload, stream, protocol=5)
    with pytest.raises(MainEvaluationError, match="replay check"):
        _verify_snapshot(bad, expected_candidates=4)


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
    assert payload["resource"]["memory_gib"] == 1400
    assert payload["resource"]["resource_alias"] == "idle-a800-robot-stage-s-graphics-8gpu"
    assert payload["resource"]["resource_id"] == "quota1ssrabud0bh"
    assert payload["resource"]["quota_name"] == "exp-robot"
    assert payload["daily_no_job_windows"][0]["timezone"] == "Asia/Shanghai"
    assert payload["checkpoint_contract"]["retain_and_audit_steps"] == list(C_RETAIN_STEPS)


def test_task64_factory_maps_logical_seed_to_numpy_uint32(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    class FakeTask64Environment:
        def __init__(self, config: dict[str, object], seed: int) -> None:
            observed["config"] = config
            observed["seed"] = seed

    monkeypatch.setattr(libero, "_configure_stage_r_sources", lambda *_: (tmp_path, tmp_path))
    monkeypatch.setattr(
        libero.importlib,
        "import_module",
        lambda name: types.SimpleNamespace(Task64Environment=FakeTask64Environment),
    )
    factory = libero.make_stage_r_task64_factory(tmp_path, tmp_path)
    logical_seed = (1 << 48) + 17
    factory(task_id=0, init_state=0, candidate_id=0, seed=logical_seed)

    assert observed["seed"] == 17
