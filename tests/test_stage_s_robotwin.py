from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from scripts.stage_s_robotwin_audit import audit
from r142_stage_s.robotwin import (
    AtomicFamilyWriter,
    CandidateRecord,
    CapabilityError,
    ConcreteRoboTwinRuntime,
    ExactReplayVerifier,
    PUBLISHED_CLEAN_SUCCESS,
    select_published_tasks,
)


def test_task_selection_is_lexical_and_exact():
    selected = select_published_tasks()
    assert selected == (
        "blocks_ranking_size",
        "pick_diverse_bottles",
        "place_a2b_left",
        "place_a2b_right",
        "place_bread_basket",
        "place_bread_skillet",
        "place_can_basket",
        "place_fan",
        "place_object_scale",
        "place_shoe",
    )
    assert len(selected) == 10
    assert all(0.25 <= PUBLISHED_CLEAN_SUCCESS[x] <= 0.65 for x in selected)


class FakeEnv:
    def __init__(self):
        self.state = np.array([1.0, 2.0])
        self.rng = {"seed": 7}

    def capture_simulator_state(self):
        return self.state.copy()

    def restore_simulator_state(self, value):
        self.state = np.asarray(value, dtype=float).copy()

    def capture_rng_state(self):
        return dict(self.rng)

    def restore_rng_state(self, value):
        self.rng = dict(value)

    def state_for_verification(self):
        return self.state.copy()

    def take_action(self, action):
        self.state += np.asarray(action)


class FakePolicy:
    def __init__(self):
        self.history = ["obs"]
        self.queue = [0.25]
        self.rng = {"seed": 11}

    def capture_observation_history(self):
        return list(self.history)

    def restore_observation_history(self, value):
        self.history = list(value)

    def capture_action_queue(self):
        return list(self.queue)

    def restore_action_queue(self, value):
        self.queue = list(value)

    def capture_rng_state(self):
        return dict(self.rng)

    def restore_rng_state(self, value):
        self.rng = dict(value)

    def act(self, observation):
        return np.array([0.1, 0.2])


class FakeActor:
    def __init__(self, name):
        self.name = name
        self.pose = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.angular_velocity = np.array([0.0, 0.0, 0.0])

    def get_name(self):
        return self.name

    def get_pose(self):
        return self.pose.copy()

    def set_pose(self, value):
        self.pose = np.asarray(value, dtype=float).copy()

    def get_velocity(self):
        return self.velocity.copy()

    def set_velocity(self, value):
        self.velocity = np.asarray(value, dtype=float).copy()

    def get_angular_velocity(self):
        return self.angular_velocity.copy()

    def set_angular_velocity(self, value):
        self.angular_velocity = np.asarray(value, dtype=float).copy()


class FakeArticulation:
    def __init__(self):
        self.root_pose = np.array([0.0, 0.0, 0.0])
        self.qpos = np.array([0.1, 0.2])
        self.qvel = np.array([0.0, 0.0])
        self.qacc = np.array([0.0, 0.0])

    def get_root_pose(self):
        return self.root_pose.copy()

    def set_root_pose(self, value):
        self.root_pose = np.asarray(value, dtype=float).copy()

    def get_qpos(self):
        return self.qpos.copy()

    def set_qpos(self, value):
        self.qpos = np.asarray(value, dtype=float).copy()

    def get_qvel(self):
        return self.qvel.copy()

    def set_qvel(self, value):
        self.qvel = np.asarray(value, dtype=float).copy()

    def get_qacc(self):
        return self.qacc.copy()

    def set_qacc(self, value):
        self.qacc = np.asarray(value, dtype=float).copy()


class FakeScene:
    def __init__(self):
        self.actor = FakeActor("object")
        self.articulation = FakeArticulation()

    def get_all_actors(self):
        return [self.actor]

    def get_all_articulations(self):
        return [self.articulation]


class FakeTaskEnv:
    def __init__(self):
        self.scene = FakeScene()
        self.take_action_cnt = 3
        self.step_lim = 20
        self.eval_success = False
        self.plan_success = True
        self.stage_success_tag = False
        self.left_cnt = 2
        self.right_cnt = 4
        self.now_obs = {"state": np.array([1.0])}

    def state_for_verification(self):
        return {
            "actor": self.scene.actor.get_pose(),
            "qpos": self.scene.articulation.get_qpos(),
            "step": self.take_action_cnt,
        }

    def take_action(self, action):
        self.scene.actor.pose += np.asarray(action)
        self.take_action_cnt += 1


def test_concrete_runtime_restores_sapien_actor_articulation_and_step_count():
    env, policy = FakeTaskEnv(), FakePolicy()
    runtime = ConcreteRoboTwinRuntime(env, policy, require_torch=False)
    snapshot = runtime.capture_snapshot()
    env.scene.actor.pose[:] = 9.0
    env.scene.articulation.qpos[:] = 8.0
    env.take_action_cnt = 99
    runtime.restore_snapshot(snapshot)
    assert np.array_equal(env.scene.actor.pose, [0.0, 0.0, 0.0])
    assert np.array_equal(env.scene.articulation.qpos, [0.1, 0.2])
    assert env.take_action_cnt == 3


def test_audit_rejects_sources_without_concrete_wrapper(tmp_path):
    blocked = audit(
        robotwin_root=tmp_path / "robotwin",
        evo_root=tmp_path / "evo",
        checkpoint_dir=tmp_path / "checkpoint",
    )
    assert blocked["status"] == "BLOCKED_CAPABILITY"
    assert not blocked["concrete_wrapper_verified"]
    assert "concrete wrapper" in blocked["capability_error"]


def test_audit_accepts_explicit_concrete_wrapper_marker(tmp_path):
    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text(
        "class ConcreteRoboTwinRuntime: pass\n"
        "class EvoProxyStateAdapter: pass\n",
        encoding="utf-8",
    )
    result = audit(
        robotwin_root=tmp_path / "robotwin",
        evo_root=tmp_path / "evo",
        checkpoint_dir=tmp_path / "checkpoint",
        runtime_wrapper=wrapper,
    )
    assert result["concrete_wrapper_verified"]
    assert "concrete wrapper" not in result["capability_error"]


def test_exact_replay_is_verified_at_one_e_minus_nine():
    result = ExactReplayVerifier(FakeEnv(), FakePolicy()).verify_restore()
    assert result["passed"]
    assert result["action_error"] <= 1e-9
    assert result["next_state_error"] <= 1e-9


def test_missing_hook_fails_closed():
    env, policy = FakeEnv(), FakePolicy()
    del policy.history
    with pytest.raises(CapabilityError, match="policy observation history"):
        ExactReplayVerifier(env, policy).capture()


def test_atomic_outcome_genealogy_and_sha(tmp_path):
    writer = AtomicFamilyWriter(tmp_path)
    record = CandidateRecord(
        candidate_id="family-0/candidate-0000",
        parent_id=None,
        generation_step=0,
        action_prefix=[[1, 2]],
        pose_trajectory=[[0.0, 0.1]],
        final_success=True,
        task_name="blocks_ranking_size",
        family_id="family-0",
        initial_state_id="state-0",
        seed=123,
        policy_forwards=1,
        env_steps=1,
    )
    manifest = writer.write("family-0", [record], metadata={"protocol_sha": "abc"})
    directory = tmp_path / "family-0"
    assert (directory / "family.json").is_file()
    assert (directory / "genealogy.jsonl").is_file()
    marker = json.loads((directory / "COMPLETED_FAMILY.json").read_text())
    assert marker["files"]["family.json"] == hashlib.sha256(
        (directory / "family.json").read_bytes()
    ).hexdigest()
    assert manifest["candidate_count"] == 1
