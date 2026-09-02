from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from r142_stage_s.robotwin import (
    AtomicFamilyWriter,
    CandidateRecord,
    CapabilityError,
    ExactReplayVerifier,
    PUBLISHED_CLEAN_SUCCESS,
    select_published_tasks,
)


def test_task_selection_is_lexical_and_exact():
    selected = select_published_tasks()
    assert selected == (
        "blocks_ranking_size",
        "place_a2b_left",
        "place_a2b_right",
        "place_bread_basket",
        "place_bread_skillet",
        "place_can_basket",
        "place_fan",
        "place_object_scale",
        "place_shoe",
        "put_object_cabinet",
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
