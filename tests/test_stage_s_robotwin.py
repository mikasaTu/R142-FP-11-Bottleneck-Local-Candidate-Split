from __future__ import annotations

import asyncio
import hashlib
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from scripts.stage_s_robotwin_audit import audit
from r142_stage_s.robotwin import (
    AtomicFamilyWriter,
    CandidateRecord,
    CapabilityError,
    ConcreteRoboTwinRuntime,
    ExactReplayVerifier,
    EVO_EXACT_REPLAY_PROTOCOL,
    EvoExactReplayServerControl,
    EvoProxyStateAdapter,
    FamilyRolloutRunner,
    PUBLISHED_CLEAN_SUCCESS,
    select_published_tasks,
)
from scripts.stage_s_robotwin_evo_server_patch import build_handle_request


class ControlSocket:
    """In-process WebSocket fake; control state is owned by the server shim."""

    def __init__(self, controller):
        self.controller = controller
        self.responses = []

    async def send(self, payload):
        response = self.controller.handle_message(json.loads(payload))
        if response is not None:
            self.responses.append(json.dumps(response))

    async def recv(self):
        if not self.responses:
            raise RuntimeError("control response queue is empty")
        return self.responses.pop(0)


class ControlProxy:
    def __init__(self, controller):
        self.ws = ControlSocket(controller)
        self.loop = asyncio.new_event_loop()

    def infer(self, *_args, **_kwargs):
        # This is only the transport fake.  The exact replay implementation
        # never treats this as evidence of a real model rollout.
        return [[0.1, 0.2]]


class HandlerSocket:
    def __init__(self, incoming):
        self.incoming = list(incoming)
        self.sent = []

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.incoming:
            raise StopAsyncIteration
        return self.incoming.pop(0)

    async def send(self, payload):
        self.sent.append(payload)


def test_evo_control_server_rejects_unversioned_and_ignores_inference():
    controller = EvoExactReplayServerControl(require_torch=False)
    assert controller.handle_message({"image": []}) is None
    result = controller.handle_message({"r142_control": "capture_rng"})
    assert result["r142_control"] == "error"
    assert "protocol" in result["error"]


def test_evo_proxy_protocol_restores_policy_rng_history_and_action_queue():
    controller = EvoExactReplayServerControl(require_torch=False)
    proxy = ControlProxy(controller)
    adapter = EvoProxyStateAdapter(proxy)
    try:
        adapter.set_seed(424242)
        adapter.observation_history = [{"state": [1.0], "prompt": "pick"}]
        adapter.action_queue = [[0.25, 0.5], [0.75, 1.0]]
        snapshot = {
            "history": adapter.capture_observation_history(),
            "queue": adapter.capture_action_queue(),
            "rng": adapter.capture_rng_state(),
        }
        # Advance both local and server-side streams and mutate the queues.
        adapter.rng.random()
        random.random()
        np.random.random()
        adapter.observation_history.append({"state": [9.0]})
        adapter.action_queue.pop()
        adapter.restore_observation_history(snapshot["history"])
        adapter.restore_action_queue(snapshot["queue"])
        adapter.restore_rng_state(snapshot["rng"])
        assert adapter.capture_observation_history() == snapshot["history"]
        assert adapter.capture_action_queue() == snapshot["queue"]
        restored = adapter.capture_rng_state()
        assert json.dumps(restored, sort_keys=True, default=str) == json.dumps(
            snapshot["rng"], sort_keys=True, default=str
        )
    finally:
        proxy.loop.close()


def test_evo_proxy_protocol_wires_exact_same_action_next_state_contract():
    controller = EvoExactReplayServerControl(require_torch=False)
    proxy = ControlProxy(controller)
    adapter = EvoProxyStateAdapter(proxy)

    class ProtocolPolicy:
        def capture_observation_history(self):
            return adapter.capture_observation_history()

        def restore_observation_history(self, value):
            adapter.restore_observation_history(value)

        def capture_action_queue(self):
            return adapter.capture_action_queue()

        def restore_action_queue(self, value):
            adapter.restore_action_queue(value)

        def capture_rng_state(self):
            return adapter.capture_rng_state()

        def restore_rng_state(self, value):
            adapter.restore_rng_state(value)

        def act(self, _observation):
            raw = adapter.infer(None, None, None, [0.0], "pick")
            return np.asarray(raw[0])

    try:
        adapter.set_seed(1234)
        result = ExactReplayVerifier(FakeEnv(), ProtocolPolicy()).verify_restore()
        assert result["passed"]
        assert result["action_error"] <= 1e-9
        assert result["next_state_error"] <= 1e-9
    finally:
        proxy.loop.close()


def test_evo_server_drop_in_handler_keeps_inference_branch_unchanged():
    calls = []

    def infer(payload, model, normalizer, arm_key, dataset_key):
        calls.append((payload, model, normalizer, arm_key, dataset_key))
        return [[7.0]]

    handler = build_handle_request(
        infer_from_json_dict=infer,
        model="model",
        normalizer="normalizer",
        arm_key="aloha_joint",
        dataset_key="robotwin_blocks_ranking_size",
        require_torch=False,
    )
    inference_payload = {"image": [], "state": [], "prompt": "pick"}
    control_payload = {
        "r142_control": "capture_rng",
        "protocol": EVO_EXACT_REPLAY_PROTOCOL,
        "request_id": "test-request",
    }
    websocket = HandlerSocket(
        [json.dumps(inference_payload), json.dumps(control_payload)]
    )
    asyncio.run(handler(websocket))
    assert calls == [(
        inference_payload,
        "model",
        "normalizer",
        "aloha_joint",
        "robotwin_blocks_ranking_size",
    )]
    assert json.loads(websocket.sent[0]) == [[7.0]]
    control_response = json.loads(websocket.sent[1])
    assert control_response["r142_control"] == "ok"
    assert control_response["request_id"] == "test-request"


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


class RolloutEnv(FakeEnv):
    def __init__(self):
        super().__init__()
        self.scene = FakeScene()
        self.scene.actor.pose = np.zeros(2)
        self.take_action_cnt = 0
        self.step_lim = 3
        self.eval_success = False

    def reset(self, seed, task_name=None):
        self.state = np.array([0.0, 0.0])
        self.scene.actor.pose = np.zeros(2)
        self.take_action_cnt = 0
        self.eval_success = False

    def capture_simulator_state(self):
        return {
            "state": self.state.copy(),
            "actor": self.scene.actor.pose.copy(),
            "step": self.take_action_cnt,
            "success": self.eval_success,
        }

    def restore_simulator_state(self, value):
        self.state = np.asarray(value["state"], dtype=float).copy()
        self.scene.actor.pose = np.asarray(value["actor"], dtype=float).copy()
        self.take_action_cnt = int(value["step"])
        self.eval_success = bool(value["success"])

    def get_obs(self):
        return {"endpose": self.state.copy(), "state": self.state.copy()}

    def take_action(self, action):
        delta = np.asarray(action, dtype=float)
        self.state += delta
        self.scene.actor.pose += delta
        self.take_action_cnt += 1
        self.eval_success = self.take_action_cnt >= 2


class RolloutPolicy(FakePolicy):
    def __init__(self, seed=0):
        super().__init__()
        self.rng = np.random.default_rng(seed)

    def set_rng(self, rng):
        self.rng = rng

    def capture_rng_state(self):
        return self.rng.bit_generator.state.copy()

    def restore_rng_state(self, value):
        self.rng.bit_generator.state = value.copy()

    def act(self, observation, **kwargs):
        return np.array([0.1, 0.2])


def test_runner_requires_gate_and_persists_seed_eef_object_genealogy(tmp_path):
    writer = AtomicFamilyWriter(tmp_path)
    runner = FamilyRolloutRunner(
        env_factory=RolloutEnv,
        policy_factory=lambda seed: RolloutPolicy(seed),
        writer=writer,
    )
    manifest = runner.run_family(
        task_name="blocks_ranking_size",
        family_id="family-1",
        initial_state_id="blocks_ranking_size/state-0000",
        initial_seed=14211,
        candidate_count=2,
    )
    payload = json.loads((tmp_path / "family-1" / "family.json").read_text())
    assert manifest["candidate_count"] == 2
    assert payload["metadata"]["replay_capability_gate"]["passed"]
    snapshot = json.loads((tmp_path / "family-1" / "SNAPSHOT.json").read_text())
    assert set(snapshot) == {"simulator", "policy_history", "action_queue", "rng_streams"}
    for candidate in payload["candidates"]:
        assert candidate["seed_sequence"]
        assert candidate["seed_genealogy"]["root_seed"] == 14211
        assert candidate["eef_trajectory"]
        assert candidate["object_trajectories"]["object"]
        assert candidate["policy_history"] is not None
        assert candidate["action_queue"] is not None
        assert set(candidate["rng_state"]) == {"environment", "policy"}
        assert candidate["final_success"] is True


def test_completed_family_is_idempotent_and_corruption_fails_closed(tmp_path):
    writer = AtomicFamilyWriter(tmp_path)
    record = CandidateRecord(
        candidate_id="family-0/candidate-0000",
        parent_id=None,
        generation_step=0,
        final_success=False,
    )
    first = writer.write("family-0", [record])
    second = writer.write("family-0", [record])
    assert second["skipped_existing"] is True
    assert second["completion_sha256"] == first["completion_sha256"]
    (tmp_path / "family-0" / "genealogy.jsonl").write_text("corrupted\n")
    with pytest.raises(CapabilityError, match="immutable family completion marker"):
        writer.write("family-0", [record])


def test_payload_builder_freezes_robot_idle_shard_and_guard(tmp_path):
    from scripts.stage_s_robotwin_main import BEIJING, assert_outside_blackout
    from scripts.stage_s_robotwin_payload import build_payload

    safe_now = datetime(2026, 9, 2, 12, 0, tzinfo=BEIJING)
    payload = build_payload(
        run_id="stage-s-rank-0000",
        output_root=tmp_path / "A",
        robotwin_root=Path("/dev14/robotwin"),
        evo_root=Path("/dev14/evo"),
        checkpoint_dir=Path("/dev14/checkpoint_ce8c583724706fbf7a03c17237761c65bf6813a7"),
        server_url="ws://127.0.0.1:9000",
        rank=0,
        world_size=4,
        now=safe_now,
    )
    assert payload["resources"] == {
        "pool": "robot",
        "resource_mode": "idle",
        "preemptible": True,
        "gpu_type": "A800",
        "gpu_count": 8,
        "cpu_cores": 88,
        "memory_gib": 1525,
    }
    assert payload["shard"]["assignment"].startswith("flat_task_family_index")
    assert payload["submission"]["submit"] is False
    with pytest.raises(CapabilityError, match="blackout"):
        assert_outside_blackout(datetime(2026, 9, 2, 9, 35, tzinfo=BEIJING))
    with pytest.raises(CapabilityError, match="blackout"):
        build_payload(
            run_id="stage-s-rank-0000",
            output_root=tmp_path / "B",
            robotwin_root=Path("/dev14/robotwin"),
            evo_root=Path("/dev14/evo"),
            checkpoint_dir=Path("/dev14/checkpoint_ce8c583724706fbf7a03c17237761c65bf6813a7"),
            server_url="ws://127.0.0.1:9000",
            now=datetime(2026, 9, 2, 19, 35, tzinfo=BEIJING),
        )
