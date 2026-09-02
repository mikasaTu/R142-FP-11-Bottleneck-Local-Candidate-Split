from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from r142_stage_s.s45_adapters import (
    LiberoS45Adapter,
    RoboTwinS45Adapter,
    S45AdapterError,
    make_libero_s45_adapter,
    make_robotwin_s45_adapter,
)
from r142_stage_s.s45_runtime import ProtocolAuthority, S45ProvenanceError


COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _protocol(tmp_path: Path) -> ProtocolAuthority:
    payload = {
        "protocol_id": "fixture-stage-s",
        "protocol_git_commit": COMMIT,
        "s4": {
            "anchor_rule": "candidate_index=0",
            "oracle_t_rule": "fixture-oracle-grid",
            "random_t_rule": "fixture-random-grid",
            "oracle_t_grid": list(range(1, 10)),
            "branch_count": 4,
            "search_branch_count": 4,
            "heldout_branch_count": 8,
            "random_branch_count": 8,
            "branch_seed_formula": "fixture-branch-seed-v1",
            "random_location_hash_formula": "sha256(protocol_id|s4|family_id|episode_length|pair_index)->first_8_bytes_big_endian_mod_interior",
            "branch_seeds": {
                "family-00|oracle|0": 7000,
                "family-00|oracle|1": 7001,
                "family-00|random|0": 7100,
                "family-00|random|1": 7101,
            },
        },
        "s5": {
            "base_candidate_count": 32,
            "fresh_candidate_indices": list(range(32, 64)),
            "extension_seed_formula": "fixture-extension-seed-v1",
            "extension_seeds": {f"family-00|{i}": 9000 + i for i in range(32, 64)},
        },
    }
    path = tmp_path / "FROZEN_PROTOCOL.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return ProtocolAuthority.load(path)


def _family() -> dict:
    candidates = []
    for index in range(32):
        candidates.append(
            {
                "candidate_index": index,
                "candidate_id": f"family-00/candidate-{index:04d}",
                "candidate_seed": 1000 + index,
                "actions": [[1.0, 0.0] for _ in range(4)],
                "trajectory": [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(4)],
                "success": False,
                "termination": "official-step-limit",
                "env_steps": 4,
            }
        )
    return {
        "family_id": "family-00",
        "task_id": 0,
        "init_state_id": 0,
        "metadata": {"task_id": 0, "init_state": 0, "task_name": "fixture-task", "initial_seed": 42},
        "substrate": "B",
        "candidates": candidates,
    }


class FakeLiberoEnv:
    def __init__(self, **_: object) -> None:
        self.state = 0.0
        self.step = 0
        self.seed_value = 0
        self._observation = {"observation/state": np.asarray([0.0] + [0.0] * 7, dtype=np.float64)}
        self.owner_rng = {"state": 0}

    def seed(self, seed: int) -> None:
        self.seed_value = int(seed)

    def reset(self, init_state: int) -> dict[str, np.ndarray]:
        self.state = float(init_state)
        self.step = 0
        self._observation = {"observation/state": np.asarray([self.state] + [0.0] * 7, dtype=np.float64)}
        return self._observation

    def capture_snapshot(self) -> dict[str, object]:
        return {"state": self.state, "step": self.step, "seed": self.seed_value}

    def restore_snapshot(self, value: dict[str, object]) -> None:
        self.state = float(value["state"])
        self.step = int(value["step"])
        self.seed_value = int(value["seed"])
        self._observation = {"observation/state": np.asarray([self.state] + [0.0] * 7, dtype=np.float64)}

    def capture_rng_state(self) -> dict[str, int]:
        return copy.deepcopy(self.owner_rng)

    def restore_rng_state(self, value: dict[str, int]) -> None:
        self.owner_rng = copy.deepcopy(value)

    def execute_actions(self, actions: np.ndarray) -> dict[str, object]:
        action = np.asarray(actions, dtype=np.float64)
        if action.ndim == 2:
            action = action[0]
        self.state += float(action[0])
        self.step += 1
        self._observation = {"observation/state": np.asarray([self.state] + [0.0] * 7, dtype=np.float64)}
        done = self.step >= 4
        return {"done": done, "success": False}

    def state_vector(self) -> np.ndarray:
        return np.asarray([self.state, float(self.step)], dtype=np.float64)

    def pose_vector(self) -> np.ndarray:
        return np.asarray([self.state, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def close(self) -> None:
        return None


class FakeLiberoPolicy:
    def __init__(self, **_: object) -> None:
        self.rng_state = {"seed": 0}

    def sample_action_chunk(self, observation: object, *, seed: int, counter: int) -> np.ndarray:
        del observation, counter
        self.rng_state["seed"] = int(seed)
        return np.tile(np.asarray([1.0, 0.0], dtype=np.float32), (5, 1))

    def capture_rng_state(self) -> dict[str, int]:
        return copy.deepcopy(self.rng_state)

    def restore_rng_state(self, value: dict[str, int]) -> None:
        self.rng_state = copy.deepcopy(value)


def test_libero_adapter_uses_real_snapshot_and_prefix_contract(tmp_path: Path) -> None:
    protocol = _protocol(tmp_path)
    env_factory = lambda **kwargs: FakeLiberoEnv(**kwargs)
    policy_factory = lambda **kwargs: FakeLiberoPolicy(**kwargs)
    adapter = make_libero_s45_adapter(
        env_factory,
        policy_factory,
        protocol=protocol,
        substrate="B",
        max_steps=8,
        require_torch=False,
    )
    family = _family()
    anchor = adapter.select_anchor(family, protocol=protocol)
    prefix = adapter.replay_prefix(family, anchor, 1, protocol=protocol)
    assert prefix["snapshot"]["simulator_state"]["step"] == 1
    assert set(prefix["snapshot"]) >= {"simulator_state", "observation_history", "action_queue", "rng_state"}
    execution = adapter.run_branch(family, anchor, prefix, 1, 7000, 0, "oracle", protocol=protocol)
    assert execution["terminated"] is True
    assert execution["snapshot_restore_check"]["max_abs_error"] <= 1e-9
    assert len(execution["actions"]) >= 4
    adapter.close()


def test_libero_adapter_fails_closed_without_owner_rng(tmp_path: Path) -> None:
    protocol = _protocol(tmp_path)

    class NoRng(FakeLiberoEnv):
        capture_rng_state = None

    adapter = LiberoS45Adapter(
        lambda **kwargs: NoRng(**kwargs),
        lambda **kwargs: FakeLiberoPolicy(**kwargs),
        protocol=protocol,
        substrate="C",
        max_steps=8,
        require_torch=False,
    )
    with pytest.raises(S45AdapterError, match="owner-specific RNG capture"):
        adapter.replay_prefix(_family(), _family()["candidates"][0], 1, protocol=protocol)


def test_seed_formula_is_read_from_protocol_without_adapter_salt(tmp_path: Path) -> None:
    protocol_path = _protocol(tmp_path).path
    payload = json.loads(protocol_path.read_text())
    payload["s4"]["branch_seed_formula"] = "sha256(protocol_id|s4|family_id|mode|branch_index)->first_8_bytes_big_endian"
    payload["s4"].pop("branch_seeds")
    payload["s5"]["extension_seed_formula"] = "sha256(protocol_id|s5|family_id|candidate_index)->first_8_bytes_big_endian"
    payload["s5"].pop("extension_seeds")
    protocol_path.write_text(json.dumps(payload, sort_keys=True))
    protocol = ProtocolAuthority.load(protocol_path)
    adapter = LiberoS45Adapter(
        lambda **kwargs: FakeLiberoEnv(**kwargs),
        lambda **kwargs: FakeLiberoPolicy(**kwargs),
        protocol=protocol,
        substrate="C",
        max_steps=8,
        require_torch=False,
    )
    family = _family()
    seed = adapter.branch_seed(family, family["candidates"][0], 1, 0, "oracle", protocol=protocol)
    assert isinstance(seed, int) and seed >= 0
    extension = adapter.extension_seed(family, 32, protocol=protocol)
    assert isinstance(extension, int) and extension >= 0


class FakeActor:
    def __init__(self) -> None:
        self.pose = np.asarray([0.0], dtype=np.float64)

    def get_pose(self) -> np.ndarray:
        return self.pose.copy()

    def set_pose(self, value: np.ndarray) -> None:
        self.pose = np.asarray(value, dtype=np.float64).copy()

    def get_velocity(self) -> np.ndarray:
        return np.zeros(1)

    def set_velocity(self, value: np.ndarray) -> None:
        del value

    def get_angular_velocity(self) -> np.ndarray:
        return np.zeros(1)

    def set_angular_velocity(self, value: np.ndarray) -> None:
        del value

    def get_name(self) -> str:
        return "fixture-actor"


class FakeScene:
    def __init__(self) -> None:
        self.actor = FakeActor()

    def get_all_actors(self):
        return [self.actor]

    def get_all_articulations(self):
        return []


class FakeRoboTwinEnv:
    def __init__(self, **_: object) -> None:
        self.scene = FakeScene()
        self.state = 0.0
        self.take_action_cnt = 0
        self.step_lim = 4
        self.eval_success = False
        self.owner_rng = {"seed": 0}

    def get_obs(self) -> dict[str, object]:
        value = float(self.scene.actor.pose[0])
        return {"pose": [value] + [0.0] * 13, "state": [value]}

    def state_for_verification(self) -> np.ndarray:
        # The official ConcreteRoboTwinRuntime restores scene actor poses and
        # the documented counters. Keep verification state within those
        # persisted simulator fields; an unregistered ad-hoc Python scalar
        # would correctly fail the exact replay contract.
        return np.asarray([self.scene.actor.pose[0], self.take_action_cnt], dtype=np.float64)

    def take_action(self, action: object) -> None:
        self.state = float(self.scene.actor.pose[0]) + float(np.asarray(action, dtype=np.float64).reshape(-1)[0])
        self.take_action_cnt += 1
        self.scene.actor.pose = np.asarray([self.state], dtype=np.float64)
        self.eval_success = self.take_action_cnt >= self.step_lim

    def capture_rng_state(self) -> dict[str, int]:
        return copy.deepcopy(self.owner_rng)

    def restore_rng_state(self, value: dict[str, int]) -> None:
        self.owner_rng = copy.deepcopy(value)

    def close(self) -> None:
        return None


class FakeRoboTwinPolicy:
    def __init__(self, **_: object) -> None:
        self.rng_state = {"seed": 0}
        self.history: list[object] = []
        self.queue: list[object] = []

    def act(self, observation: object) -> list[float]:
        self.history.append(copy.deepcopy(observation))
        self.queue = [[1.0, 0.0]]
        return [1.0, 0.0]

    def set_seed(self, seed: int) -> None:
        self.rng_state = {"seed": int(seed)}

    def capture_observation_history(self):
        return copy.deepcopy(self.history)

    def restore_observation_history(self, value):
        self.history = copy.deepcopy(value)

    def capture_action_queue(self):
        return copy.deepcopy(self.queue)

    def restore_action_queue(self, value):
        self.queue = copy.deepcopy(value)

    def capture_rng_state(self):
        return copy.deepcopy(self.rng_state)

    def restore_rng_state(self, value):
        self.rng_state = copy.deepcopy(value)

    def close(self) -> None:
        return None


def test_robotwin_adapter_delegates_to_concrete_snapshot_runtime(tmp_path: Path) -> None:
    protocol = _protocol(tmp_path)
    adapter = make_robotwin_s45_adapter(
        lambda **kwargs: FakeRoboTwinEnv(**kwargs),
        lambda **kwargs: FakeRoboTwinPolicy(**kwargs),
        lambda environment, **kwargs: None,
        protocol=protocol,
        max_steps=8,
        require_torch=False,
    )
    family = _family()
    anchor = adapter.select_anchor(family, protocol=protocol)
    prefix = adapter.replay_prefix(family, anchor, 1, protocol=protocol)
    assert set(prefix["snapshot"]) == {"simulator_state", "observation_history", "action_queue", "rng_state"}
    execution = adapter.run_branch(family, anchor, prefix, 1, 7000, 0, "oracle", protocol=protocol)
    assert execution["terminated"] is True
    assert execution["snapshot_restore_check"]["max_abs_error"] <= 1e-9
    adapter.close()


def test_robotwin_requires_explicit_reset_factory(tmp_path: Path) -> None:
    protocol = _protocol(tmp_path)
    with pytest.raises(S45AdapterError):
        RoboTwinS45Adapter(
            lambda **kwargs: FakeRoboTwinEnv(**kwargs),
            lambda **kwargs: FakeRoboTwinPolicy(**kwargs),
            None,  # type: ignore[arg-type]
            protocol=protocol,
            max_steps=8,
            require_torch=False,
        )
