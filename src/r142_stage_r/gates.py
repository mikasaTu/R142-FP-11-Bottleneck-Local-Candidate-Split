from __future__ import annotations

import copy
import hashlib
import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from .controls import generate_control_bank
from .phase0 import Phase0Runtime, load_task_rollouts
from .protocol import PROTOCOL_ID, atomic_json, ranked_initial_states, sha256_file
from .runtime import (
    TrajectoryRunner,
    configure_external_sources,
    infer_physical_many,
    policy_noise,
    task_config,
)

EXPECTED_CHECKPOINT_TREE_SHA256 = "42d571bd87f05f1182810f5a8bfa6d084c0d0dd277aff739bcf8f69868e6fb99"
TASK64_PROMPT = "stack the right bowl on the left bowl and place them in the tray"


def _validated_task_cache(
    output_root: Path,
    *,
    suite_name: str,
    task_id: int,
    expected_rollouts: int,
) -> dict[str, Any] | None:
    stem = f"{suite_name}_task{int(task_id):02d}"
    data_path = output_root / f"{stem}.npz"
    metadata_path = output_root / f"{stem}.json"
    if not data_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    expected = {
        "protocol_id": PROTOCOL_ID,
        "suite": suite_name,
        "task_id": int(task_id),
        "rollout_count": int(expected_rollouts),
        "data_file": data_path.name,
        "data_sha256": sha256_file(data_path),
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        return None
    return metadata


def checkpoint_tree_sha256(root: str | Path) -> str:
    base = Path(root)
    aggregate = hashlib.sha256()
    for path in sorted(value for value in base.rglob("*") if value.is_file()):
        relative = "./" + path.relative_to(base).as_posix()
        aggregate.update(f"{sha256_file(path)}  {relative}\n".encode("utf-8"))
    return aggregate.hexdigest()


def _sim_vector(environment: Any) -> np.ndarray:
    base = environment.environment.env
    pieces = [np.asarray(base.sim.get_state().flatten(), dtype=np.float64).ravel()]
    for name in ("ctrl", "qacc_warmstart", "qfrc_applied", "xfrc_applied", "mocap_pos", "mocap_quat", "act", "userdata"):
        if hasattr(base.sim.data, name):
            pieces.append(np.asarray(getattr(base.sim.data, name), dtype=np.float64).ravel())
    return np.concatenate(pieces)


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    return float(np.max(np.abs(left - right))) if left.size else 0.0


def _instrumented_action(policy: Any, observation: dict[str, Any], noise: np.ndarray) -> np.ndarray:
    return infer_physical_many(policy, [copy.deepcopy(observation)], np.asarray(noise)[None])[0]


def _execute_one(environment: Any, action: np.ndarray) -> dict[str, Any]:
    trace = environment.execute_actions(np.asarray(action, dtype=np.float32)[None])
    return {
        "state": _sim_vector(environment),
        "raw": environment.raw_observation(),
        "success": bool(trace["success"]),
        "done": bool(trace["done"]),
        "reward": float(trace["raw_reward"]),
    }


def _raw_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(np.array_equal(np.asarray(left[key]), np.asarray(right[key])) for key in ("observation/image", "observation/wrist_image", "observation/state")) and left["prompt"] == right["prompt"]


def run_engineering_gates(
    *,
    qpilots_root: str,
    libero_root: str,
    checkpoint: str,
    output: str | Path,
    microbatch: int = 4,
    run_e6: bool = True,
) -> dict[str, Any]:
    output_root = Path(output)
    output_root.mkdir(parents=True, exist_ok=True)
    configure_external_sources(qpilots_root, libero_root)
    from qpilots_libero.environment import DUMMY_ACTION, Task64Environment
    from qpilots_libero.policy import CleanPi05LiberoPolicy

    e1 = {
        "checkpoint_tree_sha256": checkpoint_tree_sha256(checkpoint),
        "expected_checkpoint_tree_sha256": EXPECTED_CHECKPOINT_TREE_SHA256,
    }
    e1["pass"] = e1["checkpoint_tree_sha256"] == e1["expected_checkpoint_tree_sha256"]
    atomic_json(output_root / "E1_manifest.json", e1)

    config = task_config("libero_90", 64, TASK64_PROMPT, 400)
    env = Task64Environment(config, seed=14264)
    init_state = ranked_initial_states("libero_90", 64)[0]
    observation = env.reset(init_state)
    policy = CleanPi05LiberoPolicy(checkpoint, default_prompt=TASK64_PROMPT)
    noise = policy_noise(policy, 142_640_001, 0)
    official = policy.infer_official(observation, noise)
    wrapped = _instrumented_action(policy, observation, noise)
    e2 = {
        "shape": list(official.shape),
        "bit_identical": bool(np.array_equal(official, wrapped)),
        "max_abs_error": _max_abs(official.astype(np.float64), wrapped.astype(np.float64)),
    }
    e2["pass"] = bool(e2["bit_identical"])
    atomic_json(output_root / "E2_bit_identity.json", e2)

    snapshot = env.capture_snapshot()
    action = official[0].copy()
    left = _execute_one(env, action)
    env.restore_snapshot(snapshot)
    right = _execute_one(env, action)
    e3 = {
        "next_state_max_abs_error": _max_abs(left["state"], right["state"]),
        "raw_observation_equal": _raw_equal(left["raw"], right["raw"]),
        "reward_equal": left["reward"] == right["reward"],
        "success_equal": left["success"] == right["success"],
        "done_equal": left["done"] == right["done"],
    }
    e3["pass"] = bool(
        e3["next_state_max_abs_error"] <= 1e-9
        and e3["raw_observation_equal"]
        and e3["reward_equal"]
        and e3["success_equal"]
        and e3["done_equal"]
    )
    atomic_json(output_root / "E3_restore.json", e3)

    runner = TrajectoryRunner(env, noise_seed=142_640_101)
    runner.reset(init_state)
    history_anchor = runner.snapshot()
    history_obs = copy.deepcopy(runner.observation_history[-1])
    anchor_noise = policy_noise(policy, runner.noise_seed, runner.noise_counter)
    reference_plan = _instrumented_action(policy, history_obs, anchor_noise)

    runner.restore(history_anchor)
    corrupted = copy.deepcopy(history_obs)
    corrupted["observation/image"] = np.zeros_like(corrupted["observation/image"])
    corrupted["observation/wrist_image"] = np.zeros_like(corrupted["observation/wrist_image"])
    runner.observation_history = deque([corrupted], maxlen=1)
    runner.restore(history_anchor, omit="history")
    history_plan = _instrumented_action(policy, runner.observation_history[-1], anchor_noise)

    runner.restore(history_anchor)
    runner.noise_counter += 1
    runner.restore(history_anchor, omit="rng")
    rng_noise = policy_noise(policy, runner.noise_seed, runner.noise_counter)
    rng_plan = _instrumented_action(policy, runner.observation_history[-1], rng_noise)

    runner.restore(history_anchor)
    runner.action_queue.extend(reference_plan[:5])
    for _ in range(2):
        queued = np.asarray(runner.action_queue.popleft()).copy()
        _execute_one(env, queued)
        runner.observation_history.append(env.raw_observation())
    queue_anchor = runner.snapshot()
    runner.restore(queue_anchor)
    reference_action = np.asarray(runner.action_queue[0]).copy()
    reference_step = _execute_one(env, reference_action)

    runner.restore(queue_anchor)
    runner.action_queue.popleft()
    runner.restore(queue_anchor, omit="queue")
    queue_action = np.asarray(runner.action_queue[0]).copy()
    queue_step = _execute_one(env, queue_action)

    runner.restore(queue_anchor)
    _execute_one(env, DUMMY_ACTION)
    runner.restore(queue_anchor, omit="simulator")
    simulator_step = _execute_one(env, reference_action)

    runner.restore(queue_anchor)
    full_step = _execute_one(env, reference_action)
    e4 = {
        "full_restore_next_state_error": _max_abs(reference_step["state"], full_step["state"]),
        "simulator_ablation_next_state_error": _max_abs(reference_step["state"], simulator_step["state"]),
        "queue_ablation_action_error": _max_abs(reference_action.astype(np.float64), queue_action.astype(np.float64)),
        "queue_ablation_next_state_error": _max_abs(reference_step["state"], queue_step["state"]),
        "policy_input_buffer_ablation_action_error": _max_abs(reference_plan.astype(np.float64), history_plan.astype(np.float64)),
        "rng_ablation_action_error": _max_abs(reference_plan.astype(np.float64), rng_plan.astype(np.float64)),
        "policy_native_recurrent_history": False,
        "captured_history_semantics": "runner current-policy-input buffer; pi0.5 consumes latest frame only",
    }
    e4["pass"] = bool(
        e4["full_restore_next_state_error"] <= 1e-9
        and e4["simulator_ablation_next_state_error"] > 1e-9
        and e4["queue_ablation_action_error"] > 0.0
        and e4["queue_ablation_next_state_error"] > 1e-9
        and e4["policy_input_buffer_ablation_action_error"] > 0.0
        and e4["rng_ablation_action_error"] > 0.0
    )
    atomic_json(output_root / "E4_component_ablations.json", e4)

    runner.restore(history_anchor)
    noise_a = policy_noise(policy, runner.noise_seed, runner.noise_counter + 1_000_001)
    noise_b = policy_noise(policy, runner.noise_seed, runner.noise_counter + 1_000_002)
    action_a = _instrumented_action(policy, runner.observation_history[-1], noise_a)
    action_b = _instrumented_action(policy, runner.observation_history[-1], noise_b)
    e5 = {
        "noise_max_abs_difference": _max_abs(noise_a.astype(np.float64), noise_b.astype(np.float64)),
        "action_max_abs_difference": _max_abs(action_a.astype(np.float64), action_b.astype(np.float64)),
        "noise_a_sha256": hashlib.sha256(noise_a.tobytes()).hexdigest(),
        "noise_b_sha256": hashlib.sha256(noise_b.tobytes()).hexdigest(),
        "action_a_sha256": hashlib.sha256(action_a.tobytes()).hexdigest(),
        "action_b_sha256": hashlib.sha256(action_b.tobytes()).hexdigest(),
    }
    e5["pass"] = bool(e5["noise_max_abs_difference"] > 0.0 and e5["action_max_abs_difference"] > 0.0)
    atomic_json(output_root / "E5_branch_rng.json", e5)
    env.close()

    positive = generate_control_bank("positive")
    positive_control = {
        "rollouts": len(positive),
        "successes": int(sum(trace.success for trace in positive)),
        "observed_modes": sorted(set(int(trace.mode) for trace in positive if trace.success)),
    }
    positive_control["pass"] = len(positive_control["observed_modes"]) >= 2
    atomic_json(output_root / "POSITIVE_CONTROL.json", positive_control)

    if run_e6:
        e6_runtime = Phase0Runtime(
            qpilots_root=qpilots_root,
            libero_root=libero_root,
            checkpoint=checkpoint,
            microbatch=microbatch,
        )
        e6_runtime.policy = policy
        e6_raw = output_root / "e6_raw"
        cached = _validated_task_cache(
            e6_raw,
            suite_name="libero_90",
            task_id=64,
            expected_rollouts=64,
        )
        if cached is None:
            task_metadata = e6_runtime.run_task("libero_90", 64, e6_raw, candidates=4)
            resumed_from_valid_cache = False
        else:
            task_metadata = cached
            resumed_from_valid_cache = True
        rollouts = load_task_rollouts(e6_raw / "libero_90_task64.npz")
        success_rate = float(np.mean([row["success"] for row in rollouts]))
        progress = np.asarray([row["final_progress"] for row in rollouts], dtype=np.float64)
        q25, median, q75 = np.quantile(progress, [0.25, 0.5, 0.75])
        ceiling = bool(median == 1.0 and q25 == 1.0 and q75 == 1.0)
        e6 = {
            "rollouts": len(rollouts),
            "success_rate": success_rate,
            "final_progress_q25": float(q25),
            "final_progress_median": float(median),
            "final_progress_q75": float(q75),
            "progress_ceiling_pile": ceiling,
            "resumed_from_valid_cache": resumed_from_valid_cache,
            "task_data_sha256": task_metadata["data_sha256"],
            "pass": bool(0.25 <= success_rate <= 0.75 and not ceiling),
        }
    else:
        e6 = {"run": False, "pass": False}
    atomic_json(output_root / "E6_baseline.json", e6)

    gates = {"E1": e1, "E2": e2, "E3": e3, "E4": e4, "E5": e5, "E6": e6, "positive_control": positive_control}
    all_pass = bool(all(value.get("pass", False) for value in gates.values()))
    decision = "ENGINEERING_GATES_PASSED" if all_pass else "STAGE_R_ENGINEERING_INCOMPLETE"
    result = {"protocol_id": PROTOCOL_ID, "decision": decision, "gates": gates}
    atomic_json(output_root / "engineering_gate_summary.json", result)
    artifacts = []
    for path in sorted(output_root.glob("*.json")):
        if path.name != "COMPLETED_ENGINEERING_GATES.json":
            artifacts.append({"path": path.name, "sha256": sha256_file(path)})
    atomic_json(
        output_root / "COMPLETED_ENGINEERING_GATES.json",
        {"protocol_id": PROTOCOL_ID, "decision": decision, "artifacts": artifacts},
    )
    return result
