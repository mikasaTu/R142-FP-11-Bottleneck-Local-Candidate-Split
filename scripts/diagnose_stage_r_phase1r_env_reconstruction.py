from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


TARGET_HISTORY_STRATEGIES = frozenset(("prior-reset-only", "candidate-prior-lifecycle"))


def action_sha256(actions: np.ndarray) -> str:
    array = np.asarray(actions, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(repr(tuple(int(value) for value in array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def target_environment_index(strategy: str, candidate_id: int) -> int:
    if strategy == "single" or strategy in TARGET_HISTORY_STRATEGIES:
        return 0
    return int(candidate_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--qpilots", required=True)
    parser.add_argument("--libero", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument(
        "--strategy",
        choices=(
            "single",
            "indexed-reset-target",
            "indexed-reset-all",
            "prior-reset-only",
            "candidate-prior-lifecycle",
        ),
        required=True,
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate-id", type=int, default=27)
    parser.add_argument("--init-state", type=int, default=27)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(args.repo) / "src"))
    sys.path.insert(0, args.qpilots)
    sys.path.insert(0, args.libero)
    os.environ["QPILOTS_LIBERO_SITE"] = str(Path(args.libero).resolve())

    from libero.libero import benchmark
    from qpilots_libero.environment import Task64Environment
    from r142_stage_r.phase0 import load_task_rollouts
    from r142_stage_r.protocol import ranked_initial_states
    from r142_stage_r.runtime import pose_vector, shared_environment_seed, stable_pose_keys, task_config

    suite_name = "libero_10"
    task_id = 9
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    config = task_config(suite_name, task_id, str(task.language), 520)
    rows = load_task_rollouts(args.raw)
    matching = [
        row
        for row in rows
        if int(row["candidate_id"]) == args.candidate_id and int(row["init_state"]) == args.init_state
    ]
    if len(matching) != 1:
        raise RuntimeError(f"expected one raw parent, got {len(matching)}")
    row = matching[0]
    target_history_strategy = args.strategy in TARGET_HISTORY_STRATEGIES
    count = 1 if args.strategy == "single" or target_history_strategy else 32
    environments = [Task64Environment(config, seed=0) for _ in range(count)]
    target = environments[target_environment_index(args.strategy, args.candidate_id)]
    common_seed = shared_environment_seed(suite_name, task_id, args.init_state)
    ordered_states = ranked_initial_states(suite_name, task_id)
    if args.init_state not in ordered_states:
        raise RuntimeError(f"target init state {args.init_state} is not in the frozen Phase-0 order")
    prior_states = ordered_states[: ordered_states.index(args.init_state)]
    prior_replay = []
    try:
        if args.strategy in ("prior-reset-only", "candidate-prior-lifecycle"):
            for prior_state in prior_states:
                prior_seed = shared_environment_seed(suite_name, task_id, prior_state)
                for environment in environments:
                    environment.environment.seed(int(prior_seed))
                    environment.evaluation_seed = int(prior_seed)
                    environment.reset(prior_state)
                if args.strategy == "candidate-prior-lifecycle":
                    prior_matching = [
                        candidate
                        for candidate in rows
                        if int(candidate["candidate_id"]) == args.candidate_id
                        and int(candidate["init_state"]) == prior_state
                    ]
                    if len(prior_matching) != 1:
                        raise RuntimeError(
                            f"expected one prior raw parent for init {prior_state}, got {len(prior_matching)}"
                        )
                    prior_row = prior_matching[0]
                    replay_length = 0
                    replay_success = False
                    done_step = None
                    for prior_index, prior_action in enumerate(
                        np.asarray(prior_row["actions"], dtype=np.float32)
                    ):
                        prior_trace = target.execute_actions(prior_action[None])
                        replay_length += int(prior_trace["executed_steps"])
                        replay_success = bool(prior_trace["success"])
                        if bool(prior_trace["done"]):
                            done_step = prior_index + 1
                            break
                    prior_replay.append(
                        {
                            "init_state": int(prior_state),
                            "source_length": int(len(prior_row["actions"])),
                            "replay_length": int(replay_length),
                            "source_success": bool(prior_row["success"]),
                            "replay_success": bool(replay_success),
                            "done_step": done_step,
                            "exact_terminal_match": bool(
                                replay_length == len(prior_row["actions"])
                                and replay_success == bool(prior_row["success"])
                            ),
                        }
                    )

        reset_set = (
            environments
            if args.strategy in (
                "indexed-reset-all",
                "prior-reset-only",
                "candidate-prior-lifecycle",
            )
            else [target]
        )
        for environment in reset_set:
            environment.environment.seed(int(common_seed))
            environment.evaluation_seed = int(common_seed)
            environment.reset(args.init_state)

        pose_keys = stable_pose_keys(target._observation)
        replay_poses = []
        progress = []
        done_step = None
        terminal_success = False
        pose_errors = []
        for index, action in enumerate(np.asarray(row["actions"], dtype=np.float32)):
            trace = target.execute_actions(np.asarray(action, dtype=np.float32)[None])
            eef, objects = pose_vector(target._observation, pose_keys)
            pose = np.concatenate([eef, objects])
            replay_poses.append(pose)
            predicates = target.official_predicates()
            progress.append(float(predicates["fraction"]))
            source_pose = np.asarray(row["poses"][index], dtype=np.float64)
            pose_errors.append(float(np.max(np.abs(pose - source_pose))))
            terminal_success = bool(trace["success"])
            if bool(trace["done"]):
                done_step = index + 1
                break

        output = {
            "strategy": args.strategy,
            "candidate_id": args.candidate_id,
            "init_state": args.init_state,
            "common_seed": int(common_seed),
            "environment_count": count,
            "prior_environment_scope": "target_only" if target_history_strategy else None,
            "reset_count": len(reset_set) * (len(prior_states) + 1)
            if args.strategy in ("prior-reset-only", "candidate-prior-lifecycle")
            else len(reset_set),
            "prior_init_states": [int(value) for value in prior_states],
            "prior_replay": prior_replay,
            "prior_replay_all_exact": bool(prior_replay)
            and all(bool(value["exact_terminal_match"]) for value in prior_replay),
            "raw_length": int(len(row["actions"])),
            "replay_length": int(len(replay_poses)),
            "done_step": done_step,
            "raw_success": bool(row["success"]),
            "replay_success": terminal_success,
            "raw_final_progress": float(row["final_progress"]),
            "replay_final_progress": float(progress[-1]) if progress else None,
            "action_sha256": action_sha256(np.asarray(row["actions"], dtype=np.float32)),
            "pose_keys": list(pose_keys),
            "pose_error_step0": pose_errors[0] if pose_errors else None,
            "pose_error_max": max(pose_errors) if pose_errors else None,
            "first_pose_error_gt_1e_12": next((i for i, value in enumerate(pose_errors) if value > 1e-12), None),
            "first_pose_error_gt_1e_6": next((i for i, value in enumerate(pose_errors) if value > 1e-6), None),
        }
        output_path = Path(args.output)
        temporary = output_path.with_name(f".{output_path.name}.tmp")
        temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(output_path)
        print(json.dumps(output, sort_keys=True))
    finally:
        for environment in environments:
            environment.close()


if __name__ == "__main__":
    main()
