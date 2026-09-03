#!/usr/bin/env python3
"""Run the frozen Stage-S substrate-A RoboTwin screen.

This is the real-runtime entry point.  It only imports the released
RoboTwin/Evo code after explicit roots are supplied, never invokes an expert
trajectory, and never accepts a synthetic callback.  A family is written
only after the exact ``restore -> same action -> next state`` gate passes.

The Evo-1 policy is server-backed.  ``--server-url`` therefore points at an
already-started official Evo deploy server whose model must be the audited
checkpoint.  This command itself does not submit PAI jobs or start a server.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import sys
import tempfile
from datetime import datetime, time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np

from r142_stage_s.frozen_protocol import (
    DEFAULT_PROTOCOL_PATH,
    FrozenProtocolError,
    load_frozen_protocol,
)
from r142_stage_s.asset_acceptance import (
    DEFAULT_ACCEPTED_ASSET_PATH,
    AssetAcceptanceError,
    load_accepted_asset_preflight,
)
from r142_stage_s.robotwin import (
    AtomicFamilyWriter,
    CapabilityError,
    ConcreteRoboTwinRuntime,
    EvoProxyStateAdapter,
    FamilyRolloutRunner,
    RoboTwinPins,
    STAGE_S_PROTOCOL_ID,
    _jsonable,
    select_published_tasks,
)


BEIJING = ZoneInfo("Asia/Shanghai")
BLACKOUT_WINDOWS: Tuple[Tuple[time, time], ...] = (
    (time(9, 30), time(9, 40)),
    (time(19, 30), time(19, 40)),
)


def assert_outside_blackout(now: Optional[datetime] = None) -> datetime:
    """Fail closed during the two frozen daily scheduler blackout windows."""
    supplied = now
    if supplied is not None and supplied.tzinfo is None:
        supplied = supplied.replace(tzinfo=BEIJING)
    current = (supplied or datetime.now(BEIJING)).astimezone(BEIJING)
    current_time = current.time().replace(tzinfo=None)
    for start, end in BLACKOUT_WINDOWS:
        if start <= current_time < end:
            raise CapabilityError(
                "Stage-S RoboTwin run is blocked during the Beijing scheduler "
                f"blackout {start.isoformat(timespec='minutes')}-"
                f"{end.isoformat(timespec='minutes')}"
            )
    return current


def _atomic_json(path: Path, value: Mapping[str, Any]) -> str:
    data = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    return _atomic_bytes(path, data)


def _atomic_bytes(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return hashlib.sha256(data).hexdigest()


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - real runtime only
        raise CapabilityError("RoboTwin runtime requires PyYAML") from exc
    if not path.is_file():
        raise CapabilityError(f"RoboTwin task configuration is missing: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CapabilityError(f"RoboTwin task configuration is not a mapping: {path}")
    return value


def _prepare_official_args(
    robotwin_root: Path,
    task_name: str,
    checkpoint_dir: Path,
    output_dir: Path,
) -> Tuple[Dict[str, Any], Any]:
    """Build the same embodiment/camera arguments as stable_2.0 eval_policy."""
    root_text = str(robotwin_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    eval_policy = importlib.import_module("script.eval_policy")
    from envs import CONFIGS_PATH

    args = _load_yaml(robotwin_root / "task_config" / "demo_clean.yml")
    args.update(
        {
            "task_name": task_name,
            "task_config": "demo_clean",
            "ckpt_setting": "step_20000",
            "eval_mode": True,
            "render_freq": 0,
            # Stage-S records trajectories directly and must not produce an
            # unrelated official video/data collection side effect.
            "eval_video_log": False,
            "collect_data": False,
            "save_path": str(output_dir),
            "checkpoint_dir": str(checkpoint_dir),
        }
    )

    embodiment_type = args.get("embodiment")
    if not isinstance(embodiment_type, (list, tuple)) or len(embodiment_type) not in (1, 3):
        raise CapabilityError("demo_clean embodiment must contain one or three entries")
    embodiment_cfg = _load_yaml(Path(CONFIGS_PATH) / "_embodiment_config.yml")

    def robot_file(name: str) -> Path:
        value = embodiment_cfg.get(name, {}).get("file_path")
        if not value:
            raise CapabilityError(f"embodiment file is missing for {name}")
        path = Path(value)
        return path if path.is_absolute() else robotwin_root / path

    if len(embodiment_type) == 1:
        left_file = right_file = robot_file(str(embodiment_type[0]))
        args["dual_arm_embodied"] = True
    else:
        left_file = robot_file(str(embodiment_type[0]))
        right_file = robot_file(str(embodiment_type[1]))
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    args["left_robot_file"] = str(left_file)
    args["right_robot_file"] = str(right_file)
    args["left_embodiment_config"] = _load_yaml(left_file / "config.yml")
    args["right_embodiment_config"] = _load_yaml(right_file / "config.yml")

    head_type = args.get("camera", {}).get("head_camera_type")
    camera = eval_policy.get_camera_config(head_type)
    args["head_camera_h"] = camera["h"]
    args["head_camera_w"] = camera["w"]
    return args, eval_policy.class_decorator(task_name)


def _read_instruction(robotwin_root: Path, task_name: str) -> str:
    path = robotwin_root / "description" / "task_instruction" / f"{task_name}.json"
    if not path.is_file():
        raise CapabilityError(f"official task instruction is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    instruction = value.get("full_description") if isinstance(value, dict) else None
    if not isinstance(instruction, str) or not instruction.strip():
        raise CapabilityError(f"official task instruction has no full_description: {path}")
    return instruction


class OfficialRoboTwinEpisode:
    """Thin facade over one stable_2.0 task environment.

    Snapshot methods delegate to :class:`ConcreteRoboTwinRuntime`, so the
    facade cannot accidentally turn an observation into a fake simulator
    snapshot.  The task's official ``eval_success`` flag is the only success
    label consumed by the runner (with its official check_success fallback).
    """

    def __init__(
        self,
        *,
        robotwin_root: Path,
        task_name: str,
        checkpoint_dir: Path,
        output_dir: Path,
    ):
        self.robotwin_root = robotwin_root
        self.task_name = task_name
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir
        self._args, self.task_env = _prepare_official_args(
            robotwin_root, task_name, checkpoint_dir, output_dir
        )
        self._instruction = _read_instruction(robotwin_root, task_name)
        self._runtime: Optional[ConcreteRoboTwinRuntime] = None
        self._policies = []

    def bind_policy(self, policy: Any) -> None:
        self._runtime = ConcreteRoboTwinRuntime(self.task_env, policy, require_torch=True)

    def register_policy(self, policy: Any) -> None:
        self._policies.append(policy)

    def reset(self, *, seed: int, task_name: Optional[str] = None) -> None:
        if task_name is not None and str(task_name) != self.task_name:
            raise CapabilityError(f"task mismatch: expected {self.task_name}, got {task_name}")
        # This is the official scene setup path.  No expert trajectory or
        # solvability oracle is called; each seed is retained as a real family.
        self.task_env.setup_demo(
            now_ep_num=0,
            seed=int(seed),
            is_test=True,
            **copy.deepcopy(self._args),
        )
        set_instruction = getattr(self.task_env, "set_instruction", None)
        if not callable(set_instruction):
            raise CapabilityError("stable_2.0 task lacks set_instruction()")
        set_instruction(instruction=self._instruction)

    @property
    def task_env_ref(self) -> Any:
        return self.task_env

    @property
    def scene(self) -> Any:
        return self.task_env.scene

    @property
    def eval_success(self) -> bool:
        return bool(getattr(self.task_env, "eval_success", False))

    @eval_success.setter
    def eval_success(self, value: bool) -> None:
        self.task_env.eval_success = value

    def get_obs(self) -> Any:
        return self.task_env.get_obs()

    def take_action(self, action: Any) -> Any:
        return self.task_env.take_action(action)

    def check_success(self) -> bool:
        check = getattr(self.task_env, "check_success", None)
        return bool(check()) if callable(check) else False

    def get_instruction(self) -> str:
        return self._instruction

    def close(self) -> None:
        for policy in self._policies:
            close_policy = getattr(policy, "close", None)
            if callable(close_policy):
                close_policy()
        close = getattr(self.task_env, "close_env", None)
        if callable(close):
            close()

    def __getattr__(self, name: str) -> Any:
        # The official runner reads counters such as take_action_cnt and
        # step_lim directly from TASK_ENV.  Delegation preserves that API
        # without copying simulator state into a synthetic facade.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.task_env, name)

    # The methods below are the concrete snapshot contract consumed by
    # ExactReplayVerifier.  Runtime binding is mandatory before capture.
    def _require_runtime(self) -> ConcreteRoboTwinRuntime:
        if self._runtime is None:
            raise CapabilityError("RoboTwin policy must be bound before snapshot capture")
        return self._runtime

    def capture_simulator_state(self) -> Any:
        return self._require_runtime().capture_simulator_state()

    def restore_simulator_state(self, value: Any) -> None:
        self._require_runtime().restore_simulator_state(value)

    def capture_rng_state(self) -> Any:
        return self._require_runtime().capture_rng_state()

    def restore_rng_state(self, value: Any) -> None:
        self._require_runtime().restore_rng_state(value)

    def state_for_verification(self) -> Dict[str, Any]:
        """Read only live simulator state for the 1e-9 replay comparison."""
        runtime = self._require_runtime()
        state = runtime.capture_simulator_state()
        actor_state = []
        for item in state["actors"]:
            actor_state.append(
                {
                    "name": item["name"],
                    "pose": _jsonable(item["pose"]),
                    "velocity": _jsonable(item.get("velocity")),
                    "angular_velocity": _jsonable(item.get("angular_velocity")),
                }
            )
        articulation_state = []
        for item in state["articulations"]:
            articulation_state.append(
                {
                    key: _jsonable(item[key])
                    for key in ("root_pose", "qpos", "qvel", "qacc")
                    if key in item
                }
            )
        return {
            "actors": actor_state,
            "articulations": articulation_state,
            "counters": _jsonable(state["counters"]),
            "scene_clock": _jsonable(
                state.get("scene_clock", {}).get("value")
                if isinstance(state.get("scene_clock"), Mapping)
                else None
            ),
        }


class OfficialEvoPolicy:
    """Policy adapter using the released Evo-1 deploy_policy plugin."""

    def __init__(self, proxy: Any, task_env: OfficialRoboTwinEpisode, horizon: int = 37):
        self.stateful = EvoProxyStateAdapter(proxy)
        self.task_env = task_env
        self.horizon = int(horizon)
        self.forward_count = 0

    def _queue_as_actions(self, actions: Any) -> list:
        array = np.asarray(actions)
        if array.ndim != 2 or array.shape[1] < 14 or array.shape[0] < self.horizon:
            raise CapabilityError(
                f"Evo-1 response must have at least ({self.horizon}, 14) actions, got {array.shape}"
            )
        return [np.asarray(row[:14]).copy() for row in array[: self.horizon]]

    def act(self, observation: Mapping[str, Any], **_: Any) -> Any:
        if self.stateful.action_queue:
            action = np.asarray(self.stateful.action_queue.pop(0)).copy()
            return action
        try:
            head = observation["observation"]["head_camera"]["rgb"]
            left = observation["observation"]["left_camera"]["rgb"]
            right = observation["observation"]["right_camera"]["rgb"]
            state = observation["joint_action"]["vector"].tolist()
        except (KeyError, TypeError, AttributeError) as exc:
            raise CapabilityError("stable_2.0 observation lacks Evo-1 camera/state fields") from exc
        actions_raw = self.stateful.infer(
            head, left, right, state, self.task_env.get_instruction()
        )
        module = importlib.import_module("policy.Evo1.deploy_policy")
        smooth = getattr(module, "smooth_actions", None)
        actions = np.asarray(actions_raw)
        if not callable(smooth):
            raise CapabilityError("released Evo-1 smooth_actions() is unavailable")
        actions = smooth(actions, kernel_size=9, smooth_type="gaussian")
        self.stateful.action_queue = self._queue_as_actions(actions)
        self.forward_count += 1
        return np.asarray(self.stateful.action_queue.pop(0)).copy()

    def set_rng(self, rng: np.random.Generator) -> None:
        self.stateful.set_rng(rng)

    def seed(self, seed: int) -> None:
        """Set the exact integer seed on the server-backed Evo policy.

        ``FamilyRolloutRunner`` calls this before its legacy Generator
        fallback, so the persisted SeedSequence value is the seed applied to
        Python/NumPy/Torch/CUDA in the server control shim.
        """

        self.stateful.set_seed(int(seed))

    def capture_observation_history(self) -> Any:
        return self.stateful.capture_observation_history()

    def restore_observation_history(self, value: Any) -> None:
        self.stateful.restore_observation_history(value)

    def capture_action_queue(self) -> Any:
        return self.stateful.capture_action_queue()

    def restore_action_queue(self, value: Any) -> None:
        self.stateful.restore_action_queue(value)

    def capture_rng_state(self) -> Any:
        return self.stateful.capture_rng_state()

    def restore_rng_state(self, value: Any) -> None:
        self.stateful.restore_rng_state(value)

    def close(self) -> None:
        close = getattr(self.stateful.proxy, "close", None)
        if callable(close):
            close()


def _make_policy_factory(
    *, evo_root: Path, task_name: str, server_url: str, checkpoint_revision: str, episode: OfficialRoboTwinEpisode
):
    policy_file = evo_root / "RoboTwin_evaluation" / "policy" / "Evo1" / "deploy_policy.py"
    if not policy_file.is_file():
        raise CapabilityError(f"released Evo-1 deploy_policy.py is missing: {policy_file}")
    plugin_root = str(evo_root / "RoboTwin_evaluation")
    if plugin_root not in sys.path:
        sys.path.insert(0, plugin_root)
    module = importlib.import_module("policy.Evo1.deploy_policy")

    def factory(seed: int) -> OfficialEvoPolicy:
        # get_model is the released plugin constructor; the checkpoint pin is
        # carried in the immutable run manifest and must match the audited
        # server deployment.  No local/synthetic policy is accepted.
        proxy = module.get_model(
            {
                "server_url": server_url,
                "horizon": 37,
                "task_name": task_name,
                "task_config": "demo_clean",
                "dataset_key_suffix": "",
                "checkpoint_revision": checkpoint_revision,
            }
        )
        policy = OfficialEvoPolicy(proxy, episode, horizon=37)
        episode.register_policy(policy)
        return policy

    return factory


def _write_rank_completion(
    output_root: Path,
    rank: int,
    world_size: int,
    manifests: Sequence[Mapping[str, Any]],
    frozen_protocol: Mapping[str, Any],
    accepted_asset_preflight: Mapping[str, Any],
) -> Dict[str, Any]:
    payload = {
        "status": "COMPLETED",
        "rank": int(rank),
        "world_size": int(world_size),
        "family_count": len(manifests),
        "families": list(manifests),
        "frozen_protocol": dict(frozen_protocol),
        "accepted_asset_preflight": dict(accepted_asset_preflight),
        "accepted_asset_run_id": accepted_asset_preflight["accepted_run_id"],
        "accepted_asset_job_id": accepted_asset_preflight["accepted_job_id"],
        "accepted_asset_completion_sha256": accepted_asset_preflight["completion_sha256"],
        "accepted_asset_sha256sums_sha256": accepted_asset_preflight["asset_sha256sums_sha256"],
        "accepted_model_sha256sums_sha256": accepted_asset_preflight["model_sha256sums_sha256"],
        "accepted_source_commits": dict(accepted_asset_preflight["source_commits"]),
        "protocol_id": STAGE_S_PROTOCOL_ID,
        "protocol_authority_path": frozen_protocol["path"],
        "protocol_authority_sha256": frozen_protocol["protocol_json_sha256"],
        "protocol_git_commit": frozen_protocol["protocol_git_commit"],
        "protocol_json_sha256": frozen_protocol["protocol_json_sha256"],
        "protocol_md_sha256": frozen_protocol["protocol_md_sha256"],
        "calibration_report_sha256": {
            name: item["sha256"]
            for name, item in frozen_protocol["calibration_reports"].items()
        },
    }
    path = output_root / f"COMPLETED_A_RANK-{rank:04d}.json"
    digest = _atomic_json(path, payload)
    sums = output_root / f"SHA256SUMS_A_RANK-{rank:04d}"
    sums_data = f"{digest}  {path.name}\n".encode()
    _atomic_bytes(sums, sums_data)
    return {**payload, "completion_file": str(path), "completion_sha256": digest}


def run(args: argparse.Namespace) -> Dict[str, Any]:
    pins = RoboTwinPins()
    assert_outside_blackout()
    if args.phase != "main":
        raise CapabilityError("Stage-S substrate-A has no Step-0 calibration; use --phase main")
    if args.rank < 0 or args.world_size <= 0 or args.rank >= args.world_size:
        raise CapabilityError("rank/world-size must satisfy 0 <= rank < world-size")
    if args.families_per_task != 16 or args.candidates != 32:
        raise CapabilityError("Stage-S A is frozen at 16 families x 32 candidates per task")
    try:
        frozen_protocol = load_frozen_protocol(args.frozen_protocol)
    except FrozenProtocolError as exc:
        raise CapabilityError(f"frozen protocol gate: {exc}") from exc
    robotwin_root = args.robotwin_root.resolve()
    evo_root = args.evo_root.resolve()
    checkpoint_dir = args.checkpoint_dir.resolve()
    output_root = args.output_root.resolve()
    if pins.checkpoint_revision not in checkpoint_dir.name and not (
        checkpoint_dir / "revision.txt"
    ).is_file():
        raise CapabilityError(
            "checkpoint path is not bound to exact HF revision "
            f"{pins.checkpoint_revision}; use the pinned snapshot directory or revision.txt"
        )
    try:
        accepted_asset_preflight = load_accepted_asset_preflight(
            args.accepted_asset_preflight,
            checkpoint_dir=checkpoint_dir,
        )
    except AssetAcceptanceError as exc:
        raise CapabilityError(f"accepted asset preflight gate: {exc}") from exc

    selected = select_published_tasks()
    # Audit is a precondition, not an optional informational report.
    from scripts.stage_s_robotwin_audit import audit

    audit_result = audit(
        robotwin_root=robotwin_root,
        evo_root=evo_root,
        checkpoint_dir=checkpoint_dir,
        runtime_wrapper=Path(__file__).resolve().parents[1] / "src" / "r142_stage_s" / "robotwin.py",
        server_runtime_wrapper=Path(__file__).resolve().parents[0]
        / "stage_s_robotwin_evo_server.py",
        pins=pins,
    )
    if not audit_result["status"].startswith("READY"):
        raise CapabilityError(f"asset audit blocked: {audit_result['capability_error']}")
    output_root.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "protocol": "R142-FP-11 Stage-S substrate A",
        "pins": pins.as_dict(),
        "selected_tasks": list(selected),
        "families_per_task": args.families_per_task,
        "candidate_budget": args.candidates,
        "rank": args.rank,
        "world_size": args.world_size,
        "seed_base": args.seed_base,
        "server_url": args.server_url,
        "synthetic_rollouts": False,
        "expert_trajectory": False,
        "termination": "official eval_success or step_lim",
        "frozen_protocol": frozen_protocol,
        "accepted_asset_preflight": accepted_asset_preflight,
        "accepted_asset_run_id": accepted_asset_preflight["accepted_run_id"],
        "accepted_asset_job_id": accepted_asset_preflight["accepted_job_id"],
        "accepted_asset_completion_sha256": accepted_asset_preflight["completion_sha256"],
        "accepted_asset_sha256sums_sha256": accepted_asset_preflight["asset_sha256sums_sha256"],
        "accepted_model_sha256sums_sha256": accepted_asset_preflight["model_sha256sums_sha256"],
        "accepted_source_commits": dict(accepted_asset_preflight["source_commits"]),
        "protocol_id": STAGE_S_PROTOCOL_ID,
        "protocol_authority_path": frozen_protocol["path"],
        "protocol_authority_sha256": frozen_protocol["protocol_json_sha256"],
        "protocol_git_commit": frozen_protocol["protocol_git_commit"],
        "protocol_json_sha256": frozen_protocol["protocol_json_sha256"],
        "protocol_md_sha256": frozen_protocol["protocol_md_sha256"],
        "calibration_report_sha256": {
            name: item["sha256"]
            for name, item in frozen_protocol["calibration_reports"].items()
        },
        "blackout_windows": [
            "09:30-09:40 Asia/Shanghai",
            "19:30-19:40 Asia/Shanghai",
        ],
        "asset_audit": audit_result,
    }
    _atomic_json(output_root / f"RUN_MANIFEST_RANK-{args.rank:04d}.json", run_manifest)

    manifests = []
    for task_index, task_name in enumerate(selected):
        for family_index in range(args.families_per_task):
            flat_index = task_index * args.families_per_task + family_index
            if flat_index % args.world_size != args.rank:
                continue
            local_family_id = f"family-{family_index:04d}"
            family_id = f"{task_name}/{local_family_id}"
            family_output = output_root / f"rank-{args.rank:04d}"
            writer = AtomicFamilyWriter(family_output / task_name)
            existing = writer.completed(local_family_id)
            if existing is not None:
                manifests.append(existing)
                continue
            episode = OfficialRoboTwinEpisode(
                robotwin_root=robotwin_root,
                task_name=task_name,
                checkpoint_dir=checkpoint_dir,
                output_dir=family_output,
            )
            try:
                policy_factory = _make_policy_factory(
                    evo_root=evo_root,
                    task_name=task_name,
                    server_url=args.server_url,
                    checkpoint_revision=pins.checkpoint_revision,
                    episode=episode,
                )
                runner = FamilyRolloutRunner(
                    env_factory=lambda episode=episode: episode,
                    policy_factory=policy_factory,
                    writer=writer,
                )
                manifests.append(
                    runner.run_family(
                        task_name=task_name,
                        family_id=family_id,
                        local_family_id=local_family_id,
                        initial_state_id=f"{task_name}/state-{family_index:04d}",
                        initial_seed=args.seed_base + task_index * 1_000_000 + family_index,
                        candidate_count=args.candidates,
                        metadata={
                            "protocol_id": STAGE_S_PROTOCOL_ID,
                            "protocol_authority_path": frozen_protocol["path"],
                            "protocol_authority_sha256": frozen_protocol["protocol_json_sha256"],
                            "protocol_git_commit": frozen_protocol["protocol_git_commit"],
                            "substrate": "A",
                            "pose_dimension": 14,
                        },
                    )
                )
            finally:
                episode.close()
    return _write_rank_completion(
        output_root,
        args.rank,
        args.world_size,
        manifests,
        frozen_protocol,
        accepted_asset_preflight,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("main", "calibration"), default="main")
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--evo-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--families-per-task", type=int, default=16)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--seed-base", type=int, default=14211)
    parser.add_argument(
        "--frozen-protocol",
        type=Path,
        default=DEFAULT_PROTOCOL_PATH,
        help="stable CPFS Stage-S protocol authority",
    )
    parser.add_argument(
        "--accepted-asset-preflight",
        type=Path,
        default=DEFAULT_ACCEPTED_ASSET_PATH,
        help="stable CPFS accepted asset-preflight authority",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = run(args)
    except CapabilityError as exc:
        print(json.dumps({"status": "BLOCKED_CAPABILITY", "error": str(exc)}, ensure_ascii=False))
        return 2
    except Exception as exc:  # pragma: no cover - concrete dependency/runtime only
        print(
            json.dumps(
                {
                    "status": "BLOCKED_RUNTIME",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
            )
        )
        return 3
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
