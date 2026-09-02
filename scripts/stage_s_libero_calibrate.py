#!/usr/bin/env python3
"""Run the real Stage-S B/C pooled-success calibration.

This command is intentionally a thin adapter around the frozen Stage-R
``CleanPi05LiberoPolicy`` and ``Task64Environment`` wrappers.  It has two
subcommands-in-one modes:

``prepare``
    Validate the frozen sources and materialize the immutable calibration
    plan exactly once before distributed workers are launched.  This avoids
    concurrent CPFS writers racing on ``CALIBRATION_PLAN.json``.

``shard``
    Run this rank's deterministic subset of the frozen
    ``4 settings x 4 tasks x 8 initial states x 8 candidates`` grid.  Only
    aggregate counters are persisted; no evaluator callback or synthetic
    fallback is accepted.

``aggregate``
    Verify every rank's completion marker and SHA, sum the aggregate-only
    rows, and write the final completion marker.

The command never submits PAI.  A PAI payload/launcher is a separate,
explicitly non-submitting artifact in this repository.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable

from r142_stage_s.libero import (
    C_RETAIN_STEPS,
    CALIBRATION_CANDIDATE_COUNT,
    CALIBRATION_INITIAL_STATES,
    CALIBRATION_PLAN_SCHEMA,
    CALIBRATION_SEED,
    CALIBRATION_TASK_IDS,
    PROXIMITY_MAGNITUDES,
    StageRPolicyAdapter,
    atomic_json,
    aggregate_calibration_shards,
    audit_undertrained_checkpoint_set,
    make_stage_r_task64_factory,
    run_calibration_shard,
    run_stage_s_calibration_episode,
    task_spec,
    validate_b_calibration_variants,
    verify_calibration_aggregate,
    write_calibration_plan,
)


def _settings(substrate: str) -> list[str]:
    if substrate == "B":
        return [f"proximity_{value:.2f}m" for value in PROXIMITY_MAGNITUDES]
    return [f"step_{value}" for value in C_RETAIN_STEPS]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--substrate", choices=("B", "C"), required=True)
    parser.add_argument("--mode", choices=("prepare", "shard", "aggregate"), default="shard")
    parser.add_argument("--output-root", type=Path, required=True, help="persistent calibration run root")
    parser.add_argument("--report", type=Path, help="aggregate result path (aggregate mode only)")
    parser.add_argument("--seed", type=int, default=CALIBRATION_SEED)
    parser.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", "1")))
    parser.add_argument("--rank", type=int, default=int(os.environ.get("RANK", "0")))
    parser.add_argument("--max-steps", type=int, default=520)
    parser.add_argument("--dry-run", action="store_true", help="print the frozen contract without writing or running episodes")

    # B: four independently generated variant roots and the unchanged
    # Stage-R policy checkpoint.  The source init root is used only for the
    # fail-closed byte/hash/qpos audit; it is never loaded by the episodes.
    parser.add_argument("--variant-root", type=Path, action="append", default=[], help="B variant root; repeat exactly four times")
    parser.add_argument("--source-init-root", type=Path, help="original B init root used only by the regeneration audit")

    # C: four exact checkpoints at the frozen steps.  No interpolation or
    # artificial degradation is accepted.
    parser.add_argument("--checkpoint", type=Path, action="append", default=[], help="C checkpoint; repeat exactly four times")
    parser.add_argument("--policy-checkpoint", type=Path, help="unchanged Stage-R policy checkpoint for B")
    parser.add_argument("--qpilots-root", type=Path, help="pinned QPILOTS Stage-R source root")
    parser.add_argument("--libero-root", type=Path, help="pinned LIBERO site/project root")
    parser.add_argument(
        "--libero-config-root",
        type=Path,
        help="run-scoped LIBERO config directory containing config.yaml (required for C)",
    )
    return parser


def _require_runtime_path(value: Path | None, name: str) -> Path:
    if value is None:
        raise SystemExit(f"{name} is required for real shard execution")
    if not value.exists():
        raise SystemExit(f"{name} does not exist: {value}")
    return value


def prepare_b_runtime_configs(
    output_root: Path,
    variant_roots: list[str],
    base_config_root: Path,
    *,
    create: bool,
) -> list[str]:
    """Materialize executable per-setting configs without mutating r7.

    The accepted r7 bundle has complete, hashed BDDL and simulator-state
    evidence, but its per-setting YAML left ``assets`` empty.  LIBERO resolves
    that value during environment construction and fails with ``Path(None)``.
    These run-scoped overlays point BDDL/init paths at the immutable r7 bytes
    and inherit only datasets/assets from the already audited pinned base
    config.  They contain no observations or outcomes.
    """

    base_path = Path(base_config_root) / "config.yaml"
    try:
        base = json.loads(base_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid pinned LIBERO base config {base_path}: {exc}") from exc
    for key in ("assets", "datasets"):
        value = base.get(key)
        if not isinstance(value, str) or not value:
            raise SystemExit(f"pinned LIBERO base config lacks {key}")
    if not Path(base["assets"]).is_dir():
        raise SystemExit(f"pinned LIBERO assets directory is missing: {base['assets']}")

    overlays: list[str] = []
    root = Path(output_root) / "runtime-variant-configs"
    for source_text in variant_roots:
        source = Path(source_text).resolve()
        payload = {
            "benchmark_root": str(source),
            "bddl_files": str((source / "bddl_files").resolve()),
            "init_states": str((source / "init_states").resolve()),
            "datasets": base["datasets"],
            "assets": base["assets"],
        }
        for key in ("benchmark_root", "bddl_files", "init_states", "assets"):
            if not Path(payload[key]).exists():
                raise SystemExit(f"B runtime config path is missing for {key}: {payload[key]}")
        overlay = root / source.name
        target = overlay / "config.yaml"
        if create:
            overlay.mkdir(parents=True, exist_ok=True)
            if target.exists():
                observed = json.loads(target.read_text(encoding="utf-8"))
                if observed != payload:
                    raise SystemExit(f"immutable B runtime config drifted: {target}")
            else:
                atomic_json(target, payload)
        if not target.is_file():
            raise SystemExit(f"prepared B runtime config is missing: {target}")
        observed = json.loads(target.read_text(encoding="utf-8"))
        if observed != payload:
            raise SystemExit(f"prepared B runtime config mismatch: {target}")
        overlays.append(str(overlay.resolve()))
    return overlays


def _print_dry_run(args: argparse.Namespace, settings: list[str]) -> int:
    sources: list[str] | None = None
    if args.substrate == "B" and args.variant_root:
        sources = [str(value.resolve()) for value in args.variant_root]
    elif args.substrate == "C" and args.checkpoint:
        sources = [str(value.resolve()) for value in args.checkpoint]
    payload: dict[str, Any] = {
        "schema": CALIBRATION_PLAN_SCHEMA,
        "protocol_id": "r142-stage-s-v1",
        "substrate": args.substrate,
        "mode": args.mode,
        "calibration_seed": int(args.seed),
        "world_size": int(args.world_size),
        "rank": int(args.rank),
        "settings": settings,
        "task_ids": list(CALIBRATION_TASK_IDS),
        "initial_states": list(CALIBRATION_INITIAL_STATES),
        "candidate_count": CALIBRATION_CANDIDATE_COUNT,
        "trials_per_setting": len(CALIBRATION_TASK_IDS) * len(CALIBRATION_INITIAL_STATES) * CALIBRATION_CANDIDATE_COUNT,
        "persisted_row_fields": ["pooled_success", "setting", "successes", "total"],
        "forbidden_persisted_fields": [
            "actions",
            "family",
            "genealogy",
            "poses",
            "trajectory",
            "S2",
            "S3",
            "S4",
            "S5",
        ],
    }
    if sources is not None:
        payload["setting_sources"] = sources
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _prepare_sources(args: argparse.Namespace, settings: list[str]) -> list[str]:
    if args.substrate == "B":
        if len(args.variant_root) != 4:
            raise SystemExit("B calibration requires exactly four --variant-root paths")
        source_init_root = _require_runtime_path(args.source_init_root, "--source-init-root")
        try:
            validate_b_calibration_variants(args.variant_root, source_init_root, expected_settings=settings)
        except Exception as exc:  # noqa: BLE001 - expose the fail-closed audit cause
            raise SystemExit(f"B variant audit failed closed: {exc}") from exc
        return [str(value.resolve()) for value in args.variant_root]
    if len(args.checkpoint) != 4:
        raise SystemExit("C calibration requires exactly four --checkpoint paths; interpolation is forbidden")
    audit = audit_undertrained_checkpoint_set(
        args.checkpoint,
        expected_steps=C_RETAIN_STEPS,
        hash_model=True,
        require_training_state=True,
    )
    if not audit["valid"]:
        raise SystemExit("C checkpoint audit failed closed: " + "; ".join(audit["errors"]))
    return [str(value.resolve()) for value in args.checkpoint]


def _make_real_evaluator(args: argparse.Namespace, settings: list[str], sources: list[str]) -> Callable[..., bool]:
    qpilots_root = _require_runtime_path(args.qpilots_root, "--qpilots-root")
    libero_root = _require_runtime_path(args.libero_root, "--libero-root")
    libero_config_root = _require_runtime_path(args.libero_config_root, "--libero-config-root")
    if args.substrate == "B" and args.policy_checkpoint is None:
        raise SystemExit("--policy-checkpoint is required for B real Stage-R inference")
    if args.substrate == "B" and not args.policy_checkpoint.exists():
        raise SystemExit(f"--policy-checkpoint does not exist: {args.policy_checkpoint}")
    if args.substrate == "B":
        policy_sources = [str(args.policy_checkpoint)] * len(settings)
        variant_roots: list[Path | None] = [None] * len(settings)
        config_roots = [Path(value) for value in sources]
        state_count = 16
    else:
        policy_sources = list(sources)
        variant_roots = [None] * len(settings)
        config_roots = [libero_config_root] * len(settings)
        # The unmodified Stage-R LIBERO suite owns its 50 init states; the
        # calibration protocol consumes only the frozen first eight indices.
        state_count = 50
    factories = [
        make_stage_r_task64_factory(
            qpilots_root,
            libero_root,
            checkpoint=policy_sources[index],
            variant_root=variant_roots[index],
            libero_config_root=config_roots[index],
            max_steps=int(args.max_steps),
            init_state_count=state_count,
        )
        for index in range(len(settings))
    ]
    policy_cache: dict[tuple[int, int], StageRPolicyAdapter] = {}

    def evaluator(
        setting_index: int,
        setting: Any,
        task_id: int,
        init_state: int,
        candidate_id: int,
        trial_seed: int,
    ) -> bool:
        del setting, trial_seed
        key = (int(setting_index), int(task_id))
        policy = policy_cache.get(key)
        if policy is None:
            policy = StageRPolicyAdapter(
                policy_sources[int(setting_index)],
                qpilots_root=qpilots_root,
                default_prompt=task_spec(int(task_id)).prompt,
                config_name="pi05_libero",
            )
            policy_cache[key] = policy
        return run_stage_s_calibration_episode(
            factories[int(setting_index)],
            policy,
            setting_index=int(setting_index),
            task_id=int(task_id),
            init_state=int(init_state),
            candidate_id=int(candidate_id),
            calibration_seed=int(args.seed),
            max_steps=int(args.max_steps),
        )

    return evaluator


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = _settings(args.substrate)
    if args.seed != CALIBRATION_SEED:
        raise SystemExit(f"calibration seed is frozen to {CALIBRATION_SEED}; got {args.seed}")
    if args.max_steps <= 0:
        raise SystemExit("--max-steps must be positive")
    if args.mode == "aggregate":
        if args.dry_run:
            return _print_dry_run(args, settings)
        try:
            payload = aggregate_calibration_shards(
                args.output_root,
                settings,
                calibration_seed=args.seed,
                world_size=args.world_size,
                report_path=args.report,
            )
            report = args.report or (args.output_root / "CALIBRATION_RESULT.json")
            verified = verify_calibration_aggregate(
                report,
                settings,
                calibration_seed=args.seed,
                world_size=args.world_size,
            )
        except Exception as exc:  # noqa: BLE001 - aggregation is fail-closed
            raise SystemExit(f"calibration aggregate failed closed: {exc}") from exc
        print(json.dumps(verified, indent=2, sort_keys=True))
        return 0

    if args.dry_run:
        return _print_dry_run(args, settings)
    sources = _prepare_sources(args, settings)
    runtime_sources = sources
    if args.substrate == "B":
        base_config_root = _require_runtime_path(args.libero_config_root, "--libero-config-root")
        runtime_sources = prepare_b_runtime_configs(
            args.output_root,
            sources,
            base_config_root,
            create=args.mode == "prepare",
        )
    if args.mode == "prepare":
        payload = write_calibration_plan(
            args.output_root,
            settings,
            calibration_seed=args.seed,
            world_size=args.world_size,
            substrate=args.substrate,
            sources=sources,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    evaluator = _make_real_evaluator(args, settings, runtime_sources)
    try:
        result = run_calibration_shard(
            evaluator,
            settings,
            args.output_root,
            calibration_seed=args.seed,
            world_size=args.world_size,
            rank=args.rank,
            substrate=args.substrate,
            sources=sources,
        )
    except Exception as exc:  # noqa: BLE001 - no synthetic fallback on runtime failure
        raise SystemExit(f"real Stage-R calibration shard failed closed: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
