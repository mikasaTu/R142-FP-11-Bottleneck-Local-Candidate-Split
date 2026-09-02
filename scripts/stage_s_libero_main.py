#!/usr/bin/env python3
"""Run the Stage-S no-intervention main screen with injected real adapters.

No fake policy or synthetic environment is provided.  For a CPU smoke or a
PAI run, supply callbacks implementing the real policy/environment contract.
Each family is written atomically and can be resumed by rerunning this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from r142_stage_s.libero import (
    LIBERO_SUITE,
    MAIN_CANDIDATE_COUNT,
    MAIN_INITIAL_STATE_COUNT,
    STAGE_S_PROTOCOL_ID,
    StageRPolicyAdapter,
    atomic_json,
    import_callback,
    make_stage_r_task64_factory,
    run_main_screen,
    task_spec,
    validate_regenerated_initial_states,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--substrate", choices=("A", "B", "C"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment-factory", help="module:function real environment factory")
    parser.add_argument("--policy", help="module:function real policy factory")
    parser.add_argument("--qpilots-root", type=Path, help="pinned Stage-R QPILOTS root for the built-in real adapter")
    parser.add_argument("--libero-root", type=Path, help="pinned LIBERO site/project root for the built-in real adapter")
    parser.add_argument("--checkpoint", type=Path, help="exact policy checkpoint for C (already audited)")
    parser.add_argument("--variant-root", type=Path, help="B variant LIBERO_CONFIG_PATH root")
    parser.add_argument("--source-init-root", type=Path, help="original LIBERO init root used to validate B qpos identity")
    parser.add_argument("--task-id", type=int, action="append", help="subset only for a smoke; omit for all ten tasks")
    parser.add_argument("--initial-state", type=int, action="append", help="subset only for a smoke; omit for all sixteen states")
    parser.add_argument("--candidate-count", type=int, default=MAIN_CANDIDATE_COUNT)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--validate-snapshots", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print protocol/accounting contract without loading runtime")
    return parser


def _make_policy(callback, args):
    kwargs = {"checkpoint": str(args.checkpoint) if args.checkpoint else None, "substrate": args.substrate}
    try:
        return callback(**kwargs)
    except TypeError:
        try:
            return callback(str(args.checkpoint)) if args.checkpoint else callback()
        except TypeError:
            return callback()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    tasks = tuple(args.task_id) if args.task_id else tuple(range(10))
    states = tuple(args.initial_state) if args.initial_state else tuple(range(MAIN_INITIAL_STATE_COUNT))
    contract = {
        "protocol_id": STAGE_S_PROTOCOL_ID,
        "substrate": args.substrate,
        "suite": LIBERO_SUITE,
        "task_ids": list(tasks),
        "initial_states": list(states),
        "candidate_count": int(args.candidate_count),
        "no_intervention": True,
        "primary_compute_unit": "policy_forward_pass",
        "secondary_compute_unit": "environment_step",
        "atomic_resume": True,
        "variant_root": str(args.variant_root.resolve()) if args.variant_root else None,
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
    }
    if args.dry_run:
        print(json.dumps(contract, indent=2, sort_keys=True))
        return 0
    if args.environment_factory is None:
        if args.qpilots_root is None or args.libero_root is None:
            raise SystemExit("supply --environment-factory or both --qpilots-root and --libero-root")
        factory = make_stage_r_task64_factory(
            args.qpilots_root,
            args.libero_root,
            checkpoint=args.checkpoint,
            variant_root=args.variant_root,
            max_steps=args.max_steps,
        )
    else:
        factory = import_callback(args.environment_factory)
    if args.policy is None:
        if args.qpilots_root is None or args.checkpoint is None:
            raise SystemExit("supply --policy or both --qpilots-root and --checkpoint")
        policy = StageRPolicyAdapter(
            args.checkpoint,
            qpilots_root=args.qpilots_root,
            default_prompt=task_spec(0).prompt,
        )
    else:
        policy = _make_policy(import_callback(args.policy), args)
    if args.substrate == "B" and args.variant_root is None:
        raise SystemExit("B main screen requires a generated variant root with regenerated init qpos")
    if args.substrate == "B":
        if args.source_init_root is None:
            raise SystemExit("B main screen requires --source-init-root for old-qpos identity validation")
        audit = validate_regenerated_initial_states(args.variant_root, args.source_init_root)
        if not audit["valid"]:
            raise SystemExit("B regenerated qpos audit failed closed: " + "; ".join(audit["errors"]))
    if args.substrate == "C" and args.checkpoint is None:
        raise SystemExit("C main screen requires one exact audited checkpoint")
    variant = SimpleNamespace(substrate=args.substrate, root=str(args.variant_root) if args.variant_root else None)
    result = run_main_screen(
        factory,
        policy,
        args.output,
        substrate=args.substrate,
        variant=variant,
        task_ids=tasks,
        initial_states=states,
        candidate_count=args.candidate_count,
        max_steps=args.max_steps,
        validate_snapshots=args.validate_snapshots,
    )
    result.update(contract)
    atomic_json(args.output / f"{args.substrate}_MAIN_SUMMARY.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
