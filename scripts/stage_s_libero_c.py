#!/usr/bin/env python3
"""Audit C's four exact under-trained pi05-LIBERO checkpoints.

When the four real checkpoints are not present this command emits a failed
audit plus a concrete OpenPI launcher contract and exits non-zero.  It never
interpolates, copies, or numerically degrades a checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from r142_stage_s.libero import (
    C_RETAIN_STEPS,
    C_SAVE_INTERVAL,
    C_TRAINING_SEED,
    C_TRAINING_STEPS,
    PAI_MAX_GPU,
    audit_undertrained_checkpoint_set,
    atomic_json,
    build_c_training_launcher_contract,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qpilots-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--expected-step", type=int, action="append", default=[])
    parser.add_argument("--base-checkpoint", type=Path, help="real pi05 base checkpoint for C training; never synthesized")
    parser.add_argument("--python", default="python")
    parser.add_argument("--gpu-count", type=int, default=PAI_MAX_GPU)
    parser.add_argument("--resource-pool", default="idle")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    expected_steps = args.expected_step or None
    if expected_steps is not None and len(expected_steps) != len(args.checkpoint):
        raise SystemExit("--expected-step must be supplied once per --checkpoint")
    contract = build_c_training_launcher_contract(
        qpilots_root=args.qpilots_root,
        output_root=args.output_root,
        checkpoint_paths=args.checkpoint,
        expected_steps=expected_steps,
        base_checkpoint=args.base_checkpoint,
        python=args.python,
        gpu_count=args.gpu_count,
        resource_pool=args.resource_pool,
    )
    atomic_json(args.output, contract)
    print(json.dumps(contract, indent=2, sort_keys=True))
    return 0 if contract["audit"]["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
