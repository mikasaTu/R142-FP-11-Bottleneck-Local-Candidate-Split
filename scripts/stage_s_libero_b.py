#!/usr/bin/env python3
"""Generate the real Stage-S B LIBERO matrix.

This command writes four frozen offset settings, ten exact Stage-R task BDDL
files per setting, and at least sixteen fresh ``.pruned_init`` qpos rows per
task.  The qpos rows come from resetting the pinned LIBERO simulator with
fixed seeds; the old init tensors are never passed to the simulator.  There
is no synthetic or source-only fallback.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from r142_stage_s.libero import (
    B_INIT_STATE_COUNT,
    B_INIT_STATE_SEED_BASE,
    PROXIMITY_MAGNITUDES,
    build_b_variant_matrix,
    make_libero_qpos_simulator_factory,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bddl-root", type=Path, required=True)
    parser.add_argument("--source-init-root", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True, help="pinned LIBERO project/site root")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path)
    parser.add_argument("--count", type=int, default=B_INIT_STATE_COUNT)
    parser.add_argument("--seed-base", type=int, default=B_INIT_STATE_SEED_BASE)
    parser.add_argument("--render-gpu-device-id", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="print the frozen matrix contract without importing MuJoCo")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "substrate": "B",
                    "settings": list(PROXIMITY_MAGNITUDES),
                    "task_ids": list(range(10)),
                    "init_state_count": int(args.count),
                    "seed_base": int(args.seed_base),
                    "qpos_source": "real_libero_offscreen_render_env_reset",
                    "old_init_reused": False,
                    "no_pai_submit_performed": True,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    simulator_factory = make_libero_qpos_simulator_factory(
        args.libero_root,
        render_gpu_device_id=args.render_gpu_device_id,
    )
    result = build_b_variant_matrix(
        args.source_bddl_root,
        args.source_init_root,
        args.output_root,
        simulator_factory=simulator_factory,
        count=args.count,
        seed_base=args.seed_base,
        assets_root=args.assets_root,
    )
    print(json.dumps({"substrate": "B", "settings": result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
