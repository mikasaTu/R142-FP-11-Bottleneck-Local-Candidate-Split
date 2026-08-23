#!/usr/bin/env python3
from __future__ import annotations

import argparse

from r142_stage_r.gates import run_engineering_gates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qpilots-root", required=True)
    parser.add_argument("--libero-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--skip-e6", action="store_true")
    args = parser.parse_args()
    result = run_engineering_gates(
        qpilots_root=args.qpilots_root,
        libero_root=args.libero_root,
        checkpoint=args.checkpoint,
        output=args.output,
        microbatch=args.microbatch,
        run_e6=not args.skip_e6,
    )
    print(f"decision={result['decision']}")


if __name__ == "__main__":
    main()
