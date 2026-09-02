#!/usr/bin/env python3
"""Render a Stage-S idle-PAI payload without submitting a job.

The output is a frozen, fail-closed hand-off for the canonical PAI
orchestrator.  This script never imports DLC credentials and never calls a
submit API.  It is suitable for static validation and for handing a payload
to the parent agent after the external mount/identity checks pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from r142_stage_s.libero import (
    atomic_json,
    build_pai_stage_s_payload,
    build_c_training_launcher_contract,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--working-directory", type=Path)
    parser.add_argument("--command", nargs="+", help="real foreground command; no shell fallback is generated")
    parser.add_argument("--c-contract", type=Path, help="JSON emitted by stage_s_libero_c.py")
    parser.add_argument("--payload", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    contract = None
    if args.c_contract is not None:
        contract = json.loads(args.c_contract.read_text(encoding="utf-8"))
    payload = build_pai_stage_s_payload(
        run_id=args.run_id,
        output_root=args.output_root,
        log_root=args.log_root,
        command=args.command,
        working_directory=args.working_directory,
        c_contract=contract,
    )
    atomic_json(args.payload, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
