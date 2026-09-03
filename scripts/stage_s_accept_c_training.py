#!/usr/bin/env python3
"""Publish the terminal Stage-S C training acceptance manifest.

This command is a read-only admission check until every input gate passes. It
never talks to PAI and never changes a training/checkpoint/status directory.
Use the exact registry evidence and the exact CPFS roots from one C run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from r142_stage_s.c_training_acceptance import (  # noqa: E402
    EXPECTED_GID,
    EXPECTED_UID,
    write_c_training_acceptance,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-run", "--registry-run-root", dest="registry_run", type=Path, required=True)
    parser.add_argument("--registry-result", "--result", dest="registry_result", type=Path, required=True)
    parser.add_argument("--submission-state", "--submission", dest="submission_state", type=Path, required=True)
    parser.add_argument("--resolved", "--resolved-payload", dest="resolved", type=Path, required=True)
    parser.add_argument("--jobs-ledger", "--ledger", dest="jobs_ledger", type=Path, required=True)
    parser.add_argument(
        "--terminal-getjob",
        "--getjob-terminal",
        "--sanitized-getjob",
        dest="terminal_getjob",
        type=Path,
        required=True,
        help="sanitized terminal GetJob JSON; raw GetJob output is not accepted",
    )
    parser.add_argument(
        "--terminal-getjob-sha",
        "--getjob-terminal-sha",
        "--sanitized-getjob-sha",
        dest="terminal_getjob_sha",
        type=Path,
        required=True,
    )
    parser.add_argument("--c-run-root", "--run-root", dest="c_run_root", type=Path, required=True)
    parser.add_argument("--c-status-root", "--status-root", dest="c_status_root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", "--c-checkpoint-root", dest="checkpoint_root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="unique ACCEPTED_C_TRAINING.json destination (usually the stable c_status parent)",
    )
    parser.add_argument("--expected-uid", type=int, default=EXPECTED_UID)
    parser.add_argument("--expected-gid", type=int, default=EXPECTED_GID)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        record = write_c_training_acceptance(
            output_path=args.output,
            registry_run=args.registry_run,
            registry_result=args.registry_result,
            submission_state=args.submission_state,
            resolved=args.resolved,
            jobs_ledger=args.jobs_ledger,
            terminal_getjob=args.terminal_getjob,
            terminal_getjob_sha=args.terminal_getjob_sha,
            c_run_root=args.c_run_root,
            c_status_root=args.c_status_root,
            checkpoint_root=args.checkpoint_root,
            expected_uid=args.expected_uid,
            expected_gid=args.expected_gid,
        )
    except Exception as exc:
        # Malformed evidence must be a quiet, non-publishing refusal rather
        # than an uncaught traceback that could be mistaken for a partial
        # acceptance attempt.  The validator has no write side effects until
        # every gate returns successfully.
        print(f"C training acceptance refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
