#!/usr/bin/env python3
"""Freeze terminal Stage-S B/C calibration and protocol acceptance.

This is a non-submitting foreground utility.  It reads terminal calibration
aggregates, completion markers, manifests, and the accepted C training
lineage; it never runs an episode or inspects S2--S5 data.  Destination
defaults point at the canonical CPFS log tree, but no output is created until
the operator invokes this command with real terminal evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from r142_stage_s.calibration_freeze import (  # noqa: E402
    freeze_calibration_reports,
    freeze_protocol,
)


DEFAULT_LOG_ROOT = Path("/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_s")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b-result", type=Path, required=True, help="terminal B CALIBRATION_RESULT.json")
    parser.add_argument("--b-completion-marker", type=Path, required=True, help="terminal B completion marker")
    parser.add_argument("--c-result", type=Path, required=True, help="terminal C CALIBRATION_RESULT.json")
    parser.add_argument("--c-completion-marker", type=Path, required=True, help="terminal C completion marker")
    parser.add_argument("--c-lineage", type=Path, required=True, help="accepted C training completion/lineage JSON")
    parser.add_argument("--protocol-md", type=Path, required=True, help="committed repo stage-s/PROTOCOL.md")
    parser.add_argument("--protocol-git-commit", required=True, help="full 40-hex GitHub protocol commit")
    parser.add_argument("--repo-root", type=Path, required=True, help="GitHub-backed checkout containing the commit")
    parser.add_argument("--b-report", type=Path, default=DEFAULT_LOG_ROOT / "b_calibration" / "CALIBRATION_REPORT.json")
    parser.add_argument("--c-report", type=Path, default=DEFAULT_LOG_ROOT / "c_calibration" / "CALIBRATION_REPORT.json")
    parser.add_argument("--protocol-output", type=Path, default=DEFAULT_LOG_ROOT / "stage_s" / "protocol" / "FROZEN_PROTOCOL.json")
    parser.add_argument(
        "--no-materialize-protocol-md",
        action="store_true",
        help="require an already-present adjacent runtime PROTOCOL.md instead of copying the committed markdown",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    reports = freeze_calibration_reports(
        b_result=args.b_result,
        b_completion_marker=args.b_completion_marker,
        c_result=args.c_result,
        c_completion_marker=args.c_completion_marker,
        c_lineage=args.c_lineage,
        b_report=args.b_report,
        c_report=args.c_report,
    )
    acceptance = freeze_protocol(
        protocol_md=args.protocol_md,
        protocol_git_commit=args.protocol_git_commit,
        b_report=args.b_report,
        c_report=args.c_report,
        output_path=args.protocol_output,
        repo_root=args.repo_root,
        materialize_protocol_md=not args.no_materialize_protocol_md,
    )
    print(json.dumps({"reports": reports, "protocol": acceptance}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
