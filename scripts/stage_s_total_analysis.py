#!/usr/bin/env python3
"""Verify and analyze complete Stage-S A/B/C outputs in one frozen report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from r142_stage_s.total_analysis import TotalAnalysisError, analyze_stage_s  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--controls", required=True, type=Path)
    parser.add_argument("--arms-json", required=True, type=Path, help="JSON mapping with exactly A/B/C arm inputs")
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        arms = json.loads(args.arms_json.read_text(encoding="utf-8"))
        if not isinstance(arms, dict):
            raise TotalAnalysisError("--arms-json root must be an object")
        result = analyze_stage_s(
            protocol_path=args.protocol,
            controls=args.controls,
            arms=arms,
            output_root=args.output_root,
        )
    except (OSError, ValueError, TypeError, TotalAnalysisError) as exc:
        print(json.dumps({"status": "BLOCKED_TOTAL_ANALYSIS", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "decision_code": result["decision_code"], "output_root": result.get("output_root")}, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
