#!/usr/bin/env python3
"""Finalize a complete Stage-S S4/S5 evaluation.

The finalizer accepts only protocol-bound, SHA-verified N=32 families; it
requires S4 coverage for every near-all-fail family and S5 coverage for every
family.  It then invokes the frozen 10,000-replicate S4 bootstrap and the
fresh-seed S5 analysis.  It never accepts a typed ``oracle_recovered`` or
``random_recovered`` value without the raw terminal branch records emitted by
``stage_s_s45.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from r142_stage_s.s45_runtime import (  # noqa: E402
    S45Error,
    finalise_s45,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--substrate", choices=("A", "B", "C"), required=True)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--n32-root", required=True, type=Path)
    parser.add_argument("--s4-root", required=True, type=Path)
    parser.add_argument("--s5-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = finalise_s45(
            args.n32_root,
            args.s4_root,
            args.s5_root,
            args.protocol,
            args.output_root,
            expected_substrate=args.substrate,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except (S45Error, OSError, ValueError, TypeError) as exc:
        print(json.dumps({"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
