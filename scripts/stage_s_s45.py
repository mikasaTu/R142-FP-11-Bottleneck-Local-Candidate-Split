#!/usr/bin/env python3
"""Execute Stage-S S4/S5 with an explicitly supplied real substrate adapter.

Example (PAI payloads should invoke the same command after installing the
official LIBERO or RoboTwin adapter):

    PYTHONPATH=src python scripts/stage_s_s45.py \
      --phase s4 --substrate B \
      --protocol /mnt/.../FROZEN_PROTOCOL.json \
      --n32-root /mnt/.../results/B \
      --output-root /mnt/.../s45/B/s4 \
      --adapter my_stage_s_adapter:build_adapter

There is intentionally no ``--synthetic`` flag and no built-in simulator or
policy.  A missing hook fails before any completion marker can be emitted.
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
    ProtocolAuthority,
    S45Error,
    discover_n32_families,
    load_adapter,
    run_s4,
    run_s5,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("s4", "s5", "both"), required=True)
    parser.add_argument("--substrate", choices=("A", "B", "C"), required=True)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument(
        "--calibration-report",
        type=Path,
        help="optional explicit B/C calibration report; its canonical SHA binding is rechecked",
    )
    parser.add_argument("--main-source-commit", type=str)
    parser.add_argument("--main-source-sha256", type=str)
    parser.add_argument("--n32-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--adapter",
        required=True,
        help="real adapter factory in module:factory form; it must return S45Adapter",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        protocol = ProtocolAuthority.load(
            args.protocol,
            require_canonical=True,
            substrate=args.substrate,
            calibration_report=args.calibration_report,
        )
        families = discover_n32_families(
            args.n32_root,
            protocol=protocol,
            expected_main_source_commit=args.main_source_commit,
            expected_main_source_sha256=args.main_source_sha256,
        )
        result: dict[str, object] = {
            "substrate": args.substrate,
            "phase": args.phase,
            "protocol": protocol.identity(),
            "n32_family_count": len(families),
        }
        if args.phase in {"s4", "both"}:
            adapter = load_adapter(args.adapter, protocol=protocol, substrate=args.substrate)
            result["s4"] = run_s4(families, protocol, adapter, args.output_root / "s4")
        if args.phase in {"s5", "both"}:
            adapter = load_adapter(args.adapter, protocol=protocol, substrate=args.substrate)
            result["s5"] = run_s5(families, protocol, adapter, args.output_root / "s5")
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except (S45Error, OSError, ValueError, TypeError) as exc:
        print(json.dumps({"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
