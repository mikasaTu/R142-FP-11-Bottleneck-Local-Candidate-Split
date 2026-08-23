#!/usr/bin/env python3
from __future__ import annotations

import argparse

from r142_stage_r.analyze import analyze_phase0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze_phase0(args.raw, args.thresholds, args.output)
    print(f"decision={result['decision']}")
    print(f"retained={result['retained_tasks']}")


if __name__ == "__main__":
    main()
