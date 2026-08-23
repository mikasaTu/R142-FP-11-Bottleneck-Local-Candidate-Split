#!/usr/bin/env python3
from __future__ import annotations

import argparse

from r142_stage_r.calibration import calibrate_thresholds
from r142_stage_r.protocol import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--shuffles", type=int, default=1000)
    args = parser.parse_args()
    payload = calibrate_thresholds(args.shuffles)
    atomic_json(args.output, payload)
    print(f"wrote frozen control/null thresholds to {args.output}")
    print(f"decision={payload['decision']}")


if __name__ == "__main__":
    main()
