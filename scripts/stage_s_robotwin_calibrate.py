#!/usr/bin/env python3
"""Explicitly refuse the out-of-scope Step-0 calibration entry point.

Stage-S substrate A is frozen as the ten-task ``10 x 16 x 32`` main screen;
it has no Step-0 calibration.  Keeping this command visible prevents a PAI
launcher from accidentally introducing an unregistered calibration phase.
"""

from __future__ import annotations

import argparse
import json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(
        json.dumps(
            {
                "status": "BLOCKED_CAPABILITY",
                "reason": "Stage-S substrate-A has no Step-0 calibration; run stage_s_robotwin_main.py --phase main",
            },
            ensure_ascii=False,
        )
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
