#!/usr/bin/env python3
"""Run the Stage-S B/C pooled-success-only calibration.

The evaluator is deliberately injected by the real-runtime launcher.  This
script has no fake-policy fallback and never writes trajectory, family, or
S2--S5 statistics during calibration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from r142_stage_s.libero import (
    CALIBRATION_CANDIDATE_COUNT,
    CALIBRATION_INITIAL_STATES,
    CALIBRATION_TASK_IDS,
    PROXIMITY_MAGNITUDES,
    audit_undertrained_checkpoint_set,
    calibration_plan,
    import_callback,
    run_pooled_calibration,
    write_pooled_calibration,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--substrate", choices=("B", "C"), required=True)
    parser.add_argument("--report", type=Path, help="aggregate-only calibration JSON output")
    parser.add_argument("--evaluator", help="module:function real-runtime eventual-success evaluator")
    parser.add_argument("--checkpoint", action="append", default=[], help="C checkpoint path; repeat exactly four times")
    parser.add_argument("--dry-run", action="store_true", help="print the frozen aggregate-only plan and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.substrate == "B":
        settings = [float(value) for value in PROXIMITY_MAGNITUDES]
    else:
        if len(args.checkpoint) != 4:
            raise SystemExit("C calibration requires exactly four --checkpoint paths; no interpolation is allowed")
        audit = audit_undertrained_checkpoint_set(args.checkpoint)
        if not audit["valid"]:
            raise SystemExit("C checkpoint audit failed closed: " + "; ".join(audit["errors"]))
        settings = list(args.checkpoint)
    plan = calibration_plan(settings)
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if args.evaluator is None or args.report is None:
        raise SystemExit("--evaluator and --report are required for calibration; dry-run has no evaluator")
    evaluator = import_callback(args.evaluator)
    result = run_pooled_calibration(
        evaluator,
        settings,
        task_ids=CALIBRATION_TASK_IDS,
        initial_states=CALIBRATION_INITIAL_STATES,
        candidate_count=CALIBRATION_CANDIDATE_COUNT,
    )
    result["substrate"] = args.substrate
    # The writer intentionally rejects this substrate label: calibration
    # persistence is restricted to aggregate counters and the selected setting.
    result.pop("substrate", None)
    write_pooled_calibration(args.report, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
