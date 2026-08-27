#!/usr/bin/env python3
"""Small command-line entry points for the frozen Stage-1R pipeline.

The subcommands are intentionally separate so that calibration can be
committed before natural outcomes are read by the analysis command.
"""

from __future__ import annotations

import argparse
import json
import os

from r142_stage_r.phase1 import (
    NaturalPhase1Runtime,
    PHASE1_PROTOCOL_ID,
    TASKS,
    select_all_phase1_episodes,
    validate_all_selection_manifest,
    validate_phase1_config,
    validate_natural_bundle,
    validate_cell,
)
from r142_stage_r.phase1_analysis import analyze_phase1r, calibrate_phase1r, validate_phase1_analysis
from r142_stage_r.phase1_controls import collect_control_bundle, validate_control_bundle
from r142_stage_r.protocol import atomic_json


def _owner(value: str) -> tuple[int, int] | None:
    if value.lower() in {"none", "skip"}:
        return None
    uid, gid = value.split(":", 1)
    return int(uid), int(gid)


def main() -> None:
    parser = argparse.ArgumentParser(description="R142 Stage-R Phase-1R collection and analysis")
    sub = parser.add_subparsers(dest="command", required=True)

    select = sub.add_parser("select")
    select.add_argument("--raw", required=True)
    select.add_argument("--output", required=True)

    controls = sub.add_parser("controls")
    controls.add_argument("--kind", choices=("positive", "null"), required=True)
    controls.add_argument("--output", required=True)

    controls_validate = sub.add_parser("validate-controls")
    controls_validate.add_argument("--root", required=True)
    controls_validate.add_argument("--kind", choices=("positive", "null"), required=True)
    controls_validate.add_argument("--owner", default="none")

    calibrate = sub.add_parser("calibrate")
    calibrate.add_argument("--controls", required=True)
    calibrate.add_argument("--output", required=True)
    calibrate.add_argument("--shuffles", type=int, default=1000)
    calibrate.add_argument("--owner", default="2254:2254")

    collect = sub.add_parser("collect-natural")
    collect.add_argument("--raw", required=True)
    collect.add_argument("--selection", required=True)
    collect.add_argument("--output", required=True)
    collect.add_argument("--suite", required=True)
    collect.add_argument("--task-id", type=int, required=True)
    collect.add_argument("--qpilots-root", required=True)
    collect.add_argument("--libero-root", required=True)
    collect.add_argument("--checkpoint", required=True)
    collect.add_argument("--microbatch", type=int, default=8)
    collect.add_argument("--max-steps", type=int, default=1000)

    analyze = sub.add_parser("analyze")
    analyze.add_argument("--natural", required=True)
    analyze.add_argument("--controls", required=True)
    analyze.add_argument("--calibration", required=True)
    analyze.add_argument("--output", required=True)
    analyze.add_argument("--bootstrap", type=int, default=10000)
    analyze.add_argument("--owner", default="2254:2254")

    verify = sub.add_parser("validate-analysis")
    verify.add_argument("--output", required=True)

    validate_selection = sub.add_parser("validate-selection")
    validate_selection.add_argument("--root", required=True)

    validate_config = sub.add_parser("validate-config")
    validate_config.add_argument("--protocol", required=True)
    validate_config.add_argument("--shards")

    validate_natural = sub.add_parser("validate-natural")
    validate_natural.add_argument("--selection", required=True)
    validate_natural.add_argument("--natural", required=True)
    validate_natural.add_argument("--owner", default="2254:2254")

    args = parser.parse_args()
    if args.command == "select":
        print(json.dumps(select_all_phase1_episodes(args.raw, args.output), indent=2))
    elif args.command == "controls":
        print(json.dumps(collect_control_bundle(args.kind, args.output), indent=2))
    elif args.command == "validate-controls":
        result = validate_control_bundle(args.root, args.kind, require_owner=_owner(args.owner))
        print(json.dumps(result, indent=2))
        if not result["valid"]:
            raise SystemExit(1)
    elif args.command == "calibrate":
        print(json.dumps(calibrate_phase1r(args.controls, args.output, shuffles=args.shuffles, require_owner=_owner(args.owner)), indent=2))
    elif args.command == "collect-natural":
        runtime = NaturalPhase1Runtime(qpilots_root=args.qpilots_root, libero_root=args.libero_root, checkpoint=args.checkpoint, microbatch=args.microbatch)
        result = runtime.collector(max_steps=args.max_steps).collect_task(args.raw, args.selection, args.output, args.suite, args.task_id)
        print(json.dumps(result, indent=2))
    elif args.command == "analyze":
        print(json.dumps(analyze_phase1r(args.natural, args.controls, args.calibration, args.output, bootstrap_replicates=args.bootstrap, require_owner=_owner(args.owner)), indent=2))
    elif args.command == "validate-analysis":
        valid, errors = validate_phase1_analysis(args.output)
        print(json.dumps({"valid": valid, "errors": errors}, indent=2))
        if not valid:
            raise SystemExit(1)
    elif args.command == "validate-selection":
        result = validate_all_selection_manifest(args.root)
        print(json.dumps(result, indent=2))
        if not result["valid"]:
            raise SystemExit(1)
    elif args.command == "validate-config":
        result = validate_phase1_config(args.protocol, args.shards)
        print(json.dumps(result, indent=2))
        if not result["valid"]:
            raise SystemExit(1)
    elif args.command == "validate-natural":
        result = validate_natural_bundle(args.selection, args.natural, require_owner=_owner(args.owner))
        print(json.dumps(result, indent=2))
        if not result["valid"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
