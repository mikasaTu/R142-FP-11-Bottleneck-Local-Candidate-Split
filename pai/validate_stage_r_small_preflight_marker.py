#!/usr/bin/env python3
"""Validate one small persisted Phase-1R preflight result."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROTOCOL = "r142-stage-r-phase1r-human-override-v1"
EXPECTED = {
    "protocol": {"protocol_id": PROTOCOL, "valid": True, "errors": []},
    "selection": {
        "protocol_id": PROTOCOL,
        "task_count": 40,
        "valid": True,
        "errors": [],
    },
    "positive": {
        "protocol_id": PROTOCOL,
        "control_kind": "positive",
        "checked_cells": 240,
        "valid": True,
        "errors": [],
    },
    "null": {
        "protocol_id": PROTOCOL,
        "control_kind": "null",
        "checked_cells": 240,
        "valid": True,
        "errors": [],
    },
}
FILES = {
    "protocol": "protocol_validation.json",
    "selection": "selection_validation.json",
    "positive": "positive_control_validation.json",
    "null": "null_control_validation.json",
}


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[2] not in EXPECTED:
        raise SystemExit("usage: validate_stage_r_small_preflight_marker.py RUNTIME KIND")
    runtime, kind = Path(sys.argv[1]), sys.argv[2]
    path = runtime / FILES[kind]
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"missing preflight marker: {path}")
    if json.loads(path.read_text(encoding="utf-8")) != EXPECTED[kind]:
        raise SystemExit(f"preflight marker drifted: {path}")
    print(json.dumps({"valid": True, "kind": kind}, sort_keys=True))


if __name__ == "__main__":
    main()
