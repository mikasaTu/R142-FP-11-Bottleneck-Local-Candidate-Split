#!/usr/bin/env python3
"""Quickly validate persisted Phase-1R preflight results."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROTOCOL = "r142-stage-r-phase1r-human-override-v1"


def load_exact(path: Path, expected: dict) -> None:
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"missing preflight result: {path}")
    if json.loads(path.read_text(encoding="utf-8")) != expected:
        raise SystemExit(f"preflight result drifted: {path}")


def main() -> None:
    if len(sys.argv) != 7:
        raise SystemExit(
            "usage: validate_stage_r_phase1r_preflight_bundle.py "
            "RUNTIME CHECKPOINT_TREE ATTESTATION_SHA UID:GID RUN_ID EXECUTION_SHARD"
        )
    runtime = Path(sys.argv[1])
    checkpoint_tree, attestation_sha, owner = sys.argv[2:5]
    run_id, execution_shard = sys.argv[5:7]
    expected_small = {
        "protocol_validation.json": {"protocol_id": PROTOCOL, "valid": True, "errors": []},
        "selection_validation.json": {
            "protocol_id": PROTOCOL,
            "task_count": 40,
            "valid": True,
            "errors": [],
        },
        "positive_control_validation.json": {
            "protocol_id": PROTOCOL,
            "control_kind": "positive",
            "checked_cells": 240,
            "valid": True,
            "errors": [],
        },
        "null_control_validation.json": {
            "protocol_id": PROTOCOL,
            "control_kind": "null",
            "checked_cells": 240,
            "valid": True,
            "errors": [],
        },
        "phase0_authority_validation.json": {
            "valid": True,
            "authority_records": 40,
            "raw_tasks": 40,
            "raw_rollouts": 20480,
            "raw_files": 80,
        },
        "checkpoint_validation.json": {
            "valid": True,
            "validation_mode": "frozen_full_content_attestation_plus_exact_metadata_and_content_probes",
            "file_count": 16,
            "bytes": 12439085481,
            "tree_sha256": checkpoint_tree,
            "attestation_sha256": attestation_sha,
            "uid": int(owner.split(":")[0]),
            "gid": int(owner.split(":")[1]),
        },
    }
    for name, expected in expected_small.items():
        load_exact(runtime / name, expected)
    marker = json.loads((runtime / "frozen_source_verified.json").read_text(encoding="utf-8"))
    if marker.get("marker_type") != "frozen_source_resume_attestation" or marker.get("owner") != owner:
        raise SystemExit("frozen-source preflight marker drifted")
    load_exact(
        runtime / "CPU_PRESEEDED_PREFLIGHT.json",
        {
            "schema_version": 1,
            "marker_type": "outcome_blind_cpu_preseeded_preflight",
            "run_id": run_id,
            "execution_shard": execution_shard,
            "outcomes_read": False,
            "natural_cells": 0,
            "scientific_contract_changed": False,
        },
    )
    print(json.dumps({"valid": True, "marker_count": len(expected_small) + 2}, sort_keys=True))


if __name__ == "__main__":
    main()
