#!/usr/bin/env python3
"""Fail-closed validator for the frozen Stage-S S4/S5 PAI contract.

The registry must run this check immediately before ``pai-job submit``.  The
checked-in templates intentionally contain explicit freeze-time placeholders;
that makes an accidental submission impossible until the protocol, source,
checkpoint, dependency and output pins have been read back and repinned.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path
from zoneinfo import ZoneInfo


RESOURCE_ALIAS = "idle-a800-robot-stage-s-graphics-8gpu"
RESOURCE_ID = "quota1ssrabud0bh"
POOL = "robot_idle"
QUOTA = "exp-robot"
OVERSOLD = "AcceptQuotaOverSold"
AIMASTER = (
    "--job-execution-mode=Sync --enable-job-restart=True "
    "--max-num-of-job-restart=50 --fault-tolerant-policy=OnFailure"
)
NEW_ROOT = "/mnt/cpfs/zbl-cpfs-new/"
PLACEHOLDER_RE = re.compile(
    r"(__REQUIRED_[A-Z0-9_]+__|<[^>]+>|\$\{[^}]+\}|\{\{(?!ARTIFACT_DIR\}\}|RUN_ID\}\})[^}]+\}\}|\b(?:TODO|TBD|REPLACE_ME)\b)",
    re.IGNORECASE,
)
SHA40_RE = re.compile(r"^[0-9a-f]{40}$")
SHA64_RE = re.compile(r"^[0-9a-f]{64}$")


class ContractError(ValueError):
    """A registry payload is unsafe to submit."""


def _required(mapping: dict, path: str):
    value = mapping
    for component in path.split("."):
        if not isinstance(value, dict) or component not in value:
            raise ContractError(f"missing required contract field: {path}")
        value = value[component]
    if value is None or value == "":
        raise ContractError(f"empty required contract field: {path}")
    return value


def _reject_placeholders(value, path: str = "contract") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_placeholders(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_placeholders(child, f"{path}[{index}]")
    elif isinstance(value, str) and PLACEHOLDER_RE.search(value):
        raise ContractError(f"unresolved freeze-time placeholder at {path}")


def _sha(value, path: str, length: int = 64) -> None:
    pattern = SHA40_RE if length == 40 else SHA64_RE
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ContractError(f"{path} must be a lowercase full SHA-{length * 4}")


def _path_pin(value, path: str) -> None:
    if not isinstance(value, str) or not value.startswith(NEW_ROOT):
        raise ContractError(f"{path} must be an absolute path under {NEW_ROOT}")


def validate(payload: dict, *, now: dt.datetime | None = None, check_files: bool = False) -> dict:
    if payload.get("schema") != "pai-job-registry-v2":
        raise ContractError("schema must be pai-job-registry-v2")
    if payload.get("status") not in {"IMPLEMENTED_NOT_SUBMITTED", "FROZEN_READY"}:
        raise ContractError("Stage-S S4/S5 template status must be IMPLEMENTED_NOT_SUBMITTED or FROZEN_READY")
    if payload.get("resource_alias") != RESOURCE_ALIAS:
        raise ContractError("resource_alias is not the frozen idle robot alias")
    if payload.get("resource_id") != RESOURCE_ID:
        raise ContractError("resource_id is not the frozen idle robot resource")
    if payload.get("quota_name", payload.get("quota")) != QUOTA:
        raise ContractError("quota must be exp-robot")

    resource = _required(payload, "resource")
    worker = _required(payload, "worker")
    for block_name, block in (("resource", resource), ("worker", worker)):
        expected = {"count": 1} if block_name == "worker" else {"workers": 1}
        expected.update(
            {
                "gpu": 8,
                "cpu": 88,
                "memory": "1400Gi",
                "shared_memory": "1400Gi",
            }
        )
        for key, expected_value in expected.items():
            if block.get(key) != expected_value:
                raise ContractError(f"{block_name}.{key} must be {expected_value!r}")
        if block.get("alias", RESOURCE_ALIAS) != RESOURCE_ALIAS:
            raise ContractError(f"{block_name}.alias drifted")
        if block.get("id", RESOURCE_ID) != RESOURCE_ID:
            raise ContractError(f"{block_name}.id drifted")
        if block.get("pool", POOL) != POOL:
            raise ContractError(f"{block_name}.pool must be {POOL}")
        if block.get("quota_name", block.get("quota", QUOTA)) != QUOTA:
            raise ContractError(f"{block_name}.quota must be {QUOTA}")
        if block.get("oversold_type", OVERSOLD) != OVERSOLD:
            raise ContractError(f"{block_name}.oversold_type must be {OVERSOLD}")
    if worker.get("count") != 1 or worker.get("gpu") != 8 or worker.get("cpu") != 88:
        raise ContractError("worker shape must be exactly 1x8 GPU/88 CPU")
    if worker.get("memory") != "1400Gi" or worker.get("shared_memory") != "1400Gi":
        raise ContractError("worker memory/shared_memory must be exactly 1400Gi")

    submission = _required(payload, "submission")
    if submission.get("disable_ecs_stock_check") is not True:
        raise ContractError("idle submission must set disable_ecs_stock_check=true")
    tags = _required(submission, "tags")
    for key in ("managed_by", "purpose", "project", "substrate", "task", "hardware", "resource_pool"):
        _required(tags, key)
    if tags.get("hardware") != "8xa800-idle" or tags.get("resource_pool") != POOL:
        raise ContractError("submission tags do not identify the idle 8xA800 robot pool")

    fault = _required(payload, "fault_tolerance")
    if fault.get("execution_mode") != "Sync" or fault.get("policy") != "OnFailure":
        raise ContractError("fault tolerance must be Sync/OnFailure")
    if fault.get("launcher_attempts") != 1 or fault.get("maximum_platform_restarts") != 50:
        raise ContractError("fault tolerance must be one launcher attempt and 50 platform restarts")
    if fault.get("max_num_of_job_restart") != 50 or fault.get("pai_automatic_fault_tolerance") is not True:
        raise ContractError("idle restart contract drifted")
    if fault.get("aimaster_args") != AIMASTER:
        raise ContractError("AIMaster args do not match Sync OnFailure50")

    # Reject unresolved values before path/SHA diagnostics.  Resource and
    # worker drift above remains independently observable in negative tests.
    _reject_placeholders(payload)
    runtime = _required(payload, "runtime")
    path_fields = (
        "project_dir", "runtime_repo", "command_file", "finalizer_file", "protocol_path",
        "checkpoint_path", "n32_root", "s4_root", "s5_root", "output_root",
    )
    for field in path_fields:
        _path_pin(_required(runtime, field), f"runtime.{field}")
    _path_pin(_required(runtime, "python"), "runtime.python")
    _required(runtime, "adapter_spec")
    adapter_spec = runtime["adapter_spec"]
    if (
        not isinstance(adapter_spec, str)
        or adapter_spec.count(":") != 1
        or any(token in adapter_spec.lower() for token in ("fake", "synthetic", "mock", "fixture"))
    ):
        raise ContractError("runtime.adapter_spec must name a real module:factory adapter")
    for field in (
        "source_sha256",
        "command_file_sha256",
        "payload_sha256",
        "finalizer_sha256",
        "protocol_sha256",
        "checkpoint_sha256",
    ):
        _sha(_required(runtime, field), f"runtime.{field}")
    calibration_reports = _required(runtime, "calibration_reports")
    if not isinstance(calibration_reports, dict) or set(calibration_reports) != {"B", "C"}:
        raise ContractError("runtime.calibration_reports must contain exactly B and C")
    for substrate in ("B", "C"):
        report = calibration_reports[substrate]
        if not isinstance(report, dict):
            raise ContractError(f"runtime.calibration_reports.{substrate} must be an object")
        _path_pin(_required(report, "path"), f"runtime.calibration_reports.{substrate}.path")
        _sha(_required(report, "sha256"), f"runtime.calibration_reports.{substrate}.sha256")
    _sha(_required(runtime, "source_commit"), "runtime.source_commit", 40)
    _sha(_required(runtime, "protocol_git_commit"), "runtime.protocol_git_commit", 40)
    dependencies = _required(runtime, "dependencies")
    if not isinstance(dependencies, dict) or not dependencies:
        raise ContractError("runtime.dependencies must contain every executable dependency")
    for name, dependency in dependencies.items():
        if not isinstance(dependency, dict):
            raise ContractError(f"runtime.dependencies.{name} must be an object")
        _path_pin(_required(dependency, "path"), f"runtime.dependencies.{name}.path")
        _sha(_required(dependency, "commit"), f"runtime.dependencies.{name}.commit", 40)
        _sha(_required(dependency, "sha256"), f"runtime.dependencies.{name}.sha256")
    for field in ("uid", "gid"):
        if runtime.get(field) != 2254:
            raise ContractError(f"runtime.{field} must be numeric 2254")
    if runtime.get("identity_mechanism") != "controller_inline_bootstrap_then_setpriv":
        raise ContractError("runtime identity mechanism must drop to 2254:2254")

    scientific = _required(payload, "scientific_contract")
    exact = {
        "search_grid_count": 9,
        "search_branch_count": 4,
        "heldout_oracle_branch_count": 8,
        "heldout_random_branch_count": 8,
        "paired_bootstrap_seed": 14211,
        "paired_bootstrap_replicates": 10000,
        "fresh_candidate_count": 32,
    }
    for field, expected_value in exact.items():
        if scientific.get(field) != expected_value:
            raise ContractError(f"scientific_contract.{field} must remain {expected_value}")
    if scientific.get("source_candidate_budget") != 32 or scientific.get("no_vla") is not True:
        raise ContractError("S4/S5 scientific budget or no-VLA boundary drifted")

    blackout = _required(payload, "blackout")
    if blackout.get("timezone") != "Asia/Shanghai" or blackout.get("windows") != ["09:30-09:40", "19:30-19:40"]:
        raise ContractError("blackout windows must be 09:30-09:40 and 19:30-19:40 Asia/Shanghai")
    if blackout.get("fail_closed") is not True:
        raise ContractError("blackout guard must fail closed")
    current = now or dt.datetime.now(ZoneInfo("Asia/Shanghai"))
    minute = current.hour * 60 + current.minute
    if (9 * 60 + 30) <= minute < (9 * 60 + 40) or (19 * 60 + 30) <= minute < (19 * 60 + 40):
        raise ContractError("PAI submission is forbidden during the Asia/Shanghai blackout window")

    if payload.get("status") == "FROZEN_READY" and _required(payload, "evidence.contract_ready") is not True:
        raise ContractError("FROZEN_READY requires evidence.contract_ready=true")
    if check_files:
        for field in path_fields:
            path = Path(runtime[field])
            if not path.is_file() and field not in {"project_dir", "runtime_repo", "n32_root", "s4_root", "s5_root", "output_root"}:
                raise ContractError(f"pinned file does not exist: runtime.{field}={path}")
    return {
        "status": "VALID",
        "resource": {"alias": RESOURCE_ALIAS, "id": RESOURCE_ID, "pool": POOL, "workers": 1, "gpu": 8, "cpu": 88, "memory": "1400Gi", "shared_memory": "1400Gi"},
        "scientific_contract": exact,
        "blackout_checked_at": current.isoformat(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--check-files", action="store_true")
    args = parser.parse_args(argv)
    try:
        with args.config.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ContractError("config root must be an object")
        result = validate(payload, check_files=args.check_files)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except (OSError, ValueError, TypeError, json.JSONDecodeError, ContractError) as exc:
        print(f"STAGE_S45_PAI_CONTRACT_REJECTED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
