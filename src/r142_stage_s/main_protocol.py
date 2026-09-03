"""Fail-closed gate for the frozen Stage-S main-screen protocol.

The B/C screen is only an observation of a pre-registered protocol.  This
module deliberately treats the protocol acceptance object as an input artifact
that must be independently read from stable CPFS on every process.  A caller
cannot replace it with a source-tree constant or with a calibration report
that was selected after looking at S2--S5 outcomes.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping


STAGE_S_PROTOCOL_ID = "r142-stage-s-v1"
PROTOCOL_ACCEPTANCE_SCHEMA = "r142-stage-s-protocol-acceptance-v1"
DEFAULT_PROTOCOL_ACCEPTANCE_PATH = Path(
    "/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_s/stage_s/protocol/FROZEN_PROTOCOL.json"
)
TASK_IDS = tuple(range(10))
INITIAL_STATE_IDS = tuple(range(16))
CANDIDATE_IDS = tuple(range(32))
WORLD_SIZE = 8

THRESHOLDS: dict[str, dict[str, object]] = {
    "S1": {"pooled_success_min": 0.30, "pooled_success_max": 0.60},
    "S2": {
        "near_all_fail_fraction_min": 0.10,
        "rho_min": 3.0,
        "near_all_fail_vs_binomial_min": 20.0,
    },
    "S3": {
        "median_t_div_fraction_min": 0.10,
        "t_div_zero_fraction_max": 0.25,
        "tau_quantile": 0.95,
    },
    "S4": {
        "recoverable_family_fraction_min": 0.30,
        "oracle_vs_random_ci_lower_min": 0.0,
        "paired_bootstrap_replicates": 10000,
    },
    "S5": {"best_of_n64_rescue_fraction_max": 0.05},
}

SEED_PLAN: dict[str, object] = {
    "namespace": STAGE_S_PROTOCOL_ID,
    "seed_base": 14211,
    "candidate_seed_rule": "SeedSequence([initial_seed, candidate_index])",
    "candidate": "sha256(r142-stage-s-v1|candidate|task_id|init_state|candidate_id)->first_8_bytes_big_endian",
    "environment": "sha256(r142-stage-s-v1|environment|task_id|init_state)->first_8_bytes_big_endian",
    "calibration": "sha256(r142-stage-s-v1|calibration|setting_index|task_id|init_state|candidate_id)->first_8_bytes_big_endian",
}
A_TASKS = (
    "blocks_ranking_size", "pick_diverse_bottles", "place_a2b_left",
    "place_a2b_right", "place_bread_basket", "place_bread_skillet",
    "place_can_basket", "place_fan", "place_object_scale", "place_shoe",
)
MAIN_BUDGET = {
    "task_count": 10, "families_per_task": 16,
    "candidates_per_family": 32, "terminal_episode_count": 5120,
    "world_size": 8,
}
DIVERGENCE_PROTOCOL: dict[str, object] = {
    "metric": "mean_pairwise_component_normalized_workspace_pose_rms_at_matched_control_step",
    "at_risk_rule": "candidate trajectory contains control step t; no interpolation or resampling",
    "A_pose": "left_xyz_wxyz_then_right_xyz_wxyz; each quaternion unit-normalized and sign-canonicalized",
    "A_scale": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
    "BC_pose": "Task64 observation/state first six values: eef_xyz plus eef_axis_angle; gripper excluded",
    "BC_scale": [1.0, 1.0, 1.0, 3.141592653589793, 3.141592653589793, 3.141592653589793],
    "tau": "per-task per-control-step 95th percentile of successful same-task matched-step pair distances",
    "tau_quantile": 0.95,
}
S4_PROTOCOL: dict[str, object] = {
    # Explicit authority-owned nine-point search grid.
    "search_t_grid": list(range(1, 10)),
    "anchor_rule": "lowest numeric candidate index among unsuccessful base-N32 candidates",
    "interior_grid_numerators": list(range(1, 10)),
    "interior_grid_denominator": 10,
    "interior_grid_rounding": "clamp(floor((j*H+5)/10),1,H-2), ordered unique; H<4 fails closed",
    "branch_count": 4,
    "search_branch_count": 4,
    "search_branches_per_step": 4,
    "heldout_branch_count": 8,
    "random_branch_count": 8,
    "evaluation_branch_count": 8,
    "oracle_t_rule": "maximize search successes/4 over interior grid; tie earliest control step",
    "random_t_rule": "for heldout branch k choose sha256 random-t digest modulo grid size",
    "random_location_hash_formula": "sha256(protocol_id|s4|family_id|episode_length|pair_index)->first_8_bytes_big_endian_mod_interior",
    "search_seed_formula": "sha256(r142-stage-s-v1|S4-search|substrate|family_id|t|branch_index)->first_8_bytes_big_endian",
    "random_t_seed_formula": "sha256(r142-stage-s-v1|S4-random-t|substrate|family_id|k)->first_8_bytes_big_endian modulo grid_size",
    "branch_seed_formula": "sha256(r142-stage-s-v1|S4-eval|substrate|family_id|k)->first_8_bytes_big_endian; paired across oracle/random",
    "paired_bootstrap_replicates": 10000,
    "paired_bootstrap_seed": 14211,
}
S5_PROTOCOL: dict[str, object] = {
    "base_candidate_count": 32,
    "fresh_candidate_indices": list(range(32, 64)),
    "extension_seed_formula": "sha256(r142-stage-s-v1|S5-extension|substrate|task_id|init_state|candidate_index)->first_8_bytes_big_endian",
}
DECISION_PROTOCOL: dict[str, object] = {
    "control_failure": "PIPELINE_INVALID",
    "headline_full_pass": "SUBSTRATE_QUALIFIED if A or B passes S1-S5",
    "weak_full_pass": "WEAK_SUBSTRATE_ONLY if only C passes S1-S5",
    "all_s1_fail": "NO_SUBSTRATE_AT_TARGET_DIFFICULTY",
    "gate_depth_order": ["S2", "S3", "S4", "S5"],
    "gate_failure_codes": {
        "S2": "NO_FAMILY_COLLAPSE",
        "S3_origin": "COLLAPSE_AT_ORIGIN",
        "S3_other": "UNRECOVERABLE_FAILURES",
        "S4": "UNRECOVERABLE_FAILURES",
        "S5": "BUDGET_SUFFICES",
    },
    "weak_arm_fallback": "when no S1-passing A/B arm exists, use the S1-passing C arm for the deeper falsification code",
}

FROZEN_SUMMARY: dict[str, object] = {
    "task_ids": list(TASK_IDS),
    "initial_state_ids": list(INITIAL_STATE_IDS),
    "candidate_ids": list(CANDIDATE_IDS),
    "initial_state_count": len(INITIAL_STATE_IDS),
    "candidate_budget": len(CANDIDATE_IDS),
    "world_size": WORLD_SIZE,
    "tasks": list(A_TASKS),
    "budget": MAIN_BUDGET,
    "seed_plan": SEED_PLAN,
    "thresholds": THRESHOLDS,
    "divergence": DIVERGENCE_PROTOCOL,
    "s4": S4_PROTOCOL,
    "s5": S5_PROTOCOL,
    "decision": DECISION_PROTOCOL,
    "compute_primary_unit": "policy_forward_pass",
    "compute_secondary_unit": "environment_step",
    "eventual_success_at_termination": True,
    "no_s2_s5_peeking": True,
}


class FrozenProtocolError(ValueError):
    """The stable protocol acceptance artifact is absent or inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FrozenProtocolError(f"{label} is missing or symlinked: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise FrozenProtocolError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise FrozenProtocolError(f"{label} must be a JSON object: {path}")
    return value


def _required(mapping: Mapping[str, Any], key: str, *, where: str) -> Any:
    if key not in mapping:
        raise FrozenProtocolError(f"{where} is missing required field {key!r}")
    return mapping[key]


def _full_sha(value: object, *, where: str, length: int) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{%d}" % length, value):
        raise FrozenProtocolError(f"{where} must be a lowercase full SHA-{length * 4}")
    return value


def _exact(value: object, expected: object, *, where: str) -> None:
    if value != expected:
        raise FrozenProtocolError(f"{where} mismatch: expected {expected!r}, got {value!r}")


def _protocol_summary(acceptance: Mapping[str, Any]) -> Mapping[str, Any]:
    summary = _required(acceptance, "frozen_summary", where="protocol acceptance")
    if not isinstance(summary, Mapping):
        raise FrozenProtocolError("protocol acceptance frozen_summary must be an object")
    for key, expected in FROZEN_SUMMARY.items():
        _exact(summary.get(key), expected, where=f"protocol acceptance frozen_summary.{key}")
    return summary


def _calibration_entries(acceptance: Mapping[str, Any]) -> Mapping[str, Any]:
    entries = _required(acceptance, "calibration_reports", where="protocol acceptance")
    if not isinstance(entries, Mapping):
        raise FrozenProtocolError("protocol acceptance calibration_reports must be an object")
    for substrate in ("B", "C"):
        entry = entries.get(substrate)
        if not isinstance(entry, Mapping):
            raise FrozenProtocolError(f"protocol acceptance calibration_reports.{substrate} is missing")
        _required(entry, "report_path", where=f"calibration_reports.{substrate}")
        _full_sha(
            _required(entry, "report_sha256", where=f"calibration_reports.{substrate}"),
            where=f"calibration_reports.{substrate}.report_sha256",
            length=64,
        )
        if substrate == "B":
            selected = _required(entry, "selected_setting", where="calibration_reports.B")
            if selected not in {"proximity_0.06m", "proximity_0.08m", "proximity_0.10m", "proximity_0.12m"}:
                raise FrozenProtocolError("calibration_reports.B.selected_setting is invalid")
            _exact(entry.get("variant_run_id"), "r142-stage-s-b-variants-20260903-r7", where="calibration_reports.B.variant_run_id")
        else:
            checkpoint = _required(entry, "selected_checkpoint", where="calibration_reports.C")
            if not isinstance(checkpoint, str) or not checkpoint:
                raise FrozenProtocolError("calibration_reports.C.selected_checkpoint is invalid")
            _full_sha(
                _required(entry, "selected_checkpoint_sha256", where="calibration_reports.C"),
                where="calibration_reports.C.selected_checkpoint_sha256",
                length=64,
            )
    return entries


def _verify_calibration_binding(
    entries: Mapping[str, Any],
    *,
    substrate: str,
    calibration_report: Path,
    freeze_report: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    entry = entries[substrate]
    report_path = calibration_report.resolve()
    declared_path = Path(str(entry["report_path"])).resolve()
    if report_path != declared_path:
        raise FrozenProtocolError(
            f"calibration_reports.{substrate}.report_path mismatch: expected {declared_path}, got {report_path}"
        )
    if report_path.is_symlink() or not report_path.is_file():
        raise FrozenProtocolError(f"calibration report is missing or symlinked: {report_path}")
    observed_sha = _sha256(report_path)
    if observed_sha != entry["report_sha256"]:
        raise FrozenProtocolError(f"calibration_reports.{substrate}.report_sha256 mismatch")
    if freeze_report is None:
        freeze_report = _read_json(report_path, label=f"{substrate} calibration report")
    if substrate == "B":
        _exact(freeze_report.get("selected_setting"), entry["selected_setting"], where="B selected_setting")
        _exact(freeze_report.get("variant_run_id"), entry["variant_run_id"], where="B variant_run_id")
    else:
        _exact(freeze_report.get("selected_checkpoint"), entry["selected_checkpoint"], where="C selected_checkpoint")
        _exact(
            freeze_report.get("selected_checkpoint_sha256"),
            entry["selected_checkpoint_sha256"],
            where="C selected_checkpoint_sha256",
        )
    return {
        "report_path": str(report_path),
        "report_sha256": observed_sha,
        "selected_setting": entry.get("selected_setting"),
        "variant_run_id": entry.get("variant_run_id"),
        "selected_checkpoint": entry.get("selected_checkpoint"),
        "selected_checkpoint_sha256": entry.get("selected_checkpoint_sha256"),
    }


def read_frozen_protocol(
    path: str | Path | None,
    *,
    substrate: str,
    calibration_report: str | Path,
    freeze_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Read and validate the immutable acceptance artifact for B/C.

    The protocol markdown is required to sit beside ``FROZEN_PROTOCOL.json``
    and its hash is checked on every invocation.  The acceptance file itself is
    also hashed and that digest is carried into all downstream artifacts.
    """

    if substrate not in {"B", "C"}:
        raise FrozenProtocolError(f"frozen protocol acceptance is only required for B/C, got {substrate!r}")
    acceptance_path = (DEFAULT_PROTOCOL_ACCEPTANCE_PATH if path is None else Path(path)).resolve()
    payload = _read_json(acceptance_path, label="frozen protocol acceptance")
    _exact(payload.get("schema"), PROTOCOL_ACCEPTANCE_SCHEMA, where="frozen protocol schema")
    _exact(payload.get("status"), "FROZEN", where="frozen protocol status")
    _exact(payload.get("protocol_id"), STAGE_S_PROTOCOL_ID, where="frozen protocol protocol_id")
    acceptance = _required(payload, "acceptance", where="frozen protocol")
    if not isinstance(acceptance, Mapping):
        raise FrozenProtocolError("frozen protocol acceptance must be an object")
    _exact(acceptance.get("status"), "ACCEPTED", where="protocol acceptance status")
    _exact(acceptance.get("frozen"), True, where="protocol acceptance frozen")
    _exact(acceptance.get("protocol_id"), STAGE_S_PROTOCOL_ID, where="protocol acceptance protocol_id")
    protocol_git_commit = _full_sha(
        _required(acceptance, "protocol_git_commit", where="protocol acceptance"),
        where="protocol acceptance.protocol_git_commit",
        length=40,
    )
    protocol_md_path = Path(str(_required(acceptance, "protocol_md_path", where="protocol acceptance"))).resolve()
    expected_md_path = (acceptance_path.parent / "PROTOCOL.md").resolve()
    if protocol_md_path != expected_md_path:
        raise FrozenProtocolError(
            f"protocol_md_path must be adjacent to FROZEN_PROTOCOL.json: expected {expected_md_path}, got {protocol_md_path}"
        )
    if protocol_md_path.is_symlink() or not protocol_md_path.is_file():
        raise FrozenProtocolError(f"frozen PROTOCOL.md is missing or symlinked: {protocol_md_path}")
    protocol_md_sha256 = _full_sha(
        _required(acceptance, "protocol_md_sha256", where="protocol acceptance"),
        where="protocol acceptance.protocol_md_sha256",
        length=64,
    )
    if _sha256(protocol_md_path) != protocol_md_sha256:
        raise FrozenProtocolError("frozen PROTOCOL.md SHA-256 mismatch")
    summary = _protocol_summary(acceptance)
    # S4/S5 are part of the same frozen object.  Read them from the envelope
    # when present, otherwise from the rich frozen_summary emitted by
    # freeze_protocol; never invent a grid or branch budget in this loader.
    s4 = payload.get("s4", payload.get("S4"))
    s5 = payload.get("s5", payload.get("S5"))
    if not isinstance(s4, Mapping):
        s4 = summary.get("s4", summary.get("S4"))
    if not isinstance(s5, Mapping):
        s5 = summary.get("s5", summary.get("S5"))
    if not isinstance(s4, Mapping) or not isinstance(s5, Mapping):
        raise FrozenProtocolError("frozen protocol acceptance lacks explicit S4/S5 contract")
    expected_s4 = summary.get("s4", summary.get("S4"))
    expected_s5 = summary.get("s5", summary.get("S5"))
    if not isinstance(expected_s4, Mapping) or not isinstance(expected_s5, Mapping):
        raise FrozenProtocolError("frozen protocol summary lacks explicit S4/S5 contract")
    if dict(s4) != dict(expected_s4) or dict(s5) != dict(expected_s5):
        raise FrozenProtocolError("frozen protocol envelope and summary S4/S5 contracts drifted")
    search_grid = s4.get("search_t_grid", s4.get("oracle_search_grid", s4.get("oracle_t_grid")))
    if not isinstance(search_grid, (list, tuple)) or len(search_grid) != 9:
        raise FrozenProtocolError("frozen protocol S4 search grid must contain exactly nine points")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in search_grid):
        raise FrozenProtocolError("frozen protocol S4 search grid must contain integer points")
    for field, expected in (("branch_count", 4), ("search_branch_count", 4), ("heldout_branch_count", 8), ("random_branch_count", 8), ("paired_bootstrap_replicates", 10000), ("paired_bootstrap_seed", 14211)):
        try:
            observed = int(s4.get(field, -1))
        except (TypeError, ValueError) as exc:
            raise FrozenProtocolError(f"frozen protocol S4 {field} is not an integer") from exc
        if observed != expected:
            raise FrozenProtocolError(f"frozen protocol S4 {field} drifted")
    if int(s5.get("base_candidate_count", -1)) != 32 or list(s5.get("fresh_candidate_indices", ())) != list(range(32, 64)):
        raise FrozenProtocolError("frozen protocol S5 candidate budget drifted")
    entries = _calibration_entries(acceptance)
    current_calibration = _verify_calibration_binding(
        entries,
        substrate=substrate,
        calibration_report=Path(calibration_report),
        freeze_report=freeze_report,
    )
    return {
        "schema": PROTOCOL_ACCEPTANCE_SCHEMA,
        "status": "FROZEN",
        "protocol_id": STAGE_S_PROTOCOL_ID,
        "protocol_acceptance_path": str(acceptance_path),
        "protocol_acceptance_sha256": _sha256(acceptance_path),
        "protocol_git_commit": protocol_git_commit,
        "protocol_md_path": str(protocol_md_path),
        "protocol_md_sha256": protocol_md_sha256,
        "frozen_summary": json.loads(json.dumps(summary)),
        "s4": json.loads(json.dumps(s4)),
        "s5": json.loads(json.dumps(s5)),
        "calibration_reports": json.loads(json.dumps(entries)),
        "calibration_binding": current_calibration,
    }


__all__ = [
    "DEFAULT_PROTOCOL_ACCEPTANCE_PATH",
    "FROZEN_SUMMARY",
    "FrozenProtocolError",
    "PROTOCOL_ACCEPTANCE_SCHEMA",
    "SEED_PLAN",
    "STAGE_S_PROTOCOL_ID",
    "THRESHOLDS",
    "read_frozen_protocol",
]
