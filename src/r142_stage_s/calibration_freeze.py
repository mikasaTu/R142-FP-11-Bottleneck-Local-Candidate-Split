"""Fail-closed Stage-S B/C calibration freeze and protocol acceptance.

The calibration screen is deliberately a small, auditable boundary between
the terminal pooled-success calibration and the B/C main screen.  This module
does not run an environment, inspect a rollout, submit PAI, or infer a result
from a partial run.  It consumes only terminal aggregate rows and immutable
artifact metadata, then writes the two selected-artifact reports and the
protocol acceptance object atomically.

The result rows are intentionally restricted to ``setting`` (or checkpoint
label), ``successes``, ``total``, and ``pooled_success``.  In particular,
S2--S5 fields are rejected before selection so that a caller cannot select a
setting after looking at downstream family/divergence/recovery outcomes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


CALIBRATION_FREEZE_SCHEMA = "r142-stage-s-calibration-freeze-v1"
CALIBRATION_RESULT_SCHEMA = "r142-stage-s-calibration-result-v1"
PROTOCOL_ACCEPTANCE_SCHEMA = "r142-stage-s-protocol-acceptance-v1"
STAGE_S_PROTOCOL_ID = "r142-stage-s-v1"
C_TRAINING_ACCEPTANCE_SCHEMA = "r142-stage-s-c-training-acceptance-v1"
C_TRAINING_COMPLETION_SCHEMA = "r142-stage-s-c-training-completion-v1"
CALIBRATION_TARGET = 0.45
CALIBRATION_SEED = 142042
CALIBRATION_WORLD_SIZE = 8
CALIBRATION_TOTAL_PER_SETTING = 4 * 8 * 8
B_VARIANT_RUN_ID = "r142-stage-s-b-variants-20260903-r7"
B_SETTINGS = (
    "proximity_0.06m",
    "proximity_0.08m",
    "proximity_0.10m",
    "proximity_0.12m",
)
C_STEPS = (1000, 3000, 6000, 10000)
C_SETTINGS = tuple(f"step_{step}" for step in C_STEPS)
_SOURCE_COMMIT_KEYS = ("stage_s_commit", "qpilots_commit", "openpi_commit", "libero_commit")

# These are the exact constants consumed by the B/C main-screen acceptance
# reader.  Keep this copy dependency-free: the freeze utility must work in a
# clean process before the heavy simulator/model environment is imported.
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
    "blocks_ranking_size",
    "pick_diverse_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_can_basket",
    "place_fan",
    "place_object_scale",
    "place_shoe",
)
MAIN_BUDGET = {
    "task_count": 10,
    "families_per_task": 16,
    "candidates_per_family": 32,
    "terminal_episode_count": 5120,
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
    "task_ids": list(range(10)),
    "initial_state_ids": list(range(16)),
    "candidate_ids": list(range(32)),
    "initial_state_count": 16,
    "candidate_budget": 32,
    "world_size": CALIBRATION_WORLD_SIZE,
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


class CalibrationFreezeError(ValueError):
    """A terminal calibration or protocol acceptance contract failed."""


# Exact key matching prevents false positives on ordinary metadata while
# catching the fields that would leak S2--S5 or post-calibration outcomes.
_FORBIDDEN_RESULT_KEYS = frozenset(
    {
        "s2",
        "s3",
        "s4",
        "s5",
        "near_fail",
        "near_all_fail",
        "nearallfail",
        "rho",
        "divergence",
        "t_div",
        "tdiv",
        "overdispersion",
        "recovery",
        "recover",
        "rescue",
        "family",
        "genealogy",
        "trajectory",
        "actions",
        "poses",
    }
)
_FORBIDDEN_RESULT_SUBSTRINGS = ("near_fail", "nearallfail", "diverg", "t_div", "tdiv", "overdisp", "recover", "rho")
_ALLOWED_FREEZE_METADATA_KEYS = frozenset({"no_s2_s5_peeking"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _full_sha(value: object, *, where: str, length: int = 64) -> str:
    if not isinstance(value, str) or re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is None:
        raise CalibrationFreezeError(f"{where} must be a lowercase full SHA-{length * 4}")
    return value


def _validate_source_commits(value: object, *, where: str) -> dict[str, str]:
    """Validate source provenance supplied by terminal acceptance evidence.

    Source pins are deliberately not hard-coded here: the accepted training
    manifest is the authority, and the calibration config/protocol must bind
    to the same four full commit ids before the protocol is frozen.
    """

    if not isinstance(value, Mapping) or set(value) != set(_SOURCE_COMMIT_KEYS):
        raise CalibrationFreezeError(f"{where} must contain exactly the four source commit fields")
    return {
        key: _full_sha(value[key], where=f"{where}.{key}", length=40)
        for key in _SOURCE_COMMIT_KEYS
    }


def _setting_order_key(setting: object, *, substrate: str) -> int | str:
    if substrate == "C":
        if not isinstance(setting, str):
            raise CalibrationFreezeError("C calibration setting must be a step label")
        match = re.fullmatch(r"step_([0-9]+)", setting)
        if match is None:
            raise CalibrationFreezeError(f"invalid C calibration setting: {setting}")
        return int(match.group(1))
    return str(setting)


def _select_calibration_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    substrate: str,
    target: float = CALIBRATION_TARGET,
) -> Mapping[str, Any]:
    """Select by pooled-success distance with a substrate-stable tie-break.

    C checkpoints are ordered by their numeric training step.  Comparing the
    labels lexicographically would select step_10000 before step_3000 on an
    exact distance tie, so both validation and protocol reproduction call
    this one helper.
    """

    if not rows:
        raise CalibrationFreezeError("cannot select from empty calibration rows")
    return min(
        rows,
        key=lambda row: (
            abs(float(row["pooled_success"]) - target),
            _setting_order_key(row.get("setting"), substrate=substrate),
        ),
    )


def _path(value: str | Path, *, label: str, directory: bool | None = None) -> Path:
    if not isinstance(value, (str, Path)) or not value:
        raise CalibrationFreezeError(f"{label} must be a non-empty path")
    candidate = Path(value).expanduser()
    if candidate.is_symlink():
        raise CalibrationFreezeError(f"{label} is symlinked: {candidate}")
    target = candidate.resolve()
    if target.is_symlink():
        raise CalibrationFreezeError(f"{label} is symlinked: {target}")
    if directory is True and not target.is_dir():
        raise CalibrationFreezeError(f"{label} is not a directory: {target}")
    if directory is False and not target.is_file():
        raise CalibrationFreezeError(f"{label} is not a regular file: {target}")
    return target


def _read_json(path: str | Path, *, label: str) -> tuple[Path, dict[str, Any]]:
    target = _path(path, label=label, directory=False)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CalibrationFreezeError(f"{label} is invalid JSON: {target}") from exc
    if not isinstance(payload, dict):
        raise CalibrationFreezeError(f"{label} must be a JSON object: {target}")
    return target, payload


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"


def _atomic_text(path: Path, text: str) -> None:
    path = path.expanduser()
    if path.is_symlink():
        raise CalibrationFreezeError(f"refusing to replace symlink: {path}")
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(path, _canonical_json(payload))


def _reject_result_leakage(value: object, *, where: str = "CALIBRATION_RESULT.json") -> None:
    """Reject downstream fields recursively, before reading any outcome row."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = re.sub(r"[^a-z0-9_]", "_", str(key).lower())
            if normalized not in _ALLOWED_FREEZE_METADATA_KEYS and (
                normalized in _FORBIDDEN_RESULT_KEYS
                or re.search(r"s[2-5]", normalized) is not None
                or any(token in normalized for token in _FORBIDDEN_RESULT_SUBSTRINGS)
            ):
                raise CalibrationFreezeError(f"forbidden post-calibration field at {where}.{key}")
            _reject_result_leakage(child, where=f"{where}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_result_leakage(child, where=f"{where}[{index}]")


def _manifest_rows(path: Path) -> list[tuple[str, str]]:
    if path.is_symlink() or not path.is_file():
        raise CalibrationFreezeError(f"missing SHA manifest: {path}")
    rows: list[tuple[str, str]] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or re.fullmatch(r"[0-9a-f]{64}", parts[0]) is None:
            raise CalibrationFreezeError(f"malformed SHA manifest line {path}:{number}")
        relative = parts[1].lstrip(" *")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts or not relative:
            raise CalibrationFreezeError(f"unsafe SHA manifest path {path}:{number}")
        rows.append((parts[0], relative_path.as_posix()))
    if not rows:
        raise CalibrationFreezeError(f"empty SHA manifest: {path}")
    return rows


def _verify_manifest(path: Path, *, root: Path) -> None:
    for expected, relative in _manifest_rows(path):
        candidate = root / relative
        if candidate.is_symlink():
            raise CalibrationFreezeError(f"SHA manifest member is symlinked: {candidate}")
        member = candidate.resolve()
        try:
            member.relative_to(root.resolve())
        except ValueError as exc:
            raise CalibrationFreezeError(f"SHA manifest escapes root: {path}:{relative}") from exc
        if member.is_symlink() or not member.is_file():
            raise CalibrationFreezeError(f"SHA manifest member missing/symlinked: {member}")
        if _sha256(member) != expected:
            raise CalibrationFreezeError(f"SHA manifest digest mismatch: {member}")


def _verify_result_manifest(result: Path) -> None:
    sums = result.with_name("SHA256SUMS")
    expected_line = f"{_sha256(result)}  {result.name}"
    if sums.is_symlink() or not sums.is_file() or sums.read_text(encoding="utf-8").splitlines() != [expected_line]:
        raise CalibrationFreezeError(f"calibration result SHA256SUMS mismatch: {sums}")


def _artifact_sha256(path: Path) -> str:
    """Match the B/C main loader: file bytes or a directory's own manifest."""

    if path.is_symlink() or not path.exists():
        raise CalibrationFreezeError(f"selected artifact is missing/symlinked: {path}")
    if path.is_file():
        return _sha256(path)
    if path.is_dir():
        manifest = path / "SHA256SUMS"
        _verify_manifest(manifest, root=path)
        return _sha256(manifest)
    raise CalibrationFreezeError(f"selected artifact is not a file/directory: {path}")


_TERMINAL_PAI_STATUSES = frozenset({"Succeeded", "Success", "JobSucceeded"})


def _sha256_sidecar(path: Path, sha_path: Path, *, label: str) -> str:
    """Verify a sha256sum-style sidecar for an external evidence file."""

    rows = sha_path.read_text(encoding="utf-8").splitlines()
    if len(rows) != 1:
        raise CalibrationFreezeError(f"{label} SHA sidecar must contain exactly one line")
    parts = rows[0].strip().split(maxsplit=1)
    if len(parts) != 2 or re.fullmatch(r"[0-9a-f]{64}", parts[0]) is None:
        raise CalibrationFreezeError(f"{label} SHA sidecar is malformed")
    declared_name = parts[1].lstrip(" *")
    if Path(declared_name).name != path.name:
        raise CalibrationFreezeError(f"{label} SHA sidecar names the wrong file")
    observed = _sha256(path)
    if observed != parts[0]:
        raise CalibrationFreezeError(f"{label} SHA sidecar digest mismatch")
    return observed


def _external_file_binding(path: Path, *, label: str) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha256(path)}


def _validate_external_pai_binding(
    *,
    registry_run: str | Path,
    jobs_ledger: str | Path,
    getjob_terminal: str | Path,
    getjob_terminal_sha: str | Path,
    artifact_dir: Path,
    application_run_id: object,
    substrate: str,
) -> dict[str, Any]:
    """Validate the external controller/job/artifact chain for B or C.

    The freeze utility deliberately accepts no inferred JobId.  The registry
    result, submission state, resolved payload, unique ledger row, terminal
    GetJob response, and its SHA sidecar are all required inputs.  For a
    restarted B controller, a controller-incarnation marker additionally
    binds the controller run to the application run; C's same-run identity is
    checked explicitly.
    """

    registry_root = _path(registry_run, label=f"{substrate} PAI registry run", directory=True)
    if not isinstance(application_run_id, str) or not application_run_id:
        raise CalibrationFreezeError(f"{substrate} completion marker lacks application run id")
    if registry_root.name != registry_root.name.strip() or not registry_root.name:
        raise CalibrationFreezeError(f"{substrate} PAI registry run directory has no stable name")

    result_path, result = _read_json(registry_root / "result.json", label=f"{substrate} registry result")
    state_path, state = _read_json(
        registry_root / "submission-state.json",
        label=f"{substrate} registry submission state",
    )
    resolved_path, resolved = _read_json(
        registry_root / "resolved.json",
        label=f"{substrate} registry resolved payload",
    )
    controller_run_id = result.get("run_id")
    job_id = result.get("job_id")
    if not isinstance(controller_run_id, str) or not controller_run_id:
        raise CalibrationFreezeError(f"{substrate} registry result lacks controller run id")
    if registry_root.name != controller_run_id:
        raise CalibrationFreezeError(f"{substrate} registry directory name does not match controller run id")
    if not isinstance(job_id, str) or re.fullmatch(r"dlc[0-9a-z]+", job_id) is None:
        raise CalibrationFreezeError(f"{substrate} registry result lacks a valid JobId")
    if result.get("submission_state") != "submitted_verified":
        raise CalibrationFreezeError(f"{substrate} registry result is not submitted_verified")
    if state.get("job_id") != job_id or state.get("state") != result.get("submission_state"):
        raise CalibrationFreezeError(f"{substrate} submission state does not bind result/job")
    if resolved.get("run_id") != controller_run_id:
        raise CalibrationFreezeError(f"{substrate} resolved payload run id mismatch")

    runtime = resolved.get("runtime")
    write_paths = runtime.get("write_paths") if isinstance(runtime, Mapping) else None
    if not isinstance(write_paths, list) or not any(
        isinstance(value, str) and Path(value).expanduser().resolve() == artifact_dir.resolve()
        for value in write_paths
    ):
        raise CalibrationFreezeError(
            f"{substrate} resolved payload does not bind the calibration artifact directory"
        )

    ledger_path = _path(jobs_ledger, label=f"{substrate} PAI jobs ledger", directory=False)
    matching: list[dict[str, Any]] = []
    for number, raw in enumerate(ledger_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            entry = json.loads(raw)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise CalibrationFreezeError(f"{substrate} jobs ledger line {number} is invalid JSON") from exc
        if not isinstance(entry, Mapping):
            raise CalibrationFreezeError(f"{substrate} jobs ledger line {number} is not an object")
        if entry.get("run_id") == controller_run_id:
            matching.append(dict(entry))
    if len(matching) != 1:
        raise CalibrationFreezeError(
            f"{substrate} jobs ledger must contain exactly one controller run binding"
        )
    if matching[0].get("job_id") != job_id:
        raise CalibrationFreezeError(f"{substrate} jobs ledger JobId disagrees with registry result")

    getjob_path = _path(
        getjob_terminal,
        label=f"{substrate} terminal GetJob response",
        directory=False,
    )
    getjob_sha_path = _path(
        getjob_terminal_sha,
        label=f"{substrate} terminal GetJob SHA sidecar",
        directory=False,
    )
    getjob_sha = _sha256_sidecar(
        getjob_path,
        getjob_sha_path,
        label=f"{substrate} terminal GetJob",
    )
    _, getjob = _read_json(getjob_path, label=f"{substrate} terminal GetJob response")
    observed_job_id = getjob.get("JobId", getjob.get("job_id"))
    observed_status = getjob.get("Status", getjob.get("status"))
    reason_code = getjob.get("ReasonCode", getjob.get("reason_code"))
    if observed_job_id != job_id or observed_status not in _TERMINAL_PAI_STATUSES:
        raise CalibrationFreezeError(
            f"{substrate} terminal GetJob is not a successful terminal response for the bound JobId"
        )
    if reason_code is not None and reason_code not in {"JobSucceeded", "Succeeded", "Success"}:
        raise CalibrationFreezeError(f"{substrate} terminal GetJob reason is not successful")

    # A B controller may have restarted under a new application run id.  The
    # incarnation record is the explicit controller->application edge.  C
    # intentionally uses one run id for both roles and must prove that
    # identity rather than silently accepting an unrelated artifact.
    if controller_run_id != application_run_id:
        incarnation = artifact_dir / "controller-incarnations" / f"{controller_run_id}.json"
        incarnation_path, incarnation_payload = _read_json(
            incarnation,
            label=f"{substrate} controller incarnation",
        )
        if (
            incarnation_payload.get("controller_run_id") != controller_run_id
            or incarnation_payload.get("application_run_id") != application_run_id
        ):
            raise CalibrationFreezeError(f"{substrate} controller incarnation does not bind application run")
    else:
        incarnation_path = None

    binding: dict[str, Any] = {
        "schema": "r142-stage-s-pai-registry-binding-v1",
        "substrate": substrate,
        "controller_run_id": controller_run_id,
        "application_run_id": application_run_id,
        "job_id": job_id,
        "terminal_getjob_status": observed_status,
        "registry_run": _external_file_binding(registry_root / "result.json", label="registry result"),
        "submission_state": _external_file_binding(
            registry_root / "submission-state.json",
            label="registry submission state",
        ),
        "resolved": _external_file_binding(
            registry_root / "resolved.json",
            label="registry resolved payload",
        ),
        "jobs_ledger": _external_file_binding(ledger_path, label="jobs ledger"),
        "terminal_getjob": {
            "path": str(getjob_path),
            "sha256": getjob_sha,
            "sha_sidecar": _external_file_binding(
                getjob_sha_path,
                label="terminal GetJob SHA sidecar",
            ),
        },
    }
    if incarnation_path is not None:
        binding["controller_incarnation"] = _external_file_binding(
            incarnation_path,
            label="controller incarnation",
        )
    else:
        binding["controller_incarnation"] = {
            "identity": "controller_run_id_equals_application_run_id"
        }
    return binding


def _verify_external_binding_from_report(
    binding: Mapping[str, Any],
    *,
    artifact_dir: Path,
    application_run_id: object,
    substrate: str,
) -> dict[str, Any]:
    """Re-read and hash-check the evidence referenced by a frozen report."""

    if binding.get("schema") != "r142-stage-s-pai-registry-binding-v1":
        raise CalibrationFreezeError(f"{substrate} external PAI binding schema mismatch")
    if binding.get("substrate") != substrate:
        raise CalibrationFreezeError(f"{substrate} external PAI binding substrate mismatch")
    controller_run_id = binding.get("controller_run_id")
    job_id = binding.get("job_id")
    if binding.get("application_run_id") != application_run_id:
        raise CalibrationFreezeError(f"{substrate} external application run binding mismatch")
    if not isinstance(controller_run_id, str) or not isinstance(job_id, str):
        raise CalibrationFreezeError(f"{substrate} external PAI binding ids are missing")
    expected_files = {
        "registry_run": "result.json",
        "submission_state": "submission-state.json",
        "resolved": "resolved.json",
        "jobs_ledger": "jobs.jsonl",
    }
    bound_paths: dict[str, Path] = {}
    for key, expected_name in expected_files.items():
        entry = binding.get(key)
        if not isinstance(entry, Mapping):
            raise CalibrationFreezeError(f"{substrate} external PAI binding lacks {key}")
        file_path = _path(entry.get("path", ""), label=f"{substrate} binding {key}", directory=False)
        if file_path.name != expected_name:
            raise CalibrationFreezeError(f"{substrate} external binding {key} names the wrong evidence file")
        if entry.get("sha256") != _sha256(file_path):
            raise CalibrationFreezeError(f"{substrate} external binding {key} SHA mismatch")
        bound_paths[key] = file_path
    getjob = binding.get("terminal_getjob")
    if not isinstance(getjob, Mapping):
        raise CalibrationFreezeError(f"{substrate} external PAI binding lacks terminal GetJob")
    getjob_path = _path(
        getjob.get("path", ""),
        label=f"{substrate} bound terminal GetJob",
        directory=False,
    )
    getjob_sha_path = _path(
        getjob.get("sha_sidecar", {}).get("path", "")
        if isinstance(getjob.get("sha_sidecar"), Mapping)
        else "",
        label=f"{substrate} bound terminal GetJob SHA sidecar",
        directory=False,
    )
    getjob_sha_sidecar = getjob.get("sha_sidecar")
    if not isinstance(getjob_sha_sidecar, Mapping):
        raise CalibrationFreezeError(f"{substrate} bound terminal GetJob SHA sidecar is missing")
    if getjob.get("sha256") != _sha256(getjob_path):
        raise CalibrationFreezeError(f"{substrate} bound terminal GetJob SHA mismatch")
    if getjob_sha_sidecar.get("sha256") != _sha256(getjob_sha_path):
        raise CalibrationFreezeError(f"{substrate} bound GetJob SHA sidecar binding mismatch")
    _sha256_sidecar(getjob_path, getjob_sha_path, label=f"{substrate} bound terminal GetJob")
    _, getjob_payload = _read_json(getjob_path, label=f"{substrate} bound terminal GetJob")
    observed_job_id = getjob_payload.get("JobId", getjob_payload.get("job_id"))
    observed_status = getjob_payload.get("Status", getjob_payload.get("status"))
    reason_code = getjob_payload.get("ReasonCode", getjob_payload.get("reason_code"))
    if observed_job_id != job_id:
        raise CalibrationFreezeError(f"{substrate} bound terminal GetJob JobId mismatch")
    if observed_status not in _TERMINAL_PAI_STATUSES:
        raise CalibrationFreezeError(f"{substrate} bound terminal GetJob status mismatch")
    if reason_code is not None and reason_code not in {"JobSucceeded", "Succeeded", "Success"}:
        raise CalibrationFreezeError(f"{substrate} bound terminal GetJob reason mismatch")

    result_path, result = _read_json(
        bound_paths["registry_run"],
        label=f"{substrate} bound registry result",
    )
    state_path, state = _read_json(
        bound_paths["submission_state"],
        label=f"{substrate} bound registry submission state",
    )
    resolved_path, resolved = _read_json(
        bound_paths["resolved"],
        label=f"{substrate} bound registry resolved payload",
    )
    if (
        result.get("run_id") != controller_run_id
        or result.get("job_id") != job_id
        or result.get("submission_state") != "submitted_verified"
    ):
        raise CalibrationFreezeError(f"{substrate} bound registry result ids/state drifted")
    if state.get("job_id") != job_id or state.get("state") != "submitted_verified":
        raise CalibrationFreezeError(f"{substrate} bound submission state drifted")
    if resolved.get("run_id") != controller_run_id:
        raise CalibrationFreezeError(f"{substrate} bound resolved run id drifted")
    ledger_path = bound_paths["jobs_ledger"]
    matching = []
    for number, raw in enumerate(ledger_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            entry = json.loads(raw)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise CalibrationFreezeError(
                f"{substrate} bound jobs ledger line {number} is invalid JSON"
            ) from exc
        if not isinstance(entry, Mapping):
            raise CalibrationFreezeError(f"{substrate} bound jobs ledger line {number} is not an object")
        if entry.get("run_id") == controller_run_id:
            matching.append(entry)
    if len(matching) != 1 or matching[0].get("job_id") != job_id:
        raise CalibrationFreezeError(f"{substrate} bound jobs ledger is not unique")
    runtime = resolved.get("runtime")
    write_paths = runtime.get("write_paths") if isinstance(runtime, Mapping) else None
    if not isinstance(write_paths, list) or not any(
        isinstance(value, str) and Path(value).expanduser().resolve() == artifact_dir.resolve()
        for value in write_paths
    ):
        raise CalibrationFreezeError(f"{substrate} bound resolved payload lost artifact binding")
    incarnation_binding = binding.get("controller_incarnation")
    if not isinstance(incarnation_binding, Mapping):
        raise CalibrationFreezeError(f"{substrate} bound controller incarnation is missing")
    if incarnation_binding.get("identity") != "controller_run_id_equals_application_run_id":
        incarnation = incarnation_binding
        incarnation_path = _path(
            incarnation.get("path", ""),
            label=f"{substrate} bound controller incarnation",
            directory=False,
        )
        if incarnation.get("sha256") != _sha256(incarnation_path):
            raise CalibrationFreezeError(f"{substrate} bound controller incarnation SHA mismatch")
        _, incarnation_payload = _read_json(
            incarnation_path,
            label=f"{substrate} bound controller incarnation",
        )
        if (
            incarnation_payload.get("controller_run_id") != controller_run_id
            or incarnation_payload.get("application_run_id") != application_run_id
        ):
            raise CalibrationFreezeError(f"{substrate} bound controller incarnation drifted")
    elif controller_run_id != application_run_id:
        raise CalibrationFreezeError(f"{substrate} same-run controller identity is false")
    return dict(binding)


def _validate_c_calibration_config(
    config_path: str | Path,
    *,
    expected_source: Mapping[str, str],
) -> tuple[Path, dict[str, str]]:
    path, payload = _read_json(config_path, label="C calibration config")
    evidence = payload.get("evidence")
    if not isinstance(evidence, Mapping):
        raise CalibrationFreezeError("C calibration config lacks evidence source binding")
    source = _validate_source_commits(
        {key: evidence.get(key) for key in _SOURCE_COMMIT_KEYS},
        where="C calibration config source",
    )
    if source != dict(expected_source):
        raise CalibrationFreezeError(
            "C calibration config source does not match accepted training source"
        )
    return path, source


def _validate_rows(payload: Mapping[str, Any], *, settings: Sequence[str], substrate: str) -> list[dict[str, Any]]:
    _reject_result_leakage(payload)
    allowed_payload = {
        "schema",
        "protocol_id",
        "calibration_seed",
        "world_size",
        "rows",
        "target_pooled_success",
        "selected_setting",
    }
    if set(payload) != allowed_payload:
        raise CalibrationFreezeError(f"calibration result schema drift: fields={sorted(payload)}")
    if payload.get("schema") != CALIBRATION_RESULT_SCHEMA or payload.get("protocol_id") != STAGE_S_PROTOCOL_ID:
        raise CalibrationFreezeError("calibration result schema/protocol mismatch")
    if payload.get("calibration_seed") != CALIBRATION_SEED or payload.get("world_size") != CALIBRATION_WORLD_SIZE:
        raise CalibrationFreezeError("calibration result seed/world-size mismatch")
    if payload.get("target_pooled_success") != CALIBRATION_TARGET:
        raise CalibrationFreezeError("calibration result target must be exactly 0.45")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != len(settings):
        raise CalibrationFreezeError("calibration result must contain exactly four rows")
    validated: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {"setting", "successes", "total", "pooled_success"}:
            raise CalibrationFreezeError(f"calibration row {index} contains non-pooled fields")
        if row.get("setting") != settings[index]:
            raise CalibrationFreezeError(f"calibration row order/setting mismatch at index {index}")
        successes, total = row.get("successes"), row.get("total")
        if isinstance(successes, bool) or not isinstance(successes, int) or isinstance(total, bool) or not isinstance(total, int):
            raise CalibrationFreezeError(f"calibration row {index} counts must be integers")
        if total != CALIBRATION_TOTAL_PER_SETTING or successes < 0 or successes > total:
            raise CalibrationFreezeError(f"calibration row {index} total/count is invalid")
        pooled = row.get("pooled_success")
        if isinstance(pooled, bool) or not isinstance(pooled, (int, float)):
            raise CalibrationFreezeError(f"calibration row {index} pooled_success must be numeric")
        expected = successes / total
        if abs(float(pooled) - expected) > 1e-12:
            raise CalibrationFreezeError(f"calibration row {index} pooled_success is not successes/total")
        validated.append(
            {
                "setting": str(row["setting"]),
                "successes": int(successes),
                "total": int(total),
                "pooled_success": float(pooled),
            }
        )
    selected = _select_calibration_row(validated, substrate=substrate)
    if payload.get("selected_setting") != selected["setting"]:
        raise CalibrationFreezeError("calibration result selected_setting is not the deterministic choice")
    return validated


def _completion_bundle_manifest(marker: Path, payload: Mapping[str, Any], *, substrate: str) -> Path:
    persistence = payload.get("persistence")
    declared = persistence.get("bundle_sha_file") if isinstance(persistence, Mapping) else None
    name = str(declared or ("B_SHA256SUMS" if substrate == "B" else "C_SHA256SUMS"))
    if Path(name).name != name:
        raise CalibrationFreezeError(f"unsafe completion bundle manifest name: {name}")
    return marker.parent / name


def _validate_rank_markers(marker: Path, payload: Mapping[str, Any], *, substrate: str) -> None:
    """Require and hash-bind all eight terminal worker completion markers."""

    ranks = payload.get("rank_markers")
    if (
        not isinstance(ranks, list)
        or len(ranks) != CALIBRATION_WORLD_SIZE
        or len(set(ranks)) != CALIBRATION_WORLD_SIZE
    ):
        raise CalibrationFreezeError(
            f"{substrate} completion marker does not enumerate all eight rank markers"
        )
    rank_digests = payload.get("rank_marker_sha256")
    if not isinstance(rank_digests, Mapping) or set(rank_digests) != set(ranks):
        raise CalibrationFreezeError(
            f"{substrate} completion marker rank SHA bindings are incomplete"
        )
    for relative in ranks:
        if not isinstance(relative, str):
            raise CalibrationFreezeError(f"{substrate} rank marker path is not a string")
        relative_path = PurePosixPath(relative)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative != relative_path.as_posix()
        ):
            raise CalibrationFreezeError(f"unsafe {substrate} rank marker path: {relative}")
        expected = _full_sha(
            rank_digests[relative],
            where=f"{substrate} rank marker {relative}",
            length=64,
        )
        rank_path = marker.parent.joinpath(*relative_path.parts)
        rank_path = _path(rank_path, label=f"{substrate} rank marker {relative}", directory=False)
        if _sha256(rank_path) != expected:
            raise CalibrationFreezeError(f"{substrate} rank marker SHA mismatch: {rank_path}")


def _validate_completion(marker: Path, *, result: Path, result_sha: str, substrate: str) -> dict[str, Any]:
    payload = _read_json(marker, label=f"{substrate} completion marker")[1]
    expected_schema = (
        "r142-stage-s-b-calibration-completion-v1"
        if substrate == "B"
        else "r142-stage-s-c-calibration-completion-v1"
    )
    if payload.get("schema") != expected_schema or payload.get("status") != "COMPLETED":
        raise CalibrationFreezeError(f"{substrate} completion marker is not a supported terminal marker")
    if payload.get("protocol_id") != STAGE_S_PROTOCOL_ID or payload.get("substrate") != substrate:
        raise CalibrationFreezeError(f"{substrate} completion marker protocol/substrate mismatch")
    failure_marker = marker.parent / f"FAILED_{substrate}_CALIBRATION.json"
    if failure_marker.exists() or failure_marker.is_symlink():
        raise CalibrationFreezeError(
            f"residual FAILED_{substrate}_CALIBRATION.json is present beside the terminal marker"
        )
    result_name = str(payload.get("calibration_result", payload.get("result_file", "")))
    if Path(result_name).name != result.name:
        raise CalibrationFreezeError(f"{substrate} completion marker result filename mismatch")
    declared_sha = payload.get("calibration_result_sha256", payload.get("result_sha256"))
    if declared_sha != result_sha:
        raise CalibrationFreezeError(f"{substrate} completion marker result SHA mismatch")
    if payload.get("calibration_result_schema", CALIBRATION_RESULT_SCHEMA) != CALIBRATION_RESULT_SCHEMA:
        raise CalibrationFreezeError(f"{substrate} completion marker result schema mismatch")
    if payload.get("calibration_seed") != CALIBRATION_SEED or payload.get("world_size") != CALIBRATION_WORLD_SIZE:
        raise CalibrationFreezeError(f"{substrate} completion marker seed/world-size mismatch")
    bundle = _completion_bundle_manifest(marker, payload, substrate=substrate)
    _verify_manifest(bundle, root=marker.parent)
    _validate_rank_markers(marker, payload, substrate=substrate)
    if substrate == "B":
        if payload.get("input_bundle_run_id") != B_VARIANT_RUN_ID:
            raise CalibrationFreezeError("B completion marker is not the frozen r7 variant bundle")
        provenance = payload.get("provenance")
        if not isinstance(provenance, Mapping) or not provenance.get("variant_root"):
            raise CalibrationFreezeError("B completion marker lacks the frozen variant_root provenance")
        if provenance.get("old_init_reused") is True:
            raise CalibrationFreezeError("B completion marker permits old init-state reuse")
    return payload


def _validate_b_variant(marker_payload: Mapping[str, Any], *, selected: str) -> tuple[Path, str]:
    provenance = marker_payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise CalibrationFreezeError("B completion marker provenance is missing")
    base = _path(str(provenance.get("variant_root", "")), label="B variant_root", directory=True)
    selected_root = _path(base / selected, label="selected B variant", directory=True)
    config = _path(selected_root / "config.yaml", label="selected B variant config", directory=False)
    return selected_root, _sha256(config)


def _read_calibration_source(
    result_path: str | Path,
    marker_path: str | Path,
    *,
    substrate: str,
) -> tuple[Path, Path, dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    result, payload = _read_json(result_path, label=f"{substrate} CALIBRATION_RESULT.json")
    marker, _ = _read_json(marker_path, label=f"{substrate} completion marker")
    if result.parent != marker.parent:
        raise CalibrationFreezeError(
            f"{substrate} result and completion marker must be in the same directory"
        )
    settings = B_SETTINGS if substrate == "B" else C_SETTINGS
    rows = _validate_rows(payload, settings=settings, substrate=substrate)
    _verify_result_manifest(result)
    result_sha = _sha256(result)
    marker_payload = _validate_completion(marker, result=result, result_sha=result_sha, substrate=substrate)
    return result, marker, payload, marker_payload, rows


def _lineage_reference(
    lineage_file: Path,
    value: object,
    *,
    label: str,
    directory: bool | None = None,
) -> Path:
    """Resolve a path from a lineage object without following an input symlink."""

    if not isinstance(value, (str, Path)) or not value:
        raise CalibrationFreezeError(f"C lineage lacks {label}")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = lineage_file.parent / candidate
    return _path(candidate, label=label, directory=directory)


def _validate_c_training_acceptance(
    lineage_file: Path,
    payload: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], list[Mapping[str, Any]], dict[str, str]]:
    """Normalize the current C ``ACCEPTED_C_TRAINING.json`` contract.

    The training launcher publishes one acceptance object, rather than the
    older wrapper-shaped lineage accepted by ``_lineage_completion``.  The
    object binds terminal training, the full checkpoint/log manifests, and
    the four native model hashes.  We verify both the manifest contents and
    the per-checkpoint hash map here; the latter is what lets the freeze
    report select a concrete file while preserving the common C main-loader
    schema.
    """

    _reject_result_leakage(payload, where=f"{lineage_file}")
    if payload.get("schema") != C_TRAINING_ACCEPTANCE_SCHEMA:
        raise CalibrationFreezeError("C accepted training lineage schema mismatch")
    if (
        payload.get("status") != "ACCEPTED"
        or payload.get("label") != "WEAK_SUBSTRATE"
        or payload.get("pai_terminal_status") != "Succeeded"
    ):
        raise CalibrationFreezeError("C accepted training lineage is not an accepted weak-substrate terminal result")
    accepted_run_id = payload.get("accepted_run_id")
    if not isinstance(accepted_run_id, str) or re.fullmatch(
        r"r142-stage-s-c-undertrained-20[0-9]{6}-r[0-9]+", accepted_run_id
    ) is None:
        raise CalibrationFreezeError("C accepted training lineage run id mismatch")
    job_id = payload.get("job_id")
    if not isinstance(job_id, str) or re.fullmatch(r"dlc[0-9a-z]+", job_id) is None:
        raise CalibrationFreezeError("C accepted training lineage job id mismatch")

    source = _validate_source_commits(
        payload.get("source"),
        where="C accepted training lineage source",
    )

    checkpoint_steps = payload.get("checkpoint_steps")
    if checkpoint_steps != list(C_STEPS):
        raise CalibrationFreezeError("C accepted training checkpoint schedule mismatch")
    if payload.get("full_reference_step") != 30000:
        raise CalibrationFreezeError("C accepted training full reference step mismatch")
    if payload.get("no_interpolation") is not True or payload.get("artificial_degradation") is not False:
        raise CalibrationFreezeError("C accepted training lineage permits interpolation or artificial degradation")

    checkpoint_root = _lineage_reference(
        lineage_file,
        payload.get("checkpoint_root"),
        label="C checkpoint_root",
        directory=True,
    )
    completion_path = _lineage_reference(
        lineage_file,
        payload.get("checkpoint_completion"),
        label="C checkpoint_completion",
        directory=False,
    )
    checkpoint_manifest = _lineage_reference(
        lineage_file,
        payload.get("checkpoint_sha256_manifest"),
        label="C checkpoint SHA256SUMS",
        directory=False,
    )
    log_root = _lineage_reference(
        lineage_file,
        payload.get("log_root"),
        label="C log_root",
        directory=True,
    )
    log_manifest = _lineage_reference(
        lineage_file,
        payload.get("log_sha256_manifest"),
        label="C log SHA256SUMS",
        directory=False,
    )
    pipeline_path = _lineage_reference(
        lineage_file,
        payload.get("training_pipeline_completion"),
        label="C training_pipeline_completion",
        directory=False,
    )

    for child, parent, label in (
        (completion_path, checkpoint_root, "C checkpoint completion"),
        (checkpoint_manifest, checkpoint_root, "C checkpoint SHA256SUMS"),
        (log_manifest, log_root, "C log SHA256SUMS"),
    ):
        try:
            child.relative_to(parent)
        except ValueError as exc:
            raise CalibrationFreezeError(f"{label} escapes its declared root") from exc

    expected_digests = {
        "checkpoint_completion_sha256": completion_path,
        "checkpoint_sha256_manifest_digest": checkpoint_manifest,
        "log_sha256_manifest_digest": log_manifest,
    }
    for field, artifact in expected_digests.items():
        expected = _full_sha(payload.get(field), where=f"C accepted training {field}")
        if _sha256(artifact) != expected:
            raise CalibrationFreezeError(f"C accepted training {field} mismatch")

    # The acceptance object itself is the immutable source of the four
    # selected-file bindings.  Require exactly the four frozen relative
    # names, and require the same names/hashes to occur in the checkpoint
    # bundle manifest.  This catches both a moved checkpoint and a stale
    # acceptance JSON even when the JSON's own digest is externally signed.
    expected_paths = {f"{step}/model.safetensors" for step in C_STEPS}
    checkpoint_hashes = payload.get("checkpoint_hashes")
    if not isinstance(checkpoint_hashes, Mapping) or set(checkpoint_hashes) != expected_paths:
        raise CalibrationFreezeError("C accepted training must carry exactly four checkpoint model hashes")
    normalized_hashes: dict[str, str] = {}
    for relative_name in sorted(expected_paths):
        if not isinstance(relative_name, str):  # defensive: keys came from JSON
            raise CalibrationFreezeError("C accepted checkpoint hash path is not a string")
        relative = PurePosixPath(relative_name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative_name != relative.as_posix()
        ):
            raise CalibrationFreezeError(f"C accepted checkpoint hash path is unsafe: {relative_name}")
        normalized_hashes[relative_name] = _full_sha(
            checkpoint_hashes[relative_name],
            where=f"C accepted checkpoint hash {relative_name}",
        )

    manifest_rows = _manifest_rows(checkpoint_manifest)
    manifest_by_path: dict[str, str] = {}
    for expected, relative in manifest_rows:
        if relative in manifest_by_path:
            raise CalibrationFreezeError(f"duplicate C checkpoint SHA manifest path: {relative}")
        manifest_by_path[relative] = expected
    for relative_name, expected in normalized_hashes.items():
        if manifest_by_path.get(relative_name) != expected:
            raise CalibrationFreezeError(
                f"C checkpoint SHA manifest does not bind {relative_name} to its accepted hash"
            )

    # Recheck the complete checkpoint and log bundles.  This is intentionally
    # fail-closed and may be I/O-heavy for a real model bundle, but it is the
    # only way to ensure a terminal marker has not outlived a mutated file.
    _verify_manifest(checkpoint_manifest, root=checkpoint_root)
    _verify_manifest(log_manifest, root=log_root)

    by_step: list[Mapping[str, Any]] = []
    for step in C_STEPS:
        relative_name = f"{step}/model.safetensors"
        checkpoint = _lineage_reference(
            lineage_file,
            checkpoint_root / Path(*PurePosixPath(relative_name).parts),
            label=f"C checkpoint {step}",
            directory=False,
        )
        if _sha256(checkpoint) != normalized_hashes[relative_name]:
            raise CalibrationFreezeError(f"C checkpoint {step} artifact hash mismatch")
        by_step.append({"step": step, "path": str(checkpoint), "sha256": normalized_hashes[relative_name]})

    completion_path, completion = _read_json(completion_path, label="C training completion marker")
    if completion.get("schema") != C_TRAINING_COMPLETION_SCHEMA or completion.get("status") != "COMPLETED":
        raise CalibrationFreezeError("C accepted training completion is not COMPLETED")
    if completion.get("openpi_commit") != source["openpi_commit"]:
        raise CalibrationFreezeError("C training completion OpenPI commit does not match accepted source")
    if completion.get("config_name") != "pi05_libero" or completion.get("seed") != 42:
        raise CalibrationFreezeError("C accepted training completion config/seed mismatch")
    if completion.get("terminal_global_step") != 10001 or completion.get("checkpoint_steps") != list(C_STEPS):
        raise CalibrationFreezeError("C accepted training completion terminal/checkpoint schedule mismatch")
    audit = completion.get("checkpoint_audit")
    if not isinstance(audit, Mapping) or audit.get("valid") is not True:
        raise CalibrationFreezeError("C accepted training completion checkpoint audit is not valid")

    pipeline_path, pipeline = _read_json(pipeline_path, label="C training pipeline completion")
    if pipeline.get("status") != "COMPLETED" or pipeline.get("stage") != "terminal":
        raise CalibrationFreezeError("C training pipeline marker is not terminal COMPLETED")
    if pipeline.get("run_id") != accepted_run_id:
        raise CalibrationFreezeError("C training pipeline run id mismatch")
    evidence_value = pipeline.get("evidence_path")
    if not isinstance(evidence_value, str) or not evidence_value:
        raise CalibrationFreezeError("C training pipeline marker lacks evidence_path")
    evidence_candidate = Path(evidence_value).expanduser()
    if not evidence_candidate.is_absolute():
        evidence_candidate = pipeline_path.parent / evidence_candidate
    evidence_path = _path(evidence_candidate, label="C training pipeline evidence", directory=False)
    if evidence_path != completion_path:
        raise CalibrationFreezeError("C training pipeline marker does not bind COMPLETED_C_TRAINING")
    if pipeline.get("evidence_sha256") != _sha256(completion_path):
        raise CalibrationFreezeError("C training pipeline evidence SHA mismatch")

    return completion_path, completion, by_step, source


def _lineage_completion(
    path: Path,
    payload: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], list[Mapping[str, Any]], dict[str, str]]:
    """Normalize only the current accepted C training contract.

    Older direct-completion and wrapper-shaped inputs did not bind source
    provenance to the accepted training manifest.  They are intentionally
    rejected rather than silently upgraded.
    """

    if payload.get("schema") != C_TRAINING_ACCEPTANCE_SCHEMA:
        raise CalibrationFreezeError(
            "C lineage must use the current accepted training manifest schema"
        )
    return _validate_c_training_acceptance(path, payload)


def _checkpoint_lineage(
    lineage_path: str | Path,
) -> tuple[Path, str, dict[int, tuple[Path, str]], dict[str, str]]:
    lineage_file, lineage = _read_json(lineage_path, label="C accepted training lineage")
    completion_path, completion, entries, source = _lineage_completion(lineage_file, lineage)
    if completion.get("status") != "COMPLETED" or completion.get("schema") != C_TRAINING_COMPLETION_SCHEMA:
        raise CalibrationFreezeError("C lineage completion is not accepted terminal training")
    if completion.get("openpi_commit") != source["openpi_commit"]:
        raise CalibrationFreezeError("C lineage OpenPI commit does not match accepted source")
    if completion.get("config_name") != "pi05_libero" or completion.get("seed") != 42:
        raise CalibrationFreezeError("C lineage config/seed mismatch")
    if completion.get("terminal_global_step") != 10001 or completion.get("checkpoint_steps") != list(C_STEPS):
        raise CalibrationFreezeError("C lineage terminal/checkpoint schedule mismatch")
    audit = completion.get("checkpoint_audit")
    if isinstance(audit, Mapping) and audit.get("valid") is not True:
        raise CalibrationFreezeError("C lineage checkpoint audit is not valid")

    # Verify the completion's persisted checkpoint manifest when declared.  A
    # caller may provide a wrapper with an explicit manifest path instead.
    manifest_value = completion.get("sha256sums") or lineage.get("checkpoint_manifest")
    manifest_sha = completion.get("sha256sums_sha256") or lineage.get("checkpoint_manifest_sha256")
    if manifest_value:
        manifest = _path(str(manifest_value), label="C checkpoint SHA256SUMS", directory=False)
        if manifest_sha is not None and manifest_sha != _sha256(manifest):
            raise CalibrationFreezeError("C checkpoint manifest SHA mismatch")
        _verify_manifest(manifest, root=manifest.parent)

    by_step: dict[int, tuple[Path, str]] = {}
    for entry in entries:
        raw_step = entry.get("step", entry.get("expected_step"))
        raw_path = entry.get("path", entry.get("checkpoint"))
        if isinstance(raw_step, bool) or not isinstance(raw_step, int) or raw_step not in C_STEPS:
            continue
        if not isinstance(raw_path, str) or not raw_path:
            raise CalibrationFreezeError(f"C lineage checkpoint {raw_step} lacks path")
        checkpoint_candidate = Path(raw_path).expanduser()
        if not checkpoint_candidate.is_absolute():
            checkpoint_candidate = completion_path.parent / checkpoint_candidate
        checkpoint = _path(checkpoint_candidate, label=f"C checkpoint {raw_step}")
        observed_sha = _artifact_sha256(checkpoint)
        declared_sha = entry.get("sha256", entry.get("artifact_sha256"))
        if declared_sha is not None and declared_sha != observed_sha:
            raise CalibrationFreezeError(f"C checkpoint {raw_step} artifact SHA mismatch")
        by_step[raw_step] = (checkpoint, observed_sha)
    if tuple(sorted(by_step)) != C_STEPS:
        raise CalibrationFreezeError("C lineage must enumerate all four exact checkpoints")
    return completion_path, _sha256(lineage_file), by_step, source


def freeze_calibration_reports(
    *,
    b_result: str | Path,
    b_completion_marker: str | Path,
    c_result: str | Path,
    c_completion_marker: str | Path,
    c_lineage: str | Path,
    c_config: str | Path,
    b_registry_run: str | Path,
    b_jobs_ledger: str | Path,
    b_getjob_terminal: str | Path,
    b_getjob_terminal_sha: str | Path,
    c_registry_run: str | Path,
    c_jobs_ledger: str | Path,
    c_getjob_terminal: str | Path,
    c_getjob_terminal_sha: str | Path,
    b_report: str | Path,
    c_report: str | Path,
) -> dict[str, dict[str, Any]]:
    """Validate terminal B/C inputs and atomically write compatible reports."""

    b_result_path, b_marker_path, b_payload, b_marker, b_rows = _read_calibration_source(
        b_result, b_completion_marker, substrate="B"
    )
    c_result_path, c_marker_path, c_payload, c_marker, c_rows = _read_calibration_source(
        c_result, c_completion_marker, substrate="C"
    )
    b_selected = _select_calibration_row(b_rows, substrate="B")
    c_selected = _select_calibration_row(c_rows, substrate="C")
    selected_variant, selected_variant_sha = _validate_b_variant(b_marker, selected=str(b_selected["setting"]))
    completion_path, lineage_sha, checkpoints, training_source = _checkpoint_lineage(c_lineage)
    if c_marker.get("source") != training_source:
        raise CalibrationFreezeError(
            "C calibration completion source does not match accepted training source"
        )
    c_config_path, c_config_source = _validate_c_calibration_config(
        c_config,
        expected_source=training_source,
    )
    selected_step = int(str(c_selected["setting"]).split("_", 1)[1])
    b_external = _validate_external_pai_binding(
        registry_run=b_registry_run,
        jobs_ledger=b_jobs_ledger,
        getjob_terminal=b_getjob_terminal,
        getjob_terminal_sha=b_getjob_terminal_sha,
        artifact_dir=b_marker_path.parent,
        application_run_id=b_marker.get("run_id"),
        substrate="B",
    )
    c_external = _validate_external_pai_binding(
        registry_run=c_registry_run,
        jobs_ledger=c_jobs_ledger,
        getjob_terminal=c_getjob_terminal,
        getjob_terminal_sha=c_getjob_terminal_sha,
        artifact_dir=c_marker_path.parent,
        application_run_id=c_marker.get("run_id"),
        substrate="C",
    )
    selected_checkpoint, selected_checkpoint_sha = checkpoints[selected_step]

    common = {
        "schema": CALIBRATION_FREEZE_SCHEMA,
        "status": "FROZEN",
        "protocol_id": STAGE_S_PROTOCOL_ID,
        "calibration_completed": True,
        "frozen": True,
        "no_s2_s5_peeking": True,
        "calibration_seed": CALIBRATION_SEED,
        "calibration_world_size": CALIBRATION_WORLD_SIZE,
        "selection_rule": "minimize abs(pooled_success - 0.45); tie-break is declared per substrate",
    }
    b_report_payload = {
        **common,
        "substrate": "B",
        "calibration_kind": "pooled_only",
        "source_result_path": str(b_result_path),
        "source_completion_marker": str(b_marker_path),
        "source_result_sha256": _sha256(b_result_path),
        "source_completion_marker_sha256": _sha256(b_marker_path),
        "calibration_rows": b_rows,
        "external_pai_binding": b_external,
        "selection_tie_break": "lexicographic_setting",
        "selected_setting": b_selected["setting"],
        "selected_pooled_success": b_selected["pooled_success"],
        "variant_run_id": B_VARIANT_RUN_ID,
        "selected_variant_root": str(selected_variant),
        "selected_variant_root_sha256": selected_variant_sha,
    }
    c_report_payload = {
        **common,
        "substrate": "C",
        "calibration_kind": "checkpoint_calibration",
        "source_result_path": str(c_result_path),
        "source_completion_marker": str(c_marker_path),
        "source_result_sha256": _sha256(c_result_path),
        "source_completion_marker_sha256": _sha256(c_marker_path),
        "calibration_rows": c_rows,
        "selection_tie_break": "ascending_checkpoint_step",
        "selected_setting": c_selected["setting"],
        "selected_pooled_success": c_selected["pooled_success"],
        "selected_checkpoint_step": selected_step,
        "selected_checkpoint": str(selected_checkpoint),
        "selected_checkpoint_sha256": selected_checkpoint_sha,
        "accepted_training_completion": str(completion_path),
        "accepted_training_lineage_path": str(Path(c_lineage).expanduser().resolve()),
        "accepted_training_lineage_sha256": lineage_sha,
        "training_source": training_source,
        "c_calibration_config_path": str(c_config_path),
        "c_calibration_config_sha256": _sha256(c_config_path),
        "c_calibration_config_source": c_config_source,
        "external_pai_binding": c_external,
    }

    # Validate all selected artifacts before touching either destination.
    _path(b_report, label="B report destination", directory=False) if Path(b_report).exists() else None
    _path(c_report, label="C report destination", directory=False) if Path(c_report).exists() else None
    _atomic_json(Path(b_report), b_report_payload)
    _atomic_json(Path(c_report), c_report_payload)
    return {"B": b_report_payload, "C": c_report_payload}


def _protocol_requirement_errors(text: str) -> list[str]:
    lowered = text.lower()
    errors: list[str] = []
    def required(label: str, *needles: str) -> None:
        if not all(needle in lowered for needle in needles):
            errors.append(label)

    required("S1 thresholds", "s1", "0.30", "0.60")
    required("S2 thresholds", "s2", "0.10", "3.0", "20")
    if not all(needle in lowered for needle in ("s3", "0.10", "0.25")) or not any(
        needle in lowered for needle in ("0.95", "95th", "95th percentile")
    ):
        errors.append("S3 thresholds")
    required("S4 thresholds", "s4", "0.30", "10000")
    required("S5 threshold", "s5", "0.05", "64")
    required("D(t) normalization", "d(t)")
    if not all(needle in lowered for needle in ("tau", "matched-t")) or not any(
        needle in lowered for needle in ("same-task", "same task")
    ) or not any(needle in lowered for needle in ("95th", "0.95")):
        errors.append("tau same-task matched-t 95th")
    if not all(needle in lowered for needle in ("family", "task", "initial")):
        errors.append("family definition")
    if not any(needle in lowered for needle in ("near-all-fail", "near_all_fail", "near all fail")):
        errors.append("near-all-fail definition")
    required("RNG literal contract", "rng", "seed")
    required("compute literal contract", "compute", "policy_forward_pass", "environment_step")
    # Permit the common Unicode <= spelling and require that the two
    # branching definitions are actually operational, not just named.
    if not any(token in lowered for token in ("1/32", "<= 1", "≤ 1", "at most one")):
        errors.append("near-all-fail operational definition")
    if not any(token in lowered for token in ("normalize", "normalization", "normalized")):
        errors.append("D(t) normalization wording")
    return errors


def _git_commit_exists(repo_root: Path, commit: str) -> None:
    if not repo_root.is_dir():
        raise CalibrationFreezeError(f"protocol repo root is missing: {repo_root}")
    result = subprocess.run(
        ["git", "-C", str(repo_root), "cat-file", "-e", f"{commit}^{{commit}}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise CalibrationFreezeError(f"protocol GitHub commit is not present locally: {commit}")


def _git_protocol_blob_matches(repo_root: Path, commit: str, source_md: Path) -> None:
    """Bind the authority to the exact protocol bytes stored by Git.

    Requiring a commit to be written inside the file that creates that commit
    is self-referential and cannot be satisfied honestly.  The immutable
    content binding is instead the Git object lookup itself.
    """

    result = subprocess.run(
        ["git", "-C", str(repo_root), "show", f"{commit}:stage-s/PROTOCOL.md"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise CalibrationFreezeError("protocol Git commit does not contain stage-s/PROTOCOL.md")
    if result.stdout != source_md.read_bytes():
        raise CalibrationFreezeError("working PROTOCOL.md differs from the declared Git commit")


def _validate_report_for_protocol(path: Path, *, substrate: str) -> dict[str, Any]:
    report_path, report = _read_json(path, label=f"{substrate} calibration freeze report")
    required = {
        "schema": CALIBRATION_FREEZE_SCHEMA,
        "status": "FROZEN",
        "protocol_id": STAGE_S_PROTOCOL_ID,
        "substrate": substrate,
        "calibration_completed": True,
        "frozen": True,
        "no_s2_s5_peeking": True,
    }
    kind = "pooled_only" if substrate == "B" else "checkpoint_calibration"
    required["calibration_kind"] = kind
    for key, expected in required.items():
        if report.get(key) != expected:
            raise CalibrationFreezeError(f"{substrate} report field {key} mismatch")
    source_result, source_marker, payload, marker, rows = _read_calibration_source(
        report["source_result_path"], report["source_completion_marker"], substrate=substrate
    )
    if report.get("source_result_sha256") != _sha256(source_result) or report.get("source_completion_marker_sha256") != _sha256(source_marker):
        raise CalibrationFreezeError(f"{substrate} report source SHA mismatch")
    expected = _select_calibration_row(rows, substrate=substrate)
    if report.get("selected_setting") != expected["setting"]:
        raise CalibrationFreezeError(f"{substrate} report selection is not reproducible")
    if substrate == "B":
        if report.get("variant_run_id") != B_VARIANT_RUN_ID:
            raise CalibrationFreezeError("B report variant run is not frozen r7")
        selected, digest = _validate_b_variant(marker, selected=str(expected["setting"]))
        if report.get("selected_variant_root") != str(selected) or report.get("selected_variant_root_sha256") != digest:
            raise CalibrationFreezeError("B report selected variant binding mismatch")
    else:
        lineage_reference = report.get("accepted_training_lineage_path", report["accepted_training_completion"])
        _, lineage_sha, checkpoints, training_source = _checkpoint_lineage(lineage_reference)
        if report.get("accepted_training_lineage_sha256") != lineage_sha:
            raise CalibrationFreezeError("C report accepted training lineage SHA mismatch")
        if report.get("training_source") != training_source:
            raise CalibrationFreezeError("C report training source binding mismatch")
        config_path, config_source = _validate_c_calibration_config(
            report.get("c_calibration_config_path", ""),
            expected_source=training_source,
        )
        if (
            report.get("c_calibration_config_sha256") != _sha256(config_path)
            or report.get("c_calibration_config_source") != config_source
        ):
            raise CalibrationFreezeError("C report calibration config binding mismatch")
        if marker.get("source") != training_source:
            raise CalibrationFreezeError("C marker source does not match accepted training source")
        step = int(str(expected["setting"]).split("_", 1)[1])
        selected, digest = checkpoints[step]
        if report.get("selected_checkpoint") != str(selected) or report.get("selected_checkpoint_sha256") != digest:
            raise CalibrationFreezeError("C report selected checkpoint binding mismatch")
    # Report metadata itself must not carry post-calibration fields.  This is
    # intentionally after source validation so only the permitted source rows
    # were consumed for selection.
    _reject_result_leakage(report, where=f"{report_path}")
    return report


def freeze_protocol(
    *,
    protocol_md: str | Path,
    protocol_git_commit: str,
    b_report: str | Path,
    c_report: str | Path,
    output_path: str | Path,
    repo_root: str | Path,
    materialize_protocol_md: bool = True,
) -> dict[str, Any]:
    """Freeze protocol acceptance after reports and markdown pass every gate."""

    source_md = _path(protocol_md, label="stage-s/PROTOCOL.md", directory=False)
    if source_md.name != "PROTOCOL.md":
        raise CalibrationFreezeError("protocol source must be named stage-s/PROTOCOL.md")
    if re.fullmatch(r"[0-9a-f]{40}", protocol_git_commit) is None:
        raise CalibrationFreezeError("protocol_git_commit must be a lowercase full 40-hex commit")
    repository = _path(repo_root, label="protocol repo root", directory=True)
    _git_commit_exists(repository, protocol_git_commit)
    _git_protocol_blob_matches(repository, protocol_git_commit, source_md)
    protocol_text = source_md.read_text(encoding="utf-8")
    missing = _protocol_requirement_errors(protocol_text)
    if missing:
        raise CalibrationFreezeError("PROTOCOL.md is missing frozen requirements: " + ", ".join(missing))
    b_path = _path(b_report, label="B calibration report", directory=False)
    c_path = _path(c_report, label="C calibration report", directory=False)
    b_payload = _validate_report_for_protocol(b_path, substrate="B")
    c_payload = _validate_report_for_protocol(c_path, substrate="C")
    source_provenance = c_payload.get("training_source")
    _validate_source_commits(source_provenance, where="C report training source")
    missing_source = [
        f"C {key}={value}"
        for key, value in source_provenance.items()
        if value not in protocol_text
    ]
    if missing_source:
        raise CalibrationFreezeError(
            "PROTOCOL.md does not bind the accepted C source commits: "
            + ", ".join(missing_source)
        )

    destination = Path(output_path).expanduser().resolve()
    if destination.name != "FROZEN_PROTOCOL.json":
        raise CalibrationFreezeError("protocol acceptance filename must be FROZEN_PROTOCOL.json")
    if destination.exists() or destination.is_symlink():
        raise CalibrationFreezeError(
            f"refusing to overwrite existing frozen protocol acceptance: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if materialize_protocol_md:
        # The main loader requires a non-symlink PROTOCOL.md beside the stable
        # acceptance object.  This is a runtime copy, not a repo commit.
        _atomic_text(destination.parent / "PROTOCOL.md", protocol_text)
    adjacent_md = destination.parent / "PROTOCOL.md"
    if not adjacent_md.is_file() or adjacent_md.is_symlink():
        raise CalibrationFreezeError("protocol acceptance requires adjacent non-symlink PROTOCOL.md")
    if adjacent_md.read_text(encoding="utf-8") != protocol_text:
        raise CalibrationFreezeError("adjacent PROTOCOL.md differs from committed stage-s/PROTOCOL.md")
    md_sha = _sha256(adjacent_md)
    acceptance = {
        "status": "ACCEPTED",
        "frozen": True,
        "protocol_id": STAGE_S_PROTOCOL_ID,
        "protocol_git_commit": protocol_git_commit,
        "protocol_md_path": str(adjacent_md),
        "protocol_md_sha256": md_sha,
        "frozen_summary": FROZEN_SUMMARY,
        "calibration_reports": {
            "B": {
                "report_path": str(b_path),
                "report_sha256": _sha256(b_path),
                "selected_setting": b_payload["selected_setting"],
                "variant_run_id": b_payload["variant_run_id"],
            },
            "C": {
                "report_path": str(c_path),
                "report_sha256": _sha256(c_path),
                "selected_checkpoint": c_payload["selected_checkpoint"],
                "selected_checkpoint_sha256": c_payload["selected_checkpoint_sha256"],
            },
        },
        "source_provenance": {
            "C": c_payload["training_source"],
            "C_calibration_config": {
                "path": c_payload["c_calibration_config_path"],
                "sha256": c_payload["c_calibration_config_sha256"],
            },
        },
    }
    payload = {
        "schema": PROTOCOL_ACCEPTANCE_SCHEMA,
        "status": "FROZEN",
        "protocol_id": STAGE_S_PROTOCOL_ID,
        "protocol_git_commit": protocol_git_commit,
        "protocol_md_path": str(adjacent_md),
        "protocol_md_sha256": md_sha,
        "frozen_summary": FROZEN_SUMMARY,
        "s4": S4_PROTOCOL,
        "s5": S5_PROTOCOL,
        "source_provenance": {
            "C": c_payload["training_source"],
            "C_calibration_config": {
                "path": c_payload["c_calibration_config_path"],
                "sha256": c_payload["c_calibration_config_sha256"],
            },
        },
        "files": {
            "PROTOCOL.md": {"path": str(adjacent_md), "sha256": md_sha},
            "B_CALIBRATION_REPORT": {"path": str(b_path), "sha256": _sha256(b_path)},
            "C_CALIBRATION_REPORT": {"path": str(c_path), "sha256": _sha256(c_path)},
        },
        "acceptance": acceptance,
    }
    _atomic_json(destination, payload)
    return payload


def read_frozen_protocol(
    path: str | Path,
    *,
    substrate: str,
    calibration_report: str | Path,
) -> dict[str, Any]:
    """Validate the same acceptance object consumed by the B/C main loader."""

    acceptance_path, payload = _read_json(path, label="frozen protocol acceptance")
    if payload.get("schema") != PROTOCOL_ACCEPTANCE_SCHEMA or payload.get("status") != "FROZEN" or payload.get("protocol_id") != STAGE_S_PROTOCOL_ID:
        raise CalibrationFreezeError("frozen protocol envelope mismatch")
    acceptance = payload.get("acceptance")
    if not isinstance(acceptance, Mapping) or acceptance.get("status") != "ACCEPTED" or acceptance.get("frozen") is not True:
        raise CalibrationFreezeError("frozen protocol acceptance status mismatch")
    commit = _full_sha(acceptance.get("protocol_git_commit"), where="protocol_git_commit", length=40)
    md = _path(acceptance_path.parent / "PROTOCOL.md", label="adjacent PROTOCOL.md", directory=False)
    if acceptance.get("protocol_md_path") != str(md) or acceptance.get("protocol_md_sha256") != _sha256(md):
        raise CalibrationFreezeError("frozen protocol markdown binding mismatch")
    if acceptance.get("frozen_summary") != FROZEN_SUMMARY:
        raise CalibrationFreezeError("frozen protocol summary drift")
    if payload.get("s4") != S4_PROTOCOL or payload.get("s5") != S5_PROTOCOL:
        raise CalibrationFreezeError("frozen protocol S4/S5 contract drift")
    reports = acceptance.get("calibration_reports")
    if not isinstance(reports, Mapping) or not isinstance(reports.get(substrate), Mapping):
        raise CalibrationFreezeError(f"frozen protocol lacks {substrate} calibration report")
    report_entry = reports[substrate]
    report_path = _path(calibration_report, label=f"{substrate} calibration report", directory=False)
    if report_entry.get("report_path") != str(report_path) or report_entry.get("report_sha256") != _sha256(report_path):
        raise CalibrationFreezeError(f"{substrate} report acceptance binding mismatch")
    report = _validate_report_for_protocol(report_path, substrate=substrate)
    if substrate == "C":
        expected_provenance = {
            "C": report["training_source"],
            "C_calibration_config": {
                "path": report["c_calibration_config_path"],
                "sha256": report["c_calibration_config_sha256"],
            },
        }
        if (
            payload.get("source_provenance") != expected_provenance
            or acceptance.get("source_provenance") != expected_provenance
        ):
            raise CalibrationFreezeError("frozen protocol C source provenance drifted")
    if substrate == "B":
        if report_entry.get("selected_setting") != report.get("selected_setting") or report_entry.get("variant_run_id") != report.get("variant_run_id"):
            raise CalibrationFreezeError("B report selection acceptance mismatch")
    else:
        if report_entry.get("selected_checkpoint") != report.get("selected_checkpoint") or report_entry.get("selected_checkpoint_sha256") != report.get("selected_checkpoint_sha256"):
            raise CalibrationFreezeError("C report selection acceptance mismatch")
    return {
        "schema": PROTOCOL_ACCEPTANCE_SCHEMA,
        "status": "FROZEN",
        "protocol_id": STAGE_S_PROTOCOL_ID,
        "protocol_acceptance_path": str(acceptance_path),
        "protocol_acceptance_sha256": _sha256(acceptance_path),
        "protocol_git_commit": commit,
        "protocol_md_path": str(md),
        "protocol_md_sha256": _sha256(md),
        "frozen_summary": json.loads(json.dumps(acceptance["frozen_summary"])),
        "s4": json.loads(json.dumps(payload["s4"])),
        "s5": json.loads(json.dumps(payload["s5"])),
        "calibration_reports": json.loads(json.dumps(reports)),
    }


# Short aliases keep the public API easy to discover for launchers while the
# descriptive names remain the canonical implementation entry points.
freeze_calibration = freeze_calibration_reports
protocol_freeze = freeze_protocol


__all__ = [
    "B_SETTINGS",
    "B_VARIANT_RUN_ID",
    "C_SETTINGS",
    "C_STEPS",
    "CALIBRATION_FREEZE_SCHEMA",
    "CALIBRATION_RESULT_SCHEMA",
    "CALIBRATION_TARGET",
    "CalibrationFreezeError",
    "C_TRAINING_ACCEPTANCE_SCHEMA",
    "C_TRAINING_COMPLETION_SCHEMA",
    "FROZEN_SUMMARY",
    "PROTOCOL_ACCEPTANCE_SCHEMA",
    "SEED_PLAN",
    "STAGE_S_PROTOCOL_ID",
    "THRESHOLDS",
    "freeze_calibration_reports",
    "freeze_calibration",
    "freeze_protocol",
    "protocol_freeze",
    "read_frozen_protocol",
]
