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
from pathlib import Path
from typing import Any, Mapping, Sequence


CALIBRATION_FREEZE_SCHEMA = "r142-stage-s-calibration-freeze-v1"
CALIBRATION_RESULT_SCHEMA = "r142-stage-s-calibration-result-v1"
PROTOCOL_ACCEPTANCE_SCHEMA = "r142-stage-s-protocol-acceptance-v1"
STAGE_S_PROTOCOL_ID = "r142-stage-s-v1"
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
    "candidate": "sha256(r142-stage-s-v1|candidate|task_id|init_state|candidate_id)->first_8_bytes_big_endian",
    "environment": "sha256(r142-stage-s-v1|environment|task_id|init_state)->first_8_bytes_big_endian",
    "calibration": "sha256(r142-stage-s-v1|calibration|setting_index|task_id|init_state|candidate_id)->first_8_bytes_big_endian",
}
FROZEN_SUMMARY: dict[str, object] = {
    "task_ids": list(range(10)),
    "initial_state_ids": list(range(16)),
    "candidate_ids": list(range(32)),
    "initial_state_count": 16,
    "candidate_budget": 32,
    "world_size": CALIBRATION_WORLD_SIZE,
    "seed_plan": SEED_PLAN,
    "thresholds": THRESHOLDS,
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
_FORBIDDEN_RESULT_SUBSTRINGS = ("near_fail", "nearallfail", "diverg", "overdisp", "recover", "rho")
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


def _path(value: str | Path, *, label: str, directory: bool | None = None) -> Path:
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


def _validate_rows(payload: Mapping[str, Any], *, settings: Sequence[str]) -> list[dict[str, Any]]:
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
    selected = min(validated, key=lambda row: (abs(row["pooled_success"] - CALIBRATION_TARGET), row["setting"]))
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


def _validate_completion(marker: Path, *, result: Path, result_sha: str, substrate: str) -> dict[str, Any]:
    payload = _read_json(marker, label=f"{substrate} completion marker")[1]
    expected_schema = (
        "r142-stage-s-b-calibration-completion-v1"
        if substrate == "B"
        else "r142-stage-s-c-calibration-completion-v1"
    )
    # C calibration was not present in the original Stage-R tree; accepting
    # the generic spelling preserves compatibility with an equivalent pinned
    # C launcher while still checking every security-relevant field.
    schemas = {expected_schema}
    if substrate == "C":
        schemas.add("r142-stage-s-calibration-completion-v1")
    if payload.get("schema") not in schemas or payload.get("status") != "COMPLETED":
        raise CalibrationFreezeError(f"{substrate} completion marker is not a supported terminal marker")
    if payload.get("protocol_id") != STAGE_S_PROTOCOL_ID or payload.get("substrate") != substrate:
        raise CalibrationFreezeError(f"{substrate} completion marker protocol/substrate mismatch")
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
    if substrate == "B":
        if payload.get("input_bundle_run_id") != B_VARIANT_RUN_ID:
            raise CalibrationFreezeError("B completion marker is not the frozen r7 variant bundle")
        ranks = payload.get("rank_markers")
        if not isinstance(ranks, list) or len(ranks) != CALIBRATION_WORLD_SIZE or len(set(map(str, ranks))) != CALIBRATION_WORLD_SIZE:
            raise CalibrationFreezeError("B completion marker does not enumerate all eight rank markers")
        rank_digests = payload.get("rank_marker_sha256")
        if not isinstance(rank_digests, Mapping):
            raise CalibrationFreezeError("B completion marker lacks rank marker SHA bindings")
        for relative in ranks:
            if not isinstance(relative, str) or relative not in rank_digests:
                raise CalibrationFreezeError("B completion marker rank SHA bindings are incomplete")
            expected = _full_sha(rank_digests[relative], where=f"B rank marker {relative}")
            rank_path = marker.parent / relative
            rank_path = _path(rank_path, label=f"B rank marker {relative}", directory=False)
            if _sha256(rank_path) != expected:
                raise CalibrationFreezeError(f"B rank marker SHA mismatch: {rank_path}")
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
    settings = B_SETTINGS if substrate == "B" else C_SETTINGS
    rows = _validate_rows(payload, settings=settings)
    _verify_result_manifest(result)
    result_sha = _sha256(result)
    marker_payload = _validate_completion(marker, result=result, result_sha=result_sha, substrate=substrate)
    return result, marker, payload, marker_payload, rows


def _lineage_completion(path: Path, payload: Mapping[str, Any]) -> tuple[Path, dict[str, Any], list[Mapping[str, Any]]]:
    """Normalize either a direct C completion marker or an explicit wrapper."""

    if payload.get("schema") == "r142-stage-s-c-training-completion-v1":
        completion_path, completion = path, payload
        audit = completion.get("checkpoint_audit")
        entries = audit.get("checkpoints") if isinstance(audit, Mapping) else None
        if not isinstance(entries, list):
            raise CalibrationFreezeError("C training completion lacks checkpoint_audit.checkpoints")
        return completion_path, completion, [entry for entry in entries if isinstance(entry, Mapping)]
    completion_ref = (
        payload.get("training_completion")
        or payload.get("training_completion_path")
        or payload.get("completion_marker")
    )
    if isinstance(completion_ref, Mapping):
        completion_value = completion_ref.get("path") or completion_ref.get("file")
        expected_sha = completion_ref.get("sha256")
    else:
        completion_value = completion_ref
        expected_sha = payload.get("training_completion_sha256")
    if not isinstance(completion_value, str) or not completion_value:
        raise CalibrationFreezeError("C lineage must name an accepted training completion marker")
    completion_candidate = Path(completion_value).expanduser()
    if not completion_candidate.is_absolute():
        completion_candidate = path.parent / completion_candidate
    completion_path, completion = _read_json(completion_candidate, label="C training completion marker")
    observed = _sha256(completion_path)
    if expected_sha is not None and expected_sha != observed:
        raise CalibrationFreezeError("C training completion marker SHA mismatch")
    if payload.get("accepted") is False or payload.get("status") not in (None, "ACCEPTED", "COMPLETED"):
        raise CalibrationFreezeError("C training lineage wrapper is not accepted")
    entries = payload.get("checkpoints") or payload.get("checkpoint_artifacts")
    if not isinstance(entries, list):
        raise CalibrationFreezeError("C lineage wrapper must enumerate checkpoints")
    return completion_path, completion, [entry for entry in entries if isinstance(entry, Mapping)]


def _checkpoint_lineage(lineage_path: str | Path) -> tuple[Path, str, dict[int, tuple[Path, str]]]:
    lineage_file, lineage = _read_json(lineage_path, label="C accepted training lineage")
    completion_path, completion, entries = _lineage_completion(lineage_file, lineage)
    if completion.get("status") != "COMPLETED" or completion.get("schema") != "r142-stage-s-c-training-completion-v1":
        raise CalibrationFreezeError("C lineage completion is not accepted terminal training")
    if completion.get("openpi_commit") != "54cbaee6ae0c010a1ed431871cdaa8f4684ac709":
        raise CalibrationFreezeError("C lineage OpenPI commit mismatch")
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
    return completion_path, _sha256(lineage_file), by_step


def freeze_calibration_reports(
    *,
    b_result: str | Path,
    b_completion_marker: str | Path,
    c_result: str | Path,
    c_completion_marker: str | Path,
    c_lineage: str | Path,
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
    b_selected = min(b_rows, key=lambda row: (abs(row["pooled_success"] - CALIBRATION_TARGET), row["setting"]))
    c_selected = min(c_rows, key=lambda row: (abs(row["pooled_success"] - CALIBRATION_TARGET), int(str(row["setting"]).split("_", 1)[1])))
    selected_variant, selected_variant_sha = _validate_b_variant(b_marker, selected=str(b_selected["setting"]))
    completion_path, lineage_sha, checkpoints = _checkpoint_lineage(c_lineage)
    selected_step = int(str(c_selected["setting"]).split("_", 1)[1])
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
    expected = min(rows, key=lambda row: (abs(row["pooled_success"] - CALIBRATION_TARGET), row["setting"] if substrate == "B" else int(str(row["setting"]).split("_", 1)[1])))
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
        _, lineage_sha, checkpoints = _checkpoint_lineage(lineage_reference)
        if report.get("accepted_training_lineage_sha256") != lineage_sha:
            raise CalibrationFreezeError("C report accepted training lineage SHA mismatch")
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
    _git_commit_exists(_path(repo_root, label="protocol repo root", directory=True), protocol_git_commit)
    protocol_text = source_md.read_text(encoding="utf-8")
    missing = _protocol_requirement_errors(protocol_text)
    if missing:
        raise CalibrationFreezeError("PROTOCOL.md is missing frozen requirements: " + ", ".join(missing))
    b_path = _path(b_report, label="B calibration report", directory=False)
    c_path = _path(c_report, label="C calibration report", directory=False)
    b_payload = _validate_report_for_protocol(b_path, substrate="B")
    c_payload = _validate_report_for_protocol(c_path, substrate="C")

    destination = Path(output_path).expanduser().resolve()
    if destination.name != "FROZEN_PROTOCOL.json":
        raise CalibrationFreezeError("protocol acceptance filename must be FROZEN_PROTOCOL.json")
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
    }
    payload = {
        "schema": PROTOCOL_ACCEPTANCE_SCHEMA,
        "status": "FROZEN",
        "protocol_id": STAGE_S_PROTOCOL_ID,
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
    if commit not in md.read_text(encoding="utf-8"):
        raise CalibrationFreezeError("protocol Git commit is not recorded in adjacent PROTOCOL.md")
    if acceptance.get("frozen_summary") != FROZEN_SUMMARY:
        raise CalibrationFreezeError("frozen protocol summary drift")
    reports = acceptance.get("calibration_reports")
    if not isinstance(reports, Mapping) or not isinstance(reports.get(substrate), Mapping):
        raise CalibrationFreezeError(f"frozen protocol lacks {substrate} calibration report")
    report_entry = reports[substrate]
    report_path = _path(calibration_report, label=f"{substrate} calibration report", directory=False)
    if report_entry.get("report_path") != str(report_path) or report_entry.get("report_sha256") != _sha256(report_path):
        raise CalibrationFreezeError(f"{substrate} report acceptance binding mismatch")
    report = _validate_report_for_protocol(report_path, substrate=substrate)
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
