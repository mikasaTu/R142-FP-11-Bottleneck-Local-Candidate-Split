#!/usr/bin/env python3
"""Run the frozen Stage-S positive and null controls from persisted evidence.

This executor is deliberately an evidence reader, not a rollout generator. The
positive control consumes only the persisted B0_best_of_n success rates and
the null control consumes only the three authoritative Stage-R arrays required
by the protocol. The logical rows passed to S2 contain success labels only;
no trajectory, action, or other rollout field is synthesized or modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from r142_stage_s.analysis import compute_s2  # noqa: E402
from r142_stage_s.integrity import (  # noqa: E402
    sha256_bytes,
    sha256_file,
    verify_completion_bundle,
    write_completion,
)


POSITIVE_DEFAULT = _REPO_ROOT / "evidence" / "formal_pai" / "r3" / "shards"
NULL_DEFAULT = Path(
    "/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_r/phase0r_merged/"
    "r142-stage-r-phase0r-authoritative-20260827/raw"
)
OUTPUT_DEFAULT = _REPO_ROOT / "stage-s" / "results" / "controls"
POSITIVE_POLICY = "B0_best_of_n"
EXPECTED_TASK_FILES = 40
EXPECTED_CANDIDATES = 32
EXPECTED_FAMILIES_PER_TASK = 16
EXPECTED_ROLLOWS_PER_TASK = 512
_JSON_FLOAT_TOL = 1e-9


class ControlEvidenceError(ValueError):
    """Raised when a control source violates the frozen evidence contract."""


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one generated report atomically, without touching source evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _manifest_digest(entries: Sequence[Mapping[str, Any]]) -> str:
    encoded = json.dumps(
        list(entries), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return sha256_bytes(encoded.encode("utf-8"))


def _inventory_entry(path: Path, root: Path) -> dict[str, Any]:
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ControlEvidenceError(f"source file is outside source root: {path}") from exc
    return {
        "path": str(path),
        "relative_path": relative,
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _stable_source_inventory(
    paths: Sequence[Path], root: Path
) -> tuple[list[dict[str, Any]], str]:
    entries = [_inventory_entry(path, root) for path in sorted(paths)]
    return entries, _manifest_digest(entries)


def _assert_source_unchanged(path: Path, before: str) -> str:
    after = sha256_file(path)
    if after != before:
        raise ControlEvidenceError(f"source changed while reading: {path}")
    return after


def _exact_success_count(rate: Any, *, family: str) -> int:
    if isinstance(rate, bool):
        raise ControlEvidenceError(f"{family}: candidate_success_rate must be numeric")
    try:
        value = float(rate)
    except (TypeError, ValueError) as exc:
        raise ControlEvidenceError(
            f"{family}: invalid candidate_success_rate {rate!r}"
        ) from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ControlEvidenceError(f"{family}: candidate_success_rate outside [0, 1]")
    scaled = value * float(EXPECTED_CANDIDATES)
    nearest = int(round(scaled))
    if abs(scaled - nearest) > _JSON_FLOAT_TOL:
        raise ControlEvidenceError(
            f"{family}: rate {value!r} does not encode an exact "
            f"{EXPECTED_CANDIDATES}-candidate count"
        )
    if not 0 <= nearest <= EXPECTED_CANDIDATES:
        raise ControlEvidenceError(f"{family}: decoded success count outside candidate budget")
    return nearest


def load_positive(source_root: str | Path) -> dict[str, Any]:
    """Load B0 family success labels from persisted JSONL rates only.

    Each JSONL record is one family whose recorded rate must be an exact
    multiple of 1/32. The 32 logical rows are a statistical representation of
    that persisted count, not newly generated trajectories.
    """

    root = Path(source_root)
    if not root.is_dir():
        raise ControlEvidenceError(f"positive source root does not exist: {root}")
    paths = sorted(root.glob("*/episode_metrics.jsonl"))
    if not paths:
        raise ControlEvidenceError(f"no shard episode_metrics.jsonl files under {root}")

    rows: list[dict[str, Any]] = []
    success_counts: list[int] = []
    inventory: list[dict[str, Any]] = []
    seen_families: set[str] = set()
    b0_records = 0
    for path in paths:
        before = sha256_file(path)
        relative = path.relative_to(root).as_posix()
        line_count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                line_count += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ControlEvidenceError(
                        f"invalid JSON at {path}:{line_number}"
                    ) from exc
                if not isinstance(record, Mapping):
                    raise ControlEvidenceError(
                        f"JSON record at {path}:{line_number} is not an object"
                    )
                # Non-B0 records are intentionally ignored; no other policy
                # statistic can enter this control.
                if record.get("policy") != POSITIVE_POLICY:
                    continue
                b0_records += 1
                if "episode_id" not in record:
                    raise ControlEvidenceError(
                        f"B0 record lacks episode_id at {path}:{line_number}"
                    )
                if record.get("candidate_count") != EXPECTED_CANDIDATES:
                    raise ControlEvidenceError(
                        f"B0 record at {path}:{line_number} has candidate_count "
                        f"{record.get('candidate_count')!r}, expected {EXPECTED_CANDIDATES}"
                    )
                family_id = (
                    f"{relative}:line-{line_number}:episode-{record['episode_id']}"
                )
                if family_id in seen_families:
                    raise ControlEvidenceError(f"duplicate positive family {family_id}")
                seen_families.add(family_id)
                count = _exact_success_count(
                    record.get("candidate_success_rate"), family=family_id
                )
                success_counts.append(count)
                # S2 consumes only eventual success. Keep IDs deterministic,
                # but do not invent actions/poses or claim these are rollouts.
                rows.extend(
                    {"family_id": family_id, "success": candidate < count}
                    for candidate in range(EXPECTED_CANDIDATES)
                )
        after = _assert_source_unchanged(path, before)
        entry = _inventory_entry(path, root)
        if entry["sha256"] != after:
            raise ControlEvidenceError(f"source digest changed while indexing: {path}")
        entry["line_count"] = line_count
        inventory.append(entry)

    if not b0_records:
        raise ControlEvidenceError(f"no {POSITIVE_POLICY} records found under {root}")
    if len(rows) != b0_records * EXPECTED_CANDIDATES:
        raise ControlEvidenceError("positive logical-row count does not match B0 family count")
    counts = Counter(success_counts)
    direct_all_fail = int(sum(value == 0 for value in success_counts))
    direct_near = int(sum(value <= 1 for value in success_counts))
    return {
        "rows": rows,
        "source": {
            "path": str(root),
            "kind": "formal_pai_episode_metrics_jsonl",
            "policy_filter": POSITIVE_POLICY,
            "files": inventory,
            "source_manifest_sha256": _manifest_digest(inventory),
        },
        "counts": {
            "source_file_count": len(paths),
            "b0_record_count": b0_records,
            "logical_family_count": len(success_counts),
            "candidate_count_per_family": EXPECTED_CANDIDATES,
            "logical_candidate_row_count": len(rows),
            "all_fail_family_count": direct_all_fail,
            "near_all_fail_family_count": direct_near,
            "exact_success_count_histogram": {
                str(key): int(counts[key]) for key in sorted(counts)
            },
        },
    }


def _as_python_identifier(value: Any) -> Any:
    """Convert NumPy scalar IDs to JSON/string-friendly Python scalars."""

    return value.item() if isinstance(value, np.generic) else value


def _validate_null_arrays(
    path: Path,
    success: Any,
    init_state: Any,
    candidate_id: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate one authoritative 512-row task file and return S2-only rows."""

    success_array = np.asarray(success)
    init_array = np.asarray(init_state)
    candidate_array = np.asarray(candidate_id)
    arrays = {
        "success": success_array,
        "init_state": init_array,
        "candidate_id": candidate_array,
    }
    for name, array in arrays.items():
        if array.shape != (EXPECTED_ROLLOWS_PER_TASK,):
            raise ControlEvidenceError(
                f"{path.name}: {name} shape {array.shape}, expected "
                f"({EXPECTED_ROLLOWS_PER_TASK},)"
            )
    try:
        success_numeric = success_array.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ControlEvidenceError(f"{path.name}: success is not numeric") from exc
    if not np.all(np.isfinite(success_numeric)):
        raise ControlEvidenceError(f"{path.name}: success contains non-finite values")
    if not np.all(np.isin(success_array, [0, 1])):
        raise ControlEvidenceError(f"{path.name}: success must contain only 0/1 labels")
    try:
        states, state_counts = np.unique(init_array, return_counts=True)
    except (TypeError, ValueError) as exc:
        raise ControlEvidenceError(f"{path.name}: init_state is not comparable") from exc
    if len(states) != EXPECTED_FAMILIES_PER_TASK or set(state_counts.tolist()) != {
        EXPECTED_CANDIDATES
    }:
        raise ControlEvidenceError(
            f"{path.name}: expected {EXPECTED_FAMILIES_PER_TASK} init_state families with "
            f"{EXPECTED_CANDIDATES} candidates each, got {len(states)} and "
            f"{sorted(set(state_counts.tolist()))}"
        )

    rows: list[dict[str, Any]] = []
    for state in states:
        state_value = _as_python_identifier(state)
        indices = np.flatnonzero(init_array == state)
        candidate_values = candidate_array[indices]
        try:
            unique_candidates, candidate_counts = np.unique(
                candidate_values, return_counts=True
            )
        except (TypeError, ValueError) as exc:
            raise ControlEvidenceError(
                f"{path.name}: candidate_id is not comparable"
            ) from exc
        if len(unique_candidates) != EXPECTED_CANDIDATES or set(
            candidate_counts.tolist()
        ) != {1}:
            raise ControlEvidenceError(
                f"{path.name}: init_state {state_value!r} does not have exactly one "
                f"row for each of {EXPECTED_CANDIDATES} candidate IDs"
            )
        family_id = f"{path.name}:init_state-{state_value}"
        rows.extend(
            {"family_id": family_id, "success": bool(success_array[index])}
            for index in indices.tolist()
        )
    return rows, {
        "file": path.name,
        "family_count": len(states),
        "candidate_rows": int(len(success_array)),
        "candidate_count_per_family": EXPECTED_CANDIDATES,
        "candidate_ids_unique_per_family": True,
        "success_count": int(np.sum(success_array.astype(np.int64, copy=False))),
    }


def load_null(source_root: str | Path) -> dict[str, Any]:
    """Load and validate the authoritative Stage-R raw NPZ null source."""

    root = Path(source_root)
    if not root.is_dir():
        raise ControlEvidenceError(f"null source root does not exist: {root}")
    paths = sorted(root.glob("*.npz"))
    if len(paths) != EXPECTED_TASK_FILES:
        raise ControlEvidenceError(
            f"null source must contain exactly {EXPECTED_TASK_FILES} NPZ files, "
            f"found {len(paths)}"
        )

    rows: list[dict[str, Any]] = []
    file_details: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    for path in paths:
        before = sha256_file(path)
        try:
            with np.load(path, allow_pickle=False) as payload:
                required = {"success", "init_state", "candidate_id"}
                missing = sorted(required - set(payload.files))
                if missing:
                    raise ControlEvidenceError(
                        f"{path.name}: missing required arrays {missing}"
                    )
                # Only these three arrays are read. Other NPZ fields (actions,
                # poses, seeds, etc.) are deliberately out of this control.
                success = payload["success"]
                init_state = payload["init_state"]
                candidate_id = payload["candidate_id"]
                task_rows, detail = _validate_null_arrays(
                    path, success, init_state, candidate_id
                )
        except ControlEvidenceError:
            raise
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ControlEvidenceError(
                f"failed reading authoritative NPZ {path}: {exc}"
            ) from exc
        after = _assert_source_unchanged(path, before)
        entry = _inventory_entry(path, root)
        if entry["sha256"] != after:
            raise ControlEvidenceError(f"source digest changed while indexing: {path}")
        detail["sha256"] = after
        rows.extend(task_rows)
        file_details.append(detail)
        inventory.append(entry)

    expected_rows = EXPECTED_TASK_FILES * EXPECTED_ROLLOWS_PER_TASK
    expected_families = EXPECTED_TASK_FILES * EXPECTED_FAMILIES_PER_TASK
    if len(rows) != expected_rows:
        raise ControlEvidenceError(f"null logical-row count {len(rows)} != {expected_rows}")
    if len(file_details) != EXPECTED_TASK_FILES:
        raise ControlEvidenceError("null task-file detail count mismatch")
    manifest_sha = _manifest_digest(inventory)
    for entry in inventory:
        matching = next(
            detail for detail in file_details if detail["file"] == entry["relative_path"]
        )
        if matching["sha256"] != entry["sha256"]:
            raise ControlEvidenceError(f"null source digest mismatch for {entry['path']}")
    return {
        "rows": rows,
        "source": {
            "path": str(root),
            "kind": "stage_r_phase0r_authoritative_raw_npz",
            "arrays_read": ["success", "init_state", "candidate_id"],
            "files": inventory,
            "source_manifest_sha256": manifest_sha,
        },
        "counts": {
            "source_file_count": len(paths),
            "task_file_count": len(paths),
            "family_count": expected_families,
            "candidate_rows": len(rows),
            "families_per_task": sorted(
                {detail["family_count"] for detail in file_details}
            ),
            "candidate_rows_per_task": sorted(
                {detail["candidate_rows"] for detail in file_details}
            ),
            "candidate_count_per_family": EXPECTED_CANDIDATES,
            "all_family_counts_exact": all(
                detail["candidate_count_per_family"] == EXPECTED_CANDIDATES
                for detail in file_details
            ),
            "candidate_ids_unique_per_family": all(
                detail["candidate_ids_unique_per_family"] for detail in file_details
            ),
            "success_count_total": int(
                sum(detail["success_count"] for detail in file_details)
            ),
        },
        "file_details": file_details,
    }


def _pipeline_commit(repo_root: Path, explicit: str | None) -> str:
    if explicit:
        return str(explicit)
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"
    value = result.stdout.strip()
    return value or "UNKNOWN"


def _positive_report(
    loaded: Mapping[str, Any], pipeline_commit: str
) -> dict[str, Any]:
    s2 = compute_s2(loaded["rows"], n=EXPECTED_CANDIDATES)
    counts = loaded["counts"]
    detected = bool(
        int(s2["strict_zero_count"]) == int(counts["all_fail_family_count"])
        and int(s2["near_all_fail_count"])
        >= int(counts["near_all_fail_family_count"])
        and int(s2["strict_zero_count"]) > 0
    )
    return {
        "schema": "r142-stage-s-positive-control-v1",
        "pipeline_commit": pipeline_commit,
        "source": loaded["source"],
        "counts": counts,
        "s2": s2,
        "all_fail_detection": {
            "constructed_from_candidate_success_rate_x32": True,
            "direct_all_fail_family_count": int(counts["all_fail_family_count"]),
            "pipeline_strict_zero_count": int(s2["strict_zero_count"]),
            "pipeline_detected": detected,
        },
        "verdict": "POSITIVE_CONTROL_PASS" if detected else "PIPELINE_INVALID",
        "scientific_role": "instrument_only",
        "supports_idea": False,
        "stage1_data_is_idea_evidence": False,
        "interpretation": (
            "This is a pipeline sensitivity control: construction all-fail families are detected. "
            "It is not evidence for the Stage-1 idea and uses no generated trajectories."
        ),
    }


def _null_report(loaded: Mapping[str, Any], pipeline_commit: str) -> dict[str, Any]:
    s2 = compute_s2(loaded["rows"], n=EXPECTED_CANDIDATES)
    counts = loaded["counts"]
    no_collapse = bool(
        not bool(s2["pass"])
        and int(s2["near_all_fail_count"]) == 0
        and bool(counts["all_family_counts_exact"])
        and bool(counts["candidate_ids_unique_per_family"])
    )
    return {
        "schema": "r142-stage-s-null-control-v1",
        "pipeline_commit": pipeline_commit,
        "source": loaded["source"],
        "counts": counts,
        "s2": s2,
        "verdict": "NO_FAMILY_COLLAPSE" if no_collapse else "PIPELINE_INVALID",
        "scientific_role": "built_in_null_control",
        "supports_idea": False,
        "stage1_data_is_idea_evidence": False,
        "interpretation": (
            "Authoritative Stage-R raw NPZ is a null control for the S2 family-collapse detector; "
            "it cannot support the Stage-1 idea."
        ),
    }


def run_controls(
    *,
    positive_root: str | Path = POSITIVE_DEFAULT,
    null_root: str | Path = NULL_DEFAULT,
    output_root: str | Path = OUTPUT_DEFAULT,
    repo_root: str | Path = _REPO_ROOT,
    pipeline_commit: str | None = None,
) -> dict[str, Any]:
    """Execute both controls and persist an auditable completion bundle."""

    repo = Path(repo_root)
    output = Path(output_root)
    commit = _pipeline_commit(repo, pipeline_commit)
    positive_loaded = load_positive(positive_root)
    null_loaded = load_null(null_root)
    positive = _positive_report(positive_loaded, commit)
    null = _null_report(null_loaded, commit)
    overall = (
        "CONTROLS_PASS"
        if positive["verdict"] == "POSITIVE_CONTROL_PASS"
        and null["verdict"] == "NO_FAMILY_COLLAPSE"
        else "CONTROLS_INVALID"
    )
    aggregate = {
        "schema": "r142-stage-s-controls-v1",
        "pipeline_commit": commit,
        "positive_verdict": positive["verdict"],
        "null_verdict": null["verdict"],
        "overall_verdict": overall,
        "positive": {
            "source_manifest_sha256": positive["source"]["source_manifest_sha256"],
            "source_file_count": positive["counts"]["source_file_count"],
            "family_count": positive["counts"]["logical_family_count"],
            "candidate_rows": positive["counts"]["logical_candidate_row_count"],
            "all_fail_family_count": positive["counts"]["all_fail_family_count"],
            "near_all_fail_family_count": positive["counts"]["near_all_fail_family_count"],
            "s2": positive["s2"],
        },
        "null": {
            "source_manifest_sha256": null["source"]["source_manifest_sha256"],
            "source_file_count": null["counts"]["source_file_count"],
            "family_count": null["counts"]["family_count"],
            "candidate_rows": null["counts"]["candidate_rows"],
            "s2": null["s2"],
        },
        "stage1_data_is_idea_evidence": False,
        "scientific_role": "controls_only",
        "interpretation": (
            "Both controls validate the S2 measurement path only. They do not constitute a Stage-1 "
            "supporting result or a new scientific claim."
        ),
    }
    _atomic_json(output / "POSITIVE_CONTROL.json", positive)
    _atomic_json(output / "NULL_CONTROL.json", null)
    _atomic_json(output / "CONTROLS_REPORT.json", aggregate)
    completion = write_completion(
        output,
        overall,
        artifacts=[
            "POSITIVE_CONTROL.json",
            "NULL_CONTROL.json",
            "CONTROLS_REPORT.json",
        ],
        completion_name="COMPLETED_CONTROLS.json",
        metadata={
            "pipeline_commit": commit,
            "positive_source_manifest_sha256": positive["source"]["source_manifest_sha256"],
            "null_source_manifest_sha256": null["source"]["source_manifest_sha256"],
        },
    )
    bundle = verify_completion_bundle(output, completion_name=completion.name)
    if not bundle["valid"]:
        raise ControlEvidenceError(f"control completion bundle failed verification: {bundle}")
    return {
        "output_root": str(output),
        "completion": str(completion),
        "bundle": bundle,
        "overall_verdict": overall,
        "positive": positive,
        "null": null,
        "aggregate": aggregate,
    }


def _summary(result: Mapping[str, Any]) -> dict[str, Any]:
    positive = result["positive"]
    null = result["null"]
    return {
        "output_root": result["output_root"],
        "completion": result["completion"],
        "bundle": result["bundle"],
        "overall_verdict": result["overall_verdict"],
        "positive": {
            "verdict": positive["verdict"],
            "family_count": positive["counts"]["logical_family_count"],
            "candidate_rows": positive["counts"]["logical_candidate_row_count"],
            "all_fail_family_count": positive["counts"]["all_fail_family_count"],
            "near_all_fail_family_count": positive["counts"]["near_all_fail_family_count"],
            "s2_pass": positive["s2"]["pass"],
            "rho": positive["s2"]["rho"],
        },
        "null": {
            "verdict": null["verdict"],
            "task_file_count": null["counts"]["task_file_count"],
            "family_count": null["counts"]["family_count"],
            "candidate_rows": null["counts"]["candidate_rows"],
            "near_all_fail_family_count": null["s2"]["near_all_fail_count"],
            "s2_pass": null["s2"]["pass"],
            "rho": null["s2"]["rho"],
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-root", type=Path, default=POSITIVE_DEFAULT)
    parser.add_argument("--null-root", type=Path, default=NULL_DEFAULT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--pipeline-commit", default=None)
    args = parser.parse_args(argv)
    try:
        result = run_controls(
            positive_root=args.positive_root,
            null_root=args.null_root,
            output_root=args.output_root,
            repo_root=args.repo_root,
            pipeline_commit=args.pipeline_commit,
        )
    except (ControlEvidenceError, OSError, ValueError, TypeError) as exc:
        print(
            json.dumps(
                {"verdict": "CONTROLS_INVALID", "error": str(exc)},
                ensure_ascii=False,
            )
        )
        return 2
    print(json.dumps(_summary(result), ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result["overall_verdict"] == "CONTROLS_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

