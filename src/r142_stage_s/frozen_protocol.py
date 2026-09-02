"""Fail-closed reader for the Stage-S protocol authority.

The A main screen is allowed to run only against the immutable protocol
authority staged on CPFS.  This module intentionally uses only the Python
standard library so the launcher can validate the authority before importing
RoboTwin, SAPIEN, Torch, or the Evo server.

The authority JSON contains the following logical fields (a small set of
explicit aliases is accepted for backwards-compatible publication tooling):

* ``status == FROZEN`` and a full 40-hex ``protocol_git_commit``;
* byte hashes for ``PROTOCOL.md`` and the B/C calibration reports;
* a frozen summary with the Stage-S thresholds, seed plan, selected tasks, and
  A budget.  The summary is normalized into the fingerprint returned here.

Missing files, malformed data, path escapes, symlinks, and any mismatch are
capability failures.  No caller should turn them into warnings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_PROTOCOL_PATH = Path(
    "/mnt/cpfs/zbl-cpfs-new/USERS/leon/stage_s/protocol/FROZEN_PROTOCOL.json"
)
EXPECTED_STATUS = "FROZEN"
EXPECTED_SEED_BASE = 14211
EXPECTED_SEED_RULE = "SeedSequence([initial_seed, candidate_index])"
EXPECTED_TASKS = (
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
EXPECTED_BUDGET = {
    "task_count": 10,
    "families_per_task": 16,
    "candidates_per_family": 32,
    "terminal_episode_count": 5120,
    "world_size": 8,
}
EXPECTED_THRESHOLDS = {
    "s1_success_rate_min": 0.30,
    "s1_success_rate_max": 0.60,
    "s2_near_all_fail_fraction_min": 0.10,
    "s2_rho_min": 3.0,
    "s2_binomial_multiplier_min": 20.0,
    "s3_median_t_div_fraction_min": 0.10,
    "s3_t_div_zero_fraction_max": 0.25,
    "s4_recoverable_family_fraction_min": 0.30,
    "s4_oracle_random_ci_lower_bound_min": 0.0,
    "s5_best_of_n64_rescue_fraction_max": 0.05,
}
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_CPFS_ROOT = Path("/mnt/cpfs/zbl-cpfs-new")


class FrozenProtocolError(RuntimeError):
    """The protocol authority is absent, malformed, or has been altered."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise FrozenProtocolError(f"cannot read frozen protocol artifact: {path}") from exc
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FrozenProtocolError(f"frozen protocol JSON is missing or symlinked: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise FrozenProtocolError(f"frozen protocol JSON is invalid: {path}") from exc
    if not isinstance(value, Mapping):
        raise FrozenProtocolError("frozen protocol JSON root must be an object")
    return value


def _norm(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _flatten(value: Mapping[str, Any], prefix: Sequence[str] = ()) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, child in value.items():
        current = (*prefix, _norm(key))
        if isinstance(child, Mapping):
            result.update(_flatten(child, current))
        else:
            result["_".join(item for item in current if item)] = child
    return result


def _mapping(value: Mapping[str, Any], *names: str) -> Mapping[str, Any]:
    for name in names:
        child = value.get(name)
        if isinstance(child, Mapping):
            return child
    return {}


def _first(flat: Mapping[str, Any], aliases: Iterable[str]) -> Any:
    normalized = tuple(_norm(alias) for alias in aliases)
    for alias in normalized:
        if alias in flat:
            return flat[alias]
    for key, value in flat.items():
        if any(key.endswith("_" + alias) for alias in normalized):
            return value
    return None


def _require_number(flat: Mapping[str, Any], aliases: Iterable[str], label: str) -> float:
    value = _first(flat, aliases)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FrozenProtocolError(f"frozen protocol summary lacks numeric {label}")
    number = float(value)
    if not math.isfinite(number):
        raise FrozenProtocolError(f"frozen protocol summary has non-finite {label}")
    return number


def _file_entry(
    authority: Mapping[str, Any],
    *,
    aliases: Sequence[str],
    path_aliases: Sequence[str],
    hash_aliases: Sequence[str],
    label: str,
) -> tuple[Path, str]:
    files = _mapping(authority, "files", "artifacts", "frozen_files")
    entry: Mapping[str, Any] = {}
    for alias in aliases:
        candidate = files.get(alias)
        if isinstance(candidate, Mapping):
            entry = candidate
            break
    if not entry:
        alias_names = {_norm(alias) for alias in aliases}
        for key, candidate in files.items():
            if _norm(key) in alias_names and isinstance(candidate, Mapping):
                entry = candidate
                break
    path_value: Any = None
    hash_value: Any = None
    for source in (entry, authority):
        flat = _flatten(source)
        if path_value is None:
            path_value = _first(flat, path_aliases)
        if hash_value is None:
            hash_value = _first(flat, hash_aliases)
    if not isinstance(path_value, str) or not path_value:
        raise FrozenProtocolError(f"frozen protocol lacks {label} path")
    if not isinstance(hash_value, str) or not _HEX64.fullmatch(hash_value):
        raise FrozenProtocolError(f"frozen protocol lacks valid {label} SHA-256")
    path = Path(path_value)
    if not path.is_absolute():
        raise FrozenProtocolError(f"frozen protocol {label} path must be absolute: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise FrozenProtocolError(f"frozen protocol {label} path is unreadable: {path}") from exc
    if resolved.is_symlink() or not resolved.is_file():
        raise FrozenProtocolError(f"frozen protocol {label} path is not a regular file: {path}")
    try:
        resolved.relative_to(_CPFS_ROOT)
    except ValueError as exc:
        raise FrozenProtocolError(f"frozen protocol {label} path escapes CPFS: {path}") from exc
    actual = _sha256(resolved)
    if actual != hash_value.lower():
        raise FrozenProtocolError(
            f"frozen protocol {label} SHA-256 mismatch: expected {hash_value}, got {actual}"
        )
    return resolved, actual


def _protocol_commit(authority: Mapping[str, Any]) -> str:
    nested = _mapping(authority, "protocol_git", "git", "protocol_source")
    flat = _flatten(authority)
    commit = _first(
        {**_flatten(nested), **flat},
        ("protocol_git_commit", "protocol_commit", "git_commit", "commit"),
    )
    if not isinstance(commit, str) or not _HEX40.fullmatch(commit):
        raise FrozenProtocolError(
            "frozen protocol must contain a full 40-hex protocol Git commit"
        )
    full_sha = _first(
        {**_flatten(nested), **flat},
        (
            "protocol_full_sha",
            "protocol_git_full_sha",
            "protocol_git_sha",
            "protocol_git_sha256",
            "full_sha",
        ),
    )
    if full_sha is not None and full_sha != commit:
        raise FrozenProtocolError("frozen protocol Git full SHA does not match commit")
    return commit


def _summary(authority: Mapping[str, Any]) -> dict[str, Any]:
    summary = _mapping(authority, "frozen_summary", "summary", "protocol_summary", "frozen")
    if not summary:
        raise FrozenProtocolError("frozen protocol lacks frozen_summary")
    thresholds_source = _mapping(summary, "thresholds", "gates", "gate_thresholds")
    if not thresholds_source:
        raise FrozenProtocolError("frozen protocol summary lacks thresholds")
    threshold_flat = _flatten(thresholds_source)
    thresholds: dict[str, float] = {}
    threshold_aliases = {
        "s1_success_rate_min": ("s1_success_rate_min", "s1_difficulty_min", "success_rate_min"),
        "s1_success_rate_max": ("s1_success_rate_max", "s1_difficulty_max", "success_rate_max"),
        "s2_near_all_fail_fraction_min": (
            "s2_near_all_fail_fraction_min", "s2_collapse_fraction_min", "near_all_fail_fraction_min"
        ),
        "s2_rho_min": ("s2_rho_min", "s2_overdispersion_rho_min", "rho_min"),
        "s2_binomial_multiplier_min": (
            "s2_binomial_multiplier_min", "s2_near_all_fail_binomial_multiplier", "near_all_fail_binomial_multiplier"
        ),
        "s3_median_t_div_fraction_min": (
            "s3_median_t_div_fraction_min", "s3_prefix_median_t_div_fraction", "median_t_div_fraction_min"
        ),
        "s3_t_div_zero_fraction_max": (
            "s3_t_div_zero_fraction_max", "s3_origin_fraction_max", "t_div_zero_fraction_max"
        ),
        "s4_recoverable_family_fraction_min": (
            "s4_recoverable_family_fraction_min", "s4_rescue_fraction_min", "recoverable_family_fraction_min"
        ),
        "s4_oracle_random_ci_lower_bound_min": (
            "s4_oracle_random_ci_lower_bound_min", "s4_random_ci_lower_bound_min", "oracle_random_ci_lower_bound_min"
        ),
        "s5_best_of_n64_rescue_fraction_max": (
            "s5_best_of_n64_rescue_fraction_max", "s5_budget_rescue_fraction_max", "budget_rescue_fraction_max"
        ),
    }
    for key, aliases in threshold_aliases.items():
        thresholds[key] = _require_number(threshold_flat, aliases, key)
        if not math.isclose(thresholds[key], EXPECTED_THRESHOLDS[key], rel_tol=0.0, abs_tol=1e-12):
            raise FrozenProtocolError(
                f"frozen threshold drift for {key}: expected {EXPECTED_THRESHOLDS[key]}, got {thresholds[key]}"
            )

    seed_source = _mapping(summary, "seed_plan", "seeds", "rng")
    seed_flat = _flatten(seed_source or summary)
    seed_base = _first(seed_flat, ("seed_base", "base_seed", "initial_seed"))
    if isinstance(seed_base, bool) or not isinstance(seed_base, int) or seed_base != EXPECTED_SEED_BASE:
        raise FrozenProtocolError(
            f"frozen seed base drift or omission: expected {EXPECTED_SEED_BASE}, got {seed_base!r}"
        )
    seed_rule = _first(
        seed_flat,
        ("candidate_seed_rule", "candidate_seed_derivation", "candidate_rng", "seed_derivation"),
    )
    if seed_rule is not None and seed_rule != EXPECTED_SEED_RULE:
        raise FrozenProtocolError(f"frozen candidate seed rule drift: {seed_rule!r}")

    tasks_value = summary.get("tasks", authority.get("tasks"))
    if not isinstance(tasks_value, list) or tuple(str(item) for item in tasks_value) != EXPECTED_TASKS:
        raise FrozenProtocolError("frozen task summary does not equal the A task selection")

    budget_source = _mapping(summary, "budget", "budgets", "compute_budget")
    budget_flat = _flatten(budget_source or summary)
    aliases = {
        "task_count": ("task_count", "tasks", "num_tasks"),
        "families_per_task": ("families_per_task", "family_count_per_task"),
        "candidates_per_family": ("candidates_per_family", "candidate_budget", "candidates"),
        "terminal_episode_count": ("terminal_episode_count", "terminal_episodes", "episode_count"),
        "world_size": ("world_size", "ranks", "rank_count"),
    }
    budget: dict[str, int] = {}
    for key, key_aliases in aliases.items():
        value = _require_number(budget_flat, key_aliases, key)
        if int(value) != value or int(value) != EXPECTED_BUDGET[key]:
            raise FrozenProtocolError(
                f"frozen budget drift for {key}: expected {EXPECTED_BUDGET[key]}, got {value}"
            )
        budget[key] = int(value)

    return {
        "thresholds": thresholds,
        "seed_plan": {"seed_base": EXPECTED_SEED_BASE, "candidate_seed_rule": EXPECTED_SEED_RULE},
        "tasks": list(EXPECTED_TASKS),
        "budget": budget,
    }


def load_frozen_protocol(path: Path = DEFAULT_PROTOCOL_PATH) -> dict[str, Any]:
    """Read and hash the complete frozen protocol authority."""

    path = Path(path)
    if path != path.resolve():
        raise FrozenProtocolError(f"frozen protocol path must be canonical: {path}")
    if path.name != "FROZEN_PROTOCOL.json" or path.parent.name != "protocol":
        raise FrozenProtocolError(f"unexpected frozen protocol path: {path}")
    try:
        path.resolve().relative_to(_CPFS_ROOT)
    except ValueError as exc:
        raise FrozenProtocolError(f"frozen protocol path escapes CPFS: {path}") from exc
    authority = _read_json(path)
    if authority.get("status") != EXPECTED_STATUS:
        raise FrozenProtocolError(
            f"frozen protocol status must be {EXPECTED_STATUS}, got {authority.get('status')!r}"
        )
    commit = _protocol_commit(authority)
    protocol_sha = _sha256(path)
    declared_sha = authority.get("protocol_json_sha256")
    if declared_sha is not None and declared_sha != protocol_sha:
        raise FrozenProtocolError(
            f"frozen protocol JSON SHA-256 mismatch: expected {declared_sha}, got {protocol_sha}"
        )
    protocol_md_path, protocol_md_sha = _file_entry(
        authority,
        aliases=("PROTOCOL.md", "protocol_md", "protocol"),
        path_aliases=("protocol_md_path", "protocol_path", "protocol_markdown_path", "path"),
        hash_aliases=("protocol_md_sha256", "protocol_sha256", "sha256", "hash"),
        label="PROTOCOL.md",
    )
    b_path, b_sha = _file_entry(
        authority,
        aliases=("B_CALIBRATION_REPORT", "B", "b_calibration_report"),
        path_aliases=(
            "b_calibration_report_path",
            "calibration_report_b_path",
            "calibration_reports_b_path",
            "b_path",
            "path",
        ),
        hash_aliases=(
            "b_calibration_report_sha256",
            "calibration_report_b_sha256",
            "calibration_reports_b_sha256",
            "b_sha256",
            "sha256",
            "hash",
        ),
        label="B calibration report",
    )
    c_path, c_sha = _file_entry(
        authority,
        aliases=("C_CALIBRATION_REPORT", "C", "c_calibration_report"),
        path_aliases=(
            "c_calibration_report_path",
            "calibration_report_c_path",
            "calibration_reports_c_path",
            "c_path",
            "path",
        ),
        hash_aliases=(
            "c_calibration_report_sha256",
            "calibration_report_c_sha256",
            "calibration_reports_c_sha256",
            "c_sha256",
            "sha256",
            "hash",
        ),
        label="C calibration report",
    )
    summary = _summary(authority)
    summary_sha = hashlib.sha256(
        (json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    ).hexdigest()
    return {
        "path": str(path),
        "status": EXPECTED_STATUS,
        "protocol_git_commit": commit,
        "protocol_full_sha": commit,
        "protocol_json_sha256": protocol_sha,
        "protocol_md_path": str(protocol_md_path),
        "protocol_md_sha256": protocol_md_sha,
        "calibration_reports": {
            "B": {"path": str(b_path), "sha256": b_sha},
            "C": {"path": str(c_path), "sha256": c_sha},
        },
        "frozen_summary": summary,
        "frozen_summary_sha256": summary_sha,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=DEFAULT_PROTOCOL_PATH)
    return parser


def main() -> int:
    try:
        print(json.dumps(load_frozen_protocol(build_parser().parse_args().path), sort_keys=True, indent=2))
    except FrozenProtocolError as exc:
        print(json.dumps({"status": "BLOCKED_FROZEN_PROTOCOL", "error": str(exc)}))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
