"""Fail-closed reader for the accepted Stage-S substrate-A asset preflight.

The asset preflight is a separate PAI job from the A main screen.  A main
rollout may consume it only through the stable CPFS acceptance pointer
``ACCEPTED_A_ASSET_PREFLIGHT.json``.  This module deliberately uses only the
Python standard library: the launcher can validate the pointer and all of its
lineage before importing RoboTwin, SAPIEN, Torch, or the Evo server.

The pointer is an atomic, human-readable acceptance record written by the
controller *after* an asset Job reaches terminal ``Succeeded`` and both its
completion marker and checksum manifest have been verified.  It is not enough
for a directory to contain ``FIRST_WORK.json`` or for a PAI job to be
``Running``.  We re-read the referenced output and checkpoint manifests here
and compare every declared source, model, and byte hash before returning a
fingerprint for run metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


_CPFS_ROOT = Path("/mnt/cpfs/zbl-cpfs-new")
DEFAULT_ACCEPTED_ASSET_PATH = (
    _CPFS_ROOT / "USERS/leon/stage_s/protocol/ACCEPTED_A_ASSET_PREFLIGHT.json"
)
_ASSET_OUTPUT_ROOT_SUFFIX = Path("USERS/leon/logs/r142_fp11_stage_s/assets")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9._:-]+$")

EXPECTED_STATUS = "ACCEPTED"
EXPECTED_TERMINAL_PAI_STATE = "Succeeded"
EXPECTED_GPUS = 8
EXPECTED_MODEL_REVISION = "ce8c583724706fbf7a03c17237761c65bf6813a7"
EXPECTED_SOURCE_COMMITS = {
    "evo": "5fd14b015013c4fd0aacf5f8f48f868ca9b870a2",
    "robotwin": "13c3c47ff4312dd62484bcd51be034af55c062d1",
    "curobo": "d64c4b005459db10c5dd867d8b30a87d5bda9bdb",
}


class AssetAcceptanceError(RuntimeError):
    """The stable asset acceptance record is absent, stale, or unverifiable."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise AssetAcceptanceError(f"cannot read asset artifact: {path}") from exc
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise AssetAcceptanceError(f"{label} is missing or symlinked: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AssetAcceptanceError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise AssetAcceptanceError(f"{label} JSON root must be an object: {path}")
    return value


def _under(path: Path, root: Path, label: str) -> Path:
    """Resolve a regular CPFS path without permitting symlink/path escapes."""

    if not path.is_absolute():
        raise AssetAcceptanceError(f"{label} must be an absolute path: {path}")
    if path.is_symlink():
        raise AssetAcceptanceError(f"{label} must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise AssetAcceptanceError(f"{label} is unreadable: {path}") from exc
    if resolved.is_symlink():
        raise AssetAcceptanceError(f"{label} must not be a symlink: {path}")
    try:
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise AssetAcceptanceError(f"{label} escapes CPFS: {path}") from exc
    return resolved


def _value(payload: Mapping[str, Any], *paths: str) -> Any:
    """Read a small explicit alias set from direct or dotted JSON keys."""

    for dotted in paths:
        current: Any = payload
        for key in dotted.split("."):
            if not isinstance(current, Mapping) or key not in current:
                current = None
                break
            current = current[key]
        if current is not None:
            return current
    return None


def _required_string(payload: Mapping[str, Any], aliases: Sequence[str], label: str) -> str:
    value = _value(payload, *aliases)
    if not isinstance(value, str) or not value.strip():
        raise AssetAcceptanceError(f"accepted asset record lacks {label}")
    return value.strip()


def _required_hash(payload: Mapping[str, Any], aliases: Sequence[str], label: str) -> str:
    value = _required_string(payload, aliases, label)
    if not _HEX64.fullmatch(value.lower()):
        raise AssetAcceptanceError(f"accepted asset record has invalid {label} SHA-256")
    return value.lower()


def _verify_checksum_manifest(manifest: Path, root: Path, label: str) -> None:
    """Verify every target listed by a sha256sum-style manifest."""

    if not manifest.is_file() or manifest.is_symlink():
        raise AssetAcceptanceError(f"{label} is missing or symlinked: {manifest}")
    seen: set[str] = set()
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise AssetAcceptanceError(f"cannot read {label}: {manifest}") from exc
    for raw in lines:
        if not raw.strip():
            continue
        parts = raw.split(None, 1)
        if len(parts) != 2:
            raise AssetAcceptanceError(f"malformed checksum line in {manifest}: {raw!r}")
        expected, relative = parts
        relative = relative.lstrip(" *")
        if not _HEX64.fullmatch(expected.lower()):
            raise AssetAcceptanceError(f"invalid checksum in {manifest}: {raw!r}")
        relative_path = Path(relative)
        if (
            not relative
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative in seen
        ):
            raise AssetAcceptanceError(f"unsafe or duplicate checksum path in {manifest}: {relative!r}")
        seen.add(relative)
        target = manifest.parent / relative_path
        resolved = _under(target, root, "checksum target")
        if not resolved.is_file():
            raise AssetAcceptanceError(f"checksum target is not a regular file: {target}")
        if _sha256(resolved) != expected.lower():
            raise AssetAcceptanceError(f"checksum mismatch for {target}")
    if not seen:
        raise AssetAcceptanceError(f"empty checksum manifest: {manifest}")
    if manifest.name in seen:
        raise AssetAcceptanceError(f"checksum manifest self-hash is forbidden: {manifest}")


def _source_commits(record: Mapping[str, Any]) -> dict[str, str]:
    nested = _value(record, "source_commits", "sources", "provenance.source_commits")
    if not isinstance(nested, Mapping):
        nested = {}
    result: dict[str, str] = {}
    aliases = {
        "evo": ("evo", "evo_commit", "Evo-1", "Evo1"),
        "robotwin": ("robotwin", "robotwin_commit", "RoboTwin"),
        "curobo": ("curobo", "curobo_commit", "CuRobo"),
    }
    for name, keys in aliases.items():
        value: Any = None
        for key in keys:
            value = nested.get(key)
            if value is not None:
                break
        if value is None:
            value = _value(record, *(f"{key}" for key in keys))
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", value):
            raise AssetAcceptanceError(f"accepted asset record lacks full {name} source commit")
        result[name] = value.lower()
    return result


def load_accepted_asset_preflight(
    path: Path = DEFAULT_ACCEPTED_ASSET_PATH,
    *,
    checkpoint_dir: Path | None = None,
) -> dict[str, Any]:
    """Validate the stable acceptance pointer and return immutable lineage.

    ``checkpoint_dir`` is supplied by the A launcher/runtime, rather than
    trusted from the acceptance JSON.  The accepted record must name the same
    directory and its live ``SHA256SUMS`` bytes are hashed again here.
    """

    path = Path(path)
    if path != path.resolve():
        raise AssetAcceptanceError(f"accepted asset path must be canonical: {path}")
    if path.name != "ACCEPTED_A_ASSET_PREFLIGHT.json" or path.parent.name != "protocol":
        raise AssetAcceptanceError(f"unexpected accepted asset path: {path}")
    try:
        path.resolve().relative_to(_CPFS_ROOT.resolve())
    except ValueError as exc:
        raise AssetAcceptanceError(f"accepted asset path escapes CPFS: {path}") from exc
    record = _read_json(path, "accepted asset record")
    if record.get("status") != EXPECTED_STATUS:
        raise AssetAcceptanceError(
            f"accepted asset status must be {EXPECTED_STATUS}, got {record.get('status')!r}"
        )
    terminal_state = _required_string(
        record,
        ("terminal_pai_state", "pai_terminal_state", "pai_state", "job_state"),
        "terminal PAI state",
    )
    if terminal_state != EXPECTED_TERMINAL_PAI_STATE:
        raise AssetAcceptanceError(
            "accepted asset requires terminal PAI state Succeeded, "
            f"got {terminal_state!r}"
        )
    run_id = _required_string(record, ("accepted_run_id", "asset_run_id", "run_id"), "accepted run id")
    job_id = _required_string(
        record,
        ("accepted_job_id", "asset_job_id", "pai_job_id", "job_id"),
        "accepted PAI JobId",
    )
    if not _SAFE_ID.fullmatch(run_id) or not _SAFE_ID.fullmatch(job_id):
        raise AssetAcceptanceError("accepted asset run/job id contains unsafe characters")

    output_value = _required_string(
        record,
        ("output_dir", "asset_output_dir", "output_root"),
        "asset output directory",
    )
    output_dir = _under(Path(output_value), _CPFS_ROOT, "asset output directory")
    expected_output_root = (_CPFS_ROOT / _ASSET_OUTPUT_ROOT_SUFFIX).resolve()
    try:
        output_dir.relative_to(expected_output_root)
    except ValueError as exc:
        raise AssetAcceptanceError(
            f"accepted asset output directory is outside the Stage-S asset root: {output_dir}"
        ) from exc
    if not output_dir.is_dir():
        raise AssetAcceptanceError(f"accepted asset output directory is not a directory: {output_dir}")

    marker_value = _required_string(
        record,
        ("completion_marker", "asset_completion_marker", "completion_file"),
        "asset completion marker",
    )
    marker_path = Path(marker_value)
    if not marker_path.is_absolute():
        marker_path = output_dir / marker_path
    marker_path = _under(marker_path, output_dir, "asset completion marker")
    if marker_path.parent != output_dir:
        raise AssetAcceptanceError("asset completion marker must be directly under output_dir")
    marker = _read_json(marker_path, "asset completion marker")
    if marker.get("status") != "COMPLETED":
        raise AssetAcceptanceError(
            f"asset completion marker is not terminal COMPLETED: {marker_path}"
        )
    if str(marker.get("job_id") or "") != job_id:
        raise AssetAcceptanceError("asset completion marker JobId does not match accepted JobId")
    if marker.get("gpus") != EXPECTED_GPUS:
        raise AssetAcceptanceError("asset completion marker does not prove the frozen 8-GPU preflight")
    if marker.get("model_revision") != EXPECTED_MODEL_REVISION:
        raise AssetAcceptanceError("asset completion marker model revision drifted")
    source_commits = _source_commits(record)
    marker_sources = _source_commits(marker)
    if source_commits != EXPECTED_SOURCE_COMMITS or marker_sources != EXPECTED_SOURCE_COMMITS:
        raise AssetAcceptanceError("accepted asset source commits drifted from the audited pins")
    if marker_sources != source_commits:
        raise AssetAcceptanceError("asset completion marker source commits disagree with acceptance record")

    asset_sums = output_dir / "SHA256SUMS"
    asset_sums_declared = _required_hash(
        record,
        ("asset_sha256sums_sha256", "output_sha256sums_sha256", "asset_manifest_sha256"),
        "asset SHA256SUMS",
    )
    asset_sums_resolved = _under(asset_sums, output_dir, "asset SHA256SUMS")
    _verify_checksum_manifest(asset_sums_resolved, output_dir, "asset SHA256SUMS")
    asset_sums_actual = _sha256(asset_sums_resolved)
    if asset_sums_actual != asset_sums_declared:
        raise AssetAcceptanceError(
            f"accepted asset SHA256SUMS hash mismatch: expected {asset_sums_declared}, got {asset_sums_actual}"
        )

    completion_declared = _required_hash(
        record,
        ("completion_sha256", "asset_completion_sha256", "completion_marker_sha256"),
        "asset completion marker",
    )
    completion_actual = _sha256(marker_path)
    if completion_actual != completion_declared:
        raise AssetAcceptanceError(
            f"accepted asset completion hash mismatch: expected {completion_declared}, got {completion_actual}"
        )

    model_value = _required_string(
        record,
        ("model_dir", "checkpoint_dir", "model_path"),
        "model/checkpoint directory",
    )
    model_dir = _under(Path(model_value), _CPFS_ROOT, "model/checkpoint directory")
    runtime_model_dir = _under(
        Path(checkpoint_dir) if checkpoint_dir is not None else model_dir,
        _CPFS_ROOT,
        "runtime model/checkpoint directory",
    )
    if model_dir != runtime_model_dir:
        raise AssetAcceptanceError(
            f"accepted model directory mismatch: accepted {model_dir}, runtime {runtime_model_dir}"
        )
    if marker.get("model_sha256sums_sha256") is not None:
        marker_model_hash = str(marker.get("model_sha256sums_sha256")).lower()
        if not _HEX64.fullmatch(marker_model_hash):
            raise AssetAcceptanceError("asset completion marker model SHA256SUMS hash is invalid")
    else:
        raise AssetAcceptanceError("asset completion marker lacks model SHA256SUMS hash")
    model_sums = _under(model_dir / "SHA256SUMS", model_dir, "model SHA256SUMS")
    _verify_checksum_manifest(model_sums, model_dir, "model SHA256SUMS")
    model_sums_actual = _sha256(model_sums)
    model_sums_declared = _required_hash(
        record,
        ("model_sha256sums_sha256", "checkpoint_sha256sums_sha256", "model_manifest_sha256"),
        "model SHA256SUMS",
    )
    if model_sums_actual != model_sums_declared or model_sums_actual != str(marker.get("model_sha256sums_sha256")).lower():
        raise AssetAcceptanceError("accepted model SHA256SUMS hash disagrees with live checkpoint bytes")
    if record.get("model_revision") != EXPECTED_MODEL_REVISION:
        raise AssetAcceptanceError("accepted asset model revision drifted")
    if record.get("gpus", EXPECTED_GPUS) != EXPECTED_GPUS:
        raise AssetAcceptanceError("accepted asset record GPU count drifted")

    return {
        "path": str(path),
        "status": EXPECTED_STATUS,
        "accepted_run_id": run_id,
        "accepted_job_id": job_id,
        "terminal_pai_state": terminal_state,
        "output_dir": str(output_dir),
        "completion_marker": str(marker_path),
        "completion_sha256": completion_actual,
        "asset_sha256sums": str(asset_sums_resolved),
        "asset_sha256sums_sha256": asset_sums_actual,
        "model_dir": str(model_dir),
        "model_revision": EXPECTED_MODEL_REVISION,
        "model_sha256sums": str(model_sums),
        "model_sha256sums_sha256": model_sums_actual,
        "source_commits": dict(source_commits),
        "accepted_manifest_sha256": _sha256(path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=DEFAULT_ACCEPTED_ASSET_PATH)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    return parser


def main() -> int:
    parsed = build_parser().parse_args()
    try:
        result = load_accepted_asset_preflight(
            parsed.path,
            checkpoint_dir=parsed.checkpoint_dir,
        )
    except AssetAcceptanceError as exc:
        print(json.dumps({"status": "BLOCKED_ACCEPTED_ASSET_PREFLIGHT", "error": str(exc)}))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
