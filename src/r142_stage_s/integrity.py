"""Fail-closed SHA-256 and completion-bundle verification."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

SHA256SUMS_NAME = "SHA256SUMS"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(root: Path, value: str | Path) -> str:
    path = Path(value)
    if path.is_absolute():
        try:
            path = path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"artifact path is outside root: {value}") from exc
    relative = PurePosixPath(path.as_posix())
    if relative.is_absolute() or ".." in relative.parts or str(relative) in ("", "."):
        raise ValueError(f"unsafe artifact path: {value}")
    return relative.as_posix()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _default_artifacts(root: Path, *, include_completion: bool = True) -> list[str]:
    output: list[str] = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == SHA256SUMS_NAME or relative.endswith(".tmp"):
            continue
        if not include_completion and PurePosixPath(relative).name.startswith("COMPLETED_"):
            continue
        output.append(relative)
    return output


def write_sha256sums(
    root: str | Path,
    paths: Iterable[str | Path] | None = None,
    *,
    filename: str = SHA256SUMS_NAME,
) -> Path:
    """Write a deterministic GNU-compatible SHA-256 manifest atomically."""

    base = Path(root)
    manifest_path = base / _relative_path(base, filename)
    if paths is None:
        relatives = _default_artifacts(base, include_completion=True)
    else:
        relatives = sorted({_relative_path(base, path) for path in paths})
    manifest_relative = manifest_path.relative_to(base).as_posix()
    if manifest_relative in relatives:
        raise ValueError("SHA256SUMS cannot contain itself")
    lines = []
    for relative in relatives:
        path = base / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        lines.append(f"{sha256_file(path)}  {relative}\n")
    _atomic_text(manifest_path, "".join(lines))
    return manifest_path


def _parse_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 2 or len(fields[0]) != 64:
            raise ValueError(f"malformed SHA256SUMS line {line_number}")
        digest, relative = fields
        if any(character not in "0123456789abcdefABCDEF" for character in digest):
            raise ValueError(f"invalid SHA-256 digest on line {line_number}")
        if relative in entries:
            raise ValueError(f"duplicate SHA256SUMS path {relative}")
        entries[relative] = digest.lower()
    return entries


def verify_sha256sums(root: str | Path, *, filename: str = SHA256SUMS_NAME) -> bool:
    """Verify every entry in a manifest and reject traversal/unknown paths."""

    base = Path(root)
    try:
        manifest = base / _relative_path(base, filename)
    except ValueError:
        return False
    if not manifest.is_file():
        return False
    try:
        entries = _parse_manifest(manifest)
        for relative, expected in entries.items():
            safe = _relative_path(base, relative)
            if safe != relative:
                return False
            path = base / safe
            if not path.is_file() or sha256_file(path) != expected:
                return False
    except (OSError, ValueError):
        return False
    return True


def write_completion(
    root: str | Path,
    decision: str,
    *,
    artifacts: Sequence[str | Path] | None = None,
    completion_name: str = "COMPLETED_STAGE_S.json",
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Persist completion metadata and then a SHA256SUMS covering the bundle."""

    base = Path(root)
    completion_relative = _relative_path(base, completion_name)
    if not PurePosixPath(completion_relative).name.startswith("COMPLETED_"):
        raise ValueError("completion file must be named COMPLETED_*.json")
    completion = base / completion_relative
    if artifacts is None:
        relatives = _default_artifacts(base, include_completion=False)
    else:
        relatives = sorted({_relative_path(base, path) for path in artifacts})
    if completion_relative in relatives or SHA256SUMS_NAME in relatives:
        raise ValueError("completion artifacts cannot include completion or SHA256SUMS")
    artifact_rows = []
    for relative in relatives:
        path = base / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        artifact_rows.append({"path": relative, "sha256": sha256_file(path)})
    payload: dict[str, Any] = {
        "schema": "r142-stage-s-completion-v1",
        "decision": str(decision),
        "artifacts": artifact_rows,
    }
    if metadata:
        payload["metadata"] = dict(metadata)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    _atomic_text(completion, encoded)
    # Include the completion record itself in the manifest after its content is
    # final; otherwise a self-referential digest would be impossible.
    write_sha256sums(base)
    return completion


def verify_completed_json(
    root: str | Path, *, completion_name: str | None = None
) -> bool:
    """Verify the completion record's artifact digests, without trusting it."""

    base = Path(root)
    try:
        if completion_name is None:
            candidates = sorted(
                path for path in base.glob("COMPLETED_*.json") if path.is_file()
            )
            if len(candidates) != 1:
                return False
            completion = candidates[0]
        else:
            relative = _relative_path(base, completion_name)
            if not PurePosixPath(relative).name.startswith("COMPLETED_"):
                return False
            completion = base / relative
        payload = json.loads(completion.read_text(encoding="utf-8"))
        artifacts = payload["artifacts"]
        if not isinstance(artifacts, list) or not artifacts:
            return False
        seen: set[str] = set()
        for row in artifacts:
            if not isinstance(row, Mapping):
                return False
            relative = _relative_path(base, str(row["path"]))
            if relative in seen or relative != str(row["path"]):
                return False
            seen.add(relative)
            path = base / relative
            expected = str(row["sha256"]).lower()
            if len(expected) != 64 or sha256_file(path) != expected:
                return False
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def verify_completion_bundle(
    root: str | Path, *, completion_name: str | None = None
) -> dict[str, Any]:
    """Return separately auditable completion/SHA status.

    A bundle is valid only when both the completion record and the manifest
    pass. Returning component status makes a partial/queued result visible to
    callers instead of collapsing it into a false scientific success.
    """

    completion_ok = verify_completed_json(root, completion_name=completion_name)
    sha_ok = verify_sha256sums(root)
    return {
        "completed_json": completion_ok,
        "sha256sums": sha_ok,
        "valid": completion_ok and sha_ok,
    }


__all__ = [
    "SHA256SUMS_NAME",
    "sha256_bytes",
    "sha256_file",
    "verify_completion_bundle",
    "verify_completed_json",
    "verify_sha256sums",
    "write_completion",
    "write_sha256sums",
]
