"""Fail-closed, read-only identity attestation for Stage-S PAI runs.

The Stage-S launchers are intentionally split between a Git checkout and a
deployed CPFS payload.  This module checks the two sides independently and
only declares an identity match when the exact configured bytes, source
revisions, input authorities, and resource contract all agree.  It never
repairs a checkout, rewrites a config, downloads an asset, or starts a PAI
job.

``attest_config`` is useful from tests and callers that already have a
``Path``.  The command line wrapper lives in
``scripts/stage_s_runtime_identity_preflight.py`` and requires an explicit
output path before it writes an attestation.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping


SCHEMA = "r142-stage-s-runtime-identity-attestation-v1"
HEX40 = re.compile(r"^[0-9a-fA-F]{40}$")
HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
PLACEHOLDER = re.compile(r"\{\{[^{}]+\}\}|<[A-Za-z][A-Za-z0-9_.-]*>")
# CPFS-backed Git status can take tens of seconds while the tree metadata is
# cold. Keep every probe bounded, but allow a clean checkout to finish
# without turning an I/O timeout into a false "dirty" result.
GIT_COMMAND_TIMEOUT_SECONDS = 60
GIT_STATUS_MAX_ATTEMPTS = 3
GIT_STATUS_RETRY_DELAY_SECONDS = 0.25
PATH_TOKENS = (
    "path",
    "root",
    "repo",
    "manifest",
    "report",
    "checkpoint",
    "model",
    "protocol",
    "asset",
)
IGNORE_KEYS = {
    "write_paths",
    "output_root",
    "log_root_template",
    "log_sha_template",
    "training_pipeline_completion_template",
    "first_work_evidence_path",
    "completed_evidence_path",
    "completion_evidence_path",
    "completion_marker",
    "integrity_manifest",
    "aggregate_sha_file",
    "bundle_sha_file",
}
# These fields describe an authority; they are not local filesystem paths.
# Keeping the classification explicit avoids interpreting identifiers such as
# ``physical-intelligence/libero`` as a relative path merely because they
# contain a slash.
IDENTIFIER_KEYS = frozenset(
    {
        "dataset_repo_id",
        "dataset_id",
        "repo_id",
        "dataset_revision",
        "revision",
        "resource_id",
        "workspace_id",
        "quota_name",
        "commit",
        "source_commit",
        "scientific_source_commit",
        "stage_s_source_commit",
        "qpilots_commit",
        "openpi_commit",
        "libero_commit",
    }
)
DESCRIPTION_KEYS = frozenset(
    {
        "description",
        "dataset_manifest_sha256_source",
        "dataset_snapshot_contract",
        "dependency_compatibility_contract",
        "stage_s_source_commit_policy",
        "source_commit_policy",
        "source_role",
        "submission_method",
        "code_source_role",
        "registry_manifest_type",
        "terminal_outcome",
        "genealogy_contract",
        "trajectory_contract",
        "replay_gate",
        "success_gate",
        "validation_method",
        "screen_protocol",
    }
)
# A template is a future value, not an input authority.  In particular,
# status_root is materialized after PAI injects RUN_ID.  A placeholder on an
# actual input path (for example dataset_manifest_path) remains a refusal;
# see _field_semantics below.
TEMPLATE_KEYS = frozenset(
    {
        "status_root",
        "log_root_template",
        "log_sha_template",
        "training_pipeline_completion_template",
    }
)
WRITE_ONLY_KEYS = frozenset(
    {
        "write_paths",
        "output_root",
        "log_root",
        "log_sha",
    }
)
ARTIFACT_TOKENS = (
    "model",
    "checkpoint",
    "protocol",
    "report",
    "manifest",
    "acceptance",
    "completion",
)


class IdentityRefusal(RuntimeError):
    """Raised only for programmer-facing API misuse.

    Mismatches are represented in the returned attestation rather than
    raised, so a caller can persist the complete failure evidence when an
    explicit output path was supplied.
    """


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_regular(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def _safe_path(path: Path) -> bool:
    """Reject symlinked or non-canonical authoritative paths."""

    try:
        return _is_regular(path) or (path.is_dir() and not path.is_symlink())
    except OSError:
        return False


def _memory_gib(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and int(value) == value:
        return int(value)
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"\s*([0-9]+)(?:\.?0*)\s*(GiB|Gi|G)\s*", value, re.I)
    return int(match.group(1)) if match else None


def _git_detailed(
    repo: Path,
    *args: str,
    timeout: float = GIT_COMMAND_TIMEOUT_SECONDS,
) -> tuple[bool, str, str]:
    """Run Git while retaining whether failure was command unavailability.

    The old two-value helper collapsed a timeout into ``clean=False``. That
    made a slow but clean CPFS checkout indistinguishable from a dirty one.
    ``reason`` is deliberately coarse and deterministic; command stderr is
    not part of the attestation.
    """

    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={"LC_ALL": "C", "LANG": "C"},
        )
    except subprocess.TimeoutExpired:
        return False, "", "timeout"
    except OSError as exc:
        return False, "", f"oserror:{type(exc).__name__}"
    if proc.returncode != 0:
        return False, proc.stdout.strip(), f"exit:{proc.returncode}"
    return True, proc.stdout.strip(), "ok"


def _git(repo: Path, *args: str) -> tuple[bool, str]:
    """Compatibility wrapper for callers that only need success and stdout."""

    ok, output, _ = _git_detailed(repo, *args)
    return ok, output


def _git_identity(repo: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(repo),
        "exists": repo.is_dir() and not repo.is_symlink(),
        "git": False,
        "head": None,
        "clean": False,
        "status_available": False,
        "status_reason": "not_checked",
        "status_attempts": 0,
    }
    if not record["exists"]:
        record["status_reason"] = "repository_missing"
        return record
    ok, top, reason = _git_detailed(repo, "rev-parse", "--show-toplevel")
    if not ok:
        record["status_reason"] = f"metadata_unavailable:{reason}"
        return record
    ok_head, head, head_reason = _git_detailed(repo, "rev-parse", "HEAD")
    status = ""
    status_reason = "not_checked"
    status_attempts = 0
    ok_status = False
    for attempt in range(1, GIT_STATUS_MAX_ATTEMPTS + 1):
        status_attempts = attempt
        ok_status, status, status_reason = _git_detailed(
            repo,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )
        if ok_status:
            break
        if attempt < GIT_STATUS_MAX_ATTEMPTS:
            time.sleep(GIT_STATUS_RETRY_DELAY_SECONDS)
    record.update(
        {
            "git": True,
            "top_level": top,
            "head": head.lower() if ok_head else None,
            "clean": ok_status and status == "",
            "status_available": ok_status,
            "status_reason": status_reason if not ok_status else "ok",
            "status_attempts": status_attempts,
        }
    )
    if not ok_head:
        record["head_reason"] = head_reason
    return record


def _manifest_entries(path: Path) -> tuple[dict[str, str], list[str]]:
    entries: dict[str, str] = {}
    errors: list[str] = []
    if not _is_regular(path):
        return entries, ["manifest_missing_or_symlinked"]
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        return entries, [f"manifest_read_error:{type(exc).__name__}"]
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        # GNU sha256sum permits one or two spaces and an optional leading '*'
        # for binary mode.  Absolute paths and traversal are never accepted.
        fields = line.split(None, 1)
        if len(fields) != 2:
            errors.append(f"manifest_malformed_line:{number}")
            continue
        digest, relative = fields
        relative = relative.lstrip("*")
        if not HEX64.fullmatch(digest):
            errors.append(f"manifest_invalid_digest:{number}")
            continue
        posix = PurePosixPath(relative)
        if posix.is_absolute() or ".." in posix.parts or str(posix) in ("", "."):
            errors.append(f"manifest_unsafe_path:{number}")
            continue
        safe = posix.as_posix()
        if safe in entries:
            errors.append(f"manifest_duplicate_path:{safe}")
            continue
        entries[safe] = digest.lower()
    return entries, errors


def verify_manifest(manifest: Path) -> dict[str, Any]:
    """Verify a GNU-compatible manifest without following artifact symlinks."""

    record: dict[str, Any] = {
        "path": str(manifest),
        "exists": _is_regular(manifest),
        "valid": False,
        "entries": 0,
        "errors": [],
    }
    entries, errors = _manifest_entries(manifest)
    record["entries"] = len(entries)
    errors = list(errors)
    for relative, expected in sorted(entries.items()):
        target = manifest.parent / relative
        if not _is_regular(target):
            errors.append(f"missing_or_symlinked:{relative}")
            continue
        try:
            actual = sha256_file(target)
        except OSError as exc:
            errors.append(f"hash_error:{relative}:{type(exc).__name__}")
            continue
        if actual != expected:
            errors.append(f"sha256_mismatch:{relative}:{actual}!={expected}")
    record["errors"] = sorted(errors)
    record["valid"] = bool(record["exists"]) and not record["errors"] and bool(entries)
    return record


def _walk(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], str, str, Mapping[str, Any] | None]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_s = str(key)
            if isinstance(child, str):
                yield path + (key_s,), key_s, child, value
            yield from _walk(child, path + (key_s,))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, path + (str(index),))


def _is_authoritative_path_key(key: str) -> bool:
    """Return whether a field is expected to contain an input path.

    A placeholder in an input authority must remain a hard refusal.  This is
    intentionally narrower than PATH_TOKENS: metadata fields such as
    ``dataset_repo_id`` also contain ``repo`` but are identifiers, not paths.
    """

    lower = key.lower()
    return lower in {
        "path",
        "root",
        "repo",
        "runtime_repo",
        "project_dir",
        "command_file",
        "report",
        "model",
        "checkpoint",
    } or lower.endswith(("_path", "_root"))


def _field_semantics(full_path: tuple[str, ...], key: str, value: str) -> str:
    """Classify a string field before applying filesystem path checks.

    Only fields classified as ``path`` are allowed to reach _path_check.
    This is a fail-closed distinction: a real input path remains a path even
    when malformed with an unresolved placeholder, while identifier,
    description, template, and write-only fields are recorded as skipped.
    """

    lower = key.lower()
    if lower in TEMPLATE_KEYS:
        return "template"
    if lower in WRITE_ONLY_KEYS or "write_paths" in full_path:
        return "write-only"
    if lower in IDENTIFIER_KEYS or lower.endswith(("_repo_id", "_revision", "_commit", "_sha256")):
        return "identifier"
    if lower in DESCRIPTION_KEYS or lower.endswith(
        ("_description", "_contract", "_policy", "_role", "_method", "_outcome")
    ):
        return "description"
    if PLACEHOLDER.search(value) and not _is_authoritative_path_key(lower):
        return "template"
    return "path"


def _potential_pathish(key: str, value: str) -> bool:
    """Detect a value that would look path-like without semantic filtering."""

    lower = key.lower()
    if value.startswith(("http://", "https://")):
        return False
    if HEX40.fullmatch(value) or HEX64.fullmatch(value):
        return False
    if not any(token in lower for token in PATH_TOKENS):
        return False
    return value.startswith("/") or "/" in value or value in (".", "..")


def _pathish(key: str, value: str, full_path: tuple[str, ...] = ()) -> bool:
    lower = key.lower()
    if lower in IGNORE_KEYS or lower.endswith("_template"):
        return False
    if _field_semantics(full_path, key, value) != "path":
        return False
    return _potential_pathish(key, value)


def _expected_sha(parent: Mapping[str, Any] | None, key: str) -> str | None:
    if not parent:
        return None
    candidates = [f"{key}_sha256"]
    if key.endswith("_path"):
        candidates.append(f"{key[:-5]}_sha256")
    if key.endswith("_root"):
        candidates.append(f"{key[:-5]}_sha256")
    for candidate in candidates:
        value = parent.get(candidate)
        if isinstance(value, str) and HEX64.fullmatch(value):
            return value.lower()
    return None


def _is_artifact_key(key: str) -> bool:
    lower = key.lower()
    return any(token in lower for token in ARTIFACT_TOKENS)


def _nearest_manifest(path: Path) -> Path | None:
    if path.is_dir():
        candidate = path / "SHA256SUMS"
    else:
        candidate = path.parent / "SHA256SUMS"
    return candidate if candidate.exists() else None


def _path_check(
    *,
    key: str,
    value: str,
    parent: Mapping[str, Any] | None,
    manifest_cache: dict[Path, dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    path = Path(value).expanduser()
    item: dict[str, Any] = {
        "key": key,
        "path": str(path),
        "exists": path.exists(),
        "symlink": path.is_symlink(),
        "kind": "directory" if path.is_dir() else "file",
    }
    errors: list[str] = []
    if not _safe_path(path):
        errors.append(f"path_missing_or_symlinked:{key}:{path}")
        return item, errors

    expected = _expected_sha(parent, key)
    if expected and path.is_file():
        actual = sha256_file(path)
        item.update({"expected_sha256": expected, "observed_sha256": actual, "sha256_match": actual == expected})
        if actual != expected:
            errors.append(f"sha256_mismatch:{key}:{actual}!={expected}")
    elif expected and path.is_dir():
        item["expected_sha256"] = expected
        errors.append(f"sha256_expected_for_directory:{key}:{path}")

    if _is_artifact_key(key):
        # A field explicitly named ``*_manifest_path`` points at the
        # checksum manifest itself (Stage-S uses DATASET_SHA256SUMS), whereas
        # model/checkpoint directories carry a conventional adjacent
        # SHA256SUMS.  Do not require a second manifest beside the manifest
        # file; verify the declared manifest bytes and all of its entries
        # directly instead.
        manifest = path if path.is_file() and "manifest" in key.lower() else _nearest_manifest(path)
        item["manifest"] = str(manifest) if manifest else None
        if manifest is None:
            errors.append(f"artifact_manifest_missing:{key}:{path}")
        else:
            manifest = manifest.resolve()
            if manifest not in manifest_cache:
                manifest_cache[manifest] = verify_manifest(manifest)
            item["manifest_check"] = manifest_cache[manifest]
            if not manifest_cache[manifest]["valid"]:
                errors.append(f"artifact_manifest_invalid:{key}:{manifest}")

    # The protocol authority carries an additional hash for adjacent
    # PROTOCOL.md.  Verify it here so a valid JSON authority cannot point at a
    # silently changed markdown file.
    if path.is_file() and path.name == "FROZEN_PROTOCOL.json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            errors.append(f"protocol_json_unreadable:{path}")
        else:
            for pth, key_name, child, obj in _walk(payload):
                if key_name not in {"protocol_md_path", "protocol_path"} or not child:
                    continue
                protocol_md = Path(child).expanduser()
                if not protocol_md.is_absolute():
                    protocol_md = path.parent / protocol_md
                expected_md = obj.get("protocol_md_sha256") if obj else None
                if not isinstance(expected_md, str) or not HEX64.fullmatch(expected_md):
                    errors.append("protocol_md_sha256_missing")
                    continue
                if not _is_regular(protocol_md):
                    errors.append(f"protocol_md_missing_or_symlinked:{protocol_md}")
                    continue
                actual_md = sha256_file(protocol_md)
                if actual_md != expected_md.lower():
                    errors.append(f"protocol_md_sha256_mismatch:{actual_md}!={expected_md.lower()}")
                break
    return item, errors


def _find_config_repo(config_path: Path) -> dict[str, Any]:
    repo = _git_identity(config_path.parent)
    if repo.get("git") and repo.get("top_level"):
        top = Path(str(repo["top_level"]))
        repo["config_path"] = str(config_path)
        # The top-level identity is the authoritative config source, not the
        # caller's current working directory.
        repo["source_commit"] = repo.get("head")
        repo["source_tree_clean"] = bool(repo.get("clean"))
        repo["repo_path"] = str(top)
    else:
        repo["config_path"] = str(config_path)
    return repo


def _config_expected_commit(config: Mapping[str, Any]) -> str | None:
    values: list[str] = []
    for path, key, value, _ in _walk(config):
        if key.lower() in {"config_source_commit", "config_commit", "config_git_commit"}:
            if HEX40.fullmatch(value):
                values.append(value.lower())
    if not values:
        return None
    if len(set(values)) != 1:
        return "__CONFLICT__"
    return values[0]


def _dependency_bindings(config: Mapping[str, Any]) -> tuple[list[dict[str, str]], list[str]]:
    bindings: list[dict[str, str]] = []
    errors: list[str] = []

    def collect(scope: str, deps: Any) -> None:
        if not isinstance(deps, Mapping):
            errors.append(f"dependency_bindings_invalid:{scope}:not_object")
            return
        for name, item in deps.items():
            label = f"{scope}:{name}"
            if not isinstance(item, Mapping):
                errors.append(f"dependency_binding_invalid:{label}:not_object")
                continue
            path = item.get("path") or item.get("root") or item.get("repo")
            commit = item.get("commit") or item.get("source_commit")
            if not isinstance(path, str) or not path:
                errors.append(f"dependency_path_missing:{label}")
            if not isinstance(commit, str) or not HEX40.fullmatch(commit):
                errors.append(f"dependency_commit_invalid:{label}")
                continue
            if isinstance(path, str) and path:
                bindings.append({"name": str(name), "path": path, "expected_commit": commit.lower()})

    # Preferred final-config form:
    # runtime.dependencies: {name: {path: ..., commit: ...}}
    runtime = config.get("runtime") if isinstance(config.get("runtime"), Mapping) else {}
    deps = runtime.get("dependencies") if isinstance(runtime, Mapping) else None
    if isinstance(deps, Mapping):
        collect("runtime.dependencies", deps)

    # Existing Stage-S configs keep roots in evidence.source_provenance and
    # commit pins alongside them (qpilots_root/qpilots_commit, etc.).
    evidence = config.get("evidence") if isinstance(config.get("evidence"), Mapping) else {}
    provenance = evidence.get("source_provenance") if isinstance(evidence, Mapping) else None
    if isinstance(provenance, Mapping) and isinstance(evidence, Mapping):
        for root_key, path in provenance.items():
            if not isinstance(path, str) or not any(token in str(root_key).lower() for token in ("root", "repo")):
                continue
            stem = str(root_key)
            if stem.endswith("_root"):
                stem = stem[:-5]
            if stem.endswith("_repo"):
                stem = stem[:-5]
            candidates = [f"{stem}_commit", f"{stem}_source_commit"]
            commit = next((evidence.get(c) for c in candidates if isinstance(evidence.get(c), str)), None)
            if not isinstance(commit, str) or not HEX40.fullmatch(commit):
                errors.append(f"dependency_commit_invalid:evidence.source_provenance:{root_key}")
            else:
                bindings.append({"name": stem, "path": path, "expected_commit": commit.lower()})

    # A lightweight generic form also works when dependency pins are under a
    # top-level dependencies object.
    top_deps = config.get("dependencies")
    if isinstance(top_deps, Mapping):
        collect("dependencies", top_deps)
    # Stable ordering and de-duplication make the attestation deterministic.
    unique: dict[tuple[str, str, str], dict[str, str]] = {}
    for binding in bindings:
        key = (binding["name"], binding["path"], binding["expected_commit"])
        unique[key] = binding
    return [unique[key] for key in sorted(unique)], sorted(set(errors))


def _source_commit_consistency(config: Mapping[str, Any], runtime_expected: str | None) -> list[str]:
    errors: list[str] = []
    if not runtime_expected:
        errors.append("runtime_source_commit_missing")
        return errors
    for _, key, value, _ in _walk(config):
        lower = key.lower()
        if lower in {"stage_s_source_commit", "scientific_source_commit"} and HEX40.fullmatch(value):
            if value.lower() != runtime_expected.lower():
                errors.append(f"source_commit_config_mismatch:{key}:{value.lower()}!={runtime_expected.lower()}")
        if lower == "source_commit_policy":
            found = re.findall(r"[0-9a-fA-F]{40}", value)
            if found and found[-1].lower() != runtime_expected.lower():
                errors.append(f"source_commit_policy_mismatch:{found[-1].lower()}!={runtime_expected.lower()}")
    return errors


def _resource_check(config: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    expected = {"worker_count": 1, "gpu": 8, "cpu": 88, "memory_gib": 1400, "shared_memory_gib": 1400}
    observed: list[dict[str, Any]] = []
    errors: list[str] = []

    def inspect(name: str, item: Mapping[str, Any], aliases: Mapping[str, tuple[str, ...]]) -> None:
        row: dict[str, Any] = {"source": name}
        for canonical, keys in aliases.items():
            value = next((item[k] for k in keys if k in item), None)
            if canonical.endswith("memory_gib") or canonical == "memory_gib":
                value = _memory_gib(value)
            row[canonical] = value
            if value is not None and value != expected[canonical]:
                errors.append(f"resource_mismatch:{name}:{canonical}:{value}!={expected[canonical]}")
        observed.append(row)

    aliases = {
        "worker_count": ("workers", "worker_count", "count"),
        "gpu": ("gpu", "gpus", "gpu_count"),
        "cpu": ("cpu", "cpus", "cpu_cores"),
        "memory_gib": ("memory", "memory_gib", "memory_gb"),
        "shared_memory_gib": ("shared_memory", "shared_memory_gib", "shared_memory_gb"),
    }
    for name in ("resource", "worker"):
        item = config.get(name)
        if isinstance(item, Mapping):
            inspect(name, item, aliases)
    evidence = config.get("evidence")
    if isinstance(evidence, Mapping):
        compute = evidence.get("compute_contract")
        if isinstance(compute, Mapping):
            inspect("evidence.compute_contract", compute, aliases)
        authorization = evidence.get("explicit_user_resource_authorization")
        if isinstance(authorization, Mapping):
            inspect("evidence.explicit_user_resource_authorization", authorization, aliases)
    if not any(row.get("gpu") is not None for row in observed):
        errors.append("resource_gpu_missing")
    if not any(row.get("cpu") is not None for row in observed):
        errors.append("resource_cpu_missing")
    if not any(row.get("memory_gib") is not None for row in observed):
        errors.append("resource_memory_missing")
    if not any(row.get("shared_memory_gib") is not None for row in observed):
        errors.append("resource_shared_memory_missing")

    # If the config includes pool identity, contradictory values are refused;
    # omission remains compatible with older final-config layouts.
    pool_expectations = {
        "resource_pool": {"robot", "robot_idle", "exp-robot"},
        "oversold_type": {"AcceptQuotaOverSold"},
    }
    for _, key, value, _ in _walk(config):
        lower = key.lower()
        if lower in pool_expectations and value not in pool_expectations[lower]:
            errors.append(f"resource_pool_identity_mismatch:{key}:{value}")
    return {"expected": expected, "observed": observed}, sorted(set(errors))


def attest_config(config_path: str | Path) -> dict[str, Any]:
    """Return a deterministic identity attestation without writing anything."""

    path = Path(config_path).expanduser().resolve()
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "REFUSED",
        "config_path": str(path),
        "config": {"exists": _is_regular(path)},
        "config_source": {},
        "runtime": {},
        "dependencies": [],
        "artifacts": [],
        "path_skips": [],
        "resource": {},
        "errors": [],
    }
    errors: list[str] = []
    if not _is_regular(path):
        errors.append(f"config_missing_or_symlinked:{path}")
        result["errors"] = errors
        return result
    try:
        raw = path.read_bytes()
        config = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f"config_unreadable:{type(exc).__name__}")
        result["errors"] = errors
        return result
    if not isinstance(config, Mapping):
        errors.append("config_root_must_be_object")
        result["errors"] = errors
        return result
    result["config"].update({"sha256": hashlib.sha256(raw).hexdigest(), "json_valid": True})

    config_source = _find_config_repo(path)
    expected_config_commit = _config_expected_commit(config)
    config_source["expected_commit"] = expected_config_commit
    result["config_source"] = config_source
    if expected_config_commit == "__CONFLICT__":
        errors.append("config_source_commit_conflicting")
    elif not config_source.get("git"):
        errors.append("config_source_git_missing")
    elif not config_source.get("status_available"):
        errors.append(
            "config_source_git_status_unavailable:"
            f"{config_source.get('status_reason', 'unknown')}"
        )
    elif not config_source.get("clean"):
        errors.append("config_source_tree_dirty")
    elif expected_config_commit and config_source.get("head") != expected_config_commit:
        errors.append(f"config_source_commit_mismatch:{config_source.get('head')}!={expected_config_commit}")

    runtime = config.get("runtime")
    if not isinstance(runtime, Mapping):
        runtime = {}
    runtime_repo_value = runtime.get("runtime_repo") or runtime.get("project_dir")
    runtime_expected = runtime.get("source_commit")
    # The C-calibration config predates the common runtime_repo field and
    # records the same authority in evidence.source_provenance.stage_s_repo.
    # Treat that form as an explicit fallback, not as an optional pass: the
    # resulting checkout is still required to be exact and clean.
    runtime_fallback = False
    evidence_for_runtime = config.get("evidence")
    if not isinstance(runtime_repo_value, str) and isinstance(evidence_for_runtime, Mapping):
        source_provenance = evidence_for_runtime.get("source_provenance")
        if isinstance(source_provenance, Mapping):
            for key in ("stage_s_repo", "runtime_repo", "project_dir"):
                candidate = source_provenance.get(key)
                if isinstance(candidate, str):
                    runtime_repo_value = candidate
                    runtime_fallback = True
                    break
    if not isinstance(runtime_expected, str) and isinstance(evidence_for_runtime, Mapping):
        candidate = evidence_for_runtime.get("stage_s_source_commit")
        if isinstance(candidate, str):
            runtime_expected = candidate
    if not isinstance(runtime_expected, str) or not HEX40.fullmatch(runtime_expected):
        runtime_expected = None
    runtime_record: dict[str, Any] = {
        "expected_commit": runtime_expected,
        "source": "evidence.source_provenance.stage_s_repo" if runtime_fallback else "runtime",
    }
    result["runtime"].update(runtime_record)
    if isinstance(runtime_repo_value, str):
        runtime_identity = _git_identity(Path(runtime_repo_value).expanduser())
        runtime_record["repository"] = runtime_identity
        # ``dict.update`` above copied the scalar runtime fields; keep the
        # observed checkout attached to the returned attestation as well.
        result["runtime"]["repository"] = runtime_identity
        if not runtime_identity.get("git"):
            errors.append("runtime_git_missing")
        elif runtime_identity.get("head") != runtime_expected:
            errors.append(f"runtime_source_commit_mismatch:{runtime_identity.get('head')}!={runtime_expected}")
        elif not runtime_identity.get("status_available"):
            errors.append(
                "runtime_git_status_unavailable:"
                f"{runtime_identity.get('status_reason', 'unknown')}"
            )
        elif not runtime_identity.get("clean"):
            errors.append("runtime_tree_dirty")
    else:
        errors.append("runtime_repo_missing")
    errors.extend(_source_commit_consistency(config, runtime_expected))

    manifest_cache: dict[Path, dict[str, Any]] = {}
    path_checks: list[dict[str, Any]] = []
    path_skips: list[dict[str, str]] = []
    seen_paths: set[tuple[str, str]] = set()
    for full_path, key, value, parent in _walk(config):
        semantics = _field_semantics(full_path, key, value)
        if semantics != "path":
            if _potential_pathish(key, value):
                path_skips.append(
                    {
                        "json_path": ".".join(full_path),
                        "key": key,
                        "semantic": semantics,
                    }
                )
            continue
        if not _pathish(key, value, full_path):
            continue
        # Write-only roots and ordinary storage mount paths are not input
        # identity authorities.  They are covered by the PAI resource config,
        # not by this read-only artifact preflight.
        if full_path[:2] == ("storage", "data_sources") or "write_paths" in full_path:
            continue
        pair = (key, value)
        if pair in seen_paths:
            continue
        seen_paths.add(pair)
        item, item_errors = _path_check(
            key=key,
            value=value,
            parent=parent,
            manifest_cache=manifest_cache,
        )
        path_checks.append(item)
        errors.extend(item_errors)
    result["artifacts"] = sorted(path_checks, key=lambda row: (str(row.get("key")), str(row.get("path"))))
    result["path_skips"] = sorted(
        path_skips,
        key=lambda row: (row["json_path"], row["key"], row["semantic"]),
    )

    dependency_records: list[dict[str, Any]] = []
    dependency_bindings, dependency_binding_errors = _dependency_bindings(config)
    errors.extend(dependency_binding_errors)
    for binding in dependency_bindings:
        identity = _git_identity(Path(binding["path"]).expanduser())
        row = dict(binding)
        row["observed"] = identity
        dependency_records.append(row)
        if not identity.get("git"):
            errors.append(f"dependency_git_missing:{binding['name']}:{binding['path']}")
        elif identity.get("head") != binding["expected_commit"]:
            errors.append(
                f"dependency_source_commit_mismatch:{binding['name']}:{identity.get('head')}!={binding['expected_commit']}"
            )
        elif not identity.get("status_available"):
            errors.append(
                "dependency_git_status_unavailable:"
                f"{binding['name']}:{identity.get('status_reason', 'unknown')}"
            )
        elif not identity.get("clean"):
            errors.append(f"dependency_tree_dirty:{binding['name']}:{binding['path']}")
    result["dependencies"] = dependency_records

    resource_record, resource_errors = _resource_check(config)
    result["resource"] = resource_record
    errors.extend(resource_errors)

    # Direct launcher hash is a first-class check, even though the launcher is
    # not required to carry an artifact SHA256SUMS of its own.
    command_file = runtime.get("command_file")
    if isinstance(command_file, str):
        launcher = Path(command_file).expanduser()
        launcher_row: dict[str, Any] = {
            "path": str(launcher),
            "exists": _is_regular(launcher),
            "expected_sha256": runtime.get("command_file_sha256"),
            "payload_expected_sha256": runtime.get("payload_sha256"),
        }
        if not _is_regular(launcher):
            errors.append(f"deployed_launcher_missing_or_symlinked:{launcher}")
        else:
            observed = sha256_file(launcher)
            launcher_row["observed_sha256"] = observed
            for field in ("command_file_sha256", "payload_sha256"):
                expected = runtime.get(field)
                if expected is not None:
                    if not isinstance(expected, str) or not HEX64.fullmatch(expected):
                        errors.append(f"deployed_launcher_invalid_expected_sha:{field}")
                    elif observed != expected.lower():
                        errors.append(f"deployed_launcher_sha256_mismatch:{field}:{observed}!={expected.lower()}")
        result["runtime"]["deployed_launcher"] = launcher_row
    else:
        errors.append("deployed_launcher_path_missing")

    result["errors"] = sorted(set(errors))
    result["status"] = "PASS" if not result["errors"] else "REFUSED"
    return result


def write_attestation(result: Mapping[str, Any], output_path: str | Path) -> Path:
    """Write a deterministic attestation to an explicitly supplied path."""

    output = Path(output_path).expanduser().resolve()
    if output == Path(str(result.get("config_path", ""))).expanduser().resolve():
        raise IdentityRefusal("attestation output cannot overwrite its config")
    if output.exists() and output.is_symlink():
        raise IdentityRefusal("attestation output cannot be a symlink")
    if not output.parent.is_dir():
        raise IdentityRefusal(f"attestation output parent does not exist: {output.parent}")
    payload = json.dumps(dict(result), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    temporary = output.with_name(f".{output.name}.runtime-identity.tmp")
    try:
        temporary.write_text(payload, encoding="utf-8", newline="\n")
        temporary.replace(output)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise IdentityRefusal(f"cannot write attestation: {type(exc).__name__}") from exc
    return output


__all__ = [
    "IdentityRefusal",
    "SCHEMA",
    "attest_config",
    "sha256_file",
    "verify_manifest",
    "write_attestation",
]
