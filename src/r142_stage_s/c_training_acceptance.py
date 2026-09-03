"""Fail-closed acceptance of the terminal Stage-S C training lineage.

The C substrate is a deliberately weak, real OpenPI/LeRobot checkpoint
lineage.  This module is the admission boundary between a completed PAI
training run and C calibration.  It consumes only immutable evidence; it
does not submit PAI, resume a job, import Torch, load a checkpoint, or infer
anything from a partial run.

The caller must provide the exact registry records for one controller run,
a *sanitized* terminal GetJob response and its checksum, the run/status
roots, and the common checkpoint root.  Every path and digest is re-read.
Only after all checks pass is ``ACCEPTED_C_TRAINING.json`` written atomically,
along with a separate ``.sha256`` sidecar.  The output is intentionally
always labelled ``WEAK_SUBSTRATE``.

This file uses the standard library only.  In particular, acceptance can be
performed before importing OpenPI, LeRobot, MuJoCo, or Torch.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "r142-stage-s-c-training-acceptance-v1"
EXPECTED_STATUS = "ACCEPTED"
EXPECTED_LABEL = "WEAK_SUBSTRATE"
EXPECTED_PAI_STATUS = "Succeeded"
EXPECTED_CONFIG = "pi05_libero"
EXPECTED_SEED = 42
EXPECTED_WORLD_SIZE = 8
EXPECTED_TERMINAL_STEP = 10001
EXPECTED_STEPS = (1000, 3000, 6000, 10000)
EXPECTED_FULL_REFERENCE_STEP = 30000
EXPECTED_EXPERIMENT = "r142_stage_s_c_undertrained_seed42"
EXPECTED_OPENPI_COMMIT = "54cbaee6ae0c010a1ed431871cdaa8f4684ac709"
EXPECTED_QPILOTS_COMMIT = "eacf47b981e3b22357f8a74902f8dad8cfcfa375"
EXPECTED_LIBERO_COMMIT = "f78abd68ee283de9fbe3c8f7e2a9ad60246e95c"
EXPECTED_DATASET_REPO = "physical-intelligence/libero"
EXPECTED_DATASET_REVISION = "9dfa69510ea9e1613fc54112bc706444b686a231"
EXPECTED_DATASET_MANIFEST_SHA256 = "02b5b3abfadb65b2f1c4823cfe7ed7b9351416934674fcf59aea1868826546bf"
EXPECTED_DATASET_FILE_COUNT = 1699
EXPECTED_UID = 2254
EXPECTED_GID = 2254

TRAINING_COMPLETION = "COMPLETED_C_TRAINING.json"
PIPELINE_COMPLETION = "COMPLETED_C_PIPELINE.json"
TRAINING_TERMINAL = "TRAINING_TERMINAL.json"
TRAINING_START = "TRAINING_START.json"
DATA_PREFLIGHT = "DATA_PREFLIGHT.json"
RUNTIME_IDENTITY = "RUNTIME_IDENTITY.json"
CHECKPOINT_READY = "CHECKPOINT_READY.json"
COMPLETE_RNG_STATE = "COMPLETE_RNG_STATE.json"
RNG_SHA256SUMS = "RNG_SHA256SUMS"
CORE_CHECKPOINT_FILES = ("model.safetensors", "optimizer.pt", "metadata.pt")

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
RUN_ID_RE = re.compile(r"^r142-stage-s-c-undertrained-20260903-r[0-9]+$")
JOB_ID_RE = re.compile(r"^dlc[0-9a-z]+$")


class CTrainingAcceptanceError(RuntimeError):
    """A C training lineage is absent, partial, stale, or inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise CTrainingAcceptanceError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _read_json(path: str | Path, label: str) -> tuple[Path, dict[str, Any]]:
    target = Path(path).expanduser()
    if target.is_symlink() or not target.is_file():
        raise CTrainingAcceptanceError(f"{label} is missing or symlinked: {target}")
    try:
        resolved = target.resolve(strict=True)
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CTrainingAcceptanceError(f"{label} is invalid JSON: {target}") from exc
    if not isinstance(value, dict):
        raise CTrainingAcceptanceError(f"{label} must be a JSON object: {target}")
    return resolved, value


def _regular_file(path: str | Path, label: str) -> Path:
    target = Path(path).expanduser()
    if target.is_symlink() or not target.is_file():
        raise CTrainingAcceptanceError(f"{label} is missing or symlinked: {target}")
    try:
        return target.resolve(strict=True)
    except OSError as exc:
        raise CTrainingAcceptanceError(f"{label} is unreadable: {target}") from exc


def _regular_dir(path: str | Path, label: str) -> Path:
    target = Path(path).expanduser()
    if target.is_symlink() or not target.is_dir():
        raise CTrainingAcceptanceError(f"{label} is missing or symlinked: {target}")
    try:
        return target.resolve(strict=True)
    except OSError as exc:
        raise CTrainingAcceptanceError(f"{label} is unreadable: {target}") from exc


def _within(path: Path, root: Path, label: str) -> None:
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise CTrainingAcceptanceError(f"{label} escapes its declared root: {path}") from exc


def _full_hex(value: Any, *, length: int, label: str) -> str:
    pattern = HEX40 if length == 40 else HEX64
    if not isinstance(value, str) or pattern.fullmatch(value.lower()) is None:
        raise CTrainingAcceptanceError(f"{label} must be a lowercase full SHA-{length * 4}")
    return value.lower()


def _owner(path: Path, *, uid: int, gid: int, label: str) -> dict[str, int]:
    try:
        info = path.lstat()
    except OSError as exc:
        raise CTrainingAcceptanceError(f"cannot stat {label}: {path}") from exc
    if stat.S_ISLNK(info.st_mode):
        raise CTrainingAcceptanceError(f"{label} is symlinked: {path}")
    if info.st_uid != uid or info.st_gid != gid:
        raise CTrainingAcceptanceError(
            f"{label} ownership is {info.st_uid}:{info.st_gid}, expected {uid}:{gid}: {path}"
        )
    return {"uid": int(info.st_uid), "gid": int(info.st_gid)}


def _audit_tree_ownership(root: Path, *, uid: int, gid: int, label: str) -> int:
    """Audit every node without following symlinks and reject partial names."""

    count = 0
    for path in (root, *sorted(root.rglob("*"))):
        _owner(path, uid=uid, gid=gid, label=label)
        count += 1
        if path.is_symlink():
            raise CTrainingAcceptanceError(f"{label} contains a symlink: {path}")
        name = path.name.lower()
        if (
            name.startswith("failed")
            or name.startswith("refused")
            or name.startswith("stopped")
            or name.startswith("queued")
            or name.startswith("running")
            or name.endswith(".incomplete")
            or name.endswith(".part")
            or ".tmp" in name
        ):
            raise CTrainingAcceptanceError(f"{label} contains partial/failure evidence: {path}")
    return count


def _verify_payload_sha(path: Path, payload: Mapping[str, Any], label: str) -> None:
    declared = payload.get("payload_sha256")
    if declared is None:
        raise CTrainingAcceptanceError(f"{label} lacks required payload_sha256")
    _full_hex(declared, length=64, label=f"{label}.payload_sha256")
    body = dict(payload)
    body.pop("payload_sha256", None)
    observed = hashlib.sha256(_canonical_bytes(body)).hexdigest()
    if observed != declared:
        raise CTrainingAcceptanceError(f"{label} payload_sha256 mismatch")


def _manifest_rows(path: Path, label: str) -> list[tuple[str, str]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise CTrainingAcceptanceError(f"cannot read {label}: {path}") from exc
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for number, raw in enumerate(lines, start=1):
        if not raw.strip():
            continue
        parts = raw.split(None, 1)
        if len(parts) != 2 or HEX64.fullmatch(parts[0].lower()) is None:
            raise CTrainingAcceptanceError(f"malformed {label} line {number}: {raw!r}")
        relative = parts[1].lstrip(" *")
        posix = PurePosixPath(relative)
        if (
            not relative
            or posix.is_absolute()
            or ".." in posix.parts
            or relative != posix.as_posix()
            or relative in seen
            or posix.name == "SHA256SUMS"
        ):
            raise CTrainingAcceptanceError(f"unsafe or duplicate {label} path at line {number}: {relative!r}")
        seen.add(relative)
        rows.append((parts[0].lower(), relative))
    if not rows:
        raise CTrainingAcceptanceError(f"empty {label}: {path}")
    return rows


def _verify_manifest(path: Path, *, root: Path, label: str) -> dict[str, str]:
    """Verify all entries and return relative POSIX path -> SHA-256."""

    path = _regular_file(path, label)
    _within(path, root, label)
    entries: dict[str, str] = {}
    for expected, relative in _manifest_rows(path, label):
        target = root / Path(*PurePosixPath(relative).parts)
        if target.is_symlink() or not target.is_file():
            raise CTrainingAcceptanceError(f"{label} member is missing or symlinked: {target}")
        _within(target, root, f"{label} member")
        observed = sha256_file(target)
        if observed != expected:
            raise CTrainingAcceptanceError(
                f"{label} digest mismatch for {relative}: {observed} != {expected}"
            )
        entries[relative] = expected
    return entries


def _regular_relpaths(root: Path) -> set[str]:
    result: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise CTrainingAcceptanceError(f"symlink in evidence tree: {path}")
        if path.is_file():
            result.add(path.relative_to(root).as_posix())
    return result


def _audit_manifest_coverage(
    root: Path,
    entries: Mapping[str, str],
    *,
    label: str,
    allow_unlisted: Iterable[str] = (),
) -> None:
    listed = set(entries)
    allowed = set(allow_unlisted)
    observed = _regular_relpaths(root)
    observed.discard("SHA256SUMS")
    extras = sorted(observed - listed - allowed)
    missing = sorted(listed - observed)
    if missing:
        raise CTrainingAcceptanceError(f"{label} manifest has missing members: {missing[:8]}")
    if extras:
        raise CTrainingAcceptanceError(f"{label} contains unlisted extra files: {extras[:8]}")


def _self_bound_hash(path: Path, payload: Mapping[str, Any], label: str) -> None:
    _verify_payload_sha(path, payload, label)


def _git(repo: Path, args: Sequence[str], label: str) -> str:
    if repo.is_symlink() or not repo.is_dir():
        raise CTrainingAcceptanceError(f"{label} checkout is missing or symlinked: {repo}")
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "LC_ALL": "C", "LANG": "C"},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CTrainingAcceptanceError(f"{label} Git identity is unavailable: {repo}") from exc
    if proc.returncode != 0:
        raise CTrainingAcceptanceError(f"{label} Git identity command failed: {repo}")
    return proc.stdout.strip()


def _require_clean_git(repo: Path, expected: str, label: str) -> None:
    observed = _git(repo, ("rev-parse", "HEAD"), label).lower()
    if observed != expected:
        raise CTrainingAcceptanceError(f"{label} commit drift: {observed} != {expected}")
    dirty = _git(repo, ("status", "--porcelain=v1", "--untracked-files=all"), label)
    if dirty:
        raise CTrainingAcceptanceError(f"{label} checkout is dirty")


def _path_binding(value: Any, *, label: str, root: Path | None = None, directory: bool | None = None) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise CTrainingAcceptanceError(f"{label} path is missing")
    target = Path(value).expanduser()
    result = _regular_dir(target, label) if directory is True else _regular_file(target, label) if directory is False else _regular_dir(target, label)
    if root is not None:
        _within(result, root, label)
    return result


def _verify_sha_sidecar(path: Path, sidecar: Path, label: str) -> str:
    sidecar = _regular_file(sidecar, f"{label} SHA sidecar")
    rows = sidecar.read_text(encoding="utf-8").splitlines()
    if len(rows) != 1:
        raise CTrainingAcceptanceError(f"{label} SHA sidecar must contain exactly one line")
    parts = rows[0].strip().split(None, 1)
    if len(parts) != 2 or HEX64.fullmatch(parts[0].lower()) is None:
        raise CTrainingAcceptanceError(f"{label} SHA sidecar is malformed")
    declared_name = parts[1].lstrip(" *")
    if Path(declared_name).name != path.name:
        raise CTrainingAcceptanceError(f"{label} SHA sidecar names the wrong file")
    observed = sha256_file(path)
    if observed != parts[0].lower():
        raise CTrainingAcceptanceError(f"{label} SHA sidecar digest mismatch")
    return observed


def _validate_registry(
    *,
    registry_run: str | Path,
    registry_result: str | Path,
    submission_state: str | Path,
    resolved: str | Path,
    jobs_ledger: str | Path,
    terminal_getjob: str | Path,
    terminal_getjob_sha: str | Path,
    c_run_root: Path,
    c_status_root: Path,
    checkpoint_root: Path,
    uid: int,
    gid: int,
) -> dict[str, Any]:
    run_dir = _regular_dir(registry_run, "C registry run")
    result_path, result = _read_json(registry_result, "C registry result")
    state_path, state = _read_json(submission_state, "C registry submission state")
    resolved_path, resolved_payload = _read_json(resolved, "C registry resolved payload")
    if result_path.parent != run_dir or state_path.parent != run_dir or resolved_path.parent != run_dir:
        raise CTrainingAcceptanceError("registry result/state/resolved must be inside the exact registry run directory")

    run_id = result.get("run_id")
    job_id = result.get("job_id")
    if not isinstance(run_id, str) or RUN_ID_RE.fullmatch(run_id) is None:
        raise CTrainingAcceptanceError("registry result has an invalid C run id")
    if run_dir.name != run_id:
        raise CTrainingAcceptanceError("registry run directory does not match result run_id")
    if not isinstance(job_id, str) or JOB_ID_RE.fullmatch(job_id) is None:
        raise CTrainingAcceptanceError("registry result has an invalid PAI JobId")
    if result.get("submission_state") != "submitted_verified":
        raise CTrainingAcceptanceError("registry result is not submitted_verified")
    if result.get("returncode", 0) != 0:
        raise CTrainingAcceptanceError("registry result has a non-zero return code")

    state_run_id = state.get("run_id", run_id)
    state_job_id = state.get("job_id")
    state_value = state.get("state", state.get("submission_state"))
    if state_run_id != run_id or state_job_id != job_id or state_value != "submitted_verified":
        raise CTrainingAcceptanceError("registry submission state does not bind run, JobId, and submitted_verified")
    if resolved_payload.get("run_id") != run_id:
        raise CTrainingAcceptanceError("registry resolved payload run_id drifted")

    resolved_artifact = resolved_payload.get("artifact_dir")
    artifact_dir: Path | None = None
    if isinstance(resolved_artifact, str) and resolved_artifact:
        artifact_dir = _regular_dir(resolved_artifact, "registry artifact_dir")
        _owner(artifact_dir, uid=uid, gid=gid, label="registry artifact_dir")

    runtime = resolved_payload.get("runtime")
    if not isinstance(runtime, Mapping):
        raise CTrainingAcceptanceError("registry resolved payload lacks runtime identity")
    write_paths = runtime.get("write_paths")
    if not isinstance(write_paths, list) or any(not isinstance(item, str) for item in write_paths):
        raise CTrainingAcceptanceError("registry runtime write_paths are missing or malformed")
    concrete_write_paths = [Path(item).expanduser() for item in write_paths if "{{" not in item]
    for required, label in (
        (c_run_root, "C run root"),
        (c_status_root, "C status root"),
        (checkpoint_root, "C checkpoint root"),
    ):
        if not any(
            candidate == required or candidate in required.parents
            for candidate in concrete_write_paths
        ):
            raise CTrainingAcceptanceError(f"resolved runtime write_paths do not cover {label}")
    if artifact_dir is not None and not any(
        candidate == artifact_dir or candidate in artifact_dir.parents for candidate in concrete_write_paths
    ):
        raise CTrainingAcceptanceError("resolved runtime write_paths do not cover artifact_dir")

    worker = resolved_payload.get("worker")
    if isinstance(worker, Mapping):
        expected_worker = {"count": 1, "gpu": 8, "cpu": 88, "memory": "1400Gi", "shared_memory": "1400Gi"}
        for key, value in expected_worker.items():
            if worker.get(key) != value:
                raise CTrainingAcceptanceError(f"registry worker {key} drifted")
    resource = resolved_payload.get("resource")
    if isinstance(resource, Mapping):
        target = resource.get("target_worker_contract")
        if isinstance(target, Mapping):
            for key, value in {"count": 1, "gpu": 8, "cpu": 88, "memory": "1400Gi", "shared_memory": "1400Gi"}.items():
                if target.get(key) != value:
                    raise CTrainingAcceptanceError(f"registry target worker {key} drifted")

    ledger_path = _regular_file(jobs_ledger, "C PAI jobs ledger")
    matches: list[dict[str, Any]] = []
    job_matches: list[dict[str, Any]] = []
    try:
        ledger_lines = ledger_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise CTrainingAcceptanceError(f"cannot read C PAI jobs ledger: {ledger_path}") from exc
    for number, raw in enumerate(ledger_lines, start=1):
        if not raw.strip():
            continue
        try:
            entry = json.loads(raw)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise CTrainingAcceptanceError(f"C PAI jobs ledger line {number} is invalid JSON") from exc
        if not isinstance(entry, Mapping):
            raise CTrainingAcceptanceError(f"C PAI jobs ledger line {number} is not an object")
        if entry.get("run_id") == run_id:
            matches.append(dict(entry))
        if entry.get("job_id") == job_id:
            job_matches.append(dict(entry))
    if len(matches) != 1 or matches[0].get("job_id") != job_id:
        raise CTrainingAcceptanceError("C PAI jobs ledger must contain exactly one run-to-JobId binding")
    if len(job_matches) != 1:
        raise CTrainingAcceptanceError("C PAI JobId is not unique in the jobs ledger")

    getjob_path = _regular_file(terminal_getjob, "sanitized terminal GetJob")
    getjob_sha_path = _regular_file(terminal_getjob_sha, "sanitized terminal GetJob SHA")
    getjob_digest = _verify_sha_sidecar(getjob_path, getjob_sha_path, "sanitized terminal GetJob")
    _, getjob = _read_json(getjob_path, "sanitized terminal GetJob")
    sensitive = {"usercommand", "command", "envs", "environment", "password", "token", "secret", "wandb_api_key", "payload_base64"}
    if any(str(key).lower().replace("_", "") in {item.replace("_", "") for item in sensitive} for key in getjob):
        raise CTrainingAcceptanceError("terminal GetJob is not sanitized")
    observed_job = getjob.get("JobId", getjob.get("job_id"))
    observed_status = getjob.get("Status", getjob.get("status"))
    reason = getjob.get("ReasonCode", getjob.get("reason_code"))
    if observed_job != job_id or observed_status != EXPECTED_PAI_STATUS:
        raise CTrainingAcceptanceError("sanitized terminal GetJob is not Succeeded for the bound JobId")
    if reason is not None and reason not in {"JobSucceeded", "Succeeded", "Success"}:
        raise CTrainingAcceptanceError("sanitized terminal GetJob reason is not successful")

    return {
        "run_id": run_id,
        "job_id": job_id,
        "registry_run": str(run_dir),
        "registry_result": {"path": str(result_path), "sha256": sha256_file(result_path)},
        "submission_state": {"path": str(state_path), "sha256": sha256_file(state_path)},
        "resolved": {"path": str(resolved_path), "sha256": sha256_file(resolved_path)},
        "jobs_ledger": {"path": str(ledger_path), "sha256": sha256_file(ledger_path)},
        "terminal_getjob": {
            "path": str(getjob_path),
            "sha256": getjob_digest,
            "sha_sidecar": str(getjob_sha_path),
            "status": EXPECTED_PAI_STATUS,
        },
        "resolved_runtime": {
            "project_dir": runtime.get("project_dir"),
            "command_file": runtime.get("command_file"),
            "command_file_sha256": runtime.get("command_file_sha256"),
            "payload_sha256": runtime.get("payload_sha256"),
            "qpilots_commit": runtime.get("qpilots_commit"),
            "openpi_commit": runtime.get("openpi_commit"),
            "uid": runtime.get("uid"),
            "gid": runtime.get("gid"),
            "output_mode": runtime.get("output_mode"),
            "create_artifact_dir": runtime.get("create_artifact_dir"),
            "recursive_repair": runtime.get("recursive_repair"),
        },
    }


def _validate_runtime_identity(
    *,
    path: Path,
    run_id: str,
    job_id: str,
    resolved_runtime: Mapping[str, Any],
    uid: int,
    gid: int,
) -> tuple[dict[str, Any], dict[str, str]]:
    _, payload = _read_json(path, "C runtime identity")
    if payload.get("schema") not in {"r142-stage-s-c-runtime-identity-v1", "r142-stage-s-c-runtime-identity-v2"}:
        raise CTrainingAcceptanceError("C runtime identity schema drifted")
    for key, expected in (("run_id", run_id), ("job_id", job_id), ("qpilots_commit", EXPECTED_QPILOTS_COMMIT), ("openpi_commit", EXPECTED_OPENPI_COMMIT)):
        if payload.get(key) != expected:
            raise CTrainingAcceptanceError(f"C runtime identity {key} drifted")
    stage_commit = _full_hex(payload.get("stage_s_source_commit"), length=40, label="C Stage-S runtime commit")
    project_dir = _regular_dir(payload.get("project_dir", ""), "C Stage-S runtime project")
    qpilots_root = _regular_dir(payload.get("qpilots_root", ""), "C QPILOTS root")
    openpi_root = _regular_dir(payload.get("openpi_root", ""), "C OpenPI root")
    if openpi_root != (qpilots_root / "third_party/openpi").resolve():
        raise CTrainingAcceptanceError("C OpenPI root is not the QPILOTS third_party/openpi checkout")
    _require_clean_git(project_dir, stage_commit, "C Stage-S runtime")
    _require_clean_git(qpilots_root, EXPECTED_QPILOTS_COMMIT, "C QPILOTS")
    _require_clean_git(openpi_root, EXPECTED_OPENPI_COMMIT, "C OpenPI")
    libero_root = _regular_dir(openpi_root / "third_party/libero", "C LIBERO source")
    _require_clean_git(libero_root, EXPECTED_LIBERO_COMMIT, "C LIBERO")

    payload_path = _regular_file(payload.get("payload_path", ""), "C deployed payload")
    payload_sha = _full_hex(payload.get("payload_sha256"), length=64, label="C deployed payload SHA")
    if sha256_file(payload_path) != payload_sha:
        raise CTrainingAcceptanceError("C deployed payload SHA mismatch")
    observed_payload_sha = payload.get("payload_sha256_observed")
    if observed_payload_sha != payload_sha:
        raise CTrainingAcceptanceError("C runtime identity observed payload SHA mismatch")
    runtime_payload = resolved_runtime.get("payload_sha256")
    runtime_command_sha = resolved_runtime.get("command_file_sha256")
    if runtime_payload != payload_sha or runtime_command_sha != payload_sha:
        raise CTrainingAcceptanceError("registry/runtime payload identity drifted")
    if resolved_runtime.get("command_file") != str(payload_path):
        raise CTrainingAcceptanceError("registry command_file does not match runtime payload")
    if resolved_runtime.get("project_dir") != str(project_dir):
        raise CTrainingAcceptanceError("registry project_dir does not match C runtime identity")
    if resolved_runtime.get("qpilots_commit") != EXPECTED_QPILOTS_COMMIT or resolved_runtime.get("openpi_commit") != EXPECTED_OPENPI_COMMIT:
        raise CTrainingAcceptanceError("registry dependency commit identity drifted")
    if resolved_runtime.get("uid") not in (None, uid) or resolved_runtime.get("gid") not in (None, gid):
        raise CTrainingAcceptanceError("registry runtime UID/GID drifted")
    if resolved_runtime.get("output_mode") not in (None, "resume") or resolved_runtime.get("create_artifact_dir") not in (None, True):
        raise CTrainingAcceptanceError("registry runtime resume/artifact contract drifted")
    if resolved_runtime.get("recursive_repair") not in (None, False):
        raise CTrainingAcceptanceError("registry runtime permits recursive repair")
    _owner(path, uid=uid, gid=gid, label="C runtime identity")
    return (
        {
            "path": str(path),
            "sha256": sha256_file(path),
            "stage_s_source_commit": stage_commit,
            "qpilots_root": str(qpilots_root),
            "qpilots_commit": EXPECTED_QPILOTS_COMMIT,
            "openpi_root": str(openpi_root),
            "openpi_commit": EXPECTED_OPENPI_COMMIT,
            "libero_root": str(libero_root),
            "libero_commit": EXPECTED_LIBERO_COMMIT,
            "payload_path": str(payload_path),
            "payload_sha256": payload_sha,
        },
        {
            "stage_s_commit": stage_commit,
            "qpilots_commit": EXPECTED_QPILOTS_COMMIT,
            "openpi_commit": EXPECTED_OPENPI_COMMIT,
            "libero_commit": EXPECTED_LIBERO_COMMIT,
        },
    )


def _validate_data_preflight(path: Path, *, runtime_identity: Mapping[str, Any], uid: int, gid: int) -> dict[str, Any]:
    _, payload = _read_json(path, "C data preflight")
    if payload.get("schema") != "r142-stage-s-c-data-preflight-v2" or payload.get("status") != "COMPLETED":
        raise CTrainingAcceptanceError("C data preflight is not COMPLETED v2")
    _self_bound_hash(path, payload, "C data preflight")
    if payload.get("no_network_fallback") is not True or payload.get("no_pai_submit_performed") is not True:
        raise CTrainingAcceptanceError("C data preflight permits network/Pai fallback")
    dataset = payload.get("dataset")
    norm_stats = payload.get("norm_stats")
    official = payload.get("official_bindings")
    if not isinstance(dataset, Mapping) or not isinstance(norm_stats, Mapping) or not isinstance(official, Mapping):
        raise CTrainingAcceptanceError("C data preflight lacks dataset/norm_stats/official bindings")
    if dataset.get("valid") is not True or dataset.get("repo_id") != EXPECTED_DATASET_REPO or dataset.get("revision") != EXPECTED_DATASET_REVISION:
        raise CTrainingAcceptanceError("official LIBERO dataset identity is invalid")
    dataset_root = _regular_dir(dataset.get("root", ""), "official LIBERO dataset root")
    dataset_manifest = _regular_file(dataset.get("manifest_path", ""), "official LIBERO dataset SHA manifest")
    _within(dataset_manifest, dataset_root, "official LIBERO dataset SHA manifest")
    manifest_digest = sha256_file(dataset_manifest)
    if manifest_digest != EXPECTED_DATASET_MANIFEST_SHA256 or dataset.get("manifest_sha256") != manifest_digest or dataset.get("manifest_file_sha256") != manifest_digest:
        raise CTrainingAcceptanceError("official LIBERO dataset manifest SHA drifted")
    dataset_entries = _verify_manifest(dataset_manifest, root=dataset_root, label="official LIBERO dataset SHA manifest")
    if dataset.get("file_count") != len(dataset_entries) or len(dataset_entries) != EXPECTED_DATASET_FILE_COUNT:
        raise CTrainingAcceptanceError("official LIBERO dataset file count drifted")
    _owner(dataset_manifest, uid=uid, gid=gid, label="official LIBERO dataset manifest")

    if norm_stats.get("valid") is not True:
        raise CTrainingAcceptanceError("official LIBERO norm stats are not valid")
    source = _regular_file(norm_stats.get("source_path", ""), "LIBERO norm-stats source")
    staged = _regular_file(norm_stats.get("staged_path", ""), "LIBERO norm-stats staged file")
    if source == staged:
        raise CTrainingAcceptanceError("LIBERO norm-stats source and staged files must be distinct")
    source_sha = sha256_file(source)
    staged_sha = sha256_file(staged)
    if source_sha != staged_sha or norm_stats.get("source_sha256") != source_sha or norm_stats.get("staged_sha256") != staged_sha:
        raise CTrainingAcceptanceError("LIBERO norm-stats source/staged SHA mismatch")
    for candidate, label in ((source, "LIBERO norm-stats source"), (staged, "LIBERO norm-stats staged file")):
        try:
            value = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CTrainingAcceptanceError(f"{label} is invalid JSON") from exc
        if not isinstance(value, dict) or not value:
            raise CTrainingAcceptanceError(f"{label} is not a non-empty JSON object")
        _owner(candidate, uid=uid, gid=gid, label=label)
    compatibility = official.get("lerobot_compatibility")
    if not isinstance(compatibility, Mapping) or compatibility.get("valid") is not True:
        raise CTrainingAcceptanceError("official LeRobot compatibility gate is not valid")
    if compatibility.get("lerobot_version") != "0.1.0" or compatibility.get("lerobot_commit") != "0cf864870cf29f4738d3ade893e6fd13fbd7cdb5":
        raise CTrainingAcceptanceError("LeRobot source identity drifted")
    if compatibility.get("datasets_version") not in {"3.6.0", "4.8.4"}:
        raise CTrainingAcceptanceError("datasets version is outside the audited compatibility contract")
    if compatibility.get("datasets_version") == "3.6.0" and compatibility.get("mode") != "native-pinned-datasets":
        raise CTrainingAcceptanceError("native datasets compatibility mode drifted")
    if compatibility.get("datasets_version") == "4.8.4" and compatibility.get("mode") != "datasets-column-numeric-bridge":
        raise CTrainingAcceptanceError("datasets column bridge compatibility mode drifted")
    resolver_path = official.get("resolver_path")
    if resolver_path != str(staged):
        raise CTrainingAcceptanceError("official OpenPI norm-stat resolver did not use staged file")
    if official.get("config_name") != EXPECTED_CONFIG or official.get("repo_id") != EXPECTED_DATASET_REPO:
        raise CTrainingAcceptanceError("official OpenPI data config identity drifted")

    for key in ("data_preflight_path", "data_preflight_sha256"):
        if key == "data_preflight_path" and runtime_identity.get(key) != str(path):
            raise CTrainingAcceptanceError("runtime identity data preflight path drifted")
        if key == "data_preflight_sha256" and runtime_identity.get(key) != sha256_file(path):
            raise CTrainingAcceptanceError("runtime identity data preflight SHA drifted")
    if runtime_identity.get("dataset_repo_id") != EXPECTED_DATASET_REPO or runtime_identity.get("dataset_revision") != EXPECTED_DATASET_REVISION:
        raise CTrainingAcceptanceError("runtime identity dataset identity drifted")
    if runtime_identity.get("dataset_manifest_sha256") != manifest_digest or runtime_identity.get("dataset_manifest_file_sha256") != manifest_digest:
        raise CTrainingAcceptanceError("runtime identity dataset manifest SHA drifted")
    if runtime_identity.get("norm_stats_source_sha256") != source_sha or runtime_identity.get("norm_stats_sha256") != staged_sha:
        raise CTrainingAcceptanceError("runtime identity norm-stats SHA drifted")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "dataset_root": str(dataset_root),
        "dataset_manifest": str(dataset_manifest),
        "dataset_manifest_sha256": manifest_digest,
        "dataset_file_count": len(dataset_entries),
        "dataset_repo_id": EXPECTED_DATASET_REPO,
        "dataset_revision": EXPECTED_DATASET_REVISION,
        "norm_stats_source": str(source),
        "norm_stats_source_sha256": source_sha,
        "norm_stats_staged": str(staged),
        "norm_stats_staged_sha256": staged_sha,
        "lerobot_version": compatibility.get("lerobot_version"),
        "lerobot_commit": compatibility.get("lerobot_commit"),
        "datasets_version": compatibility.get("datasets_version"),
        "lerobot_compatibility_mode": compatibility.get("mode"),
        "resolver_path": str(staged),
    }


def _validate_checkpoint_ready(step_dir: Path, step: int) -> dict[str, Any]:
    marker_path = _regular_file(step_dir / CHECKPOINT_READY, f"C checkpoint {step} READY marker")
    _, marker = _read_json(marker_path, f"C checkpoint {step} READY marker")
    if marker.get("schema") != "r142-stage-s-c-checkpoint-ready-v1" or marker.get("status") != "READY":
        raise CTrainingAcceptanceError(f"C checkpoint {step} READY schema/status drifted")
    if marker.get("global_step") != step or marker.get("world_size") != EXPECTED_WORLD_SIZE or marker.get("checkpoint_dir") != str(step_dir):
        raise CTrainingAcceptanceError(f"C checkpoint {step} READY binding drifted")
    entries = marker.get("core_files")
    if not isinstance(entries, list) or [item.get("name") for item in entries if isinstance(item, Mapping)] != list(CORE_CHECKPOINT_FILES):
        raise CTrainingAcceptanceError(f"C checkpoint {step} READY core-file inventory drifted")
    for item in entries:
        if not isinstance(item, Mapping) or item.get("exists") is not True:
            raise CTrainingAcceptanceError(f"C checkpoint {step} READY core-file evidence is invalid")
        path = _regular_file(step_dir / str(item.get("name")), f"C checkpoint {step} core file")
        if path.stat().st_size <= 0 or int(item.get("size", -1)) != path.stat().st_size:
            raise CTrainingAcceptanceError(f"C checkpoint {step} READY core-file size drifted")
        if item.get("sha256") is not None and item.get("sha256") != sha256_file(path):
            raise CTrainingAcceptanceError(f"C checkpoint {step} READY core-file SHA drifted")
    return {
        "path": str(marker_path),
        "sha256": sha256_file(marker_path),
        "core_files": [str(item["name"]) for item in entries],
    }


def _validate_rng_state(step_dir: Path, step: int) -> dict[str, Any]:
    sidecars = [f"rng_state.rank{rank}.pt" for rank in range(EXPECTED_WORLD_SIZE)]
    for name in sidecars:
        path = _regular_file(step_dir / name, f"C checkpoint {step} RNG sidecar {name}")
        if path.stat().st_size <= 0:
            raise CTrainingAcceptanceError(f"C checkpoint {step} RNG sidecar is empty: {name}")
    sums = _regular_file(step_dir / RNG_SHA256SUMS, f"C checkpoint {step} RNG SHA manifest")
    rows = sums.read_text(encoding="utf-8").splitlines()
    if len(rows) != EXPECTED_WORLD_SIZE:
        raise CTrainingAcceptanceError(f"C checkpoint {step} RNG SHA manifest width drifted")
    hashes: dict[str, str] = {}
    for expected_name, raw in zip(sidecars, rows, strict=True):
        parts = raw.strip().split(None, 1)
        if len(parts) != 2 or parts[1].lstrip(" *") != expected_name or HEX64.fullmatch(parts[0].lower()) is None:
            raise CTrainingAcceptanceError(f"C checkpoint {step} RNG SHA line drifted: {raw!r}")
        observed = sha256_file(step_dir / expected_name)
        if observed != parts[0].lower():
            raise CTrainingAcceptanceError(f"C checkpoint {step} RNG SHA mismatch: {expected_name}")
        hashes[expected_name] = observed
    complete_path = _regular_file(step_dir / COMPLETE_RNG_STATE, f"C checkpoint {step} RNG completion")
    _, complete = _read_json(complete_path, f"C checkpoint {step} RNG completion")
    if (
        complete.get("schema") != "r142-stage-s-c-complete-rng-state-v1"
        or complete.get("status") != "COMPLETED"
        or complete.get("global_step") != step
        or complete.get("world_size") != EXPECTED_WORLD_SIZE
        or complete.get("sidecars") != sidecars
        or complete.get("rng_sha256sums") != RNG_SHA256SUMS
        or complete.get("rng_sha256sums_sha256") != sha256_file(sums)
    ):
        raise CTrainingAcceptanceError(f"C checkpoint {step} RNG completion binding drifted")
    return {
        "sidecars": sidecars,
        "rng_sha256sums": str(sums),
        "rng_sha256sums_sha256": sha256_file(sums),
        "complete": str(complete_path),
        "complete_sha256": sha256_file(complete_path),
        "sidecar_sha256": hashes,
    }


def _validate_checkpoints(
    *,
    checkpoint_root: Path,
    completion_path: Path,
    run_root: Path,
    uid: int,
    gid: int,
) -> tuple[dict[str, str], dict[str, Any], Path, Path]:
    train_dir = _regular_dir(
        checkpoint_root / EXPECTED_CONFIG / EXPECTED_EXPERIMENT,
        "C native checkpoint directory",
    )
    numeric_dirs = sorted(
        child for child in train_dir.iterdir() if child.is_dir() and child.name.isdigit()
    )
    present_steps = {int(child.name) for child in numeric_dirs}
    if not set(EXPECTED_STEPS).issubset(present_steps):
        raise CTrainingAcceptanceError(f"C native checkpoint set is missing required steps: {sorted(set(EXPECTED_STEPS) - present_steps)}")
    # The C contract freezes the retained schedule exactly.  Even a complete
    # extra numeric directory is rejected: accepting it would make an
    # interrupted or stale lineage indistinguishable from the planned run.
    for extra in sorted(present_steps - set(EXPECTED_STEPS)):
        raise CTrainingAcceptanceError(f"extra C checkpoint directory is not allowed: {extra}")
    non_numeric_dirs = sorted(
        child.name for child in train_dir.iterdir() if child.is_dir() and not child.name.isdigit()
    )
    if non_numeric_dirs:
        raise CTrainingAcceptanceError(f"C native checkpoint directory has extra non-step entries: {non_numeric_dirs[:8]}")

    completion, completion_payload = _read_json(completion_path, "COMPLETED_C_TRAINING")
    if completion != completion_path or completion.parent != checkpoint_root:
        raise CTrainingAcceptanceError("COMPLETED_C_TRAINING must be directly under checkpoint root")
    if completion_payload.get("schema") != "r142-stage-s-c-training-completion-v1" or completion_payload.get("status") != "COMPLETED":
        raise CTrainingAcceptanceError("COMPLETED_C_TRAINING is not terminal COMPLETED")
    if completion_payload.get("openpi_commit") != EXPECTED_OPENPI_COMMIT or completion_payload.get("config_name") != EXPECTED_CONFIG:
        raise CTrainingAcceptanceError("COMPLETED_C_TRAINING OpenPI/config identity drifted")
    if completion_payload.get("seed") != EXPECTED_SEED or completion_payload.get("terminal_global_step") != EXPECTED_TERMINAL_STEP or completion_payload.get("checkpoint_steps") != list(EXPECTED_STEPS):
        raise CTrainingAcceptanceError("COMPLETED_C_TRAINING seed/terminal/checkpoint schedule drifted")
    audit = completion_payload.get("checkpoint_audit")
    if not isinstance(audit, Mapping) or audit.get("valid") is not True:
        raise CTrainingAcceptanceError("COMPLETED_C_TRAINING lacks a valid checkpoint audit")
    _self_bound_hash(completion, completion_payload, "COMPLETED_C_TRAINING")

    checkpoint_manifest_value = completion_payload.get("sha256sums")
    checkpoint_manifest = _regular_file(checkpoint_manifest_value or checkpoint_root / "SHA256SUMS", "C checkpoint SHA256SUMS")
    if checkpoint_manifest != checkpoint_root / "SHA256SUMS":
        raise CTrainingAcceptanceError("C checkpoint SHA manifest path drifted")
    declared_manifest_sha = completion_payload.get("sha256sums_sha256")
    if declared_manifest_sha != sha256_file(checkpoint_manifest):
        raise CTrainingAcceptanceError("C checkpoint SHA manifest digest drifted")
    entries = _verify_manifest(checkpoint_manifest, root=checkpoint_root, label="C checkpoint SHA256SUMS")
    _audit_manifest_coverage(checkpoint_root, entries, label="C checkpoint bundle", allow_unlisted=(TRAINING_COMPLETION,))
    expected_entries = {
        f"{EXPECTED_CONFIG}/{EXPECTED_EXPERIMENT}/{step}/{name}"
        for step in EXPECTED_STEPS
        for name in (
            *CORE_CHECKPOINT_FILES,
            CHECKPOINT_READY,
            *(f"rng_state.rank{rank}.pt" for rank in range(EXPECTED_WORLD_SIZE)),
            RNG_SHA256SUMS,
            COMPLETE_RNG_STATE,
        )
    }
    if set(entries) != expected_entries:
        missing = sorted(expected_entries - set(entries))
        extra = sorted(set(entries) - expected_entries)
        raise CTrainingAcceptanceError(
            f"C checkpoint SHA manifest has non-exact component inventory: missing={missing[:8]} extra={extra[:8]}"
        )
    _audit_tree_ownership(checkpoint_root, uid=uid, gid=gid, label="C checkpoint bundle")

    model_hashes: dict[str, str] = {}
    state: dict[str, Any] = {}
    for step in EXPECTED_STEPS:
        step_dir = train_dir / str(step)
        for name in CORE_CHECKPOINT_FILES:
            path = _regular_file(step_dir / name, f"C checkpoint {step} {name}")
            if path.stat().st_size <= 0:
                raise CTrainingAcceptanceError(f"C checkpoint {step} {name} is empty")
            relative = f"{EXPECTED_CONFIG}/{EXPECTED_EXPERIMENT}/{step}/{name}"
            if entries.get(relative) != sha256_file(path):
                raise CTrainingAcceptanceError(f"C checkpoint SHA manifest does not bind {relative}")
        ready = _validate_checkpoint_ready(step_dir, step)
        rng = _validate_rng_state(step_dir, step)
        for relative in (
            f"{EXPECTED_CONFIG}/{EXPECTED_EXPERIMENT}/{step}/{CHECKPOINT_READY}",
            f"{EXPECTED_CONFIG}/{EXPECTED_EXPERIMENT}/{step}/{COMPLETE_RNG_STATE}",
            f"{EXPECTED_CONFIG}/{EXPECTED_EXPERIMENT}/{step}/{RNG_SHA256SUMS}",
        ):
            if relative not in entries:
                raise CTrainingAcceptanceError(f"C checkpoint SHA manifest does not bind {relative}")
        model_relative = f"{step}/model.safetensors"
        model_path = step_dir / "model.safetensors"
        model_hashes[model_relative] = sha256_file(model_path)
        state[str(step)] = {"ready": ready, "rng": rng, "model_sha256": model_hashes[model_relative]}

    log_manifest_value = completion_payload.get("log_sha256sums")
    if not isinstance(log_manifest_value, str):
        raise CTrainingAcceptanceError("COMPLETED_C_TRAINING lacks log SHA manifest path")
    log_manifest = _regular_file(log_manifest_value, "C log SHA256SUMS")
    if log_manifest != run_root / "SHA256SUMS":
        raise CTrainingAcceptanceError("C log SHA manifest path does not match C run root")
    declared_log_sha = completion_payload.get("log_sha256sums_sha256")
    if declared_log_sha != sha256_file(log_manifest):
        raise CTrainingAcceptanceError("C log SHA manifest digest drifted")
    return model_hashes, state, checkpoint_manifest, log_manifest


def _validate_run_and_status(
    *,
    c_run_root: Path,
    c_status_root: Path,
    run_id: str,
    job_id: str,
    checkpoint_root: Path,
    checkpoint_completion: Path,
    uid: int,
    gid: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    if c_run_root.name != run_id or c_status_root.name != run_id:
        raise CTrainingAcceptanceError("C run/status root basename does not match accepted run id")
    _owner(c_run_root, uid=uid, gid=gid, label="C run root")
    _owner(c_status_root, uid=uid, gid=gid, label="C status root")
    terminal_path, terminal = _read_json(c_run_root / TRAINING_TERMINAL, "C TRAINING_TERMINAL")
    if terminal_path.parent != c_run_root or terminal.get("schema") != "r142-stage-s-c-training-terminal-v1" or terminal.get("status") != "COMPLETED":
        raise CTrainingAcceptanceError("C TRAINING_TERMINAL is not terminal COMPLETED")
    if terminal.get("openpi_commit") != EXPECTED_OPENPI_COMMIT or terminal.get("config_name") != EXPECTED_CONFIG or terminal.get("global_step") != EXPECTED_TERMINAL_STEP or terminal.get("checkpoint_steps") != list(EXPECTED_STEPS):
        raise CTrainingAcceptanceError("C TRAINING_TERMINAL identity/schedule drifted")
    _self_bound_hash(terminal_path, terminal, "C TRAINING_TERMINAL")

    start_path, start = _read_json(c_run_root / TRAINING_START, "C TRAINING_START")
    if start_path.parent != c_run_root or start.get("openpi_commit") != EXPECTED_OPENPI_COMMIT or start.get("config_name") != EXPECTED_CONFIG:
        raise CTrainingAcceptanceError("C TRAINING_START identity drifted")
    if start.get("checkpoint_base_dir") not in (None, str(checkpoint_root)):
        raise CTrainingAcceptanceError("C TRAINING_START checkpoint root drifted")
    _self_bound_hash(start_path, start, "C TRAINING_START")

    # The log SHA is written after the terminal marker.  It must cover all
    # regular files in the run root, including the terminal and start records.
    log_manifest = _regular_file(c_run_root / "SHA256SUMS", "C log SHA256SUMS")
    log_entries = _verify_manifest(log_manifest, root=c_run_root, label="C log SHA256SUMS")
    _audit_manifest_coverage(c_run_root, log_entries, label="C log bundle")
    _audit_tree_ownership(c_run_root, uid=uid, gid=gid, label="C log bundle")
    if set(log_entries) != {TRAINING_START, TRAINING_TERMINAL}:
        raise CTrainingAcceptanceError("C log SHA manifest must cover exactly training start and terminal records")
    if checkpoint_completion.parent != checkpoint_root:
        raise CTrainingAcceptanceError("C checkpoint completion root drifted")

    identity_path, identity = _read_json(c_status_root / RUNTIME_IDENTITY, "C RUNTIME_IDENTITY")
    if identity_path.parent != c_status_root:
        raise CTrainingAcceptanceError("C RUNTIME_IDENTITY is not in the exact status root")

    # The run wrapper and status writer are separate processes.  Bind every
    # optional identity field they emitted to the one status-root identity so
    # a stale terminal marker cannot be paired with a fresh checkout/payload.
    source_commit = identity.get("stage_s_source_commit")
    for marker_path, marker in ((start_path, start), (terminal_path, terminal)):
        if marker.get("stage_s_source_commit") not in (None, source_commit):
            raise CTrainingAcceptanceError(f"C {marker_path.name} Stage-S source commit drifted")
        if marker.get("qpilots_commit") not in (None, EXPECTED_QPILOTS_COMMIT):
            raise CTrainingAcceptanceError(f"C {marker_path.name} QPILOTS commit drifted")
        if marker.get("openpi_commit") not in (None, EXPECTED_OPENPI_COMMIT):
            raise CTrainingAcceptanceError(f"C {marker_path.name} OpenPI commit drifted")
        if marker.get("run_id") not in (None, run_id) or marker.get("job_id") not in (None, job_id):
            raise CTrainingAcceptanceError(f"C {marker_path.name} run/JobId drifted")
    pipeline_path, pipeline = _read_json(c_status_root / PIPELINE_COMPLETION, "C COMPLETED_C_PIPELINE")
    if pipeline_path.parent != c_status_root or pipeline.get("schema") != "r142-stage-s-c-pai-stage-status-v1" or pipeline.get("status") != "COMPLETED" or pipeline.get("stage") != "terminal":
        raise CTrainingAcceptanceError("C COMPLETED_C_PIPELINE is not terminal COMPLETED")
    if pipeline.get("run_id") != run_id:
        raise CTrainingAcceptanceError("C COMPLETED_C_PIPELINE run id drifted")
    evidence = pipeline.get("evidence_path")
    if not isinstance(evidence, str) or Path(evidence).expanduser().resolve() != checkpoint_completion:
        raise CTrainingAcceptanceError("C COMPLETED_C_PIPELINE does not bind COMPLETED_C_TRAINING")
    if pipeline.get("evidence_sha256") != sha256_file(checkpoint_completion):
        raise CTrainingAcceptanceError("C COMPLETED_C_PIPELINE evidence SHA drifted")
    _self_bound_hash(pipeline_path, pipeline, "C COMPLETED_C_PIPELINE")
    if pipeline.get("stage_s_source_commit") not in (None, source_commit):
        raise CTrainingAcceptanceError("C COMPLETED_C_PIPELINE Stage-S source commit drifted")
    if pipeline.get("qpilots_commit") not in (None, EXPECTED_QPILOTS_COMMIT):
        raise CTrainingAcceptanceError("C COMPLETED_C_PIPELINE QPILOTS commit drifted")
    if pipeline.get("openpi_commit") not in (None, EXPECTED_OPENPI_COMMIT):
        raise CTrainingAcceptanceError("C COMPLETED_C_PIPELINE OpenPI commit drifted")
    if pipeline.get("job_id") not in (None, job_id):
        raise CTrainingAcceptanceError("C COMPLETED_C_PIPELINE JobId drifted")

    for marker_name in ("COMPLETED_preflight.json", "COMPLETED_base_download.json", "COMPLETED_conversion.json", "COMPLETED_training.json"):
        marker_path = _regular_file(c_status_root / marker_name, f"C {marker_name}")
        _, marker = _read_json(marker_path, f"C {marker_name}")
        if marker.get("status") != "COMPLETED" or marker.get("run_id") not in (None, run_id):
            raise CTrainingAcceptanceError(f"C {marker_name} is not bound terminal evidence")
        _self_bound_hash(marker_path, marker, f"C {marker_name}")
        if marker.get("job_id") not in (None, job_id):
            raise CTrainingAcceptanceError(f"C {marker_name} JobId drifted")
        if marker.get("stage_s_source_commit") not in (None, source_commit):
            raise CTrainingAcceptanceError(f"C {marker_name} Stage-S source commit drifted")
        if marker.get("qpilots_commit") not in (None, EXPECTED_QPILOTS_COMMIT):
            raise CTrainingAcceptanceError(f"C {marker_name} QPILOTS commit drifted")
        if marker.get("openpi_commit") not in (None, EXPECTED_OPENPI_COMMIT):
            raise CTrainingAcceptanceError(f"C {marker_name} OpenPI commit drifted")
    expected_status_files = {
        DATA_PREFLIGHT,
        RUNTIME_IDENTITY,
        PIPELINE_COMPLETION,
        "COMPLETED_preflight.json",
        "COMPLETED_base_download.json",
        "COMPLETED_conversion.json",
        "COMPLETED_training.json",
    }
    observed_status_files = _regular_relpaths(c_status_root)
    if observed_status_files != expected_status_files:
        missing = sorted(expected_status_files - observed_status_files)
        extra = sorted(observed_status_files - expected_status_files)
        raise CTrainingAcceptanceError(
            f"C status bundle has non-exact file inventory: missing={missing[:8]} extra={extra[:8]}"
        )
    unexpected_status_dirs = sorted(path.name for path in c_status_root.iterdir() if path.is_dir())
    if unexpected_status_dirs:
        raise CTrainingAcceptanceError(f"C status bundle has unexpected directories: {unexpected_status_dirs[:8]}")
    _audit_tree_ownership(c_status_root, uid=uid, gid=gid, label="C status bundle")
    for path in c_status_root.rglob("*"):
        if path.is_file() and path.name.startswith(("FAILED", "REFUSED", "STOPPED", "RUNNING", "QUEUED")):
            raise CTrainingAcceptanceError(f"C status bundle contains non-terminal marker: {path}")
    return terminal, start, identity, identity_path


def build_c_training_acceptance(
    *,
    registry_run: str | Path,
    registry_result: str | Path,
    submission_state: str | Path,
    resolved: str | Path,
    jobs_ledger: str | Path,
    terminal_getjob: str | Path,
    terminal_getjob_sha: str | Path,
    c_run_root: str | Path,
    c_status_root: str | Path,
    checkpoint_root: str | Path,
    expected_uid: int = EXPECTED_UID,
    expected_gid: int = EXPECTED_GID,
) -> dict[str, Any]:
    """Validate all C inputs and return an acceptance record.

    This function has no write side effects.  ``write_c_training_acceptance``
    performs the unique atomic publication only after this function returns.
    """

    uid, gid = int(expected_uid), int(expected_gid)
    run_root = _regular_dir(c_run_root, "C run root")
    status_root = _regular_dir(c_status_root, "C status root")
    ckpt_root = _regular_dir(checkpoint_root, "C checkpoint root")
    registry = _validate_registry(
        registry_run=registry_run,
        registry_result=registry_result,
        submission_state=submission_state,
        resolved=resolved,
        jobs_ledger=jobs_ledger,
        terminal_getjob=terminal_getjob,
        terminal_getjob_sha=terminal_getjob_sha,
        c_run_root=run_root,
        c_status_root=status_root,
        checkpoint_root=ckpt_root,
        uid=uid,
        gid=gid,
    )
    run_id = registry["run_id"]
    job_id = registry["job_id"]
    completion_path = _regular_file(ckpt_root / TRAINING_COMPLETION, "COMPLETED_C_TRAINING")
    if completion_path.parent != ckpt_root:
        raise CTrainingAcceptanceError("COMPLETED_C_TRAINING must be directly under checkpoint root")
    terminal, start, identity, identity_path = _validate_run_and_status(
        c_run_root=run_root,
        c_status_root=status_root,
        run_id=run_id,
        job_id=job_id,
        checkpoint_root=ckpt_root,
        checkpoint_completion=completion_path,
        uid=uid,
        gid=gid,
    )
    runtime_identity, source = _validate_runtime_identity(
        path=identity_path,
        run_id=run_id,
        job_id=job_id,
        resolved_runtime=registry["resolved_runtime"],
        uid=uid,
        gid=gid,
    )
    data_path = _regular_file(status_root / DATA_PREFLIGHT, "C DATA_PREFLIGHT")
    _, identity_payload = _read_json(identity_path, "C RUNTIME_IDENTITY")
    data = _validate_data_preflight(path=data_path, runtime_identity=identity_payload, uid=uid, gid=gid)
    models, checkpoint_state, checkpoint_manifest, log_manifest = _validate_checkpoints(
        checkpoint_root=ckpt_root,
        completion_path=completion_path,
        run_root=run_root,
        uid=uid,
        gid=gid,
    )
    _, completion = _read_json(completion_path, "COMPLETED_C_TRAINING")
    if completion.get("log_sha256sums") != str(log_manifest):
        raise CTrainingAcceptanceError("COMPLETED_C_TRAINING log manifest binding drifted")
    if completion.get("data_preflight_path") not in (None, str(data_path)):
        raise CTrainingAcceptanceError("COMPLETED_C_TRAINING data preflight path drifted")
    if completion.get("data_preflight_sha256") not in (None, sha256_file(data_path)):
        raise CTrainingAcceptanceError("COMPLETED_C_TRAINING data preflight SHA drifted")

    return {
        "schema": SCHEMA,
        "status": EXPECTED_STATUS,
        "label": EXPECTED_LABEL,
        "pai_terminal_status": EXPECTED_PAI_STATUS,
        "accepted_run_id": run_id,
        "job_id": job_id,
        "source": source,
        "checkpoint_root": str(ckpt_root),
        "checkpoint_completion": str(completion_path),
        "checkpoint_sha256_manifest": str(checkpoint_manifest),
        "checkpoint_sha256_manifest_digest": sha256_file(checkpoint_manifest),
        "log_root": str(run_root),
        "log_sha256_manifest": str(log_manifest),
        "log_sha256_manifest_digest": sha256_file(log_manifest),
        "training_pipeline_completion": str(status_root / PIPELINE_COMPLETION),
        "checkpoint_completion_sha256": sha256_file(completion_path),
        "checkpoint_steps": list(EXPECTED_STEPS),
        "full_reference_step": EXPECTED_FULL_REFERENCE_STEP,
        "no_interpolation": True,
        "artificial_degradation": False,
        "checkpoint_hashes": models,
        "checkpoint_state": checkpoint_state,
        "registry_binding": registry,
        "runtime_identity": runtime_identity,
        "data_provenance": data,
        "training_terminal": {"path": str(run_root / TRAINING_TERMINAL), "sha256": sha256_file(run_root / TRAINING_TERMINAL), "global_step": EXPECTED_TERMINAL_STEP},
        "training_start": {"path": str(run_root / TRAINING_START), "sha256": sha256_file(run_root / TRAINING_START)},
        "retention": {
            "required_steps": list(EXPECTED_STEPS),
            "full_state": True,
            "world_size": EXPECTED_WORLD_SIZE,
            "logs_sha_verified": True,
            "ownership": {"uid": uid, "gid": gid},
            "no_partial_or_failure_markers": True,
        },
        "acceptance_contract": "terminal PAI Succeeded; one unique JobId; exact source/data/runtime identity; full native checkpoints and manifests; always WEAK_SUBSTRATE",
        "no_pai_submit_performed": True,
    }


def _atomic_bytes(path: Path, data: bytes) -> None:
    if path.is_symlink() or path.exists():
        raise CTrainingAcceptanceError(f"refusing to overwrite existing acceptance artifact: {path}")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise CTrainingAcceptanceError(f"acceptance output parent is missing or symlinked: {path.parent}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{id(data)}")
    try:
        # O_EXCL prevents two callers in one process from sharing a temp
        # inode.  A hard-link from the fully fsynced temp inode is the
        # no-replace publication primitive: unlike os.replace(), it cannot
        # overwrite an acceptance emitted by a racing caller.
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            # fdopen owns the descriptor after entering its context.  If the
            # write failed before that hand-off, close it explicitly.
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise CTrainingAcceptanceError(
                f"refusing to overwrite existing acceptance artifact: {path}"
            ) from exc
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_c_training_acceptance(
    *,
    output_path: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Validate and publish one unique acceptance JSON plus ``.sha256``."""

    output = Path(output_path).expanduser()
    if output.name != "ACCEPTED_C_TRAINING.json":
        raise CTrainingAcceptanceError("acceptance output must be named ACCEPTED_C_TRAINING.json")
    status_root = _regular_dir(kwargs["c_status_root"], "C status root")
    output_parent = output.parent.resolve()
    if output_parent not in {status_root, status_root.parent}:
        raise CTrainingAcceptanceError("acceptance output must be beside the exact C status root or its parent")
    sidecar = output.with_name(output.name + ".sha256")
    if output.exists() or output.is_symlink() or sidecar.exists() or sidecar.is_symlink():
        raise CTrainingAcceptanceError("acceptance output already exists; refusing overwrite")
    record = build_c_training_acceptance(**kwargs)
    encoded = json.dumps(record, sort_keys=True, indent=2, ensure_ascii=False).encode("utf-8") + b"\n"
    digest = hashlib.sha256(encoded).hexdigest()
    # Reserve both names before validation can be repeated concurrently.  The
    # second reservation is deliberately created only after the JSON bytes are
    # durable; consumers accept only when both files exist and the sidecar
    # matches.
    _atomic_bytes(output, encoded)
    try:
        _atomic_bytes(sidecar, f"{digest}  {output.name}\n".encode("ascii"))
    except BaseException:
        # A JSON without its sidecar is not an accepted lineage.  Do not hide
        # the original publication error behind cleanup failures.  The output
        # was newly reserved by this invocation, so it is safe to remove it;
        # a retry must never mistake an orphan JSON for acceptance.
        try:
            output.unlink()
        except OSError:
            pass
        raise
    return {**record, "acceptance_path": str(output.resolve()), "acceptance_sha256": digest, "acceptance_sha_sidecar": str(sidecar.resolve())}


def verify_c_training_acceptance(path: str | Path) -> dict[str, Any]:
    """Verify a previously published JSON and its sidecar without mutation."""

    target = _regular_file(path, "C acceptance JSON")
    if target.name != "ACCEPTED_C_TRAINING.json":
        raise CTrainingAcceptanceError("C acceptance JSON has the wrong filename")
    sidecar = _regular_file(target.with_name(target.name + ".sha256"), "C acceptance SHA sidecar")
    digest = _verify_sha_sidecar(target, sidecar, "C acceptance JSON")
    _, payload = _read_json(target, "C acceptance JSON")
    if payload.get("schema") != SCHEMA or payload.get("status") != EXPECTED_STATUS or payload.get("label") != EXPECTED_LABEL or payload.get("pai_terminal_status") != EXPECTED_PAI_STATUS:
        raise CTrainingAcceptanceError("C acceptance JSON status/schema/label drifted")
    return {"path": str(target), "sha256": digest, "payload": payload}


__all__ = [
    "SCHEMA",
    "EXPECTED_STATUS",
    "EXPECTED_LABEL",
    "EXPECTED_PAI_STATUS",
    "EXPECTED_SEED",
    "EXPECTED_WORLD_SIZE",
    "EXPECTED_TERMINAL_STEP",
    "EXPECTED_STEPS",
    "EXPECTED_FULL_REFERENCE_STEP",
    "EXPECTED_OPENPI_COMMIT",
    "EXPECTED_QPILOTS_COMMIT",
    "EXPECTED_LIBERO_COMMIT",
    "EXPECTED_DATASET_REPO",
    "EXPECTED_DATASET_REVISION",
    "EXPECTED_DATASET_MANIFEST_SHA256",
    "CTrainingAcceptanceError",
    "build_c_training_acceptance",
    "write_c_training_acceptance",
    "verify_c_training_acceptance",
    "sha256_file",
]
