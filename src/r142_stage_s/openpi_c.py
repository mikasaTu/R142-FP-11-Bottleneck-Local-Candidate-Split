"""Fail-closed assets and training contracts for Stage-S substrate C.

The C screen is deliberately based on the *official* OpenPI PyTorch path at
the pinned commit, rather than on a community SFT checkpoint.  This module
keeps the large-data operation executable on CPFS while making the source,
object, conversion, resume, and terminal-marker contracts inspectable without
importing JAX, Torch, PAI, or the simulator.

No function in this module submits a PAI job.  ``download_base_checkpoint``
and ``run_conversion`` are explicit foreground operations; callers must run
them only after a path/ownership preflight in the target environment.
"""

from __future__ import annotations

import base64
import ast
import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

from .libero import (
    C_RETAIN_STEPS,
    C_SAVE_INTERVAL,
    C_TRAINING_SEED,
    C_TRAINING_STEPS,
    C_FULL_REFERENCE_STEP,
    PAI_MAX_CPU,
    PAI_MAX_GPU,
    PAI_MAX_MEMORY_GIB,
    STAGE_S_NO_JOB_WINDOWS,
    atomic_bytes,
    atomic_json,
    canonical_json,
    sha256_file,
)


OPENPI_COMMIT = "54cbaee6ae0c010a1ed431871cdaa8f4684ac709"
OPENPI_SOURCE_URL = "https://github.com/Physical-Intelligence/openpi.git"
DEFAULT_OPENPI_PYTHON = "/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/openpi_py311/bin/python"
OPENPI_CONFIG_NAME = "pi05_libero"
OPENPI_CONVERTER = "examples/convert_jax_model_to_pytorch.py"
OPENPI_TRAINER = "scripts/train_pytorch.py"
LIBERO_DATASET_REPO = "physical-intelligence/libero"
# The C substrate is frozen to the published v2.0 dataset snapshot.  The
# revision is recorded by Hugging Face's local-download sidecars and is
# checked before the official LeRobot loader is allowed to run.
LIBERO_DATASET_REVISION = "9dfa69510ea9e1613fc54112bc706444b686a231"
DEFAULT_HF_LEROBOT_HOME = "/mnt/cpfs/zbl-cpfs-new/USERS/leon/cache/lerobot"
DEFAULT_LIBERO_DATASET_ROOT = f"{DEFAULT_HF_LEROBOT_HOME}/{LIBERO_DATASET_REPO}"
LIBERO_DATASET_MANIFEST_NAME = "DATASET_SHA256SUMS"
LIBERO_DATASET_MANIFEST_SHA256 = "02b5b3abfadb65b2f1c4823cfe7ed7b9351416934674fcf59aea1868826546bf"
LIBERO_DATASET_EXPECTED_INFO = {
    "codebase_version": "v2.0",
    "robot_type": "panda",
    "total_episodes": 1693,
    "total_frames": 273465,
    "total_tasks": 40,
    "total_videos": 0,
    "total_chunks": 2,
    "chunks_size": 1000,
    "fps": 10,
    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
}
DEFAULT_STAGED_LIBERO_ASSETS_BASE_DIR = "/mnt/cpfs/zbl-cpfs-new/USERS/leon/cache/r142_stage_s/c_libero_assets"
LIBERO_NORM_STATS_SOURCE_RELATIVE = f"{OPENPI_CONFIG_NAME}/assets/{LIBERO_DATASET_REPO}/norm_stats.json"
# OpenPI's pinned DataConfigFactory resolves assets_base_dir/config_name/repo_id
# (without the source checkpoint's extra ``assets`` component).
LIBERO_NORM_STATS_RUNTIME_RELATIVE = f"{OPENPI_CONFIG_NAME}/{LIBERO_DATASET_REPO}/norm_stats.json"
LIBERO_NORM_STATS_MARKER_NAME = ".r142_stage_s_libero_norm_stats.json"
DEFAULT_PI05_ASSETS_BASE_DIR = "/mnt/cpfs/zbl-cpfs-new/USERS/leon/cache/openpi/r16p15/openpi-assets/checkpoints"
PI05_BASE_BUCKET = "openpi-assets"
PI05_BASE_PREFIX = "checkpoints/pi05_base/"
PI05_BASE_GCS_URI = f"gs://{PI05_BASE_BUCKET}/{PI05_BASE_PREFIX}"
PI05_BASE_HTTP_PREFIX = f"https://storage.googleapis.com/{PI05_BASE_BUCKET}/{PI05_BASE_PREFIX}"
PI05_BASE_LISTING_URL = (
    f"https://storage.googleapis.com/storage/v1/b/{PI05_BASE_BUCKET}/o?"
    f"prefix={urllib.parse.quote(PI05_BASE_PREFIX, safe='')}&maxResults=1000"
)
PI05_BASE_OBJECT_COUNT = 29
PI05_BASE_TOTAL_BYTES = 12_441_749_581
PI05_BASE_MANIFEST_NAME = "BASE_OBJECT_MANIFEST.json"
PI05_BASE_COMPLETION_NAME = "BASE_DOWNLOAD_COMPLETED.json"
PI05_BASE_SHA_NAME = "SHA256SUMS"
CONVERSION_PROVENANCE_NAME = "CONVERSION_PROVENANCE.json"
CONVERSION_COMPLETION_NAME = "CONVERSION_COMPLETED.json"
TRAINING_START_NAME = "TRAINING_START.json"
TRAINING_TERMINAL_NAME = "TRAINING_TERMINAL.json"
TRAINING_COMPLETION_NAME = "COMPLETED_C_TRAINING.json"
TRAINING_FAILED_NAME = "FAILED_C_TRAINING.json"
C_NUM_WORKERS = 0


@dataclass(frozen=True)
class GCSObject:
    """The immutable server-side identity of one public GCS object."""

    name: str
    size: int
    md5_base64: str | None
    crc32c: str
    generation: str
    updated: str

    @property
    def relative_name(self) -> str:
        if not self.name.startswith(PI05_BASE_PREFIX):
            raise ValueError(f"object is outside the frozen prefix: {self.name}")
        return self.name[len(PI05_BASE_PREFIX) :]

    def as_dict(self) -> dict[str, Any]:
        return {
            "gcs_name": self.name,
            "name": self.relative_name,
            "size": int(self.size),
            "md5_base64": self.md5_base64,
            "crc32c": self.crc32c,
            "generation": self.generation,
            "updated": self.updated,
            "sha256": None,
            "downloaded": False,
        }


# This is a checked-in copy of the public JSON API listing observed on
# 2026-09-02.  It intentionally carries both the object metadata supplied by
# GCS and a null SHA-256 field: GCS does not expose SHA-256 for these objects,
# so the downloader must calculate it from the bytes actually persisted on
# CPFS.  The list is sorted by full GCS object name and is not selected after
# any model/evaluation outcome.
PI05_BASE_OBJECTS: tuple[GCSObject, ...] = (
    GCSObject("checkpoints/pi05_base/assets/arx/norm_stats.json", 3476, "Qy/oPSHJD94BHxRaLllgsg==", "6nyXHA==", "1757354310096104", "2025-09-08T17:58:30.180Z"),
    GCSObject("checkpoints/pi05_base/assets/arx_mobile/norm_stats.json", 3677, "Cs4bLtdB+15zZI0ObagbEQ==", "3XZHoA==", "1757354310086315", "2025-09-08T17:58:30.159Z"),
    GCSObject("checkpoints/pi05_base/assets/droid/norm_stats.json", 2062, "HQC4cGUtEo+86jMc1vBR7w==", "sr4eWw==", "1757354310090122", "2025-09-08T17:58:30.180Z"),
    GCSObject("checkpoints/pi05_base/assets/fibocom_mobile/norm_stats.json", 3951, "w0SZXV8c/ogHVtS7Ckh++Q==", "Fh5IlA==", "1757354310088094", "2025-09-08T17:58:30.180Z"),
    GCSObject("checkpoints/pi05_base/assets/franka/norm_stats.json", 2081, "LC6+FS5cfmXjBJKG9hderA==", "LDm8qg==", "1757354310090331", "2025-09-08T17:58:30.180Z"),
    GCSObject("checkpoints/pi05_base/assets/trossen/norm_stats.json", 3394, "OfbauPUY0C6yjwF1jFjdsg==", "ej6a+A==", "1757354310095901", "2025-09-08T17:58:30.180Z"),
    GCSObject("checkpoints/pi05_base/assets/trossen_mobile/norm_stats.json", 3699, "PEW+oUP3c2vo1fwB0Qtp+A==", "l2F3EA==", "1757354310090749", "2025-09-08T17:58:30.180Z"),
    GCSObject("checkpoints/pi05_base/assets/ur5e/norm_stats.json", 1851, "ljxT2aK0MQVTC87Sw+ZeQw==", "o9bKwg==", "1757354310089361", "2025-09-08T17:58:30.180Z"),
    GCSObject("checkpoints/pi05_base/assets/ur5e_dual/norm_stats.json", 3459, "YxVr2ClglFN0p/7g43ycPw==", "X5QwQA==", "1757354310089903", "2025-09-08T17:58:30.180Z"),
    GCSObject("checkpoints/pi05_base/params/_CHECKPOINT_METADATA", 258, "uDrD0GJ9/soyYRTjoi8CTw==", "3q5lUQ==", "1757354310086279", "2025-09-08T17:58:30.180Z"),
    GCSObject("checkpoints/pi05_base/params/_METADATA", 21402, "2KbT3SeW+TztM+bI4gRvTQ==", "frXw2w==", "1757354310086139", "2025-09-08T17:58:30.180Z"),
    GCSObject("checkpoints/pi05_base/params/_sharding", 11420, "fWlnNNk1SS+IQh+CEMIX6g==", "4PKBJg==", "1757354310089508", "2025-09-08T17:58:30.180Z"),
    GCSObject("checkpoints/pi05_base/params/array_metadatas/process_0", 8856, "/YmvRR2S0+o29qZiGEPt0w==", "CS/yRg==", "1757354310090622", "2025-09-08T17:58:30.180Z"),
    GCSObject("checkpoints/pi05_base/params/commit_success.txt", 0, "1B2M2Y8AsgTpgAmY7PhCfg==", "AAAAAA==", "1757354310089792", "2025-09-08T17:58:30.180Z"),
    GCSObject("checkpoints/pi05_base/params/d/1c4302d2d2000b5f3eb4fa1350fdef9a", 2132, "M9VQtg9mmNpNxUk3PFqKaA==", "SGdLxw==", "1757354310119002", "2025-09-08T17:58:30.181Z"),
    GCSObject("checkpoints/pi05_base/params/manifest.ocdbt", 117, "JHMt8vA2+4zTj7e08rGstQ==", "SGdLxw==", "1757354310086335", "2025-09-08T17:58:30.180Z"),
    GCSObject("checkpoints/pi05_base/params/ocdbt.process_0/d/0832cad6c37f82d4eedd897dcbb8da9d", 2240590484, None, "4YAIHA==", "1757354310220587", "2025-09-08T17:58:30.321Z"),
    GCSObject("checkpoints/pi05_base/params/ocdbt.process_0/d/247d4b7c814d8b1a23fa8a20f36a88f7", 34945019, "OXwd3VCCtNmDWnq+Av+/8w==", "ceM1dg==", "1757354310092980", "2025-09-08T17:58:30.180Z"),
    GCSObject("checkpoints/pi05_base/params/ocdbt.process_0/d/35a545f74995e511808d4f94dfbef3b6", 70041571, "H9E9107OxwddcrM7+5WzAw==", "Iq8YUA==", "1757354310090576", "2025-09-08T17:58:30.180Z"),
    GCSObject("checkpoints/pi05_base/params/ocdbt.process_0/d/73bbae8a4deba6498bf07d96b215a574", 2935855, "HVlWzNlQc0i4zG5dA3jWrA==", "2IEHcA==", "1757354310087457", "2025-09-08T17:58:30.180Z"),
    GCSObject("checkpoints/pi05_base/params/ocdbt.process_0/d/7bc9d3296d23a6fb83a6b3778ac6e964", 2240315383, None, "R6AiOQ==", "1757354310232242", "2025-09-08T17:58:30.351Z"),
    GCSObject("checkpoints/pi05_base/params/ocdbt.process_0/d/7d78f38c5d8be1eea31406644dde9bd6", 133821, "qA3aTOnuJOm8TSyveL+ScA==", "3ljcDg==", "1757354310097613", "2025-09-08T17:58:30.180Z"),
    GCSObject("checkpoints/pi05_base/params/ocdbt.process_0/d/828bee85475e37c61e1cc19e32d1c5ef", 929, "l5njiIkDe+WMHZv9hjz9yQ==", "SGdLxw==", "1757354310090811", "2025-09-08T17:58:30.180Z"),
    GCSObject("checkpoints/pi05_base/params/ocdbt.process_0/d/8c5d7070ea57bdce2f0a19f95b8a21b4", 3077149148, None, "gQSvJQ==", "1757354310234847", "2025-09-08T17:58:30.385Z"),
    GCSObject("checkpoints/pi05_base/params/ocdbt.process_0/d/b4349aaadb7dfa45c3a53fc67c04b8f6", 1120156687, None, "FlwBrA==", "1757354310188883", "2025-09-08T17:58:30.248Z"),
    GCSObject("checkpoints/pi05_base/params/ocdbt.process_0/d/caf01a82962cbd4651563d2ac1063e0b", 27445152, "eKeUW1e7wkNVuJ7OtsXgXQ==", "Yosa6g==", "1757354310098105", "2025-09-08T17:58:30.180Z"),
    GCSObject("checkpoints/pi05_base/params/ocdbt.process_0/d/deefd3c43390a50472cbcd317b0fff58", 2393670849, None, "JyiHIA==", "1757354310196805", "2025-09-08T17:58:30.322Z"),
    GCSObject("checkpoints/pi05_base/params/ocdbt.process_0/d/ec484cf8f02dcf59e1892180f0862e40", 1234292390, None, "TP2sFA==", "1757354310192419", "2025-09-08T17:58:30.280Z"),
    GCSObject("checkpoints/pi05_base/params/ocdbt.process_0/manifest.ocdbt", 458, "76clmla5BpNyk9xZmuw44g==", "SGdLxw==", "1757354310097331", "2025-09-08T17:58:30.180Z"),
)


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def expected_base_manifest() -> dict[str, Any]:
    """Return the checked-in object contract with no unverified SHA values."""

    return {
        "schema": "r142-stage-s-c-pi05-base-object-manifest-v1",
        "status": "EXPECTED_NOT_DOWNLOADED",
        "source": {
            "bucket": PI05_BASE_BUCKET,
            "prefix": PI05_BASE_PREFIX,
            "uri": PI05_BASE_GCS_URI,
            "http_prefix": PI05_BASE_HTTP_PREFIX,
            "listing_url": PI05_BASE_LISTING_URL,
            "listing_observed": "2026-09-02",
            "sha256_policy": "compute_after_persisting_each_object; GCS md5 is not a SHA256 substitute",
        },
        "object_count": PI05_BASE_OBJECT_COUNT,
        "total_bytes": PI05_BASE_TOTAL_BYTES,
        "objects": [item.as_dict() for item in PI05_BASE_OBJECTS],
        "manifest_sha256": None,
    }


def _safe_relative(name: str) -> str:
    path = Path(name)
    if not name or path.is_absolute() or ".." in path.parts or name.endswith("/"):
        raise ValueError(f"unsafe relative object name: {name!r}")
    return path.as_posix()


def validate_base_manifest(
    manifest: Mapping[str, Any], *, strict_source: bool = True
) -> tuple[dict[str, Any], ...]:
    """Validate object names, sizes, cardinality, and optional SHA fields."""

    objects = manifest.get("objects")
    if not isinstance(objects, list):
        raise ValueError("base manifest must contain an objects list")
    if int(manifest.get("object_count", -1)) != len(objects):
        raise ValueError("base manifest object_count does not match objects")
    if int(manifest.get("total_bytes", -1)) != sum(int(row.get("size", -1)) for row in objects if isinstance(row, Mapping)):
        raise ValueError("base manifest total_bytes does not match object sizes")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    expected_by_name = {item.relative_name: item for item in PI05_BASE_OBJECTS}
    for raw in objects:
        if not isinstance(raw, Mapping):
            raise ValueError("base manifest object row is not an object")
        row = dict(raw)
        relative = _safe_relative(str(row.get("name", "")))
        if relative in seen:
            raise ValueError(f"duplicate object name: {relative}")
        seen.add(relative)
        if int(row.get("size", -1)) < 0:
            raise ValueError(f"invalid object size for {relative}")
        gcs_name = str(row.get("gcs_name", PI05_BASE_PREFIX + relative))
        if gcs_name != PI05_BASE_PREFIX + relative:
            raise ValueError(f"gcs_name does not match frozen prefix for {relative}")
        if strict_source:
            expected = expected_by_name.get(relative)
            if expected is None:
                raise ValueError(f"unexpected object under frozen prefix: {relative}")
            if int(row["size"]) != expected.size:
                raise ValueError(f"size drift for {relative}: {row['size']} != {expected.size}")
            if row.get("md5_base64") != expected.md5_base64:
                raise ValueError(f"md5 metadata drift for {relative}")
            if row.get("crc32c") != expected.crc32c:
                raise ValueError(f"crc32c metadata drift for {relative}")
        digest = row.get("sha256")
        if digest is not None and (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
            raise ValueError(f"invalid SHA-256 for {relative}")
        rows.append(row)
    if strict_source and set(seen) != set(expected_by_name):
        missing = sorted(set(expected_by_name) - seen)
        extra = sorted(seen - set(expected_by_name))
        raise ValueError(f"frozen object set mismatch; missing={missing}, extra={extra}")
    return tuple(rows)


def manifest_from_gcs_listing(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a live JSON API listing into the frozen manifest format."""

    items = payload.get("items")
    if not isinstance(items, list) or payload.get("nextPageToken"):
        raise ValueError("GCS listing is missing items or requires pagination")
    expected = {item.name: item for item in PI05_BASE_OBJECTS}
    if {str(item.get("name")) for item in items} != set(expected):
        raise ValueError("live GCS listing does not contain exactly the frozen 29 objects")
    rows: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda row: str(row.get("name"))):
        name = str(item.get("name"))
        frozen = expected[name]
        if int(item.get("size", -1)) != frozen.size:
            raise ValueError(f"live GCS size drift for {name}")
        row = {
            "gcs_name": name,
            "name": frozen.relative_name,
            "size": int(item["size"]),
            "md5_base64": item.get("md5Hash"),
            "crc32c": str(item.get("crc32c", "")),
            "generation": str(item.get("generation", "")),
            "updated": str(item.get("updated", "")),
            "sha256": None,
            "downloaded": False,
        }
        rows.append(row)
    result = expected_base_manifest()
    result["source"] = dict(result["source"])
    result["source"]["live_listing_verified"] = True
    result["objects"] = rows
    validate_base_manifest(result, strict_source=True)
    result["manifest_sha256"] = _canonical_sha256({key: value for key, value in result.items() if key != "manifest_sha256"})
    return result


def fetch_gcs_manifest(
    *, opener: Callable[..., Any] = urllib.request.urlopen, timeout: int = 60
) -> dict[str, Any]:
    request = urllib.request.Request(PI05_BASE_LISTING_URL, headers={"Accept": "application/json"})
    response = opener(request, timeout=timeout)
    with response:
        payload = json.load(response)
    return manifest_from_gcs_listing(payload)


def write_expected_base_manifest(path: str | Path) -> dict[str, Any]:
    result = expected_base_manifest()
    result["manifest_sha256"] = _canonical_sha256({key: value for key, value in result.items() if key != "manifest_sha256"})
    atomic_json(path, result)
    return result


def _file_digests(path: Path) -> tuple[str, str]:
    sha = hashlib.sha256()
    md5 = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
            md5.update(chunk)
    return sha.hexdigest(), base64.b64encode(md5.digest()).decode("ascii")


def _urlopen(opener: Callable[..., Any], request: urllib.request.Request, timeout: int) -> Any:
    try:
        return opener(request, timeout=timeout)
    except TypeError:
        return opener(request)


def _status(response: Any) -> int:
    value = getattr(response, "status", None)
    if value is None:
        value = getattr(response, "code", 200)
    return int(value)


def download_base_checkpoint(
    output_root: str | Path,
    manifest: Mapping[str, Any] | str | Path | None = None,
    *,
    opener: Callable[..., Any] = urllib.request.urlopen,
    timeout: int = 120,
    chunk_size: int = 8 * 1024 * 1024,
    strict_source: bool = True,
) -> dict[str, Any]:
    """Download the exact base object set with resumable per-object ``.part`` files.

    A complete destination is never silently overwritten.  A short partial
    is resumed with HTTP Range; a server that ignores Range is rejected rather
    than appended, which prevents byte duplication after an interruption.
    Completion and SHA files are written only after every object has passed
    its expected size and any available GCS MD5 check.
    """

    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if manifest is None:
        payload = expected_base_manifest()
    elif isinstance(manifest, (str, Path)):
        payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
    else:
        payload = dict(manifest)
    rows = list(validate_base_manifest(payload, strict_source=strict_source))
    persisted_rows: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        relative = _safe_relative(str(row["name"]))
        destination = root / relative
        if not destination.resolve().is_relative_to(root):
            raise ValueError(f"object escapes output root: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        expected_size = int(row["size"])
        if destination.exists():
            if not destination.is_file() or destination.stat().st_size != expected_size:
                raise ValueError(f"existing destination has wrong size/type: {destination}")
        else:
            partial = destination.with_name(destination.name + ".part")
            if partial.exists() and partial.stat().st_size > expected_size:
                raise ValueError(f"partial destination exceeds expected size: {partial}")
            start = partial.stat().st_size if partial.exists() else 0
            url = PI05_BASE_HTTP_PREFIX + urllib.parse.quote(relative, safe="/")
            headers = {"Accept-Encoding": "identity"}
            if start:
                headers["Range"] = f"bytes={start}-"
            request = urllib.request.Request(url, headers=headers)
            response = _urlopen(opener, request, timeout)
            status = _status(response)
            expected_status = 206 if start else 200
            if status != expected_status:
                try:
                    response.close()
                except Exception:
                    pass
                raise IOError(f"unexpected HTTP status {status} for {relative} at offset {start}")
            with response, partial.open("ab") as handle:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    if handle.tell() + len(chunk) > expected_size:
                        raise IOError(f"download exceeded expected size for {relative}")
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            if partial.stat().st_size != expected_size:
                raise IOError(f"short download for {relative}: {partial.stat().st_size} != {expected_size}")
            partial.replace(destination)
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        sha256, md5_base64 = _file_digests(destination)
        expected_md5 = row.get("md5_base64")
        if expected_md5 is not None and md5_base64 != expected_md5:
            raise IOError(f"MD5 mismatch for {relative}: {md5_base64} != {expected_md5}")
        expected_sha = row.get("sha256")
        if expected_sha is not None and sha256 != expected_sha:
            raise IOError(f"SHA256 mismatch for {relative}: {sha256} != {expected_sha}")
        row.update({"sha256": sha256, "downloaded": True, "bytes_observed": destination.stat().st_size})
        persisted_rows.append(row)
    persisted = dict(payload)
    persisted["status"] = "DOWNLOADED"
    persisted["objects"] = persisted_rows
    persisted["manifest_sha256"] = _canonical_sha256({key: value for key, value in persisted.items() if key != "manifest_sha256"})
    manifest_path = root / PI05_BASE_MANIFEST_NAME
    atomic_json(manifest_path, persisted)
    checksum_lines = "".join(f"{row['sha256']}  {row['name']}\n" for row in sorted(persisted_rows, key=lambda item: str(item["name"])))
    atomic_text = checksum_lines.encode("utf-8")
    atomic_bytes(root / PI05_BASE_SHA_NAME, atomic_text)
    marker = {
        "schema": "r142-stage-s-c-pi05-base-download-v1",
        "status": "COMPLETED",
        "object_count": len(persisted_rows),
        "total_bytes": sum(int(row["bytes_observed"]) for row in persisted_rows),
        "manifest": PI05_BASE_MANIFEST_NAME,
        "manifest_sha256": sha256_file(manifest_path),
        "sha256sums": PI05_BASE_SHA_NAME,
        "sha256sums_sha256": sha256_file(root / PI05_BASE_SHA_NAME),
        "root": str(root),
    }
    marker["payload_sha256"] = _canonical_sha256(marker)
    atomic_json(root / PI05_BASE_COMPLETION_NAME, marker)
    return marker


def audit_base_download(root: str | Path) -> dict[str, Any]:
    """Verify all 29 persisted files and both terminal integrity artifacts."""

    base = Path(root).expanduser().resolve()
    errors: list[str] = []
    manifest_path = base / PI05_BASE_MANIFEST_NAME
    marker_path = base / PI05_BASE_COMPLETION_NAME
    sums_path = base / PI05_BASE_SHA_NAME
    if not manifest_path.is_file():
        errors.append(f"missing {manifest_path.name}")
        return {"valid": False, "errors": errors, "root": str(base)}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = validate_base_manifest(manifest, strict_source=True)
    except Exception as exc:  # noqa: BLE001 - report all capability errors
        return {"valid": False, "errors": [f"invalid base manifest: {exc}"], "root": str(base)}
    expected_manifest_sha = _canonical_sha256({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    if manifest.get("manifest_sha256") != expected_manifest_sha:
        errors.append("base manifest self-digest mismatch")
    for row in rows:
        path = base / _safe_relative(str(row["name"]))
        if not path.is_file():
            errors.append(f"missing object file: {row['name']}")
            continue
        if path.stat().st_size != int(row["size"]):
            errors.append(f"size mismatch: {row['name']}")
            continue
        observed, _ = _file_digests(path)
        if row.get("sha256") != observed:
            errors.append(f"sha256 mismatch: {row['name']}")
    if not sums_path.is_file():
        errors.append(f"missing {sums_path.name}")
    else:
        expected_lines = {f"{row.get('sha256')}  {row['name']}" for row in rows}
        actual_lines = {line.strip() for line in sums_path.read_text(encoding="utf-8").splitlines() if line.strip()}
        if actual_lines != expected_lines:
            errors.append("SHA256SUMS does not match the manifest")
    marker: dict[str, Any] | None = None
    if not marker_path.is_file():
        errors.append(f"missing {marker_path.name}")
    else:
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"invalid {marker_path.name}: {exc}")
        else:
            if marker.get("status") != "COMPLETED":
                errors.append("base completion marker is not COMPLETED")
            if int(marker.get("object_count", -1)) != PI05_BASE_OBJECT_COUNT:
                errors.append("base completion marker object count mismatch")
            if int(marker.get("total_bytes", -1)) != PI05_BASE_TOTAL_BYTES:
                errors.append("base completion marker byte count mismatch")
            if marker.get("manifest_sha256") != sha256_file(manifest_path):
                errors.append("base completion marker manifest hash mismatch")
            if marker.get("sha256sums_sha256") != sha256_file(sums_path):
                errors.append("base completion marker SHA256SUMS hash mismatch")
            if marker.get("payload_sha256") != _canonical_sha256(
                {key: value for key, value in marker.items() if key != "payload_sha256"}
            ):
                errors.append("base completion marker payload hash mismatch")
    return {"valid": not errors, "errors": errors, "root": str(base), "marker": marker, "manifest": manifest}


def _libero_expected_snapshot_files(info: Mapping[str, Any]) -> list[str]:
    """Return the exact files required by the frozen v2.0 LIBERO snapshot."""

    total_episodes = int(info["total_episodes"])
    data_path = str(info["data_path"])
    files = [
        ".gitattributes",
        "README.md",
        "meta/info.json",
        "meta/tasks.jsonl",
        "meta/episodes.jsonl",
        "meta/stats.json",
    ]
    files.extend(
        data_path.format(episode_chunk=episode_index // int(info["chunks_size"]), episode_index=episode_index)
        for episode_index in range(total_episodes)
    )
    return files


def _jsonl_line_count(path: Path, expected: int) -> str | None:
    """Parse a JSONL file and return an error instead of trusting line count."""

    count = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    return f"blank line in {path} at line {line_number}"
                try:
                    json.loads(line)
                except json.JSONDecodeError as exc:
                    return f"invalid JSON in {path} at line {line_number}: {exc}"
                count += 1
    except OSError as exc:
        return f"cannot read {path}: {exc}"
    if count != expected:
        return f"{path} has {count} JSONL rows, expected {expected}"
    return None


def _read_sha256sum_manifest(path: Path, expected_files: Sequence[str]) -> tuple[dict[str, str], list[str]]:
    """Parse the existing cwd-relative ``sha256sum`` manifest strictly."""

    rows: dict[str, str] = {}
    errors: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return {}, [f"cannot read dataset checksum manifest {path}: {exc}"]
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            errors.append(f"blank line in dataset checksum manifest at line {line_number}")
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            errors.append(f"invalid dataset checksum manifest line {line_number}")
            continue
        relative = parts[1]
        if relative.startswith("*" ):
            relative = relative[1:]
        if Path(relative).is_absolute() or ".." in Path(relative).parts or not relative:
            errors.append(f"unsafe path in dataset checksum manifest line {line_number}: {relative!r}")
            continue
        relative = Path(relative).as_posix()
        if relative in rows:
            errors.append(f"duplicate path in dataset checksum manifest: {relative}")
        rows[relative] = parts[0]
    expected = set(expected_files)
    observed = set(rows)
    missing = sorted(expected - observed)
    extras = sorted(observed - expected)
    if missing:
        errors.append(f"dataset checksum manifest missing files: {missing[:8]}")
    if extras:
        errors.append(f"dataset checksum manifest has unexpected files: {extras[:8]}")
    return rows, errors


def audit_libero_dataset_snapshot(
    dataset_root: str | Path = DEFAULT_LIBERO_DATASET_ROOT,
    *,
    persist_manifest: bool = False,
    verify_hashes: bool = False,
) -> dict[str, Any]:
    """Audit the complete local ``physical-intelligence/libero`` snapshot.

    The audit is intentionally stricter than ``LeRobotDatasetMetadata``.  A
    missing ``meta`` file would otherwise make the pinned LeRobot class call
    ``snapshot_download``.  This function refuses that state, verifies every
    Hugging Face download sidecar is for the frozen commit, and reuses the
    pre-generated ``DATASET_SHA256SUMS`` manifest.  It never downloads,
    rewrites, or repairs the dataset.
    """

    root = Path(dataset_root).expanduser().resolve()
    manifest_path = root / LIBERO_DATASET_MANIFEST_NAME
    errors: list[str] = []
    info: dict[str, Any] = {}
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        errors.append(f"missing LIBERO metadata; network fallback is forbidden: {info_path}")
    else:
        try:
            parsed = json.loads(info_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid LIBERO metadata {info_path}: {exc}")
        else:
            if not isinstance(parsed, dict):
                errors.append(f"LIBERO metadata is not an object: {info_path}")
            else:
                info = parsed
                for key, expected in LIBERO_DATASET_EXPECTED_INFO.items():
                    if info.get(key) != expected:
                        errors.append(
                            f"LIBERO metadata {key}={info.get(key)!r} does not match frozen {expected!r}"
                        )

    expected_files: list[str] = []
    if info:
        try:
            expected_files = _libero_expected_snapshot_files(info)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"cannot derive LIBERO data file set from metadata: {exc}")
    else:
        # Keep the result useful for diagnostics while preserving fail-closed
        # behavior when metadata is absent or malformed.
        expected_files = list(_libero_expected_snapshot_files(LIBERO_DATASET_EXPECTED_INFO))

    for relative in expected_files:
        path = root / relative
        if not path.is_file():
            errors.append(f"missing frozen LIBERO snapshot file: {path}")
        elif path.is_symlink():
            errors.append(f"LIBERO snapshot file must not be a symlink: {path}")

    observed_files: set[str] = set()
    if root.is_dir():
        try:
            for path in root.rglob("*"):
                relative = path.relative_to(root).as_posix()
                if relative == LIBERO_DATASET_MANIFEST_NAME or relative.startswith(".cache/"):
                    continue
                if path.is_file() or path.is_symlink():
                    observed_files.add(relative)
                    if path.is_symlink():
                        errors.append(f"LIBERO snapshot file must not be a symlink: {path}")
        except OSError as exc:
            errors.append(f"cannot enumerate LIBERO snapshot: {exc}")
    else:
        errors.append(f"missing LIBERO dataset root: {root}")
    extras = sorted(observed_files - set(expected_files))
    if extras:
        errors.append(f"unexpected files in frozen LIBERO snapshot: {extras[:8]}")

    if info:
        tasks_path = root / "meta" / "tasks.jsonl"
        episodes_path = root / "meta" / "episodes.jsonl"
        if tasks_path.is_file():
            error = _jsonl_line_count(tasks_path, int(LIBERO_DATASET_EXPECTED_INFO["total_tasks"]))
            if error:
                errors.append(error)
        if episodes_path.is_file():
            error = _jsonl_line_count(episodes_path, int(LIBERO_DATASET_EXPECTED_INFO["total_episodes"]))
            if error:
                errors.append(error)
        stats_path = root / "meta" / "stats.json"
        if stats_path.is_file():
            try:
                if not isinstance(json.loads(stats_path.read_text(encoding="utf-8")), dict):
                    errors.append(f"LIBERO stats metadata is not an object: {stats_path}")
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"invalid LIBERO stats metadata {stats_path}: {exc}")

    # ``snapshot_download(local_dir=...)`` writes one .metadata sidecar per
    # file.  Its first line is the resolved revision.  Requiring all sidecars
    # prevents a locally assembled tree from masquerading as the pinned repo.
    sidecar_root = root / ".cache" / "huggingface" / "download"
    sidecar_files: set[str] = set()
    if not sidecar_root.is_dir():
        errors.append(f"missing Hugging Face local-download metadata: {sidecar_root}")
    else:
        try:
            for sidecar in sidecar_root.rglob("*.metadata"):
                sidecar_files.add(sidecar.relative_to(sidecar_root).as_posix()[: -len(".metadata")])
            partial = list(sidecar_root.rglob("*.incomplete"))
            if partial:
                errors.append(f"incomplete Hugging Face downloads remain: {partial[:4]}")
        except OSError as exc:
            errors.append(f"cannot enumerate Hugging Face local-download metadata: {exc}")
        for relative in expected_files:
            sidecar = sidecar_root / f"{relative}.metadata"
            if not sidecar.is_file():
                errors.append(f"missing Hugging Face revision sidecar: {sidecar}")
                continue
            try:
                lines = sidecar.read_text(encoding="utf-8").splitlines()
            except OSError as exc:
                errors.append(f"cannot read Hugging Face revision sidecar {sidecar}: {exc}")
                continue
            if not lines or lines[0].strip() != LIBERO_DATASET_REVISION:
                errors.append(
                    f"Hugging Face sidecar revision mismatch for {relative}: "
                    f"{lines[0].strip() if lines else '<empty>'}"
                )
            if len(lines) < 2 or not lines[1].strip():
                errors.append(f"Hugging Face sidecar has no content digest for {relative}: {sidecar}")
        extra_sidecars = sorted(sidecar_files - set(expected_files))
        if extra_sidecars:
            errors.append(f"unexpected Hugging Face sidecars: {extra_sidecars[:8]}")

    manifest: dict[str, Any] | None = None
    checksum_rows: dict[str, str] = {}
    if not manifest_path.is_file():
        errors.append(f"missing pre-generated dataset checksum manifest: {manifest_path}")
    else:
        checksum_rows, checksum_errors = _read_sha256sum_manifest(manifest_path, expected_files)
        errors.extend(checksum_errors)
        observed_manifest_sha = sha256_file(manifest_path)
        if observed_manifest_sha != LIBERO_DATASET_MANIFEST_SHA256:
            errors.append(
                "dataset checksum manifest SHA-256 mismatch: "
                f"expected {LIBERO_DATASET_MANIFEST_SHA256}, got {observed_manifest_sha}"
            )
        if verify_hashes or persist_manifest:
            for relative, expected_digest in checksum_rows.items():
                path = root / relative
                if not path.is_file():
                    continue
                observed_digest = sha256_file(path)
                if observed_digest != expected_digest:
                    errors.append(
                        f"dataset file SHA-256 mismatch for {relative}: "
                        f"expected {expected_digest}, got {observed_digest}"
                    )
        manifest = {
            "schema": "r142-stage-s-c-libero-dataset-sha256sum-v1",
            "status": "COMPLETED",
            "repo_id": LIBERO_DATASET_REPO,
            "revision": LIBERO_DATASET_REVISION,
            "root": str(root),
            "manifest_path": str(manifest_path),
            "file_count": len(checksum_rows),
            "manifest_sha256": observed_manifest_sha,
            "checksum_format": "sha256sum -c DATASET_SHA256SUMS",
            "pre_generated": True,
        }

    return {
        "valid": not errors,
        "root": str(root),
        "repo_id": LIBERO_DATASET_REPO,
        "revision": LIBERO_DATASET_REVISION,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest.get("manifest_sha256") if manifest else None,
        "manifest_file_sha256": sha256_file(manifest_path) if manifest_path.is_file() else None,
        "file_count": manifest.get("file_count") if manifest else len(expected_files),
        "total_bytes": (
            sum(
                (root / relative).stat().st_size
                for relative in checksum_rows
                if (root / relative).is_file()
            )
            if manifest
            else None
        ),
        "errors": errors,
    }


def stage_libero_norm_stats(
    assets_base_dir: str | Path,
    staged_assets_base_dir: str | Path = DEFAULT_STAGED_LIBERO_ASSETS_BASE_DIR,
) -> dict[str, Any]:
    """Stage LIBERO norm stats at the path OpenPI actually resolves.

    The downloaded base checkpoint stores the file under
    ``pi05_libero/assets/physical-intelligence/libero``.  The pinned
    ``TrainConfig.assets_dirs`` resolver looks under
    ``assets_base_dir/pi05_libero/physical-intelligence/libero``.  Copying to
    a separate, stable CPFS staging root makes that path explicit and avoids
    mutating the downloaded base artifact.  Source and destination hashes,
    plus the marker hash, are checked on every resume.
    """

    source_root = Path(assets_base_dir).expanduser().resolve()
    staged_root = Path(staged_assets_base_dir).expanduser().resolve()
    source = source_root / LIBERO_NORM_STATS_SOURCE_RELATIVE
    destination = staged_root / LIBERO_NORM_STATS_RUNTIME_RELATIVE
    marker_path = staged_root / LIBERO_NORM_STATS_MARKER_NAME
    errors: list[str] = []
    if not source.is_file():
        errors.append(f"missing source LIBERO norm stats: {source}")
        return {
            "valid": False,
            "source_path": str(source),
            "staged_path": str(destination),
            "marker_path": str(marker_path),
            "source_sha256": None,
            "staged_sha256": None,
            "errors": errors,
        }
    if source.is_symlink():
        errors.append(f"source LIBERO norm stats must not be a symlink: {source}")
    try:
        source_payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"invalid source LIBERO norm stats {source}: {exc}")
        source_payload = None
    if not isinstance(source_payload, dict) or not source_payload:
        errors.append(f"source LIBERO norm stats is not a non-empty JSON object: {source}")
    source_digest = sha256_file(source)
    staged_root.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if not destination.is_file() or destination.is_symlink():
            errors.append(f"staged LIBERO norm stats must be a regular file: {destination}")
    elif not errors:
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        try:
            shutil.copyfile(source, temporary)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            temporary.replace(destination)
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError as exc:
            errors.append(f"cannot stage LIBERO norm stats: {exc}")
            with contextlib.suppress(OSError):
                temporary.unlink()
    staged_digest = sha256_file(destination) if destination.is_file() else None
    if staged_digest != source_digest:
        errors.append(
            f"staged LIBERO norm stats hash mismatch: source={source_digest} staged={staged_digest}"
        )
    marker_payload: dict[str, Any] = {
        "schema": "r142-stage-s-c-libero-norm-stats-v1",
        "status": "COMPLETED",
        "repo_id": LIBERO_DATASET_REPO,
        "dataset_revision": LIBERO_DATASET_REVISION,
        "source_path": str(source),
        "source_sha256": source_digest,
        "staged_path": str(destination),
        "staged_sha256": staged_digest,
        "runtime_relative_path": LIBERO_NORM_STATS_RUNTIME_RELATIVE,
        "resolver_contract": "assets_base_dir/pi05_libero/physical-intelligence/libero/norm_stats.json",
    }
    if marker_path.is_file():
        try:
            existing_marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid persisted LIBERO norm stats marker {marker_path}: {exc}")
        else:
            if not isinstance(existing_marker, dict) or existing_marker.get("payload_sha256") != _canonical_sha256(
                {key: value for key, value in existing_marker.items() if key != "payload_sha256"}
            ):
                errors.append(f"persisted LIBERO norm stats marker payload hash mismatch: {marker_path}")
            elif any(existing_marker.get(key) != value for key, value in marker_payload.items()):
                errors.append(f"persisted LIBERO norm stats marker differs from current bytes: {marker_path}")
    elif not errors:
        _status_marker(marker_path, marker_payload)
    return {
        "valid": not errors,
        "source_path": str(source),
        "staged_path": str(destination),
        "marker_path": str(marker_path),
        "source_sha256": source_digest,
        "staged_sha256": staged_digest,
        "runtime_relative_path": LIBERO_NORM_STATS_RUNTIME_RELATIVE,
        "errors": errors,
    }


def audit_libero_data_assets(
    assets_base_dir: str | Path,
    staged_assets_base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Check source and, when supplied, runtime-resolved norm-stat paths."""

    root = Path(assets_base_dir).expanduser().resolve()
    source = root / LIBERO_NORM_STATS_SOURCE_RELATIVE
    errors: list[str] = []
    if not source.is_file():
        errors.append(f"missing pinned LeRobot LIBERO norm stats: {source}")
    source_digest = sha256_file(source) if source.is_file() else None
    result: dict[str, Any] = {
        "valid": not errors,
        "root": str(root),
        "asset_id": LIBERO_DATASET_REPO,
        "norm_stats": str(source),
        "norm_stats_sha256": source_digest,
        "runtime_norm_stats": None,
        "runtime_norm_stats_sha256": None,
        "errors": errors,
    }
    if staged_assets_base_dir is not None:
        staged = Path(staged_assets_base_dir).expanduser().resolve() / LIBERO_NORM_STATS_RUNTIME_RELATIVE
        result["runtime_norm_stats"] = str(staged)
        if not staged.is_file():
            errors.append(f"missing staged OpenPI LIBERO norm stats: {staged}")
        else:
            result["runtime_norm_stats_sha256"] = sha256_file(staged)
            if result["runtime_norm_stats_sha256"] != source_digest:
                errors.append("source and runtime LIBERO norm stats hashes differ")
        result["valid"] = not errors
    return result


def _ast_function_parameters(path: Path, function_name: str) -> tuple[str, ...]:
    """Return a function's declared parameters without importing OpenPI."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            arguments = node.args
            return tuple(
                argument.arg
                for argument in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)
            )
    raise ValueError(f"function {function_name!r} not found in {path}")


def _ast_dataclass_fields(path: Path, class_name: str) -> tuple[str, ...]:
    """Return annotated/assigned dataclass fields from a source class."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            fields: list[str] = []
            for child in node.body:
                if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                    fields.append(child.target.id)
                elif isinstance(child, ast.Assign):
                    fields.extend(
                        target.id for target in child.targets if isinstance(target, ast.Name)
                    )
            return tuple(fields)
    raise ValueError(f"class {class_name!r} not found in {path}")


def _audit_official_cli(converter: Path, trainer: Path, config: Path) -> dict[str, Any]:
    """Audit the actual Tyro signatures/config fields used by the pinned CLIs.

    This deliberately parses the pinned source instead of trusting a copied
    command string.  The converter exposes its five Tyro parameters directly;
    the trainer's command-line overrides are the fields on ``TrainConfig``
    consumed by ``tyro.extras.overridable_config_cli``.
    """

    errors: list[str] = []
    converter_parameters: tuple[str, ...] = ()
    trainer_fields: tuple[str, ...] = ()
    try:
        converter_parameters = _ast_function_parameters(converter, "main")
    except (OSError, SyntaxError, ValueError) as exc:
        errors.append(f"cannot parse converter CLI: {exc}")
    try:
        trainer_fields = _ast_dataclass_fields(config, "TrainConfig")
    except (OSError, SyntaxError, ValueError) as exc:
        errors.append(f"cannot parse trainer config CLI: {exc}")
    converter_expected = ("checkpoint_dir", "config_name", "output_path", "precision", "inspect_only")
    trainer_expected = (
        "exp_name",
        "checkpoint_base_dir",
        "save_interval",
        "num_train_steps",
        "seed",
        "keep_period",
        "pytorch_weight_path",
        "assets_base_dir",
        "num_workers",
        "resume",
    )
    converter_valid = converter_parameters == converter_expected and "tyro.cli(main)" in converter.read_text(
        encoding="utf-8", errors="replace"
    )
    trainer_text = trainer.read_text(encoding="utf-8", errors="replace")
    trainer_valid = all(field in trainer_fields for field in trainer_expected) and (
        "overridable_config_cli" in config.read_text(encoding="utf-8", errors="replace")
    )
    if not converter_valid:
        errors.append(
            f"converter CLI signature drift: {converter_parameters!r} != {converter_expected!r}"
        )
    if not trainer_valid:
        errors.append("trainer CLI config fields or overridable_config_cli entry drifted")
    return {
        "valid": not errors,
        "converter_parameters": list(converter_parameters),
        "converter_expected_parameters": list(converter_expected),
        "trainer_config_fields": list(trainer_fields),
        "trainer_required_overrides": list(trainer_expected),
        "trainer_entry": "scripts/train_pytorch.py:main -> openpi.training.config.cli",
        "errors": errors,
    }


def _resolve_python_executable(value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate
    resolved = shutil.which(str(candidate))
    return Path(resolved).resolve() if resolved else candidate.resolve()


def audit_openpi_checkout(
    root: str | Path,
    *,
    python: str | Path = DEFAULT_OPENPI_PYTHON,
) -> dict[str, Any]:
    """Audit the exact source tree, runtime, and source-level C contracts."""

    path = Path(root).expanduser().resolve()
    errors: list[str] = []
    observed_commit: str | None = None
    dirty = None
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        observed_commit = result.stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        errors.append(f"cannot read OpenPI git HEAD: {exc}")
    try:
        status_result = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
        dirty = bool(status_result.stdout.strip())
    except (OSError, subprocess.CalledProcessError) as exc:
        errors.append(f"cannot read OpenPI git status: {exc}")
    if dirty:
        errors.append("OpenPI checkout is dirty; a clean exact pin is required")
    if observed_commit != OPENPI_COMMIT:
        errors.append(f"OpenPI commit mismatch: {observed_commit!r} != {OPENPI_COMMIT}")
    converter = path / OPENPI_CONVERTER
    trainer = path / OPENPI_TRAINER
    config = path / "src/openpi/training/config.py"
    for required in (converter, trainer, config):
        if not required.is_file():
            errors.append(f"missing pinned OpenPI source file: {required}")
    config_text = config.read_text(encoding="utf-8", errors="replace") if config.is_file() else ""
    trainer_text = trainer.read_text(encoding="utf-8", errors="replace") if trainer.is_file() else ""
    converter_text = converter.read_text(encoding="utf-8", errors="replace") if converter.is_file() else ""
    runtime_python = _resolve_python_executable(python)
    runtime_ok = runtime_python.is_file() and os.access(runtime_python, os.X_OK)
    runtime_version: str | None = None
    if not runtime_ok:
        errors.append(f"pinned OpenPI Python is not executable: {runtime_python}")
    else:
        try:
            version_result = subprocess.run(
                [str(runtime_python), "--version"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            runtime_version = (version_result.stdout or version_result.stderr).strip()
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            errors.append(f"cannot execute pinned OpenPI Python: {exc}")
    cli_audit = _audit_official_cli(converter, trainer, config) if all(
        required.is_file() for required in (converter, trainer, config)
    ) else {"valid": False, "errors": ["CLI source files missing"]}
    errors.extend(f"OpenPI CLI audit failed: {error}" for error in cli_audit.get("errors", []))
    checks = {
        "config_name_pi05_libero": 'name="pi05_libero"' in config_text,
        "dataset_repo_id": f'repo_id="{LIBERO_DATASET_REPO}"' in config_text,
        "libero_no_extra_delta": "extra_delta_transform=False" in config_text,
        "jax_base_uri": f'gs://{PI05_BASE_BUCKET}/{PI05_BASE_PREFIX}params' in config_text,
        "pi05_model": "Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False)" in config_text,
        "official_converter_entry": "def convert_pi0_checkpoint" in converter_text and "tyro.cli(main)" in converter_text,
        "official_trainer_entry": "def save_checkpoint" in trainer_text and "def train_loop" in trainer_text,
        "full_state_model": '"model.safetensors"' in trainer_text,
        "full_state_optimizer": '"optimizer.pt"' in trainer_text,
        "full_state_metadata": '"metadata.pt"' in trainer_text,
        "atomic_checkpoint_rename": "tmp_ckpt_dir.rename(final_ckpt_dir)" in trainer_text,
        "resume_native": "load_checkpoint(model, optim, config.checkpoint_dir, device)" in trainer_text,
        "converter_cli_exact": bool(cli_audit.get("valid")) and tuple(cli_audit.get("converter_parameters", ()))
        == ("checkpoint_dir", "config_name", "output_path", "precision", "inspect_only"),
        "trainer_cli_exact": bool(cli_audit.get("valid")),
        "trainer_cursor_loop_audited": all(
            snippet in trainer_text
            for snippet in (
                "while global_step < config.num_train_steps",
                "global_step // len(loader)",
                "for observation, actions in loader",
            )
        ),
    }
    errors.extend(f"OpenPI source check failed: {name}" for name, passed in checks.items() if not passed)
    return {
        "ready": not errors,
        "root": str(path),
        "source_url": OPENPI_SOURCE_URL,
        "expected_commit": OPENPI_COMMIT,
        "observed_commit": observed_commit,
        "dirty": dirty,
        "required_files": [str(converter), str(trainer), str(config)],
        "python": str(runtime_python),
        "python_version": runtime_version,
        "python_executable": runtime_ok,
        "cli_audit": cli_audit,
        "source_checks": checks,
        "errors": errors,
    }


def build_conversion_contract(
    *,
    openpi_root: str | Path,
    base_jax_root: str | Path,
    base_pytorch_root: str | Path,
    python: str | Path = DEFAULT_OPENPI_PYTHON,
    precision: str = "bfloat16",
) -> dict[str, Any]:
    """Build the exact official JAX-to-PyTorch command; no community base."""

    if precision not in {"bfloat16", "float32"}:
        raise ValueError("pinned converter supports only bfloat16 or float32 output")
    source = Path(openpi_root).expanduser().resolve()
    jax_root = Path(base_jax_root).expanduser().resolve()
    pytorch_root = Path(base_pytorch_root).expanduser().resolve()
    command = [
        str(_resolve_python_executable(python)),
        str(source / OPENPI_CONVERTER),
        "--checkpoint_dir",
        str(jax_root),
        "--config_name",
        OPENPI_CONFIG_NAME,
        "--output_path",
        str(pytorch_root),
        "--precision",
        precision,
    ]
    return {
        "schema": "r142-stage-s-c-openpi-conversion-contract-v1",
        "openpi_commit": OPENPI_COMMIT,
        "openpi_root": str(source),
        "base_jax_root": str(jax_root),
        "base_pytorch_root": str(pytorch_root),
        "config_name": OPENPI_CONFIG_NAME,
        "precision": precision,
        "official_converter": OPENPI_CONVERTER,
        "command": command,
        "source_audit": audit_openpi_checkout(source, python=python),
        "base_download_audit": audit_base_download(jax_root),
        "expected_outputs": ["model.safetensors", "config.json", CONVERSION_PROVENANCE_NAME, CONVERSION_COMPLETION_NAME],
        "community_checkpoint_forbidden": True,
    }


def _conversion_audit(path: Path) -> dict[str, Any]:
    errors: list[str] = []
    model = path / "model.safetensors"
    config = path / "config.json"
    provenance = path / CONVERSION_PROVENANCE_NAME
    marker = path / CONVERSION_COMPLETION_NAME
    for required in (model, config, provenance, marker):
        if not required.is_file():
            errors.append(f"missing conversion output: {required.name}")
    if provenance.is_file():
        try:
            payload = json.loads(provenance.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"invalid conversion provenance: {exc}")
        else:
            if payload.get("openpi_commit") != OPENPI_COMMIT:
                errors.append("conversion provenance OpenPI commit mismatch")
            if payload.get("config_name") != OPENPI_CONFIG_NAME:
                errors.append("conversion provenance config mismatch")
            observed_model_sha = sha256_file(model) if model.is_file() else None
            if payload.get("model_sha256") != observed_model_sha:
                errors.append("conversion provenance model hash mismatch")
    if marker.is_file():
        try:
            marker_payload = json.loads(marker.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"invalid conversion completion marker: {exc}")
        else:
            if marker_payload.get("status") != "COMPLETED":
                errors.append("conversion marker is not COMPLETED")
            if marker_payload.get("payload_sha256") != _canonical_sha256(
                {key: value for key, value in marker_payload.items() if key != "payload_sha256"}
            ):
                errors.append("conversion marker payload hash mismatch")
    return {"valid": not errors, "path": str(path), "errors": errors}


def run_conversion(
    contract: Mapping[str, Any],
    *,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Run the pinned converter and atomically publish conversion provenance."""

    if not contract.get("source_audit", {}).get("ready"):
        raise RuntimeError("OpenPI source audit failed; conversion is fail-closed")
    base_audit = contract.get("base_download_audit", {})
    if not base_audit.get("valid"):
        raise RuntimeError("base object audit failed; conversion is fail-closed")
    output = Path(str(contract["base_pytorch_root"])).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        existing = _conversion_audit(output)
        if existing["valid"]:
            return existing
        raise RuntimeError(f"refusing to overwrite incomplete conversion output: {existing['errors']}")
    output.mkdir(parents=True, exist_ok=True)
    runner(contract["command"], cwd=contract["openpi_root"], check=True)
    raw_errors = [
        f"missing official conversion output: {name}"
        for name in ("model.safetensors", "config.json")
        if not (output / name).is_file()
    ]
    if raw_errors:
        raise RuntimeError("official conversion returned without complete output: " + "; ".join(raw_errors))
    provenance = {
        "schema": "r142-stage-s-c-conversion-provenance-v1",
        "status": "COMPLETED",
        "openpi_source_url": OPENPI_SOURCE_URL,
        "openpi_commit": OPENPI_COMMIT,
        "official_converter": OPENPI_CONVERTER,
        "config_name": OPENPI_CONFIG_NAME,
        "precision": contract["precision"],
        "base_jax_root": contract["base_jax_root"],
        "base_manifest_sha256": contract["base_download_audit"].get("manifest", {}).get("manifest_sha256"),
        "model_sha256": sha256_file(output / "model.safetensors"),
    }
    atomic_json(output / CONVERSION_PROVENANCE_NAME, provenance)
    marker = {
        "schema": "r142-stage-s-c-conversion-completion-v1",
        "status": "COMPLETED",
        "openpi_commit": OPENPI_COMMIT,
        "config_name": OPENPI_CONFIG_NAME,
        "model_sha256": sha256_file(output / "model.safetensors"),
        "provenance_sha256": sha256_file(output / CONVERSION_PROVENANCE_NAME),
    }
    marker["payload_sha256"] = _canonical_sha256(marker)
    atomic_json(output / CONVERSION_COMPLETION_NAME, marker)
    audit = _conversion_audit(output)
    if not audit["valid"]:
        raise RuntimeError(f"conversion provenance audit failed after publish: {audit['errors']}")
    return {"valid": True, "path": str(output), "errors": [], "marker": marker}


def build_official_training_command(
    *,
    openpi_root: str | Path,
    base_pytorch_root: str | Path,
    checkpoint_base_dir: str | Path,
    python: str | Path = DEFAULT_OPENPI_PYTHON,
    world_size: int = PAI_MAX_GPU,
    assets_base_dir: str | Path | None = None,
) -> list[str]:
    """Return flags verified against the pinned ``train_pytorch.py`` source."""

    if int(world_size) != PAI_MAX_GPU:
        raise ValueError(f"C production command is frozen to {PAI_MAX_GPU} GPUs")
    experiment = "r142_stage_s_c_undertrained_seed42"
    command = [
        str(_resolve_python_executable(python)),
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={PAI_MAX_GPU}",
        str(Path(openpi_root).expanduser().resolve() / OPENPI_TRAINER),
        OPENPI_CONFIG_NAME,
        "--exp_name",
        experiment,
        "--checkpoint_base_dir",
        str(Path(checkpoint_base_dir).expanduser().resolve()),
        "--save_interval",
        str(C_SAVE_INTERVAL),
        "--num_train_steps",
        str(C_TRAINING_STEPS),
        "--seed",
        str(C_TRAINING_SEED),
        "--keep_period",
        str(C_SAVE_INTERVAL),
        "--num_workers",
        str(C_NUM_WORKERS),
        "--pytorch_weight_path",
        str(Path(base_pytorch_root).expanduser().resolve()),
    ]
    if assets_base_dir is not None:
        command.extend(["--assets_base_dir", str(Path(assets_base_dir).expanduser().resolve())])
    return command


def build_patched_training_command(
    *,
    worker_path: str | Path,
    openpi_root: str | Path,
    base_pytorch_root: str | Path,
    checkpoint_base_dir: str | Path,
    world_size: int = PAI_MAX_GPU,
    resume: bool = False,
    assets_base_dir: str | Path | None = None,
    python: str | Path = DEFAULT_OPENPI_PYTHON,
) -> list[str]:
    """Run the official trainer through the save/load-only RNG sidecar worker.

    The pinned trainer saves model/optimizer/metadata but not process RNG
    state.  The worker imports that exact trainer and wraps only its
    ``save_checkpoint``/``load_checkpoint`` functions, preserving the model,
    data, optimizer, and loss semantics while making same-directory resume a
    genuine full-state resume for all eight ranks.
    """

    command = [
        str(_resolve_python_executable(python)),
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={int(world_size)}",
        str(Path(worker_path).expanduser().resolve()),
        "--openpi-root",
        str(Path(openpi_root).expanduser().resolve()),
        "--",
        OPENPI_CONFIG_NAME,
        "--exp_name",
        "r142_stage_s_c_undertrained_seed42",
        "--checkpoint_base_dir",
        str(Path(checkpoint_base_dir).expanduser().resolve()),
        "--save_interval",
        str(C_SAVE_INTERVAL),
        "--num_train_steps",
        str(C_TRAINING_STEPS),
        "--seed",
        str(C_TRAINING_SEED),
        "--keep_period",
        str(C_SAVE_INTERVAL),
        "--num_workers",
        str(C_NUM_WORKERS),
        "--pytorch_weight_path",
        str(Path(base_pytorch_root).expanduser().resolve()),
    ]
    if assets_base_dir is not None:
        command.extend(["--assets_base_dir", str(Path(assets_base_dir).expanduser().resolve())])
    if resume:
        command.append("--resume")
    return command


def build_c_chain_contract(
    *,
    openpi_root: str | Path,
    base_jax_root: str | Path,
    base_pytorch_root: str | Path,
    checkpoint_base_dir: str | Path,
    log_root: str | Path,
    repo_root: str | Path,
    python: str | Path = DEFAULT_OPENPI_PYTHON,
    assets_base_dir: str | Path = DEFAULT_PI05_ASSETS_BASE_DIR,
    dataset_root: str | Path = DEFAULT_LIBERO_DATASET_ROOT,
    staged_assets_base_dir: str | Path = DEFAULT_STAGED_LIBERO_ASSETS_BASE_DIR,
) -> dict[str, Any]:
    """Describe the complete C asset -> conversion -> training hand-off."""

    source = Path(openpi_root).expanduser().resolve()
    jax_root = Path(base_jax_root).expanduser().resolve()
    pytorch_root = Path(base_pytorch_root).expanduser().resolve()
    checkpoint_root = Path(checkpoint_base_dir).expanduser().resolve()
    logs = Path(log_root).expanduser().resolve()
    repo = Path(repo_root).expanduser().resolve()
    data_snapshot = audit_libero_dataset_snapshot(dataset_root)
    data_assets = audit_libero_data_assets(assets_base_dir, staged_assets_base_dir)
    conversion = build_conversion_contract(
        openpi_root=source,
        base_jax_root=jax_root,
        base_pytorch_root=pytorch_root,
        python=python,
    )
    official_command = build_official_training_command(
        openpi_root=source,
        base_pytorch_root=pytorch_root,
        checkpoint_base_dir=checkpoint_root,
        python=python,
        assets_base_dir=assets_base_dir,
    )
    train_dir = checkpoint_root / OPENPI_CONFIG_NAME / "r142_stage_s_c_undertrained_seed42"
    wrapper = repo / "scripts" / "stage_s_libero_c_train.py"
    wrapper_command = [
        str(_resolve_python_executable(python)),
        str(wrapper),
        "--openpi-root",
        str(source),
        "--base-jax-root",
        str(jax_root),
        "--base-pytorch-root",
        str(pytorch_root),
        "--checkpoint-base-dir",
        str(checkpoint_root),
        "--log-root",
        str(logs),
        "--assets-base-dir",
        str(Path(assets_base_dir).expanduser().resolve()),
    ]
    worker_command = build_patched_training_command(
        worker_path=repo / "scripts" / "stage_s_libero_c_train_worker.py",
        openpi_root=source,
        base_pytorch_root=pytorch_root,
        checkpoint_base_dir=checkpoint_root,
        assets_base_dir=assets_base_dir,
        python=python,
    )
    return {
        "schema": "r142-stage-s-c-undertrained-chain-v1",
        "status": "READY_IF_PREFLIGHTS_PASS",
        "substrate": "C",
        "label": "WEAK_SUBSTRATE",
        "source": {
            "openpi_url": OPENPI_SOURCE_URL,
            "openpi_commit": OPENPI_COMMIT,
            "config_name": OPENPI_CONFIG_NAME,
            "dataset_repo_id": LIBERO_DATASET_REPO,
            "dataset_revision": LIBERO_DATASET_REVISION,
            "dataset_source_contract": "HuggingFace LeRobot repo resolved by pinned OpenPI LeRobotLiberoDataConfig; no synthetic/fake data",
            "dataset_root": str(Path(dataset_root).expanduser().resolve()),
            "dataset_manifest": data_snapshot,
            "assets_base_dir": str(Path(assets_base_dir).expanduser().resolve()),
            "staged_assets_base_dir": str(Path(staged_assets_base_dir).expanduser().resolve()),
            "base_jax_gcs_uri": PI05_BASE_GCS_URI,
            "base_object_count": PI05_BASE_OBJECT_COUNT,
            "base_total_bytes": PI05_BASE_TOTAL_BYTES,
        },
        "paths": {
            "openpi_root": str(source),
            "base_jax_root": str(jax_root),
            "base_pytorch_root": str(pytorch_root),
            "checkpoint_base_dir": str(checkpoint_root),
            "training_checkpoint_dir": str(train_dir),
            "log_root": str(logs),
            "repo_root": str(repo),
        },
        "conversion": conversion,
        "data_assets": data_assets,
        "data_snapshot": data_snapshot,
        "training": {
            "official_command": official_command,
            "resume_command": official_command + ["--resume"],
            "worker_command": worker_command,
            "wrapper_command": wrapper_command,
            "seed": C_TRAINING_SEED,
            "num_train_steps": C_TRAINING_STEPS,
            "optimizer_step_terminal": C_TRAINING_STEPS,
            "save_interval_steps": C_SAVE_INTERVAL,
            "checkpoint_steps": list(C_RETAIN_STEPS),
            "full_training_reference_steps": C_FULL_REFERENCE_STEP,
            "checkpoint_layout": "<checkpoint_base_dir>/pi05_libero/r142_stage_s_c_undertrained_seed42/<step>/",
            "full_state_components": ["model.safetensors", "optimizer.pt", "metadata.pt", "rng_state.rank{0..7}.pt"],
            "native_save_semantics": "pinned trainer saves 1000..10000; num_train_steps=10001 reaches terminal global_step 10001",
            "same_directory_resume": "--resume discovers newest numeric checkpoint and restores model+optimizer+metadata plus per-rank Python/NumPy/Torch/CUDA RNG sidecars",
            "num_workers": C_NUM_WORKERS,
            "exact_data_cursor": "worker wraps the pinned finite DataLoader epoch; resume skips global_step % epoch_length batches after setting DistributedSampler epoch=global_step // epoch_length",
            "data_cursor_fail_closed": "wrapper refuses resume when loader epoch length or checkpoint metadata cursor cannot be proven",
            "scheduler_state": "pinned trainer computes LR from config and global_step; the sidecar does not change this deterministic schedule",
            "community_sft_forbidden": True,
        },
        "resource": {
            "pool": "robot_idle",
            "worker": 1,
            "gpu": PAI_MAX_GPU,
            "cpu": PAI_MAX_CPU,
            "memory_gib": PAI_MAX_MEMORY_GIB,
            "shared_memory_gib": PAI_MAX_MEMORY_GIB,
            "no_pai_submit_performed": True,
        },
        "daily_no_job_windows": [dict(window) for window in STAGE_S_NO_JOB_WINDOWS],
        "fail_closed": {
            "blackout_timezone": "Asia/Shanghai",
            "blackout_windows": ["09:30-09:40", "19:30-19:40"],
            "missing_source_or_asset": True,
            "partial_checkpoint": True,
            "marker_written_only_after_complete_sha": True,
        },
        "ready_for_pai_submission": bool(
            conversion["source_audit"].get("ready")
            and conversion["base_download_audit"].get("valid")
            and data_snapshot.get("valid")
            and data_assets.get("valid")
        ),
    }


def assert_outside_blackout(now: datetime | None = None) -> datetime:
    """Fail closed during either Beijing no-submit interval."""

    current = now.astimezone(ZoneInfo("Asia/Shanghai")) if now is not None else datetime.now(ZoneInfo("Asia/Shanghai"))
    minute = current.hour * 60 + current.minute
    windows = ((9 * 60 + 30, 9 * 60 + 40), (19 * 60 + 30, 19 * 60 + 40))
    if any(start <= minute < end for start, end in windows):
        raise RuntimeError(f"Beijing blackout active at {current.isoformat()}; C operation is fail-closed")
    return current


def _checkpoint_audit(checkpoint_dir: Path) -> dict[str, Any]:
    """Use the existing real-checkpoint audit without importing Torch."""

    from .libero import audit_c_checkpoint_schedule

    return audit_c_checkpoint_schedule(checkpoint_dir, require_training_state=True)


def _write_sha256_manifest(root: Path) -> Path:
    """Write a cwd-relative manifest that is valid for ``sha256sum -c``.

    Checkpoint and log trees are intentionally separate bundles.  Keeping
    each manifest in the root it hashes avoids the previous mixed-root
    manifest, whose log entries could not be checked from the checkpoint cwd.
    """

    root.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != PI05_BASE_SHA_NAME:
            rows.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n")
    destination = root / PI05_BASE_SHA_NAME
    atomic_bytes(destination, "".join(rows).encode("utf-8"))
    return destination


def _status_marker(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Persist a status marker with a self-excluding canonical payload hash."""

    result = dict(payload)
    result["payload_sha256"] = _canonical_sha256(result)
    atomic_json(path, result)
    return result


def _audit_data_preflight(path: str | Path) -> tuple[dict[str, Any] | None, list[str]]:
    """Recheck the immutable local data gate before publishing training."""

    preflight = Path(path).expanduser().resolve()
    errors: list[str] = []
    if not preflight.is_file():
        return None, [f"missing C data preflight: {preflight}"]
    try:
        payload = json.loads(preflight.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"invalid C data preflight {preflight}: {exc}"]
    if not isinstance(payload, dict) or payload.get("status") != "COMPLETED":
        return None, ["C data preflight is not COMPLETED"]
    if payload.get("payload_sha256") != _canonical_sha256(
        {key: value for key, value in payload.items() if key != "payload_sha256"}
    ):
        return payload, ["C data preflight payload hash mismatch"]
    dataset = payload.get("dataset")
    norm_stats = payload.get("norm_stats")
    if not isinstance(dataset, dict) or not isinstance(norm_stats, dict):
        return None, ["C data preflight lacks dataset/norm_stats provenance"]
    if dataset.get("repo_id") != LIBERO_DATASET_REPO:
        errors.append("C data preflight dataset repo mismatch")
    if dataset.get("revision") != LIBERO_DATASET_REVISION:
        errors.append("C data preflight dataset revision mismatch")
    manifest_path = Path(str(dataset.get("manifest_path", ""))).expanduser().resolve()
    if not manifest_path.is_file():
        errors.append(f"C data preflight manifest is missing: {manifest_path}")
    else:
        # ``manifest_path`` is the pre-generated sha256sum text file, not the
        # JSON status marker.  Its two recorded hashes intentionally refer to
        # the same immutable bytes; parsing it as JSON would make every valid
        # training completion fail closed after the data gate had passed.
        observed_manifest_sha = sha256_file(manifest_path)
        if observed_manifest_sha != LIBERO_DATASET_MANIFEST_SHA256:
            errors.append(
                "C data preflight manifest SHA-256 mismatch: "
                f"expected {LIBERO_DATASET_MANIFEST_SHA256}, got {observed_manifest_sha}"
            )
        if dataset.get("manifest_sha256") != observed_manifest_sha:
            errors.append("C data preflight manifest content hash mismatch")
        if dataset.get("manifest_file_sha256") != observed_manifest_sha:
            errors.append("C data preflight manifest file hash mismatch")
    staged_path = Path(str(norm_stats.get("staged_path", ""))).expanduser().resolve()
    if not staged_path.is_file():
        errors.append(f"C data preflight staged norm stats is missing: {staged_path}")
    else:
        if sha256_file(staged_path) != norm_stats.get("staged_sha256"):
            errors.append("C data preflight staged norm stats hash mismatch")
    if norm_stats.get("source_sha256") != norm_stats.get("staged_sha256"):
        errors.append("C data preflight source/staged norm stats hashes differ")
    if errors:
        return payload, errors
    return payload, []


def finalize_training(
    *,
    checkpoint_base_dir: str | Path,
    log_root: str | Path,
    base_manifest_sha256: str,
    openpi_root: str | Path,
    data_preflight_path: str | Path | None = None,
) -> dict[str, Any]:
    """Publish terminal C evidence only after all four native checkpoints pass."""

    checkpoint_base = Path(checkpoint_base_dir).expanduser().resolve()
    train_dir = checkpoint_base / OPENPI_CONFIG_NAME / "r142_stage_s_c_undertrained_seed42"
    logs = Path(log_root).expanduser().resolve()
    terminal = logs / TRAINING_TERMINAL_NAME
    errors: list[str] = []
    data_preflight: dict[str, Any] | None = None
    if data_preflight_path is not None:
        data_preflight, data_errors = _audit_data_preflight(data_preflight_path)
        errors.extend(data_errors)
    audit = _checkpoint_audit(train_dir)
    if not audit.get("valid"):
        errors.extend(str(value) for value in audit.get("errors", []))
    if not terminal.is_file():
        errors.append(f"missing {terminal}")
    else:
        try:
            terminal_payload = json.loads(terminal.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"invalid training terminal marker: {exc}")
            terminal_payload = {}
        else:
            if terminal_payload.get("status") != "COMPLETED":
                errors.append("training terminal marker is not COMPLETED")
            if terminal_payload.get("openpi_commit") != OPENPI_COMMIT:
                errors.append("training terminal marker OpenPI commit mismatch")
            if int(terminal_payload.get("global_step", -1)) != C_TRAINING_STEPS:
                errors.append(f"terminal global_step != {C_TRAINING_STEPS}")
    # The pinned official trainer has no RNG save.  Our worker adds one
    # atomic sidecar per rank; require all eight before claiming resumability.
    for step in C_RETAIN_STEPS:
        step_dir = train_dir / str(step)
        for rank in range(PAI_MAX_GPU):
            if not (step_dir / f"rng_state.rank{rank}.pt").is_file():
                errors.append(f"missing full-state RNG sidecar: {step}/rng_state.rank{rank}.pt")
    if errors:
        _status_marker(
            logs / TRAINING_FAILED_NAME,
            {
                "schema": "r142-stage-s-c-training-failure-v1",
                "status": "FAILED",
                "openpi_commit": OPENPI_COMMIT,
                "errors": errors,
            },
        )
        raise RuntimeError("C finalization refused: " + "; ".join(errors))
    if checkpoint_base == logs:
        raise RuntimeError("C finalization requires separate checkpoint and log bundle roots")
    checkpoint_sums = _write_sha256_manifest(checkpoint_base)
    log_sums = _write_sha256_manifest(logs)
    data_provenance = {}
    if data_preflight is not None:
        dataset = data_preflight["dataset"]
        norm_stats = data_preflight["norm_stats"]
        data_provenance = {
            "data_preflight_path": str(Path(data_preflight_path).expanduser().resolve()),
            "data_preflight_sha256": sha256_file(Path(data_preflight_path).expanduser().resolve()),
            "dataset_repo_id": dataset.get("repo_id"),
            "dataset_revision": dataset.get("revision"),
            "dataset_root": dataset.get("root"),
            "dataset_manifest_path": dataset.get("manifest_path"),
            "dataset_manifest_sha256": dataset.get("manifest_sha256"),
            "dataset_manifest_file_sha256": dataset.get("manifest_file_sha256"),
            "norm_stats_source_path": norm_stats.get("source_path"),
            "norm_stats_source_sha256": norm_stats.get("source_sha256"),
            "norm_stats_staged_path": norm_stats.get("staged_path"),
            "norm_stats_sha256": norm_stats.get("staged_sha256"),
        }
    marker = _status_marker(
        checkpoint_base / TRAINING_COMPLETION_NAME,
        {
        "schema": "r142-stage-s-c-training-completion-v1",
        "status": "COMPLETED",
        "openpi_commit": OPENPI_COMMIT,
        "openpi_root": str(Path(openpi_root).expanduser().resolve()),
        "config_name": OPENPI_CONFIG_NAME,
        "seed": C_TRAINING_SEED,
        "terminal_global_step": C_TRAINING_STEPS,
        "checkpoint_steps": list(C_RETAIN_STEPS),
        "checkpoint_audit": audit,
        "base_manifest_sha256": base_manifest_sha256,
        "sha256sums": str(checkpoint_sums),
        "sha256sums_sha256": sha256_file(checkpoint_sums),
        "log_sha256sums": str(log_sums),
        "log_sha256sums_sha256": sha256_file(log_sums),
        "sha256_manifest_contract": "two cwd-relative manifests; run sha256sum -c SHA256SUMS separately in each root",
        **data_provenance,
        },
    )
    return marker


__all__ = [
    "C_NUM_WORKERS",
    "CONVERSION_COMPLETION_NAME",
    "CONVERSION_PROVENANCE_NAME",
    "DEFAULT_OPENPI_PYTHON",
    "DEFAULT_HF_LEROBOT_HOME",
    "DEFAULT_LIBERO_DATASET_ROOT",
    "DEFAULT_STAGED_LIBERO_ASSETS_BASE_DIR",
    "DEFAULT_PI05_ASSETS_BASE_DIR",
    "GCSObject",
    "LIBERO_DATASET_REPO",
    "LIBERO_DATASET_REVISION",
    "LIBERO_DATASET_MANIFEST_NAME",
    "LIBERO_DATASET_MANIFEST_SHA256",
    "LIBERO_DATASET_EXPECTED_INFO",
    "LIBERO_NORM_STATS_SOURCE_RELATIVE",
    "LIBERO_NORM_STATS_RUNTIME_RELATIVE",
    "LIBERO_NORM_STATS_MARKER_NAME",
    "OPENPI_COMMIT",
    "OPENPI_CONFIG_NAME",
    "OPENPI_CONVERTER",
    "OPENPI_SOURCE_URL",
    "OPENPI_TRAINER",
    "PI05_BASE_COMPLETION_NAME",
    "PI05_BASE_GCS_URI",
    "PI05_BASE_HTTP_PREFIX",
    "PI05_BASE_LISTING_URL",
    "PI05_BASE_MANIFEST_NAME",
    "PI05_BASE_OBJECT_COUNT",
    "PI05_BASE_OBJECTS",
    "PI05_BASE_PREFIX",
    "PI05_BASE_SHA_NAME",
    "PI05_BASE_TOTAL_BYTES",
    "TRAINING_COMPLETION_NAME",
    "TRAINING_FAILED_NAME",
    "TRAINING_START_NAME",
    "TRAINING_TERMINAL_NAME",
    "assert_outside_blackout",
    "audit_base_download",
    "audit_libero_data_assets",
    "audit_libero_dataset_snapshot",
    "audit_openpi_checkout",
    "_audit_data_preflight",
    "build_c_chain_contract",
    "build_conversion_contract",
    "build_official_training_command",
    "build_patched_training_command",
    "download_base_checkpoint",
    "expected_base_manifest",
    "fetch_gcs_manifest",
    "finalize_training",
    "manifest_from_gcs_listing",
    "run_conversion",
    "stage_libero_norm_stats",
    "validate_base_manifest",
    "write_expected_base_manifest",
]
