"""Fail-closed execution runtime for Stage-S S4/S5.

Stage-S S4 and S5 are deliberately implemented as an execution layer above
the existing A/B/C runners.  This module does not manufacture a trajectory,
turn a hand-entered boolean into a rollout, or choose an oracle location from
the observed outcomes.  A substrate adapter must provide the real simulator,
policy, snapshot and terminal-state hooks; the frozen protocol must provide
every oracle/random/seed constant before an execution can start.

The public surface is intentionally small:

``ProtocolAuthority``
    Reads the machine-readable frozen protocol and rejects an authority which
    does not contain the complete S4/S5 seed and branch contract.
``discover_n32_families``
    Loads only immutable, SHA-verified N=32 family bundles and returns every
    family, including all near-all-fail families.
``S45Adapter``
    Interface used by real LIBERO/Robotwin adapters.  No synthetic fallback
    exists: an unimplemented hook raises ``S45CapabilityError``.
``run_s4`` / ``run_s5``
    Execute the prefix-preserving oracle/random probes and fresh 32-candidate
    extension, respectively, with atomic completion markers and manifests.

The fixture tests implement this interface with a tiny deterministic simulator
only to test persistence and fail-closed behavior.  Production callers must
bind the official simulator/policy and real snapshot hooks.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

import numpy as np

from .analysis import compute_s4, compute_s5


PROTOCOL_ID = "r142-stage-s-v1"
BASE_CANDIDATE_COUNT = 32
EXTENDED_CANDIDATE_COUNT = 64
FRESH_CANDIDATE_INDICES = tuple(range(BASE_CANDIDATE_COUNT, EXTENDED_CANDIDATE_COUNT))
NEAR_ALL_FAIL_MAX_SUCCESS = 1
S4_BOOTSTRAP_REPLICATES = 10_000
SNAPSHOT_REPLAY_TOLERANCE = 1e-9


class S45Error(RuntimeError):
    """Base class for fail-closed S4/S5 errors."""


class S45ProtocolError(S45Error):
    """The frozen protocol authority is absent or incomplete."""


class S45CapabilityError(S45Error):
    """A real adapter did not expose a required simulator/policy hook."""


class S45BundleError(S45Error):
    """A source or output artifact is incomplete, inconsistent, or tampered."""


class S45ProvenanceError(S45Error):
    """A result cannot be tied to its family, prefix, protocol, or terminal run."""


def _jsonable(value: Any) -> Any:
    """Convert common array/tensor/pose values to deterministic JSON data."""

    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape), "data": value.tolist()}
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if hasattr(value, "detach") and callable(value.detach):
        tensor = value.detach().cpu()
        return {"dtype": str(tensor.dtype), "shape": list(tensor.shape), "data": tensor.tolist()}
    if hasattr(value, "p") and hasattr(value, "q"):
        return {"p": _jsonable(value.p), "q": _jsonable(value.q)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(child) for child in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_file(path: str | Path, *, label: str) -> Path:
    value = Path(path)
    if value.is_symlink() or not value.is_file():
        raise S45BundleError(f"{label} is missing or symlinked: {value}")
    return value


def _atomic_bytes(path: str | Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=str(destination.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: str | Path, payload: Any) -> None:
    _atomic_bytes(path, (json.dumps(_jsonable(payload), ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))


def _relative(root: Path, path: str | Path) -> str:
    value = Path(path)
    if value.is_absolute():
        try:
            value = value.relative_to(root)
        except ValueError as exc:
            raise S45BundleError(f"artifact outside bundle root: {path}") from exc
    relative = PurePosixPath(value.as_posix())
    if relative.is_absolute() or ".." in relative.parts or str(relative) in ("", "."):
        raise S45BundleError(f"unsafe artifact path: {path}")
    return relative.as_posix()


def write_atomic_bundle(
    root: str | Path,
    artifacts: Mapping[str, bytes | bytearray | str],
    *,
    marker_name: str,
    marker_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Write artifacts, marker, and SHA256SUMS in a fixed order.

    The marker is the last payload artifact and is itself included in the
    manifest.  A caller must not overwrite an already valid marker: repeated
    execution is a read-only verification/resume operation.
    """

    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    marker_path = base / marker_name
    if marker_path.exists():
        return verify_atomic_bundle(base, marker_name=marker_name)
    if not marker_name.startswith("COMPLETED_") or not marker_name.endswith(".json"):
        raise ValueError("completion marker must be COMPLETED_*.json")
    normalized: dict[str, bytes] = {}
    for relative, payload in artifacts.items():
        safe = _relative(base, relative)
        if safe in {"SHA256SUMS", marker_name} or safe.startswith("COMPLETED_"):
            raise ValueError("artifacts cannot contain completion or manifest files")
        normalized[safe] = payload.encode("utf-8") if isinstance(payload, str) else bytes(payload)
    if not normalized:
        raise ValueError("at least one artifact is required")
    for relative, payload in sorted(normalized.items()):
        _atomic_bytes(base / relative, payload)
    hashes = {relative: sha256_bytes(payload) for relative, payload in sorted(normalized.items())}
    marker: dict[str, Any] = dict(marker_payload)
    marker["files"] = dict(hashes)
    marker_bytes = (json.dumps(_jsonable(marker), ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    _atomic_bytes(marker_path, marker_bytes)
    hashes[marker_name] = sha256_bytes(marker_bytes)
    manifest = "".join(f"{digest}  {relative}\n" for relative, digest in sorted(hashes.items()))
    _atomic_bytes(base / "SHA256SUMS", manifest.encode("utf-8"))
    return verify_atomic_bundle(base, marker_name=marker_name)


def _manifest_entries(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 2 or not re.fullmatch(r"[0-9a-fA-F]{64}", fields[0]):
            raise S45BundleError(f"malformed SHA256SUMS line {line_number}: {path}")
        relative = PurePosixPath(fields[1])
        if relative.is_absolute() or ".." in relative.parts or fields[1] in entries:
            raise S45BundleError(f"unsafe or duplicate SHA256SUMS path: {fields[1]}")
        entries[fields[1]] = fields[0].lower()
    return entries


def verify_atomic_bundle(root: str | Path, *, marker_name: str) -> dict[str, Any]:
    base = Path(root)
    marker_path = _strict_file(base / marker_name, label="completion marker")
    manifest_path = _strict_file(base / "SHA256SUMS", label="SHA256SUMS")
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise S45BundleError(f"invalid completion marker: {marker_path}") from exc
    if not isinstance(marker, Mapping) or not isinstance(marker.get("files"), Mapping):
        raise S45BundleError(f"completion marker lacks files: {marker_path}")
    manifest = _manifest_entries(manifest_path)
    expected_files = {str(name): str(digest).lower() for name, digest in marker["files"].items()}
    if any(PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts for name in expected_files):
        raise S45BundleError("completion file path traversal")
    for relative, expected in expected_files.items():
        path = _strict_file(base / relative, label=f"completion artifact {relative}")
        if not re.fullmatch(r"[0-9a-f]{64}", expected) or sha256_file(path) != expected:
            raise S45BundleError(f"completion artifact hash mismatch: {relative}")
    if set(manifest) != set(expected_files) | {marker_name}:
        raise S45BundleError("SHA256SUMS does not match completion artifact set")
    for relative, expected in manifest.items():
        path = _strict_file(base / relative, label=f"manifest artifact {relative}")
        if sha256_file(path) != expected:
            raise S45BundleError(f"SHA256SUMS hash mismatch: {relative}")
    return {**dict(marker), "path": str(base), "completion_sha256": sha256_file(marker_path), "manifest_sha256": sha256_file(manifest_path)}


def _read_json(path: str | Path, *, label: str) -> Mapping[str, Any]:
    value = _strict_file(path, label=label)
    try:
        payload = json.loads(value.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise S45ProtocolError(f"{label} is not valid JSON: {value}") from exc
    if not isinstance(payload, Mapping):
        raise S45ProtocolError(f"{label} must be a JSON object")
    return payload


def _full_sha(value: Any, *, field: str, length: int = 64) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{%d}" % length, value):
        raise S45ProtocolError(f"{field} must be lowercase full SHA-{length * 4}")
    return value


@dataclass(frozen=True)
class ProtocolAuthority:
    """Complete machine-readable S4/S5 contract.

    The parser intentionally does not supply defaults for oracle grids,
    branch count, randomization, or extension seeds.  Those values belong to
    the pre-registered authority and missing values are a hard stop.
    """

    path: Path
    sha256: str
    payload: Mapping[str, Any]
    protocol_id: str
    git_commit: str
    s4: Mapping[str, Any]
    s5: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "ProtocolAuthority":
        source = _strict_file(path, label="frozen protocol authority")
        payload = _read_json(source, label="frozen protocol authority")
        protocol_id = payload.get("protocol_id", payload.get("id"))
        if not isinstance(protocol_id, str) or not protocol_id:
            raise S45ProtocolError("protocol authority lacks protocol_id")
        commit = payload.get("protocol_git_commit", payload.get("git_commit"))
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
            raise S45ProtocolError("protocol authority lacks a full 40-character git commit")
        s4 = payload.get("s4", payload.get("S4"))
        s5 = payload.get("s5", payload.get("S5"))
        if not isinstance(s4, Mapping) or not isinstance(s5, Mapping):
            frozen = payload.get("frozen_summary")
            if isinstance(frozen, Mapping):
                s4 = frozen.get("s4", frozen.get("S4", s4))
                s5 = frozen.get("s5", frozen.get("S5", s5))
        if not isinstance(s4, Mapping) or not isinstance(s5, Mapping):
            raise S45ProtocolError("protocol authority must contain explicit s4 and s5 objects")
        required_s4 = ("anchor_rule", "oracle_t_rule", "random_t_rule", "oracle_t_grid", "random_t_grid", "branch_count", "branch_seed_formula")
        required_s5 = ("base_candidate_count", "fresh_candidate_indices", "extension_seed_formula")
        for field in required_s4:
            if field not in s4:
                raise S45ProtocolError(f"protocol authority s4 lacks {field}")
        for field in required_s5:
            if field not in s5:
                raise S45ProtocolError(f"protocol authority s5 lacks {field}")
        if not isinstance(s4["anchor_rule"], str) or not s4["anchor_rule"].strip():
            raise S45ProtocolError("s4.anchor_rule must be explicit text")
        if not isinstance(s4["oracle_t_rule"], str) or not s4["oracle_t_rule"].strip():
            raise S45ProtocolError("s4.oracle_t_rule must be explicit text")
        if not isinstance(s4["random_t_rule"], str) or not s4["random_t_rule"].strip():
            raise S45ProtocolError("s4.random_t_rule must be explicit text")
        if not isinstance(s4["branch_seed_formula"], str) or not s4["branch_seed_formula"].strip():
            raise S45ProtocolError("s4.branch_seed_formula must be explicit text")
        try:
            branch_count = int(s4["branch_count"])
        except (TypeError, ValueError) as exc:
            raise S45ProtocolError("s4.branch_count must be a positive integer") from exc
        if branch_count <= 0:
            raise S45ProtocolError("s4.branch_count must be positive")
        for grid_name in ("oracle_t_grid", "random_t_grid"):
            grid = s4[grid_name]
            if not isinstance(grid, (list, tuple)) or len(grid) != branch_count:
                raise S45ProtocolError(f"s4.{grid_name} must contain exactly branch_count entries")
            if not all(isinstance(item, (int, np.integer)) and not isinstance(item, bool) for item in grid):
                raise S45ProtocolError(f"s4.{grid_name} must contain integer control steps")
        try:
            base_count = int(s5["base_candidate_count"])
        except (TypeError, ValueError) as exc:
            raise S45ProtocolError("s5.base_candidate_count must be an integer") from exc
        if base_count != BASE_CANDIDATE_COUNT:
            raise S45ProtocolError("s5.base_candidate_count is frozen at 32")
        if list(s5["fresh_candidate_indices"]) != list(FRESH_CANDIDATE_INDICES):
            raise S45ProtocolError("s5.fresh_candidate_indices must be exactly 32..63")
        if not isinstance(s5["extension_seed_formula"], str) or not s5["extension_seed_formula"].strip():
            raise S45ProtocolError("s5.extension_seed_formula must be explicit text")
        return cls(
            path=source,
            sha256=sha256_file(source),
            payload=payload,
            protocol_id=protocol_id,
            git_commit=commit.lower(),
            s4=s4,
            s5=s5,
        )

    def identity(self) -> dict[str, str]:
        return {
            "protocol_id": self.protocol_id,
            "protocol_authority_path": str(self.path),
            "protocol_authority_sha256": self.sha256,
            "protocol_git_commit": self.git_commit,
        }

    def _grid(self, name: str, *, episode_length: int, family_id: str) -> tuple[int, ...]:
        raw = self.s4[name]
        if isinstance(raw, Mapping):
            if family_id not in raw and str(family_id) not in raw:
                raise S45ProtocolError(f"s4.{name} has no entry for family {family_id}")
            raw = raw.get(family_id, raw.get(str(family_id)))
        if not isinstance(raw, (list, tuple)) or len(raw) != int(self.s4["branch_count"]):
            raise S45ProtocolError(f"s4.{name} length is not frozen branch_count")
        values = tuple(int(item) for item in raw)
        if any(not (0 < item < int(episode_length) - 1) for item in values):
            raise S45ProtocolError(f"s4.{name} contains a non-interior control step for family {family_id}")
        return values

    def oracle_steps(self, family_id: str, episode_length: int) -> tuple[int, ...]:
        return self._grid("oracle_t_grid", episode_length=episode_length, family_id=family_id)

    def random_steps(self, family_id: str, episode_length: int) -> tuple[int, ...]:
        return self._grid("random_t_grid", episode_length=episode_length, family_id=family_id)

    @property
    def branch_count(self) -> int:
        return int(self.s4["branch_count"])


def _call(fn: Callable[..., Any], **kwargs: Any) -> Any:
    """Call a required adapter hook without masking exceptions from its body."""

    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(**kwargs)
    parameters = signature.parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return fn(**kwargs)
    accepted = {key: value for key, value in kwargs.items() if key in parameters}
    missing = [name for name, parameter in parameters.items() if parameter.default is inspect.Parameter.empty and parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD) and name not in accepted]
    if missing:
        raise S45CapabilityError(f"adapter hook {fn!r} does not accept required arguments {missing}")
    return fn(**accepted)


class S45Adapter:
    """Abstract real-substrate adapter.

    The methods are intentionally concrete failures rather than no-op defaults.
    A production implementation may delegate them to the existing
    ``libero.py``/``robotwin.py`` snapshot hooks, but every method required by
    S4/S5 must remain observable in the result provenance.
    """

    def select_anchor(self, family: Mapping[str, Any], *, protocol: ProtocolAuthority) -> Mapping[str, Any]:
        raise S45CapabilityError("adapter must implement select_anchor under the frozen s4.anchor_rule")

    def replay_prefix(self, family: Mapping[str, Any], anchor: Mapping[str, Any], split_step: int, *, protocol: ProtocolAuthority) -> Mapping[str, Any]:
        raise S45CapabilityError("adapter must implement replay_prefix with simulator/history/queue/all-RNG snapshot")

    def branch_seed(self, family: Mapping[str, Any], anchor: Mapping[str, Any], split_step: int, branch_index: int, mode: str, *, protocol: ProtocolAuthority) -> int:
        raise S45CapabilityError("adapter must implement branch_seed from the frozen s4.branch_seed_formula")

    def run_branch(self, family: Mapping[str, Any], anchor: Mapping[str, Any], prefix: Mapping[str, Any], split_step: int, branch_seed: int, branch_index: int, mode: str, *, protocol: ProtocolAuthority) -> Mapping[str, Any]:
        raise S45CapabilityError("adapter must implement run_branch using a restored real snapshot and official termination")

    def extension_seed(self, family: Mapping[str, Any], candidate_index: int, *, protocol: ProtocolAuthority) -> int:
        raise S45CapabilityError("adapter must implement extension_seed from the frozen s5.extension_seed_formula")

    def run_fresh_candidate(self, family: Mapping[str, Any], candidate_index: int, candidate_seed: int, *, protocol: ProtocolAuthority) -> Mapping[str, Any]:
        raise S45CapabilityError("adapter must implement run_fresh_candidate from the same task/checkpoint")

    def close(self) -> None:
        return None


def _coerce_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise S45ProvenanceError(f"{label} must be a mapping")
    return value


def _candidate_success(row: Mapping[str, Any]) -> bool:
    if "success" in row:
        value = row["success"]
    elif "final_success" in row:
        value = row["final_success"]
    else:
        raise S45BundleError("candidate lacks eventual terminal success label")
    if not isinstance(value, (bool, np.bool_, int, np.integer)):
        raise S45BundleError("candidate success must be a persisted boolean")
    return bool(value)


def _candidate_id(row: Mapping[str, Any], fallback: int) -> str:
    value = row.get("candidate_id", row.get("id", fallback))
    return str(value)


def _candidate_seed(row: Mapping[str, Any], fallback: int) -> int:
    value = row.get("candidate_seed", row.get("seed"))
    if value is None:
        raise S45BundleError(f"candidate {fallback} lacks its independent seed")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise S45BundleError(f"candidate {fallback} seed is invalid") from exc


def _normalise_candidate(row: Mapping[str, Any], index: int, *, success: Any | None = None, seed: Any | None = None) -> dict[str, Any]:
    output = {str(key): copy.deepcopy(value) for key, value in row.items()}
    output["candidate_index"] = int(row.get("candidate_index", index))
    output["candidate_id"] = _candidate_id(row, index)
    output["candidate_seed"] = _candidate_seed(row, index) if seed is None else int(seed)
    output["success"] = _candidate_success(row) if success is None else bool(success)
    if "actions" not in output and "action_prefix" in output:
        output["actions"] = output["action_prefix"]
    if "trajectory" not in output:
        for key in ("poses", "pose_trajectory", "eef_trajectory"):
            if key in output:
                output["trajectory"] = output[key]
                break
    if "actions" not in output or "trajectory" not in output:
        raise S45BundleError(f"candidate {output['candidate_id']} lacks raw actions and trajectory")
    if "termination" not in output and "termination_reason" not in output:
        # Family metadata can supply the official termination contract; the
        # loader copies it into each row below.
        output["termination"] = None
    return output


def _load_family_candidates(directory: Path, marker: Mapping[str, Any]) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    metadata: dict[str, Any] = {}
    metadata_path = directory / "metadata.json"
    if metadata_path.is_file() and not metadata_path.is_symlink():
        raw = _read_json(metadata_path, label="family metadata")
        metadata.update(raw)
    family_json = directory / "family.json"
    rollouts_npz = directory / "rollouts.npz"
    candidates: list[dict[str, Any]] = []
    if family_json.is_file() and not family_json.is_symlink():
        payload = _read_json(family_json, label="family raw result")
        raw_candidates = payload.get("candidates")
        if not isinstance(raw_candidates, list):
            raise S45BundleError(f"family.json candidates is not a list: {family_json}")
        if isinstance(payload.get("metadata"), Mapping):
            metadata = {**dict(payload["metadata"]), **metadata}
        for index, row in enumerate(raw_candidates):
            if not isinstance(row, Mapping):
                raise S45BundleError(f"family candidate {index} is not a mapping")
            candidates.append(_normalise_candidate(row, index))
    elif rollouts_npz.is_file() and not rollouts_npz.is_symlink():
        genealogy_path = directory / "genealogy.json"
        if not genealogy_path.is_file() or genealogy_path.is_symlink():
            raise S45BundleError(f"rollouts.npz family lacks genealogy.json: {directory}")
        genealogy = json.loads(genealogy_path.read_text(encoding="utf-8"))
        if not isinstance(genealogy, list) or len(genealogy) != int(marker.get("candidate_count", -1)):
            raise S45BundleError(f"genealogy count mismatch: {genealogy_path}")
        with np.load(rollouts_npz, allow_pickle=False) as arrays:
            required = ("success", "candidate_seed", "actions", "poses", "lengths", "offsets")
            missing = [name for name in required if name not in arrays]
            if missing:
                raise S45BundleError(f"rollouts.npz lacks {missing}: {rollouts_npz}")
            success_values = np.asarray(arrays["success"]).reshape(-1)
            seeds = np.asarray(arrays["candidate_seed"]).reshape(-1)
            lengths = np.asarray(arrays["lengths"], dtype=np.int64).reshape(-1)
            offsets = np.asarray(arrays["offsets"], dtype=np.int64).reshape(-1)
            actions = np.asarray(arrays["actions"])
            poses = np.asarray(arrays["poses"])
            if len(genealogy) != len(success_values) or len(success_values) != len(seeds) or len(lengths) != len(success_values) or len(offsets) != len(success_values) + 1:
                raise S45BundleError(f"rollouts.npz candidate arrays disagree: {rollouts_npz}")
            for index, row in enumerate(genealogy):
                if not isinstance(row, Mapping):
                    raise S45BundleError(f"genealogy row {index} is not a mapping")
                start, stop = int(offsets[index]), int(offsets[index + 1])
                if stop - start != int(lengths[index]) or start < 0 or stop > len(actions) or stop > len(poses):
                    raise S45BundleError(f"trajectory offsets invalid for candidate {index}")
                candidate = dict(row)
                candidate["actions"] = actions[start:stop].tolist()
                candidate["trajectory"] = poses[start:stop].tolist()
                candidates.append(_normalise_candidate(candidate, index, success=success_values[index], seed=seeds[index]))
    else:
        raise S45BundleError(f"family lacks family.json or rollouts.npz: {directory}")
    if len(candidates) != int(marker.get("candidate_count", -1)):
        raise S45BundleError(f"candidate count mismatch in {directory}")
    termination = metadata.get("termination", metadata.get("termination_reason"))
    for candidate in candidates:
        if candidate.get("termination") is None and termination is not None:
            candidate["termination"] = termination
        if candidate.get("termination") is None:
            raise S45BundleError(f"candidate {candidate['candidate_id']} lacks official termination evidence")
    family_id = str(marker.get("family_id", metadata.get("family_id", directory.name)))
    metadata.setdefault("family_id", family_id)
    metadata.setdefault("source_directory", str(directory))
    return family_id, metadata, candidates


@dataclass(frozen=True)
class N32Family:
    family_id: str
    directory: Path
    marker: Mapping[str, Any]
    metadata: Mapping[str, Any]
    candidates: tuple[Mapping[str, Any], ...]
    source_marker_sha256: str
    source_bundle_sha256: str

    def as_mapping(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "directory": str(self.directory),
            "marker": dict(self.marker),
            "metadata": dict(self.metadata),
            "candidates": [dict(candidate) for candidate in self.candidates],
            "source_marker_sha256": self.source_marker_sha256,
            "source_bundle_sha256": self.source_bundle_sha256,
            "task_id": self.metadata.get("task_id"),
            "init_state_id": self.metadata.get("init_state", self.metadata.get("initial_state_id")),
        }


def _verify_source_marker(marker_path: Path) -> Mapping[str, Any]:
    try:
        marker = json.loads(_strict_file(marker_path, label="N32 completion marker").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise S45BundleError(f"invalid N32 completion marker: {marker_path}") from exc
    if not isinstance(marker, Mapping) or not isinstance(marker.get("files"), Mapping):
        raise S45BundleError(f"N32 completion marker lacks file hashes: {marker_path}")
    directory = marker_path.parent
    manifest = directory / "SHA256SUMS"
    _strict_file(manifest, label="N32 SHA256SUMS")
    entries = _manifest_entries(manifest)
    expected = {str(name): str(value).lower() for name, value in marker["files"].items()}
    for relative, digest in expected.items():
        path = _strict_file(directory / relative, label=f"N32 artifact {relative}")
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or sha256_file(path) != digest:
            raise S45BundleError(f"N32 artifact hash mismatch: {marker_path}/{relative}")
    if set(entries) not in ({*expected}, {*expected, "COMPLETED_FAMILY.json"}):
        raise S45BundleError(f"N32 SHA256SUMS does not match marker files: {manifest}")
    for relative, digest in entries.items():
        path = _strict_file(directory / relative, label=f"N32 manifest artifact {relative}")
        if sha256_file(path) != digest:
            raise S45BundleError(f"N32 manifest hash mismatch: {manifest}/{relative}")
    return marker


def discover_n32_families(root: str | Path, *, protocol: ProtocolAuthority | None = None) -> tuple[N32Family, ...]:
    """Read every complete N=32 family under ``root``.

    The function never selects a convenient subset.  Duplicate family IDs,
    missing provenance, partial markers, candidate count drift, and a missing
    terminal/action/trajectory record are all hard failures.
    """

    base = Path(root)
    if not base.is_dir() or base.is_symlink():
        raise S45BundleError(f"N32 root is missing or symlinked: {base}")
    marker_paths = sorted(path for path in base.rglob("COMPLETED_FAMILY.json") if path.is_file() and not path.is_symlink())
    if not marker_paths:
        raise S45BundleError(f"no COMPLETED_FAMILY.json found below {base}")
    result: list[N32Family] = []
    seen: set[str] = set()
    for marker_path in marker_paths:
        marker = _verify_source_marker(marker_path)
        if int(marker.get("candidate_count", -1)) != BASE_CANDIDATE_COUNT:
            raise S45BundleError(f"N32 source candidate_count is not 32: {marker_path}")
        family_id, metadata, candidates = _load_family_candidates(marker_path.parent, marker)
        if family_id in seen:
            raise S45BundleError(f"duplicate N32 family id: {family_id}")
        seen.add(family_id)
        if protocol is not None:
            if metadata.get("protocol_id", marker.get("protocol_id")) != protocol.protocol_id:
                raise S45ProvenanceError(f"N32 family {family_id} has a different protocol id")
            declared_sha = metadata.get("protocol_authority_sha256", marker.get("protocol_authority_sha256"))
            if declared_sha != protocol.sha256:
                raise S45ProvenanceError(f"N32 family {family_id} is not bound to the frozen protocol authority")
            declared_commit = metadata.get("protocol_git_commit", marker.get("protocol_git_commit"))
            if declared_commit != protocol.git_commit:
                raise S45ProvenanceError(f"N32 family {family_id} has a different protocol git commit")
        result.append(
            N32Family(
                family_id=family_id,
                directory=marker_path.parent,
                marker=dict(marker),
                metadata=metadata,
                candidates=tuple(candidates),
                source_marker_sha256=sha256_file(marker_path),
                source_bundle_sha256=sha256_file(marker_path.parent / "SHA256SUMS"),
            )
        )
    return tuple(sorted(result, key=lambda family: family.family_id))


def _family_from_mapping(family: N32Family | Mapping[str, Any]) -> dict[str, Any]:
    return family.as_mapping() if isinstance(family, N32Family) else dict(family)


def _row_actions(row: Mapping[str, Any]) -> list[Any]:
    value = row.get("actions", row.get("action_prefix"))
    if value is None:
        raise S45ProvenanceError(f"candidate {row.get('candidate_id')} lacks actions")
    return _jsonable(value)


def _row_trajectory(row: Mapping[str, Any]) -> list[Any]:
    value = row.get("trajectory", row.get("poses", row.get("pose_trajectory")))
    if value is None:
        raise S45ProvenanceError(f"candidate {row.get('candidate_id')} lacks trajectory")
    return _jsonable(value)


def _trajectory_length(row: Mapping[str, Any]) -> int:
    value = row.get("env_steps")
    if value is not None:
        try:
            length = int(value)
            if length > 0:
                return length
        except (TypeError, ValueError):
            pass
    actions = _row_actions(row)
    if isinstance(actions, Mapping) and "data" in actions:
        actions = actions["data"]
    if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)) or len(actions) <= 0:
        raise S45ProvenanceError(f"candidate {row.get('candidate_id')} has no finite trajectory length")
    return len(actions)


def _canonical_digest(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def _require_branch_execution(value: Any, *, label: str) -> Mapping[str, Any]:
    result = _coerce_mapping(value, label=label)
    for field in ("actions", "trajectory", "terminated", "success", "policy_forwards", "env_steps", "snapshot_restore_check"):
        if field not in result:
            raise S45ProvenanceError(f"{label} lacks raw execution field {field}")
    if not bool(result["terminated"]):
        raise S45ProvenanceError(f"{label} did not reach official termination")
    replay = _coerce_mapping(result["snapshot_restore_check"], label=f"{label}.snapshot_restore_check")
    if replay.get("same_action") is not True or not bool(replay.get("passed")):
        raise S45ProvenanceError(f"{label} lacks a passing restore->same-action replay check")
    try:
        if float(replay.get("max_abs_error")) > SNAPSHOT_REPLAY_TOLERANCE:
            raise S45ProvenanceError(f"{label} restore replay error exceeds 1e-9")
    except (TypeError, ValueError) as exc:
        raise S45ProvenanceError(f"{label} has invalid replay error") from exc
    return result


def _require_full_snapshot(value: Any, *, label: str) -> Mapping[str, Any]:
    """Require persisted simulator/history/queue/all-RNG snapshot components."""

    snapshot = _coerce_mapping(value, label=label)
    aliases = {
        "simulator": ("simulator_state", "environment", "env_state"),
        "history": ("observation_history", "policy_observation_history", "history"),
        "queue": ("action_queue", "policy_action_queue", "queue"),
        "rng": ("rng_state", "all_rng", "random_state", "torch_rng_state"),
    }
    missing = [name for name, names in aliases.items() if not any(key in snapshot for key in names)]
    if missing:
        raise S45ProvenanceError(f"{label} lacks full-state components: {', '.join(missing)}")
    return snapshot


def _make_branch_record(
    family: Mapping[str, Any],
    anchor: Mapping[str, Any],
    prefix: Mapping[str, Any],
    execution: Mapping[str, Any],
    *,
    split_step: int,
    branch_seed: int,
    branch_index: int,
    mode: str,
    protocol: ProtocolAuthority,
) -> dict[str, Any]:
    anchor_actions = _row_actions(anchor)
    anchor_trajectory = _row_trajectory(anchor)
    snapshot = _require_full_snapshot(prefix.get("snapshot"), label="S4 prefix snapshot")
    prefix_actions = _jsonable(prefix.get("actions", prefix.get("prefix_actions")))
    prefix_trajectory = _jsonable(prefix.get("trajectory", prefix.get("prefix_trajectory")))
    if prefix_actions is None or prefix_trajectory is None:
        raise S45ProvenanceError("replay_prefix must return actions and trajectory through split_step")
    if len(prefix_actions) != int(split_step) or len(prefix_trajectory) != int(split_step):
        raise S45ProvenanceError("prefix snapshot length does not equal split_step")
    branch_actions = _jsonable(execution["actions"])
    branch_trajectory = _jsonable(execution["trajectory"])
    if not isinstance(branch_actions, Sequence) or len(branch_actions) < int(split_step):
        raise S45ProvenanceError("branch execution lacks the full action trajectory")
    if not isinstance(branch_trajectory, Sequence) or len(branch_trajectory) < int(split_step):
        raise S45ProvenanceError("branch execution lacks the full state trajectory")
    prefix_preserving = canonical_json(branch_actions[:split_step]) == canonical_json(prefix_actions) and canonical_json(branch_trajectory[:split_step]) == canonical_json(prefix_trajectory)
    horizon = _trajectory_length(anchor)
    if not (0 < int(split_step) < int(horizon) - 1):
        raise S45ProvenanceError("S4 branch location is not interior to the anchor episode")
    result = {
        "family_id": str(family["family_id"]),
        "task_id": family.get("task_id", family.get("metadata", {}).get("task_id")),
        "init_state_id": family.get("init_state_id", family.get("metadata", {}).get("init_state", family.get("metadata", {}).get("initial_state_id"))),
        "anchor_candidate_id": str(anchor.get("candidate_id")),
        "anchor_candidate_seed": int(anchor.get("candidate_seed", anchor.get("seed"))),
        "anchor_success": _candidate_success(anchor),
        "split_step": int(split_step),
        "episode_length": int(horizon),
        "branch_id": f"{mode}-{int(branch_index):04d}",
        "branch_index": int(branch_index),
        "mode": str(mode),
        "branch_seed": int(branch_seed),
        "branch_seed_formula": protocol.s4["branch_seed_formula"],
        "prefix_actions": prefix_actions,
        "prefix_trajectory": prefix_trajectory,
        "prefix_digest": _canonical_digest({"actions": prefix_actions, "trajectory": prefix_trajectory}),
        # Keep generic names as well as the explicit branch_* names.  The
        # former lets the finalizer re-run the same raw-execution contract
        # without converting a persisted probe into a summary boolean.
        "actions": branch_actions,
        "trajectory": branch_trajectory,
        "branch_actions": branch_actions,
        "branch_trajectory": branch_trajectory,
        "prefix_preserving": bool(prefix_preserving),
        "success": bool(execution["success"]),
        "terminated": True,
        "termination": execution.get("termination", execution.get("termination_reason", "official")),
        "policy_forwards": int(execution["policy_forwards"]),
        "env_steps": int(execution["env_steps"]),
        "snapshot_restore_check": _jsonable(execution["snapshot_restore_check"]),
        "snapshot": _jsonable(snapshot),
        "source_n32_directory": str(family.get("directory", family.get("metadata", {}).get("source_directory", ""))),
        "source_n32_marker_sha256": str(family.get("source_marker_sha256", "")),
        "source_n32_bundle_sha256": str(family.get("source_bundle_sha256", "")),
        **protocol.identity(),
    }
    if not result["prefix_preserving"]:
        # A non-preserving branch is still retained as raw negative evidence,
        # but it cannot count toward S4.  The field is computed here, never
        # trusted from the adapter.
        result["prefix_preserving_reason"] = "branch prefix differs from anchor replay"
    return result


def _write_jsonl(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return ("".join(json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows)).encode("utf-8")


def run_s4(
    families: Sequence[N32Family | Mapping[str, Any]],
    protocol: ProtocolAuthority,
    adapter: S45Adapter,
    output_root: str | Path,
) -> dict[str, Any]:
    """Run S4 on every near-all-fail N32 family.

    Oracle and random use the exact same frozen K and the same family/anchor;
    only their pre-registered control-step grids differ.  No branch is run if
    a required protocol field or real adapter hook is missing.
    """

    family_list = [_family_from_mapping(value) for value in families]
    if not family_list:
        raise S45BundleError("S4 requires at least one accepted N32 family")
    near = [family for family in family_list if sum(_candidate_success(row) for row in family["candidates"]) <= NEAR_ALL_FAIL_MAX_SUCCESS]
    if not near:
        raise S45BundleError("S4 requires the complete near-all-fail set; none were found")
    root = Path(output_root)
    completed = 0
    try:
        for family in sorted(near, key=lambda item: str(item["family_id"])):
            family_id = str(family["family_id"])
            directory = root / family_id
            marker_name = "COMPLETED_S4_FAMILY.json"
            if (directory / marker_name).is_file():
                verify_atomic_bundle(directory, marker_name=marker_name)
                completed += 1
                continue
            anchor = _coerce_mapping(_call(adapter.select_anchor, family=family, protocol=protocol), label="S4 anchor")
            anchor_id = str(anchor.get("candidate_id"))
            base_ids = {str(row.get("candidate_id")) for row in family["candidates"]}
            if anchor_id not in base_ids:
                raise S45ProvenanceError(f"S4 anchor {anchor_id} is not an N32 candidate in {family_id}")
            horizon = _trajectory_length(anchor)
            oracle_steps = protocol.oracle_steps(family_id, horizon)
            random_steps = protocol.random_steps(family_id, horizon)
            if len(oracle_steps) != protocol.branch_count or len(random_steps) != protocol.branch_count:
                raise S45ProtocolError("S4 oracle/random branch counts differ")
            oracle_records: list[dict[str, Any]] = []
            random_records: list[dict[str, Any]] = []
            for mode, steps, records in (("oracle", oracle_steps, oracle_records), ("random", random_steps, random_records)):
                for branch_index, split_step in enumerate(steps):
                    prefix = _coerce_mapping(_call(adapter.replay_prefix, family=family, anchor=anchor, split_step=int(split_step), protocol=protocol), label="S4 replay_prefix")
                    seed_value = _call(adapter.branch_seed, family=family, anchor=anchor, split_step=int(split_step), branch_index=int(branch_index), mode=mode, protocol=protocol)
                    try:
                        branch_seed = int(seed_value)
                    except (TypeError, ValueError) as exc:
                        raise S45ProvenanceError("adapter returned a non-integer S4 branch seed") from exc
                    execution = _require_branch_execution(
                        _call(adapter.run_branch, family=family, anchor=anchor, prefix=prefix, split_step=int(split_step), branch_seed=branch_seed, branch_index=int(branch_index), mode=mode, protocol=protocol),
                        label=f"{family_id}/{mode}/{branch_index}",
                    )
                    records.append(_make_branch_record(family, anchor, prefix, execution, split_step=int(split_step), branch_seed=branch_seed, branch_index=int(branch_index), mode=mode, protocol=protocol))
            probe = {
                "family_id": family_id,
                "anchor_candidate_id": anchor_id,
                "oracle_branches": oracle_records,
                "random_branches": random_records,
                "oracle_branch_count": protocol.branch_count,
                "random_branch_count": protocol.branch_count,
                **protocol.identity(),
            }
            artifacts = {
                "S4_BRANCHES.jsonl": _write_jsonl([*oracle_records, *random_records]),
                "S4_PROBE.json": (json.dumps(_jsonable(probe), ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"),
            }
            marker = {
                "schema": "r142-stage-s-s4-family-v1",
                "marker_type": "completed_s4_family",
                "family_id": family_id,
                "candidate_count": BASE_CANDIDATE_COUNT,
                "branch_count_per_mode": protocol.branch_count,
                "near_all_fail": True,
                **protocol.identity(),
                "source_n32_marker_sha256": family.get("source_marker_sha256"),
            }
            write_atomic_bundle(directory, artifacts, marker_name=marker_name, marker_payload=marker)
            completed += 1
    finally:
        adapter.close()
    return {
        "schema": "r142-stage-s-s4-run-v1",
        "protocol_id": protocol.protocol_id,
        "protocol_authority_sha256": protocol.sha256,
        "near_all_fail_family_count": len(near),
        "completed_family_count": completed,
        "output_root": str(root),
    }


def _fresh_record(
    family: Mapping[str, Any],
    execution: Mapping[str, Any],
    *,
    candidate_index: int,
    candidate_seed: int,
    protocol: ProtocolAuthority,
) -> dict[str, Any]:
    required = ("actions", "trajectory", "terminated", "success", "policy_forwards", "env_steps", "snapshot_restore_check")
    for field in required:
        if field not in execution:
            raise S45ProvenanceError(f"S5 candidate {candidate_index} lacks {field}")
    if not bool(execution["terminated"]):
        raise S45ProvenanceError(f"S5 candidate {candidate_index} did not reach official termination")
    replay = _coerce_mapping(execution["snapshot_restore_check"], label="S5 snapshot_restore_check")
    if replay.get("same_action") is not True or not bool(replay.get("passed")) or float(replay.get("max_abs_error", float("inf"))) > SNAPSHOT_REPLAY_TOLERANCE:
        raise S45ProvenanceError(f"S5 candidate {candidate_index} lacks passing full-state replay evidence")
    return {
        "candidate_index": int(candidate_index),
        "candidate_id": f"{family['family_id']}/candidate-{int(candidate_index):04d}",
        "parent_id": None,
        "generation_step": 0,
        "candidate_seed": int(candidate_seed),
        "seed_formula": protocol.s5["extension_seed_formula"],
        "actions": _jsonable(execution["actions"]),
        "trajectory": _jsonable(execution["trajectory"]),
        "success": bool(execution["success"]),
        "terminated": True,
        "termination": execution.get("termination", execution.get("termination_reason", "official")),
        "policy_forwards": int(execution["policy_forwards"]),
        "env_steps": int(execution["env_steps"]),
        "snapshot_restore_check": _jsonable(execution["snapshot_restore_check"]),
        "genealogy": _jsonable(execution.get("genealogy", {"root_family_id": family["family_id"], "candidate_index": int(candidate_index)})),
        "source_n32_directory": str(family.get("directory", "")),
        "source_n32_marker_sha256": str(family.get("source_marker_sha256", "")),
        **protocol.identity(),
    }


def _base_row_for_s5(family: Mapping[str, Any], row: Mapping[str, Any], index: int) -> dict[str, Any]:
    candidate = _normalise_candidate(row, index)
    if int(candidate.get("candidate_index", index)) != int(index):
        raise S45ProvenanceError(f"S5 base candidate index is not immutable 0..31 in {family['family_id']}")
    candidate["candidate_index"] = int(index)
    candidate["parent_id"] = candidate.get("parent_id")
    candidate["generation_step"] = int(candidate.get("generation_step", 0))
    candidate["actions"] = _row_actions(candidate)
    candidate["trajectory"] = _row_trajectory(candidate)
    candidate["success"] = _candidate_success(candidate)
    if candidate.get("termination") is None:
        raise S45ProvenanceError(f"S5 base candidate {index} lacks terminal evidence")
    return candidate


def run_s5(
    families: Sequence[N32Family | Mapping[str, Any]],
    protocol: ProtocolAuthority,
    adapter: S45Adapter,
    output_root: str | Path,
) -> dict[str, Any]:
    """Run fresh candidates 32..63 for every accepted N32 family."""

    family_list = [_family_from_mapping(value) for value in families]
    if not family_list:
        raise S45BundleError("S5 requires accepted N32 families")
    root = Path(output_root)
    completed = 0
    try:
        for family in sorted(family_list, key=lambda item: str(item["family_id"])):
            family_id = str(family["family_id"])
            directory = root / family_id
            marker_name = "COMPLETED_S5_FAMILY.json"
            if (directory / marker_name).is_file():
                verify_atomic_bundle(directory, marker_name=marker_name)
                completed += 1
                continue
            base_rows = [_base_row_for_s5(family, row, index) for index, row in enumerate(family["candidates"])]
            if len(base_rows) != BASE_CANDIDATE_COUNT:
                raise S45BundleError(f"S5 base family {family_id} is not exactly 32 candidates")
            base_digest = _canonical_digest(base_rows)
            fresh_rows: list[dict[str, Any]] = []
            seeds: set[int] = {int(row["candidate_seed"]) for row in base_rows}
            for candidate_index in FRESH_CANDIDATE_INDICES:
                seed_value = _call(adapter.extension_seed, family=family, candidate_index=int(candidate_index), protocol=protocol)
                try:
                    candidate_seed = int(seed_value)
                except (TypeError, ValueError) as exc:
                    raise S45ProvenanceError(f"S5 extension seed is not an integer for {family_id}/{candidate_index}") from exc
                if candidate_seed in seeds:
                    raise S45ProvenanceError(f"S5 fresh seed collides with base/another fresh seed: {family_id}/{candidate_index}")
                seeds.add(candidate_seed)
                execution = _coerce_mapping(_call(adapter.run_fresh_candidate, family=family, candidate_index=int(candidate_index), candidate_seed=candidate_seed, protocol=protocol), label="S5 fresh candidate execution")
                fresh_rows.append(_fresh_record(family, execution, candidate_index=int(candidate_index), candidate_seed=candidate_seed, protocol=protocol))
            extended = [*base_rows, *fresh_rows]
            if [int(row["candidate_index"]) for row in extended] != list(range(EXTENDED_CANDIDATE_COUNT)):
                raise S45ProvenanceError(f"S5 family {family_id} does not contain immutable base32+fresh32 indices")
            payload = {
                "schema": "r142-stage-s-s5-family-v1",
                "family_id": family_id,
                "base_candidate_count": BASE_CANDIDATE_COUNT,
                "extended_candidate_count": EXTENDED_CANDIDATE_COUNT,
                "fresh_candidate_indices": list(FRESH_CANDIDATE_INDICES),
                "base_digest": base_digest,
                "base_rows": base_rows,
                "fresh_rows": fresh_rows,
                "extended_rows": extended,
                **protocol.identity(),
                "source_n32_directory": str(family.get("directory", "")),
                "source_n32_marker_sha256": str(family.get("source_marker_sha256", "")),
                "source_n32_bundle_sha256": str(family.get("source_bundle_sha256", "")),
            }
            artifacts = {
                "S5_FAMILY.json": (json.dumps(_jsonable(payload), ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"),
                "S5_GENEALOGY.jsonl": _write_jsonl(extended),
            }
            marker = {
                "schema": "r142-stage-s-s5-family-v1",
                "marker_type": "completed_s5_family",
                "family_id": family_id,
                "base_candidate_count": BASE_CANDIDATE_COUNT,
                "extended_candidate_count": EXTENDED_CANDIDATE_COUNT,
                "fresh_candidate_indices": list(FRESH_CANDIDATE_INDICES),
                "base_digest": base_digest,
                **protocol.identity(),
                "source_n32_marker_sha256": family.get("source_marker_sha256"),
            }
            write_atomic_bundle(directory, artifacts, marker_name=marker_name, marker_payload=marker)
            completed += 1
    finally:
        adapter.close()
    return {
        "schema": "r142-stage-s-s5-run-v1",
        "protocol_id": protocol.protocol_id,
        "protocol_authority_sha256": protocol.sha256,
        "family_count": len(family_list),
        "completed_family_count": completed,
        "output_root": str(root),
    }


def load_s4_probes(root: str | Path, *, protocol: ProtocolAuthority, expected_family_ids: Sequence[str]) -> list[Mapping[str, Any]]:
    base = Path(root)
    probes: list[Mapping[str, Any]] = []
    expected = {str(value) for value in expected_family_ids}
    found: set[str] = set()
    for family_id in sorted(expected):
        directory = base / family_id
        marker = verify_atomic_bundle(directory, marker_name="COMPLETED_S4_FAMILY.json")
        if marker.get("protocol_authority_sha256") != protocol.sha256 or marker.get("protocol_git_commit") != protocol.git_commit:
            raise S45ProvenanceError(f"S4 family {family_id} has mismatched protocol authority")
        probe = _read_json(directory / "S4_PROBE.json", label="S4 probe")
        if str(probe.get("family_id")) != family_id or probe.get("protocol_authority_sha256") != protocol.sha256:
            raise S45ProvenanceError(f"S4 probe provenance mismatch: {family_id}")
        for mode in ("oracle_branches", "random_branches"):
            branches = probe.get(mode)
            if not isinstance(branches, list) or len(branches) != protocol.branch_count:
                raise S45ProvenanceError(f"S4 {family_id} lacks exactly K {mode}")
            for branch in branches:
                _require_branch_execution(branch, label=f"S4 persisted {family_id}/{mode}")
                if branch.get("family_id") != family_id or branch.get("protocol_authority_sha256") != protocol.sha256:
                    raise S45ProvenanceError(f"S4 branch provenance mismatch: {family_id}")
        probes.append(probe)
        found.add(family_id)
    if found != expected:
        raise S45BundleError("S4 output does not cover exactly all near-all-fail families")
    return probes


def load_s5_extended(root: str | Path, *, protocol: ProtocolAuthority, families: Sequence[N32Family]) -> tuple[dict[str, list[Mapping[str, Any]]], dict[str, list[Mapping[str, Any]]]]:
    base = Path(root)
    all_base: dict[str, list[Mapping[str, Any]]] = {}
    all_extended: dict[str, list[Mapping[str, Any]]] = {}
    for family in families:
        family_id = family.family_id
        directory = base / family_id
        marker = verify_atomic_bundle(directory, marker_name="COMPLETED_S5_FAMILY.json")
        if marker.get("protocol_authority_sha256") != protocol.sha256 or marker.get("protocol_git_commit") != protocol.git_commit:
            raise S45ProvenanceError(f"S5 family {family_id} has mismatched protocol authority")
        payload = _read_json(directory / "S5_FAMILY.json", label="S5 family")
        if str(payload.get("family_id")) != family_id or payload.get("protocol_authority_sha256") != protocol.sha256:
            raise S45ProvenanceError(f"S5 payload provenance mismatch: {family_id}")
        base_rows = payload.get("base_rows")
        fresh_rows = payload.get("fresh_rows")
        extended_rows = payload.get("extended_rows")
        if not isinstance(base_rows, list) or not isinstance(fresh_rows, list) or not isinstance(extended_rows, list):
            raise S45BundleError(f"S5 family {family_id} lacks base/fresh/extended rows")
        if len(base_rows) != BASE_CANDIDATE_COUNT or len(fresh_rows) != BASE_CANDIDATE_COUNT or len(extended_rows) != EXTENDED_CANDIDATE_COUNT:
            raise S45BundleError(f"S5 family {family_id} has incorrect row counts")
        expected_base = [_base_row_for_s5(family.as_mapping(), row, index) for index, row in enumerate(family.candidates)]
        if _canonical_digest(expected_base) != str(payload.get("base_digest")) or _canonical_digest(expected_base) != _canonical_digest(base_rows):
            raise S45ProvenanceError(f"S5 family {family_id} rewrote immutable base32")
        if [int(row.get("candidate_index", -1)) for row in fresh_rows] != list(FRESH_CANDIDATE_INDICES):
            raise S45ProvenanceError(f"S5 family {family_id} fresh indices are not 32..63")
        if [int(row.get("candidate_index", -1)) for row in extended_rows] != list(range(EXTENDED_CANDIDATE_COUNT)):
            raise S45ProvenanceError(f"S5 family {family_id} extended indices are not 0..63")
        for index, row in enumerate(extended_rows):
            _normalise_candidate(row, index, success=row.get("success"), seed=row.get("candidate_seed"))
            # Base rows are preserved from the accepted N32 bundle and may
            # predate the row-level identity fields.  Fresh rows are created
            # by this runtime and must carry the current authority directly.
            if index >= BASE_CANDIDATE_COUNT and row.get("protocol_authority_sha256") != protocol.sha256:
                raise S45ProvenanceError(f"S5 row protocol mismatch: {family_id}/{index}")
        all_base[family_id] = list(base_rows)
        all_extended[family_id] = list(extended_rows)
    return all_base, all_extended


def finalise_s45(
    n32_root: str | Path,
    s4_root: str | Path,
    s5_root: str | Path,
    protocol_path: str | Path,
    output_root: str | Path,
    *,
    expected_substrate: str | None = None,
) -> dict[str, Any]:
    """Verify all persisted inputs and compute S4/S5 gates.

    This is the only function in this module that calls the statistical
    analysis.  It first verifies provenance and all family coverage, so the
    pure analysis functions cannot be fed a manually typed recovery boolean or
    a convenient subset of the near-all-fail denominator.
    """

    protocol = ProtocolAuthority.load(protocol_path)
    families = discover_n32_families(n32_root, protocol=protocol)
    if expected_substrate is not None:
        for family in families:
            observed = family.metadata.get("substrate", family.marker.get("substrate"))
            if observed != expected_substrate:
                raise S45ProvenanceError(f"N32 family {family.family_id} substrate mismatch")
    near = [family for family in families if sum(_candidate_success(row) for row in family.candidates) <= NEAR_ALL_FAIL_MAX_SUCCESS]
    probes = load_s4_probes(s4_root, protocol=protocol, expected_family_ids=[family.family_id for family in near])
    base_data, extended_data = load_s5_extended(s5_root, protocol=protocol, families=list(families))
    rollouts = {family.family_id: list(family.candidates) for family in families}
    s4 = compute_s4(probes, replicates=S4_BOOTSTRAP_REPLICATES)
    s5 = compute_s5(base_data, extended_data)
    result: dict[str, Any] = {
        "schema": "r142-stage-s-s45-result-v1",
        "protocol": protocol.identity(),
        "substrate": expected_substrate,
        "n32_family_count": len(families),
        "near_all_fail_family_count": len(near),
        "S4": s4,
        "S5": s5,
        "s4": s4,
        "s5": s5,
        "all_inputs_complete": True,
        "pass": bool(s4.get("pass") and s5.get("pass")),
    }
    output = Path(output_root)
    artifacts = {"S45_RESULT.json": (json.dumps(_jsonable(result), ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")}
    marker = {"schema": "r142-stage-s-s45-completion-v1", "marker_type": "completed_s45_evaluation", **protocol.identity(), "substrate": expected_substrate, "n32_family_count": len(families), "near_all_fail_family_count": len(near)}
    completion = write_atomic_bundle(output, artifacts, marker_name="COMPLETED_EVALUATION_RESULT.json", marker_payload=marker)
    return {**result, "completion": completion}


def load_adapter(spec: str, *, protocol: ProtocolAuthority, substrate: str | None = None) -> S45Adapter:
    """Load a real adapter from ``module:factory`` for the CLI."""

    if ":" not in spec:
        raise S45CapabilityError("adapter must be specified as module:factory")
    module_name, factory_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name, None)
    if not callable(factory):
        raise S45CapabilityError(f"adapter factory is not callable: {spec}")
    adapter = _call(factory, protocol=protocol, substrate=substrate)
    if not isinstance(adapter, S45Adapter):
        raise S45CapabilityError("adapter factory must return an S45Adapter instance")
    return adapter


__all__ = [
    "BASE_CANDIDATE_COUNT",
    "EXTENDED_CANDIDATE_COUNT",
    "FRESH_CANDIDATE_INDICES",
    "N32Family",
    "ProtocolAuthority",
    "S45Adapter",
    "S45BundleError",
    "S45CapabilityError",
    "S45Error",
    "S45ProtocolError",
    "S45ProvenanceError",
    "SNAPSHOT_REPLAY_TOLERANCE",
    "canonical_json",
    "discover_n32_families",
    "finalise_s45",
    "load_adapter",
    "load_s4_probes",
    "load_s5_extended",
    "run_s4",
    "run_s5",
    "sha256_file",
    "write_atomic_bundle",
    "verify_atomic_bundle",
]
