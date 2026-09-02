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
        # S4 is a two-stage procedure.  The nine search locations, four
        # search branches per location, and eight paired held-out suffixes
        # are protocol-owned constants.  They must never be inferred from a
        # source-tree default or from observed outcomes.
        required_s4 = ("anchor_rule", "oracle_t_rule", "random_t_rule", "branch_count", "branch_seed_formula")
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
        search_grid = s4.get("search_t_grid", s4.get("oracle_search_grid", s4.get("oracle_t_grid")))
        search_branch_count = s4.get(
            "search_branch_count",
            s4.get("search_branches_per_location", s4.get("branch_count")),
        )
        heldout_branch_count = s4.get(
            "heldout_branch_count",
            s4.get("oracle_heldout_branch_count", s4.get("heldout_branches")),
        )
        random_branch_count = s4.get(
            "random_branch_count",
            s4.get("random_heldout_branch_count", heldout_branch_count),
        )
        random_location_formula = s4.get(
            "random_location_hash_formula",
            s4.get("random_t_hash_formula", s4.get("random_location_formula", s4.get("random_t_rule"))),
        )
        if search_grid is None:
            raise S45ProtocolError("protocol authority s4 lacks search_t_grid/oracle_t_grid")
        if search_branch_count is None:
            raise S45ProtocolError("protocol authority s4 lacks search_branch_count")
        if heldout_branch_count is None:
            raise S45ProtocolError("protocol authority s4 lacks heldout_branch_count")
        if random_location_formula is None or not isinstance(random_location_formula, str) or not random_location_formula.strip():
            raise S45ProtocolError("protocol authority s4 lacks random_location_hash_formula")
        try:
            branch_count = int(s4["branch_count"])
        except (TypeError, ValueError) as exc:
            raise S45ProtocolError("s4.branch_count must be a positive integer") from exc
        if branch_count <= 0:
            raise S45ProtocolError("s4.branch_count must be positive")
        try:
            search_branch_count = int(search_branch_count)
            heldout_branch_count = int(heldout_branch_count)
            random_branch_count = int(random_branch_count)
        except (TypeError, ValueError) as exc:
            raise S45ProtocolError("s4 search/heldout branch counts must be integers") from exc
        if search_branch_count != 4:
            raise S45ProtocolError("s4.search_branch_count is frozen at 4")
        if heldout_branch_count != 8 or random_branch_count != 8:
            raise S45ProtocolError("s4 held-out oracle/random branch counts are frozen at 8")
        if branch_count != search_branch_count:
            raise S45ProtocolError("legacy s4.branch_count must equal search_branch_count=4")
        search_grid_values = search_grid.values() if isinstance(search_grid, Mapping) else (search_grid,)
        if isinstance(search_grid, Mapping) and not search_grid:
            raise S45ProtocolError("s4 search grid mapping cannot be empty")
        for grid in search_grid_values:
            if not isinstance(grid, (list, tuple)) or len(grid) != 9:
                raise S45ProtocolError("s4 search grid must contain exactly 9 interior control steps")
            if not all(isinstance(item, (int, np.integer)) and not isinstance(item, bool) for item in grid):
                raise S45ProtocolError("s4 search grid must contain integer control steps")
        random_grid = s4.get("random_t_grid")
        random_grid_values = random_grid.values() if isinstance(random_grid, Mapping) else (random_grid,)
        for grid in random_grid_values:
            if isinstance(grid, (list, tuple)) and grid:
                if len(grid) != random_branch_count:
                    raise S45ProtocolError("s4.random_t_grid must contain exactly 8 paired locations")
                if not all(isinstance(item, (int, np.integer)) and not isinstance(item, bool) for item in grid):
                    raise S45ProtocolError("s4.random_t_grid must contain integer control steps")
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
        return self.search_steps(family_id, episode_length)

    def search_steps(self, family_id: str, episode_length: int) -> tuple[int, ...]:
        raw = self.s4.get("search_t_grid", self.s4.get("oracle_search_grid", self.s4.get("oracle_t_grid")))
        if isinstance(raw, Mapping):
            if family_id not in raw and str(family_id) not in raw:
                raise S45ProtocolError(f"s4 search grid has no entry for family {family_id}")
            raw = raw.get(family_id, raw.get(str(family_id)))
        if not isinstance(raw, (list, tuple)) or len(raw) != 9:
            raise S45ProtocolError("s4 search grid must contain exactly 9 entries")
        values = tuple(int(item) for item in raw)
        if any(not (0 < item < int(episode_length) - 1) for item in values):
            raise S45ProtocolError(f"s4 search grid contains a non-interior control step for family {family_id}")
        if len(set(values)) != len(values):
            raise S45ProtocolError(f"s4 search grid contains duplicate locations for family {family_id}")
        return values

    def random_steps(self, family_id: str, episode_length: int) -> tuple[int, ...]:
        """Return eight frozen hash-selected random interior locations."""

        count = self.heldout_branch_count
        raw = self.s4.get("random_t_grid")
        if isinstance(raw, Mapping):
            if family_id not in raw and str(family_id) not in raw:
                raw = None
            else:
                raw = raw.get(family_id, raw.get(str(family_id)))
        formula = self.s4.get(
            "random_location_hash_formula",
            self.s4.get("random_t_hash_formula", self.s4.get("random_location_formula", self.s4.get("random_t_rule"))),
        )
        if not isinstance(formula, str) or not formula.strip():
            raise S45ProtocolError("s4 random locations require a frozen hash formula or explicit grid")
        match = re.fullmatch(
            r"sha256\(([^)]*)\)\s*[-=]>?\s*(?:first_8_bytes_(big|little)_endian_)?mod_interior",
            formula.strip(), flags=re.IGNORECASE,
        )
        hashed: tuple[int, ...] | None = None
        if match is not None:
            tokens = [token.strip() for token in match.group(1).split("|")]
            derived: list[int] = []
            interior = int(episode_length) - 2
            if interior <= 0:
                raise S45ProtocolError("episode has no interior control step")
            for pair_index in range(count):
                context = {
                    "protocol_id": self.protocol_id,
                    "family_id": str(family_id),
                    "episode_length": int(episode_length),
                    "pair_index": int(pair_index),
                    "index": int(pair_index),
                }
                material: list[str] = []
                for token in tokens:
                    if token in context:
                        material.append(str(context[token]))
                    elif token.startswith(("'", '"')) and token[-1:] == token[0]:
                        material.append(token[1:-1])
                    elif token in {"s4", "random", "random_location", "pair"}:
                        material.append(token)
                    else:
                        raise S45ProtocolError(f"unknown token {token!r} in random location hash formula")
                digest = hashlib.sha256("|".join(material).encode("utf-8")).digest()
                endian = "little" if (match.group(1) or "big").lower().startswith("little") else "big"
                derived.append(int.from_bytes(digest[:8], endian, signed=False) % interior + 1)
            hashed = tuple(derived)
        if isinstance(raw, (list, tuple)) and raw:
            if len(raw) != count:
                raise S45ProtocolError("s4.random_t_grid must contain exactly heldout_branch_count entries")
            explicit = tuple(int(item) for item in raw)
            if any(not (0 < item < int(episode_length) - 1) for item in explicit):
                raise S45ProtocolError(f"s4.random_t_grid contains a non-interior control step for family {family_id}")
            if hashed is not None and explicit != hashed:
                raise S45ProtocolError(f"s4.random_t_grid disagrees with the frozen hash formula for family {family_id}")
            return hashed if hashed is not None else explicit
        if hashed is None:
            raise S45ProtocolError("unsupported s4 random location hash formula")
        return hashed

    @property
    def branch_count(self) -> int:
        return self.search_branch_count

    @property
    def search_branch_count(self) -> int:
        return int(self.s4.get("search_branch_count", self.s4.get("search_branches_per_location", self.s4["branch_count"])))

    @property
    def heldout_branch_count(self) -> int:
        value = self.s4.get("heldout_branch_count", self.s4.get("oracle_heldout_branch_count", self.s4.get("heldout_branches")))
        if value is None:
            raise S45ProtocolError("s4 heldout_branch_count is missing")
        return int(value)


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


def _decode_numpy_wrapper(value: Any, *, label: str) -> Any:
    """Decode the canonical JSON wrapper used for persisted NumPy arrays.

    The A producer serializes workspace poses as a wrapper rather than a
    Python list.  Decode only for validation/adapter consumption; the source
    row itself remains untouched so S5 can bind its original file digest.
    """

    if not isinstance(value, Mapping):
        return value
    if "__ndarray__" in value:
        data = value["__ndarray__"]
    elif "data" in value and "shape" in value:
        data = value["data"]
    else:
        return value
    try:
        array = np.asarray(data)
        shape = tuple(int(item) for item in value.get("shape", array.shape))
    except (TypeError, ValueError) as exc:
        raise S45BundleError(f"{label} has an invalid NumPy wrapper") from exc
    if array.shape != shape:
        raise S45BundleError(f"{label} NumPy wrapper shape does not match data")
    if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
        raise S45BundleError(f"{label} NumPy wrapper contains non-finite/non-numeric data")
    return array.tolist()


def _validate_pose_trajectory(value: Any, *, label: str, dimension: int) -> None:
    decoded = _decode_numpy_wrapper(value, label=label)
    try:
        array = np.asarray(decoded, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise S45BundleError(f"{label} is not a numeric pose trajectory") from exc
    if array.ndim != 2 or array.shape[1] != int(dimension) or array.shape[0] <= 0:
        raise S45BundleError(
            f"{label} must have shape [T,{int(dimension)}] with T>0; got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise S45BundleError(f"{label} contains non-finite pose values")


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
    termination = metadata.get("termination", metadata.get("termination_reason", marker.get("termination")))
    substrate = metadata.get("substrate", marker.get("substrate"))
    if substrate not in {"A", "B", "C"}:
        raise S45BundleError(f"family {directory.name} lacks a valid substrate identity A/B/C")
    pose_dimension = 14 if substrate == "A" else 6
    ids: set[str] = set()
    seeds: set[int] = set()
    for index, candidate in enumerate(candidates):
        if int(candidate.get("candidate_index", -1)) != index:
            raise S45BundleError(f"candidate indices are not exactly 0..31 in {directory}")
        candidate_id = str(candidate.get("candidate_id", ""))
        if not candidate_id or candidate_id in ids:
            raise S45BundleError(f"candidate ids are missing or duplicated in {directory}")
        ids.add(candidate_id)
        candidate_seed = int(candidate.get("candidate_seed"))
        if candidate_seed in seeds:
            raise S45BundleError(f"candidate seeds are duplicated in {directory}")
        seeds.add(candidate_seed)
        if candidate.get("terminated") is not True:
            raise S45BundleError(f"candidate {candidate_id} lacks terminal termination evidence")
        if candidate.get("termination") is None and candidate.get("termination_reason") is not None:
            candidate["termination"] = candidate["termination_reason"]
        if candidate.get("termination") is None and termination is not None:
            candidate["termination"] = termination
        if candidate.get("termination") is None:
            raise S45BundleError(f"candidate {candidate['candidate_id']} lacks official termination evidence")
        if not isinstance(candidate.get("termination"), (str, Mapping)) or not candidate.get("termination"):
            raise S45BundleError(f"candidate {candidate_id} has invalid official termination evidence")
        _validate_pose_trajectory(candidate.get("trajectory"), label=f"candidate {candidate_id} trajectory", dimension=pose_dimension)
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
    source_family_file_sha256: str = ""

    def as_mapping(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "directory": str(self.directory),
            "marker": dict(self.marker),
            "metadata": dict(self.metadata),
            "candidates": [dict(candidate) for candidate in self.candidates],
            "source_marker_sha256": self.source_marker_sha256,
            "source_bundle_sha256": self.source_bundle_sha256,
            "source_family_file_sha256": self.source_family_file_sha256,
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
                source_family_file_sha256=sha256_file(
                    marker_path.parent / "family.json"
                    if (marker_path.parent / "family.json").is_file()
                    else marker_path.parent / "rollouts.npz"
                ),
            )
        )
    return tuple(sorted(result, key=lambda family: family.family_id))


def _family_from_mapping(family: N32Family | Mapping[str, Any]) -> dict[str, Any]:
    return family.as_mapping() if isinstance(family, N32Family) else dict(family)


def _row_actions(row: Mapping[str, Any]) -> list[Any]:
    value = row.get("actions", row.get("action_prefix"))
    if value is None:
        raise S45ProvenanceError(f"candidate {row.get('candidate_id')} lacks actions")
    value = _decode_numpy_wrapper(value, label=f"candidate {row.get('candidate_id')} actions")
    if isinstance(value, Mapping) or not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise S45ProvenanceError(f"candidate {row.get('candidate_id')} actions are not a sequence")
    return _jsonable(value)


def _row_trajectory(row: Mapping[str, Any]) -> list[Any]:
    value = row.get("trajectory", row.get("poses", row.get("pose_trajectory")))
    if value is None:
        raise S45ProvenanceError(f"candidate {row.get('candidate_id')} lacks trajectory")
    value = _decode_numpy_wrapper(value, label=f"candidate {row.get('candidate_id')} trajectory")
    if isinstance(value, Mapping) or not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise S45ProvenanceError(f"candidate {row.get('candidate_id')} trajectory is not a sequence")
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
    for field in ("actions", "trajectory", "terminated", "success", "termination", "policy_forwards", "env_steps", "snapshot_restore_check"):
        if field not in result:
            raise S45ProvenanceError(f"{label} lacks raw execution field {field}")
    if not bool(result["terminated"]):
        raise S45ProvenanceError(f"{label} did not reach official termination")
    replay = _coerce_mapping(result["snapshot_restore_check"], label=f"{label}.snapshot_restore_check")
    if replay.get("same_action") is not True or not bool(replay.get("passed")):
        raise S45ProvenanceError(f"{label} lacks a passing restore->same-action replay check")
    if not isinstance(result["termination"], (str, Mapping)) or not result["termination"]:
        raise S45ProvenanceError(f"{label} lacks official termination evidence")
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
    for name, names in aliases.items():
        key = next(key for key in names if key in snapshot)
        if snapshot[key] is None:
            raise S45ProvenanceError(f"{label} has a null {name} component")
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
    prefix_actions = _row_actions({"actions": prefix.get("actions", prefix.get("prefix_actions"))})
    prefix_trajectory = _row_trajectory({"trajectory": prefix.get("trajectory", prefix.get("prefix_trajectory"))})
    if prefix_actions is None or prefix_trajectory is None:
        raise S45ProvenanceError("replay_prefix must return actions and trajectory through split_step")
    if len(prefix_actions) != int(split_step) or len(prefix_trajectory) != int(split_step):
        raise S45ProvenanceError("prefix snapshot length does not equal split_step")
    branch_actions = _row_actions({"actions": execution["actions"]})
    branch_trajectory = _row_trajectory({"trajectory": execution["trajectory"]})
    if not isinstance(branch_actions, Sequence) or len(branch_actions) < int(split_step):
        raise S45ProvenanceError("branch execution lacks the full action trajectory")
    if not isinstance(branch_trajectory, Sequence) or len(branch_trajectory) < int(split_step):
        raise S45ProvenanceError("branch execution lacks the full state trajectory")
    substrate = family.get("substrate", family.get("metadata", {}).get("substrate"))
    pose_dimension = 14 if substrate == "A" else 6 if substrate in {"B", "C"} else None
    if pose_dimension is None:
        raise S45ProvenanceError(f"family {family.get('family_id')} lacks substrate pose contract")
    _validate_pose_trajectory(prefix_trajectory, label="S4 prefix trajectory", dimension=pose_dimension)
    _validate_pose_trajectory(branch_trajectory, label="S4 branch trajectory", dimension=pose_dimension)
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
        "branch_id": f"{mode}-t{int(split_step):04d}-b{int(branch_index):04d}",
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
        "source_n32_family_file_sha256": str(family.get("source_family_file_sha256", "")),
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
    """Run the frozen two-stage S4 probe on every near-all-fail family.

    Stage one searches all nine interior locations with exactly four branches
    at each location.  The selected location is the maximum number of
    prefix-preserving successes, breaking ties by the earliest control step.
    Stage two is held out: eight oracle suffixes share the selected location,
    while eight paired suffixes use the locations selected by the frozen hash
    rule.  The same suffix seed is passed to both members of every pair.
    """

    family_list = [_family_from_mapping(value) for value in families]
    if not family_list:
        raise S45BundleError("S4 requires at least one accepted N32 family")
    near = [
        family
        for family in family_list
        if sum(_candidate_success(row) for row in family["candidates"]) <= NEAR_ALL_FAIL_MAX_SUCCESS
    ]
    if not near:
        raise S45BundleError("S4 requires the complete near-all-fail set; none were found")
    root = Path(output_root)
    completed = 0

    def seed_for(
        family: Mapping[str, Any],
        anchor: Mapping[str, Any],
        split_step: int,
        branch_index: int,
        mode: str,
    ) -> int:
        value = _call(
            adapter.branch_seed,
            family=family,
            anchor=anchor,
            split_step=int(split_step),
            branch_index=int(branch_index),
            mode=str(mode),
            protocol=protocol,
        )
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise S45ProvenanceError(
                f"adapter returned a non-integer S4 seed for {family['family_id']}/{mode}/{branch_index}"
            ) from exc

    try:
        for family in sorted(near, key=lambda item: str(item["family_id"])):
            family_id = str(family["family_id"])
            directory = root / family_id
            marker_name = "COMPLETED_S4_FAMILY.json"
            if (directory / marker_name).is_file():
                verify_atomic_bundle(directory, marker_name=marker_name)
                completed += 1
                continue
            anchor = _coerce_mapping(
                _call(adapter.select_anchor, family=family, protocol=protocol), label="S4 anchor"
            )
            anchor_id = str(anchor.get("candidate_id"))
            base_ids = {str(row.get("candidate_id")) for row in family["candidates"]}
            if anchor_id not in base_ids:
                raise S45ProvenanceError(f"S4 anchor {anchor_id} is not an N32 candidate in {family_id}")
            try:
                anchor_index = int(anchor.get("candidate_index", -1))
            except (TypeError, ValueError) as exc:
                raise S45ProvenanceError(f"S4 anchor index is invalid in {family_id}") from exc
            if not 0 <= anchor_index < BASE_CANDIDATE_COUNT:
                raise S45ProvenanceError(f"S4 anchor index is outside immutable 0..31 in {family_id}")
            source_anchor = family["candidates"][anchor_index]
            if str(source_anchor.get("candidate_id")) != anchor_id or int(source_anchor.get("candidate_seed")) != int(anchor.get("candidate_seed")):
                raise S45ProvenanceError(f"S4 anchor identity is not the immutable source row in {family_id}")
            if canonical_json(_row_actions(source_anchor)) != canonical_json(_row_actions(anchor)) or canonical_json(_row_trajectory(source_anchor)) != canonical_json(_row_trajectory(anchor)):
                raise S45ProvenanceError(f"S4 anchor raw trajectory was rewritten in {family_id}")
            horizon = _trajectory_length(anchor)
            search_steps = protocol.search_steps(family_id, horizon)
            random_steps = protocol.random_steps(family_id, horizon)
            if len(search_steps) != 9 or len(random_steps) != protocol.heldout_branch_count:
                raise S45ProtocolError("S4 protocol grid lengths are not frozen at 9 search/8 held-out")

            search_records: list[dict[str, Any]] = []
            for grid_index, split_step in enumerate(search_steps):
                branches: list[dict[str, Any]] = []
                for branch_index in range(protocol.search_branch_count):
                    prefix = _coerce_mapping(
                        _call(
                            adapter.replay_prefix,
                            family=family,
                            anchor=anchor,
                            split_step=int(split_step),
                            protocol=protocol,
                        ),
                        label="S4 search replay_prefix",
                    )
                    branch_seed = seed_for(
                        family, anchor, int(split_step), int(branch_index), "search"
                    )
                    execution = _require_branch_execution(
                        _call(
                            adapter.run_branch,
                            family=family,
                            anchor=anchor,
                            prefix=prefix,
                            split_step=int(split_step),
                            branch_seed=branch_seed,
                            branch_index=int(branch_index),
                            mode="search",
                            protocol=protocol,
                        ),
                        label=f"{family_id}/search/t{split_step}/b{branch_index}",
                    )
                    branches.append(
                        _make_branch_record(
                            family,
                            anchor,
                            prefix,
                            execution,
                            split_step=int(split_step),
                            branch_seed=branch_seed,
                            branch_index=int(branch_index),
                            mode="search",
                            protocol=protocol,
                        )
                    )
                success_count = int(
                    sum(bool(row["success"]) and bool(row["prefix_preserving"]) for row in branches)
                )
                search_records.append(
                    {
                        "grid_index": int(grid_index),
                        "split_step": int(split_step),
                        "success_count": success_count,
                        "branch_count": protocol.search_branch_count,
                        "branches": branches,
                    }
                )

            # The ordering is explicit rather than relying on a pre-sorted
            # grid: success count is primary and numerical t is the tie-break.
            chosen = max(search_records, key=lambda row: (int(row["success_count"]), -int(row["split_step"])))
            chosen_step = int(chosen["split_step"])

            oracle_records: list[dict[str, Any]] = []
            random_records: list[dict[str, Any]] = []
            paired_seeds: list[int] = []
            for pair_index in range(protocol.heldout_branch_count):
                pair_seed = seed_for(
                    family, anchor, chosen_step, int(pair_index), "heldout_pair"
                )
                if pair_seed in paired_seeds:
                    raise S45ProvenanceError(
                        f"S4 heldout suffix seed collision in {family_id}/{pair_index}"
                    )
                paired_seeds.append(pair_seed)
                oracle_prefix = _coerce_mapping(
                    _call(
                        adapter.replay_prefix,
                        family=family,
                        anchor=anchor,
                        split_step=chosen_step,
                        protocol=protocol,
                    ),
                    label="S4 heldout oracle replay_prefix",
                )
                oracle_execution = _require_branch_execution(
                    _call(
                        adapter.run_branch,
                        family=family,
                        anchor=anchor,
                        prefix=oracle_prefix,
                        split_step=chosen_step,
                        branch_seed=pair_seed,
                        branch_index=int(pair_index),
                        mode="oracle_heldout",
                        protocol=protocol,
                    ),
                    label=f"{family_id}/oracle_heldout/{pair_index}",
                )
                oracle_records.append(
                    _make_branch_record(
                        family,
                        anchor,
                        oracle_prefix,
                        oracle_execution,
                        split_step=chosen_step,
                        branch_seed=pair_seed,
                        branch_index=int(pair_index),
                        mode="oracle_heldout",
                        protocol=protocol,
                    )
                )

                random_step = int(random_steps[pair_index])
                random_prefix = _coerce_mapping(
                    _call(
                        adapter.replay_prefix,
                        family=family,
                        anchor=anchor,
                        split_step=random_step,
                        protocol=protocol,
                    ),
                    label="S4 heldout random replay_prefix",
                )
                random_execution = _require_branch_execution(
                    _call(
                        adapter.run_branch,
                        family=family,
                        anchor=anchor,
                        prefix=random_prefix,
                        split_step=random_step,
                        branch_seed=pair_seed,
                        branch_index=int(pair_index),
                        mode="random_heldout",
                        protocol=protocol,
                    ),
                    label=f"{family_id}/random_heldout/{pair_index}",
                )
                record = _make_branch_record(
                    family,
                    anchor,
                    random_prefix,
                    random_execution,
                    split_step=random_step,
                    branch_seed=pair_seed,
                    branch_index=int(pair_index),
                    mode="random_heldout",
                    protocol=protocol,
                )
                record["paired_oracle_branch_index"] = int(pair_index)
                random_records.append(record)

            probe = {
                "schema": "r142-stage-s-s4-probe-v2",
                "family_id": family_id,
                "anchor_candidate_id": anchor_id,
                "search_grid": [int(item) for item in search_steps],
                "search_branch_count": protocol.search_branch_count,
                "search": search_records,
                "chosen_oracle_t": chosen_step,
                "chosen_oracle_search_success_count": int(chosen["success_count"]),
                "oracle_branches": oracle_records,
                "random_branches": random_records,
                "oracle_branch_count": protocol.heldout_branch_count,
                "random_branch_count": protocol.heldout_branch_count,
                "random_locations": [int(item) for item in random_steps],
                "paired_suffix_seeds": [int(item) for item in paired_seeds],
                "oracle_t_rule": protocol.s4["oracle_t_rule"],
                "random_t_rule": protocol.s4["random_t_rule"],
                **protocol.identity(),
            }
            artifacts = {
                "S4_SEARCH.json": (json.dumps(_jsonable({"search": search_records, "chosen_oracle_t": chosen_step}), ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"),
                "S4_BRANCHES.jsonl": _write_jsonl(
                    [
                        branch
                        for row in search_records
                        for branch in row["branches"]
                    ]
                    + oracle_records
                    + random_records
                ),
                "S4_PROBE.json": (json.dumps(_jsonable(probe), ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"),
            }
            marker = {
                "schema": "r142-stage-s-s4-family-v2",
                "marker_type": "completed_s4_family",
                "family_id": family_id,
                "candidate_count": BASE_CANDIDATE_COUNT,
                "search_grid_count": 9,
                "search_branch_count": protocol.search_branch_count,
                "heldout_branch_count_per_mode": protocol.heldout_branch_count,
                "near_all_fail": True,
                "chosen_oracle_t": chosen_step,
                **protocol.identity(),
                "source_n32_marker_sha256": family.get("source_marker_sha256"),
                "source_n32_family_file_sha256": family.get("source_family_file_sha256"),
            }
            write_atomic_bundle(directory, artifacts, marker_name=marker_name, marker_payload=marker)
            completed += 1
    finally:
        adapter.close()
    return {
        "schema": "r142-stage-s-s4-run-v2",
        "protocol_id": protocol.protocol_id,
        "protocol_authority_sha256": protocol.sha256,
        "near_all_fail_family_count": len(near),
        "completed_family_count": completed,
        "search_grid_count": 9,
        "search_branch_count": protocol.search_branch_count,
        "heldout_branch_count": protocol.heldout_branch_count,
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
    required = ("actions", "trajectory", "terminated", "success", "termination", "policy_forwards", "env_steps", "snapshot_restore_check")
    for field in required:
        if field not in execution:
            raise S45ProvenanceError(f"S5 candidate {candidate_index} lacks {field}")
    if not bool(execution["terminated"]):
        raise S45ProvenanceError(f"S5 candidate {candidate_index} did not reach official termination")
    if not isinstance(execution["termination"], (str, Mapping)) or not execution["termination"]:
        raise S45ProvenanceError(f"S5 candidate {candidate_index} lacks official termination evidence")
    substrate = family.get("substrate", family.get("metadata", {}).get("substrate"))
    pose_dimension = 14 if substrate == "A" else 6 if substrate in {"B", "C"} else None
    if pose_dimension is None:
        raise S45ProvenanceError(f"family {family.get('family_id')} lacks substrate pose contract")
    _validate_pose_trajectory(execution["trajectory"], label=f"S5 candidate {candidate_index} trajectory", dimension=pose_dimension)
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
        "source_n32_bundle_sha256": str(family.get("source_bundle_sha256", "")),
        "source_n32_family_file_sha256": str(family.get("source_family_file_sha256", "")),
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
            if [int(row["candidate_index"]) for row in base_rows] != list(range(BASE_CANDIDATE_COUNT)):
                raise S45ProvenanceError(f"S5 base candidate indices are not exactly 0..31 in {family_id}")
            base_ids = [str(row["candidate_id"]) for row in base_rows]
            if len(set(base_ids)) != BASE_CANDIDATE_COUNT:
                raise S45ProvenanceError(f"S5 base candidate ids are not unique in {family_id}")
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
            extended_ids = [str(row["candidate_id"]) for row in extended]
            if len(set(extended_ids)) != EXTENDED_CANDIDATE_COUNT:
                raise S45ProvenanceError(f"S5 candidate ids are not unique in {family_id}")
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
                "source_n32_family_file_sha256": str(family.get("source_family_file_sha256", "")),
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
                "source_n32_bundle_sha256": family.get("source_bundle_sha256"),
                "source_n32_family_file_sha256": family.get("source_family_file_sha256"),
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
        search_grid = probe.get("search_grid")
        search = probe.get("search")
        if not isinstance(search_grid, list) or len(search_grid) != 9:
            raise S45ProvenanceError(f"S4 {family_id} lacks the frozen nine-point search grid")
        if not isinstance(search, list) or len(search) != 9:
            raise S45ProvenanceError(f"S4 {family_id} lacks nine search-location records")
        if [int(row.get("split_step", -1)) for row in search] != [int(item) for item in search_grid]:
            raise S45ProvenanceError(f"S4 {family_id} search records do not preserve grid order")
        for row in search:
            if not isinstance(row.get("branches"), list) or len(row["branches"]) != protocol.search_branch_count:
                raise S45ProvenanceError(f"S4 {family_id} search location lacks exactly four branches")
            count = 0
            for branch in row["branches"]:
                _require_branch_execution(branch, label=f"S4 persisted {family_id}/search")
                if branch.get("family_id") != family_id or branch.get("protocol_authority_sha256") != protocol.sha256:
                    raise S45ProvenanceError(f"S4 branch provenance mismatch: {family_id}")
                count += int(bool(branch.get("success")) and bool(branch.get("prefix_preserving")))
            if int(row.get("success_count", -1)) != count:
                raise S45ProvenanceError(f"S4 {family_id} search success count was rewritten")
        chosen_step = int(probe.get("chosen_oracle_t", -1))
        if chosen_step not in {int(item) for item in search_grid}:
            raise S45ProvenanceError(f"S4 {family_id} chosen oracle t is outside search grid")
        chosen_rows = [row for row in search if int(row["split_step"]) == chosen_step]
        if len(chosen_rows) != 1 or int(probe.get("chosen_oracle_search_success_count", -1)) != int(chosen_rows[0]["success_count"]):
            raise S45ProvenanceError(f"S4 {family_id} chosen oracle success count is inconsistent")
        maxima = max(int(row["success_count"]) for row in search)
        earliest_max = min(int(row["split_step"]) for row in search if int(row["success_count"]) == maxima)
        if chosen_step != earliest_max:
            raise S45ProvenanceError(f"S4 {family_id} chosen oracle t does not implement max-success/earliest-t rule")

        oracle_branches = probe.get("oracle_branches")
        random_branches = probe.get("random_branches")
        if not isinstance(oracle_branches, list) or len(oracle_branches) != protocol.heldout_branch_count:
            raise S45ProvenanceError(f"S4 {family_id} lacks exactly eight held-out oracle branches")
        if not isinstance(random_branches, list) or len(random_branches) != protocol.heldout_branch_count:
            raise S45ProvenanceError(f"S4 {family_id} lacks exactly eight held-out random branches")
        random_locations = probe.get("random_locations")
        paired_seeds = probe.get("paired_suffix_seeds")
        if not isinstance(random_locations, list) or len(random_locations) != protocol.heldout_branch_count:
            raise S45ProvenanceError(f"S4 {family_id} lacks eight frozen random locations")
        if not isinstance(paired_seeds, list) or len(paired_seeds) != protocol.heldout_branch_count:
            raise S45ProvenanceError(f"S4 {family_id} lacks eight paired suffix seeds")
        expected_random = protocol.random_steps(family_id, int(oracle_branches[0].get("episode_length", 0)))
        if [int(item) for item in random_locations] != [int(item) for item in expected_random]:
            raise S45ProvenanceError(f"S4 {family_id} random locations do not match frozen hash selection")
        for index, (oracle_branch, random_branch) in enumerate(zip(oracle_branches, random_branches)):
            _require_branch_execution(oracle_branch, label=f"S4 persisted {family_id}/oracle_heldout/{index}")
            _require_branch_execution(random_branch, label=f"S4 persisted {family_id}/random_heldout/{index}")
            for branch in (oracle_branch, random_branch):
                if branch.get("family_id") != family_id or branch.get("protocol_authority_sha256") != protocol.sha256:
                    raise S45ProvenanceError(f"S4 branch provenance mismatch: {family_id}")
            if int(oracle_branch.get("split_step", -1)) != chosen_step:
                raise S45ProvenanceError(f"S4 {family_id} oracle held-out branch is not at chosen t")
            if int(random_branch.get("split_step", -1)) != int(random_locations[index]):
                raise S45ProvenanceError(f"S4 {family_id} random held-out location mismatch")
            if int(oracle_branch.get("branch_seed", -1)) != int(random_branch.get("branch_seed", -2)) or int(paired_seeds[index]) != int(oracle_branch.get("branch_seed", -3)):
                raise S45ProvenanceError(f"S4 {family_id} suffix pair does not share the frozen seed")
        if int(probe.get("oracle_branch_count", -1)) != protocol.heldout_branch_count or int(probe.get("random_branch_count", -1)) != protocol.heldout_branch_count:
            raise S45ProvenanceError(f"S4 {family_id} held-out branch counts are not equal eight")
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
        source_fields = {
            "source_n32_marker_sha256": "source_marker_sha256",
            "source_n32_bundle_sha256": "source_bundle_sha256",
            "source_n32_family_file_sha256": "source_family_file_sha256",
        }
        for field, family_field in source_fields.items():
            if isinstance(family, N32Family):
                expected_source = str(getattr(family, family_field, ""))
            else:
                expected_source = str(family.get(family_field, ""))
            if str(payload.get(field, "")) != expected_source:
                raise S45ProvenanceError(f"S5 family {family_id} source SHA mismatch for {field}")
            if str(marker.get(field, "")) != expected_source:
                raise S45ProvenanceError(f"S5 marker {family_id} source SHA mismatch for {field}")
        expected_base = [_base_row_for_s5(family.as_mapping(), row, index) for index, row in enumerate(family.candidates)]
        if _canonical_digest(expected_base) != str(payload.get("base_digest")) or _canonical_digest(expected_base) != _canonical_digest(base_rows):
            raise S45ProvenanceError(f"S5 family {family_id} rewrote immutable base32")
        if [int(row.get("candidate_index", -1)) for row in fresh_rows] != list(FRESH_CANDIDATE_INDICES):
            raise S45ProvenanceError(f"S5 family {family_id} fresh indices are not 32..63")
        if [int(row.get("candidate_index", -1)) for row in extended_rows] != list(range(EXTENDED_CANDIDATE_COUNT)):
            raise S45ProvenanceError(f"S5 family {family_id} extended indices are not 0..63")
        candidate_ids = [str(row.get("candidate_id", "")) for row in extended_rows]
        candidate_seeds = [int(row.get("candidate_seed")) for row in extended_rows]
        if len(set(candidate_ids)) != EXTENDED_CANDIDATE_COUNT or len(set(candidate_seeds)) != EXTENDED_CANDIDATE_COUNT:
            raise S45ProvenanceError(f"S5 family {family_id} candidate ids/seeds are not unique")
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
