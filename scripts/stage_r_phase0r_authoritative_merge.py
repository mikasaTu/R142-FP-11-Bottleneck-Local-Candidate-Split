#!/usr/bin/env python3
"""Fail-closed, outcome-blind merge for the frozen Phase-0R authority map."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np


PROTOCOL_ID = "r142-stage-r-phase0r-v1"
TASKS = [
    f"{suite}_task{task_id:02d}"
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    for task_id in range(10)
]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_checksum_manifest(root: Path) -> str:
    manifest = root / "SHA256SUMS"
    if not manifest.is_file():
        raise RuntimeError(f"missing checksum manifest: {manifest}")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError(f"unsafe checksum entry: {relative}")
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"unsafe or missing checksum entry: {relative}")
        if digest(path) != expected:
            raise RuntimeError(f"checksum mismatch: {path}")
    return digest(manifest)


def validate_pair(root: Path, stem: str, uid: int, gid: int) -> dict[str, object]:
    metadata_path = root / f"{stem}.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    data_path = root / metadata["data_file"]
    match = re.fullmatch(r"(libero_(?:spatial|object|goal|10))_task(\d{2})", stem)
    if match is None:
        raise RuntimeError(f"invalid task stem: {stem}")
    data_sha = digest(data_path)
    if metadata != {
        **metadata,
        "protocol_id": PROTOCOL_ID,
        "suite": match.group(1),
        "task_id": int(match.group(2)),
        "rollout_count": 512,
        "data_file": f"{stem}.npz",
        "data_sha256": data_sha,
    }:
        raise RuntimeError(f"metadata contract mismatch: {stem}")
    for path in (metadata_path, data_path):
        stat = path.stat()
        if (stat.st_uid, stat.st_gid) != (uid, gid):
            raise RuntimeError(f"owner mismatch: {path}")
    with np.load(data_path, allow_pickle=False) as data:
        required = {
            "lengths", "offsets", "actions", "eef", "objects", "progress",
            "success", "init_state", "candidate_id", "rollout_seed", "policy_forwards",
        }
        if not required.issubset(data.files):
            raise RuntimeError(f"missing arrays: {stem}")
        for key in ("success", "init_state", "candidate_id", "rollout_seed", "policy_forwards", "lengths"):
            if data[key].shape != (512,):
                raise RuntimeError(f"shape mismatch {stem}:{key}")
        if data["offsets"].shape != (513,) or int(data["offsets"][0]) != 0:
            raise RuntimeError(f"offset shape mismatch: {stem}")
        if not np.array_equal(np.diff(data["offsets"]), data["lengths"]):
            raise RuntimeError(f"offset/length mismatch: {stem}")
        if int(data["offsets"][-1]) != len(data["actions"]):
            raise RuntimeError(f"terminal offset mismatch: {stem}")
        if not (len(data["actions"]) == len(data["eef"]) == len(data["objects"]) == len(data["progress"])):
            raise RuntimeError(f"trajectory length mismatch: {stem}")
        for key in ("actions", "eef", "objects", "progress"):
            if not np.isfinite(data[key]).all():
                raise RuntimeError(f"non-finite trajectory array {stem}:{key}")
        counts = collections.Counter(map(int, data["init_state"]))
        if len(counts) != 16 or set(counts.values()) != {32}:
            raise RuntimeError(f"state/candidate coverage mismatch: {stem}")
        observed: set[int] = set()
        for init_state in counts:
            selector = data["init_state"] == init_state
            candidates = data["candidate_id"][selector]
            seeds = data["rollout_seed"][selector]
            if sorted(map(int, candidates)) != list(range(32)):
                raise RuntimeError(f"candidate IDs mismatch: {stem}/{init_state}")
            for candidate, seed in zip(candidates, seeds, strict=True):
                expected = int.from_bytes(
                    hashlib.sha256(
                        f"{PROTOCOL_ID}|{metadata['suite']}|{metadata['task_id']}|{int(init_state)}|{int(candidate)}".encode()
                    ).digest()[:8],
                    "big",
                )
                if int(seed) != expected:
                    raise RuntimeError(f"rollout seed mismatch: {stem}/{init_state}/{int(candidate)}")
                observed.add(int(seed))
        if len(observed) != 512:
            raise RuntimeError(f"rollout seeds are not unique: {stem}")
    return {
        "stem": stem,
        "npz_sha256": data_sha,
        "metadata_sha256": digest(metadata_path),
    }


def validate_subset(root: Path, shard: str, targets: list[str], contract: dict[str, object]) -> dict[str, str]:
    checksum_sha = validate_checksum_manifest(root)
    raw = json.loads((root / "COMPLETED_SUBSET_RAW.json").read_text(encoding="utf-8"))
    complete = json.loads((root / "COMPLETED_EVALUATION_RESULT.json").read_text(encoding="utf-8"))
    expected = contract["shards"][shard]
    if raw["protocol_id"] != PROTOCOL_ID or raw["shard"] != shard or raw["target_tasks"] != targets:
        raise RuntimeError(f"subset authority mismatch: {shard}")
    if raw["task_count"] != 4 or raw["rollout_count"] != 2048 or raw["outcomes_unblinded"] is not False:
        raise RuntimeError(f"subset completion mismatch: {shard}")
    if complete["decision"] != "RAW_SUBSET_COMPLETE_NO_UNBLINDING" or complete["phase1_authorized"] is not False:
        raise RuntimeError(f"subset decision mismatch: {shard}")
    for rank, stem in zip(expected["global_ranks"], targets, strict=True):
        marker = json.loads((root / "raw" / f"COMPLETE.rank{rank}.json").read_text(encoding="utf-8"))
        expected_tasks = expected["prerequisites"][str(rank)] + [stem]
        if marker != {"completed_tasks": expected_tasks, "protocol_id": PROTOCOL_ID, "rank": rank}:
            raise RuntimeError(f"rank marker mismatch: {shard}/{rank}")
    return {
        "completed_subset_sha256": digest(root / "COMPLETED_SUBSET_RAW.json"),
        "completed_result_sha256": digest(root / "COMPLETED_EVALUATION_RESULT.json"),
        "sha256sums_sha256": checksum_sha,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--shard-a", type=Path, required=True)
    parser.add_argument("--shard-b", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--uid", type=int, default=2254)
    parser.add_argument("--gid", type=int, default=2254)
    args = parser.parse_args()
    if (os.getuid(), os.getgid()) != (args.uid, args.gid):
        raise RuntimeError("merge must run as the frozen numeric owner")
    if args.output.exists():
        for required in ("AUTHORITY_MANIFEST.json", "COMPLETED_PHASE0R_RAW.json", "SHA256SUMS"):
            if not (args.output / required).is_file():
                raise RuntimeError(f"partial existing merge directory: missing {required}")
        validate_checksum_manifest(args.output)
        print("AUTHORITATIVE_MERGE_ALREADY_COMPLETE_VALIDATED")
        return
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if contract["protocol_id"] != PROTOCOL_ID or contract["shards"]["A"]["target_tasks"] != TASKS[32:36] or contract["shards"]["B"]["target_tasks"] != TASKS[36:40]:
        raise RuntimeError("frozen authority contract mismatch")
    subset_evidence = {
        "A": validate_subset(args.shard_a, "A", TASKS[32:36], contract),
        "B": validate_subset(args.shard_b, "B", TASKS[36:40], contract),
    }
    sources = [("parent", args.parent / "raw")] * 32 + [("shard_a", args.shard_a / "raw")] * 4 + [("shard_b", args.shard_b / "raw")] * 4
    records = []
    for index, (stem, (label, source)) in enumerate(zip(TASKS, sources, strict=True)):
        record = validate_pair(source, stem, args.uid, args.gid)
        records.append({"index": index, "source": label, **record})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.tmp.", dir=args.output.parent))
    try:
        raw_dir = temporary / "raw"
        raw_dir.mkdir(mode=0o700)
        for stem, (_, source) in zip(TASKS, sources, strict=True):
            for suffix in ("json", "npz"):
                os.link(source / f"{stem}.{suffix}", raw_dir / f"{stem}.{suffix}")
        authority = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "scientific_source_commit": contract["scientific_source_commit"],
            "authority_rule": "parent[0:32]+shard_A[32:36]+shard_B[36:40]",
            "outcome_selection_permitted": False,
            "parent_run_id": contract["parent_run_id"],
            "parent_job_id": contract["parent_job_id"],
            "subset_evidence": subset_evidence,
            "records": records,
        }
        atomic_json(temporary / "AUTHORITY_MANIFEST.json", authority)
        raw_complete = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "task_count": 40,
            "rollout_count": 40 * 16 * 32,
            "outcomes_unblinded": False,
            "authority_manifest_sha256": digest(temporary / "AUTHORITY_MANIFEST.json"),
            "artifacts": [
                {"path": f"raw/{record['stem']}.npz", "sha256": record["npz_sha256"]}
                for record in records
            ] + [
                {"path": f"raw/{record['stem']}.json", "sha256": record["metadata_sha256"]}
                for record in records
            ],
        }
        atomic_json(temporary / "COMPLETED_PHASE0R_RAW.json", raw_complete)
        paths = sorted(path for path in temporary.rglob("*") if path.is_file())
        lines = [f"{digest(path)}  {path.relative_to(temporary)}\n" for path in paths]
        checksum_path = temporary / "SHA256SUMS"
        with checksum_path.open("w", encoding="utf-8") as handle:
            handle.writelines(lines)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, args.output)
        directory_fd = os.open(args.output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    validate_checksum_manifest(args.output)
    print("AUTHORITATIVE_MERGE_COMPLETE_OUTCOME_BLIND")


if __name__ == "__main__":
    main()
