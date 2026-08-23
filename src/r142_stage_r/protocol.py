from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

PROTOCOL_ID = "r142-stage-r-phase0r-v1"
SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def ranked_initial_states(suite: str, task_id: int, *, count: int = 16, total: int = 50) -> list[int]:
    if suite not in SUITE_MAX_STEPS and suite != "libero_90":
        raise ValueError(f"unsupported suite {suite!r}")
    if not 0 < count <= total:
        raise ValueError("count must be in [1,total]")
    values = range(int(total))
    return sorted(
        values,
        key=lambda index: sha256_text(f"{PROTOCOL_ID}|{suite}|{int(task_id)}|{index}"),
    )[: int(count)]


def rollout_seed(suite: str, task_id: int, init_state_index: int, candidate_id: int) -> int:
    digest = hashlib.sha256(
        f"{PROTOCOL_ID}|{suite}|{int(task_id)}|{int(init_state_index)}|{int(candidate_id)}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def environment_seed(suite: str, task_id: int, init_state_index: int) -> int:
    digest = hashlib.sha256(
        f"{PROTOCOL_ID}|environment|{suite}|{int(task_id)}|{int(init_state_index)}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


def atomic_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    encoded = (json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)
    directory_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
