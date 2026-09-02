#!/usr/bin/env python3
"""Bind one torchrun worker to one visible GPU before importing JAX/OpenPI.

The parent launcher exposes all GPUs to torchrun.  JAX otherwise initializes
every visible device in every child, so eight workers can each preallocate on
all eight GPUs.  This entrypoint selects the LOCAL_RANK-th device first, then
executes the requested Python script without importing Torch, JAX or OpenPI.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def bind_local_rank() -> str:
    raw_rank = os.environ.get("LOCAL_RANK")
    if raw_rank is None:
        raise SystemExit("LOCAL_RANK is required")
    try:
        local_rank = int(raw_rank)
    except ValueError as exc:
        raise SystemExit(f"invalid LOCAL_RANK: {raw_rank!r}") from exc
    visible = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    if not visible:
        raise SystemExit("CUDA_VISIBLE_DEVICES must enumerate the allocated GPUs")
    if len(visible) == 1:
        if local_rank != 0:
            raise SystemExit("one visible GPU is incompatible with LOCAL_RANK > 0")
        selected = visible[0]
    else:
        if not 0 <= local_rank < len(visible):
            raise SystemExit(
                f"LOCAL_RANK {local_rank} is outside {len(visible)} visible GPUs"
            )
        selected = visible[local_rank]
    os.environ["CUDA_VISIBLE_DEVICES"] = selected
    return selected


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: stage_s_gpu_rank_entry.py TARGET.py [args ...]")
    bind_local_rank()
    target = Path(sys.argv[1]).resolve()
    if not target.is_file() or target.is_symlink():
        raise SystemExit(f"target script is missing or symlinked: {target}")
    sys.argv = [str(target), *sys.argv[2:]]
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
