#!/usr/bin/env python3
"""Bind one torchrun worker to one visible GPU before importing JAX/OpenPI.

The parent launcher exposes all GPUs to torchrun.  JAX otherwise initializes
every visible device in every child, so eight workers can each preallocate on
all eight GPUs.  This entrypoint selects the LOCAL_RANK-th device first, then
executes the requested Python script without importing Torch, JAX or OpenPI.
"""

from __future__ import annotations

import importlib
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
    # robosuite 1.4 validates MUJOCO_EGL_DEVICE_ID against the literal
    # CUDA_VISIBLE_DEVICES string at binding_utils import time.  With one
    # physical device selected, satisfy that legacy check first.  The actual
    # EGL context sees the narrowed allocation as logical device zero; main()
    # primes binding_utils and then performs that second mapping below.
    os.environ["EGL_DEVICE_ID"] = selected
    os.environ["MUJOCO_EGL_DEVICE_ID"] = selected
    return selected


def prime_robosuite_then_bind_logical_zero(
    selected: str, *, importer=importlib.import_module
) -> None:
    """Bridge robosuite's import-time physical check and EGL's local index.

    This is deliberately executed before the target imports JAX/OpenPI.  It
    imports only robosuite's binding module while CUDA visibility names the
    selected physical allocation, then switches both render selectors to the
    sole logical device exposed to the worker.
    """

    if os.environ.get("CUDA_VISIBLE_DEVICES") != selected:
        raise SystemExit("CUDA_VISIBLE_DEVICES changed before robosuite EGL priming")
    if os.environ.get("MUJOCO_EGL_DEVICE_ID") != selected:
        raise SystemExit("physical MUJOCO_EGL_DEVICE_ID priming contract drifted")
    importer("robosuite.utils.binding_utils")
    os.environ["EGL_DEVICE_ID"] = "0"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: stage_s_gpu_rank_entry.py TARGET.py [args ...]")
    selected = bind_local_rank()
    prime_robosuite_then_bind_logical_zero(selected)
    target = Path(sys.argv[1]).resolve()
    if not target.is_file() or target.is_symlink():
        raise SystemExit(f"target script is missing or symlinked: {target}")
    sys.argv = [str(target), *sys.argv[2:]]
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
