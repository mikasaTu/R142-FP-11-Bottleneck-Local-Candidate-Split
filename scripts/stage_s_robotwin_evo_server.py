#!/usr/bin/env python3
"""Run one pinned Evo-1 server for the Stage-S RoboTwin substrate-A screen.

The released Evo server is loaded from ``--evo-root`` without modifying its
source.  The only added request branch is the versioned Stage-S exact-replay
dispatcher installed by :mod:`scripts.stage_s_robotwin_evo_server_patch`.
That module's ``EvoServerReplayDispatcher.control_response`` branch is reached
inside ``build_handle_request`` before the unchanged pinned
``infer_from_json_dict`` call.
There is deliberately one process, one CUDA-visible device, and one
WebSocket port per rank.  A client must use the matching port; sharing a
server between ranks would share the process-global Torch/CUDA RNG stream and
is therefore rejected by the launcher.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping


CHECKPOINT_REVISION = "ce8c583724706fbf7a03c17237761c65bf6813a7"
EVO_REVISION = "5fd14b015013c4fd0aacf5f8f48f868ca9b870a2"
STAGE_S_SERVER_PROTOCOL = "r142-evo-exact-replay/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _git_head(path: Path) -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot resolve pinned git head: {path}") from exc


def _verify_checkpoint(checkpoint_dir: Path) -> str:
    if CHECKPOINT_REVISION not in checkpoint_dir.name and not (
        checkpoint_dir / "revision.txt"
    ).is_file():
        raise RuntimeError(
            "checkpoint path is not bound to the exact HF revision "
            f"{CHECKPOINT_REVISION}"
        )
    required = ("config.json", "norm_stats.json", "mp_rank_00_model_states.pt")
    for filename in required:
        path = checkpoint_dir / filename
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"checkpoint file is missing or empty: {path}")
    sums = checkpoint_dir / "SHA256SUMS"
    if not sums.is_file():
        raise RuntimeError(f"checkpoint integrity manifest is missing: {sums}")
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise RuntimeError(f"malformed checkpoint SHA256SUMS line: {line!r}")
        expected, name = parts
        name = name.lstrip(" *")
        path = checkpoint_dir / name
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"checkpoint hash mismatch: {path}")
    return _sha256(sums)


def _load_released_server(evo_root: Path) -> Any:
    """Load Evo1_server.py by path, keeping the released source immutable."""

    server_path = evo_root / "Evo_1" / "scripts" / "Evo1_server.py"
    if not server_path.is_file():
        raise RuntimeError(f"pinned Evo server is missing: {server_path}")
    source = server_path.read_text(encoding="utf-8", errors="replace")
    if "EvoServerReplayDispatcher" in source:
        raise RuntimeError(
            "pinned Evo server source is unexpectedly modified; use the external dispatcher"
        )
    # Evo1_server.py imports top-level config/dataset/scripts modules from the
    # Evo_1 tree.  Insert those roots before executing the source module; the
    # Stage-S dispatcher is imported first by _load_dispatcher below.
    for item in (evo_root / "Evo_1", evo_root / "Evo_1" / "scripts"):
        text = str(item)
        while text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)
    spec = importlib.util.spec_from_file_location("r142_pinned_evo1_server", server_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pinned Evo server module: {server_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_dispatcher(stage_s_root: Path) -> Any:
    stage_text = str(stage_s_root)
    if stage_text not in sys.path:
        sys.path.insert(0, stage_text)
    from scripts.stage_s_robotwin_evo_server_patch import build_handle_request

    return build_handle_request


def _runtime_provenance(
    *,
    output_root: Path,
    stage_s_root: Path,
    evo_root: Path,
    checkpoint_dir: Path,
    rank: int,
    world_size: int,
    port: int,
    gpu_id: int,
    checkpoint_sums_sha: str,
) -> dict[str, Any]:
    server_source = evo_root / "Evo_1" / "scripts" / "Evo1_server.py"
    dispatcher_source = stage_s_root / "scripts" / "stage_s_robotwin_evo_server_patch.py"
    return {
        "protocol": "R142-FP-11 Stage-S substrate A",
        "server_protocol": STAGE_S_SERVER_PROTOCOL,
        "rank": int(rank),
        "world_size": int(world_size),
        "gpu_id_physical": int(gpu_id),
        "gpu_id_logical": 0,
        "port": int(port),
        "server_bind": "127.0.0.1",
        "server_client_ownership": "one_server_one_client_one_gpu_one_port",
        "evo_root": str(evo_root),
        "evo_revision": EVO_REVISION,
        "evo_head": _git_head(evo_root),
        "released_server_source": str(server_source),
        "released_server_sha256": _sha256(server_source),
        "dispatcher_source": str(dispatcher_source),
        "dispatcher_sha256": _sha256(dispatcher_source),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_sha256s_sha256": checkpoint_sums_sha,
        "synthetic_rollouts": False,
        "expert_trajectory": False,
        "inference_branch": "pinned_infer_from_json_dict_unchanged",
    }


async def _serve(
    *,
    server_module: Any,
    dispatcher_factory: Any,
    provenance: Mapping[str, Any],
    output_root: Path,
    rank: int,
    port: int,
) -> None:
    import websockets

    handler = dispatcher_factory(
        infer_from_json_dict=server_module.infer_from_json_dict,
        model=server_module._stage_s_model,
        normalizer=server_module._stage_s_normalizer,
        arm_key="aloha_joint",
        dataset_key="robotwin_blocks_ranking_size",
        require_torch=True,
    )
    async with websockets.serve(
        handler,
        "127.0.0.1",
        int(port),
        max_size=100_000_000,
        ping_interval=None,
        ping_timeout=None,
    ):
        ready = dict(provenance)
        ready.update({"status": "READY", "pid": os.getpid()})
        _atomic_json(output_root / f"SERVER_READY_RANK-{rank:04d}.json", ready)
        print(json.dumps(ready, sort_keys=True), flush=True)
        await asyncio.Future()


def run(args: argparse.Namespace) -> int:
    if args.world_size != 8:
        raise RuntimeError("Stage-S substrate A requires exactly world_size=8")
    if args.rank < 0 or args.rank >= args.world_size:
        raise RuntimeError("rank must satisfy 0 <= rank < world_size")
    if args.gpu_id < 0 or args.gpu_id >= 8:
        raise RuntimeError("gpu_id must be one of the eight Stage-S A devices")
    if args.port < 1024 or args.port > 65535:
        raise RuntimeError("server port is outside the user-space range")
    if os.getuid() != 2254 or os.getgid() != 2254:
        raise RuntimeError("Stage-S persistent server work must run as UID/GID 2254:2254")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible != str(args.gpu_id):
        raise RuntimeError(
            "server CUDA ownership mismatch: expected CUDA_VISIBLE_DEVICES="
            f"{args.gpu_id}, got {visible!r}"
        )

    stage_s_root = args.stage_s_root.resolve()
    evo_root = args.evo_root.resolve()
    checkpoint_dir = args.checkpoint_dir.resolve()
    output_root = args.output_root.resolve()
    if _git_head(evo_root) != EVO_REVISION:
        raise RuntimeError(f"Evo source is not pinned to {EVO_REVISION}")
    checkpoint_sums_sha = _verify_checkpoint(checkpoint_dir)

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Evo server requires Torch") from exc
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "each Evo server must see exactly one CUDA device after ownership binding"
        )
    if torch.cuda.current_device() != 0:
        raise RuntimeError("the server's logical CUDA device must be index 0")

    dispatcher_factory = _load_dispatcher(stage_s_root)
    server_module = _load_released_server(evo_root)
    print("STAGE_S_LOAD_MODEL_BEGIN", flush=True)
    server_module._stage_s_model, server_module._stage_s_normalizer = (
        server_module.load_model_and_normalizer(str(checkpoint_dir))
    )
    provenance = _runtime_provenance(
        output_root=output_root,
        stage_s_root=stage_s_root,
        evo_root=evo_root,
        checkpoint_dir=checkpoint_dir,
        rank=args.rank,
        world_size=args.world_size,
        port=args.port,
        gpu_id=args.gpu_id,
        checkpoint_sums_sha=checkpoint_sums_sha,
    )
    _atomic_json(output_root / f"SERVER_PROVENANCE_RANK-{args.rank:04d}.json", provenance)
    asyncio.run(
        _serve(
            server_module=server_module,
            dispatcher_factory=dispatcher_factory,
            provenance=provenance,
            output_root=output_root,
            rank=args.rank,
            port=args.port,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-s-root", type=Path, required=True)
    parser.add_argument("--evo-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    return parser


if __name__ == "__main__":
    try:
        raise SystemExit(run(build_parser().parse_args()))
    except Exception as exc:
        print(
            json.dumps(
                {"status": "BLOCKED_SERVER_CAPABILITY", "error_type": type(exc).__name__, "error": str(exc)},
                ensure_ascii=False,
            ),
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(2)
