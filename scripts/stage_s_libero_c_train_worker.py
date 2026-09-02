#!/usr/bin/env python3
"""Execute the pinned OpenPI trainer with full-state RNG sidecars.

The worker is launched by ``torchrun`` and imports the exact
``scripts/train_pytorch.py`` file from the audited OpenPI checkout.  It wraps
only checkpoint save/load: model, data, optimizer, learning-rate calculation,
and loss code remain the pinned upstream implementation.  A missing sidecar
on resume is fatal, so an interrupted or legacy weights-only tree cannot be
silently presented as a resumable C run.
"""

from __future__ import annotations

import importlib.util
import os
import random
import sys
from pathlib import Path
from typing import Any


def _parse_worker_args(argv: list[str]) -> tuple[Path, list[str]]:
    if "--openpi-root" not in argv:
        raise SystemExit("worker requires --openpi-root")
    index = argv.index("--openpi-root")
    if index + 1 >= len(argv):
        raise SystemExit("--openpi-root requires a path")
    root = Path(argv[index + 1]).expanduser().resolve()
    remaining = argv[:index] + argv[index + 2 :]
    if "--" in remaining:
        remaining = remaining[remaining.index("--") + 1 :]
    return root, remaining


def _load_trainer(root: Path) -> Any:
    path = root / "scripts" / "train_pytorch.py"
    if not path.is_file():
        raise SystemExit(f"missing pinned trainer: {path}")
    spec = importlib.util.spec_from_file_location("r142_stage_s_pinned_openpi_trainer", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import pinned trainer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rank(torch: Any) -> int:
    if torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return int(os.environ.get("RANK", "0"))


def _atomic_torch_save(torch: Any, value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(value, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _capture_rng(torch: Any) -> dict[str, Any]:
    cuda_states = None
    if torch.cuda.is_available():
        cuda_states = [state.cpu() for state in torch.cuda.get_rng_state_all()]
    return {
        "schema": "r142-stage-s-c-rng-state-v1",
        "python": random.getstate(),
        "numpy": __import__("numpy").random.get_state(),
        "torch": torch.get_rng_state().cpu(),
        "cuda": cuda_states,
    }


def _restore_rng(torch: Any, state: dict[str, Any]) -> None:
    if state.get("schema") != "r142-stage-s-c-rng-state-v1":
        raise RuntimeError("RNG sidecar schema mismatch")
    random.setstate(state["python"])
    __import__("numpy").random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available():
        cuda = state.get("cuda")
        if not isinstance(cuda, list) or not cuda:
            raise RuntimeError("CUDA RNG sidecar is absent")
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda])


def _patch_checkpoint_io(trainer: Any, torch: Any) -> None:
    original_save = trainer.save_checkpoint
    original_load = trainer.load_checkpoint

    def save_checkpoint(model: Any, optimizer: Any, global_step: int, config: Any, is_main: bool, data_config: Any) -> None:
        original_save(model, optimizer, global_step, config, is_main, data_config)
        should_save = (global_step % int(config.save_interval) == 0 and global_step > 0) or global_step == int(config.num_train_steps) - 1
        if not should_save:
            return
        distributed = torch.distributed.is_initialized()
        if distributed:
            torch.distributed.barrier()
        final_dir = Path(config.checkpoint_dir) / str(global_step)
        if final_dir.is_dir():
            _atomic_torch_save(torch, _capture_rng(torch), final_dir / f"rng_state.rank{_rank(torch)}.pt")
        if distributed:
            torch.distributed.barrier()

    def load_checkpoint(model: Any, optimizer: Any, checkpoint_dir: Any, device: Any) -> int:
        global_step = original_load(model, optimizer, checkpoint_dir, device)
        target = Path(checkpoint_dir) / str(global_step)
        sidecar = target / f"rng_state.rank{_rank(torch)}.pt"
        if not sidecar.is_file():
            raise RuntimeError(f"full-state resume refused; missing RNG sidecar {sidecar}")
        try:
            state = torch.load(sidecar, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(sidecar, map_location="cpu")
        if not isinstance(state, dict):
            raise RuntimeError(f"invalid RNG sidecar {sidecar}")
        _restore_rng(torch, state)
        return global_step

    trainer.save_checkpoint = save_checkpoint
    trainer.load_checkpoint = load_checkpoint


def main(argv: list[str] | None = None) -> int:
    root, trainer_args = _parse_worker_args(list(sys.argv[1:] if argv is None else argv))
    trainer = _load_trainer(root)
    import torch

    _patch_checkpoint_io(trainer, torch)
    sys.argv = [str(root / "scripts" / "train_pytorch.py"), *trainer_args]
    trainer.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
