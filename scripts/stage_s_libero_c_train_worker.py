#!/usr/bin/env python3
"""Execute the pinned OpenPI trainer with full-state RNG/data-cursor sidecars.

The worker is launched by ``torchrun`` and imports the exact
``scripts/train_pytorch.py`` file from the audited OpenPI checkout.  It wraps
only checkpoint save/load plus a deterministic finite-epoch data-loader
adapter: model, transforms, optimizer, learning-rate calculation, and loss
code remain the pinned upstream implementation.  A missing sidecar or an
unprovable loader cursor on resume is fatal, so an interrupted or legacy
weights-only tree cannot be silently presented as a resumable C run.
"""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterator


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


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _write_rng_completion(step_dir: Path, *, global_step: int, world_size: int) -> None:
    sidecars = [step_dir / f"rng_state.rank{rank}.pt" for rank in range(int(world_size))]
    missing = [path.name for path in sidecars if not path.is_file()]
    if missing:
        raise RuntimeError(f"full-state checkpoint is missing RNG sidecars: {missing}")
    lines = [f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}" for path in sidecars]
    sums = step_dir / "RNG_SHA256SUMS"
    _atomic_text("\n".join(lines) + "\n", sums)
    marker = {
        "schema": "r142-stage-s-c-complete-rng-state-v1",
        "status": "COMPLETED",
        "global_step": int(global_step),
        "world_size": int(world_size),
        "sidecars": [path.name for path in sidecars],
        "rng_sha256sums": sums.name,
        "rng_sha256sums_sha256": hashlib.sha256(sums.read_bytes()).hexdigest(),
    }
    _atomic_text(json.dumps(marker, sort_keys=True, indent=2) + "\n", step_dir / "COMPLETE_RNG_STATE.json")


def _verify_rng_completion(step_dir: Path, *, global_step: int, world_size: int) -> None:
    marker_path = step_dir / "COMPLETE_RNG_STATE.json"
    sums_path = step_dir / "RNG_SHA256SUMS"
    if not marker_path.is_file() or not sums_path.is_file():
        raise RuntimeError(f"full-state resume refused; incomplete RNG checkpoint {step_dir}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected_names = [f"rng_state.rank{rank}.pt" for rank in range(int(world_size))]
    if (
        marker.get("schema") != "r142-stage-s-c-complete-rng-state-v1"
        or marker.get("status") != "COMPLETED"
        or int(marker.get("global_step", -1)) != int(global_step)
        or int(marker.get("world_size", -1)) != int(world_size)
        or marker.get("sidecars") != expected_names
        or marker.get("rng_sha256sums") != sums_path.name
        or marker.get("rng_sha256sums_sha256") != hashlib.sha256(sums_path.read_bytes()).hexdigest()
    ):
        raise RuntimeError(f"full-state resume refused; RNG completion marker drifted at {step_dir}")
    expected_lines = sums_path.read_text(encoding="utf-8").splitlines()
    if len(expected_lines) != int(world_size):
        raise RuntimeError(f"full-state resume refused; RNG SHA manifest width drifted at {step_dir}")
    for name, line in zip(expected_names, expected_lines, strict=True):
        path = step_dir / name
        if not path.is_file():
            raise RuntimeError(f"full-state resume refused; missing RNG sidecar {path}")
        expected = f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {name}"
        if line != expected:
            raise RuntimeError(f"full-state resume refused; RNG sidecar SHA mismatch {path}")


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


def _torch_data_loader(base_loader: Any) -> Any:
    """Resolve the pinned DataLoader without relying on private guessing."""

    implementation = getattr(base_loader, "_data_loader", None)
    candidate = getattr(implementation, "torch_loader", None)
    if candidate is None:
        raise RuntimeError(
            "exact C resume refused; pinned loader does not expose a finite torch_loader"
        )
    if not hasattr(candidate, "__iter__") or not hasattr(candidate, "__len__"):
        raise RuntimeError("exact C resume refused; pinned torch_loader lacks iteration/length")
    return candidate


def _latest_checkpoint_step(checkpoint_dir: Path) -> int:
    steps = sorted(
        int(child.name)
        for child in checkpoint_dir.iterdir()
        if child.is_dir() and child.name.isdigit()
    )
    if not steps:
        raise RuntimeError(f"exact C resume refused; no numeric checkpoint in {checkpoint_dir}")
    return steps[-1]


def _resume_step(config: Any, torch: Any) -> int:
    """Read and validate the native checkpoint cursor before data creation."""

    checkpoint_dir = Path(config.checkpoint_dir).expanduser().resolve()
    step = _latest_checkpoint_step(checkpoint_dir)
    metadata_path = checkpoint_dir / str(step) / "metadata.pt"
    if not metadata_path.is_file():
        raise RuntimeError(f"exact C resume refused; missing checkpoint metadata {metadata_path}")
    try:
        metadata = torch.load(metadata_path, map_location="cpu", weights_only=False)
    except TypeError:
        metadata = torch.load(metadata_path, map_location="cpu")
    if not isinstance(metadata, dict):
        raise RuntimeError(f"exact C resume refused; invalid metadata {metadata_path}")
    observed = int(metadata.get("global_step", -1))
    if observed != step:
        raise RuntimeError(
            f"exact C resume refused; metadata global_step={observed} differs from directory step={step}"
        )
    return step


class ExactCursorDataLoader:
    """Expose one finite sampler epoch and skip the resumable batch cursor.

    The pinned ``TorchDataLoader`` yields indefinitely, while its trainer
    computes an epoch from ``global_step // len(loader)``.  This adapter makes
    that intended contract executable: each iterator consumes exactly one
    finite underlying DataLoader epoch, forwards ``DistributedSampler``'s
    ``set_epoch``, and skips ``resume_step % epoch_length`` batches exactly
    once after a resume.  C fixes ``num_workers=0`` so no hidden worker cursor
    or worker RNG state is omitted from the checkpoint contract.
    """

    def __init__(self, base_loader: Any, *, resume_step: int = 0, require_sampler: bool = False):
        self._base_loader = base_loader
        self._torch_loader = _torch_data_loader(base_loader)
        self._epoch_length = int(len(self._torch_loader))
        if self._epoch_length <= 0:
            raise RuntimeError("exact C resume refused; DataLoader epoch length is not positive")
        self._pending_skip = int(resume_step) % self._epoch_length
        self._require_sampler = bool(require_sampler)
        self._sampler = getattr(self._torch_loader, "sampler", None)
        if self._require_sampler and (
            self._sampler is None or not callable(getattr(self._sampler, "set_epoch", None))
        ):
            raise RuntimeError(
                "exact C resume refused; distributed loader has no DistributedSampler.set_epoch"
            )

    def __len__(self) -> int:
        return self._epoch_length

    def data_config(self) -> Any:
        return self._base_loader.data_config()

    def set_epoch(self, epoch: int) -> None:
        if self._sampler is not None and callable(getattr(self._sampler, "set_epoch", None)):
            self._sampler.set_epoch(int(epoch))
        elif self._require_sampler:
            raise RuntimeError("exact C resume refused; sampler epoch cannot be restored")

    def __iter__(self) -> Iterator[Any]:
        iterator = iter(self._base_loader)
        skip = self._pending_skip
        self._pending_skip = 0
        for _ in range(skip):
            try:
                next(iterator)
            except StopIteration as exc:
                raise RuntimeError(
                    "exact C resume refused; loader ended before cursor skip"
                ) from exc
        remaining = self._epoch_length - skip
        yielded = 0
        try:
            while yielded < remaining:
                try:
                    batch = next(iterator)
                except StopIteration as exc:
                    raise RuntimeError(
                        "exact C resume refused; loader ended before one full epoch"
                    ) from exc
                yield batch
                yielded += 1
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()


def _patch_data_cursor(trainer: Any, torch: Any) -> None:
    """Patch only the pinned loader boundary needed for exact resume."""

    original_build = trainer.build_datasets

    def build_datasets(config: Any) -> tuple[Any, Any]:
        if int(getattr(config, "num_workers", -1)) != 0:
            raise RuntimeError("exact C resume requires frozen --num_workers 0")
        if bool(config.resume) and not torch.distributed.is_initialized():
            raise RuntimeError("exact C resume requires the frozen 8-GPU distributed sampler")
        if torch.distributed.is_initialized() and int(torch.distributed.get_world_size()) != 8:
            raise RuntimeError("exact C training is frozen to an 8-rank distributed sampler")
        base_loader, data_config = original_build(config)
        start_step = _resume_step(config, torch) if bool(config.resume) else 0
        wrapped = ExactCursorDataLoader(
            base_loader,
            resume_step=start_step,
            require_sampler=bool(torch.distributed.is_initialized()),
        )
        return wrapped, data_config

    trainer.build_datasets = build_datasets


def _patch_checkpoint_io(trainer: Any, torch: Any) -> None:
    original_save = trainer.save_checkpoint
    original_load = trainer.load_checkpoint

    def save_checkpoint(model: Any, optimizer: Any, global_step: int, config: Any, is_main: bool, data_config: Any) -> None:
        should_save = (global_step % int(config.save_interval) == 0 and global_step > 0) or global_step == int(config.num_train_steps) - 1
        if not should_save:
            original_save(model, optimizer, global_step, config, is_main, data_config)
            return
        distributed = torch.distributed.is_initialized()
        rank = _rank(torch)
        world_size = int(torch.distributed.get_world_size()) if distributed else 1
        checkpoint_root = Path(config.checkpoint_dir)
        staging_dir = checkpoint_root / f".rng_stage_{global_step}"
        _atomic_torch_save(torch, _capture_rng(torch), staging_dir / f"rng_state.rank{rank}.pt")
        if distributed:
            torch.distributed.barrier()
        # Rank zero may atomically replace the final checkpoint directory;
        # keep every rank's RNG bytes outside tmp_<step> until that completes.
        original_save(model, optimizer, global_step, config, is_main, data_config)
        if distributed:
            torch.distributed.barrier()
        final_dir = checkpoint_root / str(global_step)
        if not final_dir.is_dir():
            raise RuntimeError(f"native checkpoint directory is absent after save: {final_dir}")
        staged = staging_dir / f"rng_state.rank{rank}.pt"
        destination = final_dir / staged.name
        os.replace(staged, destination)
        if distributed:
            torch.distributed.barrier()
        if rank == 0:
            _write_rng_completion(final_dir, global_step=global_step, world_size=world_size)
            try:
                staging_dir.rmdir()
            except OSError:
                pass
        if distributed:
            torch.distributed.barrier()

    def load_checkpoint(model: Any, optimizer: Any, checkpoint_dir: Any, device: Any) -> int:
        checkpoint_root = Path(checkpoint_dir)
        expected_step = _latest_checkpoint_step(checkpoint_root)
        world_size = int(torch.distributed.get_world_size()) if torch.distributed.is_initialized() else 1
        _verify_rng_completion(checkpoint_root / str(expected_step), global_step=expected_step, world_size=world_size)
        global_step = original_load(model, optimizer, checkpoint_dir, device)
        if int(global_step) != int(expected_step):
            raise RuntimeError(
                f"full-state resume refused; loaded step {global_step} differs from verified step {expected_step}"
            )
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
    # Install the narrow, provenance-gated datasets Column bridge before the
    # pinned OpenPI module imports LeRobot's dataset class.  Unknown package
    # versions and constructor ABIs fail closed in this call.
    from r142_stage_s.lerobot_compat import install_column_compat_bridge

    install_column_compat_bridge()
    trainer = _load_trainer(root)
    import torch

    _patch_data_cursor(trainer, torch)
    _patch_checkpoint_io(trainer, torch)
    sys.argv = [str(root / "scripts" / "train_pytorch.py"), *trainer_args]
    trainer.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
