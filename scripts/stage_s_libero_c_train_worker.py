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
import time
from pathlib import Path
from typing import Any, Callable, Iterator


# The native OpenPI saver writes the model/optimizer/metadata tree from rank 0
# and returns before every rank necessarily observes that CPFS rename. Keep
# this visibility contract explicit and bounded. These values affect only the
# post-save visibility wait; they do not alter the frozen training schedule.
CHECKPOINT_READY_NAME = "CHECKPOINT_READY.json"
CHECKPOINT_READY_SCHEMA = "r142-stage-s-c-checkpoint-ready-v1"
CHECKPOINT_READY_CORE_FILES = ("model.safetensors", "optimizer.pt", "metadata.pt")
CHECKPOINT_READY_TIMEOUT_S = 300.0
CHECKPOINT_READY_POLL_INTERVAL_S = 0.5


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


def _checkpoint_core_inventory(step_dir: Path, *, include_sha256: bool = False) -> list[dict[str, Any]]:
    """Return the native checkpoint files that must be visible before sidecars.

    The native trainer already writes these files into a temporary directory
    and atomically renames that directory. This inventory is deliberately
    small and size-based so the READY marker does not add a multi-gigabyte
    hashing pass to every checkpoint. Tests and offline audits may request
    optional SHA-256 values for the same files.
    """

    inventory: list[dict[str, Any]] = []
    for name in CHECKPOINT_READY_CORE_FILES:
        path = step_dir / name
        if not path.is_file():
            raise RuntimeError(f"native checkpoint core file is absent: {path}")
        size = int(path.stat().st_size)
        if size <= 0:
            raise RuntimeError(f"native checkpoint core file is empty: {path}")
        entry: dict[str, Any] = {"name": name, "exists": True, "size": size}
        if include_sha256:
            entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        inventory.append(entry)
    return inventory


def _write_checkpoint_ready(
    step_dir: Path,
    *,
    global_step: int,
    world_size: int,
    include_sha256: bool = False,
) -> dict[str, Any]:
    """Publish an atomic, content-bound READY marker after native save."""

    step_dir = Path(step_dir)
    core_files = _checkpoint_core_inventory(step_dir, include_sha256=include_sha256)
    marker: dict[str, Any] = {
        "schema": CHECKPOINT_READY_SCHEMA,
        "status": "READY",
        "global_step": int(global_step),
        "world_size": int(world_size),
        "checkpoint_dir": str(step_dir.resolve()),
        "core_files": core_files,
    }
    _atomic_text(json.dumps(marker, sort_keys=True, indent=2) + "\n", step_dir / CHECKPOINT_READY_NAME)
    return marker


def _read_checkpoint_ready_once(
    step_dir: Path,
    *,
    global_step: int,
    world_size: int,
) -> tuple[dict[str, Any] | None, str]:
    """Read READY and verify its files, returning a transient reason if absent.

    A marker that parses but binds to another step/world/checkpoint is a hard
    failure (stale or corrupt evidence). Missing files, short reads and JSON
    decode errors are treated as transient CPFS visibility failures by the
    bounded polling caller.
    """

    step_dir = Path(step_dir)
    marker_path = step_dir / CHECKPOINT_READY_NAME
    if not marker_path.is_file():
        return None, f"missing {marker_path.name}"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"READY marker is not yet readable: {exc}"
    if not isinstance(marker, dict):
        raise RuntimeError(f"checkpoint READY marker is not an object: {marker_path}")
    if marker.get("schema") != CHECKPOINT_READY_SCHEMA or marker.get("status") != "READY":
        raise RuntimeError(f"checkpoint READY marker schema/status mismatch: {marker_path}")
    if int(marker.get("global_step", -1)) != int(global_step):
        raise RuntimeError(f"checkpoint READY marker global_step mismatch: {marker_path}")
    if int(marker.get("world_size", -1)) != int(world_size):
        raise RuntimeError(f"checkpoint READY marker world_size mismatch: {marker_path}")
    if marker.get("checkpoint_dir") != str(step_dir.resolve()):
        raise RuntimeError(f"checkpoint READY marker checkpoint_dir mismatch: {marker_path}")

    entries = marker.get("core_files")
    expected_names = list(CHECKPOINT_READY_CORE_FILES)
    observed_names = [entry.get("name") for entry in entries if isinstance(entry, dict)] if isinstance(entries, list) else []
    if not isinstance(entries, list) or observed_names != expected_names:
        raise RuntimeError(f"checkpoint READY marker core-file list mismatch: {marker_path}")
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("exists") is not True:
            raise RuntimeError(f"checkpoint READY marker core-file evidence is invalid: {marker_path}")
        name = entry["name"]
        path = step_dir / name
        if not path.is_file():
            return None, f"core file is not yet visible: {name}"
        try:
            observed_size = int(path.stat().st_size)
        except OSError as exc:
            return None, f"core file stat is not yet visible: {name}: {exc}"
        if observed_size != int(entry.get("size", -1)) or observed_size <= 0:
            return None, f"core file size is not stable: {name}"
        expected_sha = entry.get("sha256")
        if expected_sha is not None:
            if not isinstance(expected_sha, str) or len(expected_sha) != 64:
                raise RuntimeError(f"checkpoint READY marker SHA evidence is invalid: {name}")
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
                return None, f"core file SHA is not stable: {name}"
    return marker, "ready"


def _wait_for_checkpoint_ready(
    step_dir: Path,
    *,
    global_step: int,
    world_size: int,
    timeout_s: float = CHECKPOINT_READY_TIMEOUT_S,
    poll_interval_s: float = CHECKPOINT_READY_POLL_INTERVAL_S,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Wait for rank-0 READY plus stable core files, or fail closed."""

    timeout_s = float(timeout_s)
    poll_interval_s = float(poll_interval_s)
    if timeout_s < 0:
        raise ValueError("checkpoint READY timeout must be non-negative")
    if timeout_s > 0 and poll_interval_s <= 0:
        raise ValueError("checkpoint READY poll interval must be positive")
    started = monotonic()
    deadline = started + timeout_s
    last_reason = "not checked"
    while True:
        marker, last_reason = _read_checkpoint_ready_once(
            step_dir,
            global_step=global_step,
            world_size=world_size,
        )
        if marker is not None:
            return marker
        now = monotonic()
        if now >= deadline:
            raise RuntimeError(
                f"checkpoint READY visibility timeout after {timeout_s:.1f}s at {Path(step_dir)}: {last_reason}"
            )
        sleep(min(poll_interval_s, max(0.0, deadline - now)))


def _verify_checkpoint_ready(step_dir: Path, *, global_step: int, world_size: int) -> dict[str, Any]:
    """Verify READY synchronously before any checkpoint can be resumed."""

    marker, reason = _read_checkpoint_ready_once(
        step_dir,
        global_step=global_step,
        world_size=world_size,
    )
    if marker is None:
        raise RuntimeError(
            f"full-state resume refused; checkpoint READY is incomplete at {step_dir}: {reason}"
        )
    return marker


def _write_rng_completion(step_dir: Path, *, global_step: int, world_size: int) -> None:
    _verify_checkpoint_ready(step_dir, global_step=global_step, world_size=world_size)
    sidecars = [step_dir / f"rng_state.rank{rank}.pt" for rank in range(int(world_size))]
    missing = [path.name for path in sidecars if not path.is_file() or path.stat().st_size <= 0]
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
    _verify_checkpoint_ready(step_dir, global_step=global_step, world_size=world_size)
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
    world_size = int(torch.distributed.get_world_size()) if torch.distributed.is_initialized() else 1
    # Validate the complete marker before the loader is constructed. This
    # prevents a weights-only or rank-partial checkpoint from influencing the
    # data cursor even if the later native load would reject it.
    _verify_rng_completion(checkpoint_dir / str(step), global_step=step, world_size=world_size)
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
        final_dir = checkpoint_root / str(global_step)
        if rank == 0:
            # The native saver performs an atomic directory rename only on
            # rank 0. Publish READY after the rename and after checking the
            # native model/optimizer/metadata files. Other ranks may observe
            # this marker before the CPFS directory contents, so every rank
            # below re-checks marker-bound sizes with a bounded wait.
            _write_checkpoint_ready(
                final_dir,
                global_step=global_step,
                world_size=world_size,
            )
        if distributed:
            # Synchronize after rank 0 publishes READY so a stale marker from
            # an interrupted attempt cannot be consumed by another rank
            # before this save attempt has finished its native rename.
            torch.distributed.barrier()
        _wait_for_checkpoint_ready(
            final_dir,
            global_step=global_step,
            world_size=world_size,
        )
        staged = staging_dir / f"rng_state.rank{rank}.pt"
        destination = final_dir / staged.name
        if not staged.is_file():
            raise RuntimeError(f"staged RNG sidecar is absent for rank {rank}: {staged}")
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
