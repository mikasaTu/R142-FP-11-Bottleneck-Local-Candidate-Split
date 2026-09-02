"""Fail-closed compatibility bridge for the pinned OpenPI/LeRobot ABI.

The frozen OpenPI lock resolves LeRobot commit ``0cf8648`` with
``datasets==3.6.0``.  The dev14 training image contains that LeRobot source
but ``datasets==4.8.4``.  In the latter API, ``Dataset["timestamp"]`` is a
``datasets.arrow_dataset.Column`` of Python scalars; the pinned LeRobot
constructor still passes it to ``torch.stack``, which only accepts tensors.

The bridge is deliberately narrow: it is enabled only for the exact observed
package provenance and only converts a real datasets ``Column`` whose values
are scalar real numbers.  It does not rewrite the dataset, touch parquet
bytes, or alter model/training values.  Any unknown package/version/source or
non-scalar column fails closed.
"""

from __future__ import annotations

import contextlib
import functools
import importlib.metadata
import inspect
import json
import numbers
import threading
from pathlib import Path
from typing import Any, Iterator


PINNED_LEROBOT_COMMIT = "0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
PINNED_LEROBOT_VERSION = "0.1.0"
PINNED_DATASETS_VERSION = "3.6.0"
SUPPORTED_COLUMN_DATASETS_VERSION = "4.8.4"
COMPATIBILITY_CONTRACT = (
    f"lerobot=={PINNED_LEROBOT_VERSION}@{PINNED_LEROBOT_COMMIT}; "
    f"datasets=={PINNED_DATASETS_VERSION} native or "
    f"datasets=={SUPPORTED_COLUMN_DATASETS_VERSION} scalar-column bridge"
)

_PATCH_LOCK = threading.RLock()


def _distribution_source(package: str) -> dict[str, Any]:
    distribution = importlib.metadata.distribution(package)
    raw = distribution.read_text("direct_url.json")
    direct_url: dict[str, Any] = {}
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            direct_url = {"raw": raw}
        else:
            if isinstance(parsed, dict):
                direct_url = parsed
            else:
                direct_url = {"raw": raw}
    return {
        "version": distribution.version,
        "direct_url": direct_url,
        "source_text": json.dumps(direct_url, sort_keys=True, separators=(",", ":")),
    }


def runtime_dependency_contract() -> dict[str, Any]:
    """Read and validate the installed dependency provenance without mutation."""

    errors: list[str] = []
    try:
        lerobot = _distribution_source("lerobot")
    except importlib.metadata.PackageNotFoundError as exc:
        lerobot = {"version": None, "direct_url": {}, "source_text": ""}
        errors.append(f"lerobot distribution is missing: {exc}")
    try:
        datasets = _distribution_source("datasets")
    except importlib.metadata.PackageNotFoundError as exc:
        datasets = {"version": None, "direct_url": {}, "source_text": ""}
        errors.append(f"datasets distribution is missing: {exc}")

    if lerobot.get("version") != PINNED_LEROBOT_VERSION:
        errors.append(
            f"lerobot version mismatch: expected {PINNED_LEROBOT_VERSION}, got {lerobot.get('version')}"
        )
    if PINNED_LEROBOT_COMMIT not in str(lerobot.get("source_text", "")):
        errors.append(
            "lerobot provenance does not identify the pinned commit "
            f"{PINNED_LEROBOT_COMMIT}"
        )
    datasets_version = datasets.get("version")
    if datasets_version == PINNED_DATASETS_VERSION:
        mode = "native-pinned-datasets"
    elif datasets_version == SUPPORTED_COLUMN_DATASETS_VERSION:
        mode = "datasets-column-scalar-bridge"
    else:
        mode = "unsupported"
        errors.append(
            "datasets version is outside the frozen compatibility contract: "
            f"expected {PINNED_DATASETS_VERSION} or {SUPPORTED_COLUMN_DATASETS_VERSION}, got {datasets_version}"
        )
    return {
        "valid": not errors,
        "contract": COMPATIBILITY_CONTRACT,
        "lerobot_version": lerobot.get("version"),
        "lerobot_commit": PINNED_LEROBOT_COMMIT,
        "lerobot_direct_url": lerobot.get("direct_url"),
        "datasets_version": datasets_version,
        "datasets_direct_url": datasets.get("direct_url"),
        "mode": mode,
        "bridge_installed": False,
        "errors": errors,
    }


def _datasets_column_type() -> type[Any] | None:
    try:
        from datasets.arrow_dataset import Column  # type: ignore[import-not-found]
    except (ImportError, AttributeError):
        return None
    return Column


def _stack_scalar_column(
    original_stack: Any,
    torch_module: Any,
    column_type: type[Any],
    values: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Adapt only scalar datasets Columns at LeRobot's timestamp check."""

    if not isinstance(values, column_type):
        return original_stack(values, *args, **kwargs)
    dim = 0
    if args:
        if len(args) != 1:
            raise TypeError("Stage-S Column bridge only supports torch.stack(..., dim=0)")
        dim = args[0]
    if "dim" in kwargs:
        if len(kwargs) != 1:
            raise TypeError("Stage-S Column bridge rejects unsupported torch.stack kwargs")
        dim = kwargs["dim"]
    if dim not in (0, -1):
        raise TypeError(f"Stage-S Column bridge rejects torch.stack dim={dim}")
    materialized = list(values)
    if not materialized:
        raise TypeError(
            "Stage-S Column bridge accepts only non-empty scalar real timestamp/index columns"
        )
    tensor_type = getattr(torch_module, "Tensor", ())
    if all(
        isinstance(value, tensor_type) and getattr(value, "ndim", None) == 0
        for value in materialized
    ):
        # torch.stack in the image rejects a datasets Column container even
        # though its transformed elements are already scalar tensors.  A
        # tuple is the only adaptation needed; tensor values stay untouched.
        return original_stack(tuple(materialized), *args, **kwargs)
    if not all(isinstance(value, numbers.Real) and not isinstance(value, bool) for value in materialized):
        raise TypeError(
            "Stage-S Column bridge accepts only non-empty scalar real timestamp/index columns"
        )
    # ``as_tensor`` preserves the scalar values and produces the same 1-D
    # tensor shape that stacking scalar tensors would produce.  No dataset
    # object or parquet file is changed.
    return torch_module.as_tensor(materialized)


@contextlib.contextmanager
def _temporary_column_stack(torch_module: Any, column_type: type[Any]) -> Iterator[None]:
    original_stack = torch_module.stack

    @functools.wraps(original_stack)
    def stack(values: Any, *args: Any, **kwargs: Any) -> Any:
        return _stack_scalar_column(original_stack, torch_module, column_type, values, *args, **kwargs)

    torch_module.stack = stack
    try:
        yield
    finally:
        torch_module.stack = original_stack


def install_column_compat_bridge() -> dict[str, Any]:
    """Install the exact datasets-4.8.4 bridge on LeRobot's constructor."""

    contract = runtime_dependency_contract()
    if not contract["valid"]:
        raise RuntimeError("Stage-S dependency contract refused: " + "; ".join(contract["errors"]))
    if contract["mode"] == "native-pinned-datasets":
        return contract

    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset  # type: ignore[import-not-found]
    import torch

    dataset_class = lerobot_dataset.LeRobotDataset
    if getattr(dataset_class, "_r142_stage_s_column_bridge", False):
        contract["bridge_installed"] = True
        return contract
    source = inspect.getsource(dataset_class.__init__)
    required_anchors = (
        'torch.stack(self.hf_dataset["timestamp"])',
        'torch.stack(self.hf_dataset["episode_index"])',
    )
    missing = [anchor for anchor in required_anchors if anchor not in source]
    if missing:
        raise RuntimeError(
            "Stage-S Column bridge refuses an unknown LeRobot constructor ABI: " + ", ".join(missing)
        )
    column_type = _datasets_column_type()
    if column_type is None:
        raise RuntimeError("Stage-S Column bridge cannot resolve datasets.arrow_dataset.Column")
    original_init = dataset_class.__init__

    @functools.wraps(original_init)
    def compatible_init(self: Any, *args: Any, **kwargs: Any) -> Any:
        # LeRobot constructs synchronously in each torchrun rank.  Keep the
        # patch scoped to this constructor and restore torch.stack even on an
        # exception; no global behavior remains after dataset construction.
        with _PATCH_LOCK, _temporary_column_stack(torch, column_type):
            return original_init(self, *args, **kwargs)

    compatible_init._r142_stage_s_column_bridge = True  # type: ignore[attr-defined]
    dataset_class.__init__ = compatible_init
    contract["bridge_installed"] = True
    return contract


def smoke_test_lerobot_dataset(dataset_root: str | Path, revision: str, *, episode_index: int = 0) -> dict[str, Any]:
    """Construct one real frozen episode after installing the narrow bridge."""

    contract = install_column_compat_bridge()
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # type: ignore[import-not-found]

    root = Path(dataset_root).expanduser().resolve()
    dataset = LeRobotDataset(
        "physical-intelligence/libero",
        root=root,
        revision=revision,
        episodes=[episode_index],
        download_videos=False,
    )
    length = len(dataset)
    if length <= 0:
        raise RuntimeError("Stage-S LeRobot compatibility smoke produced an empty dataset")
    return {
        "valid": True,
        "episode_index": episode_index,
        "length": length,
        "revision": revision,
        "root": str(root),
        "dependency_contract": contract,
    }


__all__ = [
    "COMPATIBILITY_CONTRACT",
    "PINNED_DATASETS_VERSION",
    "PINNED_LEROBOT_COMMIT",
    "PINNED_LEROBOT_VERSION",
    "SUPPORTED_COLUMN_DATASETS_VERSION",
    "install_column_compat_bridge",
    "runtime_dependency_contract",
    "smoke_test_lerobot_dataset",
]
