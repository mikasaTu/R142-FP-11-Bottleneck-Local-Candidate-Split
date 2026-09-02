"""Fail-closed compatibility bridge for the pinned OpenPI/LeRobot ABI.

The frozen OpenPI lock resolves LeRobot commit ``0cf8648`` with
``datasets==3.6.0``.  The dev14 training image contains that LeRobot source
but ``datasets==4.8.4``.  In the latter API, ``Dataset["timestamp"]`` is a
``datasets.arrow_dataset.Column`` of Python scalars; the pinned LeRobot
constructor still passes it to ``torch.stack``, which only accepts tensors.

The bridge is deliberately narrow: it is enabled only for the exact observed
package provenance and only materializes a real datasets ``Column`` through
the same per-item ``torch.tensor`` conversion used by LeRobot's pinned
``hf_transform_to_torch``. It does not rewrite the dataset, touch parquet
bytes, or alter model/training values. Any unknown package/version/source,
constructor/query ABI, or non-tensorizable column fails closed.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import importlib.metadata
import inspect
import json
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
    f"datasets=={SUPPORTED_COLUMN_DATASETS_VERSION} numeric-column bridge"
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
        mode = "datasets-column-numeric-bridge"
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
    """Adapt only tensorizable numeric datasets Columns at pinned stack sites."""

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
    source = getattr(values, "source", None)
    column_name = getattr(values, "column_name", None)
    if callable(getattr(source, "with_format", None)) and isinstance(column_name, str):
        # With datasets 4.8.4, iterating a Column backed by LeRobot's custom
        # transform evaluates that transform for every complete row.  On the
        # full LIBERO snapshot that needlessly decodes image fields hundreds
        # of thousands of times before training starts.  Read the exact same
        # scalar Arrow column through a transform-free Dataset view instead;
        # this changes only materialization cost, never the stored values.
        raw_column = source.with_format(None)[column_name]
        materialized = list(raw_column)
    else:
        materialized = list(values)
    if not materialized:
        raise TypeError("Stage-S Column bridge accepts only non-empty numeric columns")
    tensor_type = getattr(torch_module, "Tensor", ())
    converted = []
    for value in materialized:
        if isinstance(value, tensor_type):
            converted.append(value)
            continue
        if isinstance(value, (str, bytes)) or value is None:
            raise TypeError("Stage-S Column bridge rejects string/bytes/None columns")
        try:
            tensor = torch_module.tensor(value)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise TypeError(
                "Stage-S Column bridge accepts only tensorizable numeric columns"
            ) from exc
        converted.append(tensor)
    shapes = {tuple(getattr(value, "shape", ())) for value in converted}
    if len(shapes) != 1:
        raise TypeError("Stage-S Column bridge rejects ragged numeric columns")
    # This exactly mirrors the pinned LeRobot transform: torch.tensor(item)
    # for each non-string item, followed by the original torch.stack call.
    return original_stack(tuple(converted), *args, **kwargs)


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
    """Install the exact datasets-4.8.4 bridge on pinned stack call sites."""

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
    method_anchors = {
        "__init__": (
            'torch.stack(self.hf_dataset["timestamp"])',
            'torch.stack(self.hf_dataset["episode_index"])',
        ),
        "_get_query_timestamps": ("torch.stack(timestamps)",),
        "_query_hf_dataset": ("torch.stack(self.hf_dataset.select(q_idx)[key])",),
    }
    for method_name, anchors in method_anchors.items():
        method = getattr(dataset_class, method_name, None)
        if method is None:
            raise RuntimeError(f"Stage-S Column bridge cannot resolve LeRobot method {method_name}")
        source = inspect.getsource(method)
        missing = [anchor for anchor in anchors if anchor not in source]
        if missing:
            raise RuntimeError(
                "Stage-S Column bridge refuses an unknown LeRobot "
                f"{method_name} ABI: " + ", ".join(missing)
            )
    column_type = _datasets_column_type()
    if column_type is None:
        raise RuntimeError("Stage-S Column bridge cannot resolve datasets.arrow_dataset.Column")
    for method_name in method_anchors:
        original = getattr(dataset_class, method_name)

        @functools.wraps(original)
        def compatible(self: Any, *args: Any, __original: Any = original, **kwargs: Any) -> Any:
            # Each pinned method is synchronous in one torchrun rank. Keep the
            # global torch function patched only for that method call and
            # restore it on every success/failure path.
            with _PATCH_LOCK, _temporary_column_stack(torch, column_type):
                return __original(self, *args, **kwargs)

        compatible._r142_stage_s_column_bridge = True  # type: ignore[attr-defined]
        setattr(dataset_class, method_name, compatible)
    contract["bridge_installed"] = True
    contract["patched_methods"] = sorted(method_anchors)
    return contract


def smoke_test_lerobot_dataset(dataset_root: str | Path, revision: str, *, episode_index: int = 0) -> dict[str, Any]:
    """Construct and query one real frozen episode with byte preservation."""

    contract = install_column_compat_bridge()
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # type: ignore[import-not-found]

    root = Path(dataset_root).expanduser().resolve()
    parquet_paths = sorted((root / "data").rglob("*.parquet"))
    if not parquet_paths:
        raise RuntimeError("Stage-S LeRobot compatibility smoke found no parquet input")
    representative = parquet_paths[0]
    parquet_sha_before = hashlib.sha256(representative.read_bytes()).hexdigest()
    dataset = LeRobotDataset(
        "physical-intelligence/libero",
        root=root,
        revision=revision,
        episodes=[episode_index],
        delta_timestamps={"actions": [0.0]},
        download_videos=False,
    )
    length = len(dataset)
    if length <= 0:
        raise RuntimeError("Stage-S LeRobot compatibility smoke produced an empty dataset")
    sample = dataset[0]
    action = sample.get("actions")
    if action is None or tuple(getattr(action, "shape", ())) == ():
        raise RuntimeError("Stage-S LeRobot compatibility smoke did not query an action sequence")
    parquet_sha_after = hashlib.sha256(representative.read_bytes()).hexdigest()
    if parquet_sha_after != parquet_sha_before:
        raise RuntimeError("Stage-S LeRobot compatibility smoke changed parquet bytes")
    return {
        "valid": True,
        "episode_index": episode_index,
        "length": length,
        "revision": revision,
        "root": str(root),
        "sample_keys": sorted(sample),
        "action_shape": list(action.shape),
        "representative_parquet": str(representative),
        "representative_parquet_sha256": parquet_sha_after,
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
