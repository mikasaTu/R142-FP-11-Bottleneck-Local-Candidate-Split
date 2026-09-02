#!/usr/bin/env python3
"""Fail-closed local LIBERO and norm-stat preflight for Stage-S C.

This command performs only local reads plus one integrity-preserving write:
the norm-stat staging marker.  It reuses the pre-generated dataset checksum
manifest and never calls a
Hugging Face download API.  The caller must set the offline environment before
this module imports LeRobot/OpenPI.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from r142_stage_s.openpi_c import (  # noqa: E402
    DEFAULT_HF_LEROBOT_HOME,
    DEFAULT_LIBERO_DATASET_ROOT,
    DEFAULT_PI05_ASSETS_BASE_DIR,
    DEFAULT_STAGED_LIBERO_ASSETS_BASE_DIR,
    LIBERO_DATASET_EXPECTED_INFO,
    LIBERO_DATASET_REPO,
    LIBERO_DATASET_REVISION,
    _status_marker,
    audit_libero_dataset_snapshot,
    stage_libero_norm_stats,
)
from r142_stage_s.lerobot_compat import smoke_test_lerobot_dataset  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path(DEFAULT_LIBERO_DATASET_ROOT))
    parser.add_argument("--assets-source-root", type=Path, default=Path(DEFAULT_PI05_ASSETS_BASE_DIR))
    parser.add_argument("--staged-assets-base-dir", type=Path, default=Path(DEFAULT_STAGED_LIBERO_ASSETS_BASE_DIR))
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _assert_offline_environment() -> None:
    expected_home = Path(os.environ.get("HF_LEROBOT_HOME", "")).expanduser().resolve()
    if expected_home != Path(DEFAULT_HF_LEROBOT_HOME).resolve():
        raise RuntimeError(
            "HF_LEROBOT_HOME must be the frozen CPFS path "
            f"{Path(DEFAULT_HF_LEROBOT_HOME).resolve()}, got {expected_home}"
        )
    for name in ("HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name}=1 is required; network fallback is forbidden")


def _load_official_bindings(openpi_root: Path, dataset_root: Path, staged_assets: Path) -> dict[str, Any]:
    """Exercise the pinned LeRobot metadata and OpenPI norm-stat resolver."""

    # Imports happen only after the launcher has installed the frozen offline
    # environment.  This makes the LeRobot module capture the requested home.
    from lerobot.common.constants import HF_LEROBOT_HOME  # type: ignore[import-not-found]
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata  # type: ignore[import-not-found]

    observed_home = Path(HF_LEROBOT_HOME).expanduser().resolve()
    expected_home = Path(os.environ["HF_LEROBOT_HOME"]).expanduser().resolve()
    if observed_home != expected_home:
        raise RuntimeError(f"LeRobot imported the wrong HF_LEROBOT_HOME: {observed_home}")
    expected_root = expected_home / LIBERO_DATASET_REPO
    if dataset_root != expected_root:
        raise RuntimeError(f"dataset root must be {expected_root}, got {dataset_root}")

    metadata = LeRobotDatasetMetadata(
        LIBERO_DATASET_REPO,
        root=dataset_root,
        revision=LIBERO_DATASET_REVISION,
    )
    if metadata.revision != LIBERO_DATASET_REVISION:
        raise RuntimeError(
            f"LeRobot revision mismatch: expected {LIBERO_DATASET_REVISION}, got {metadata.revision}"
        )
    if metadata.total_episodes != int(LIBERO_DATASET_EXPECTED_INFO["total_episodes"]):
        raise RuntimeError(f"LeRobot metadata episode count mismatch: {metadata.total_episodes}")
    if metadata.total_tasks != int(LIBERO_DATASET_EXPECTED_INFO["total_tasks"]):
        raise RuntimeError(f"LeRobot metadata task count mismatch: {metadata.total_tasks}")

    # Exercise the real LeRobot constructor before any training process is
    # launched.  This catches the datasets-4.8 Column ABI mismatch while the
    # bridge remains scoped to the constructor and records the actual package
    # provenance/mechanism in the preflight marker.
    dataset_smoke = smoke_test_lerobot_dataset(dataset_root, LIBERO_DATASET_REVISION, episode_index=0)

    # The official config is loaded from the pinned OpenPI checkout.  Replacing
    # only assets_base_dir leaves the model/data/optimizer scientific choices
    # untouched while testing the actual DataConfigFactory resolver.
    openpi_src = (openpi_root / "src").resolve()
    if str(openpi_src) not in sys.path:
        sys.path.insert(0, str(openpi_src))
    from openpi.training import config as openpi_config  # type: ignore[import-not-found]

    config = openpi_config.get_config("pi05_libero")
    config = dataclasses.replace(config, assets_base_dir=str(staged_assets))
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id != LIBERO_DATASET_REPO:
        raise RuntimeError(f"official pi05_libero repo_id changed: {data_config.repo_id}")
    if data_config.norm_stats is None:
        raise RuntimeError("official pi05_libero config did not load local norm_stats")
    resolved = staged_assets / "pi05_libero" / LIBERO_DATASET_REPO / "norm_stats.json"
    if not resolved.is_file():
        raise RuntimeError(f"official norm-stat resolver path is absent: {resolved}")
    return {
        "config_name": config.name,
        "assets_base_dir": str(staged_assets),
        "assets_dirs": str(config.assets_dirs),
        "resolver_path": str(resolved),
        "repo_id": data_config.repo_id,
        "norm_stats_keys": sorted(data_config.norm_stats),
        "dataset_metadata_root": str(metadata.root),
        "dataset_metadata_revision": metadata.revision,
        "dataset_total_episodes": metadata.total_episodes,
        "dataset_total_tasks": metadata.total_tasks,
        "lerobot_compatibility": dataset_smoke["dependency_contract"],
        "dataset_constructor_smoke": {
            key: dataset_smoke[key]
            for key in ("episode_index", "length", "revision", "root")
        },
    }


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    _assert_offline_environment()
    dataset_root = args.dataset_root.expanduser().resolve()
    staged_assets = args.staged_assets_base_dir.expanduser().resolve()
    snapshot = audit_libero_dataset_snapshot(
        dataset_root,
        persist_manifest=True,
        verify_hashes=True,
    )
    if not snapshot["valid"]:
        raise RuntimeError("LIBERO local snapshot audit failed: " + "; ".join(snapshot["errors"]))
    norm_stats = stage_libero_norm_stats(args.assets_source_root, staged_assets)
    if not norm_stats["valid"]:
        raise RuntimeError("LIBERO norm-stat staging failed: " + "; ".join(norm_stats["errors"]))
    official = _load_official_bindings(args.openpi_root.expanduser().resolve(), dataset_root, staged_assets)
    record = {
        "schema": "r142-stage-s-c-data-preflight-v2",
        "status": "COMPLETED",
        "dataset": snapshot,
        "norm_stats": norm_stats,
        "official_bindings": official,
        "offline_environment": {
            name: os.environ.get(name)
            for name in (
                "HF_LEROBOT_HOME",
                "HF_HOME",
                "HF_DATASETS_CACHE",
                "HUGGINGFACE_HUB_CACHE",
                "HF_DATASETS_OFFLINE",
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
            )
        },
        "no_network_fallback": True,
        "no_pai_submit_performed": True,
    }
    _status_marker(args.output.expanduser().resolve(), record)
    return record


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        print(json.dumps(_execute(args), indent=2, sort_keys=True))
        return 0
    except BaseException as exc:
        print(f"C data preflight refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
