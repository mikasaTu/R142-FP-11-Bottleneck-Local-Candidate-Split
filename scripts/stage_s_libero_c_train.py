#!/usr/bin/env python3
"""Run or resume the pinned OpenPI C under-training lineage.

The foreground process is the official ``scripts/train_pytorch.py`` from
OpenPI commit ``54cbaee6...``.  This wrapper performs source/base/conversion
preflight, enforces the two Beijing blackout windows, writes a durable start
record, and only publishes a terminal C completion marker after the native
1000/3000/6000/10000 checkpoints pass the full-state audit.  It never submits
PAI and never manufactures a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from r142_stage_s.libero import atomic_json  # noqa: E402
from r142_stage_s.openpi_c import (  # noqa: E402
    CONVERSION_COMPLETION_NAME,
    DEFAULT_OPENPI_PYTHON,
    OPENPI_COMMIT,
    OPENPI_CONFIG_NAME,
    TRAINING_START_NAME,
    TRAINING_FAILED_NAME,
    TRAINING_TERMINAL_NAME,
    _status_marker,
    assert_outside_blackout,
    audit_base_download,
    audit_openpi_checkout,
    build_patched_training_command,
    DEFAULT_PI05_ASSETS_BASE_DIR,
    finalize_training,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--base-jax-root", type=Path, required=True)
    parser.add_argument("--base-pytorch-root", type=Path, required=True)
    parser.add_argument("--checkpoint-base-dir", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--assets-base-dir", type=Path, default=Path(DEFAULT_PI05_ASSETS_BASE_DIR))
    parser.add_argument("--python", default=DEFAULT_OPENPI_PYTHON)
    parser.add_argument("--resume", action="store_true")
    return parser


def _has_numeric_checkpoint(path: Path) -> bool:
    return any(child.is_dir() and child.name.isdigit() for child in path.iterdir()) if path.is_dir() else False


def _execute(args: argparse.Namespace) -> int:
    now = assert_outside_blackout()
    source_audit = audit_openpi_checkout(args.openpi_root, python=args.python)
    if not source_audit["ready"]:
        raise SystemExit("OpenPI source audit failed: " + "; ".join(source_audit["errors"]))
    base_audit = audit_base_download(args.base_jax_root)
    if not base_audit["valid"]:
        raise SystemExit("base object audit failed: " + "; ".join(base_audit["errors"]))
    conversion_dir = args.base_pytorch_root.resolve()
    conversion_marker = conversion_dir / CONVERSION_COMPLETION_NAME
    if not conversion_marker.is_file():
        raise SystemExit(f"missing conversion completion marker: {conversion_marker}")
    try:
        conversion = json.loads(conversion_marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid conversion completion marker: {exc}") from exc
    if conversion.get("status") != "COMPLETED" or conversion.get("openpi_commit") != OPENPI_COMMIT:
        raise SystemExit("conversion marker is not a completed artifact from the pinned OpenPI commit")

    checkpoint_base = args.checkpoint_base_dir.resolve()
    train_dir = checkpoint_base / OPENPI_CONFIG_NAME / "r142_stage_s_c_undertrained_seed42"
    if train_dir.exists() and not args.resume and _has_numeric_checkpoint(train_dir):
        raise SystemExit("numeric checkpoints already exist; pass --resume to preserve same-directory lineage")
    command = build_patched_training_command(
        worker_path=Path(__file__).resolve().with_name("stage_s_libero_c_train_worker.py"),
        openpi_root=args.openpi_root,
        base_pytorch_root=args.base_pytorch_root,
        checkpoint_base_dir=checkpoint_base,
        resume=args.resume,
        assets_base_dir=args.assets_base_dir,
        python=args.python,
    )
    args.log_root.mkdir(parents=True, exist_ok=True)
    start = {
        "schema": "r142-stage-s-c-training-start-v1",
        "status": "RUNNING",
        "started_at": now.isoformat(),
        "openpi_commit": OPENPI_COMMIT,
        "config_name": OPENPI_CONFIG_NAME,
        "command": command,
        "resume": bool(args.resume),
        "base_manifest_sha256": base_audit.get("manifest", {}).get("manifest_sha256"),
        "checkpoint_base_dir": str(checkpoint_base),
        "no_pai_submit_performed": True,
    }
    start_path = args.log_root / TRAINING_START_NAME
    if start_path.exists():
        previous = json.loads(start_path.read_text(encoding="utf-8"))
        if previous.get("openpi_commit") != OPENPI_COMMIT or previous.get("config_name") != OPENPI_CONFIG_NAME:
            raise SystemExit("existing training start record belongs to a different source/config")
    else:
        _status_marker(start_path, start)

    environment = dict(os.environ)
    environment.setdefault("WANDB_MODE", "disabled")
    completed = False
    try:
        subprocess.run(command, cwd=args.openpi_root.resolve(), check=True, env=environment)
        completed = True
    finally:
        if completed:
            _status_marker(
                args.log_root / TRAINING_TERMINAL_NAME,
                {
                    "schema": "r142-stage-s-c-training-terminal-v1",
                    "status": "COMPLETED",
                    "openpi_commit": OPENPI_COMMIT,
                    "config_name": OPENPI_CONFIG_NAME,
                    "global_step": 10001,
                    "checkpoint_steps": [1000, 3000, 6000, 10000],
                    "no_pai_submit_performed": True,
                },
            )
            marker = finalize_training(
                checkpoint_base_dir=checkpoint_base,
                log_root=args.log_root,
                base_manifest_sha256=str(base_audit.get("manifest", {}).get("manifest_sha256") or ""),
                openpi_root=args.openpi_root,
            )
            print(json.dumps(marker, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _execute(args)
    except BaseException as exc:
        # Persist a hashable terminal failure record.  If finalization already
        # wrote a detailed failure marker, preserve it rather than replacing
        # its checkpoint-audit evidence with this outer exception summary.
        try:
            args.log_root.mkdir(parents=True, exist_ok=True)
            failure_path = args.log_root / TRAINING_FAILED_NAME
            if not failure_path.exists():
                _status_marker(
                    failure_path,
                    {
                        "schema": "r142-stage-s-c-training-failure-v1",
                        "status": "FAILED",
                        "stage": "training",
                        "openpi_commit": OPENPI_COMMIT,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "resume": bool(args.resume),
                    },
                )
        except OSError:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
