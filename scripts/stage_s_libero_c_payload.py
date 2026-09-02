#!/usr/bin/env python3
"""Render the complete Stage-S C chain and a non-submitting PAI payload."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from r142_stage_s.libero import atomic_json  # noqa: E402
from r142_stage_s.openpi_c import (  # noqa: E402
    DEFAULT_PI05_ASSETS_BASE_DIR,
    build_c_chain_contract,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--base-jax-root", type=Path, required=True)
    parser.add_argument("--base-pytorch-root", type=Path, required=True)
    parser.add_argument("--checkpoint-base-dir", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--assets-base-dir", type=Path, default=Path(DEFAULT_PI05_ASSETS_BASE_DIR))
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    chain = build_c_chain_contract(
        openpi_root=args.openpi_root,
        base_jax_root=args.base_jax_root,
        base_pytorch_root=args.base_pytorch_root,
        checkpoint_base_dir=args.checkpoint_base_dir,
        log_root=args.log_root,
        repo_root=args.repo_root,
        assets_base_dir=args.assets_base_dir,
    )
    atomic_json(args.contract, chain)
    from r142_stage_s.libero import build_pai_stage_s_payload

    payload = build_pai_stage_s_payload(
        run_id=args.run_id,
        output_root=args.checkpoint_base_dir,
        log_root=args.log_root,
        command=chain["training"]["wrapper_command"],
        working_directory=args.repo_root,
        c_contract={
            "training_contract": {
                "retain_and_audit_steps": chain["training"]["checkpoint_steps"],
                "save_interval_steps": chain["training"]["save_interval_steps"],
                "full_reference_step": chain["training"]["full_training_reference_steps"],
            }
        },
    )
    atomic_json(args.payload, payload)
    print(json.dumps({"contract": chain, "payload": payload}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
