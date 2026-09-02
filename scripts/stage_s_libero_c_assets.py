#!/usr/bin/env python3
"""Materialize and audit the exact public pi05 base asset set.

This command has no PAI integration.  ``manifest`` performs a read-only live
GCS listing check; ``download`` resumes only per-object partial files and
publishes ``BASE_DOWNLOAD_COMPLETED.json`` after all SHA-256 checks pass;
``convert`` invokes the pinned OpenPI converter and records provenance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_src() -> Path:
    return Path(__file__).resolve().parents[1] / "src"


sys.path.insert(0, str(_repo_src()))

from r142_stage_s.openpi_c import (  # noqa: E402
    audit_base_download,
    build_conversion_contract,
    download_base_checkpoint,
    expected_base_manifest,
    fetch_gcs_manifest,
    run_conversion,
    write_expected_base_manifest,
)
from r142_stage_s.libero import atomic_json  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)

    manifest = sub.add_parser("manifest", help="write the checked-in contract or verify live GCS metadata")
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--live", action="store_true", help="query the public GCS JSON API before writing")

    download = sub.add_parser("download", help="resume exact object downloads and publish completion markers")
    download.add_argument("--output-root", type=Path, required=True)
    download.add_argument("--manifest", type=Path)

    convert = sub.add_parser("convert", help="invoke the pinned JAX-to-PyTorch converter")
    convert.add_argument("--openpi-root", type=Path, required=True)
    convert.add_argument("--base-jax-root", type=Path, required=True)
    convert.add_argument("--base-pytorch-root", type=Path, required=True)
    convert.add_argument("--python", default="python")
    convert.add_argument("--precision", choices=("bfloat16", "float32"), default="bfloat16")

    audit = sub.add_parser("audit", help="verify an already downloaded base root")
    audit.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.operation == "manifest":
        payload = fetch_gcs_manifest() if args.live else write_expected_base_manifest(args.output)
        if args.live:
            atomic_json(args.output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.operation == "download":
        payload = download_base_checkpoint(args.output_root, args.manifest)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.operation == "convert":
        contract = build_conversion_contract(
            openpi_root=args.openpi_root,
            base_jax_root=args.base_jax_root,
            base_pytorch_root=args.base_pytorch_root,
            python=args.python,
            precision=args.precision,
        )
        result = run_conversion(contract)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.operation == "audit":
        result = audit_base_download(args.output_root)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["valid"] else 2
    raise AssertionError(args.operation)


if __name__ == "__main__":
    raise SystemExit(main())
