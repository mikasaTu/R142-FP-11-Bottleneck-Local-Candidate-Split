#!/usr/bin/env python3
"""Read-only Stage-S runtime/config/deployment identity preflight.

No output file is created unless ``--output`` is explicitly supplied.  A
non-zero exit means the attestation is ``REFUSED``; an explicit output still
receives the complete deterministic failure record for audit purposes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from r142_stage_s.runtime_identity import (  # noqa: E402
    IdentityRefusal,
    attest_config,
    write_attestation,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="explicit JSON attestation path; no file is written when omitted",
    )
    args = parser.parse_args(argv)

    if args.output is None:
        print(
            "RUNTIME_IDENTITY_REFUSED: --output is required to emit an attestation; no files written",
            file=sys.stderr,
        )
        return 2
    try:
        result = attest_config(args.config)
        write_attestation(result, args.output)
    except IdentityRefusal as exc:
        print(f"RUNTIME_IDENTITY_REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
