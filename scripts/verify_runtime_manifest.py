#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    failures: list[str] = []
    for relative, expected in payload["files"].items():
        path = ROOT / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "MISSING"
        if actual != expected:
            failures.append(f"{relative}: {actual} != {expected}")
    if failures:
        raise SystemExit("runtime manifest mismatch:\n" + "\n".join(failures))
    print(f"RUNTIME_MANIFEST_OK files={len(payload['files'])}")


if __name__ == "__main__":
    main()
