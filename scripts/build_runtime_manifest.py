#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = [
    "configs/stage1.json",
    "scripts/pai_entry.py",
    "scripts/render_artifacts.py",
    "scripts/run_experiment.py",
    "scripts/verify_runtime_manifest.py",
    "src/r142_bottleneck/__init__.py",
    "src/r142_bottleneck/benchmark.py",
    "src/r142_bottleneck/detector.py",
    "src/r142_bottleneck/experiment.py",
    "src/r142_bottleneck/genealogy.py",
    "src/r142_bottleneck/methods.py",
    "src/r142_bottleneck/metrics.py",
    "src/r142_bottleneck/visualize.py",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    payload = {"schema_version": 1, "files": {relative: digest(ROOT / relative) for relative in INCLUDE}}
    destination = ROOT / "evidence" / "runtime_manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
