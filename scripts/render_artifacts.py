#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from r142_bottleneck.visualize import render_example, render_quantitative_table  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render_example(args.output)
    render_quantitative_table(
        args.results / "quantitative_table.csv", args.output / "quantitative_results.png"
    )


if __name__ == "__main__":
    main()
