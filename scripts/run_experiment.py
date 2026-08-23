#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from r142_bottleneck.benchmark import BenchmarkConfig  # noqa: E402
from r142_bottleneck.experiment import (  # noqa: E402
    ExperimentConfig,
    aggregate_results,
    load_episode_metrics,
    run_experiment,
)


def configs(path: Path) -> tuple[BenchmarkConfig, ExperimentConfig]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    benchmark = dict(payload["benchmark"])
    benchmark["bottleneck_steps"] = tuple(benchmark["bottleneck_steps"])
    benchmark["uniform_split_steps"] = tuple(benchmark["uniform_split_steps"])
    experiment = dict(payload["experiment"])
    experiment.pop("calibration_seeds", None)
    experiment.update(
        {
            "minimum_success_gain": payload["gate"]["minimum_success_gain"],
            "minimum_winning_blocks": payload["gate"]["minimum_winning_blocks"],
            "maximum_mode_discovery_drop": payload["gate"]["maximum_mode_discovery_drop"],
            "maximum_localization_median_error": payload["gate"]["maximum_localization_median_error"],
            "minimum_localization_within_one": payload["gate"]["minimum_localization_within_one"],
            "minimum_valid_collapse_fraction": payload["gate"]["minimum_valid_collapse_fraction"],
        }
    )
    return BenchmarkConfig(**benchmark), ExperimentConfig(**experiment)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "stage1.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--no-genealogy", action="store_true")
    parser.add_argument("--aggregate", nargs="*", type=Path)
    args = parser.parse_args()
    benchmark, experiment = configs(args.config)
    if args.episodes is not None:
        experiment = ExperimentConfig(**{**experiment.__dict__, "evaluation_seeds": args.episodes})
    if args.aggregate is not None:
        records = load_episode_metrics(path / "episode_metrics.jsonl" for path in args.aggregate)
        result = aggregate_results(records, args.output, experiment)
        print(json.dumps(result["gate"], indent=2, sort_keys=True))
        return
    indices = [
        index
        for index in range(experiment.evaluation_seeds)
        if index % args.num_shards == args.shard_id
    ]
    run_experiment(
        args.output,
        benchmark,
        experiment,
        indices,
        save_genealogy=not args.no_genealogy,
    )
    print(json.dumps({"complete": True, "episodes": len(indices), "output": str(args.output)}))


if __name__ == "__main__":
    main()
