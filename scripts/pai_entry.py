#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from r142_bottleneck.experiment import ExperimentConfig, aggregate_results, load_episode_metrics  # noqa: E402
from r142_bottleneck.visualize import render_example, render_quantitative_table  # noqa: E402


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_shard(
    shard_id: int,
    workers: int,
    episodes: int,
    output_root: Path,
    python_bin: str,
    config: Path,
) -> Path:
    final = output_root / "shards" / f"shard-{shard_id:02d}"
    if (final / "COMPLETE").is_file():
        return final
    attempt = output_root / "shards" / f".shard-{shard_id:02d}.attempt-{os.getpid()}"
    attempt.mkdir(parents=True, exist_ok=False)
    command = [
        python_bin,
        str(ROOT / "scripts" / "run_experiment.py"),
        "--config",
        str(config),
        "--output",
        str(attempt),
        "--episodes",
        str(episodes),
        "--shard-id",
        str(shard_id),
        "--num-shards",
        str(workers),
    ]
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    (attempt / "subprocess.stdout").write_text(completed.stdout, encoding="utf-8")
    (attempt / "subprocess.stderr").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0 or not (attempt / "COMPLETE").is_file():
        raise RuntimeError(f"shard {shard_id} failed with exit {completed.returncode}: {completed.stderr[-1000:]}")
    try:
        os.replace(attempt, final)
    except OSError:
        if not (final / "COMPLETE").is_file():
            raise
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "stage1.json")
    parser.add_argument("--skip-gpu-check", action="store_true")
    parser.add_argument("--allow-local-uid", action="store_true")
    args = parser.parse_args()
    if args.allow_local_uid and not args.skip_gpu_check:
        raise SystemExit("--allow-local-uid is restricted to --skip-gpu-check smoke")
    if not args.allow_local_uid and (os.getuid() != 2254 or os.getgid() != 2254):
        raise SystemExit(f"expected runtime 2254:2254, got {os.getuid()}:{os.getgid()}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    owner = args.output_root.stat()
    expected_owner = (os.getuid(), os.getgid()) if args.allow_local_uid else (2254, 2254)
    if (owner.st_uid, owner.st_gid) != expected_owner:
        raise SystemExit(
            f"output owner is {owner.st_uid}:{owner.st_gid}, expected {expected_owner[0]}:{expected_owner[1]}"
        )
    inventory_path = args.output_root / "gpu_inventory.txt"
    if args.skip_gpu_check:
        inventory = "SKIPPED_LOCAL_CPU_SMOKE\n"
    else:
        completed = subprocess.run(["nvidia-smi", "-L"], check=True, text=True, capture_output=True)
        inventory = completed.stdout
        if len([line for line in inventory.splitlines() if line.startswith("GPU ")]) != 8:
            raise SystemExit("formal evaluation requires exactly 8 visible GPUs")
    inventory_path.write_text(inventory, encoding="utf-8")

    first_work = args.output_root / "results" / "FIRST_WORK.json"
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_shard,
                shard_id,
                args.workers,
                args.episodes,
                args.output_root,
                args.python_bin,
                args.config,
            ): shard_id
            for shard_id in range(args.workers)
        }
        shard_paths: list[Path] = []
        for future in as_completed(futures):
            shard = future.result()
            shard_paths.append(shard)
            if not first_work.exists():
                metrics = shard / "episode_metrics.jsonl"
                atomic_json(
                    first_work,
                    {
                        "run_id": args.run_id,
                        "workload_type": "evaluation",
                        "completed_shard": shard.name,
                        "metrics_sha256": sha256(metrics),
                        "timestamp_unix": time.time(),
                        "uid": os.getuid(),
                        "gid": os.getgid(),
                        "persisted_completed_evaluation_result": True,
                    },
                )

    shard_paths = sorted(args.output_root.glob("shards/shard-*"))
    if len(shard_paths) != args.workers or not all((path / "COMPLETE").is_file() for path in shard_paths):
        raise SystemExit("not all formal shards are complete")
    records = load_episode_metrics(path / "episode_metrics.jsonl" for path in shard_paths)
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    exp_data = dict(payload["experiment"])
    exp_data.pop("calibration_seeds", None)
    exp_data["evaluation_seeds"] = args.episodes
    exp_data.update(
        {
            "minimum_success_gain": payload["gate"]["minimum_success_gain"],
            "minimum_winning_blocks": payload["gate"]["minimum_winning_blocks"],
            "maximum_mode_discovery_drop": payload["gate"]["maximum_mode_discovery_drop"],
            "maximum_localization_median_error": payload["gate"]["maximum_localization_median_error"],
            "minimum_localization_within_one": payload["gate"]["minimum_localization_within_one"],
            "minimum_valid_collapse_fraction": payload["gate"]["minimum_valid_collapse_fraction"],
        }
    )
    results = args.output_root / "results"
    aggregate = aggregate_results(records, results, ExperimentConfig(**exp_data))
    figures = args.output_root / "figures"
    render_example(figures)
    render_quantitative_table(results / "quantitative_table.csv", figures / "quantitative_results.png")
    completed_files = [
        results / "gate_decision.json",
        results / "quantitative_table.csv",
        results / "mechanism_diagnostics.json",
        results / "failure_cases.json",
    ]
    atomic_json(
        results / "COMPLETED_EVALUATION.json",
        {
            "run_id": args.run_id,
            "workload_type": "evaluation",
            "completed": True,
            "episode_count": args.episodes,
            "shard_count": args.workers,
            "policy_count": 10,
            "decision": aggregate["gate"]["decision"],
            "accepted": aggregate["gate"]["accepted"],
            "artifacts": {str(path.relative_to(args.output_root)): sha256(path) for path in completed_files},
            "uid": os.getuid(),
            "gid": os.getgid(),
            "pai_probe_created": False,
            "timestamp_unix": time.time(),
        },
    )
    print(json.dumps(aggregate["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
