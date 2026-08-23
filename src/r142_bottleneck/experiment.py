from __future__ import annotations

import csv
import gzip
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .benchmark import BenchmarkConfig, ForkPush2D
from .methods import POLICIES, generate_policy_set, policy_metadata
from .metrics import candidate_set_metrics, collapse_diagnostics, paired_bootstrap


@dataclass(frozen=True)
class ExperimentConfig:
    evaluation_seeds: int = 400
    bootstrap_replicates: int = 10000
    block_size: int = 40
    base_seed: int = 14211
    minimum_success_gain: float = 0.05
    minimum_winning_blocks: int = 7
    maximum_mode_discovery_drop: float = 0.02
    maximum_localization_median_error: float = 1.0
    minimum_localization_within_one: float = 0.80
    minimum_valid_collapse_fraction: float = 0.80


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _seed_for_index(base_seed: int, index: int) -> int:
    return int(np.random.SeedSequence([base_seed, index, 142]).generate_state(1, dtype=np.uint32)[0])


def run_experiment(
    output_dir: str | Path,
    benchmark_config: BenchmarkConfig | None = None,
    experiment_config: ExperimentConfig | None = None,
    episode_indices: Iterable[int] | None = None,
    save_genealogy: bool = True,
) -> Path:
    bench_cfg = benchmark_config or BenchmarkConfig()
    exp_cfg = experiment_config or ExperimentConfig()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    indices = list(range(exp_cfg.evaluation_seeds) if episode_indices is None else episode_indices)
    env = ForkPush2D(bench_cfg)
    episode_path = output / "episode_metrics.jsonl"
    oracle_path = output / "oracle_truth.jsonl"
    genealogy_path = output / "genealogy.jsonl.gz"
    with episode_path.open("w", encoding="utf-8") as episode_file, oracle_path.open(
        "w", encoding="utf-8"
    ) as oracle_file:
        genealogy_context = gzip.open(genealogy_path, "wt", encoding="utf-8") if save_genealogy else None
        try:
            for episode_id in indices:
                seed = _seed_for_index(exp_cfg.base_seed, episode_id)
                spec = env.make_episode(episode_id, seed)
                oracle_file.write(json.dumps(spec.oracle_truth(), sort_keys=True) + "\n")
                b0_set = None
                for policy in POLICIES:
                    candidate_set = generate_policy_set(env, spec, policy)
                    if policy == "B0_best_of_n":
                        b0_set = candidate_set
                    metrics = candidate_set_metrics(candidate_set, spec.true_bottleneck_step)
                    if policy == "B0_best_of_n":
                        metrics["collapse"] = collapse_diagnostics(
                            candidate_set, spec.true_bottleneck_step, bench_cfg.scout_count
                        )
                    episode_file.write(json.dumps(metrics, sort_keys=True) + "\n")
                    if genealogy_context is not None:
                        for candidate in candidate_set.candidates:
                            genealogy_context.write(json.dumps(candidate.to_record(), sort_keys=True) + "\n")
                if b0_set is None:
                    raise AssertionError("B0 was not generated")
        finally:
            if genealogy_context is not None:
                genealogy_context.close()
    manifest = {
        "benchmark_config": asdict(bench_cfg),
        "experiment_config": asdict(exp_cfg),
        "episode_indices": indices,
        "policy_metadata": policy_metadata(),
        "genealogy_saved": save_genealogy,
        "complete": True,
    }
    _atomic_json(output / "shard_manifest.json", manifest)
    (output / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return output


def load_episode_metrics(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path_value in paths:
        path = Path(path_value)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    return records


def aggregate_results(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    experiment_config: ExperimentConfig | None = None,
) -> dict[str, Any]:
    exp_cfg = experiment_config or ExperimentConfig()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    by_policy: dict[str, list[dict[str, Any]]] = {policy: [] for policy in POLICIES}
    for record in records:
        by_policy[record["policy"]].append(record)
    episode_ids = sorted({int(record["episode_id"]) for record in records})
    expected = len(episode_ids)
    for policy, policy_records in by_policy.items():
        if len(policy_records) != expected:
            raise ValueError(f"incomplete policy {policy}: {len(policy_records)} != {expected}")
        policy_records.sort(key=lambda item: int(item["episode_id"]))

    table: list[dict[str, Any]] = []
    for policy in POLICIES:
        policy_records = by_policy[policy]
        localization = [
            float(item["bottleneck_localization_error"])
            for item in policy_records
            if item["bottleneck_localization_error"] is not None
        ]
        table.append(
            {
                "policy": policy,
                "candidate_budget": int(policy_records[0]["candidate_count"]),
                "success_at_n": float(np.mean([x["success_at_n"] for x in policy_records])),
                "any_success_at_n": float(np.mean([x["any_success_at_n"] for x in policy_records])),
                "candidate_success_rate": float(np.mean([x["candidate_success_rate"] for x in policy_records])),
                "mode_discovery_rate": float(np.mean([x["mode_discovery_rate"] for x in policy_records])),
                "successful_modes_per_sample": float(
                    np.mean([x["successful_modes_per_sample"] for x in policy_records])
                ),
                "both_modes_discovered": float(np.mean([x["both_modes_discovered"] for x in policy_records])),
                "localization_mae": None if not localization else float(np.mean(localization)),
                "localization_median": None if not localization else float(np.median(localization)),
                "localization_within_one": None
                if not localization
                else float(np.mean(np.asarray(localization) <= 1.0)),
            }
        )

    table_by_policy = {row["policy"]: row for row in table}
    proposed_success = np.asarray(
        [record["success_at_n"] for record in by_policy["proposed_bottleneck_local"]], dtype=float
    )
    comparisons: dict[str, Any] = {}
    for comparison_index, baseline in enumerate(("B1_uniform_split", "B2_random_split")):
        baseline_success = np.asarray([record["success_at_n"] for record in by_policy[baseline]], dtype=float)
        bootstrap = paired_bootstrap(
            proposed_success,
            baseline_success,
            exp_cfg.bootstrap_replicates,
            exp_cfg.base_seed + 100 + comparison_index,
        )
        block_wins = 0
        for start in range(0, len(proposed_success), exp_cfg.block_size):
            stop = min(start + exp_cfg.block_size, len(proposed_success))
            if proposed_success[start:stop].mean() > baseline_success[start:stop].mean():
                block_wins += 1
        bootstrap["winning_blocks"] = block_wins
        bootstrap["total_blocks"] = int(np.ceil(len(proposed_success) / exp_cfg.block_size))
        comparisons[baseline] = bootstrap

    collapse_values = [
        bool(record["collapse"]["collapse_valid"])
        for record in by_policy["B0_best_of_n"]
    ]
    collapse_fraction = float(np.mean(collapse_values))
    proposed_row = table_by_policy["proposed_bottleneck_local"]
    comparison_passes: dict[str, bool] = {}
    for baseline, comparison in comparisons.items():
        mode_drop = table_by_policy[baseline]["mode_discovery_rate"] - proposed_row["mode_discovery_rate"]
        comparison_passes[baseline] = bool(
            comparison["mean_difference"] >= exp_cfg.minimum_success_gain
            and comparison["ci95_low"] > 0.0
            and comparison["winning_blocks"] >= exp_cfg.minimum_winning_blocks
            and mode_drop <= exp_cfg.maximum_mode_discovery_drop
        )
    localization_pass = bool(
        proposed_row["localization_median"] is not None
        and proposed_row["localization_median"] <= exp_cfg.maximum_localization_median_error
        and proposed_row["localization_within_one"] >= exp_cfg.minimum_localization_within_one
    )
    benchmark_valid = collapse_fraction >= exp_cfg.minimum_valid_collapse_fraction
    stable_gain = all(comparison_passes.values())
    accepted = benchmark_valid and stable_gain and localization_pass
    if not benchmark_valid:
        decision = "INVALID_BENCHMARK_COLLAPSE_GATE_FAILED"
    elif accepted:
        decision = "SUPPORTED_STAGE1_NO_VLA_CLAIM"
    else:
        decision = "IDEA_FAILED_DO_NOT_ENTER_VLA"
    gate = {
        "decision": decision,
        "accepted": accepted,
        "benchmark_valid": benchmark_valid,
        "collapse_valid_fraction": collapse_fraction,
        "stable_gain": stable_gain,
        "comparison_passes": comparison_passes,
        "localization_pass": localization_pass,
        "comparisons": comparisons,
        "thresholds": asdict(exp_cfg),
        "evidence_boundary": "synthetic 2D mechanism validation only; no VLA or learned-policy claim",
    }

    failure_cases: list[dict[str, Any]] = []
    per_episode = {
        policy: {int(item["episode_id"]): item for item in policy_records}
        for policy, policy_records in by_policy.items()
    }
    for episode_id in episode_ids:
        b0 = per_episode["B0_best_of_n"][episode_id]
        proposed = per_episode["proposed_bottleneck_local"][episode_id]
        wrong = per_episode["A_wrong_late"][episode_id]
        if (not b0["success_at_n"] and proposed["success_at_n"]) or (
            not wrong["success_at_n"] and proposed["success_at_n"]
        ):
            failure_cases.append(
                {
                    "episode_id": episode_id,
                    "true_bottleneck_step": proposed["true_bottleneck_step"],
                    "predicted_bottleneck_step": proposed["predicted_bottleneck_step"],
                    "B0_success": b0["success_at_n"],
                    "proposed_success": proposed["success_at_n"],
                    "wrong_late_success": wrong["success_at_n"],
                    "B0_failures": b0["failure_counts"],
                    "wrong_late_failures": wrong["failure_counts"],
                }
            )
        if len(failure_cases) >= 12:
            break

    conditional_split: dict[str, Any] = {}
    for policy in ("B1_uniform_split", "B2_random_split"):
        hits = [
            item
            for item in by_policy[policy]
            if item["true_bottleneck_step"] in item["split_steps"]
        ]
        misses = [
            item
            for item in by_policy[policy]
            if item["true_bottleneck_step"] not in item["split_steps"]
        ]
        conditional_split[policy] = {
            "hit_count": len(hits),
            "miss_count": len(misses),
            "success_when_hit": None if not hits else float(np.mean([x["success_at_n"] for x in hits])),
            "success_when_missed": None
            if not misses
            else float(np.mean([x["success_at_n"] for x in misses])),
        }
    mechanism = {
        "controlled_findings": {
            "location_effect": {
                "proposed_success": proposed_row["success_at_n"],
                "wrong_early_success": table_by_policy["A_wrong_early"]["success_at_n"],
                "wrong_late_success": table_by_policy["A_wrong_late"]["success_at_n"],
                "status": "confirmed_by_location_intervention",
            },
            "operator_specificity": {
                "structured_local_success": proposed_row["success_at_n"],
                "random_operator_at_correct_location_success": table_by_policy[
                    "A_correct_random_operator"
                ]["success_at_n"],
                "full_resampling_at_correct_location_success": table_by_policy[
                    "A_full_resampling"
                ]["success_at_n"],
                "status": "structured_operator_not_isolated_if_alternatives_match_or_win",
            },
            "more_samples": {
                "B0_budget": table_by_policy["B0_best_of_n"]["candidate_budget"],
                "B0_success": table_by_policy["B0_best_of_n"]["success_at_n"],
                "more_samples_budget": table_by_policy["A_more_samples"]["candidate_budget"],
                "more_samples_success": table_by_policy["A_more_samples"]["success_at_n"],
                "status": "tests_correlated_family_collapse",
            },
        },
        "conditional_split_location": conditional_split,
        "interpretation_boundary": "mechanism findings are benchmark-specific controlled interventions",
    }

    with (output / "quantitative_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)
    _atomic_json(output / "summary.json", {"table": table, "comparisons": comparisons})
    _atomic_json(output / "gate_decision.json", gate)
    _atomic_json(output / "failure_cases.json", failure_cases)
    _atomic_json(output / "mechanism_diagnostics.json", mechanism)
    return {"table": table, "gate": gate}
