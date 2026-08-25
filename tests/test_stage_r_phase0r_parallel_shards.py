from __future__ import annotations

import json
from pathlib import Path


TASKS = [
    f"{suite}_task{task_id:02d}"
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    for task_id in range(10)
]


def test_parallel_shards_are_exact_and_disjoint() -> None:
    contract = json.loads(
        (Path(__file__).parents[1] / "configs/stage_r_phase0r_parallel_shards.json").read_text()
    )
    assert contract["world_size"] == 8
    shards = contract["shards"]
    assert set(shards) == {"A", "B"}

    targets: list[str] = []
    prerequisites: list[str] = []
    for shard in shards.values():
        ranks = shard["global_ranks"]
        shard_targets = shard["target_tasks"]
        assert len(ranks) == len(shard_targets) == 4
        for rank, target in zip(ranks, shard_targets):
            index = TASKS.index(target)
            assert index % contract["world_size"] == rank
            expected_prerequisites = [
                task
                for prior_index, task in enumerate(TASKS[:index])
                if prior_index % contract["world_size"] == rank
            ]
            assert shard["prerequisites"][str(rank)] == expected_prerequisites
        targets.extend(shard_targets)
        prerequisites.extend(task for values in shard["prerequisites"].values() for task in values)

    assert targets == TASKS[32:]
    assert sorted(prerequisites) == sorted(TASKS[:32])
    assert set(targets).isdisjoint(prerequisites)
