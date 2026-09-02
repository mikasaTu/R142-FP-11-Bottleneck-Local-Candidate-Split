from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.stage_s_controls import (
    ControlEvidenceError,
    _validate_null_arrays,
    load_positive,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def test_positive_loader_uses_only_b0_exact_rates_and_detects_all_fail(tmp_path: Path) -> None:
    source = tmp_path / "shards" / "shard-00" / "episode_metrics.jsonl"
    _write_jsonl(
        source,
        [
            {"policy": "other", "episode_id": 1, "candidate_count": 32},
            {
                "policy": "B0_best_of_n",
                "episode_id": 10,
                "candidate_count": 32,
                "candidate_success_rate": 0.0,
            },
            {
                "policy": "B0_best_of_n",
                "episode_id": 11,
                "candidate_count": 32,
                "candidate_success_rate": 1.0 / 32.0,
            },
        ],
    )
    loaded = load_positive(tmp_path / "shards")
    assert loaded["counts"]["source_file_count"] == 1
    assert loaded["counts"]["b0_record_count"] == 2
    assert loaded["counts"]["logical_family_count"] == 2
    assert loaded["counts"]["logical_candidate_row_count"] == 64
    assert loaded["counts"]["all_fail_family_count"] == 1
    assert loaded["counts"]["near_all_fail_family_count"] == 2
    assert sum(bool(row["success"]) for row in loaded["rows"]) == 1
    assert all("poses" not in row and "actions" not in row for row in loaded["rows"])


def test_positive_loader_rejects_non_integral_rate(tmp_path: Path) -> None:
    source = tmp_path / "shards" / "shard-00" / "episode_metrics.jsonl"
    _write_jsonl(
        source,
        [
            {
                "policy": "B0_best_of_n",
                "episode_id": 10,
                "candidate_count": 32,
                "candidate_success_rate": 0.1,
            }
        ],
    )
    with pytest.raises(ControlEvidenceError, match="exact"):
        load_positive(tmp_path / "shards")


def test_null_validation_requires_one_row_for_each_candidate_id(tmp_path: Path) -> None:
    path = tmp_path / "task.npz"
    init_state = np.repeat(np.arange(16, dtype=np.int64), 32)
    candidate_id = np.tile(np.arange(32, dtype=np.int64), 16)
    success = np.zeros(512, dtype=np.uint8)
    rows, detail = _validate_null_arrays(path, success, init_state, candidate_id)
    assert len(rows) == 512
    assert detail["family_count"] == 16
    assert detail["candidate_count_per_family"] == 32
    candidate_id[0] = candidate_id[1]
    with pytest.raises(ControlEvidenceError, match="candidate IDs"):
        _validate_null_arrays(path, success, init_state, candidate_id)

