from __future__ import annotations

import json
from pathlib import Path

import pytest

from r142_stage_s.libero import (
    CALIBRATION_CANDIDATE_COUNT,
    CALIBRATION_INITIAL_STATES,
    CALIBRATION_PROGRESS_FILE,
    CALIBRATION_PROGRESS_SHA_FILE,
    CALIBRATION_SEED,
    CALIBRATION_TASK_IDS,
    StageSError,
    atomic_json,
    run_calibration_shard,
    sha256_file,
)


SETTINGS = ["s0", "s1", "s2", "s3"]


def _evaluator(log: list[tuple[int, int, int, int, int]], *, fail_after: int | None = None):
    def evaluate(
        setting_index: int,
        setting: str,
        task_id: int,
        init_state: int,
        candidate_id: int,
        trial_seed: int,
    ) -> bool:
        del setting
        if fail_after is not None and len(log) >= int(fail_after):
            raise RuntimeError("simulated spot interruption")
        log.append((setting_index, task_id, init_state, candidate_id, trial_seed))
        return trial_seed % 11 in (0, 3)

    return evaluate


def _run(root: Path, evaluator) -> dict:
    return run_calibration_shard(
        evaluator,
        SETTINGS,
        root,
        calibration_seed=CALIBRATION_SEED,
        world_size=2,
        rank=0,
        substrate="B",
        sources=["variant0", "variant1", "variant2", "variant3"],
    )


def test_interruption_resume_is_byte_identical_and_evaluates_only_remaining_trials(tmp_path: Path) -> None:
    interrupted_root = tmp_path / "interrupted"
    interrupted_calls: list[tuple[int, int, int, int, int]] = []
    with pytest.raises(RuntimeError, match="spot interruption"):
        _run(interrupted_root, _evaluator(interrupted_calls, fail_after=37))
    assert len(interrupted_calls) == 37

    progress_path = interrupted_root / "shards" / "rank-00000" / CALIBRATION_PROGRESS_FILE
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["next_ordinal"] == 74
    assert len(progress["finished_ordinals"]) == 37
    assert progress["totals"] == [37, 0, 0, 0]

    resumed_calls: list[tuple[int, int, int, int, int]] = []
    resumed = _run(interrupted_root, _evaluator(resumed_calls))
    assert resumed["status"] == "completed"
    assert len(resumed_calls) == 512 - 37
    assert set(call[0] for call in resumed_calls) == set(range(4))

    uninterrupted_root = tmp_path / "uninterrupted"
    uninterrupted_calls: list[tuple[int, int, int, int, int]] = []
    uninterrupted = _run(uninterrupted_root, _evaluator(uninterrupted_calls))
    assert len(uninterrupted_calls) == 512
    assert resumed["payload"] == uninterrupted["payload"]
    assert (
        (interrupted_root / "shards" / "rank-00000" / "RESULT.json").read_bytes()
        == (uninterrupted_root / "shards" / "rank-00000" / "RESULT.json").read_bytes()
    )


def test_corrupt_progress_sha_and_identity_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "corrupt"
    calls: list[tuple[int, int, int, int, int]] = []
    with pytest.raises(RuntimeError):
        _run(root, _evaluator(calls, fail_after=3))
    shard = root / "shards" / "rank-00000"
    progress_path = shard / CALIBRATION_PROGRESS_FILE
    sha_path = shard / CALIBRATION_PROGRESS_SHA_FILE

    progress_path.write_text(progress_path.read_text(encoding="utf-8").replace("\"rank\": 0", "\"rank\": 1"), encoding="utf-8")
    with pytest.raises(StageSError, match="SHA mismatch"):
        _run(root, _evaluator([]))

    # Re-seal the intentionally modified JSON to exercise the identity guard,
    # rather than the transport-integrity guard above.
    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    payload["rank"] = 1
    atomic_json(progress_path, payload)
    sha_path.write_text(f"{sha256_file(progress_path)}  {CALIBRATION_PROGRESS_FILE}\n", encoding="utf-8")
    with pytest.raises(StageSError, match="identity"):
        _run(root, _evaluator([]))


def test_progress_contains_no_calibration_lookahead_fields(tmp_path: Path) -> None:
    root = tmp_path / "fields"
    calls: list[tuple[int, int, int, int, int]] = []
    with pytest.raises(RuntimeError):
        _run(root, _evaluator(calls, fail_after=1))
    progress_path = root / "shards" / "rank-00000" / CALIBRATION_PROGRESS_FILE
    payload = json.loads(progress_path.read_text(encoding="utf-8"))

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            result = {str(key).lower() for key in value}
            for child in value.values():
                result.update(keys(child))
            return result
        if isinstance(value, list):
            result: set[str] = set()
            for child in value:
                result.update(keys(child))
            return result
        return set()

    assert keys(payload).isdisjoint(
        {
            "success",
            "final_success",
            "family",
            "trajectory",
            "genealogy",
            "actions",
            "action_prefix",
            "poses",
            "s2",
            "s3",
            "s4",
            "s5",
            "divergence",
            "overdispersion",
            "all_fail",
        }
    )
    assert 0 <= payload["successes"][0] <= payload["totals"][0]
    assert payload["totals"] == [1, 0, 0, 0]
    assert CALIBRATION_TASK_IDS == (0, 3, 6, 9)
    assert CALIBRATION_INITIAL_STATES == tuple(range(8))
    assert CALIBRATION_CANDIDATE_COUNT == 8


def test_completed_shard_is_idempotent_after_resume_seal(tmp_path: Path) -> None:
    root = tmp_path / "idempotent"
    calls: list[tuple[int, int, int, int, int]] = []
    result = _run(root, _evaluator(calls))
    result_bytes = (root / "shards" / "rank-00000" / "RESULT.json").read_bytes()

    def should_not_run(*args: object, **kwargs: object) -> bool:
        raise AssertionError("completed calibration shard executed an evaluator")

    repeated = _run(root, should_not_run)
    assert repeated["status"] == "already_complete"
    assert repeated["payload"] == result["payload"]
    assert (root / "shards" / "rank-00000" / "RESULT.json").read_bytes() == result_bytes
