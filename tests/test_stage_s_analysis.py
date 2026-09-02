from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from r142_stage_s.analysis import (
    BASE_CANDIDATE_COUNT,
    S4_BOOTSTRAP_REPLICATES,
    compute_s1,
    compute_s2,
    compute_s3,
    compute_s4,
    compute_s5,
    decide_stage_s,
    matched_time_tau,
    normalized_workspace_pose_rms,
    paired_bootstrap_recovery,
)
from r142_stage_s.calibration import (
    CALIBRATION_SCHEMA,
    make_calibration_record,
    persist_calibration_report,
    select_calibration_setting,
)
from r142_stage_s.integrity import (
    verify_completion_bundle,
    verify_completed_json,
    verify_sha256sums,
    write_completion,
)


def _rows(success_counts: list[int], *, horizon: int = 10) -> list[dict]:
    output = []
    for family, count in enumerate(success_counts):
        for candidate in range(BASE_CANDIDATE_COUNT):
            output.append(
                {
                    "family_id": family,
                    "task_id": family % 2,
                    "candidate_id": f"{family}-{candidate}",
                    "seed": 100000 + family * 100 + candidate,
                    "success": candidate < count,
                    "poses": np.zeros((horizon, 2), dtype=np.float64),
                }
            )
    return output


def test_calibration_uses_pooled_success_only_and_selects_closest() -> None:
    records = [
        make_calibration_record("mag-1", [True] * 3 + [False] * 7),
        make_calibration_record("mag-2", [True] * 5 + [False] * 5),
    ]
    selected = select_calibration_setting(records, target=0.45)
    assert selected["setting_id"] == "mag-2"
    assert selected["selection_statistic"] == "pooled_success_only"
    assert all("rho" not in key and "divergence" not in key for key in selected)
    with pytest.raises(ValueError):
        make_calibration_record("bad", [True], context={"rho": 10.0})
    with pytest.raises(ValueError):
        select_calibration_setting([{**records[0], "near_all_fail_fraction": 0.2}])


def test_calibration_persisted_schema_has_no_gate_statistics(tmp_path: Path) -> None:
    report = persist_calibration_report(
        tmp_path / "calibration.json",
        [make_calibration_record("a", [True, False]), make_calibration_record("b", [True])],
    )
    assert report["schema"] == CALIBRATION_SCHEMA
    encoded = (tmp_path / "calibration.json").read_text()
    assert "near_all_fail" not in encoded
    assert "divergence" not in encoded
    assert "bootstrap" not in encoded


def test_s1_s2_near_all_fail_strict_zero_rho_and_binomial_tail() -> None:
    rows = _rows([0, 1, 16, 16, 16, 16, 16, 16, 16, 16])
    s1 = compute_s1(rows)
    s2 = compute_s2(rows)
    assert s1["pass"]
    assert s2["near_all_fail_count"] == 2
    assert s2["strict_zero_count"] == 1
    assert s2["rho"] >= 3.0
    assert s2["observed_to_binomial_expected"] > 20.0
    assert s2["pass"]


def test_normalized_workspace_pose_rms_and_matched_tau() -> None:
    same = np.zeros((3, 2), dtype=np.float64)
    shifted = np.ones((3, 2), dtype=np.float64)
    curve = normalized_workspace_pose_rms([same, shifted], workspace_bounds=[[0, 0], [2, 2]])
    assert np.allclose(curve, 0.5)
    rows = [
        {"task_id": 0, "success": True, "poses": same},
        {"task_id": 0, "success": True, "poses": shifted},
        {"task_id": 1, "success": True, "poses": same},
        {"task_id": 1, "success": True, "poses": same},
    ]
    assert matched_time_tau(rows, workspace_scale=[2, 2]) == pytest.approx(0.5)


def test_s3_uses_late_divergence_and_reports_origin_fraction() -> None:
    rows = _rows([0, 0, 16, 16], horizon=10)
    # Successful reference episodes are tightly clustered. Near-fail family 0
    # stays together for two steps then separates at a meaningful interior time.
    for row in rows:
        if row["family_id"] == 0:
            row["poses"] = np.zeros((10, 2), dtype=np.float64)
            if row["candidate_id"].endswith("-31"):
                row["poses"][3:] = 2.0
        elif row["family_id"] == 1:
            row["poses"] = np.zeros((10, 2), dtype=np.float64)
            if row["candidate_id"].endswith("-31"):
                row["poses"][0:] = 2.0
    result = compute_s3(rows, tau=0.1, workspace_scale=[1, 1])
    assert result["near_all_fail_family_count"] == 2
    assert result["t_div_records"][0]["t_div"] == 3
    assert result["origin_t_div_fraction"] == pytest.approx(0.5)
    assert not result["pass"]


def test_s4_requires_interior_prefix_preserving_branch_and_10000_bootstrap() -> None:
    probes = [
        {
            "family_id": i,
            "oracle_branches": [
                {"split_step": 2, "episode_length": 8, "prefix_preserving": True, "success": i < 8},
                {"split_step": 0, "episode_length": 8, "prefix_preserving": True, "success": True},
            ],
            "random_branches": [
                {"split_step": 2, "episode_length": 8, "prefix_preserving": True, "success": i < 2},
                {"split_step": 0, "episode_length": 8, "prefix_preserving": True, "success": True},
            ],
        }
        for i in range(10)
    ]
    result = compute_s4(probes)
    assert result["oracle_recovered_count"] == 8
    assert result["random_recovered_count"] == 2
    assert result["bootstrap"]["replicates"] == S4_BOOTSTRAP_REPLICATES
    assert result["pass"]
    with pytest.raises(ValueError):
        paired_bootstrap_recovery([True], [False], replicates=999)


def test_s5_requires_fresh_seed_and_counts_rescues() -> None:
    base = _rows([0, 1] + [16] * 8)
    extended = []
    for family in range(10):
        original = [row for row in base if row["family_id"] == family]
        for row in original:
            extended.append(dict(row))
        for candidate in range(32):
            extended.append(
                {
                    "family_id": family,
                    "candidate_id": f"fresh-{family}-{candidate}",
                    "seed": 600000 + family * 100 + candidate,
                    "fresh_seed": True,
                    "success": candidate < (2 if family == 0 else (1 if family == 1 else 0)),
                }
            )
    result = compute_s5(base, extended)
    assert result["near_all_fail_family_count"] == 2
    assert result["rescued_family_count"] == 2
    assert result["rescue_fraction"] == pytest.approx(1.0)
    assert result["fresh_seed_verified"]
    assert not result["pass"]


def _all_pass() -> dict:
    gate = {"pass": True}
    return {"S1": gate, "S2": gate, "S3": {**gate, "origin_dominant": False}, "S4": gate, "S5": gate}


def _all_fail(stage: str) -> dict:
    result = _all_pass()
    result[stage] = {"pass": False}
    return result


def test_decision_has_one_code_and_weak_substrate_semantics() -> None:
    assert decide_stage_s({"A": _all_pass(), "B": _all_fail("S1"), "C": _all_fail("S1")}, positive_control_pass=True) == "SUBSTRATE_QUALIFIED"
    assert decide_stage_s({"A": _all_fail("S1"), "B": _all_fail("S1"), "C": _all_pass()}, positive_control_pass=True) == "WEAK_SUBSTRATE_ONLY"
    assert decide_stage_s({"A": _all_pass(), "B": _all_pass(), "C": _all_pass()}, positive_control_pass=False) == "PIPELINE_INVALID"
    assert decide_stage_s({"A": _all_fail("S2"), "B": _all_fail("S2"), "C": _all_fail("S2")}, positive_control_pass=True) == "NO_FAMILY_COLLAPSE"


def test_completion_and_sha_manifests_are_fail_closed(tmp_path: Path) -> None:
    (tmp_path / "result.json").write_text(json.dumps({"success": 1}))
    completion = write_completion(tmp_path, "NO_FAMILY_COLLAPSE")
    assert completion.name.startswith("COMPLETED_")
    assert verify_completed_json(tmp_path)
    assert verify_sha256sums(tmp_path)
    assert verify_completion_bundle(tmp_path)["valid"]
    (tmp_path / "result.json").write_text(json.dumps({"success": 0}))
    assert not verify_completion_bundle(tmp_path)["valid"]


# Corrected fixtures: N=64 contains the original seed identities plus 32 new
# seeds. The late definition is intentionally used by the tests above too.
def _rows(success_counts: list[int], *, horizon: int = 10) -> list[dict]:
    output = []
    for family, count in enumerate(success_counts):
        for candidate in range(BASE_CANDIDATE_COUNT):
            output.append(
                {
                    "family_id": family,
                    "task_id": family % 2,
                    "candidate_id": f"{family}-{candidate}",
                    "seed": 500000 + family * 100 + candidate,
                    "success": candidate < count,
                    "poses": np.zeros((horizon, 2), dtype=np.float64),
                }
            )
    return output


def test_tau_is_separate_for_each_task_and_matched_control_step() -> None:
    zero = np.zeros((3, 2), dtype=np.float64)
    task0_other = zero.copy()
    task0_other[1] = [0.2, 0.0]
    task0_other[2] = [0.4, 0.0]
    task1_other = zero.copy()
    task1_other[1] = [2.0, 0.0]
    task1_other[2] = [4.0, 0.0]
    rows = [
        {"task_id": 0, "success": True, "poses": zero},
        {"task_id": 0, "success": True, "poses": task0_other},
        {"task_id": 1, "success": True, "poses": zero},
        {"task_id": 1, "success": True, "poses": task1_other},
    ]
    from r142_stage_s.analysis import matched_time_tau_curve

    curves = matched_time_tau_curve(rows, workspace_scale=[1.0, 1.0])
    assert np.allclose(curves[0], [0.0, 0.1414213562, 0.2828427125])
    assert np.allclose(curves[1], [0.0, 1.4142135624, 2.8284271247])
    assert not np.isclose(curves[0][1], curves[1][1])


def test_s3_uses_the_near_family_task_tau_curve() -> None:
    refs = [
        {"task_id": 0, "success": True, "poses": np.zeros((3, 2))},
        {"task_id": 0, "success": True, "poses": np.asarray([[0, 0], [0.2, 0], [0.2, 0]], dtype=float)},
        {"task_id": 1, "success": True, "poses": np.zeros((3, 2))},
        {"task_id": 1, "success": True, "poses": np.asarray([[0, 0], [2.0, 0], [2.0, 0]], dtype=float)},
    ]
    rollouts = []
    for family, task in [(10, 0), (11, 1)]:
        for candidate in range(32):
            pose = np.zeros((3, 2), dtype=float)
            if candidate >= 16:
                pose[1:] = 1.0
            rollouts.append(
                {
                    "family_id": family,
                    "task_id": task,
                    "candidate_id": f"{family}-{candidate}",
                    "success": False,
                    "poses": pose,
                }
            )
    result = compute_s3(rollouts, successful_episodes=refs, workspace_scale=[1, 1])
    by_task = {row["task_id"]: row for row in result["t_div_records"]}
    assert by_task["0"]["t_div"] == 1
    assert by_task["1"]["t_div"] == 2
    assert set(result["tau_by_task"]) == {"0", "1"}
    assert result["tau_source"] == "successful_same_task_matched_time"


def test_s3_censors_at_last_comparable_tau_step_and_rejects_missing_task_tau() -> None:
    refs = [
        {"task_id": 0, "success": True, "poses": np.zeros((3, 2))},
        {"task_id": 0, "success": True, "poses": np.zeros((3, 2))},
    ]
    rollouts = []
    for candidate in range(32):
        rollouts.append(
            {
                "family_id": "task0-family",
                "task_id": 0,
                "candidate_id": f"t0-{candidate}",
                "success": False,
                "poses": np.zeros((6, 2)),
            }
        )
        rollouts.append(
            {
                "family_id": "task1-family",
                "task_id": 1,
                "candidate_id": f"t1-{candidate}",
                "success": False,
                "poses": np.zeros((6, 2)),
            }
        )
    result = compute_s3(rollouts, successful_episodes=refs, workspace_scale=[1, 1])
    assert result["near_all_fail_family_count"] == 2
    assert result["evaluable_family_count"] == 1
    assert result["missing_tau_family_count"] == 1
    assert result["t_div_records"][0]["t_div"] == 2
    assert result["t_div_records"][0]["last_comparable_step"] == 2
    assert result["t_div_records"][0]["fraction"] == pytest.approx(2 / 6)
    assert not result["pass"]


def test_s5_accepts_only_32_added_disjoint_seeds() -> None:
    base = _rows([0] + [16] * 9)
    extended = [dict(row) for row in base]
    for family in range(10):
        extended.extend(
            {
                "family_id": family,
                "candidate_id": f"new-{family}-{candidate}",
                "seed": 900000 + family * 100 + candidate,
                "fresh_seed": True,
                "success": False,
            }
            for candidate in range(32)
        )
    result = compute_s5(base, extended)
    assert result["total_candidate_count"] == 64
    assert result["added_fresh_candidate_count"] == 32
    assert result["fresh_seed_verified"]
    assert result["rescue_fraction"] == 0.0
    assert result["pass"]


def test_s5_rejects_mutated_base_success_in_n64_view() -> None:
    base = _rows([0] + [16] * 9)
    extended = [dict(row) for row in base]
    extended[0]["success"] = True
    for family in range(10):
        extended.extend(
            {
                "family_id": family,
                "candidate_id": f"new-mut-{family}-{candidate}",
                "seed": 990000 + family * 100 + candidate,
                "fresh_seed": True,
                "success": False,
            }
            for candidate in range(32)
        )
    result = compute_s5(base, extended)
    assert not result["fresh_seed_verified"]
    assert not result["pass"]


def test_decision_prunes_failed_headline_before_deeper_gate() -> None:
    a = _all_fail("S2")
    b = _all_fail("S3")
    b["S3"] = {"pass": False, "origin_dominant": False}
    c = _all_fail("S1")
    assert decide_stage_s({"A": a, "B": b, "C": c}, positive_control_pass=True) == "UNRECOVERABLE_FAILURES"
    a2 = _all_fail("S1")
    b2 = _all_fail("S3")
    b2["S3"] = {"pass": False, "origin_dominant": True}
    assert decide_stage_s({"A": a2, "B": b2, "C": c}, positive_control_pass=True) == "COLLAPSE_AT_ORIGIN"


def test_decision_uses_deepest_plan_gate_when_only_c_reaches_target_difficulty() -> None:
    a = _all_fail("S1")
    b = _all_fail("S1")
    c = _all_fail("S2")
    assert decide_stage_s({"A": a, "B": b, "C": c}, positive_control_pass=True) == "NO_FAMILY_COLLAPSE"
