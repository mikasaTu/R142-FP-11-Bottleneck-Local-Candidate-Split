from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from r142_stage_s import frozen_protocol as frozen_protocol_module
from r142_stage_s.analysis import (
    S3_SUBSTRATE_WORKSPACE_SCALES,
    compute_s3,
    compute_s3_production,
    compute_s4_from_protocol,
)
from r142_stage_s.frozen_protocol import load_frozen_protocol
from r142_stage_s.main_protocol import (
    FROZEN_SUMMARY,
    PROTOCOL_ACCEPTANCE_SCHEMA,
    S4_PROTOCOL,
    S5_PROTOCOL,
    STAGE_S_PROTOCOL_ID,
    read_frozen_protocol as read_main_protocol,
)
from r142_stage_s.s45_runtime import ProtocolAuthority, S45ProtocolError
from r142_stage_s.total_analysis import (
    EXPECTED_S4_BOOTSTRAP_REPLICATES,
    EXPECTED_S4_BOOTSTRAP_SEED,
    TOTAL_COMPLETION_FILE,
    TOTAL_RESULT_FILE,
    TotalAnalysisError,
    _extract_arm_inputs,
    _load_controls,
    _normalise_row,
    _verify_replay_check,
    _verify_snapshot,
    analyze_stage_s,
)


COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _protocol(tmp_path: Path) -> Path:
    payload = {
        "protocol_id": STAGE_S_PROTOCOL_ID,
        "protocol_git_commit": COMMIT,
        "s4": {
            **S4_PROTOCOL,
        },
        "s5": {
            **S5_PROTOCOL,
        },
    }
    path = tmp_path / "FROZEN_PROTOCOL.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _row(family: str, index: int, *, substrate: str, success: bool, task: str = "task-0") -> dict:
    width = 14 if substrate == "A" else 6
    actions = [[float(index), 0.0] for _ in range(4)]
    poses = np.zeros((4, width), dtype=float).tolist()
    return {
        "family_id": family,
        "task_id": task,
        "init_state": "init-0",
        "candidate_index": index,
        "candidate_id": f"{family}/candidate-{index:04d}",
        "candidate_seed": 100000 + index,
        "seed": 100000 + index,
        "success": success,
        "final_success": success,
        "terminated": True,
        "termination": "official_step_limit",
        "actions": actions,
        "action_prefix": actions,
        "poses": poses,
        "env_steps": 4,
        "policy_forwards": 4,
        "compute": {
            "policy_forwards": 4,
            "environment_steps": 4,
            "env_steps": 4,
            "primary_unit": "policy_forward_pass",
            "secondary_unit": "environment_step",
        },
        "parent_id": None,
        "generation_step": 0,
        "genealogy": {
            "parent_id": None,
            "generation_step": 0,
            "action_prefix": actions,
            "final_success": success,
        },
    }


def _arm_rows(substrate: str, *, count: int = 16) -> list[dict]:
    family = f"{substrate.lower()}-family-0"
    # A is deliberately all-fail (near-all-fail), while B/C are at the
    # difficulty-range midpoint. This drives the depth-ordered decision and
    # still leaves every arm eligible for all five calculations.
    successes = 0 if substrate == "A" else count
    return [
        _row(family, index, substrate=substrate, success=index < successes)
        for index in range(32)
    ]


def _extended_a() -> dict[str, list[dict]]:
    rows = []
    for index in range(64):
        rows.append(
            {
                "family_id": "a-family-0",
                "candidate_id": f"a-family-0/candidate-{index:04d}",
                "candidate_seed": 100000 + index if index < 32 else 200000 + index,
                "seed": 100000 + index if index < 32 else 200000 + index,
                "success": False,
                "fresh_seed": index >= 32,
            }
        )
    return {"a-family-0": rows}


def _probe() -> dict:
    return {
        "family_id": "a-family-0",
        "oracle_recovered": True,
        "random_recovered": False,
        "oracle_branch_count": 8,
        "random_branch_count": 8,
    }


def _verified_arms() -> dict[str, dict]:
    flags = {
        "terminal_markers": True,
        "sha256": True,
        "genealogy": True,
        "compute": True,
    }
    return {
        "A": {
            "substrate": "A",
            "records": _arm_rows("A"),
            "probes": [_probe()],
            "extended_rollouts": _extended_a(),
            "pipeline_identity": "pipeline-v1",
            "artifact_verification": flags,
        },
        "B": {
            "substrate": "B",
            "records": _arm_rows("B"),
            "probes": [],
            "extended_rollouts": {},
            "pipeline_identity": "pipeline-v1",
            "artifact_verification": flags,
        },
        "C": {
            "substrate": "C",
            "records": _arm_rows("C"),
            "probes": [],
            "extended_rollouts": {},
            "pipeline_identity": "pipeline-v1",
            "artifact_verification": flags,
        },
    }


def test_schema_round_trip_uses_one_frozen_object(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cpfs_root = tmp_path / "cpfs"
    protocol_root = cpfs_root / "logs" / "stage-s" / "protocol"
    protocol_root.mkdir(parents=True)
    protocol_md = protocol_root / "PROTOCOL.md"
    protocol_md.write_text("# Stage-S frozen protocol\n", encoding="utf-8")
    reports = {}
    for substrate, payload in (
        ("B", {"selected_setting": "proximity_0.08m", "variant_run_id": "r142-stage-s-b-variants-20260903-r7"}),
        ("C", {"selected_checkpoint": "step_10000", "selected_checkpoint_sha256": "a" * 64}),
    ):
        report = cpfs_root / f"{substrate}-CALIBRATION_REPORT.json"
        report.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        reports[substrate] = {
            "report_path": str(report),
            "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
            **payload,
        }
    md_sha = hashlib.sha256(protocol_md.read_bytes()).hexdigest()
    acceptance = {
        "status": "ACCEPTED",
        "frozen": True,
        "protocol_id": STAGE_S_PROTOCOL_ID,
        "protocol_git_commit": COMMIT,
        "protocol_md_path": str(protocol_md),
        "protocol_md_sha256": md_sha,
        "frozen_summary": FROZEN_SUMMARY,
        "calibration_reports": reports,
    }
    payload = {
        "schema": PROTOCOL_ACCEPTANCE_SCHEMA,
        "status": "FROZEN",
        "protocol_id": STAGE_S_PROTOCOL_ID,
        "protocol_git_commit": COMMIT,
        "protocol_md_path": str(protocol_md),
        "protocol_md_sha256": md_sha,
        "frozen_summary": FROZEN_SUMMARY,
        "s4": S4_PROTOCOL,
        "s5": S5_PROTOCOL,
        "files": {
            "PROTOCOL.md": {"path": str(protocol_md), "sha256": md_sha},
            "B_CALIBRATION_REPORT": {"path": str(reports["B"]["report_path"]), "sha256": reports["B"]["report_sha256"]},
            "C_CALIBRATION_REPORT": {"path": str(reports["C"]["report_path"]), "sha256": reports["C"]["report_sha256"]},
        },
        "acceptance": acceptance,
    }
    authority_path = protocol_root / "FROZEN_PROTOCOL.json"
    authority_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    main = read_main_protocol(authority_path, substrate="B", calibration_report=reports["B"]["report_path"])
    monkeypatch.setattr(frozen_protocol_module, "_CPFS_ROOT", cpfs_root)
    frozen = load_frozen_protocol(authority_path)
    runtime = ProtocolAuthority.load(authority_path)

    assert main["s4"]["search_t_grid"] == list(range(1, 10))
    assert frozen["s4"]["search_t_grid"] == main["s4"]["search_t_grid"]
    assert frozen["s5"] == main["s5"]
    assert runtime.search_steps("family-0", 20) == tuple(range(1, 10))
    assert runtime.paired_bootstrap_seed == EXPECTED_S4_BOOTSTRAP_SEED


def test_production_s3_uses_substrate_scale_and_forbids_scalar_tau() -> None:
    rows = _arm_rows("B") + [
        _row("b-family-1", index, substrate="B", success=index < 16)
        for index in range(32)
    ]
    result = compute_s3_production(rows, substrate="B", successful_episodes=[row for row in rows if row["success"]])
    assert result["substrate"] == "B"
    assert result["workspace_scale"] == list(S3_SUBSTRATE_WORKSPACE_SCALES["B"])
    assert result["tau_source"] == "successful_same_task_matched_time"
    assert result["tau_override_forbidden"] is True
    with pytest.raises(ValueError, match="scalar tau"):
        compute_s3(rows, substrate="B", tau=0.1)


def test_s4_reads_seed_from_protocol_and_requires_equal_eight_pairs(tmp_path: Path) -> None:
    protocol = ProtocolAuthority.load(_protocol(tmp_path))
    result = compute_s4_from_protocol([_probe()], protocol)
    assert result["bootstrap"]["seed"] == EXPECTED_S4_BOOTSTRAP_SEED
    assert result["bootstrap"]["replicates"] == EXPECTED_S4_BOOTSTRAP_REPLICATES
    assert result["protocol_grid_point_count"] == 9
    assert result["protocol_branch_count_oracle"] == 8
    assert result["protocol_branch_count_random"] == 8
    bad = _probe()
    bad["random_branch_count"] = 7
    with pytest.raises(ValueError, match="exactly 8"):
        compute_s4_from_protocol([bad], protocol)


def test_total_analyzer_evaluates_all_arms_and_writes_complete_bundle(tmp_path: Path) -> None:
    protocol = _protocol(tmp_path)
    report = analyze_stage_s(
        protocol_path=protocol,
        controls={"overall_verdict": "CONTROLS_PASS", "positive_verdict": "POSITIVE_CONTROL_PASS", "null_verdict": "NO_FAMILY_COLLAPSE", "pipeline_commit": "pipeline-v1"},
        arms=_verified_arms(),
        output_root=tmp_path / "total",
    )
    assert report["decision_code"] == "NO_FAMILY_COLLAPSE"
    assert report["decision_code_count"] == 1
    assert report["all_arms_evaluated"] is True
    for name in ("A", "B", "C"):
        assert all(key in report["arms"][name] for key in ("S1", "S2", "S3", "S4", "S5"))
    assert report["arms"]["A"]["S4"]["bootstrap"]["seed"] == EXPECTED_S4_BOOTSTRAP_SEED
    assert (tmp_path / "total" / TOTAL_RESULT_FILE).is_file()
    assert (tmp_path / "total" / TOTAL_COMPLETION_FILE).is_file()
    assert report["bundle"]["valid"] is True


def test_total_analyzer_keeps_other_arms_when_one_arm_is_malformed(tmp_path: Path) -> None:
    arms = _verified_arms()
    arms["B"]["records"] = arms["B"]["records"][:-1]
    report = analyze_stage_s(
        protocol_path=_protocol(tmp_path),
        controls={"overall_verdict": "CONTROLS_PASS", "positive_verdict": "POSITIVE_CONTROL_PASS", "null_verdict": "NO_FAMILY_COLLAPSE", "pipeline_commit": "pipeline-v1"},
        arms=arms,
    )
    assert report["decision_code"] == "PIPELINE_INVALID"
    assert report["all_arms_evaluated"] is True
    assert all(key in report["arms"]["B"] for key in ("S1", "S2", "S3", "S4", "S5"))
    assert "status" not in report["arms"]["A"]["S1"]
    assert "status" not in report["arms"]["C"]["S1"]


def test_production_authority_without_bootstrap_seed_fails_closed(tmp_path: Path) -> None:
    path = _protocol(tmp_path)
    payload = json.loads(path.read_text())
    del payload["s4"]["paired_bootstrap_seed"]
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(S45ProtocolError, match="paired_bootstrap_seed"):
        ProtocolAuthority.load(path)


def test_production_arm_rejects_in_memory_artifacts_even_with_verification_flags(tmp_path: Path) -> None:
    protocol = ProtocolAuthority.load(_protocol(tmp_path))
    with pytest.raises(TotalAnalysisError, match="unsupported/in-memory"):
        _extract_arm_inputs(
            "A",
            {
                "substrate": "A",
                "records": _arm_rows("A"),
                "artifact_verification": {
                    "terminal_markers": True,
                    "sha256": True,
                    "genealogy": True,
                    "compute": True,
                },
                "main_root": tmp_path,
                "s4_root": tmp_path,
                "s5_root": tmp_path,
            },
            protocol=protocol,
            strict=True,
        )


def test_production_controls_reject_in_memory_mapping() -> None:
    with pytest.raises(TotalAnalysisError, match="audited directory"):
        _load_controls(
            {
                "overall_verdict": "CONTROLS_PASS",
                "positive_verdict": "POSITIVE_CONTROL_PASS",
                "null_verdict": "NO_FAMILY_COLLAPSE",
                "pipeline_commit": COMMIT,
            },
            strict=True,
        )


def test_snapshot_requires_all_rng_streams_and_replay_error_is_tight() -> None:
    snapshot = {
        "simulator": {"state": 1},
        "observation_history": [],
        "action_queue": [],
        "python_rng_state": {"state": 1},
        "numpy_rng_state": {"state": 1},
        "policy_rng_state": {"state": 1},
        "torch_rng_state": {"cpu": [1], "cuda": [2]},
    }
    _verify_snapshot(snapshot, label="test snapshot")
    del snapshot["torch_rng_state"]["cuda"]
    with pytest.raises(TotalAnalysisError, match="CPU/CUDA"):
        _verify_snapshot(snapshot, label="test snapshot")
    with pytest.raises(TotalAnalysisError, match="exceeds 1e-9"):
        _verify_replay_check(
            {"same_action": True, "passed": True, "max_abs_error": 1.1e-9},
            label="test replay",
        )


def test_robotwin_snapshot_requires_runtime_and_policy_rng_streams() -> None:
    stream = {"python": [1], "numpy": [2], "torch": [3], "torch_cuda": []}
    _verify_snapshot(
        {
            "simulator": {"state": 1},
            "policy_history": [],
            "action_queue": [],
            "rng_streams": {"runtime": stream, "policy": stream},
        },
        label="Robotwin snapshot",
    )


def test_strict_normalise_rejects_missing_task_id_and_replay() -> None:
    row = _row("a-family-0", 0, substrate="A", success=False)
    row.pop("task_id")
    with pytest.raises(TotalAnalysisError, match="lacks task_id"):
        _normalise_row(
            row,
            family={"family_id": "a-family-0", "task_id": None, "init_state": 0},
            index=0,
            substrate="A",
            strict=True,
        )
    row = _row("a-family-0", 0, substrate="A", success=False)
    with pytest.raises(TotalAnalysisError, match="snapshot_restore_check"):
        _normalise_row(
            row,
            family={"family_id": "a-family-0", "task_id": 0, "init_state": 0},
            index=0,
            substrate="A",
            strict=True,
        )


def test_strict_normalise_rejects_genealogy_without_explicit_root() -> None:
    row = _row("a-family-0", 0, substrate="A", success=False)
    row["task_id"] = 0
    row["init_state"] = 0
    row["snapshot_restore_check"] = {"same_action": True, "passed": True, "max_abs_error": 0.0}
    row["genealogy"].update(
        {
            "candidate_id": row["candidate_id"],
            "candidate_index": 0,
            "candidate_seed": row["candidate_seed"],
        }
    )
    with pytest.raises(TotalAnalysisError, match="lacks root binding"):
        _normalise_row(
            row,
            family={"family_id": "a-family-0", "task_id": 0, "init_state": 0},
            index=0,
            substrate="A",
            genealogy=row["genealogy"],
            strict=True,
        )
