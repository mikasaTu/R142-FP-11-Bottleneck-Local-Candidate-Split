from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from r142_stage_s.calibration_freeze import (
    B_SETTINGS,
    B_VARIANT_RUN_ID,
    C_SETTINGS,
    CALIBRATION_RESULT_SCHEMA,
    CALIBRATION_SEED,
    CALIBRATION_TARGET,
    CALIBRATION_WORLD_SIZE,
    FROZEN_SUMMARY,
    STAGE_S_PROTOCOL_ID,
    CalibrationFreezeError,
    freeze_calibration_reports,
    freeze_protocol,
    read_frozen_protocol,
)


ROOT = Path(__file__).resolve().parents[1]
OPENPI_COMMIT = "54cbaee6ae0c010a1ed431871cdaa8f4684ac709"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_manifest(root: Path, name: str, paths: list[Path]) -> Path:
    manifest = root / name
    lines = [f"{_sha(path)}  {path.relative_to(root).as_posix()}" for path in paths]
    manifest.write_text("\n".join(sorted(lines)) + "\n", encoding="utf-8")
    return manifest


def _result(root: Path, settings: tuple[str, ...], counts: list[int]) -> Path:
    result = root / "CALIBRATION_RESULT.json"
    rows = [
        {
            "setting": setting,
            "successes": count,
            "total": 256,
            "pooled_success": count / 256,
        }
        for setting, count in zip(settings, counts)
    ]
    selected = min(rows, key=lambda row: (abs(row["pooled_success"] - CALIBRATION_TARGET), row["setting"]))
    _write_json(
        result,
        {
            "schema": CALIBRATION_RESULT_SCHEMA,
            "protocol_id": STAGE_S_PROTOCOL_ID,
            "calibration_seed": CALIBRATION_SEED,
            "world_size": CALIBRATION_WORLD_SIZE,
            "rows": rows,
            "target_pooled_success": CALIBRATION_TARGET,
            "selected_setting": selected["setting"],
        },
    )
    _write_manifest(root, "SHA256SUMS", [result])
    return result


def _make_terminal_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    b_root = tmp_path / "b_calibration"
    b_root.mkdir(parents=True)
    variant_root = tmp_path / "b_variants" / B_VARIANT_RUN_ID / "variants"
    for setting in B_SETTINGS:
        setting_root = variant_root / setting
        setting_root.mkdir(parents=True)
        (setting_root / "config.yaml").write_text(f"setting: {setting}\n", encoding="utf-8")
    b_result = _result(b_root, B_SETTINGS, [100, 112, 128, 100])
    rank_paths = []
    for rank in range(8):
        rank_path = b_root / "shards" / f"rank-{rank:05d}" / "COMPLETED_SHARD.json"
        _write_json(rank_path, {"rank": rank, "status": "COMPLETED"})
        rank_paths.append(rank_path)
    b_marker = b_root / "COMPLETED_B_CALIBRATION.json"
    b_marker_payload = {
        "schema": "r142-stage-s-b-calibration-completion-v1",
        "status": "COMPLETED",
        "protocol_id": STAGE_S_PROTOCOL_ID,
        "substrate": "B",
        "run_id": "test-b-calibration",
        "input_bundle_run_id": B_VARIANT_RUN_ID,
        "calibration_result": b_result.name,
        "calibration_result_sha256": _sha(b_result),
        "calibration_result_schema": CALIBRATION_RESULT_SCHEMA,
        "calibration_seed": CALIBRATION_SEED,
        "world_size": CALIBRATION_WORLD_SIZE,
        "rank_markers": [f"shards/rank-{rank:05d}/COMPLETED_SHARD.json" for rank in range(8)],
        "rank_marker_sha256": {
            path.relative_to(b_root).as_posix(): _sha(path) for path in rank_paths
        },
        "provenance": {"variant_root": str(variant_root)},
        "persistence": {"bundle_sha_file": "B_SHA256SUMS"},
    }
    _write_json(b_marker, b_marker_payload)
    _write_manifest(b_root, "B_SHA256SUMS", [b_result, b_marker, *rank_paths])

    c_root = tmp_path / "c_calibration"
    c_root.mkdir(parents=True)
    c_result = _result(c_root, C_SETTINGS, [120, 80, 110, 115])
    c_marker = c_root / "COMPLETED_C_CALIBRATION.json"
    c_marker_payload = {
        "schema": "r142-stage-s-c-calibration-completion-v1",
        "status": "COMPLETED",
        "protocol_id": STAGE_S_PROTOCOL_ID,
        "substrate": "C",
        "calibration_result": c_result.name,
        "calibration_result_sha256": _sha(c_result),
        "calibration_result_schema": CALIBRATION_RESULT_SCHEMA,
        "calibration_seed": CALIBRATION_SEED,
        "world_size": CALIBRATION_WORLD_SIZE,
        "persistence": {"bundle_sha_file": "C_SHA256SUMS"},
    }
    _write_json(c_marker, c_marker_payload)
    _write_manifest(c_root, "C_SHA256SUMS", [c_result, c_marker])

    train_root = tmp_path / "accepted_c_training"
    checkpoint_entries = []
    for step in (1000, 3000, 6000, 10000):
        checkpoint = train_root / str(step)
        checkpoint.mkdir(parents=True)
        weight = checkpoint / "model.safetensors"
        weight.write_bytes(f"model-step-{step}".encode())
        manifest = _write_manifest(checkpoint, "SHA256SUMS", [weight])
        checkpoint_entries.append({"path": str(checkpoint), "step": step, "sha256": _sha(manifest)})
    completion = train_root / "COMPLETED_C_TRAINING.json"
    _write_json(
        completion,
        {
            "schema": "r142-stage-s-c-training-completion-v1",
            "status": "COMPLETED",
            "openpi_commit": OPENPI_COMMIT,
            "config_name": "pi05_libero",
            "seed": 42,
            "terminal_global_step": 10001,
            "checkpoint_steps": [1000, 3000, 6000, 10000],
            "checkpoint_audit": {"valid": True, "checkpoints": checkpoint_entries},
        },
    )
    return b_result, b_marker, c_result, c_marker, completion


def _protocol_markdown(commit: str) -> str:
    return f"""# Stage-S protocol

Protocol GitHub commit: {commit}

S1 threshold: pooled_success in [0.30, 0.60].
S2 thresholds: near-all-fail fraction >= 0.10; rho >= 3.0; near-all-fail vs binomial >= 20.
S3 thresholds: median t_div fraction >= 0.10; t_div zero fraction <= 0.25; tau quantile = 0.95.
S4 thresholds: recoverable family fraction >= 0.30; paired bootstrap replicates = 10000.
S5 threshold: best-of-N=64 rescue fraction <= 0.05.

D(t) normalization is fixed by the declared action-space normalization.
tau is the same-task matched-t 95th percentile, estimated before the main screen.
A family is one task/initial-state unit. near-all-fail means <= 1/32 successful
candidates in a family (the 1/32 operational definition).

RNG literal contract: Python, NumPy, Torch CPU/CUDA, environment, and policy
seeds are derived from the frozen seed plan. Compute literal contract:
policy_forward_pass is the primary unit and environment_step is the secondary unit.
"""


def _freeze_inputs(tmp_path: Path):
    paths = _make_terminal_inputs(tmp_path)
    reports = freeze_calibration_reports(
        b_result=paths[0],
        b_completion_marker=paths[1],
        c_result=paths[2],
        c_completion_marker=paths[3],
        c_lineage=paths[4],
        b_report=tmp_path / "out" / "b" / "CALIBRATION_REPORT.json",
        c_report=tmp_path / "out" / "c" / "CALIBRATION_REPORT.json",
    )
    return paths, reports


def test_terminal_selection_and_loader_schema(tmp_path: Path) -> None:
    paths, reports = _freeze_inputs(tmp_path)
    assert reports["B"]["selected_setting"] == "proximity_0.08m"
    assert reports["C"]["selected_setting"] == "step_10000"
    assert reports["B"]["selected_variant_root"].endswith("/proximity_0.08m")
    assert len(reports["C"]["selected_checkpoint_sha256"]) == 64
    assert reports["B"]["source_result_sha256"] == _sha(paths[0])
    assert reports["C"]["source_completion_marker_sha256"] == _sha(paths[3])


def test_result_tamper_and_forbidden_lookahead_fail_closed(tmp_path: Path) -> None:
    paths = _make_terminal_inputs(tmp_path)
    result = paths[0]
    result.write_text(result.read_text(encoding="utf-8").replace("\"successes\": 112", "\"successes\": 113"), encoding="utf-8")
    with pytest.raises(CalibrationFreezeError, match="calibration"):
        freeze_calibration_reports(
            b_result=paths[0],
            b_completion_marker=paths[1],
            c_result=paths[2],
            c_completion_marker=paths[3],
            c_lineage=paths[4],
            b_report=tmp_path / "tamper-out-b.json",
            c_report=tmp_path / "tamper-out-c.json",
        )

    paths = _make_terminal_inputs(tmp_path / "leak")
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    payload["rows"][0]["rho"] = 4.0
    _write_json(paths[0], payload)
    with pytest.raises(CalibrationFreezeError, match="forbidden"):
        freeze_calibration_reports(
            b_result=paths[0],
            b_completion_marker=paths[1],
            c_result=paths[2],
            c_completion_marker=paths[3],
            c_lineage=paths[4],
            b_report=tmp_path / "leak-out-b.json",
            c_report=tmp_path / "leak-out-c.json",
        )


def test_symlinked_terminal_input_is_rejected(tmp_path: Path) -> None:
    paths = _make_terminal_inputs(tmp_path)
    alias = tmp_path / "b-result-alias.json"
    alias.symlink_to(paths[0])
    with pytest.raises(CalibrationFreezeError, match="symlinked"):
        freeze_calibration_reports(
            b_result=alias,
            b_completion_marker=paths[1],
            c_result=paths[2],
            c_completion_marker=paths[3],
            c_lineage=paths[4],
            b_report=tmp_path / "symlink-out-b.json",
            c_report=tmp_path / "symlink-out-c.json",
        )


def test_training_lineage_tamper_fails_closed(tmp_path: Path) -> None:
    paths = _make_terminal_inputs(tmp_path)
    completion = paths[4]
    payload = json.loads(completion.read_text(encoding="utf-8"))
    payload["openpi_commit"] = "0" * 40
    _write_json(completion, payload)
    with pytest.raises(CalibrationFreezeError, match="OpenPI commit"):
        freeze_calibration_reports(
            b_result=paths[0],
            b_completion_marker=paths[1],
            c_result=paths[2],
            c_completion_marker=paths[3],
            c_lineage=paths[4],
            b_report=tmp_path / "tamper-out-b.json",
            c_report=tmp_path / "tamper-out-c.json",
        )


def test_protocol_freeze_and_acceptance_tamper_detection(tmp_path: Path) -> None:
    paths, _ = _freeze_inputs(tmp_path)
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    source_md = tmp_path / "repo" / "stage-s" / "PROTOCOL.md"
    source_md.parent.mkdir(parents=True)
    source_md.write_text(_protocol_markdown(commit), encoding="utf-8")
    acceptance_path = tmp_path / "logs" / "stage_s" / "protocol" / "FROZEN_PROTOCOL.json"
    payload = freeze_protocol(
        protocol_md=source_md,
        protocol_git_commit=commit,
        b_report=tmp_path / "out" / "b" / "CALIBRATION_REPORT.json",
        c_report=tmp_path / "out" / "c" / "CALIBRATION_REPORT.json",
        output_path=acceptance_path,
        repo_root=ROOT,
    )
    assert payload["schema"] == "r142-stage-s-protocol-acceptance-v1"
    read_frozen_protocol(
        acceptance_path,
        substrate="B",
        calibration_report=tmp_path / "out" / "b" / "CALIBRATION_REPORT.json",
    )
    acceptance_path.parent.joinpath("PROTOCOL.md").write_text("tampered", encoding="utf-8")
    with pytest.raises(CalibrationFreezeError, match="markdown"):
        read_frozen_protocol(
            acceptance_path,
            substrate="B",
            calibration_report=tmp_path / "out" / "b" / "CALIBRATION_REPORT.json",
        )


def test_protocol_missing_requirement_or_commit_is_rejected(tmp_path: Path) -> None:
    _freeze_inputs(tmp_path)
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    bad_md = tmp_path / "bad" / "PROTOCOL.md"
    bad_md.parent.mkdir(parents=True)
    bad_md.write_text("S1 0.30 0.60\n", encoding="utf-8")
    with pytest.raises(CalibrationFreezeError, match="requirements"):
        freeze_protocol(
            protocol_md=bad_md,
            protocol_git_commit=commit,
            b_report=tmp_path / "out" / "b" / "CALIBRATION_REPORT.json",
            c_report=tmp_path / "out" / "c" / "CALIBRATION_REPORT.json",
            output_path=tmp_path / "bad-out" / "FROZEN_PROTOCOL.json",
            repo_root=ROOT,
        )
