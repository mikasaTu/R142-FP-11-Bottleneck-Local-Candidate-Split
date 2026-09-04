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
    C_TRAINING_ACCEPTANCE_SCHEMA,
    _select_calibration_row,
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


from r142_stage_s.libero import _calibration_selection_key


ROOT = Path(__file__).resolve().parents[1]
OPENPI_COMMIT = "54cbaee6ae0c010a1ed431871cdaa8f4684ac709"
SOURCE = {
    "stage_s_commit": "59581b09ce974a7080aaf6660f7619be465ce19d",
    "qpilots_commit": "eacf47b981e3b22357f8a74902f8dad8cfcfa375",
    "openpi_commit": OPENPI_COMMIT,
    "libero_commit": "f78abd68ee283de9f9be3c8f7e2a9ad60246e95c",
}


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


def _make_registry_binding(
    tmp_path: Path,
    *,
    substrate: str,
    controller_run_id: str,
    application_run_id: str,
    artifact_dir: Path,
) -> dict[str, Path]:
    registry_root = tmp_path / "pai-registry" / substrate / controller_run_id
    registry_root.mkdir(parents=True)
    job_id = f"dlctest{substrate.lower()}{len(controller_run_id)}"
    _write_json(
        registry_root / "result.json",
        {
            "run_id": controller_run_id,
            "job_id": job_id,
            "submission_state": "submitted_verified",
        },
    )
    _write_json(
        registry_root / "submission-state.json",
        {
            "run_id": controller_run_id,
            "job_id": job_id,
            "state": "submitted_verified",
        },
    )
    _write_json(
        registry_root / "resolved.json",
        {
            "run_id": controller_run_id,
            "artifact_dir": str(artifact_dir),
            "runtime": {"write_paths": [str(artifact_dir)]},
        },
    )
    ledger = registry_root / "jobs.jsonl"
    ledger.write_text(
        json.dumps({"run_id": controller_run_id, "job_id": job_id}) + "\n",
        encoding="utf-8",
    )
    getjob = registry_root / "getjob-terminal.json"
    _write_json(getjob, {"JobId": job_id, "Status": "Succeeded", "ReasonCode": "JobSucceeded"})
    getjob_sha = registry_root / "getjob-terminal.json.sha256"
    getjob_sha.write_text(f"{_sha(getjob)}  {getjob.name}\n", encoding="utf-8")
    if controller_run_id != application_run_id:
        incarnation = artifact_dir / "controller-incarnations" / f"{controller_run_id}.json"
        _write_json(
            incarnation,
            {
                "controller_run_id": controller_run_id,
                "application_run_id": application_run_id,
            },
        )
    return {
        "registry_run": registry_root,
        "jobs_ledger": ledger,
        "getjob_terminal": getjob,
        "getjob_terminal_sha": getjob_sha,
    }


def _external_kwargs(paths: tuple) -> dict[str, Path]:
    b_binding, c_binding = paths[6], paths[7]
    return {
        "c_config": paths[5],
        "b_registry_run": b_binding["registry_run"],
        "b_jobs_ledger": b_binding["jobs_ledger"],
        "b_getjob_terminal": b_binding["getjob_terminal"],
        "b_getjob_terminal_sha": b_binding["getjob_terminal_sha"],
        "c_registry_run": c_binding["registry_run"],
        "c_jobs_ledger": c_binding["jobs_ledger"],
        "c_getjob_terminal": c_binding["getjob_terminal"],
        "c_getjob_terminal_sha": c_binding["getjob_terminal_sha"],
    }


def _make_terminal_inputs(tmp_path: Path) -> tuple:
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
    c_rank_paths = []
    for rank in range(8):
        rank_path = c_root / "shards" / f"rank-{rank:05d}" / "COMPLETED_SHARD.json"
        _write_json(rank_path, {"rank": rank, "status": "COMPLETED"})
        c_rank_paths.append(rank_path)
    c_marker = c_root / "COMPLETED_C_CALIBRATION.json"
    c_marker_payload = {
        "schema": "r142-stage-s-c-calibration-completion-v1",
        "status": "COMPLETED",
        "protocol_id": STAGE_S_PROTOCOL_ID,
        "substrate": "C",
        "run_id": "test-c-calibration",
        "calibration_result": c_result.name,
        "calibration_result_sha256": _sha(c_result),
        "calibration_result_schema": CALIBRATION_RESULT_SCHEMA,
        "calibration_seed": CALIBRATION_SEED,
        "world_size": CALIBRATION_WORLD_SIZE,
        "rank_markers": [
            f"shards/rank-{rank:05d}/COMPLETED_SHARD.json" for rank in range(8)
        ],
        "rank_marker_sha256": {
            path.relative_to(c_root).as_posix(): _sha(path) for path in c_rank_paths
        },
        "source": dict(SOURCE),
        "persistence": {"bundle_sha_file": "C_SHA256SUMS"},
    }
    _write_json(c_marker, c_marker_payload)
    _write_manifest(c_root, "C_SHA256SUMS", [c_result, c_marker, *c_rank_paths])

    acceptance, _ = _make_current_c_acceptance(tmp_path)
    c_config = tmp_path / "c_calibration_config.json"
    _write_json(c_config, {"evidence": dict(SOURCE)})
    b_binding = _make_registry_binding(
        tmp_path,
        substrate="B",
        controller_run_id="test-b-controller",
        application_run_id="test-b-calibration",
        artifact_dir=b_root,
    )
    c_binding = _make_registry_binding(
        tmp_path,
        substrate="C",
        controller_run_id="test-c-calibration",
        application_run_id="test-c-calibration",
        artifact_dir=c_root,
    )
    return b_result, b_marker, c_result, c_marker, acceptance, c_config, b_binding, c_binding


def _make_current_c_acceptance(tmp_path: Path) -> tuple[Path, Path]:
    """Build the current accepted-training schema with real small artifacts."""

    checkpoint_root = tmp_path / "current_c" / "checkpoints"
    model_paths: dict[int, Path] = {}
    for step in (1000, 3000, 6000, 10000):
        model = checkpoint_root / str(step) / "model.safetensors"
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_bytes(f"accepted-model-step-{step}".encode("ascii"))
        model_paths[step] = model
    checkpoint_manifest = _write_manifest(
        checkpoint_root,
        "SHA256SUMS",
        list(model_paths.values()),
    )

    completion = checkpoint_root / "COMPLETED_C_TRAINING.json"
    _write_json(
        completion,
        {
            "schema": "r142-stage-s-c-training-completion-v1",
            "status": "COMPLETED",
            "openpi_commit": SOURCE["openpi_commit"],
            "config_name": "pi05_libero",
            "seed": 42,
            "terminal_global_step": 10001,
            "checkpoint_steps": [1000, 3000, 6000, 10000],
            "checkpoint_audit": {"valid": True},
            "sha256sums": str(checkpoint_manifest),
            "sha256sums_sha256": _sha(checkpoint_manifest),
        },
    )

    log_root = tmp_path / "current_c" / "logs"
    log_file = log_root / "train.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text("terminal training evidence\n", encoding="utf-8")
    log_manifest = _write_manifest(log_root, "SHA256SUMS", [log_file])

    # Exercise the post-blackout calendar-day continuation accepted by the
    # terminal C lineage contract.
    accepted_run_id = "r142-stage-s-c-undertrained-20260904-r99"
    pipeline = tmp_path / "current_c" / "c_status" / "COMPLETED_C_PIPELINE.json"
    _write_json(
        pipeline,
        {
            "schema": "r142-stage-s-c-training-pipeline-v1",
            "status": "COMPLETED",
            "stage": "terminal",
            "run_id": accepted_run_id,
            "evidence_path": str(completion),
            "evidence_sha256": _sha(completion),
        },
    )

    acceptance = tmp_path / "current_c" / "ACCEPTED_C_TRAINING.json"
    _write_json(
        acceptance,
        {
            "schema": C_TRAINING_ACCEPTANCE_SCHEMA,
            "status": "ACCEPTED",
            "label": "WEAK_SUBSTRATE",
            "pai_terminal_status": "Succeeded",
            "accepted_run_id": accepted_run_id,
            "job_id": "dlctestcurrentc99",
            "source": dict(SOURCE),
            "checkpoint_root": str(checkpoint_root),
            "checkpoint_completion": str(completion),
            "checkpoint_sha256_manifest": str(checkpoint_manifest),
            "checkpoint_sha256_manifest_digest": _sha(checkpoint_manifest),
            "log_root": str(log_root),
            "log_sha256_manifest": str(log_manifest),
            "log_sha256_manifest_digest": _sha(log_manifest),
            "training_pipeline_completion": str(pipeline),
            "checkpoint_completion_sha256": _sha(completion),
            "checkpoint_steps": [1000, 3000, 6000, 10000],
            "full_reference_step": 30000,
            "no_interpolation": True,
            "artificial_degradation": False,
            "checkpoint_hashes": {
                f"{step}/model.safetensors": _sha(model_paths[step])
                for step in (1000, 3000, 6000, 10000)
            },
        },
    )
    return acceptance, model_paths[10000]


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

Accepted C source commits:
{SOURCE["stage_s_commit"]} {SOURCE["qpilots_commit"]} {SOURCE["openpi_commit"]} {SOURCE["libero_commit"]}
"""


def _freeze_inputs(tmp_path: Path):
    paths = _make_terminal_inputs(tmp_path)
    reports = freeze_calibration_reports(
        b_result=paths[0],
        b_completion_marker=paths[1],
        c_result=paths[2],
        c_completion_marker=paths[3],
        c_lineage=paths[4],
        **_external_kwargs(paths),
        b_report=tmp_path / "out" / "b" / "CALIBRATION_REPORT.json",
        c_report=tmp_path / "out" / "c" / "CALIBRATION_REPORT.json",
    )
    return paths, reports


def test_c_numeric_checkpoint_tie_break_is_shared_by_producer_and_freezer() -> None:
    rows = [
        {"setting": "step_3000", "pooled_success": 0.40},
        {"setting": "step_10000", "pooled_success": 0.60},
    ]
    assert _select_calibration_row(rows, substrate="C", target=0.50)["setting"] == "step_3000"
    assert min(rows, key=_calibration_selection_key)["setting"] == "step_3000"


def test_result_and_completion_must_share_directory(tmp_path: Path) -> None:
    paths = _make_terminal_inputs(tmp_path)
    detached_marker = tmp_path / "detached" / paths[1].name
    detached_marker.parent.mkdir(parents=True)
    detached_marker.write_text(paths[1].read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(CalibrationFreezeError, match="same directory"):
        freeze_calibration_reports(
            b_result=paths[0],
            b_completion_marker=detached_marker,
            c_result=paths[2],
            c_completion_marker=paths[3],
            c_lineage=paths[4],
            **_external_kwargs(paths),
            b_report=tmp_path / "detached-out-b.json",
            c_report=tmp_path / "detached-out-c.json",
        )


@pytest.mark.parametrize("substrate", ["B", "C"])
def test_residual_failed_calibration_marker_is_rejected(tmp_path: Path, substrate: str) -> None:
    paths = _make_terminal_inputs(tmp_path)
    marker = paths[1] if substrate == "B" else paths[3]
    (marker.parent / f"FAILED_{substrate}_CALIBRATION.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    with pytest.raises(CalibrationFreezeError, match="residual FAILED"):
        freeze_calibration_reports(
            b_result=paths[0],
            b_completion_marker=paths[1],
            c_result=paths[2],
            c_completion_marker=paths[3],
            c_lineage=paths[4],
            **_external_kwargs(paths),
            b_report=tmp_path / "failed-out-b.json",
            c_report=tmp_path / "failed-out-c.json",
        )


def test_c_requires_all_eight_rank_markers_and_hashes(tmp_path: Path) -> None:
    paths = _make_terminal_inputs(tmp_path)
    missing = paths[3].parent / "shards" / "rank-00007" / "COMPLETED_SHARD.json"
    missing.unlink()
    with pytest.raises(CalibrationFreezeError, match="SHA manifest member missing|rank"):
        freeze_calibration_reports(
            b_result=paths[0],
            b_completion_marker=paths[1],
            c_result=paths[2],
            c_completion_marker=paths[3],
            c_lineage=paths[4],
            **_external_kwargs(paths),
            b_report=tmp_path / "rank-out-b.json",
            c_report=tmp_path / "rank-out-c.json",
        )


def test_external_registry_binding_must_be_unique_and_terminal(tmp_path: Path) -> None:
    paths = _make_terminal_inputs(tmp_path)
    ledger = paths[7]["jobs_ledger"]
    line = ledger.read_text(encoding="utf-8")
    ledger.write_text(line + line, encoding="utf-8")
    with pytest.raises(CalibrationFreezeError, match="exactly one controller run binding"):
        freeze_calibration_reports(
            b_result=paths[0],
            b_completion_marker=paths[1],
            c_result=paths[2],
            c_completion_marker=paths[3],
            c_lineage=paths[4],
            **_external_kwargs(paths),
            b_report=tmp_path / "registry-out-b.json",
            c_report=tmp_path / "registry-out-c.json",
        )


def test_c_config_source_must_match_accepted_training(tmp_path: Path) -> None:
    paths = _make_terminal_inputs(tmp_path)
    config = json.loads(paths[5].read_text(encoding="utf-8"))
    config["evidence"]["stage_s_commit"] = "0" * 40
    _write_json(paths[5], config)
    with pytest.raises(CalibrationFreezeError, match="config source"):
        freeze_calibration_reports(
            b_result=paths[0],
            b_completion_marker=paths[1],
            c_result=paths[2],
            c_completion_marker=paths[3],
            c_lineage=paths[4],
            **_external_kwargs(paths),
            b_report=tmp_path / "config-out-b.json",
            c_report=tmp_path / "config-out-c.json",
        )


def test_terminal_selection_and_loader_schema(tmp_path: Path) -> None:
    paths, reports = _freeze_inputs(tmp_path)
    assert reports["B"]["selected_setting"] == "proximity_0.08m"
    assert reports["C"]["selected_setting"] == "step_10000"
    assert reports["B"]["selected_variant_root"].endswith("/proximity_0.08m")
    assert len(reports["C"]["selected_checkpoint_sha256"]) == 64
    assert reports["B"]["source_result_sha256"] == _sha(paths[0])
    assert reports["C"]["source_completion_marker_sha256"] == _sha(paths[3])


def test_current_accepted_training_schema_is_native_lineage(tmp_path: Path) -> None:
    paths = _make_terminal_inputs(tmp_path)
    acceptance, selected_model = _make_current_c_acceptance(tmp_path)
    reports = freeze_calibration_reports(
        b_result=paths[0],
        b_completion_marker=paths[1],
        c_result=paths[2],
        c_completion_marker=paths[3],
        c_lineage=acceptance,
        **_external_kwargs(paths),
        b_report=tmp_path / "accepted-out" / "b" / "CALIBRATION_REPORT.json",
        c_report=tmp_path / "accepted-out" / "c" / "CALIBRATION_REPORT.json",
    )
    assert reports["C"]["selected_setting"] == "step_10000"
    assert reports["C"]["selected_checkpoint"] == str(selected_model.resolve())
    assert reports["C"]["selected_checkpoint_sha256"] == _sha(selected_model)


def test_current_accepted_training_artifact_tamper_fails_closed(tmp_path: Path) -> None:
    paths = _make_terminal_inputs(tmp_path)
    acceptance, selected_model = _make_current_c_acceptance(tmp_path)
    selected_model.write_bytes(b"tampered-model")
    with pytest.raises(CalibrationFreezeError, match="SHA manifest digest mismatch|checkpoint .*hash"):
        freeze_calibration_reports(
            b_result=paths[0],
            b_completion_marker=paths[1],
            c_result=paths[2],
            c_completion_marker=paths[3],
            c_lineage=acceptance,
            **_external_kwargs(paths),
            b_report=tmp_path / "accepted-tamper-out-b.json",
            c_report=tmp_path / "accepted-tamper-out-c.json",
        )


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
            **_external_kwargs(paths),
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
            **_external_kwargs(paths),
            b_report=tmp_path / "leak-out-b.json",
            c_report=tmp_path / "leak-out-c.json",
        )

    paths = _make_terminal_inputs(tmp_path / "s2-s5")
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    payload["S2-S5"] = {"metric": 1}
    _write_json(paths[0], payload)
    with pytest.raises(CalibrationFreezeError, match="forbidden"):
        freeze_calibration_reports(
            b_result=paths[0],
            b_completion_marker=paths[1],
            c_result=paths[2],
            c_completion_marker=paths[3],
            c_lineage=paths[4],
            **_external_kwargs(paths),
            b_report=tmp_path / "s2-s5-out-b.json",
            c_report=tmp_path / "s2-s5-out-c.json",
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
            **_external_kwargs(paths),
            b_report=tmp_path / "symlink-out-b.json",
            c_report=tmp_path / "symlink-out-c.json",
        )


def test_training_lineage_tamper_fails_closed(tmp_path: Path) -> None:
    paths = _make_terminal_inputs(tmp_path)
    completion = paths[4]
    payload = json.loads(completion.read_text(encoding="utf-8"))
    payload["source"]["openpi_commit"] = "0" * 40
    _write_json(completion, payload)
    with pytest.raises(CalibrationFreezeError, match="OpenPI commit"):
        freeze_calibration_reports(
            b_result=paths[0],
            b_completion_marker=paths[1],
            c_result=paths[2],
            c_completion_marker=paths[3],
            c_lineage=paths[4],
            **_external_kwargs(paths),
            b_report=tmp_path / "tamper-out-b.json",
            c_report=tmp_path / "tamper-out-c.json",
        )


def test_protocol_freeze_and_acceptance_tamper_detection(tmp_path: Path) -> None:
    paths, _ = _freeze_inputs(tmp_path)
    repo = tmp_path / "protocol-repo"
    source_md = repo / "stage-s" / "PROTOCOL.md"
    # The declared commit must contain the exact protocol bytes.  The file
    # cannot honestly contain its own future commit hash, so the binding is
    # verified with git-show rather than a self-referential text field.
    source_md.parent.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "test"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"], check=True)
    source_md.write_text(_protocol_markdown("1" * 40), encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "stage-s/PROTOCOL.md"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "protocol content"], check=True)
    protocol_commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    acceptance_path = tmp_path / "logs" / "stage_s" / "protocol" / "FROZEN_PROTOCOL.json"
    payload = freeze_protocol(
        protocol_md=source_md,
        protocol_git_commit=protocol_commit,
        b_report=tmp_path / "out" / "b" / "CALIBRATION_REPORT.json",
        c_report=tmp_path / "out" / "c" / "CALIBRATION_REPORT.json",
        output_path=acceptance_path,
        repo_root=repo,
    )
    assert payload["schema"] == "r142-stage-s-protocol-acceptance-v1"
    assert payload["protocol_git_commit"] == protocol_commit
    assert payload["frozen_summary"] == FROZEN_SUMMARY
    assert payload["s4"] == FROZEN_SUMMARY["s4"]
    assert payload["s5"] == FROZEN_SUMMARY["s5"]
    assert set(payload["files"]) == {
        "PROTOCOL.md", "B_CALIBRATION_REPORT", "C_CALIBRATION_REPORT"
    }
    with pytest.raises(CalibrationFreezeError, match="overwrite existing"):
        freeze_protocol(
            protocol_md=source_md,
            protocol_git_commit=protocol_commit,
            b_report=tmp_path / "out" / "b" / "CALIBRATION_REPORT.json",
            c_report=tmp_path / "out" / "c" / "CALIBRATION_REPORT.json",
            output_path=acceptance_path,
            repo_root=repo,
        )
    read_frozen_protocol(
        acceptance_path,
        substrate="B",
        calibration_report=tmp_path / "out" / "b" / "CALIBRATION_REPORT.json",
    )
    read_frozen_protocol(
        acceptance_path,
        substrate="C",
        calibration_report=tmp_path / "out" / "c" / "CALIBRATION_REPORT.json",
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
    repo = tmp_path / "bad-repo"
    bad_md = repo / "stage-s" / "PROTOCOL.md"
    bad_md.parent.mkdir(parents=True)
    bad_md.write_text("S1 0.30 0.60\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "test"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "stage-s/PROTOCOL.md"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "bad protocol fixture"], check=True)
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    with pytest.raises(CalibrationFreezeError, match="requirements"):
        freeze_protocol(
            protocol_md=bad_md,
            protocol_git_commit=commit,
            b_report=tmp_path / "out" / "b" / "CALIBRATION_REPORT.json",
            c_report=tmp_path / "out" / "c" / "CALIBRATION_REPORT.json",
            output_path=tmp_path / "bad-out" / "FROZEN_PROTOCOL.json",
            repo_root=repo,
        )


def test_protocol_freeze_rejects_worktree_bytes_not_in_declared_commit(tmp_path: Path) -> None:
    _freeze_inputs(tmp_path)
    repo = tmp_path / "drift-repo"
    protocol = repo / "stage-s" / "PROTOCOL.md"
    protocol.parent.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "test"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"], check=True)
    protocol.write_text(_protocol_markdown("1" * 40), encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "stage-s/PROTOCOL.md"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "frozen bytes"], check=True)
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    protocol.write_text(_protocol_markdown("2" * 40), encoding="utf-8")
    with pytest.raises(CalibrationFreezeError, match="differs from the declared Git commit"):
        freeze_protocol(
            protocol_md=protocol,
            protocol_git_commit=commit,
            b_report=tmp_path / "out" / "b" / "CALIBRATION_REPORT.json",
            c_report=tmp_path / "out" / "c" / "CALIBRATION_REPORT.json",
            output_path=tmp_path / "drift-out" / "FROZEN_PROTOCOL.json",
            repo_root=repo,
        )
