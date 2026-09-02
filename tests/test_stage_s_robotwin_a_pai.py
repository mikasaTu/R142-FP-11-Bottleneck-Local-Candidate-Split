from __future__ import annotations

import json
import hashlib
import subprocess
from pathlib import Path

import pytest

import r142_stage_s.frozen_protocol as frozen_protocol_module
import r142_stage_s.asset_acceptance as asset_acceptance_module
from r142_stage_s.asset_acceptance import (
    AssetAcceptanceError,
    EXPECTED_MODEL_REVISION,
    EXPECTED_SOURCE_COMMITS,
    load_accepted_asset_preflight,
)
from r142_stage_s.frozen_protocol import (
    EXPECTED_BUDGET,
    EXPECTED_SEED_RULE,
    EXPECTED_TASKS,
    EXPECTED_THRESHOLDS,
    FrozenProtocolError,
    load_frozen_protocol,
)
from scripts.stage_s_robotwin_finalize import EvaluationBundleError, finalize
from scripts.stage_s_robotwin_main import _write_rank_completion


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "stage_s_robotwin_a_pai.sh"
SERVER = ROOT / "scripts" / "stage_s_robotwin_evo_server.py"
FINALIZER = ROOT / "scripts" / "stage_s_robotwin_finalize.py"
CONFIG = ROOT / "configs" / "pai" / "stage_s_robotwin_a.json"


def test_launcher_is_valid_shell_and_binds_all_eight_pairs() -> None:
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "WORLD_SIZE=8" in text
    assert "EXPECTED_GPUS\" -eq 8" in text
    assert "for rank in $(seq 0 7)" in text
    assert "CUDA_VISIBLE_DEVICES=\"$rank\"" in text
    assert "stage_s_robotwin_evo_server.py" in text
    assert "stage_s_robotwin_main.py" in text
    assert "--world-size \"$WORLD_SIZE\"" in text
    assert "COMPLETED_EVALUATION_RESULT.json" in text
    assert "sha256sum --check --quiet SHA256SUMS" in text
    assert "09[3][0-9]|19[3][0-9]" in text
    assert "2254:2254" in text
    assert 'REQUIRED_RUNTIME_REPO="$ROOT/code/r142-stage-s-a-runtime-20260903"' in text
    assert "r142-stage-s-runtime-20260902" not in text
    assert "FROZEN_SOURCE_COMMIT=\"047f15c5cbad99e76ac47e2ce608a40bae22c49e\"" in text
    assert "ACCEPTED_ASSET_PREFLIGHT_PATH=\"$ROOT/stage_s/protocol/ACCEPTED_A_ASSET_PREFLIGHT.json\"" in text
    assert "ASSET_PREFLIGHT_RUN_ID" not in text
    assert "r142-stage-s-a-assets-20260902-r15" not in text
    assert "COMPLETED_ASSET_PREFLIGHT.json" in text
    assert "asset_acceptance.py" in text
    assert "accepted run id" in text.lower()
    assert "checkpoint-dir \"$CHECKPOINT_DIR\"" in text
    assert "stage_s/protocol/FROZEN_PROTOCOL.json" in text
    assert "frozen_protocol.py" in text
    assert "--frozen-protocol \"$FROZEN_PROTOCOL_PATH\"" in text
    assert "synthetic" not in text.lower() or "mock rollout" in text.lower()


def test_config_freezes_real_resource_and_evaluation_contract() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["resource_alias"] == "idle-a800-robot-stage-s-graphics-8gpu"
    assert config["resource_id"] == "quota1ssrabud0bh"
    assert config["quota"] == "exp-robot"
    assert config["worker"] == {
        "count": 1,
        "gpu": 8,
        "cpu": 88,
        "memory": "1525Gi",
        "shared_memory": "1525Gi",
        "image": config["worker"]["image"],
    }
    assert config["shard"]["world_size"] == 8
    assert config["shard"]["server_processes"] == 8
    assert config["shard"]["client_processes"] == 8
    assert config["runtime_contract"]["terminal_episode_count"] == 5120
    assert config["runtime_contract"]["families_per_task"] == 16
    assert config["runtime_contract"]["candidates_per_family"] == 32
    assert config["runtime_contract"]["synthetic_rollouts"] is False
    assert config["runtime_contract"]["expert_trajectory"] is False
    assert config["evidence"]["success_gate"] == "persisted_completed_evaluation_result"
    assert config["runtime"]["runtime_repo"] == (
        "/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/r142-stage-s-a-runtime-20260903"
    )
    assert config["runtime"]["source_commit"] == (
        "047f15c5cbad99e76ac47e2ce608a40bae22c49e"
    )
    assert config["runtime"]["frozen_protocol_path"] == (
        "/mnt/cpfs/zbl-cpfs-new/USERS/leon/stage_s/protocol/FROZEN_PROTOCOL.json"
    )
    assert config["runtime"]["command_file"] == (
        "/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/r142-stage-s-pai-20260902/"
        "stage_s_robotwin_a_pai.sh"
    )
    assert config["assets"]["asset_preflight_acceptance"]["manifest"] == (
        "/mnt/cpfs/zbl-cpfs-new/USERS/leon/stage_s/protocol/ACCEPTED_A_ASSET_PREFLIGHT.json"
    )
    assert config["assets"]["asset_preflight_acceptance"]["completion_marker"] == (
        "COMPLETED_ASSET_PREFLIGHT.json"
    )
    assert config["assets"]["asset_preflight_acceptance"]["integrity_manifest"] == "SHA256SUMS"
    assert "r142-stage-s-a-assets-20260902-r15" not in json.dumps(config)
    assert config["fault_tolerance"]["maximum_platform_restarts"] == 50
    assert config["fault_tolerance"]["launcher_attempts"] == 1
    for field, path in (
        ("command_file_sha256", LAUNCHER),
        ("server_wrapper_sha256", SERVER),
        ("aggregate_verifier_sha256", FINALIZER),
    ):
        assert config["runtime"][field] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert config["runtime"]["asset_acceptance_helper_sha256"] == hashlib.sha256(
        (ROOT / "src" / "r142_stage_s" / "asset_acceptance.py").read_bytes()
    ).hexdigest()
    assert config["evidence"]["validated_payload_sha256"] == hashlib.sha256(
        LAUNCHER.read_bytes()
    ).hexdigest()


def test_server_wrapper_keeps_released_inference_and_external_dispatch() -> None:
    text = SERVER.read_text(encoding="utf-8")
    assert "EvoServerReplayDispatcher" in text
    assert "control_response" in text
    assert "infer_from_json_dict" in text
    assert "load_model_and_normalizer" in text
    assert "one_server_one_client_one_gpu_one_port" in text
    assert "EvoServerReplayDispatcher" in text
    assert "Evo1_server.py" in text


def test_incomplete_or_partial_root_never_becomes_evaluation_completion(tmp_path: Path) -> None:
    (tmp_path / "FIRST_WORK.json").write_text('{"status":"FIRST_WORK"}\n', encoding="utf-8")
    with pytest.raises(EvaluationBundleError, match="frozen protocol"):
        finalize(tmp_path, run_id="stage-s-a-test")
    assert not (tmp_path / "COMPLETED_EVALUATION_RESULT.json").exists()
    assert not (tmp_path / "SHA256SUMS").exists()


def test_finalizer_source_contains_no_synthetic_fallback() -> None:
    text = FINALIZER.read_text(encoding="utf-8").lower()
    assert "synthetic_rollouts" in text
    assert "expert_trajectory" in text
    assert "official eval_success or step_lim" in text
    assert "partial top-level completion bundle" in text


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _protocol_fixture(tmp_path: Path) -> tuple[Path, dict]:
    protocol_dir = tmp_path / "stage_s" / "protocol"
    protocol_dir.mkdir(parents=True)
    protocol_md = protocol_dir / "PROTOCOL.md"
    b_report = tmp_path / "b_calibration_report.json"
    c_report = tmp_path / "c_calibration_report.json"
    protocol_md.write_text("frozen protocol bytes\n", encoding="utf-8")
    b_report.write_text('{"status":"COMPLETED","substrate":"B"}\n', encoding="utf-8")
    c_report.write_text('{"status":"COMPLETED","substrate":"C"}\n', encoding="utf-8")
    authority = {
        "status": "FROZEN",
        "protocol_git_commit": "a" * 40,
        "files": {
            "PROTOCOL.md": {"path": str(protocol_md), "sha256": _sha256(protocol_md)},
            "B_CALIBRATION_REPORT": {"path": str(b_report), "sha256": _sha256(b_report)},
            "C_CALIBRATION_REPORT": {"path": str(c_report), "sha256": _sha256(c_report)},
        },
        "frozen_summary": {
            "thresholds": dict(EXPECTED_THRESHOLDS),
            "seed_plan": {
                "seed_base": 14211,
                "candidate_seed_rule": EXPECTED_SEED_RULE,
            },
            "tasks": list(EXPECTED_TASKS),
            "budget": {
                "task_count": EXPECTED_BUDGET["task_count"],
                "families_per_task": EXPECTED_BUDGET["families_per_task"],
                "candidates_per_family": EXPECTED_BUDGET["candidates_per_family"],
                "terminal_episode_count": EXPECTED_BUDGET["terminal_episode_count"],
                "world_size": EXPECTED_BUDGET["world_size"],
            },
        },
    }
    path = protocol_dir / "FROZEN_PROTOCOL.json"
    path.write_text(json.dumps(authority, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path, authority


def test_frozen_protocol_reader_returns_complete_fingerprint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path, _ = _protocol_fixture(tmp_path)
    monkeypatch.setattr(frozen_protocol_module, "_CPFS_ROOT", tmp_path)
    fingerprint = load_frozen_protocol(path)
    assert fingerprint["status"] == "FROZEN"
    assert fingerprint["protocol_git_commit"] == "a" * 40
    assert len(fingerprint["protocol_json_sha256"]) == 64
    assert fingerprint["protocol_md_sha256"] == _sha256(tmp_path / "stage_s" / "protocol" / "PROTOCOL.md")
    assert set(fingerprint["calibration_reports"]) == {"B", "C"}
    assert fingerprint["frozen_summary"]["tasks"] == list(EXPECTED_TASKS)
    assert fingerprint["frozen_summary"]["budget"] == EXPECTED_BUDGET


@pytest.mark.parametrize("tamper", ("status", "protocol", "b_report", "summary"))
def test_frozen_protocol_reader_rejects_missing_or_tampered_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    path, authority = _protocol_fixture(tmp_path)
    monkeypatch.setattr(frozen_protocol_module, "_CPFS_ROOT", tmp_path)
    if tamper == "status":
        authority["status"] = "DRAFT"
        path.write_text(json.dumps(authority, sort_keys=True) + "\n", encoding="utf-8")
    elif tamper == "protocol":
        (tmp_path / "stage_s" / "protocol" / "PROTOCOL.md").write_text("altered\n", encoding="utf-8")
    elif tamper == "b_report":
        Path(authority["files"]["B_CALIBRATION_REPORT"]["path"]).write_text("altered\n", encoding="utf-8")
    elif tamper == "summary":
        authority["frozen_summary"]["budget"]["world_size"] = 4
        path.write_text(json.dumps(authority, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(FrozenProtocolError):
        load_frozen_protocol(path)


def test_rank_completion_persists_protocol_fingerprint() -> None:
    fingerprint = {
        "protocol_git_commit": "b" * 40,
        "protocol_json_sha256": "c" * 64,
        "protocol_md_sha256": "d" * 64,
        "calibration_reports": {
            "B": {"path": "/mnt/cpfs/zbl-cpfs-new/b.json", "sha256": "e" * 64},
            "C": {"path": "/mnt/cpfs/zbl-cpfs-new/c.json", "sha256": "f" * 64},
        },
    }
    # The writer is tested only for metadata persistence here; family data
    # verification remains the responsibility of the real finalizer suite.
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        accepted = {
            "accepted_run_id": "asset-run",
            "accepted_job_id": "dlc-terminal",
            "completion_sha256": "1" * 64,
            "asset_sha256sums_sha256": "2" * 64,
            "model_sha256sums_sha256": "3" * 64,
            "source_commits": {
                "evo": "4" * 40,
                "robotwin": "5" * 40,
                "curobo": "6" * 40,
            },
        }
        result = _write_rank_completion(Path(directory), 0, 8, [], fingerprint, accepted)
        marker = json.loads(
            (Path(directory) / "COMPLETED_A_RANK-0000.json").read_text(encoding="utf-8")
        )
    assert result["frozen_protocol"] == fingerprint
    assert marker["protocol_git_commit"] == "b" * 40
    assert marker["protocol_json_sha256"] == "c" * 64
    assert marker["protocol_md_sha256"] == "d" * 64
    assert marker["calibration_report_sha256"] == {"B": "e" * 64, "C": "f" * 64}
    assert marker["accepted_asset_preflight"] == accepted
    assert marker["accepted_asset_job_id"] == "dlc-terminal"


def test_main_and_finalizer_require_protocol_fingerprint() -> None:
    main_text = (ROOT / "scripts" / "stage_s_robotwin_main.py").read_text(encoding="utf-8")
    finalizer_text = FINALIZER.read_text(encoding="utf-8")
    for text in (main_text, finalizer_text):
        assert "frozen_protocol" in text
        assert "protocol_git_commit" in text
        assert "protocol_json_sha256" in text
        assert "protocol_md_sha256" in text
        assert "calibration_report_sha256" in text
        assert "--frozen-protocol" in text
        assert "accepted_asset_preflight" in text
        assert "accepted_asset_job_id" in text
        assert "--accepted-asset-preflight" in text


def _accepted_asset_fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    """Build a byte-complete accepted pointer without creating a PAI job."""

    output_dir = tmp_path / "USERS" / "leon" / "logs" / "r142_fp11_stage_s" / "assets" / "asset-run"
    protocol_dir = tmp_path / "USERS" / "leon" / "stage_s" / "protocol"
    model_dir = tmp_path / "USERS" / "leon" / "cache" / "r142_stage_s" / "models" / EXPECTED_MODEL_REVISION
    output_dir.mkdir(parents=True)
    protocol_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    (model_dir / "weights.bin").write_bytes(b"pinned checkpoint bytes\n")
    model_sums = f"{_sha256(model_dir / 'weights.bin')}  weights.bin\n"
    (model_dir / "SHA256SUMS").write_text(model_sums, encoding="utf-8")

    marker = {
        "status": "COMPLETED",
        "job_id": "dlc-terminal",
        "gpus": 8,
        "model_revision": EXPECTED_MODEL_REVISION,
        "model_sha256sums_sha256": _sha256(model_dir / "SHA256SUMS"),
        "robotwin_commit": EXPECTED_SOURCE_COMMITS["robotwin"],
        "evo_commit": EXPECTED_SOURCE_COMMITS["evo"],
        "curobo_commit": EXPECTED_SOURCE_COMMITS["curobo"],
    }
    marker_path = output_dir / "COMPLETED_ASSET_PREFLIGHT.json"
    marker_path.write_text(json.dumps(marker, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    first_work = output_dir / "FIRST_WORK.json"
    first_work.write_text('{"status":"FIRST_WORK"}\n', encoding="utf-8")
    asset_sums_text = "\n".join(
        f"{_sha256(item)}  {item.name}" for item in (marker_path, first_work)
    ) + "\n"
    asset_sums = output_dir / "SHA256SUMS"
    asset_sums.write_text(asset_sums_text, encoding="utf-8")
    record = {
        "status": "ACCEPTED",
        "accepted_run_id": "asset-run",
        "accepted_job_id": "dlc-terminal",
        "terminal_pai_state": "Succeeded",
        "output_dir": str(output_dir),
        "completion_marker": "COMPLETED_ASSET_PREFLIGHT.json",
        "completion_sha256": _sha256(marker_path),
        "asset_sha256sums_sha256": _sha256(asset_sums),
        "model_dir": str(model_dir),
        "model_revision": EXPECTED_MODEL_REVISION,
        "model_sha256sums_sha256": _sha256(model_dir / "SHA256SUMS"),
        "source_commits": dict(EXPECTED_SOURCE_COMMITS),
    }
    pointer = protocol_dir / "ACCEPTED_A_ASSET_PREFLIGHT.json"
    pointer.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return pointer, model_dir, record


def test_accepted_asset_reader_rechecks_terminal_lineage_and_live_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pointer, model_dir, _ = _accepted_asset_fixture(tmp_path)
    monkeypatch.setattr(asset_acceptance_module, "_CPFS_ROOT", tmp_path)
    fingerprint = load_accepted_asset_preflight(pointer, checkpoint_dir=model_dir)
    assert fingerprint["status"] == "ACCEPTED"
    assert fingerprint["accepted_run_id"] == "asset-run"
    assert fingerprint["accepted_job_id"] == "dlc-terminal"
    assert fingerprint["terminal_pai_state"] == "Succeeded"
    assert fingerprint["model_revision"] == EXPECTED_MODEL_REVISION
    assert fingerprint["source_commits"] == EXPECTED_SOURCE_COMMITS
    assert len(fingerprint["accepted_manifest_sha256"]) == 64


def test_accepted_asset_reader_rejects_running_job_and_tampered_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pointer, model_dir, record = _accepted_asset_fixture(tmp_path)
    monkeypatch.setattr(asset_acceptance_module, "_CPFS_ROOT", tmp_path)
    record["terminal_pai_state"] = "Running"
    pointer.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(AssetAcceptanceError, match="terminal PAI state"):
        load_accepted_asset_preflight(pointer, checkpoint_dir=model_dir)

    pointer, model_dir, _ = _accepted_asset_fixture(tmp_path / "tampered")
    monkeypatch.setattr(asset_acceptance_module, "_CPFS_ROOT", tmp_path / "tampered")
    marker_path = Path(json.loads(pointer.read_text(encoding="utf-8"))["output_dir"]) / "COMPLETED_ASSET_PREFLIGHT.json"
    marker_path.write_text(marker_path.read_text(encoding="utf-8").replace("}\n", "} \n"), encoding="utf-8")
    with pytest.raises(AssetAcceptanceError, match="checksum mismatch"):
        load_accepted_asset_preflight(pointer, checkpoint_dir=model_dir)


def test_accepted_asset_reader_rejects_missing_stable_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(asset_acceptance_module, "_CPFS_ROOT", tmp_path)
    missing = tmp_path / "USERS" / "leon" / "stage_s" / "protocol" / "ACCEPTED_A_ASSET_PREFLIGHT.json"
    with pytest.raises(AssetAcceptanceError, match="missing"):
        load_accepted_asset_preflight(missing, checkpoint_dir=tmp_path)
