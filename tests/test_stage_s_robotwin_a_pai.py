from __future__ import annotations

import json
import hashlib
import subprocess
from pathlib import Path

import pytest

from scripts.stage_s_robotwin_finalize import EvaluationBundleError, finalize


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "stage_s_robotwin_a_pai.sh"
SERVER = ROOT / "scripts" / "stage_s_robotwin_evo_server.py"
FINALIZER = ROOT / "scripts" / "stage_s_robotwin_finalize.py"
CONFIG = ROOT / "configs" / "pai" / "stage_s_robotwin_a.json"
ASSET_LAUNCHER = ROOT / "scripts" / "stage_s_asset_preflight.sh"
ASSET_CONFIG = ROOT / "configs" / "pai" / "stage_s_asset_preflight.json"


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
    assert "FROZEN_SOURCE_COMMIT=\"c2bd51db6de0e22d09827d06460cbac8d47bb6ae\"" in text
    assert "ASSET_PREFLIGHT_RUN_ID=\"r142-stage-s-a-assets-20260902-r15\"" in text
    assert "COMPLETED_ASSET_PREFLIGHT.json" in text
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
        "c2bd51db6de0e22d09827d06460cbac8d47bb6ae"
    )
    assert config["runtime"]["command_file"] == (
        "/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/r142-stage-s-pai-20260902/"
        "stage_s_robotwin_a_pai.sh"
    )
    assert config["assets"]["asset_preflight_required"]["run_id"] == (
        "r142-stage-s-a-assets-20260902-r15"
    )
    assert config["assets"]["asset_preflight_required"]["completion_marker"] == (
        "COMPLETED_ASSET_PREFLIGHT.json"
    )
    assert config["assets"]["asset_preflight_required"]["integrity_manifest"] == "SHA256SUMS"
    assert config["fault_tolerance"]["maximum_platform_restarts"] == 50
    assert config["fault_tolerance"]["launcher_attempts"] == 1
    for field, path in (
        ("command_file_sha256", LAUNCHER),
        ("server_wrapper_sha256", SERVER),
        ("aggregate_verifier_sha256", FINALIZER),
    ):
        assert config["runtime"][field] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert config["evidence"]["validated_payload_sha256"] == hashlib.sha256(
        LAUNCHER.read_bytes()
    ).hexdigest()


def test_asset_preflight_flash_attn_install_avoids_cross_filesystem_rename() -> None:
    subprocess.run(["bash", "-n", str(ASSET_LAUNCHER)], check=True)
    text = ASSET_LAUNCHER.read_text(encoding="utf-8")
    assert 'FLASH_ATTN_TMP="$PIP_CACHE/flash-attn-tmp/$RUN_ID"' in text
    assert 'TMPDIR="$FLASH_ATTN_TMP" PIP_NO_CACHE_DIR=1 MAX_JOBS=32' in text
    assert "--no-cache-dir flash-attn --no-build-isolation" in text
    assert 'stat -c \'%d\' "$FLASH_ATTN_TMP"' in text
    assert 'stat -c \'%d\' "$PIP_CACHE"' in text
    assert '"flash_attn_install"' in text
    assert '"tmpdir_under_new_root": str(flash_tmp).startswith' in text
    assert "export HOME" not in text
    assert "HOME=" not in text

    config = json.loads(ASSET_CONFIG.read_text(encoding="utf-8"))
    runtime = config["runtime"]
    evidence = config["evidence"]
    digest = hashlib.sha256(ASSET_LAUNCHER.read_bytes()).hexdigest()
    assert runtime["command_file_sha256"] == digest
    assert runtime["payload_sha256"] == digest
    assert config["evidence"]["validated_payload_sha256"] == digest
    assert evidence["first_work_evidence_path"].endswith(
        "/assets/{{RUN_ID}}/FIRST_WORK.json"
    )
    assert evidence["explicit_user_resource_authorization"]["scope"] == "{{RUN_ID}}"
    assert runtime["flash_attn_install"] == {
        "package": "flash-attn",
        "cache_policy": "disabled",
        "pip_flag": "--no-cache-dir",
        "tmpdir_template": "/mnt/cpfs/zbl-cpfs-new/USERS/leon/cache/r142_stage_s/pip/flash-attn-tmp/<run_id>",
        "tmpdir_same_filesystem_as_pip_cache": True,
        "home_unchanged": True,
    }
    assert config["evidence"]["asset_manifest_contract"] == {
        "completion_marker": "COMPLETED_ASSET_PREFLIGHT.json",
        "integrity_manifest": "SHA256SUMS",
        "required_flash_attn_install": {
            "cache_policy": "disabled",
            "pip_no_cache_dir": True,
            "tmpdir_under_new_root": True,
            "home_unchanged": True,
        },
    }


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
    with pytest.raises(EvaluationBundleError, match="required JSON file|rank"):
        finalize(tmp_path, run_id="stage-s-a-test")
    assert not (tmp_path / "COMPLETED_EVALUATION_RESULT.json").exists()
    assert not (tmp_path / "SHA256SUMS").exists()


def test_finalizer_source_contains_no_synthetic_fallback() -> None:
    text = FINALIZER.read_text(encoding="utf-8").lower()
    assert "synthetic_rollouts" in text
    assert "expert_trajectory" in text
    assert "official eval_success or step_lim" in text
    assert "partial top-level completion bundle" in text
