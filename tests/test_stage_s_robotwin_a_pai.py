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
    assert "synthetic" not in text.lower() or "mock rollout" in text.lower()


def test_config_freezes_real_resource_and_evaluation_contract() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
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
