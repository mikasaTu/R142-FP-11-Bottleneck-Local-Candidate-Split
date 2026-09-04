from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "stage_s_libero_c_calibration_pai.sh"
CONFIG = ROOT / "configs" / "pai" / "stage_s_c_calibration.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_c_launcher_is_shell_valid_and_non_submitting() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    text = SCRIPT.read_text(encoding="utf-8")
    assert "pai-job submit" not in text
    assert "CreateJob" not in text
    assert '"$PYTHON" -m torch.distributed.run --standalone' in text
    assert "torchrun --standalone" not in text
    assert "stage_s_gpu_rank_entry.py scripts/stage_s_libero_calibrate.py" in text
    assert "$OPENPI/src" in text
    assert "nvidia-smi --query-gpu=index --format=csv,noheader,nounits" in text
    assert 'CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPU_INDEXES[*]}")"' in text
    assert "--substrate C --mode prepare" in text
    assert text.index("--substrate C --mode prepare") < text.index("-m torch.distributed.run")
    assert "--nproc_per_node=\"$WORLD_SIZE\"" in text
    assert "--substrate C --mode shard" in text
    assert "COMPLETED_C_CALIBRATION.json" in text
    assert "C_SHA256SUMS" in text
    assert "WEAK_SUBSTRATE" in text


def test_config_binds_payload_graphics_env_and_compute_contract() -> None:
    config = _config()
    digest = hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    runtime = config["runtime"]
    evidence = config["evidence"]
    assert runtime["payload_sha256"] == digest
    assert runtime["command_file_sha256"] == digest
    assert evidence["validated_payload_sha256"] == digest
    assert config["resource_alias"] == "idle-a800-robot-stage-s-graphics-8gpu"
    assert runtime["pod_env"] == {
        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics"
    }
    assert runtime["command_file"] == (
        "/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/r142-stage-s-pai-20260902/"
        "stage_s_libero_c_calibration_pai.sh"
    )
    assert config["worker"] == {
        "count": 1,
        "gpu": 8,
        "cpu": 88,
        "memory": "1400Gi",
        "shared_memory": "1400Gi",
        "image": "dsw-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pai/modelscope:1.29.0-pytorch2.3.1tensorflow2.16.1-gpu-py311-cu121-ubuntu22.04",
    }
    assert runtime["output_mode"] == "resume"
    assert runtime["create_artifact_dir"] is True
    assert runtime["recursive_repair"] is False
    assert runtime["write_paths"] == ["{{ARTIFACT_DIR}}"]
    assert runtime["uid"] == runtime["gid"] == 2254


def test_c_calibration_axes_and_weak_labels_are_frozen() -> None:
    contract = _config()["evidence"]["calibration_contract"]
    assert contract["settings"] == [
        "step_1000",
        "step_3000",
        "step_6000",
        "step_10000",
    ]
    assert contract["task_ids"] == [0, 3, 6, 9]
    assert contract["initial_state_indices"] == list(range(8))
    assert contract["candidate_count"] == 8
    assert contract["trials_per_setting"] == 256
    assert contract["world_size"] == 8
    assert contract["persisted_row_fields"] == [
        "setting",
        "successes",
        "total",
        "pooled_success",
    ]
    assert contract["result_label"] == "WEAK_SUBSTRATE"
    assert contract["setting_labels"] == {
        "step_1000": "WEAK_SUBSTRATE",
        "step_3000": "WEAK_SUBSTRATE",
        "step_6000": "WEAK_SUBSTRATE",
        "step_10000": "WEAK_SUBSTRATE",
    }
    assert contract["completion_marker"] == "COMPLETED_C_CALIBRATION.json"
    assert contract["bundle_sha_file"] == "C_SHA256SUMS"
    text = SCRIPT.read_text(encoding="utf-8")
    for required in (
        "--checkpoint \"$C_TRAIN_DIR/1000\"",
        "--checkpoint \"$C_TRAIN_DIR/3000\"",
        "--checkpoint \"$C_TRAIN_DIR/6000\"",
        "--checkpoint \"$C_TRAIN_DIR/10000\"",
        "--world-size \"$WORLD_SIZE\"",
        "persisted_row_fields",
        "forbidden_trial_fields",
    ):
        assert required in text


def test_c_training_input_gate_requires_acceptance_manifest_and_full_state() -> None:
    contract = _config()["evidence"]["input_contract"]
    assert contract["acceptance_manifest"] == (
        "/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_s/"
        "c_status/ACCEPTED_C_TRAINING.json"
    )
    assert contract["acceptance_manifest_schema"] == "r142-stage-s-c-training-acceptance-v1"
    assert contract["accepted_terminal_status"] == "Succeeded"
    assert contract["training_run_id_source"] == "accepted_run_id"
    assert contract["training_job_id_source"] == "job_id"
    assert contract["expected_steps"] == [1000, 3000, 6000, 10000]
    assert contract["full_reference_step"] == 30000
    assert contract["full_training_state_required"] is True
    assert contract["no_interpolation"] is True
    assert contract["artificial_degradation"] is False
    assert contract["completion_and_sha_required"] is True
    assert contract["checkpoint_hashes"] == {
        "format": "accepted_manifest_relative_checkpoint_path_to_sha256",
        "required_paths": [
            "1000/model.safetensors",
            "3000/model.safetensors",
            "6000/model.safetensors",
            "10000/model.safetensors",
        ],
    }
    assert contract["checkpoint_sha"].endswith("/r142_stage_s_c/SHA256SUMS")
    assert contract["training_pipeline_completion_template"].endswith(
        "c_status/<accepted_run_id>/COMPLETED_C_PIPELINE.json"
    )
    assert contract["log_root_template"].endswith("c/<accepted_run_id>")
    assert contract["log_sha_template"].endswith("c/<accepted_run_id>/SHA256SUMS")
    text = SCRIPT.read_text(encoding="utf-8")
    for required in (
        "C_ACCEPTANCE_MANIFEST=",
        "accepted_run_id",
        "job_id",
        "pai_terminal_status",
        "checkpoint_hashes",
        "C acceptance checkpoint hash mismatch",
        "C_ACCEPTANCE_MANIFEST_GATE_PASS",
        "COMPLETED_C_TRAINING.json",
        "COMPLETED_C_PIPELINE.json",
        "sha256sum --check --quiet SHA256SUMS",
        "audit_c_checkpoint_schedule",
        "require_training_state=True",
        "no_interpolation",
        "artificial_degradation",
    ):
        assert required in text


def test_c_source_resume_fault_tolerance_and_blackout_contract() -> None:
    config = _config()
    evidence = config["evidence"]
    assert evidence["stage_s_source_commit"] == "59581b09ce974a7080aaf6660f7619be465ce19d"
    assert evidence["qpilots_commit"] == "eacf47b981e3b22357f8a74902f8dad8cfcfa375"
    assert evidence["openpi_commit"] == "54cbaee6ae0c010a1ed431871cdaa8f4684ac709"
    assert evidence["libero_commit"] == "f78abd68ee283de9f9be3c8f7e2a9ad60246e95c"
    assert evidence["compute_contract"] == {
        "worker_count": 1,
        "gpu_count": 8,
        "cpu_cores": 88,
        "memory_gib": 1400,
        "shared_memory_gib": 1400,
        "rank_world_size_fixed": 8,
        "resource_pool": "exp-robot",
        "resource_alias": "idle-a800-robot-stage-s-graphics-8gpu",
        "resource_id": "quota1ssrabud0bh",
    }
    assert evidence["resume_contract"] == {
        "same_artifact_dir": True,
        "same_run_id": True,
        "idempotent_rank_markers": True,
        "aggregate_requires_all_ranks": True,
        "fail_closed_on_preemption": True,
    }
    assert evidence["daily_no_job_windows"] == [
        {"start": "09:30", "end": "09:40", "timezone": "Asia/Shanghai"},
        {"start": "19:30", "end": "19:40", "timezone": "Asia/Shanghai"},
    ]
    fault = config["fault_tolerance"]
    assert fault["maximum_platform_restarts"] == 50
    assert fault["application_auto_resume"] is True
    assert fault["pai_automatic_fault_tolerance"] is True
    assert "--enable-job-restart=True" in fault["aimaster_args"]
    assert "--max-num-of-job-restart=50" in fault["aimaster_args"]


def test_c_no_secrets_or_legacy_storage_prefix() -> None:
    config = _config()
    assert config["runtime"]["secret_env_names"] == []
    serialized = CONFIG.read_text(encoding="utf-8") + SCRIPT.read_text(encoding="utf-8")
    assert "/mnt/cpfs/leon" not in serialized
    assert "/workspace/leon" not in serialized
