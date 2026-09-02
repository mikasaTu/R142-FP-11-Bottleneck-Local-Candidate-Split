from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "stage_s_libero_b_calibration_pai.sh"
CONFIG = ROOT / "configs" / "pai" / "stage_s_b_calibration.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_b_launcher_is_shell_valid_and_non_submitting() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    text = SCRIPT.read_text(encoding="utf-8")
    assert "pai-job submit" not in text
    assert "CreateJob" not in text
    assert "torchrun --standalone" in text
    assert "--nproc_per_node=\"$WORLD_SIZE\"" in text
    assert "--world-size \"$WORLD_SIZE\"" in text
    assert "COMPLETED_B_CALIBRATION.json" in text
    assert "B_SHA256SUMS" in text


def test_config_binds_launcher_hash_and_registry_contract() -> None:
    config = _config()
    digest = hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    runtime = config["runtime"]
    evidence = config["evidence"]
    assert runtime["payload_sha256"] == digest
    assert runtime["command_file_sha256"] == digest
    assert evidence["validated_payload_sha256"] == digest
    assert config["resource_alias"] == "idle-a800-robot-native3-8gpu"
    assert config["worker"] == {
        "count": 1,
        "gpu": 8,
        "cpu": 88,
        "memory": "1525Gi",
        "shared_memory": "1525Gi",
        "image": "dsw-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pai/modelscope:1.29.0-pytorch2.3.1tensorflow2.16.1-gpu-py311-cu121-ubuntu22.04",
    }
    assert runtime["output_mode"] == "resume"
    assert runtime["create_artifact_dir"] is True
    assert runtime["recursive_repair"] is False
    assert runtime["write_paths"] == ["{{ARTIFACT_DIR}}"]
    assert runtime["uid"] == runtime["gid"] == 2254
    assert config["storage"]["output_root"] == (
        "/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/"
        "pai_registry/r142_stage_s/b_calibration"
    )


def test_calibration_axes_and_aggregate_only_persistence_are_frozen() -> None:
    contract = _config()["evidence"]["calibration_contract"]
    assert contract["settings"] == [
        "proximity_0.06m",
        "proximity_0.08m",
        "proximity_0.10m",
        "proximity_0.12m",
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
    assert contract["completion_marker"] == "COMPLETED_B_CALIBRATION.json"

    text = SCRIPT.read_text(encoding="utf-8")
    # The launcher records only source/provenance/compute metadata and the
    # runtime's four aggregate counters; no trial trace is assembled here.
    assert '"gpu_count": 8' in text
    assert '"state_format": "torch.save(sim.get_state().flatten())"' in text
    assert "--variant-root \"$B_VARIANT_RUN_ROOT/variants/proximity_0.06m\"" in text
    assert "--variant-root \"$B_VARIANT_RUN_ROOT/variants/proximity_0.08m\"" in text
    assert "--variant-root \"$B_VARIANT_RUN_ROOT/variants/proximity_0.10m\"" in text
    assert "--variant-root \"$B_VARIANT_RUN_ROOT/variants/proximity_0.12m\"" in text


def test_r7_completion_sha_and_flattened_state_input_are_mandatory() -> None:
    config = _config()["evidence"]["input_contract"]
    assert config["variant_run_id"] == "r142-stage-s-b-variants-20260903-r7"
    assert config["completion_and_sha_required"] is True
    assert config["requires_flattened_state"] is True
    assert config["state_format"] == "torch.save(sim.get_state().flatten())"
    assert config["old_init_reused"] is False
    text = SCRIPT.read_text(encoding="utf-8")
    for required in (
        "COMPLETED_B_VARIANTS.json",
        "SHA256SUMS",
        "validate_b_calibration_variants",
        "state_format",
        "old_init_reused",
        "matrix_sha256",
    ):
        assert required in text
    assert "B variant completion marker is not completed" in text
    assert "r7 consumed artifact escapes variant root" in text


def test_source_provenance_compute_and_daily_resume_contract() -> None:
    config = _config()
    evidence = config["evidence"]
    assert evidence["stage_s_source_commit"] == "afe353bbc5997355f35cb0c77c5446fd4df5f1e3"
    assert evidence["qpilots_commit"] == "eacf47b981e3b22357f8a74902f8dad8cfcfa375"
    assert evidence["openpi_commit"] == "54cbaee6ae0c010a1ed431871cdaa8f4684ac709"
    assert evidence["libero_commit"] == "f78abd68ee283de9f9be3c8f7e2a9ad60246e95c"
    assert evidence["compute_contract"] == {
        "worker_count": 1,
        "gpu_count": 8,
        "cpu_cores": 88,
        "memory_gib": 1525,
        "shared_memory_gib": 1525,
        "rank_world_size_fixed": 8,
        "resource_pool": "exp-robot",
        "resource_alias": "idle-a800-robot-native3-8gpu",
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


def test_no_secrets_and_no_legacy_storage_prefix() -> None:
    config = _config()
    assert config["runtime"]["secret_env_names"] == []
    serialized = CONFIG.read_text(encoding="utf-8") + SCRIPT.read_text(encoding="utf-8")
    assert "/mnt/cpfs/leon" not in serialized
    assert "/workspace/leon" not in serialized
