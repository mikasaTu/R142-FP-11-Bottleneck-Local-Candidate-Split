from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MAIN = ROOT / "scripts" / "stage_s_libero_main.py"


def _payload(name: str) -> Path:
    return ROOT / "scripts" / name


@pytest.mark.parametrize("substrate", ["b", "c"])
def test_main_config_binds_external_payload_and_idle_shape(substrate: str) -> None:
    config = json.loads((ROOT / "configs" / "pai" / f"stage_s_{substrate}_main.json").read_text())
    runtime = config["runtime"]
    payload = _payload(f"stage_s_{substrate}_main_pai.sh")
    observed = hashlib.sha256(payload.read_bytes()).hexdigest()
    assert runtime["command_file_sha256"] == observed
    assert runtime["payload_sha256"] == observed
    assert config["worker"] == {
        "count": 1,
        "gpu": 8,
        "cpu": 88,
        "memory": "1525Gi",
        "shared_memory": "1525Gi",
        "image": "dsw-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pai/modelscope:1.29.0-pytorch2.3.1tensorflow2.16.1-gpu-py311-cu121-ubuntu22.04",
    }
    assert runtime["pod_env"] == {"NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics"}
    assert config["resource_id"] == "quota1ssrabud0bh"
    assert config["provenance"] == {
        "contract_source_job_id": "dlckjz66iwcv38gw",
        "resource_source_job_id": "dlckjz66iwcv38gw",
        "source_role": "readback_reference",
        "submission_method": "cli_create",
        "pai_clone_performed": False,
    }
    assert config["evidence"]["success_gate"] == "persisted_completed_evaluation_result"
    assert config["evidence"]["validated_payload_sha256"] == observed


def test_payloads_are_same_bytes_and_fail_closed_before_calibration() -> None:
    b = _payload("stage_s_b_main_pai.sh").read_bytes()
    c = _payload("stage_s_c_main_pai.sh").read_bytes()
    assert b == c
    text = b.decode()
    for fragment in (
        "PAI_CANARY_RUN_ID",
        "REFUSED_DAILY_NO_JOB_WINDOW",
        "torch.distributed.run",
        "--validate-snapshots",
        "stage_s_libero_main_finalize.py",
        "COMPLETED_EVALUATION_RESULT.json",
        "SHA256SUMS",
        "source_commit",
    ):
        assert fragment in text
    assert "S2" not in text and "S3" not in text and "S4" not in text and "S5" not in text


def test_main_cli_shards_160_families_without_overlap() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC)
    shards: list[set[tuple[int, int]]] = []
    for rank in range(8):
        completed = subprocess.run(
            [
                "python3",
                str(MAIN),
                "--substrate",
                "A",
                "--output",
                "/tmp/stage-s-bc-test-output",
                "--dry-run",
                "--rank",
                str(rank),
                "--world-size",
                "8",
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        payload = json.loads(completed.stdout)
        shards.append({tuple(pair) for pair in payload["family_pairs"]})
    assert all(len(shard) == 20 for shard in shards)
    assert len(set.union(*shards)) == 160
    assert sum(len(shard) for shard in shards) == 160


def test_c_main_requires_weak_annotation_and_frozen_report() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC)
    without_annotation = subprocess.run(
        ["python3", str(MAIN), "--substrate", "C", "--output", "/tmp/stage-s-c", "--dry-run"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert without_annotation.returncode != 0
    assert "WEAK_SUBSTRATE" in without_annotation.stderr
    with_annotation = subprocess.run(
        [
            "python3",
            str(MAIN),
            "--substrate",
            "C",
            "--output",
            "/tmp/stage-s-c",
            "--dry-run",
            "--weak-substrate",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert with_annotation.returncode != 0
    assert "calibration-report" in with_annotation.stderr or "calibration-report" in with_annotation.stdout


def test_finalizer_rejects_partial_top_level_bundle(tmp_path: Path) -> None:
    import importlib.util

    path = ROOT / "scripts" / "stage_s_libero_main_finalize.py"
    spec = importlib.util.spec_from_file_location("stage_s_main_finalize", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "SHA256SUMS").write_text("partial\n")
    with pytest.raises(module.MainEvaluationError, match="partial top-level"):
        module.finalize(
            tmp_path,
            substrate="B",
            run_id="r142-stage-s-b-main-20260903-r1",
            source_commit="d85bdfb2e5f4d934de2dc4a754d0fb2df30b4246",
            calibration_report="/frozen/B/CALIBRATION_REPORT.json",
        )
