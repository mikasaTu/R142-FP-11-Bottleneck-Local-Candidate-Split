from __future__ import annotations

import base64
import hashlib
import io
import json
import importlib.util
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

from r142_stage_s.openpi_c import (
    OPENPI_COMMIT,
    OPENPI_CONFIG_NAME,
    DEFAULT_OPENPI_PYTHON,
    C_NUM_WORKERS,
    C_RETAIN_STEPS,
    C_TRAINING_STEPS,
    TRAINING_TERMINAL_NAME,
    PI05_BASE_OBJECT_COUNT,
    PI05_BASE_OBJECTS,
    PI05_BASE_TOTAL_BYTES,
    assert_outside_blackout,
    build_c_chain_contract,
    build_conversion_contract,
    build_patched_training_command,
    download_base_checkpoint,
    expected_base_manifest,
    finalize_training,
    manifest_from_gcs_listing,
    run_conversion,
    validate_base_manifest,
)

import r142_stage_s.openpi_c as openpi_c


def _load_worker_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "stage_s_libero_c_train_worker.py"
    spec = importlib.util.spec_from_file_location("stage_s_c_train_worker_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_gcs_manifest_has_exact_cardinality_and_bytes() -> None:
    manifest = expected_base_manifest()
    assert len(PI05_BASE_OBJECTS) == PI05_BASE_OBJECT_COUNT == 29
    assert manifest["object_count"] == 29
    assert manifest["total_bytes"] == PI05_BASE_TOTAL_BYTES == 12_441_749_581
    assert validate_base_manifest(manifest)
    assert manifest["objects"][0]["name"] == "assets/arx/norm_stats.json"
    assert manifest["objects"][-1]["name"].endswith("manifest.ocdbt")


def test_live_listing_must_match_frozen_object_set() -> None:
    manifest = expected_base_manifest()
    payload = {
        "items": [
            {
                "name": row["gcs_name"],
                "size": str(row["size"]),
                "md5Hash": row["md5_base64"],
                "crc32c": row["crc32c"],
                "generation": row["generation"],
                "updated": row["updated"],
            }
            for row in manifest["objects"]
        ]
    }
    observed = manifest_from_gcs_listing(payload)
    assert observed["source"]["live_listing_verified"] is True
    assert observed["manifest_sha256"]
    assert all(row["sha256"] is None for row in observed["objects"])
    payload["items"].pop()
    with pytest.raises(ValueError, match="exactly the frozen 29 objects"):
        manifest_from_gcs_listing(payload)


class _Response(io.BytesIO):
    status = 200

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def test_downloader_persists_sha_and_is_idempotent(tmp_path: Path) -> None:
    contents = {"assets/a": b"abc", "params/b": b"012345"}
    rows = []
    for name, data in contents.items():
        rows.append(
            {
                "gcs_name": "checkpoints/pi05_base/" + name,
                "name": name,
                "size": len(data),
                "md5_base64": base64.b64encode(hashlib.md5(data).digest()).decode(),
                "crc32c": "fake",
                "sha256": None,
            }
        )
    manifest = {
        "schema": "test",
        "status": "EXPECTED_NOT_DOWNLOADED",
        "object_count": len(rows),
        "total_bytes": sum(row["size"] for row in rows),
        "objects": rows,
    }
    requests: list[tuple[str, str | None]] = []

    def opener(request: object, timeout: int = 0) -> _Response:
        del timeout
        url = str(getattr(request, "full_url", getattr(request, "_Request__original", "")))
        if not url:
            url = str(request)
        name = url.split("pi05_base/", 1)[1]
        headers = getattr(request, "headers", {})
        range_value = headers.get("Range") or headers.get("range")
        requests.append((name, range_value))
        response = _Response(contents[name])
        response.status = 200 if range_value is None else 206
        return response

    marker = download_base_checkpoint(tmp_path / "base", manifest, opener=opener, strict_source=False)
    assert marker["status"] == "COMPLETED"
    assert marker["object_count"] == 2
    persisted = json.loads((tmp_path / "base" / "BASE_OBJECT_MANIFEST.json").read_text())
    assert all(row["downloaded"] is True and len(row["sha256"]) == 64 for row in persisted["objects"])
    completion = json.loads((tmp_path / "base" / "BASE_DOWNLOAD_COMPLETED.json").read_text())
    assert completion["status"] == "COMPLETED"
    assert len(completion["payload_sha256"]) == 64
    first_request_count = len(requests)
    download_base_checkpoint(tmp_path / "base", tmp_path / "base" / "BASE_OBJECT_MANIFEST.json", opener=opener, strict_source=False)
    assert len(requests) == first_request_count


def test_conversion_and_training_contract_pin_official_flags(tmp_path: Path) -> None:
    conversion = build_conversion_contract(
        openpi_root=tmp_path / "openpi",
        base_jax_root=tmp_path / "jax",
        base_pytorch_root=tmp_path / "pytorch",
        precision="bfloat16",
    )
    assert conversion["openpi_commit"] == OPENPI_COMMIT
    assert conversion["config_name"] == OPENPI_CONFIG_NAME
    assert conversion["command"][1].endswith("examples/convert_jax_model_to_pytorch.py")
    assert conversion["command"][conversion["command"].index("--checkpoint_dir") + 1].endswith("/jax")
    chain = build_c_chain_contract(
        openpi_root=tmp_path / "openpi",
        base_jax_root=tmp_path / "jax",
        base_pytorch_root=tmp_path / "pytorch",
        checkpoint_base_dir=tmp_path / "checkpoints",
        log_root=tmp_path / "logs",
        repo_root=tmp_path / "repo",
    )
    command = chain["training"]["official_command"]
    assert command[0] == DEFAULT_OPENPI_PYTHON
    assert command[1:3] == ["-m", "torch.distributed.run"]
    assert "scripts/train_pytorch.py" in command[6]
    assert command[command.index("--num_train_steps") + 1] == "10001"
    assert command[command.index("--save_interval") + 1] == "1000"
    assert command[command.index("--seed") + 1] == "42"
    assert command[command.index("--num_workers") + 1] == str(C_NUM_WORKERS)
    assert "--pytorch_weight_path" in command
    assert chain["training"]["checkpoint_steps"] == [1000, 3000, 6000, 10000]
    worker = chain["training"]["worker_command"]
    assert any("stage_s_libero_c_train_worker.py" in value for value in worker)
    assert "--" in worker
    assert "--resume" not in worker
    direct = build_patched_training_command(
        worker_path=tmp_path / "worker.py",
        openpi_root=tmp_path / "openpi",
        base_pytorch_root=tmp_path / "pytorch",
        checkpoint_base_dir=tmp_path / "checkpoints",
        assets_base_dir=tmp_path / "assets",
        resume=True,
    )
    assert direct[-1] == "--resume"
    assert "--assets_base_dir" in direct
    assert chain["training"]["full_state_components"][-1] == "rng_state.rank{0..7}.pt"
    assert chain["training"]["num_workers"] == 0
    assert "global_step % epoch_length" in chain["training"]["exact_data_cursor"]
    assert chain["ready_for_pai_submission"] is False


def test_conversion_publishes_provenance_after_official_outputs(tmp_path: Path) -> None:
    output = tmp_path / "converted"
    command = ["python", "official-converter"]

    def runner(*args: object, **kwargs: object) -> None:
        del args, kwargs
        output.mkdir(parents=True, exist_ok=True)
        (output / "model.safetensors").write_bytes(b"converted")
        (output / "config.json").write_text("{}", encoding="utf-8")

    result = run_conversion(
        {
            "source_audit": {"ready": True},
            "base_download_audit": {"valid": True, "manifest": {"manifest_sha256": "base-sha"}},
            "base_pytorch_root": str(output),
            "base_jax_root": str(tmp_path / "jax"),
            "openpi_root": str(tmp_path / "openpi"),
            "command": command,
            "precision": "bfloat16",
        },
        runner=runner,
    )
    assert result["valid"] is True
    assert (output / "CONVERSION_PROVENANCE.json").is_file()
    conversion_marker = json.loads((output / "CONVERSION_COMPLETED.json").read_text())
    assert conversion_marker["status"] == "COMPLETED"
    assert len(conversion_marker["payload_sha256"]) == 64


def test_blackout_guard_is_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="blackout"):
        assert_outside_blackout(datetime(2026, 9, 2, 9, 35))
    with pytest.raises(RuntimeError, match="blackout"):
        assert_outside_blackout(datetime(2026, 9, 2, 19, 30))
    assert assert_outside_blackout(datetime(2026, 9, 2, 9, 40)).hour == 9


def _finalize_fixture(tmp_path: Path, terminal_payload: dict[str, object]) -> tuple[Path, Path]:
    checkpoint_base = tmp_path / "checkpoint"
    train_dir = checkpoint_base / OPENPI_CONFIG_NAME / "r142_stage_s_c_undertrained_seed42"
    for step in C_RETAIN_STEPS:
        step_dir = train_dir / str(step)
        step_dir.mkdir(parents=True, exist_ok=True)
        for rank in range(8):
            (step_dir / f"rng_state.rank{rank}.pt").write_bytes(f"{step}:{rank}".encode())
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / TRAINING_TERMINAL_NAME).write_text(json.dumps(terminal_payload), encoding="utf-8")
    return checkpoint_base, logs


def test_finalize_checks_valid_terminal_json_status_and_commit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    checkpoint_base, logs = _finalize_fixture(
        tmp_path,
        {"status": "FAILED", "openpi_commit": "wrong", "global_step": C_TRAINING_STEPS},
    )
    monkeypatch.setattr(openpi_c, "_checkpoint_audit", lambda _: {"valid": True, "errors": []})
    with pytest.raises(RuntimeError, match="terminal marker is not COMPLETED.*commit mismatch"):
        finalize_training(
            checkpoint_base_dir=checkpoint_base,
            log_root=logs,
            base_manifest_sha256="base",
            openpi_root=tmp_path / "openpi",
        )
    failure = json.loads((logs / "FAILED_C_TRAINING.json").read_text(encoding="utf-8"))
    assert failure["status"] == "FAILED"
    assert len(failure["payload_sha256"]) == 64


def test_finalize_manifests_are_independently_sha256sum_checkable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint_base, logs = _finalize_fixture(
        tmp_path,
        {"status": "COMPLETED", "openpi_commit": OPENPI_COMMIT, "global_step": C_TRAINING_STEPS},
    )
    monkeypatch.setattr(openpi_c, "_checkpoint_audit", lambda _: {"valid": True, "errors": []})
    marker = finalize_training(
        checkpoint_base_dir=checkpoint_base,
        log_root=logs,
        base_manifest_sha256="base",
        openpi_root=tmp_path / "openpi",
    )
    assert marker["sha256sums"] != marker["log_sha256sums"]
    subprocess.run(["sha256sum", "--check", "--quiet", "SHA256SUMS"], cwd=checkpoint_base, check=True)
    subprocess.run(["sha256sum", "--check", "--quiet", "SHA256SUMS"], cwd=logs, check=True)
    assert not any(str(logs) in line for line in (checkpoint_base / "SHA256SUMS").read_text().splitlines())


def test_exact_cursor_loader_skips_resume_offset_and_sets_sampler_epoch() -> None:
    worker = _load_worker_module()

    class FakeSampler:
        def __init__(self) -> None:
            self.epochs: list[int] = []

        def set_epoch(self, epoch: int) -> None:
            self.epochs.append(epoch)

    class FakeTorchLoader:
        def __init__(self, sampler: FakeSampler) -> None:
            self.sampler = sampler

        def __len__(self) -> int:
            return 5

        def __iter__(self):
            return iter(range(5))

    class FakeImplementation:
        def __init__(self, torch_loader: FakeTorchLoader) -> None:
            self.torch_loader = torch_loader

    class FakeBase:
        def __init__(self, implementation: FakeImplementation) -> None:
            self._data_loader = implementation

        def data_config(self) -> str:
            return "config"

        def __iter__(self):
            yield from range(5)

    sampler = FakeSampler()
    base = FakeBase(FakeImplementation(FakeTorchLoader(sampler)))
    loader = worker.ExactCursorDataLoader(base, resume_step=7, require_sampler=True)
    loader.set_epoch(1)
    assert list(loader) == [2, 3, 4]
    assert list(loader) == [0, 1, 2, 3, 4]
    assert sampler.epochs == [1]


def test_registry_v2_payload_binds_pinned_runtime_and_stages() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "configs/pai/stage_s_c_undertrained.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["resource_alias"] == "idle-a800-robot-stage-s-graphics-8gpu"
    assert payload["resource_id"] == "quota1ssrabud0bh"
    assert payload["quota_name"] == "exp-robot"
    assert payload["worker"]["gpu"] == 8
    assert payload["worker"]["cpu"] == 88
    assert payload["worker"]["memory"] == "1525Gi"
    assert payload["runtime"]["python"] == DEFAULT_OPENPI_PYTHON
    assert payload["runtime"]["uid"] == payload["runtime"]["gid"] == 2254
    assert payload["runtime"]["project_dir"] == \
        "/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/r142-stage-s-c-runtime-20260903"
    assert payload["runtime"]["project_dir_env"] == "STAGE_S_C_PROJECT_DIR"
    assert payload["runtime"]["command_file"] == (
        payload["runtime"]["project_dir"] + "/" + payload["runtime"]["command_file_relative"]
    )
    assert payload["runtime"]["payload_sha256_env"] == "STAGE_S_C_PAYLOAD_SHA256"
    assert payload["runtime"]["stage_s_source_commit_env"] == "STAGE_S_SOURCE_COMMIT"
    assert payload["runtime"]["qpilots_commit"] == "eacf47b981e3b22357f8a74902f8dad8cfcfa375"
    assert payload["runtime"]["openpi_commit"] == OPENPI_COMMIT
    assert payload["runtime"]["required_env_names"] == [
        "PAI_RUN_ID",
        "STAGE_S_SOURCE_COMMIT",
        "STAGE_S_C_PROJECT_DIR",
        "STAGE_S_C_PAYLOAD_SHA256",
    ]
    assert payload["runtime"]["pod_env"]["STAGE_S_C_PROJECT_DIR"] == payload["runtime"]["project_dir"]
    assert payload["runtime"]["pod_env"]["STAGE_S_C_PAYLOAD_SHA256"] == payload["runtime"]["payload_sha256"]
    assert payload["fault_tolerance"]["application_auto_resume"] is True
    assert payload["fault_tolerance"]["same_directory_on_resume"] is True
    assert payload["fault_tolerance"]["max_num_of_job_restart"] == 50
    assert all(path.startswith("/mnt/cpfs/zbl-cpfs-new/") for path in payload["runtime"]["write_paths"])
    script = root / "scripts/stage_s_c_undertrained_pai.sh"
    assert hashlib.sha256(script.read_bytes()).hexdigest() == payload["runtime"]["command_file_sha256"]
    assert payload["runtime"]["home_policy"].startswith("inherit")
    assert "R142-FP-11-Bottleneck-Local-Candidate-Split" not in script.read_text(encoding="utf-8")
    assert "STAGE_S_SOURCE_COMMIT" in script.read_text(encoding="utf-8")
    assert "EXPECTED_QPILOTS_COMMIT" in script.read_text(encoding="utf-8")
    assert "STAGE_S_C_PAYLOAD_SHA256" in script.read_text(encoding="utf-8")
    assert "RUNTIME_IDENTITY.json" in script.read_text(encoding="utf-8")
    assert [stage["name"] for stage in payload["stage_pipeline"]["stages"]] == [
        "base_download",
        "conversion",
        "training",
    ]
    assert "export HOME" not in script.read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(script)], check=True)
