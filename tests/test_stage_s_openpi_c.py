from __future__ import annotations

import base64
import hashlib
import io
import json
from datetime import datetime
from pathlib import Path

import pytest

from r142_stage_s.openpi_c import (
    OPENPI_COMMIT,
    OPENPI_CONFIG_NAME,
    PI05_BASE_OBJECT_COUNT,
    PI05_BASE_OBJECTS,
    PI05_BASE_TOTAL_BYTES,
    assert_outside_blackout,
    build_c_chain_contract,
    build_conversion_contract,
    build_patched_training_command,
    download_base_checkpoint,
    expected_base_manifest,
    manifest_from_gcs_listing,
    run_conversion,
    validate_base_manifest,
)


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
    assert (tmp_path / "base" / "BASE_DOWNLOAD_COMPLETED.json").is_file()
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
    assert "scripts/train_pytorch.py" in command[4]
    assert command[command.index("--num_train_steps") + 1] == "10001"
    assert command[command.index("--save_interval") + 1] == "1000"
    assert command[command.index("--seed") + 1] == "42"
    assert "--pytorch_weight_path" in command
    assert chain["training"]["checkpoint_steps"] == [1000, 3000, 6000, 10000]
    worker = chain["training"]["worker_command"]
    assert "stage_s_libero_c_train_worker.py" in worker[4]
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
    assert (output / "CONVERSION_COMPLETED.json").is_file()


def test_blackout_guard_is_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="blackout"):
        assert_outside_blackout(datetime(2026, 9, 2, 9, 35))
    with pytest.raises(RuntimeError, match="blackout"):
        assert_outside_blackout(datetime(2026, 9, 2, 19, 30))
    assert assert_outside_blackout(datetime(2026, 9, 2, 9, 40)).hour == 9
