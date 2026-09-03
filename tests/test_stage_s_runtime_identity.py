from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from r142_stage_s.runtime_identity import attest_config, sha256_file, verify_manifest, write_attestation


def _git_repo(path: Path, name: str = "source.txt") -> str:
    path.mkdir(parents=True, exist_ok=True)
    (path / name).write_text("pinned\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.name=Stage-S test",
            "-c",
            "user.email=stage-s-test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def _manifest(root: Path, *files: Path) -> Path:
    rows = []
    for path in sorted(files):
        rows.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n")
    manifest = root / "SHA256SUMS"
    manifest.write_text("".join(rows), encoding="utf-8")
    return manifest


@pytest.fixture()
def fixture_config(tmp_path: Path) -> tuple[Path, dict[str, Path], dict[str, str]]:
    runtime = tmp_path / "runtime"
    runtime_head = _git_repo(runtime)
    dependency = tmp_path / "dependency"
    dependency_head = _git_repo(dependency)

    launcher = tmp_path / "deployed" / "stage_s_payload.sh"
    launcher.parent.mkdir()
    launcher.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    launcher_sha = hashlib.sha256(launcher.read_bytes()).hexdigest()

    model = tmp_path / "model"
    model.mkdir()
    model_file = model / "model.safetensors"
    model_file.write_bytes(b"model fixture\n")
    _manifest(model, model_file)

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    checkpoint_file = checkpoint / "model.safetensors"
    checkpoint_file.write_bytes(b"checkpoint fixture\n")
    _manifest(checkpoint, checkpoint_file)

    protocol = tmp_path / "protocol"
    protocol.mkdir()
    protocol_md = protocol / "PROTOCOL.md"
    protocol_md.write_text("# Frozen\n", encoding="utf-8")
    protocol_json = protocol / "FROZEN_PROTOCOL.json"
    protocol_json.write_text(
        json.dumps(
            {
                "status": "FROZEN",
                "protocol_md_path": str(protocol_md),
                "protocol_md_sha256": sha256_file(protocol_md),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _manifest(protocol, protocol_md, protocol_json)

    report_dir = tmp_path / "report"
    report_dir.mkdir()
    report = report_dir / "CALIBRATION_REPORT.json"
    report.write_text('{"status":"FROZEN"}\n', encoding="utf-8")
    _manifest(report_dir, report)

    config_repo = tmp_path / "config-repo"
    config_head = _git_repo(config_repo, name="placeholder.txt")
    config = {
        "schema": "r142-stage-s",
        "resource_alias": "idle-a800-robot-stage-s-graphics-8gpu",
        "worker": {
            "count": 1,
            "gpu": 8,
            "cpu": 88,
            "memory": "1400Gi",
            "shared_memory": "1400Gi",
        },
        "resource": {
            "pool_family": "robot",
            "pool": "robot_idle",
            "gpu": 8,
            "cpu": 88,
            "memory": "1400Gi",
            "shared_memory": "1400Gi",
            "workers": 1,
            "oversold_type": "AcceptQuotaOverSold",
        },
        "runtime": {
            "runtime_repo": str(runtime),
            "source_commit": runtime_head,
            "command_file": str(launcher),
            "command_file_sha256": launcher_sha,
            "payload_sha256": launcher_sha,
            "frozen_protocol_path": str(protocol_json),
            "dependencies": {
                "dependency": {"path": str(dependency), "commit": dependency_head}
            },
        },
        "assets": {
            "model_path": str(model),
            "checkpoint_path": str(checkpoint),
        },
        "evidence": {
            "protocol_acceptance_path": str(protocol_json),
            "calibration_gate": {"report": str(report)},
        },
    }
    config_path = config_repo / "stage_s.json"
    config_path.write_text(json.dumps(config, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    # The config repository intentionally remains clean after adding the
    # fixture; the config source commit is therefore derived from its HEAD.
    subprocess.run(["git", "-C", str(config_repo), "add", "stage_s.json"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(config_repo),
            "-c",
            "user.name=Stage-S test",
            "-c",
            "user.email=stage-s-test@example.invalid",
            "commit",
            "-qm",
            "config fixture",
        ],
        check=True,
    )
    config_head = subprocess.check_output(
        ["git", "-C", str(config_repo), "rev-parse", "HEAD"], text=True
    ).strip()
    return config_path, {
        "runtime": runtime,
        "dependency": dependency,
        "launcher": launcher,
        "model": model,
        "checkpoint": checkpoint,
        "protocol": protocol,
        "report": report,
        "config_repo": config_repo,
    }, {"runtime": runtime_head, "dependency": dependency_head, "config": config_head}


def test_valid_fixture_passes_and_is_deterministic(fixture_config) -> None:
    config, _, _ = fixture_config
    first = attest_config(config)
    second = attest_config(config)
    assert first["status"] == "PASS"
    assert first == second
    assert first["runtime"]["deployed_launcher"]["observed_sha256"] == first["runtime"]["deployed_launcher"]["expected_sha256"]
    assert first["runtime"]["repository"]["head"] == fixture_config[2]["runtime"]
    assert first["runtime"]["repository"]["clean"]
    assert all(row["observed"]["clean"] for row in first["dependencies"])
    assert {row["manifest_check"]["valid"] for row in first["artifacts"] if "manifest_check" in row} == {True}


def test_identifier_description_and_template_fields_skip_path_existence(fixture_config) -> None:
    config, paths, _ = fixture_config
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["provenance"] = {
        "dataset_repo_id": "physical-intelligence/libero",
        "dataset_manifest_sha256_source": (
            "pre-generated DATASET_SHA256SUMS; rechecked in DATA_PREFLIGHT.json/"
            "RUNTIME_IDENTITY.json"
        ),
    }
    payload["stage_pipeline"] = {
        "status_root": "/mnt/cpfs/future/r142/c_status/<RUN_ID>",
    }
    payload["evidence"]["dataset_manifest_path"] = str(paths["model"] / "SHA256SUMS")
    config.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    result = attest_config(config)
    skips = {
        (row["key"], row["semantic"])
        for row in result["path_skips"]
    }
    assert ("dataset_repo_id", "identifier") in skips
    assert ("dataset_manifest_sha256_source", "description") in skips
    assert ("status_root", "template") in skips
    assert not any(
        error.startswith("path_missing_or_symlinked:dataset_repo_id")
        or error.startswith("path_missing_or_symlinked:dataset_manifest_sha256_source")
        or error.startswith("path_missing_or_symlinked:status_root")
        for error in result["errors"]
    )
    # The real manifest path remains an artifact authority and is still
    # checked, proving that semantic skips do not weaken dataset validation.
    assert any(
        row["key"] == "dataset_manifest_path" and row["path"] == str(paths["model"] / "SHA256SUMS")
        for row in result["artifacts"]
    )


def test_placeholder_on_authoritative_path_is_refused(fixture_config) -> None:
    config, _, _ = fixture_config
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["evidence"]["dataset_manifest_path"] = "/missing/dataset/<RUN_ID>/DATASET_SHA256SUMS"
    config.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    result = attest_config(config)
    assert any(
        error.startswith("path_missing_or_symlinked:dataset_manifest_path")
        for error in result["errors"]
    )
    assert not any(row["key"] == "dataset_manifest_path" for row in result["path_skips"])


def test_declared_dataset_manifest_is_verified_without_adjacent_sha256sum(fixture_config, tmp_path: Path) -> None:
    config, _, _ = fixture_config
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    data_file = dataset / "episode-000.bin"
    data_file.write_bytes(b"official dataset fixture\n")
    manifest = dataset / "DATASET_SHA256SUMS"
    manifest.write_text(f"{sha256_file(data_file)}  {data_file.name}\n", encoding="utf-8")

    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["evidence"]["dataset_manifest_path"] = str(manifest)
    payload["evidence"]["dataset_manifest_sha256"] = sha256_file(manifest)
    config.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    result = attest_config(config)
    row = next(row for row in result["artifacts"] if row["key"] == "dataset_manifest_path")
    assert row["manifest"] == str(manifest)
    assert row["manifest_check"]["valid"]
    assert row["sha256_match"]


def test_c_undertrained_config_binds_runtime_and_official_dependencies() -> None:
    config = Path(__file__).parents[1] / "configs" / "pai" / "stage_s_c_undertrained.json"
    payload = json.loads(config.read_text(encoding="utf-8"))
    runtime = payload["runtime"]
    assert runtime["source_commit"] == "cd3bfb4f1d2e392f071140dc7b02ec4ea3c3d0bc"
    assert payload["evidence"]["dataset_manifest_sha256"] == (
        "02b5b3abfadb65b2f1c4823cfe7ed7b9351416934674fcf59aea1868826546bf"
    )
    dependencies = runtime["dependencies"]
    assert dependencies["qpilots"] == {
        "path": "/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/QPILOTS-r16p15-stage1-task64-20260812",
        "commit": "eacf47b981e3b22357f8a74902f8dad8cfcfa375",
    }
    assert dependencies["openpi"] == {
        "path": "/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/QPILOTS-r16p15-stage1-task64-20260812/third_party/openpi",
        "commit": "54cbaee6ae0c010a1ed431871cdaa8f4684ac709",
    }
    assert "config_source_commit" not in payload


def test_missing_explicit_output_does_not_write(tmp_path: Path) -> None:
    output = tmp_path / "attestation.json"
    script = Path(__file__).parents[1] / "scripts" / "stage_s_runtime_identity_preflight.py"
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    proc = subprocess.run(
        ["python3", str(script), "--config", str(config)],
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 2
    assert not output.exists()
    assert "--output is required" in proc.stderr


def test_launcher_sha_mismatch_is_refused(fixture_config) -> None:
    config, paths, _ = fixture_config
    paths["launcher"].write_text(paths["launcher"].read_text(encoding="utf-8") + "# tamper\n", encoding="utf-8")
    result = attest_config(config)
    assert result["status"] == "REFUSED"
    assert any("deployed_launcher_sha256_mismatch" in error for error in result["errors"])


def test_command_and_payload_sha_disagreement_is_refused(fixture_config) -> None:
    config, _, _ = fixture_config
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["runtime"]["payload_sha256"] = "0" * 64
    config.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    result = attest_config(config)
    assert result["status"] == "REFUSED"
    assert any("deployed_launcher_sha256_mismatch:payload_sha256" in error for error in result["errors"])


def test_runtime_commit_and_dirty_tree_mismatch_are_refused(fixture_config) -> None:
    config, paths, _ = fixture_config
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["runtime"]["source_commit"] = "f" * 40
    config.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    result = attest_config(config)
    assert any("runtime_source_commit_mismatch" in error for error in result["errors"])
    # The uncommitted config itself is independently refused as a config-source
    # identity mismatch, not silently treated as a new config.
    assert "config_source_tree_dirty" in result["errors"]


def test_explicit_config_source_commit_mismatch_is_refused(fixture_config) -> None:
    config, paths, _ = fixture_config
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["config_source_commit"] = "f" * 40
    config.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(paths["config_repo"]), "add", "stage_s.json"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(paths["config_repo"]),
            "-c",
            "user.name=Stage-S test",
            "-c",
            "user.email=stage-s-test@example.invalid",
            "commit",
            "-qm",
            "bad source pin",
        ],
        check=True,
    )
    result = attest_config(config)
    assert result["status"] == "REFUSED"
    assert any("config_source_commit_mismatch" in error for error in result["errors"])


def test_dependency_commit_mismatch_is_refused(fixture_config) -> None:
    config, paths, _ = fixture_config
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["runtime"]["dependencies"]["dependency"]["commit"] = "0" * 40
    config.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    result = attest_config(config)
    assert any("dependency_source_commit_mismatch" in error for error in result["errors"])


def test_invalid_dependency_pin_is_refused(fixture_config) -> None:
    config, _, _ = fixture_config
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["runtime"]["dependencies"]["dependency"]["commit"] = "not-a-git-commit"
    config.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    result = attest_config(config)
    assert "dependency_commit_invalid:runtime.dependencies:dependency" in result["errors"]


def test_manifest_tamper_and_missing_manifest_are_refused(fixture_config) -> None:
    config, paths, _ = fixture_config
    model_file = paths["model"] / "model.safetensors"
    model_file.write_bytes(b"tampered\n")
    result = attest_config(config)
    assert any("artifact_manifest_invalid" in error for error in result["errors"])

    # Remove only the model manifest.  Other authorities stay intact, proving
    # this is a specific artifact-manifest gate rather than a generic failure.
    (paths["model"] / "SHA256SUMS").unlink()
    result = attest_config(config)
    assert any("artifact_manifest_missing" in error for error in result["errors"])


def test_protocol_markdown_hash_mismatch_is_refused(fixture_config) -> None:
    config, paths, _ = fixture_config
    (paths["protocol"] / "PROTOCOL.md").write_text("# changed\n", encoding="utf-8")
    result = attest_config(config)
    assert any("protocol_md_sha256_mismatch" in error for error in result["errors"])


def test_resource_1525_gib_is_refused(fixture_config) -> None:
    config, _, _ = fixture_config
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["worker"]["memory"] = "1525Gi"
    config.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    result = attest_config(config)
    assert any("resource_mismatch:worker:memory_gib:1525!=1400" in error for error in result["errors"])


def test_symlink_authorities_are_refused(fixture_config, tmp_path: Path) -> None:
    config, paths, _ = fixture_config
    link = tmp_path / "protocol-link.json"
    link.symlink_to(paths["protocol"] / "FROZEN_PROTOCOL.json")
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["runtime"]["frozen_protocol_path"] = str(link)
    config.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    result = attest_config(config)
    assert any("path_missing_or_symlinked:frozen_protocol_path" in error for error in result["errors"])


def test_explicit_output_is_the_only_write(tmp_path: Path) -> None:
    config_repo = tmp_path / "config"
    _git_repo(config_repo)
    config = config_repo / "bad.json"
    config.write_text("{}\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(config_repo), "add", "bad.json"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(config_repo),
            "-c",
            "user.name=Stage-S test",
            "-c",
            "user.email=stage-s-test@example.invalid",
            "commit",
            "-qm",
            "bad config",
        ],
        check=True,
    )
    output = tmp_path / "attestation.json"
    result = attest_config(config)
    write_attestation(result, output)
    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8")) == result


def test_verify_manifest_rejects_traversal(tmp_path: Path) -> None:
    manifest = tmp_path / "SHA256SUMS"
    manifest.write_text(f"{'a' * 64}  ../outside\n", encoding="utf-8")
    result = verify_manifest(manifest)
    assert not result["valid"]
    assert any("manifest_unsafe_path" in error for error in result["errors"])
