from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

import r142_stage_s.c_training_acceptance as ca


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(payload: dict) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def _json(path: Path, payload: dict, *, self_hash: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload)
    if self_hash:
        data["payload_sha256"] = hashlib.sha256(_canonical(data)).hexdigest()
    path.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _manifest(root: Path, paths: list[Path], name: str = "SHA256SUMS") -> Path:
    target = root / name
    target.write_text(
        "".join(f"{_sha(path)}  {path.relative_to(root).as_posix()}\n" for path in sorted(paths)),
        encoding="utf-8",
    )
    return target


def _git(path: Path, *, nested: bool = False) -> str:
    path.mkdir(parents=True, exist_ok=True)
    (path / "source.txt").write_text("pinned\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(path), "-c", "user.name=Stage-S", "-c", "user.email=stage-s@example.invalid", "commit", "-qm", "fixture"],
        check=True,
    )
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


@pytest.fixture()
def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path | str]:
    """Build a small but structurally complete acceptance lineage."""

    monkeypatch.setattr(ca, "EXPECTED_UID", os.getuid())
    monkeypatch.setattr(ca, "EXPECTED_GID", os.getgid())
    monkeypatch.setattr(ca, "EXPECTED_DATASET_FILE_COUNT", 1)

    runtime = tmp_path / "runtime"
    runtime_commit = _git(runtime)
    qpilots = tmp_path / "qpilots"
    # The real QPILOTS checkout contains nested third_party repositories.  A
    # gitignore keeps those nested .git directories out of the parent status,
    # so the acceptance gate can verify all three repositories independently.
    qpilots.mkdir(parents=True)
    (qpilots / ".gitignore").write_text("third_party/\n", encoding="utf-8")
    qpilots_commit = _git(qpilots)
    openpi = qpilots / "third_party" / "openpi"
    openpi.mkdir(parents=True)
    (openpi / ".gitignore").write_text("third_party/\n", encoding="utf-8")
    openpi_commit = _git(openpi)
    libero = openpi / "third_party" / "libero"
    libero_commit = _git(libero)
    monkeypatch.setattr(ca, "EXPECTED_QPILOTS_COMMIT", qpilots_commit)
    monkeypatch.setattr(ca, "EXPECTED_OPENPI_COMMIT", openpi_commit)
    monkeypatch.setattr(ca, "EXPECTED_LIBERO_COMMIT", libero_commit)

    payload = tmp_path / "deployed" / "stage_s_c_undertrained_pai.sh"
    payload.parent.mkdir(parents=True)
    payload.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    payload_sha = _sha(payload)

    dataset = tmp_path / "dataset"
    data_file = dataset / "data" / "episode-000.parquet"
    data_file.parent.mkdir(parents=True)
    data_file.write_bytes(b"official fixture bytes\n")
    dataset_manifest = _manifest(dataset, [data_file], "DATASET_SHA256SUMS")
    monkeypatch.setattr(ca, "EXPECTED_DATASET_MANIFEST_SHA256", _sha(dataset_manifest))

    norm_source = tmp_path / "assets" / "source" / "norm_stats.json"
    norm_staged = tmp_path / "assets" / "staged" / "norm_stats.json"
    norm_source.parent.mkdir(parents=True)
    norm_staged.parent.mkdir(parents=True)
    norm_source.write_text('{"state": {"mean": [0.0]}}\n', encoding="utf-8")
    norm_staged.write_bytes(norm_source.read_bytes())

    status_root = tmp_path / "logs" / "c_status" / "r142-stage-s-c-undertrained-20260903-r99"
    run_root = tmp_path / "logs" / "c" / "r142-stage-s-c-undertrained-20260903-r99"
    checkpoint_root = tmp_path / "CKPT" / "r142_stage_s_c"
    status_root.mkdir(parents=True)
    run_root.mkdir(parents=True)
    checkpoint_root.mkdir(parents=True)
    artifact_dir = tmp_path / "registry-artifact"
    artifact_dir.mkdir()
    run_id = run_root.name
    job_id = "dlctestc99"

    data_payload = {
        "schema": "r142-stage-s-c-data-preflight-v2",
        "status": "COMPLETED",
        "dataset": {
            "valid": True,
            "repo_id": ca.EXPECTED_DATASET_REPO,
            "revision": ca.EXPECTED_DATASET_REVISION,
            "root": str(dataset),
            "manifest_path": str(dataset_manifest),
            "manifest_sha256": _sha(dataset_manifest),
            "manifest_file_sha256": _sha(dataset_manifest),
            "file_count": 1,
        },
        "norm_stats": {
            "valid": True,
            "source_path": str(norm_source),
            "source_sha256": _sha(norm_source),
            "staged_path": str(norm_staged),
            "staged_sha256": _sha(norm_staged),
        },
        "official_bindings": {
            "config_name": ca.EXPECTED_CONFIG,
            "repo_id": ca.EXPECTED_DATASET_REPO,
            "resolver_path": str(norm_staged),
            "lerobot_compatibility": {
                "valid": True,
                "lerobot_version": "0.1.0",
                "lerobot_commit": "0cf864870cf29f4738d3ade893e6fd13fbd7cdb5",
                "datasets_version": "3.6.0",
                "mode": "native-pinned-datasets",
            },
        },
        "no_network_fallback": True,
        "no_pai_submit_performed": True,
    }
    data_path = status_root / ca.DATA_PREFLIGHT
    _json(data_path, data_payload, self_hash=True)

    identity_payload = {
        "schema": "r142-stage-s-c-runtime-identity-v2",
        "project_dir": str(runtime),
        "stage_s_source_commit": runtime_commit,
        "qpilots_root": str(qpilots),
        "qpilots_commit": qpilots_commit,
        "openpi_root": str(openpi),
        "openpi_commit": openpi_commit,
        "payload_path": str(payload),
        "payload_sha256": payload_sha,
        "payload_sha256_observed": payload_sha,
        "run_id": run_id,
        "job_id": job_id,
        "data_preflight_path": str(data_path),
        "data_preflight_sha256": _sha(data_path),
        "dataset_repo_id": ca.EXPECTED_DATASET_REPO,
        "dataset_revision": ca.EXPECTED_DATASET_REVISION,
        "dataset_manifest_sha256": _sha(dataset_manifest),
        "dataset_manifest_file_sha256": _sha(dataset_manifest),
        "norm_stats_source_sha256": _sha(norm_source),
        "norm_stats_sha256": _sha(norm_staged),
    }
    identity_path = status_root / ca.RUNTIME_IDENTITY
    _json(identity_path, identity_payload)

    train_dir = checkpoint_root / ca.EXPECTED_CONFIG / ca.EXPECTED_EXPERIMENT
    train_dir.mkdir(parents=True)
    checkpoint_files: list[Path] = []
    for step in ca.EXPECTED_STEPS:
        step_dir = train_dir / str(step)
        step_dir.mkdir()
        for name in ca.CORE_CHECKPOINT_FILES:
            path = step_dir / name
            path.write_bytes(f"{name}-{step}\n".encode())
            checkpoint_files.append(path)
        _json(
            step_dir / ca.CHECKPOINT_READY,
            {
                "schema": "r142-stage-s-c-checkpoint-ready-v1",
                "status": "READY",
                "global_step": step,
                "world_size": ca.EXPECTED_WORLD_SIZE,
                "checkpoint_dir": str(step_dir),
                "core_files": [
                    {"name": name, "exists": True, "size": (step_dir / name).stat().st_size}
                    for name in ca.CORE_CHECKPOINT_FILES
                ],
            },
        )
        rng_paths = []
        for rank in range(ca.EXPECTED_WORLD_SIZE):
            path = step_dir / f"rng_state.rank{rank}.pt"
            path.write_bytes(f"rng-{step}-{rank}\n".encode())
            rng_paths.append(path)
        rng_manifest = step_dir / ca.RNG_SHA256SUMS
        rng_manifest.write_text("".join(f"{_sha(path)}  {path.name}\n" for path in rng_paths), encoding="utf-8")
        _json(
            step_dir / ca.COMPLETE_RNG_STATE,
            {
                "schema": "r142-stage-s-c-complete-rng-state-v1",
                "status": "COMPLETED",
                "global_step": step,
                "world_size": ca.EXPECTED_WORLD_SIZE,
                "sidecars": [path.name for path in rng_paths],
                "rng_sha256sums": ca.RNG_SHA256SUMS,
                "rng_sha256sums_sha256": _sha(rng_manifest),
            },
        )
        checkpoint_files.extend([step_dir / ca.CHECKPOINT_READY, *rng_paths, rng_manifest, step_dir / ca.COMPLETE_RNG_STATE])
    checkpoint_manifest = _manifest(checkpoint_root, checkpoint_files)

    completion_payload = {
        "schema": "r142-stage-s-c-training-completion-v1",
        "status": "COMPLETED",
        "openpi_commit": openpi_commit,
        "openpi_root": str(openpi),
        "config_name": ca.EXPECTED_CONFIG,
        "seed": ca.EXPECTED_SEED,
        "terminal_global_step": ca.EXPECTED_TERMINAL_STEP,
        "checkpoint_steps": list(ca.EXPECTED_STEPS),
        "checkpoint_audit": {"valid": True},
        "sha256sums": str(checkpoint_manifest),
        "sha256sums_sha256": _sha(checkpoint_manifest),
        "log_sha256sums": str(run_root / "SHA256SUMS"),
        "log_sha256sums_sha256": "pending",
        "data_preflight_path": str(data_path),
        "data_preflight_sha256": _sha(data_path),
    }
    completion_path = checkpoint_root / ca.TRAINING_COMPLETION
    # The log manifest digest is known only after run markers are materialized.
    terminal_path = run_root / ca.TRAINING_TERMINAL
    start_path = run_root / ca.TRAINING_START
    _json(start_path, {"schema": "r142-stage-s-c-training-start-v1", "status": "RUNNING", "openpi_commit": openpi_commit, "config_name": ca.EXPECTED_CONFIG, "checkpoint_base_dir": str(checkpoint_root)}, self_hash=True)
    _json(terminal_path, {"schema": "r142-stage-s-c-training-terminal-v1", "status": "COMPLETED", "openpi_commit": openpi_commit, "config_name": ca.EXPECTED_CONFIG, "global_step": ca.EXPECTED_TERMINAL_STEP, "checkpoint_steps": list(ca.EXPECTED_STEPS)}, self_hash=True)
    log_manifest = _manifest(run_root, [start_path, terminal_path])
    completion_payload["log_sha256sums_sha256"] = _sha(log_manifest)
    _json(completion_path, completion_payload, self_hash=True)

    pipeline_path = status_root / ca.PIPELINE_COMPLETION
    _json(
        pipeline_path,
        {
            "schema": "r142-stage-s-c-pai-stage-status-v1",
            "status": "COMPLETED",
            "stage": "terminal",
            "run_id": run_id,
            "evidence_path": str(completion_path),
            "evidence_sha256": _sha(completion_path),
        },
        self_hash=True,
    )
    for name in ("COMPLETED_preflight.json", "COMPLETED_base_download.json", "COMPLETED_conversion.json", "COMPLETED_training.json"):
        _json(status_root / name, {"schema": "r142-stage-s-c-pai-stage-status-v1", "status": "COMPLETED", "stage": name.removeprefix("COMPLETED_").removesuffix(".json"), "run_id": run_id}, self_hash=True)

    registry_root = tmp_path / "registry" / run_id
    registry_root.mkdir(parents=True)
    _json(registry_root / "result.json", {"run_id": run_id, "job_id": job_id, "submission_state": "submitted_verified", "returncode": 0})
    _json(registry_root / "submission-state.json", {"run_id": run_id, "job_id": job_id, "state": "submitted_verified"})
    _json(registry_root / "resolved.json", {"run_id": run_id, "artifact_dir": str(artifact_dir), "runtime": {"project_dir": str(runtime), "command_file": str(payload), "command_file_sha256": payload_sha, "payload_sha256": payload_sha, "qpilots_commit": qpilots_commit, "openpi_commit": openpi_commit, "uid": os.getuid(), "gid": os.getgid(), "output_mode": "resume", "create_artifact_dir": True, "recursive_repair": False, "write_paths": [str(artifact_dir), str(run_root.parent), str(status_root.parent), str(checkpoint_root.parent)]}})
    ledger = registry_root / "jobs.jsonl"
    ledger.write_text(json.dumps({"run_id": run_id, "job_id": job_id}) + "\n", encoding="utf-8")
    getjob = registry_root / "getjob-terminal.json"
    _json(getjob, {"JobId": job_id, "Status": "Succeeded", "ReasonCode": "JobSucceeded"})
    getjob_sha = registry_root / "getjob-terminal.json.sha256"
    getjob_sha.write_text(f"{_sha(getjob)}  {getjob.name}\n", encoding="utf-8")

    return {
        "registry_run": registry_root,
        "registry_result": registry_root / "result.json",
        "submission_state": registry_root / "submission-state.json",
        "resolved": registry_root / "resolved.json",
        "jobs_ledger": ledger,
        "terminal_getjob": getjob,
        "terminal_getjob_sha": getjob_sha,
        "run_root": run_root,
        "status_root": status_root,
        "checkpoint_root": checkpoint_root,
        "runtime": runtime,
        "qpilots": qpilots,
        "openpi": openpi,
        "payload": payload,
        "data": data_path,
        "completion": completion_path,
        "pipeline": pipeline_path,
        "dataset_manifest": dataset_manifest,
        "output": status_root.parent / "ACCEPTED_C_TRAINING.json",
    }


def _kwargs(fixture: dict[str, Path | str]) -> dict[str, object]:
    return {
        "registry_run": fixture["registry_run"],
        "registry_result": fixture["registry_result"],
        "submission_state": fixture["submission_state"],
        "resolved": fixture["resolved"],
        "jobs_ledger": fixture["jobs_ledger"],
        "terminal_getjob": fixture["terminal_getjob"],
        "terminal_getjob_sha": fixture["terminal_getjob_sha"],
        "c_run_root": fixture["run_root"],
        "c_status_root": fixture["status_root"],
        "checkpoint_root": fixture["checkpoint_root"],
        "expected_uid": os.getuid(),
        "expected_gid": os.getgid(),
    }


def test_terminal_acceptance_is_complete_and_sidecar_bound(fixture) -> None:
    result = ca.write_c_training_acceptance(output_path=fixture["output"], **_kwargs(fixture))
    assert result["status"] == "ACCEPTED"
    assert result["label"] == "WEAK_SUBSTRATE"
    assert result["pai_terminal_status"] == "Succeeded"
    assert result["checkpoint_steps"] == [1000, 3000, 6000, 10000]
    assert set(result["checkpoint_hashes"]) == {f"{step}/model.safetensors" for step in ca.EXPECTED_STEPS}
    verified = ca.verify_c_training_acceptance(fixture["output"])
    assert verified["sha256"] == result["acceptance_sha256"]


def test_acceptance_has_no_write_side_effect_on_running_job(fixture) -> None:
    getjob = json.loads(Path(fixture["terminal_getjob"]).read_text())
    getjob["Status"] = "Running"
    _json(Path(fixture["terminal_getjob"]), getjob)
    Path(fixture["terminal_getjob_sha"]).write_text(
        f"{_sha(Path(fixture['terminal_getjob']))}  {Path(fixture['terminal_getjob']).name}\n",
        encoding="utf-8",
    )
    with pytest.raises(ca.CTrainingAcceptanceError, match="not Succeeded"):
        ca.write_c_training_acceptance(output_path=fixture["output"], **_kwargs(fixture))
    assert not Path(fixture["output"]).exists()


def test_duplicate_job_id_is_rejected(fixture) -> None:
    ledger = Path(fixture["jobs_ledger"])
    line = ledger.read_text()
    ledger.write_text(line + line)
    with pytest.raises(ca.CTrainingAcceptanceError, match="exactly one|unique"):
        ca.build_c_training_acceptance(**_kwargs(fixture))


def test_missing_rng_sidecar_is_rejected(fixture) -> None:
    target = Path(fixture["checkpoint_root"]) / ca.EXPECTED_CONFIG / ca.EXPECTED_EXPERIMENT / "6000" / "rng_state.rank7.pt"
    target.unlink()
    with pytest.raises(ca.CTrainingAcceptanceError, match="missing or symlinked|SHA manifest"):
        ca.build_c_training_acceptance(**_kwargs(fixture))


def test_extra_partial_checkpoint_is_rejected(fixture) -> None:
    extra = Path(fixture["checkpoint_root"]) / ca.EXPECTED_CONFIG / ca.EXPECTED_EXPERIMENT / "11000"
    extra.mkdir()
    (extra / "model.safetensors").write_bytes(b"partial\n")
    with pytest.raises(ca.CTrainingAcceptanceError, match="extra C checkpoint"):
        ca.build_c_training_acceptance(**_kwargs(fixture))


def test_extra_status_file_is_rejected(fixture) -> None:
    extra = Path(fixture["status_root"]) / "unexpected-note.txt"
    extra.write_text("not part of the terminal status contract\n", encoding="utf-8")
    with pytest.raises(ca.CTrainingAcceptanceError, match="non-exact file inventory"):
        ca.build_c_training_acceptance(**_kwargs(fixture))


def test_completion_without_self_sha_is_rejected(fixture) -> None:
    completion = Path(fixture["completion"])
    payload = json.loads(completion.read_text())
    payload.pop("payload_sha256")
    _json(completion, payload)
    pipeline_path = Path(fixture["pipeline"])
    pipeline = json.loads(pipeline_path.read_text())
    pipeline["evidence_sha256"] = _sha(completion)
    pipeline.pop("payload_sha256", None)
    _json(pipeline_path, pipeline, self_hash=True)
    with pytest.raises(ca.CTrainingAcceptanceError, match="lacks required payload_sha256"):
        ca.build_c_training_acceptance(**_kwargs(fixture))


def test_payload_drift_is_rejected(fixture) -> None:
    resolved = Path(fixture["resolved"])
    payload = json.loads(resolved.read_text())
    payload["runtime"]["payload_sha256"] = "0" * 64
    _json(resolved, payload)
    with pytest.raises(ca.CTrainingAcceptanceError, match="payload identity"):
        ca.build_c_training_acceptance(**_kwargs(fixture))


def test_output_is_unique_and_never_overwritten(fixture) -> None:
    output = Path(fixture["output"])
    output.write_text("sentinel\n")
    with pytest.raises(ca.CTrainingAcceptanceError, match="already exists"):
        ca.write_c_training_acceptance(output_path=output, **_kwargs(fixture))
    assert output.read_text() == "sentinel\n"


def test_cli_is_non_submitting_and_has_no_pai_create_call() -> None:
    cli = Path(__file__).parents[1] / "scripts" / "stage_s_accept_c_training.py"
    text = cli.read_text(encoding="utf-8")
    lowered = text.lower()
    assert "createjob" not in lowered
    assert "submitjob" not in lowered
    assert "pai-job submit" not in lowered
