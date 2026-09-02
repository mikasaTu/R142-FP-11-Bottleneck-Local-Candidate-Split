from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from r142_stage_s.openpi_c import (
    LIBERO_DATASET_EXPECTED_INFO,
    LIBERO_DATASET_MANIFEST_NAME,
    LIBERO_DATASET_REPO,
    LIBERO_DATASET_REVISION,
    LIBERO_NORM_STATS_RUNTIME_RELATIVE,
    _audit_data_preflight,
    audit_libero_dataset_snapshot,
    stage_libero_norm_stats,
)
import r142_stage_s.openpi_c as openpi_c
import r142_stage_s.lerobot_compat as lerobot_compat
from r142_stage_s.lerobot_compat import (
    COMPATIBILITY_CONTRACT,
    PINNED_LEROBOT_COMMIT,
    PINNED_LEROBOT_VERSION,
    _stack_scalar_column,
)


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "stage_s_c_undertrained_pai.sh"
PREFLIGHT = ROOT / "scripts" / "stage_s_c_data_preflight.py"
TRAIN = ROOT / "scripts" / "stage_s_libero_c_train.py"
WORKER = ROOT / "scripts" / "stage_s_libero_c_train_worker.py"
CONFIG = ROOT / "configs" / "pai" / "stage_s_c_undertrained.json"


def _make_dataset(root: Path) -> None:
    root.mkdir(parents=True)
    (root / ".gitattributes").write_text("*.parquet filter=lfs\n", encoding="utf-8")
    (root / "README.md").write_text("local frozen fixture\n", encoding="utf-8")
    meta = root / "meta"
    data = root / "data"
    meta.mkdir()
    (data / "chunk-000").mkdir(parents=True)
    (data / "chunk-001").mkdir(parents=True)
    info = dict(LIBERO_DATASET_EXPECTED_INFO)
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (meta / "tasks.jsonl").write_text(
        "".join(json.dumps({"task_index": i, "task": f"task-{i}"}) + "\n" for i in range(40)),
        encoding="utf-8",
    )
    (meta / "episodes.jsonl").write_text(
        "".join(json.dumps({"episode_index": i}) + "\n" for i in range(1693)),
        encoding="utf-8",
    )
    (meta / "stats.json").write_text("{}", encoding="utf-8")
    expected = [
        ".gitattributes",
        "README.md",
        "meta/info.json",
        "meta/tasks.jsonl",
        "meta/episodes.jsonl",
        "meta/stats.json",
    ]
    expected.extend(
        info["data_path"].format(episode_chunk=i // 1000, episode_index=i)
        for i in range(1693)
    )
    sidecars = root / ".cache" / "huggingface" / "download"
    for relative in expected:
        path = root / relative
        if relative.startswith("data/"):
            path.touch()
        sidecar = sidecars / f"{relative}.metadata"
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(f"{LIBERO_DATASET_REVISION}\nfixture-etag\n0\n", encoding="utf-8")
    lines = []
    for relative in sorted(expected):
        digest = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        lines.append(f"{digest}  {relative}\n")
    (root / "DATASET_SHA256SUMS").write_text("".join(lines), encoding="utf-8")


def test_snapshot_audit_requires_complete_revision_pinned_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / LIBERO_DATASET_REPO
    _make_dataset(root)
    monkeypatch.setattr(
        openpi_c,
        "LIBERO_DATASET_MANIFEST_SHA256",
        hashlib.sha256((root / "DATASET_SHA256SUMS").read_bytes()).hexdigest(),
    )
    result = audit_libero_dataset_snapshot(root, persist_manifest=True, verify_hashes=True)
    assert result["valid"] is True
    assert result["revision"] == LIBERO_DATASET_REVISION
    assert result["manifest_sha256"]
    manifest = root / LIBERO_DATASET_MANIFEST_NAME
    assert manifest.is_file()
    changed = root / "data/chunk-001/episode_001692.parquet"
    changed.write_bytes(b"changed")
    invalid = audit_libero_dataset_snapshot(root, verify_hashes=True)
    assert invalid["valid"] is False
    assert any("file SHA-256 mismatch" in error for error in invalid["errors"])


def test_snapshot_audit_refuses_missing_metadata_without_network_fallback(tmp_path: Path) -> None:
    root = tmp_path / LIBERO_DATASET_REPO
    root.mkdir(parents=True)
    result = audit_libero_dataset_snapshot(root, persist_manifest=True, verify_hashes=True)
    assert result["valid"] is False
    assert any("network fallback is forbidden" in error for error in result["errors"])
    assert not (root / LIBERO_DATASET_MANIFEST_NAME).exists()


def test_norm_stats_staging_is_hash_constrained(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source = source_root / "pi05_libero/assets" / LIBERO_DATASET_REPO / "norm_stats.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps({"state": {"mean": [0.0]}}), encoding="utf-8")
    staged_root = tmp_path / "staged"
    result = stage_libero_norm_stats(source_root, staged_root)
    assert result["valid"] is True
    destination = staged_root / LIBERO_NORM_STATS_RUNTIME_RELATIVE
    assert destination.is_file()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == hashlib.sha256(destination.read_bytes()).hexdigest()
    source.write_text(json.dumps({"state": {"mean": [1.0]}}), encoding="utf-8")
    invalid = stage_libero_norm_stats(source_root, staged_root)
    assert invalid["valid"] is False
    assert any("marker differs" in error or "hash mismatch" in error for error in invalid["errors"])


def test_training_gate_rechecks_text_checksum_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The persisted dataset manifest is sha256sum text, not a JSON marker."""

    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    manifest_path = dataset_root / LIBERO_DATASET_MANIFEST_NAME
    manifest_path.write_text("a" * 64 + "  README.md\n", encoding="utf-8")
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    monkeypatch.setattr(openpi_c, "LIBERO_DATASET_MANIFEST_SHA256", manifest_sha)
    staged_norm = tmp_path / "staged" / LIBERO_NORM_STATS_RUNTIME_RELATIVE
    staged_norm.parent.mkdir(parents=True)
    staged_norm.write_text(json.dumps({"state": {"mean": [0.0]}}), encoding="utf-8")
    norm_sha = hashlib.sha256(staged_norm.read_bytes()).hexdigest()
    preflight_path = tmp_path / "DATA_PREFLIGHT.json"
    payload = {
        "schema": "r142-stage-s-c-data-preflight-v2",
        "status": "COMPLETED",
        "dataset": {
            "repo_id": LIBERO_DATASET_REPO,
            "revision": LIBERO_DATASET_REVISION,
            "root": str(dataset_root),
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha,
            "manifest_file_sha256": manifest_sha,
        },
        "norm_stats": {
            "staged_path": str(staged_norm),
            "source_sha256": norm_sha,
            "staged_sha256": norm_sha,
        },
        "official_bindings": {
            "lerobot_compatibility": {
                "valid": True,
                "contract": (
                    "lerobot==0.1.0@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5; "
                    "datasets==3.6.0 native or datasets==4.8.4 numeric-column bridge"
                ),
                "lerobot_version": "0.1.0",
                "lerobot_commit": "0cf864870cf29f4738d3ade893e6fd13fbd7cdb5",
                "datasets_version": "4.8.4",
                "mode": "datasets-column-numeric-bridge",
            }
        },
    }
    payload["payload_sha256"] = openpi_c._canonical_sha256(payload)
    preflight_path.write_text(json.dumps(payload), encoding="utf-8")
    loaded, errors = _audit_data_preflight(preflight_path)
    assert loaded == payload
    assert errors == []
    manifest_path.write_text("b" * 64 + "  README.md\n", encoding="utf-8")
    _, invalid_errors = _audit_data_preflight(preflight_path)
    assert any("manifest SHA-256 mismatch" in error for error in invalid_errors)


def test_launcher_and_training_wrapper_bind_offline_data_gate() -> None:
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    launcher = LAUNCHER.read_text(encoding="utf-8")
    train = TRAIN.read_text(encoding="utf-8")
    preflight = PREFLIGHT.read_text(encoding="utf-8")
    assert "HF_LEROBOT_HOME" in launcher
    for name in ("HF_DATASETS_OFFLINE=1", "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1"):
        assert name in launcher
    assert "stage_s_c_data_preflight.py" in launcher
    assert "--data-preflight" in launcher and "--data-preflight" in train
    assert "snapshot_download" not in launcher
    assert "network fallback is forbidden" in preflight
    assert "LeRobotDatasetMetadata" in preflight
    assert "official pi05_libero config did not load local norm_stats" in preflight
    assert "data_preflight_path" in train
    assert 'cd "$PROJECT_DIR"' in launcher
    assert launcher.index('cd "$PROJECT_DIR"') < launcher.index("CURRENT_STAGE=data_preflight")
    worker = WORKER.read_text(encoding="utf-8")
    assert "install_column_compat_bridge" in worker
    assert "_patch_lerobot_dataset_column_stack" not in worker
    compat = (ROOT / "src/r142_stage_s/lerobot_compat.py").read_text(encoding="utf-8")
    assert '"_query_hf_dataset"' in compat
    assert '"_get_query_timestamps"' in compat


def test_column_bridge_matches_pinned_numeric_transform_and_rejects_ragged() -> None:
    class Column(list):
        pass

    class FakeTensor:
        def __init__(self, value):
            self.value = value
            if isinstance(value, list):
                self.shape = (len(value),)
            else:
                self.shape = ()

    class FakeTorch:
        Tensor = FakeTensor

        @staticmethod
        def tensor(value):
            return FakeTensor(value)

    calls = []

    def original_stack(values, *args, **kwargs):
        calls.append((values, args, kwargs))
        return "native"

    assert _stack_scalar_column(original_stack, FakeTorch, Column, Column([0.0, 1.0])) == "native"
    assert [value.value for value in calls[-1][0]] == [0.0, 1.0]

    class TransformFreeSource:
        def __init__(self) -> None:
            self.formats = []

        def with_format(self, value):
            self.formats.append(value)
            return {"timestamp": [2.0, 3.0]}

    source = TransformFreeSource()
    transformed = Column([99.0])
    transformed.source = source
    transformed.column_name = "timestamp"
    assert _stack_scalar_column(original_stack, FakeTorch, Column, transformed) == "native"
    assert [value.value for value in calls[-1][0]] == [2.0, 3.0]
    assert source.formats == [None]
    assert _stack_scalar_column(original_stack, FakeTorch, Column, [0.0, 1.0]) == "native"
    assert calls[-1] == ([0.0, 1.0], (), {})
    assert _stack_scalar_column(
        original_stack, FakeTorch, Column, Column([FakeTensor(0.0), FakeTensor(1.0)])
    ) == "native"
    assert isinstance(calls[-1][0], tuple)
    assert _stack_scalar_column(
        original_stack, FakeTorch, Column, Column([[0.0, 1.0], [2.0, 3.0]])
    ) == "native"
    assert [value.shape for value in calls[-1][0]] == [(2,), (2,)]
    with pytest.raises(TypeError, match="ragged"):
        _stack_scalar_column(original_stack, FakeTorch, Column, Column([[0.0], [1.0, 2.0]]))
    with pytest.raises(TypeError, match="string"):
        _stack_scalar_column(original_stack, FakeTorch, Column, Column(["bad"]))


def test_dependency_contract_fails_closed_on_unknown_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_distribution(package: str) -> dict[str, object]:
        if package == "lerobot":
            return {"version": PINNED_LEROBOT_VERSION, "direct_url": {}, "source_text": "other-source"}
        return {"version": "4.8.5", "direct_url": {}, "source_text": ""}

    monkeypatch.setattr(lerobot_compat, "_distribution_source", fake_distribution)
    contract = lerobot_compat.runtime_dependency_contract()
    assert contract["valid"] is False
    assert contract["mode"] == "unsupported"
    assert any(PINNED_LEROBOT_COMMIT in error or "outside the frozen" in error for error in contract["errors"])


def test_config_preserves_graphics_contract_and_records_data_contract() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["resource_alias"] == "idle-a800-robot-stage-s-graphics-8gpu"
    assert config["runtime"]["pod_env"] == {
        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics"
    }
    assert config["runtime"]["uid"] == config["runtime"]["gid"] == 2254
    assert config["runtime"]["output_mode"] == "resume"
    assert config["evidence"]["dataset_revision"] == LIBERO_DATASET_REVISION
    assert config["evidence"]["dataset_metadata_missing_network_fallback"] is False
    assert config["evidence"]["hf_offline_environment"] == {
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    assert config["evidence"]["num_train_steps"] == 10001
    assert config["evidence"]["seed"] == 42
    assert config["evidence"]["retained_checkpoint_steps"] == [1000, 3000, 6000, 10000]
    assert config["evidence"]["dataset_manifest_path"].endswith("DATASET_SHA256SUMS")
    assert config["evidence"]["dependency_compatibility_contract"] == COMPATIBILITY_CONTRACT
    assert config["evidence"]["lerobot_expected_commit"] == PINNED_LEROBOT_COMMIT


@pytest.mark.parametrize("path", [LAUNCHER, PREFLIGHT, TRAIN])
def test_c_runtime_source_has_no_pai_submit(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert "pai-job submit" not in text
    assert "CreateJob" not in text
