from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from r142_stage_s.s45_runtime import (
    PROTOCOL_ID,
    ProtocolAuthority,
    S45Adapter,
    S45BundleError,
    S45CapabilityError,
    S45ProvenanceError,
    discover_n32_families,
    load_adapter,
    verify_atomic_bundle,
    write_atomic_bundle,
)


SOURCE_COMMIT = "0" * 40
SOURCE_SHA256 = "1" * 64


def _snapshot(index: int) -> dict:
    return {
        "simulator_state": {"state": [index]},
        "observation_history": [],
        "action_queue": [],
        "rng_state": {
            "python": {"state": [index]},
            "numpy": {"state": [index]},
            "torch": {"cpu": [index], "cuda": []},
            "environment_owner": {"state": [index]},
            "policy_owner": {"state": [index]},
        },
        "snapshot_restore_check": {
            "same_action": True,
            "passed": True,
            "max_abs_error": 0.0,
        },
    }


def _family_payload(*, bad_index: int | None = None, bad_snapshot: bool = False) -> tuple[dict, dict]:
    candidates = []
    for index in range(32):
        action = [[float(index), 0.0]]
        row = {
            "candidate_index": index,
            "candidate_id": str(index),
            "candidate_seed": 1000 + index,
            "success": False,
            "final_success": False,
            "family_id": "family-b",
            "root_family_id": "family-b",
            "parent_id": None,
            "generation_step": 0,
            "action_prefix": action,
            "actions": action,
            "trajectory": [[0.0] * 6],
            "terminated": True,
            "termination": "official_terminal",
            "snapshot": _snapshot(index),
        }
        if bad_index == index:
            if bad_snapshot:
                row["snapshot"]["rng_state"].pop("policy_owner")
            else:
                row["parent_id"] = "31"
                row["generation_step"] = 1
        candidates.append(row)
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "family_id": "family-b",
        "substrate": "B",
        "termination": "official_terminal",
        "main_source_commit": SOURCE_COMMIT,
        "main_source_sha256": SOURCE_SHA256,
    }
    marker = {
        "schema": "r142-stage-s-main-family-v1",
        "marker_type": "completed_family",
        "protocol_id": PROTOCOL_ID,
        "family_id": "family-b",
        "substrate": "B",
        "candidate_count": 32,
        "main_source_commit": SOURCE_COMMIT,
        "main_source_sha256": SOURCE_SHA256,
    }
    return {"family_id": "family-b", "metadata": metadata, "candidates": candidates}, marker


def _write_family(root: Path, *, bad_index: int | None = None, bad_snapshot: bool = False) -> Path:
    payload, marker = _family_payload(bad_index=bad_index, bad_snapshot=bad_snapshot)
    family_root = root / "family-b"
    artifacts = {
        "family.json": json.dumps(payload, sort_keys=True).encode(),
        "metadata.json": json.dumps(payload["metadata"], sort_keys=True).encode(),
    }
    write_atomic_bundle(family_root, artifacts, marker_name="COMPLETED_FAMILY.json", marker_payload=marker)
    return family_root


def test_bundle_rejects_unregistered_extra_file(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    write_atomic_bundle(root, {"payload.json": b"{}"}, marker_name="COMPLETED_RESULT.json", marker_payload={"schema": "x"})
    (root / "unregistered.json").write_text("{}", encoding="utf-8")
    with pytest.raises(S45BundleError, match="not closed"):
        verify_atomic_bundle(root, marker_name="COMPLETED_RESULT.json")


def test_production_discovery_accepts_complete_genealogy_and_snapshots(tmp_path: Path) -> None:
    family_root = _write_family(tmp_path)
    families = discover_n32_families(tmp_path)
    assert len(families) == 1
    assert families[0].main_source_commit == SOURCE_COMMIT
    assert families[0].main_source_sha256 == SOURCE_SHA256
    assert family_root.exists()


def test_production_discovery_rejects_parent_out_of_order(tmp_path: Path) -> None:
    _write_family(tmp_path, bad_index=3)
    with pytest.raises((S45ProvenanceError, S45BundleError), match="parent|genealogy"):
        discover_n32_families(tmp_path)


def test_production_discovery_rejects_missing_owner_rng(tmp_path: Path) -> None:
    _write_family(tmp_path, bad_index=4, bad_snapshot=True)
    with pytest.raises(S45ProvenanceError, match="RNG streams"):
        discover_n32_families(tmp_path)


class FakeProductionAdapter(S45Adapter):
    pass


def fake_factory(**_kwargs):
    return FakeProductionAdapter()


def _production_protocol(tmp_path: Path) -> ProtocolAuthority:
    path = tmp_path / "authority.json"
    path.write_text("{}", encoding="utf-8")
    return ProtocolAuthority(
        path=path,
        sha256="2" * 64,
        payload={},
        protocol_id=PROTOCOL_ID,
        git_commit="3" * 40,
        s4={},
        s5={},
    )


def test_production_adapter_rejects_fake_factory(tmp_path: Path) -> None:
    protocol = _production_protocol(tmp_path)
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader("fake_stage_s_module", loader=None))
    module.fake_factory = fake_factory
    import sys

    sys.modules[module.__name__] = module
    try:
        with pytest.raises(S45CapabilityError, match="fake/synthetic"):
            load_adapter(f"{module.__name__}:fake_factory", protocol=protocol, substrate="B")
    finally:
        sys.modules.pop(module.__name__, None)


def test_pai_template_rejects_unresolved_pins() -> None:
    validator_path = Path(__file__).parents[1] / "scripts" / "validate_stage_s45_pai_contract.py"
    spec = importlib.util.spec_from_file_location("stage_s45_pai_validator", validator_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config_path = Path(__file__).parents[1] / "configs" / "pai" / "stage_s_s4.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    with pytest.raises(module.ContractError, match="placeholder"):
        module.validate(payload)


def test_pai_validator_rejects_resource_drift() -> None:
    validator_path = Path(__file__).parents[1] / "scripts" / "validate_stage_s45_pai_contract.py"
    spec = importlib.util.spec_from_file_location("stage_s45_pai_validator_resource", validator_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config_path = Path(__file__).parents[1] / "configs" / "pai" / "stage_s_s5.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["resource_alias"] = "wrong-resource"
    with pytest.raises(module.ContractError, match="resource_alias"):
        module.validate(payload)


def test_pai_validator_accepts_frozen_repinned_contract() -> None:
    validator_path = Path(__file__).parents[1] / "scripts" / "validate_stage_s45_pai_contract.py"
    spec = importlib.util.spec_from_file_location("stage_s45_pai_validator_positive", validator_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config_path = Path(__file__).parents[1] / "configs" / "pai" / "stage_s_s5.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))

    def repin(value, key: str = ""):
        if isinstance(value, dict):
            return {name: repin(child, name) for name, child in value.items()}
        if isinstance(value, list):
            return [repin(child, key) for child in value]
        if value == "__REQUIRED_AFTER_FREEZE_REAL_ADAPTER_MODULE_FACTORY__":
            return "r142_stage_s.s45_adapters:make_robo_twin_s45_adapter"
        if isinstance(value, str) and value.startswith("__REQUIRED_"):
            if "COMMIT" in value:
                return "a" * 40
            return "b" * 64
        return value

    payload = repin(payload)
    payload["status"] = "FROZEN_READY"
    payload["evidence"]["contract_ready"] = True
    payload["evidence"]["validation_status"] = "passed"
    result = module.validate(payload, now=module.dt.datetime(2026, 9, 3, 12, 0, tzinfo=module.ZoneInfo("Asia/Shanghai")))
    assert result["status"] == "VALID"
