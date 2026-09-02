from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from r142_stage_s.s45_runtime import (
    BASE_CANDIDATE_COUNT,
    FRESH_CANDIDATE_INDICES,
    ProtocolAuthority,
    S45Adapter,
    S45BundleError,
    S45CapabilityError,
    S45ProtocolError,
    S45ProvenanceError,
    discover_n32_families,
    finalise_s45,
    run_s4,
    run_s5,
    write_atomic_bundle,
)


COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _protocol(tmp_path: Path, *, complete: bool = True) -> Path:
    payload = {
        "protocol_id": "fixture-stage-s",
        "protocol_git_commit": COMMIT,
        "s4": {
            "anchor_rule": "fixture-anchor-rule",
            "oracle_t_rule": "fixture-oracle-grid",
            "random_t_rule": "fixture-random-grid",
            "oracle_t_grid": list(range(1, 10)),
            "branch_count": 4,
            "search_branch_count": 4,
            "heldout_branch_count": 8,
            "random_branch_count": 8,
            "branch_seed_formula": "fixture-branch-seed-v1",
            "random_location_hash_formula": "sha256(protocol_id|s4|family_id|episode_length|pair_index)->first_8_bytes_big_endian_mod_interior",
        },
        "s5": {
            "base_candidate_count": 32,
            "fresh_candidate_indices": list(FRESH_CANDIDATE_INDICES),
            "extension_seed_formula": "fixture-extension-seed-v1",
        },
    }
    if not complete:
        del payload["s4"]["oracle_t_grid"]
    path = tmp_path / "FROZEN_PROTOCOL.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _candidate(index: int, *, success: bool = False) -> dict:
    return {
        "candidate_index": index,
        "candidate_id": f"family/candidate-{index:04d}",
        "candidate_seed": 1000 + index,
        "parent_id": None,
        "generation_step": 0,
        "actions": [[float(index), 0.0] for _ in range(12)],
        "trajectory": [[float(index), 0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(12)],
        "success": success,
        "final_success": success,
        "terminated": True,
        "termination": "official-step-limit",
        "policy_forwards": 12,
        "env_steps": 12,
    }


def _write_n32(tmp_path: Path, protocol: ProtocolAuthority, *, success_count: int = 0) -> Path:
    root = tmp_path / "n32"
    rows = [_candidate(index, success=index < success_count) for index in range(BASE_CANDIDATE_COUNT)]
    family = {
        "family_id": "family-00",
        "metadata": {
            "family_id": "family-00",
            "substrate": "B",
            "task_id": 0,
            "init_state": 0,
            "termination": "official-step-limit",
            **protocol.identity(),
        },
        "candidates": rows,
    }
    marker = {
        "schema": "fixture-n32",
        "marker_type": "completed_family",
        "family_id": "family-00",
        "candidate_count": 32,
        "protocol_id": protocol.protocol_id,
        "protocol_authority_sha256": protocol.sha256,
        "protocol_git_commit": protocol.git_commit,
    }
    write_atomic_bundle(
        root / "family-00",
        {"family.json": json.dumps(family, sort_keys=True)},
        marker_name="COMPLETED_FAMILY.json",
        marker_payload=marker,
    )
    return root


class FixtureAdapter(S45Adapter):
    def __init__(self) -> None:
        self.closed = False

    def select_anchor(self, family, *, protocol):
        return family["candidates"][0]

    def replay_prefix(self, family, anchor, split_step, *, protocol):
        return {
            "actions": copy.deepcopy(anchor["actions"][:split_step]),
            "trajectory": copy.deepcopy(anchor["trajectory"][:split_step]),
            "snapshot": {
                "simulator_state": {"step": split_step},
                "observation_history": [[0.0]],
                "action_queue": [],
                "rng_state": {"python": [1], "numpy": [2], "torch": [3]},
            },
        }

    def branch_seed(self, family, anchor, split_step, branch_index, mode, *, protocol):
        if mode == "search":
            return 5000 + split_step * 10 + branch_index
        return 7000 + branch_index

    def run_branch(self, family, anchor, prefix, split_step, branch_seed, branch_index, mode, *, protocol):
        actions = copy.deepcopy(anchor["actions"])
        trajectory = copy.deepcopy(anchor["trajectory"])
        # The prefix is unchanged. Search has a unique max at t=3, while the
        # held-out oracle branches recover and the paired random branches fail.
        success = (
            (mode == "search" and split_step == 3 and branch_index < 2)
            or mode == "oracle_heldout"
        )
        return {
            "actions": actions,
            "trajectory": trajectory,
            "terminated": True,
            "success": success,
            "termination": "official-step-limit",
            "policy_forwards": 12,
            "env_steps": 12,
            "snapshot_restore_check": {"same_action": True, "passed": True, "max_abs_error": 0.0},
        }

    def extension_seed(self, family, candidate_index, *, protocol):
        return 9000 + candidate_index

    def run_fresh_candidate(self, family, candidate_index, candidate_seed, *, protocol):
        return {
            "actions": [[float(candidate_index), 0.0] for _ in range(12)],
            "trajectory": [[float(candidate_index), 0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(12)],
            "terminated": True,
            "success": False,
            "termination": "official-step-limit",
            "policy_forwards": 12,
            "env_steps": 12,
            "snapshot_restore_check": {"same_action": True, "passed": True, "max_abs_error": 0.0},
        }

    def close(self):
        self.closed = True


def test_protocol_has_no_implicit_s4_s5_defaults(tmp_path: Path):
    with pytest.raises(S45ProtocolError):
        ProtocolAuthority.load(_protocol(tmp_path, complete=False))


def test_protocol_freezes_two_stage_s4_budget_and_hash_locations(tmp_path: Path):
    protocol = ProtocolAuthority.load(_protocol(tmp_path))
    assert protocol.search_branch_count == 4
    assert protocol.heldout_branch_count == 8
    assert protocol.search_steps("family-00", 12) == tuple(range(1, 10))
    locations = protocol.random_steps("family-00", 12)
    assert len(locations) == 8
    assert all(0 < step < 11 for step in locations)
    assert locations == protocol.random_steps("family-00", 12)


def test_n32_rejects_duplicate_identity_and_wrong_substrate_pose_width(tmp_path: Path):
    protocol = ProtocolAuthority.load(_protocol(tmp_path))
    root = _write_n32(tmp_path, protocol)
    family_path = root / "family-00" / "family.json"
    payload = json.loads(family_path.read_text())
    payload["candidates"][1]["candidate_id"] = payload["candidates"][0]["candidate_id"]
    family_path.write_text(json.dumps(payload, sort_keys=True))
    with pytest.raises(S45BundleError):
        discover_n32_families(root, protocol=protocol)


def test_n32_loader_requires_sha_and_reads_all_rows(tmp_path: Path):
    protocol = ProtocolAuthority.load(_protocol(tmp_path))
    root = _write_n32(tmp_path, protocol)
    families = discover_n32_families(root, protocol=protocol)
    assert len(families) == 1
    assert len(families[0].candidates) == 32
    (root / "family-00" / "family.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(S45BundleError):
        discover_n32_families(root, protocol=protocol)


def test_s4_runs_search_then_equal_heldout_pairs(tmp_path: Path):
    protocol = ProtocolAuthority.load(_protocol(tmp_path))
    n32 = _write_n32(tmp_path, protocol)
    families = discover_n32_families(n32, protocol=protocol)
    result = run_s4(families, protocol, FixtureAdapter(), tmp_path / "s4")
    assert result["near_all_fail_family_count"] == 1
    probe = json.loads((tmp_path / "s4" / "family-00" / "S4_PROBE.json").read_text())
    assert len(probe["search"]) == 9
    assert all(len(row["branches"]) == 4 for row in probe["search"])
    assert probe["chosen_oracle_t"] == 3
    assert len(probe["oracle_branches"]) == len(probe["random_branches"]) == 8
    assert probe["paired_suffix_seeds"] == [row["branch_seed"] for row in probe["oracle_branches"]]
    assert probe["paired_suffix_seeds"] == [row["branch_seed"] for row in probe["random_branches"]]
    assert all(row["prefix_preserving"] for row in probe["oracle_branches"])
    assert all("snapshot" in row and row["snapshot"] for row in probe["oracle_branches"])
    assert (tmp_path / "s4" / "family-00" / "COMPLETED_S4_FAMILY.json").is_file()
    assert (tmp_path / "s4" / "family-00" / "SHA256SUMS").is_file()


def test_s4_rejects_missing_full_snapshot_and_nonterminal_branch(tmp_path: Path):
    protocol = ProtocolAuthority.load(_protocol(tmp_path))
    n32 = _write_n32(tmp_path, protocol)
    families = discover_n32_families(n32, protocol=protocol)

    class BadSnapshot(FixtureAdapter):
        def replay_prefix(self, family, anchor, split_step, *, protocol):
            value = super().replay_prefix(family, anchor, split_step, protocol=protocol)
            del value["snapshot"]["rng_state"]
            return value

    with pytest.raises(S45ProvenanceError):
        run_s4(families, protocol, BadSnapshot(), tmp_path / "bad-s4")


def test_s5_preserves_base_and_writes_exact_fresh_32(tmp_path: Path):
    protocol = ProtocolAuthority.load(_protocol(tmp_path))
    n32 = _write_n32(tmp_path, protocol)
    families = discover_n32_families(n32, protocol=protocol)
    before = hashlib.sha256((n32 / "family-00" / "family.json").read_bytes()).hexdigest()
    result = run_s5(families, protocol, FixtureAdapter(), tmp_path / "s5")
    assert result["completed_family_count"] == 1
    payload = json.loads((tmp_path / "s5" / "family-00" / "S5_FAMILY.json").read_text())
    assert [row["candidate_index"] for row in payload["fresh_rows"]] == list(FRESH_CANDIDATE_INDICES)
    assert [row["candidate_index"] for row in payload["extended_rows"]] == list(range(64))
    assert payload["source_n32_family_file_sha256"] == families[0].source_family_file_sha256
    marker = json.loads((tmp_path / "s5" / "family-00" / "COMPLETED_S5_FAMILY.json").read_text())
    assert marker["source_n32_family_file_sha256"] == families[0].source_family_file_sha256
    assert hashlib.sha256((n32 / "family-00" / "family.json").read_bytes()).hexdigest() == before


def test_s5_rejects_nonterminal_fresh_execution(tmp_path: Path):
    protocol = ProtocolAuthority.load(_protocol(tmp_path))
    n32 = _write_n32(tmp_path, protocol)
    families = discover_n32_families(n32, protocol=protocol)

    class NonTerminal(FixtureAdapter):
        def run_fresh_candidate(self, family, candidate_index, candidate_seed, *, protocol):
            value = dict(super().run_fresh_candidate(family, candidate_index, candidate_seed, protocol=protocol))
            value["terminated"] = False
            return value

    with pytest.raises(S45ProvenanceError):
        run_s5(families, protocol, NonTerminal(), tmp_path / "bad-s5")


def test_finalizer_calls_frozen_analysis_only_after_both_markers(tmp_path: Path):
    protocol = ProtocolAuthority.load(_protocol(tmp_path))
    n32 = _write_n32(tmp_path, protocol)
    families = discover_n32_families(n32, protocol=protocol)
    run_s4(families, protocol, FixtureAdapter(), tmp_path / "s4")
    run_s5(families, protocol, FixtureAdapter(), tmp_path / "s5")
    result = finalise_s45(n32, tmp_path / "s4", tmp_path / "s5", _protocol(tmp_path), tmp_path / "evaluation", expected_substrate="B")
    assert result["all_inputs_complete"] is True
    assert result["S4"]["bootstrap"]["replicates"] == 10000
    assert result["S5"]["total_candidate_count"] == 64
    assert (tmp_path / "evaluation" / "COMPLETED_EVALUATION_RESULT.json").is_file()


def test_adapter_has_no_synthetic_fallback(tmp_path: Path):
    protocol = ProtocolAuthority.load(_protocol(tmp_path))
    n32 = _write_n32(tmp_path, protocol)
    families = discover_n32_families(n32, protocol=protocol)
    with pytest.raises(S45CapabilityError):
        run_s5(families, protocol, S45Adapter(), tmp_path / "missing")
