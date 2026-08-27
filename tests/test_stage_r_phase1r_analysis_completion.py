from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import r142_stage_r.phase1_analysis as analysis
from r142_stage_r.phase1 import PHASE1_PROTOCOL_ID, TASKS


def _matrix(sensitivity: float = 0.5) -> np.ndarray:
    values = np.zeros((12, 10, 16), dtype=np.float64)
    values[:, -1, :] = float(sensitivity)
    return values


def _metadata_accounting(cell_count: int = 120) -> dict[str, int]:
    return {
        "cell_count": int(cell_count),
        "policy_forwards": int(cell_count * 10),
        "logical_policy_forwards": int(cell_count * 10),
        "policy_batches": int(cell_count * 5),
        "physical_policy_batches": int(cell_count * 3),
        "environment_steps": int(cell_count * 100),
        "branch_count": int(cell_count * 16),
    }


def _calibration_file(path: Path, threshold: float = 0.5) -> Path:
    path.write_text(
        json.dumps(
            {
                "protocol_id": PHASE1_PROTOCOL_ID,
                "artifact": "BLINDED_PHASE1R_CALIBRATION",
                "shuffles": 1000,
                "seed": analysis.PERMUTATION_SEED,
                "thresholds_by_control": {"positive": threshold, "null": 0.0},
                "location_sensitivity_threshold": threshold,
                "unpermuted_curve_present": False,
                "natural_curve_present": False,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_task_permutation_is_deterministic_and_has_frozen_minimum() -> None:
    matrix = _matrix(1.0)
    first = analysis.task_permutation_statistics(matrix, "libero_spatial", 0, shuffles=1000)
    second = analysis.task_permutation_statistics(matrix, "libero_spatial", 0, shuffles=1000)
    assert first["shuffles"] == 1000
    assert len(first["null_distribution"]) == 1000
    assert first["permutation_seed"] == second["permutation_seed"]
    assert first["null_distribution_sha256"] == second["null_distribution_sha256"]
    assert first["p95"] == second["p95"]
    assert first["empirical_p_value"] == second["empirical_p_value"]
    digest = hashlib.sha256(
        np.asarray(first["null_distribution"], dtype=np.float64).tobytes()
    ).hexdigest()
    assert digest == first["null_distribution_sha256"]
    with pytest.raises(ValueError, match="at least 1000"):
        analysis.task_permutation_statistics(matrix, "libero_spatial", 0, shuffles=999)


def test_decision_tree_inclusive_threshold_and_all_branches() -> None:
    assert analysis.phase1r_decision_label(False, 0.5, 0.5) == "PIPELINE_INVALID"
    assert (
        analysis.phase1r_decision_label(True, 0.499999, 0.5)
        == "NO_TRAJECTORY_BOTTLENECK_ON_PINNED_PI05_LIBERO"
    )
    assert (
        analysis.phase1r_decision_label(True, 0.5, 0.5)
        == "TRAJECTORY_NONFLATNESS_DETECTED_CHECKPOINT_2"
    )
    assert (
        analysis.phase1r_decision_label(True, 0.75, 0.5)
        == "TRAJECTORY_NONFLATNESS_DETECTED_CHECKPOINT_2"
    )


def test_analysis_contract_counts_permutations_and_checksum_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    natural = tmp_path / "natural"
    controls = tmp_path / "controls"
    calibration = _calibration_file(tmp_path / "calibration.json")
    output = tmp_path / "analysis"
    natural_matrix = _matrix(0.5)
    control_positive = _matrix(0.75)
    control_null = _matrix(0.0)
    calls: list[tuple[str, int, int]] = []

    def fake_natural(
        root: str | Path,
        *,
        suite: str,
        task_id: int,
        stream: str,
        require_owner: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, dict[str, int]]:
        del root, require_owner
        assert stream in {"calibration", "heldout"}
        return natural_matrix.copy(), _metadata_accounting()

    def fake_control(
        root: str | Path,
        kind: str,
        *,
        stream: str,
        require_owner: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, dict[str, int]]:
        del root, require_owner
        assert stream in {"calibration", "heldout"}
        return (
            (control_positive if kind == "positive" else control_null).copy(),
            _metadata_accounting(),
        )

    def fake_permutation(
        matrix: np.ndarray,
        suite: str,
        task_id: int,
        *,
        shuffles: int,
        seed: int,
    ) -> dict[str, object]:
        assert matrix.shape == (12, 10, 16)
        assert shuffles == 1000
        calls.append((suite, int(task_id), int(shuffles)))
        samples = np.linspace(0.0, 1.0, shuffles, dtype=np.float64)
        observed = float(np.ptp(analysis.location_curve(matrix)))
        return {
            "suite": suite,
            "task_id": int(task_id),
            "shuffles": int(shuffles),
            "seed": int(seed),
            "permutation_seed": analysis._stable_seed(
                PHASE1_PROTOCOL_ID,
                "natural",
                "branch-location-permutation",
                suite,
                int(task_id),
                int(seed),
            ),
            "permutation_unit": "within_episode_branch_location_labels",
            "observed_sensitivity": observed,
            "null_p95": float(np.quantile(samples, 0.95)),
            "p95": float(np.quantile(samples, 0.95)),
            "empirical_p_value": float(
                (np.count_nonzero(samples >= observed) + 1) / (len(samples) + 1)
            ),
            "null_distribution_sha256": analysis._float64_digest(samples),
            "null_distribution": [float(value) for value in samples],
        }

    monkeypatch.setattr(analysis, "_load_success_matrix", fake_natural)
    monkeypatch.setattr(analysis, "_load_control_matrix", fake_control)
    monkeypatch.setattr(analysis, "task_permutation_statistics", fake_permutation)
    summary = analysis.analyze_phase1r(
        natural,
        controls,
        calibration,
        output,
        bootstrap_replicates=10000,
    )
    assert len(calls) == 40
    assert {(suite, task_id) for suite, task_id, _ in calls} == set(TASKS)
    assert summary["decision_label"] == "TRAJECTORY_NONFLATNESS_DETECTED_CHECKPOINT_2"
    assert summary["natural_cells_total"] == 9600
    assert summary["natural_heldout_cells"] == 4800
    assert summary["natural_calibration_cells"] == 4800
    assert summary["control_cells"] == 480
    assert summary["control_heldout_cells"] == 240
    assert summary["control_calibration_cells"] == 240
    assert summary["bootstrap_replicates"] == 10000
    assert summary["null_control_pass"] is True
    assert summary["compute"]["natural"]["both_streams"]["cell_count"] == 9600
    assert summary["compute"]["natural"]["heldout"]["cell_count"] == 4800
    assert summary["compute"]["natural"]["calibration"]["cell_count"] == 4800
    assert summary["compute"]["zero_budget_slack"] is True

    valid, errors = analysis.validate_phase1_analysis(output)
    assert valid, errors
    cli_env = os.environ.copy()
    src_path = str(Path(__file__).parents[1] / "src")
    cli_env["PYTHONPATH"] = src_path + os.pathsep + cli_env.get("PYTHONPATH", "")
    cli = subprocess.run(
        [
            sys.executable,
            "scripts/stage_r_phase1r.py",
            "validate-analysis",
            "--output",
            str(output),
        ],
        cwd=Path(__file__).parents[1],
        env=cli_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert cli.returncode == 0, cli.stdout + cli.stderr
    assert json.loads(cli.stdout)["valid"] is True

    curves_path = output / "phase1r_task_curves.json"
    curves_payload = json.loads(curves_path.read_text(encoding="utf-8"))
    curves_payload["tasks"][0]["analysis_curve"][0] += 0.125
    curves_path.write_text(json.dumps(curves_payload), encoding="utf-8")
    valid, errors = analysis.validate_phase1_analysis(output)
    assert not valid
    assert any("SHA" in error or "checksum" in error for error in errors)


def test_paired_bootstrap_rejects_non_frozen_replicate_count() -> None:
    with pytest.raises(ValueError, match="exactly 10000"):
        analysis.paired_episode_bootstrap([_matrix()], seed=1, replicates=999)
