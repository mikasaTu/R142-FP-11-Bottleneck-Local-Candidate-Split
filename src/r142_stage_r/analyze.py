from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import task_metrics
from .phase0 import load_task_rollouts
from .protocol import PROTOCOL_ID, atomic_json, sha256_file


def _bootstrap_success(rollouts: list[dict[str, Any]], seed: int, replicates: int = 10000) -> list[float]:
    by_state: dict[int, list[bool]] = {}
    for row in rollouts:
        by_state.setdefault(int(row["init_state"]), []).append(bool(row["success"]))
    state_values = np.asarray([np.mean(by_state[key]) for key in sorted(by_state)], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    samples = rng.choice(state_values, size=(int(replicates), len(state_values)), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def analyze_phase0(raw_dir: str | Path, thresholds_file: str | Path, output_dir: str | Path) -> dict[str, Any]:
    raw_root = Path(raw_dir)
    output_root = Path(output_dir)
    threshold_payload = json.loads(Path(thresholds_file).read_text(encoding="utf-8"))
    if threshold_payload.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("threshold protocol mismatch")
    if not threshold_payload.get("positive_control_pass"):
        decision = "PIPELINE_INVALID"
    else:
        decision = None
    rows = []
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        for task_id in range(10):
            stem = f"{suite}_task{task_id:02d}"
            metadata_path = raw_root / f"{stem}.json"
            npz_path = raw_root / f"{stem}.npz"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata["data_sha256"] != sha256_file(npz_path):
                raise RuntimeError(f"SHA mismatch for {npz_path}")
            rollouts = load_task_rollouts(npz_path)
            metrics = task_metrics(rollouts, threshold_payload["thresholds"])
            metrics["success_rate_ci95"] = _bootstrap_success(rollouts, seed=142000 + task_id)
            rows.append({"suite": suite, "task_id": task_id, "prompt": metadata["prompt"], **metrics})
    rows.append(
        {
            "suite": "robotwin",
            "task_id": None,
            "prompt": None,
            "retained": False,
            "source_status": "SOURCE_LIMITATION_UNVERIFIABLE",
            "rollout_count": 0,
        }
    )
    retained = [row for row in rows if row.get("retained")]
    retained.sort(
        key=lambda row: (
            -float(row["rho"]),
            -float(row["low_p_fraction"]),
            -float(row["median_t_div_episode_fraction"]),
            -int(row["stable_modes"]),
            str(row["suite"]),
            int(row["task_id"]),
        )
    )
    retained = retained[:3]
    if decision is None:
        decision = "CHECKPOINT1_TASKS_RETAINED" if retained else "NO_STAGE_R_PRECONDITION_ON_PINNED_PI05_LIBERO"
    payload = {
        "protocol_id": PROTOCOL_ID,
        "decision": decision,
        "positive_control_pass": bool(threshold_payload.get("positive_control_pass")),
        "threshold_file": str(Path(thresholds_file)),
        "threshold_sha256": sha256_file(thresholds_file),
        "candidate_rows": rows,
        "retained_tasks": [{"suite": row["suite"], "task_id": row["task_id"]} for row in retained],
        "checkpoint": "CHECKPOINT_1_STOP",
        "phase1_authorized": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "phase0r_summary.json"
    atomic_json(summary_path, payload)
    completed = {
        "protocol_id": PROTOCOL_ID,
        "decision": decision,
        "summary": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "checkpoint": "CHECKPOINT_1_STOP",
    }
    atomic_json(output_root / "COMPLETED_PHASE0R.json", completed)
    return payload
