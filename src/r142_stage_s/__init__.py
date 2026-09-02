"""Frozen statistical and artifact-integrity helpers for R142 Stage-S.

The package intentionally contains no rollout adapter. It consumes persisted
episode/family records and applies the Stage-S preregistered rules without
changing the sampling or substrate implementation.
"""

from .analysis import (
    DECISION_CODES,
    compute_divergence_curve,
    compute_s1,
    compute_s2,
    compute_s3,
    compute_s4,
    compute_s5,
    decide_stage_s,
    evaluate_substrate,
    family_collapse_metrics,
    matched_time_tau,
    normalized_workspace_pose_rms,
)
from .calibration import (
    CALIBRATION_SCHEMA,
    make_calibration_record,
    select_calibration_setting,
)
from .integrity import (
    sha256_file,
    verify_completion_bundle,
    verify_completed_json,
    verify_sha256sums,
    write_completion,
    write_sha256sums,
)

__all__ = [
    "CALIBRATION_SCHEMA",
    "DECISION_CODES",
    "compute_divergence_curve",
    "compute_s1",
    "compute_s2",
    "compute_s3",
    "compute_s4",
    "compute_s5",
    "decide_stage_s",
    "evaluate_substrate",
    "family_collapse_metrics",
    "make_calibration_record",
    "matched_time_tau",
    "normalized_workspace_pose_rms",
    "select_calibration_setting",
    "sha256_file",
    "verify_completion_bundle",
    "verify_completed_json",
    "verify_sha256sums",
    "write_completion",
    "write_sha256sums",
]
