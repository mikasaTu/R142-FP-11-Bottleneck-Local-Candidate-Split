"""Frozen analysis helpers and fail-closed substrate adapters for Stage-S."""

from .analysis import (
    DECISION_CODES,
    compute_divergence_curve,
    compute_s1,
    compute_s2,
    compute_s3,
    compute_s3_production,
    compute_s4,
    compute_s4_from_protocol,
    compute_s5,
    S3_SUBSTRATE_WORKSPACE_SCALES,
    S4_BOOTSTRAP_SEED,
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
from .robotwin import (
    AtomicFamilyWriter,
    CandidateRecord,
    CapabilityError,
    ExactReplayVerifier,
    ConcreteRoboTwinRuntime,
    EvoProxyStateAdapter,
    FamilyRolloutRunner,
    RoboTwinPins,
    select_published_tasks,
)

__all__ = [
    "AtomicFamilyWriter",
    "CALIBRATION_SCHEMA",
    "CandidateRecord",
    "CapabilityError",
    "DECISION_CODES",
    "ExactReplayVerifier",
    "ConcreteRoboTwinRuntime",
    "EvoProxyStateAdapter",
    "FamilyRolloutRunner",
    "RoboTwinPins",
    "compute_divergence_curve",
    "compute_s1",
    "compute_s2",
    "compute_s3",
    "compute_s3_production",
    "compute_s4",
    "compute_s4_from_protocol",
    "compute_s5",
    "S3_SUBSTRATE_WORKSPACE_SCALES",
    "S4_BOOTSTRAP_SEED",
    "decide_stage_s",
    "evaluate_substrate",
    "family_collapse_metrics",
    "make_calibration_record",
    "matched_time_tau",
    "normalized_workspace_pose_rms",
    "select_calibration_setting",
    "select_published_tasks",
    "sha256_file",
    "verify_completion_bundle",
    "verify_completed_json",
    "verify_sha256sums",
    "write_completion",
    "write_sha256sums",
]

from .analysis import matched_time_tau_curve

if "matched_time_tau_curve" not in __all__:
    __all__.append("matched_time_tau_curve")

from .total_analysis import (
    TOTAL_ANALYSIS_SCHEMA,
    TotalAnalysisError,
    analyze_all_stage_s,
    analyze_stage_s,
    finalise_stage_s,
    finalize_stage_s,
)

for _name in (
    "TOTAL_ANALYSIS_SCHEMA", "TotalAnalysisError", "analyze_all_stage_s",
    "analyze_stage_s", "finalise_stage_s", "finalize_stage_s",
):
    if _name not in __all__:
        __all__.append(_name)
