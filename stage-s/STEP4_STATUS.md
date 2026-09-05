# R142-FP-11 Stage-S / Step-4 checkpoint status

Status: CHECKPOINT_BLOCKED_ON_C_SERIALIZATION

As of: 2026-09-05 (Asia/Shanghai)

## Verified

- C calibration JobId dlc16ic2rudi8xli (r142-stage-s-c-calibration-20260905-r14) reached FIRST_WORK and completed the immutable input, source, checkpoint-schedule, and checkpoint-integrity audits.
- PAI terminal state is Stopped; the master and aimaster pods are both stopped. Failure evidence is retained at: /mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_s/failures/c-calibration-r14-policy-checkpoint-format-20260905.
- All eight ranks failed closed before an episode because CleanPi05LiberoPolicy requires an OpenPI params/ tree, while the accepted C training lineage provides native model.safetensors plus optimizer/RNG sidecars and no params/ directory.
- No C pooled-success row, calibration report, frozen protocol, A/B/C main run, or S4/S5 scientific result is claimed. No checkpoint conversion, interpolation, or artificial degradation was performed.
- Static contract suite: 25 tests passed across test_stage_s_c_calibration_pai.py, test_stage_s_bc_main_pai.py, and test_stage_s_s45.py with PYTHONPATH=src python3 -m pytest.

## Remaining blocker

The pinned Stage-R inference adapter and the accepted C training artifact use incompatible serialization contracts. This is an interface failure before trajectory execution, so success rate, family collapse, recovery, and substrate qualification are not scientifically identifiable. The only valid next step is an explicitly authorized native checkpoint/interface repair; silently fabricating params/ would change the frozen substrate.

The broader freeze test module currently has a stale import (_calibration_selection_key) and fails at collection; this is recorded as a tooling issue, not treated as a scientific result.

## Evidence and publication

- Machine-readable failure: stage-s/results/calibration/C/CALIBRATION_FAILURE.json
- Detailed mechanism analysis: stage-s/CALIBRATION_REPORT.md
- Feishu Step-4 report contains the same JobId, evidence path, terminal state, and mechanism interpretation.
- This status is published on GitHub main together with the complete Step-4 plan, reports, scripts, and preserved failure evidence references.

