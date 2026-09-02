# Stage-S calibration report contract

This file documents the report produced by the terminal-only calibration
freezer. It is a schema/operations document, not a scientific result. No
current B or C calibration result is stored in this repository.

The freezer reads only terminal `CALIBRATION_RESULT.json` rows with the exact
fields `setting`, `successes`, `total`, and `pooled_success`, plus the source
completion marker and its SHA manifests. It rejects any S2--S5, near-all-fail,
rho, divergence, recovery, family, trajectory, genealogy, action, or pose
field before making a selection. Partial, queued, running, or first-work
artifacts are not accepted.

## Selection contract

For each substrate, select the row minimizing
`abs(pooled_success - 0.45)`. B ties use lexicographic setting order over the
four frozen proximity settings. C ties use ascending checkpoint step over
`step_1000`, `step_3000`, `step_6000`, and `step_10000`. The B bundle is fixed
to `r142-stage-s-b-variants-20260903-r7`; C additionally requires the accepted
OpenPI training lineage at commit
`54cbaee6ae0c010a1ed431871cdaa8f4684ac709`, seed 42, terminal step 10001,
and all four real checkpoint artifacts with independently verified manifests.

## Generated report shape

At runtime, the tool atomically writes:

* `/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_s/b_calibration/CALIBRATION_REPORT.json`
* `/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_s/c_calibration/CALIBRATION_REPORT.json`

Both reports use schema `r142-stage-s-calibration-freeze-v1`, status `FROZEN`,
protocol `r142-stage-s-v1`, `calibration_completed=true`, `frozen=true`, and
`no_s2_s5_peeking=true`. They bind source result/marker paths and SHA-256
digests. B binds `selected_setting`, `variant_run_id`,
`selected_variant_root`, and the selected `config.yaml` digest. C binds
`selected_checkpoint`, its artifact digest, and the accepted training lineage
digest. These are the fields consumed by the B/C main-screen loader.

The generated JSON is intentionally absent from this repository until real
terminal calibration evidence exists. The freezer never executes calibration,
loads a model, or submits PAI.
