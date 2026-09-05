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

For C, the native lineage input is the current
`ACCEPTED_C_TRAINING.json` (`r142-stage-s-c-training-acceptance-v1`) schema.
It must be `status=ACCEPTED`, `label=WEAK_SUBSTRATE`, and
`pai_terminal_status=Succeeded`, with exact Stage-S/QPILOTS/OpenPI/LIBERO
source pins. The freezer follows its `checkpoint_completion`,
`training_pipeline_completion`, `checkpoint_sha256_manifest`, and
`log_sha256_manifest` references, verifies their declared SHA-256 values and
terminal bindings, then rechecks every manifest member. The four exact
`checkpoint_hashes` entries (`<step>/model.safetensors`) must agree with both
the checkpoint files and the checkpoint bundle manifest. A stale marker or
mutated model therefore cannot produce a C report. The older direct
`COMPLETED_C_TRAINING.json`/`checkpoint_audit.checkpoints` form remains
accepted for fixture/backward compatibility, but it is held to the same
terminal OpenPI, schedule, and artifact checks.

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


## Step-S C calibration outcome (2026-09-05)

The C calibration submission reached FIRST_WORK and completed the immutable input/checkpoint audits, but all eight shard ranks failed closed when loading the first checkpoint. The pinned `CleanPi05LiberoPolicy` loader requires an OpenPI checkpoint containing a `params/` directory; the accepted C training lineage contains native `model.safetensors` plus optimizer/RNG sidecars and no `params/` directory. This is a serialization/interface incompatibility, not a scientific success or failure measurement. No C pooled-success row is accepted, no `CALIBRATION_REPORT.json` is emitted, and the Step-S protocol remains unfrozen. Exact PAI evidence is preserved under `logs/r142_fp11_stage_s/failures/c-calibration-r14-policy-checkpoint-format-20260905` and JobId `dlc16ic2rudi8xli` was stopped fail-closed.

Mechanism interpretation: the C training artifact and Stage-R inference adapter serialize different parameter trees. The adapter rejects before an episode because it cannot reconstruct the policy parameter tree; therefore there is no valid success-rate, collapse, or recovery statistic to interpret. Converting or fabricating a `params/` tree would change the frozen substrate and is intentionally not performed.
