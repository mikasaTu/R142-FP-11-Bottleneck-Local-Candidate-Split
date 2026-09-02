# Stage-S B calibration PAI contract

This is the non-submitting PAI hand-off for the Stage-S B calibration. It
uses the maintained Stage-R policy and the real LIBERO simulator. The
launcher is `scripts/stage_s_libero_b_calibration_pai.sh`; the corresponding
registry manifest is `configs/pai/stage_s_b_calibration.json`.

## Frozen experiment

The runtime uses the four B settings
`proximity_0.06m`, `proximity_0.08m`, `proximity_0.10m`, and
`proximity_0.12m`. Each setting evaluates task IDs `0, 3, 6, 9`, initial
state indices `0..7`, and eight candidates, for 256 episodes per setting.
The calibration seed is `142042`, and the global shard world size is exactly
8. Rank assignment is deterministic and independent of the trial seed.

The only calibration rows that are written are `setting`, `successes`,
`total`, and `pooled_success`. The final completion record additionally
contains source, provenance, compute, rank-marker, and SHA metadata. No
trial trace is used as a calibration output.

## Input gate

The read-only B input is the completed r7 bundle:

```text
/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_s/b_variants/
  r142-stage-s-b-variants-20260903-r7/
```

Before a single episode, the launcher requires:

* `COMPLETED_B_VARIANTS.json` with completed status, four settings, ten tasks,
  at least sixteen states per task, and `old_init_reused=false`;
* `SHA256SUMS` passing from the bundle root;
* all four variant `REGENERATED_INIT_STATES.json` manifests and ten task
  BDDL/state pairs passing `validate_b_calibration_variants`;
* every consumed state declaring the real flattened simulator-state format
  `torch.save(sim.get_state().flatten())`, with manifest hashes and state width
  checks; and
* pinned clean source trees:
  `Stage-S=/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/r142-stage-s-bcal-runtime-20260903`,
  `Stage-S=afe353bbc5997355f35cb0c77c5446fd4df5f1e3`,
  `QPILOTS=eacf47b981e3b22357f8a74902f8dad8cfcfa375`,
  `OpenPI=54cbaee6ae0c010a1ed431871cdaa8f4684ac709`, and
  `LIBERO=f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`.

The r7 directory was not yet complete when this contract was authored. A
missing completion marker, a stale SHA, a missing state, or any source drift
fails closed; it is not a scientific result.

## Resource and controller contract

The manifest selects the registered robot idle alias
`idle-a800-robot-stage-s-graphics-8gpu` (`quota1ssrabud0bh`, `exp-robot`) with one
worker, exactly 8 A800 GPUs, 88 CPU cores, 1525 GiB memory, and 1525 GiB
shared memory. The manifest has no secrets, uses the controller's numeric
`2254:2254` identity, and declares only `{{ARTIFACT_DIR}}` as a write path.
Both `contract_source_job_id` and `resource_source_job_id` are pinned to the
Stage-S carrier readback `dlckjz66iwcv38gw`; they are provenance only and do
not represent a PAI clone.
The artifact directory is exactly

```text
/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/pai_registry/
  r142_stage_s/b_calibration/{{RUN_ID}}
```

The registry manifest binds the required public
`NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics` environment for the
registered graphics alias. The launcher additionally sets `MUJOCO_GL=egl`,
`PYOPENGL_PLATFORM=egl`, and `EGL_PLATFORM=device` inside the workload.

The launcher records `FIRST_WORK.json`, runs the eight ranks with
`torchrun --standalone --nnodes=1 --nproc_per_node=8`, and calls the existing
real `stage_s_libero_calibrate.py` shard/aggregate modes. It never calls a
submit API. PAI's Sync/OnFailure max-50 policy and application-idempotent
markers allow a preempted incarnation to resume the same artifact directory
and same run ID. Existing complete rank markers are verified and skipped by
the calibration runtime.

After all eight rank markers have passed their own `SHA256SUMS`, the aggregate
is verified and the launcher writes:

```text
CALIBRATION_RESULT.json
COMPLETED_CALIBRATION.json
COMPLETED_B_CALIBRATION.json
SHA256SUMS
B_SHA256SUMS
```

`COMPLETED_B_CALIBRATION.json` lists all eight rank markers, the aggregate
digest, and source/provenance/compute metadata. `B_SHA256SUMS` covers the
aggregate, all rank results/markers, `FIRST_WORK.json`, and the B completion
marker. There is no completion claim without all eight ranks and both SHA
checks.

## Daily no-job windows

The launcher refuses to begin/resume during `09:30-09:40` and
`19:30-19:40` Beijing time (`Asia/Shanghai`). The external PAI controller must
stop this exact job before `09:30` and `19:30`, then resume the same run after
`09:40` and `19:40`; it must not create a new run ID or output directory.

## Static hand-off checks

The following checks do not submit PAI:

```bash
PYTHONPATH=src python -m pytest -q tests/test_stage_s_b_calibration_pai.py
bash -n scripts/stage_s_libero_b_calibration_pai.sh
```

The registry shape was checked with `pai-job validate` using the exact
robot-idle resource contract. Before an authorized submission, this launcher
must be installed at the external payload path named by the manifest:

```text
/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/r142-stage-s-pai-20260902/
  stage_s_libero_b_calibration_pai.sh
```

The installed payload must match SHA256
`a9c1db905e1dbc9c7c732fbe62b279130142233d1b4873fdc1e7cc965b429d49`, and its
independent runtime source must be the B-only tree
`/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/r142-stage-s-bcal-runtime-20260903`
at the pinned commit above. No PAI job was submitted by this change.
No PAI job was submitted by this change.
