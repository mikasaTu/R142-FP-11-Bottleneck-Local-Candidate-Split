# Stage-S B/C main screen runtime contract

Status: **prepared, validated statically, not submitted**.  This document is
the operational contract for the independent B and C main screens.  It does
not contain scientific results and does not authorize a PAI submission.

## Frozen screen

Both variants run the existing `scripts/stage_s_libero_main.py` and the
Stage-R pinned `StageRPolicyAdapter` against the real LIBERO environment:

* ten frozen `libero_10` tasks;
* initial states `0..15`;
* 32 independently seeded candidates per task/state family;
* eight foreground ranks, with the flat `(task,state)` list assigned by
  `ordinal % 8`, hence 20 families per rank and 160 families total;
* each policy candidate runs until the official environment returns a terminal
  result or the frozen 520-step limit; the official success label is retained;
* no scripted expert, synthetic callback, S2--S5 statistic, or post-hoc
  candidate selection is allowed.

Every candidate family atomically persists `rollouts.npz`, `genealogy.json`,
`snapshots.pkl`, `metadata.json`, `SHA256SUMS`, and the final
`COMPLETED_FAMILY.json`.  The genealogy contains candidate id, parent id,
generation step, seed, complete action prefix, and terminal success.  NPZ
arrays contain action/pose trajectories and per-candidate policy-forward and
environment-step accounting.

The replay gate is mandatory for B/C: restore simulator state, observation
history, action queue, Python RNG, NumPy RNG, Torch CPU/CUDA RNG, and policy RNG;
execute the same action and require the next-state maximum absolute error to be
at most `1e-9`.  A missing concrete capture/restore hook or incomplete RNG
stream fails closed before any family is committed.

## Calibration gates

The main screen cannot start from a calibration result, running job, partial
shard, or arbitrary caller path.

* B requires the frozen
  `/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_s/b_calibration/CALIBRATION_REPORT.json`.
  It must prove pooled-only calibration completion, be `FROZEN`, select the
  exact r7 variant run `r142-stage-s-b-variants-20260903-r7`, and identify one
  of the four settings.  The launcher passes only that selected variant root;
  all other variant roots are rejected.
* C requires the analogous frozen
  `/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_s/c_calibration/CALIBRATION_REPORT.json`.
  It must select one complete, SHA-verified real checkpoint after calibration
  completion.  The caller cannot override it.  Every C family and the top
  report carry the literal `WEAK_SUBSTRATE` annotation.

The report source result, completion marker, and SHA256SUMS are independently
verified.  Freeze reports reject S2, S3, S4, S5, trajectory, genealogy, and
divergence fields to prevent pre-screen lookahead.

## Frozen protocol acceptance gate

Before either B or C rank enters the main rollout, the launcher reads the
stable CPFS artifact
`/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_s/stage_s/protocol/FROZEN_PROTOCOL.json`.
The adjacent `PROTOCOL.md` is required.  The JSON must have schema
`r142-stage-s-protocol-acceptance-v1`, top-level `status=FROZEN`, and an
`acceptance` object with `status=ACCEPTED`, `frozen=true`, the full 40-hex
`protocol_git_commit`, and the SHA-256 of the adjacent markdown.  The commit
must also be recorded in that markdown; a missing, malformed, or mismatched
commit/hash fails closed.

The acceptance object freezes the exact ten task IDs, initial states `0..15`,
candidate IDs `0..31`, eight-rank budget, SHA-256 seed namespace/formulas,
S1--S5 thresholds, eventual-termination label, and policy-forward/env-step
accounting.  It also contains both B and C calibration entries (report path,
report SHA, and selected setting/checkpoint).  The active launcher verifies the
selected entry against the actual calibration report and its SHA before any
rollout.  The acceptance path, acceptance SHA, protocol Git commit, and
`PROTOCOL.md` SHA are copied into every family metadata/completion marker,
rank summary/marker, and final `COMPLETED_EVALUATION_RESULT.json`.

## Runtime and evidence

The independent runtime source is
`/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/r142-stage-s-bc-main-runtime-20260903`
at the exact Stage-S source commit bound in each registry config.  The pinned dependencies are:

* QPILOTS `eacf47b981e3b22357f8a74902f8dad8cfcfa375`;
* OpenPI `54cbaee6ae0c010a1ed431871cdaa8f4684ac709`;
* LIBERO `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`.

The external payloads and companion configs are deployed under
`/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/r142-stage-s-bc-pai-20260903/` and
are hash-bound in the two registry configs.  Both configs use the exact robot
idle carrier: one worker, 8 A800, 88 CPU, 1525 GiB memory/shared memory,
`quota1ssrabud0bh`, `AcceptQuotaOverSold`, and the only public pod environment
`NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`.  Runtime identity is
2254:2254; `HOME` is inherited and never rewritten.

The application resumes in the same run directory.  Complete family markers
are immutable and skipped; incomplete families rerun atomically.  Each rank
publishes `COMPLETED_{B,C}_MAIN_RANK-0000..0007.json` only after its 20 family
paths and summaries are complete.  The finalizer verifies all eight markers,
all 160 families, all hashes and snapshots, then writes
`COMPLETED_EVALUATION_RESULT.json` and the top-level `SHA256SUMS`.  No top
marker is written for a partial screen.

At 09:30--09:40 and 19:30--19:40 Asia/Shanghai the payload refuses to start
or resume and records `REFUSED_DAILY_NO_JOB_WINDOW.json`.  Operations must stop
the exact PAI job before 09:30/19:30 and resume it only after 09:40/19:40.
Failure evidence is preserved as `FAILED_{B,C}_MAIN.json` and archived on a
later same-directory incarnation.

## Validation performed

The branch was checked with targeted pytest, Python compilation, `bash -n`,
JSON parsing, and `git diff --check`.  No PAI job was submitted and no B/C
scientific result is claimed until both calibration reports and terminal
evaluation evidence exist.
