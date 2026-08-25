# Phase-0R parallel supplemental preemption and resume evidence

This log records operational failures and recoveries for the pre-registered
parallel supplements. It does not change the scientific contract or authorize
outcome-dependent artifact selection.

## 2026-08-25 shard B rollback to queue

- exact JobId: `dlcap6ioa9w2ht2u`
- run ID: `r142-stage-r-phase0r-supp-b-20260825-idle4-r2`
- authoritative task set: indices 36--39,
  `libero_10_task06`--`libero_10_task09`
- PAI state transition observed: `Running` -> `Restarting` -> `Queuing`
- latest exact GetJob reason: `JobEnqueued`, `Rollback to queue`
- AIMaster contract: Sync, OnFailure, at most 50 platform restarts
- actual idle placement had already been sealed with
  `UseOversoldResource=true`

The job had produced no target NPZ/metadata pair, subset completion marker, or
SHA256SUMS before rollback. Its 16 imported prerequisite pairs remained in the
same CPFS artifact directory. The directory identity at the rollback snapshot
was:

```text
path  /mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_r/phase0r_parallel/r142-stage-r-phase0r-supp-b-20260825-idle4-r2
inode 1183268080862
owner 2254:2254
mode  0700
```

No new PAI job was submitted, no target assignment changed, and no parent-run
redundant outcome was promoted. Recovery must occur under the same JobId,
run ID, artifact directory, frozen source, seed schedule, candidate budget,
thresholds, and statistics protocol. `Queuing`, `Restarting`, and later
`Running` remain recovery milestones rather than completion evidence.

### Same-Job recovery checkpoint

At the 2026-08-25T19:27Z checkpoint, exact GetJob returned `Running` for the
same JobId and run ID. The artifact directory still had inode
`1183268080862`, owner `2254:2254`, and the four global-rank logs had been
recreated with newer mtimes and no launcher fatal, traceback, assertion, or
CUDA-OOM marker. No shard-B target pair existed yet. This verifies rescheduling
and same-directory restart only; it is not first target work or completion.

At the same checkpoint, shard A produced its first authoritative target,
`libero_10_task05`. A full outcome-blind structure audit verified protocol and
task identity, 512 rollouts, exact 16-by-32 state/candidate coverage, rollout
seed formula, offsets, finite trajectory arrays, and owner `2254:2254`:

```text
NPZ SHA256      2e78dda37b3e0ec5ae1099e1ff1b80e25d63f7803a90fc764f70c5699a1094f1
metadata SHA256 96055a9879e726d0ae8a2d03a8e5452b867146893ad49807923ec48668e164e9
```

This is one of four shard-A targets. It is partial work, not a subset or
Phase-0R completion claim.

At the 2026-08-25T20:00Z checkpoint, shard A completed a second authoritative
target, `libero_10_task04`. The same full outcome-blind audit passed:

```text
NPZ SHA256      e105c61b7b360b45ff09c4f5adf6aa0a4b76c7ccb7f5cda0157c3b48becd5e97
metadata SHA256 f347560d2f5de3ec9a752e4b19763bdc2e9ede567f4f88792f9258f004e00072
```

Shard A is therefore 2/4 target pairs at this checkpoint. It still has no
subset completion marker or SHA256SUMS and remains `Running`.

At the 2026-08-25T21:03Z checkpoint, shard A completed and passed the same
full outcome-blind audit for `libero_10_task03`:

```text
NPZ SHA256      9a680aa8ef70bd560d73481bad320a833f05c769f2cf4fe3a1db317c595d99ca
metadata SHA256 e9b62b1bf0938a3da9c523afb31928ad9034e9e071cf775f34f6a00065fb187f
```

Shard A is now 3/4 target pairs. `Running`, a partial target set, and valid
individual hashes remain insufficient for subset completion.
