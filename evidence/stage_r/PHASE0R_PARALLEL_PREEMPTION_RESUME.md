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
