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

At the 2026-08-25T23:09Z checkpoint, the resumed shard B completed its first
authoritative target, `libero_10_task06`. The full outcome-blind audit verified
protocol/task identity, 512 rollouts, exact 16-by-32 coverage, rollout seeds,
offsets, finite arrays, and owner `2254:2254`:

```text
NPZ SHA256      8448a6abcb09c1b83895c720b2801b9e0125e56f091760da87146c06a47aaee8
metadata SHA256 41b63046ce941e536721b4dc48b414771cd78cacd78df2226affa14e93e79643
```

This supplies persisted first-target evidence after the rollback, but shard B
is only 1/4 and has no subset completion marker or SHA256SUMS.

## Shard A terminal completion

At the 2026-08-25T23:41Z checkpoint, exact GetJob reported terminal
`Succeeded` for shard A. All four authoritative targets passed the full
outcome-blind schema, seed, coverage, owner, and SHA audit. The missing final
target was `libero_10_task02`:

```text
NPZ SHA256      888401afd0c4597e9f665358d31dabff3aa10a833417ecf107b9c959ec7a6ff8
metadata SHA256 072021805fa0ba6ec965599ff6f5caa0378363c00e0b12cc07b409e9ca805fc5
```

The four exact rank markers matched the frozen global-rank assignments, all 16
imported prerequisite records were present, and the subset manifests stated
4 targets, 2,048 rollouts, no outcome unblinding, no global Phase-0R
completion, and no Phase-1 authorization. The immutable completion hashes are:

```text
COMPLETED_SUBSET_RAW.json         416d655d3520c04019fb7ea285db23c91f31d783c8e441e6facd7b4b13ada03c
COMPLETED_EVALUATION_RESULT.json  c65b70daf07634f8c54b5ab92a41c8480fb41f68abed14c6253cc777de562410
SHA256SUMS                        e127d911b436a010fe8fa70671d64f8fb68049369f7e0a0dd93ef3e3ed2b429b
```

`sha256sum --check` passed and every immutable file was owned by
`2254:2254`. This completes shard A only. Global merge and analysis remain
blocked until shard B independently reaches the same gates and terminal
`Succeeded`.

## Shard B progress after same-directory recovery

At the 2026-08-26T00:45Z checkpoint, exact GetJob still reported `Running`
for shard B under the same JobId, run ID, and CPFS artifact directory. Two
additional authoritative targets, `libero_10_task07` and
`libero_10_task09`, passed the full outcome-blind protocol/task identity,
512-rollout, exact 16-by-32 state/candidate coverage, rollout-seed, offsets,
finite-array, owner `2254:2254`, and metadata/data SHA audit:

```text
libero_10_task07 NPZ SHA256      d61278b60f15a617fb29c6fdfae91695dbe6b2fa64ea3bcc24822e6e03f4500a
libero_10_task07 metadata SHA256 8a55a7147ab1561b558d54be4230b99d0fc36212891cc018a2aeb307a0fcb9f1
libero_10_task09 NPZ SHA256      725d8bca12eeeb7e54fe0df75c536a6b8b6081c08a180c9228a43596983dd8e8
libero_10_task09 metadata SHA256 937f404a15e75ad6d90b137cd4e73613148ef04d3fa3715f7ca77d6a63919841
```

Together with the previously validated `libero_10_task06`, shard B is now
3/4 authoritative targets. Exact rank markers exist for global ranks 4, 5,
and 7; global rank 6 is still running `libero_10_task08`. There is no shard-B
`COMPLETED_SUBSET_RAW.json`, `COMPLETED_EVALUATION_RESULT.json`, or
`SHA256SUMS`, and PAI has not reached terminal `Succeeded`. This checkpoint is
persisted progress only, not subset or global Phase-0R completion.

### Shard B second preemption and same-Job recovery

At the 2026-08-26T06:22Z checkpoint, exact GetJob showed that shard-B master
UID `1a6121bc-fe5d-4961-bf80-489f24da15dc` failed after running from
`2026-08-25T19:08:09Z` to `2026-08-26T06:15:15Z`. AIMaster retained the exact
Job and created replacement master UID `c1508148-e71f-4d63-b59c-378b21d40262`,
which was `Running` from `2026-08-26T06:16:38Z`.

The shard directory retained inode `1183268080862`, owner `2254:2254`, and
mode `0700`. The three authoritative targets `libero_10_task06`, `task07`,
and `task09` retained their previously sealed SHA values and owner; no
`task08` pair, rank-6 marker, subset completion record, or root SHA256SUMS had
appeared. No replacement Job was submitted and the frozen target, seed,
candidate-budget, threshold, statistical, and authority contracts were not
changed. This is same-directory recovery progress only; shard B remains 3/4
and global analysis stays sealed.

At the 2026-08-26T07:58Z checkpoint, shard-B replacement master UID
`c1508148-e71f-4d63-b59c-378b21d40262` had failed after running from
`2026-08-26T06:16:38Z` to `2026-08-26T07:31:35Z`. AIMaster created master
UID `43be3a75-1b1d-4c37-84dd-a03d7be39eca`, which exact GetJob reported
`Running` from `2026-08-26T07:32:58Z`. The exact JobId, run ID, and artifact
directory inode `1183268080862` were unchanged; no fatal marker was detected
in the recreated rank logs. The three sealed target pairs remained intact,
while `task08`, rank 6, subset markers, and terminal `Succeeded` were still
missing. This additional same-directory restart is not completion and does
not authorize a source substitution or global analysis.
