# Phase-0R parallel acceleration protocol

This operational protocol was frozen before either supplemental shard produced
an outcome. It changes scheduling only and does not change the Stage-R
scientific contract.

## Frozen scientific contract

- scientific source commit:
  `24423e8114ace80e6a76f22bee29992cea420cfc`
- protocol: `r142-stage-r-phase0r-v1`
- model/checkpoint, 40-task order, 16 initial states per task, 32 candidates
  per state, rollout seeds, action horizon, microbatch 4, thresholds, metrics,
  and Checkpoint-1 stop: unchanged
- Phase-1 remains unauthorized

## Parent evidence boundary

Parent run `r142-stage-r-phase0r-20260824-r4-idle4`, JobId
`dlcyuv28a0djtgxd`, produced SHA-valid tasks at indices 0--31 before this
parallel protocol was introduced. Those 32 task pairs are frozen as the
authoritative source for indices 0--31.

No supplemental job may write the parent's artifact directory. Each writes a
separate staging directory and imports the frozen 0--31 prerequisites by
hardlink only after protocol, suite/task identity, rollout count, candidate and
initial-state structure, numeric ownership, and SHA validation.

## Pre-registered authoritative split

- shard A is authoritative for indices 32--35:
  `libero_10_task02` through `libero_10_task05`
- shard B is authoritative for indices 36--39:
  `libero_10_task06` through `libero_10_task09`

The two shards are disjoint and use global `world_size=8` assignments while
each 4-GPU job executes four global ranks. The imported prerequisites make the
unchanged frozen runner skip indices 0--31, so every supplemental GPU computes
exactly one target task.

Parent-run outputs for indices 32--39 are redundancy only. They will not be
selected based on success labels, metrics, completion time, or agreement with
the hypothesis. If a supplemental shard is preempted or fails operationally,
the same shard and staging directory must resume or be repaired without
changing its task set. Parent redundancy is not an outcome-dependent fallback.

## Merge and completion gate

Global merge is permitted only after both supplemental shards satisfy all of
the following:

1. their exact PAI jobs reach terminal `Succeeded`;
2. all four target NPZ/metadata pairs per shard pass full SHA and structure
   validation;
3. `COMPLETED_SUBSET_RAW.json`, `COMPLETED_EVALUATION_RESULT.json`, and root
   `SHA256SUMS` exist and validate;
4. numeric ownership is `2254:2254`;
5. the exact idle placement and restart lineage are sealed.

The merged authoritative set is fixed as parent indices 0--31 plus shard A
indices 32--35 plus shard B indices 36--39. Only then may the frozen Phase-0R
aggregate and analysis run. The workflow must stop at `CHECKPOINT_1`; it must
not enter Phase-1.
