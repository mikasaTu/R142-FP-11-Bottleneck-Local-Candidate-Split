# Phase-0R parallel supplemental submission evidence

Recorded on 2026-08-25. This is submission and first-work evidence, not a
Phase-0R completion claim.

## Frozen operational split

The scientific source remains commit
`24423e8114ace80e6a76f22bee29992cea420cfc`. The supplemental jobs write
isolated staging directories and do not write the parent raw directory.
Authority was registered before outcomes were available:

- parent run `r142-stage-r-phase0r-20260824-r4-idle4`, JobId
  `dlcyuv28a0djtgxd`: indices 0--31;
- shard A: indices 32--35 (`libero_10_task02`--`task05`);
- shard B: indices 36--39 (`libero_10_task06`--`task09`).

The parent outputs at indices 32--39 are redundancy only and cannot be chosen
based on their result. Phase-1 remains unauthorized.

## Validation before submission

- local and dev14 shard-contract pytest: passed;
- launcher `bash -n`: passed;
- payload SHA256:
  `35bfcf683f7c70225fce23bbef1ca2745eb0b691b66d1c39af85bf90a3ecd694`;
- shard-contract SHA256:
  `c9ab70c12fd226381f622d279dd6e5a884f03ada7d88b9446ab62eee3a168da0`;
- 32 parent task pairs were validated live for protocol, suite/task identity,
  512 rollouts, exact 16-by-32 coverage, candidate IDs, rollout-seed formula,
  offsets, finite trajectory arrays, ownership, and NPZ metadata SHA;
- both r2 templates passed the canonical controller's `pai-job validate`.

## Submission attempts and exact jobs

The first shard-A r1 attempt was refused before CreateJob because the new
`phase0r_parallel` parent directory did not exist. Its registry state is
`preflight_failed_sealed`; no PAI JobId exists. The parent directory was then
created as `2254:2254`, mode `0700`, and fresh r2 run IDs were validated.

| Shard | Run ID | JobId | Requested resource | Initial state |
|---|---|---|---|---|
| A | `r142-stage-r-phase0r-supp-a-20260825-idle4-r2` | `dlc9b8jreh4e8fy3` | 1 worker, 4 A800, 46 CPU, 800 GiB memory/shared-memory | submitted and exact readback verified |
| B | `r142-stage-r-phase0r-supp-b-20260825-idle4-r2` | `dlcap6ioa9w2ht2u` | 1 worker, 4 A800, 46 CPU, 800 GiB memory/shared-memory | submitted and exact readback verified |

Both use quota `quotaewyznuc7b9l`, `AcceptQuotaOverSold`, priority 9, empty
reservation, `DisableEcsStockCheck=true`, and AIMaster Sync/OnFailure with at
most 50 platform restarts.

## Actual idle placement

Exact-JobId PAI OpenAPI/ListJobs evidence sealed
`UseOversoldResource=true` for the parent and both supplemental jobs:

- parent placement evidence SHA256:
  `3c1171e275558060d8062ffbf71f0f6c40679604662a7dff837d941e937229b5`;
- shard A placement evidence SHA256:
  `ba89a568e485b4072eca2d4792698ea50fbe7ec94b1dbb474c705629d79e3b71`;
- shard B placement evidence SHA256:
  `6b487fb66cda2f34d99cdc125f0f25d37236695a5f30c97c9b50233b09b0db54`.

`AcceptQuotaOverSold` alone was not counted as placement evidence.

## Controller snapshot used

The canonical registry worktree contained unrelated existing modifications, so
the exact runtime files used for this submission were hashed:

- `bin/pai-job`:
  `6c110f413f177d47c39bfae4a08954c75003f8c8b6f37472f760325776b16df3`;
- `config/resources.json`:
  `bd420cbb1c775750926e85c09877950a395534034ef8f877b5f0cf1a465e1fb5`;
- `config/toolchain.json`:
  `d29b212125d733aa6bb682b4200e5e518f9135bd77b242fde8312b8226678a03`;
- pinned DLC binary:
  `09fac825e088dfeee7f55919d6ad8421d4f46c2a6554da1827b664a08518473c`.

## Evidence boundary

At this snapshot both PAI jobs had reached `Running`; shard B had started all
four global ranks and shard A had completed its validated prerequisite import.
These are milestones only. No target task, subset completion, global 40-task
completion, analysis, or Checkpoint-1 result is claimed here. Completion still
requires target NPZ/metadata SHA validation, subset markers and SHA256SUMS,
terminal PAI `Succeeded`, a fail-closed global merge, frozen analysis, and the
explicit Checkpoint-1 stop.

## User resource-ceiling audit (2026-08-26)

The user required stopping and resubmitting any current Phase-0R PAI task
whose single-job specification exceeded either 88 CPU cores or 1.49 T of
memory. Fresh exact-JobId GetJob readback reported the same shape for the
parent, shard A, and shard B:

```text
PodCount=1, GPU=4, CPU=46, Memory=800Gi, SharedMemory=800Gi
```

Thus every exact job is below both ceilings (`46 < 88`; `800Gi < 1.49T`,
whether T is interpreted as decimal TB or TiB). No Job was stopped or
resubmitted, and the frozen task, seed, candidate-budget, threshold,
statistics, authority, and artifact-directory contracts remain unchanged.

## Concurrent GPU ceiling (2026-08-26)

After network recovery, the user set a hard ceiling of 16 concurrently used
GPUs for the remaining plan. Exact GetJob showed the only active Phase-0R
workers were the 4-GPU parent and 4-GPU shard B; shard A was terminal
`Succeeded`. Effective active use was therefore 8 GPUs, below the 16-GPU
ceiling. The recurring monitor was updated to enforce this cap for any later
recovery or supplemental submission. No new job was created and the frozen
authority map was not changed.
