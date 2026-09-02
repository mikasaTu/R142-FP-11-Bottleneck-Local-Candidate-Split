# Stage-S substrate-A execution ledger

This is an operations ledger, not scientific evidence. `Queuing`, `Running`,
`FIRST_WORK`, server readiness, a partial family shard, or a local contract
test never count as a RoboTwin result.

| Attempt | JobId | State | Evidence / disposition |
|---|---|---|---|
| static-20260902-r1 | not created | validated, no PAI submission | Canonical clean `pai-job validate` accepted the updated `configs/pai/stage_s_robotwin_a.json` for run id `r142-stage-s-a-main-20260902-r1`; exact resource `quota1ssrabud0bh`, `AcceptQuotaOverSold`, 1 worker, 8 GPU, 88 CPU, 1525 GiB memory/shared-memory, three new-root mounts, and Sync OnFailure max-50 were resolved. Independent runtime pin: `c2bd51db6de0e22d09827d06460cbac8d47bb6ae`; payload hash: `4e37de86b0e9e5eb7bb37990cb29bcbad93db2e9483f330d5a73ac4155f2e179`. |
| dev14-contract-20260902 | not created | passed, no PAI submission | `bash -n`, Python compile, JSON parse, and the Stage-S directed suite passed (`41 passed`, including the existing RoboTwin replay tests). No mock rollout was treated as evidence. |
| asset-acceptance-20260903 | not eligible | blocked before submission | Stable CPFS `stage_s/protocol/ACCEPTED_A_ASSET_PREFLIGHT.json` is absent. Current readback: r15 is stopped with only `FIRST_WORK.json`, r16 has no Job and is sealed, and r17 JobId `dlc17mybd6alknp3` is only `Running`; no terminal `Succeeded` completion/SHA evidence is accepted, so the controller must not create the acceptance pointer. |
| runtime-independent-20260903 | not created | prepared, no PAI submission | A runtime is frozen to `/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/r142-stage-s-a-runtime-20260903` @ `ea06996168bd460c98cb3e2ee77b929aafced12e`; external payload path remains `/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/r142-stage-s-pai-20260902/stage_s_robotwin_a_pai.sh`. Updated payload hash: `4e37de86b0e9e5eb7bb37990cb29bcbad93db2e9483f330d5a73ac4155f2e179`. |
| protocol-authority-20260903 | not eligible | prepared, no PAI submission | A main now requires stable CPFS `stage_s/protocol/FROZEN_PROTOCOL.json` and stable `ACCEPTED_A_ASSET_PREFLIGHT.json` before any server/client; rank metadata, rank completion, and aggregate completion carry both complete fingerprints. Current authorities are absent, so this is intentionally fail-closed. |

## Submission gate

The formal A job is intentionally not submitted by this branch worker. Before
the parent submits a unique run id, the controller must atomically create
`ACCEPTED_A_ASSET_PREFLIGHT.json` only after one asset preflight reaches
terminal PAI `Succeeded` and its exact output directory contains
`COMPLETED_ASSET_PREFLIGHT.json` plus a verified `SHA256SUMS`. The current
readback has no accepted pointer, so this gate is presently unmet. The parent
must copy the runtime files into the independent A runtime checkout,
verify the exact frozen commit, update the run-scoped
`explicit_user_resource_authorization.scope` and `validated_payload_sha256`
binding if the launcher bytes change, then run the canonical validator again.

The stable protocol authority is an additional hard gate. It is not satisfied
by a local copy, a GitHub plan, a `FIRST_WORK.json`, or a partial B/C report;
the exact CPFS JSON and all three referenced file hashes must verify before
the first server starts. The accepted asset authority is likewise checked
against the live run/job, source commits, checkpoint revision, and asset/model
hashes on every rank and during finalization.

The launcher uses `/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_s/a_main/<RUN_ID>`
as the same-directory resume root. It starts eight independent Evo server
processes and eight matching clients. A completed run requires every rank and
family marker, 160 family directories, 5,120 terminal candidates, and the
aggregate `COMPLETED_EVALUATION_RESULT.json`/`SHA256SUMS` pair. The runtime
server and launcher must remain pinned to the source/model revisions listed in
`A_MAIN_RUNBOOK.md`.

## Failure and blackout handling

Any pre-step capability failure writes `FAILED_A_MAIN.json`; a later
same-directory incarnation archives that marker under `failures/` before
resuming. Hash-invalid family data is never silently repaired. A job that
starts or restarts in 09:30--09:40 or 19:30--19:40 Beijing time exits with
`REFUSED_DAILY_NO_JOB_WINDOW`; the controller must stop before each window and
resume after it with a unique PAI run id while retaining the same output root.
