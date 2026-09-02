# Stage-S substrate-A execution ledger

This is an operations ledger, not scientific evidence. `Queuing`, `Running`,
`FIRST_WORK`, server readiness, a partial family shard, or a local contract
test never count as a RoboTwin result.

| Attempt | JobId | State | Evidence / disposition |
|---|---|---|---|
| static-20260902-r1 | not created | validated, no PAI submission | Canonical clean `pai-job validate` accepted `configs/pai/stage_s_robotwin_a.json` for run id `r142-stage-s-a-main-20260902-r1`; exact resource `quota1ssrabud0bh`, `AcceptQuotaOverSold`, 1 worker, 8 GPU, 88 CPU, 1525 GiB memory/shared-memory, three new-root mounts, and Sync OnFailure max-50 were resolved. Payload hash: `78b8a4393c6867d2a5105d86b9cc45b5c66a18836f88d023a1c135351382730a`. |
| dev14-contract-20260902 | not created | passed, no PAI submission | `bash -n`, Python compile, JSON parse, and the Stage-S directed suite passed (`41 passed`, including the existing RoboTwin replay tests). No mock rollout was treated as evidence. |

## Submission gate

The formal A job is intentionally not submitted by this branch worker. Before
the parent submits a unique run id, the A asset preflight must be in terminal
PAI `Succeeded` state and its exact output directory must contain
`COMPLETED_ASSET_PREFLIGHT.json` plus a verified `SHA256SUMS`. The parent must
copy the three runtime files into the frozen runtime checkout, set
`STAGE_S_SOURCE_COMMIT` to that exact commit, update the run-scoped
`explicit_user_resource_authorization.scope` and `validated_payload_sha256`
binding if the launcher bytes change, then run the canonical validator again.

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
