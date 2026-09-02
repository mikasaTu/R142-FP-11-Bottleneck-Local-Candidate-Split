# Stage-S substrate-A execution ledger

This is an operations ledger, not scientific evidence. `Queuing`, `Running`,
`FIRST_WORK`, server readiness, a partial family shard, or a local contract
test never count as a RoboTwin result.

| Attempt | JobId | State | Evidence / disposition |
|---|---|---|---|
| static-20260902-r1 | not created | validated, no PAI submission | Canonical clean `pai-job validate` accepted the updated `configs/pai/stage_s_robotwin_a.json` for run id `r142-stage-s-a-main-20260902-r1`; exact resource `quota1ssrabud0bh`, `AcceptQuotaOverSold`, 1 worker, 8 GPU, 88 CPU, 1525 GiB memory/shared-memory, three new-root mounts, and Sync OnFailure max-50 were resolved. Independent runtime pin: `c2bd51db6de0e22d09827d06460cbac8d47bb6ae`; payload hash: `4e37de86b0e9e5eb7bb37990cb29bcbad93db2e9483f330d5a73ac4155f2e179`. |
| dev14-contract-20260902 | not created | passed, no PAI submission | `bash -n`, Python compile, JSON parse, and the Stage-S directed suite passed (`41 passed`, including the existing RoboTwin replay tests). No mock rollout was treated as evidence. |
| r15 | `dlc11rl91mtxp2wq` | `Stopped`; asset preflight failed | Controller readback is terminal `Stopped` after the worker hit `flash-attn` pip build `Errno 18` (invalid cross-device link). Exact evidence: `/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_s/assets/r142-stage-s-a-assets-20260902-r15/pai_logs/master-final-before-stop.log` (SHA-256 `d5a9db6b3451a6ddcf1497eaf1973dbde6c405a0088f91a01fe05cf7e05dc4e6`), `FAILED_ASSET_PREFLIGHT.json` (SHA-256 `e0f20f3df46f791e6f625964b6c53c7b0869a3f1f8913c8ce7f7bf2c40c28beb`), and `FIRST_WORK.json` (SHA-256 `692b6b73fbeb0847fbf28d2cb70c751d7ede031bf6ff60e7c671ad82b8080107`). No `COMPLETED_ASSET_PREFLIGHT.json` or `SHA256SUMS` exists; this is not scientific evidence. |
| r16 | no JobId | controller `REFUSED` before CreateJob; sealed | The controller rejected the run because the required pre-created resume directory was missing. No PAI JobId was issued and no CPFS artifact was produced. The exact configured paths checked were `/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/pai_registry/r142_stage_s/assets/r142-stage-s-a-assets-20260902-r16/` and `/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_s/assets/r142-stage-s-a-assets-20260902-r16/`; both are absent, so no file SHA exists to report. |
| r17 | `dlc17mybd6alknp3` | `Stopped`; infrastructure preflight failed | Controller readback is terminal `Stopped`. The same log proves the cross-filesystem repair itself worked: `flash-attn` built and installed successfully, then the new setuptools environment failed at the infrastructure gate with `ModuleNotFoundError: pkg_resources`. Exact log `/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_s/assets/r142-stage-s-a-assets-20260902-r17/pai_logs/master-final-before-stop.log` SHA-256 `89c97f91389e845c2d5df9de47166a7bfc6bdb8d9d0716bd4dfdc09edae4f3d9`; `FAILED_ASSET_PREFLIGHT.json` SHA-256 `a49fffcd2c34221e2b46f2605edcafcb740932c97b7497ce07137a766b61e5af`; `FIRST_WORK.json` SHA-256 `0b55dbc7294bbf00eca356e6cad1e9df84a2c5edab62905d749c03253a99cfef`. The registry-side path `/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/pai_registry/r142_stage_s/assets/r142-stage-s-a-assets-20260902-r17/` has no completion marker or integrity manifest. This is an infrastructure gate failure, not a scientific result. |
| runtime-independent-20260903 | not created | prepared, no PAI submission | A runtime is frozen to `/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/r142-stage-s-a-runtime-20260903` @ `c2bd51db6de0e22d09827d06460cbac8d47bb6ae`; external payload path remains `/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/r142-stage-s-pai-20260902/stage_s_robotwin_a_pai.sh`. Updated payload hash: `4e37de86b0e9e5eb7bb37990cb29bcbad93db2e9483f330d5a73ac4155f2e179`. |

## Submission gate

The formal A job is intentionally not submitted by this branch worker. The
asset gate is presently unmet: r15 is terminal `Stopped` after the
cross-device `flash-attn` failure, r16 was refused before CreateJob and
sealed, and r17 is terminal `Stopped` after the `pkg_resources`
infrastructure failure. A valid gate requires a
terminal PAI `Succeeded` state plus `COMPLETED_ASSET_PREFLIGHT.json` and a
verified `SHA256SUMS` in the exact run directory. None of those incomplete
states is a scientific result or permission to start the formal A screen.
The parent must copy the three runtime files into the independent A runtime
checkout, verify the exact frozen commit, update the run-scoped
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
