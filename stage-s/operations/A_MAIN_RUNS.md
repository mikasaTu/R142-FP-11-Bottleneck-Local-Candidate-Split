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
| r18 | `dlc1nggsgakz69g7` | active; incomplete | The Evo-only `setuptools<81` compatibility repair is deployed. Current `Running`, `FIRST_WORK`, logs, or partial imports do not satisfy the gate; terminal PAI `Succeeded`, `COMPLETED_ASSET_PREFLIGHT.json`, verified `SHA256SUMS`, and a controller-created stable acceptance pointer are all still required. |
| asset-acceptance-20260903 | not eligible | blocked before completion evidence | A main requires stable CPFS `logs/r142_fp11_stage_s/stage_s/protocol/ACCEPTED_A_ASSET_PREFLIGHT.json`. It may be atomically created only after one exact asset Job reaches terminal `Succeeded` and its completion and integrity manifests verify; no earlier attempt is accepted. |
| runtime-independent-20260903 | not created | prepared, no PAI submission | A runtime is frozen to `/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/r142-stage-s-a-runtime-20260903` @ `01f964ff3cfa2b1d99eb8f76d06d9971e096977b`; external payload path remains `/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/r142-stage-s-pai-20260902/stage_s_robotwin_a_pai.sh`. Final authority-root payload SHA-256: `a768925b1e2536ae28acdbfa2bc7718bdf7aa729034ddf7a5e9cba32b997367b`. |
| protocol-authority-20260903 | not eligible | prepared, no PAI submission | A main now requires both stable CPFS `logs/r142_fp11_stage_s/stage_s/protocol/FROZEN_PROTOCOL.json` and `logs/r142_fp11_stage_s/stage_s/protocol/ACCEPTED_A_ASSET_PREFLIGHT.json` before any server/client; rank metadata, rank completion, and aggregate completion must carry both fingerprints. Current authorities are absent, so this is intentionally fail-closed. |
| accepted-lineage-tests-20260903 | not created | passed static checks, no PAI submission | The directed Stage-S suite passed (`58 passed`), including terminal-state, source/checkpoint, asset/model SHA, missing-pointer, tamper, metadata-lineage, replay, shell, and config tests. Canonical validation must be repeated against the clean controller registry after every final runtime/payload update; no PAI main task was created. |

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

The stable protocol authority is an additional hard gate. It is not satisfied
by a local copy, a GitHub plan, a `FIRST_WORK.json`, or a partial B/C report;
the exact CPFS JSON and all three referenced file hashes must verify before
the first server starts.

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
