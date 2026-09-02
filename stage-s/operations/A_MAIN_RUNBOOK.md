# Stage-S substrate A: RoboTwin/Evo PAI runbook

This runbook is the frozen execution contract for substrate A. It is an
evaluation, not a calibration or training job. The scientific workload is
exactly the ten lexically selected RoboTwin tasks, 16 initial-state families
per task, and 32 independently seeded candidates per family: 160 families and
5,120 terminal episodes.

## Pinned source and model

The formal job must use the following CPFS inputs and must fail closed on any
revision mismatch:

| input | path / revision |
|---|---|
| Stage-S runtime | `/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/r142-stage-s-a-runtime-20260903` @ `047f15c5cbad99e76ac47e2ce608a40bae22c49e` |
| RoboTwin | `cache/r142_stage_s/runtime/RoboTwin` @ `13c3c47ff4312dd62484bcd51be034af55c062d1` |
| CuRobo | `cache/r142_stage_s/runtime/RoboTwin/envs/curobo` @ `d64c4b005459db10c5dd867d8b30a87d5bda9bdb` |
| Evo-1 | `code/r142_stage_s_deps/Evo-1` @ `5fd14b015013c4fd0aacf5f8f48f868ca9b870a2` |
| checkpoint | `cache/r142_stage_s/models/Evo1_RoboTwin2_clean_ce8c583724706fbf7a03c17237761c65bf6813a7`, HF commit `ce8c583724706fbf7a03c17237761c65bf6813a7` |

The checkpoint `SHA256SUMS` is checked before any server starts. The public
Evo server source remains unmodified. Each server imports the released
`load_model_and_normalizer` and `infer_from_json_dict`, while the external
Stage-S dispatcher handles only the three versioned exact-replay controls.

The formal screen also has a hard asset gate. Before any server or client is
started, the launcher must read the stable CPFS acceptance pointer
`/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_s/stage_s/protocol/ACCEPTED_A_ASSET_PREFLIGHT.json`.
The controller may create this file atomically only after one asset-preflight
PAI Job reaches terminal `Succeeded` and its exact output directory contains
`COMPLETED_ASSET_PREFLIGHT.json` plus a verified `SHA256SUMS`. The launcher
then dynamically rechecks the accepted run id and JobId, marker/manifest byte
hashes, exact Evo/RoboTwin/CuRobo source commits, exact checkpoint revision,
and the live model `SHA256SUMS` under the runtime checkpoint directory. A
`FIRST_WORK.json`, an active/queued/`Running` PAI job, a partial cache, or a
hardcoded historical run directory does not satisfy this gate. The accepted
record and all rechecked fields are persisted in every rank manifest,
completion marker, and aggregate result.

The A main launcher also reads the stable CPFS authority
`/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_s/stage_s/protocol/FROZEN_PROTOCOL.json`
before starting any Evo server. It must report `status=FROZEN`, a full
40-hex protocol Git commit, matching hashes for `PROTOCOL.md` and both B/C
calibration reports, and the frozen Stage-S threshold, seed, task, and budget
summary. The launcher and each rank client fail closed on any missing,
symlinked, malformed, or hash-mismatched authority. The rank run manifest and
`COMPLETED_A_RANK-<rank>.json` persist the complete protocol and accepted asset
fingerprints;
the aggregate result repeats it and re-verifies the stable authority.

## Eight-GPU ownership

The PAI worker is one node with exactly 8 A800 GPUs, 88 CPU cores, 1525 GiB
memory, and 1525 GiB shared memory on the explicitly requested robot idle
resource (`ResourceId=quota1ssrabud0bh`, `quota=exp-robot`,
`OversoldType=AcceptQuotaOverSold`). The launcher creates exactly eight pairs:

```text
rank r: CUDA_VISIBLE_DEVICES=r
        server: 127.0.0.1:19000+r
        client:  stage_s_robotwin_main.py --rank r --world-size 8
```

One server process and one client process own each GPU and port. No server is
shared across ranks, so a Torch/CUDA RNG stream cannot be interleaved by two
clients. The dispatcher uses an `asyncio.Lock` inside each server as a second
line of protection. A non-eight-GPU allocation or a client/server ownership
mismatch exits before a rollout.

The client shard is `flat_task_family_index % 8 == rank`, so every rank has 20
families. The aggregate verifier rejects missing or extra families and requires
all eight rank markers.

## Replay and completion contract

Before candidate 0 in every family, the concrete RoboTwin adapter captures and
restores simulator state, policy observation history, queued actions, and
environment/policy Python, NumPy, Torch CPU, and CUDA RNG state. It then runs
the same action twice and requires action and next-state errors no greater than
`1e-9`. The released official `eval_success`/`check_success` path is the only
success label; no expert trajectory, solvability oracle, synthetic policy, or
synthetic success callback is accepted.

Each family is immutable after `COMPLETED_FAMILY.json` and its `SHA256SUMS`
verify. A resumed incarnation skips only a valid completed family; a partial
or hash-invalid family is a capability failure at the same path. Each rank
finishes with `COMPLETED_A_RANK-<rank>.json` plus its rank checksum. The
aggregate command then writes `COMPLETED_EVALUATION_RESULT.json` and a
recursive top-level `SHA256SUMS`. Queueing, `Running`, `FIRST_WORK`, a partial
shard, or a server-ready marker is never completion evidence.

## Scheduler and preemption rules

The launcher refuses to start or resume during both Beijing windows:

```text
09:30 <= Asia/Shanghai < 09:40
19:30 <= Asia/Shanghai < 19:40
```

The controller must stop the exact active job before 09:30 and 19:30 and may
resume it only after 09:40 and 19:40 with the same run directory and a unique
replacement PAI run ID. The idle fault policy is `Sync OnFailure`, 50 platform
restarts, and one foreground launcher attempt per platform incarnation. The
launcher itself has no retry loop around the workload. Failure records are
preserved under `failures/` when a later incarnation resumes the directory.

## Controller handoff

Copy the three runtime files to the independent A runtime checkout and the
launcher to the controller payload directory, then freeze their bytes and
hashes:

```text
scripts/stage_s_robotwin_a_pai.sh
scripts/stage_s_robotwin_evo_server.py
scripts/stage_s_robotwin_finalize.py
configs/pai/stage_s_robotwin_a.json
```

The launcher hard-codes the independent runtime path and rejects every source
revision except the source commit recorded in the config. It passes both the
stable protocol authority and stable accepted-asset authority to every rank
and to the aggregate verifier. Do not submit while
`ACCEPTED_A_ASSET_PREFLIGHT.json` is missing or while its accepted run/job is
not backed by terminal PAI `Succeeded`, `COMPLETED_ASSET_PREFLIGHT.json`, a
verified asset `SHA256SUMS`, and a live checkpoint `SHA256SUMS` matching the
declared source/model pins. In the current CPFS readback, r15 is stopped with
only `FIRST_WORK.json`, r16 has no Job and is sealed, and r17 JobId
`dlc17mybd6alknp3` is only `Running`; none can be used to create the accepted
pointer. Do not submit a PAI probe: static and bounded dev14 contract tests
are sufficient for this launcher; the next PAI operation is the formal A
screen after the controller writes the stable acceptance record.

After a formal attempt, preserve the exact JobId, raw failure/log evidence,
and output directory. A replacement may reuse the same output directory only
when it is the same frozen task and the family markers pass verification. Do
not delete successful or active PAI service records as part of recovery.
