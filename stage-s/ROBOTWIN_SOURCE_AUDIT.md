# Stage-S RoboTwin substrate A source audit

Audit date: 2026-09-02 (Asia/Shanghai). This is a source/capability audit,
not a Stage-S outcome and not a substitute for real RoboTwin rollouts.

## Frozen pins

| Component | Pin |
|---|---|
| checkpoint | MINT-SJTU/Evo1_RoboTwin2_clean @ ce8c583724706fbf7a03c17237761c65bf6813a7 |
| Evo-1 source | https://github.com/MINT-SJTU/Evo-1.git @ 5fd14b015013c4fd0aacf5f8f48f868ca9b870a2 |
| RoboTwin source | https://github.com/RoboTwin-Platform/RoboTwin.git, stable_2.0 @ 13c3c47ff4312dd62484bcd51be034af55c062d1 |
| CuRobo source | https://github.com/NVlabs/curobo.git, v0.7.8 @ d64c4b005459db10c5dd867d8b30a87d5bda9bdb |

The checkpoint revision is the Hugging Face model commit. Its model card
states that the policy is trained on RoboTwin 2.0, aloha-agilex, absolute
14-D joint control, with horizon=37, 50 inference timesteps, and Gaussian
kernel-9 smoothing.

## Pre-registered lexical task selection

The published clean success values below are copied from the pinned
[Evo-1 RoboTwin evaluation table](https://github.com/MINT-SJTU/Evo-1/blob/evo1-flash/RoboTwin_evaluation/README.md).
Eligibility is [0.25, 0.65]; exactly the first ten eligible task names in
lexical order are frozen before any Stage-S outcome is observed.

| Task | Published clean success |
|---|---:|
| blocks_ranking_size | 0.58 |
| pick_diverse_bottles | 0.49 |
| place_a2b_left | 0.48 |
| place_a2b_right | 0.38 |
| place_bread_basket | 0.63 |
| place_bread_skillet | 0.63 |
| place_can_basket | 0.50 |
| place_fan | 0.34 |
| place_object_scale | 0.49 |
| place_shoe | 0.33 |

The next five eligible names (put_object_cabinet, rotate_qrcode, scan_object,
stamp_seal, turn_switch) are not selected because the frozen rule takes the
first ten.

## Dev14 inspection result

The adapter was inspected against dev14 CPFS on the audit date. The explicit
source paths
/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/r142_stage_s_deps/RoboTwin and
/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/r142_stage_s_deps/Evo-1 resolve to
RoboTwin stable_2.0 at 13c3c47ff4312dd62484bcd51be034af55c062d1 and Evo-1 at
5fd14b015013c4fd0aacf5f8f48f868ca9b870a2. All ten selected task modules and
instruction files are present. The explicit checkpoint path
/mnt/cpfs/zbl-cpfs-new/USERS/leon/cache/r142_stage_s/models/Evo1_RoboTwin2_clean_ce8c583724706fbf7a03c17237761c65bf6813a7
contains `config.json` and `norm_stats.json`, but the required
`mp_rank_00_model_states.pt` is still absent while the checkpoint download is
pending. RoboTwin
stable 2.0 requires SAPIEN/CuRobo and the Evo-1 plugin/server.

Therefore the current live asset status is `BLOCKED_CAPABILITY` because the
three checkpoint files are not yet present. This is not a scientific failure
and no synthetic rollout is emitted. Independently, the deployed policy
server must expose the exact RNG snapshot/restore and candidate-seeding hooks
required by the concrete adapter; the public unpatched proxy does not. The
remaining prerequisites are to verify checkpoint file hashes at the pinned HF
revision and run the exact replay preflight through the concrete wrapper.

The audit script requires the concrete wrapper path explicitly via
--runtime-wrapper; source and weight presence alone never yields READY.
The wrapper must export ConcreteRoboTwinRuntime and EvoProxyStateAdapter.
The public Evo-1 deploy_policy.py has neither exact snapshot hooks nor server
RNG restore, so it is correctly rejected.

## Substrate-A execution contract

`scripts/stage_s_robotwin_main.py` is the only A execution entry point. It
freezes the ten tasks above, 16 initial-state families per task, and 32
independently seeded policy candidates per family. The official RoboTwin
`setup_demo`/`get_obs`/`take_action` path is used directly; no expert
trajectory or synthetic success callback is accepted. Each candidate stores
the complete action prefix, EEF trajectory, rigid-actor/object trajectories,
policy-forward count, environment-step count, final official success flag,
and seed genealogy. A family is written atomically as `family.json`,
`genealogy.jsonl`, `SNAPSHOT.json` (the initial simulator/policy/RNG replay
state), `SHA256SUMS`, then `COMPLETED_FAMILY.json`. Each candidate also
persists its terminal policy history, action queue, and environment/policy RNG
states in `family.json`.

Before candidate 0 the runner must pass the concrete
`restore -> same action -> next-state` check at tolerance `1e-9`. Failure is a
capability block and produces no family completion marker. A valid marker is
immutable and is skipped on same-directory retry; a hash mismatch is
fail-closed. `scripts/stage_s_robotwin_payload.py` remains a non-submitting
rank template. The formal launcher is `scripts/stage_s_robotwin_a_pai.sh`:
one robot-idle 8×A800 worker starts exactly eight one-server/one-client pairs,
binds rank `r` to `CUDA_VISIBLE_DEVICES=r` and `127.0.0.1:19000+r`, and shards
`flat_task_family_index % 8 == rank`. It uses the same output directory across
platform restarts and invokes `stage_s_robotwin_finalize.py` only after all
eight rank markers and all 160 family markers verify. The launcher enforces
88 CPU/1525 GiB memory limits and Beijing 09:30–09:40 and 19:30–19:40
fail-closed scheduler guards. The visible calibration command refuses
Step-0 because substrate A has no registered calibration phase.

## Evo exact-replay control protocol

Inspection of the pinned `Evo_1/scripts/Evo1_server.py` and
`RoboTwin_evaluation/policy/Evo1/deploy_policy.py` found that the released
server accepts only inference JSON and the released `Evo1Proxy` exposes only
`infer`/`close`; its `reset_model` is a no-op. In particular, the pinned
flow-matching action head samples its initial action with Torch, so a
client-local NumPy seed cannot establish candidate independence or replay.

`src/r142_stage_s/robotwin.py` now supplies an opt-in, versioned control
protocol on the *same* WebSocket used by `infer`:

| Control | Purpose |
|---|---|
| `set_seed` | Set Python, NumPy, Torch CPU, and all CUDA streams from the persisted integer candidate seed |
| `capture_rng` | Return exact serialized Python/NumPy/Torch/CUDA states |
| `restore_rng` | Validate protocol, device count/availability, byte shape, and restore all streams |

Every message has `r142-evo-exact-replay/v1` and a request id. Inference
payloads are not changed. `scripts/stage_s_robotwin_evo_server_patch.py`
contains the minimal dispatcher: call `control_response()` before the
unchanged pinned `infer_from_json_dict()` branch and send a control response
on the same socket. `scripts/stage_s_robotwin_evo_server.py` loads the
released server by path, checks its immutable revision and checkpoint
manifest, installs this dispatcher with `require_torch=True`, and records
server/model/source provenance per rank. A malformed, unversioned, or
rejected control fails closed; there is no local-RNG or synthetic fallback.
The runner prefers the exact integer `SeedSequence([initial_seed, candidate_index])` via
`OfficialEvoPolicy.seed()` and retains `set_rng(Generator)` only for local
test adapters.

The bridge implementation is unit-tested in-process, including policy
history/action-queue restoration and the exact same-action/next-state
`1e-9` wiring. It is not evidence that a server has been deployed: the
currently inspected public pinned server source is still unpatched, and the
audit reports `server_control_deployed=false` until the dispatcher is
installed in its message loop (or an equivalent in-process server wrapper is
explicitly supplied). Together with missing checkpoint files/revision
evidence, this keeps a real A rollout `BLOCKED_CAPABILITY`; it is an
infrastructure precondition, not a scientific result.
