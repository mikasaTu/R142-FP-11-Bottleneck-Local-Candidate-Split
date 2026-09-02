# Stage-S substrate source audit

Audit date: 2026-09-02 (Asia/Shanghai).  This file records source and asset
availability only; it is not a Stage-S gate result.  No PAI job or large
rollout was submitted by this implementation pass.

## Frozen Stage-R LIBERO lineage

The Stage-R comparison suite is the ten-task `libero_10` suite, task IDs
`0..9`, in this exact order:

| ID | task stem | B target referent |
|---:|---|---|
| 0 | `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket` | `alphabet_soup_1` |
| 1 | `LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket` | `cream_cheese_1` |
| 2 | `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it` | `moka_pot_1` |
| 3 | `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it` | `akita_black_bowl_1` |
| 4 | `LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate` | `porcelain_mug_1` |
| 5 | `STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy` | `black_book_1` |
| 6 | `LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate` | `porcelain_mug_1` |
| 7 | `LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket` | `alphabet_soup_1` |
| 8 | `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove` | `moka_pot_1` |
| 9 | `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it` | `white_yellow_mug_1` |

Pinned runtime lineage used by the existing Stage-R artifacts:

* QPILOTS: `/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/QPILOTS-r16p15-stage1-task64-20260812`, commit `eacf47b981e3b22357f8a74902f8dad8cfcfa375`.
* OpenPI: `$QPILOTS/third_party/openpi`, commit `54cbaee6ae0c010a1ed431871cdaa8f4684ac709`.
* LIBERO: `$OPENPI/third_party/libero`, commit `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`.
* Python: `/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/openpi_py311/bin/python`.
* Official Stage-R policy checkpoint: `/mnt/cpfs/zbl-cpfs-new/USERS/leon/cache/openpi/r16p15/openpi-assets/checkpoints/pi05_libero`.
* Stage-R LIBERO configuration: `LIBERO_CONFIG_PATH=/mnt/cpfs/zbl-cpfs-new/USERS/leon/cache/libero/r16p15-stage1-task64`.
* Existing unperturbed raw archive: `/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_r/phase0r_merged/r142-stage-r-phase0r-authoritative-20260827/raw/`.

The original BDDL files are present under
`$OPENPI/third_party/libero/libero/libero/bddl_files/libero_10/`, and the ten
original `.pruned_init` files are present under the matching
`init_files/libero_10/` directory.  `Task64Environment` resolves both through
`LIBERO_CONFIG_PATH`; its qpos layout is changed by adding a free duplicate
object.

## Substrate A: RoboTwin

No locally evidenced learned/pinned RoboTwin policy satisfying the required
published success interval `[0.25, 0.65]` was found in the Leon CPFS code,
checkpoint, or cache roots.  The only directly available RoboTwin checkout is
the R22 responsibility-routing source at
`/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/R22-P19-Responsibility-Routed-Multi-Effector-Steering/stage2_robotwin`, parent commit
`05600234df39367424fcb8036533b5e111d2a0aa`; its manifest identifies RoboTwin
commit `266f3aadf505a4f7fe9af0faa41a20f5f47cd123` and task `handover_block`.
That evidence is scripted/expert simulator smoke (10 selected seeds out of an
11-seed search), explicitly not a learned-policy evaluation.  It cannot be
promoted to A or silently substituted with another task.

**A status: unavailable for the frozen screen.**  The Stage-S runner therefore
has no A result and no A success claim.

## Substrate B: exact-one visual duplicate

`src/r142_stage_s/libero.py` generates one same-type duplicate for each table
task, leaves `(:language)` and `(:goal)` unchanged, excludes the duplicate from
`obj_of_interest`, and places it in a deterministic translated copy of the
target init region.  The four frozen center offsets are `0.06`, `0.08`, `0.10`,
and `0.12` meters.  The target mapping is the table above (for task 8 the
target region is `moka_pot_right_init_region`).

The generated BDDL is executable only with a separately regenerated init-state
tensor.  Reusing the old `.pruned_init` is explicitly rejected because adding a
free object changes MuJoCo qpos dimensionality and object ordering.  The
required hand-off is a directory containing all ten new `.pruned_init` files
and `REGENERATED_INIT_STATES.json` asserting:

```json
{"regenerated": true, "old_init_reused": false}
```

No such regenerated B init-state directory was present during this audit.
Consequently B generation is implemented and unit-tested, but a real B
rollout is **blocked fail-closed** until a simulator-generated qpos bundle is
provided.  The existing unperturbed Stage-R archive remains the B null
control; it is not relabeled as the perturbed arm.

## Substrate C: under-trained exact policy checkpoints

The real community checkpoint at
`/mnt/cpfs/zbl-cpfs-new/USERS/leon/checkpoints/openpi/community_madokalif_pi05_libero_sft`
declares `steps: 60000` and reports near-ceiling LIBERO evaluation.  It is not
an under-trained C checkpoint.  The official complete checkpoint under
`/mnt/cpfs/zbl-cpfs-new/USERS/leon/checkpoints/openpi/pi05_libero_official_complete_20260731`
is likewise not under-trained.  The local incomplete mirror
`/mnt/cpfs/zbl-cpfs-new/USERS/leon/checkpoints/openpi/pi05_libero` lacks a
complete checkpoint tree.  No four exact early pi05-LIBERO actor checkpoints
were found; unrelated small LIBERO actors and critics do not satisfy the
same-policy contract and are not substituted.

`scripts/stage_s_libero_c.py` emits a fail-closed audit and the real OpenPI
PyTorch launcher contract. The frozen calibration steps are 1000, 3000,
6000, and 10000, so the launcher saves every 1000 steps and ends at 10001:

```text
cd <QPILOTS>/third_party/openpi && torchrun --standalone --nproc_per_node=4 \
  scripts/train_pytorch.py pi05_libero \
  --exp_name r142_stage_s_c_undertrained \
  --checkpoint-dir <output> --save-interval 1000 --num-train-steps 10001
```

The contract labels all C-derived output `WEAK_SUBSTRATE`, requires exactly
four unique real checkpoints with declared steps below 30,000, and explicitly
sets interpolation and artificial degradation to false.  It does not submit
the launcher.  **C status: unavailable pending four exact real checkpoints.**

## Snapshot and execution boundary

The Stage-S adapter captures the Stage-R-compatible simulator snapshot,
observation history, action queue, Python RNG, NumPy RNG, policy RNG state
when exposed, seed/counter, and control step.  `validate_restore_same_action`
restores and executes one identical action twice, requiring max absolute
next-state error `<= 1e-9`.  Main families are `10 x 16 x 32`; each family is
committed only after atomic `rollouts.npz`, full `genealogy.json`,
`snapshots.pkl`, `metadata.json`, `SHA256SUMS`, and the final
`COMPLETED_FAMILY.json` marker are present and hash-valid.  Calibration uses
exactly `4 x 8 x 8 x 4` and persists only setting, successes, total, and pooled
success.

This audit contains no scientific gate verdict.  It records that B and C are
contract-complete but not currently executable with real assets, and A lacks a
qualifying local learned-policy source.
