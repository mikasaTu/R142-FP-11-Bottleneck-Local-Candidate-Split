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
| place_a2b_left | 0.48 |
| place_a2b_right | 0.38 |
| place_bread_basket | 0.63 |
| place_bread_skillet | 0.63 |
| place_can_basket | 0.50 |
| place_fan | 0.34 |
| place_object_scale | 0.49 |
| place_shoe | 0.33 |
| put_object_cabinet | 0.39 |

The next four eligible names (rotate_qrcode, scan_object, stamp_seal,
turn_switch) are not selected because the frozen rule takes the first ten.

## Dev14 inspection result

The adapter was inspected against dev14 CPFS paths
/mnt/cpfs/zbl-cpfs-new/USERS/leon and /workspace/leon on the audit date.
No checkout at the exact RoboTwin pin, no Evo-1 checkout at its exact pin,
and no local copy of the three checkpoint files were found. RoboTwin stable
2.0 requires SAPIEN/CuRobo and the Evo-1 plugin/server; these are not present
in the inspected assets. The checkpoint was intentionally not downloaded
(the delegated work is static/unit-only).

Therefore the live substrate is BLOCKED_CAPABILITY. This is not a scientific
failure and no synthetic rollout is emitted. The precise next prerequisites
are: install/read-only checkout of all pinned sources, verify the checkpoint
file hashes at the pinned HF revision, and supply an Evo-1 policy wrapper
exposing exact simulator/history/action-queue/environment-RNG/policy-RNG
snapshot and restore hooks.
