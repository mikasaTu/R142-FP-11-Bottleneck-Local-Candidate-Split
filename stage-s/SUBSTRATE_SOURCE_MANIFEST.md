# Stage-S substrate source manifest

This manifest is frozen before any Stage-S outcome is observed. It fixes the
three input substrates; it is not the post-calibration `PROTOCOL.md`.

## A — RoboTwin

- RoboTwin `stable_2.0` at `13c3c47ff4312dd62484bcd51be034af55c062d1`.
- Evo-1 `evo1-flash` at `5fd14b015013c4fd0aacf5f8f48f868ca9b870a2`.
- Released `MINT-SJTU/Evo1_RoboTwin2_clean` checkpoint revision
  `ce8c583724706fbf7a03c17237761c65bf6813a7`.
- `demo_clean`, horizon 37, the published policy recipe without degradation.
- Selection rule: alphabetically first ten tasks with published clean-policy
  success in the inclusive interval [0.25, 0.65]. The exact task list and
  published rates are in `SUBSTRATE_SOURCE_MANIFEST.json`.

The source branch matters: the Evo-1 authors state that their published
RoboTwin numbers use `stable_2.0`; RoboTwin `main` is not interchangeable.

## B — LIBERO referential ambiguity

The task set is exactly `libero_10` task IDs 0–9 from Stage-R. The pinned final
`pi05_libero` checkpoint and OpenPI revision are unchanged. Calibration uses
the outcome-blind, evenly spaced task IDs 0, 3, 6, and 9.

## C — under-trained policy

C uses the same `pi05_libero` architecture, data configuration, task IDs and
OpenPI revision as Stage-R. It is a real training lineage from the official
pi0.5 base, not weight interpolation. The four preselected calibration
checkpoints are steps 1000, 3000, 6000, and 10000 from seed 42. Every C result
must carry `WEAK_SUBSTRATE` and cannot be a headline qualification.
