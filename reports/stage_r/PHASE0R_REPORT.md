# R142-FP-11 Stage-R Phase-0R report

## Decision label

- **observed fact** — `NO_STAGE_R_PRECONDITION_ON_PINNED_PI05_LIBERO`.
- **observed fact** — the frozen positive control passed, all 40 LIBERO tasks
  were analyzed, zero tasks met the conjunctive retention rule, and the
  workflow stopped at `CHECKPOINT_1_STOP` with `phase1_authorized=false`.
- **observed fact** — RoboTwin remained
  `SOURCE_LIMITATION_UNVERIFIABLE`; no substitute trajectories were used.
- **interpretation** — on this pinned policy/task distribution, there is no
  eligible shared-failing-prefix population on which a bottleneck-local split
  intervention can be tested. This is a negative precondition result, not a
  claim that every possible policy or environment lacks bottlenecks.

## Frozen lineage

- **observed fact** — protocol `r142-stage-r-phase0r-v1`; scientific source
  commit `24423e8114ace80e6a76f22bee29992cea420cfc`.
- **observed fact** — OpenPI `pi05_libero`, OpenPI commit
  `54cbaee6ae0c010a1ed431871cdaa8f4684ac709`, QPILOTS commit
  `eacf47b981e3b22357f8a74902f8dad8cfcfa375`, LIBERO commit
  `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`.
- **observed fact** — the fixed authority map was parent indices 0--31,
  supplemental shard A indices 32--35, and shard B indices 36--39. The merge
  did not inspect success values and never selected parent redundancy for
  indices 32--39.
- **observed fact** — authority-manifest SHA-256:
  `3d5a37ec8a7e2c0dfd0c808ad59553c43a13c846b90f99c1afaa3529a072469c`.

## Gates

| Gate | Passing tasks / 40 | Result |
|---|---:|---|
| `rho >= 3` | 14 | partial |
| fraction of initial states with `p_e <= 1/32` >= 0.25 | 0 | failed globally |
| median `t_div` / episode length >= 0.10 | 0 | undefined globally |
| stable successful modes >= 2 | 30 | partial |
| E6 success/progress window | 1 | partial |
| all gates jointly | 0 | failed |

- **observed fact** — the positive control passed all divergence, action-split,
  and mode checks. Its `rho=25.453`, low-p fraction was `0.5`, and pooled
  success was `0.410`.
- **observed fact** — all natural tasks had low-p fraction `0.0` and zero
  all-fail initial states. Consequently, the preregistered all-fail-only
  divergence and action-prefix split measurements had no eligible natural
  groups and every natural `t_div` was undefined.

## Method

- **observed fact** — 16 deterministically ranked initial states per task and
  32 independent policy-noise candidates per state were run to eventual
  termination: 512 rollouts per task, 20,480 total.
- **observed fact** — all selected NPZ/metadata pairs passed protocol identity,
  SHA, exact 16-by-32 coverage, rollout-seed formula, offset/length consistency,
  finite trajectory arrays, and numeric-owner validation.
- **observed fact** — thresholds were frozen from the null/permutation control
  before outcome analysis: divergence RMS `0.0070381425`, action-prefix
  silhouette `0.7127923129`, and successful-mode silhouette `0.3565303449`.
- **observed fact** — the analyzer used eventual official success for `p_e`,
  `p_bar`, and `rho`; it computed failing-prefix divergence only for initial
  states whose 32 descendants all failed; and it clustered successful
  mid-trajectory action/pose features for mode multiplicity.

## Quantitative table

| Task | p̄ | rho | low-p frac | all-fail states | median t_div frac | modes | E6 | retained |
|---|---:|---:|---:|---:|---:|---:|:---:|:---:|
| libero_spatial/00 | 1.000 | undefined | 0.000 | 0 | undefined | 2 | fail | no |
| libero_spatial/01 | 1.000 | undefined | 0.000 | 0 | undefined | 2 | fail | no |
| libero_spatial/02 | 1.000 | undefined | 0.000 | 0 | undefined | 2 | fail | no |
| libero_spatial/03 | 0.992 | 2.419 | 0.000 | 0 | undefined | 2 | fail | no |
| libero_spatial/04 | 0.951 | 6.815 | 0.000 | 0 | undefined | 2 | fail | no |
| libero_spatial/05 | 0.930 | 14.182 | 0.000 | 0 | undefined | 1 | fail | no |
| libero_spatial/06 | 0.998 | 1.002 | 0.000 | 0 | undefined | 2 | fail | no |
| libero_spatial/07 | 0.998 | 1.002 | 0.000 | 0 | undefined | 2 | fail | no |
| libero_spatial/08 | 1.000 | undefined | 0.000 | 0 | undefined | 1 | fail | no |
| libero_spatial/09 | 0.990 | 1.602 | 0.000 | 0 | undefined | 2 | fail | no |
| libero_object/00 | 0.977 | 1.183 | 0.000 | 0 | undefined | 1 | fail | no |
| libero_object/01 | 0.990 | 3.326 | 0.000 | 0 | undefined | 2 | fail | no |
| libero_object/02 | 0.953 | 7.741 | 0.000 | 0 | undefined | 2 | fail | no |
| libero_object/03 | 0.975 | 1.047 | 0.000 | 0 | undefined | 2 | fail | no |
| libero_object/04 | 0.980 | 8.241 | 0.000 | 0 | undefined | 1 | fail | no |
| libero_object/05 | 0.994 | 1.587 | 0.000 | 0 | undefined | 2 | fail | no |
| libero_object/06 | 1.000 | undefined | 0.000 | 0 | undefined | 2 | fail | no |
| libero_object/07 | 0.998 | 1.002 | 0.000 | 0 | undefined | 2 | fail | no |
| libero_object/08 | 0.998 | 1.002 | 0.000 | 0 | undefined | 2 | fail | no |
| libero_object/09 | 0.988 | 2.114 | 0.000 | 0 | undefined | 1 | fail | no |
| libero_goal/00 | 0.977 | 3.004 | 0.000 | 0 | undefined | 2 | fail | no |
| libero_goal/01 | 0.994 | 0.872 | 0.000 | 0 | undefined | 2 | fail | no |
| libero_goal/02 | 0.916 | 4.318 | 0.000 | 0 | undefined | 2 | fail | no |
| libero_goal/03 | 0.934 | 7.989 | 0.000 | 0 | undefined | 2 | fail | no |
| libero_goal/04 | 0.992 | 0.806 | 0.000 | 0 | undefined | 2 | fail | no |
| libero_goal/05 | 0.988 | 2.833 | 0.000 | 0 | undefined | 2 | fail | no |
| libero_goal/06 | 0.994 | 3.018 | 0.000 | 0 | undefined | 1 | fail | no |
| libero_goal/07 | 1.000 | undefined | 0.000 | 0 | undefined | 2 | fail | no |
| libero_goal/08 | 1.000 | undefined | 0.000 | 0 | undefined | 2 | fail | no |
| libero_goal/09 | 0.963 | 1.425 | 0.000 | 0 | undefined | 2 | fail | no |
| libero_10/00 | 0.965 | 1.704 | 0.000 | 0 | undefined | 1 | fail | no |
| libero_10/01 | 0.992 | 1.344 | 0.000 | 0 | undefined | 2 | fail | no |
| libero_10/02 | 0.980 | 1.931 | 0.000 | 0 | undefined | 1 | fail | no |
| libero_10/03 | 0.979 | 1.530 | 0.000 | 0 | undefined | 2 | fail | no |
| libero_10/04 | 0.969 | 3.991 | 0.000 | 0 | undefined | 2 | fail | no |
| libero_10/05 | 0.980 | 6.500 | 0.000 | 0 | undefined | 2 | fail | no |
| libero_10/06 | 0.918 | 5.084 | 0.000 | 0 | undefined | 1 | fail | no |
| libero_10/07 | 0.992 | 0.806 | 0.000 | 0 | undefined | 2 | fail | no |
| libero_10/08 | 0.500 | 6.867 | 0.000 | 0 | undefined | 1 | pass | no |
| libero_10/09 | 0.967 | 4.085 | 0.000 | 0 | undefined | 2 | fail | no |

## Negative controls and failure cases

- **observed fact** — 14 tasks showed `rho>=3`, so between-initial-state
  heterogeneity existed, but none had even one all-fail state. High `rho` was
  therefore insufficient evidence of candidate collapse.
- **observed fact** — 30 tasks had at least two stable modes among successful
  trajectories, but no task combined this with a measurable shared failing
  prefix.
- **observed fact** — `libero_10/08` was the only E6 task (`p_bar=0.5`, progress
  median `0.833`, IQR `[0.667,1.0]`), but it had one stable successful mode,
  zero all-fail states, and undefined `t_div`.
- **observed fact** — most tasks were near ceiling: pooled success ranged
  `[0.5,1.0]` with median `0.9893`; all but `libero_10/08` failed E6.

## Mechanism explanation

- **observed fact** — the code computes `t_div` only inside a guard that skips
  every initial state with any successful descendant. Because every natural
  state had at least one success, the failing-prefix branch of the analyzer was
  never entered for natural tasks.
- **interpretation** — this is not a detector malfunction: the positive control
  exercised the same code and passed. The natural learned-policy rollouts
  supplied heterogeneity and successful modes, but not the conjunction required
  by the idea: a family collapsed onto a common failure prefix followed by a
  meaningful trajectory-axis split.
- **interpretation** — the apparent positive ingredients (`rho` and successful
  mode count) live in different populations. `rho` reflects varying success
  probability across initial states; mode count is measured only on successful
  rollouts. Neither implies an all-fail family or a local failure bottleneck.
- **untested hypothesis** — a different, harder checkpoint/task distribution
  could produce all-fail families. This run does not test that distribution and
  does not license selecting one after seeing these outcomes.

## Compute accounting

| Suite | rollouts | policy forward passes | environment steps |
|---|---:|---:|---:|
| LIBERO-Spatial | 5,120 | 109,844 | 539,184 |
| LIBERO-Object | 5,120 | 147,736 | 728,460 |
| LIBERO-Goal | 5,120 | 119,255 | 586,237 |
| LIBERO-10 | 5,120 | 281,514 | 1,397,875 |
| **total** | **20,480** | **658,349** | **3,251,756** |

- **observed fact** — branch count was zero in observational Phase-0R.
- **observed fact** — supplemental A and B each used one 4-A800 idle-pool
  worker with 46 CPU cores and 800 GiB memory/shared memory, below the user
  ceilings. Their exact PAI jobs ended `Succeeded`; repeated shard-B
  preemptions resumed in the same directory.

## Evidence boundary

- **observed fact** — final analysis summary SHA-256:
  `7da6f4751f64f08bdfa50e7f37ac1dd1a2f80b7f3f36862f12e7e95ef475c299`.
- **observed fact** — final completion SHA-256:
  `9daff7a544ebb7b1c4e3f6fbf10e38b124c067aa6c56a417f822a028dd5fb10a`.
- **observed fact** — the final CPFS `SHA256SUMS` verified all 87 declared
  files; its SHA-256 is
  `2d887069054098d569c4f42260d419f7bcc9377c41c51d46fb14a7ef8139f924`.
- **interpretation** — this result does not license Phase-1R, any split
  intervention, any VLA claim beyond the pinned policy/tasks, or changing the
  frozen retention rule after unblinding.

