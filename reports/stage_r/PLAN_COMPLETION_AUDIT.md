# R142-FP-11 Stage-R plan completion audit

## Decision label

- **observed fact** — every unconditional Stage-R experiment and publication
  deliverable in the frozen plan is complete.
- **observed fact** — the scientific decision is
  `NO_STAGE_R_PRECONDITION_ON_PINNED_PI05_LIBERO`; the workflow is at
  `CHECKPOINT_1_STOP`, `retained_tasks=[]`, and `phase1_authorized=false`.
- **interpretation** — Phase-1R is not an unfinished experiment. The frozen
  plan permits it only for tasks approved at Checkpoint 1 and explicitly says
  to write the report and stop when zero tasks pass. Starting Phase-1R now
  would violate both the task-eligibility rule and the human-approval gate.

## Frozen lineage

- **observed fact** — the original user-supplied Stage-R plan is preserved at
  `docs/steps/step3/PLAN.md`; its source attachment SHA-256 is
  `cd2101cae904c20b7af39333b73607ecdfb3e568e3bdb228c7a1b590a6c24d26`.
- **observed fact** — the original idea and top-level experiment plan are
  preserved under `docs/steps/context/`.
- **observed fact** — the scientific source commit was
  `24423e8114ace80e6a76f22bee29992cea420cfc`; Phase-0R protocol id was
  `r142-stage-r-phase0r-v1`.
- **observed fact** — the outcome-blind authority mapping was parent indices
  0--31, supplemental A 32--35, and supplemental B 36--39. Authority manifest
  SHA-256 is
  `3d5a37ec8a7e2c0dfd0c808ad59553c43a13c846b90f99c1afaa3529a072469c`.

## Plan-by-plan completion matrix

| Frozen requirement | Evidence | Status |
|---|---|:---:|
| Phase E: E1 manifest | `results/stage_r/gates/pai_r4_idle4/gates/E1_manifest.json` | PASS |
| E2 instrumentation disabled is bit-identical | `E2_bit_identity.json`: bit identical, max error 0 | PASS |
| E3 restore/action/next-state equality | `E3_restore.json`: next-state error 0 | PASS |
| E4 restore all state components with separate ablations | `E4_component_ablations.json`: full restore error 0; buffer, queue, RNG, simulator ablations diverged | PASS |
| E5 branch RNG is non-degenerate | `E5_branch_rng.json`: executed actions changed | PASS |
| E6 non-ceiling baseline | 64 rollouts, success 0.46875, progress median 0.5, Q25 0.5, Q75 1.0 | PASS |
| Phase E positive control | 512 rollouts, 210 successes, observed modes -1 and +1 | PASS |
| Phase-0R frozen protocol before outcomes | `docs/stage_r/PHASE0R_PROTOCOL.md` plus invalid/deviation history | PASS |
| Four LIBERO suites, E=16 and N=32 | 40 tasks x 512 full rollouts = 20,480 | PASS |
| Eventual success and trajectory-axis measurements | full-to-termination raw arrays and frozen analyzer outputs | PASS |
| Null/permutation threshold with at least 1000 shuffles | frozen `docs/stage_r/PHASE0R_THRESHOLDS.json` and control records | PASS |
| Phase-0R positive control | `positive_control_pass=true` | PASS |
| Full table including rejected candidates | 40 LIBERO rows plus explicit RoboTwin source-limitation row | PASS |
| Raw per-task records | 40 NPZ plus 40 paired metadata JSON under `results/stage_r/phase0r/raw/` | PASS |
| Raw integrity | 80/80 files pass `results/stage_r/phase0r/RAW_SHA256SUMS` | PASS |
| Gate-bundle integrity | all 16 files pass `results/stage_r/gates/pai_r4_idle4/BUNDLE_SHA256SUMS` | PASS |
| Completion artifacts | `COMPLETED_PHASE0R.json`, `COMPLETED_EVALUATION_RESULT.json`, and canonical SHA manifests | PASS |
| Negative controls, flat cases, compute accounting | report sections and `MECHANISM_REVERSE_EXPLANATION.md` | PASS |
| RoboTwin candidate-source accounting | zero rollouts and `SOURCE_LIMITATION_UNVERIFIABLE` | REPORTED LIMITATION |
| Checkpoint 1 stop | zero retained tasks; report published; no Phase-1 work started | PASS |

## RoboTwin evidence boundary

- **observed fact** — the pinned pi0.5-LIBERO policy/checkpoint has no
  compatible RoboTwin observation, action, normalization, and checkpoint
  contract. This was frozen in `docs/stage_r/SOURCE_INTERFACE_AUDIT.md` before
  candidate outcomes were inspected.
- **observed fact** — the authoritative Phase-0R summary contains a dedicated
  RoboTwin row with `rollout_count=0`, `retained=false`, and
  `SOURCE_LIMITATION_UNVERIFIABLE`.
- **interpretation** — running a scripted RoboTwin policy would change the
  scientific object rather than complete the pinned-policy comparison.
  Fabricating or adapting trajectories after outcome inspection was prohibited.

## Quantitative closure

| Item | Value |
|---|---:|
| Natural LIBERO tasks | 40 |
| Initial states per task | 16 |
| Candidates per state | 32 |
| Full rollouts | 20,480 |
| Policy forward passes | 658,349 |
| Environment steps | 3,251,756 |
| Tasks with rho at least 3 | 14 |
| Tasks with at least two stable successful modes | 30 |
| Tasks with any all-fail initial state | 0 |
| Tasks retained by all frozen gates | 0 |

## Mechanism explanation

- **observed fact** — `task_metrics()` measures failing-prefix divergence only
  for an initial-state family whose 32 descendants all fail. Every natural
  family had at least one eventual success, so all natural `t_div` values were
  undefined.
- **observed fact** — the same code path passed the positive control, so the
  result is not explained by a dead detector.
- **interpretation** — success overdispersion and successful-mode multiplicity
  were present in different populations, but neither implies the shared
  all-fail prefix required by the proposed split mechanism. The prerequisite
  population was absent; no intervention improvement or degradation can be
  truthfully attributed.
- **untested hypothesis** — another policy/checkpoint/task distribution might
  contain eligible all-fail families. Selecting one after seeing these results
  is outside this frozen plan and is not licensed by this audit.

## Evidence boundary

- **observed fact** — Phase-0R analysis summary SHA-256 is
  `7da6f4751f64f08bdfa50e7f37ac1dd1a2f80b7f3f36862f12e7e95ef475c299`;
  completion SHA-256 is
  `9daff7a544ebb7b1c4e3f6fbf10e38b124c067aa6c56a417f822a028dd5fb10a`.
- **observed fact** — the canonical CPFS 87-file final manifest passed and has
  SHA-256
  `2d887069054098d569c4f42260d419f7bcc9377c41c51d46fb14a7ef8139f924`.
  It preserves CPFS-relative `analysis/` paths; repository-local verification
  is instead provided by `BUNDLE_SHA256SUMS` and `RAW_SHA256SUMS`.
- **interpretation** — completion of the frozen plan does not convert its
  negative precondition result into support for the idea, does not authorize
  Phase-1R or Phase-2R, and does not license a VLA claim beyond the pinned
  policy/task distribution.
