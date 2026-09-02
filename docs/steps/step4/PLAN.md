# TASK: R142-FP-11 Stage-S — substrate qualification screen

## CONTEXT
Repo: mikasaTu/R142-FP-11-Bottleneck-Local-Candidate-Split
Frozen hypothesis: docs/steps/context/ORIGINAL_IDEA.md

CLOSED tracks. Do not extend, rerun, or cite any of them as evidence FOR the idea:
- Stage-1 (toy ForkPush2D). The location effect is hard-coded in `rollout()`
  control flow: injections at t<t* are discarded, the mode commits once at t*,
  and injections at t>t* cannot change it. Its headline numbers are therefore
  tautological. Its detector fires on 75.7% of no-bottleneck episodes.
- Stage-2A (DDPM denoising axis, DiffusionPolicy PushT). Wrong axis,
  ceiling-bound outcome, precondition never verified. 0/24 natural snapshots
  reached the location-sensitivity threshold; max spread-range 0.041 vs 0.10.
- Stage-R Phase 0R (pinned pi0.5-LIBERO, 40 tasks, 640 families, 20480 rollouts).
  0/640 families were all-fail. Pooled task success median 0.9893.

Stage-S does NOT test the hypothesis. Stage-S tests whether any candidate
substrate satisfies the hypothesis's PRECONDITION. Read that sentence twice.

Start a new track `stage-s/` in the same repo. Reuse ONLY the Stage-R
screening harness, replay/snapshot machinery, genealogy logging, PAI launcher,
and SHA-256 manifest infrastructure.

## OBJECTIVE
Screen three candidate substrates against five pre-committed gates and emit
exactly one decision code. Deliver a substrate, or deliver a kill.

## SUBSTRATES — run all three. Do not stop early on a favorable result.
A. RoboTwin. 10 tasks whose published pinned-policy success is in [0.25, 0.65].
B. LIBERO referential-ambiguity variant. The SAME 10 LIBERO-Long tasks used in
   Stage-R, with one visually similar distractor object added for the target
   referent. The existing unperturbed Stage-R run on these tasks is the
   built-in null control.
C. Under-trained checkpoint of the same policy on the Stage-R LIBERO tasks.
   Tag every C-derived result `WEAK_SUBSTRATE` in the report. C is a
   mechanism-isolation arm only. It may never be the headline substrate.

## STEP 0 — CALIBRATION (B and C only). Do this, commit it, then continue.
Pilot: 4 tasks x 8 initial states x 8 candidates, over 4 perturbation
magnitudes (B) / checkpoint steps (C). Select the setting whose POOLED SUCCESS
is closest to 0.45. Freeze and commit it before the main screen.

HARD RULE: calibrate on pooled success ONLY. Do not compute, inspect, or log
all-fail-family fraction, overdispersion, divergence onset, or any Gate S2-S5
statistic during calibration. Doing so selects the substrate on the outcome
under test and invalidates the whole stage.

## STEP 1 — FREEZE THE PROTOCOL. Commit before any main-screen rollout.
Commit a single `stage-s/PROTOCOL.md` containing, as literal numbers:
all five gate thresholds, the divergence metric D(t) and its normalization,
the tau derivation procedure, the family definition, the "near-all-fail"
definition (<=1 success in 32; strict 0/32 reported as secondary), the RNG
seed plan, and the compute accounting unit. Record the commit SHA in every
downstream artifact.

## STEP 2 — MAIN SCREEN
Per substrate: 10 tasks x 16 initial states x 32 independent candidates, each
run to episode termination. No intervention. Observation only.

## STEP 3 — GATES (all thresholds already committed in STEP 1)
S1 DIFFICULTY   pooled success in [0.30, 0.60]
S2 COLLAPSE     fraction of families with <=1/32 success >= 0.10
                AND rho = Var_obs / Var_binom >= 3
                AND observed near-all-fail count > 20x binomial expectation
S3 PREFIX       among near-all-fail families, median divergence onset t_div
                >= 0.10 * episode length, AND families with t_div = 0 are < 25%.
                tau = 95th percentile of D(t) among successful same-task
                episodes at matched t. Committed in STEP 1.
S4 RECOVERABLE  for >= 30% of probed near-all-fail families, at least one
                prefix-preserving branch at some INTERIOR t reaches success;
                AND oracle-t recovery rate exceeds an equal-branch-count
                random-t probe, paired bootstrap (10000 replicates) 95% CI
                lower bound > 0.
                S4 is an ORACLE UPPER BOUND, not a deployable method.
                Passing S4 means the idea is not dead. Nothing more.
S5 BUDGET       extending best-of-N from 32 to 64 with fresh seeds rescues
                < 5% of near-all-fail families.

## STEP 4 — CONTROLS. Both are mandatory. Report before the gate verdicts.
POSITIVE (instrument): run the unmodified Stage-S statistics pipeline on the
  Stage-1 toy, where all-fail families exist by construction. The pipeline must
  detect them. This is the ONLY permitted use of Stage-1: testing the
  instrument, never supporting the idea.
NULL (free): run the unmodified pipeline on the existing unperturbed Stage-R
  pi0.5-LIBERO data. It must return NO_FAMILY_COLLAPSE.
If the positive control fails, the stage outcome is PIPELINE_INVALID, not a
scientific finding. Stop and report.

## DECISION CODES — emit exactly one
SUBSTRATE_QUALIFIED               A or B passes all five gates
NO_SUBSTRATE_AT_TARGET_DIFFICULTY S1 fails on all three
NO_FAMILY_COLLAPSE                S2 fails
COLLAPSE_AT_ORIGIN                S3 fails, t_div = 0 dominant
UNRECOVERABLE_FAILURES            S4 fails
BUDGET_SUFFICES                   S5 fails
WEAK_SUBSTRATE_ONLY               only C passes
PIPELINE_INVALID                  positive control failed

## HARD CONSTRAINTS
1. Dependent variable is eventual episode success, run to termination. No
   short-horizon proxy may gate anything.
2. Branch location is a CONTROL STEP in the trajectory. Never a denoising step.
3. Thresholds are frozen in STEP 1 and are not renegotiable after unblinding.
4. Never use latent / z pairwise distance as a signal. Already falsified in
   Stage-2A (pooled correlation -0.0031).
5. State restore must include simulator state, policy observation history
   buffer, action chunk queue, and every RNG stream. Verify explicitly:
   restore -> same action -> next-state error <= 1e-9. This is the gate that
   fails silently.
6. Report every substrate, every gate, every flat and negative result. No
   filtering, no example selection.
7. Compute accounting in policy forward passes (primary) and env steps
   (secondary).
8. Completion means a persisted COMPLETED_*.json with SHA-256 verification.
   Queued, running, or one finished shard is not a result.

## PROHIBITED
- Fabricating, interpolating, or resampling any trajectory, state, or outcome.
- Inspecting any S2-S5 statistic during STEP 0 calibration.
- Dropping, swapping, or re-tuning a substrate after seeing its gate outcomes.
- Relaxing or reinterpreting a committed threshold after unblinding.
- Artificially degrading the pinned policy on substrate A or B to manufacture
  failure families. Degradation is confined to substrate C and is labeled.
- Advancing past the CHECKPOINT below without explicit human approval.

## DELIVERABLES
stage-s/PROTOCOL.md (committed first)
stage-s/CALIBRATION_REPORT.md
stage-s/results/{A,B,C}/  raw per-family outcomes + genealogy
stage-s/CONTROLS_REPORT.md
stage-s/STAGE_S_REPORT.md  gate table per substrate, decision code, compute
                           accounting, and every negative result

## CHECKPOINT — STOP HERE
After emitting the decision code, STOP. Do not design Stage-T. Do not start any
intervention experiment. Do not propose an alternative substrate. Post the gate
table and the decision code and wait for human approval.