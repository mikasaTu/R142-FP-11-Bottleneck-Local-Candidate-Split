# TASK: R142-FP-11 Stage-R — trajectory-axis revalidation

## CONTEXT
Prior work is in `mikasaTu/R142-FP-11-Bottleneck-Local-Candidate-Split`.
Read `docs/steps/context/ORIGINAL_IDEA.md` for the frozen hypothesis.
Stage-1 (toy) and Stage-2A (denoising axis on PushT) are CLOSED. Do not extend,
rerun, or cite them as support. Their known defects:
- Stage-1 benchmark hard-codes the location effect in `rollout()` control flow;
  its detector has a 75.7% false-positive rate on a no-spike environment.
- Stage-2A branched along the DDPM denoising axis, not the trajectory axis, used
  an 8-control-step ceiling-bound outcome, and never verified the precondition.

Start a new track `stage-r/` in the same repo. Reuse only the replay/snapshot,
genealogy-logging, PAI-launcher and SHA-256 manifest infrastructure.

## OBJECTIVE
Test the ORIGINAL hypothesis on the correct axis:
> When a candidate family collapses onto a shared failing prefix, does there
> exist an earliest control step t* at which concentrating a fixed diversity
> budget beats uniform/random split locations and beats plain best-of-N, as
> measured by EVENTUAL episode success?

## HARD CONSTRAINTS
1. Dependent variable is EVENTUAL episode success (run to termination). Any
   short-horizon proxy is a secondary metric only. Never gate on one.
2. Branch location is a CONTROL STEP in the trajectory. Never a denoising step.
3. Freeze every protocol file BEFORE inspecting outcomes. Commit it first.
4. Effect-size thresholds are NOT hand-picked numbers. Compute a permutation
   null (>=1000 shuffles) and a null-control task, set every threshold at the
   95th percentile of that null, and commit the threshold before unblinding.
5. Every phase runs a POSITIVE CONTROL. A negative result without a passing
   positive control is `PIPELINE_INVALID`, not a scientific finding.
6. Aggregate-first statistics. Primary test on the task-level pooled curve with
   episode as the resampling unit (paired bootstrap, 10000 replicates).
   Per-episode prevalence is secondary. Do not gate on per-unit significance
   with small M.
7. Report all negative controls and flat cases. Never filter examples.
8. Compute accounting in policy forward passes (primary) and env steps
   (secondary). Report branch counts and slack per strategy.
9. Completion means a persisted `COMPLETED_*.json` with SHA-256 verification.
   Queued / Running / first-work / one finished shard is NOT a result.

## PROHIBITED
- Fabricating, interpolating, or resampling any trajectory, state, or outcome.
- Substituting a cheaper proxy for eventual success in any gate.
- Selecting episodes, snapshots, or examples after seeing descendant outcomes.
- Relaxing, rewriting, or reinterpreting a committed threshold after unblinding.
- Using latent/z pairwise distance as a detector signal (already falsified).
- Advancing past a CHECKPOINT without explicit human approval.
- Reporting a partial run as complete, or claiming a gate passed without the
  persisted artifact that proves it.

---

## PHASE E — engineering gates (blocking)

Pin one policy + checkpoint (revision + weights SHA-256) and one simulator
stack (exact commit + package versions). Then prove:

- E1 environment/checkpoint manifest persisted.
- E2 with instrumentation disabled, the path is bit-identical to the original.
- E3 state restore -> same action -> next state error <= 1e-9.
- E4 restore covers simulator state AND policy observation-history buffer AND
    action-chunk queue AND all RNG streams. Prove each component separately by
    ablating it and showing divergence. THIS IS THE MOST COMMON SILENT FAILURE.
- E5 changing only branch RNG changes the executed actions (non-degenerate).
- E6 baseline success rate in [0.25, 0.75] and final-progress distribution not
    piled at the upper bound.

Any failure -> `STAGE_R_ENGINEERING_INCOMPLETE`. Fix or change checkpoint. Do
not proceed to science.

Write `docs/stage_r/ENGINEERING_GATES.md` + `results/stage_r/gates/`.

---

## PHASE 0R — precondition screen and task selection (observational only)

NO intervention in this phase.

Candidate tasks: LIBERO suites (spatial / object / goal / 10) and RoboTwin.
For each task: E=16 initial states, N=32 independent full rollouts each.

Compute per task:
1. Overdispersion `rho = Var(p_e) / [p_bar(1-p_bar)/N]`, plus the fraction of
   episodes with `p_e <= 1/N`.
2. Shared failing prefix: for all-fail episodes, per-step pairwise divergence
   D(t) over end-effector and object pose, workspace-normalized; first crossing
   t_div. Also hierarchical clustering of executed action sequences -> step at
   which the merge tree splits into >=2 clusters.
3. Mode multiplicity: cluster mid-trajectory behavior of successful rollouts;
   count stable modes.
4. Ceiling check per E6.

RETAIN a task only if: rho >= 3 AND >=25% episodes with p_e <= 1/N AND median
t_div >= 10% of episode length AND modes >= 2 AND E6 holds.

Keep at most 3 tasks. If zero tasks pass, write the report and STOP.

Deliverables:
- `docs/stage_r/PHASE0R_PROTOCOL.md` (committed before running)
- `reports/stage_r/PHASE0R_REPORT.md` with the full table for ALL candidate
  tasks including rejected ones
- raw per-rollout records under `results/stage_r/phase0r/`

### >>> CHECKPOINT 1 — STOP HERE <
Post the Phase-0R table and the retained task list. Wait for explicit human
approval before Phase 1R. Do not start Phase 1R work of any kind.

---

## PHASE 1R — recoverability profile along the trajectory axis

Only for tasks approved at Checkpoint 1.

Per task: select 12 episodes descendant-blind by deterministic SHA-256 rank,
stratified 4 failed / 4 marginal / 4 successful from the Phase-0R baseline.
Branch grid: 10 control steps at episode-length deciles.
Per (episode, t): restore, resample M=16 action chunks with independent RNG,
execute, then CONTINUE WITH THE FROZEN POLICY TO EPISODE TERMINATION.
Split RNG into disjoint calibration and held-out streams.

Primary: `R(t) = P(eventual success | branch at t)`, pooled per task.
Derived: location sensitivity `max R - min R`; largest adjacent cliff;
commit point t_c = earliest t where R(t) < (Rmax+Rmin)/2.

Run all three controls in the SAME pipeline:
- POSITIVE CONTROL: a variant with a verified irreversible commitment.
  Construct it explicitly and document the construction.
- NULL CONTROL: a single-mode, highly recoverable task.
- PERMUTATION NULL: >=1000 within-episode branch-label shuffles per task.

Threshold: 95th percentile of the permutation null. Commit it before unblinding.

Decision:
- positive control below threshold -> `PIPELINE_INVALID`. Stop, fix tooling.
- positive control passes, natural below threshold ->
  `NO_TRAJECTORY_BOTTLENECK_ON_<policy>_<task>`. This is a valid negative
  result. Write it up fully. Do not proceed to Phase 2R.
- positive control passes, natural at/above threshold -> proceed.

Deliverables: frozen protocol, `reports/stage_r/PHASE1R_REPORT.md`, per-task
R(t) curves, full genealogy, all control results, compute accounting.

### >>> CHECKPOINT 2 — STOP HERE <
Post R(t) curves for natural + positive + null controls, the permutation-null
threshold, and the decision label. Wait for explicit human approval before
Phase 2R. Phase 2R is expensive; do not pre-run any part of it.

---

## OUTPUT FORMAT (every report)
Sections: Decision label / Frozen lineage / Gates / Method / Quantitative
tables / Negative controls and failure cases / Mechanism explanation /
Compute accounting / Evidence boundary.

Label every claim as exactly one of:
`observed fact` | `controlled intervention` | `interpretation` |
`untested hypothesis`.

State explicitly what the result does NOT license.
