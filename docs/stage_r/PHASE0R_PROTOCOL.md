# R142-FP-11 Stage-R Phase-0R frozen protocol

Protocol ID: `r142-stage-r-phase0r-v1`

This file is outcome-blind and immutable after its first Git commit. Any
correction requires a new protocol ID and an explicit deviation record. Stage-1
and Stage-2A are closed and provide no scientific support for Stage-R.

## Question and evidence boundary

Phase-0R tests only whether the precondition for the original hypothesis is
present on the **control-step trajectory axis** under one frozen learned policy:
candidate families that share a failing prefix, later split into multiple
behaviour modes, and have substantial between-initial-state failure
overdispersion. There is no split intervention in this phase. Eventual success
at environment termination is the primary outcome; no short-horizon metric can
select a task or pass a gate.

## Frozen lineage

- Policy: official OpenPI `pi05_libero`, OpenPI Git
  `54cbaee6ae0c010a1ed431871cdaa8f4684ac709`.
- Read-only integration: QPILOTS Git
  `eacf47b981e3b22357f8a74902f8dad8cfcfa375`.
- Checkpoint:
  `/mnt/cpfs/zbl-cpfs-new/USERS/leon/cache/openpi/r16p15/openpi-assets/checkpoints/pi05_libero`.
- Checkpoint-tree SHA-256: `42d571bd87f05f1182810f5a8bfa6d084c0d0dd277aff739bcf8f69868e6fb99`.
- Orbax root-manifest SHA-256:
  `65246951e69bd2b5118e609646bc9e8c439229bccbc1325643833aeb74f77104`.
- Action norm-stat SHA-256:
  `b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84`.
- LIBERO Git: `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`.
- Simulator: Python 3.11.11, MuJoCo 3.6.0, robosuite 1.4.1.
- Four-suite task-metadata SHA-256:
  `36ec2bb54444491059e749e6f1234b03aaf5e020aba0269dad06d3afe9b17772`.
- Policy contract: 10-step output, execute/replan every 5 control steps,
  physical action dimension 7, state dimension 8, 10 denoising steps.

The policy and checkpoint do not provide a compatible RoboTwin observation,
action, normalization, or checkpoint contract. RoboTwin remains in the
candidate table as `SOURCE_LIMITATION_UNVERIFIABLE` and receives no substituted
scripted/oracle trajectories.

## Candidate universe and deterministic sampling

The candidate universe is all task IDs 0--9 in each of `libero_spatial`,
`libero_object`, `libero_goal`, and `libero_10`, plus the preregistered RoboTwin
source row. All LIBERO tasks have 50 released initial states.

For every LIBERO task, rank init-state indices 0--49 by the bytewise SHA-256 of
`r142-stage-r-phase0r-v1|suite|task_id|init_state_index`; use the lowest 16.
For each selected state, run N=32 independent policy-noise streams. A rollout
seed is the unsigned 64-bit integer formed from the first eight bytes of
SHA-256(`r142-stage-r-phase0r-v1|suite|task_id|init_state_index|candidate_id`).
Candidate IDs are 0--31. No outcome-dependent retry is allowed.

Use 10 dummy stabilization actions before policy execution. Maximum policy
control steps are 220/280/300/520 for spatial/object/goal/10 respectively,
matching the pinned official evaluator. Stop early only on official task
success or simulator termination. The primary label is official eventual
success.

## Recorded trajectory and compute contract

For every executed control step record candidate ID, parent ID (null at root),
generation/control step, action-chunk index, executed action, end-effector pose,
all stable object pose observation keys, official predicate vector and
fraction, reward, termination reason, policy-noise seed/counter, and final
eventual-success label. Preserve the complete per-rollout record; summaries do
not replace raw data.

Count candidate-equivalent policy forward passes as the primary compute unit,
even when implementation micro-batches model inference. Also record physical
batched calls, environment steps, branch count (zero in Phase-0R), and unused
budget slack.

## Metrics

For task initial state e, `p_e` is successes/32 and `p_bar` is the pooled
success rate. Compute
`rho = Var_e(p_e) / [p_bar(1-p_bar)/32]` using sample variance. If the
denominator is zero, rho is undefined and the task fails retention. Report the
fraction of initial states with `p_e <= 1/32`.

For initial states whose 32 descendants all fail, compute D(t) as the mean
pairwise RMS distance of the workspace-normalized concatenation of EEF and
object position plus quaternion-chord coordinates. Pad no trajectory: only
pairs alive at t contribute, and report the at-risk count. `t_div` is the first
control step at which D(t) exceeds the frozen 95th-percentile null threshold.
Separately apply average-linkage hierarchical clustering to executed action
prefixes and report the first step yielding at least two stable clusters.

For successful rollouts, cluster middle-third executed action/pose summaries by
average linkage. A mode is stable only when its bootstrap stability exceeds the
frozen null threshold and it contains at least two real rollouts. No trajectory
may be fabricated, interpolated, padded, or resampled to create a mode.

Final progress is the official LIBERO predicate fraction. “Piled at the upper
bound” means median=1 and IQR=0 exactly. E6 holds only when pooled eventual
success is in [0.25, 0.75] and progress is not piled at the upper bound.

## Controls and threshold freeze

Before candidate outcomes are unblinded, the same analysis code must run:

- Positive control `GeometricCommit2D-v1`: a separately labelled continuous
  point-mass simulator with a physical fork barrier and one-way gate. Its two
  reachable goals and irreversible geometry are fixed in code before running.
- Null control `OpenPlane2D-v1`: the same integrator, horizon, action noise and
  terminal radius, but with no barrier and one central goal.
- At least 1000 deterministic within-initial-state candidate-label
  permutations, seeded from the protocol ID.

These controls validate only the metric pipeline and are not evidence for the
natural-policy hypothesis. Every derived D(t), merge split, or mode-stability
effect threshold is the 95th percentile of the maximum null-control/permutation
statistic. The generated `docs/stage_r/PHASE0R_THRESHOLDS.json` must be committed
before the candidate outcome analyzer is run. The explicit retention cutoffs
below are supplied by the Stage-R plan and are not editable.

## Retention, ranking, and stopping rule

Retain a task only if all hold: rho >= 3; at least 25% of initial states have
`p_e <= 1/32`; median t_div is at least 10% of observed episode length; at least
two stable successful modes; and E6 holds. If more than three pass, rank by
descending `(rho, low-p fraction, median t_div fraction, stable-mode count)`,
then ascending `(suite, task_id)`, and keep the first three.

Report every task, all controls, all undefined metrics, flat cases, and source
limitations. A failed positive control yields `PIPELINE_INVALID`. Zero retained
natural tasks yields `NO_STAGE_R_PRECONDITION_ON_PINNED_PI05_LIBERO` and STOP.
Otherwise report the retained list and STOP at CHECKPOINT 1. Phase-1R code,
rollouts, thresholding, or precomputation are forbidden without explicit human
approval.

## Statistics and completion

Task-level pooled curves are primary. Where uncertainty is reported, resample
initial-state episodes as paired units with 10,000 deterministic bootstrap
replicates. Per-initial-state prevalence is secondary. Completion requires
atomic rank markers, a persisted `COMPLETED_PHASE0R.json`, and successful
SHA-256 verification of every declared artifact. Queueing, Running,
`FIRST_WORK`, or a partial shard is not a result.
