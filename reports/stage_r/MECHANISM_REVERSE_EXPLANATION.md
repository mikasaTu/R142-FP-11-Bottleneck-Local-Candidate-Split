# Stage-R Phase-0R mechanism reverse explanation

This note follows the code-first reverse-explanation step only. It does not
generate a new idea or modify the frozen protocol.

## What the implementation actually measures

1. **observed fact** — `overdispersion()` groups the 32 descendants of each
   initial state, computes their success probability `p_e`, and reports
   `rho = Var(p_e) / [p_bar(1-p_bar)/32]` plus the fraction with
   `p_e <= 1/32`.
2. **observed fact** — `task_metrics()` skips a state as soon as any descendant
   succeeds. Trajectory divergence, threshold crossing, and action-prefix
   clustering are therefore defined only for a 32-of-32 failure family.
3. **observed fact** — successful mode multiplicity is computed separately:
   only successful trajectories enter standardized mid-trajectory
   action/pose clustering.
4. **observed fact** — task retention is a conjunction of `rho>=3`, low-p
   fraction `>=0.25`, defined late-enough `t_div`, at least two successful
   modes, and E6.

## Why superficially positive metrics did not support the mechanism

- **observed fact** — 14/40 tasks passed `rho>=3`, and 30/40 had at least two
  stable successful modes.
- **observed fact** — nevertheless, every one of the 640 natural initial-state
  families had at least one successful descendant. Thus all 40 tasks had
  low-p fraction `0`, all-fail count `0`, and undefined `t_div`.
- **interpretation** — high `rho` came from unequal but nonzero success rates
  across initial states, not from whole candidate families collapsing into a
  shared failure route.
- **interpretation** — successful multimodality showed that the policy could
  realize distinct successful behaviors, but it did not show that injecting
  diversity at one control step would recover a collapsed failure family.
- **observed fact** — the only non-ceiling E6 task, `libero_10/08`, had
  `p_bar=0.5` and `rho=6.867`, but only one stable successful mode and no
  all-fail family. It therefore isolated outcome heterogeneity without the
  proposed bottleneck structure.

## Why this is a valid negative rather than pipeline failure

- **observed fact** — the frozen positive control passed the same divergence,
  action-split, mode, low-p, and overdispersion machinery.
- **observed fact** — raw coverage, seeds, offsets, finite arrays, authority
  mapping, SHA manifests, and numeric ownership all validated before analysis.
- **interpretation** — the machinery was capable of detecting the preregistered
  pattern, but the pinned natural rollouts did not instantiate it. The causal
  reason for stopping is absence of the prerequisite population, not a failed
  intervention and not lack of successful behavior diversity.

## Boundary

- **observed fact** — the frozen decision is
  `NO_STAGE_R_PRECONDITION_ON_PINNED_PI05_LIBERO` and Phase-1 is unauthorized.
- **untested hypothesis** — other policies or harder task distributions may
  contain all-fail families; this run supplies no evidence for or against
  them.
- **interpretation** — no mechanism-specific improvement or degradation was
  measured because the split intervention was correctly never run. Any claim
  about bottleneck-local split efficacy would exceed the evidence.

