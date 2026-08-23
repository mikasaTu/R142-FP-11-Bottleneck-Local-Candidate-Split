# Stage-2A mechanism reverse explanation

Date: 2026-08-24

Decision: `R142_FP11_CORE_HYPOTHESIS_WEAKENED`

This document applies the code-first mechanism audit to the completed code and
measurements. It does not generate a new idea and does not recommend a new
detector.

## 1. Code facts

1. `generate_roots` samples eight full `(horizon=16, action_dim=2)` noisy action
   tensors and runs the official 100-step DDPM loop. A checkpoint stores the
   real tensor immediately before the official UNet/scheduler call.
2. `resume_suffix` copies the same saved `z_s`, keeps the conditioning and
   simulator prefix fixed, and changes only the suffix RNG. Checkpoint 0 has 100
   remaining model evaluations; checkpoint 99 has one.
3. Only the official policy slice
   `[n_obs_steps-1 : n_obs_steps-1+n_action_steps]`, i.e. eight actions, is
   unnormalized and executed. Raw disagreement is measured on the complete
   16-action latent tensor, whereas outcome support is measured on the executed
   eight-action chunk.
4. Fixed-NFE cost is
   `K*T + K*M_b*(T-b)`. Later checkpoints can therefore produce more terminal
   suffixes under the same cap, but each suffix has fewer remaining denoising
   transitions.
5. Cross-fitted oracle-local chooses a checkpoint only with calibration suffixes
   and is scored with disjoint held-out suffixes. It is an offline upper bound,
   not a deployable detector.

## 2. Observed facts

- Across 24 descendant-blind natural snapshots, the maximum location range of
  held-out progress spread was 0.040996; the preregistered threshold was 0.10.
  No snapshot was location-sensitive and none of eight hard snapshots had a
  valid recoverability cliff.
- Median root `z_s` disagreement fell from 1.4001 at checkpoint 0 to 0.9588 at
  checkpoint 53 and 0.0379 at checkpoint 99. Median held-out progress spread
  remained 0.00177, 0.00178 and 0.00194 respectively.
- Pooled correlation was −0.0031 between `z_s` disagreement and held-out
  progress spread, −0.0913 between executed-action support diversity and
  progress spread, and 0.0247 between `z_s` disagreement and executed-action
  support diversity.
- Calibration and held-out progress-spread curves were highly reproducible
  (`r=0.9971`); their best-gain checkpoint agreed exactly for 19/24 snapshots.
  Thus the negative result is not explained by a failed cross-fit alone.
- At the 7200 sample-NFE cap, mean gain over no-branch was 0.009655 for
  always-early, 0.012011 for random, 0.012364 for uniform-three-quantiles, and
  0.011913 for cross-fitted oracle-local.
- Oracle-local minus always-early was +0.002258 (paired bootstrap 95% CI
  [0.000345, 0.004919]), far below the required +0.10. Oracle-local minus
  random was −0.000099 (CI [−0.002204, 0.002766]); minus uniform was −0.000451
  (CI [−0.001419, 0.000361]).
- The representative eventual-continuation subset produced 7/12, 8/12 and 6/12
  successes at checkpoints 0, 53 and 99. Their Wilson intervals overlap widely.
  This 36-case subset is descriptive, descendant-blind and not the fixed-NFE
  gate.

## 3. Controlled mechanism interpretation

### Why oracle-local is slightly above always-early

This small positive difference is consistent with a compute-allocation effect,
not evidence for a bottleneck. Always-early spends all 7200 NFE on 64 expensive
100-step suffixes. Oracle-local used a mean of 7090 NFE and generated 97.7
suffixes because selected checkpoints often had fewer remaining steps. The
calibration curve was stable enough to avoid arbitrary locations, and the
additional order-statistic opportunities can raise the per-root maximum by a
small amount. However, the effect is only 0.0023 and there is no local cliff.

### Why oracle-local does not beat random or uniform

Once location carries almost no outcome signal, random and uniform allocations
lose little by choosing the “wrong” checkpoint. They can also generate more
terminal suffixes under the same cap: 748.3 on average for random (highly skewed
when a very late checkpoint is sampled) and exactly 216 for uniform, versus
97.7 for oracle-local. Uniform additionally hedges across three frozen regions.
The observed oracle deficits are tiny and their confidence intervals include
zero, so the correct interpretation is equivalence at the resolution of this
pilot, not a reliable harmful oracle mechanism.

### Why latent collapse does not become an outcome bottleneck

The DDPM scheduler and denoiser contract the full latent substantially, but the
executed eight-action support remains similar across checkpoints. Part of raw
latent variation can reside in the unexecuted portion of the 16-action horizon
or in directions to which short-horizon PushT coverage is insensitive.
Moreover, later branching trades away stochastic transitions while receiving
more suffix draws. These two effects largely cancel at the outcome level. The
experiment therefore observes natural diffusion-state collapse without the
specific generation-local recoverability collapse required by R142-FP-11.

### Why some curves are high but still not location-sensitive

Individual snapshots can have large descendant spread at every checkpoint
(the maximum single-checkpoint spread is 0.1257), but the largest within-snapshot
change across locations is only 0.0410. This distinguishes state-dependent task
uncertainty from a generation-location bottleneck: some observations are more
branchable than others, yet no particular denoising region uniquely unlocks
that branchability.

## 4. Evidence boundary

Confirmed code facts and measured outcomes support the narrow conclusion that
this official PushT DiffusionPolicy does not exhibit the preregistered,
generation-local mechanism at meaningful prevalence or effect size. Whether a
different policy family, task, larger natural-state sample, or VLA has another
structure is untested. Per the frozen decision rule, Stage-2B and VLA expansion
must not proceed from this result.
