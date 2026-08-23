# R142-FP-11 Stage-1 preregistered protocol

## Frozen claim

This stage tests only the following minimal mechanism:

> When a correlated candidate family collapses onto a shared failing prefix,
> can an unlabeled detector locate the earliest meaningful decision point and
> concentrate a fixed diversity budget there more effectively than uniform or
> random split locations?

It does not test a VLA, a learned policy, or deployment performance. Every
baseline and ablation is executed even if an intermediate gate fails. A failed
gate blocks later VLA work; it does not truncate this Stage-1 matrix.

## Benchmark

`ForkPush2D-v1` is a symmetric two-dimensional manipulation task. A point-like
object follows a shared stem, then must choose an upper or lower gate around a
central obstacle. Both modes have equal geometry and terminal reward. The true
bottleneck varies over action steps 4, 5, and 6. The environment truth is the
earliest step at which prefix-preserving positive and negative interventions
can reach the two successful gate basins.

Ordinary candidates share a family selector. In 82% of episodes it lies in the
central collision basin; increasing independent candidate count does not remove
that family-level correlation. Small candidate-specific uncertainty becomes
visible at the decision step, creating the disagreement onset used by the
detector.

## Candidate genealogy and label boundary

Each terminal record contains `candidate_id`, `parent_id`,
`generation_step`, `split_action_step`, full state/action/latent prefixes,
terminal score, final success, final mode, and failure reason. The proposed
detector receives only same-time state, action, and latent-prefix values. It
cannot receive terminal score, success/mode labels, failure reason, future
suffixes, or oracle truth.

## Methods and fixed compute

The fixed-budget methods each produce exactly 32 terminal candidates:

- B0: standard best-of-N.
- B1: uniform split at steps 2, 5, and 8; perturbation energy is distributed
  over those locations.
- B2: three random distinct split locations; same total perturbation energy.
- Proposed: eight scouts plus 24 children split at the earliest label-free
  disagreement spike; the scouts are included in the 32-candidate total.

Ablations are no detector, wrong location at `t*-2`, wrong location at `t*+2`,
correct location with a Gaussian random operator, full resampling at the correct
location, and B0 with 64 candidates (explicitly 2x compute).

## Metrics

For 400 paired evaluation seeds, report:

- success@N of the top terminal-score candidate;
- any-success@N and candidate success fraction;
- mode discovery rate (`successful modes / 2`);
- successful modes per sample;
- localization MAE, median error, and probability of error <= 1;
- both-mode discovery, mode entropy, and failure-reason counts.

Paired bootstrap confidence intervals use 10,000 replicates. Ten consecutive
40-seed blocks measure stability.

## Gates

The benchmark is valid only if at least 80% of B0 episodes satisfy both:

- disagreement before the true bottleneck divided by disagreement at it <= 0.35;
- candidate-family purity >= 0.80.

The proposed method stably exceeds B1 and B2 only if both comparisons have:

- success@N gain >= 0.05;
- paired 95% CI lower bound > 0;
- at least 7/10 winning blocks;
- mode discovery degradation no worse than 0.02.

Localization additionally requires median absolute error <= 1 and at least
80% of predictions within one action step. Failure of stable gain or
localization yields `IDEA_FAILED_DO_NOT_ENTER_VLA`. Passing yields only
`SUPPORTED_STAGE1_NO_VLA_CLAIM`.

## Mechanism reverse explanation

The location effect is isolated by correct-location versus `t*-2`/`t*+2`
interventions. Operator specificity is isolated by structured, Gaussian, and
full-resampling operators at the same correct location. The 64-sample B0
ablation tests whether gains are merely due to more samples. Only controlled
intervention differences are marked confirmed; correlational explanations are
marked hypotheses.
