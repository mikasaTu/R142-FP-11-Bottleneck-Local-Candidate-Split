# Stage-2A preregistered falsification protocol

Status: frozen before descendant outcomes are inspected.

## Scientific question

Does the official learned `lerobot/diffusion_pusht` policy exhibit a natural,
generation-local recoverability bottleneck: are suffix branches resumed from
some real intermediate DDPM states substantially more able to change downstream
PushT progress/support than branches resumed elsewhere?

This stage does not use ForkPush2D, change PushT, train a detector, or enter a
VLA. Oracle outcomes are offline analysis only.

## Frozen source and environment

- LeRobot commit: `3c0a209f9fac4d2a57617e686a7f2a2309144ba2`
- checkpoint: `lerobot/diffusion_pusht`
- checkpoint revision: `84a7c23178445c6bbf7e1a884ff497017910f653`
- checkpoint weights SHA-256:
  `995d14d35db57d95c35ad9704c3d79c8612b7bc45f3877e5c46c2cdc516856a8`
- Python 3.10; torch 2.6.0; torchvision 0.21.0; diffusers 0.32.2
- gymnasium 0.29.1; gym-pusht 0.1.5; pygame 2.6.1; pymunk 6.11.0
- standard `gym_pusht/PushT-v0`, pixels plus agent position, 300 control steps
- Pymunk is the simulator. MuJoCo is not used and is recorded as `n/a`.

Checkpoint inference is unmodified: DDPM, 100 inference steps, horizon 16,
two observation steps, and eight executed action steps.

## Hard engineering gates

A. With instrumentation disabled, the wrapper delegates to the official policy
and is bit-identical to the frozen original for the same input and RNG state.

B. Passive tracing observes the official loop without modifying tensors and is
bit-identical to tracing off.

C. A saved real `z_s`, real scheduler location, and saved RNG continuation
reproduce the original suffix.

D. Replacing only the suffix RNG keeps the prefix and `z_s` fixed and changes a
non-zero fraction of final trajectories/actions.

E. Restoring a simulator snapshot by deterministic replay from the episode seed
and complete action prefix reproduces the full observable/native body state and
the result of the same action chunk. `PushTEnv.reset_to_state` alone is forbidden
because it omits velocities, angular velocity, RNG and collision cache.

Failure of C or D is `STAGE2A_SOURCE_LIMITATION`; failure of installation,
snapshot restoration or execution is `STAGE2A_PILOT_INCOMPLETE`. Neither is a
scientific result and no fake genealogy may be created.

## Baseline and natural snapshot frame

Run 50 fixed seeds `0..49`. Save every observation/action trajectory, native
reward/progress, success (`max coverage >= 0.95`), action chunk boundary, actual
NFE, timing, and RNG/state hashes.

Snapshot candidates are captured before looking at descendant outcomes at
control steps `{50, 100, 150, 200, 250}`. Select 24 by deterministic SHA-256
rank, targeting 8 successful/easy, 8 ambiguous/stalled, and 8 natural
failed/near-failure snapshots. If a stratum has fewer than eight candidates,
use every real candidate and report the shortfall; do not fabricate states.
The complete candidate frame, including unselected rows, is retained.

## Counterfactual tree

For every selected snapshot:

- K=8 independently seeded root trajectories;
- all 100 real denoising states retained for each root;
- 16 frozen, noise-level-stratified checkpoint indices:
  `{0, 7, 13, 20, 26, 33, 40, 46, 53, 59, 66, 73, 79, 86, 92, 99}`;
- M=8 independent, real scheduler suffixes per root/checkpoint;
- every descendant eight-action chunk starts from the same replay-restored
  simulator snapshot;
- calibration and held-out suffix seed streams are disjoint.

No true bottleneck step is defined.

## Outcomes and accounting

For every descendant record immediate coverage/progress change, block
`dx,dy,dtheta`, contacts, normalized/unnormalized action trajectory, pairwise
support diversity, root/checkpoint/suffix ancestry, seed/RNG hashes, remaining
NFE, wall clock, GPU time and peak VRAM.

Branchability is a vector, not a detector score:

- progress spread `q90-q10`;
- rescue probability;
- action/block-motion support diversity;
- held-out best-descendant gain over an independent no-branch control.

Report raw benefit, benefit/additional-NFE, and a genuinely fixed-NFE secondary
comparison. Discovery uses K=8,M=8. Fixed-NFE keeps K=8 and allocates suffixes
using `M_b=floor((B-K*T)/(K*(T-b)))`, reports slack, and clearly distinguishes
values with M>8 from the discovery tree.

Strategies: no-branch, always-early, three frozen uniform noise quantiles,
pre-seeded random, and cross-fitted oracle-local. Oracle-local selects a
checkpoint with calibration suffixes and evaluates it only with held-out
suffixes.

## Negative controls and cliff rule

Report all observed no-bottleneck, smooth-decay, fake-disagreement,
silent-bottleneck, and multiple-cliff cases. Classification rules use frozen
disagreement/curve criteria; examples are not hand-picked for the proposal.

A recoverability cliff requires a nearby-checkpoint primary branchability drop
of at least 0.10, paired-bootstrap 95% lower bound above zero, and two subsequent
checkpoints remaining low. Report prevalence over all preregistered natural
hard/failing snapshots.

## Conservative decision

Stage-2B is supported only if all evidence is complete, at least 30% and at
least six natural hard/failing snapshots show location-sensitive branchability,
cross-fitted oracle-local fixed-NFE normalized progress exceeds always-early,
uniform and random by at least 0.10 with paired 95% CI lower bound above zero,
always-early is not equivalent to oracle-local, and at least one natural cliff
exists.

Flat branchability, always-early equivalent to oracle, ineffective intermediate
branching, no meaningful prevalence, or no fixed-NFE advantage yields
`R142_FP11_CORE_HYPOTHESIS_WEAKENED`. Pilot completion never authorizes VLA or
automatic expansion.
