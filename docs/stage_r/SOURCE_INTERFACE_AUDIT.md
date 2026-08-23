# Stage-R source/interface audit (outcome blind)

## Accepted real-policy path

The only currently verified source contract is official OpenPI pi0.5-LIBERO at
OpenPI commit `54cbaee6ae0c010a1ed431871cdaa8f4684ac709`, loaded through the clean
read-only QPILOTS integration at
`eacf47b981e3b22357f8a74902f8dad8cfcfa375`. The integration exposes explicit
policy noise, preserves official input/output transforms, and has a real
MuJoCo/robosuite snapshot implementation. Stage-R adds trajectory-axis runner
state (history, action queue and RNG counters) but may not use latent/z distance
as a detector.

The official evaluator executes five actions from each ten-action prediction;
Stage-R preserves that contract. It uses the released task-specific instruction,
two 256x256 cameras rotated exactly as in official evaluation, the 8-D EEF and
gripper state, and the checkpoint's pinned action normalization.

## RoboTwin source limitation

The audited RoboTwin checkout is Git
`05600234df39367424fcb8036533b5e111d2a0aa` and has unrelated untracked files
under `stage2_robotwin/stage2e/`, which Stage-R does not touch. It contains
scripted/oracle SAPIEN task infrastructure, not a runnable adapter for the same
pi0.5-LIBERO checkpoint. Observation images, proprioception, action semantics,
normalization and checkpoint lineage therefore fail the single-policy contract.

Using its scripted policy would change the scientific object; pretending the
LIBERO checkpoint can act in RoboTwin would fabricate an interface. RoboTwin is
therefore retained in the full candidate table with the preregistered label
`SOURCE_LIMITATION_UNVERIFIABLE`, zero rollouts and no scientific conclusion.

## Allowed reuse and forbidden reuse

Allowed: snapshot mechanics, raw genealogy schema, PAI launcher/registry
patterns and SHA-256 completion manifests. Forbidden: Stage-1's hard-coded
`rollout()` location effect, Stage-2A denoising-axis branches, latent/z pairwise
detectors, short-horizon success proxies, and any prior Stage-1/2 outcome as
support for Stage-R.
