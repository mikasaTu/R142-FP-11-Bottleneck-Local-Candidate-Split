# Stage-2A dev14 exact-runtime smoke report

## Scope

This report records engineering feasibility checks only. It is not the
Stage-2A scientific evaluation and cannot support or reject R142-FP-11.

The checks ran on one dev14 A800 using the frozen LeRobot commit
`3c0a209f9fac4d2a57617e686a7f2a2309144ba2`, checkpoint revision
`84a7c23178445c6bbf7e1a884ff497017910f653`, and unmodified checkpoint weights
with SHA-256
`995d14d35db57d95c35ad9704c3d79c8612b7bc45f3877e5c46c2cdc516856a8`.

## Runtime

- Python 3.10.20
- PyTorch 2.6.0+cu124
- torchvision 0.21.0+cu124
- diffusers 0.32.2
- gymnasium 0.29.1
- gym-pusht 0.1.5
- Pymunk 6.11.0

The complete checkpoint-file hashes and runtime manifest are in
`stage2a_dev14_exact_smoke/pinned_environment_manifest.json`.

## Results

All five pre-registered source-interface gates passed:

1. tracing disabled delegates to the original implementation;
2. passive tracing captures all 100 real DDPM steps without changing output;
3. the same intermediate latent and the same suffix RNG state reproduce the
   original final sample bit-for-bit;
4. the same latent with a new suffix RNG changes the final sample (maximum
   absolute difference `0.043298251926898956`);
5. exact episode seed plus full action-prefix replay restores the native PushT
   state with maximum absolute error `0.0`.

Machine-readable evidence is in
`stage2a_dev14_exact_smoke/resume_equivalence_tests.json` and
`stage2a_dev14_exact_smoke/simulator_snapshot_tests.json`.

## Evidence boundary

These results establish that the real learned-policy inference loop can be
traced and resumed, and that standard PushT snapshots can be reconstructed by
seed-plus-action-prefix replay. They do not show that a natural bottleneck
exists, that the detector localizes one, or that bottleneck-local splitting
improves any outcome. Those claims require the complete fixed-budget formal
evaluation.
