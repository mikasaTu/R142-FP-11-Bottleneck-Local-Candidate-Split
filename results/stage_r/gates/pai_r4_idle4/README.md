# Stage-R engineering gates: PAI 4-GPU idle run

- `observed fact`: PAI job `dlcm2n1u21b258la` finished with status
  `Succeeded` on resource `quotaewyznuc7b9l` with
  `OversoldType=AcceptQuotaOverSold`.
- `observed fact`: The live worker inventory contained four
  `NVIDIA A800-SXM4-80GB` devices. The gate workload intentionally executed on
  GPU 0; the subsequent Phase-0R workload uses all four devices as four ranks.
- `observed fact`: `COMPLETED_EVALUATION_RESULT.json` SHA-256 is
  `da15e15ba56daa74a399316c91cea3e3dd0017433b533b930cd2298ae14748f1`.
- `observed fact`: The complete artifact-root `SHA256SUMS` verification passed
  before this export was made.
- `observed fact`: The engineering decision is `ENGINEERING_GATES_PASSED`.
  E6 completed 64 closed-loop rollouts with success rate `0.46875`, progress
  quartiles `(0.5, 0.5, 1.0)`, and no ceiling pile.
- `controlled intervention`: E4 separately omitted simulator, action queue,
  current-policy-input buffer, and RNG restoration. Each omission produced the
  preregistered divergence while full restoration remained exact.
- `interpretation`: The snapshot/branch instrumentation is sufficiently
  faithful for the frozen Stage-R Phase-0R protocol. This is an engineering
  authorization, not evidence that bottleneck-local splitting improves task
  success.

The exported `e6_raw` NPZ is the complete 64-rollout E6 dataset. JSON metadata
binds it by SHA-256. The PAI artifact root additionally contains an archived
frozen source tree; that duplicate tree is omitted from Git because its source
commit is already versioned.
