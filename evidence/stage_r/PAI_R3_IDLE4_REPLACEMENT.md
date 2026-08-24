# Stage-R PAI replacement audit: 8-GPU Efficiency to 4-GPU idle

- `observed fact`: The superseded gate job was `dlc1r0r956yc0m7r`, run
  `r142-stage-r-gates-20260824-r3`, on the 8-GPU Efficiency carrier.
- `observed fact`: The exact PAI stop readback returned `Status=Stopped`,
  `ReasonCode=StoppedByUser`, before the job ever left `JobEnqueued`.
- `controlled intervention`: The replacement contract allocates one worker with
  4 A800 GPUs, 46 CPUs, 800 GiB memory and 800 GiB shared memory from resource
  `quotaewyznuc7b9l`, with `AcceptQuotaOverSold` required at live readback.
- `observed fact`: The first replacement validation was refused before job
  creation because the formal-idle controller requires the exact synchronous
  OnFailure max-50 AIMaster contract.
- `observed fact`: A second pre-create validation rejected the semantically
  invalid `expected_platform_restarts` field; idle preemption is possible, not
  an event count that can be promised in advance. The field was removed.
- `observed fact`: A third pre-create validation showed that the WRC-named
  four-card alias is workload-restricted. The controller-authorized generic
  four-card formal alias is `idle-a800-wallx-plug-native5-4gpu`; its name comes
  from its source contract, while its generic evidence profile permits this
  formal evaluation and still resolves to the same idle A800 quota.
- `observed fact`: A fourth pre-create validation required the alias's exact
  five-mount source contract. The added `x2robot_data` and `share` mounts are
  carrier-contract mounts only and are not read by the Stage-R launcher.
- `controlled intervention`: AIMaster synchronous restart is enabled with at
  most 50 platform restarts. E1-E5 replay deterministically; a completed E6
  shard is reused only when its metadata contract and data SHA-256 validate.
- `interpretation`: Queueing, `Running`, platform restart, and first work remain
  operational milestones only. Scientific completion still requires persisted
  `COMPLETED_EVALUATION_RESULT.json` and a valid `SHA256SUMS` manifest.

## Completed replacement and Phase-0R handoff

- `observed fact`: Replacement gate job `dlcm2n1u21b258la` reached PAI
  `Succeeded`; its persisted completion SHA-256 is
  `da15e15ba56daa74a399316c91cea3e3dd0017433b533b930cd2298ae14748f1`.
- `observed fact`: All gates E1-E6 and the positive control passed. E6 success
  was 30/64 (`0.46875`) and progress was not at ceiling.
- `observed fact`: Phase-0R run `r1` was sealed before PAI job creation because
  the output parent directory did not exist. It has no JobId and was not reused.
- `controlled intervention`: After creating only the exact output parent with
  owner `2254:2254` and mode `0700`, the identical validated payload was
  submitted as run `r2`.
- `observed fact`: Phase-0R PAI job `dlcj19s0anqjw4mo` was accepted on the same
  four-card idle contract. Its completion status is tracked separately and must
  not be inferred from submission or `Running`.
