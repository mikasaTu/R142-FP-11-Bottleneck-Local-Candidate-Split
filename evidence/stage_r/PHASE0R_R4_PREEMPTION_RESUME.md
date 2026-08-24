# Phase-0R r4 idle preemption and same-artifact resume

- `observed fact`: The formal Phase-0R job is `dlcyuv28a0djtgxd`, run
  `r142-stage-r-phase0r-20260824-r4-idle4`, on resource
  `quotaewyznuc7b9l` with `OversoldType=AcceptQuotaOverSold`.
- `observed fact`: The original master pod UID
  `a7ca3db5-8b4d-4a0e-99a4-35866866f661` ran from
  `2026-08-24T04:40:32Z` until `2026-08-24T09:47:54Z` and is recorded by PAI
  as `Failed`. The AIMaster remained live.
- `observed fact`: AIMaster created replacement master pod UID
  `51314019-b159-41e4-9280-805dace31a36`, created at
  `2026-08-24T09:48:12Z` and running from `2026-08-24T09:49:27Z`.
- `observed fact`: Before preemption, four complete task artifacts existed:
  `libero_spatial_task00` through `libero_spatial_task03`. Each contains 512
  rollouts; all four NPZ SHA-256 values still match their metadata after the
  replacement pod started.
- `observed fact`: The replacement pod reused source commit
  `24423e8114ace80e6a76f22bee29992cea420cfc` and the same persistent artifact
  directory. The four rank logs were recreated by the replacement payload and
  contain no `Traceback`, `AssertionError`, `RuntimeError`, or launcher fatal
  marker at this checkpoint.
- `controlled intervention`: No scientific configuration, task, seed,
  candidate budget, threshold, or completed task artifact was changed. The
  pre-registered per-task resume path skipped only SHA-valid completed task
  pairs and restarted each rank at its next assigned task.
- `interpretation`: This is positive operational evidence for the application
  resume contract, not scientific evidence for the bottleneck-local mechanism.
  The job remains incomplete until all 40 tasks, four rank-complete markers,
  `COMPLETED_EVALUATION_RESULT.json`, root `SHA256SUMS`, and PAI `Succeeded`
  agree.
