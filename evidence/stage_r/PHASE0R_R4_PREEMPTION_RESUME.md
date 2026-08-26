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

## Second preemption sequence (2026-08-25)

- `observed fact`: Replacement master pod UID
  `51314019-b159-41e4-9280-805dace31a36` ran until
  `2026-08-25T13:47:46Z` and is recorded by PAI as `Failed`; the AIMaster UID
  `ef751241-4569-4882-9de3-a4f9c4e458ef` remained `Running`.
- `observed fact`: AIMaster then created master pod UID
  `61a87db3-953e-487b-937c-79d4ee61deef`, which ran from
  `2026-08-25T13:49:20Z` until `2026-08-25T13:56:42Z` and is also recorded as
  `Failed`.
- `observed fact`: AIMaster created the current replacement master pod UID
  `46574d95-e00d-414d-b046-99e79404a16d`, created at
  `2026-08-25T13:57:00Z` and `Running` from `2026-08-25T13:58:05Z` at the
  evidence checkpoint.
- `observed fact`: All 32 task artifacts completed before this sequence remain
  in the same persistent directory. They contain 16,384 rollouts in total;
  every NPZ SHA-256 still matches `data_sha256` in its task metadata, with no
  incomplete or mismatched pair observed.
- `observed fact`: The current replacement recreated all four rank logs and
  all four contain no `Traceback`, `fatal`, segmentation-fault, OOM, CUDA-error,
  or uncaught-exception marker at the evidence checkpoint.
- `controlled intervention`: No scientific task, seed, candidate budget,
  threshold, metric protocol, source snapshot, or completed artifact was
  changed. PAI automatic fault tolerance resumed the unchanged foreground
  payload in the same artifact directory.
- `interpretation`: The consecutive eviction sequence is additional positive
  operational evidence for same-directory application resume. It is not a
  completed evaluation and not evidence for or against the scientific
  mechanism. Analysis remains sealed until all completion gates agree.

## Third rollback sequence (2026-08-25)

- `observed fact`: At the 2026-08-25T19:27Z checkpoint, exact GetJob returned
  `Queuing`, reason `JobEnqueued`, message `Rollback to queue`, for the same
  JobId `dlcyuv28a0djtgxd`.
- `observed fact`: The persistent artifact directory retained inode
  `1183269075169`, owner `2254:2254`, the 32 pre-registered authoritative task
  pairs, and three later redundant pairs (`libero_10_task03`--`task05`) with no
  observed SHA mismatch.
- `observed fact`: Exact-JobId OpenAPI evidence for the parent had already
  sealed `UseOversoldResource=true`; this rollback is therefore consistent
  with idle-resource reclamation.
- `controlled intervention`: No replacement job was submitted and no
  scientific or authoritative-source mapping changed. AIMaster remains
  responsible for restoring the same run and directory.
- `interpretation`: The parent indices 32--39 are pre-registered redundancy,
  so this rollback does not change the authoritative Phase-0R merge. `Queuing`
  is not recovery completion; a subsequent `Running` readback and persisted
  work are still required.

### Third same-Job recovery checkpoint

At the 2026-08-25T20:00Z checkpoint, exact GetJob returned `Running` for the
same parent JobId. The artifact directory retained inode `1183269075169`, and
all four rank logs were recreated with newer mtimes and no detected traceback,
assertion, CUDA-OOM, or launcher-fatal marker. This verifies operational
rescheduling only. The parent still has no global completion marker or root
SHA256SUMS, and its post-index-31 task pairs remain non-authoritative
redundancy.

## Fourth preemption and same-Job recovery (2026-08-26)

- `observed fact`: Parent replacement master pod UID
  `bb402e17-21ee-49dd-8fb8-d37e007d3d5e` ran from
  `2026-08-25T19:50:10Z` until `2026-08-26T05:24:42Z` and is recorded by
  PAI as `Failed`; the same AIMaster remained live.
- `observed fact`: AIMaster created replacement master pod UID
  `b02400ec-dd85-4243-86eb-2a6fe0e8209b`, created at
  `2026-08-26T05:24:59Z` and `Running` from `2026-08-26T05:26:06Z` under the
  exact same JobId and run ID.
- `observed fact`: The persistent artifact directory still has inode
  `1183269075169`, owner `2254:2254`, and mode `0700`. The 32 frozen
  authoritative pairs remain in place. Parent-side `libero_10_task02`,
  `task03`--`task05`, `task07`, and `task09` are later redundancy and are not
  promoted over the pre-registered shard-A/B authority mapping.
- `observed fact`: The current parent rank logs contain no detected traceback,
  assertion, CUDA-OOM, or launcher-fatal marker at this checkpoint.
- `controlled intervention`: No new parent job was submitted and no task,
  seed, candidate budget, threshold, statistical protocol, or source mapping
  changed. Resume occurred through AIMaster in the same artifact directory.
- `interpretation`: This is another idle-resource preemption/recovery event,
  not scientific completion. The parent remains `Running`; shard B is still
  independently missing authoritative `libero_10_task08`, its rank-6 marker,
  subset completion records, and terminal `Succeeded`.
