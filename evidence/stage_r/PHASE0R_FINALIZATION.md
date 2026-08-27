# Phase-0R authoritative merge and finalization evidence

## Terminal inputs

| authority | exact JobId | exact PAI state | task indices |
|---|---|---|---:|
| parent | `dlcyuv28a0djtgxd` | running redundancy; only pre-frozen pairs used | 0--31 |
| shard A | `dlc9b8jreh4e8fy3` | `Succeeded` | 32--35 |
| shard B | `dlcap6ioa9w2ht2u` | `Succeeded` | 36--39 |

Both supplemental jobs passed their full root `SHA256SUMS`, exact rank-marker,
metadata/data SHA, 16-by-32 coverage, rollout-seed, finite-array, and owner
checks before merge. The parent job's later task indices were never eligible
for selection.

## Outcome-blind authoritative merge

Command implementation:
`scripts/stage_r_phase0r_authoritative_merge.py`.

Canonical output:

```text
/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_r/phase0r_merged/r142-stage-r-phase0r-authoritative-20260827
```

The merge validated all 40 authoritative pairs before hardlinking them into a
temporary directory and atomically publishing the result. It never read a
success value. Verification returned 40 NPZ files, 40 metadata files, owner
`2254:2254`, and a passing checksum manifest.

```text
AUTHORITY_MANIFEST.json  3d5a37ec8a7e2c0dfd0c808ad59553c43a13c846b90f99c1afaa3529a072469c
COMPLETED_PHASE0R_RAW.json 9255838afefec38d76f9de6bbac962ffc2a8538d283e30949eb4c67e5b8d4675
merge SHA256SUMS         0016527de4d82f8c430664f02cfd4634b09b12121e32912c85526a15a160219d
```

## Frozen analysis and final seal

The analyzer came from scientific source commit
`24423e8114ace80e6a76f22bee29992cea420cfc` and used its committed
`PHASE0R_THRESHOLDS.json`. It ran on the merged raw directory without code,
threshold, task, seed, budget, or authority changes.

```text
decision                 NO_STAGE_R_PRECONDITION_ON_PINNED_PI05_LIBERO
positive_control_pass    true
retained_tasks           []
checkpoint               CHECKPOINT_1_STOP
phase1_authorized        false
phase0r_summary.json     7da6f4751f64f08bdfa50e7f37ac1dd1a2f80b7f3f36862f12e7e95ef475c299
COMPLETED_PHASE0R.json   61f926ad2408e314b8cbd41dcc23a4f59cfe48c5102f8847df1074733fe168eb
top-level completion     9daff7a544ebb7b1c4e3f6fbf10e38b124c067aa6c56a417f822a028dd5fb10a
final SHA256SUMS         2d887069054098d569c4f42260d419f7bcc9377c41c51d46fb14a7ef8139f924
```

The final 87-file checksum manifest passed. The workflow stopped at
Checkpoint 1; no Phase-1 code, rollout, threshold, or precomputation was run.

## Verification commands

- local and dev14 test selection:
  `PYTHONPATH=src python -m pytest -q tests/test_stage_r_phase0r_parallel_shards.py tests/test_stage_r.py`
- result: `6 passed`
- `python -m py_compile` passed for both operational merge/finalize scripts.
- both merge and final root checksum manifests passed complete verification.

## Publication readback

- GitHub main result commit:
  `c0ada3144e2055d2c75b65704e5c78db284f99b6`.
- A fresh clone from GitHub reproduced all eight bundle hashes and the frozen
  decision/checkpoint/Phase-1 fields.
- Feishu `step3/实验报告` doc token:
  `UNtUdgZYfoCMhGxZ3CQcOo3RnAg`.
- Feishu update result: success, revision 2, no warnings.
- Outline readback returned all ten report sections. Keyword readback returned
  the exact decision, `CHECKPOINT_1_STOP`, GitHub result commit, authority
  manifest SHA, and the `libero_10/08` failure-case row.
