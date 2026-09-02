# Stage-S asset preflight run ledger

This is operational evidence only. Queueing, `Running`, `FIRST_WORK.json`, or a partial download is not a scientific result and does not qualify substrate A.

| Run ID | Job ID | Outcome | Evidence / action |
|---|---|---|---|
| `r142-stage-s-a-assets-20260902-r4` | none | `preflight_failed_sealed` | Submission was rejected before CreateJob because a shared existing cache was declared under `output_mode=new`. The run ID was sealed and never reused. |
| `r142-stage-s-a-assets-20260902-r5` | none | `preflight_failed_sealed` | Submission was rejected before CreateJob because a resume write directory did not yet exist. The run ID was sealed and never reused. |
| `r142-stage-s-a-assets-20260902-r6` | `dlc76kft6inr2xbi` | `Stopped` after fail-closed runtime failure | Exact readback: robot resource `quota1ssrabud0bh`, 8 GPUs, 88 CPU, 1525 GiB memory/shared-memory, `AcceptQuotaOverSold`, graphics capability. `FIRST_WORK.json` proved uid/gid 2254 and 8 visible GPUs, but this is not completion. The payload then failed because Bash parsed `0930` as octal and the pre-existing tools environment lacked `huggingface_hub`. `FAILED_ASSET_PREFLIGHT.json` and pod logs were preserved; the exact JobId was stopped before retry. |
| `r142-stage-s-a-assets-20260902-r7` | `dlcjo3klmgje5jtb` | `Stopped` after fail-closed runtime failure | The time guard was fixed, but the inherited tools environment had neither an importable `huggingface_hub` nor a working `pip` module. The failed marker and pod log were preserved and the exact JobId was stopped. |
| `r142-stage-s-a-assets-20260902-r8` | `dlczrhfd6ef6cbre` | `Stopped` after fail-closed bootstrap failure | Moving conda setup ahead of the failure trap caused the worker to exit before `FIRST_WORK.json` or a failed marker. Repeated non-retryable exit-code-1 events were preserved; no scientific work occurred. The exact JobId was stopped. |
| `r142-stage-s-a-assets-20260902-r9` | `dlcjem8lmdlzpoe2` | `Stopped` after fail-closed runtime failure | FIRST_WORK was persisted, then the payload established that conda was unavailable in the pinned image. The failed marker and logs were preserved; the exact JobId was stopped. |
| `r142-stage-s-a-assets-20260902-r10` | pending submission | prepared | Removes the conda assumption: system Python creates an isolated tools venv, pinned uv installs Python 3.10 under the Stage-S cache, and uv seeds independent RoboTwin/Evo environments. Completion still requires terminal PAI success plus `COMPLETED_ASSET_PREFLIGHT.json` and `SHA256SUMS` verification. |

The payload does not run scientific rollouts. Its only purpose is to pin and verify the exact RoboTwin/Evo-1/CuRobo sources, the published checkpoint revision, required assets, independent environments, GPU/graphics visibility, and imports before any calibration or main-screen rollout.
