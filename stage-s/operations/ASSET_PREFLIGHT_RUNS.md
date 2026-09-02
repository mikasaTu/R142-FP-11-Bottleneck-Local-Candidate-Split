# Stage-S asset preflight run ledger

This is operational evidence only. Queueing, `Running`, `FIRST_WORK.json`, or a partial download is not a scientific result and does not qualify substrate A.

| Run ID | Job ID | Outcome | Evidence / action |
|---|---|---|---|
| `r142-stage-s-a-assets-20260902-r4` | none | `preflight_failed_sealed` | Submission was rejected before CreateJob because a shared existing cache was declared under `output_mode=new`. The run ID was sealed and never reused. |
| `r142-stage-s-a-assets-20260902-r5` | none | `preflight_failed_sealed` | Submission was rejected before CreateJob because a resume write directory did not yet exist. The run ID was sealed and never reused. |
| `r142-stage-s-a-assets-20260902-r6` | `dlc76kft6inr2xbi` | `Stopped` after fail-closed runtime failure | Exact readback: robot resource `quota1ssrabud0bh`, 8 GPUs, 88 CPU, 1525 GiB memory/shared-memory, `AcceptQuotaOverSold`, graphics capability. `FIRST_WORK.json` proved uid/gid 2254 and 8 visible GPUs, but this is not completion. The payload then failed because Bash parsed `0930` as octal and the pre-existing tools environment lacked `huggingface_hub`. `FAILED_ASSET_PREFLIGHT.json` and pod logs were preserved; the exact JobId was stopped before retry. |
| `r142-stage-s-a-assets-20260902-r7` | `dlcjo3klmgje5jtb` | active as of 2026-09-02 19:52 CST | New unique run after fixing the time guard to string matching and adding a fail-closed `huggingface_hub==0.36.2` bootstrap. Completion still requires terminal PAI success plus `COMPLETED_ASSET_PREFLIGHT.json` and `SHA256SUMS` verification. |

The payload does not run scientific rollouts. Its only purpose is to pin and verify the exact RoboTwin/Evo-1/CuRobo sources, the published checkpoint revision, required assets, independent environments, GPU/graphics visibility, and imports before any calibration or main-screen rollout.
