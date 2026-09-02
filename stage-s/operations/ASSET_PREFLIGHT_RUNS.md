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
| `r142-stage-s-a-assets-20260902-r10` | `dlceewcuiircdnhx` | `Stopped` after fail-closed runtime failure | The new tools venv reached `FIRST_WORK.json` and survived one idle-pool eviction under the same JobId. Three worker attempts then timed out while downloading the pinned 21.1 MB uv wheel from files.pythonhosted.org. `FAILED_ASSET_PREFLIGHT.json`, pod histories, and logs were preserved. The exact JobId was stopped before retry; no asset-completion marker or scientific rollout exists. |
| `r142-stage-s-a-assets-20260902-r11` | none | `preflight_failed_sealed` | Submission was rejected before CreateJob because an empty controller artifact directory had been pre-created. The empty directory was removed, but the run ID was sealed and never reused. |
| `r142-stage-s-a-assets-20260902-r12` | `dlcm738y2anr1vrd` | `Stopped` after fail-closed runtime failure | The Alibaba mirror repaired the pinned tools bootstrap and the persistent pip cache survived an idle eviction. The worker then proved that `huggingface.co` was network-unreachable before any checkpoint file was resolved. The failed marker and logs were preserved and the exact JobId was stopped; no completion marker or scientific rollout exists. |
| `r142-stage-s-a-assets-20260902-r13` | `dlc1qpllai3167ml` | `Stopped` after fail-closed runtime failure | The reachable mirror completed the pinned Evo checkpoint and all three large RoboTwin asset archives, and extraction reached the final archive. The payload then executed `popd` under uid/gid 2254 and attempted to return to the inherited unreadable `/root`, producing `Permission denied`. `FAILED_ASSET_PREFLIGHT.json`, downloaded caches, extracted assets, and pod logs were preserved; no completion manifest was written. The exact JobId was stopped. |
| `r142-stage-s-a-assets-20260902-r14` | `dlc1ielngql7vqaw` | `Stopped` after fail-closed runtime failure | The readable asset subshell succeeded and all downloads were reused, but `uv` searched the inherited unreadable `/root/uv.toml` after the uid drop. PAI restarted the task repeatedly under the same JobId; every attempt failed at the same environment bootstrap before completion. Logs and failed markers were preserved and the exact JobId was stopped. |
| `r142-stage-s-a-assets-20260902-r15` | `dlc11rl91mtxp2wq` | `Stopped` after `flash-attn` `Errno 18` | Controller readback is terminal `Stopped`. The exact log `/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_s/assets/r142-stage-s-a-assets-20260902-r15/pai_logs/master-final-before-stop.log` has SHA-256 `d5a9db6b3451a6ddcf1497eaf1973dbde6c405a0088f91a01fe05cf7e05dc4e6`; the failure marker `/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_s/assets/r142-stage-s-a-assets-20260902-r15/FAILED_ASSET_PREFLIGHT.json` has SHA-256 `e0f20f3df46f791e6f625964b6c53c7b0869a3f1f8913c8ce7f7bf2c40c28beb`, and `FIRST_WORK.json` has SHA-256 `692b6b73fbeb0847fbf28d2cb70c751d7ede031bf6ff60e7c671ad82b8080107`. No `COMPLETED_ASSET_PREFLIGHT.json` or `SHA256SUMS` was written; this is an operational failure, not a scientific result. |
| `r142-stage-s-a-assets-20260902-r16` | no JobId | controller `REFUSED` before CreateJob; sealed | The required pre-created resume directory was missing, so the controller refused the request before CreateJob. No JobId or CPFS artifact was produced. The exact configured paths checked were `/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/pai_registry/r142_stage_s/assets/r142-stage-s-a-assets-20260902-r16/` and `/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_s/assets/r142-stage-s-a-assets-20260902-r16/`; both are absent and therefore have no file SHA to report. |
| `r142-stage-s-a-assets-20260902-r17` | `dlc17mybd6alknp3` | `Stopped` after infrastructure gate failure | Controller readback is terminal `Stopped`. The log proves the cross-filesystem repair itself succeeded: `flash-attn` built and installed successfully, then bootstrap failed with `ModuleNotFoundError: pkg_resources` under the new setuptools environment. Exact log `/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_s/assets/r142-stage-s-a-assets-20260902-r17/pai_logs/master-final-before-stop.log` has SHA-256 `89c97f91389e845c2d5df9de47166a7bfc6bdb8d9d0716bd4dfdc09edae4f3d9`; `FAILED_ASSET_PREFLIGHT.json` has SHA-256 `a49fffcd2c34221e2b46f2605edcafcb740932c97b7497ce07137a766b61e5af`; `FIRST_WORK.json` has SHA-256 `0b55dbc7294bbf00eca356e6cad1e9df84a2c5edab62905d749c03253a99cfef`. The registry-side path `/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/pai_registry/r142_stage_s/assets/r142-stage-s-a-assets-20260902-r17/` has no completion marker or integrity manifest. This is an infrastructure gate failure, not a scientific result. |

The payload does not run scientific rollouts. Its only purpose is to pin and verify the exact RoboTwin/Evo-1/CuRobo sources, the published checkpoint revision, required assets, independent environments, GPU/graphics visibility, and imports before any calibration or main-screen rollout.

## Flash-attn cross-filesystem repair

The asset launcher now installs only `flash-attn` with `--no-cache-dir` and
`PIP_NO_CACHE_DIR=1`. Its `TMPDIR` is the run-scoped directory
`/mnt/cpfs/zbl-cpfs-new/USERS/leon/cache/r142_stage_s/pip/flash-attn-tmp/<run_id>`;
the launcher verifies that this directory and the persistent pip-cache parent
have the same filesystem device before the build. This addresses the observed
pip wheel-cache `Errno 18` rename without changing `HOME` or any scientific
source/checkpoint/task contract. `COMPLETED_ASSET_PREFLIGHT.json` records the
cache policy and temporary-root evidence, and the normal completion
`SHA256SUMS` remains mandatory.

## Current recovery boundary

r15 is terminal `Stopped` after the cross-device wheel-build failure, r16 was
refused before CreateJob and is sealed, and r17 is terminal `Stopped` after
the `pkg_resources` infrastructure gate failure. Do not reuse any of these
run IDs or treat their partial directories as a
scientific lineage. A future attempt must use a fresh controller-approved run
ID outside the Beijing no-job windows (`09:30--09:40` and `19:30--19:40`,
Asia/Shanghai), retain the exact same artifact/cache roots across an idle
restart, and bind monitoring to the returned JobId. Qualification still
requires terminal PAI success, a complete `COMPLETED_ASSET_PREFLIGHT.json`,
and a passing asset-root `SHA256SUMS`; queueing, running, first-work, and
partial-cache markers remain incomplete.
