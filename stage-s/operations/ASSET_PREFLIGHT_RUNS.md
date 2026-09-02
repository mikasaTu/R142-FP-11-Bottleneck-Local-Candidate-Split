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
| `r142-stage-s-a-assets-20260902-r13` | pending submission | prepared | Retains the r12 package mirror and adds the reachable `hf-mirror.com` endpoint with explicit metadata/download timeouts for the exact immutable checkpoint revision. It reuses only the explicit persistent cache; completion still requires terminal PAI success plus `COMPLETED_ASSET_PREFLIGHT.json` and `SHA256SUMS` verification. |

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

## Safe r16 recovery recommendation

After the static checks and canonical registry validation pass, use a fresh
run ID `r142-stage-s-a-assets-20260902-r16`; do not reuse r13 or any stopped
run. Submit only outside the Beijing no-job windows (`09:30--09:40` and
`19:30--19:40`, Asia/Shanghai), retain the exact same artifact/cache roots on
every idle restart, and bind monitoring to the newly returned JobId. Treat
`FIRST_WORK.json`, `Running`, queueing, or a partial cache as incomplete. The
run qualifies only after terminal PAI success, a complete
`COMPLETED_ASSET_PREFLIGHT.json`, and a passing asset-root `SHA256SUMS`; if it
is preempted, resume the same run/artifact directory rather than creating a
new scientific lineage.
