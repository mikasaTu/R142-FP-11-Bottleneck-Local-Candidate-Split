# Stage-S substrate C operations

This document is the non-submitting runbook for the exact under-trained
pi05-LIBERO lineage. It is intended for the PAI orchestrator after the parent
agent performs the external credential, mount, UID/GID, and resource readback.

## Frozen paths and lineage

```text
RUNTIME_PROJECT=/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/r142-stage-s-c-runtime-20260903
STAGE_S_C_PROJECT_DIR=$RUNTIME_PROJECT
STAGE_S_SOURCE_COMMIT=7575da585be31eb369a604d90048b338bbbf2c92
STAGE_S_C_PAYLOAD_SHA256=85efff1581bd4428d5b64700bf677efeb53eed412a137c71519deb7d5d078da6
PAI_CANARY_RUN_ID=<registry-injected run id>
OPENPI=/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/QPILOTS-r16p15-stage1-task64-20260812/third_party/openpi
OPENPI_PYTHON=/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/openpi_py311/bin/python
BASE_JAX=/mnt/cpfs/zbl-cpfs-new/USERS/leon/cache/r142_stage_s/pi05_base
BASE_PT=/mnt/cpfs/zbl-cpfs-new/USERS/leon/cache/r142_stage_s/pi05_base_pytorch
ASSETS=/mnt/cpfs/zbl-cpfs-new/USERS/leon/cache/openpi/r16p15/openpi-assets/checkpoints
CKPT=/mnt/cpfs/zbl-cpfs-new/CKPT/leon/r142_stage_s_c
LOG=/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_s/c
STATUS=/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_s/c_status/<RUN_ID>
```

`OPENPI` must report commit
`54cbaee6ae0c010a1ed431871cdaa8f4684ac709` and contain the three audited
source files. `BASE_JAX` must have a completed 29-object manifest and
`BASE_DOWNLOAD_COMPLETED.json`; `BASE_PT` must have a completed conversion
marker from the same OpenPI commit. The base is the public JAX `pi05_base`,
not any community full-SFT actor.

The idle resource provenance is readback-only: both
`contract_source_job_id` and `resource_source_job_id` are
`dlckjz66iwcv38gw`, with `source_role=readback_reference`,
`submission_method=cli_create`, and `pai_clone_performed=false`. The target
contract separately requires `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`.
The clean-registry readback currently enforces `runtime.pod_env` equality with
that one-key `required_pod_env`; consequently the requested additional
`STAGE_S_SOURCE_COMMIT`/project/payload keys are recorded in the C manifest but
are refused by the unmodified registry validator. A graphics-only control
manifest validates (`valid=true`); the source-bound manifest remains
fail-closed until the registry controller adds an allowlisted task-environment
injection path. This is an admission-layer blocker, not a C training result.

## Asset and conversion commands

Write/read the checked-in manifest or verify the live public listing:

```text
PYTHONPATH="$PWD/src:$OPENPI/src" "$OPENPI_PYTHON" scripts/stage_s_libero_c_assets.py manifest \
  --output stage-s/C_PI05_BASE_GCS_MANIFEST.json --live
```

Download is a foreground, resumable operation; it must not be run in either
Beijing blackout window:

```text
PYTHONPATH="$PWD/src:$OPENPI/src" "$OPENPI_PYTHON" scripts/stage_s_libero_c_assets.py download \
  --output-root "$BASE_JAX" \
  --manifest stage-s/C_PI05_BASE_GCS_MANIFEST.json
```

Convert only after the base audit passes, using the pinned OpenPI Python:

```text
PYTHONPATH="$PWD/src:$OPENPI/src" "$OPENPI_PYTHON" scripts/stage_s_libero_c_assets.py convert \
  --openpi-root "$OPENPI" --base-jax-root "$BASE_JAX" \
  --base-pytorch-root "$BASE_PT" --python "$OPENPI_PYTHON" --precision bfloat16
```

To hand the parent orchestrator an auditable, non-submitting chain and PAI
payload, render both from the same paths:

```text
PYTHONPATH="$PWD/src:$OPENPI/src" "$OPENPI_PYTHON" scripts/stage_s_libero_c_payload.py \
  --run-id r142-stage-s-c-undertrained-20260902 \
  --openpi-root "$OPENPI" --base-jax-root "$BASE_JAX" \
  --base-pytorch-root "$BASE_PT" --checkpoint-base-dir "$CKPT" \
  --log-root "$LOG" --repo-root "$RUNTIME_PROJECT" \
  --assets-base-dir "$ASSETS" \
  --contract /mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_s/c/CHAIN.json \
  --payload /mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_s/c/PAI_PAYLOAD.json
```

The payload is deliberately marked `ready_for_submission=false` and
`no_pai_submit_performed=true`; external PAI submission is a parent-agent
operation after live resource and identity readback.

## Training command and resume

The registry payload `scripts/stage_s_c_undertrained_pai.sh` runs asset
download, conversion, and training sequentially in one foreground job. Each
stage has an atomic `COMPLETED_*.json` or `FAILED_*.json` status marker under
`STATUS`, including a canonical `payload_sha256` and an evidence SHA when an
evidence file exists. The training wrapper performs blackout, source, base,
data-asset, and conversion preflight, then launches one 8-GPU worker with the
pinned Python. The worker imports the pinned official trainer, adds per-rank
RNG sidecars, and wraps the loader to restore the exact data cursor. The
direct upstream command represented inside the contract is:

```text
"$OPENPI_PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
  "$OPENPI/scripts/train_pytorch.py" pi05_libero \
  --exp_name r142_stage_s_c_undertrained_seed42 \
  --checkpoint_base_dir "$CKPT" \
  --save_interval 1000 --num_train_steps 10001 --seed 42 \
  --keep_period 1000 --num_workers 0 --pytorch_weight_path "$BASE_PT" \
  --assets_base_dir "$ASSETS"
```

Use the checked-in worker wrapper so resume captures all rank RNG state:

```text
PYTHONPATH="$PWD/src:$OPENPI/src" "$OPENPI_PYTHON" scripts/stage_s_libero_c_train.py \
  --openpi-root "$OPENPI" --base-jax-root "$BASE_JAX" \
  --base-pytorch-root "$BASE_PT" --checkpoint-base-dir "$CKPT" \
  --log-root "$LOG" --assets-base-dir "$ASSETS" --python "$OPENPI_PYTHON"
```

After a spot interruption, retain the same `CKPT`, `LOG`, and run ID and
rerun with `--resume`. The wrapper refuses a non-resume start over an existing
numeric checkpoint tree. It also refuses resume if any selected checkpoint
lacks the per-rank RNG sidecars.

## Resource and blackout contract

* robot idle pool, one worker;
* 8 A800 GPUs, 88 CPU cores, 1525 GiB memory/shared-memory ceiling;
* runtime UID/GID 2254:2254, checked by the parent PAI orchestrator;
* Sync/OnFailure platform recovery may restart up to 50 times, but every
  incarnation must use the same `CKPT`/`LOG` directory and the wrapper's
  application-level resume contract;
* no job submission or resume mutation in `09:30–09:40` or `19:30–19:40`
  Asia/Shanghai;
* the wrapper fails closed inside either interval and writes a hashed
  `FAILED_BLACKOUT.json` status marker, never a completion marker;
* spot retries must preserve the same CPFS output directory and exact run
  lineage, with no checkpoint deletion/replacement by a new base.

## Terminal evidence

Every periodic checkpoint must contain `model.safetensors`, `optimizer.pt`,
`metadata.pt`, and `rng_state.rank{0..7}.pt`. The retained C set is exactly:

```text
CKPT/pi05_libero/r142_stage_s_c_undertrained_seed42/{1000,3000,6000,10000}/
```

The wrapper writes `TRAINING_TERMINAL.json` with global step 10001 only after
the official process exits successfully. `finalize_training` then audits all
four complete native checkpoints and writes `COMPLETED_C_TRAINING.json` plus
two cwd-relative manifests: `$CHECKPOINT_BASE/SHA256SUMS` covers only the
checkpoint bundle and `$LOG/SHA256SUMS` covers only the log bundle. Verify
with `cd "$CHECKPOINT_BASE" && sha256sum -c SHA256SUMS` and the analogous
command in `$LOG`; no mixed-root entries are accepted. A missing component,
source drift, partial download, or failed checkpoint audit writes a hashed
`FAILED_C_TRAINING.json` and leaves no completion marker.

The pinned trainer's loop is audited at source. Its native loader is an
infinite iterator and does not expose `len`/`DistributedSampler.set_epoch` at
the wrapper boundary. The C worker therefore exposes one finite epoch,
forwards `set_epoch(global_step // epoch_length)`, and skips exactly
`global_step % epoch_length` batches on the first resumed iterator. It also
freezes `--num_workers 0`; if epoch length, sampler epoch, or checkpoint
metadata cannot be proven, resume fails closed rather than being labelled
full-state.

For registry execution, submit only the checked-in
`configs/pai/stage_s_c_undertrained.json` (schema v2) through the canonical
`pai-job-registry`; it binds the robot-idle alias/id/quota, UID/GID, CPFS
write paths, pinned Python, sequential stage payload, and application
autoresume. Before execution the controller must bind
`STAGE_S_C_PROJECT_DIR`, `STAGE_S_SOURCE_COMMIT`, and
`STAGE_S_C_PAYLOAD_SHA256` exactly as recorded in the registry config. The
launcher checks the source Git HEAD and cleanliness, the QPILOTS parent
commit, the OpenPI commit, and both the registry and invoked-payload SHA;
missing or mismatched values fail closed. The payload itself is
non-interactive and never changes `HOME`.

No scientific C success/gate claim is made until these artifacts are read
back from CPFS and the parent agent separately verifies the terminal PAI JobId.
