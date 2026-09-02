# Stage-S substrate C operations

This document is the non-submitting runbook for the exact under-trained
pi05-LIBERO lineage. It is intended for the PAI orchestrator after the parent
agent performs the external credential, mount, UID/GID, and resource readback.

## Frozen paths and lineage

```text
OPENPI=/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/QPILOTS-r16p15-stage1-task64-20260812/third_party/openpi
BASE_JAX=/mnt/cpfs/zbl-cpfs-new/open_data/r142_stage_s/pi05_base
BASE_PT=/mnt/cpfs/zbl-cpfs-new/Models/r142_stage_s/pi05_base_pytorch
ASSETS=/mnt/cpfs/zbl-cpfs-new/USERS/leon/cache/openpi/r16p15/openpi-assets/checkpoints
CKPT=/mnt/cpfs/zbl-cpfs-new/CKPT/leon/r142_stage_s_c
LOG=/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_s/c
```

`OPENPI` must report commit
`54cbaee6ae0c010a1ed431871cdaa8f4684ac709` and contain the three audited
source files. `BASE_JAX` must have a completed 29-object manifest and
`BASE_DOWNLOAD_COMPLETED.json`; `BASE_PT` must have a completed conversion
marker from the same OpenPI commit. The base is the public JAX `pi05_base`,
not any community full-SFT actor.

## Asset and conversion commands

Write/read the checked-in manifest or verify the live public listing:

```text
PYTHONPATH=src python scripts/stage_s_libero_c_assets.py manifest \
  --output stage-s/C_PI05_BASE_GCS_MANIFEST.json --live
```

Download is a foreground, resumable operation; it must not be run in either
Beijing blackout window:

```text
PYTHONPATH=src python scripts/stage_s_libero_c_assets.py download \
  --output-root "$BASE_JAX" \
  --manifest stage-s/C_PI05_BASE_GCS_MANIFEST.json
```

Convert only after the base audit passes:

```text
PYTHONPATH=src python scripts/stage_s_libero_c_assets.py convert \
  --openpi-root "$OPENPI" --base-jax-root "$BASE_JAX" \
  --base-pytorch-root "$BASE_PT" --precision bfloat16
```

To hand the parent orchestrator an auditable, non-submitting chain and PAI
payload, render both from the same paths:

```text
PYTHONPATH=src python scripts/stage_s_libero_c_payload.py \
  --run-id r142-stage-s-c-undertrained-20260902 \
  --openpi-root "$OPENPI" --base-jax-root "$BASE_JAX" \
  --base-pytorch-root "$BASE_PT" --checkpoint-base-dir "$CKPT" \
  --log-root "$LOG" --repo-root /mnt/cpfs/zbl-cpfs-new/USERS/leon/code/r142-stage-s \
  --assets-base-dir "$ASSETS" \
  --contract /mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_s/c/CHAIN.json \
  --payload /mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_s/c/PAI_PAYLOAD.json
```

The payload is deliberately marked `ready_for_submission=false` and
`no_pai_submit_performed=true`; external PAI submission is a parent-agent
operation after live resource and identity readback.

## Training command and resume

The wrapper performs blackout, source, base, data-asset, and conversion
preflight, then launches one 8-GPU `torchrun` worker. The worker imports the
pinned official trainer and adds only per-rank RNG sidecars to its checkpoint
I/O. The direct upstream command represented inside the contract is:

```text
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  "$OPENPI/scripts/train_pytorch.py" pi05_libero \
  --exp_name r142_stage_s_c_undertrained_seed42 \
  --checkpoint_base_dir "$CKPT" \
  --save_interval 1000 --num_train_steps 10001 --seed 42 \
  --keep_period 1000 --pytorch_weight_path "$BASE_PT" \
  --assets_base_dir "$ASSETS"
```

Use the checked-in worker wrapper so resume captures all rank RNG state:

```text
PYTHONPATH=src python scripts/stage_s_libero_c_train.py \
  --openpi-root "$OPENPI" --base-jax-root "$BASE_JAX" \
  --base-pytorch-root "$BASE_PT" --checkpoint-base-dir "$CKPT" \
  --log-root "$LOG" --assets-base-dir "$ASSETS"
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
* the wrapper fails closed inside either interval and writes no terminal
  completion marker;
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
the root `SHA256SUMS`. A missing component, source drift, partial download,
or failed checkpoint audit leaves no `COMPLETED_C_TRAINING.json`.

No scientific C success/gate claim is made until these artifacts are read
back from CPFS and the parent agent separately verifies the terminal PAI JobId.
