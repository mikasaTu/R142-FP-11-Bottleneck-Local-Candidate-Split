# Stage-S C under-trained training run ledger

This ledger records the C training pipeline separately from the later C
pooled calibration. It is operational evidence only. A queueing/running state,
`FIRST_WORK`, stage-level completion, a partial checkpoint directory, or a
successful download/conversion stage is not a scientific training result.

## Controller and CPFS readback

| Run ID | JobId | State | Evidence / disposition |
|---|---|---|---|
| `r142-stage-s-c-undertrained-20260903-r3` | `dlccjv5e2snfm9m4` | terminal `Stopped`; training failed | Base download, conversion, and preflight completed, but training failed when PAI could not access the pinned Hugging Face dataset reference. The preserved master log `/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_s/c_status/r142-stage-s-c-undertrained-20260903-r3/pai_logs/master-final-before-stop.log` has SHA-256 `f5189a93c3171c59144fbbbb3a66560580ce6a337c16e1a2df0033b5fe8841d9`; it records the missing `.../xdg-cache/huggingface/lerobot/physical-intelligence/libero/meta/info.json`, followed by `OSError: [Errno 101] Network is unreachable` / Hugging Face connection failures. This is an environment/data-access failure, not a scientific negative result. |

The controller JobId above is the authoritative PAI lineage. The run-scoped
status marker records `job_id: null` because the status writer did not receive
the controller identifier; this does not turn the stopped run into a
successful run.

## Stage markers and hashes

All paths below were read from CPFS. The status root is:

```text
/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_s/c_status/
  r142-stage-s-c-undertrained-20260903-r3/
```

The exact marker hashes are:

| Marker | SHA-256 |
|---|---|
| `COMPLETED_preflight.json` | `87c88c010fb0c5d07dc82e973353a82da041a103828c529cc61c5507d77ce8ea` |
| `COMPLETED_base_download.json` | `63c440864113060d41c060af5a4148dcd8db5574c1afc521bf3074dc6e04187` |
| `COMPLETED_conversion.json` | `2df7806ba5e42cb4ae58422ebb25cc67fea9e39f4dea21d2bf2712ec4737fbc6` |
| `RUNTIME_IDENTITY.json` | `4346f584bf485dc0cf1ac78f7a98c707d56ff5a26b9943b36f6404c0b325a129` |
| `FAILED_training.json` | `d810903b75d3e84da648a54ba4dce93d8607f0048d709134cce0f1489c9ef541` |

The stage-level evidence paths declared by the markers are:

```text
/mnt/cpfs/zbl-cpfs-new/USERS/leon/cache/r142_stage_s/pi05_base/BASE_DOWNLOAD_COMPLETED.json
/mnt/cpfs/zbl-cpfs-new/USERS/leon/cache/r142_stage_s/pi05_base_pytorch/CONVERSION_COMPLETED.json
/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_s/c_status/r142-stage-s-c-undertrained-20260903-r3/RUNTIME_IDENTITY.json
```

`FAILED_training.json` is the terminal stage marker (`status=FAILED`, exit
code 1). Its pinned runtime is
`/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/r142-stage-s-c-runtime-20260903` at
Stage-S commit `7575da585be31eb369a604d90048b338bbbf2c92`, with OpenPI
`54cbaee6ae0c010a1ed431871cdaa8f4684ac709` and QPILOTS
`eacf47b981e3b22357f8a74902f8dad8cfcfa375`.

## No training completion and recovery boundary

The run log root is
`/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_s/c/r142-stage-s-c-undertrained-20260903-r3/`;
its preserved files are `TRAINING_START.json` and `FAILED_C_TRAINING.json`.
The registry root
`/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/pai_registry/r142_stage_s/c/r142-stage-s-c-undertrained-20260903-r3/`
contains no completion artifact. The scientific checkpoint root
`/mnt/cpfs/zbl-cpfs-new/CKPT/leon/r142_stage_s_c/pi05_libero/r142_stage_s_c_undertrained_seed42/`
contains only `wandb_id.txt` on this readback; there is no
`COMPLETED_C_TRAINING.json`, training `SHA256SUMS`, or retained 1k/3k/6k/10k
checkpoint set attributable to r3.

r3 is sealed and must not be resumed. After the pinned dataset is made
locally available and independently verified, a new controller-approved run
must use the same frozen scientific checkpoint directory/configuration while
preserving the source, seed, data sequence, and checkpoint schedule. The
dataset-access repair is an environment prerequisite, not evidence for or
against the R142 mechanism.
