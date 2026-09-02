# Stage-S substrate C: pinned OpenPI and data audit

Audit date: 2026-09-03 (Asia/Shanghai). This is an implementation/source
audit, not a C scientific result. No PAI job was submitted and the 12.4 GB
base was not downloaded during this pass.

## Exact source and checkpoint lineage

| item | frozen value | dev14 evidence |
|---|---|---|
| OpenPI source | `https://github.com/Physical-Intelligence/openpi.git` | checkout `/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/QPILOTS-r16p15-stage1-task64-20260812/third_party/openpi` |
| OpenPI commit | `54cbaee6ae0c010a1ed431871cdaa8f4684ac709` | `git rev-parse HEAD` exact; clean detached checkout |
| OpenPI Python | `/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/openpi_py311/bin/python` | executable, `Python 3.11.11`; launcher uses this absolute path |
| configuration | `pi05_libero` | `src/openpi/training/config.py` at the pin |
| official converter | `examples/convert_jax_model_to_pytorch.py` | AST signature audit: `checkpoint_dir`, `config_name`, `output_path`, `precision`, `inspect_only`; `tyro.cli(main)` |
| official trainer | `scripts/train_pytorch.py` | AST `TrainConfig` override audit: `exp_name`, checkpoint/save/seed/weight/assets/worker/resume fields; `train_loop`, native model/optimizer/metadata saves |
| dataset | `physical-intelligence/libero` | pinned `LeRobotLiberoDataConfig`, `prompt_from_task=True`, `extra_delta_transform=False` |
| base source | `gs://openpi-assets/checkpoints/pi05_base/params` | the C input is the public JAX base, never the community full SFT |

The pinned `pi05_libero` config is `Pi0Config(pi05=True,
action_horizon=10, discrete_state_input=False)`, global batch size 256,
CosineDecay warmup 10,000, peak/end learning rate `5e-5`, AdamW, and the
official LIBERO LeRobot transforms. The C run changes only the frozen seed,
terminal step, save cadence, and PyTorch base path required by this screen;
it does not change model/data semantics.

The exact conversion invocation (with the pinned runtime) is:

```text
/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/openpi_py311/bin/python \
  examples/convert_jax_model_to_pytorch.py \
  --checkpoint_dir /mnt/cpfs/zbl-cpfs-new/USERS/leon/cache/r142_stage_s/pi05_base \
  --config_name pi05_libero \
  --output_path /mnt/cpfs/zbl-cpfs-new/USERS/leon/cache/r142_stage_s/pi05_base_pytorch \
  --precision bfloat16
```

The converter restores the Orbax `params/` tree and writes
`model.safetensors`, `config.json`, and copied assets. The wrapper adds
`CONVERSION_PROVENANCE.json` and `CONVERSION_COMPLETED.json`, including the
source commit, config, source-manifest digest, and model SHA-256.

## Runtime checkout and launcher admission

The registry command file is the external launcher
`/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/r142-stage-s-pai-20260902/stage_s_c_undertrained_pai.sh`;
its companion manifest is in the same external directory. The launcher
hard-codes the independent C runtime clone
`/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/r142-stage-s-c-runtime-20260903` and
refuses a missing, non-canonical, dirty, or differently pinned checkout. The
frozen runtime commit is
`7575da585be31eb369a604d90048b338bbbf2c92`; the launcher also verifies the
QPILOTS parent commit
`eacf47b981e3b22357f8a74902f8dad8cfcfa375`, the OpenPI commit above, the
external companion manifest's payload SHA-256, and the actual executed `$0`
payload SHA-256. Only `PAI_CANARY_RUN_ID` is controller-injected; no custom
payload/source/project environment is required. It writes
`RUNTIME_IDENTITY.json` and a hashed `COMPLETED_preflight.json` before any
download, conversion, or training mutation. Missing or mismatched identity
fails closed and cannot produce a stage completion marker.

## Canonical registry admission readback

The C manifest carries the exact resource provenance required by the clean
registry: `contract_source_job_id` and `resource_source_job_id` are both
`dlckjz66iwcv38gw`, `source_role` is `readback_reference`,
`submission_method` is `cli_create`, and `pai_clone_performed` is `false`.
The frozen runtime public environment contains exactly
`NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`, matching the canonical
`required_pod_env`; `required_env_names` contains only the
controller-injected `PAI_CANARY_RUN_ID`. The external companion config and
payload were deployed to the canonical code root, and
`pai-job validate ... --run-id r142-stage-s-c-undertrained-20260903-r1
--no-wandb` returned `{"valid": true}`. No PAI submission was made; this is
an admission validation result, not a training result.

## Public GCS object contract

The live public JSON API listing was checked against the frozen manifest in
`stage-s/C_PI05_BASE_GCS_MANIFEST.json`:

* exactly 29 objects under `checkpoints/pi05_base/`;
* total bytes exactly `12,441,749,581`;
* GCS `generation`, `size`, MD5 (where provided), and CRC32C are persisted;
* GCS does not provide SHA-256 for this object set, so each SHA-256 is
  calculated only after the corresponding bytes are persisted on CPFS.

The downloader uses one `.part` file per object, resumes with an HTTP Range,
rejects a server that ignores Range, checks expected size and available MD5,
then atomically renames the object. It writes `BASE_OBJECT_MANIFEST.json`,
`SHA256SUMS`, and `BASE_DOWNLOAD_COMPLETED.json` only after all 29 objects
pass. A short or corrupt object leaves evidence and no completed marker.

## Data asset path

The pinned data config resolves its normalization asset from

```text
/mnt/cpfs/zbl-cpfs-new/USERS/leon/cache/openpi/r16p15/openpi-assets/checkpoints/
  pi05_libero/assets/physical-intelligence/libero/norm_stats.json
```

This file existed during the audit with SHA-256
`b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84`.
The training command passes the parent `assets_base_dir` explicitly; it does
not rely on the current working directory or on an implicit local dataset.

## Native checkpoint and resume semantics

The pinned PyTorch trainer atomically renames `tmp_<step>` to `<step>` and
saves `model.safetensors`, `optimizer.pt`, and `metadata.pt`. With
`num_train_steps=10001` it executes optimizer steps through terminal global
step 10001 and natively saves the periodic checkpoints 1000 through 10000;
the final retained C selection is exactly 1000, 3000, 6000, and 10000. There
is intentionally no interpolated or numerically degraded checkpoint.

The pinned trainer does not save process RNG state itself. The checked-in C
worker imports that exact trainer and wraps only `save_checkpoint` and
`load_checkpoint`, adding atomic per-rank `rng_state.rank0.pt` through
`rng_state.rank7.pt` sidecars. On `--resume`, a missing sidecar is fatal. The
source audit also records the native loop (`while global_step < ...`,
`global_step // len(loader)`, `for observation, actions in loader`). Because
the pinned `TorchDataLoader` is an infinite iterator and the public wrapper
does not expose its finite length/sampler epoch, the worker's
`ExactCursorDataLoader` exposes one finite epoch, forwards
`DistributedSampler.set_epoch(epoch)`, and skips exactly
`global_step % epoch_length` batches on the first resumed iterator. `--num_workers 0`
is frozen so no hidden worker cursor/RNG is omitted. If length, sampler epoch,
or metadata cursor cannot be proven, resume fails closed. The learning rate in
this trainer is a deterministic function of the saved global step and frozen
config, so no separate scheduler object is silently omitted.

Stage and terminal markers are persisted atomically. Every registry stage
writes a hashed `COMPLETED_*.json` or `FAILED_*.json` under the run's status
root; the training wrapper additionally writes `TRAINING_TERMINAL.json`,
`COMPLETED_C_TRAINING.json`, or `FAILED_C_TRAINING.json` only after the
corresponding evidence audit. Finalization writes separate cwd-relative
checkpoint/log `SHA256SUMS` manifests, each verified with `sha256sum -c` from
its own root.

## Current execution status and blockers

Source and static contracts are ready. A real C result remains pending:

1. download the exact public 29-object base to the CPFS path and read back all
   SHA-256 values;
2. run the pinned converter and read back its model/provenance marker;
3. run the 8-GPU idle training lineage to global step 10001, preserving the
   same directory across any spot interruption;
4. audit all four selected full-state checkpoints and publish the terminal
   marker.

The registry-submit contract is checked in as
`configs/pai/stage_s_c_undertrained.json` (schema v2) with the executable
`scripts/stage_s_c_undertrained_pai.sh`; it binds the robot-idle alias/id/quota,
UID/GID, CPFS write paths, pinned Python, sequential stage resume, and the
09:30–09:40 / 19:30–19:40 Asia/Shanghai fail-closed windows. No PAI job was
submitted in this audit.

The community checkpoint
`/mnt/cpfs/zbl-cpfs-new/USERS/leon/checkpoints/openpi/community_madokalif_pi05_libero_sft`
is explicitly forbidden as a C base; its declared 60,000-step SFT lineage is
not used or relabeled.
