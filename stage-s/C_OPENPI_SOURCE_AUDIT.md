# Stage-S substrate C: pinned OpenPI and data audit

Audit date: 2026-09-02 (Asia/Shanghai). This is an implementation/source
audit, not a C scientific result. No PAI job was submitted and the 12.4 GB
base was not downloaded during this pass.

## Exact source and checkpoint lineage

| item | frozen value | dev14 evidence |
|---|---|---|
| OpenPI source | `https://github.com/Physical-Intelligence/openpi.git` | checkout `/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/QPILOTS-r16p15-stage1-task64-20260812/third_party/openpi` |
| OpenPI commit | `54cbaee6ae0c010a1ed431871cdaa8f4684ac709` | `git rev-parse HEAD` exact; clean detached checkout |
| configuration | `pi05_libero` | `src/openpi/training/config.py` at the pin |
| official converter | `examples/convert_jax_model_to_pytorch.py` | `tyro.cli(main)`, `convert_pi0_checkpoint` |
| official trainer | `scripts/train_pytorch.py` | `train_loop`, native model/optimizer/metadata saves |
| dataset | `physical-intelligence/libero` | pinned `LeRobotLiberoDataConfig`, `prompt_from_task=True`, `extra_delta_transform=False` |
| base source | `gs://openpi-assets/checkpoints/pi05_base/params` | the C input is the public JAX base, never the community full SFT |

The pinned `pi05_libero` config is `Pi0Config(pi05=True,
action_horizon=10, discrete_state_input=False)`, global batch size 256,
CosineDecay warmup 10,000, peak/end learning rate `5e-5`, AdamW, and the
official LIBERO LeRobot transforms. The C run changes only the frozen seed,
terminal step, save cadence, and PyTorch base path required by this screen;
it does not change model/data semantics.

The exact conversion invocation is:

```text
python examples/convert_jax_model_to_pytorch.py \
  --checkpoint_dir /mnt/cpfs/zbl-cpfs-new/open_data/r142_stage_s/pi05_base \
  --config_name pi05_libero \
  --output_path /mnt/cpfs/zbl-cpfs-new/Models/r142_stage_s/pi05_base_pytorch \
  --precision bfloat16
```

The converter restores the Orbax `params/` tree and writes
`model.safetensors`, `config.json`, and copied assets. The wrapper adds
`CONVERSION_PROVENANCE.json` and `CONVERSION_COMPLETED.json`, including the
source commit, config, source-manifest digest, and model SHA-256.

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
learning rate in this trainer is a deterministic function of the saved global
step and frozen config, so no separate scheduler object is silently omitted.

## Current execution status and blockers

Source and static contracts are ready. A real C result remains pending:

1. download the exact public 29-object base to the CPFS path and read back all
   SHA-256 values;
2. run the pinned converter and read back its model/provenance marker;
3. run the 8-GPU idle training lineage to global step 10001, preserving the
   same directory across any spot interruption;
4. audit all four selected full-state checkpoints and publish the terminal
   marker.

The community checkpoint
`/mnt/cpfs/zbl-cpfs-new/USERS/leon/checkpoints/openpi/community_madokalif_pi05_libero_sft`
is explicitly forbidden as a C base; its declared 60,000-step SFT lineage is
not used or relabeled.
