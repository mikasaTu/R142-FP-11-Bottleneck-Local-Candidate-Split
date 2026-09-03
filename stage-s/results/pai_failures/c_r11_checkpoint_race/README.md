# Stage-S C r11 checkpoint race evidence

- Scientific run: `r142-stage-s-c-undertrained-20260903-r11`
- PAI JobId: `dlc1mg3n62bspsuc`
- PAI terminal status after containment: `Stopped`
- Frozen training contract: 8 GPU ranks, seed 42, checkpoints at steps 1000/3000/6000/10000, 10001 training steps.
- Source commit: `19867e93ca9d3c197ee3fc4b8db8ca8efa371af6`
- Deployed worker SHA-256: `147a6a9bd01023a3a58dbae4348f4d3c73d6769d7399a6d428693bfe8d025e51`
- Failure type: application checkpoint-save race, not an accepted PAI eviction.

At step 1000, ranks 1-7 raised `native checkpoint directory is absent after save` before rank 0 published the native checkpoint directory. The final directory contained only `rng_state.rank0.pt`; the other seven RNG sidecars remained in `.rng_stage_1000`. `RNG_SHA256SUMS` and `COMPLETE_RNG_STATE.json` were absent. Therefore this checkpoint is incomplete and MUST NOT be used for resume or calibration.

After the job and both recorded pods reached `Stopped`, the partial checkpoint and RNG staging directory were moved to the recoverable evidence directory:

`/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_s/interruptions/r142-stage-s-c-undertrained-20260903-r11-checkpoint-race`

The original active checkpoint root no longer contains step 1000. The scientific task, seed, topology, save interval, and training-step target remain frozen. Recovery must use corrected cross-rank readiness synchronization and restart from the last complete state; because r11 produced no complete state, that means a fresh seed-42 start.

