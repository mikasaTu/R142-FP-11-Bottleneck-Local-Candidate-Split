# Stage-S B-variant PAI run ledger

This ledger is operational evidence only. A validation refusal, `EnvPreparing`,
`Running`, `FIRST_WORK.json`, or a partial variant directory is not a scientific
result and must not be promoted to completion.

| Attempt | JobId | State | Evidence / disposition |
|---|---|---|---|
| r1 | not created | sealed | Controller rejected a pod-environment mismatch before creation. |
| r2 | not created | sealed | Controller rejected an evidence-kind mismatch before creation. |
| r3 | not created | sealed | Controller rejected the corrected-but-still-incompatible evidence contract before creation. |
| r4 | not created | sealed | Controller rejected the run because the required registry output parent was absent. |
| r5 | `dlck9preyo3symau` | stopped after failure | The task wrote `FIRST_WORK.json`, then pinned LIBERO attempted its first-import interactive dataset prompt and raised `EOFError`. `FAILED_B_VARIANTS.json` was preserved. The exact job was stopped; partial artifacts are not reusable as completion. |
| r6 | `dlc1bqps0a2dhbcw` | stopped after failure | Run-scoped LIBERO configuration succeeded and EGL initialized, but the generator compared source flattened MuJoCo states (123 values for the first task) against `sim.data.qpos` (72 values). The strict dimension check stopped the run. The exact job, log, `FAILED_B_VARIANTS.json`, and partial artifacts were preserved; none is completion evidence. |
| r7 | `dlc1t8igi22mqwq6` | `Succeeded`; accepted | Pins Stage-S source `afe353bbc5997355f35cb0c77c5446fd4df5f1e3`. The terminal job produced 4 settings x 10 tasks, 40 BDDL files and 16 regenerated flattened simulator states per task/setting. `COMPLETED_B_VARIANTS.json` and `SHA256SUMS` both passed readback. Every task grew by 13 state values for the added free-joint distractor, `old_init_reused=false`, and the generator required `set_init_state` round-trip error <= 1e-9 with finite observations before accepting each row. |

## Recovery rules

The replacement launcher exports a run-scoped `LIBERO_CONFIG_PATH` and writes
the exact pinned LIBERO benchmark, BDDL, init-state, dataset, and asset paths
before importing LIBERO. This fixes only non-interactive path initialization;
it does not alter tasks, seeds, distractor settings, initial-state counts,
policy behavior, calibration budgets, or Stage-S gates. Recovery uses a new run
ID and requires terminal PAI success plus `COMPLETED_B_VARIANTS.json` and a
verified `SHA256SUMS` before the generated variants are accepted.

The r6 failure additionally showed that LIBERO `.pruned_init` rows follow
`ControlEnv.get_sim_state() == sim.get_state().flatten()`, not raw qpos. The
next recovery must generate that same flattened state schema, prove each row
can be restored through `set_init_state`, and retain the dimension-growth gate
for the added free-joint object. Merely relaxing the 123-versus-72 comparison
is prohibited.

## Accepted r7 bundle

The accepted persistent root is
`/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_s/b_variants/r142-stage-s-b-variants-20260903-r7`.
Its settings are exactly `proximity_0.06m`, `0.08m`, `0.10m`, and `0.12m`;
each `REGENERATED_INIT_STATES.json` records ten tasks, sixteen states per task,
the flattened-state format, deterministic seeds, source/variant dimensions,
and `old_init_reused=false`. The top-level matrix SHA is
`db84103b15a1f091cc6f9a2dbab92e72f9644a0e3d55aab116c7cfef3b49e651`.
This is input qualification only, not calibration or Stage-S gate evidence.
