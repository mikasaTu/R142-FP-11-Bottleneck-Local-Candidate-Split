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

## r5 recovery rule

The replacement launcher exports a run-scoped `LIBERO_CONFIG_PATH` and writes
the exact pinned LIBERO benchmark, BDDL, init-state, dataset, and asset paths
before importing LIBERO. This fixes only non-interactive path initialization;
it does not alter tasks, seeds, distractor settings, initial-state counts,
policy behavior, calibration budgets, or Stage-S gates. Recovery uses a new run
ID and requires terminal PAI success plus `COMPLETED_B_VARIANTS.json` and a
verified `SHA256SUMS` before the generated variants are accepted.
