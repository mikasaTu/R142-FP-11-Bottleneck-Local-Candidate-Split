# Formal PAI predecessor r2

- Run ID: `r142-fp11-stage1-eval-20260823-1323-r2`
- Job ID: `dlc1yavs3uab0cm0`
- Resource: idle A800, exact `UseOversoldResource=true`
- First real work: not reached
- Earliest decisive error:
  `/tmp/pai-payload.../payload.sh: line 8: RUN_ID: RUN_ID is required`
- Cause: the generic controller passes `PAI_CANARY_RUN_ID` and
  `PAI_CANARY_RUN_DIR`; the initial launcher incorrectly expected `RUN_ID` and
  `ARTIFACT_DIR` directly.
- Repair: bind the launcher to the controller-provided names without changing
  benchmark, seeds, budgets, metrics, or gate thresholds.
- Stop receipt: pinned DLC CLI returned
  `Job [dlc1yavs3uab0cm0] was stopped successfully`.
- Terminal readback: `Stopped`, `ReasonCode=StoppedByUser`.
- PAI probe created: no.

This run is sealed as the exact same-workflow predecessor of replacement r3.
Its PAI service row may be deleted only after r3 proves persisted first work
and remains healthy. Registry, logs, placement evidence, and CPFS artifacts are
preserved.
