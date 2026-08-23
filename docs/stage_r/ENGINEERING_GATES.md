# Stage-R engineering-gate protocol

Protocol ID: `r142-stage-r-engineering-v1`. This file is frozen before gate
outcomes. Observations are written only beneath `results/stage_r/gates/` and in
the report.

E1 persists and verifies the exact lineage declared in
`PHASE0R_PROTOCOL.md`, including full checkpoint-tree and task-metadata hashes.

E2 compares the original pinned `CleanPi05LiberoPolicy.infer_official` and the
Stage-R wrapper with instrumentation disabled, using the same raw observation
and explicit noise. Pass requires array shape equality and bitwise equality of
all 10x7 float32 actions.

E3 captures a simulator snapshot, executes a fixed float32 action, restores,
executes the same action, and compares the complete MuJoCo flat state, auxiliary
arrays, controller fields, robot buffers, observables and rendered policy
observation. Pass requires maximum absolute next-state error <=1e-9 and exact
image equality.

E4 snapshots simulator state, the runner's one-observation policy-history
buffer, a non-empty five-action execution queue, Python/NumPy/JAX policy-noise
RNG state and all counters. Full restore must reproduce the continuation.
Separately omit each component and continue through the next replan; each
ablation must diverge in executed action or simulator state. Missing a
component, or an ablation that does not diverge, fails E4.

E5 holds all observations and states fixed and changes only the branch RNG.
Pass requires a bitwise difference in sampled model action and executed physical
action with finite values and identical shapes.

E6 uses the preregistered calibration task `libero_90` task 64, “stack the right
bowl on the left bowl and place them in the tray”, with the same pinned policy,
the 16 lowest SHA-ranked initial states and four independent seeds per state
(64 full rollouts). Pass requires eventual success in [0.25,0.75] and official
predicate progress not having median=1 and IQR=0.

Every gate phase also runs the frozen geometric positive-control smoke. Any E1
through E6 failure yields `STAGE_R_ENGINEERING_INCOMPLETE`; Phase-0R science may
not start. Fixing a defect requires a new implementation commit and complete
gate rerun. Changing checkpoint requires an explicit lineage revision; no
outcome-based silent substitution is allowed.

Completion requires `results/stage_r/gates/COMPLETED_ENGINEERING_GATES.json`
and SHA-256 verification. A smoke, first-work marker or a passing subset is not
gate completion.
