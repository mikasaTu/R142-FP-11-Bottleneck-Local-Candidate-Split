# Stage-S calibration spot-resume runbook

`run_calibration_shard` writes one aggregate-only progress pair under the
rank's shard directory while it evaluates the frozen calibration grid:

* `CALIBRATION_PROGRESS.json`
* `CALIBRATION_PROGRESS_SHA256SUMS`

The state contains the frozen-plan identity hashes, the round-robin
`next_ordinal`/`finished_ordinals`, and per-setting aggregate `successes` and
`totals`. It never contains a family id, candidate outcome, trajectory,
genealogy, or S2--S5 statistic. The plan hash binds progress to the exact
settings, tasks, initial states, candidate count, substrate sources, and
protocol that were prepared before the worker started.

For a spot interruption, resubmit the same shard command with the same
output root, seed, rank, world size, settings, and source arguments. The
worker verifies the progress SHA and all identity hashes, reconstructs the
round-robin assignment, skips only the sealed completed ordinals, and reuses
the exact deterministic trial seeds. A missing pair, bad SHA, plan drift,
out-of-assignment ordinal, or inconsistent counter is a fail-closed error;
do not repair the JSON manually. Preserve the directory as evidence and
start a fresh run id if the state cannot be verified.

After the last assigned trial, the existing aggregate-only sealing sequence
remains unchanged: `RESULT.json`, `SHA256SUMS`, and `COMPLETED_SHARD.json`.
Rerunning a shard after that marker performs verification only and does not
call the evaluator again.
