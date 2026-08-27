# R142-FP-11 Stage-R Phase-1R human-override protocol

Status: frozen before any Phase-1R descendant outcome is generated.

Protocol ID: `r142-stage-r-phase1r-human-override-v1`

## Why this amendment exists

Phase-0R completed on all 40 executable LIBERO tasks and retained zero tasks.
The original Stage-R plan therefore recorded `CHECKPOINT_1_STOP`. On
2026-08-27 the human owner explicitly overrode every intermediate gate and
instructed that all planned experiments be completed. This amendment preserves
the Phase-0R negative result; it does not reinterpret a failed gate as a pass.

The override authorizes Phase-1R collection and analysis for every executable
LIBERO task. No task is selected using a Phase-1R descendant outcome. The
RoboTwin candidate remains `SOURCE_LIMITATION_UNVERIFIABLE`: the frozen pi0.5
policy/checkpoint has no verified RoboTwin observation/action adapter or
snapshot implementation. No synthetic RoboTwin result may be substituted.

The supplied plan defines Phase-1R but does not define a Phase-2R experiment.
Phase-2R is therefore outside the executable protocol and must not be invented.

## Frozen lineage

- Natural policy: pinned pi0.5 LIBERO policy and checkpoint from Phase-E.
- Phase-0R raw authority: the outcome-blind authoritative merge containing all
  40 LIBERO task NPZ/metadata pairs.
- Natural task order: `libero_spatial`, `libero_object`, `libero_goal`,
  `libero_10`; task IDs 0 through 9 in each suite.
- Natural task scope: all 40 tasks, irrespective of Phase-0R retention.
- Per-node ceiling: at most 8 GPU, 88 CPU, 1525 GiB memory, and 1525 GiB
  shared memory.
- Global active Phase-1R ceiling: at most 16 GPU.

## Episode selection

The Phase-0R archive contains 16 initial states and 32 independently sampled
rollouts per initial state. The original phrase "4 failed / 4 marginal / 4
successful episodes" is infeasible as an initial-state stratification: no
Phase-0R initial state was all-fail. To execute all tasks without fabricating
or duplicating trajectories, the selection unit is frozen here as an
individual Phase-0R rollout.

Each rollout receives exactly one baseline stratum:

- `failed`: eventual success is false and final official predicate fraction is
  exactly zero;
- `marginal`: eventual success is false and final official predicate fraction
  is greater than zero;
- `successful`: eventual success is true.

Within each stratum, rank without replacement by SHA-256 of
`protocol_id|episode|suite|task_id|init_state|candidate_id|rollout_seed` and
take up to four. If a stratum has fewer than four members, fill every remaining
slot from all still-unselected rollouts using the same SHA rank, without
replacement and without looking at any Phase-1R descendant. Each task always
has 12 unique selected rollouts. Persist requested/observed stratum counts and
every fallback fill.

## Natural branch intervention

For each selected baseline rollout with length `L`, use ten control-step
locations. For decile `q` in `0.1, 0.2, ..., 1.0`, freeze
`t(q) = max(0, min(L-1, ceil(q*L)-1))`. A replay reconstructs the exact
initial state, common environment seed, policy-noise seed, observation buffer,
action-chunk queue, simulator state, and RNG state. The snapshot is captured
immediately before executing control step `t`.

At each snapshot, create `M=16` descendants. Each descendant:

1. restores the complete snapshot;
2. discards the inherited unexecuted action queue at the intervention instant;
3. samples a new action chunk from an independent SHA-derived branch RNG;
4. executes the same five-action receding-horizon convention as Phase-0R;
5. continues with the same frozen policy and branch RNG stream until the
   environment reports termination;
6. records eventual success, full executed action genealogy, official progress,
   policy-forward count, environment-step count, and immutable IDs.

Calibration and held-out RNG namespaces are disjoint. Natural headline curves
use the held-out namespace. Calibration descendants may only be used for null
threshold construction and engineering checks.

## Same-pipeline controls

Run both controls through the identical 12 episode x 10 location x 16 branch
cell schema and the same analysis code:

- Positive: `GeometricCommit2D-Phase1R-v1`, with a collision barrier and a
  one-way aperture. The simulator records the first irreversible side crossing;
  restoring after that crossing cannot change sides. This construction is
  verified directly from simulator state transitions.
- Null: `OpenPlane2D-Phase1R-v1`, a single-mode recoverable integrator with no
  irreversible barrier.

The controls are diagnostics only and are not evidence about the learned
policy.

## Blinding, permutation null, and thresholds

Raw collection is complete before natural analysis. A calibration program
computes, for each task/control, at least 1000 within-episode permutations of
the branch-location labels while preserving the 16 descendant outcomes in each
episode. The test statistic is location sensitivity
`S = max_t R(t) - min_t R(t)` on the task-level pooled curve. The threshold is
the 95th percentile of the corresponding permutation distribution.

The calibration artifact contains thresholds and hashes but no unpermuted
natural curve. It must be committed before the unblinded analysis program is
run. Thresholds, task scope, seeds, branch budget, and statistics cannot change
afterward.

## Statistics and decisions

- Primary curve: `R(t)`, pooled over all 12 selected episodes and 16 descendants
  per location.
- Primary uncertainty: paired bootstrap over episodes, 10000 replicates.
- Derived: location sensitivity, largest adjacent cliff, and earliest commit
  point where `R(t) < (Rmax+Rmin)/2`.
- Compute: candidate-equivalent policy forwards (primary), physical batched
  calls, environment steps, branch count, and budget slack.

The positive-control decision is recorded but is not a stop under the explicit
human override. If it is below threshold, label the pipeline
`PIPELINE_INVALID_WITH_HUMAN_CONTINUE_OVERRIDE` and still complete all natural,
null, permutation, and reporting work. Natural task decisions retain the
original meanings and are reported even when the pipeline control fails.

## Resume and completion

The atomic recovery unit is one `(task, selected_episode, branch_location,
stream)` cell containing all 16 descendants. A cell is reusable only after its
metadata schema, NPZ structure, owner `2254:2254`, and SHA-256 verify. Invalid
or partial cells are preserved as failure evidence and recomputed under the
same protocol. A task, shard, job, and global result each require their own
`COMPLETED_*.json` and SHA manifest. Queueing, `Running`, first work, or a
partial shard is never completion.

## Evidence boundary

The experiment can establish only whether branch-location recoverability is
non-flat under this frozen pi0.5/LIBERO lineage. It does not by itself establish
that bottleneck-local candidate split beats uniform, random, or best-of-N;
that intervention comparison belongs to an undefined Phase-2R and cannot be
claimed from Phase-1R.
