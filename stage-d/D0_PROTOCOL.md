# R142-FP-11 Phase D0 Protocol (frozen)

CPU-only offline screen using repository rollouts; freeze commit is recorded in downstream artifacts.

## Frozen labels
- Mixed family: at least 2 successes and at least 2 failures among exactly 32 candidate rollouts (one task and initial state).
- `t_c`: earliest normalized control step at which proprioceptive EEF pose separation between failing and succeeding groups crosses the frozen task-specific 95th-percentile within-task label-permutation null; normalized step is `step/(comparable episode-window length-1)`.
- Permutation null: shuffle candidate success labels within each family, 1000 shuffles per task (95th percentile of family separation maxima).
- Censoring: no crossing inside comparable window => `CENSORED`, excluded; never assign the last step.
- Fork/failure-event rule: compute progress-separation onset separately for validation only; require pose onset to lead progress onset. Non-positive lead => `FAILURE_EVENT_NOT_FORK`, excluded.

## Frozen detector whitelist
Only within-rollout `eef` and executed `actions`, plus quantities derived from them: (a) EEF spread steepest rise, (b) stepwise pairwise action-sequence distance, (c) first hierarchical action-cluster split, (d) proprioceptive velocity/acceleration changepoint, (e) action norm/direction discontinuity. Objects, progress, labels, latent distances, family-level statistics and future information are prohibited detector inputs.

## Frozen metrics and gates
Primary metric is `lead = t_c - t_detect`, normalized by episode length. Report median lead, fraction lead > 0, correlation(t_detect,t_c), and per-task permutation-null 95th percentile for that correlation.

- D0-1 LABELS: at least 50 retained families after `NO_SEPARATION`, `CENSORED`, and `FAILURE_EVENT_NOT_FORK` exclusions.
- D0-2 PROSPECTIVE: some signal has median lead > 0 and fraction lead > 0 at least 0.60.
- D0-3 SIGNIFICANT: that signal correlation exceeds its per-task permutation 95th percentile.
- D0-4 SPECIFIC: each signal firing rate on all-success families is at most 0.20.

Exactly one decision code is emitted: `DETECTOR_CANDIDATE_FOUND`, `LOCALIZATION_INFEASIBLE_OFFLINE`, `INSUFFICIENT_LABELS`, or `PIPELINE_INVALID` (baseline (a) fails to reproduce known failure corr +0.174 versus null 95th +0.260). `t_c` is observational, not causal.

No PAI, GPU, policy loading, remote download, or new infrastructure is permitted. New Python under `src/` and `scripts/` remains below 1500 lines. Stop at this D0 checkpoint; no D1.
