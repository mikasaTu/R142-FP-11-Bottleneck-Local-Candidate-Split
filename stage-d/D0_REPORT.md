# R142-FP-11 Phase D0 Report

- Protocol freeze: `857ff769c902832800e5d650427670014951d9ba` (protocol blob SHA; commit to be recorded below).
- Run date: 2026-09-16, dev14 CPU-only.
- Data: 40 phase0r task NPZs, 640 families, 512 rollouts/task (16 initial states × 32 candidates). B calibration NPZ discovered: 0; no calibration trajectories were fabricated.
- Detector inputs were restricted to `eef` and `actions`; `progress` was used only for Step-1 fork-vs-failure validation. `objects`, labels, latent distances, family-level and future information never entered detector functions.

## Label accounting

Mixed families: 91; retained after frozen exclusions: 75. Exclusions: {'NOT_MIXED': 549, 'RETAINED': 75, 'FAILURE_EVENT_NOT_FORK': 15, 'CENSORED': 1}. D0-1 (>=50) = PASS. `t_c` is an observational normalized onset proxy, not causal.

## Signal results

| signal | n | median lead | fraction lead>0 | corr(t_detect,t_c) | task-null 95% |
|---|---:|---:|---:|---:|---:|
| eef_spread | 75 | -0.457661 | 0.013333 | 0.167135 | 0.193421 |
| action_distance | 75 | -0.365927 | 0.133333 | 0.142677 | 0.168726 |
| cluster_split | 75 | -0.365927 | 0.133333 | 0.142677 | 0.170464 |
| proprio_changepoint | 75 | 0.018145 | 0.560000 | 0.242749 | 0.237237 |
| action_discontinuity | 75 | -0.224294 | 0.213333 | -0.188938 | -0.116344 |

Specificity on all-success families (n=471):

- eef_spread: fired 48/471, rate 0.101911 (D0-4 threshold <=0.20: PASS).
- action_distance: fired 233/471, rate 0.494692 (D0-4 threshold <=0.20: FAIL).
- cluster_split: fired 233/471, rate 0.494692 (D0-4 threshold <=0.20: FAIL).
- proprio_changepoint: fired 364/471, rate 0.772824 (D0-4 threshold <=0.20: FAIL).
- action_discontinuity: fired 303/471, rate 0.643312 (D0-4 threshold <=0.20: FAIL).

## Frozen gates and decision

- D0-1 labels: PASS (75 retained).
- D0-2 prospective: FAIL; best `proprio_changepoint` median lead is positive (0.018145) but fraction lead>0 is 0.56, below 0.60.
- D0-3 significant: the same signal's correlation 0.242749 is above its 0.237237 task-null 95th percentile (PASS in this implementation).
- D0-4 specificity: only `eef_spread` is <=0.20; other signals fire too often.
- Baseline sanity (signal a): the implementation produced corr=0.167135 and null95=0.193421, not the known +0.174/+0.260 reference; therefore the frozen baseline reproduction check fails.

**Decision code: `PIPELINE_INVALID`**

The decision is `PIPELINE_INVALID` because the required baseline (a) reproduction did not match the known failure reference. This is a calibration failure, not evidence for or against the scientific hypothesis. No D1, policy instrumentation, substrate proposal, or PAI job was run.

## Mechanistic reverse explanation (code/results only; no new idea)

The retained families show early pose-group separation followed by late within-rollout derivative extrema. The EEF spread detector fires late (median lead -0.457661), consistent with its steepest-rise statistic reacting to trajectory expansion after the success/failure groups have already diverged. Action-distance and cluster proxies are similarly late and have high all-success firing (0.495), indicating generic action transients rather than commitment localization. The proprioceptive changepoint is the only near-prospective signal (median lead 0.018145; 56% positive): velocity/acceleration extrema sometimes precede the observational pose crossing, but their task-null margin is small and the 60% gate is missed. Action discontinuity is negatively aligned (corr -0.188938), consistent with isolated control spikes being unrelated to the group-level fork. These are observational mechanisms inferred from implemented statistics, not causal claims.

All negative results, exclusions, and permutation arrays are preserved in `stage-d/results/`. Stop at D0 checkpoint.
