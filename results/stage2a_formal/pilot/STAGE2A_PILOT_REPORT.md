# R142-FP-11 Stage-2A Pilot Report

Scientific decision: `R142_FP11_CORE_HYPOTHESIS_WEAKENED`

This report evaluates the official learned LeRobot Diffusion Policy on unmodified standard PushT. It does not use ForkPush2D, a VLA, a learned detector, or oracle labels at deployment time.

## Evidence boundary

- Baseline rollouts: 50 fixed seeds; success rate 0.660; mean max progress 0.994.
- Natural snapshots: 24 selected before descendant outcomes.
- A-E real-source equivalence gates: PASS.
- Discovery tree: K=8 roots, 16 real DDPM checkpoints, M=8 calibration and M=8 held-out suffixes per root/checkpoint.
- Fixed sample-NFE budget: 7200 per snapshot for every branching strategy, with slack reported.

## Q1-Q6 quantitative answer

1. Location-sensitive branchability prevalence over all snapshots: 0.000 (0/24).
2. Natural recoverability-cliff prevalence in hard/failing snapshots: 0.000 (0/8).
3. Oracle checkpoint indices: {'20': 3, '46': 4, '13': 2, '53': 2, '0': 10, '86': 1, '26': 1, '7': 1}.
4. Always-early equivalent-to-oracle flag: True.
5. Raw disagreement/branchability correlation (snapshot mean): -0.029.
6. Raw, benefit/NFE and fixed-NFE results are all retained; the decision uses only held-out fixed-NFE results.

## Fixed-NFE held-out comparisons

| Comparison | Mean oracle gain | Paired 95% CI | Required |
|---|---:|---:|---:|
| oracle-local - always_early | 0.0023 | [0.0003, 0.0049] | >=0.10 and lower>0 |
| oracle-local - uniform_three_quantiles | -0.0005 | [-0.0014, 0.0004] | >=0.10 and lower>0 |
| oracle-local - random | -0.0001 | [-0.0022, 0.0028] | >=0.10 and lower>0 |

## Negative controls

- `disagreement-outcome-decoupled`: 17 snapshots
- `fake-disagreement`: 11 snapshots
- `no-bottleneck`: 24 snapshots
- `smooth-decay`: 8 snapshots

## Mechanism reverse explanation (no new idea)

### observed_fact

Across 24 preregistered natural snapshots, 0 had a held-out progress-spread range of at least 0.10; 0 hard snapshots met the non-oracle cliff screen.

### controlled_intervention

Every comparison copied the same real z_s, varied only the DDPM suffix RNG, and replayed the same simulator seed/action prefix. Calibration selected oracle-local locations and disjoint held-out suffixes scored them.

### interpretation

The proposed local mechanism is weakened because natural location effects did not jointly satisfy prevalence, cliff, always-early and fixed-NFE superiority gates. Any isolated gain is insufficient evidence for a deployable bottleneck-local split.

### untested_hypothesis

Whether another learned policy family, task, or substantially larger natural-state sample has a different branchability structure remains untested and is not inferred here.

## Failure cases

All negative-control snapshot IDs and complete raw genealogies are retained. Flat curves, early-is-best curves, disagreement/outcome decoupling and any multiple-cliff cases are included rather than filtered out.

Pilot completion does not authorize Stage-2B or VLA expansion.
