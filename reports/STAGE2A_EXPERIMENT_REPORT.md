# R142-FP-11 Stage-2A experiment report

Date: 2026-08-24

Final decision: `R142_FP11_CORE_HYPOTHESIS_WEAKENED`

VLA expansion: stopped by preregistered gate

## Executive result

The official learned `lerobot/diffusion_pusht` policy supports faithful tracing,
real intermediate-state resume and complete candidate genealogy. Nevertheless,
the proposed scientific mechanism did not survive the independent falsification:
0/24 natural snapshots showed the required location-sensitive branchability,
0/8 naturally hard snapshots showed a recoverability cliff, and a cross-fitted
oracle location did not meaningfully outperform always-early, uniform or random
branching at fixed sample-NFE.

## Frozen benchmark and lineage

- Unmodified standard `gym_pusht/PushT-v0`; no artificial obstacles, routes or
  bottleneck labels.
- Official LeRobot commit:
  `3c0a209f9fac4d2a57617e686a7f2a2309144ba2`.
- Checkpoint: `lerobot/diffusion_pusht`, revision
  `84a7c23178445c6bbf7e1a884ff497017910f653`.
- `model.safetensors` SHA-256:
  `995d14d35db57d95c35ad9704c3d79c8612b7bc45f3877e5c46c2cdc516856a8`.
- Python 3.10.20, torch 2.6.0+cu124, torchvision 0.21.0+cu124,
  diffusers 0.32.2, gymnasium 0.29.1, gym-pusht 0.1.5, pymunk 6.11.0.
- Frozen inference: DDPM, 100 inference steps, horizon 16, two observation
  steps and eight executed action steps.

The exact source interface is recorded in
`docs/STAGE2A_SOURCE_INTERFACE_AUDIT.md`; the descendant-blind protocol is in
`docs/STAGE2A_PREREGISTERED_PROTOCOL.md`.

## Engineering gates A–E

All gates passed before descendants were generated:

| Gate | Requirement | Result |
|---|---|---|
| A | instrumentation disabled delegates to frozen original | bit-identical |
| B | passive tracing changes no tensor/output | bit-identical, 100 real steps |
| C | saved `z_s` + same RNG reproduces suffix | passed |
| D | same `z_s` + new suffix RNG changes output | passed; max abs action difference 0.0433 in exact smoke |
| E | simulator seed + complete prefix replay + same chunk | passed; native-state error 0 |

No surrogate genealogy or artificial simulator reset was used.

## Baseline reproduction and natural snapshot frame

The frozen baseline ran seeds 0–49. Success rate was 0.66 (33/50), mean maximum
coverage/progress was 0.994274, shared-batch model NFE was 3800 and wall time was
71.91 s. Every trajectory, action, chunk boundary, seed, progress, success and
snapshot candidate was retained.

Snapshots were selected before descendant outcomes by deterministic SHA-256
rank from control steps {50,100,150,200,250}: eight easy, eight ambiguous and
eight naturally hard snapshots, 24 total. No terminal label was used to select
a favorable bottleneck.

## Counterfactual genealogy

For each snapshot the experiment used K=8 root diffusion trajectories, 16
frozen real DDPM checkpoints and M=8 suffixes per root/checkpoint, with disjoint
calibration and held-out RNG streams. This produced exactly 49,344 genealogy
records (24,672 per stream). Every record includes ancestry, real scheduler
position, RNG/seed, normalized and unnormalized actions, progress, block motion,
contacts, support diversity, NFE and timing.

## Main quantitative results

| Quantity | Result | Gate/interpretation |
|---|---:|---|
| Location-sensitive snapshots | 0/24 (0%) | required spread-range ≥0.10 |
| Hard-state recoverability cliffs | 0/8 (0%) | required ≥6 and ≥30% |
| Maximum location spread-range | 0.040996 | below 0.10 |
| Median location spread-range | 0.000562 | effectively flat |
| Mean within-snapshot disagreement/branchability correlation | −0.0286 | no predictive coupling |
| Pooled `z_s` disagreement/progress-spread correlation | −0.0031 | decoupled |
| Always-early equivalent to oracle | true | kill condition met |

Fixed sample-NFE cap was 7200 per snapshot:

| Strategy | Mean NFE | Mean suffixes | Mean gain over no-branch | Mean wall time (s) |
|---|---:|---:|---:|---:|
| Always early | 7200 | 64.0 | 0.009655 | 25.18 |
| Random location | 7052.7 | 748.3 | 0.012011 | 239.74 |
| Uniform three quantiles | 6728 | 216.0 | 0.012364 | 83.28 |
| Oracle local, cross-fit | 7090 | 97.7 | 0.011913 | 33.71 |

Paired cross-fitted oracle-local differences:

| Comparison | Mean difference | Paired bootstrap 95% CI |
|---|---:|---:|
| oracle − always early | +0.002258 | [0.000345, 0.004919] |
| oracle − random | −0.000099 | [−0.002204, 0.002766] |
| oracle − uniform | −0.000451 | [−0.001419, 0.000361] |

The only positive difference is two orders of magnitude below the required
+0.10. It is insufficient for Stage-2B.

## Eventual continuation subset

The descendant-blind representative subset used two lexicographically first
snapshot IDs per stratum, root indices {0,7}, checkpoints {0,53,99}, and suffix
0: 36 cases across six snapshots. Each branch chunk was executed and then the
frozen baseline policy continued to episode termination.

| Checkpoint | Eventual success | Mean max progress | Wilson 95% CI |
|---:|---:|---:|---:|
| 0 | 7/12 (58.3%) | 0.984820 | [0.320, 0.807] |
| 53 | 8/12 (66.7%) | 0.987764 | [0.391, 0.862] |
| 99 | 6/12 (50.0%) | 0.980311 | [0.254, 0.746] |

The intervals overlap widely. This subset is descriptive, small, and not a
fixed-NFE superiority test; it cannot rescue the failed main gate.

## Negative controls and failure cases

- no-bottleneck: 24/24 snapshots;
- smooth-decay: 8/24;
- disagreement/outcome decoupled: 17/24;
- fake disagreement: 11/24;
- valid natural cliffs: none;
- multiple cliffs: none.
- silent bottlenecks under the frozen classifier: none.

Some observations had high descendant spread at all locations, while the
within-snapshot location range stayed small. These are state-dependent task
uncertainty, not generation-local bottlenecks. Flat cases and all raw negative
examples are retained, not filtered.

## Mechanism explanation

The latent diffusion trajectory visibly contracts: median `z_s` disagreement
falls from 1.400 at checkpoint 0 to 0.038 at 99. The executed-action/outcome
support does not contract in the same way; median progress spread remains about
0.0018–0.0019. Raw latent variation can lie outside the eight executed actions
or in behaviorally insensitive directions. Later branching also exchanges
remaining stochastic transitions for more suffix draws under fixed NFE. Thus
latent collapse exists, but the specific generation-local recoverability
collapse does not.

Oracle-local's +0.002258 over always-early is consistent with modest
compute-allocation/order-statistic benefit from later, cheaper suffixes. It does
not beat random or uniform; with an almost flat location signal those baselines
lose little, generate more suffixes and uniform hedges across three regions.
The tiny negative differences are statistically unresolved, not evidence that
oracle location is reliably harmful. Full reasoning and evidence categories are
in `reports/STAGE2A_MECHANISM_REVERSE_EXPLANATION.md`.

## PAI execution evidence

- Main formal run: `r142-stage2a-formal-20260823-r2`, job
  `dlcm4saves6zi30f`, exact terminal `Succeeded`, 2×A800 Efficiency pool.
- Eventual continuation: `r142-stage2a-continuation-20260824-r1`, job
  `dlcs1330q3a1eijb`, exact terminal `Succeeded`, 2×A800 Efficiency pool.
- Failed predecessor `dlc1xar28asuv34o` failed before first work because the
  exact runtime lacked `pip`; no scientific record came from it. Its CPFS and
  registry evidence was preserved and only its failed PAI service row was
  deleted with two-phase absence verification.
- Both successful outputs have persisted completion records, complete SHA-256
  verification and uid/gid 2254 ownership evidence. `Running` or first-work
  milestones were not treated as completion.

Compute accounting is kept in the unit used by each stage rather than mixing
batched model calls with per-sample work. The discovery trees consumed
2,501,376 sample-NFE, including eight 100-step roots and both 51,712-NFE suffix
streams for every snapshot. The four fixed-NFE strategies consumed 673,696
sample-NFE in total with discrete slack retained per strategy/snapshot. Baseline
reproduction used 3,800 shared-batch model NFE; eventual continuation used
53,200 model NFE. PAI reported 16,562 s for the main 2-GPU allocation (9.201
allocated GPU-hours) and 257 s for continuation (0.143 allocated GPU-hours).
Observed peak allocated tensor memory was 1.115 GiB in discovery and 1.071 GiB
in continuation; these are model-process peaks, not total device reservation.

## Strict evidence categories

**Observed fact.** All numerical tables above are persisted outputs from the
frozen official policy and standard simulator.

**Controlled intervention.** Each comparison copied the same real `z_s`, held
conditioning, simulator seed and action prefix fixed, changed only suffix RNG,
and used disjoint calibration/held-out streams.

**Interpretation.** Branchability is observation-dependent but not meaningfully
generation-location-dependent in this policy/task, so R142-FP-11's minimal
learned-policy mechanism is weakened.

**Untested hypothesis.** Other learned policy families, tasks, larger natural
snapshot frames and VLA-scale models may differ; this experiment makes no claim
about them.

## Final decision

The preregistered kill conditions are met. The formal result is
`R142_FP11_CORE_HYPOTHESIS_WEAKENED`. Do not enter Stage-2B detector design and
do not enter VLA/π0.5 validation on the basis of this idea.
