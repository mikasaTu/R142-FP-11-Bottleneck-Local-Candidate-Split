# R142-FP-11 Bottleneck-Local Candidate Split

Reproducible Stage-1 validation of candidate collapse and bottleneck-local
candidate splitting in a small synthetic 2D manipulation environment. This
repository deliberately does not use a VLA.

The frozen protocol is in `docs/PREREGISTERED_PROTOCOL.md`. The benchmark has
two symmetric feasible modes, a shared early prefix, a randomized intermediate
decision point, and a correlated central failure basin. Every candidate keeps a
complete genealogy.

## Stage-2A learned-policy falsification

Stage-2A is an independent follow-up on the official learned LeRobot Diffusion
Policy and unmodified standard PushT. It does not reuse the artificial Stage-1
bottleneck. The frozen protocol and exact source interface are documented in
`docs/STAGE2A_PREREGISTERED_PROTOCOL.md` and
`docs/STAGE2A_SOURCE_INTERFACE_AUDIT.md`.

The formal workflow requires the pinned LeRobot checkout and checkpoint listed
in the protocol. A local one-GPU engineering gate is:

```bash
PYTHONPATH=/path/to/lerobot:src \
python scripts/stage2a_validate.py \
  --checkpoint /path/to/lerobot_diffusion_pusht_84a7c231 \
  --output outputs/stage2a_gates \
  --device cuda \
  --mode gates
```

The PAI Efficiency template is
`pai/r142_stage2a_formal_evaluation_efficiency_2gpu.json`. It runs all 50 frozen
baseline episodes, descendant-blind natural snapshot selection, K=8/M=8 real
DDPM genealogies at 16 checkpoints, calibration/held-out suffix streams,
negative controls and actual fixed-sample-NFE comparisons. Completion requires
`COMPLETED_EVALUATION_RESULT.json`; queueing, `Running`, first work, or one
completed snapshot is not a scientific result.

Stage-2A is now complete. The final decision is
`R142_FP11_CORE_HYPOTHESIS_WEAKENED`: 0/24 natural snapshots met the frozen
location-sensitivity threshold and cross-fitted oracle-local branching did not
meaningfully beat always-early, random or uniform branching at a 7200
sample-NFE cap. The full report is `reports/STAGE2A_EXPERIMENT_REPORT.md`, the
code-level mechanism audit is
`reports/STAGE2A_MECHANISM_REVERSE_EXPLANATION.md`, and all persisted formal
artifacts are under `results/`. Per the preregistered gate, no VLA expansion is
authorized.

## Step plans and reports

The original Feishu idea/hypothesis, top-level experiment plan, step snapshots,
frozen executable protocols, corresponding reports, publication XML, source
revisions, and SHA-256 manifest for Step 1 and Step 2 are grouped under
[`docs/steps/`](docs/steps/README.md).

## Stage-R trajectory-axis revalidation

Stage-R re-tested the hypothesis on trajectory control steps with eventual
episode success. The frozen plan is
[`docs/steps/step3/PLAN.md`](docs/steps/step3/PLAN.md), the complete result is
[`reports/stage_r/PHASE0R_REPORT.md`](reports/stage_r/PHASE0R_REPORT.md), and
the plan-by-plan closure audit is
[`reports/stage_r/PLAN_COMPLETION_AUDIT.md`](reports/stage_r/PLAN_COMPLETION_AUDIT.md).

Phase E passed all engineering gates. Phase-0R completed 20,480 LIBERO
rollouts and retained zero tasks; RoboTwin is explicitly reported as
`SOURCE_LIMITATION_UNVERIFIABLE`. The frozen stopping rule therefore produced
`CHECKPOINT_1_STOP` and `phase1_authorized=false`. Phase-1R was conditional on
an approved retained task, so starting it would violate the preregistered
protocol. All 40 NPZ files, all 40 metadata files, manifests, reports, negative
controls, failure records, and mechanism analysis are archived under
[`results/stage_r/`](results/stage_r/) and [`reports/stage_r/`](reports/stage_r/).

## Local CPU smoke

```bash
PYTHONPATH=src python3 -m pytest -q
python3 scripts/run_experiment.py \
  --output outputs/cpu_smoke/shard0 \
  --episodes 8
python3 scripts/run_experiment.py \
  --output outputs/cpu_smoke/aggregate \
  --episodes 8 \
  --aggregate outputs/cpu_smoke/shard0
python3 scripts/render_artifacts.py \
  --results outputs/cpu_smoke/aggregate \
  --output outputs/cpu_smoke/figures
```

Smoke output is functional evidence only. It cannot pass the 10-block
scientific gate.

## Formal evaluation

The PAI launcher runs 400 paired seeds in eight resumable shards and writes a
durable first-work record only after one complete shard. It aggregates only
after all shards have atomic `COMPLETE` markers.

```bash
python3 scripts/build_runtime_manifest.py
python3 scripts/verify_runtime_manifest.py --manifest evidence/runtime_manifest.json
```

The formal template is `pai/r142_stage1_formal_evaluation_idle8.json` and the
foreground command is `pai/run_formal_evaluation.sh`. The formal PAI job is an
evaluation workload, not training; idle restart recovery is shard-based and no
checkpoint/training claim is made.

## Evidence boundary

Passing the gate means the toy mechanism is supported under this controlled
benchmark. It does not establish usefulness for a learned policy or VLA.
