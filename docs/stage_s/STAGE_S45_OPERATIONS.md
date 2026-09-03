# Stage-S S4/S5 production runbook

This runbook is an operational contract, not a scientific parameter source.
S4/S5 remain bound to the frozen Stage-S authority and to the accepted N=32
producer bundles.  No VLA, synthetic rollout, expert trajectory, fake adapter,
or typed success boolean may be substituted.

## Before freeze

`configs/pai/stage_s_s4.json` and `configs/pai/stage_s_s5.json` are deliberately
`IMPLEMENTED_NOT_SUBMITTED` templates.  Every
`__REQUIRED_AFTER_FREEZE_*__` value must remain unresolved until the live
calibration/protocol/checkpoint/source readback is complete.  The registry
wrapper must reject the templates in this state:

```bash
python3 scripts/validate_stage_s45_pai_contract.py \
  --config configs/pai/stage_s_s4.json
```

The expected result before freeze is `STAGE_S45_PAI_CONTRACT_REJECTED`.

## Freeze and repin

After the canonical B/C calibration reports and protocol are accepted, make a
new immutable run snapshot and repin *all* of the following in both configs:

1. the clean runtime source commit and complete source-tree SHA;
2. the S4/S5 launcher, decoded foreground payload and finalizer SHA;
3. `FROZEN_PROTOCOL.json`, adjacent `PROTOCOL.md`, and `SHA256SUMS`;
4. both B and C calibration report paths and SHA-256 values (the registry
   `runtime.calibration_reports.B/C` entries are mandatory);
5. the actual substrate checkpoint/model path and SHA;
6. every executable dependency path, commit and SHA;
7. N32, S4, S5, output roots and numeric `2254:2254` ownership probes;
8. the exact substrate (`A`, `B`, or `C`) and main source commit/SHA.

Do not copy a hash from an earlier run, a prose note, or a mutable symlink.
Re-run the validator after all replacements.  It must also be followed by the
canonical registry `pai-job validate`; only then may `submit_stage_s_s45.sh`
call `pai-job submit`.  A changed source, protocol, calibration, checkpoint,
dependency, adapter, finalizer, or output path requires a fresh freeze and
repin, never an in-place resume under the old payload identity.

## Exact scientific contract

The configs and runtime enforce the existing plan without changing it:

* source N=32 candidates;
* S4 nine search locations × four search branches;
* held-out paired oracle/random branches are 8/8;
* paired bootstrap seed `14211`, replicates `10000`;
* S5 fresh candidates are exactly `32..63` (fresh32);
* accepted producer genealogy, action prefix, terminal label, full snapshot,
  all Python/NumPy/Torch CPU/CUDA/environment/policy RNG streams, and
  restore-to-same-action error `<= 1e-9` are mandatory.

`discover_n32_families` reads every complete family below the pinned root and
does not select a convenient subset.  It rejects duplicate/out-of-order
parents, root drift, seed collisions, action-prefix rewrites, incomplete
snapshots, mismatched source SHA, and unregistered bundle files.

## Real adapter boundary

The `module:factory` must construct the maintained environment adapter:

* substrate A: concrete `RoboTwinS45Adapter` and official RoboTwin/Evo hooks;
* substrates B/C: concrete `LiberoS45Adapter` and official LIBERO/policy hooks.

Missing simulator, policy history, action queue, snapshot/restore, RNG or
official terminal hooks fail before a completion marker.  Production loading
rejects module/class names containing `fake`, `synthetic`, `mock`, or
`fixture`, and rejects the abstract base adapter.

## Idle worker and blackout windows

Both registry templates request exactly one worker with 8 A800 GPUs, 88 CPU
cores, 1400 GiB memory and 1400 GiB shared memory on
`idle-a800-robot-stage-s-graphics-8gpu` / `quota1ssrabud0bh`,
`AcceptQuotaOverSold`, `robot_idle`, and Sync OnFailure with at most 50
platform restarts.  This is queue-admission permission; actual idle placement
must be proven by exact-JobId readback (`UseOversoldResource=true`).

No task may be submitted or resumed during 09:30–09:40 or 19:30–19:40
Asia/Shanghai.  Stop the exact run before 09:30/19:30 and resume the same
directory/run only after 09:40/19:40.  Queueing, Running, partial shards, or
log activity is not completion evidence.

## Completion and recovery

Use exact JobId monitoring and persist the earliest decisive error.  A spot
eviction without an application error is scheduling; restart the same frozen
run directory and payload.  An application failure requires a new run ID only
after the cause is repaired and the same source/protocol/checkpoint contract is
revalidated.  Never overwrite a valid completion marker.  Finalization is
valid only after `COMPLETED_EVALUATION_RESULT.json` and its closed
`SHA256SUMS` verify, and after the result is cross-bound to the expected
substrate and main source SHA.
