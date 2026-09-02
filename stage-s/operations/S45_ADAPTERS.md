# Stage-S S4/S5 substrate adapters

`src/r142_stage_s/s45_adapters.py` binds the generic S4/S5 executor to the
maintained Stage-S runtimes. It does not add a simulator, policy, or success
heuristic.

## LIBERO B/C

Create the adapter from the existing Stage-R factories:

```python
from r142_stage_s.libero import (
    make_stage_r_policy_factory,
    make_stage_r_task64_factory,
)
from r142_stage_s.s45_adapters import make_libero_s45_adapter

environment_factory = make_stage_r_task64_factory(
    qpilots_root,
    libero_root,
    checkpoint=checkpoint,
    variant_root=variant_root,              # B only
    libero_config_root=libero_config_root,  # C only
    max_steps=max_steps,
    init_state_count=16,
)
policy_factory = make_stage_r_policy_factory(qpilots_root, checkpoint)
adapter = make_libero_s45_adapter(
    environment_factory,
    policy_factory,
    protocol=protocol,
    substrate="B",  # or "C"
    max_steps=max_steps,
)
```

The adapter uses `seeded_reset`, `_sample_chunk`, `_execute_one`,
`capture_stage_r_snapshot`, and `restore_stage_r_snapshot` from `libero.py`.
It additionally requires an owner-specific environment RNG hook and a policy
RNG hook. The maintained Task64 `_observation` buffer is treated as the
policy observation-history buffer only because it is the state consumed by the
Stage-R policy; if that buffer or either restore hook is absent, execution
stops. The adapter replays the persisted anchor actions and rejects the family
if official inference diverges at any prefix step.

## RoboTwin A

The official task setup is deployment-specific, so the adapter receives
explicit callbacks:

```python
from r142_stage_s.s45_adapters import make_robotwin_s45_adapter

adapter = make_robotwin_s45_adapter(
    environment_factory=official_env_factory,
    policy_factory=official_policy_factory,
    reset_factory=official_reset_factory,
    protocol=protocol,
    max_steps=max_steps,
)
```

`environment_factory` must return the official stable_2.0 SAPIEN task env;
`policy_factory` must return a policy exposing `act`, history/queue capture and
restore, RNG capture and restore, and an explicit `set_seed`/`seed`/`set_rng`.
`reset_factory` must perform the official task reset. The adapter delegates
simulator/history/queue/all-RNG snapshot and restore to
`ConcreteRoboTwinRuntime` and checks `state_for_verification()` or `get_obs()`
under the existing 1e-9 replay contract.

The released Evo proxy alone is insufficient: it has no policy `act` contract
or complete server-side RNG restore unless the exact-replay control patch is
connected. In that case the adapter fails closed; it does not substitute local
NumPy randomness.

## Seed and location authority

Branch and extension seeds are read from the frozen protocol. The adapters
accept either explicit protocol tables (`s4.branch_seeds` /
`s5.extension_seeds`) or the restricted hash-formula grammar implemented in
`_frozen_seed`; an unknown formula fails closed. No adapter-side seed, salt,
grid, anchor, or threshold is introduced. The generic runtime still validates
the protocol's literal interior grids and equal K before any branch runs.

## Remaining integration gaps

The source repository currently does not contain a deployment-independent
official RoboTwin scene factory or policy `act` wrapper, and Stage-R's
maintained LIBERO policy wrapper does not expose a separate policy-history
accessor or owner-specific environment RNG accessor on every installation.
The concrete adapter therefore requires the exact callbacks/hooks above and
will stop with a named capability error where the installed runtime cannot
prove them. These are integration blockers, not scientific outcomes.
