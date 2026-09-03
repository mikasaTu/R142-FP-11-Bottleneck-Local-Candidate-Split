# Stage-S LIBERO replay-integrity gate

The LIBERO S4/S5 adapter does not treat an exposed observation, pose, or
`_state_vector(env)` as a sufficient exact-replay witness. For each
restore → same-action → next-state probe it captures the complete simulator
snapshot and compares every nested numeric leaf, including hidden simulator
state. The comparison also covers the Stage-R observation-history buffer, the
runner/policy action queues, Python and NumPy RNG states, Torch CPU and every
visible CUDA RNG state, and both environment- and policy-owner RNG states.

The validator first checks nested structure (mapping keys, dataclass fields,
sequence lengths, array dtypes/shapes, and nonnumeric values), then compares
numeric leaves at absolute tolerance `1e-9`. Only the explicitly documented
process-local `object` and `scene` handle fields are omitted from simulator
comparison. Missing simulator numeric leaves, a schema/path mismatch,
non-finite values, unavailable required RNGs, or an error above tolerance
raises `S45AdapterError`; no branch/fresh-candidate completion marker can be
written after such a failure.

Successful `snapshot_restore_check` records retain `component_evidence` for
each component: maximum absolute error, numeric leaf paths/count, structural
schema SHA-256, and first/second numeric SHA-256. They also retain the visible
state error and explicit validation flags for history, queue, and each RNG
stream. Thus a passing visible state vector cannot hide a replay mismatch,
and a failure can be diagnosed from the exact component and path rather than
from a single aggregate scalar.

The adversarial tests in `tests/test_stage_s_replay_integrity.py` hold the
visible state/observation fixed while changing hidden simulator state,
observation history, queued actions, Python/NumPy/Torch RNG, or an owner RNG;
all are rejected. An adapter-level test exercises hidden simulator drift
through the same-action gate.
