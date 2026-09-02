# Stage-S S4/S5 execution contract

This runbook describes the independent execution layer in
`src/r142_stage_s/s45_runtime.py`. It is deliberately not a synthetic
benchmark. The caller must bind an official LIBERO or RoboTwin simulator and
the exact policy/checkpoint used for the accepted N=32 screen.

## Preconditions

1. `FROZEN_PROTOCOL.json` must be the machine-readable authority committed by
   Stage-S step 1. It must contain literal `s4` and `s5` objects. S4 must
   declare `anchor_rule`, `oracle_t_rule`, `random_t_rule`, a nine-entry
   interior search grid, `search_branch_count=4`,
   `heldout_branch_count=8`, `branch_seed_formula`, and a frozen
   `random_location_hash_formula` (or an explicitly matching published
   random grid). S5 must declare `base_candidate_count=32`, the exact
   `fresh_candidate_indices=[32,...,63]`, and `extension_seed_formula`. The
   runtime supplies no defaults for these values; a missing field is a
   fail-closed protocol error.
2. `discover_n32_families()` recursively reads every
   `COMPLETED_FAMILY.json` under the N=32 root. Each marker, listed artifact,
   and `SHA256SUMS` entry is verified. Every family must have exactly 32
   persisted candidates, eventual terminal success, raw actions and raw
   trajectory, official termination metadata, unique candidate ids/seeds,
   the substrate pose contract (A=14D; B/C=6D), and the protocol authority
   identity. NumPy wrappers are decoded and shape-checked without replacing
   the source file. A subset is never accepted.
3. S4 requires at least one near-all-fail family (`successes <= 1/32`). The
   runtime includes *all* such families in the denominator. S5 runs on all
   accepted N=32 families, not only the near-all-fail subset.

## Real adapter interface

Pass `--adapter module:factory` to `scripts/stage_s_s45.py`. The factory must
return `r142_stage_s.s45_runtime.S45Adapter`. The abstract methods are
intentional hard failures:

| Hook | Required evidence |
| --- | --- |
| `select_anchor(family, protocol=...)` | A candidate already present in the source N=32 family, selected under the frozen `anchor_rule`. |
| `replay_prefix(family, anchor, split_step, protocol=...)` | Real replay through the control step and a persisted snapshot containing simulator state, policy observation history, action queue, and all RNG streams. |
| `branch_seed(...)` | Integer derived from the literal frozen `branch_seed_formula`; search and held-out pair modes must be distinguishable, with no runtime default. |
| `run_branch(...)` | Restore the snapshot, preserve the prefix, replace only the suffix seed, run to official termination, and return full actions/trajectory, eventual success, policy-forward count, env-step count, and `snapshot_restore_check`. |
| `extension_seed(family, candidate_index, protocol=...)` | Integer derived from the literal frozen S5 formula. |
| `run_fresh_candidate(...)` | Candidate 32..63 from the same task/state/policy/checkpoint, to official termination, with full actions/trajectory/success/compute and passing replay check. |

`snapshot_restore_check` must contain `same_action=true`, `passed=true`, and
`max_abs_error <= 1e-9`. The runtime computes `prefix_preserving` by comparing
the branch's recorded prefix with the replayed prefix; an adapter-supplied
boolean is ignored. The branch's success is read from the returned terminal
execution record and cannot be supplied as a probe-level shortcut.

## S4 output and finalization

For each near-all-fail family, S4 first executes all nine search locations with
exactly four branches per location. The location with the largest number of
prefix-preserving successes is selected, with numerical `t` as the earliest
tie-break. This search result is persisted in `S4_SEARCH.json` and in the
probe; it is not reused as held-out evidence. At the selected location, eight
held-out oracle suffixes are paired with eight suffixes at the protocol's
hash-selected random locations. Each pair receives the identical suffix seed.
S4 writes raw search and held-out branches to `S4_BRANCHES.jsonl` and
`S4_PROBE.json`, then atomically writes `COMPLETED_S4_FAMILY.json` and
`SHA256SUMS`. All locations are strict interior control steps (`0 < t < H-1`).
The finalizer calls the analysis with exactly 10,000 paired bootstrap
replicates only after checking every near-all-fail family, all 9x4 search
branches, and all 8+8 held-out branches.

## S5 output and finalization

For every accepted family, S5 keeps the source base rows byte-for-byte in the
logical payload and adds exactly candidate indices 32..63 with disjoint fresh
seeds. `S5_FAMILY.json` includes base/fresh/extended rows, complete genealogy,
trajectory, eventual success and compute fields, the canonical base digest,
and the source family/marker/bundle SHA-256 values. It is followed by
`COMPLETED_S5_FAMILY.json` and `SHA256SUMS`. The finalizer rejects missing
rows, rewritten base rows, index overlap, seed/id overlap, source/protocol
mismatch, or incomplete terminal records before calling `compute_s5`.

## Commands

```bash
PYTHONPATH=src python scripts/stage_s_s45.py \
  --phase both --substrate B \
  --protocol /path/to/FROZEN_PROTOCOL.json \
  --n32-root /path/to/n32/B \
  --output-root /path/to/s45/B \
  --adapter official_stage_s_adapter:build_adapter

PYTHONPATH=src python scripts/stage_s_s45_finalize.py \
  --substrate B \
  --protocol /path/to/FROZEN_PROTOCOL.json \
  --n32-root /path/to/n32/B \
  --s4-root /path/to/s45/B/s4 \
  --s5-root /path/to/s45/B/s5 \
  --output-root /path/to/s45/B/evaluation
```

The CLI has no synthetic mode. Queued/running shards and partial directories
are not results; downstream publication may use only the verified
`COMPLETED_EVALUATION_RESULT.json` and its manifest.
