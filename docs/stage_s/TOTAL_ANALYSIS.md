# Stage-S total analysis contract

`r142_stage_s.total_analysis.analyze_stage_s` is the fail-closed boundary for the completed A/B/C substrate screen. It is deliberately separate from the per-arm producers and never treats a queued/running/partial artifact as an evaluation result.

## Inputs

- One canonical frozen `protocol/FROZEN_PROTOCOL.json`. The production CLI
  validates it with both the signed frozen-protocol envelope and the S4/S5
  `ProtocolAuthority`, including `PROTOCOL.md`, B/C calibration hashes, the
  frozen status, and the exact budget/threshold summary.
- A positive-control report with `overall_verdict=CONTROLS_PASS` and one
  full-length pipeline commit. Controls remain compatible with the existing
  marker schema, but production accepts them only from an audited directory.
- Exactly three production arms (`A`, `B`, `C`). The CLI accepts only
  directory references (`main_root`, `s4_root`, `s5_root`); raw records,
  probes, extended rows, and `artifact_verification` overrides are rejected.
- Each main root must contain a terminal `COMPLETED_EVALUATION_RESULT.json`
  with `status=COMPLETED`, an exact root `SHA256SUMS` coverage set, eight
  ranks, the frozen 10-task x 16-initial-state x 32-candidate grid, and the
  correct task/state/rank/world-size binding.
- Each persisted family must carry an immutable completion marker, a SHA256 manifest, 32 terminal candidates, full genealogy, terminal actions/poses, full replay snapshot, and policy/environment compute counts.
- Every candidate genealogy must explicitly bind its root family (or the
  producer's equivalent `family_id` root alias) and is checked against its
  raw candidate ID, root, order, generation, seed, action prefix, and
  final-success label. Every
  candidate snapshot must contain all replay components and Python/NumPy/
  Torch CPU/CUDA RNG streams, plus a passing restore-to-same-action check with
  error at most `1e-9`.
- S4 must cover exactly every near-all-fail family with the authority-owned nine-point search grid, four search branches per location, and equal eight/eight held-out oracle/random branches.
- S4 receives the accepted main family objects and therefore binds its marker
  to the exact source family SHA values. S5 is loaded through
  `load_s5_extended`, binding immutable main base rows/source SHAs and checking
  the complete 64-candidate extension, termination, trajectory/action, and
  replay evidence.

The Python API retains a non-production, explicitly verified in-memory mode
for fixture/legacy callers. It is not reachable through the production CLI;
`production=True` rejects every in-memory artifact and verification override.

## Frozen production metrics

S3 uses the substrate pose contract (`A`: 14 components; `B/C`: six components), and derives `tau(t)` from successful same-task, matched-control-step episode pairs at the 95th percentile. A scalar `tau`, external tau curve, or workspace-bound override is rejected by the production entry point.

S4 reads bootstrap seed `14211` and 10,000 paired-bootstrap replicates from the authority. The search grid has exactly nine points; the held-out oracle and random arms each have exactly eight branches. S5 compares base N=32 with fresh indices 32--63.

## Evaluation and decision

The analyzer computes S1, S2, S3, S4, and S5 for all three arms before applying any gate or decision. A malformed arm is represented by five explicit `INVALID_INPUT` gate records; other arms are still evaluated. Pipeline identity is compared against the controls commit after all arm computations.

The report contains one `decision_code`, `decision_code_count=1`, all arm gates, artifact verification summaries, protocol metric contracts, and any verification errors. `C` is mechanism-isolation evidence only and can produce `WEAK_SUBSTRATE_ONLY`; it can never establish headline `SUBSTRATE_QUALIFIED`.

## CLI

```bash
PYTHONPATH=src python scripts/stage_s_total_analysis.py \
  --protocol /path/to/FROZEN_PROTOCOL.json \
  --controls /path/to/controls \
  --arms-json /path/to/arms.json \
  --output-root /path/to/total-analysis
```

`arms.json` must contain only directory references, for example:

```json
{
  "A": {"substrate": "A", "main_root": "/cpfs/A-main", "s4_root": "/cpfs/A-s4", "s5_root": "/cpfs/A-s5"},
  "B": {"substrate": "B", "main_root": "/cpfs/B-main", "s4_root": "/cpfs/B-s4", "s5_root": "/cpfs/B-s5"},
  "C": {"substrate": "C", "main_root": "/cpfs/C-main", "s4_root": "/cpfs/C-s4", "s5_root": "/cpfs/C-s5"}
}
```

On an output directory, `STAGE_S_TOTAL_ANALYSIS.json`, `COMPLETED_EVALUATION_RESULT.json`, and `SHA256SUMS` are written atomically and read back before the call returns.
