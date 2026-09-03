# Stage-S total analysis contract

`r142_stage_s.total_analysis.analyze_stage_s` is the fail-closed boundary for the completed A/B/C substrate screen. It is deliberately separate from the per-arm producers and never treats a queued/running/partial artifact as an evaluation result.

## Inputs

- One frozen `FROZEN_PROTOCOL.json`, loaded by `ProtocolAuthority`.
- A positive-control report with `overall_verdict=CONTROLS_PASS` and one pipeline commit.
- Exactly three arms (`A`, `B`, `C`). Each arm must provide either verified in-memory rows or a persisted completion tree.
- Each persisted family must carry an immutable completion marker, a SHA256 manifest, 32 terminal candidates, full genealogy, terminal actions/poses, full replay snapshot, and policy/environment compute counts.
- S4 must cover exactly every near-all-fail family with the authority-owned nine-point search grid, four search branches per location, and equal eight/eight held-out oracle/random branches.
- S5 must bind the fresh candidate extension to indices 32--63.

In-memory handoffs are accepted only with explicit `artifact_verification` flags for terminal markers, SHA256, genealogy, and compute. This keeps a Python list of booleans from becoming a substitute for an audited disk bundle.

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

On an output directory, `STAGE_S_TOTAL_ANALYSIS.json`, `COMPLETED_EVALUATION_RESULT.json`, and `SHA256SUMS` are written atomically and read back before the call returns.
