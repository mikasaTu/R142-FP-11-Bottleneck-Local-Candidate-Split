# Stage-S calibration freeze operations

This runbook is for the terminal evidence hand-off between B/C calibration
and the main screen. It is deliberately non-submitting: the command below
does not call PAI, run LIBERO, or inspect S2--S5 data.

## Preconditions

1. B and C calibration jobs have terminal `COMPLETED` markers. A queue,
   running process, first-work marker, or partial shard is insufficient.
2. Each `CALIBRATION_RESULT.json` has the adjacent `SHA256SUMS`, and its
   completion bundle has a verified SHA manifest. The result has only the
   four pooled row fields; any downstream field is a hard failure.
3. B and C completion markers enumerate exactly eight rank markers and
bind every marker with a full SHA-256. Both the calibration result and its
completion marker must be regular files in the same directory, with no
residual FAILED_B_CALIBRATION.json or FAILED_C_CALIBRATION.json beside them.
B still names the frozen r7 variant bundle; C uses only the current
ACCEPTED_C_TRAINING.json lineage schema.
4. The C source mapping is read from the accepted training manifest, then
matched exactly by the C calibration config evidence. All four source SHAs
must also occur in the committed PROTOCOL.md. No source SHA is hard-coded by
the freezer.
5. Each substrate has external PAI registry evidence: result.json,
submission-state.json, resolved.json, a jobs.jsonl with exactly one
controller-run-to-JobId row, and a successful terminal GetJob JSON plus its
sha256sum sidecar. The resolved write paths must include the calibration
artifact. A restarted B controller additionally needs its
controller-incarnations/<controller-run-id>.json edge; C must prove its
same-run controller/application identity.
6. The GitHub-backed checkout already contains the full commit that added
stage-s/PROTOCOL.md. The markdown must declare S1--S5 thresholds,
D(t) normalization, the same-task matched-t tau 95th percentile,
family/near-all-fail definitions, literal RNG/compute contracts, and the four
accepted C source SHAs.
## Freeze command

Run outside the Beijing no-job windows and set `PYTHONPATH=src`:

```bash
PYTHONPATH=src python scripts/stage_s_freeze_calibration.py \
  --b-result <B_RUN>/CALIBRATION_RESULT.json \
  --b-completion-marker <B_RUN>/COMPLETED_B_CALIBRATION.json \
  --c-result <C_RUN>/CALIBRATION_RESULT.json \
  --c-completion-marker <C_RUN>/COMPLETED_C_CALIBRATION.json \
  --c-lineage <ACCEPTED_C_TRAINING.json> \
  --c-config configs/pai/stage_s_c_calibration.json \
  --b-registry-run <B_CONTROLLER_RUN_DIR> \
  --b-jobs-ledger <B_CONTROLLER_RUN_DIR>/jobs.jsonl \
  --b-getjob-terminal <B_CONTROLLER_RUN_DIR>/getjob-terminal.json \
  --b-getjob-terminal-sha <B_CONTROLLER_RUN_DIR>/getjob-terminal.json.sha256 \
  --c-registry-run <C_CONTROLLER_RUN_DIR> \
  --c-jobs-ledger <C_CONTROLLER_RUN_DIR>/jobs.jsonl \
  --c-getjob-terminal <C_CONTROLLER_RUN_DIR>/getjob-terminal.json \
  --c-getjob-terminal-sha <C_CONTROLLER_RUN_DIR>/getjob-terminal.json.sha256 \
  --protocol-md stage-s/PROTOCOL.md \
  --protocol-git-commit <40-hex-protocol-commit> \
  --repo-root "$PWD"
```

The default destinations are the canonical CPFS paths shown in
`stage-s/CALIBRATION_REPORT.md`, plus
`logs/r142_fp11_stage_s/stage_s/protocol/FROZEN_PROTOCOL.json`. The tool
atomically materializes a runtime copy of `PROTOCOL.md` beside the acceptance
object because the main loader verifies that adjacent file; this copy is not
a repository protocol commit. Use `--no-materialize-protocol-md` only when
that adjacent non-symlink file already exists and has the same bytes. An existing
FROZEN_PROTOCOL.json is never overwritten; start a new evidence directory.

## Read-back checks

After the command, run `sha256sum -c SHA256SUMS` in each source bundle and
re-run the B/C main loader's frozen-report/protocol read gate. The acceptance
object binds the exact report SHA, selected B setting or C checkpoint and
artifact SHA, protocol commit, protocol markdown SHA, and the literal frozen
summary. Changing any source result, marker, selected artifact, report, or
markdown invalidates the acceptance object; rerun the freezer only from a new
terminal evidence set. This branch intentionally contains no current result
JSON and does not submit or resume any PAI job. Rechecking the complete C
checkpoint bundle is intentionally I/O-heavy for a real model directory; that
cost is part of the fail-closed lineage gate rather than a scientific result.
