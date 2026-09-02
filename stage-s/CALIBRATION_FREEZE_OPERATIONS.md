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
3. B's completion marker names the frozen
   `r142-stage-s-b-variants-20260903-r7` bundle and all eight rank marker
   digests. C's accepted training lineage is the current
   `ACCEPTED_C_TRAINING.json` schema
   (`r142-stage-s-c-training-acceptance-v1`), with
   `status=ACCEPTED`, `label=WEAK_SUBSTRATE`, terminal PAI status
   `Succeeded`, and the pinned OpenPI
   `54cbaee6ae0c010a1ed431871cdaa8f4684ac709`, `pi05_libero`, seed 42,
   terminal step 10001, with all four exact early checkpoints. Its
   checkpoint/log manifests are rechecked from their declared roots, and
   every `checkpoint_hashes` model entry must match both the file and the
   bundle manifest. The legacy direct completion marker is supported only
   for backward-compatible fixtures and receives equivalent terminal and
   artifact checks.
4. The GitHub-backed checkout already contains the full commit that added
   `stage-s/PROTOCOL.md`. The markdown must declare S1--S5 thresholds,
   `D(t)` normalization, the same-task matched-t tau 95th percentile,
   family/near-all-fail definitions, and literal RNG/compute contracts.

## Freeze command

Run outside the Beijing no-job windows and set `PYTHONPATH=src`:

```bash
PYTHONPATH=src python scripts/stage_s_freeze_calibration.py \
  --b-result <B_RUN>/CALIBRATION_RESULT.json \
  --b-completion-marker <B_RUN>/COMPLETED_B_CALIBRATION.json \
  --c-result <C_RUN>/CALIBRATION_RESULT.json \
  --c-completion-marker <C_RUN>/COMPLETED_C_CALIBRATION.json \
  --c-lineage <ACCEPTED_C_TRAINING_LINEAGE.json> \
  --protocol-md stage-s/PROTOCOL.md \
  --protocol-git-commit "$(git rev-parse HEAD)" \
  --repo-root "$PWD"
```

The default destinations are the canonical CPFS paths shown in
`stage-s/CALIBRATION_REPORT.md`, plus
`logs/r142_fp11_stage_s/stage_s/protocol/FROZEN_PROTOCOL.json`. The tool
atomically materializes a runtime copy of `PROTOCOL.md` beside the acceptance
object because the main loader verifies that adjacent file; this copy is not
a repository protocol commit. Use `--no-materialize-protocol-md` only when
that adjacent non-symlink file already exists and has the same bytes.

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
