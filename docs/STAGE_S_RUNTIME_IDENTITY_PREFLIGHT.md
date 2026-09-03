# Stage-S runtime identity preflight

This runbook defines the read-only identity check for the Stage-S A/B/C main
screens and C calibration.  It is an admission check, not a training or
evaluation command.  It never repairs a checkout, mutates a payload, creates
a CPFS directory, downloads a checkpoint, or submits a PAI job.

## Invocation

From a clean Stage-S checkout:

```bash
PYTHONPATH=src python3 scripts/stage_s_runtime_identity_preflight.py \
  --config configs/pai/stage_s_robotwin_a.json \
  --output /mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_s/stage_s/runtime_identity/a.json
```

The output path is intentionally mandatory for an attestation.  Omitting it
returns exit code 2 and writes nothing.  With an explicit path, the command
writes a deterministic JSON record even when the result is `REFUSED`; a
`PASS` result returns 0, while any mismatch returns 1.

## Checks and fail-closed behavior

The check binds the config to a clean config Git source (or to an explicitly
recorded `config_source_commit`), then verifies the configured runtime checkout
at its exact `runtime.source_commit` and a clean tree.  Dependency bindings may
use the final-config form `runtime.dependencies.{name}.path/commit` or the
existing Stage-S `evidence.source_provenance` roots with sibling commit pins.
Every bound dependency must be a clean Git checkout at the expected commit.

The deployed launcher is read from the exact configured `runtime.command_file`.
Its observed SHA-256 must match both `command_file_sha256` and
`payload_sha256` when both are present.  Symlinked or missing launchers are
refused before a hash is accepted.

Model, checkpoint, protocol, acceptance, and calibration-report authorities
must exist as regular files/directories and have a valid adjacent
`SHA256SUMS` manifest.  Manifest paths are GNU-compatible relative paths; path
traversal, duplicate entries, symlinked entries, missing files, and digest
mismatches all fail closed.  A `FROZEN_PROTOCOL.json` also has to validate the
SHA-256 recorded for its adjacent `PROTOCOL.md`.

The resource contract is exact: one worker, 8 GPUs, 88 CPU cores, and
1400 GiB memory plus 1400 GiB shared memory.  Any explicit contradictory
resource, compute-contract, or authorization field is refused.  Optional pool
identity fields are checked when present, including the robot idle pool and
`AcceptQuotaOverSold`.

## Evidence discipline

The JSON record contains each observed source head/clean flag, launcher digest,
artifact-manifest result, dependency identity, normalized resource values, and
sorted deterministic error strings.  It contains no timestamps, no secret
environment values, and no command output.  Queueing, `Running`,
`FIRST_WORK.json`, partial shards, or a successful import smoke cannot turn a
`REFUSED` identity record into a scientific completion result; the ordinary
Stage-S completion markers and PAI terminal-state evidence remain required.

## Final-config requirements

Before repinning the final A/B/C configs, add explicit dependency path/commit
bindings for every source commit that is currently only recorded as a bare
field.  Keep the actual deployed command file path and its two SHA fields in
the same config.  Add adjacent manifests for the protocol, selected model or
checkpoint, accepted asset/training authority, and selected calibration
report.  Re-run this preflight after every payload or config change and retain
the JSON beside the corresponding PAI evidence; this tool itself does not
alter any of those artifacts.
