# Stage-S C training acceptance

`r142_stage_s.c_training_acceptance` is the terminal, fail-closed admission
gate for the deliberately undertrained C substrate.  It is an evidence
consumer, not a launcher: it never calls PAI, resumes a job, imports Torch, or
modifies a run, status, data, or checkpoint directory.  The published record
is always labelled `WEAK_SUBSTRATE` and therefore cannot be interpreted as a
full-training or model-quality claim.

## Required evidence

The command must receive one exact registry lineage and the three CPFS roots
belonging to that lineage:

* a registry run directory containing `result.json`, `submission-state.json`,
  and `resolved.json`, plus the jobs JSONL ledger;
* a sanitized terminal `GetJob` JSON and its one-line SHA-256 sidecar;
* the C log root, C status root, and common checkpoint root.

The registry evidence must bind exactly one run ID to one unique PAI JobId,
with `submission_state=submitted_verified`, a terminal sanitized GetJob
`Status=Succeeded`, and the resolved runtime write paths covering all three
CPFS roots.  Raw commands, environment values, or base64 payloads in a
GetJob record are refused.

The gate then re-hashes every referenced artifact.  It requires the pinned
Stage-S/QPILOTS/OpenPI/LIBERO source checkouts and deployed payload, the
official immutable LIBERO revision and dataset manifest, staged norm stats,
and the audited LeRobot compatibility mode.  It requires seed 42, world size
8, terminal global step 10001, and the native C checkpoint schedule
1000/3000/6000/10000.  Each retained step must contain and be bound by the
checkpoint SHA manifest to:

```
model.safetensors
optimizer.pt
metadata.pt
CHECKPOINT_READY.json
rng_state.rank0.pt ... rng_state.rank7.pt
RNG_SHA256SUMS
COMPLETE_RNG_STATE.json
```

The log SHA manifest must cover the start and terminal records exactly, and
all run/status/checkpoint evidence must have the expected UID/GID and contain
no failed, stopped, running, queued, temporary, or partial artifacts.  Extra
numeric checkpoints are accepted only when they have the core model,
optimizer, and metadata files; an incomplete extra checkpoint is rejected.

## CLI

Run from the repository root after the PAI controller has produced the
terminal evidence (the command itself does not query PAI):

```bash
python3 scripts/stage_s_accept_c_training.py \
  --registry-run /path/to/registry/<RUN_ID> \
  --registry-result /path/to/registry/<RUN_ID>/result.json \
  --submission-state /path/to/registry/<RUN_ID>/submission-state.json \
  --resolved /path/to/registry/<RUN_ID>/resolved.json \
  --jobs-ledger /path/to/registry/jobs.jsonl \
  --terminal-getjob /path/to/registry/<RUN_ID>/getjob-terminal.json \
  --terminal-getjob-sha /path/to/registry/<RUN_ID>/getjob-terminal.json.sha256 \
  --c-run-root /path/to/logs/r142_fp11_stage_s/c/<RUN_ID> \
  --c-status-root /path/to/logs/r142_fp11_stage_s/c_status/<RUN_ID> \
  --checkpoint-root /path/to/CKPT/leon/r142_stage_s_c \
  --output /path/to/logs/r142_fp11_stage_s/c_status/ACCEPTED_C_TRAINING.json
```

On success the command emits exactly one new
`ACCEPTED_C_TRAINING.json` and one
`ACCEPTED_C_TRAINING.json.sha256`.  Both names are unique: existing files are
never overwritten, and publication uses a durable temporary inode followed by
a no-replace atomic link.  A JSON is removed if its companion sidecar cannot
be published, so an orphan JSON is never treated as acceptance.

The JSON schema is `r142-stage-s-c-training-acceptance-v1`; its terminal
fields are `status=ACCEPTED`, `pai_terminal_status=Succeeded`, and
`label=WEAK_SUBSTRATE`.  Any missing, partial, running, stopped, extra
partial, or drifted input returns a non-zero exit code and leaves no output.

`verify_c_training_acceptance` independently re-hashes the published JSON and
sidecar.  It is intended for the next calibration freeze and for publication
readback; it does not relax any of the admission checks.
