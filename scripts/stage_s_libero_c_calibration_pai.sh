#!/usr/bin/env bash
set -Eeuo pipefail

# Stage-S C Step-0 is an eight-rank, pooled-only evaluation over four real
# under-trained pi05_libero checkpoints. This
# launcher never submits a PAI job; a registry/controller owns the artifact
# directory and injects PAI_CANARY_RUN_ID and PAI_CANARY_RUN_DIR.
umask 077

readonly LEON_UID=2254
readonly LEON_GID=2254
readonly ROOT=/mnt/cpfs/zbl-cpfs-new/USERS/leon
readonly NEW_ROOT=/mnt/cpfs/zbl-cpfs-new
readonly RUN_ID="${PAI_CANARY_RUN_ID:?controller must inject PAI_CANARY_RUN_ID}"
readonly ARTIFACT_DIR="${PAI_CANARY_RUN_DIR:?controller must inject PAI_CANARY_RUN_DIR}"
readonly EXPECTED_GPUS=8
readonly WORLD_SIZE=8

# Pinned real Stage-R/LIBERO source trees. A dirty tree or commit drift is
# refused before the first simulator import.
readonly STAGE_S_REPO="$ROOT/code/r142-stage-s-c-runtime-racefix-20260903"
readonly STAGE_S_SOURCE_COMMIT=59581b09ce974a7080aaf6660f7619be465ce19d
readonly QPILOTS="$ROOT/code/QPILOTS-r16p15-stage1-task64-20260812"
readonly QPILOTS_COMMIT=eacf47b981e3b22357f8a74902f8dad8cfcfa375
readonly OPENPI="$QPILOTS/third_party/openpi"
readonly OPENPI_COMMIT=54cbaee6ae0c010a1ed431871cdaa8f4684ac709
readonly LIBERO="$OPENPI/third_party/libero"
readonly LIBERO_COMMIT=f78abd68ee283de9f9be3c8f7e2a9ad60246e95c
readonly PYTHON="$ROOT/envs/openpi_py311/bin/python"

# C training is a read-only input. The main thread publishes this stable
# acceptance manifest only after one exact PAI JobId reaches terminal success;
# its accepted_run_id follows a future blackout/resume lineage and is not
# hard-coded here.
readonly C_ACCEPTANCE_MANIFEST="$ROOT/logs/r142_fp11_stage_s/c_status/ACCEPTED_C_TRAINING.json"
readonly C_CHECKPOINT_BASE="$NEW_ROOT/CKPT/leon/r142_stage_s_c"
readonly C_TRAIN_DIR="$C_CHECKPOINT_BASE/pi05_libero/r142_stage_s_c_undertrained_seed42"
readonly C_COMPLETION="$C_CHECKPOINT_BASE/COMPLETED_C_TRAINING.json"
readonly C_CHECKPOINT_SHA="$C_CHECKPOINT_BASE/SHA256SUMS"
readonly LIBERO_CONFIG_SOURCE="$LIBERO/libero/configs/config.yaml"

readonly OUT_ROOT="$ROOT/logs/pai_registry/r142_stage_s/c_calibration"
readonly EXPECTED_ARTIFACT_DIR="$OUT_ROOT/$RUN_ID"
readonly OUT="$ARTIFACT_DIR"
readonly LIBERO_CONFIG_ROOT="$OUT/libero-config"
readonly C_ACCEPTANCE_SNAPSHOT="$OUT/C_TRAINING_ACCEPTANCE.json"

PHASE=bootstrap
FAILURE_RECORDED=0
C_TRAINING_RUN_ID=""
C_TRAINING_JOB_ID=""
C_TRAINING_LOG_ROOT=""
C_TRAINING_STATUS_ROOT=""
C_LOG_SHA=""
C_PIPELINE_MARKER=""

write_failure_marker() {
  local rc="$1"
  [[ "$FAILURE_RECORDED" == 1 ]] && return 0
  [[ -d "$OUT" && ! -L "$OUT" ]] || return 0
  FAILURE_RECORDED=1
  python3 - "$OUT/FAILED_C_CALIBRATION.json" "$rc" "$RUN_ID" "$PHASE" <<'PY'
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "r142-stage-s-c-calibration-failure-v1",
    "status": "FAILED",
    "label": "WEAK_SUBSTRATE",
    "run_id": sys.argv[3],
    "phase": sys.argv[4],
    "exit_code": int(sys.argv[2]),
    "uid": os.getuid(),
    "gid": os.getgid(),
    "world_size": 8,
    "gpu_count": 8,
    "acceptance_manifest": "/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_s/c_status/ACCEPTED_C_TRAINING.json",
    "accepted_run_id": os.environ.get("C_ACCEPTED_RUN_ID"),
    "accepted_job_id": os.environ.get("C_ACCEPTED_JOB_ID"),
}
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
os.chmod(tmp, 0o600)
os.replace(tmp, path)
PY
}

on_error() {
  local rc=$?
  write_failure_marker "$rc" || true
  exit "$rc"
}
trap on_error ERR

in_daily_no_job_window() {
  local hm hour minute total
  hm="$(TZ=Asia/Shanghai date +%H%M)"
  hour=$((10#${hm:0:2}))
  minute=$((10#${hm:2:2}))
  total=$((hour * 60 + minute))
  (( (total >= 9 * 60 + 30 && total < 9 * 60 + 40) ||
     (total >= 19 * 60 + 30 && total < 19 * 60 + 40) ))
}

guard_daily_no_job_window() {
  if in_daily_no_job_window; then
    PHASE=daily_no_job_window
    python3 - "$OUT/REFUSED_DAILY_NO_JOB_WINDOW.json" "$RUN_ID" <<'PY'
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "r142-stage-s-c-calibration-window-refusal-v1",
    "status": "REFUSED_DAILY_NO_JOB_WINDOW",
    "label": "WEAK_SUBSTRATE",
    "run_id": sys.argv[2],
    "timezone": "Asia/Shanghai",
    "windows": ["09:30-09:40", "19:30-19:40"],
    "uid": os.getuid(),
    "gid": os.getgid(),
}
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
os.chmod(tmp, 0o600)
os.replace(tmp, path)
PY
    trap - ERR
    exit 75
  fi
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "C_CALIBRATION_REFUSED missing command: $1" >&2
    return 69
  }
}

PHASE=bootstrap
[[ "$RUN_ID" =~ ^r142-stage-s-c-calibration-20[0-9]{6}-r[0-9]+$ ]]
[[ "$ARTIFACT_DIR" == "$EXPECTED_ARTIFACT_DIR" ]]
[[ -d "$OUT" && ! -L "$OUT" ]]
[[ "$(realpath -e "$OUT")" == "$EXPECTED_ARTIFACT_DIR" ]]
[[ "$(stat -c '%u:%g' "$OUT")" == "$LEON_UID:$LEON_GID" ]]
[[ "$(stat -c '%a' "$OUT")" == 700 ]]
[[ "$(id -u):$(id -g)" == "$LEON_UID:$LEON_GID" ]]
for command_name in bash date find git nvidia-smi realpath sha256sum stat "$PYTHON" python3; do
  if [[ "$command_name" == /* ]]; then
    [[ -x "$command_name" ]] || { echo "C_CALIBRATION_REFUSED missing executable: $command_name" >&2; exit 69; }
  else
    require_command "$command_name"
  fi
done
guard_daily_no_job_window

# Materialize the allocation-local device list when the PAI device plugin
# omits CUDA_VISIBLE_DEVICES.  The child entrypoint selects one device per
# LOCAL_RANK before importing JAX/OpenPI.
mapfile -t GPU_INDEXES < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
[[ "${#GPU_INDEXES[@]}" -eq "$EXPECTED_GPUS" ]]
for index in "${GPU_INDEXES[@]}"; do
  [[ "$index" =~ ^[0-9]+$ ]]
done
export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPU_INDEXES[*]}")"

PHASE=source_readback
for source_repo in "$STAGE_S_REPO" "$QPILOTS" "$OPENPI" "$LIBERO"; do
  [[ -e "$source_repo/.git" ]] || { echo "C_CALIBRATION_REFUSED missing git source: $source_repo" >&2; exit 66; }
  [[ -z "$(git -C "$source_repo" status --porcelain)" ]] || {
    echo "C_CALIBRATION_REFUSED dirty source tree: $source_repo" >&2
    exit 66
  }
done
[[ "$(git -C "$STAGE_S_REPO" rev-parse HEAD)" == "$STAGE_S_SOURCE_COMMIT" ]]
[[ "$(git -C "$QPILOTS" rev-parse HEAD)" == "$QPILOTS_COMMIT" ]]
[[ "$(git -C "$OPENPI" rev-parse HEAD)" == "$OPENPI_COMMIT" ]]
[[ "$(git -C "$LIBERO" rev-parse HEAD)" == "$LIBERO_COMMIT" ]]
[[ -s "$LIBERO_CONFIG_SOURCE" && ! -L "$LIBERO_CONFIG_SOURCE" ]]

PHASE=training_input_audit
[[ -s "$C_ACCEPTANCE_MANIFEST" && ! -L "$C_ACCEPTANCE_MANIFEST" ]]
C_TRAINING_RUN_ID="$($PYTHON - "$C_ACCEPTANCE_MANIFEST" <<'PY'
import json
import pathlib
import re
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("status") != "ACCEPTED":
    raise SystemExit("C acceptance manifest is not ACCEPTED")
value = payload.get("accepted_run_id")
if not isinstance(value, str) or not re.fullmatch(r"r142-stage-s-c-undertrained-20[0-9]{6}-r[0-9]+", value):
    raise SystemExit("C acceptance manifest has invalid accepted_run_id")
print(value)
PY
)"
C_TRAINING_JOB_ID="$($PYTHON - "$C_ACCEPTANCE_MANIFEST" <<'PY'
import json
import pathlib
import re
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
value = payload.get("job_id")
if not isinstance(value, str) or not re.fullmatch(r"dlc[0-9a-z]+", value):
    raise SystemExit("C acceptance manifest has invalid job_id")
print(value)
PY
)"
export C_ACCEPTED_RUN_ID="$C_TRAINING_RUN_ID" C_ACCEPTED_JOB_ID="$C_TRAINING_JOB_ID"
C_TRAINING_LOG_ROOT="$ROOT/logs/r142_fp11_stage_s/c/$C_TRAINING_RUN_ID"
C_TRAINING_STATUS_ROOT="$ROOT/logs/r142_fp11_stage_s/c_status/$C_TRAINING_RUN_ID"
C_LOG_SHA="$C_TRAINING_LOG_ROOT/SHA256SUMS"
C_PIPELINE_MARKER="$C_TRAINING_STATUS_ROOT/COMPLETED_C_PIPELINE.json"
[[ -s "$C_COMPLETION" && ! -L "$C_COMPLETION" ]]
[[ -s "$C_CHECKPOINT_SHA" && ! -L "$C_CHECKPOINT_SHA" ]]
[[ -s "$C_LOG_SHA" && ! -L "$C_LOG_SHA" ]]
[[ -s "$C_PIPELINE_MARKER" && ! -L "$C_PIPELINE_MARKER" ]]
(cd "$C_CHECKPOINT_BASE" && sha256sum --check --quiet SHA256SUMS)
(cd "$C_TRAINING_LOG_ROOT" && sha256sum --check --quiet SHA256SUMS)

# Bind the common checkpoint bundle, both SHA manifests, and terminal status
# to the stable acceptance manifest. This launcher never creates acceptance.
"$PYTHON" - "$C_ACCEPTANCE_MANIFEST" "$C_COMPLETION" "$C_CHECKPOINT_SHA" "$C_LOG_SHA" "$C_TRAINING_LOG_ROOT" "$C_PIPELINE_MARKER" "$C_TRAIN_DIR" "$C_TRAINING_RUN_ID" "$C_TRAINING_JOB_ID" <<'PY'
import hashlib
import json
import pathlib
import re
import sys

acceptance_path = pathlib.Path(sys.argv[1]).resolve()
completion_path = pathlib.Path(sys.argv[2]).resolve()
checkpoint_sha_path = pathlib.Path(sys.argv[3]).resolve()
log_sha_path = pathlib.Path(sys.argv[4]).resolve()
log_root = pathlib.Path(sys.argv[5]).resolve()
pipeline_path = pathlib.Path(sys.argv[6]).resolve()
checkpoint_base = pathlib.Path(sys.argv[7]).resolve()
run_id, job_id = sys.argv[8:10]
acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
if acceptance.get("schema") != "r142-stage-s-c-training-acceptance-v1":
    raise SystemExit("C acceptance manifest schema drifted")
if (
    acceptance.get("status") != "ACCEPTED"
    or acceptance.get("label") != "WEAK_SUBSTRATE"
    or acceptance.get("pai_terminal_status") != "Succeeded"
):
    raise SystemExit("C acceptance manifest is not an accepted weak-substrate result")
if acceptance.get("accepted_run_id") != run_id or not re.fullmatch(r"r142-stage-s-c-undertrained-20[0-9]{6}-r[0-9]+", run_id):
    raise SystemExit("C acceptance run id does not match its derived lineage")
if acceptance.get("job_id") != job_id or not re.fullmatch(r"dlc[0-9a-z]+", job_id):
    raise SystemExit("C acceptance job id does not match its derived terminal job")
source = {
    "stage_s_commit": "59581b09ce974a7080aaf6660f7619be465ce19d",
    "qpilots_commit": "eacf47b981e3b22357f8a74902f8dad8cfcfa375",
    "openpi_commit": "54cbaee6ae0c010a1ed431871cdaa8f4684ac709",
    "libero_commit": "f78abd68ee283de9f9be3c8f7e2a9ad60246e95c",
}
if acceptance.get("source") != source:
    raise SystemExit("C acceptance source commits drifted")
expected_paths = {
    "checkpoint_root": str(checkpoint_base.parents[1]),
    "checkpoint_completion": str(completion_path),
    "checkpoint_sha256_manifest": str(checkpoint_sha_path),
    "log_root": str(log_root),
    "log_sha256_manifest": str(log_sha_path),
    "training_pipeline_completion": str(pipeline_path),
}
for key, expected in expected_paths.items():
    if acceptance.get(key) != expected:
        raise SystemExit(f"C acceptance path mismatch for {key}")
hash_fields = {
    "checkpoint_completion_sha256": completion_path,
    "checkpoint_sha256_manifest_digest": checkpoint_sha_path,
    "log_sha256_manifest_digest": log_sha_path,
}
for key, path in hash_fields.items():
    if acceptance.get(key) != hashlib.sha256(path.read_bytes()).hexdigest():
        raise SystemExit(f"C acceptance hash mismatch for {key}")
if acceptance.get("checkpoint_steps") != [1000, 3000, 6000, 10000]:
    raise SystemExit("C acceptance checkpoint schedule drifted")
if acceptance.get("full_reference_step") != 30000:
    raise SystemExit("C acceptance full reference step drifted")
if acceptance.get("no_interpolation") is not True or acceptance.get("artificial_degradation") is not False:
    raise SystemExit("C acceptance permits interpolation or artificial degradation")
checkpoint_hashes = acceptance.get("checkpoint_hashes")
expected_checkpoint_hash_paths = {
    f"{step}/model.safetensors" for step in (1000, 3000, 6000, 10000)
}
if not isinstance(checkpoint_hashes, dict) or set(checkpoint_hashes) != expected_checkpoint_hash_paths:
    raise SystemExit("C acceptance must carry exactly four checkpoint model hashes")

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

for relative_path, expected_hash in checkpoint_hashes.items():
    relative = pathlib.PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts or relative_path != relative.as_posix():
        raise SystemExit(f"C acceptance checkpoint hash path is not relative: {relative_path}")
    if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise SystemExit(f"C acceptance checkpoint hash is invalid: {relative_path}")
    checkpoint_path = checkpoint_base / pathlib.Path(*relative.parts)
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise SystemExit(f"C acceptance checkpoint model is missing or symlinked: {checkpoint_path}")
    if sha256_file(checkpoint_path) != expected_hash:
        raise SystemExit(f"C acceptance checkpoint hash mismatch: {relative_path}")
print("C_ACCEPTANCE_MANIFEST_GATE_PASS")
PY

if [[ -e "$C_ACCEPTANCE_SNAPSHOT" || -L "$C_ACCEPTANCE_SNAPSHOT" ]]; then
  [[ -f "$C_ACCEPTANCE_SNAPSHOT" && ! -L "$C_ACCEPTANCE_SNAPSHOT" ]]
  cmp -s "$C_ACCEPTANCE_SNAPSHOT" "$C_ACCEPTANCE_MANIFEST"
else
  cp -- "$C_ACCEPTANCE_MANIFEST" "$C_ACCEPTANCE_SNAPSHOT"
  chmod 600 "$C_ACCEPTANCE_SNAPSHOT"
fi

export PYTHONPATH="$STAGE_S_REPO/src:$QPILOTS:$OPENPI/src:$OPENPI:$LIBERO"
"$PYTHON" - "$C_PIPELINE_MARKER" "$C_COMPLETION" "$C_TRAINING_RUN_ID" <<'PY'
import json
import pathlib
import sys

from hashlib import sha256

pipeline_path, completion_path, run_id = map(pathlib.Path, sys.argv[1:])
pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
if pipeline.get("status") != "COMPLETED" or pipeline.get("stage") != "terminal":
    raise SystemExit("C training pipeline marker is not terminal COMPLETED")
if pipeline.get("run_id") != str(run_id):
    raise SystemExit("C training pipeline marker run id mismatch")
if pathlib.Path(str(pipeline.get("evidence_path", ""))).resolve() != completion_path.resolve():
    raise SystemExit("C training pipeline marker does not bind COMPLETED_C_TRAINING")
if pipeline.get("evidence_sha256") != sha256(completion_path.read_bytes()).hexdigest():
    raise SystemExit("C training pipeline evidence SHA mismatch")
completion = json.loads(completion_path.read_text(encoding="utf-8"))
if completion.get("status") != "COMPLETED":
    raise SystemExit("COMPLETED_C_TRAINING is not COMPLETED")
if completion.get("openpi_commit") != "54cbaee6ae0c010a1ed431871cdaa8f4684ac709":
    raise SystemExit("C training OpenPI commit drifted")
if completion.get("config_name") != "pi05_libero":
    raise SystemExit("C training config is not pi05_libero")
if completion.get("seed") != 42 or completion.get("terminal_global_step") != 10001:
    raise SystemExit("C training seed or terminal step drifted")
if completion.get("checkpoint_steps") != [1000, 3000, 6000, 10000]:
    raise SystemExit("C retained checkpoint schedule drifted")
if completion.get("checkpoint_audit", {}).get("valid") is not True:
    raise SystemExit("C completion marker lacks a valid full checkpoint audit")
print("C_TRAINING_COMPLETION_GATE_PASS")
PY

# Audit all four exact C checkpoints with the pinned runtime before simulator
# import, and persist only source/qualification metadata in the artifact.
"$PYTHON" - "$C_TRAIN_DIR" "$OUT/C_INPUT_AUDIT.json" "$C_TRAINING_RUN_ID" "$C_TRAINING_JOB_ID" "$C_ACCEPTANCE_MANIFEST" "$C_ACCEPTANCE_SNAPSHOT" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

from r142_stage_s.libero import C_FULL_REFERENCE_STEP, C_RETAIN_STEPS, audit_c_checkpoint_schedule

checkpoint_base = pathlib.Path(sys.argv[1]).resolve()
out = pathlib.Path(sys.argv[2]).resolve()
training_run_id, training_job_id = sys.argv[3:5]
acceptance_path = pathlib.Path(sys.argv[5]).resolve()
acceptance_snapshot = pathlib.Path(sys.argv[6]).resolve()
audit = audit_c_checkpoint_schedule(checkpoint_base, expected_steps=C_RETAIN_STEPS, require_training_state=True)
if not audit.get("valid"):
    raise SystemExit("C checkpoint schedule audit failed: " + "; ".join(audit.get("errors", [])))
if audit.get("label") != "WEAK_SUBSTRATE" or audit.get("full_reference_step") != C_FULL_REFERENCE_STEP:
    raise SystemExit("C checkpoint label/reference drifted")
for row, expected in zip(audit.get("checkpoints", []), C_RETAIN_STEPS):
    if row.get("step") != expected or row.get("complete") is not True or row.get("undertrained") is not True:
        raise SystemExit(f"C checkpoint is not complete exact weak substrate at step {expected}: {row}")
payload = {
    "schema": "r142-stage-s-c-calibration-input-audit-v1",
    "status": "COMPLETED",
    "label": "WEAK_SUBSTRATE",
    "training_run_id": training_run_id,
    "training_job_id": training_job_id,
    "acceptance_manifest": str(acceptance_path),
    "acceptance_manifest_sha256": hashlib.sha256(acceptance_path.read_bytes()).hexdigest(),
    "acceptance_snapshot": str(acceptance_snapshot),
    "checkpoint_root": str(checkpoint_base),
    "expected_steps": list(C_RETAIN_STEPS),
    "full_reference_step": C_FULL_REFERENCE_STEP,
    "no_interpolation": True,
    "artificial_degradation": False,
    "require_training_state": True,
    "audit": audit,
}
out.parent.mkdir(parents=True, exist_ok=True)
tmp = out.with_suffix(out.suffix + ".tmp")
tmp.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
os.chmod(tmp, 0o600)
os.replace(tmp, out)
print("C_CHECKPOINT_SCHEDULE_AUDIT_PASS")
PY

# Copy the pinned LIBERO config into this run's output once, then require byte
# identity on every resumed incarnation. This prevents process-global config
# state from changing the unmodified C substrate.
PHASE=run_scoped_config
if [[ -e "$LIBERO_CONFIG_ROOT/config.yaml" || -L "$LIBERO_CONFIG_ROOT/config.yaml" ]]; then
  [[ -f "$LIBERO_CONFIG_ROOT/config.yaml" && ! -L "$LIBERO_CONFIG_ROOT/config.yaml" ]]
  cmp -s "$LIBERO_CONFIG_ROOT/config.yaml" "$LIBERO_CONFIG_SOURCE"
else
  mkdir -p "$LIBERO_CONFIG_ROOT"
  cp -- "$LIBERO_CONFIG_SOURCE" "$LIBERO_CONFIG_ROOT/config.yaml"
  chmod 600 "$LIBERO_CONFIG_ROOT/config.yaml"
fi
[[ -s "$LIBERO_CONFIG_ROOT/config.yaml" && ! -L "$LIBERO_CONFIG_ROOT/config.yaml" ]]
export LIBERO_CONFIG_PATH="$LIBERO_CONFIG_ROOT"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=device
export NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
export XDG_CACHE_HOME="$OUT/xdg-cache"
mkdir -p "$XDG_CACHE_HOME"

PHASE=first_work
if [[ -e "$OUT/FIRST_WORK.json" || -L "$OUT/FIRST_WORK.json" ]]; then
  [[ -f "$OUT/FIRST_WORK.json" && ! -L "$OUT/FIRST_WORK.json" ]]
  "$PYTHON" - "$OUT/FIRST_WORK.json" "$RUN_ID" "$C_TRAINING_RUN_ID" "$C_TRAINING_JOB_ID" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("status") != "FIRST_WORK" or payload.get("label") != "WEAK_SUBSTRATE":
    raise SystemExit("FIRST_WORK label/status drifted")
if payload.get("run_id") != sys.argv[2] or payload.get("world_size") != 8 or payload.get("gpu_count") != 8:
    raise SystemExit("FIRST_WORK identity drifted")
if payload.get("training_run_id") != sys.argv[3] or payload.get("training_job_id") != sys.argv[4]:
    raise SystemExit("FIRST_WORK training input drifted")
PY
else
  "$PYTHON" - "$OUT/FIRST_WORK.json" "$RUN_ID" "$C_TRAINING_RUN_ID" "$C_TRAINING_JOB_ID" <<'PY'
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "r142-stage-s-c-calibration-first-work-v1",
    "status": "FIRST_WORK",
    "label": "WEAK_SUBSTRATE",
    "run_id": sys.argv[2],
    "training_run_id": sys.argv[3],
    "training_job_id": sys.argv[4],
    "checkpoint_steps": [1000, 3000, 6000, 10000],
    "uid": os.getuid(),
    "gid": os.getgid(),
    "gpu_count": 8,
    "world_size": 8,
    "stage_s_source_commit": "59581b09ce974a7080aaf6660f7619be465ce19d",
    "qpilots_commit": "eacf47b981e3b22357f8a74902f8dad8cfcfa375",
    "openpi_commit": "54cbaee6ae0c010a1ed431871cdaa8f4684ac709",
    "libero_commit": "f78abd68ee283de9f9be3c8f7e2a9ad60246e95c",
}
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
os.chmod(tmp, 0o600)
os.replace(tmp, path)
PY
fi

PHASE=calibration_shards
CHECKPOINT_ARGS=(
  --checkpoint "$C_TRAIN_DIR/1000"
  --checkpoint "$C_TRAIN_DIR/3000"
  --checkpoint "$C_TRAIN_DIR/6000"
  --checkpoint "$C_TRAIN_DIR/10000"
)

# Materialize the immutable plan once before distributed ranks start.  This
# avoids concurrent CPFS create/replace races on CALIBRATION_PLAN.json.
cd "$STAGE_S_REPO"
"$PYTHON" scripts/stage_s_libero_calibrate.py \
  --substrate C --mode prepare --output-root "$OUT" \
  --world-size "$WORLD_SIZE" --seed "$CALIBRATION_SEED" \
  "${CHECKPOINT_ARGS[@]}"

# The pinned OpenPI interpreter owns Torch, JAX and policy dependencies; do
# not allow PATH's system torchrun to select another Python.
"$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$WORLD_SIZE" \
  scripts/stage_s_gpu_rank_entry.py scripts/stage_s_libero_calibrate.py \
  --substrate C --mode shard --output-root "$OUT" \
  --world-size "$WORLD_SIZE" --seed "$CALIBRATION_SEED" --max-steps 520 \
  "${CHECKPOINT_ARGS[@]}" \
  --qpilots-root "$QPILOTS" --libero-root "$LIBERO" \
  --libero-config-root "$LIBERO_CONFIG_ROOT"

guard_daily_no_job_window

PHASE=calibration_aggregate
"$PYTHON" scripts/stage_s_libero_calibrate.py \
  --substrate C --mode aggregate --output-root "$OUT" \
  --world-size "$WORLD_SIZE" --seed "$CALIBRATION_SEED"

PHASE=completion_publish
"$PYTHON" - "$OUT" "$RUN_ID" "$C_TRAINING_RUN_ID" "$C_TRAINING_JOB_ID" "$C_ACCEPTANCE_MANIFEST" "$C_ACCEPTANCE_SNAPSHOT" "$C_CHECKPOINT_BASE" "$C_TRAINING_LOG_ROOT" "$C_COMPLETION" "$C_CHECKPOINT_SHA" "$C_LOG_SHA" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

from r142_stage_s.libero import (
    CALIBRATION_RESULT_SCHEMA,
    CALIBRATION_SEED,
    C_RETAIN_STEPS,
    STAGE_S_PROTOCOL_ID,
    verify_calibration_aggregate,
)

root = pathlib.Path(sys.argv[1]).resolve()
run_id, training_run_id, training_job_id = sys.argv[2], sys.argv[3], sys.argv[4]
acceptance_path = pathlib.Path(sys.argv[5]).resolve()
acceptance_snapshot = pathlib.Path(sys.argv[6]).resolve()
checkpoint_base = pathlib.Path(sys.argv[7]).resolve()
training_log_root = pathlib.Path(sys.argv[8]).resolve()
completion_path = pathlib.Path(sys.argv[9]).resolve()
checkpoint_sha_path = pathlib.Path(sys.argv[10]).resolve()
log_sha_path = pathlib.Path(sys.argv[11]).resolve()
settings = [f"step_{value}" for value in C_RETAIN_STEPS]
result = root / "CALIBRATION_RESULT.json"
verify_calibration_aggregate(result, settings, calibration_seed=CALIBRATION_SEED, world_size=8)
core_marker = root / "COMPLETED_CALIBRATION.json"
core_payload = json.loads(core_marker.read_text(encoding="utf-8"))
if core_payload.get("schema") != CALIBRATION_RESULT_SCHEMA:
    raise SystemExit("core calibration marker schema drifted")

if len(core_payload.get("rows", [])) not in (0, 4):
    raise SystemExit("core calibration marker has unexpected row shape")

rank_markers = []
rank_marker_sha256 = {}
for rank in range(8):
    marker = root / "shards" / f"rank-{rank:05d}" / "COMPLETED_SHARD.json"
    result_path = marker.parent / "RESULT.json"
    sums = marker.parent / "SHA256SUMS"
    if not marker.is_file() or not result_path.is_file() or not sums.is_file():
        raise SystemExit(f"missing rank completion evidence: {marker}")
    expected = f"{hashlib.sha256(result_path.read_bytes()).hexdigest()}  RESULT.json\n"
    if sums.read_text(encoding="utf-8") != expected:
        raise SystemExit(f"rank SHA mismatch: {marker.parent}")
    rel = marker.relative_to(root).as_posix()
    rank_markers.append(rel)
    rank_marker_sha256[rel] = hashlib.sha256(marker.read_bytes()).hexdigest()

payload = {
    "schema": "r142-stage-s-c-calibration-completion-v1",
    "status": "COMPLETED",
    "label": "WEAK_SUBSTRATE",
    "result_label": "WEAK_SUBSTRATE",
    "setting_labels": {setting: "WEAK_SUBSTRATE" for setting in settings},
    "protocol_id": STAGE_S_PROTOCOL_ID,
    "substrate": "C",
    "run_id": run_id,
    "input_training_run_id": training_run_id,
    "input_training_job_id": training_job_id,
    "checkpoint_steps": list(C_RETAIN_STEPS),
    "full_reference_step": 30000,
    "calibration_result": "CALIBRATION_RESULT.json",
    "calibration_result_sha256": hashlib.sha256(result.read_bytes()).hexdigest(),
    "calibration_result_schema": CALIBRATION_RESULT_SCHEMA,
    "calibration_seed": CALIBRATION_SEED,
    "world_size": 8,
    "rank_markers": rank_markers,
    "rank_marker_sha256": rank_marker_sha256,
    "source": {
        "stage_s_commit": "59581b09ce974a7080aaf6660f7619be465ce19d",
        "qpilots_commit": "eacf47b981e3b22357f8a74902f8dad8cfcfa375",
        "openpi_commit": "54cbaee6ae0c010a1ed431871cdaa8f4684ac709",
        "libero_commit": "f78abd68ee283de9f9be3c8f7e2a9ad60246e95c",
    },
    "input_provenance": {
        "acceptance_manifest": str(acceptance_path),
        "acceptance_manifest_sha256": hashlib.sha256(acceptance_path.read_bytes()).hexdigest(),
        "acceptance_snapshot": str(acceptance_snapshot),
        "checkpoint_root": str(checkpoint_base),
        "checkpoint_completion": str(completion_path),
        "checkpoint_sha256_manifest": str(checkpoint_sha_path),
        "checkpoint_sha256_manifest_digest": hashlib.sha256(checkpoint_sha_path.read_bytes()).hexdigest(),
        "training_log_root": str(training_log_root),
        "training_log_sha256_manifest": str(log_sha_path),
        "training_log_sha256_manifest_digest": hashlib.sha256(log_sha_path.read_bytes()).hexdigest(),
        "no_interpolation": True,
        "artificial_degradation": False,
        "full_training_state_required": True,
    },
    "compute": {
        "worker_count": 1,
        "gpu_count": 8,
        "cpu_cores": 88,
        "memory_gib": 1400,
        "shared_memory_gib": 1400,
        "resource_pool": "exp-robot",
        "resource_alias": "idle-a800-robot-stage-s-graphics-8gpu",
        "resource_id": "quota1ssrabud0bh",
    },
    "persistence": {
        "artifact_dir": str(root),
        "resume_same_run_id": True,
        "rank_shards_required": 8,
        "persisted_row_fields": ["setting", "successes", "total", "pooled_success"],
        "forbidden_trial_fields": ["S2", "S3", "S4", "S5", "genealogy", "trajectory"],
        "aggregate_sha_file": "SHA256SUMS",
        "bundle_sha_file": "C_SHA256SUMS",
    },
}

marker = root / "COMPLETED_C_CALIBRATION.json"
if marker.is_file():
    if json.loads(marker.read_text(encoding="utf-8")) != payload:
        raise SystemExit("existing COMPLETED_C_CALIBRATION.json drifted")
else:
    tmp = marker.with_suffix(marker.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, marker)
files = [
    root / "CALIBRATION_RESULT.json",
    root / "COMPLETED_CALIBRATION.json",
    root / "COMPLETED_C_CALIBRATION.json",
    root / "FIRST_WORK.json",
    root / "C_INPUT_AUDIT.json",
    root / "C_TRAINING_ACCEPTANCE.json",
    root / "libero-config" / "config.yaml",
]
for rank in range(8):
    files.extend(
        [
            root / "shards" / f"rank-{rank:05d}" / "RESULT.json",
            root / "shards" / f"rank-{rank:05d}" / "COMPLETED_SHARD.json",
        ]
    )
lines = []
for path in files:
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"missing final C calibration file: {path}")
    lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root).as_posix()}")
bundle_sums = root / "C_SHA256SUMS"
tmp = bundle_sums.with_suffix(bundle_sums.suffix + ".tmp")
tmp.write_text("\n".join(sorted(lines)) + "\n", encoding="utf-8")
os.chmod(tmp, 0o600)
os.replace(tmp, bundle_sums)
PY
(cd "$OUT" && sha256sum --check --quiet C_SHA256SUMS)

rm -f -- "$OUT/FAILED_C_CALIBRATION.json"
trap - ERR
printf 'C_CALIBRATION_COMPLETED label=WEAK_SUBSTRATE run_id=%s output=%s\n' "$RUN_ID" "$OUT"
