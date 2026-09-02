#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

# This is a registry payload, not a submission helper. The PAI registry
# invokes it as the single foreground process. Every stage is idempotent in
# its own CPFS directory and the same RUN_ID is reused after a spot restart.
# HOME is intentionally inherited; this script never changes it.

NEW_ROOT=/mnt/cpfs/zbl-cpfs-new
USER_ROOT=$NEW_ROOT/USERS/leon
# The registry resource contract permits only its exact graphics pod env.
# Keep the scientific/runtime identity in this immutable payload + companion
# registry manifest instead of injecting custom pod environment variables.
STAGE_S_C_PROJECT_DIR=$USER_ROOT/code/r142-stage-s-c-runtime-20260903
STAGE_S_C_REGISTRY_CONFIG=$USER_ROOT/code/r142-stage-s-pai-20260902/stage_s_c_undertrained.json
STAGE_S_SOURCE_COMMIT=7575da585be31eb369a604d90048b338bbbf2c92
PROJECT_DIR=$(realpath -e -- "$STAGE_S_C_PROJECT_DIR") || {
  echo "C runtime clone path does not exist: $STAGE_S_C_PROJECT_DIR" >&2
  exit 43
}
export STAGE_S_C_PROJECT_DIR
PYTHON_BIN=$USER_ROOT/envs/openpi_py311/bin/python
QPILOTS=$USER_ROOT/code/QPILOTS-r16p15-stage1-task64-20260812
OPENPI=$QPILOTS/third_party/openpi
EXPECTED_QPILOTS_COMMIT=eacf47b981e3b22357f8a74902f8dad8cfcfa375
EXPECTED_OPENPI_COMMIT=54cbaee6ae0c010a1ed431871cdaa8f4684ac709
PAYLOAD_FILE=$(realpath -e -- "$0") || { echo "cannot resolve invoked C payload" >&2; exit 43; }
CONFIG_FILE=$(realpath -e -- "$STAGE_S_C_REGISTRY_CONFIG") || {
  echo "missing external C registry companion config: $STAGE_S_C_REGISTRY_CONFIG" >&2
  exit 43
}
STAGE_S_C_PAYLOAD_SHA256=$($PYTHON_BIN - "$CONFIG_FILE" <<'PY'
import json
import pathlib
import sys

runtime = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))["runtime"]
command_sha = runtime.get("command_file_sha256")
payload_sha = runtime.get("payload_sha256")
if not isinstance(command_sha, str) or command_sha != payload_sha:
    raise SystemExit("registry payload SHA fields are missing or disagree")
print(payload_sha)
PY
)
export STAGE_S_SOURCE_COMMIT STAGE_S_C_PAYLOAD_SHA256 STAGE_S_C_REGISTRY_CONFIG
BASE_JAX=$USER_ROOT/cache/r142_stage_s/pi05_base
BASE_PT=$USER_ROOT/cache/r142_stage_s/pi05_base_pytorch
ASSETS=$USER_ROOT/cache/openpi/r16p15/openpi-assets/checkpoints
CHECKPOINT_BASE=$NEW_ROOT/CKPT/leon/r142_stage_s_c
RUN_ID=${PAI_RUN_ID:-${PAI_CANARY_RUN_ID:?registry must inject PAI_RUN_ID}}
LOG_ROOT=$USER_ROOT/logs/r142_fp11_stage_s/c/$RUN_ID
STATUS_ROOT=$USER_ROOT/logs/r142_fp11_stage_s/c_status/$RUN_ID

CURRENT_STAGE=preflight
mkdir -p "$LOG_ROOT" "$STATUS_ROOT"

write_status_marker() {
  local path=$1 status=$2 stage=$3 exit_code=${4:-0} evidence=${5:-}
  "$PYTHON_BIN" - "$path" "$status" "$stage" "$exit_code" "$evidence" <<'PY'
import hashlib
import json
import os
import pathlib
import sys
import time

path, status, stage, exit_code, evidence = sys.argv[1:]
payload = {
    "schema": "r142-stage-s-c-pai-stage-status-v1",
    "status": status,
    "stage": stage,
    "exit_code": int(exit_code),
    "run_id": os.environ.get("PAI_RUN_ID") or os.environ.get("PAI_CANARY_RUN_ID"),
    "job_id": os.environ.get("PAI_TASK_JOB_ID") or os.environ.get("PAI_JOB_ID"),
    "openpi_commit": "54cbaee6ae0c010a1ed431871cdaa8f4684ac709",
    "qpilots_commit": "eacf47b981e3b22357f8a74902f8dad8cfcfa375",
    "stage_s_source_commit": os.environ.get("STAGE_S_SOURCE_COMMIT"),
    "launcher_payload_sha256": os.environ.get("STAGE_S_C_PAYLOAD_SHA256"),
    "project_dir": os.environ.get("STAGE_S_C_PROJECT_DIR"),
    "evidence_path": evidence or None,
    "evidence_sha256": (
        hashlib.sha256(pathlib.Path(evidence).read_bytes()).hexdigest()
        if evidence and pathlib.Path(evidence).is_file()
        else None
    ),
    "written_at": time.time(),
}
payload["payload_sha256"] = hashlib.sha256(
    json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
).hexdigest()
destination = pathlib.Path(path)
destination.parent.mkdir(parents=True, exist_ok=True)
temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
with temporary.open("w", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True, indent=2, ensure_ascii=False)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
temporary.replace(destination)
directory_fd = os.open(destination.parent, os.O_RDONLY)
try:
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)
PY
}

on_error() {
  local rc=$?
  set +e
  write_status_marker "$STATUS_ROOT/FAILED_${CURRENT_STAGE}.json" FAILED "$CURRENT_STAGE" "$rc" "" || true
  exit "$rc"
}
trap on_error ERR

blocked_window() {
  local hm
  hm=$(TZ=Asia/Shanghai date +%H%M)
  case "$hm" in
    0930|0931|0932|0933|0934|0935|0936|0937|0938|0939|1930|1931|1932|1933|1934|1935|1936|1937|1938|1939)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

if blocked_window; then
  write_status_marker "$STATUS_ROOT/FAILED_BLACKOUT.json" FAILED blackout 75 ""
  printf '%s\n' 'REFUSED_DAILY_NO_JOB_WINDOW' >"$STATUS_ROOT/REFUSED_WINDOW.txt"
  exit 75
fi

[[ "$(id -u):$(id -g)" == "2254:2254" ]] || {
  echo "expected runtime UID/GID 2254:2254, got $(id -u):$(id -g)" >&2
  exit 41
}
[[ -x "$PYTHON_BIN" ]] || { echo "missing pinned OpenPI Python: $PYTHON_BIN" >&2; exit 42; }
for checkout in "$PROJECT_DIR" "$QPILOTS" "$OPENPI"; do
  git -C "$checkout" rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
    echo "missing independent C, QPILOTS, or OpenPI git checkout: $checkout" >&2
    exit 43
  }
done
[[ -f "$PAYLOAD_FILE" && -f "$CONFIG_FILE" ]] || {
  echo "missing invoked C payload or companion registry config" >&2
  exit 43
}
[[ "$STAGE_S_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || {
  echo "STAGE_S_SOURCE_COMMIT must be a full lowercase Git commit" >&2
  exit 44
}
[[ "$(git -C "$PROJECT_DIR" rev-parse HEAD)" == "$STAGE_S_SOURCE_COMMIT" ]] || {
  echo "Stage-S source checkout does not match injected STAGE_S_SOURCE_COMMIT" >&2
  exit 44
}
[[ -z "$(git -C "$PROJECT_DIR" status --porcelain)" ]] || {
  echo "Stage-S source checkout is dirty" >&2
  exit 45
}
[[ "$(git -C "$QPILOTS" rev-parse HEAD)" == "$EXPECTED_QPILOTS_COMMIT" ]] || {
  echo "QPILOTS checkout is not pinned to $EXPECTED_QPILOTS_COMMIT" >&2
  exit 46
}
[[ -z "$(git -C "$QPILOTS" status --porcelain)" ]] || { echo "QPILOTS checkout is dirty" >&2; exit 47; }
[[ "$(git -C "$OPENPI" rev-parse HEAD)" == "$EXPECTED_OPENPI_COMMIT" ]] || {
  echo "OpenPI checkout is not pinned to $EXPECTED_OPENPI_COMMIT" >&2
  exit 48
}
[[ -z "$(git -C "$OPENPI" status --porcelain)" ]] || { echo "OpenPI checkout is dirty" >&2; exit 49; }
[[ "$STAGE_S_C_PAYLOAD_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
  echo "STAGE_S_C_PAYLOAD_SHA256 must be a full lowercase SHA-256" >&2
  exit 50
}
CONFIG_PAYLOAD_SHA256=$("$PYTHON_BIN" - "$CONFIG_FILE" <<'PY'
import json
import os
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
runtime = payload.get("runtime", {})
project_dir = runtime.get("project_dir")
if project_dir != os.environ.get("STAGE_S_C_PROJECT_DIR"):
    raise SystemExit("registry project_dir does not match STAGE_S_C_PROJECT_DIR")
command_file = runtime.get("command_file")
if command_file != str(pathlib.Path(sys.argv[1]).with_name("stage_s_c_undertrained_pai.sh")):
    raise SystemExit("registry command_file is not beside the companion config")
command_sha = runtime.get("command_file_sha256")
payload_sha = runtime.get("payload_sha256")
if not isinstance(command_sha, str) or not isinstance(payload_sha, str) or command_sha != payload_sha:
    raise SystemExit("registry payload SHA fields are missing or disagree")
print(payload_sha)
PY
)
[[ "$CONFIG_PAYLOAD_SHA256" == "$STAGE_S_C_PAYLOAD_SHA256" ]] || {
  echo "injected payload SHA differs from pinned registry config" >&2
  exit 51
}
OBSERVED_PAYLOAD_SHA256=$(sha256sum "$PAYLOAD_FILE" | awk '{print $1}')
[[ "$OBSERVED_PAYLOAD_SHA256" == "$STAGE_S_C_PAYLOAD_SHA256" ]] || {
  echo "invoked C payload SHA differs from injected SHA" >&2
  exit 52
}
for writable in "$BASE_JAX" "$BASE_PT" "$CHECKPOINT_BASE" "$LOG_ROOT" "$STATUS_ROOT"; do
  mkdir -p "$writable"
  probe="$writable/.r142-owner-probe.$$"
  : >"$probe"
  [[ "$(stat -c '%u:%g' "$probe")" == "2254:2254" ]] || exit 46
  rm -f -- "$probe"
done

# Persist the admission identity before any asset, conversion, or training
# mutation.  This records the exact bindings checked above; it is not a
# substitute for the source, cleanliness, and digest checks.
"$PYTHON_BIN" - "$STATUS_ROOT/RUNTIME_IDENTITY.json" "$PROJECT_DIR" "$QPILOTS" "$OPENPI" \
  "$STAGE_S_SOURCE_COMMIT" "$EXPECTED_QPILOTS_COMMIT" "$EXPECTED_OPENPI_COMMIT" "$STAGE_S_C_PAYLOAD_SHA256" "$PAYLOAD_FILE" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

destination = pathlib.Path(sys.argv[1])
project, qpilots, openpi, stage_commit, qpilots_commit, openpi_commit, payload_sha, payload_path = sys.argv[2:]
payload_file = pathlib.Path(payload_path)
record = {
    "schema": "r142-stage-s-c-runtime-identity-v1",
    "project_dir": project,
    "stage_s_source_commit": stage_commit,
    "qpilots_root": qpilots,
    "qpilots_commit": qpilots_commit,
    "openpi_root": openpi,
    "openpi_commit": openpi_commit,
    "payload_path": str(payload_file),
    "payload_sha256": payload_sha,
    "payload_sha256_observed": hashlib.sha256(payload_file.read_bytes()).hexdigest(),
    "run_id": os.environ.get("PAI_RUN_ID") or os.environ.get("PAI_CANARY_RUN_ID"),
    "job_id": os.environ.get("PAI_TASK_JOB_ID") or os.environ.get("PAI_JOB_ID"),
}
if record["payload_sha256"] != record["payload_sha256_observed"]:
    raise SystemExit("runtime identity payload digest changed during admission")
destination.parent.mkdir(parents=True, exist_ok=True)
temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
with temporary.open("w", encoding="utf-8") as handle:
    json.dump(record, handle, sort_keys=True, indent=2)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
temporary.replace(destination)
directory_fd = os.open(destination.parent, os.O_RDONLY)
try:
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)
PY
write_status_marker "$STATUS_ROOT/COMPLETED_preflight.json" COMPLETED preflight 0 "$STATUS_ROOT/RUNTIME_IDENTITY.json"

export PYTHONPATH="$PROJECT_DIR/src:$OPENPI/src"
export WANDB_MODE=disabled

CURRENT_STAGE=base_download
if [[ -f "$BASE_JAX/BASE_DOWNLOAD_COMPLETED.json" ]]; then
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/stage_s_libero_c_assets.py" audit --output-root "$BASE_JAX"
else
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/stage_s_libero_c_assets.py" download \
    --output-root "$BASE_JAX" \
    --manifest "$PROJECT_DIR/stage-s/C_PI05_BASE_GCS_MANIFEST.json"
fi
write_status_marker "$STATUS_ROOT/COMPLETED_base_download.json" COMPLETED base_download 0 "$BASE_JAX/BASE_DOWNLOAD_COMPLETED.json"

CURRENT_STAGE=conversion
"$PYTHON_BIN" "$PROJECT_DIR/scripts/stage_s_libero_c_assets.py" convert \
  --openpi-root "$OPENPI" --base-jax-root "$BASE_JAX" \
  --base-pytorch-root "$BASE_PT" --python "$PYTHON_BIN" --precision bfloat16
write_status_marker "$STATUS_ROOT/COMPLETED_conversion.json" COMPLETED conversion 0 "$BASE_PT/CONVERSION_COMPLETED.json"

CURRENT_STAGE=training
TRAIN_ARGS=(
  --openpi-root "$OPENPI"
  --base-jax-root "$BASE_JAX"
  --base-pytorch-root "$BASE_PT"
  --checkpoint-base-dir "$CHECKPOINT_BASE"
  --log-root "$LOG_ROOT"
  --assets-base-dir "$ASSETS"
  --python "$PYTHON_BIN"
)
TRAIN_DIR="$CHECKPOINT_BASE/pi05_libero/r142_stage_s_c_undertrained_seed42"
if [[ -f "$CHECKPOINT_BASE/COMPLETED_C_TRAINING.json" ]]; then
  (cd "$CHECKPOINT_BASE" && sha256sum --check --quiet SHA256SUMS)
  (cd "$LOG_ROOT" && sha256sum --check --quiet SHA256SUMS)
else
  if [[ -d "$TRAIN_DIR" ]] && find "$TRAIN_DIR" -mindepth 1 -maxdepth 1 -type d -name '[0-9]*' -print -quit | grep -q .; then
    TRAIN_ARGS+=(--resume)
  fi
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/stage_s_libero_c_train.py" "${TRAIN_ARGS[@]}"
  [[ -f "$CHECKPOINT_BASE/COMPLETED_C_TRAINING.json" ]]
  (cd "$CHECKPOINT_BASE" && sha256sum --check --quiet SHA256SUMS)
  (cd "$LOG_ROOT" && sha256sum --check --quiet SHA256SUMS)
fi
write_status_marker "$STATUS_ROOT/COMPLETED_training.json" COMPLETED training 0 "$CHECKPOINT_BASE/COMPLETED_C_TRAINING.json"
write_status_marker "$STATUS_ROOT/COMPLETED_C_PIPELINE.json" COMPLETED terminal 0 "$CHECKPOINT_BASE/COMPLETED_C_TRAINING.json"
trap - ERR
