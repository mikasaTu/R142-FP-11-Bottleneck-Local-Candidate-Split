#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

# This is a registry payload, not a submission helper. The PAI registry
# invokes it as the single foreground process. Every stage is idempotent in
# its own CPFS directory and the same RUN_ID is reused after a spot restart.
# HOME is intentionally inherited; this script never changes it.

NEW_ROOT=/mnt/cpfs/zbl-cpfs-new
USER_ROOT=$NEW_ROOT/USERS/leon
PROJECT_DIR=$USER_ROOT/code/R142-FP-11-Bottleneck-Local-Candidate-Split
PYTHON_BIN=$USER_ROOT/envs/openpi_py311/bin/python
OPENPI=$USER_ROOT/code/QPILOTS-r16p15-stage1-task64-20260812/third_party/openpi
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
[[ -d "$PROJECT_DIR" && -d "$OPENPI" ]] || { echo "missing project/OpenPI checkout" >&2; exit 43; }
[[ "$(git -C "$OPENPI" rev-parse HEAD)" == 54cbaee6ae0c010a1ed431871cdaa8f4684ac709 ]] || {
  echo "OpenPI checkout is not pinned to 54cbaee6ae0c010a1ed431871cdaa8f4684ac709" >&2
  exit 44
}
[[ -z "$(git -C "$OPENPI" status --porcelain)" ]] || { echo "OpenPI checkout is dirty" >&2; exit 45; }
for writable in "$BASE_JAX" "$BASE_PT" "$CHECKPOINT_BASE" "$LOG_ROOT" "$STATUS_ROOT"; do
  mkdir -p "$writable"
  probe="$writable/.r142-owner-probe.$$"
  : >"$probe"
  [[ "$(stat -c '%u:%g' "$probe")" == "2254:2254" ]] || exit 46
  rm -f -- "$probe"
done

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
