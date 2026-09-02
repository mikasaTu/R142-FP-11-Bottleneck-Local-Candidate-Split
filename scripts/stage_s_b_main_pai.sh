#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

# External PAI payload for the two Stage-S LIBERO main-screen variants.  The
# same bytes are deployed under stage_s_b_main_pai.sh and
# stage_s_c_main_pai.sh; the basename is the only variant selector.  No
# scientific result is accepted before its frozen calibration report gate.

readonly ROOT=/mnt/cpfs/zbl-cpfs-new/USERS/leon
readonly CODE_ROOT="$ROOT/code"
readonly DEP_ROOT="$CODE_ROOT/QPILOTS-r16p15-stage1-task64-20260812"
readonly OPENPI="$DEP_ROOT/third_party/openpi"
readonly LIBERO="$OPENPI/third_party/libero"
readonly PY="$ROOT/envs/openpi_py311/bin/python"
readonly RUNTIME_REPO="$CODE_ROOT/r142-stage-s-bc-main-runtime-20260903"
readonly SOURCE_COMMIT=b9c4f2eced140fb2b4711bdbfd86439cec41e291
readonly QPILOTS_COMMIT=eacf47b981e3b22357f8a74902f8dad8cfcfa375
readonly OPENPI_COMMIT=54cbaee6ae0c010a1ed431871cdaa8f4684ac709
readonly LIBERO_COMMIT=f78abd68ee283de9f9be3c8f7e2a9ad60246e95c
readonly POLICY_CHECKPOINT="$ROOT/cache/openpi/r16p15/openpi-assets/checkpoints/pi05_libero"
readonly B_REPORT="$ROOT/logs/r142_fp11_stage_s/b_calibration/CALIBRATION_REPORT.json"
readonly B_VARIANT_ROOT="$ROOT/logs/r142_fp11_stage_s/b_variants/r142-stage-s-b-variants-20260903-r7/variants"
readonly C_REPORT="$ROOT/logs/r142_fp11_stage_s/c_calibration/CALIBRATION_REPORT.json"
readonly PROTOCOL_ACCEPTANCE="$ROOT/logs/r142_fp11_stage_s/stage_s/protocol/FROZEN_PROTOCOL.json"
readonly MAIN_FINALIZER="$RUNTIME_REPO/scripts/stage_s_libero_main_finalize.py"

case "$(basename "$0")" in
  stage_s_b_main_pai.sh)
    readonly SUBSTRATE=B
    readonly RUN_PREFIX=r142-stage-s-b-main-20260903-r
    readonly REPORT="$B_REPORT"
    readonly OUT_ROOT="$ROOT/logs/pai_registry/r142_stage_s/b_main"
    ;;
  stage_s_c_main_pai.sh)
    readonly SUBSTRATE=C
    readonly RUN_PREFIX=r142-stage-s-c-main-20260903-r
    readonly REPORT="$C_REPORT"
    readonly OUT_ROOT="$ROOT/logs/pai_registry/r142_stage_s/c_main"
    ;;
  *)
    echo "refuse: payload must execute as stage_s_b_main_pai.sh or stage_s_c_main_pai.sh" >&2
    exit 64
    ;;
esac

readonly RUN_ID="${PAI_CANARY_RUN_ID:?controller must inject PAI_CANARY_RUN_ID}"
readonly ARTIFACT_DIR="${ARTIFACT_DIR:?controller must inject ARTIFACT_DIR}"
readonly EXPECTED_OUT="$OUT_ROOT/$RUN_ID"
readonly WORLD_SIZE=8
readonly CANDIDATE_COUNT=32
readonly EXPECTED_PAYLOAD_CONFIG="$CODE_ROOT/r142-stage-s-bc-pai-20260903/stage_s_${SUBSTRATE,,}_main.json"

FAILURE_WRITTEN=0
PHASE=bootstrap

write_json() {
  local path="$1"
  local json_payload="$2"
  python3 - "$path" "$json_payload" <<'PY'
import json
import os
import pathlib
import sys
import tempfile

path = pathlib.Path(sys.argv[1])
payload = json.loads(sys.argv[2])
path.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
try:
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
PY
}

write_failure() {
  local rc="$1"
  [[ "$FAILURE_WRITTEN" == 1 ]] && return 0
  FAILURE_WRITTEN=1
  [[ -d "$OUT" && ! -L "$OUT" ]] || return 0
  local payload
  payload=$(python3 - "$rc" "$PHASE" <<'PY'
import json
import os
import sys
import time

print(json.dumps({
    "status": "FAILED",
    "marker_type": "failed_stage_s_main",
    "substrate": os.environ.get("STAGE_S_SUBSTRATE"),
    "run_id": os.environ.get("PAI_CANARY_RUN_ID"),
    "job_id": os.environ.get("PAI_TASK_JOB_ID"),
    "phase": sys.argv[2],
    "exit_code": int(sys.argv[1]),
    "source_commit": os.environ.get("STAGE_S_SOURCE_COMMIT"),
    "uid": os.getuid(),
    "gid": os.getgid(),
    "time": time.time(),
}, sort_keys=True))
PY
  )
  write_json "$OUT/FAILED_${SUBSTRATE}_MAIN.json" "$payload" || true
}

on_error() {
  local rc=$?
  trap - ERR
  write_failure "$rc" || true
  exit "$rc"
}

in_blackout() {
  local hm hour minute total
  hm="$(TZ=Asia/Shanghai date +%H%M)"
  hour=$((10#${hm:0:2}))
  minute=$((10#${hm:2:2}))
  total=$((hour * 60 + minute))
  (( (total >= 9 * 60 + 30 && total < 9 * 60 + 40) ||
     (total >= 19 * 60 + 30 && total < 19 * 60 + 40) ))
}

export STAGE_S_SUBSTRATE="$SUBSTRATE"
export STAGE_S_SOURCE_COMMIT="$SOURCE_COMMIT"
export STAGE_S_CALIBRATION_REPORT="$REPORT"
export STAGE_S_PROTOCOL_ACCEPTANCE="$PROTOCOL_ACCEPTANCE"
mkdir -p "$EXPECTED_OUT"
readonly OUT="$EXPECTED_OUT"
[[ "$ARTIFACT_DIR" == "$EXPECTED_OUT" ]]
[[ -d "$OUT" && ! -L "$OUT" ]]

if in_blackout; then
  write_json "$OUT/REFUSED_DAILY_NO_JOB_WINDOW.json" "$(python3 - <<PY
import json, os
print(json.dumps({
  "status":"REFUSED_DAILY_NO_JOB_WINDOW",
  "substrate":os.environ["STAGE_S_SUBSTRATE"],
  "run_id":os.environ["PAI_CANARY_RUN_ID"],
  "timezone":"Asia/Shanghai",
  "windows":["09:30-09:40","19:30-19:40"],
  "uid":os.getuid(),
  "gid":os.getgid(),
}, sort_keys=True))
PY
)"
  exit 75
fi

if [[ -f "$OUT/FAILED_${SUBSTRATE}_MAIN.json" ]]; then
  mkdir -p "$OUT/failures"
  mv -- "$OUT/FAILED_${SUBSTRATE}_MAIN.json" "$OUT/failures/FAILED_${SUBSTRATE}_MAIN-$(date +%Y%m%d_%H%M%S)-$$.json"
fi
if [[ -f "$OUT/REFUSED_DAILY_NO_JOB_WINDOW.json" ]]; then
  mkdir -p "$OUT/failures"
  mv -- "$OUT/REFUSED_DAILY_NO_JOB_WINDOW.json" "$OUT/failures/REFUSED_DAILY_NO_JOB_WINDOW-$(date +%Y%m%d_%H%M%S)-$$.json"
fi
trap on_error ERR

PHASE=identity
[[ "$RUN_ID" =~ ^${RUN_PREFIX}[0-9]+$ ]]
[[ "$ARTIFACT_DIR" == "$EXPECTED_OUT" ]]
[[ "$(id -u):$(id -g)" == 2254:2254 ]]
[[ "$(stat -c '%u:%g' "$OUT")" == 2254:2254 ]]
[[ "$(stat -c '%a' "$OUT")" == 700 ]]
[[ "$(nvidia-smi -L | wc -l)" -eq 8 ]]
mapfile -t GPU_INDEXES < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
[[ "${#GPU_INDEXES[@]}" -eq "$WORLD_SIZE" ]]
for index in "${GPU_INDEXES[@]}"; do
  [[ "$index" =~ ^[0-9]+$ ]]
done
export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPU_INDEXES[*]}")"
[[ -x "$PY" ]]
[[ -d "$RUNTIME_REPO/.git" && ! -L "$RUNTIME_REPO" ]]
[[ "$(git -C "$RUNTIME_REPO" rev-parse HEAD)" == "$SOURCE_COMMIT" ]]
[[ -z "$(git -C "$RUNTIME_REPO" status --porcelain)" ]]
[[ "$(git -C "$DEP_ROOT" rev-parse HEAD)" == "$QPILOTS_COMMIT" ]]
[[ "$(git -C "$OPENPI" rev-parse HEAD)" == "$OPENPI_COMMIT" ]]
[[ "$(git -C "$LIBERO" rev-parse HEAD)" == "$LIBERO_COMMIT" ]]
[[ -f "$EXPECTED_PAYLOAD_CONFIG" && ! -L "$EXPECTED_PAYLOAD_CONFIG" ]]

PHASE=payload_binding
CONFIG_PAYLOAD_SHA256=$(python3 - "$EXPECTED_PAYLOAD_CONFIG" "$0" "$SOURCE_COMMIT" <<'PY'
import hashlib
import json
import pathlib
import sys

config = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
runtime = config.get("runtime", {})
if runtime.get("source_commit") != sys.argv[3]:
    raise SystemExit("payload companion source commit mismatch")
if runtime.get("command_file") != str(pathlib.Path(sys.argv[2]).resolve()):
    raise SystemExit("payload companion command_file mismatch")
payload_sha = runtime.get("payload_sha256")
if not isinstance(payload_sha, str) or len(payload_sha) != 64 or any(c not in "0123456789abcdef" for c in payload_sha):
    raise SystemExit("payload companion SHA is not a lowercase SHA-256")
observed = hashlib.sha256(pathlib.Path(sys.argv[2]).read_bytes()).hexdigest()
if observed != payload_sha:
    raise SystemExit("executed payload SHA differs from companion config")
print(payload_sha)
PY
)
[[ "$(sha256sum "$0" | awk '{print $1}')" == "$CONFIG_PAYLOAD_SHA256" ]]

PHASE=runtime_import
export PYTHONPATH="$RUNTIME_REPO/src:$RUNTIME_REPO:$DEP_ROOT:$OPENPI/src:$OPENPI:$LIBERO"
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export EGL_PLATFORM=device
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility,graphics}"
[[ "$NVIDIA_DRIVER_CAPABILITIES" == "compute,utility,graphics" ]]
export XDG_CACHE_HOME="$OUT/xdg-cache"
mkdir -p "$XDG_CACHE_HOME"
"$PY" - <<'PY'
import importlib
for name in ("numpy", "torch", "r142_stage_s.libero", "qpilots_libero.policy", "qpilots_libero.environment"):
    module = importlib.import_module(name)
    print("STAGE_S_MAIN_IMPORT", name, getattr(module, "__version__", "ok"))
PY

if [[ -f "$OUT/COMPLETED_EVALUATION_RESULT.json" || -f "$OUT/SHA256SUMS" ]]; then
  [[ -f "$OUT/COMPLETED_EVALUATION_RESULT.json" && -f "$OUT/SHA256SUMS" ]]
  PHASE=verify_existing_completion
  "$PY" "$MAIN_FINALIZER" \
    --output-root "$OUT" --substrate "$SUBSTRATE" --run-id "$RUN_ID" \
    --source-commit "$SOURCE_COMMIT" --calibration-report "$REPORT" \
    --protocol-acceptance "$PROTOCOL_ACCEPTANCE"
  trap - ERR
  exit 0
fi

PHASE=first_work
if [[ ! -f "$OUT/FIRST_WORK.json" ]]; then
  write_json "$OUT/FIRST_WORK.json" "$(python3 - <<PY
import json, os, time
print(json.dumps({
  "status":"FIRST_WORK",
  "marker_type":"stage_s_main_runtime_started",
  "substrate":os.environ["STAGE_S_SUBSTRATE"],
  "run_id":os.environ["PAI_CANARY_RUN_ID"],
  "uid":os.getuid(),
  "gid":os.getgid(),
  "world_size":8,
  "candidate_budget":32,
  "source_commit":os.environ["STAGE_S_SOURCE_COMMIT"],
  "calibration_report":os.environ.get("STAGE_S_CALIBRATION_REPORT"),
  "protocol_acceptance":os.environ.get("STAGE_S_PROTOCOL_ACCEPTANCE"),
  "replay_gate":"restore -> same action -> next-state <= 1e-9",
  "time":time.time(),
}, sort_keys=True))
PY
)"
fi
PHASE=main_screen
if [[ "$SUBSTRATE" == B ]]; then
  "$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
    "$RUNTIME_REPO/scripts/stage_s_gpu_rank_entry.py" "$RUNTIME_REPO/scripts/stage_s_libero_main.py" \
    --substrate B --output "$OUT" \
    --qpilots-root "$DEP_ROOT" --libero-root "$LIBERO" \
    --checkpoint "$POLICY_CHECKPOINT" \
    --variant-root "$B_VARIANT_ROOT" \
    --source-init-root "$LIBERO/libero/libero/init_files/libero_10" \
    --calibration-report "$REPORT" --protocol-acceptance "$PROTOCOL_ACCEPTANCE" \
    --candidate-count "$CANDIDATE_COUNT" \
    --max-steps 520 --validate-snapshots --source-commit "$SOURCE_COMMIT"
else
  "$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
    "$RUNTIME_REPO/scripts/stage_s_gpu_rank_entry.py" "$RUNTIME_REPO/scripts/stage_s_libero_main.py" \
    --substrate C --output "$OUT" \
    --qpilots-root "$DEP_ROOT" --libero-root "$LIBERO" \
    --calibration-report "$REPORT" --protocol-acceptance "$PROTOCOL_ACCEPTANCE" \
    --candidate-count "$CANDIDATE_COUNT" \
    --max-steps 520 --validate-snapshots --weak-substrate --source-commit "$SOURCE_COMMIT"
fi

PHASE=aggregate
[[ ! -f "$OUT/FAILED_${SUBSTRATE}_MAIN.json" ]]
if in_blackout; then
  write_json "$OUT/REFUSED_DAILY_NO_JOB_WINDOW.json" "$(python3 - <<PY
import json, os
print(json.dumps({"status":"REFUSED_DAILY_NO_JOB_WINDOW","substrate":os.environ["STAGE_S_SUBSTRATE"],"run_id":os.environ["PAI_CANARY_RUN_ID"],"timezone":"Asia/Shanghai","windows":["09:30-09:40","19:30-19:40"]}, sort_keys=True))
PY
)"
  exit 75
fi
"$PY" "$MAIN_FINALIZER" \
  --output-root "$OUT" --substrate "$SUBSTRATE" --run-id "$RUN_ID" \
  --source-commit "$SOURCE_COMMIT" --calibration-report "$REPORT" \
  --protocol-acceptance "$PROTOCOL_ACCEPTANCE"

trap - ERR
