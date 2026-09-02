#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

# One formal idle job uses one worker with eight explicit server/client pairs.
# PAI restarts the same foreground command; completed family directories are
# immutable and are skipped by stage_s_robotwin_main.py on the next incarnation.
RUN_ID="${PAI_STAGE_S_RUN_ID:-${PAI_CANARY_RUN_ID:?controller must inject run id}}"
EXPECTED_GPUS="${PAI_CANARY_EXPECTED_GPUS:-8}"
WORLD_SIZE=8
ROOT=/mnt/cpfs/zbl-cpfs-new/USERS/leon
readonly REQUIRED_RUNTIME_REPO="$ROOT/code/r142-stage-s-a-runtime-20260903"
REPO="$REQUIRED_RUNTIME_REPO"
DEPS="$ROOT/code/r142_stage_s_deps"
EVO_ROOT="$DEPS/Evo-1"
ROBOTWIN_ROOT="$ROOT/cache/r142_stage_s/runtime/RoboTwin"
CUROBO_ROOT="$ROBOTWIN_ROOT/envs/curobo"
CHECKPOINT_DIR="$ROOT/cache/r142_stage_s/models/Evo1_RoboTwin2_clean_ce8c583724706fbf7a03c17237761c65bf6813a7"
CLIENT_PY="$ROOT/cache/r142_stage_s/envs/robotwin_py310/bin/python"
SERVER_PY="$ROOT/cache/r142_stage_s/envs/evo1_py310/bin/python"
OUT="$ROOT/logs/r142_fp11_stage_s/a_main/$RUN_ID"
BASE_PORT="${STAGE_S_EVO_SERVER_BASE_PORT:-19000}"
readonly FROZEN_SOURCE_COMMIT="c2bd51db6de0e22d09827d06460cbac8d47bb6ae"
STAGE_S_SOURCE_COMMIT="${STAGE_S_SOURCE_COMMIT:-$FROZEN_SOURCE_COMMIT}"
readonly ASSET_PREFLIGHT_RUN_ID="r142-stage-s-a-assets-20260902-r15"
ASSET_PREFLIGHT_DIR="$ROOT/logs/r142_fp11_stage_s/assets/$ASSET_PREFLIGHT_RUN_ID"
readonly FROZEN_PROTOCOL_PATH="$ROOT/stage_s/protocol/FROZEN_PROTOCOL.json"
export STAGE_S_SOURCE_COMMIT

SERVER_PIDS=()
CLIENT_PIDS=()
FAILURE_WRITTEN=0

blocked_window() {
  local hm
  hm=$(TZ=Asia/Shanghai date +%H%M)
  case "$hm" in
    09[3][0-9]|19[3][0-9]) return 0 ;;
    *) return 1 ;;
  esac
}

write_json() {
  local path="$1"
  shift
  python3 - "$path" "$@" <<'PY'
import json, os, pathlib, sys, tempfile
path = pathlib.Path(sys.argv[1])
payload = json.loads(sys.argv[2])
path.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
try:
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2)
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
  [[ "$FAILURE_WRITTEN" -eq 0 ]] || return 0
  FAILURE_WRITTEN=1
  local payload
  payload=$(python3 - "$rc" <<'PY'
import json, os, sys, time
print(json.dumps({
    "status": "FAILED",
    "stage": "stage_s_a_main",
    "exit_code": int(sys.argv[1]),
    "job_id": os.environ.get("PAI_TASK_JOB_ID"),
    "run_id": os.environ.get("PAI_STAGE_S_RUN_ID") or os.environ.get("PAI_CANARY_RUN_ID"),
    "source_commit": os.environ.get("STAGE_S_SOURCE_COMMIT"),
    "time": time.time(),
}, sort_keys=True))
PY
  )
  write_json "$OUT/FAILED_A_MAIN.json" "$payload"
}

cleanup() {
  local pid
  for pid in "${CLIENT_PIDS[@]}" "${SERVER_PIDS[@]}"; do
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${CLIENT_PIDS[@]}" "${SERVER_PIDS[@]}"; do
    if [[ -n "${pid:-}" ]]; then
      wait "$pid" 2>/dev/null || true
    fi
  done
}

on_error() {
  local rc=$?
  trap - ERR
  write_failure "$rc" || true
  cleanup
  exit "$rc"
}

if blocked_window; then
  mkdir -p "$OUT"
  printf '%s\n' 'REFUSED_DAILY_NO_JOB_WINDOW' >"$OUT/REFUSED_WINDOW.txt"
  exit 75
fi

mkdir -p "$OUT/logs"
# A previous incarnation may have failed before PAI resumed the same run
# directory. Preserve that evidence under a stable failure history path so a
# later complete aggregate is not confused with an active failure marker.
if [[ -f "$OUT/FAILED_A_MAIN.json" ]]; then
  mkdir -p "$OUT/failures"
  mv "$OUT/FAILED_A_MAIN.json" "$OUT/failures/FAILED_A_MAIN-$(date +%Y%m%d_%H%M%S)-$$.json"
fi
if [[ -f "$OUT/REFUSED_WINDOW.txt" ]]; then
  mkdir -p "$OUT/failures"
  mv "$OUT/REFUSED_WINDOW.txt" "$OUT/failures/REFUSED_WINDOW-$(date +%Y%m%d_%H%M%S)-$$.txt"
fi
trap on_error ERR

[[ "$EXPECTED_GPUS" -eq 8 ]]
[[ "$WORLD_SIZE" -eq 8 ]]
[[ "$BASE_PORT" =~ ^[0-9]+$ ]] && [[ "$BASE_PORT" -ge 1024 ]] && [[ "$BASE_PORT" -le 65527 ]]
[[ "$REPO" == "$REQUIRED_RUNTIME_REPO" ]]
[[ "$STAGE_S_SOURCE_COMMIT" == "$FROZEN_SOURCE_COMMIT" ]]
[[ "$(id -u):$(id -g)" == 2254:2254 ]]
[[ "$(stat -c '%u:%g' "$OUT")" == 2254:2254 ]]
[[ "$(nvidia-smi -L | wc -l)" -eq 8 ]]
[[ -x "$CLIENT_PY" && -x "$SERVER_PY" ]]

[[ -d "$REPO/.git" ]]
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$STAGE_S_SOURCE_COMMIT" ]]
[[ -z "$(git -C "$REPO" status --porcelain)" ]]
[[ "$(git -C "$EVO_ROOT" rev-parse HEAD)" == 5fd14b015013c4fd0aacf5f8f48f868ca9b870a2 ]]
[[ "$(git -C "$ROBOTWIN_ROOT" rev-parse HEAD)" == 13c3c47ff4312dd62484bcd51be034af55c062d1 ]]
[[ "$(git -C "$CUROBO_ROOT" rev-parse HEAD)" == d64c4b005459db10c5dd867d8b30a87d5bda9bdb ]]
[[ -f "$EVO_ROOT/Evo_1/scripts/Evo1_server.py" ]]
[[ -f "$REPO/scripts/stage_s_robotwin_evo_server.py" ]]
[[ -f "$REPO/scripts/stage_s_robotwin_evo_server_patch.py" ]]
[[ -f "$REPO/scripts/stage_s_robotwin_main.py" ]]
[[ -f "$REPO/scripts/stage_s_robotwin_finalize.py" ]]
[[ -f "$REPO/src/r142_stage_s/frozen_protocol.py" ]]
[[ -d "$ROBOTWIN_ROOT/assets" ]]

# Every A main incarnation reads the same stable CPFS authority before a
# server is started.  The helper verifies the status, commit, referenced
# PROTOCOL.md/B/C report bytes, and frozen threshold/seed/task/budget summary.
# Its non-zero exit is intentionally not recoverable within this incarnation.
[[ -f "$FROZEN_PROTOCOL_PATH" ]]
"$CLIENT_PY" "$REPO/src/r142_stage_s/frozen_protocol.py" \
  --path "$FROZEN_PROTOCOL_PATH" >/dev/null

# The r15 asset preflight is a hard prerequisite for the formal screen.  Its
# output is immutable evidence from the separate asset job; a FIRST_WORK file,
# a running job, or a partial cache never satisfies this gate.
[[ -d "$ASSET_PREFLIGHT_DIR" ]]
[[ -s "$ASSET_PREFLIGHT_DIR/COMPLETED_ASSET_PREFLIGHT.json" ]]
[[ -s "$ASSET_PREFLIGHT_DIR/SHA256SUMS" ]]
(
  cd "$ASSET_PREFLIGHT_DIR"
  sha256sum --check --quiet SHA256SUMS
)
python3 - "$ASSET_PREFLIGHT_DIR/COMPLETED_ASSET_PREFLIGHT.json" <<'PY'
import json, pathlib, sys
marker = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "status": "COMPLETED",
    "gpus": 8,
    "model_revision": "ce8c583724706fbf7a03c17237761c65bf6813a7",
    "robotwin_commit": "13c3c47ff4312dd62484bcd51be034af55c062d1",
    "evo_commit": "5fd14b015013c4fd0aacf5f8f48f868ca9b870a2",
    "curobo_commit": "d64c4b005459db10c5dd867d8b30a87d5bda9bdb",
}
for key, value in expected.items():
    if marker.get(key) != value:
        raise SystemExit(f"asset preflight r15 mismatch: {key}={marker.get(key)!r}")
if not marker.get("job_id"):
    raise SystemExit("asset preflight r15 marker lacks terminal PAI JobId")
PY

for file in config.json norm_stats.json mp_rank_00_model_states.pt SHA256SUMS; do
  [[ -s "$CHECKPOINT_DIR/$file" ]]
done
(cd "$CHECKPOINT_DIR" && sha256sum --check --quiet SHA256SUMS)

export PYTHONPATH="$REPO/src:$REPO:$ROBOTWIN_ROOT:$EVO_ROOT"
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export EGL_PLATFORM=device
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility,graphics}"
export XDG_CACHE_HOME="$OUT/xdg-cache"
export PAI_STAGE_S_RUN_ID="$RUN_ID"
mkdir -p "$XDG_CACHE_HOME"

# A successful directory is immutable. On a platform restart after completion,
# verify it and exit before starting servers that would rewrite PID-bearing
# readiness records and invalidate the aggregate checksum.
if [[ -f "$OUT/COMPLETED_EVALUATION_RESULT.json" && -f "$OUT/SHA256SUMS" ]]; then
  "$CLIENT_PY" "$REPO/scripts/stage_s_robotwin_finalize.py" \
    --output-root "$OUT" --run-id "$RUN_ID" \
    --job-id "${PAI_TASK_JOB_ID:-}" --source-commit "$STAGE_S_SOURCE_COMMIT"
  [[ "$(stat -c '%u:%g' "$OUT/COMPLETED_EVALUATION_RESULT.json")" == 2254:2254 ]]
  [[ "$(stat -c '%u:%g' "$OUT/SHA256SUMS")" == 2254:2254 ]]
  trap - ERR
  exit 0
fi

# The client import covers only the real pinned adapter; it does not create a
# mock rollout or grant scientific completion.
"$CLIENT_PY" - <<'PY'
import importlib
for name in ("numpy", "torch", "sapien", "websockets", "r142_stage_s.robotwin"):
    module = importlib.import_module(name)
    print("STAGE_S_CLIENT_IMPORT", name, getattr(module, "__version__", "ok"))
PY

probe="$OUT/.ownership-probe-$$"
printf '%s\n' ownership >"$probe"
[[ "$(stat -c '%u:%g' "$probe")" == 2254:2254 ]]
rm -f "$probe"

if [[ ! -f "$OUT/FIRST_WORK.json" ]]; then
  payload=$(python3 - <<'PY'
import json, os, time
print(json.dumps({
    "status": "FIRST_WORK",
    "uid": os.getuid(), "gid": os.getgid(), "gpus": 8,
    "world_size": 8,
    "run_id": os.environ["PAI_STAGE_S_RUN_ID"],
    "source_commit": os.environ["STAGE_S_SOURCE_COMMIT"],
    "server_client_ownership": "one_server_one_client_one_gpu_one_port",
    "time": time.time(),
}, sort_keys=True))
PY
  )
  write_json "$OUT/FIRST_WORK.json" "$payload"
fi

python3 - "$OUT/SERVER_ASSIGNMENT.json" "$BASE_PORT" "$STAGE_S_SOURCE_COMMIT" <<'PY'
import json, os, pathlib, sys, tempfile
path, base, commit = pathlib.Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
payload = {
    "protocol": "R142-FP-11 Stage-S substrate A",
    "status": "FROZEN",
    "world_size": 8,
    "run_id": os.environ["PAI_STAGE_S_RUN_ID"],
    "source_commit": commit,
    "server_client_ownership": "one_server_one_client_one_gpu_one_port",
    "assignments": [
        {"rank": rank, "gpu_id": rank, "server_port": base + rank,
         "server_url": f"ws://127.0.0.1:{base + rank}"}
        for rank in range(8)
    ],
}
fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
try:
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)
finally:
    if os.path.exists(tmp): os.unlink(tmp)
PY

wait_for_server() {
  local rank="$1"
  local port=$((BASE_PORT + rank))
  local pid="${SERVER_PIDS[$rank]}"
  local attempt
  for ((attempt=0; attempt<180; attempt++)); do
    if ! kill -0 "$pid" 2>/dev/null; then
      return 1
    fi
    if [[ -f "$OUT/SERVER_READY_RANK-$(printf '%04d' "$rank").json" ]] && \
       CUDA_VISIBLE_DEVICES="$rank" "$CLIENT_PY" - "$port" "$rank" <<'PY'
import asyncio, json, sys
import websockets

async def probe():
    port, rank = int(sys.argv[1]), int(sys.argv[2])
    async with websockets.connect(
        f"ws://127.0.0.1:{port}", ping_interval=None, ping_timeout=None,
        max_size=100_000_000,
    ) as ws:
        await ws.send(json.dumps({
            "r142_control": "capture_rng",
            "protocol": "r142-evo-exact-replay/v1",
            "request_id": f"ready-rank-{rank}",
        }, sort_keys=True, separators=(",", ":")))
        response = json.loads(await ws.recv())
        if response.get("r142_control") != "ok" or response.get("protocol") != "r142-evo-exact-replay/v1":
            raise RuntimeError(f"server control probe failed: {response}")

asyncio.run(probe())
PY
    then
      [[ "$(stat -c '%u:%g' "$OUT/SERVER_READY_RANK-$(printf '%04d' "$rank").json")" == 2254:2254 ]]
      return 0
    fi
    sleep 2
  done
  return 1
}

for rank in $(seq 0 7); do
  port=$((BASE_PORT + rank))
  (
    export CUDA_VISIBLE_DEVICES="$rank"
    exec "$SERVER_PY" "$REPO/scripts/stage_s_robotwin_evo_server.py" \
      --stage-s-root "$REPO" \
      --evo-root "$EVO_ROOT" \
      --checkpoint-dir "$CHECKPOINT_DIR" \
      --output-root "$OUT" \
      --rank "$rank" --world-size "$WORLD_SIZE" \
      --gpu-id "$rank" --port "$port"
  ) >"$OUT/logs/server-rank-$(printf '%04d' "$rank").log" 2>&1 &
  SERVER_PIDS[$rank]=$!
done

for rank in $(seq 0 7); do
  wait_for_server "$rank"
done

for rank in $(seq 0 7); do
  port=$((BASE_PORT + rank))
  (
    export CUDA_VISIBLE_DEVICES="$rank"
    exec "$CLIENT_PY" "$REPO/scripts/stage_s_robotwin_main.py" \
      --phase main \
      --robotwin-root "$ROBOTWIN_ROOT" \
      --evo-root "$EVO_ROOT" \
      --checkpoint-dir "$CHECKPOINT_DIR" \
      --output-root "$OUT" \
      --server-url "ws://127.0.0.1:$port" \
      --rank "$rank" --world-size "$WORLD_SIZE" \
      --families-per-task 16 --candidates 32 --seed-base 14211 \
      --frozen-protocol "$FROZEN_PROTOCOL_PATH"
  ) >"$OUT/logs/client-rank-$(printf '%04d' "$rank").log" 2>&1 &
  CLIENT_PIDS[$rank]=$!
done

for rank in $(seq 0 7); do
  pid="${CLIENT_PIDS[$rank]}"
  if wait "$pid"; then
    :
  else
    rc=$?
    exit "$rc"
  fi
done

# No server may continue mutating evidence while the aggregate verifier hashes
# it. Clients have completed all requests, so this bounded shutdown is safe.
for pid in "${SERVER_PIDS[@]}"; do
  kill "$pid" 2>/dev/null || true
done
for pid in "${SERVER_PIDS[@]}"; do
  wait "$pid" 2>/dev/null || true
done
SERVER_PIDS=()

"$CLIENT_PY" "$REPO/scripts/stage_s_robotwin_finalize.py" \
  --output-root "$OUT" --run-id "$RUN_ID" \
  --job-id "${PAI_TASK_JOB_ID:-}" --source-commit "$STAGE_S_SOURCE_COMMIT" \
  --frozen-protocol "$FROZEN_PROTOCOL_PATH"
[[ -f "$OUT/COMPLETED_EVALUATION_RESULT.json" && -f "$OUT/SHA256SUMS" ]]
[[ "$(stat -c '%u:%g' "$OUT/COMPLETED_EVALUATION_RESULT.json")" == 2254:2254 ]]
[[ "$(stat -c '%u:%g' "$OUT/SHA256SUMS")" == 2254:2254 ]]
(cd "$OUT" && sha256sum --check --quiet SHA256SUMS)

trap - ERR
exit 0
