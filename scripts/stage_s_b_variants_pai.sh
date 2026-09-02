#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

RUN_ID="${PAI_CANARY_RUN_ID:?controller must inject run id}"
EXPECTED_GPUS="${PAI_CANARY_EXPECTED_GPUS:?controller must inject GPU count}"
STAGE_S_SOURCE_COMMIT=afe353bbc5997355f35cb0c77c5446fd4df5f1e3
ROOT=/mnt/cpfs/zbl-cpfs-new/USERS/leon
REPO="$ROOT/code/r142-stage-s-runtime-20260902"
QPILOTS="$ROOT/code/QPILOTS-r16p15-stage1-task64-20260812"
OPENPI="$QPILOTS/third_party/openpi"
LIBERO="$OPENPI/third_party/libero"
PYTHON="$ROOT/envs/openpi_py311/bin/python"
OUT="$ROOT/logs/r142_fp11_stage_s/b_variants/$RUN_ID"
VARIANTS="$OUT/variants"
SOURCE_BDDL="$LIBERO/libero/libero/bddl_files/libero_10"
SOURCE_INIT="$LIBERO/libero/libero/init_files/libero_10"
mkdir -p "$OUT"

blocked_window() {
  local hm
  hm=$(TZ=Asia/Shanghai date +%H%M)
  case "$hm" in
    09[3][0-9]|19[3][0-9]) return 0 ;;
    *) return 1 ;;
  esac
}
if blocked_window; then
  printf '%s\n' 'REFUSED_DAILY_NO_JOB_WINDOW' >"$OUT/REFUSED_WINDOW.txt"
  exit 75
fi

on_error() {
  local rc=$?
  "$PYTHON" - "$OUT/FAILED_B_VARIANTS.json" "$rc" <<'PY'
import json, os, pathlib, sys, time
path, rc = pathlib.Path(sys.argv[1]), int(sys.argv[2])
tmp = path.with_suffix(path.suffix + ".tmp")
with tmp.open("w", encoding="utf-8") as f:
    json.dump({"status": "FAILED", "exit_code": rc, "job_id": os.getenv("PAI_TASK_JOB_ID"), "time": time.time()}, f, sort_keys=True, indent=2)
    f.write("\n"); f.flush(); os.fsync(f.fileno())
os.replace(tmp, path)
PY
  exit "$rc"
}
trap on_error ERR

if [[ -f "$OUT/COMPLETED_B_VARIANTS.json" && -f "$OUT/SHA256SUMS" ]]; then
  (cd "$OUT" && sha256sum --check --quiet SHA256SUMS)
  exit 0
fi

[[ "$(id -u):$(id -g)" == 2254:2254 ]]
[[ "$(nvidia-smi -L | wc -l)" -eq "$EXPECTED_GPUS" ]]
[[ "$EXPECTED_GPUS" -eq 8 ]]
[[ -x "$PYTHON" ]]
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$STAGE_S_SOURCE_COMMIT" ]]
[[ -z "$(git -C "$REPO" status --porcelain)" ]]
[[ "$(git -C "$QPILOTS" rev-parse HEAD)" == eacf47b981e3b22357f8a74902f8dad8cfcfa375 ]]
[[ -z "$(git -C "$QPILOTS" status --porcelain)" ]]
[[ "$(git -C "$OPENPI" rev-parse HEAD)" == 54cbaee6ae0c010a1ed431871cdaa8f4684ac709 ]]
[[ "$(git -C "$LIBERO" rev-parse HEAD)" == f78abd68ee283de9f9be3c8f7e2a9ad60246e95c ]]
[[ -d "$SOURCE_BDDL" && -d "$SOURCE_INIT" ]]

"$PYTHON" - "$OUT/FIRST_WORK.json" "$STAGE_S_SOURCE_COMMIT" <<'PY'
import json, os, pathlib, sys, time
path, commit = pathlib.Path(sys.argv[1]), sys.argv[2]
tmp = path.with_suffix(path.suffix + ".tmp")
with tmp.open("w", encoding="utf-8") as f:
    json.dump({"status": "FIRST_WORK", "uid": os.getuid(), "gid": os.getgid(), "gpus": 8, "stage_s_source_commit": commit, "time": time.time()}, f, sort_keys=True, indent=2)
    f.write("\n"); f.flush(); os.fsync(f.fileno())
os.replace(tmp, path)
PY

export PYTHONPATH="$REPO/src:$QPILOTS:$OPENPI:$LIBERO"
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export EGL_PLATFORM=device
export XDG_CACHE_HOME="$OUT/xdg-cache"
export LIBERO_CONFIG_PATH="$OUT/libero-config"
mkdir -p "$XDG_CACHE_HOME" "$LIBERO_CONFIG_PATH"

# Pinned LIBERO prompts on first import when ~/.libero/config.yaml is absent.
# A PAI task is non-interactive, so materialize the exact source-tree paths
# explicitly instead of accepting an implicit per-node HOME state.
"$PYTHON" - "$LIBERO_CONFIG_PATH/config.yaml" "$LIBERO" <<'PY'
import json, os, pathlib, sys
path, libero = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
root = libero / "libero" / "libero"
payload = {
    "benchmark_root": str(root),
    "bddl_files": str(root / "bddl_files"),
    "init_states": str(root / "init_files"),
    "datasets": str(libero / "libero" / "datasets"),
    "assets": str(root / "assets"),
}
tmp = path.with_suffix(path.suffix + ".tmp")
with tmp.open("w", encoding="utf-8") as f:
    json.dump(payload, f, sort_keys=True, indent=2)
    f.write("\n"); f.flush(); os.fsync(f.fileno())
os.replace(tmp, path)
PY

cd "$REPO"
"$PYTHON" scripts/stage_s_libero_b.py \
  --source-bddl-root "$SOURCE_BDDL" \
  --source-init-root "$SOURCE_INIT" \
  --libero-root "$LIBERO" \
  --output-root "$VARIANTS" \
  --count 16 \
  --seed-base 142011 \
  --render-gpu-device-id 0

"$PYTHON" - "$VARIANTS" "$OUT/COMPLETED_B_VARIANTS.json" "$STAGE_S_SOURCE_COMMIT" <<'PY'
import hashlib, json, os, pathlib, sys, time
root, marker, commit = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
matrix = json.loads((root / "B_VARIANT_MATRIX.json").read_text(encoding="utf-8"))
settings = matrix.get("settings", [])
if len(settings) != 4:
    raise SystemExit(f"expected 4 settings, got {len(settings)}")
for setting in settings:
    if len(setting.get("tasks", [])) != 10:
        raise SystemExit("every B setting must contain 10 tasks")
    for task in setting["tasks"]:
        for key in ("bddl_path", "init_states_path"):
            path = pathlib.Path(task[key])
            if not path.is_file() or path.stat().st_size <= 0:
                raise SystemExit(f"missing generated artifact: {path}")
payload = {
    "status": "COMPLETED",
    "marker_type": "completed_b_variant_generation",
    "job_id": os.getenv("PAI_TASK_JOB_ID"),
    "stage_s_source_commit": commit,
    "settings": 4,
    "tasks_per_setting": 10,
    "init_states_per_task": 16,
    "old_init_reused": False,
    "matrix_sha256": hashlib.sha256((root / "B_VARIANT_MATRIX.json").read_bytes()).hexdigest(),
    "completed_at": time.time(),
}
tmp = marker.with_suffix(marker.suffix + ".tmp")
with tmp.open("w", encoding="utf-8") as f:
    json.dump(payload, f, sort_keys=True, indent=2)
    f.write("\n"); f.flush(); os.fsync(f.fileno())
os.replace(tmp, marker)
PY

rm -f "$OUT/FAILED_B_VARIANTS.json"
(cd "$OUT" && { find variants -type f -print0 | sort -z | xargs -0 sha256sum; sha256sum FIRST_WORK.json COMPLETED_B_VARIANTS.json; } >SHA256SUMS.tmp && mv SHA256SUMS.tmp SHA256SUMS)
(cd "$OUT" && sha256sum --check --quiet SHA256SUMS)
trap - ERR
