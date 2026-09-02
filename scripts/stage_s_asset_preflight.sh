#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

RUN_ID="${PAI_CANARY_RUN_ID:?controller must inject run id}"
EXPECTED_GPUS="${PAI_CANARY_EXPECTED_GPUS:?controller must inject GPU count}"
ROOT=/mnt/cpfs/zbl-cpfs-new/USERS/leon
DEPS="$ROOT/code/r142_stage_s_deps"
RT="$ROOT/cache/r142_stage_s/runtime/RoboTwin"
OUT="$ROOT/logs/r142_fp11_stage_s/assets/$RUN_ID"
MODEL="$ROOT/cache/r142_stage_s/models/Evo1_RoboTwin2_clean_ce8c583724706fbf7a03c17237761c65bf6813a7"
TOOLS_ENV="$ROOT/cache/r142_stage_s/envs/tools_py311"
RT_ENV="$ROOT/cache/r142_stage_s/envs/robotwin_py310"
EVO_ENV="$ROOT/cache/r142_stage_s/envs/evo1_py310"
PIP_CACHE="$ROOT/cache/r142_stage_s/pip"
FLASH_ATTN_TMP="$PIP_CACHE/flash-attn-tmp/$RUN_ID"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
mkdir -p "$OUT" "$MODEL" "$PIP_CACHE" "$FLASH_ATTN_TMP"
export PIP_CACHE_DIR="$PIP_CACHE"
export PIP_INDEX_URL
export PIP_DEFAULT_TIMEOUT=120
export HF_ENDPOINT
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_ETAG_TIMEOUT=60

# pip's wheel cache is on CPFS while its default temporary build directory is
# commonly pod-local /tmp.  A flash-attn build can therefore hit Errno 18 when
# pip renames a cached wheel across filesystems.  Keep only this wheel's build
# temporary files on the same CPFS device and disable its cache entirely; HOME
# remains inherited and is never rewritten.
[[ "$(realpath -e "$FLASH_ATTN_TMP")" == "$FLASH_ATTN_TMP" ]]
[[ "$(stat -c '%d' "$FLASH_ATTN_TMP")" == "$(stat -c '%d' "$PIP_CACHE")" ]]

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
  python3 - "$OUT/FAILED_ASSET_PREFLIGHT.json" "$rc" <<'PY'
import json, os, sys, time
path, rc = sys.argv[1], int(sys.argv[2])
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump({"status": "FAILED", "exit_code": rc, "job_id": os.getenv("PAI_TASK_JOB_ID"), "time": time.time()}, f, sort_keys=True, indent=2)
    f.write("\n"); f.flush(); os.fsync(f.fileno())
os.replace(tmp, path)
PY
  exit "$rc"
}
trap on_error ERR

actual_gpus=$(nvidia-smi -L | wc -l)
[[ "$actual_gpus" -eq "$EXPECTED_GPUS" ]]
[[ "$EXPECTED_GPUS" -eq 8 ]]
[[ "$(id -u):$(id -g)" == 2254:2254 ]]
[[ "$(git -C "$DEPS/Evo-1" rev-parse HEAD)" == 5fd14b015013c4fd0aacf5f8f48f868ca9b870a2 ]]
[[ "$(git -C "$DEPS/RoboTwin" rev-parse HEAD)" == 13c3c47ff4312dd62484bcd51be034af55c062d1 ]]
[[ "$(git -C "$DEPS/RoboTwin/envs/curobo" rev-parse HEAD)" == d64c4b005459db10c5dd867d8b30a87d5bda9bdb ]]
python3 - "$OUT/FIRST_WORK.json" "$actual_gpus" <<'PY'
import json, os, pathlib, sys, time
path, gpus = pathlib.Path(sys.argv[1]), int(sys.argv[2])
tmp = path.with_suffix(path.suffix + ".tmp")
with tmp.open("w", encoding="utf-8") as f:
    json.dump({"uid": os.getuid(), "gid": os.getgid(), "gpus": gpus, "time": time.time()}, f, sort_keys=True)
    f.write("\n"); f.flush(); os.fsync(f.fileno())
os.replace(tmp, path)
PY

if [[ ! -d "$RT/.git" ]]; then
  mkdir -p "$(dirname "$RT")"
  git clone --no-local "$DEPS/RoboTwin" "$RT"
fi
git -C "$RT" checkout --detach 13c3c47ff4312dd62484bcd51be034af55c062d1
if [[ ! -d "$RT/envs/curobo/.git" ]]; then
  git clone --no-local "$DEPS/RoboTwin/envs/curobo" "$RT/envs/curobo"
fi
git -C "$RT/envs/curobo" checkout --detach d64c4b005459db10c5dd867d8b30a87d5bda9bdb

if [[ ! -x "$TOOLS_ENV/bin/python" ]]; then
  python3 -m venv "$TOOLS_ENV"
elif ! "$TOOLS_ENV/bin/python" -m pip --version >/dev/null 2>&1; then
  "$TOOLS_ENV/bin/python" -m ensurepip --upgrade
fi
if ! "$TOOLS_ENV/bin/python" -c 'import huggingface_hub, uv' >/dev/null 2>&1; then
  "$TOOLS_ENV/bin/python" -m pip install --retries 8 'huggingface_hub==0.36.2' 'uv==0.8.17'
fi
"$TOOLS_ENV/bin/hf" download MINT-SJTU/Evo1_RoboTwin2_clean \
  --revision ce8c583724706fbf7a03c17237761c65bf6813a7 --local-dir "$MODEL"

(
  cd "$RT/assets"
  "$TOOLS_ENV/bin/python" _download.py
  for archive in background_texture.zip embodiments.zip objects.zip; do
    [[ -s "$archive" ]]
    unzip -q -n "$archive"
  done
)

export UV_PYTHON_INSTALL_DIR="$ROOT/cache/r142_stage_s/uv-python"
export UV_CACHE_DIR="$ROOT/cache/r142_stage_s/uv-cache"
export UV_NO_CONFIG=1
mkdir -p "$UV_PYTHON_INSTALL_DIR" "$UV_CACHE_DIR"
if [[ ! -x "$RT_ENV/bin/python" ]]; then
  "$TOOLS_ENV/bin/uv" python install 3.10
  "$TOOLS_ENV/bin/uv" venv --seed --python 3.10 "$RT_ENV"
fi
"$RT_ENV/bin/pip" install --retries 8 -r "$RT/script/requirements.txt" websockets
# SAPIEN imports the legacy pkg_resources module from setuptools during the
# RoboTwin smoke test.  New setuptools releases omit that module, so pin the
# provider in the exact runtime environment before importing SAPIEN.
"$RT_ENV/bin/pip" install --retries 8 "setuptools<81"
"$RT_ENV/bin/pip" install --retries 8 -e "$RT/envs/curobo" --no-build-isolation

if [[ ! -x "$EVO_ENV/bin/python" ]]; then
  "$TOOLS_ENV/bin/uv" venv --seed --python 3.10 "$EVO_ENV"
fi
"$EVO_ENV/bin/pip" install --retries 8 -r "$DEPS/Evo-1/Evo_1/requirements.txt"
# Keep the same compatibility provider in Evo-1 as well.  These two explicit
# environment pins are infrastructure constraints, not model or protocol
# changes.
"$EVO_ENV/bin/pip" install --retries 8 "setuptools<81"
# Do not let this wheel use the persistent pip cache.  TMPDIR is deliberately
# below the same CPFS root so build/install renames never cross filesystems.
TMPDIR="$FLASH_ATTN_TMP" PIP_NO_CACHE_DIR=1 MAX_JOBS=32 \
  "$EVO_ENV/bin/pip" install --retries 8 --no-cache-dir flash-attn --no-build-isolation

rm -rf "$RT/policy/Evo1.tmp"
cp -a "$DEPS/Evo-1/RoboTwin_evaluation/policy/Evo1" "$RT/policy/Evo1.tmp"
rm -rf "$RT/policy/Evo1"
mv "$RT/policy/Evo1.tmp" "$RT/policy/Evo1"

NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-} "$RT_ENV/bin/python" - <<'PY' >"$OUT/robotwin_import_smoke.txt"
import importlib
for name in ("torch", "sapien", "mplib", "curobo", "websockets"):
    module = importlib.import_module(name)
    print(name, getattr(module, "__version__", "unknown"))
PY
"$EVO_ENV/bin/python" - <<'PY' >"$OUT/evo_import_smoke.txt"
import importlib
for name in ("torch", "transformers", "flash_attn", "websockets"):
    module = importlib.import_module(name)
    print(name, getattr(module, "__version__", "unknown"))
PY
nvidia-smi -L >"$OUT/nvidia-smi-L.txt"
if command -v vulkaninfo >/dev/null 2>&1; then vulkaninfo --summary >"$OUT/vulkan-summary.txt" 2>&1; fi

for file in config.json norm_stats.json mp_rank_00_model_states.pt; do [[ -s "$MODEL/$file" ]]; done
find "$MODEL" -maxdepth 1 -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >"$MODEL/SHA256SUMS"

python3 - "$OUT/COMPLETED_ASSET_PREFLIGHT.json" "$MODEL" "$actual_gpus" "$FLASH_ATTN_TMP" <<'PY'
import hashlib, json, os, pathlib, sys, time
path, model, gpus, flash_tmp = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), int(sys.argv[3]), pathlib.Path(sys.argv[4])
payload = {
  "status": "COMPLETED",
  "job_id": os.getenv("PAI_TASK_JOB_ID"),
  "gpus": gpus,
  "model_revision": "ce8c583724706fbf7a03c17237761c65bf6813a7",
  "model_sha256sums_sha256": hashlib.sha256((model / "SHA256SUMS").read_bytes()).hexdigest(),
  "robotwin_commit": "13c3c47ff4312dd62484bcd51be034af55c062d1",
  "evo_commit": "5fd14b015013c4fd0aacf5f8f48f868ca9b870a2",
  "curobo_commit": "d64c4b005459db10c5dd867d8b30a87d5bda9bdb",
  "flash_attn_install": {
    "package": "flash-attn",
    "cache_policy": "disabled",
    "pip_no_cache_dir": True,
    "tmpdir": str(flash_tmp),
    "tmpdir_under_new_root": str(flash_tmp).startswith("/mnt/cpfs/zbl-cpfs-new/"),
    "home_unchanged": True,
  },
  "pkg_resources_compat": {
    "provider": "setuptools",
    "version_constraint": "<81",
    "environments": ["robotwin_py310", "evo1_py310"],
  },
  "completed_at": time.time(),
}
tmp = path.with_suffix(path.suffix + ".tmp")
with tmp.open("w", encoding="utf-8") as f:
    json.dump(payload, f, sort_keys=True, indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())
os.replace(tmp, path)
PY
(cd "$OUT" && sha256sum COMPLETED_ASSET_PREFLIGHT.json FIRST_WORK.json evo_import_smoke.txt nvidia-smi-L.txt robotwin_import_smoke.txt >SHA256SUMS)
trap - ERR
