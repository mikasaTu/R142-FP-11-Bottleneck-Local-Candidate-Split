#!/usr/bin/env bash
set -euo pipefail

NEW_ROOT=/mnt/cpfs/zbl-cpfs-new
USER_ROOT=$NEW_ROOT/USERS/leon
PROJECT_DIR=$USER_ROOT/code/R142-FP-11-Bottleneck-Local-Candidate-Split
PYTHON_BIN=$USER_ROOT/envs/openpi_py311/bin/python
RUN_ID=${RUN_ID:?RUN_ID is required}
OUTPUT_ROOT=$USER_ROOT/logs/r142_fp11/$RUN_ID
CACHE_DIR=$USER_ROOT/cache/r142_fp11/$RUN_ID
ARTIFACT_DIR=${ARTIFACT_DIR:?ARTIFACT_DIR is required}

[[ "$(id -u):$(id -g)" == "2254:2254" ]] || {
  echo "[r142] expected 2254:2254, got $(id -u):$(id -g)" >&2
  exit 41
}
for required in "$NEW_ROOT" "$USER_ROOT" "$PROJECT_DIR" "$PYTHON_BIN" "$ARTIFACT_DIR" "$OUTPUT_ROOT" "$CACHE_DIR"; do
  test -e "$required" || {
    echo "[r142] required path missing: $required" >&2
    exit 42
  }
done
for writable in "$ARTIFACT_DIR" "$OUTPUT_ROOT" "$CACHE_DIR"; do
  test -d "$writable" && test -w "$writable" || {
    echo "[r142] path is not writable: $writable" >&2
    exit 43
  }
  probe="$writable/.r142-owner-probe.$$"
  (umask 077; : >"$probe")
  [[ "$(stat -c '%u:%g' "$probe")" == "2254:2254" ]] || exit 44
  rm -f -- "$probe"
done

export HOME=$USER_ROOT
export USER=leon
export LOGNAME=leon
export XDG_CACHE_HOME=$CACHE_DIR/xdg
export MPLCONFIGDIR=$CACHE_DIR/matplotlib
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

cd "$PROJECT_DIR"
"$PYTHON_BIN" scripts/verify_runtime_manifest.py --manifest evidence/runtime_manifest.json
exec "$PYTHON_BIN" scripts/pai_entry.py \
  --output-root "$OUTPUT_ROOT" \
  --run-id "$RUN_ID" \
  --workers 8 \
  --episodes 400 \
  --python-bin "$PYTHON_BIN" \
  --config configs/stage1.json
