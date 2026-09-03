#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
: "${STAGE_S45_CONFIG:?STAGE_S45_CONFIG is required}"
: "${STAGE_S45_RUN_ID:?STAGE_S45_RUN_ID is required}"
: "${STAGE_S45_SUBSTRATE:?STAGE_S45_SUBSTRATE is required}"
: "${STAGE_S45_PYTHON:?STAGE_S45_PYTHON is required}"
: "${STAGE_S45_PROTOCOL:?STAGE_S45_PROTOCOL is required}"
: "${STAGE_S45_N32_ROOT:?STAGE_S45_N32_ROOT is required}"
: "${STAGE_S45_S4_ROOT:?STAGE_S45_S4_ROOT is required}"
: "${STAGE_S45_S5_ROOT:?STAGE_S45_S5_ROOT is required}"
: "${STAGE_S45_OUTPUT_ROOT:?STAGE_S45_OUTPUT_ROOT is required}"
: "${STAGE_S45_SOURCE_COMMIT:?STAGE_S45_SOURCE_COMMIT is required}"
: "${STAGE_S45_SOURCE_SHA256:?STAGE_S45_SOURCE_SHA256 is required}"

if [[ "$(id -u):$(id -g)" != "2254:2254" ]]; then
  if [[ "$(id -u)" -ne 0 || ! -x "$(command -v setpriv)" ]]; then
    echo "STAGE_S45_FINALIZE_REJECTED: workload must run as UID:GID 2254:2254" >&2
    exit 2
  fi
  mkdir -p "$STAGE_S45_OUTPUT_ROOT" && chown 2254:2254 "$STAGE_S45_OUTPUT_ROOT"
  exec setpriv --reuid=2254 --regid=2254 --clear-groups "$0"
fi

if [[ ! -x "$STAGE_S45_PYTHON" ]]; then
  echo "STAGE_S45_FINALIZE_REJECTED: pinned Python is unavailable" >&2
  exit 2
fi
cmd=("$STAGE_S45_PYTHON" "$SCRIPT_DIR/stage_s_s45_finalize.py" \
  --substrate "$STAGE_S45_SUBSTRATE" \
  --protocol "$STAGE_S45_PROTOCOL" \
  --n32-root "$STAGE_S45_N32_ROOT" \
  --s4-root "$STAGE_S45_S4_ROOT" \
  --s5-root "$STAGE_S45_S5_ROOT" \
  --output-root "$STAGE_S45_OUTPUT_ROOT/$STAGE_S45_RUN_ID" \
  --main-source-commit "$STAGE_S45_SOURCE_COMMIT" \
  --main-source-sha256 "$STAGE_S45_SOURCE_SHA256")
if [[ -n "${STAGE_S45_CALIBRATION_REPORT:-}" ]]; then
  cmd+=(--calibration-report "$STAGE_S45_CALIBRATION_REPORT")
fi
exec "${cmd[@]}"
