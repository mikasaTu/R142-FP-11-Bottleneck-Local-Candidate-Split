#!/usr/bin/env bash
# Common fail-closed foreground entrypoint for Stage-S S4/S5 PAI workers.
# PAI's outer interpreter may be /bin/sh; the registry wraps this file with
# an explicit Bash exec.  No model/simulator work is permitted before the
# bounded UID/GID transition.
set -euo pipefail

PHASE="${1:-}"
if [[ "$PHASE" != "s4" && "$PHASE" != "s5" ]]; then
  echo "STAGE_S45_ENTRY_REJECTED: phase must be s4 or s5" >&2
  exit 2
fi

: "${STAGE_S45_CONFIG:?STAGE_S45_CONFIG is required}"
: "${STAGE_S45_SCRIPT:?STAGE_S45_SCRIPT is required}"
: "${STAGE_S45_RUN_ID:?STAGE_S45_RUN_ID is required}"
: "${STAGE_S45_SUBSTRATE:?STAGE_S45_SUBSTRATE is required}"
: "${STAGE_S45_PROTOCOL:?STAGE_S45_PROTOCOL is required}"
: "${STAGE_S45_N32_ROOT:?STAGE_S45_N32_ROOT is required}"
: "${STAGE_S45_S4_ROOT:?STAGE_S45_S4_ROOT is required}"
: "${STAGE_S45_S5_ROOT:?STAGE_S45_S5_ROOT is required}"
: "${STAGE_S45_OUTPUT_ROOT:?STAGE_S45_OUTPUT_ROOT is required}"
: "${STAGE_S45_ADAPTER:?STAGE_S45_ADAPTER is required}"
: "${STAGE_S45_PYTHON:?STAGE_S45_PYTHON is required}"
: "${STAGE_S45_SOURCE_COMMIT:?STAGE_S45_SOURCE_COMMIT is required}"
: "${STAGE_S45_SOURCE_SHA256:?STAGE_S45_SOURCE_SHA256 is required}"
: "${STAGE_S45_PAYLOAD_SHA256:?STAGE_S45_PAYLOAD_SHA256 is required}"

for value in "$STAGE_S45_CONFIG" "$STAGE_S45_SCRIPT" "$STAGE_S45_RUN_ID" "$STAGE_S45_SUBSTRATE" "$STAGE_S45_PROTOCOL" \
  "$STAGE_S45_N32_ROOT" "$STAGE_S45_S4_ROOT" "$STAGE_S45_S5_ROOT" \
  "$STAGE_S45_OUTPUT_ROOT" "$STAGE_S45_ADAPTER" "$STAGE_S45_PYTHON" \
  "$STAGE_S45_SOURCE_COMMIT" "$STAGE_S45_SOURCE_SHA256" "$STAGE_S45_PAYLOAD_SHA256"; do
  if [[ "$value" == *"__REQUIRED_"* || "$value" == *"<"* || "$value" == *">"* || "$value" == *'${'* || "$value" == *"TODO"* || "$value" == *"TBD"* || "$value" == *"REPLACE_ME"* ]]; then
    echo "STAGE_S45_ENTRY_REJECTED: unresolved pin" >&2
    exit 2
  fi
done

for path in "$STAGE_S45_CONFIG" "$STAGE_S45_SCRIPT" "$STAGE_S45_PROTOCOL" \
  "$STAGE_S45_N32_ROOT" "$STAGE_S45_S4_ROOT" "$STAGE_S45_S5_ROOT" "$STAGE_S45_OUTPUT_ROOT"; do
  case "$path" in
    /mnt/cpfs/zbl-cpfs-new/*) ;;
    *) echo "STAGE_S45_ENTRY_REJECTED: all persistent paths must use new CPFS root" >&2; exit 2 ;;
  esac
done

if [[ "$(id -u)" -ne 2254 || "$(id -g)" -ne 2254 ]]; then
  if [[ "$(id -u)" -ne 0 || ! -x "$(command -v setpriv)" ]]; then
    echo "STAGE_S45_ENTRY_REJECTED: workload must run as UID:GID 2254:2254" >&2
    exit 2
  fi
  # Root is allowed only to prepare this exact run's output directories.  It
  # must never import Python, touch source/data roots, or start a subprocess
  # other than the bounded setpriv transition.
  mkdir -p "$STAGE_S45_OUTPUT_ROOT" "$STAGE_S45_S4_ROOT" "$STAGE_S45_S5_ROOT"
  chown 2254:2254 "$STAGE_S45_OUTPUT_ROOT" "$STAGE_S45_S4_ROOT" "$STAGE_S45_S5_ROOT"
  exec setpriv --reuid=2254 --regid=2254 --clear-groups "$0" "$PHASE"
fi

if [[ "$(id -u):$(id -g)" != "2254:2254" ]]; then
  echo "STAGE_S45_ENTRY_REJECTED: setpriv identity transition failed" >&2
  exit 2
fi

if [[ ! -x "$STAGE_S45_PYTHON" || ! -f "$STAGE_S45_PROTOCOL" || ! -d "$STAGE_S45_N32_ROOT" ]]; then
  echo "STAGE_S45_ENTRY_REJECTED: pinned Python/protocol/N32 root is unavailable" >&2
  exit 2
fi

mkdir -p "$STAGE_S45_OUTPUT_ROOT/$STAGE_S45_RUN_ID"
if [[ ! -w "$STAGE_S45_OUTPUT_ROOT/$STAGE_S45_RUN_ID" ]]; then
  echo "STAGE_S45_ENTRY_REJECTED: exact run output is not writable" >&2
  exit 2
fi

cmd=("$STAGE_S45_PYTHON" "$STAGE_S45_SCRIPT" \
  --phase "$PHASE" \
  --substrate "$STAGE_S45_SUBSTRATE" \
  --protocol "$STAGE_S45_PROTOCOL" \
  --n32-root "$STAGE_S45_N32_ROOT" \
  --output-root "$STAGE_S45_OUTPUT_ROOT/$STAGE_S45_RUN_ID" \
  --adapter "$STAGE_S45_ADAPTER" \
  --main-source-commit "$STAGE_S45_SOURCE_COMMIT" \
  --main-source-sha256 "$STAGE_S45_SOURCE_SHA256")
if [[ -n "${STAGE_S45_CALIBRATION_REPORT:-}" ]]; then
  cmd+=(--calibration-report "$STAGE_S45_CALIBRATION_REPORT")
fi
exec "${cmd[@]}"
