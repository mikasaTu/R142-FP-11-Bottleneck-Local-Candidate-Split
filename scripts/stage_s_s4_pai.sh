#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
export STAGE_S45_SCRIPT="${STAGE_S45_SCRIPT:-$SCRIPT_DIR/stage_s_s45.py}"
exec "$SCRIPT_DIR/stage_s_s45_entry.sh" s4
