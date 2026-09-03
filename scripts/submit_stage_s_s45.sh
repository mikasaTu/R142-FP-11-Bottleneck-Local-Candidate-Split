#!/usr/bin/env bash
# Submit exactly one frozen Stage-S S4/S5 registry job.  This wrapper is never
# a substitute for the canonical registry; it adds the local contract gate
# immediately before pai-job validate/submit.
set -euo pipefail

: "${STAGE_S45_CONFIG:?set STAGE_S45_CONFIG to stage_s_s4.json or stage_s_s5.json}"
: "${STAGE_S45_RUN_ID:?set a fresh unique STAGE_S45_RUN_ID}"
REGISTRY_ROOT="${PAI_REGISTRY_ROOT:-/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/pai-job-registry}"
PROJECT_ROOT="${STAGE_S45_PROJECT_ROOT:-$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)}"
VALIDATOR="$PROJECT_ROOT/scripts/validate_stage_s45_pai_contract.py"

if [[ ! -x "$REGISTRY_ROOT/bin/pai-job" || ! -f "$VALIDATOR" ]]; then
  echo "STAGE_S45_SUBMIT_REJECTED: canonical registry or contract validator is unavailable" >&2
  exit 2
fi

"${STAGE_S45_PYTHON:-python3}" "$VALIDATOR" --config "$STAGE_S45_CONFIG"
"$REGISTRY_ROOT/bin/pai-job" validate "$STAGE_S45_CONFIG" --run-id "$STAGE_S45_RUN_ID"
exec "$REGISTRY_ROOT/bin/pai-job" submit "$STAGE_S45_CONFIG" --run-id "$STAGE_S45_RUN_ID" --config "${DLC_CONFIG:-/workspace/leon/.dlc/config}"
