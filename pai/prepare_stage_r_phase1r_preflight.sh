#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

readonly REPO=/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/R142-FP-11-Bottleneck-Local-Candidate-Split
readonly PYTHON=/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/openpi_py311/bin/python
readonly OUTPUT_ROOT=/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_r/phase1r
readonly SOURCE_COMMIT=57859fcbb36776e0049ce24fb1abbadab0de46d5
readonly CHECKPOINT=/mnt/cpfs/zbl-cpfs-new/USERS/leon/cache/openpi/r16p15/openpi-assets/checkpoints/pi05_libero
readonly CHECKPOINT_TREE_SHA=42d571bd87f05f1182810f5a8bfa6d084c0d0dd277aff739bcf8f69868e6fb99
readonly CHECKPOINT_ATTESTATION="$REPO/configs/stage_r_pi05_libero_checkpoint_attestation.json"
readonly CHECKPOINT_ATTESTATION_SHA=d050805b0c1e9e8d8e879c7443bb10504859c654d0ba031bbbc6ce3635b02fca
readonly PHASE0_ROOT=/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_r/phase0r_merged/r142-stage-r-phase0r-authoritative-20260827
readonly AUTHORITY_SHA=3d5a37ec8a7e2c0dfd0c808ad59553c43a13c846b90f99c1afaa3529a072469c
readonly EXECUTION="$REPO/configs/stage_r_phase1r_execution_idle4.json"

ARTIFACT_DIR="${1:?artifact directory is required}"
RUN_ID="${2:?run id is required}"
[[ "$ARTIFACT_DIR" == "$OUTPUT_ROOT/$RUN_ID" ]] || { echo "artifact/run identity mismatch" >&2; exit 64; }
case "$RUN_ID" in
  r142-stage-r-phase1r-shard-a0-*) EXECUTION_SHARD=A0 ;;
  r142-stage-r-phase1r-shard-a1-*) EXECUTION_SHARD=A1 ;;
  r142-stage-r-phase1r-shard-b0-*) EXECUTION_SHARD=B0 ;;
  r142-stage-r-phase1r-shard-b1-*) EXECUTION_SHARD=B1 ;;
  *) echo "unsupported run id" >&2; exit 64 ;;
esac
[[ "$(id -u):$(id -g)" == 2254:2254 ]] || { echo "owner mismatch" >&2; exit 77; }
for forbidden in FIRST_WORK.json COMPLETED_EVALUATION_RESULT.json SHA256SUMS; do
  [[ ! -e "$ARTIFACT_DIR/$forbidden" ]] || { echo "outcome/completion evidence already exists" >&2; exit 79; }
done
if [[ -d "$ARTIFACT_DIR/natural" ]] && find "$ARTIFACT_DIR/natural" -type f -print -quit | grep -q .; then
  echo "natural outcomes already exist" >&2
  exit 79
fi
mkdir -p "$ARTIFACT_DIR/frozen_source" "$ARTIFACT_DIR/runtime" "$ARTIFACT_DIR/natural"
if find "$ARTIFACT_DIR/frozen_source" -mindepth 1 -print -quit | grep -q .; then
  echo "frozen_source must be empty before CPU preseed" >&2
  exit 79
fi

git -C "$REPO" archive "$SOURCE_COMMIT" | tar -xf - -C "$ARTIFACT_DIR/frozen_source"
printf '%s\n' "$SOURCE_COMMIT" > "$ARTIFACT_DIR/runtime/source_commit.txt"
git -C "$REPO" rev-parse "$SOURCE_COMMIT^{tree}" > "$ARTIFACT_DIR/runtime/source_tree.txt"
(
  cd "$ARTIFACT_DIR/frozen_source"
  find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
) > "$ARTIFACT_DIR/runtime/frozen_source.sha256"
find "$ARTIFACT_DIR/frozen_source" -type f -exec chmod 0444 {} +
find "$ARTIFACT_DIR/frozen_source" -type d -exec chmod 0555 {} +
"$PYTHON" "$REPO/pai/validate_stage_r_frozen_source_resume.py" seal \
  "$ARTIFACT_DIR/frozen_source" "$ARTIFACT_DIR/runtime/frozen_source.sha256" \
  "$ARTIFACT_DIR/runtime/frozen_source_verified.json" "$SOURCE_COMMIT" \
  "$(cat "$ARTIFACT_DIR/runtime/source_tree.txt")" >/dev/null

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ARTIFACT_DIR/frozen_source/src"
PROTOCOL="$ARTIFACT_DIR/frozen_source/configs/stage_r_phase1r_protocol.json"
SHARDS="$ARTIFACT_DIR/frozen_source/configs/stage_r_phase1r_shards.json"
SELECTION="$ARTIFACT_DIR/frozen_source/results/stage_r/phase1r/selection"
CONTROLS="$ARTIFACT_DIR/frozen_source/results/stage_r/phase1r/controls"
CALIBRATION="$ARTIFACT_DIR/frozen_source/results/stage_r/phase1r/calibration"
"$PYTHON" "$ARTIFACT_DIR/frozen_source/scripts/stage_r_phase1r.py" validate-config \
  --protocol "$PROTOCOL" --shards "$SHARDS" > "$ARTIFACT_DIR/runtime/protocol_validation.json"
"$PYTHON" "$ARTIFACT_DIR/frozen_source/scripts/stage_r_phase1r.py" validate-selection \
  --root "$SELECTION" > "$ARTIFACT_DIR/runtime/selection_validation.json"
"$PYTHON" "$ARTIFACT_DIR/frozen_source/scripts/stage_r_phase1r.py" validate-controls \
  --root "$CONTROLS" --kind positive --owner 2254:2254 > "$ARTIFACT_DIR/runtime/positive_control_validation.json"
"$PYTHON" "$ARTIFACT_DIR/frozen_source/scripts/stage_r_phase1r.py" validate-controls \
  --root "$CONTROLS" --kind null --owner 2254:2254 > "$ARTIFACT_DIR/runtime/null_control_validation.json"
"$PYTHON" "$REPO/pai/validate_stage_r_phase1r_prerequisites.py" calibration \
  "$CALIBRATION/BLINDED_PHASE1R_CALIBRATION.json" "$CALIBRATION/SHA256SUMS" >/dev/null
"$PYTHON" "$REPO/pai/validate_stage_r_phase1r_prerequisites.py" phase0 \
  "$PHASE0_ROOT" "$PHASE0_ROOT/raw" "$AUTHORITY_SHA" > "$ARTIFACT_DIR/runtime/phase0_authority_validation.json"
"$PYTHON" "$REPO/pai/validate_stage_r_checkpoint_attestation.py" \
  "$CHECKPOINT" "$CHECKPOINT_TREE_SHA" "$CHECKPOINT_ATTESTATION" "$CHECKPOINT_ATTESTATION_SHA" \
  > "$ARTIFACT_DIR/runtime/checkpoint_validation.json"
"$PYTHON" "$REPO/pai/stage_r_phase1r_task_mapping.py" write "$SHARDS" "$EXECUTION" \
  "$EXECUTION_SHARD" "$SELECTION" "$ARTIFACT_DIR/runtime/task_mapping.tsv" >/dev/null
"$PYTHON" "$REPO/pai/stage_r_phase1r_task_mapping.py" validate "$SHARDS" "$EXECUTION" \
  "$EXECUTION_SHARD" "$SELECTION" "$ARTIFACT_DIR/runtime/task_mapping.tsv" >/dev/null
"$PYTHON" - "$RUN_ID" "$EXECUTION_SHARD" <<'PY' > "$ARTIFACT_DIR/runtime/CPU_PRESEEDED_PREFLIGHT.json"
import json, sys
print(json.dumps({
    "schema_version": 1,
    "marker_type": "outcome_blind_cpu_preseeded_preflight",
    "run_id": sys.argv[1],
    "execution_shard": sys.argv[2],
    "outcomes_read": False,
    "natural_cells": 0,
    "scientific_contract_changed": False,
}, indent=2, sort_keys=True))
PY
"$PYTHON" "$REPO/pai/validate_stage_r_phase1r_preflight_bundle.py" \
  "$ARTIFACT_DIR/runtime" "$CHECKPOINT_TREE_SHA" "$CHECKPOINT_ATTESTATION_SHA" 2254:2254 \
  "$RUN_ID" "$EXECUTION_SHARD" >/dev/null
echo "CPU_PRESEEDED_PREFLIGHT_OK run_id=$RUN_ID execution_shard=$EXECUTION_SHARD"
