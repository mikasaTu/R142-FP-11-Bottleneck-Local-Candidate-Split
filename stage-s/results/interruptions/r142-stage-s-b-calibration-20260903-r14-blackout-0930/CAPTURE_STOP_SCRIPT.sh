#!/usr/bin/env bash
set -euo pipefail

readonly REGISTRY=/mnt/cpfs/zbl-cpfs-new/share/leon/codex-archives/Explore-claude-local-worktrees/r142-stage-s-pai-registry-clean-20260902
readonly DLC="$REGISTRY/bin/dlc-2598c3119-202512111654"
readonly DLC_CONFIG=/workspace/leon/.dlc/config
readonly JOB_ID=dlc2njrbes3vyx3z
readonly RUN_ID=r142-stage-s-b-calibration-20260903-r14
readonly SCI_ROOT=/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/pai_registry/r142_stage_s/b_calibration/$RUN_ID
readonly EVIDENCE=/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_s/interruptions/${RUN_ID}-blackout-0930

umask 077
mkdir -p "$EVIDENCE"

capture_job() {
  local output=$1
  local raw
  raw=$(mktemp)
  "$DLC" -c "$DLC_CONFIG" get job "$JOB_ID" >"$raw" 2>/dev/null
  sed -n '/^{/,$ p' "$raw" | jq '{JobId:.JobId,Status:.Status,Pods:(.Pods // .PodStatus // [] | map({PodId,Status,Type,GmtCreateTime,GmtStartTime})),ResourceConfig:(.JobSpecs // .ResourceConfig // null)}' >"$output"
  rm -f "$raw"
}

capture_job "$EVIDENCE/JOB_BEFORE_STOP.json"
"$DLC" -c "$DLC_CONFIG" logs "$JOB_ID" "${JOB_ID}-master-0" -n 400 >"$EVIDENCE/MASTER_LOG_TAIL_BEFORE_STOP.txt" 2>&1 || true

find "$SCI_ROOT" -maxdepth 5 -type f ! -path '*/xdg-cache/*' -printf '%P\t%s\t%TY-%Tm-%TdT%TH:%TM:%TS\n' | LC_ALL=C sort >"$EVIDENCE/SCIENTIFIC_FILE_SNAPSHOT.tsv"
find "$SCI_ROOT/shards" -type f -name COMPLETED_SHARD.json -print 2>/dev/null | LC_ALL=C sort >"$EVIDENCE/COMPLETED_SHARDS.txt" || true
find "$SCI_ROOT/shards" -type f \( -name RESULT.json -o -name COMPLETED_SHARD.json -o -name SHA256SUMS \) -print0 2>/dev/null | LC_ALL=C sort -z | xargs -0 -r sha256sum >"$EVIDENCE/PARTIAL_SHARD_SHA256SUMS" || true
find "$SCI_ROOT" -maxdepth 2 -type f ! -path '*/xdg-cache/*' -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum >"$EVIDENCE/ROOT_ARTIFACT_SHA256SUMS" || true

"$DLC" -c "$DLC_CONFIG" stop job "$JOB_ID" -f >"$EVIDENCE/STOP_COMMAND.txt" 2>&1
for _ in $(seq 1 40); do
  capture_job "$EVIDENCE/JOB_AFTER_STOP.json"
  status=$(jq -r '.Status' "$EVIDENCE/JOB_AFTER_STOP.json")
  case "$status" in
    Stopped|Failed|Succeeded) break ;;
  esac
  sleep 5
done

status=$(jq -r '.Status' "$EVIDENCE/JOB_AFTER_STOP.json")
case "$status" in
  Stopped|Failed|Succeeded) ;;
  *) echo "job did not reach a terminal state before blackout: $status" >&2; exit 1 ;;
esac

jq -n \
  --arg schema r142-stage-s-planned-blackout-interruption-v1 \
  --arg job_id "$JOB_ID" \
  --arg run_id "$RUN_ID" \
  --arg scientific_root "$SCI_ROOT" \
  --arg stopped_at "$(date --iso-8601=seconds)" \
  --arg status "$status" \
  --argjson completed_shards "$(wc -l <"$EVIDENCE/COMPLETED_SHARDS.txt")" \
  '{schema:$schema,reason:"mandatory user blackout 09:30-09:40 Asia/Shanghai",job_id:$job_id,run_id:$run_id,scientific_root:$scientific_root,stopped_at:$stopped_at,terminal_status:$status,completed_shards:$completed_shards,resume_policy:"new PAI controller, exact same scientific directory and frozen protocol after 09:40"}' \
  >"$EVIDENCE/INTERRUPTION.json"

(cd "$EVIDENCE" && find . -maxdepth 1 -type f ! -name SHA256SUMS -print0 | LC_ALL=C sort -z | xargs -0 sha256sum >SHA256SUMS)
(cd "$EVIDENCE" && sha256sum --check --quiet SHA256SUMS)
printf 'STOPPED job_id=%s status=%s evidence=%s\n' "$JOB_ID" "$status" "$EVIDENCE"
