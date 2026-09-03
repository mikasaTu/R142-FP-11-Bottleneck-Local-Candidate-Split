#!/usr/bin/env bash
set -euo pipefail

export TZ=Asia/Shanghai
readonly NOW_HM=$(date +%H%M)
readonly NOW_HM_DEC=$((10#$NOW_HM))
if (( NOW_HM_DEC >= 930 && NOW_HM_DEC < 940 )) || (( NOW_HM_DEC >= 1930 && NOW_HM_DEC < 1940 )); then
  echo "refusing Stage-S submission during mandatory blackout: $(date --iso-8601=seconds)" >&2
  exit 75
fi

readonly REGISTRY=/mnt/cpfs/zbl-cpfs-new/share/leon/codex-archives/Explore-claude-local-worktrees/r142-stage-s-pai-registry-clean-20260902
readonly REPO=/mnt/cpfs/zbl-cpfs-new/share/leon/codex-archives/Explore-claude-local-worktrees/r142-stage-s-20260902
readonly SUBMISSION_ROOT=/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_s/submissions
readonly B_RUN=r142-stage-s-b-calibration-20260903-r16
readonly C_RUN=r142-stage-s-c-undertrained-20260903-r11

umask 077
mkdir -p "$SUBMISSION_ROOT"

"$REGISTRY/bin/pai-job" submit "$REPO/configs/pai/stage_s_b_calibration.json" --run-id "$B_RUN" \
  >"$SUBMISSION_ROOT/${B_RUN}.txt" 2>&1
"$REGISTRY/bin/pai-job" submit "$REPO/configs/pai/stage_s_c_undertrained.json" --run-id "$C_RUN" \
  >"$SUBMISSION_ROOT/${C_RUN}.txt" 2>&1

printf 'B_SUBMISSION=%s\n' "$SUBMISSION_ROOT/${B_RUN}.txt"
printf 'C_SUBMISSION=%s\n' "$SUBMISSION_ROOT/${C_RUN}.txt"
