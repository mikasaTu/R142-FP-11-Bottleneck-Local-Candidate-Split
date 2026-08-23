#!/usr/bin/env bash
set -Eeuo pipefail

umask 077

REPO=/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/R142-FP-11-Bottleneck-Local-Candidate-Split
LERO=/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/lerobot-r142-stage2a-3c0a209
ENV=/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/r142-stage2a-py310
PYTHON="$ENV/bin/python"
CHECKPOINT=/mnt/cpfs/zbl-cpfs-new/USERS/leon/cache/r142_fp11_stage2a/checkpoints/lerobot_diffusion_pusht_84a7c231
INPUT=/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage2a/r142-stage2a-formal-20260823-r2
INPUT_JOB_ID=dlcm4saves6zi30f
ARTIFACT_DIR=${PAI_CANARY_RUN_DIR:?PAI_CANARY_RUN_DIR is required}
RUN_ID=${PAI_CANARY_RUN_ID:?PAI_CANARY_RUN_ID is required}
EXPECTED_WEIGHT_SHA256=995d14d35db57d95c35ad9704c3d79c8612b7bc45f3877e5c46c2cdc516856a8
EXPECTED_LEROBOT_COMMIT=3c0a209f9fac4d2a57617e686a7f2a2309144ba2

on_error() {
  local exit_code=$?
  printf 'STAGE2A_CONTINUATION_FAILED line=%s exit_code=%s command=%q\n' \
    "${BASH_LINENO[0]:-unknown}" "$exit_code" "$BASH_COMMAND" >&2
  exit "$exit_code"
}
trap on_error ERR

test "$(id -u):$(id -g)" = 2254:2254
test "${PAI_CANARY_EXPECTED_GPUS:-}" = 2
test -x "$PYTHON"
test "$(git -C "$LERO" rev-parse HEAD)" = "$EXPECTED_LEROBOT_COMMIT"
test -z "$(git -C "$LERO" status --porcelain)"
test -z "$(git -C "$REPO" status --porcelain)"
test "$(sha256sum "$CHECKPOINT/model.safetensors" | cut -d' ' -f1)" = "$EXPECTED_WEIGHT_SHA256"
test -s "$INPUT/COMPLETED_EVALUATION_RESULT.json"
test -s "$INPUT/pilot/STAGE2A_PILOT_REPORT.md"
test -s "$INPUT/pilot/stage2a_summary.json"
test -s "$INPUT/SHA256SUMS"
case "$ARTIFACT_DIR" in
  /mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage2a/*) ;;
  *) echo "artifact directory escaped frozen output root" >&2; exit 71 ;;
esac
test "$(basename "$ARTIFACT_DIR")" = "$RUN_ID"
test "$(stat -c '%u:%g' "$ARTIFACT_DIR")" = 2254:2254
test ! -e "$ARTIFACT_DIR/COMPLETED_EVALUATION_RESULT.json"
cd "$ARTIFACT_DIR"

mkdir frozen_source frozen_lerobot continuation runtime
mkdir runtime/tmp runtime/cache
git -C "$REPO" archive HEAD | tar -xf - -C frozen_source
git -C "$LERO" archive "$EXPECTED_LEROBOT_COMMIT" -- lerobot pyproject.toml \
  | tar -xf - -C frozen_lerobot
git -C "$REPO" rev-parse HEAD > runtime/project_commit.txt
git -C "$REPO" rev-parse 'HEAD^{tree}' > runtime/project_tree.txt
git -C "$LERO" rev-parse HEAD > runtime/lerobot_commit.txt
printf '%s\n' "$INPUT_JOB_ID" > runtime/input_job_id.txt
printf '%s\n' "$INPUT" > runtime/input_artifact_dir.txt
"$PYTHON" - <<'PY' > runtime/pip_freeze.txt
from importlib.metadata import distributions

rows = []
for distribution in distributions():
    name = distribution.metadata.get("Name")
    if name:
        rows.append((name.casefold(), f"{name}=={distribution.version}"))
for _, row in sorted(set(rows)):
    print(row)
PY
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv,noheader > runtime/gpu_identity.csv

export PYTHONPATH="$ARTIFACT_DIR/frozen_lerobot:$ARTIFACT_DIR/frozen_source/src:$ARTIFACT_DIR/frozen_source/scripts"
export TMPDIR="$ARTIFACT_DIR/runtime/tmp"
export XDG_CACHE_HOME="$ARTIFACT_DIR/runtime/cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false

"$PYTHON" frozen_source/scripts/stage2a_eventual_continuation.py \
  --baseline-output "$INPUT/baseline" --pilot-output "$INPUT/pilot" \
  --output "$ARTIFACT_DIR/continuation" --manifest-only \
  2>&1 | tee runtime/manifest.log

CUDA_VISIBLE_DEVICES=0 "$PYTHON" frozen_source/scripts/stage2a_eventual_continuation.py \
  --checkpoint "$CHECKPOINT" --baseline-output "$INPUT/baseline" \
  --pilot-output "$INPUT/pilot" --output "$ARTIFACT_DIR/continuation" \
  --device cuda --rank 0 --world-size 2 > runtime/continuation.rank0.log 2>&1 &
rank0=$!
CUDA_VISIBLE_DEVICES=1 "$PYTHON" frozen_source/scripts/stage2a_eventual_continuation.py \
  --checkpoint "$CHECKPOINT" --baseline-output "$INPUT/baseline" \
  --pilot-output "$INPUT/pilot" --output "$ARTIFACT_DIR/continuation" \
  --device cuda --rank 1 --world-size 2 > runtime/continuation.rank1.log 2>&1 &
rank1=$!
wait "$rank0"
wait "$rank1"

"$PYTHON" frozen_source/scripts/stage2a_eventual_continuation.py \
  --baseline-output "$INPUT/baseline" --pilot-output "$INPUT/pilot" \
  --output "$ARTIFACT_DIR/continuation" --aggregate \
  2>&1 | tee runtime/aggregate.log

"$PYTHON" - "$ARTIFACT_DIR" "$INPUT" "$INPUT_JOB_ID" <<'PY'
import json, os, pathlib, sys

root = pathlib.Path(sys.argv[1])
source = pathlib.Path(sys.argv[2])
summary = json.loads((root / "continuation/eventual_continuation_summary.json").read_text())
first_work = {
    "schema_version": 1,
    "milestone": "representative_eventual_continuation_complete",
    "input_job_id": sys.argv[3],
    "case_count": summary["case_count"],
    "snapshot_count": summary["snapshot_count"],
    "uid": os.getuid(),
    "gid": os.getgid(),
}
value = {
    "schema_version": 1,
    "success_gate": "persisted_completed_evaluation_result",
    "engineering_complete": True,
    "scientific_role": "predeclared representative eventual-success continuation subset",
    "input_job_id": sys.argv[3],
    "input_run_id": source.name,
    "case_count": summary["case_count"],
    "snapshot_count": summary["snapshot_count"],
    "uid": os.getuid(),
    "gid": os.getgid(),
}
first_path = root / "FIRST_WORK.json"
first_tmp = first_path.with_suffix(".tmp")
first_tmp.write_text(json.dumps(first_work, indent=2, sort_keys=True) + "\n")
os.replace(first_tmp, first_path)
path = root / "COMPLETED_EVALUATION_RESULT.json"
tmp = path.with_suffix(".tmp")
tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
os.replace(tmp, path)
PY
(
  find . -type f ! -name SHA256SUMS -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
) > SHA256SUMS
sync -f "$ARTIFACT_DIR/COMPLETED_EVALUATION_RESULT.json"
sync -f "$ARTIFACT_DIR/FIRST_WORK.json"
sync -f "$ARTIFACT_DIR/SHA256SUMS"
echo STAGE2A_EVENTUAL_CONTINUATION_COMPLETE
