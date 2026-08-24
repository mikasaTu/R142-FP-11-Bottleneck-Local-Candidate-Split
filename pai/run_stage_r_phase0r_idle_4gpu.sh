#!/usr/bin/env bash
set -Eeuo pipefail

umask 077

REPO=/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/R142-FP-11-Bottleneck-Local-Candidate-Split
SOURCE_COMMIT=24423e8114ace80e6a76f22bee29992cea420cfc
QPILOTS=/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/QPILOTS-r16p15-stage1-task64-20260812
QPILOTS_COMMIT=eacf47b981e3b22357f8a74902f8dad8cfcfa375
OPENPI="$QPILOTS/third_party/openpi"
OPENPI_COMMIT=54cbaee6ae0c010a1ed431871cdaa8f4684ac709
LIBERO="$OPENPI/third_party/libero"
LIBERO_COMMIT=f78abd68ee283de9f9be3c8f7e2a9ad60246e95c
PYTHON=/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/openpi_py311/bin/python
CHECKPOINT=/mnt/cpfs/zbl-cpfs-new/USERS/leon/cache/openpi/r16p15/openpi-assets/checkpoints/pi05_libero
GATE_DIR=/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_r/gates/r142-stage-r-gates-20260824-r4-idle4
ARTIFACT_DIR=${PAI_CANARY_RUN_DIR:?PAI_CANARY_RUN_DIR is required}
RUN_ID=${PAI_CANARY_RUN_ID:?PAI_CANARY_RUN_ID is required}

trap 'code=$?; printf "STAGE_R_PHASE0R_COMMAND_FAILED line=%s exit_code=%s command=%q\n" "${BASH_LINENO[0]:-unknown}" "$code" "$BASH_COMMAND" >&2; exit "$code"' ERR

test "$(id -u):$(id -g)" = 2254:2254
test "${PAI_CANARY_EXPECTED_GPUS:-}" = 4
test -x "$PYTHON"
test "$(git -C "$REPO" rev-parse "$SOURCE_COMMIT^{commit}")" = "$SOURCE_COMMIT"
test "$(git -C "$QPILOTS" rev-parse HEAD)" = "$QPILOTS_COMMIT"
test "$(git -C "$OPENPI" rev-parse HEAD)" = "$OPENPI_COMMIT"
test "$(git -C "$LIBERO" rev-parse HEAD)" = "$LIBERO_COMMIT"
git -C "$REPO" diff --quiet
git -C "$REPO" diff --cached --quiet
test -z "$(git -C "$QPILOTS" status --porcelain)"
test -f "$GATE_DIR/COMPLETED_EVALUATION_RESULT.json"
(cd "$GATE_DIR" && sha256sum --check --quiet SHA256SUMS)
test "$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["decision"])' "$GATE_DIR/COMPLETED_EVALUATION_RESULT.json")" = ENGINEERING_GATES_PASSED
case "$ARTIFACT_DIR" in
  /mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_r/phase0r/*) ;;
  *) echo "artifact directory escaped frozen Phase-0R root" >&2; exit 71 ;;
esac
test "$(basename "$ARTIFACT_DIR")" = "$RUN_ID"
test "$(stat -c '%u:%g' "$ARTIFACT_DIR")" = 2254:2254
cd "$ARTIFACT_DIR"
if [ -f COMPLETED_EVALUATION_RESULT.json ] && [ -f SHA256SUMS ]; then
  sha256sum --check --quiet SHA256SUMS
  echo "STAGE_R_PHASE0R_ALREADY_COMPLETE_VALIDATED"
  exit 0
fi
mkdir -p frozen_source runtime raw analysis
mkdir -p runtime/tmp runtime/cache runtime/logs
if [ ! -e runtime/source_commit.txt ]; then
  git -C "$REPO" archive "$SOURCE_COMMIT" | tar -xf - -C frozen_source
  printf '%s\n' "$SOURCE_COMMIT" > runtime/source_commit.txt
  git -C "$REPO" rev-parse "$SOURCE_COMMIT^{tree}" > runtime/source_tree.txt
  (
    cd frozen_source
    find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
  ) > runtime/frozen_source.sha256
else
  test "$(cat runtime/source_commit.txt)" = "$SOURCE_COMMIT"
fi
nvidia-smi -L > runtime/gpu_inventory.txt
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv,noheader > runtime/gpu_identity.csv

export PYTHONPATH="$ARTIFACT_DIR/frozen_source/src:$QPILOTS:$OPENPI/src"
export LIBERO_CONFIG_PATH=/mnt/cpfs/zbl-cpfs-new/USERS/leon/cache/libero/r16p15-stage1-task64
export LD_LIBRARY_PATH=/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/openpi_py311/lib:${LD_LIBRARY_PATH:-}
export TMPDIR="$ARTIFACT_DIR/runtime/tmp"
export XDG_CACHE_HOME="$ARTIFACT_DIR/runtime/cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70

pids=()
for rank in 0 1 2 3; do
  visible_devices="$rank"
  if [ "$rank" != 0 ]; then
    visible_devices="$rank,0"
  fi
  CUDA_VISIBLE_DEVICES="$visible_devices" EGL_DEVICE_ID=0 MUJOCO_EGL_DEVICE_ID=0 \
    "$PYTHON" frozen_source/scripts/stage_r_phase0r.py \
    --qpilots-root "$QPILOTS" \
    --libero-root "$LIBERO" \
    --checkpoint "$CHECKPOINT" \
    --output "$ARTIFACT_DIR/raw" \
    --microbatch 4 \
    --rank "$rank" --world-size 4 \
    > "runtime/logs/phase0r.rank${rank}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
test "$failed" = 0

"$PYTHON" frozen_source/scripts/stage_r_phase0r.py \
  --output "$ARTIFACT_DIR/raw" --world-size 4 --aggregate-only \
  2>&1 | tee runtime/raw_aggregate.log

"$PYTHON" frozen_source/scripts/stage_r_analyze.py \
  --raw "$ARTIFACT_DIR/raw" \
  --thresholds "$ARTIFACT_DIR/frozen_source/docs/stage_r/PHASE0R_THRESHOLDS.json" \
  --output "$ARTIFACT_DIR/analysis" \
  2>&1 | tee runtime/analysis.log

"$PYTHON" - "$ARTIFACT_DIR" <<'PY'
import json, os, pathlib, sys
root = pathlib.Path(sys.argv[1])
raw = json.loads((root / "raw/COMPLETED_PHASE0R_RAW.json").read_text())
summary = json.loads((root / "analysis/phase0r_summary.json").read_text())
analysis_complete = json.loads((root / "analysis/COMPLETED_PHASE0R.json").read_text())
first = {
    "schema_version": 1,
    "milestone": "all_raw_shards_complete_before_unblinding",
    "raw_task_count": raw["task_count"],
    "raw_rollout_count": raw["rollout_count"],
    "uid": os.getuid(),
    "gid": os.getgid(),
}
completed = {
    "schema_version": 1,
    "success_gate": "persisted_complete_phase0r_evaluation",
    "decision": summary["decision"],
    "retained_tasks": summary["retained_tasks"],
    "checkpoint": "CHECKPOINT_1_STOP",
    "phase1_authorized": False,
    "analysis_completion": analysis_complete,
    "uid": os.getuid(),
    "gid": os.getgid(),
}
for name, value in (("FIRST_WORK.json", first), ("COMPLETED_EVALUATION_RESULT.json", completed)):
    path = root / name
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
PY
(
  find . -type f ! -name SHA256SUMS -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
) > SHA256SUMS
sync -f "$ARTIFACT_DIR/FIRST_WORK.json"
sync -f "$ARTIFACT_DIR/COMPLETED_EVALUATION_RESULT.json"
sync -f "$ARTIFACT_DIR/SHA256SUMS"
echo "STAGE_R_PHASE0R_EVALUATION_COMPLETE_CHECKPOINT1_STOP"
