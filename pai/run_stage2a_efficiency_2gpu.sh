#!/usr/bin/env bash
set -Eeuo pipefail

umask 077

REPO=/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/R142-FP-11-Bottleneck-Local-Candidate-Split
LERO=/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/lerobot-r142-stage2a-3c0a209
ENV=/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/r142-stage2a-py310
PYTHON="$ENV/bin/python"
CHECKPOINT=/mnt/cpfs/zbl-cpfs-new/USERS/leon/cache/r142_fp11_stage2a/checkpoints/lerobot_diffusion_pusht_84a7c231
ARTIFACT_DIR=${PAI_CANARY_RUN_DIR:?PAI_CANARY_RUN_DIR is required}
RUN_ID=${PAI_CANARY_RUN_ID:?PAI_CANARY_RUN_ID is required}
EXPECTED_WEIGHT_SHA256=995d14d35db57d95c35ad9704c3d79c8612b7bc45f3877e5c46c2cdc516856a8
EXPECTED_LEROBOT_COMMIT=3c0a209f9fac4d2a57617e686a7f2a2309144ba2

on_error() {
  local exit_code=$?
  printf 'STAGE2A_COMMAND_FAILED line=%s exit_code=%s command=%q\n' \
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
case "$ARTIFACT_DIR" in
  /mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage2a/*) ;;
  *) echo "artifact directory escaped frozen output root" >&2; exit 71 ;;
esac
test "$(basename "$ARTIFACT_DIR")" = "$RUN_ID"
test "$(stat -c '%u:%g' "$ARTIFACT_DIR")" = 2254:2254
test ! -e "$ARTIFACT_DIR/COMPLETED_EVALUATION_RESULT.json"
cd "$ARTIFACT_DIR"

mkdir frozen_source frozen_lerobot baseline pilot runtime
mkdir runtime/tmp runtime/cache
git -C "$REPO" archive HEAD | tar -xf - -C frozen_source
git -C "$LERO" archive "$EXPECTED_LEROBOT_COMMIT" | tar -xf - -C frozen_lerobot
git -C "$REPO" rev-parse HEAD > runtime/project_commit.txt
git -C "$REPO" rev-parse 'HEAD^{tree}' > runtime/project_tree.txt
git -C "$LERO" rev-parse HEAD > runtime/lerobot_commit.txt
"$PYTHON" -m pip freeze --all > runtime/pip_freeze.txt
nvidia-smi -L > runtime/gpu_inventory.txt
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv,noheader > runtime/gpu_identity.csv
(
  cd frozen_source
  find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
) > runtime/frozen_source.sha256
(
  cd frozen_lerobot
  find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
) > runtime/frozen_lerobot.sha256

export PYTHONPATH="$ARTIFACT_DIR/frozen_lerobot:$ARTIFACT_DIR/frozen_source/src:$ARTIFACT_DIR/frozen_source/scripts"
export TMPDIR="$ARTIFACT_DIR/runtime/tmp"
export XDG_CACHE_HOME="$ARTIFACT_DIR/runtime/cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES=0 "$PYTHON" frozen_source/scripts/stage2a_validate.py \
  --checkpoint "$CHECKPOINT" \
  --output "$ARTIFACT_DIR/baseline" \
  --device cuda \
  --mode all \
  --seed-start 0 \
  --episodes 50 \
  2>&1 | tee runtime/baseline.log

"$PYTHON" frozen_source/scripts/stage2a_select_snapshots.py \
  --frame "$ARTIFACT_DIR/baseline/snapshot_sampling_frame.jsonl" \
  --target-per-stratum 8 \
  2>&1 | tee runtime/snapshot_selection.log

"$PYTHON" - "$ARTIFACT_DIR" <<'PY'
import json, pathlib, sys, os
root=pathlib.Path(sys.argv[1])
selection=json.loads((root/'baseline/snapshot_selection_manifest.json').read_text())
count=selection['selected_count']
if count < 15 or count > 30:
    raise SystemExit(f'natural snapshot count outside pilot range: {count}')
value={
  'schema_version':1,
  'milestone':'baseline_and_snapshot_frame_complete',
  'baseline_summary':json.loads((root/'baseline/baseline_summary.json').read_text()),
  'selection_manifest':selection,
  'uid':os.getuid(),
  'gid':os.getgid(),
}
path=root/'FIRST_WORK.json'
tmp=path.with_suffix('.tmp')
tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
os.replace(tmp,path)
PY
sync -f "$ARTIFACT_DIR/FIRST_WORK.json"

CUDA_VISIBLE_DEVICES=0 "$PYTHON" frozen_source/scripts/stage2a_pilot.py \
  --checkpoint "$CHECKPOINT" --baseline-output "$ARTIFACT_DIR/baseline" \
  --output "$ARTIFACT_DIR/pilot" --device cuda --rank 0 --world-size 2 \
  > runtime/pilot.rank0.log 2>&1 &
pilot0=$!
CUDA_VISIBLE_DEVICES=1 "$PYTHON" frozen_source/scripts/stage2a_pilot.py \
  --checkpoint "$CHECKPOINT" --baseline-output "$ARTIFACT_DIR/baseline" \
  --output "$ARTIFACT_DIR/pilot" --device cuda --rank 1 --world-size 2 \
  > runtime/pilot.rank1.log 2>&1 &
pilot1=$!
wait "$pilot0"
wait "$pilot1"

CUDA_VISIBLE_DEVICES=0 "$PYTHON" frozen_source/scripts/stage2a_fixed_nfe.py \
  --checkpoint "$CHECKPOINT" --baseline-output "$ARTIFACT_DIR/baseline" \
  --pilot-output "$ARTIFACT_DIR/pilot" --device cuda --rank 0 --world-size 2 \
  > runtime/fixed_nfe.rank0.log 2>&1 &
fixed0=$!
CUDA_VISIBLE_DEVICES=1 "$PYTHON" frozen_source/scripts/stage2a_fixed_nfe.py \
  --checkpoint "$CHECKPOINT" --baseline-output "$ARTIFACT_DIR/baseline" \
  --pilot-output "$ARTIFACT_DIR/pilot" --device cuda --rank 1 --world-size 2 \
  > runtime/fixed_nfe.rank1.log 2>&1 &
fixed1=$!
wait "$fixed0"
wait "$fixed1"

"$PYTHON" frozen_source/scripts/stage2a_analyze.py \
  --baseline-output "$ARTIFACT_DIR/baseline" \
  --pilot-output "$ARTIFACT_DIR/pilot" \
  2>&1 | tee runtime/analysis.log

"$PYTHON" - "$ARTIFACT_DIR" <<'PY'
import json, pathlib, sys, os
root=pathlib.Path(sys.argv[1])
summary=json.loads((root/'pilot/stage2a_summary.json').read_text())
required=[
 root/'baseline/resume_equivalence_tests.json',
 root/'baseline/simulator_snapshot_tests.json',
 root/'baseline/baseline_reproduction.jsonl',
 root/'baseline/snapshot_sampling_frame.jsonl',
 root/'pilot/descendant_genealogy.rank0.jsonl',
 root/'pilot/descendant_genealogy.rank1.jsonl',
 root/'pilot/cost_aware_utility.json',
 root/'pilot/negative_controls.json',
 root/'pilot/STAGE2A_PILOT_REPORT.md',
]
missing=[str(p) for p in required if not p.is_file() or p.stat().st_size == 0]
if missing:
    raise SystemExit(f'missing required artifacts: {missing}')
value={
 'schema_version':1,
 'success_gate':'persisted_completed_evaluation_result',
 'engineering_complete':True,
 'scientific_decision':summary['scientific_decision'],
 'snapshot_count':summary['snapshot_count'],
 'all_source_gates_pass':summary['all_source_gates_pass'],
 'uid':os.getuid(),
 'gid':os.getgid(),
}
path=root/'COMPLETED_EVALUATION_RESULT.json'
tmp=path.with_suffix('.tmp')
tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
os.replace(tmp,path)
PY
(
  find . -type f ! -name SHA256SUMS ! -path './frozen_source/.git/*' -print0 \
    | LC_ALL=C sort -z | xargs -0 sha256sum
) > SHA256SUMS
sync -f "$ARTIFACT_DIR/COMPLETED_EVALUATION_RESULT.json"
sync -f "$ARTIFACT_DIR/SHA256SUMS"
echo "STAGE2A_FORMAL_EVALUATION_COMPLETE"
