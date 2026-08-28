#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

readonly SOURCE_COMMIT=1e3ac9b7c9bdf7c5a628911aee320ab0c631ac07
readonly REPO=/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/R142-FP-11-Bottleneck-Local-Candidate-Split
readonly QPILOTS=/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/QPILOTS-r16p15-stage1-task64-20260812
readonly QPILOTS_COMMIT=eacf47b981e3b22357f8a74902f8dad8cfcfa375
readonly OPENPI="$QPILOTS/third_party/openpi"
readonly OPENPI_COMMIT=54cbaee6ae0c010a1ed431871cdaa8f4684ac709
readonly LIBERO="$OPENPI/third_party/libero"
readonly LIBERO_CONFIG=/mnt/cpfs/zbl-cpfs-new/USERS/leon/cache/libero/r16p15-stage1-task64
readonly LIBERO_COMMIT=f78abd68ee283de9f9be3c8f7e2a9ad60246e95c
readonly PYTHON=/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/openpi_py311/bin/python
readonly RAW=/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_r/phase0r_merged/r142-stage-r-phase0r-authoritative-20260827/raw/libero_10_task09.npz
readonly RAW_SHA256=725d8bca12eeeb7e54fe0df75c536a6b8b6081c08a180c9228a43596983dd8e8
readonly OUTPUT=${PAI_CANARY_RUN_DIR:?PAI_CANARY_RUN_DIR is required}

[[ "${PAI_CANARY_EXPECTED_GPUS:-}" == 4 ]]
[[ "$(id -u):$(id -g)" == 2254:2254 ]]
[[ "$(git -C "$REPO" rev-parse "$SOURCE_COMMIT^{commit}")" == "$SOURCE_COMMIT" ]]
[[ "$(git -C "$QPILOTS" rev-parse HEAD)" == "$QPILOTS_COMMIT" ]]
[[ "$(git -C "$OPENPI" rev-parse HEAD)" == "$OPENPI_COMMIT" ]]
[[ "$(git -C "$LIBERO" rev-parse HEAD)" == "$LIBERO_COMMIT" ]]
[[ "$(sha256sum "$RAW" | awk '{print $1}')" == "$RAW_SHA256" ]]

mkdir -p "$OUTPUT/runtime" "$OUTPUT/results"
readonly SOURCE_DIR="$OUTPUT/runtime/source"
if [[ ! -f "$OUTPUT/runtime/source_commit.txt" ]]; then
  [[ -z "$(find "$SOURCE_DIR" -mindepth 1 -print -quit 2>/dev/null)" ]]
  mkdir -p "$SOURCE_DIR"
  git -C "$REPO" archive "$SOURCE_COMMIT" | tar -xf - -C "$SOURCE_DIR"
  printf '%s\n' "$SOURCE_COMMIT" > "$OUTPUT/runtime/source_commit.txt"
fi
[[ "$(cat "$OUTPUT/runtime/source_commit.txt")" == "$SOURCE_COMMIT" ]]
find "$SOURCE_DIR" -type f -name '*.pyc' -o -type d -name __pycache__ | grep -q . && exit 79 || true

run_strategy() {
  local strategy=$1
  local result="$OUTPUT/results/$strategy.json"
  local log="$OUTPUT/results/$strategy.stdout.log"
  if [[ -f "$result" ]]; then
    "$PYTHON" - "$result" "$strategy" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert payload["strategy"] == sys.argv[2]
assert payload["candidate_id"] == 27 and payload["init_state"] == 27
assert payload["raw_length"] == 520
PY
    return
  fi
  LIBERO_CONFIG_PATH="$LIBERO_CONFIG" CUDA_VISIBLE_DEVICES=0 EGL_DEVICE_ID=0 MUJOCO_EGL_DEVICE_ID=0 PYTHONDONTWRITEBYTECODE=1 \
    "$PYTHON" -u "$SOURCE_DIR/scripts/diagnose_stage_r_phase1r_env_reconstruction.py" \
      --repo "$SOURCE_DIR" --qpilots "$QPILOTS" --libero "$LIBERO" --raw "$RAW" \
      --strategy "$strategy" --output "$result" > "$log" 2>&1
}

run_strategy prior-reset-only
"$PYTHON" - "$OUTPUT" "$SOURCE_COMMIT" <<'PY'
import json, os, sys
from pathlib import Path
root, source_commit = Path(sys.argv[1]), sys.argv[2]
path = root / "FIRST_WORK.json"
if not path.exists():
    temporary = root / ".FIRST_WORK.json.tmp"
    temporary.write_text(json.dumps({
        "schema_version": 1,
        "milestone": "first_complete_environment_reconstruction_strategy",
        "source_commit": source_commit,
        "strategy": "prior-reset-only",
        "uid": os.getuid(),
        "gid": os.getgid(),
    }, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
PY
run_strategy candidate-prior-lifecycle

(
  cd "$OUTPUT"
  find results runtime -type f ! -name SHA256SUMS -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
) > "$OUTPUT/SHA256SUMS"
(
  cd "$OUTPUT"
  sha256sum --check --strict SHA256SUMS
)

"$PYTHON" - "$OUTPUT" "$SOURCE_COMMIT" "$RAW_SHA256" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
root, source_commit, raw_sha = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
strategies = ("prior-reset-only", "candidate-prior-lifecycle")
results = {name: json.loads((root / "results" / f"{name}.json").read_text()) for name in strategies}
payload = {
    "schema_version": 1,
    "diagnostic": "phase0_prior_init_lifecycle_reconstruction",
    "source_commit": source_commit,
    "raw_sha256": raw_sha,
    "candidate_id": 27,
    "init_state": 27,
    "results": results,
    "uid": os.getuid(),
    "gid": os.getgid(),
    "decision": "DIAGNOSTIC_COMPLETE_NO_SCIENTIFIC_UNBLINDING",
}
for name in ("COMPLETED_DIAGNOSTIC.json", "COMPLETED_EVALUATION_RESULT.json"):
    path = root / name
    temporary = root / f".{name}.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
PY
sha256sum "$OUTPUT/COMPLETED_DIAGNOSTIC.json" >> "$OUTPUT/SHA256SUMS"
sha256sum "$OUTPUT/COMPLETED_EVALUATION_RESULT.json" >> "$OUTPUT/SHA256SUMS"
sync -f "$OUTPUT/COMPLETED_DIAGNOSTIC.json"
sync -f "$OUTPUT/COMPLETED_EVALUATION_RESULT.json"
sync -f "$OUTPUT/SHA256SUMS"
