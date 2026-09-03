#!/usr/bin/env bash
set -Eeuo pipefail

# Stage-S B calibration is an eight-rank, aggregate-only evaluation. This
# launcher never submits a PAI job; a registry/controller supplies the unique
# PAI run identity, and the launcher derives the exact CPFS artifact directory
# from that identity.
umask 077

readonly LEON_UID=2254
readonly LEON_GID=2254
readonly ROOT=/mnt/cpfs/zbl-cpfs-new/USERS/leon
readonly CONTROLLER_RUN_ID="${PAI_CANARY_RUN_ID:?controller must inject PAI_CANARY_RUN_ID}"
# A resumed PAI incarnation gets a fresh controller run id, while the
# scientific application keeps one immutable CPFS directory.  This launcher
# is pinned to the already-started r14 application lineage; keeping that
# identity in the hashed payload avoids widening the registry pod-env
# contract for a one-off controller restart.
readonly RUN_ID=r142-stage-s-b-calibration-20260903-r14
readonly OUT_ROOT="$ROOT/logs/pai_registry/r142_stage_s/b_calibration"
readonly ARTIFACT_DIR="$OUT_ROOT/$RUN_ID"
readonly EXPECTED_GPUS=8
readonly WORLD_SIZE=8

# Pinned real Stage-R/LIBERO source trees. A dirty tree or commit drift is
# refused before the first simulator import.
# Scientific generation code remains bound to the r14 FIRST_WORK commit.
# The newer detached runtime changes only aggregate-only trial checkpointing
# so spot eviction cannot erase an entire calibration rank.
readonly STAGE_S_REPO="$ROOT/code/r142-stage-s-bcal-resume-runtime-20260903"
readonly STAGE_S_SOURCE_COMMIT=cb1281d43151e9436ae400fbbfa42b264fdfda29
readonly STAGE_S_RUNTIME_COMMIT=cfba9f13dfedde304e88961c7db12ab98c762c07
readonly QPILOTS="$ROOT/code/QPILOTS-r16p15-stage1-task64-20260812"
readonly QPILOTS_COMMIT=eacf47b981e3b22357f8a74902f8dad8cfcfa375
readonly OPENPI="$QPILOTS/third_party/openpi"
readonly OPENPI_COMMIT=54cbaee6ae0c010a1ed431871cdaa8f4684ac709
readonly LIBERO="$OPENPI/third_party/libero"
readonly LIBERO_COMMIT=f78abd68ee283de9f9be3c8f7e2a9ad60246e95c
readonly PYTHON="$ROOT/envs/openpi_py311/bin/python"

# r7 is read-only input. It must be a complete, hash-verified four-setting
# bundle before any calibration episode is allowed to start.
readonly B_VARIANT_RUN_ID=r142-stage-s-b-variants-20260903-r7
readonly B_VARIANT_RUN_ROOT="$ROOT/logs/r142_fp11_stage_s/b_variants/$B_VARIANT_RUN_ID"
readonly B_VARIANT_MARKER="$B_VARIANT_RUN_ROOT/COMPLETED_B_VARIANTS.json"
readonly B_VARIANT_SHA="$B_VARIANT_RUN_ROOT/SHA256SUMS"
readonly SOURCE_INIT_ROOT="$LIBERO/libero/libero/init_files/libero_10"
readonly RUN_SCOPED_CONFIG_SOURCE="$B_VARIANT_RUN_ROOT/libero-config/config.yaml"

readonly EXPECTED_ARTIFACT_DIR="$OUT_ROOT/$RUN_ID"
readonly OUT="$ARTIFACT_DIR"
readonly LIBERO_CONFIG_ROOT="$OUT/libero-config"

PHASE=bootstrap
FAILURE_RECORDED=0

write_failure_marker() {
  local rc="$1"
  [[ "$FAILURE_RECORDED" == 1 ]] && return 0
  [[ -d "$OUT" && ! -L "$OUT" ]] || return 0
  FAILURE_RECORDED=1
  python3 - "$OUT/FAILED_B_CALIBRATION.json" "$rc" "$RUN_ID" "$PHASE" <<'PY'
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "r142-stage-s-b-calibration-failure-v1",
    "status": "FAILED",
    "run_id": sys.argv[3],
    "phase": sys.argv[4],
    "exit_code": int(sys.argv[2]),
    "uid": os.getuid(),
    "gid": os.getgid(),
    "world_size": 8,
    "gpu_count": 8,
}
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
os.chmod(tmp, 0o600)
os.replace(tmp, path)
PY
}

on_error() {
  local rc=$?
  write_failure_marker "$rc" || true
  exit "$rc"
}
trap on_error ERR

in_daily_no_job_window() {
  local hm hour minute total
  hm="$(TZ=Asia/Shanghai date +%H%M)"
  hour=$((10#${hm:0:2}))
  minute=$((10#${hm:2:2}))
  total=$((hour * 60 + minute))
  (( (total >= 9 * 60 + 30 && total < 9 * 60 + 40) ||
     (total >= 19 * 60 + 30 && total < 19 * 60 + 40) ))
}

guard_daily_no_job_window() {
  if in_daily_no_job_window; then
    PHASE=daily_no_job_window
    python3 - "$OUT/REFUSED_DAILY_NO_JOB_WINDOW.json" "$RUN_ID" <<'PY'
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "r142-stage-s-b-calibration-window-refusal-v1",
    "status": "REFUSED_DAILY_NO_JOB_WINDOW",
    "run_id": sys.argv[2],
    "timezone": "Asia/Shanghai",
    "windows": ["09:30-09:40", "19:30-19:40"],
    "uid": os.getuid(),
    "gid": os.getgid(),
}
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
os.chmod(tmp, 0o600)
os.replace(tmp, path)
PY
    trap - ERR
    exit 75
  fi
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "B_CALIBRATION_REFUSED missing command: $1" >&2
    return 69
  }
}

PHASE=bootstrap
[[ "$RUN_ID" =~ ^r142-stage-s-b-calibration-20260903-r[0-9]+$ ]]
[[ "$CONTROLLER_RUN_ID" =~ ^r142-stage-s-b-calibration-20260903-r[0-9]+$ ]]
[[ "$ARTIFACT_DIR" == "$EXPECTED_ARTIFACT_DIR" ]]
[[ -d "$OUT" && ! -L "$OUT" ]]
[[ "$(realpath -e "$OUT")" == "$EXPECTED_ARTIFACT_DIR" ]]
[[ "$(stat -c '%u:%g' "$OUT")" == "$LEON_UID:$LEON_GID" ]]
[[ "$(stat -c '%a' "$OUT")" == 700 ]]
[[ "$(id -u):$(id -g)" == "$LEON_UID:$LEON_GID" ]]
for command_name in bash date find git nvidia-smi realpath sha256sum stat "$PYTHON" python3; do
  if [[ "$command_name" == /* ]]; then
    [[ -x "$command_name" ]] || { echo "B_CALIBRATION_REFUSED missing executable: $command_name" >&2; exit 69; }
  else
    require_command "$command_name"
  fi
done
guard_daily_no_job_window

PHASE=controller_lineage
mkdir -p "$OUT/controller-incarnations"
"$PYTHON" - "$OUT/controller-incarnations/$CONTROLLER_RUN_ID.json" "$CONTROLLER_RUN_ID" "$RUN_ID" "$STAGE_S_SOURCE_COMMIT" "$STAGE_S_RUNTIME_COMMIT" <<'PY'
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "r142-stage-s-pai-controller-incarnation-v1",
    "controller_run_id": sys.argv[2],
    "application_run_id": sys.argv[3],
    "scientific_source_commit": sys.argv[4],
    "runtime_source_commit": sys.argv[5],
    "runtime_change_scope": "aggregate_only_trial_resume_sidecars",
    "pai_job_id": os.environ.get("PAI_JOB_ID", ""),
}
if path.exists():
    if json.loads(path.read_text(encoding="utf-8")) != payload:
        raise SystemExit("controller incarnation identity drifted")
else:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
PY

# PAI's GPU device plugin does not always export CUDA_VISIBLE_DEVICES even
# though nvidia-smi exposes the allocated devices.  Enumerate the exact eight
# allocation-local indices before torchrun; each child then narrows this list
# to LOCAL_RANK in stage_s_gpu_rank_entry.py.
mapfile -t GPU_INDEXES < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
[[ "${#GPU_INDEXES[@]}" -eq "$EXPECTED_GPUS" ]]
for index in "${GPU_INDEXES[@]}"; do
  [[ "$index" =~ ^[0-9]+$ ]]
done
export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPU_INDEXES[*]}")"

PHASE=source_readback
for source_repo in "$STAGE_S_REPO" "$QPILOTS" "$OPENPI" "$LIBERO"; do
  [[ "$(git -C "$source_repo" rev-parse --is-inside-work-tree 2>/dev/null)" == true ]] || {
    echo "B_CALIBRATION_REFUSED missing git worktree: $source_repo" >&2
    exit 66
  }
  [[ -z "$(git -C "$source_repo" status --porcelain)" ]] || {
    echo "B_CALIBRATION_REFUSED dirty source tree: $source_repo" >&2
    exit 66
  }
done
[[ "$(git -C "$STAGE_S_REPO" rev-parse HEAD)" == "$STAGE_S_RUNTIME_COMMIT" ]]
[[ "$(git -C "$QPILOTS" rev-parse HEAD)" == "$QPILOTS_COMMIT" ]]
[[ "$(git -C "$OPENPI" rev-parse HEAD)" == "$OPENPI_COMMIT" ]]
[[ "$(git -C "$LIBERO" rev-parse HEAD)" == "$LIBERO_COMMIT" ]]
[[ -d "$SOURCE_INIT_ROOT" ]]

PHASE=r7_input_audit
[[ -s "$B_VARIANT_MARKER" && ! -L "$B_VARIANT_MARKER" ]]
[[ -s "$B_VARIANT_SHA" && ! -L "$B_VARIANT_SHA" ]]
(cd "$B_VARIANT_RUN_ROOT" && sha256sum --check --quiet SHA256SUMS)
[[ -s "$RUN_SCOPED_CONFIG_SOURCE" && ! -L "$RUN_SCOPED_CONFIG_SOURCE" ]]

# Delegate tensor/BDDL checks to the real Stage-S runtime, then check the r7
# completion identity, exact setting order, and path containment. The audit
# is read-only and persists no trial-level information.
export PYTHONPATH="$STAGE_S_REPO/src:$QPILOTS:$OPENPI/src:$OPENPI:$LIBERO"
"$PYTHON" - "$B_VARIANT_RUN_ROOT" "$SOURCE_INIT_ROOT" <<'PY'
import hashlib
import json
import pathlib
import sys

from r142_stage_s.libero import PROXIMITY_MAGNITUDES, validate_b_calibration_variants

run_root = pathlib.Path(sys.argv[1]).resolve()
source_init = pathlib.Path(sys.argv[2]).resolve()
expected = [f"proximity_{value:.2f}m" for value in PROXIMITY_MAGNITUDES]
marker_path = run_root / "COMPLETED_B_VARIANTS.json"
marker = json.loads(marker_path.read_text(encoding="utf-8"))
if marker.get("status") != "COMPLETED" or marker.get("marker_type") != "completed_b_variant_generation":
    raise SystemExit("r7 B variant completion marker is not completed")
if marker.get("settings") != 4 or marker.get("tasks_per_setting") != 10:
    raise SystemExit("r7 B variant completion dimensions drifted")
if marker.get("init_states_per_task", 0) < 16 or marker.get("old_init_reused") is not False:
    raise SystemExit("r7 B variant completion does not prove fresh states")
matrix_path = run_root / "variants" / "B_VARIANT_MATRIX.json"
if not matrix_path.is_file() or marker.get("matrix_sha256") != hashlib.sha256(matrix_path.read_bytes()).hexdigest():
    raise SystemExit("r7 B variant matrix completion hash mismatch")
matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
settings = matrix.get("settings")
if not isinstance(settings, list) or [row.get("setting_id") for row in settings] != expected:
    raise SystemExit("r7 B variant setting order drifted")
roots = [run_root / "variants" / setting for setting in expected]
for root in roots:
    if not root.is_dir() or root.is_symlink():
        raise SystemExit(f"missing r7 variant root: {root}")
    config = root / "config.yaml"
    if not config.is_file() or config.is_symlink():
        raise SystemExit(f"missing run-scoped B variant config: {config}")
    state_marker = json.loads((root / "REGENERATED_INIT_STATES.json").read_text(encoding="utf-8"))
    if "flatten" not in str(state_marker.get("state_format", "")):
        raise SystemExit(f"variant does not declare flattened simulator state: {root}")
    if state_marker.get("regenerated") is not True or state_marker.get("old_init_reused") is not False:
        raise SystemExit(f"variant state marker is not fresh/real: {root}")
    row = next(item for item in settings if item.get("setting_id") == root.name)
    if len(row.get("tasks", [])) != 10:
        raise SystemExit(f"variant task count drifted: {root}")
    for task in row["tasks"]:
        for key in ("bddl_path", "init_states_path"):
            path = pathlib.Path(task[key]).resolve()
            if not path.is_file() or path.is_symlink() or not path.is_relative_to(root.resolve()):
                raise SystemExit(f"r7 consumed artifact escapes variant root: {path}")
validate_b_calibration_variants(roots, source_init, expected_settings=expected)
print("B_R7_INPUT_AUDIT_PASS")
PY

# Copy the r7 config into this run's output once, then require byte identity on
# every resumed incarnation. This prevents process-global HOME/LIBERO config
# state deciding which variant was imported first.
PHASE=run_scoped_config
if [[ -e "$LIBERO_CONFIG_ROOT/config.yaml" || -L "$LIBERO_CONFIG_ROOT/config.yaml" ]]; then
  [[ -f "$LIBERO_CONFIG_ROOT/config.yaml" && ! -L "$LIBERO_CONFIG_ROOT/config.yaml" ]]
  cmp -s "$LIBERO_CONFIG_ROOT/config.yaml" "$RUN_SCOPED_CONFIG_SOURCE"
else
  mkdir -p "$LIBERO_CONFIG_ROOT"
  cp -- "$RUN_SCOPED_CONFIG_SOURCE" "$LIBERO_CONFIG_ROOT/config.yaml"
  chmod 600 "$LIBERO_CONFIG_ROOT/config.yaml"
fi
[[ -s "$LIBERO_CONFIG_ROOT/config.yaml" && ! -L "$LIBERO_CONFIG_ROOT/config.yaml" ]]
export LIBERO_CONFIG_PATH="$LIBERO_CONFIG_ROOT"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=device
export NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
export XDG_CACHE_HOME="$OUT/xdg-cache"
mkdir -p "$XDG_CACHE_HOME"

PHASE=first_work
if [[ -e "$OUT/FIRST_WORK.json" || -L "$OUT/FIRST_WORK.json" ]]; then
  [[ -f "$OUT/FIRST_WORK.json" && ! -L "$OUT/FIRST_WORK.json" ]]
  "$PYTHON" - "$OUT/FIRST_WORK.json" "$RUN_ID" "$STAGE_S_SOURCE_COMMIT" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("status") != "FIRST_WORK" or payload.get("run_id") != sys.argv[2] or payload.get("world_size") != 8 or payload.get("gpu_count") != 8 or payload.get("stage_s_source_commit") != sys.argv[3]:
    raise SystemExit("FIRST_WORK.json identity drifted")
PY
else
  "$PYTHON" - "$OUT/FIRST_WORK.json" "$RUN_ID" "$STAGE_S_SOURCE_COMMIT" "$QPILOTS_COMMIT" "$OPENPI_COMMIT" "$LIBERO_COMMIT" "$B_VARIANT_RUN_ID" <<'PY'
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "r142-stage-s-b-calibration-first-work-v1",
    "status": "FIRST_WORK",
    "run_id": sys.argv[2],
    "uid": os.getuid(),
    "gid": os.getgid(),
    "gpu_count": 8,
    "world_size": 8,
    "stage_s_source_commit": sys.argv[3],
    "qpilots_commit": sys.argv[4],
    "openpi_commit": sys.argv[5],
    "libero_commit": sys.argv[6],
    "input_bundle_run_id": sys.argv[7],
}
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
os.chmod(tmp, 0o600)
os.replace(tmp, path)
PY
fi

PHASE=calibration_shards
VARIANT_ARGS=(
  --variant-root "$B_VARIANT_RUN_ROOT/variants/proximity_0.06m"
  --variant-root "$B_VARIANT_RUN_ROOT/variants/proximity_0.08m"
  --variant-root "$B_VARIANT_RUN_ROOT/variants/proximity_0.10m"
  --variant-root "$B_VARIANT_RUN_ROOT/variants/proximity_0.12m"
)

# Only the controller writes the immutable plan.  CPFS does not guarantee
# coherent concurrent create/replace semantics for eight writers using the
# same temporary path, so plan creation must finish before torch workers start.
cd "$STAGE_S_REPO"
"$PYTHON" scripts/stage_s_libero_calibrate.py \
  --substrate B --mode prepare --output-root "$OUT" \
  --world-size "$WORLD_SIZE" \
  "${VARIANT_ARGS[@]}" \
  --source-init-root "$SOURCE_INIT_ROOT" \
  --libero-config-root "$LIBERO_CONFIG_ROOT"

# The pinned OpenPI interpreter owns torch, JAX and the policy dependencies.
# Invoking its module entrypoint prevents PATH's system torchrun from silently
# selecting /usr/local/bin/python.  WORLD_SIZE remains frozen to eight.
"$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$WORLD_SIZE" \
  scripts/stage_s_gpu_rank_entry.py scripts/stage_s_libero_calibrate.py \
  --substrate B --mode shard --output-root "$OUT" \
  --world-size "$WORLD_SIZE" \
  "${VARIANT_ARGS[@]}" \
  --source-init-root "$SOURCE_INIT_ROOT" \
  --libero-config-root "$LIBERO_CONFIG_ROOT" \
  --policy-checkpoint "$ROOT/cache/openpi/r16p15/openpi-assets/checkpoints/pi05_libero" \
  --qpilots-root "$QPILOTS" --libero-root "$LIBERO"

guard_daily_no_job_window

PHASE=calibration_aggregate
"$PYTHON" scripts/stage_s_libero_calibrate.py \
  --substrate B --mode aggregate --output-root "$OUT" \
  --world-size "$WORLD_SIZE"

PHASE=completion_publish
"$PYTHON" - "$OUT" "$RUN_ID" "$B_VARIANT_RUN_ID" "$STAGE_S_SOURCE_COMMIT" "$QPILOTS_COMMIT" "$OPENPI_COMMIT" "$LIBERO_COMMIT" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

from r142_stage_s.libero import (
    CALIBRATION_RESULT_SCHEMA,
    CALIBRATION_SEED,
    PROXIMITY_MAGNITUDES,
    STAGE_S_PROTOCOL_ID,
    verify_calibration_aggregate,
)

root = pathlib.Path(sys.argv[1]).resolve()
run_id, input_run_id = sys.argv[2], sys.argv[3]
stage_s_commit, qpilots_commit, openpi_commit, libero_commit = sys.argv[4:8]
settings = [f"proximity_{value:.2f}m" for value in PROXIMITY_MAGNITUDES]
result = root / "CALIBRATION_RESULT.json"
verify_calibration_aggregate(result, settings, calibration_seed=CALIBRATION_SEED, world_size=8)
core_marker = root / "COMPLETED_CALIBRATION.json"
core_payload = json.loads(core_marker.read_text(encoding="utf-8"))
if core_payload.get("schema") != CALIBRATION_RESULT_SCHEMA:
    raise SystemExit("core calibration marker schema drifted")

rank_markers = []
rank_marker_sha256 = {}
for rank in range(8):
    marker = root / "shards" / f"rank-{rank:05d}" / "COMPLETED_SHARD.json"
    result_path = marker.parent / "RESULT.json"
    sums = marker.parent / "SHA256SUMS"
    if not marker.is_file() or not result_path.is_file() or not sums.is_file():
        raise SystemExit(f"missing rank completion evidence: {marker}")
    expected = f"{hashlib.sha256(result_path.read_bytes()).hexdigest()}  RESULT.json\n"
    if sums.read_text(encoding="utf-8") != expected:
        raise SystemExit(f"rank SHA mismatch: {marker.parent}")
    rel = marker.relative_to(root).as_posix()
    rank_markers.append(rel)
    rank_marker_sha256[rel] = hashlib.sha256(marker.read_bytes()).hexdigest()

payload = {
    "schema": "r142-stage-s-b-calibration-completion-v1",
    "status": "COMPLETED",
    "protocol_id": STAGE_S_PROTOCOL_ID,
    "substrate": "B",
    "run_id": run_id,
    "input_bundle_run_id": input_run_id,
    "calibration_result": "CALIBRATION_RESULT.json",
    "calibration_result_sha256": hashlib.sha256(result.read_bytes()).hexdigest(),
    "calibration_result_schema": CALIBRATION_RESULT_SCHEMA,
    "calibration_seed": CALIBRATION_SEED,
    "world_size": 8,
    "rank_markers": rank_markers,
    "rank_marker_sha256": rank_marker_sha256,
    "source": {
        "stage_s_commit": stage_s_commit,
        "qpilots_commit": qpilots_commit,
        "openpi_commit": openpi_commit,
        "libero_commit": libero_commit,
        "variant_bundle": input_run_id,
    },
    "provenance": {
        "variant_root": "/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_s/b_variants/" + input_run_id + "/variants",
        "source_init_root": "/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/QPILOTS-r16p15-stage1-task64-20260812/third_party/openpi/third_party/libero/libero/libero/init_files/libero_10",
        "state_format": "torch.save(sim.get_state().flatten())",
        "old_init_reused": False,
    },
    "compute": {
        "worker_count": 1,
        "gpu_count": 8,
        "cpu_cores": 88,
        "memory_gib": 1400,
        "shared_memory_gib": 1400,
        "resource_pool": "exp-robot",
        "resource_alias": "idle-a800-robot-stage-s-graphics-8gpu",
        "resource_id": "quota1ssrabud0bh",
    },
    "persistence": {
        "artifact_dir": str(root),
        "resume_same_run_id": True,
        "rank_shards_required": 8,
        "aggregate_sha_file": "SHA256SUMS",
        "bundle_sha_file": "B_SHA256SUMS",
    },
}

marker = root / "COMPLETED_B_CALIBRATION.json"
if marker.is_file():
    if json.loads(marker.read_text(encoding="utf-8")) != payload:
        raise SystemExit("existing COMPLETED_B_CALIBRATION.json drifted")
else:
    tmp = marker.with_suffix(marker.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, marker)
files = [
    root / "CALIBRATION_RESULT.json",
    root / "COMPLETED_CALIBRATION.json",
    root / "COMPLETED_B_CALIBRATION.json",
    root / "FIRST_WORK.json",
]
for rank in range(8):
    files.extend(
        [
            root / "shards" / f"rank-{rank:05d}" / "RESULT.json",
            root / "shards" / f"rank-{rank:05d}" / "COMPLETED_SHARD.json",
        ]
    )
lines = []
for path in files:
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"missing final B calibration file: {path}")
    lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root).as_posix()}")
bundle_sums = root / "B_SHA256SUMS"
tmp = bundle_sums.with_suffix(bundle_sums.suffix + ".tmp")
tmp.write_text("\n".join(sorted(lines)) + "\n", encoding="utf-8")
os.chmod(tmp, 0o600)
os.replace(tmp, bundle_sums)
PY
(cd "$OUT" && sha256sum --check --quiet B_SHA256SUMS)

rm -f -- "$OUT/FAILED_B_CALIBRATION.json"
trap - ERR
printf 'B_CALIBRATION_COMPLETED run_id=%s output=%s\n' "$RUN_ID" "$OUT"
