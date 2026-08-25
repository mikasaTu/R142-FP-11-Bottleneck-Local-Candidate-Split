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
PARENT_DIR=/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_r/phase0r/r142-stage-r-phase0r-20260824-r4-idle4
SHARD_CONTRACT="$REPO/configs/stage_r_phase0r_parallel_shards.json"
SHARD_CONTRACT_SHA256=c9ab70c12fd226381f622d279dd6e5a884f03ada7d88b9446ab62eee3a168da0
ARTIFACT_DIR=${PAI_CANARY_RUN_DIR:?PAI_CANARY_RUN_DIR is required}
RUN_ID=${PAI_CANARY_RUN_ID:?PAI_CANARY_RUN_ID is required}
SHARD=${PAI_TASK_SHARD:?PAI_TASK_SHARD is required}

trap 'code=$?; printf "STAGE_R_PHASE0R_SUBSET_FAILED line=%s exit_code=%s command=%q\n" "${BASH_LINENO[0]:-unknown}" "$code" "$BASH_COMMAND" >&2; exit "$code"' ERR

test "$(id -u):$(id -g)" = 2254:2254
test "${PAI_CANARY_EXPECTED_GPUS:-}" = 4
test "$SHARD" = A || test "$SHARD" = B
test -x "$PYTHON"
test "$(git -C "$REPO" rev-parse "$SOURCE_COMMIT^{commit}")" = "$SOURCE_COMMIT"
test "$(git -C "$QPILOTS" rev-parse HEAD)" = "$QPILOTS_COMMIT"
test "$(git -C "$OPENPI" rev-parse HEAD)" = "$OPENPI_COMMIT"
test "$(git -C "$LIBERO" rev-parse HEAD)" = "$LIBERO_COMMIT"
git -C "$REPO" diff --quiet
git -C "$REPO" diff --cached --quiet
test -z "$(git -C "$QPILOTS" status --porcelain)"
test "$(sha256sum "$SHARD_CONTRACT" | awk '{print $1}')" = "$SHARD_CONTRACT_SHA256"
test "$(cat "$PARENT_DIR/runtime/source_commit.txt")" = "$SOURCE_COMMIT"
test -f "$GATE_DIR/COMPLETED_EVALUATION_RESULT.json"
(cd "$GATE_DIR" && sha256sum --check --quiet SHA256SUMS)
test "$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["decision"])' "$GATE_DIR/COMPLETED_EVALUATION_RESULT.json")" = ENGINEERING_GATES_PASSED
case "$ARTIFACT_DIR" in
  /mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_r/phase0r_parallel/*) ;;
  *) echo "artifact directory escaped frozen Phase-0R parallel root" >&2; exit 71 ;;
esac
test "$(basename "$ARTIFACT_DIR")" = "$RUN_ID"
test "$(stat -c '%u:%g' "$ARTIFACT_DIR")" = 2254:2254

cd "$ARTIFACT_DIR"
if [ -f COMPLETED_EVALUATION_RESULT.json ] && [ -f SHA256SUMS ]; then
  sha256sum --check --quiet SHA256SUMS
  echo "STAGE_R_PHASE0R_SUBSET_ALREADY_COMPLETE_VALIDATED shard=$SHARD"
  exit 0
fi

mkdir -p frozen_source runtime raw
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

mapfile -t shard_rows < <("$PYTHON" - "$SHARD_CONTRACT" "$SHARD" <<'PY'
import json, sys
contract = json.load(open(sys.argv[1]))
shard = contract["shards"][sys.argv[2]]
for rank, target in zip(shard["global_ranks"], shard["target_tasks"], strict=True):
    prerequisites = shard["prerequisites"][str(rank)]
    print(f"{rank}\t{target}\t{','.join(prerequisites)}")
PY
)
test "${#shard_rows[@]}" = 4

validate_pair() {
  local directory=$1 stem=$2
  "$PYTHON" - "$directory" "$stem" <<'PY'
import collections, hashlib, json, os, pathlib, re, sys
import numpy as np
root = pathlib.Path(sys.argv[1])
stem = sys.argv[2]
metadata = json.loads((root / f"{stem}.json").read_text())
path = root / metadata["data_file"]
digest = hashlib.sha256(path.read_bytes()).hexdigest()
match = re.fullmatch(r"(libero_(?:spatial|object|goal|10))_task(\d{2})", stem)
assert match is not None
assert metadata["protocol_id"] == "r142-stage-r-phase0r-v1"
assert metadata["suite"] == match.group(1)
assert metadata["task_id"] == int(match.group(2))
assert metadata["rollout_count"] == 512
assert path.name == f"{stem}.npz"
assert digest == metadata["data_sha256"]
assert (root / f"{stem}.json").stat().st_uid == 2254
assert (root / f"{stem}.json").stat().st_gid == 2254
assert path.stat().st_uid == 2254
assert path.stat().st_gid == 2254
with np.load(path, allow_pickle=False) as data:
    required = {"lengths", "offsets", "actions", "eef", "objects", "progress", "success", "init_state", "candidate_id", "rollout_seed", "policy_forwards"}
    assert required.issubset(data.files)
    assert data["success"].shape == (512,)
    assert data["init_state"].shape == (512,)
    assert data["candidate_id"].shape == (512,)
    assert data["rollout_seed"].shape == (512,)
    assert data["policy_forwards"].shape == (512,)
    assert data["lengths"].shape == (512,)
    assert data["offsets"].shape == (513,)
    assert int(data["offsets"][0]) == 0
    assert np.all(np.diff(data["offsets"]) >= 0)
    assert np.array_equal(np.diff(data["offsets"]), data["lengths"])
    assert int(data["offsets"][-1]) == len(data["actions"])
    assert len(data["actions"]) == len(data["eef"]) == len(data["objects"]) == len(data["progress"])
    assert np.isfinite(data["actions"]).all()
    assert np.isfinite(data["eef"]).all()
    assert np.isfinite(data["objects"]).all()
    assert np.isfinite(data["progress"]).all()
    counts = collections.Counter(map(int, data["init_state"]))
    assert len(counts) == 16 and set(counts.values()) == {32}
    observed = set()
    for init_state in counts:
        selector = data["init_state"] == init_state
        candidates = data["candidate_id"][selector]
        seeds = data["rollout_seed"][selector]
        assert sorted(map(int, candidates)) == list(range(32))
        for candidate, seed in zip(candidates, seeds, strict=True):
            expected = int.from_bytes(hashlib.sha256(
                f"r142-stage-r-phase0r-v1|{metadata['suite']}|{metadata['task_id']}|{int(init_state)}|{int(candidate)}".encode()
            ).digest()[:8], "big")
            assert int(seed) == expected
            observed.add(int(seed))
    assert len(observed) == 512
PY
}

for row in "${shard_rows[@]}"; do
  IFS=$'\t' read -r global_rank target prerequisites <<<"$row"
  IFS=',' read -r -a prerequisite_array <<<"$prerequisites"
  for stem in "${prerequisite_array[@]}"; do
    validate_pair "$PARENT_DIR/raw" "$stem"
    for suffix in npz json; do
      destination="raw/${stem}.${suffix}"
      if [ ! -e "$destination" ]; then
        ln "$PARENT_DIR/raw/${stem}.${suffix}" "$destination"
      fi
    done
    validate_pair raw "$stem"
  done
done

"$PYTHON" - "$ARTIFACT_DIR" "$PARENT_DIR" "$SHARD_CONTRACT" "$SHARD" <<'PY'
import hashlib, json, os, pathlib, sys

root = pathlib.Path(sys.argv[1])
parent = pathlib.Path(sys.argv[2])
contract = json.loads(pathlib.Path(sys.argv[3]).read_text())
shard_name = sys.argv[4]
shard = contract["shards"][shard_name]
imports = []
for rank in shard["global_ranks"]:
    for stem in shard["prerequisites"][str(rank)]:
        metadata_path = root / "raw" / f"{stem}.json"
        metadata = json.loads(metadata_path.read_text())
        npz_path = root / "raw" / metadata["data_file"]
        imports.append({
            "stem": stem,
            "source_metadata_path": str(parent / "raw" / metadata_path.name),
            "source_npz_path": str(parent / "raw" / npz_path.name),
            "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
            "npz_sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
        })
manifest = {
    "schema_version": 1,
    "protocol_id": contract["protocol_id"],
    "shard": shard_name,
    "parent_run_id": contract["parent_run_id"],
    "parent_job_id": contract["parent_job_id"],
    "scientific_source_commit": contract["scientific_source_commit"],
    "imports": sorted(imports, key=lambda value: value["stem"]),
    "target_tasks": shard["target_tasks"],
    "outcomes_unblinded": False,
    "uid": os.getuid(),
    "gid": os.getgid(),
}
path = root / "IMPORTED_TASKS_MANIFEST.json"
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
os.replace(temporary, path)
PY

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
local_rank=0
for row in "${shard_rows[@]}"; do
  IFS=$'\t' read -r global_rank target prerequisites <<<"$row"
  visible_devices="$local_rank"
  if [ "$local_rank" != 0 ]; then
    visible_devices="$local_rank,0"
  fi
  CUDA_VISIBLE_DEVICES="$visible_devices" EGL_DEVICE_ID=0 MUJOCO_EGL_DEVICE_ID=0 \
    "$PYTHON" frozen_source/scripts/stage_r_phase0r.py \
    --qpilots-root "$QPILOTS" \
    --libero-root "$LIBERO" \
    --checkpoint "$CHECKPOINT" \
    --output "$ARTIFACT_DIR/raw" \
    --microbatch 4 \
    --rank "$global_rank" --world-size 8 \
    > "runtime/logs/phase0r.globalrank${global_rank}.log" 2>&1 &
  pids+=("$!")
  local_rank=$((local_rank + 1))
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
test "$failed" = 0

"$PYTHON" - "$ARTIFACT_DIR" "$SHARD_CONTRACT" "$SHARD" <<'PY'
import hashlib, json, os, pathlib, sys

root = pathlib.Path(sys.argv[1])
contract_path = pathlib.Path(sys.argv[2])
shard_name = sys.argv[3]
contract = json.loads(contract_path.read_text())
shard = contract["shards"][shard_name]
artifacts = []
for rank, stem in zip(shard["global_ranks"], shard["target_tasks"], strict=True):
    metadata_path = root / "raw" / f"{stem}.json"
    metadata = json.loads(metadata_path.read_text())
    npz_path = root / "raw" / metadata["data_file"]
    digest = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    assert metadata["protocol_id"] == contract["protocol_id"]
    assert metadata["suite"] == stem.rsplit("_task", 1)[0]
    assert metadata["task_id"] == int(stem.rsplit("_task", 1)[1])
    assert metadata["data_file"] == f"{stem}.npz"
    assert metadata["rollout_count"] == 512
    assert digest == metadata["data_sha256"]
    marker = root / "raw" / f"COMPLETE.rank{rank}.json"
    assert marker.is_file()
    marker_payload = json.loads(marker.read_text())
    expected_tasks = shard["prerequisites"][str(rank)] + [stem]
    assert marker_payload == {
        "completed_tasks": expected_tasks,
        "protocol_id": contract["protocol_id"],
        "rank": rank,
    }
    artifacts.extend([
        {"path": str(npz_path.relative_to(root)), "sha256": digest},
        {"path": str(metadata_path.relative_to(root)), "sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest()},
        {"path": str(marker.relative_to(root)), "sha256": hashlib.sha256(marker.read_bytes()).hexdigest()},
    ])

raw_complete = {
    "schema_version": 1,
    "protocol_id": contract["protocol_id"],
    "shard": shard_name,
    "target_tasks": shard["target_tasks"],
    "task_count": len(shard["target_tasks"]),
    "rollout_count": 512 * len(shard["target_tasks"]),
    "parent_run_id": contract["parent_run_id"],
    "parent_job_id": contract["parent_job_id"],
    "outcomes_unblinded": False,
    "artifacts": artifacts,
}
first = {
    "schema_version": 1,
    "milestone": "parallel_raw_subset_complete",
    "shard": shard_name,
    "task_count": raw_complete["task_count"],
    "rollout_count": raw_complete["rollout_count"],
    "uid": os.getuid(),
    "gid": os.getgid(),
}
completed = {
    "schema_version": 1,
    "success_gate": "persisted_complete_phase0r_raw_subset",
    "decision": "RAW_SUBSET_COMPLETE_NO_UNBLINDING",
    "shard": shard_name,
    "target_tasks": shard["target_tasks"],
    "global_phase0r_complete": False,
    "checkpoint": "CHECKPOINT_1_PENDING_GLOBAL_MERGE",
    "phase1_authorized": False,
    "uid": os.getuid(),
    "gid": os.getgid(),
}
for name, value in (
    ("COMPLETED_SUBSET_RAW.json", raw_complete),
    ("FIRST_WORK.json", first),
    ("COMPLETED_EVALUATION_RESULT.json", completed),
):
    path = root / name
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
PY
"$PYTHON" - "$ARTIFACT_DIR" <<'PY'
import os, pathlib, subprocess, sys
root = pathlib.Path(sys.argv[1])
paths = [
    root / "IMPORTED_TASKS_MANIFEST.json",
    root / "COMPLETED_SUBSET_RAW.json",
    root / "FIRST_WORK.json",
    root / "COMPLETED_EVALUATION_RESULT.json",
]
paths.extend(sorted((root / "raw").glob("*.json")))
paths.extend(sorted((root / "raw").glob("*.npz")))
lines = []
for path in sorted(set(paths)):
    digest = subprocess.check_output(["sha256sum", str(path)], text=True).split()[0]
    lines.append(f"{digest}  {path.relative_to(root)}\n")
destination = root / "SHA256SUMS"
temporary = root / ".SHA256SUMS.tmp"
with temporary.open("w", encoding="utf-8") as handle:
    handle.writelines(lines)
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, destination)
directory_fd = os.open(root, os.O_RDONLY)
try:
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)
PY
(cd "$ARTIFACT_DIR" && sha256sum --check --quiet SHA256SUMS)
sync -f "$ARTIFACT_DIR/FIRST_WORK.json"
sync -f "$ARTIFACT_DIR/COMPLETED_EVALUATION_RESULT.json"
sync -f "$ARTIFACT_DIR/SHA256SUMS"
echo "STAGE_R_PHASE0R_PARALLEL_SUBSET_COMPLETE shard=$SHARD"
