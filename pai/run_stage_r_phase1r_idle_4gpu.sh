#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

trap 'code=$?; printf "STAGE_R_PHASE1R_NATURAL_FAILED line=%s exit_code=%s command=%q\n" "${BASH_LINENO[0]:-unknown}" "$code" "$BASH_COMMAND" >&2; exit "$code"' ERR

# The final scientific source pin is intentionally supplied by the controller.
# A PAI run must provide one exact commit, so optimization cannot silently
# change the source used by a natural shard.
readonly REQUIRED_SOURCE_COMMIT=57859fcbb36776e0049ce24fb1abbadab0de46d5
# The committed default is the final pre-launch scientific source. A runtime
# variable is accepted only for controlled local validation; the formal PAI
# templates use this exact committed default.
SOURCE_COMMIT="${PAI_PHASE1R_SOURCE_COMMIT:-$REQUIRED_SOURCE_COMMIT}"
if [[ -z "$SOURCE_COMMIT" || "$SOURCE_COMMIT" =~ ^0{40}$ || ! "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "SOURCE_COMMIT must be one nonzero exact 40-hex git commit" >&2
  exit 64
fi

readonly REPO=/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/R142-FP-11-Bottleneck-Local-Candidate-Split
readonly QPILOTS=/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/QPILOTS-r16p15-stage1-task64-20260812
readonly QPILOTS_COMMIT=eacf47b981e3b22357f8a74902f8dad8cfcfa375
readonly OPENPI="$QPILOTS/third_party/openpi"
readonly OPENPI_COMMIT=54cbaee6ae0c010a1ed431871cdaa8f4684ac709
readonly LIBERO="$OPENPI/third_party/libero"
readonly LIBERO_COMMIT=f78abd68ee283de9f9be3c8f7e2a9ad60246e95c
readonly PYTHON=/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/openpi_py311/bin/python
readonly CHECKPOINT=/mnt/cpfs/zbl-cpfs-new/USERS/leon/cache/openpi/r16p15/openpi-assets/checkpoints/pi05_libero
readonly CHECKPOINT_TREE_SHA=42d571bd87f05f1182810f5a8bfa6d084c0d0dd277aff739bcf8f69868e6fb99
readonly CHECKPOINT_ATTESTATION="$REPO/configs/stage_r_pi05_libero_checkpoint_attestation.json"
readonly CHECKPOINT_ATTESTATION_SHA256=d050805b0c1e9e8d8e879c7443bb10504859c654d0ba031bbbc6ce3635b02fca
readonly OUTPUT_ROOT=/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_r/phase1r
readonly PHASE0_MERGE_ROOT=/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_r/phase0r_merged/r142-stage-r-phase0r-authoritative-20260827
readonly PHASE0_RAW="$PHASE0_MERGE_ROOT/raw"
readonly PROTOCOL_SHA256=7e4de68cba5c0fdb288ee25d81f30b72d65483753973f06f44b395e3db0b9cb4
readonly SHARDS_SHA256=f0e4b137d5f5b39737f671dc273428fdd5b646327863f03a542c2c7c5e2977d6
readonly EXECUTION_CONFIG="$REPO/configs/stage_r_phase1r_execution_idle4.json"
readonly EXECUTION_CONFIG_SHA256=0279da07d96f243a71503230903a55aa2027997ed3b422f379a040523ba63dd3
readonly SELECTION_MANIFEST_SHA256=082f1f28f6ed8bddb1ed2ef87a3b848ac3daccec5c333f7f2cf1c4ef5d988231
readonly SELECTION_SHA256SUMS_SHA256=e7aa31741834ae5ac1064e3fab5a2b307e8202ad543b2d11f1e4e5fb337b3893
readonly CALIBRATION_SHA256=f8a6486a96b9fc02071c391c4971ac2251d5c5e89dbb405f9d51dcd44fbfad6a
readonly CALIBRATION_SHA256SUMS_SHA256=eab9085ceea2eabe0e61a939131582847bfd88d850a519d638eb8625f70f4cd5
readonly AUTHORITY_MANIFEST_SHA256=3d5a37ec8a7e2c0dfd0c808ad59553c43a13c846b90f99c1afaa3529a072469c
ARTIFACT_DIR="${PAI_CANARY_RUN_DIR:?PAI_CANARY_RUN_DIR is required}"
RUN_ID="${PAI_CANARY_RUN_ID:?PAI_CANARY_RUN_ID is required}"

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$ ]]; then
  echo "PAI_CANARY_RUN_ID contains unsupported characters" >&2
  exit 64
fi
case "$RUN_ID" in
  r142-stage-r-phase1r-shard-a0-*) EXECUTION_SHARD=A0; SHARD=A ;;
  r142-stage-r-phase1r-shard-a1-*) EXECUTION_SHARD=A1; SHARD=A ;;
  r142-stage-r-phase1r-shard-b0-*) EXECUTION_SHARD=B0; SHARD=B ;;
  r142-stage-r-phase1r-shard-b1-*) EXECUTION_SHARD=B1; SHARD=B ;;
  *) echo "run id must identify execution shard A0, A1, B0, or B1" >&2; exit 64 ;;
esac
if [[ "${PAI_CANARY_EXPECTED_GPUS:-}" != 4 ]]; then
  echo "natural Phase-1R execution shards require PAI_CANARY_EXPECTED_GPUS=4" >&2
  exit 77
fi
if [[ "$(id -u):$(id -g)" != 2254:2254 ]]; then
  echo "natural Phase-1R must run as UID:GID 2254:2254" >&2
  exit 77
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "pinned Python interpreter is unavailable: $PYTHON" >&2
  exit 77
fi
case "$ARTIFACT_DIR" in
  "$OUTPUT_ROOT"/*) ;;
  *) echo "artifact directory escaped frozen Phase-1R output root" >&2; exit 71 ;;
esac
if [[ "$(basename -- "$ARTIFACT_DIR")" != "$RUN_ID" ]]; then
  echo "artifact basename does not equal PAI_CANARY_RUN_ID" >&2
  exit 71
fi
[[ -d "$ARTIFACT_DIR" ]] || { echo "controller did not create the artifact directory" >&2; exit 71; }
[[ "$(stat -c '%u:%g' "$ARTIFACT_DIR")" == 2254:2254 ]] || { echo "artifact owner is not 2254:2254" >&2; exit 77; }

[[ "$(git -C "$REPO" rev-parse "$SOURCE_COMMIT^{commit}")" == "$SOURCE_COMMIT" ]] || { echo "source commit is not present" >&2; exit 78; }
git -C "$REPO" diff --quiet
git -C "$REPO" diff --cached --quiet
[[ "$(git -C "$QPILOTS" rev-parse HEAD)" == "$QPILOTS_COMMIT" ]] || { echo "QPILOTS commit drifted" >&2; exit 78; }
[[ "$(git -C "$OPENPI" rev-parse HEAD)" == "$OPENPI_COMMIT" ]] || { echo "OpenPI commit drifted" >&2; exit 78; }
[[ "$(git -C "$LIBERO" rev-parse HEAD)" == "$LIBERO_COMMIT" ]] || { echo "LIBERO commit drifted" >&2; exit 78; }
[[ -z "$(git -C "$QPILOTS" status --porcelain)" ]]
[[ -z "$(git -C "$OPENPI" status --porcelain)" ]]
[[ -z "$(git -C "$LIBERO" status --porcelain)" ]]

cd "$ARTIFACT_DIR"
mkdir -p frozen_source runtime runtime/tmp runtime/cache runtime/logs natural

sha256_file() { sha256sum -- "$1" | awk '{print $1}'; }
expect_sha256() {
  local path="$1" expected="$2" observed
  [[ -f "$path" && ! -L "$path" ]] || { echo "missing pinned file: $path" >&2; return 1; }
  observed="$(sha256_file "$path")"
  [[ "$observed" == "$expected" ]] || { echo "SHA256 mismatch for $path: $observed != $expected" >&2; return 1; }
}
expect_sha256 "$EXECUTION_CONFIG" "$EXECUTION_CONFIG_SHA256"
expect_sha256 "$CHECKPOINT_ATTESTATION" "$CHECKPOINT_ATTESTATION_SHA256"

if [[ ! -e runtime/source_commit.txt ]]; then
  if find frozen_source -mindepth 1 -print -quit | grep -q .; then
    echo "frozen_source is nonempty without a source commit marker" >&2
    exit 79
  fi
  git -C "$REPO" archive "$SOURCE_COMMIT" | tar -xf - -C frozen_source
  printf '%s\n' "$SOURCE_COMMIT" > runtime/source_commit.txt
  git -C "$REPO" rev-parse "$SOURCE_COMMIT^{tree}" > runtime/source_tree.txt
  ( cd frozen_source; find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum ) > runtime/frozen_source.sha256
else
  [[ "$(cat runtime/source_commit.txt)" == "$SOURCE_COMMIT" ]] || { echo "resume source commit differs from frozen archive" >&2; exit 79; }
  [[ "$(cat runtime/source_tree.txt)" == "$(git -C "$REPO" rev-parse "$SOURCE_COMMIT^{tree}")" ]] || { echo "resume source tree differs from pinned commit" >&2; exit 79; }
  [[ -s runtime/source_tree.txt && -s runtime/frozen_source.sha256 ]] || { echo "source archive marker is incomplete" >&2; exit 79; }
  [[ -n "$(find frozen_source -type f -print -quit)" ]] || { echo "frozen source archive is empty" >&2; exit 79; }
  if find frozen_source -type l -print -quit | grep -q .; then
    echo "frozen source contains a symlink" >&2
    exit 79
  fi
  (
    cd frozen_source
    sha256sum --check --strict ../runtime/frozen_source.sha256 >/dev/null
    diff -u \
      <(awk '{print $2}' ../runtime/frozen_source.sha256 | LC_ALL=C sort) \
      <(find . -type f -print | LC_ALL=C sort) >/dev/null
  )
fi

# Resume is fail-closed.  Validate the immutable completion manifest before
# writing any fresh runtime diagnostics; otherwise preserve both old markers
# and rebuild them from the resumed cells.
if [[ -f COMPLETED_EVALUATION_RESULT.json && -f SHA256SUMS ]]; then
  if "$PYTHON" - "$ARTIFACT_DIR" "$SHARD" "$SOURCE_COMMIT" <<'PY'
import hashlib, json, re, sys
from pathlib import Path, PurePosixPath
root, shard, source = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
marker = json.loads((root / "COMPLETED_EVALUATION_RESULT.json").read_text())
if marker.get("shard") != shard or marker.get("source_commit") != source or marker.get("success_gate") != "persisted_completed_evaluation_result":
    raise SystemExit("completion identity/gate mismatch")
if marker.get("phase1_authorized") is not False or marker.get("analysis_performed") is not False:
    raise SystemExit("natural completion attempted analysis/authorization")
names = []
for line in (root / "SHA256SUMS").read_text().splitlines():
    m = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
    if m is None: raise SystemExit("unsafe completion digest line")
    digest, name = m.groups(); rel = PurePosixPath(name)
    if rel.is_absolute() or ".." in rel.parts or "\\" in name or name in {"SHA256SUMS", ""}:
        raise SystemExit("unsafe completion digest path")
    path = root.joinpath(*rel.parts)
    if path.resolve() != path or not path.is_file() or path.is_symlink():
        raise SystemExit(f"missing completion file: {path}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise SystemExit(f"completion digest mismatch: {path}")
    names.append(name)
actual = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and not p.is_symlink() and p.name != "SHA256SUMS" and not p.name.endswith(".tmp"))
if sorted(names) != actual: raise SystemExit("completion SHA256SUMS is not exhaustive")
PY
  then
    echo "STAGE_R_PHASE1R_NATURAL_SHARD_ALREADY_COMPLETE_VALIDATED"
    exit 0
  fi
fi
if [[ -f COMPLETED_EVALUATION_RESULT.json || -f SHA256SUMS ]]; then
  stale_tag="$(date -u +%Y%m%dT%H%M%SZ)_$$"
  for marker in COMPLETED_EVALUATION_RESULT.json SHA256SUMS; do
    [[ -f "$marker" ]] && mv -- "$marker" "runtime/stale_completion_${stale_tag}_$(basename -- "$marker")"
  done
fi

export PYTHONPATH="$ARTIFACT_DIR/frozen_source/src:$QPILOTS:$OPENPI/src"
export LIBERO_CONFIG_PATH=/mnt/cpfs/zbl-cpfs-new/USERS/leon/cache/libero/r16p15-stage1-task64
export LD_LIBRARY_PATH=/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/openpi_py311/lib:${LD_LIBRARY_PATH:-}
export TMPDIR="$ARTIFACT_DIR/runtime/tmp"
export XDG_CACHE_HOME="$ARTIFACT_DIR/runtime/cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70
export OMP_NUM_THREADS=11
export MKL_NUM_THREADS=11
export PAI_PHASE1R_SHARD="$SHARD"

nvidia-smi -L > runtime/gpu_inventory.txt
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv,noheader > runtime/gpu_identity.csv

PROTOCOL_PATH=frozen_source/configs/stage_r_phase1r_protocol.json
SHARDS_PATH=frozen_source/configs/stage_r_phase1r_shards.json
SELECTION_ROOT=frozen_source/results/stage_r/phase1r/selection
CONTROLS_ROOT=frozen_source/results/stage_r/phase1r/controls
CALIBRATION_ROOT=frozen_source/results/stage_r/phase1r/calibration
CALIBRATION_FILE="$CALIBRATION_ROOT/BLINDED_PHASE1R_CALIBRATION.json"
expect_sha256 "$PROTOCOL_PATH" "$PROTOCOL_SHA256"
expect_sha256 "$SHARDS_PATH" "$SHARDS_SHA256"
expect_sha256 "$SELECTION_ROOT/SELECTION_MANIFEST.json" "$SELECTION_MANIFEST_SHA256"
expect_sha256 "$SELECTION_ROOT/SELECTION_SHA256SUMS" "$SELECTION_SHA256SUMS_SHA256"
expect_sha256 "$CALIBRATION_FILE" "$CALIBRATION_SHA256"
expect_sha256 "$CALIBRATION_ROOT/SHA256SUMS" "$CALIBRATION_SHA256SUMS_SHA256"

"$PYTHON" frozen_source/scripts/stage_r_phase1r.py validate-config --protocol "$PROTOCOL_PATH" --shards "$SHARDS_PATH" > runtime/protocol_validation.json
"$PYTHON" frozen_source/scripts/stage_r_phase1r.py validate-selection --root "$SELECTION_ROOT" > runtime/selection_validation.json
# These commands validate already committed controls only; they never collect
# controls or regenerate calibration.
"$PYTHON" frozen_source/scripts/stage_r_phase1r.py validate-controls --root "$CONTROLS_ROOT" --kind positive --owner 2254:2254 > runtime/positive_control_validation.json
"$PYTHON" frozen_source/scripts/stage_r_phase1r.py validate-controls --root "$CONTROLS_ROOT" --kind null --owner 2254:2254 > runtime/null_control_validation.json

"$PYTHON" - "$CALIBRATION_FILE" "$CALIBRATION_ROOT/SHA256SUMS" <<'PY'
import hashlib, json, re, sys
from pathlib import Path, PurePosixPath
calibration, seal = Path(sys.argv[1]), Path(sys.argv[2])
root = calibration.parent
for line in seal.read_text(encoding="utf-8").splitlines():
    if not line.strip(): continue
    match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
    if match is None: raise SystemExit(f"unsafe calibration seal line: {line!r}")
    digest, name = match.groups()
    rel = PurePosixPath(name)
    if rel.is_absolute() or ".." in rel.parts or "\\" in name:
        raise SystemExit(f"unsafe calibration seal path: {name!r}")
    path = root.joinpath(*rel.parts)
    if path.resolve() != path or not path.is_file() or path.is_symlink():
        raise SystemExit(f"missing/noncanonical calibration artifact: {path}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise SystemExit(f"calibration digest mismatch: {path}")
payload = json.loads(calibration.read_text(encoding="utf-8"))
if payload.get("protocol_id") != "r142-stage-r-phase1r-human-override-v1":
    raise SystemExit("calibration protocol mismatch")
if int(payload.get("shuffles", 0)) < 1000:
    raise SystemExit("calibration shuffles below frozen minimum")
if payload.get("unpermuted_curve_present") is not False or payload.get("natural_curve_present") is not False:
    raise SystemExit("calibration is not blinded")
if any(key in payload for key in ("curve", "curves", "natural")):
    raise SystemExit("calibration contains unblinded data")
marker = json.loads((root / "COMPLETED_CALIBRATION.json").read_text(encoding="utf-8"))
if marker.get("protocol_id") != payload["protocol_id"] or marker.get("owner") != "2254:2254":
    raise SystemExit("calibration completion marker drifted")
PY

"$PYTHON" - "$PHASE0_MERGE_ROOT" "$PHASE0_RAW" "$AUTHORITY_MANIFEST_SHA256" <<'PY' > runtime/phase0_authority_validation.json
import hashlib, json, re, sys
from pathlib import Path, PurePosixPath
merge_root, raw_root, expected_sha = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""): h.update(chunk)
    return h.hexdigest()
def check_sums(path, base):
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        m = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if m is None: raise SystemExit(f"unsafe digest line: {line!r}")
        digest, name = m.groups(); rel = PurePosixPath(name)
        if rel.is_absolute() or ".." in rel.parts or "\\" in name or not name:
            raise SystemExit(f"unsafe digest path: {name!r}")
        target = base.joinpath(*rel.parts)
        if target.resolve() != target or not target.is_file() or target.is_symlink():
            raise SystemExit(f"missing/noncanonical file: {target}")
        if sha(target) != digest: raise SystemExit(f"digest mismatch: {target}")
        names.append(name)
    if len(names) != len(set(names)): raise SystemExit(f"duplicate path in {path}")
    return names
authority = merge_root / "AUTHORITY_MANIFEST.json"
if sha(authority) != expected_sha: raise SystemExit("authority manifest SHA mismatch")
a = json.loads(authority.read_text(encoding="utf-8"))
if a.get("protocol_id") != "r142-stage-r-phase0r-v1" or a.get("outcome_selection_permitted") is not False:
    raise SystemExit("authority protocol/outcome contract drifted")
if a.get("authority_rule") != "parent[0:32]+shard_A[32:36]+shard_B[36:40]":
    raise SystemExit("authority range rule drifted")
records = a.get("records")
if not isinstance(records, list) or [r.get("index") for r in records] != list(range(40)):
    raise SystemExit("authority records do not cover indices 0..39")
for r in records[:32]:
    if r.get("source") != "parent": raise SystemExit("parent authority source drift")
for r in records[32:36]:
    if r.get("source") != "shard_a": raise SystemExit("shard A authority source drift")
for r in records[36:]:
    if r.get("source") != "shard_b": raise SystemExit("shard B authority source drift")
raw_marker = json.loads((merge_root / "COMPLETED_PHASE0R_RAW.json").read_text(encoding="utf-8"))
if raw_marker.get("protocol_id") != "r142-stage-r-phase0r-v1" or raw_marker.get("task_count") != 40 or raw_marker.get("rollout_count") != 20480:
    raise SystemExit("raw completion marker cardinality/protocol drift")
if raw_marker.get("outcomes_unblinded") is not False or raw_marker.get("authority_manifest_sha256") != expected_sha:
    raise SystemExit("raw completion marker unblinding/authority drift")
names = check_sums(merge_root / "MERGE_SHA256SUMS", merge_root)
npz = sorted(p.name for p in raw_root.glob("*.npz"))
meta = sorted(p.name for p in raw_root.glob("*.json"))
if len(npz) != 40 or len(meta) != 40: raise SystemExit("raw input must contain exactly 40 NPZ and 40 metadata files")
expected_raw = sorted("raw/" + n for n in npz + meta)
if sorted(n for n in names if n.startswith("raw/")) != expected_raw:
    raise SystemExit("merge seal raw file set drifted")
print(json.dumps({"valid": True, "authority_records": 40, "raw_tasks": 40, "raw_rollouts": 20480, "raw_files": 80}, sort_keys=True))
PY

"$PYTHON" - "$CHECKPOINT" "$CHECKPOINT_TREE_SHA" "$CHECKPOINT_ATTESTATION" "$CHECKPOINT_ATTESTATION_SHA256" <<'PY' > runtime/checkpoint_validation.json
import hashlib, json, os, sys
from pathlib import Path
root, expected_tree, attestation_path, expected_attestation = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4]
attestation_bytes = attestation_path.read_bytes()
if hashlib.sha256(attestation_bytes).hexdigest() != expected_attestation:
    raise SystemExit("checkpoint attestation SHA mismatch")
attestation = json.loads(attestation_bytes)
expected_header = {
    "schema_version": 1,
    "marker_type": "full_content_checkpoint_attestation",
    "checkpoint": str(root),
    "tree_sha256": expected_tree,
    "file_count": 16,
    "bytes": 12439085481,
    "probe_scheme": "sha256_first_and_last_1MiB_plus_full_file_metadata",
    "uid": 2254,
    "gid": 2254,
}
if any(attestation.get(key) != value for key, value in expected_header.items()):
    raise SystemExit("checkpoint attestation header mismatch")
rows = attestation.get("files")
if not isinstance(rows, list) or len(rows) != 16:
    raise SystemExit("checkpoint attestation file inventory mismatch")
expected_paths = [str(row.get("path", "")) for row in rows]
actual_paths = []
for path in sorted(root.rglob("*")):
    if path.is_symlink(): raise SystemExit(f"checkpoint contains symlink: {path}")
    if path.is_file(): actual_paths.append(path.relative_to(root).as_posix())
if actual_paths != expected_paths:
    raise SystemExit("checkpoint file inventory drifted from full-content attestation")
probe_bytes = 1024 * 1024
for row in rows:
    path = root / row["path"]
    metadata = path.stat()
    observed_metadata = (metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns)
    expected_metadata = (row.get("size"), row.get("mtime_ns"), row.get("ctime_ns"))
    if observed_metadata != expected_metadata:
        raise SystemExit(f"checkpoint metadata drifted: {path}")
    with path.open("rb") as handle:
        head = handle.read(probe_bytes)
        if metadata.st_size > probe_bytes:
            handle.seek(max(0, metadata.st_size - probe_bytes))
            tail = handle.read(probe_bytes)
        else:
            tail = head
    if hashlib.sha256(head).hexdigest() != row.get("head_sha256") or hashlib.sha256(tail).hexdigest() != row.get("tail_sha256"):
        raise SystemExit(f"checkpoint probe digest drifted: {path}")
    if metadata.st_size <= 2 * probe_bytes and hashlib.sha256(path.read_bytes()).hexdigest() != row.get("sha256"):
        raise SystemExit(f"checkpoint small-file full digest drifted: {path}")
print(json.dumps({"valid": True, "validation_mode": "frozen_full_content_attestation_plus_exact_metadata_and_content_probes", "file_count": len(rows), "bytes": attestation["bytes"], "tree_sha256": expected_tree, "attestation_sha256": expected_attestation, "uid": os.getuid(), "gid": os.getgid()}, sort_keys=True))
PY

"$PYTHON" - "$SHARDS_PATH" "$EXECUTION_CONFIG" "$EXECUTION_SHARD" "$SELECTION_ROOT" runtime/task_mapping.tsv <<'PY'
import json, re, sys
from pathlib import Path
shards = json.loads(Path(sys.argv[1]).read_text())
execution = json.loads(Path(sys.argv[2]).read_text())
execution_shard, selection_root, out = sys.argv[3], Path(sys.argv[4]), Path(sys.argv[5])
execution_entry = execution["execution_shards"].get(execution_shard)
if not isinstance(execution_entry, dict): raise SystemExit(f"missing execution shard {execution_shard}")
shard = execution_entry.get("logical_shard")
payload = shards["shards"].get(shard)
if not isinstance(payload, dict): raise SystemExit(f"missing shard {shard}")
global_ranks = [int(v) for v in execution_entry.get("global_ranks", [])]
expected_ranks = {"A0": list(range(0, 4)), "A1": list(range(4, 8)), "B0": list(range(8, 12)), "B1": list(range(12, 16))}[execution_shard]
if global_ranks != expected_ranks: raise SystemExit(f"shard global ranks drifted: {global_ranks}")
rank_tasks = payload.get("rank_tasks")
if not isinstance(rank_tasks, dict): raise SystemExit("rank_tasks missing")
lines = ["local_rank\tglobal_rank\ttask_name\tsuite\ttask_id\tselection_path"]
seen = set()
for local_rank, global_rank in enumerate(global_ranks):
    names = rank_tasks.get(str(global_rank))
    if not isinstance(names, list) or not names: raise SystemExit(f"rank {global_rank} has no fixed tasks")
    for task_name in names:
        m = re.fullmatch(r"(libero_(?:spatial|object|goal|10))_task(\d{2})", str(task_name))
        if m is None or task_name in seen: raise SystemExit(f"invalid/duplicate task {task_name!r}")
        seen.add(task_name); suite, task_text = m.groups(); task_id = int(task_text)
        selection = selection_root / f"{task_name}.json"
        if not selection.is_file() or selection.is_symlink(): raise SystemExit(f"missing selection {selection}")
        lines.append(f"{local_rank}\t{global_rank}\t{task_name}\t{suite}\t{task_id}\t{selection}")
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
expected_count = {"A0": 7, "A1": 10, "B0": 11, "B1": 12}[execution_shard]
if len(seen) != expected_count: raise SystemExit(f"shard task count drifted: {len(seen)}")
PY

# Resume marker handling is fail-closed: preserve invalid markers as evidence.
if [[ -f COMPLETED_EVALUATION_RESULT.json && -f SHA256SUMS ]]; then
  if "$PYTHON" - "$ARTIFACT_DIR" "$SHARD" "$EXECUTION_SHARD" "$SOURCE_COMMIT" <<'PY'
import hashlib, json, re, sys
from pathlib import Path, PurePosixPath
root, shard, execution_shard, source = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
marker = json.loads((root / "COMPLETED_EVALUATION_RESULT.json").read_text())
if marker.get("shard") != shard or marker.get("execution_shard") != execution_shard or marker.get("source_commit") != source or marker.get("success_gate") != "persisted_completed_evaluation_result":
    raise SystemExit("completion identity/gate mismatch")
if marker.get("phase1_authorized") is not False or marker.get("analysis_performed") is not False:
    raise SystemExit("natural completion attempted analysis/authorization")
names = []
for line in (root / "SHA256SUMS").read_text().splitlines():
    m = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
    if m is None: raise SystemExit("unsafe completion digest line")
    digest, name = m.groups(); rel = PurePosixPath(name)
    if rel.is_absolute() or ".." in rel.parts or "\\" in name or name in {"SHA256SUMS", ""}:
        raise SystemExit("unsafe completion digest path")
    path = root.joinpath(*rel.parts)
    if path.resolve() != path or not path.is_file() or path.is_symlink():
        raise SystemExit(f"missing completion file: {path}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise SystemExit(f"completion digest mismatch: {path}")
    names.append(name)
actual = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and not p.is_symlink() and p.name != "SHA256SUMS" and not p.name.endswith(".tmp"))
if sorted(names) != actual: raise SystemExit("completion SHA256SUMS is not exhaustive")
print("STAGE_R_PHASE1R_NATURAL_SHARD_ALREADY_COMPLETE_VALIDATED")
PY
  then
    exit 0
  fi
fi
if [[ -f COMPLETED_EVALUATION_RESULT.json || -f SHA256SUMS ]]; then
  stale_tag="$(date -u +%Y%m%dT%H%M%SZ)_$$"
  for marker in COMPLETED_EVALUATION_RESULT.json SHA256SUMS; do
    [[ -f "$marker" ]] && mv -- "$marker" "runtime/stale_completion_${stale_tag}_$(basename -- "$marker")"
  done
fi

first_work_once() {
  local local_rank="$1" global_rank="$2" task_name="$3"
  "$PYTHON" - "$ARTIFACT_DIR/FIRST_WORK.json" "$SOURCE_COMMIT" "$SHARD" "$EXECUTION_SHARD" "$local_rank" "$global_rank" "$task_name" <<'PY'
import json, os, sys
from pathlib import Path
path = Path(sys.argv[1])
payload = {"schema_version": 1, "milestone": "first_valid_natural_task_cell", "source_commit": sys.argv[2], "shard": sys.argv[3], "execution_shard": sys.argv[4], "local_rank": int(sys.argv[5]), "global_rank": int(sys.argv[6]), "task_name": sys.argv[7], "completed_cells": 240, "streams": ["calibration", "heldout"], "uid": os.getuid(), "gid": os.getgid()}
try:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
except FileExistsError:
    raise SystemExit(0)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
PY
}

validate_task() {
  local task_local="$1" task_global="$2" task_name="$3" suite="$4" task_id="$5" selection="$6"
  "$PYTHON" - "$ARTIFACT_DIR" "$suite" "$task_id" "$selection" "$ARTIFACT_DIR/runtime/task_validation.globalrank${task_global}.${task_name}.json" <<'PY'
import json, os, sys
from pathlib import Path
from r142_stage_r.phase1 import validate_task_completion_marker
root, suite, task_id, selection, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], Path(sys.argv[5])
ok, errors, marker = validate_task_completion_marker(root + "/natural", suite, task_id, selection, require_owner=(2254, 2254))
out.write_text(json.dumps({"valid": bool(ok), "errors": errors, "marker": marker, "uid": os.getuid(), "gid": os.getgid()}, indent=2, sort_keys=True) + "\n")
if not ok: raise SystemExit("task completion validation failed: " + "; ".join(errors))
PY
}

run_rank() {
  local local_rank="$1" global_rank="$2" task_name suite task_id selection
  local visible_devices
  if [[ "$local_rank" == 0 ]]; then
    visible_devices=0
  else
    # robosuite resolves MUJOCO_EGL_DEVICE_ID against the CUDA-visible list.
    # Keep the rank-local GPU first (for the model) and expose physical GPU 0
    # as the EGL fallback required by the pinned Phase-0R pattern.
    visible_devices="$local_rank,0"
  fi
  while IFS=$'\t' read -r row_local row_global task_name suite task_id selection; do
    [[ "$row_local" == "$local_rank" && "$row_global" == "$global_rank" ]] || continue
    echo "START_NATURAL_TASK local_rank=$local_rank global_rank=$global_rank task=$task_name" >&2
    CUDA_VISIBLE_DEVICES="$visible_devices" EGL_DEVICE_ID=0 MUJOCO_EGL_DEVICE_ID=0 \
      "$PYTHON" frozen_source/scripts/stage_r_phase1r.py collect-natural \
      --raw "$PHASE0_RAW" --selection "$selection" --output "$ARTIFACT_DIR/natural" \
      --suite "$suite" --task-id "$task_id" --qpilots-root "$QPILOTS" --libero-root "$LIBERO" \
      --checkpoint "$CHECKPOINT" --microbatch 8 --max-steps 1000 \
      > "runtime/logs/task.globalrank${global_rank}.${task_name}.stdout.log" 2>&1
    validate_task "$local_rank" "$global_rank" "$task_name" "$suite" "$task_id" "$selection"
    first_work_once "$local_rank" "$global_rank" "$task_name"
    echo "COMPLETE_NATURAL_TASK local_rank=$local_rank global_rank=$global_rank task=$task_name" >&2
  done < runtime/task_mapping.tsv
}

pids=()
declare -A started
while IFS=$'\t' read -r local_rank global_rank task_name suite task_id selection; do
  [[ "$local_rank" == local_rank ]] && continue
  if [[ -z "${started[$local_rank]:-}" ]]; then
    started["$local_rank"]=1
    run_rank "$local_rank" "$global_rank" > "runtime/logs/rank${global_rank}.launcher.log" 2>&1 &
    pids+=("$!")
  fi
done < runtime/task_mapping.tsv
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
[[ "$failed" == 0 ]] || { echo "one or more natural rank workers failed" >&2; exit 80; }

"$PYTHON" - "$ARTIFACT_DIR" "$SHARD" "$EXECUTION_SHARD" "$SOURCE_COMMIT" "$PROTOCOL_SHA256" "$SHARDS_SHA256" "$EXECUTION_CONFIG_SHA256" "$SELECTION_MANIFEST_SHA256" "$CALIBRATION_SHA256" "$AUTHORITY_MANIFEST_SHA256" <<'PY'
import json, os, sys
from pathlib import Path
from r142_stage_r.phase1 import validate_task_completion_marker
root, shard, execution_shard = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
source_commit, protocol_sha, shards_sha, execution_sha, selection_sha, calibration_sha, authority_sha = sys.argv[4:]
mapping = []
for line in (root / "runtime/task_mapping.tsv").read_text().splitlines()[1:]:
    local_rank, global_rank, task_name, suite, task_id, selection = line.split("\t")
    task_id_int = int(task_id)
    ok, errors, marker = validate_task_completion_marker(root / "natural", suite, task_id_int, selection, require_owner=(2254, 2254))
    if not ok: raise SystemExit(f"final task validation failed for {task_name}: {'; '.join(errors)}")
    mapping.append({"local_rank": int(local_rank), "global_rank": int(global_rank), "task_name": task_name, "suite": suite, "task_id": task_id_int, "selection": Path(selection).name, "completed_cells": int(marker["completed_cells"])})
expected_cells, expected_descendants = len(mapping) * 12 * 10 * 2, len(mapping) * 12 * 10 * 16
payload = {"schema_version": 1, "marker_type": "natural_phase1r_execution_shard", "protocol_id": "r142-stage-r-phase1r-human-override-v1", "source_commit": source_commit, "protocol_sha256": protocol_sha, "shards_sha256": shards_sha, "execution_config_sha256": execution_sha, "selection_manifest_sha256": selection_sha, "calibration_sha256": calibration_sha, "authority_manifest_sha256": authority_sha, "shard": shard, "execution_shard": execution_shard, "global_ranks": sorted({row["global_rank"] for row in mapping}), "task_names": [row["task_name"] for row in mapping], "mapping": mapping, "streams": ["calibration", "heldout"], "natural_cells": expected_cells, "natural_descendants_per_stream": expected_descendants, "uid": os.getuid(), "gid": os.getgid(), "checkpoint": "EXECUTION_SHARD_NATURAL_COMPLETE"}
temporary = root / "runtime/.SHARD_NATURAL_COMPLETE.json.tmp"; temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n"); temporary.replace(root / "SHARD_NATURAL_COMPLETE.json")
completed = {"schema_version": 1, "success_gate": "persisted_completed_evaluation_result", "decision": "NATURAL_EXECUTION_SHARD_COMPLETE_NO_UNBLINDING", "checkpoint": "CHECKPOINT_1_PENDING_GLOBAL_MERGE", "phase1_authorized": False, "analysis_performed": False, "protocol_id": "r142-stage-r-phase1r-human-override-v1", "source_commit": source_commit, "protocol_sha256": protocol_sha, "shards_sha256": shards_sha, "execution_config_sha256": execution_sha, "selection_manifest_sha256": selection_sha, "calibration_sha256": calibration_sha, "authority_manifest_sha256": authority_sha, "shard": shard, "execution_shard": execution_shard, "task_names": [row["task_name"] for row in mapping], "task_count": len(mapping), "natural_cells": expected_cells, "natural_descendants_per_stream": expected_descendants, "streams": ["calibration", "heldout"], "uid": os.getuid(), "gid": os.getgid()}
temporary = root / "runtime/.COMPLETED_EVALUATION_RESULT.json.tmp"; temporary.write_text(json.dumps(completed, indent=2, sort_keys=True) + "\n"); temporary.replace(root / "COMPLETED_EVALUATION_RESULT.json")
PY

"$PYTHON" - "$ARTIFACT_DIR" <<'PY'
import hashlib, os, sys
from pathlib import Path
root = Path(sys.argv[1])
lines = []
for path in sorted(root.rglob("*")):
    if not path.is_file() or path.is_symlink() or path.name == "SHA256SUMS" or path.name.endswith(".tmp"): continue
    lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root).as_posix()}\n")
temporary = root / ".SHA256SUMS.tmp"; temporary.write_text("".join(lines), encoding="utf-8"); os.chmod(temporary, 0o600); temporary.replace(root / "SHA256SUMS")
PY
sync -f "$ARTIFACT_DIR/FIRST_WORK.json"
sync -f "$ARTIFACT_DIR/SHARD_NATURAL_COMPLETE.json"
sync -f "$ARTIFACT_DIR/COMPLETED_EVALUATION_RESULT.json"
sync -f "$ARTIFACT_DIR/SHA256SUMS"

"$PYTHON" - "$ARTIFACT_DIR" "$SHARD" "$EXECUTION_SHARD" "$SOURCE_COMMIT" <<'PY'
import hashlib, json, re, sys
from pathlib import Path, PurePosixPath
root, shard, execution_shard, source = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
marker = json.loads((root / "COMPLETED_EVALUATION_RESULT.json").read_text())
if marker.get("shard") != shard or marker.get("execution_shard") != execution_shard or marker.get("source_commit") != source or marker.get("success_gate") != "persisted_completed_evaluation_result":
    raise SystemExit("completion marker identity/gate mismatch")
if marker.get("phase1_authorized") is not False or marker.get("analysis_performed") is not False:
    raise SystemExit("natural marker attempted analysis/authorization")
names = []
for line in (root / "SHA256SUMS").read_text().splitlines():
    m = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
    if m is None: raise SystemExit("unsafe completion digest line")
    digest, name = m.groups(); rel = PurePosixPath(name)
    if rel.is_absolute() or ".." in rel.parts or "\\" in name or name in {"SHA256SUMS", ""}: raise SystemExit("unsafe completion digest path")
    path = root.joinpath(*rel.parts)
    if path.resolve() != path or not path.is_file() or path.is_symlink(): raise SystemExit(f"missing completion file: {path}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest: raise SystemExit(f"completion digest mismatch: {path}")
    names.append(name)
actual = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and not p.is_symlink() and p.name != "SHA256SUMS" and not p.name.endswith(".tmp"))
if sorted(names) != actual: raise SystemExit("completion SHA256SUMS is not exhaustive")
print("STAGE_R_PHASE1R_NATURAL_SHARD_COMPLETE_VALIDATED")
PY
echo "STAGE_R_PHASE1R_NATURAL_EXECUTION_SHARD_COMPLETE_NO_UNBLINDING execution_shard=$EXECUTION_SHARD logical_shard=$SHARD checkpoint=CHECKPOINT_1_PENDING_GLOBAL_MERGE"
