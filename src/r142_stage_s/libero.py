"""Stage-S LIBERO substrate adapters and evidence-safe collection helpers.

This module deliberately contains no synthetic rollout path.  The B adapter
changes the simulator task (BDDL plus regenerated initial states), while the
C adapter only accepts complete, real pi05-LIBERO checkpoints.  The collection
helpers are adapter based so unit tests can use small fakes without importing
MuJoCo, JAX, or the PAI runtime.

The public functions are intentionally conservative:

* calibration returns and persists pooled counters only;
* a main-screen family is committed only after all data, genealogy, snapshot,
  metadata, and SHA files have been written and verified;
* restoring a Stage-R snapshot is checked by executing the same action twice
  and requiring an absolute next-state error of at most 1e-9.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import os
import pickle
import random
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


STAGE_S_PROTOCOL_ID = "r142-stage-s-v1"
LIBERO_SUITE = "libero_10"
TASK_COUNT = 10
MAIN_INITIAL_STATE_COUNT = 16
MAIN_CANDIDATE_COUNT = 32
CALIBRATION_TASK_IDS = (0, 1, 2, 3)
CALIBRATION_INITIAL_STATES = tuple(range(8))
CALIBRATION_CANDIDATE_COUNT = 8
CALIBRATION_TARGET_SUCCESS = 0.45
MAIN_ACTION_HORIZON = 10
REPLAN_STEPS = 5
SNAPSHOT_REPLAY_TOLERANCE = 1e-9
FULL_PI05_LIBERO_TRAINING_STEPS = 60_000
UNDERTRAINED_CHECKPOINT_COUNT = 4

# The four B settings are measured in the table's XY coordinate system.  They
# are frozen here rather than selected after observing any S2--S5 statistic.
# The minimum 0.06 m keeps a same-mesh duplicate from being intentionally
# interpenetrating at reset; all settings remain a local referential cue.
PROXIMITY_MAGNITUDES = (0.06, 0.08, 0.10, 0.12)


LIBERO_10_TASK_NAMES = (
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
)


@dataclass(frozen=True)
class LiberoTaskSpec:
    """Frozen Stage-R task identity and the target used by B."""

    task_id: int
    name: str
    prompt: str
    target_object: str
    target_region: str


LIBERO_TASK_SPECS = (
    LiberoTaskSpec(0, LIBERO_10_TASK_NAMES[0], "put both the alphabet soup and the tomato sauce in the basket", "alphabet_soup_1", "alphabet_soup_init_region"),
    LiberoTaskSpec(1, LIBERO_10_TASK_NAMES[1], "put both the cream cheese box and the butter in the basket", "cream_cheese_1", "cream_cheese_init_region"),
    LiberoTaskSpec(2, LIBERO_10_TASK_NAMES[2], "turn on the stove and put the moka pot on it", "moka_pot_1", "moka_pot_init_region"),
    LiberoTaskSpec(3, LIBERO_10_TASK_NAMES[3], "put the black bowl in the bottom drawer of the cabinet and close it", "akita_black_bowl_1", "akita_black_bowl_init_region"),
    LiberoTaskSpec(4, LIBERO_10_TASK_NAMES[4], "put the white mug on the left plate and put the yellow and white mug on the right plate", "porcelain_mug_1", "porcelain_mug_init_region"),
    LiberoTaskSpec(5, LIBERO_10_TASK_NAMES[5], "pick up the book and place it in the back compartment of the caddy", "black_book_1", "black_book_init_region"),
    LiberoTaskSpec(6, LIBERO_10_TASK_NAMES[6], "put the white mug on the plate and put the chocolate pudding to the right of the plate", "porcelain_mug_1", "porcelain_mug_init_region"),
    LiberoTaskSpec(7, LIBERO_10_TASK_NAMES[7], "put both the alphabet soup and the cream cheese box in the basket", "alphabet_soup_1", "alphabet_soup_init_region"),
    LiberoTaskSpec(8, LIBERO_10_TASK_NAMES[8], "put both moka pots on the stove", "moka_pot_1", "moka_pot_right_init_region"),
    LiberoTaskSpec(9, LIBERO_10_TASK_NAMES[9], "put the yellow and white mug in the microwave and close it", "white_yellow_mug_1", "white_yellow_mug_init_region"),
)


class StageSError(RuntimeError):
    """Base class for fail-closed Stage-S errors."""


class VariantGenerationError(StageSError):
    pass


class MissingRegeneratedInitialStates(StageSError):
    pass


class CheckpointQualificationError(StageSError):
    pass


class SnapshotReplayError(StageSError):
    pass


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def stable_seed(*parts: object, bytes_count: int = 8) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:bytes_count], "big", signed=False)


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_bytes(path: str | Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)
    _fsync_directory(destination.parent)


def atomic_json(path: str | Path, payload: Any) -> None:
    atomic_bytes(path, (json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


def atomic_text(path: str | Path, payload: str) -> None:
    atomic_bytes(path, payload.encode("utf-8"))


def atomic_npz(path: str | Path, payload: Mapping[str, np.ndarray]) -> None:
    import io

    buffer = io.BytesIO()
    np.savez_compressed(buffer, **payload)
    atomic_bytes(path, buffer.getvalue())


def _find_block(text: str, name: str) -> tuple[int, int]:
    """Find a balanced BDDL form whose first symbol is ``name``."""

    pattern = re.compile(rf"(?m)^[ \t]*\({re.escape(name)}(?=[\s)])")
    match = pattern.search(text)
    if match is None:
        raise VariantGenerationError(f"BDDL form {name!r} not found")
    start = match.start() + len(match.group(0)) - len(match.group(0).lstrip())
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return start, index + 1
    raise VariantGenerationError(f"unterminated BDDL form {name!r}")


def _insert_before_form_close(text: str, block: tuple[int, int], line: str) -> str:
    start, end = block
    prefix = "" if text[: end - 1].endswith("\n") else "\n"
    return text[: end - 1] + prefix + line.rstrip("\n") + "\n" + text[end - 1 :]


def _object_type(objects_form: str, target_object: str) -> str:
    for line in objects_form.splitlines():
        if "-" not in line:
            continue
        lhs, rhs = line.split("-", 1)
        if target_object in lhs.split():
            values = rhs.strip().split()
            if values:
                return values[0]
    raise VariantGenerationError(f"target object {target_object!r} absent from (:objects)")


def _region_range(region_form: str) -> tuple[float, float, float, float]:
    ranges = re.search(
        r"\(:ranges\s*\(\s*\(\s*([^()]+?)\s*\)\s*\)",
        region_form,
        flags=re.DOTALL,
    )
    if ranges is None:
        raise VariantGenerationError("target region has no one-rectangle (:ranges) form")
    values = [float(value) for value in ranges.group(1).split()]
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise VariantGenerationError(f"invalid target region range: {values!r}")
    return tuple(values)  # type: ignore[return-value]


def _format_range(values: Sequence[float]) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def _range_form(name: str, target: str, values: Sequence[float]) -> str:
    return (
        f"      ({name}\n"
        f"          (:target {target})\n"
        f"          (:ranges (\n"
        f"              ({_format_range(values)})\n"
        f"            )\n"
        f"          )\n"
        f"          (:yaw_rotation (\n"
        f"              (0.0 0.0)\n"
        f"            )\n"
        f"          )\n"
        f"      )"
    )


@dataclass(frozen=True)
class BVariant:
    """One deterministic BDDL variant for one task and magnitude."""

    task: LiberoTaskSpec
    magnitude: float
    source_bddl_path: str
    source_bddl_sha256: str
    bddl_text: str
    target_type: str
    distractor_object: str
    distractor_region: str
    target_center: tuple[float, float]
    distractor_center: tuple[float, float]
    offset_direction: tuple[float, float]
    prompt_unchanged: bool = True

    @property
    def setting_id(self) -> str:
        return f"proximity_{self.magnitude:.2f}m"

    def manifest(self) -> dict[str, Any]:
        return {
            "protocol_id": STAGE_S_PROTOCOL_ID,
            "substrate": "B",
            "suite": LIBERO_SUITE,
            "task_id": self.task.task_id,
            "task_name": self.task.name,
            "prompt": self.task.prompt,
            "prompt_unchanged": self.prompt_unchanged,
            "magnitude_m": self.magnitude,
            "target_object": self.task.target_object,
            "target_type": self.target_type,
            "distractor_object": self.distractor_object,
            "distractor_region": self.distractor_region,
            "target_center": list(self.target_center),
            "distractor_center": list(self.distractor_center),
            "offset_direction": list(self.offset_direction),
            "source_bddl_path": self.source_bddl_path,
            "source_bddl_sha256": self.source_bddl_sha256,
            "bddl_sha256": sha256_bytes(self.bddl_text.encode("utf-8")),
            "initial_states_contract": "regenerated_variant_qpos_required; original init qpos is incompatible",
        }


def task_spec(task_id: int) -> LiberoTaskSpec:
    value = int(task_id)
    if not 0 <= value < TASK_COUNT:
        raise ValueError(f"task_id must be in [0,{TASK_COUNT}), got {task_id!r}")
    return LIBERO_TASK_SPECS[value]


def generate_b_variant(
    source_bddl_path: str | Path,
    task_id: int,
    magnitude: float,
    *,
    offset_direction: tuple[float, float] | None = None,
) -> BVariant:
    """Generate an exact-one same-type duplicate without changing the goal.

    The insertion is purely textual and preserves the original ``(:language)``
    and ``(:goal)`` forms.  The duplicate is not put in ``obj_of_interest`` or
    ``goal``; it is therefore a visually identical referential distractor, not
    a second goal object.
    """

    magnitude = float(magnitude)
    if not any(math.isclose(magnitude, value, rel_tol=0.0, abs_tol=1e-12) for value in PROXIMITY_MAGNITUDES):
        raise VariantGenerationError(f"magnitude {magnitude!r} is not one of frozen settings {PROXIMITY_MAGNITUDES}")
    task = task_spec(task_id)
    path = Path(source_bddl_path)
    source = path.read_text(encoding="utf-8")
    source_sha = sha256_file(path)
    language = re.search(r"\(:language\s+([^\n)]+)\)", source)
    if language is None or language.group(1).strip() != task.prompt:
        raise VariantGenerationError(f"prompt mismatch in {path}: expected {task.prompt!r}")
    objects_start, objects_end = _find_block(source, ":objects")
    # _find_block expects a symbol that starts after '('; sections with a
    # colon use a direct fallback because BDDL names the form (:objects ...).
    objects_form = source[objects_start:objects_end]
    target_type = _object_type(objects_form, task.target_object)
    region_start, region_end = _find_block(source, task.target_region)
    target_range = _region_range(source[region_start:region_end])
    regions_start, regions_end = _find_block(source, ":regions")
    region_targets = re.findall(
        r"\(:target\s+([^\s)]+)\)", source[regions_start:regions_end]
    )
    table_target = next(
        (value for value in region_targets if value.endswith("_table")), None
    )
    if table_target is None:
        raise VariantGenerationError("could not identify the task table fixture in (:regions)")
    target_center = ((target_range[0] + target_range[2]) / 2.0, (target_range[1] + target_range[3]) / 2.0)
    direction = offset_direction or ((1.0, 0.0) if task_id % 2 == 0 else (0.0, 1.0))
    norm = math.hypot(*direction)
    if not math.isfinite(norm) or norm <= 0.0:
        raise VariantGenerationError("offset_direction must be finite and nonzero")
    direction = (float(direction[0] / norm), float(direction[1] / norm))
    distractor_center = (
        target_center[0] + magnitude * direction[0],
        target_center[1] + magnitude * direction[1],
    )
    width = target_range[2] - target_range[0]
    height = target_range[3] - target_range[1]
    duplicate_range = (
        distractor_center[0] - width / 2.0,
        distractor_center[1] - height / 2.0,
        distractor_center[0] + width / 2.0,
        distractor_center[1] + height / 2.0,
    )
    base_object = task.target_object.removesuffix("_1")
    distractor_object = f"{base_object}_distractor_1"
    distractor_region = f"{base_object}_distractor_init_region"
    if re.search(rf"\b{re.escape(distractor_object)}\b", source) or re.search(rf"\b{re.escape(distractor_region)}\b", source):
        raise VariantGenerationError("source already contains the reserved distractor names")
    object_line = f"    {distractor_object} - {target_type}"
    source = _insert_before_form_close(source, _find_block(source, ":objects"), object_line)
    # Re-find blocks after the object insertion because offsets changed.
    source = _insert_before_form_close(
        source,
        _find_block(source, ":regions"),
        _range_form(distractor_region, table_target, duplicate_range),
    )
    source = _insert_before_form_close(source, _find_block(source, ":init"), f"    (On {distractor_object} living_room_table_{distractor_region})")
    # Exact one duplicate: one declaration, one placement, zero goal mentions.
    if source.count(distractor_object) != 2 or distractor_object in source[source.find("(:goal") :]:
        raise VariantGenerationError("generated BDDL does not contain exactly one non-goal distractor")
    return BVariant(
        task=task,
        magnitude=magnitude,
        source_bddl_path=str(path),
        source_bddl_sha256=source_sha,
        bddl_text=source,
        target_type=target_type,
        distractor_object=distractor_object,
        distractor_region=distractor_region,
        target_center=target_center,
        distractor_center=distractor_center,
        offset_direction=direction,
    )


def _candidate_state_file(root: Path, task: LiberoTaskSpec) -> Path:
    return root / f"{task.name}.pruned_init"


def validate_regenerated_initial_states(
    regenerated_root: str | Path,
    source_init_root: str | Path,
    tasks: Iterable[int] = range(TASK_COUNT),
    *,
    manifest_name: str = "REGENERATED_INIT_STATES.json",
) -> dict[str, Any]:
    """Require an explicit regenerated-qpos manifest before a B run.

    A BDDL edit changes MuJoCo's free-joint layout.  Copying the old tensor is
    invalid even if its first elements happen to load.  The manifest is an
    auditable hand-off from a real simulator regeneration job and must assert
    that old qpos was not reused.
    """

    regenerated = Path(regenerated_root)
    source = Path(source_init_root)
    marker = regenerated / manifest_name
    errors: list[str] = []
    if not marker.is_file():
        errors.append(f"missing {manifest_name}; old init qpos cannot be accepted")
        return {"valid": False, "errors": errors, "files": []}
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"valid": False, "errors": [f"invalid regenerated-state manifest: {exc}"], "files": []}
    if payload.get("regenerated") is not True or payload.get("old_init_reused") is not False:
        errors.append("manifest must assert regenerated=true and old_init_reused=false")
    files: list[dict[str, Any]] = []
    for task_id in tasks:
        task = task_spec(task_id)
        path = _candidate_state_file(regenerated, task)
        old = _candidate_state_file(source, task)
        if not path.is_file():
            errors.append(f"missing regenerated init file: {path}")
            continue
        new_sha = sha256_file(path)
        old_sha = sha256_file(old) if old.is_file() else None
        if old_sha is not None and new_sha == old_sha:
            errors.append(f"regenerated init file is byte-identical to old qpos: {path.name}")
        files.append({"task_id": int(task_id), "path": str(path), "sha256": new_sha, "old_sha256": old_sha})
    return {"valid": not errors, "errors": errors, "files": files, "manifest": payload}


def write_b_variant(
    variant: BVariant,
    output_root: str | Path,
    *,
    regenerated_initial_states_root: str | Path | None,
    source_init_root: str | Path,
    write_config: bool = True,
    assets_root: str | Path | None = None,
) -> dict[str, Any]:
    """Write one B variant and its *separately regenerated* init states."""

    if regenerated_initial_states_root is None:
        raise MissingRegeneratedInitialStates("B requires regenerated initial qpos; no source-only fallback is permitted")
    audit = validate_regenerated_initial_states(
        regenerated_initial_states_root,
        source_init_root,
        tasks=(variant.task.task_id,),
    )
    if not audit["valid"]:
        raise MissingRegeneratedInitialStates("; ".join(audit["errors"]))
    root = Path(output_root)
    bddl_dir = root / "bddl_files" / LIBERO_SUITE
    init_dir = root / "init_states" / LIBERO_SUITE
    bddl_dir.mkdir(parents=True, exist_ok=True)
    init_dir.mkdir(parents=True, exist_ok=True)
    bddl_path = bddl_dir / f"{variant.task.name}.bddl"
    init_source = _candidate_state_file(Path(regenerated_initial_states_root), variant.task)
    init_path = init_dir / init_source.name
    atomic_text(bddl_path, variant.bddl_text)
    atomic_bytes(init_path, init_source.read_bytes())
    if write_config:
        config = {
            "benchmark_root": str(root.resolve()),
            "bddl_files": str((root / "bddl_files").resolve()),
            "init_states": str((root / "init_states").resolve()),
            "datasets": str((root / "datasets").resolve()),
            "assets": str(Path(assets_root).resolve()) if assets_root else "",
        }
        atomic_text(root / "config.yaml", "\n".join(f"{key}: {value}" for key, value in config.items()) + "\n")
    result = variant.manifest()
    result.update({"bddl_path": str(bddl_path), "init_states_path": str(init_path), "init_states_sha256": sha256_file(init_path)})
    atomic_json(root / f"{variant.task.name}.{variant.setting_id}.json", result)
    return result


def build_b_variant_suite(
    source_bddl_root: str | Path,
    output_root: str | Path,
    magnitude: float,
    *,
    regenerated_initial_states_root: str | Path | None,
    source_init_root: str | Path,
    assets_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Build all ten task files, failing before writes if qpos is absent."""

    if regenerated_initial_states_root is None:
        raise MissingRegeneratedInitialStates("B suite build requires regenerated init states for every task")
    audit = validate_regenerated_initial_states(regenerated_initial_states_root, source_init_root)
    if not audit["valid"]:
        raise MissingRegeneratedInitialStates("; ".join(audit["errors"]))
    source_root = Path(source_bddl_root)
    results = []
    for task_id in range(TASK_COUNT):
        task = task_spec(task_id)
        source_path = source_root / f"{task.name}.bddl"
        variant = generate_b_variant(source_path, task_id, magnitude)
        results.append(
            write_b_variant(
                variant,
                output_root,
                regenerated_initial_states_root=regenerated_initial_states_root,
                source_init_root=source_init_root,
                assets_root=assets_root,
                write_config=task_id == 0,
            )
        )
    return results


def seeded_reset(environment: Any, init_state: int, seed: int) -> Any:
    """Set every exposed environment seed before a deterministic reset."""

    target = getattr(environment, "environment", None)
    seed_owner = target if target is not None and hasattr(target, "seed") else environment
    if hasattr(seed_owner, "seed"):
        seed_owner.seed(int(seed))
    if hasattr(environment, "evaluation_seed"):
        environment.evaluation_seed = int(seed)
    reset = getattr(environment, "reset", None)
    if reset is None:
        raise StageSError("environment has no reset method")
    try:
        return reset(int(init_state))
    except TypeError:
        return reset(seed=int(seed))


def _observation(environment: Any) -> Any:
    if hasattr(environment, "raw_observation"):
        return environment.raw_observation()
    value = getattr(environment, "observation", None)
    return value() if callable(value) else value


def _state_vector(environment: Any, observation: Any = None) -> np.ndarray:
    for name in ("state_vector", "get_state_vector", "current_state_vector"):
        if hasattr(environment, name):
            value = getattr(environment, name)
            return np.asarray(value() if callable(value) else value, dtype=np.float64).reshape(-1)
    if isinstance(observation, Mapping):
        pieces = []
        for key in sorted(observation):
            if "image" in key.lower() or key == "prompt":
                continue
            value = np.asarray(observation[key])
            if np.issubdtype(value.dtype, np.number):
                pieces.append(value.astype(np.float64).reshape(-1))
        if pieces:
            return np.concatenate(pieces)
    if observation is not None:
        value = np.asarray(observation)
        if np.issubdtype(value.dtype, np.number):
            return value.astype(np.float64).reshape(-1)
    return np.zeros(1, dtype=np.float64)


def _pose_vector(environment: Any, observation: Any = None) -> np.ndarray:
    for name in ("pose_vector", "get_pose_vector", "current_pose_vector"):
        if hasattr(environment, name):
            value = getattr(environment, name)
            return np.asarray(value() if callable(value) else value, dtype=np.float64).reshape(-1)
    return _state_vector(environment, observation)


def _execute_one(environment: Any, action: np.ndarray) -> dict[str, Any]:
    action = np.asarray(action, dtype=np.float32)
    if hasattr(environment, "execute_actions"):
        try:
            result = environment.execute_actions(action[None, ...])
        except (TypeError, ValueError, IndexError):
            result = environment.execute_actions(action)
    elif hasattr(environment, "step"):
        result = environment.step(action)
    else:
        raise StageSError("environment has neither execute_actions nor step")
    if isinstance(result, tuple):
        # gym-style (obs, reward, terminated, truncated, info)
        if len(result) >= 5:
            result = {"done": bool(result[2] or result[3]), "success": bool(result[4].get("success", False)) if isinstance(result[4], Mapping) else False, "info": result[4]}
        elif len(result) >= 4:
            result = {"done": bool(result[2]), "success": False, "info": result[3]}
        else:
            result = {}
    if result is None:
        result = {}
    if not isinstance(result, Mapping):
        result = {"done": bool(getattr(result, "done", False))}
    output = dict(result)
    output.setdefault("success", bool(output.get("is_success", False)))
    output.setdefault("done", bool(output["success"]))
    return output


def _sample_chunk(policy: Any, observation: Any, seed: int, counter: int, action_count: int = REPLAN_STEPS) -> np.ndarray:
    if hasattr(policy, "sample_action_chunk"):
        actions = policy.sample_action_chunk(observation, seed=int(seed), counter=int(counter))
    elif hasattr(policy, "sample_action"):
        actions = [policy.sample_action(observation, seed=int(seed), counter=int(counter)) for _ in range(action_count)]
    else:
        raise StageSError("policy adapter must expose sample_action_chunk or sample_action")
    array = np.asarray(actions, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[0] < action_count:
        raise StageSError(f"policy returned invalid action chunk shape {array.shape}")
    return array[:action_count].copy()


@dataclass
class StageRSnapshot:
    """Stage-R-compatible snapshot, with policy RNG state when exposed."""

    environment: Any
    observation_history: Any
    action_queue: list[np.ndarray]
    python_rng_state: object
    numpy_rng_state: object
    baseline_noise_seed: int
    baseline_noise_counter: int
    step: int
    policy_rng_state: Any = None


def _policy_rng_state(policy: Any) -> Any:
    for name in ("get_rng_state", "rng_state"):
        if hasattr(policy, name):
            value = getattr(policy, name)
            return copy.deepcopy(value() if callable(value) else value)
    return None


def _restore_policy_rng(policy: Any, state: Any) -> None:
    if state is None or policy is None:
        return
    setter = getattr(policy, "set_rng_state", None)
    if setter is not None:
        setter(copy.deepcopy(state))


def capture_stage_r_snapshot(environment: Any, action_queue: Sequence[np.ndarray], seed: int, counter: int, step: int, *, policy: Any = None) -> StageRSnapshot:
    if not hasattr(environment, "capture_snapshot"):
        raise SnapshotReplayError("Stage-R snapshot requires environment.capture_snapshot")
    history = getattr(environment, "_observation", None)
    if history is None:
        history = getattr(environment, "observation_history", None)
    return StageRSnapshot(
        environment=copy.deepcopy(environment.capture_snapshot()),
        observation_history=copy.deepcopy(history),
        action_queue=[np.asarray(value, dtype=np.float32).copy() for value in action_queue],
        python_rng_state=copy.deepcopy(random.getstate()),
        numpy_rng_state=copy.deepcopy(np.random.get_state()),
        baseline_noise_seed=int(seed),
        baseline_noise_counter=int(counter),
        step=int(step),
        policy_rng_state=_policy_rng_state(policy),
    )


def restore_stage_r_snapshot(environment: Any, snapshot: StageRSnapshot, *, policy: Any = None) -> list[np.ndarray]:
    if not hasattr(environment, "restore_snapshot"):
        raise SnapshotReplayError("Stage-R snapshot requires environment.restore_snapshot")
    if hasattr(environment, "evaluation_seed") and hasattr(snapshot.environment, "evaluation_seed"):
        environment.evaluation_seed = int(snapshot.environment.evaluation_seed)
    environment.restore_snapshot(copy.deepcopy(snapshot.environment))
    if snapshot.observation_history is not None:
        if hasattr(environment, "_observation"):
            environment._observation = copy.deepcopy(snapshot.observation_history)
        elif hasattr(environment, "observation_history"):
            environment.observation_history = copy.deepcopy(snapshot.observation_history)
    random.setstate(copy.deepcopy(snapshot.python_rng_state))
    np.random.set_state(copy.deepcopy(snapshot.numpy_rng_state))
    _restore_policy_rng(policy, snapshot.policy_rng_state)
    return [np.asarray(value, dtype=np.float32).copy() for value in snapshot.action_queue]


def _numeric_leaves(value: Any) -> list[np.ndarray]:
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number):
            return [value.astype(np.float64).reshape(-1)]
        return []
    if isinstance(value, Mapping):
        result: list[np.ndarray] = []
        for key in sorted(value, key=str):
            result.extend(_numeric_leaves(value[key]))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(_numeric_leaves(item))
        return result
    if isinstance(value, (bool, int, float, np.number)):
        return [np.asarray([value], dtype=np.float64)]
    return []


def validate_restore_same_action(
    environment: Any,
    snapshot: StageRSnapshot,
    action: Iterable[float],
    *,
    second_environment: Any | None = None,
    policy: Any = None,
    tolerance: float = SNAPSHOT_REPLAY_TOLERANCE,
) -> dict[str, Any]:
    """Verify restore -> same action -> identical next state <= 1e-9."""

    action_array = np.asarray(action, dtype=np.float32)
    restore_stage_r_snapshot(environment, snapshot, policy=policy)
    _execute_one(environment, action_array)
    first = _state_vector(environment, _observation(environment))
    if second_environment is None:
        restore_stage_r_snapshot(environment, snapshot, policy=policy)
        _execute_one(environment, action_array)
        second = _state_vector(environment, _observation(environment))
    else:
        restore_stage_r_snapshot(second_environment, snapshot, policy=policy)
        _execute_one(second_environment, action_array)
        second = _state_vector(second_environment, _observation(second_environment))
    if first.shape != second.shape:
        error = float("inf")
    else:
        error = float(np.max(np.abs(first - second))) if first.size else 0.0
    result = {"passed": bool(error <= float(tolerance)), "max_abs_error": error, "tolerance": float(tolerance), "same_action": True}
    if not result["passed"]:
        raise SnapshotReplayError(f"restore same-action next-state error {error} > {tolerance}")
    return result


@dataclass
class CandidateOutcome:
    candidate_id: int
    candidate_seed: int
    actions: np.ndarray
    poses: np.ndarray
    success: bool
    policy_forwards: int
    environment_steps: int
    snapshot: StageRSnapshot | None = None


def _factory_call(factory: Callable[..., Any], task_id: int, init_state: int, candidate_id: int, seed: int, variant: Any) -> Any:
    try:
        return factory(task_id=task_id, init_state=init_state, candidate_id=candidate_id, seed=seed, variant=variant)
    except TypeError:
        try:
            return factory(task_id, init_state, candidate_id, seed, variant)
        except TypeError:
            return factory(task_id, candidate_id, variant)


def collect_family(
    environment_factory: Callable[..., Any],
    policy: Any,
    *,
    task_id: int,
    init_state: int,
    candidate_count: int = MAIN_CANDIDATE_COUNT,
    max_steps: int = 1000,
    variant: Any = None,
    seed_namespace: str = STAGE_S_PROTOCOL_ID,
    validate_snapshots: bool = False,
) -> dict[str, Any]:
    """Collect one complete no-intervention family using real adapters."""

    if candidate_count <= 0 or max_steps <= 0:
        raise ValueError("candidate_count and max_steps must be positive")
    envs = []
    outcomes: list[CandidateOutcome] = []
    try:
        for candidate_id in range(int(candidate_count)):
            seed = stable_seed(seed_namespace, "candidate", task_id, init_state, candidate_id)
            env = _factory_call(environment_factory, task_id, init_state, candidate_id, seed, variant)
            envs.append(env)
            observation = seeded_reset(env, init_state, stable_seed(seed_namespace, "environment", task_id, init_state))
            initial_snapshot = None
            if hasattr(env, "capture_snapshot"):
                initial_snapshot = capture_stage_r_snapshot(env, [], seed, 0, 0, policy=policy)
            queue: list[np.ndarray] = []
            actions: list[np.ndarray] = []
            poses: list[np.ndarray] = []
            forwards = 0
            success = False
            done = False
            for step in range(int(max_steps)):
                if not queue:
                    queue.extend(_sample_chunk(policy, observation, seed, forwards))
                    forwards += 1
                action = np.asarray(queue.pop(0), dtype=np.float32)
                result = _execute_one(env, action)
                observation = _observation(env)
                actions.append(action.copy())
                poses.append(_pose_vector(env, observation))
                success = bool(result.get("success", False))
                done = bool(result.get("done", False))
                if done:
                    break
            if not done:
                raise StageSError(f"family task={task_id} init={init_state} candidate={candidate_id} exceeded max_steps={max_steps}")
            if validate_snapshots and initial_snapshot is not None and actions:
                validate_restore_same_action(env, initial_snapshot, actions[0], policy=policy)
            outcomes.append(CandidateOutcome(candidate_id, seed, np.asarray(actions, dtype=np.float32), np.asarray(poses, dtype=np.float64), success, forwards, len(actions), initial_snapshot))
    finally:
        for env in envs:
            if hasattr(env, "close"):
                env.close()
    family_id = f"task{int(task_id):02d}_init{int(init_state):03d}"
    return {
        "protocol_id": STAGE_S_PROTOCOL_ID,
        "substrate": getattr(variant, "substrate", None) or ("B" if variant is not None else "A_OR_C"),
        "suite": LIBERO_SUITE,
        "task_id": int(task_id),
        "task_name": task_spec(task_id).name,
        "init_state": int(init_state),
        "family_id": family_id,
        "candidate_count": int(candidate_count),
        "outcomes": outcomes,
        "policy_forwards": int(sum(value.policy_forwards for value in outcomes)),
        "environment_steps": int(sum(value.environment_steps for value in outcomes)),
    }


def _snapshot_payload(snapshot: StageRSnapshot | None) -> Any:
    if snapshot is None:
        return None
    return {
        "environment": snapshot.environment,
        "observation_history": snapshot.observation_history,
        "action_queue": snapshot.action_queue,
        "python_rng_state": snapshot.python_rng_state,
        "numpy_rng_state": snapshot.numpy_rng_state,
        "baseline_noise_seed": snapshot.baseline_noise_seed,
        "baseline_noise_counter": snapshot.baseline_noise_counter,
        "step": snapshot.step,
        "policy_rng_state": snapshot.policy_rng_state,
    }


def _pack_family(family: Mapping[str, Any]) -> dict[str, np.ndarray]:
    outcomes: Sequence[CandidateOutcome] = family["outcomes"]
    lengths = np.asarray([len(value.actions) for value in outcomes], dtype=np.int32)
    offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    pose_width = max((value.poses.shape[1] if value.poses.ndim == 2 else 0) for value in outcomes)
    padded_poses = []
    for value in outcomes:
        poses = value.poses if value.poses.ndim == 2 else np.empty((len(value.actions), 0), dtype=np.float64)
        if poses.shape[1] < pose_width:
            poses = np.pad(poses, ((0, 0), (0, pose_width - poses.shape[1])))
        padded_poses.append(poses)
    return {
        "lengths": lengths,
        "offsets": offsets,
        "actions": np.concatenate([value.actions for value in outcomes], axis=0).astype(np.float32),
        "poses": np.concatenate(padded_poses, axis=0).astype(np.float64),
        "success": np.asarray([value.success for value in outcomes], dtype=np.bool_),
        "candidate_id": np.asarray([value.candidate_id for value in outcomes], dtype=np.int16),
        "candidate_seed": np.asarray([value.candidate_seed for value in outcomes], dtype=np.uint64),
        "generation_step": np.zeros(len(outcomes), dtype=np.int32),
        "policy_forwards": np.asarray([value.policy_forwards for value in outcomes], dtype=np.int32),
        "environment_steps": np.asarray([value.environment_steps for value in outcomes], dtype=np.int32),
    }


def family_is_complete(directory: str | Path, *, expected_candidates: int = MAIN_CANDIDATE_COUNT) -> bool:
    root = Path(directory)
    marker_path = root / "COMPLETED_FAMILY.json"
    sums_path = root / "SHA256SUMS"
    if not marker_path.is_file() or not sums_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        lines = sums_path.read_text(encoding="utf-8").splitlines()
        for path in (root / "rollouts.npz", root / "genealogy.json", root / "snapshots.pkl", root / "metadata.json"):
            if not path.is_file():
                return False
        expected = {f"{sha256_file(root / name)}  {name}" for name in ("rollouts.npz", "genealogy.json", "snapshots.pkl", "metadata.json")}
        if set(lines) != expected or marker.get("protocol_id") != STAGE_S_PROTOCOL_ID:
            return False
        if marker.get("files") != {name: sha256_file(root / name) for name in ("rollouts.npz", "genealogy.json", "snapshots.pkl", "metadata.json")}:
            return False
        with np.load(root / "rollouts.npz", allow_pickle=False) as data:
            return len(data["success"]) == int(expected_candidates)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def write_family_atomic(directory: str | Path, family: Mapping[str, Any]) -> dict[str, Any]:
    """Persist a family; marker is written last and makes resume fail-closed."""

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    outcomes: Sequence[CandidateOutcome] = family["outcomes"]
    if len(outcomes) != int(family["candidate_count"]):
        raise ValueError("family candidate count does not match outcomes")
    npz_path = root / "rollouts.npz"
    genealogy = []
    for value in outcomes:
        genealogy.append(
            {
                "candidate_id": int(value.candidate_id),
                "parent_id": None,
                "generation_step": 0,
                "candidate_seed": int(value.candidate_seed),
                "action_prefix": value.actions.astype(float).tolist(),
                "final_success": bool(value.success),
            }
        )
    metadata = {
        "protocol_id": STAGE_S_PROTOCOL_ID,
        "schema_version": 1,
        "substrate": family.get("substrate"),
        "suite": family["suite"],
        "task_id": int(family["task_id"]),
        "task_name": family["task_name"],
        "init_state": int(family["init_state"]),
        "family_id": family["family_id"],
        "candidate_count": int(family["candidate_count"]),
        "policy_forwards": int(family["policy_forwards"]),
        "environment_steps": int(family["environment_steps"]),
        "genealogy_file": "genealogy.json",
        "rollouts_file": "rollouts.npz",
        "snapshots_file": "snapshots.pkl",
        "written_at_unix": time.time(),
    }
    snapshots = {str(value.candidate_id): _snapshot_payload(value.snapshot) for value in outcomes}
    atomic_npz(npz_path, _pack_family(family))
    atomic_json(root / "genealogy.json", genealogy)
    atomic_bytes(root / "snapshots.pkl", pickle.dumps({"schema_version": 1, "candidates": snapshots}, protocol=5))
    atomic_json(root / "metadata.json", metadata)
    names = ("rollouts.npz", "genealogy.json", "snapshots.pkl", "metadata.json")
    hashes = {name: sha256_file(root / name) for name in names}
    atomic_text(root / "SHA256SUMS", "".join(f"{hashes[name]}  {name}\n" for name in names))
    marker = {
        "protocol_id": STAGE_S_PROTOCOL_ID,
        "marker_type": "completed_family",
        "family_id": family["family_id"],
        "candidate_count": int(family["candidate_count"]),
        "files": hashes,
        "checkpoint": "FAMILY_COMPLETE",
    }
    atomic_json(root / "COMPLETED_FAMILY.json", marker)
    return marker


def run_main_screen(
    environment_factory: Callable[..., Any],
    policy: Any,
    output_root: str | Path,
    *,
    substrate: str,
    variant: Any = None,
    initial_states: Sequence[int] = tuple(range(MAIN_INITIAL_STATE_COUNT)),
    task_ids: Sequence[int] = tuple(range(TASK_COUNT)),
    candidate_count: int = MAIN_CANDIDATE_COUNT,
    max_steps: int = 1000,
    validate_snapshots: bool = False,
) -> dict[str, Any]:
    if substrate not in {"A", "B", "C"}:
        raise ValueError("substrate must be A, B, or C")
    completed = skipped = 0
    for task_id in task_ids:
        for init_state in initial_states:
            directory = Path(output_root) / substrate / f"task{int(task_id):02d}" / f"init{int(init_state):03d}"
            if family_is_complete(directory, expected_candidates=candidate_count):
                skipped += 1
                continue
            family = collect_family(
                environment_factory,
                policy,
                task_id=int(task_id),
                init_state=int(init_state),
                candidate_count=int(candidate_count),
                max_steps=int(max_steps),
                variant=variant,
                validate_snapshots=validate_snapshots,
            )
            family["substrate"] = substrate
            write_family_atomic(directory, family)
            completed += 1
    return {"protocol_id": STAGE_S_PROTOCOL_ID, "substrate": substrate, "completed_families": completed, "skipped_complete_families": skipped, "task_count": len(task_ids), "initial_state_count": len(initial_states), "candidate_count": int(candidate_count)}


def calibration_plan(settings: Sequence[Any], *, task_ids: Sequence[int] = CALIBRATION_TASK_IDS, initial_states: Sequence[int] = CALIBRATION_INITIAL_STATES, candidate_count: int = CALIBRATION_CANDIDATE_COUNT) -> dict[str, Any]:
    return {
        "protocol_id": STAGE_S_PROTOCOL_ID,
        "task_ids": [int(value) for value in task_ids],
        "initial_states": [int(value) for value in initial_states],
        "candidate_count": int(candidate_count),
        "settings": list(settings),
        "total_trials_per_setting": int(len(task_ids) * len(initial_states) * candidate_count),
        "persisted_fields_only": ["setting", "successes", "total", "pooled_success"],
        "forbidden_persisted_fields": ["family", "trajectory", "genealogy", "S2", "S3", "S4", "S5", "divergence", "overdispersion", "all_fail"],
    }


def run_pooled_calibration(
    evaluator: Callable[..., bool],
    settings: Sequence[Any],
    *,
    task_ids: Sequence[int] = CALIBRATION_TASK_IDS,
    initial_states: Sequence[int] = CALIBRATION_INITIAL_STATES,
    candidate_count: int = CALIBRATION_CANDIDATE_COUNT,
) -> dict[str, Any]:
    """Run B/C calibration while retaining only aggregate counters."""

    if len(settings) != 4:
        raise ValueError("Stage-S calibration fixes exactly four settings")
    rows = []
    for setting_index, setting in enumerate(settings):
        successes = 0
        total = 0
        for task_id in task_ids:
            for init_state in initial_states:
                for candidate_id in range(int(candidate_count)):
                    seed = stable_seed(STAGE_S_PROTOCOL_ID, "calibration", setting_index, task_id, init_state, candidate_id)
                    successes += int(bool(evaluator(setting, int(task_id), int(init_state), int(candidate_id), int(seed))))
                    total += 1
        rows.append({"setting": setting, "successes": int(successes), "total": int(total), "pooled_success": float(successes / total if total else float("nan"))})
    selected = min(rows, key=lambda row: (abs(float(row["pooled_success"]) - CALIBRATION_TARGET_SUCCESS), str(row["setting"])))
    return {"protocol_id": STAGE_S_PROTOCOL_ID, "target_pooled_success": CALIBRATION_TARGET_SUCCESS, "rows": rows, "selected_setting": selected["setting"]}


def write_pooled_calibration(path: str | Path, result: Mapping[str, Any]) -> dict[str, Any]:
    """Persist only the calibration allow-list; reject accidental leakage."""

    allowed = {"protocol_id", "target_pooled_success", "rows", "selected_setting"}
    if set(result) - allowed:
        raise ValueError(f"calibration result contains forbidden top-level fields: {sorted(set(result) - allowed)}")
    rows = []
    for row in result.get("rows", []):
        if set(row) != {"setting", "successes", "total", "pooled_success"}:
            raise ValueError("calibration row contains non-aggregate fields")
        rows.append({"setting": row["setting"], "successes": int(row["successes"]), "total": int(row["total"]), "pooled_success": float(row["pooled_success"])})
    payload = {"protocol_id": result.get("protocol_id", STAGE_S_PROTOCOL_ID), "target_pooled_success": float(result.get("target_pooled_success", CALIBRATION_TARGET_SUCCESS)), "rows": rows, "selected_setting": result.get("selected_setting")}
    atomic_json(path, payload)
    return payload


def _read_json_candidates(path: Path) -> list[dict[str, Any]]:
    values = []
    for name in ("CHECKPOINT_PROVENANCE.json", "metadata.json", "checkpoint.json", "manifest.json"):
        candidate = path / name if path.is_dir() else path.parent / name
        if not candidate.is_file():
            continue
        try:
            value = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            values.append(value)
    return values


def _find_step(value: Any) -> int | None:
    if isinstance(value, Mapping):
        for key in ("step", "global_step", "checkpoint_step", "steps"):
            if key in value and isinstance(value[key], (int, float)):
                return int(value[key])
        for child in value.values():
            found = _find_step(child)
            if found is not None:
                return found
    return None


@dataclass(frozen=True)
class CheckpointAudit:
    path: str
    exists: bool
    complete: bool
    exact_policy: bool
    step: int | None
    undertrained: bool
    model_sha256: str | None
    status: str
    errors: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "exists": self.exists,
            "complete": self.complete,
            "exact_policy": self.exact_policy,
            "step": self.step,
            "undertrained": self.undertrained,
            "model_sha256": self.model_sha256,
            "status": self.status,
            "errors": list(self.errors),
        }


def audit_undertrained_checkpoint(path: str | Path, *, expected_step: int | None = None, hash_model: bool = False) -> CheckpointAudit:
    target = Path(path)
    errors: list[str] = []
    exists = target.exists()
    if not exists:
        return CheckpointAudit(str(target), False, False, False, None, False, None, "MISSING", ("checkpoint path does not exist",))
    model = target / "model.safetensors" if target.is_dir() else target
    if not model.is_file():
        errors.append("missing model.safetensors")
    provenance = _read_json_candidates(target)
    text = " ".join(canonical_json(value).lower() for value in provenance) + " " + str(target).lower()
    exact_policy = "pi05" in text and "libero" in text
    if not exact_policy:
        errors.append("checkpoint is not identified as exact pi05_libero")
    step = _find_step(provenance[0]) if provenance else None
    if step is None:
        match = re.search(r"(?:step|checkpoint)[_-]?(\d+)", str(target), flags=re.IGNORECASE)
        step = int(match.group(1)) if match else None
    if expected_step is not None and step != int(expected_step):
        errors.append(f"step {step!r} != expected {expected_step}")
    if step is None:
        errors.append("checkpoint step is not declared; refusing to infer under-training")
    undertrained = step is not None and step < FULL_PI05_LIBERO_TRAINING_STEPS
    if not undertrained:
        errors.append(f"step {step!r} is not below full training step {FULL_PI05_LIBERO_TRAINING_STEPS}")
    # Real checkpoint completeness requires weights, metadata, and norm stats;
    # optimizer is not required for inference but missing weights is fatal.
    complete = model.is_file() and bool(provenance)
    if not complete:
        errors.append("checkpoint lacks a complete real-weight/provenance pair")
    model_sha = sha256_file(model) if hash_model and model.is_file() else None
    return CheckpointAudit(str(target), True, complete, exact_policy, step, undertrained, model_sha, "PASS" if not errors else "FAIL", tuple(errors))


def audit_undertrained_checkpoint_set(paths: Sequence[str | Path], *, expected_steps: Sequence[int] | None = None, hash_model: bool = False) -> dict[str, Any]:
    if len(paths) != UNDERTRAINED_CHECKPOINT_COUNT:
        return {"valid": False, "errors": [f"exactly four real undertrained checkpoints are required, got {len(paths)}"], "checkpoints": []}
    if len({str(Path(path).resolve()) for path in paths}) != len(paths):
        return {"valid": False, "errors": ["checkpoint paths must be unique"], "checkpoints": []}
    audits = [audit_undertrained_checkpoint(path, expected_step=(expected_steps[index] if expected_steps else None), hash_model=hash_model) for index, path in enumerate(paths)]
    errors = [f"{audit.path}: {error}" for audit in audits for error in audit.errors]
    return {"valid": not errors, "errors": errors, "checkpoints": [audit.as_dict() for audit in audits], "no_interpolation": True}


def build_c_training_launcher_contract(
    *,
    qpilots_root: str | Path,
    output_root: str | Path,
    checkpoint_paths: Sequence[str | Path],
    expected_steps: Sequence[int] | None = None,
    python: str = "python",
    gpu_count: int = 4,
    resource_pool: str = "idle",
) -> dict[str, Any]:
    """Return a real OpenPI PyTorch training command; never synthesize C."""

    audit = audit_undertrained_checkpoint_set(checkpoint_paths, expected_steps=expected_steps)
    openpi = Path(qpilots_root) / "third_party" / "openpi"
    command = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={int(gpu_count)}",
        str(openpi / "scripts" / "train_pytorch.py"),
        "pi05_libero",
        "--exp_name",
        "r142_stage_s_c_undertrained",
        "--checkpoint_dir",
        str(Path(output_root).resolve()),
        "--save_interval",
        "5000",
    ]
    return {
        "protocol_id": STAGE_S_PROTOCOL_ID,
        "substrate": "C",
        "label": "WEAK_SUBSTRATE",
        "same_policy": "pi05_libero",
        "audit": audit,
        "launcher": {
            "working_directory": str(openpi.resolve()),
            "python": str(python),
            "command": command,
            "shell": "cd " + str(openpi.resolve()) + " && " + " ".join(command),
            "gpu_count": int(gpu_count),
            "resource_pool": resource_pool,
            "no_pai_submit_performed": True,
        },
        "checkpoint_contract": {
            "count": UNDERTRAINED_CHECKPOINT_COUNT,
            "paths": [str(Path(path).resolve()) for path in checkpoint_paths],
            "expected_steps": list(expected_steps) if expected_steps is not None else None,
            "interpolation": False,
            "artificial_degradation": False,
        },
    }


def import_callback(spec: str) -> Callable[..., Any]:
    """Load ``module:function`` for the real-runtime scripts."""

    if ":" not in spec:
        raise ValueError("callback must have module:function form")
    module, function = spec.split(":", 1)
    value = getattr(importlib.import_module(module), function)
    if not callable(value):
        raise TypeError(f"callback {spec} is not callable")
    return value


__all__ = [
    "BVariant",
    "CALIBRATION_CANDIDATE_COUNT",
    "CALIBRATION_INITIAL_STATES",
    "CALIBRATION_TASK_IDS",
    "CandidateOutcome",
    "CheckpointAudit",
    "CheckpointQualificationError",
    "LIBERO_10_TASK_NAMES",
    "LIBERO_TASK_SPECS",
    "MAIN_CANDIDATE_COUNT",
    "MAIN_INITIAL_STATE_COUNT",
    "MissingRegeneratedInitialStates",
    "PROXIMITY_MAGNITUDES",
    "SNAPSHOT_REPLAY_TOLERANCE",
    "STAGE_S_PROTOCOL_ID",
    "StageRSnapshot",
    "audit_undertrained_checkpoint",
    "audit_undertrained_checkpoint_set",
    "build_b_variant_suite",
    "build_c_training_launcher_contract",
    "calibration_plan",
    "capture_stage_r_snapshot",
    "collect_family",
    "family_is_complete",
    "generate_b_variant",
    "import_callback",
    "run_main_screen",
    "run_pooled_calibration",
    "seeded_reset",
    "stable_seed",
    "validate_regenerated_initial_states",
    "validate_restore_same_action",
    "write_b_variant",
    "write_family_atomic",
    "write_pooled_calibration",
    "_execute_one",
    "_observation",
    "_pose_vector",
    "_sample_chunk",
    "_state_vector",
]
