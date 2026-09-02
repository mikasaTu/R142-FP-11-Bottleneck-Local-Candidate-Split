#!/usr/bin/env python3
"""Run the Stage-S no-intervention main screen with injected real adapters.

No fake policy or synthetic environment is provided.  For a CPU smoke or a
PAI run, supply callbacks implementing the real policy/environment contract.
Each family is written atomically and can be resumed by rerunning this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

from r142_stage_s.libero import (
    LIBERO_SUITE,
    MAIN_CANDIDATE_COUNT,
    MAIN_INITIAL_STATE_COUNT,
    STAGE_S_PROTOCOL_ID,
    StageRPolicyAdapter,
    atomic_json,
    family_is_complete,
    import_callback,
    make_stage_r_task64_factory,
    run_main_screen,
    task_spec,
    validate_regenerated_initial_states,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--substrate", choices=("A", "B", "C"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment-factory", help="module:function real environment factory")
    parser.add_argument("--policy", help="module:function real policy factory")
    parser.add_argument("--qpilots-root", type=Path, help="pinned Stage-R QPILOTS root for the built-in real adapter")
    parser.add_argument("--libero-root", type=Path, help="pinned LIBERO site/project root for the built-in real adapter")
    parser.add_argument("--checkpoint", type=Path, help="exact policy checkpoint for C (already audited)")
    parser.add_argument("--variant-root", type=Path, help="B variant LIBERO_CONFIG_PATH root")
    parser.add_argument("--source-init-root", type=Path, help="original LIBERO init root used to validate B qpos identity")
    parser.add_argument("--task-id", type=int, action="append", help="subset only for a smoke; omit for all ten tasks")
    parser.add_argument("--initial-state", type=int, action="append", help="subset only for a smoke; omit for all sixteen states")
    parser.add_argument("--candidate-count", type=int, default=MAIN_CANDIDATE_COUNT)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--validate-snapshots", action="store_true")
    parser.add_argument("--calibration-report", type=Path, help="frozen pooled-only B/C calibration report; required for B/C")
    parser.add_argument("--rank", type=int, default=None, help="rank shard; defaults to LOCAL_RANK")
    parser.add_argument("--world-size", type=int, default=None, help="rank count; defaults to WORLD_SIZE")
    parser.add_argument("--weak-substrate", action="store_true", help="required for C and persisted in every family metadata/report")
    parser.add_argument("--source-commit", help="exact Stage-S runtime commit persisted in family metadata")
    parser.add_argument("--dry-run", action="store_true", help="print protocol/accounting contract without loading runtime")
    return parser


_FREEZE_SCHEMA = "r142-stage-s-calibration-freeze-v1"
_FORBIDDEN_LOOKAHEAD_KEYS = frozenset({"s2", "s3", "s4", "s5", "trajectory", "genealogy", "actions", "poses", "divergence", "overdispersion"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_sha256(path: Path) -> str:
    """Hash a file or a checkpoint directory's own immutable manifest."""

    if path.is_file():
        return _sha256(path)
    if path.is_dir() and not path.is_symlink():
        manifest = path / "SHA256SUMS"
        if not manifest.is_file() or manifest.is_symlink():
            raise SystemExit(f"checkpoint directory lacks SHA256SUMS: {path}")
        for raw in manifest.read_text(encoding="utf-8").splitlines():
            parts = raw.split(None, 1)
            if len(parts) != 2:
                raise SystemExit(f"malformed checkpoint SHA256SUMS line: {path}")
            expected, relative = parts
            relative_path = Path(relative.lstrip(" *"))
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise SystemExit(f"unsafe checkpoint SHA256SUMS path: {path}")
            member = path / relative_path
            if not member.is_file() or member.is_symlink() or _sha256(member) != expected:
                raise SystemExit(f"checkpoint SHA256SUMS mismatch: {member}")
        return _sha256(manifest)
    raise SystemExit(f"checkpoint artifact is missing or symlinked: {path}")


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _reject_lookahead(value: object, *, path: str = "report") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in _FORBIDDEN_LOOKAHEAD_KEYS:
                raise SystemExit(f"calibration freeze contains forbidden pre-screen field: {path}.{key}")
            _reject_lookahead(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_lookahead(child, path=f"{path}[{index}]")


def _read_frozen_calibration(
    report_path: Path | None,
    *,
    substrate: str,
    variant_root: Path | None,
    checkpoint: Path | None,
) -> tuple[Path | None, Path | None, dict[str, object] | None]:
    """Resolve only the frozen calibration-selected B variant/C checkpoint.

    The report is intentionally a hard gate.  Its source completion marker and
    hash must be independently present; a partial/running calibration or a
    caller-provided unselected artifact cannot enter S2--S5 main observation.
    """

    if substrate == "A":
        if report_path is not None or checkpoint is None:
            return variant_root, checkpoint, None
        return variant_root, checkpoint, None
    if report_path is None:
        raise SystemExit(f"{substrate} main screen requires --calibration-report; no pre-calibration main run is allowed")
    report_path = report_path.resolve()
    if report_path.is_symlink() or not report_path.is_file():
        raise SystemExit(f"frozen calibration report is missing or symlinked: {report_path}")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid frozen calibration report: {report_path}: {exc}") from exc
    if not isinstance(report, dict):
        raise SystemExit("frozen calibration report must be a JSON object")
    _reject_lookahead(report)
    required = {
        "schema": _FREEZE_SCHEMA,
        "status": "FROZEN",
        "protocol_id": STAGE_S_PROTOCOL_ID,
        "substrate": substrate,
        "calibration_kind": "pooled_only" if substrate == "B" else "checkpoint_calibration",
        "calibration_completed": True,
        "frozen": True,
        "no_s2_s5_peeking": True,
    }
    for key, expected in required.items():
        if report.get(key) != expected:
            raise SystemExit(f"frozen calibration report gate mismatch: {key}={report.get(key)!r}")
    source_result = Path(str(report.get("source_result_path", ""))).resolve()
    source_marker = Path(str(report.get("source_completion_marker", ""))).resolve()
    source_sha = str(report.get("source_result_sha256", ""))
    marker_sha = str(report.get("source_completion_marker_sha256", ""))
    if source_result.is_symlink() or not source_result.is_file() or not source_marker.is_file() or source_marker.is_symlink():
        raise SystemExit("frozen calibration report source completion/result is missing or symlinked")
    if not hashlib.sha256(source_result.read_bytes()).hexdigest() == source_sha:
        raise SystemExit("frozen calibration source result SHA mismatch")
    if not hashlib.sha256(source_marker.read_bytes()).hexdigest() == marker_sha:
        raise SystemExit("frozen calibration completion marker SHA mismatch")
    sums = source_result.with_name("SHA256SUMS")
    if not sums.is_file() or f"{source_sha}  {source_result.name}" not in sums.read_text(encoding="utf-8").splitlines():
        raise SystemExit("frozen calibration source result SHA256SUMS is missing or mismatched")
    if substrate == "B":
        selected = report.get("selected_setting")
        if not isinstance(selected, str) or selected not in {"proximity_0.06m", "proximity_0.08m", "proximity_0.10m", "proximity_0.12m"}:
            raise SystemExit("B frozen calibration selected_setting is invalid")
        if report.get("variant_run_id") != "r142-stage-s-b-variants-20260903-r7":
            raise SystemExit("B frozen calibration must select the frozen r7 variant run")
        selected_root = Path(str(report.get("selected_variant_root", ""))).resolve()
        if variant_root is None:
            raise SystemExit("B main screen requires the r7 variant root")
        variant_base = variant_root.resolve()
        if selected_root != (variant_base / selected).resolve() or not selected_root.is_relative_to(variant_base):
            raise SystemExit("B main screen attempted to load a non-selected or escaping variant root")
        if not selected_root.is_dir() or selected_root.is_symlink() or not (selected_root / "config.yaml").is_file():
            raise SystemExit("selected B variant root is incomplete")
        if report.get("selected_variant_root_sha256") != _sha256(selected_root / "config.yaml"):
            raise SystemExit("selected B variant config SHA mismatch")
        return selected_root, checkpoint, report
    selected_checkpoint = Path(str(report.get("selected_checkpoint", ""))).resolve()
    if checkpoint is not None and checkpoint.resolve() != selected_checkpoint:
        raise SystemExit("C caller checkpoint differs from frozen calibration-selected checkpoint")
    if not selected_checkpoint.is_file() or selected_checkpoint.is_symlink():
        raise SystemExit("frozen C calibration-selected checkpoint is missing or symlinked")
    if report.get("selected_checkpoint_sha256") != _artifact_sha256(selected_checkpoint):
        raise SystemExit("frozen C selected checkpoint SHA mismatch")
    return variant_root, selected_checkpoint, report


def _make_policy(callback, args):
    kwargs = {"checkpoint": str(args.checkpoint) if args.checkpoint else None, "substrate": args.substrate}
    try:
        return callback(**kwargs)
    except TypeError:
        try:
            return callback(str(args.checkpoint)) if args.checkpoint else callback()
        except TypeError:
            return callback()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.substrate == "C" and not args.weak_substrate:
        raise SystemExit("C main screen must explicitly carry WEAK_SUBSTRATE annotation")
    if args.substrate != "C" and args.weak_substrate:
        raise SystemExit("--weak-substrate is reserved for C")
    rank = int(os.environ.get("LOCAL_RANK", "0")) if args.rank is None else int(args.rank)
    world_size = int(os.environ.get("WORLD_SIZE", "1")) if args.world_size is None else int(args.world_size)
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise SystemExit(f"invalid rank/world-size: rank={rank} world_size={world_size}")
    all_tasks = tuple(args.task_id) if args.task_id else tuple(range(10))
    all_states = tuple(args.initial_state) if args.initial_state else tuple(range(MAIN_INITIAL_STATE_COUNT))
    all_pairs = tuple((int(task), int(state)) for task in all_tasks for state in all_states)
    pairs = all_pairs[rank::world_size]
    tasks = tuple(sorted({task for task, _ in pairs}))
    states = tuple(sorted({state for _, state in pairs}))
    resolved_variant_root, resolved_checkpoint, freeze_report = _read_frozen_calibration(
        args.calibration_report,
        substrate=args.substrate,
        variant_root=args.variant_root,
        checkpoint=args.checkpoint,
    )
    # From this point on the calibration report, not a caller-provided path,
    # is the sole checkpoint/variant source of truth.
    args.checkpoint = resolved_checkpoint
    if not args.dry_run and args.substrate in {"B", "C"}:
        if not isinstance(args.source_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
            raise SystemExit("B/C main screen requires an exact 40-hex --source-commit")
    contract = {
        "protocol_id": STAGE_S_PROTOCOL_ID,
        "substrate": args.substrate,
        "suite": LIBERO_SUITE,
        "task_ids": list(tasks),
        "initial_states": list(states),
        "candidate_count": int(args.candidate_count),
        "rank": rank,
        "world_size": world_size,
        "family_pairs": [[task, state] for task, state in pairs],
        "no_intervention": True,
        "primary_compute_unit": "policy_forward_pass",
        "secondary_compute_unit": "environment_step",
        "atomic_resume": True,
        "variant_root": str(resolved_variant_root.resolve()) if resolved_variant_root else None,
        "checkpoint": str(resolved_checkpoint.resolve()) if resolved_checkpoint else None,
        "calibration_report": str(args.calibration_report.resolve()) if args.calibration_report else None,
        "substrate_annotation": "WEAK_SUBSTRATE" if args.substrate == "C" else None,
        "source_commit": args.source_commit,
        "replay_gate": "restore -> same action -> next-state <= 1e-9; Python/NumPy/Torch CPU/CUDA/policy RNG plus history/action queue",
    }
    if args.dry_run:
        print(json.dumps(contract, indent=2, sort_keys=True))
        return 0
    if args.environment_factory is None:
        if args.qpilots_root is None or args.libero_root is None:
            raise SystemExit("supply --environment-factory or both --qpilots-root and --libero-root")
        factory = make_stage_r_task64_factory(
            args.qpilots_root,
            args.libero_root,
            checkpoint=resolved_checkpoint,
            variant_root=resolved_variant_root,
            max_steps=args.max_steps,
        )
    else:
        factory = import_callback(args.environment_factory)
    if args.policy is None:
        if args.qpilots_root is None or args.checkpoint is None:
            raise SystemExit("supply --policy or both --qpilots-root and --checkpoint")
        policy = StageRPolicyAdapter(
            resolved_checkpoint,
            qpilots_root=args.qpilots_root,
            default_prompt=task_spec(0).prompt,
        )
    else:
        policy = _make_policy(import_callback(args.policy), args)
    if args.substrate == "B" and resolved_variant_root is None:
        raise SystemExit("B main screen requires a generated variant root with regenerated init qpos")
    if args.substrate == "B":
        if args.source_init_root is None:
            raise SystemExit("B main screen requires --source-init-root for old-qpos identity validation")
        audit = validate_regenerated_initial_states(resolved_variant_root.parent, args.source_init_root)
        if not audit["valid"]:
            raise SystemExit("B regenerated qpos audit failed closed: " + "; ".join(audit["errors"]))
    if args.substrate == "C" and resolved_checkpoint is None:
        raise SystemExit("C main screen requires one exact audited checkpoint")
    variant = SimpleNamespace(substrate=args.substrate, root=str(resolved_variant_root) if resolved_variant_root else None)
    metadata_extra = {
        "rank": rank,
        "world_size": world_size,
        "source_commit": args.source_commit,
        "termination": "official eval_success or step_limit",
        "replay_gate": "restore -> same action -> next-state <= 1e-9",
        "policy_rng_streams": "python_numpy_torch_cpu_torch_cuda_policy",
        "calibration_report": str(args.calibration_report.resolve()) if args.calibration_report else None,
    }
    if args.substrate == "C":
        metadata_extra["substrate_annotation"] = "WEAK_SUBSTRATE"
    result = run_main_screen(
        factory,
        policy,
        args.output,
        substrate=args.substrate,
        variant=variant,
        task_ids=tasks,
        initial_states=states,
        family_pairs=pairs,
        candidate_count=args.candidate_count,
        max_steps=args.max_steps,
        validate_snapshots=args.validate_snapshots,
        metadata_extra=metadata_extra,
    )
    result.update(contract)
    result["freeze_report"] = freeze_report
    result["rank"] = rank
    result["world_size"] = world_size
    result["family_paths"] = [
        str((args.output / args.substrate / f"task{task:02d}" / f"init{state:03d}").resolve())
        for task, state in pairs
    ]
    for path in result["family_paths"]:
        if not family_is_complete(path, expected_candidates=args.candidate_count):
            raise SystemExit(f"rank completed without a complete family: {path}")
    summary_path = args.output / f"{args.substrate}_MAIN_SUMMARY_RANK-{rank:04d}.json"
    atomic_json(summary_path, result)
    rank_manifest_path = args.output / f"SHA256SUMS_{args.substrate}_MAIN_RANK-{rank:04d}"
    output_root = args.output.resolve()
    manifest_lines = [f"{_sha256(summary_path)}  {summary_path.name}"]
    for family_path in result["family_paths"]:
        family_marker = Path(family_path) / "COMPLETED_FAMILY.json"
        manifest_lines.append(
            f"{_sha256(family_marker)}  {family_marker.resolve().relative_to(output_root).as_posix()}"
        )
    _atomic_text(rank_manifest_path, "\n".join(manifest_lines) + "\n")
    rank_marker = {
        "status": "COMPLETED",
        "marker_type": "completed_stage_s_main_rank",
        "substrate": args.substrate,
        "protocol_id": STAGE_S_PROTOCOL_ID,
        "rank": rank,
        "world_size": world_size,
        "family_count": len(pairs),
        "candidate_budget": int(args.candidate_count),
        "families": result["family_paths"],
        "summary": summary_path.name,
        "sha256sums": rank_manifest_path.name,
        "source_commit": args.source_commit,
        "substrate_annotation": "WEAK_SUBSTRATE" if args.substrate == "C" else None,
        "calibration_report": str(args.calibration_report.resolve()) if args.calibration_report else None,
        "replay_gate": "restore -> same action -> next-state <= 1e-9",
    }
    atomic_json(args.output / f"COMPLETED_{args.substrate}_MAIN_RANK-{rank:04d}.json", rank_marker)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
