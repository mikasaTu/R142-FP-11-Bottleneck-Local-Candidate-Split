#!/usr/bin/env python3
"""Static contract tests for the four natural Phase-1R idle execution shards."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PAI = ROOT / "pai"
LAUNCHER = PAI / "run_stage_r_phase1r_idle_4gpu.sh"
CHECKPOINT_VALIDATOR = PAI / "validate_stage_r_checkpoint_attestation.py"
FROZEN_SOURCE_VALIDATOR = PAI / "validate_stage_r_frozen_source_resume.py"
SMALL_PREFLIGHT_VALIDATOR = PAI / "validate_stage_r_small_preflight_marker.py"
PREREQUISITE_VALIDATOR = PAI / "validate_stage_r_phase1r_prerequisites.py"
TEMPLATES = (
    PAI / "r142_stage_r_phase1r_natural_shard_a0_idle4.json",
    PAI / "r142_stage_r_phase1r_natural_shard_a1_idle4.json",
    PAI / "r142_stage_r_phase1r_natural_shard_b0_idle4.json",
    PAI / "r142_stage_r_phase1r_natural_shard_b1_idle4.json",
)
EXECUTION = ROOT / "configs" / "stage_r_phase1r_execution_idle4.json"
SCIENTIFIC_SHARDS = ROOT / "configs" / "stage_r_phase1r_shards.json"
CHECKPOINT_ATTESTATION = ROOT / "configs" / "stage_r_pi05_libero_checkpoint_attestation.json"
EXPECTED_MOUNTS = (
    ("d-mkixtohdn75dp8x9tb", "/mnt/cpfs/zbl-cpfs-new/USERS/leon"),
    ("d-36p023eg0f2vuqny8y", "/mnt/cpfs/zbl-cpfs-new/CKPT/leon"),
    ("d-ejgj2ej7io1t2t32uc", "/mnt/cpfs/zbl-cpfs-new/dataset/leon"),
    ("d-wf28n33a829hb5kvne", "/mnt/cpfs/zbl-cpfs-new/x2robot_data/"),
    ("d-n0h4g9qcnu9i4acnya", "/mnt/cpfs/zbl-cpfs-new/share/"),
)
EXPECTED_AIMASTER = (
    "--job-execution-mode=Sync --enable-job-restart=True "
    "--max-num-of-job-restart=50 --fault-tolerant-policy=OnFailure"
)
EXPECTED_LAUNCHER_SHA = hashlib.sha256(LAUNCHER.read_bytes()).hexdigest()
EXPECTED_SCIENCE_COMMIT = "18847dfc2dab91b18edcd296b9d7e363b5e48570"


class Phase1RPAIContractTest(unittest.TestCase):
    def test_launcher_is_strict_bash_and_bound_to_runtime_source_pin(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertTrue(text.startswith("#!/usr/bin/env bash"))
        self.assertIn("set -Eeuo pipefail", text)
        self.assertIn("export PYTHONDONTWRITEBYTECODE=1", text)
        self.assertIn("PAI_PHASE1R_SOURCE_COMMIT", text)
        self.assertIn(f"readonly REQUIRED_SOURCE_COMMIT={EXPECTED_SCIENCE_COMMIT}", text)
        self.assertNotIn("REQUIRED_SOURCE_COMMIT=" + "0" * 40, text)
        self.assertIn("sha256sum --check --strict ../runtime/frozen_source.sha256", text)
        self.assertIn('visible_devices="$local_rank,0"', text)
        self.assertIn('CUDA_VISIBLE_DEVICES="$visible_devices" EGL_DEVICE_ID=0 MUJOCO_EGL_DEVICE_ID=0', text)
        self.assertIn("robosuite resolves MUJOCO_EGL_DEVICE_ID", text)
        self.assertIn("PAI_CANARY_EXPECTED_GPUS:-}\" != 4", text)
        self.assertIn("EXECUTION_SHARD=A0", text)
        self.assertIn("EXECUTION_SHARD=B1", text)
        self.assertIn("git -C \"$REPO\" archive \"$SOURCE_COMMIT\"", text)
        self.assertIn("collect-natural", text)
        # Validators reject relative paths as noncanonical.  The PAI launcher
        # must use the same artifact-rooted paths as the CPU preseed helper,
        # including for task-mapping equality and blinded calibration checks.
        for relative in (
            "configs/stage_r_phase1r_protocol.json",
            "configs/stage_r_phase1r_shards.json",
            "results/stage_r/phase1r/selection",
            "results/stage_r/phase1r/controls",
            "results/stage_r/phase1r/calibration",
        ):
            self.assertIn(f'\"$ARTIFACT_DIR/frozen_source/{relative}\"', text)
        self.assertIn("CHECKPOINT_ATTESTATION_SHA256=d050805b0c1e9e8d8e879c7443bb10504859c654d0ba031bbbc6ce3635b02fca", text)
        self.assertIn(
            "CHECKPOINT_VALIDATOR_SHA256=1b32a626d34bcb25bd81927f24c44579686d2945ab4f36525d1bad1c6dc639c4",
            text,
        )
        self.assertEqual(
            hashlib.sha256(CHECKPOINT_VALIDATOR.read_bytes()).hexdigest(),
            "1b32a626d34bcb25bd81927f24c44579686d2945ab4f36525d1bad1c6dc639c4",
        )
        subprocess.run(["python3", "-m", "py_compile", str(CHECKPOINT_VALIDATOR)], check=True)
        validator_text = CHECKPOINT_VALIDATOR.read_text(encoding="utf-8")
        self.assertIn(
            "frozen_full_content_attestation_plus_exact_metadata_and_content_probes",
            validator_text,
        )
        self.assertIn(
            "checkpoint file inventory drifted from full-content attestation",
            validator_text,
        )
        helper_contracts = (
            (
                FROZEN_SOURCE_VALIDATOR,
                "9dafe2e0bf089ffd788b81c7e4baf41c2b90faae4cdad73e2e11a107bab62775",
            ),
            (
                SMALL_PREFLIGHT_VALIDATOR,
                "5eb36792ff6cabe8ed6f94c142864faf1173ed52fe917b5e4bd265347b1a0fe2",
            ),
            (
                PREREQUISITE_VALIDATOR,
                "9c45933f5052684fad52a90de1e05a01594db555cf9aff45b68680995e510c46",
            ),
        )
        for helper, expected_sha in helper_contracts:
            with self.subTest(helper=helper.name):
                self.assertEqual(hashlib.sha256(helper.read_bytes()).hexdigest(), expected_sha)
                self.assertIn(expected_sha, text)
                subprocess.run(["python3", "-m", "py_compile", str(helper)], check=True)
        self.assertIn("runtime/frozen_source_verified.json", text)
        extra_helpers = (
            (PAI / "stage_r_phase1r_task_mapping.py", "e0dff96a034122719794441d02ba177c7f82564aae088c236eca2c5a0943f671"),
            (PAI / "validate_stage_r_phase1r_preflight_bundle.py", "2fc83d485ebc5634e01ce3933ff9b72837b1f0a142aea292e093cfd91aeddbfc"),
        )
        for helper, expected_sha in extra_helpers:
            with self.subTest(helper=helper.name):
                self.assertEqual(hashlib.sha256(helper.read_bytes()).hexdigest(), expected_sha)
                self.assertIn(expected_sha, text)
                subprocess.run(["python3", "-m", "py_compile", str(helper)], check=True)
        preparer = PAI / "prepare_stage_r_phase1r_preflight.sh"
        subprocess.run(["bash", "-n", str(preparer)], check=True)
        self.assertIn("outcome_blind_cpu_preseeded_preflight", preparer.read_text(encoding="utf-8"))
        self.assertNotIn("digest = hashlib.sha256(path.read_bytes()).hexdigest()", text)
        self.assertIn("COMPLETED_EVALUATION_RESULT.json", text)
        self.assertIn("SHA256SUMS", text)
        self.assertIn("CHECKPOINT_1_PENDING_GLOBAL_MERGE", text)
        self.assertNotIn("phase0r_parallel", text)
        # Natural workers may read-only validate pre-committed controls, but
        # they must never create controls or a calibration artifact.
        self.assertNotIn("collect_control_bundle", text)
        self.assertNotIn("calibrate_phase1r", text)
        self.assertNotIn(" controls --", text)
        self.assertNotIn(" calibrate --", text)

    def test_exact_templates(self) -> None:
        self.assertTrue(LAUNCHER.stat().st_mode & 0o111)
        for path in TEMPLATES:
            with self.subTest(path=path.name):
                manifest = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(manifest["schema_version"], 2)
                self.assertEqual(manifest["kind"], "pytorchjob")
                self.assertEqual(manifest["resource_alias"], "idle-a800-wallx-plug-native5-4gpu")
                self.assertEqual(manifest["workspace_id"], "179169")
                self.assertEqual(manifest["network"], {})
                self.assertEqual(
                    [(item["id"], item["mount_path"]) for item in manifest["storage"]["data_sources"]],
                    list(EXPECTED_MOUNTS),
                )
                self.assertEqual(
                    manifest["storage"]["output_root"],
                    "/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_r/phase1r",
                )
                worker = manifest["worker"]
                self.assertEqual(worker["count"], 1)
                self.assertEqual(worker["gpu"], 4)
                self.assertEqual(worker["cpu"], 46)
                self.assertEqual(worker["memory"], "800Gi")
                self.assertEqual(worker["shared_memory"], "800Gi")
                runtime = manifest["runtime"]
                self.assertEqual(
                    runtime["command_file"],
                    "/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/R142-FP-11-Bottleneck-Local-Candidate-Split/pai/run_stage_r_phase1r_idle_4gpu.sh",
                )
                self.assertEqual(runtime["write_paths"], ["{{ARTIFACT_DIR}}"])
                self.assertEqual(runtime["output_mode"], "resume")
                self.assertIs(runtime["create_artifact_dir"], True)
                self.assertIs(runtime["recursive_repair"], False)
                self.assertEqual(runtime["identity_mechanism"], "controller_inline_bootstrap_then_setpriv")
                self.assertEqual(runtime["uid"], 2254)
                self.assertEqual(runtime["gid"], 2254)
                self.assertEqual(runtime["pod_env"], {})
                self.assertEqual(runtime["secret_env_names"], [])
                self.assertNotIn("root_symlinks", runtime)
                self.assertEqual(manifest["fault_tolerance"]["aimaster_args"], EXPECTED_AIMASTER)
                self.assertEqual(manifest["fault_tolerance"]["maximum_platform_restarts"], 50)
                self.assertEqual(manifest["fault_tolerance"]["launcher_attempts"], 1)
                self.assertEqual(manifest["fault_tolerance"]["retry_sleep_seconds"], 30)
                self.assertIs(manifest["fault_tolerance"]["pai_automatic_fault_tolerance"], True)
                evidence = manifest["evidence"]
                execution_shard = evidence["execution_shard"]
                expected_ranges = {"A0": [0, 3], "A1": [4, 7], "B0": [8, 11], "B1": [12, 15]}
                expected_counts = {"A0": 7, "A1": 10, "B0": 11, "B1": 12}
                self.assertEqual(evidence["idle_4gpu_contract"], "generic_formal_idle_4gpu_v1")
                self.assertEqual(evidence["workload_type"], "evaluation")
                self.assertTrue(evidence["kind"].endswith("_formal_evaluation"))
                self.assertEqual(evidence["success_gate"], "persisted_completed_evaluation_result")
                self.assertEqual(evidence["validated_payload_sha256"], EXPECTED_LAUNCHER_SHA)
                self.assertIs(evidence["require_actual_idle"], True)
                self.assertIs(evidence["pai_probe_created"], False)
                self.assertIs(evidence["contract_ready"], True)
                self.assertEqual(evidence["expected_first_work_uid"], 2254)
                self.assertEqual(evidence["expected_first_work_gid"], 2254)
                self.assertEqual(evidence["global_rank_range"], expected_ranges[execution_shard])
                self.assertEqual(evidence["natural_tasks"], expected_counts[execution_shard])
                self.assertEqual(evidence["execution_config_sha256"], hashlib.sha256(EXECUTION.read_bytes()).hexdigest())
                self.assertIs(evidence["no_controls_regeneration"], True)
                self.assertIs(evidence["no_calibration_regeneration"], True)
                self.assertIs(evidence["no_unblinding"], True)
                self.assertEqual(
                    evidence["first_work_evidence_path"],
                    "{{ARTIFACT_DIR}}/FIRST_WORK.json",
                )
                self.assertEqual(
                    evidence["completed_evidence_path"],
                    "{{ARTIFACT_DIR}}/COMPLETED_EVALUATION_RESULT.json",
                )
                tags = manifest["submission"]["tags"]
                self.assertEqual(tags["managed_by"], "pai-job-registry")
                self.assertEqual(tags["purpose"], "formal-evaluation")
                self.assertEqual(tags["hardware"], "4xa800-idle")
                self.assertEqual(tags["resource_pool"], "idle-a800")
                self.assertEqual(tags["task"], evidence["task_id"])
                self.assertEqual(tags["model"], evidence["model_id"])
                self.assertEqual(manifest["submission"]["priority"], 9)
                self.assertIs(manifest["submission"]["disable_ecs_stock_check"], True)
                self.assertEqual(manifest["submission"]["job_reserved_policy"], "")
                self.assertEqual(manifest["submission"]["job_reserved_minutes"], 0)
                self.assertEqual(manifest["submission"]["job_max_running_time_minutes"], 0)

    def test_execution_split_exactly_preserves_frozen_scientific_mapping(self) -> None:
        execution = json.loads(EXECUTION.read_text(encoding="utf-8"))
        scientific = json.loads(SCIENTIFIC_SHARDS.read_text(encoding="utf-8"))
        expected = {
            "A0": ("A", list(range(0, 4)), 7),
            "A1": ("A", list(range(4, 8)), 10),
            "B0": ("B", list(range(8, 12)), 11),
            "B1": ("B", list(range(12, 16)), 12),
        }
        execution_names = []
        scientific_names = []
        for logical in ("A", "B"):
            rank_tasks = scientific["shards"][logical]["rank_tasks"]
            for rank in scientific["shards"][logical]["global_ranks"]:
                scientific_names.extend(rank_tasks[str(rank)])
        for name, (logical, ranks, task_count) in expected.items():
            self.assertEqual(
                execution["execution_shards"][name],
                {"logical_shard": logical, "global_ranks": ranks},
            )
            rank_tasks = scientific["shards"][logical]["rank_tasks"]
            names = [task for rank in ranks for task in rank_tasks[str(rank)]]
            self.assertEqual(len(names), task_count)
            execution_names.extend(names)
        self.assertEqual(execution_names, scientific_names)
        self.assertEqual(len(execution_names), 40)
        self.assertEqual(len(set(execution_names)), 40)

    def test_checkpoint_attestation_is_frozen_and_content_complete(self) -> None:
        self.assertEqual(
            hashlib.sha256(CHECKPOINT_ATTESTATION.read_bytes()).hexdigest(),
            "d050805b0c1e9e8d8e879c7443bb10504859c654d0ba031bbbc6ce3635b02fca",
        )
        attestation = json.loads(CHECKPOINT_ATTESTATION.read_text(encoding="utf-8"))
        self.assertEqual(attestation["marker_type"], "full_content_checkpoint_attestation")
        self.assertEqual(attestation["tree_sha256"], "42d571bd87f05f1182810f5a8bfa6d084c0d0dd277aff739bcf8f69868e6fb99")
        self.assertEqual(attestation["file_count"], 16)
        self.assertEqual(attestation["bytes"], 12439085481)
        self.assertEqual(len(attestation["files"]), 16)
        self.assertEqual(len({row["path"] for row in attestation["files"]}), 16)
        for row in attestation["files"]:
            self.assertEqual(len(row["sha256"]), 64)
            self.assertEqual(len(row["head_sha256"]), 64)
            self.assertEqual(len(row["tail_sha256"]), 64)

    def test_merge_utility_is_outcome_blind_and_fail_closed(self) -> None:
        utility = PAI / "merge_stage_r_phase1r_natural_shards.py"
        self.assertTrue(utility.is_file())
        text = utility.read_text(encoding="utf-8")
        self.assertIn("STAGE_R_PHASE1R_NATURAL_MERGE_COMPLETE_VALIDATED", text)
        self.assertIn("PAI_TERMINAL_COMPLETION.json", text)
        self.assertIn("outcome_blind", text)
        self.assertNotIn("os.link", text)
        self.assertIn("shutil.copy2", text)
        for shard in ("a0", "a1", "b0", "b1"):
            self.assertIn(f"--expected-job-id-{shard}", text)
            self.assertIn(f'"--expected-job-id-{shard}", required=True', text)
        self.assertNotIn("baseline_success", text)
        self.assertNotIn("success]", text)
        self.assertIn("execution shard {execution_shard} marker mapping mismatch", text)
        self.assertIn('"task_count": expected_task_count', text)
        subprocess.run(["python3", "-m", "py_compile", str(utility)], check=True)

    def test_terminal_sealer_requires_exact_succeeded_idle_readback(self) -> None:
        utility = PAI / "seal_stage_r_phase1r_terminal.py"
        self.assertTrue(utility.is_file())
        text = utility.read_text(encoding="utf-8")
        self.assertIn('job.get("Status") != "Succeeded"', text)
        self.assertIn('"GPU": "4"', text)
        self.assertIn('"CPU": "46"', text)
        self.assertIn('"Memory": "800Gi"', text)
        self.assertIn('"SharedMemory": "800Gi"', text)
        self.assertIn('"AcceptQuotaOverSold"', text)
        self.assertIn('"PAI_JOB_TERMINAL_READBACK.json"', text)
        self.assertIn('"completion_marker_sha256"', text)
        self.assertIn("write_exhaustive_sums(root)", text)
        subprocess.run(["python3", "-m", "py_compile", str(utility)], check=True)


if __name__ == "__main__":
    unittest.main()
