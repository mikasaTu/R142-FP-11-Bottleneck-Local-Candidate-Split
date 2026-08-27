#!/usr/bin/env python3
"""Static contract tests for the two natural Phase-1R idle shard manifests."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PAI = ROOT / "pai"
LAUNCHER = PAI / "run_stage_r_phase1r_idle_8gpu.sh"
TEMPLATES = (
    PAI / "r142_stage_r_phase1r_natural_shard_a_idle8.json",
    PAI / "r142_stage_r_phase1r_natural_shard_b_idle8.json",
)
EXPECTED_MOUNTS = (
    ("d-mkixtohdn75dp8x9tb", "/mnt/cpfs/zbl-cpfs-new/USERS/leon"),
    ("d-36p023eg0f2vuqny8y", "/mnt/cpfs/zbl-cpfs-new/CKPT/leon"),
    ("d-ejgj2ej7io1t2t32uc", "/mnt/cpfs/zbl-cpfs-new/dataset/leon"),
)
EXPECTED_AIMASTER = (
    "--job-execution-mode=Sync --enable-job-restart=True "
    "--max-num-of-job-restart=50 --fault-tolerant-policy=OnFailure"
)
EXPECTED_LAUNCHER_SHA = hashlib.sha256(LAUNCHER.read_bytes()).hexdigest()


class Phase1RPAIContractTest(unittest.TestCase):
    def test_launcher_is_strict_bash_and_bound_to_runtime_source_pin(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertTrue(text.startswith("#!/usr/bin/env bash"))
        self.assertIn("set -Eeuo pipefail", text)
        self.assertIn("PAI_PHASE1R_SOURCE_COMMIT", text)
        self.assertIn("readonly REQUIRED_SOURCE_COMMIT=0000000000000000000000000000000000000000", text)
        self.assertIn('visible_devices="$local_rank,0"', text)
        self.assertIn('CUDA_VISIBLE_DEVICES="$visible_devices" EGL_DEVICE_ID=0 MUJOCO_EGL_DEVICE_ID=0', text)
        self.assertIn("robosuite resolves MUJOCO_EGL_DEVICE_ID", text)
        self.assertIn("git -C \"$REPO\" archive \"$SOURCE_COMMIT\"", text)
        self.assertIn("collect-natural", text)
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
                self.assertEqual(manifest["resource_alias"], "idle-a800")
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
                self.assertEqual(worker["gpu"], 8)
                self.assertEqual(worker["cpu"], 88)
                self.assertEqual(worker["memory"], "1525Gi")
                self.assertEqual(worker["shared_memory"], "1525Gi")
                runtime = manifest["runtime"]
                self.assertEqual(
                    runtime["command_file"],
                    "/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/R142-FP-11-Bottleneck-Local-Candidate-Split/pai/run_stage_r_phase1r_idle_8gpu.sh",
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
                shard = "a" if "_shard_a_" in evidence["kind"] else "b"
                self.assertEqual(evidence["idle_8gpu_contract"], "generic_formal_idle_8gpu_v1")
                self.assertEqual(evidence["workload_type"], "evaluation")
                self.assertTrue(evidence["kind"].endswith("_formal_evaluation"))
                self.assertEqual(evidence["success_gate"], "persisted_completed_evaluation_result")
                self.assertEqual(evidence["validated_payload_sha256"], EXPECTED_LAUNCHER_SHA)
                self.assertIs(evidence["require_actual_idle"], True)
                self.assertIs(evidence["pai_probe_created"], False)
                self.assertIs(evidence["contract_ready"], True)
                self.assertEqual(evidence["expected_first_work_uid"], 2254)
                self.assertEqual(evidence["expected_first_work_gid"], 2254)
                self.assertEqual(evidence["global_rank_range"], [0, 7] if shard == "a" else [8, 15])
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
                self.assertEqual(tags["hardware"], "8xa800-idle")
                self.assertEqual(tags["resource_pool"], "idle-a800")
                self.assertEqual(tags["task"], evidence["task_id"])
                self.assertEqual(tags["model"], evidence["model_id"])
                self.assertEqual(manifest["submission"]["priority"], 9)
                self.assertIs(manifest["submission"]["disable_ecs_stock_check"], True)
                self.assertEqual(manifest["submission"]["job_reserved_policy"], "")
                self.assertEqual(manifest["submission"]["job_reserved_minutes"], 0)
                self.assertEqual(manifest["submission"]["job_max_running_time_minutes"], 0)

    def test_merge_utility_is_outcome_blind_and_fail_closed(self) -> None:
        utility = PAI / "merge_stage_r_phase1r_natural_shards.py"
        self.assertTrue(utility.is_file())
        text = utility.read_text(encoding="utf-8")
        self.assertIn("STAGE_R_PHASE1R_NATURAL_MERGE_COMPLETE_VALIDATED", text)
        self.assertIn("PAI_TERMINAL_COMPLETION.json", text)
        self.assertIn("outcome_blind", text)
        self.assertIn("os.link", text)
        self.assertIn("--expected-job-id-a", text)
        self.assertNotIn("baseline_success", text)
        self.assertNotIn("success]", text)
        subprocess.run(["python3", "-m", "py_compile", str(utility)], check=True)


if __name__ == "__main__":
    unittest.main()
