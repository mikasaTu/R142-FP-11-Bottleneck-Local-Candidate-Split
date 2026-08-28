from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC = ROOT / "scripts" / "diagnose_stage_r_phase1r_env_reconstruction.py"


def _diagnostic_module():
    spec = importlib.util.spec_from_file_location("stage_r_phase1r_env_diagnostic", DIAGNOSTIC)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_target_only_lifecycle_strategies_select_the_only_environment() -> None:
    module = _diagnostic_module()
    assert module.target_environment_index("single", 27) == 0
    assert module.target_environment_index("prior-reset-only", 27) == 0
    assert module.target_environment_index("candidate-prior-lifecycle", 27) == 0
    assert module.target_environment_index("indexed-reset-target", 27) == 27
    assert module.target_environment_index("indexed-reset-all", 27) == 27
