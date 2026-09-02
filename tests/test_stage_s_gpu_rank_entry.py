from __future__ import annotations

import os
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "stage_s_gpu_rank_entry.py"


def _module():
    return runpy.run_path(str(SCRIPT), run_name="stage_s_gpu_rank_entry_test")


def test_rank_selects_exact_allocated_device(monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_RANK", "2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5,7,9")
    selected = _module()["bind_local_rank"]()
    assert selected == "7"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "7"


def test_rank_binding_fails_closed_without_allocation(monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    try:
        _module()["bind_local_rank"]()
    except SystemExit as exc:
        assert "CUDA_VISIBLE_DEVICES" in str(exc)
    else:
        raise AssertionError("missing GPU allocation must fail closed")
