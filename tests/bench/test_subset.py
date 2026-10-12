"""Boundary failures must be rejected before the GPU or ledger is touched."""

import fcntl
import json
import os
import sys
from pathlib import Path

import pytest

from abc_bench.subset import load_job, require_subset_lease


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": True},
        {"module": "arbitrary.module"},
        {"max_wall_s": 0},
        {"max_wall_s": float("inf")},
        {"max_wall_s": True},
        {"max_wall_s": 7201},
        {"python": "relative/python"},
        {"arguments": ["--output", "/other"]},
        {"arguments": [17]},
        {"claim_scope": False},
        {"claim_scope": "   "},
    ],
)
def test_invalid_job_is_rejected(tmp_path, change):
    job = {
        "schema_version": 1,
        "label": "original VLA smoke",
        "python": sys.executable,
        "module": "abc_author_subset.host",
        "arguments": ["--mode", "smoke"],
        "max_wall_s": 120,
        **change,
    }
    path = tmp_path / "job.json"
    path.write_text(json.dumps(job))
    with pytest.raises(ValueError):
        load_job(path)


def test_worker_requires_a_held_ancestor_lease(tmp_path, monkeypatch):
    import abc_bench.runner

    monkeypatch.setattr(abc_bench.runner, "RESULTS", tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "test-reserved-uuid")
    monkeypatch.setenv("MUJOCO_GL", "disable")
    (tmp_path / "gpu_budget.json").write_text(
        json.dumps(
            {
                "gpu": 0,
                "active_parent_pid": os.getppid(),
                "active_gpu_uuid": "test-reserved-uuid",
            }
        )
    )
    lock_path = tmp_path / ".gpu-budget.lock"
    lock_path.touch()
    with pytest.raises(RuntimeError, match="not held"):
        require_subset_lease()
    with lock_path.open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        require_subset_lease()
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "wrong-uuid")
        with pytest.raises(RuntimeError, match="visibility differs"):
            require_subset_lease()


def test_original_pi05_stage_keeps_empty_output_and_truthful_scope(
    tmp_path, monkeypatch
):
    """Explicit CPU execution double checks cross-runtime receipt admission."""
    import argparse

    from abc_bench import runner, subset

    monkeypatch.setattr(runner, "RESULTS", tmp_path)
    (tmp_path / "gpu_budget.json").write_text(
        json.dumps({"gpu": 0, "charged_seconds": 17, "limit_seconds": 1000})
    )
    monkeypatch.setattr(
        subset,
        "inspect_reserved_gpu",
        lambda _: {"uuid": "explicit-cpu-double", "ordinal": 0},
    )
    claim = "Original pi0.5 SFT runtime qualification; no simulator task score"
    job = {
        "schema_version": 1,
        "label": "original pi0.5 SFT qualification",
        "claim_scope": claim,
        "python": sys.executable,
        "module": "expo_abc_adapter.stage",
        "arguments": [],
        "max_wall_s": 120,
    }
    path = tmp_path / "job.json"
    path.write_text(json.dumps(job))

    def cpu_execution_double(command, log, timeout, *, gpu_uuid):
        assert gpu_uuid == "explicit-cpu-double"
        output = Path(command[-1])
        assert output.name == "expoft" and not output.exists()
        assert (output.parent / "job.json").is_file()
        output.mkdir()
        (output / "receipt.json").write_text(
            json.dumps({"status": "completed", "metrics": {"actual_steps": 2}})
        )
        return 0

    monkeypatch.setattr(runner, "execute_command", cpu_execution_double)
    result = subset.run_job(
        argparse.Namespace(training_config=path, timeout_seconds=120)
    )
    assert result["status"] == "completed" and result["claim_scope"] == claim
    assert result["metrics"] == {"actual_steps": 2}
    ledger = json.loads((tmp_path / "gpu_budget.json").read_text())
    assert ledger["charged_seconds"] >= 17
    assert "active_parent_pid" not in ledger
