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


@pytest.mark.parametrize(
    "module, task, embodiment, empty_child, explicit_scope",
    [
        (
            "expo_abc_adapter.stage",
            "put_plastic_bottles_in_bin",
            "native_yam",
            True,
            True,
        ),
        (
            "sadhana_fasttd3_hb.stage",
            "h1hand-stand-v0",
            "unitree_h1_with_hands",
            False,
            True,
        ),
        (
            "sadhana_fasttd3_hb.stage",
            "h1hand-stand-v0",
            "unitree_h1_with_hands",
            False,
            False,
        ),
    ],
)
def test_author_stage_keeps_output_contract_and_truthful_task(
    tmp_path, monkeypatch, module, task, embodiment, empty_child, explicit_scope
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
        "module": module,
        "arguments": [],
        "max_wall_s": 120,
    }
    if not explicit_scope:
        del job["claim_scope"]
    path = tmp_path / "job.json"
    path.write_text(json.dumps(job))

    def cpu_execution_double(command, log, timeout, *, gpu_uuid):
        assert gpu_uuid == "explicit-cpu-double"
        output = Path(command[-1])
        if empty_child:
            assert output.name == "expoft" and not output.exists()
            assert (output.parent / "job.json").is_file()
            output.mkdir()
        else:
            assert (output / "job.json").is_file()
            assert {item.name for item in output.iterdir()} == {"job.json"}
        (output / "receipt.json").write_text(
            json.dumps({"status": "completed", "metrics": {"actual_steps": 2}})
        )
        return 0

    monkeypatch.setattr(runner, "execute_command", cpu_execution_double)
    result = subset.run_job(
        argparse.Namespace(training_config=path, timeout_seconds=120)
    )
    assert result["status"] == "completed"
    if explicit_scope:
        assert result["claim_scope"] == claim
    else:
        assert result["claim_scope"].startswith("Bounded original FastTD3 Stand")
        assert (
            "no convergence, ABC transfer or paper reproduction claim"
            in result["claim_scope"]
        )
    assert result["task"] == task and result["embodiment"] == embodiment
    assert result["metrics"] == {"actual_steps": 2}
    ledger = json.loads((tmp_path / "gpu_budget.json").read_text())
    assert ledger["charged_seconds"] >= 17
    assert "active_parent_pid" not in ledger
