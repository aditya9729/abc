"""Metadata and explicit CPU execution doubles; no simulator or learner imports."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pytest

from abc_bench import runner, subset


@pytest.fixture
def inputs(tmp_path):
    task = tmp_path / "task.json"
    task.write_text('{"scene": "explicit metadata double"}\n')
    config = {
        "schema_version": 1,
        "task_config": str(task),
        "total_timesteps": 12,
        "seed": 1,
        "timeout_s": 100,
        "phase": "original_fasttd3_r1lite",
        "author_source_commit": "dce87035e2d58608c015f01f3ee332d6ba84a86b",
        "adapter_version": "0.1.0",
    }
    worker = tmp_path / "worker.json"
    worker.write_text(json.dumps(config))
    job = {
        "schema_version": 1,
        "module": "sadhana_r1lite.fasttd3",
        "python": sys.executable,
        "label": "R1 Lite CPU boundary double",
        "task": "r1lite_remote_into_tray",
        "embodiment": "r1lite_2025",
        "arguments": ["--config", str(worker)],
        "max_wall_s": 160,
    }
    path = tmp_path / "job.json"
    path.write_text(json.dumps(job))
    return path, job, worker, config, task


@pytest.mark.parametrize("calls", [12, 10000])
def test_admits_exact_boundary_without_runtime_import(inputs, calls):
    path, job, worker, config, _ = inputs
    worker.write_text(json.dumps({**config, "total_timesteps": calls}))
    assert subset.load_job(path) == job
    assert "sadhana_r1lite" not in sys.modules


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": True},
        {"seed": True},
        {"seed": 2},
        {"total_timesteps": 11},
        {"total_timesteps": 10001},
        {"total_timesteps": 12.0},
        {"timeout_s": 0},
        {"timeout_s": 7101},
        {"timeout_s": float("nan")},
        {"timeout_s": float("inf")},
        {"timeout_s": True},
        {"phase": "other"},
        {"adapter_version": "0.2.0"},
        {"author_source_commit": "wrong"},
        {"extra": True},
        {"task_config": "task.json"},
    ],
)
def test_rejects_worker_drift(inputs, change):
    path, _, worker, config, _ = inputs
    worker.write_text(json.dumps({**config, **change}))
    with pytest.raises(ValueError):
        subset.load_job(path)


@pytest.mark.parametrize(
    "change",
    [
        {"task": "put_plastic_bottles_in_bin"},
        {"embodiment": "native_yam"},
        {"arguments": ["--help"]},
        {"arguments": ["--config", "worker.json"]},
        {"max_wall_s": 159},
        {"max_wall_s": 7201},
        {"max_wall_s": float("inf")},
        {"max_wall_s": True},
    ],
)
def test_rejects_job_scope_and_watchdog(inputs, change):
    path, job, _, _, _ = inputs
    path.write_text(json.dumps({**job, **change}))
    with pytest.raises(ValueError):
        subset.load_job(path)


@pytest.mark.parametrize("which", ["worker", "task"])
def test_rejects_config_symlink(inputs, which):
    path, _, worker, _, task = inputs
    selected = worker if which == "worker" else task
    target = selected.with_suffix(".original")
    selected.rename(target)
    selected.symlink_to(target)
    with pytest.raises(ValueError, match="absolute regular"):
        subset.load_job(path)


def test_effective_watchdog_rejected_before_ledger(inputs, monkeypatch, tmp_path):
    path, _, _, _, _ = inputs
    results = tmp_path / "not-created"
    monkeypatch.setattr(runner, "RESULTS", results)
    with pytest.raises(ValueError, match="Effective R1 Lite"):
        subset.run_job(argparse.Namespace(training_config=path, timeout_seconds=159))
    assert not results.exists()


def test_execution_uses_frozen_configs_and_correct_identity(
    inputs, monkeypatch, tmp_path
):
    path, _, worker, _, task = inputs
    original_worker, original_task = worker.read_bytes(), task.read_bytes()
    results = tmp_path / "results"
    results.mkdir()
    (results / "gpu_budget.json").write_text(
        json.dumps({"gpu": 0, "charged_seconds": 7, "limit_seconds": 1000})
    )
    monkeypatch.setattr(runner, "RESULTS", results)
    monkeypatch.setattr(
        subset, "inspect_reserved_gpu", lambda _: {"uuid": "cpu-double", "ordinal": 0}
    )

    def execute(command, log, timeout, *, gpu_uuid):
        assert gpu_uuid == "cpu-double" and timeout == 160
        config_path = Path(command[command.index("--config") + 1])
        snapshot = json.loads(config_path.read_text())
        task_path = Path(snapshot["task_config"])
        worker.write_text("changed after admission")
        task.write_text("changed after admission")
        assert task_path.read_bytes() == original_task
        assert config_path.is_absolute() and task_path.is_absolute()
        assert config_path.stat().st_mode & 0o222 == 0
        assert task_path.stat().st_mode & 0o222 == 0
        output = Path(command[-1])
        assert config_path.parent == output == task_path.parent
        (output / "receipt.json").write_text(
            json.dumps({"status": "completed", "metrics": {"calls": 12}})
        )
        return 0

    monkeypatch.setattr(runner, "execute_command", execute)
    result = subset.run_job(
        argparse.Namespace(training_config=path, timeout_seconds=160)
    )
    assert result["status"] == "completed"
    assert result["task"] == "r1lite_remote_into_tray"
    assert result["embodiment"] == "r1lite_2025"
    assert (
        result["config_snapshots"]["worker_original_sha256"]
        == hashlib.sha256(original_worker).hexdigest()
    )
    assert (
        result["config_snapshots"]["task_sha256"]
        == hashlib.sha256(original_task).hexdigest()
    )
    assert result["claim_scope"].startswith("Original FastTD3 learner")
