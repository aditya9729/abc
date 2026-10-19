"""Actual canonical loader and process contract; explicit CPU execution double."""

import argparse
import json
import sys
from pathlib import Path

import pytest
from sadhana_fasttd3_wbc.caller import validate_job

from abc_bench import runner, subset

CONFIGS = Path(
    "/home/user/aditya/RL/sadhana-fasttd3-wbc-20261019/"
    "packages/r1lite-fasttd3-wbc/configs"
)


def job(phase="qualification"):
    return {
        "schema_version": 1,
        "python": sys.executable,
        "module": "sadhana_fasttd3_wbc.caller",
        "arguments": ["--config", str(CONFIGS / f"{phase}.json")],
        "max_wall_s": 1300 if phase == "qualification" else 28900,
        "label": "Original FastTD3 WBC " + phase,
        "task": "r1lite_moving_wbc_tracking",
        "embodiment": "r1lite",
        "claim_scope": "Original12-command scratch; no frozen task score yet",
    }


@pytest.mark.parametrize("phase", ["qualification", "reference"])
def test_actual_loader_accepts_both_finite_paths(tmp_path, phase):
    data = job(phase)
    path = tmp_path / "job.json"
    path.write_text(json.dumps(data))
    assert subset.load_job(path) == data
    with pytest.raises(ValueError, match="100s cleanup"):
        validate_job(data, data["max_wall_s"] - 1)
    data["max_wall_s"] += 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="at most"):
        subset.load_job(path)


@pytest.mark.parametrize(
    "change",
    [
        {"task": "remote_placement"},
        {"embodiment": "r1pro"},
        {"max_wall_s": True},
        {"arguments": []},
    ],
)
def test_actual_loader_rejects_wrong_task_or_unbounded_call(tmp_path, change):
    path = tmp_path / "job.json"
    path.write_text(json.dumps(job() | change))
    with pytest.raises(ValueError):
        subset.load_job(path)


def test_fresh_child_receipt_task_and_lease_cleanup(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RESULTS", tmp_path)
    (tmp_path / "gpu_budget.json").write_text(
        json.dumps(
            {
                "gpu": 0,
                "charged_seconds": 17,
                "limit_seconds": 2000,
            }
        )
    )
    monkeypatch.setattr(
        subset,
        "inspect_reserved_gpu",
        lambda _: {"uuid": "explicit-cpu-double", "ordinal": 0},
    )
    path = tmp_path / "job.json"
    path.write_text(json.dumps(job()))

    def cpu_double(command, log, timeout, *, gpu_uuid):
        assert gpu_uuid == "explicit-cpu-double" and timeout == 1300
        output = Path(command[-1])
        assert output.name == "fasttd3-wbc" and not output.exists()
        assert (output.parent / "job.json").is_file()
        output.mkdir()
        (output / "receipt.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "metrics": {"critic_updates": 2, "actor_updates": 1},
                }
            )
        )
        return 0

    monkeypatch.setattr(runner, "execute_command", cpu_double)
    result = subset.run_job(
        argparse.Namespace(training_config=path, timeout_seconds=1300)
    )
    assert result["status"] == "completed"
    assert result["task"] == "r1lite_moving_wbc_tracking"
    assert result["embodiment"] == "r1lite"
    assert result["phase"] == "original_fasttd3_wbc"
    assert result["metrics"] == {"critic_updates": 2, "actor_updates": 1}
    ledger = json.loads((tmp_path / "gpu_budget.json").read_text())
    assert ledger["charged_seconds"] >= 17 and "active_parent_pid" not in ledger
