"""CPU orchestration doubles; no simulator, learner, or GPU evidence."""

import copy
import json
from argparse import Namespace
from pathlib import Path

import pytest

from abc_bench import runner


@pytest.mark.parametrize("stage", ["collect", "train"])
def test_sadhana_worker_uses_shared_budget_and_resolved_config(
    monkeypatch, tmp_path, stage
):
    campaign = tmp_path / "campaign"
    monkeypatch.setattr(runner, "RESULTS", campaign)
    runner.write_json(
        campaign / "gpu_budget.json",
        {
            "limit_seconds": 7200,
            "charged_seconds": 7100,
            "gpu": 0,
        },
    )
    monkeypatch.setattr(runner, "execution_provenance", lambda: {"evidence": "fixture"})
    gpu_uuid = "GPU-00000000-0000-0000-0000-000000000000"
    monkeypatch.setattr(
        runner, "inspect_reserved_gpu", lambda ordinal: {"uuid": gpu_uuid}
    )
    publications = []
    monkeypatch.setattr(
        runner,
        "publish",
        lambda receipt, results: publications.append(copy.deepcopy(receipt)),
    )
    config = {
        "schema_version": 1,
        "stage": stage,
        "device": "cuda:0",
        "max_wall_s": 500,
        "base": {"checkpoint_path": "/fixture/base.pt", "task_id": "fixture-task"},
        "collect": {"seeds": [701, 702], "episode_steps": 32},
        "train": {"target_steps": 128},
    }
    config_path = tmp_path / "input.json"
    runner.write_json(config_path, config)

    def fixture_worker(command, log_path, timeout, *, gpu_uuid):
        assert timeout == 100
        assert gpu_uuid == "GPU-00000000-0000-0000-0000-000000000000"
        assert command[1:3] == ["-m", "nrh.abc_training"]
        child_config = json.loads(
            Path(command[command.index("--config") + 1]).read_text()
        )
        assert child_config["max_wall_s"] == 40
        ledger = json.loads((campaign / "gpu_budget.json").read_text())
        assert ledger["charged_seconds"] == 7200
        assert ledger["active_gpu_uuid"] == gpu_uuid
        destination = Path(command[command.index("--out") + 1])
        runner.write_json(
            destination / "receipt.json",
            {
                "stage": stage,
                "status": "completed",
                "metrics": {"simulation_steps": 32},
            },
        )
        return 0

    monkeypatch.setattr(runner, "execute_command", fixture_worker)
    args = Namespace(
        results=tmp_path / "results",
        timeout_seconds=500,
        algorithm="sustained-resfit",
        training_config=config_path,
        checkpoint=tmp_path / "unused.pt",
        task="unused",
        worlds=1,
        chunks=2,
        seed=0,
        video=False,
    )
    receipt = runner.run_baseline(args)
    assert receipt["status"] == "completed"
    assert receipt["training_stage"] == stage
    assert receipt["task"] == "fixture-task"
    assert receipt["checkpoint_path"] == "/fixture/base.pt"
    assert receipt["budget"]["steps"] == (64 if stage == "collect" else 128)
    assert receipt["metrics"]["simulation_steps"] == 32
    assert json.loads(config_path.read_text())["max_wall_s"] == 500
    ledger = json.loads((campaign / "gpu_budget.json").read_text())
    assert 7100 <= ledger["charged_seconds"] < 7101
    assert "active_gpu_uuid" not in ledger
    assert publications[-1]["status"] == "completed"
