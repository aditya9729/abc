"""CPU orchestration doubles; no simulator, learner, or GPU evidence."""

import copy
import json
from argparse import Namespace
from pathlib import Path

import pytest

from abc_bench import runner


@pytest.mark.parametrize(
    "stage,worker_status",
    [
        ("collect", "collected_unreviewed"),
        ("train", "completed"),
        ("train", "budget_stopped"),
    ],
)
def test_sadhana_worker_uses_shared_budget_and_resolved_config(
    monkeypatch, tmp_path, stage, worker_status
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
                "status": worker_status,
                "independent_admission_required": stage == "collect",
                "metrics": {"sim_steps": 32, "critic_updates": 0, "actor_updates": 0},
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
    assert receipt["status"] == (
        "partial" if worker_status == "budget_stopped" else "completed"
    )
    assert receipt["worker_status"] == worker_status
    assert receipt["independent_admission_required"] == (stage == "collect")
    assert receipt["training_target_reached"] == (
        stage == "train" and worker_status == "completed"
    )
    assert receipt["training_stage"] == stage
    assert receipt["task"] == "fixture-task"
    assert receipt["checkpoint_path"] == "/fixture/base.pt"
    assert receipt["budget"]["steps"] == (64 if stage == "collect" else 128)
    assert receipt["metrics"]["simulation_steps"] == 32
    assert json.loads(config_path.read_text())["max_wall_s"] == 500
    ledger = json.loads((campaign / "gpu_budget.json").read_text())
    assert 7100 <= ledger["charged_seconds"] < 7101
    assert "active_gpu_uuid" not in ledger
    assert publications[-1]["status"] == receipt["status"]


@pytest.mark.parametrize(
    "stage,status",
    [
        ("native_check", "native_check_completed"),
        ("train", "budget_stopped"),
        ("evaluate", "completed"),
        ("native_check", "failed"),
    ],
)
def test_qf3_vla_dispatch_shares_existing_lease_and_preserves_paper_recipe(
    monkeypatch, tmp_path, stage, status
):
    campaign = tmp_path / "campaign"
    monkeypatch.setattr(runner, "RESULTS", campaign)
    runner.write_json(
        campaign / "gpu_budget.json",
        {"limit_seconds": 7200, "charged_seconds": 7100, "gpu": 0},
    )
    gpu_uuid = "GPU-00000000-0000-0000-0000-000000000000"
    monkeypatch.setattr(
        runner, "inspect_reserved_gpu", lambda ordinal: {"uuid": gpu_uuid}
    )
    monkeypatch.setattr(runner, "execution_provenance", lambda: {"evidence": "fixture"})
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
        "base": {
            "checkpoint_path": "/fixture/vla.pt",
            "task_id": "put_plastic_bottles_in_bin",
            "seed": 901,
        },
        "seed": 903,
        "training": {
            "target_control_steps": 400000,
            "train_worlds": 16,
            "critic_updates_per_iteration": 1600,
            "actor_updates_per_iteration": 200,
        },
        "evaluation": {
            "seeds": [10001, 10002],
            "sampler_seed": 91001,
            "heads": ["frozen"],
            "max_control_steps_per_world": None,
        },
    }
    path = tmp_path / "qf3.json"
    runner.write_json(path, config)

    def worker(command, log_path, timeout, *, gpu_uuid):
        assert command[1:3] == ["-m", "nrh.qf3_training"]
        assert timeout == 100
        child = json.loads(Path(command[command.index("--config") + 1]).read_text())
        assert child["campaign_directory"] == str(campaign.resolve())
        assert child["max_wall_s"] == 40
        assert child["training"] == config["training"]
        ledger = json.loads((campaign / "gpu_budget.json").read_text())
        assert (
            ledger["charged_seconds"] == 7200 and ledger["active_gpu_uuid"] == gpu_uuid
        )
        out = Path(command[command.index("--out") + 1])
        runner.write_json(
            out / "receipt.json",
            {
                "stage": stage,
                "status": status,
                "metrics": {
                    "simulation_steps": 30,
                    "critic_updates": 2,
                    "actor_updates": 1,
                },
                **(
                    {"error": {"type": "ContractError", "message": "metadata differs"}}
                    if status == "failed"
                    else {}
                ),
            },
        )
        return 1 if status == "failed" else 0

    monkeypatch.setattr(runner, "execute_command", worker)
    args = Namespace(
        results=tmp_path / "results",
        timeout_seconds=500,
        algorithm="qf3-vla",
        training_config=path,
        checkpoint=tmp_path / "unused.pt",
        task="unused",
        worlds=1,
        chunks=2,
        seed=0,
        video=False,
    )
    receipt = runner.run_baseline(args)
    if status == "failed":
        assert receipt["status"] == "failed"
        assert receipt["worker_error"]["message"] == "metadata differs"
        assert "metadata differs" in receipt["blockers"]
        artifact = next(
            a
            for a in receipt["artifacts"]
            if a["label"] == "Preserved worker failure evidence"
        )
        assert artifact["path"].endswith("training/receipt.json")
        assert len(receipt["worker_receipt_sha256"]) == 64
        ledger = json.loads((campaign / "gpu_budget.json").read_text())
        assert 7100 <= ledger["charged_seconds"] < 7101
        assert "active_parent_pid" not in ledger
        assert publications[-1]["status"] == "failed"
        return
    assert receipt["status"] == (
        "partial" if status == "budget_stopped" else "completed"
    )
    assert receipt["phase"] == "smoke"
    assert receipt["budget"]["steps"] == (2000 if stage == "evaluate" else 400000)
    assert receipt["training_target_control_steps"] == 400000
    assert receipt["seed"] == (None if stage == "evaluate" else 903)
    assert receipt["policy_seed"] == 901
    if stage == "evaluate":
        assert receipt["evaluation_horizon_steps"] == 1000
        assert receipt["evaluation_seeds"] == [10001, 10002]
        assert receipt["evaluation_heads"] == ["frozen"]
        assert receipt["evaluation_sampler_seed"] == 91001
    assert receipt["metrics"]["simulation_steps"] == 30
    assert receipt["metrics"]["updates"] == 2
    assert receipt["metrics"]["successes"] is receipt["metrics"]["episodes"] is None
    assert (
        receipt["task"] == "put_plastic_bottles_in_bin"
        and receipt["checkpoint_path"] == "/fixture/vla.pt"
    )
    assert json.loads(path.read_text()) == config
    ledger = json.loads((campaign / "gpu_budget.json").read_text())
    assert 7100 <= ledger["charged_seconds"] < 7101
    assert "active_parent_pid" not in ledger and "active_gpu_uuid" not in ledger
    assert publications[-1]["status"] == receipt["status"]
