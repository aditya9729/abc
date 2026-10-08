"""CPU coordinator doubles only; no native learner or GPU evidence."""

import copy
import json
from argparse import Namespace
from pathlib import Path

import pytest

from abc_bench import expoft_dispatch, runner


def input_config():
    return {
        "schema_version": 1,
        "domain": "sim",
        "profile": expoft_dispatch.PROFILE,
        "mode": "paused_simulation_fixed_tick",
        "episodes": 12,
        "max_wall_s": 500,
        "seed": 903,
        "sampler_seed": 901,
        "artifacts": {"checkpoint_path": "/fixture/candidate.pt"},
    }


def worker_record(config, status="completed"):
    completed = 12 if status == "completed" else 11
    discarded = 0 if status == "completed" else 3
    return {
        "revision": expoft_dispatch.REVISION,
        "domain": "sim",
        "status": status,
        "identity": {
            "revision": expoft_dispatch.REVISION,
            "config": {k: v for k, v in config.items() if k != "max_wall_s"},
        },
        "clock_mode": config["mode"],
        "independent_native_admission": False,
        "author_reproduction": False,
        "wall_time_reactivity_verified": False,
        "wrapper": {
            "completed_episodes": completed,
            "accepted_update_calls": 3,
            "phase": "ready" if status == "completed" else "collecting",
        },
        "invocation": {
            "initial_wrapper": {"completed_episodes": 10, "accepted_update_calls": 1},
            "physical_control_steps": (completed - 10) * 30 + discarded,
            "accepted_completed_control_steps": (completed - 10) * 30,
            "discarded_partial_control_steps": discarded,
        },
        "accepted_update_groups": [
            {"ordinal": i, "pin": {"fixture": True}} for i in range(3)
        ],
        "successful_base_updates": 0,
        "learning_status": "awaiting_success_imitation_data",
        "error": "fixture clock miss" if status != "completed" else None,
    }


@pytest.mark.parametrize(
    "status", ["completed", "budget_stopped", "deadline_missed_partial", "failed"]
)
def test_worker_dispatch_reserves_shared_lease_and_preserves_partial_counts(
    monkeypatch, tmp_path, status
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
    config = input_config()
    config_path = tmp_path / "input.json"
    runner.write_json(config_path, config)

    def fixture_worker(command, log_path, timeout, *, gpu_uuid):
        assert command[1:3] == ["-m", "nrh.realtime_expoft_run"]
        assert timeout == 100 and gpu_uuid == "GPU-00000000-0000-0000-0000-000000000000"
        assert command[command.index("--campaign-directory") + 1] == str(
            campaign.resolve()
        )
        assert command[command.index("--resume-pin") + 1] == str(
            (tmp_path / "resume-pin.json").resolve()
        )
        child = json.loads(Path(command[command.index("--config") + 1]).read_bytes())
        assert child["max_wall_s"] == 40
        assert all(child[k] == v for k, v in config.items() if k != "max_wall_s")
        ledger = json.loads((campaign / "gpu_budget.json").read_bytes())
        assert (
            ledger["charged_seconds"] == 7200 and ledger["active_gpu_uuid"] == gpu_uuid
        )
        assert type(ledger["active_parent_pid"]) is int
        destination = Path(command[command.index("--out") + 1])
        runner.write_json(destination / "receipt.json", worker_record(child, status))
        return 1 if status == "failed" else 0

    monkeypatch.setattr(runner, "execute_command", fixture_worker)
    args = Namespace(
        results=tmp_path / "results",
        timeout_seconds=500,
        algorithm="realtime-expoft-abc",
        training_config=config_path,
        checkpoint=tmp_path / "unused.pt",
        task="unused",
        worlds=1,
        chunks=2,
        seed=0,
        video=False,
        expoft_resume_pin=tmp_path / "resume-pin.json",
    )
    receipt = runner.run_baseline(args)
    expected = (
        "failed"
        if status == "failed"
        else "completed"
        if status == "completed"
        else "partial"
    )
    assert receipt["status"] == expected and receipt["worker_status"] == status
    assert receipt["phase"] == "training" and receipt["training_stage"] == "train"
    assert (
        receipt["task"] == "put_plastic_bottles_in_bin"
        and receipt["checkpoint_path"] == config["artifacts"]["checkpoint_path"]
    )
    assert receipt["metrics"]["simulation_steps"] == (
        60 if status == "completed" else 33
    )
    assert receipt["metrics"]["accepted_completed_control_steps"] == (
        60 if status == "completed" else 30
    )
    assert receipt["metrics"]["discarded_partial_control_steps"] == (
        0 if status == "completed" else 3
    )
    assert (
        receipt["metrics"]["updates"] == 2
        and receipt["metrics"]["cumulative_accepted_update_calls"] == 3
    )
    assert (
        receipt["metrics"]["successes"] is None
        and receipt["metrics"]["episodes"] is None
    )
    assert receipt["learning_status"] == "awaiting_success_imitation_data"
    assert receipt["episode_schedule_completed"] is (status == "completed")
    assert receipt["independent_admission_required"] is True
    assert receipt["wall_time_reactivity_verified"] is False
    assert config == json.loads(config_path.read_bytes())
    ledger = json.loads((campaign / "gpu_budget.json").read_bytes())
    assert (
        7100 <= ledger["charged_seconds"] < 7101 and "active_parent_pid" not in ledger
    )
    assert publications[-1]["status"] == expected
    assert any(
        a["path"].endswith("training/receipt.json") for a in receipt["artifacts"]
    )
    if status == "failed":
        assert receipt["worker_error"] == "fixture clock miss"
        assert len(receipt["worker_receipt_sha256"]) == 64


@pytest.mark.parametrize(
    "fault",
    [
        "controls",
        "counter_bool",
        "regressed",
        "group_order",
        "config_type",
        "clock",
        "self_admit",
        "unfinished",
        "unknown",
        "list_learning",
    ],
)
def test_invalid_worker_evidence_has_no_counter_admission(fault):
    config = input_config()
    worker = worker_record(config)
    if fault == "controls":
        worker["invocation"]["discarded_partial_control_steps"] = 1
    elif fault == "counter_bool":
        worker["invocation"]["discarded_partial_control_steps"] = False
    elif fault == "regressed":
        worker["invocation"]["initial_wrapper"]["accepted_update_calls"] = 4
    elif fault == "group_order":
        worker["accepted_update_groups"][1]["ordinal"] = 0
    elif fault == "config_type":
        worker["identity"]["config"]["seed"] = 903.0
    elif fault == "clock":
        worker["clock_mode"] = "strict_wall"
    elif fault == "self_admit":
        worker["independent_native_admission"] = True
    elif fault == "unfinished":
        worker["wrapper"]["phase"] = "pending_updates"
    elif fault == "unknown":
        worker["status"] = []
    else:
        worker["learning_status"] = []
    with pytest.raises(ValueError):
        expoft_dispatch.summarize_worker(worker, config)


@pytest.mark.parametrize("content", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}', "[]"])
def test_duplicate_nonfinite_and_nonobject_json_rejected(tmp_path, content):
    p = tmp_path / "input.json"
    p.write_text(content)
    with pytest.raises(ValueError):
        expoft_dispatch.read_json(p)


@pytest.mark.parametrize(
    "field,value",
    [
        ("domain", "fixture"),
        ("schema_version", True),
        ("episodes", False),
        ("max_wall_s", float("inf")),
        ("mode", []),
    ],
)
def test_coordinator_refuses_invalid_native_request(tmp_path, field, value):
    config = input_config()
    config[field] = value
    p = tmp_path / "input.json"
    p.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        expoft_dispatch.training_input(p)
