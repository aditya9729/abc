"""CPU coordinator doubles only; no native learner or GPU evidence."""

import copy
import json
from argparse import Namespace
from pathlib import Path

import pytest

from abc_bench import expoft_dispatch, runner


def input_config():
    return {
        "schema_version": 2,
        "checkpoint_storage": {
            "format": "immutable_episode_blocks",
            "schema_version": 1,
        },
        "domain": "sim",
        "profile": expoft_dispatch.PROFILE,
        "mode": "paused_simulation_fixed_tick",
        "episodes": 13,
        "max_wall_s": 500,
        "seed": 903,
        "sampler_seed": 901,
        "artifacts": {"checkpoint_path": "/fixture/candidate.pt"},
    }


def learner_counts(calls):
    return {
        "critic": 20 * calls,
        "critic_target": 20 * calls,
        "noise": 20 * calls,
        "editor": calls,
        "temperature": calls,
        "update_calls": calls,
        "base": 0,
        "auxiliary": 0,
        "skipped_base": calls,
    }


def worker_record(config, status="completed"):
    completed = 13 if status == "completed" else 12
    discarded = 0 if status == "completed" else 3
    calls = 3 if status == "completed" else 2
    return {
        "revision": expoft_dispatch.REVISION,
        "domain": "sim",
        "status": status,
        "identity": {
            "revision": expoft_dispatch.REVISION,
            "config": {k: v for k, v in config.items() if k != "max_wall_s"},
            "learning_starts": 10,
            "step_interval": 30,
            "diagnostic_fixture": False,
        },
        "clock_mode": config["mode"],
        "independent_native_admission": False,
        "author_reproduction": False,
        "wall_time_reactivity_verified": False,
        "wrapper": {
            "completed_episodes": completed,
            "accepted_update_calls": calls,
            "issued_steps": 330 + (completed - 11) * 30 + discarded,
            "update_debt": 0,
            "checkpoint_serial": 5,
            "phase": "ready" if status == "completed" else "collecting",
        },
        "invocation": {
            "initial_wrapper": {
                "completed_episodes": 11,
                "accepted_update_calls": 1,
                "issued_steps": 330,
                "update_debt": 0,
                "checkpoint_serial": 3,
                "phase": "ready",
            },
            "initial_learner_counters": learner_counts(1),
            "physical_control_steps": (completed - 11) * 30 + discarded,
            "accepted_completed_control_steps": (completed - 11) * 30,
            "accepted_warmup_control_steps": 0,
            "accepted_learning_control_steps": (completed - 11) * 30,
            "discarded_partial_control_steps": discarded,
        },
        "accepted_update_groups": [
            {"ordinal": i, "pin": {"fixture": True}} for i in range(calls)
        ],
        "successful_base_updates": 0,
        "skipped_base_updates": calls,
        "learner_counters": learner_counts(calls),
        "learning_status": "awaiting_success_imitation_data",
        "error": "fixture clock miss" if status != "completed" else None,
    }


@pytest.mark.parametrize(
    "status,exit_code,expected",
    [
        ("completed", 0, "completed"),
        ("budget_stopped", 2, "partial"),
        ("deadline_missed_partial", 2, "partial"),
        ("failed", 1, "failed"),
        ("completed", 2, "failed"),
        ("budget_stopped", 0, "failed"),
        ("deadline_missed_partial", 0, "failed"),
        ("budget_stopped", 1, "failed"),
    ],
)
def test_worker_dispatch_reserves_shared_lease_and_preserves_partial_counts(
    monkeypatch, tmp_path, status, exit_code, expected
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
        return exit_code

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
    assert receipt["status"] == expected and receipt["worker_exit_code"] == exit_code
    assert receipt["phase"] == "training" and receipt["training_stage"] == "train"
    assert (
        receipt["task"] == "put_plastic_bottles_in_bin"
        and receipt["checkpoint_path"] == config["artifacts"]["checkpoint_path"]
    )
    ledger = json.loads((campaign / "gpu_budget.json").read_bytes())
    assert (
        7100 <= ledger["charged_seconds"] < 7101 and "active_parent_pid" not in ledger
    )
    assert publications[-1]["status"] == expected
    assert any(
        a["path"].endswith("training/receipt.json") for a in receipt["artifacts"]
    )
    if expected == "failed" and status != "failed":
        assert receipt["metrics"].get("updates") is None
        assert "episode_schedule_completed" not in receipt
        assert any("exit code and status differ" in b for b in receipt["blockers"])
        return
    assert receipt["worker_status"] == status
    assert receipt["metrics"]["simulation_steps"] == (
        60 if status == "completed" else 33
    )
    assert receipt["metrics"]["accepted_completed_control_steps"] == (
        60 if status == "completed" else 30
    )
    assert receipt["metrics"]["discarded_partial_control_steps"] == (
        0 if status == "completed" else 3
    )
    assert receipt["metrics"]["updates"] == (
        2 if status == "completed" else 1
    ) and receipt["metrics"]["cumulative_accepted_update_calls"] == (
        3 if status == "completed" else 2
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
        "issued_delta",
        "due_debt",
        "critic_short",
        "noise_short",
        "critic_target_short",
        "editor_short",
        "temperature_short",
        "auxiliary_short",
        "base_total",
        "initial_counts",
        "initial_schema",
        "fixture_clock",
        "clock_bool",
        "extra_coherent_group",
        "unchanged_episode_prefix",
        "partition_sum",
        "partition_bool",
        "wrong_learning_partition",
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
    elif fault == "list_learning":
        worker["learning_status"] = []
    elif fault == "issued_delta":
        worker["invocation"]["physical_control_steps"] += 1
        worker["invocation"]["accepted_completed_control_steps"] += 1
    elif fault == "due_debt":
        worker["wrapper"]["update_debt"] = 30
    elif fault.endswith("_short"):
        key = fault.removesuffix("_short")
        if key == "auxiliary":
            worker["learner_counters"].update(base=1, skipped_base=2)
            worker.update(
                successful_base_updates=1,
                skipped_base_updates=2,
                learning_status="components_exercised_no_gain_claim",
            )
        else:
            worker["learner_counters"][key] -= 1
    elif fault == "base_total":
        worker["learner_counters"]["skipped_base"] = 4
        worker["skipped_base_updates"] = 4
    elif fault == "initial_counts":
        worker["invocation"]["initial_learner_counters"]["noise"] = 19
    elif fault == "initial_schema":
        worker["invocation"]["initial_wrapper"]["unknown"] = 0
    elif fault == "fixture_clock":
        worker["identity"].update(
            diagnostic_fixture=True, learning_starts=0, step_interval=1
        )
    elif fault == "clock_bool":
        worker["identity"]["step_interval"] = True
    elif fault == "extra_coherent_group":
        worker["wrapper"]["accepted_update_calls"] = 4
        worker.update(learner_counters=learner_counts(4), skipped_base_updates=4)
        worker["accepted_update_groups"].append(
            {"ordinal": 3, "pin": {"fixture": True}}
        )
    elif fault == "unchanged_episode_prefix":
        worker["invocation"]["initial_wrapper"]["completed_episodes"] = 13
    elif fault == "partition_sum":
        worker["invocation"]["accepted_warmup_control_steps"] = 1
    elif fault == "partition_bool":
        worker["invocation"]["accepted_warmup_control_steps"] = False
    elif fault == "wrong_learning_partition":
        worker["invocation"].update(
            accepted_warmup_control_steps=30, accepted_learning_control_steps=30
        )
    with pytest.raises(ValueError):
        expoft_dispatch.summarize_worker(worker, config)


def test_native_warmup_stop_with_no_update_groups_is_valid():
    config = input_config()
    worker = worker_record(config, "budget_stopped")
    worker["wrapper"].update(
        completed_episodes=8, issued_steps=240, accepted_update_calls=0, phase="ready"
    )
    worker["invocation"].update(
        initial_wrapper={
            "completed_episodes": 0,
            "issued_steps": 0,
            "accepted_update_calls": 0,
            "update_debt": 0,
            "checkpoint_serial": 0,
            "phase": "ready",
        },
        initial_learner_counters=learner_counts(0),
        physical_control_steps=240,
        accepted_completed_control_steps=240,
        accepted_warmup_control_steps=240,
        accepted_learning_control_steps=0,
        discarded_partial_control_steps=0,
    )
    worker.update(
        accepted_update_groups=[],
        learner_counters=learner_counts(0),
        skipped_base_updates=0,
        learning_status="no_accepted_update_groups",
    )
    result = expoft_dispatch.summarize_worker(worker, config, exit_code=2)
    assert result["learning_status"] == "no_accepted_update_groups"
    assert result["metrics"]["simulation_steps"] == 240
    assert result["metrics"]["updates"] == 0
    assert result["metrics"]["accepted_warmup_control_steps"] == 240
    assert result["metrics"]["accepted_learning_control_steps"] == 0
    assert result["episode_schedule_completed"] is False


def test_partial_optimizer_failure_keeps_only_accepted_group_credit():
    config = input_config()
    worker = worker_record(config, "failed")
    worker["learner_counters"]["critic"] += 1
    worker["learner_counters"]["base"] += 1
    worker.update(
        successful_base_updates=1, learning_status="components_exercised_no_gain_claim"
    )
    # A returned native command can fail during replay append. Preserve its
    # physical evidence separately from the wrapper's successfully added rows.
    worker["wrapper"]["issued_steps"] -= 1
    result = expoft_dispatch.summarize_worker(worker, config, exit_code=1)
    assert result["metrics"]["updates"] == 1
    assert result["metrics"]["cumulative_accepted_update_calls"] == 2
    assert result["metrics"]["simulation_steps"] == 33
    assert result["episode_schedule_completed"] is False


def test_completed_group_with_actual_successful_imitation_is_supported():
    config = input_config()
    worker = worker_record(config)
    worker["learner_counters"].update(base=1, auxiliary=1, skipped_base=2)
    worker.update(
        successful_base_updates=1,
        skipped_base_updates=2,
        learning_status="components_exercised_no_gain_claim",
    )
    result = expoft_dispatch.summarize_worker(worker, config, exit_code=0)
    assert result["learning_status"] == "components_exercised_no_gain_claim"
    assert result["metrics"]["updates"] == 2
    assert result["metrics"]["successes"] is None


def test_pending_update_only_resume_credits_no_new_controls():
    config = input_config()
    worker = worker_record(config)
    worker["invocation"]["initial_wrapper"].update(
        completed_episodes=13, issued_steps=390, phase="pending_updates", update_debt=60
    )
    worker["invocation"].update(
        physical_control_steps=0,
        accepted_completed_control_steps=0,
        accepted_warmup_control_steps=0,
        accepted_learning_control_steps=0,
        discarded_partial_control_steps=0,
    )
    result = expoft_dispatch.summarize_worker(worker, config, exit_code=0)
    assert result["metrics"]["simulation_steps"] == 0
    assert result["metrics"]["completed_training_episodes"] == 0
    assert result["metrics"]["updates"] == 2


def test_crossing_warmup_receipt_has_exact_learning_debt_partition():
    config = input_config()
    worker = worker_record(config, "budget_stopped")
    worker["wrapper"].update(
        completed_episodes=11, issued_steps=330, accepted_update_calls=1, phase="ready"
    )
    worker["invocation"].update(
        initial_wrapper={
            "completed_episodes": 0,
            "issued_steps": 0,
            "accepted_update_calls": 0,
            "update_debt": 0,
            "checkpoint_serial": 0,
            "phase": "ready",
        },
        initial_learner_counters=learner_counts(0),
        physical_control_steps=330,
        accepted_completed_control_steps=330,
        accepted_warmup_control_steps=300,
        accepted_learning_control_steps=30,
        discarded_partial_control_steps=0,
    )
    worker.update(
        accepted_update_groups=[{"ordinal": 0, "pin": {"fixture": True}}],
        learner_counters=learner_counts(1),
        skipped_base_updates=1,
    )
    result = expoft_dispatch.summarize_worker(worker, config, exit_code=2)
    assert result["metrics"]["updates"] == 1
    assert result["metrics"]["accepted_completed_control_steps"] == 330
    assert result["metrics"]["accepted_warmup_control_steps"] == 300
    assert result["metrics"]["accepted_learning_control_steps"] == 30
    assert result["independent_admission_required"] is True
    # Preserve the total control credit, but claim a coherent extra group using
    # warmup controls. Only the exact learning partition can reject this case.
    worker["wrapper"]["accepted_update_calls"] = 2
    worker.update(learner_counters=learner_counts(2), skipped_base_updates=2)
    worker["accepted_update_groups"].append({"ordinal": 1, "pin": {"fixture": True}})
    with pytest.raises(ValueError, match="update debt"):
        expoft_dispatch.summarize_worker(worker, config, exit_code=2)


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
        ("schema_version", 1),
        ("checkpoint_storage", {"format": "inline", "schema_version": 1}),
        (
            "checkpoint_storage",
            {"format": "immutable_episode_blocks", "schema_version": True},
        ),
    ],
)
def test_coordinator_refuses_invalid_native_request(tmp_path, field, value):
    config = input_config()
    config[field] = value
    p = tmp_path / "input.json"
    p.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        expoft_dispatch.training_input(p)
