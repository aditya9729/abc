"""Explicit coordinator doubles; no simulated/native learner performance claims."""

import copy
import json
from argparse import Namespace
from pathlib import Path

import pytest
from abc_bench import resfit_dispatch, runner


@pytest.mark.parametrize("algorithm", ["resfit-abc-vla", "realtime-expoft-abc"])
def test_main_dispatches_config_owned_checkpoint_without_legacy_candidate(
    monkeypatch, tmp_path, capsys, algorithm
):
    """Exercise the real CLI; the dispatch sentinel never launches a worker."""
    candidate = tmp_path / "reviewed-vla.pt"
    candidate.write_bytes(b"explicit non-model fixture")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"artifacts": {"checkpoint_path": str(candidate)}})
    )
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    calls = []

    def dispatch(args):
        calls.append(args)
        assert args.training_config == config_path
        assert not args.checkpoint.is_file()
        return {"status": "completed", "scope": "CLI dispatch fixture only"}

    monkeypatch.setattr(runner, "run_baseline", dispatch)
    monkeypatch.setattr(
        "sys.argv",
        ["abc-bench", "--algorithm", algorithm, "--training-config", str(config_path)],
    )
    runner.main()
    assert len(calls) == 1 and calls[0].algorithm == algorithm
    assert json.loads(capsys.readouterr().out)["status"] == "completed"


@pytest.mark.parametrize(
    "algorithm",
    ["baseline", "qf3", "resfit", "comparison", "sustained-resfit", "qf3-vla"],
)
def test_main_keeps_legacy_candidate_precondition(monkeypatch, tmp_path, algorithm):
    monkeypatch.setattr(runner, "ROOT", tmp_path)

    def unexpected_dispatch(args):
        raise AssertionError("A missing legacy candidate must fail before dispatch")

    monkeypatch.setattr(runner, "run_baseline", unexpected_dispatch)
    monkeypatch.setattr("sys.argv", ["abc-bench", "--algorithm", algorithm])
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 2


def input_config():
    return {
        "schema_version": 1,
        "domain": "sim",
        "profile": resfit_dispatch.PROFILE,
        "data_profile": resfit_dispatch.DATA_PROFILE,
        "seed": 903,
        "sampler_seed": 901,
        "replay_seed": 904,
        "learner_seed": 902,
        "episodes": 20,
        "target_controls": 10008,
        "replay_capacity": 200000,
        "max_wall_s": 500,
        "diagnostic": None,
        "artifacts": {"checkpoint_path": "/fixture/candidate.pt"},
        "sources": {},
        "native_inventory": {},
    }


def wrapper(controls=10000, episodes=10, ordinary=0, phase="ready", warmup=10000):
    return {
        "controls": controls,
        "episodes": episodes,
        "critic_warmup_updates": warmup,
        "ordinary_critic_updates": ordinary,
        "actor_updates": ordinary // 4,
        "debt": max(0, controls - 10000) * 4 - ordinary,
        "checkpoint_serial": 5,
        "phase": phase,
    }


def worker_record(config, status="completed"):
    final = (
        wrapper(10008, 11, 32)
        if status == "completed"
        else wrapper(10004, 10, 16, "collecting")
    )
    physical = 8 if status == "completed" else 4
    return {
        "schema_version": 1,
        "revision": resfit_dispatch.REVISION,
        "domain": "sim",
        "status": status,
        "identity": {
            "revision": resfit_dispatch.REVISION,
            "config": {k: v for k, v in config.items() if k != "max_wall_s"},
            "schedule": resfit_dispatch.SCHEDULE.copy(),
            "data_profile": resfit_dispatch.DATA_PROFILE,
            "n_step": 3,
            "gamma": 0.99,
        },
        "wrapper": final,
        "learner_counters": {
            "critic_updates": 10000 + final["ordinary_critic_updates"],
            "critic_warmup_updates": 10000,
            "ordinary_critic_updates": final["ordinary_critic_updates"],
            "actor_updates": final["actor_updates"],
            "critic_target_updates": 10000 + final["ordinary_critic_updates"],
            "actor_target_updates": final["actor_updates"],
        },
        "invocation": {
            "initial_wrapper": wrapper(),
            "physical_control_steps": physical,
            "accepted_complete_control_steps": physical if status == "completed" else 0,
            "discarded_partial_control_steps": 0 if status == "completed" else physical,
            "warmup_control_steps": 0,
            "learning_control_steps": physical,
            "control_dispatch_attempts": physical,
            "native_failed_step_calls_with_unknown_physics": 0,
        },
        **{
            f: False
            for f in [
                "native_independent_admission",
                "native_physics_resume",
                "task_gain",
                "paper_reproduction",
                "matched_evaluation_implemented",
                "expert_ground_truth",
            ]
        },
        "data_profile": resfit_dispatch.DATA_PROFILE,
        "full_recipe": True,
        "reference_replay_capacity": 200000,
        "offline_rows": 0,
        "error": "fixture dispatch failed" if status == "failed" else None,
    }


@pytest.mark.parametrize(
    "status,exit_code,expected",
    [
        ("completed", 0, "completed"),
        ("budget_stopped", 2, "partial"),
        ("failed", 1, "failed"),
        ("completed", 2, "failed"),
        ("budget_stopped", 0, "failed"),
        ("budget_stopped", 1, "failed"),
        ("failed", 2, "failed"),
    ],
)
def test_dispatch_lease_watchdog_partial_work_and_exit_semantics(
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
    monkeypatch.setattr(
        runner, "execution_provenance", lambda: {"evidence": "explicit_CPU_double"}
    )
    publications = []
    monkeypatch.setattr(
        runner,
        "publish",
        lambda receipt, results: publications.append(copy.deepcopy(receipt)),
    )
    config = input_config()
    config_path = tmp_path / "input.json"
    runner.write_json(config_path, config)

    def fixture_child(command, log_path, timeout, *, gpu_uuid):
        assert command[1:3] == ["-m", "nrh.resfit_abc_vla_run"]
        assert timeout == 100 and gpu_uuid == "GPU-00000000-0000-0000-0000-000000000000"
        assert command[command.index("--campaign-directory") + 1] == str(
            campaign.resolve()
        )
        assert command[command.index("--resume-pin") + 1] == str(
            (tmp_path / "resume.json").resolve()
        )
        child = json.loads(Path(command[command.index("--config") + 1]).read_text())
        assert child["max_wall_s"] == 40 and all(
            child[k] == v for k, v in config.items() if k != "max_wall_s"
        )
        ledger = json.loads((campaign / "gpu_budget.json").read_text())
        assert (
            ledger["charged_seconds"] == 7200 and ledger["active_gpu_uuid"] == gpu_uuid
        )
        assert type(ledger["active_parent_pid"]) is int
        destination = Path(command[command.index("--out") + 1])
        runner.write_json(destination / "receipt.json", worker_record(child, status))
        return exit_code

    monkeypatch.setattr(runner, "execute_command", fixture_child)
    args = Namespace(
        results=tmp_path / "results",
        timeout_seconds=500,
        algorithm="resfit-abc-vla",
        training_config=config_path,
        checkpoint=tmp_path / "unused.pt",
        task="unused",
        worlds=1,
        chunks=2,
        seed=0,
        video=False,
        resfit_resume_pin=tmp_path / "resume.json",
    )
    receipt = runner.run_baseline(args)
    assert (
        receipt["status"] == expected
        and receipt["phase"] == "training"
        and receipt["worker_exit_code"] == exit_code
    )
    assert (
        receipt["task"] == "put_plastic_bottles_in_bin"
        and receipt["checkpoint_path"] == config["artifacts"]["checkpoint_path"]
    )
    ledger = json.loads((campaign / "gpu_budget.json").read_text())
    assert (
        7100 <= ledger["charged_seconds"] < 7101 and "active_parent_pid" not in ledger
    )
    assert publications[-1]["status"] == expected
    assert any(
        a["path"].endswith("training/receipt.json") for a in receipt["artifacts"]
    )
    if expected == "failed" and (status != "failed" or exit_code == 2):
        assert receipt["metrics"].get("ordinary_critic_updates") is None
        assert any("exit code and status differ" in b for b in receipt["blockers"])
        return
    m = receipt["metrics"]
    assert m["simulation_steps"] == (8 if status == "completed" else 4)
    assert m["accepted_complete_control_steps"] == (8 if status == "completed" else 0)
    assert m["ordinary_critic_updates"] == (32 if status == "completed" else 16)
    assert m["successes"] is None and m["episodes"] is None
    assert receipt["independent_admission_required"] is True
    assert "partial work is not admitted checkpoint credit" in m["counters_scope"]


@pytest.mark.parametrize(
    "fault",
    [
        "physical_inflation",
        "Boolean_counter",
        "accepted_partial",
        "extra_critic",
        "extra_actor",
        "borrowed_controls",
        "target_not_reached",
        "fixture_clock",
        "capacity_alias",
        "self_admission",
        "config_alias",
        "warmup_partition",
        "unknown_physics",
        "early_warmup",
        "due_ready",
    ],
)
def test_untrusted_receipt_cannot_gain_control_or_update_credit(fault):
    config = input_config()
    worker = worker_record(config)
    if fault == "physical_inflation":
        worker["invocation"]["physical_control_steps"] += 1
    elif fault == "Boolean_counter":
        worker["invocation"]["discarded_partial_control_steps"] = False
    elif fault == "accepted_partial":
        worker = worker_record(config, "budget_stopped")
        worker["invocation"]["accepted_complete_control_steps"] = 4
        worker["invocation"]["discarded_partial_control_steps"] = 0
    elif fault == "extra_critic":
        worker["learner_counters"]["ordinary_critic_updates"] += 1
    elif fault == "extra_actor":
        worker["wrapper"]["actor_updates"] += 1
    elif fault == "borrowed_controls":
        worker["wrapper"]["ordinary_critic_updates"] += 4
        worker["wrapper"]["actor_updates"] += 1
    elif fault == "target_not_reached":
        config["target_controls"] = 10009
        worker["identity"]["config"]["target_controls"] = 10009
    elif fault == "fixture_clock":
        worker["identity"]["schedule"]["warmup_controls"] = 1
    elif fault == "capacity_alias":
        worker["reference_replay_capacity"] = 200000.0
    elif fault == "self_admission":
        worker["task_gain"] = True
    elif fault == "config_alias":
        worker["identity"]["config"]["seed"] = 903.0
    elif fault == "warmup_partition":
        worker["invocation"]["warmup_control_steps"] = 1
        worker["invocation"]["learning_control_steps"] -= 1
    elif fault == "unknown_physics":
        worker["invocation"]["native_failed_step_calls_with_unknown_physics"] = 1
    elif fault == "early_warmup":
        worker["invocation"]["initial_wrapper"] = wrapper(9000, 9, 0, "ready", 10000)
    elif fault == "due_ready":
        worker["wrapper"]["debt"] = 1
    with pytest.raises(ValueError):
        resfit_dispatch.summarize_worker(worker, config)


def test_pending_only_resume_keeps_new_controls_zero():
    config = input_config()
    worker = worker_record(config, "budget_stopped")
    worker["wrapper"] = wrapper(10008, 11, 32)
    worker["invocation"].update(
        initial_wrapper=wrapper(10008, 11, 24, "pending_updates"),
        physical_control_steps=0,
        accepted_complete_control_steps=0,
        discarded_partial_control_steps=0,
        warmup_control_steps=0,
        learning_control_steps=0,
        control_dispatch_attempts=0,
    )
    worker["learner_counters"] = worker_record(config)["learner_counters"]
    summary = resfit_dispatch.summarize_worker(worker, config, exit_code=2)
    assert (
        summary["metrics"]["simulation_steps"] == 0
        and summary["metrics"]["ordinary_critic_updates"] == 8
    )
    assert summary["metrics"]["completed_training_episodes"] == 0


@pytest.mark.parametrize(
    "kind", ["returned_uncaptured", "unknown_physics", "interrupted_optimizer"]
)
def test_failed_receipt_separates_returned_controls_missing_capture_and_optimizer(kind):
    config = input_config()
    worker = worker_record(config, "failed")
    if kind == "returned_uncaptured":
        worker["invocation"].update(
            physical_control_steps=5,
            discarded_partial_control_steps=5,
            learning_control_steps=5,
            control_dispatch_attempts=5,
        )
    elif kind == "unknown_physics":
        worker["invocation"].update(
            control_dispatch_attempts=5, native_failed_step_calls_with_unknown_physics=1
        )
    else:
        for key in [
            "critic_updates",
            "ordinary_critic_updates",
            "critic_target_updates",
        ]:
            worker["learner_counters"][key] += 1
    m = resfit_dispatch.summarize_worker(worker, config, exit_code=1)["metrics"]
    assert m["validated_replay_rows"] == 4 and m["accepted_complete_control_steps"] == 0
    assert m["ordinary_critic_updates"] == 16
    if kind == "returned_uncaptured":
        assert m["simulation_steps"] == 5
    if kind == "unknown_physics":
        assert (
            m["simulation_steps"] == 4
            and m["native_failed_step_calls_with_unknown_physics"] == 1
        )
    if kind == "interrupted_optimizer":
        assert m["producer_learner_counters"]["ordinary_critic_updates"] == 17


@pytest.mark.parametrize(
    "fault",
    [
        "fixture",
        "reduced_capacity",
        "capacity_float",
        "diagnostic",
        "negative_seed",
        "seed_bool",
        "zero_target",
        "watchdog_bool",
        "unknown_field",
    ],
)
def test_invalid_native_inputs_fail_before_model_loading(tmp_path, fault):
    config = input_config()
    if fault == "fixture":
        config["domain"] = "fixture"
    elif fault == "reduced_capacity":
        config["replay_capacity"] = 100
    elif fault == "capacity_float":
        config["replay_capacity"] = 200000.0
    elif fault == "diagnostic":
        config["diagnostic"] = {"warmup_controls": 1}
    elif fault == "negative_seed":
        config["seed"] = -1
    elif fault == "seed_bool":
        config["seed"] = False
    elif fault == "zero_target":
        config["target_controls"] = 0
    elif fault == "watchdog_bool":
        config["max_wall_s"] = True
    else:
        config["unknown"] = None
    path = tmp_path / "input.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        resfit_dispatch.training_input(path)


@pytest.mark.parametrize("text", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":1e999}', "[]"])
def test_json_boundary_rejects_ambiguous_nonfinite_inputs(tmp_path, text):
    path = tmp_path / "input.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        resfit_dispatch.read_json(path)


def test_explicit_storage_schema2_input(tmp_path):
    value = input_config()
    value.update(
        schema_version=2,
        storage={
            "rgb_codec": "lossless_rgb_zlib/1",
            "complete_milestone_controls": 10000,
        },
    )
    path = tmp_path / "input.json"
    path.write_text(json.dumps(value))
    assert resfit_dispatch.training_input(path) == value
    value["storage"]["complete_milestone_controls"] = 9999
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="storage"):
        resfit_dispatch.training_input(path)


def test_storage_milestone_type_coordinator(tmp_path):
    value = input_config()
    value.update(
        schema_version=2,
        storage={
            "rgb_codec": "lossless_rgb_zlib/1",
            "complete_milestone_controls": 10000,
        },
    )
    path = tmp_path / "strict-storage.json"
    path.write_text(json.dumps(value))
    assert resfit_dispatch.training_input(path) == value
    value["storage"]["complete_milestone_controls"] = 10000.0
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="storage"):
        resfit_dispatch.training_input(path)
