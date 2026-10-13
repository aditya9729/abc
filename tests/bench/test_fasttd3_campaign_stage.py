"""Canonical stage accounting/lifecycle over explicit child and RPC doubles."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from abc_bench import fasttd3_campaign_frozen_stage as frozen
from abc_bench import fasttd3_campaign_host as host
from abc_bench import fasttd3_campaign_stage as stage
from abc_bench.bottle_success_reward import SPARSE_PROFILE, reward_spec
from abc_bench.fasttd3_campaign_contract import PROFILE, PROFILE64, campaign_spec
from abc_bench.subset import load_job


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return stage.artifact(path)


@pytest.fixture(params=[128, 64])
def evidence(tmp_path, request):
    count = request.param
    config = {
        "schema_version": 1,
        "profile": PROFILE if count == 128 else PROFILE64,
        "phase": "runtime_gate",
        "num_envs": count,
        "seed_base": 200000,
        "seed_stride": 128,
        "max_episode_steps": 3540,
        "vector_steps": 64,
        "reward": reward_spec(SPARSE_PROFILE),
        "campaign": campaign_spec("runtime_gate"),
        "timeouts": {
            "ready_s": 600,
            "step_s": 180,
            "shutdown_s": 30,
            "wall_s": 2400,
            "rpc_s": 180,
        },
    }
    description = {
        **stage.qualification.TASK,
        "profile": config["profile"],
        "num_envs": count,
        "reset_options": host.native.RESET_OPTIONS,
        "reset_seed_bank": {
            "base": 200000,
            "rank_stride": 1,
            "episode_stride": 128,
            "max_episode_index": 780,
            "fallback_offset": 100000,
        },
        "reward": config["reward"],
        "campaign": config["campaign"],
        "physics_failure_policy": "fail_on_BADQPOS_BADQVEL_BADQACC_BADCTRL_nonfinite_or_internal_time_reset",
    }
    path = tmp_path / "run"
    checkpoint = path / "learner/models/original_final.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"explicit original checkpoint artifact double")
    checkpoint_pin = stage.artifact(checkpoint)
    decision = {
        "reason": "runtime_gate_completed",
        "controls": 64,
        "completed_episodes_per_rank": [0] * count,
        "stale_windows": 0,
    }
    learner = {
        "status": "completed",
        "controls": 64,
        "configuration": stage.recipe(config),
        "actual_updates": {"critic": 106, "actor": 53},
        "finite_gradient_samples": {"critic": 3, "actor": 2},
        "checkpoint": checkpoint_pin,
    }
    campaign = {
        "status": "completed",
        "phase": "runtime_gate",
        "controls": 64,
        "transitions": count * 64,
        "declared_maximum_args": stage.recipe(config),
        "effective_final_args": stage.recipe(config),
        "actual_updates": learner["actual_updates"],
        "checkpoint": checkpoint_pin,
        "stopping": {"spec": config["campaign"], "decision": decision, "windows": []},
        "episodes": [],
    }
    if count == 64:
        campaign["parallelism"] = stage.parallelism(config)
    native = {
        "status": "completed",
        "config": config,
        "description": description,
        "vector_steps": 64,
        "transitions": count * 64,
        "worker_cleanup": [
            {"rank": i, "pid": 1900000000 + i, "exitcode": 0, "alive": False}
            for i in range(count)
        ],
    }
    physics = {
        "time": 2.176,
        "time_finite": True,
        "finite": dict.fromkeys(("qpos", "qvel", "qacc", "ctrl"), True),
        "warnings": {
            name: {"number": 0, "lastinfo": 0}
            for name in ("BADQPOS", "BADQVEL", "BADQACC", "BADCTRL")
        },
    }
    for rank in range(count):
        write(
            path / "native_host" / f"world-{rank:03d}" / "reward-summary.json",
            {
                "rank": rank,
                "controls": 64,
                "completed_episodes": 0,
                "reward": config["reward"],
                "physics": physics,
            },
        )
    write(path / "learner/receipt.json", learner)
    write(path / "learner/campaign.json", campaign)
    write(path / "native_host/receipt.json", native)
    return config, path, description


def test_completed_gate_binds_exact_world_cleanup_counts_and_final_export(evidence):
    config, path, _ = evidence
    result = stage.verify_result({}, config, path)
    assert result["metrics"] == stage.counters(64, num_envs=config["num_envs"])
    assert len(result["reward_summaries"]) == config["num_envs"]
    assert result["completed_training_episodes"] == 0
    assert result["training_successes"] == 0
    assert result["first_geometry_proofs"] == []


def test_runtime_gate_cannot_admit_a_different_parallel_world_count(evidence, tmp_path):
    config, _, _ = evidence
    other = 64 if config["num_envs"] == 128 else 128
    worker_pin = write(tmp_path / "other-worker.json", {})
    parent = {
        "status": "completed",
        "embodiment": "native_yam",
        "task": "put_plastic_bottles_in_bin",
        "worker_exit_code": 0,
        "worker_receipt_sha256": worker_pin["sha256"],
        "metrics": stage.counters(64, num_envs=other),
    }
    with pytest.raises(ValueError, match="runtime gate"):
        stage.runtime_gate(
            write(tmp_path / "other-parent.json", parent), worker_pin, config, {}
        )


def test_completed_training_export_binds_exact_world_frozen_producer(evidence):
    config, path, _ = evidence
    count = config["num_envs"]
    calls = 42480
    config.update(
        phase="training", vector_steps=calls, campaign=campaign_spec("training")
    )
    native_path = path / "native_host/receipt.json"
    native = json.loads(native_path.read_text())
    native.update(config=config, vector_steps=calls, transitions=calls * count)
    native["description"]["campaign"] = config["campaign"]
    write(native_path, native)
    episodes = []
    for rank in range(count):
        directory = path / "native_host" / f"world-{rank:03d}"
        summary_path = directory / "reward-summary.json"
        summary = json.loads(summary_path.read_text())
        summary.update(controls=calls, completed_episodes=6)
        write(summary_path, summary)
        for index in range(6):
            episode = {"rank": rank, "episode_index": index, "success": False}
            write(directory / f"episode-{index:04d}.json", episode)
            episodes.append(episode)
    learner_path = path / "learner/receipt.json"
    learner = json.loads(learner_path.read_text())
    updates = {"critic": 84938, "actor": 42469}
    learner.update(
        controls=calls, configuration=stage.recipe(config), actual_updates=updates
    )
    write(learner_path, learner)
    campaign_path = path / "learner/campaign.json"
    campaign = json.loads(campaign_path.read_text())
    campaign.update(
        phase="training",
        controls=calls,
        transitions=calls * count,
        declared_maximum_args=stage.recipe(config),
        effective_final_args=stage.recipe(config),
        actual_updates=updates,
        episodes=episodes,
    )
    if count == 64:
        campaign["parallelism"] = stage.parallelism(config)
    campaign["stopping"].update(
        spec=config["campaign"],
        decision={
            "reason": "declared_maximum_completed",
            "controls": calls,
            "completed_episodes_per_rank": [6] * count,
            "stale_windows": 0,
        },
    )
    write(campaign_path, campaign)
    host_pin = write(path / "host.json", config)
    config_pin = write(path / "input.json", {"host_config": host_pin})
    result = stage.verify_result({}, config, path)
    export_pin = write(
        path / "export.json",
        {
            "global_step": calls,
            "state_fingerprints": {
                name: {"sha256": "explicit export fingerprint double"}
                for name in (
                    "actor_state_dict",
                    "qnet_state_dict",
                    "qnet_target_state_dict",
                    "obs_normalizer_state",
                    "critic_obs_normalizer_state",
                )
            },
        },
    )
    worker_pin = write(
        path / "receipt.json",
        {
            "status": "completed",
            "phase": "fasttd3_reward_v2_training",
            "child_exit_codes": {"host": 0, "learner": 0},
            "input_config": config_pin,
            "host_config": host_pin,
            "export_check": export_pin,
            **result,
        },
    )
    parent_pin = write(
        path / "parent.json",
        {
            "status": "completed",
            "embodiment": "native_yam",
            "task": "put_plastic_bottles_in_bin",
            "worker_exit_code": 0,
            "worker_receipt_sha256": worker_pin["sha256"],
            "metrics": result["metrics"],
        },
    )
    producer = stage.training_producer(parent_pin, worker_pin)
    assert producer["metrics"]["transitions"] == 42480 * count
    assert producer["completed_training_episodes"] == 6 * count
    if count == 64:
        assert producer["parallelism"]["actual_rank_statistics_denominator"] == 64
    else:
        assert "parallelism" not in producer


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_actor_updates",
        "bad_physics",
        "failed_child",
        "invented_early_stop",
        "missing_episode",
    ],
)
def test_bad_evidence_cannot_become_completed_gate(evidence, mutation):
    config, path, _ = evidence
    if mutation == "wrong_actor_updates":
        name = path / "learner/receipt.json"
        row = json.loads(name.read_text())
        row["actual_updates"]["actor"] = 52
    elif mutation == "bad_physics":
        name = path / "native_host/world-012/reward-summary.json"
        row = json.loads(name.read_text())
        row["physics"]["warnings"]["BADQACC"]["number"] = 1
    elif mutation == "failed_child":
        name = path / "native_host/receipt.json"
        row = json.loads(name.read_text())
        row["worker_cleanup"][0]["exitcode"] = 1
    elif mutation == "invented_early_stop":
        name = path / "learner/campaign.json"
        row = json.loads(name.read_text())
        row["stopping"]["decision"]["reason"] = "loss_plateau"
    else:
        name = path / "native_host/world-010/reward-summary.json"
        row = json.loads(name.read_text())
        row["completed_episodes"] = 1
    write(name, row)
    with pytest.raises((RuntimeError, ValueError)):
        stage.verify_result({}, config, path)


def test_canonical_lifecycle_uses_finite_samegroup_children_and_one_shutdown(
    evidence, tmp_path, monkeypatch
):
    host_config, produced, description = evidence
    output = tmp_path / "fresh"
    config = {
        "host_python": "explicit-host-python",
        "learner_python": "explicit-fast-python",
        "host_config": write(tmp_path / "host.json", host_config),
        "cpu_admission": {},
        "qualification_parent": {},
        "qualification_worker": {},
        "source_pins": {"abc": [], "fast_adapter": []},
        "limits": {
            "ready_seconds": 600,
            "learner_seconds": 1500,
            "shutdown_seconds": 30,
        },
    }
    config_path = tmp_path / "config.json"
    write(config_path, config)
    monkeypatch.setattr(stage, "require_subset_lease", lambda: None)
    monkeypatch.setattr(
        stage,
        "load_config",
        lambda _path: (config, {"explicit": "CPU admission double"}),
    )
    monkeypatch.setattr(
        stage,
        "memory_admission",
        lambda _value, *, count: {"explicit": "RAM double", "count": count},
    )
    monkeypatch.setattr(stage, "ROOT", tmp_path)
    (tmp_path / "builds").mkdir()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-explicit-CPU-test-double")
    monkeypatch.setenv("MUJOCO_GL", "disable")
    monkeypatch.setattr(stage.os, "getpgid", lambda _pid: os.getpgrp())
    monkeypatch.setattr(stage, "wait_ready", lambda *_args: description)
    verify_result = stage.verify_result
    monkeypatch.setattr(
        stage, "verify_result", lambda *_args: verify_result({}, host_config, produced)
    )
    processes = []
    operations = []

    class Child:
        def __init__(self, command, **kwargs):
            assert (
                kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-explicit-CPU-test-double"
            )
            assert "start_new_session" not in kwargs
            self.pid, self.returncode = 1900000000 + len(processes), None
            self.role = (
                "host" if "abc_bench.fasttd3_campaign_host" in command else "learner"
            )
            assert command[:4] == [
                config[self.role + "_python"]
                if self.role == "host"
                else config["learner_python"],
                "-I",
                "-u",
                "-B",
            ]
            if self.role == "learner":
                assert "sadhana_fasttd3_abc.campaign" in command
            processes.append(self)

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            assert timeout == (30 if self.role == "host" else 1500)
            self.returncode = 0
            return 0

        def terminate(self):
            self.returncode = -15

    def rpc(_path, operation, *, timeout):
        operations.append(operation)
        assert timeout == 30
        return {"status": "closed"}

    def process_run(command, **kwargs):
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert kwargs["timeout"] in (30, 60)
        destination = command[5] if command[4] == stage.PREFLIGHT else command[6]
        write(Path(destination), {"explicit": "subprocess double"})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(stage.subprocess, "Popen", Child)
    monkeypatch.setattr(stage.subprocess, "run", process_run)
    monkeypatch.setattr(stage, "rpc", rpc)
    result = stage.run(config_path, output)
    assert result["status"] == "completed"
    assert result["child_exit_codes"] == {"host": 0, "learner": 0}
    assert operations == ["shutdown"]
    assert list((tmp_path / "builds").iterdir()) == []


def test_frozen_facade_authenticates_campaign_and_preserves_declared_val_cohort(
    tmp_path, monkeypatch
):
    def reject():
        raise RuntimeError("explicit missing canonical lease")

    monkeypatch.setattr(frozen, "require_subset_lease", reject)
    with pytest.raises(RuntimeError, match="missing canonical lease"):
        frozen.run(tmp_path / "absent.json", tmp_path / "output")
    assert not (tmp_path / "output").exists()
    parent = {
        "status": "completed",
        "embodiment": "native_yam",
        "task": "put_plastic_bottles_in_bin",
        "worker_exit_code": 0,
        "metrics": stage.counters(64),
    }
    worker = {
        "status": "completed",
        "phase": "fasttd3_reward_v2_runtime_gate",
        "metrics": stage.counters(64),
        "child_exit_codes": {"host": 0, "learner": 0},
    }
    worker_pin = write(tmp_path / "worker.json", worker)
    parent["worker_receipt_sha256"] = worker_pin["sha256"]
    with pytest.raises(
        ValueError, match="completed canonical reward-v2 training producer"
    ):
        stage.training_producer(write(tmp_path / "parent.json", parent), worker_pin)


def test_only_new_campaign_can_use_the_declared_long_watchdog(tmp_path):
    import sys

    job = {
        "schema_version": 1,
        "label": "reward-v2 task-transfer diagnostic",
        "python": sys.executable,
        "module": "abc_bench.fasttd3_campaign_stage",
        "arguments": ["--config", "/explicit/config.json"],
        "max_wall_s": 12600,
    }
    path = tmp_path / "job.json"
    write(path, job)
    assert load_job(path)["max_wall_s"] == 12600
    for module, maximum in (
        ("abc_bench.fasttd3_scratch_stage", 7201),
        ("abc_bench.fasttd3_campaign_stage", 12601),
    ):
        write(path, {**job, "module": module, "max_wall_s": maximum})
        with pytest.raises(ValueError):
            load_job(path)
