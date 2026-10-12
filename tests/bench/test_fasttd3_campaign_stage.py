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
from abc_bench.fasttd3_campaign_contract import PROFILE, campaign_spec
from abc_bench.subset import load_job


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return stage.artifact(path)


@pytest.fixture
def evidence(tmp_path):
    config = {
        "schema_version": 1,
        "profile": PROFILE,
        "phase": "runtime_gate",
        "num_envs": 128,
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
        "profile": PROFILE,
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
        "completed_episodes_per_rank": [0] * 128,
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
        "transitions": 8192,
        "declared_maximum_args": stage.recipe(config),
        "effective_final_args": stage.recipe(config),
        "actual_updates": learner["actual_updates"],
        "checkpoint": checkpoint_pin,
        "stopping": {"spec": config["campaign"], "decision": decision, "windows": []},
        "episodes": [],
    }
    native = {
        "status": "completed",
        "config": config,
        "description": description,
        "vector_steps": 64,
        "transitions": 8192,
        "worker_cleanup": [
            {"rank": i, "pid": 1900000000 + i, "exitcode": 0, "alive": False}
            for i in range(128)
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
    for rank in range(128):
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


def test_completed_gate_binds128_cleanup_original_counts_and_final_export(evidence):
    config, path, _ = evidence
    result = stage.verify_result({}, config, path)
    assert result["metrics"] == stage.counters(64)
    assert len(result["reward_summaries"]) == 128
    assert result["completed_training_episodes"] == 0
    assert result["training_successes"] == 0
    assert result["first_geometry_proofs"] == []


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
        stage, "memory_admission", lambda _value: {"explicit": "RAM double"}
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
