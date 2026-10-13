"""Installed-stage lifecycle tests with explicit child/RPC doubles; no GPU."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from abc_bench import fasttd3_vector_host as host
from abc_bench import fasttd3_vector_stage as stage


def write(path, value):
    path.write_text(json.dumps(value))
    return stage.artifact(path)


@pytest.fixture
def inputs(tmp_path):
    description = {
        **stage.TASK,
        "profile": "cpu_acceptance2",
        "num_envs": 2,
        "reset_options": host.RESET_OPTIONS,
        "reset_seed_bank": {
            "base": 200000,
            "rank_stride": 1,
            "episode_stride": 128,
            "max_episode_index": 780,
            "fallback_offset": 100000,
        },
    }
    resources = []
    for rank in range(2):
        measurement = {
            "rank": rank,
            "pid": 2000000000 + rank,
            "status": "closed",
            "cuda_visible_devices": "",
            "torch_imported": False,
            "warp_imported": False,
            "render_cameras": False,
            "maxrss_kib": 1000,
        }
        resources.append(
            {
                "measurement": measurement,
                "evidence": write(tmp_path / f"worker-{rank}.json", measurement),
            }
        )
    evidence = {
        "schema_version": 1,
        "phase": "abc_fasttd3_vector_native_cpu_acceptance",
        "status": "completed",
        "vector_steps": 2,
        "transitions": 4,
        "description": description,
        "original_fast_bootstrap_mask": [False, True],
        "terminal_timeout_step": {
            "terminated": [True, False],
            "truncated": [False, True],
        },
        "worker_cleanup": [
            {"rank": rank, "pid": 2000000000 + rank, "exitcode": 0, "alive": False}
            for rank in range(2)
        ],
        "worker_resources": resources,
        "carried_arm_acceptance": [
            {
                "rank": rank,
                "same_seed_pose_preserved": True,
                "joint_comparison": "exact_float32",
                "absolute_difference": [0.0] * 14,
            }
            for rank in range(2)
        ],
    }
    host_config = {
        "schema_version": 1,
        "profile": "training128",
        "num_envs": 128,
        "seed_base": 200000,
        "seed_stride": 128,
        "max_episode_steps": 3540,
        "vector_steps": 64,
        "timeouts": {
            "ready_s": 600,
            "step_s": 120,
            "shutdown_s": 30,
            "wall_s": 1800,
            "rpc_s": 180,
        },
    }
    config = {
        "schema_version": 1,
        "host_python": str(stage.HOST_PYTHON),
        "learner_python": str(stage.LEARNER_PYTHON),
        "host_config": write(tmp_path / "host.json", host_config),
        "cpu_admission": write(tmp_path / "cpu.json", evidence),
        "source_pins": {
            "abc_host": {
                key: value
                for key, value in stage.artifact(Path(host.__file__)).items()
                if key != "path"
            },
            "fast_adapter": [
                {"path": "sadhana_fasttd3_abc/" + name, "bytes": 1, "sha256": "0" * 64}
                for name in (
                    "vector.py",
                    "environment.py",
                    "learner.py",
                    "provenance.json",
                )
            ],
        },
        "limits": {
            "ready_seconds": 600,
            "learner_seconds": 1000,
            "shutdown_seconds": 30,
        },
    }
    path = tmp_path / "config.json"
    write(path, config)
    return path, config, evidence


def test_actual_schema_accepts_exact_native_cpu_boundaries(inputs):
    path, expected, evidence = inputs
    assert stage.load_config(path) == (expected, evidence)


@pytest.mark.parametrize(
    "fault", ["bootstrap", "carry", "gpu", "duplicate", "pin", "wall"]
)
def test_admission_rejects_changed_physics_cleanup_and_pins(inputs, fault):
    path, config, evidence = inputs
    if fault == "bootstrap":
        evidence["original_fast_bootstrap_mask"] = [False, False]
    elif fault == "carry":
        evidence["carried_arm_acceptance"][0]["absolute_difference"][0] = 0.001
    elif fault == "gpu":
        row = evidence["worker_resources"][0]
        row["measurement"]["torch_imported"] = True
        row["evidence"] = write(Path(row["evidence"]["path"]), row["measurement"])
    elif fault == "duplicate":
        evidence["worker_cleanup"][1]["pid"] = evidence["worker_cleanup"][0]["pid"]
    elif fault == "pin":
        config["cpu_admission"]["sha256"] = "0" * 64
    elif fault == "wall":
        config["limits"]["learner_seconds"] = 1800
    if fault not in ("pin", "wall"):
        config["cpu_admission"] = write(Path(config["cpu_admission"]["path"]), evidence)
    write(path, config)
    with pytest.raises(ValueError):
        stage.load_config(path)


def test_memory_refusal_uses_fresh_available_memory(inputs, tmp_path):
    evidence = inputs[2]
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 999999999 kB\nMemAvailable: 1 kB\n")
    with pytest.raises(MemoryError, match="observed 1 KiB"):
        stage.memory_admission(evidence, meminfo=meminfo)
    meminfo.write_text("MemAvailable: 999999999 kB\n")
    result = stage.memory_admission(evidence, meminfo=meminfo)
    assert result["required_kib"] == 128 * 1000 * 1.25 + 8 * 1024**2


def test_64_world_memory_admission_keeps_conservative_margin_and_128_default(tmp_path):
    evidence = {"worker_resources": [{"measurement": {"maxrss_kib": 3707260}}]}
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemAvailable: 366146160 kB\n")
    result = stage.memory_admission(evidence, meminfo=meminfo, count=64)
    assert result["required_kib"] == 304969408
    assert result["cpu_worker_measured_peak_kib"] == 3707260
    with pytest.raises(MemoryError, match="ABC128 requires estimated 601550208"):
        stage.memory_admission(evidence, meminfo=meminfo)
    meminfo.write_text("MemAvailable: 304969407 kB\n")
    with pytest.raises(MemoryError, match="ABC64"):
        stage.memory_admission(evidence, meminfo=meminfo, count=64)


@pytest.mark.parametrize("count", [2, 32, 64.0, True, 256])
def test_memory_admission_rejects_undeclared_world_counts(tmp_path, count):
    with pytest.raises(ValueError, match="world counts"):
        stage.memory_admission({}, meminfo=tmp_path / "absent", count=count)


def test_lease_refusal_precedes_output_or_child_creation(inputs, tmp_path, monkeypatch):
    output = tmp_path / "run"

    def reject():
        raise RuntimeError("No live lease")

    monkeypatch.setattr(stage, "require_subset_lease", reject)
    with pytest.raises(RuntimeError, match="No live lease"):
        stage.run(inputs[0], output)
    assert not output.exists()


def install_child_doubles(
    monkeypatch, tmp_path, *, learner_failure=False, bad_counts=False
):
    processes = []
    monkeypatch.setattr(stage, "ROOT", tmp_path)
    (tmp_path / "builds").mkdir()
    monkeypatch.setattr(stage, "require_subset_lease", lambda: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-explicit-test-double")
    monkeypatch.setattr(stage.os, "getpgid", lambda _pid: stage.os.getpgrp())
    monkeypatch.setattr(
        stage, "memory_admission", lambda _evidence: {"test_double": True}
    )
    description = {
        **stage.TASK,
        "reset_options": host.RESET_OPTIONS,
        "reset_seed_bank": {
            "base": 200000,
            "rank_stride": 1,
            "episode_stride": 128,
            "max_episode_index": 780,
            "fallback_offset": 100000,
        },
    }
    monkeypatch.setattr(stage, "wait_ready", lambda *_args: description)

    def preflight(command, **kwargs):
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        destination = command[5] if command[4] == stage.PREFLIGHT else command[6]
        write(Path(destination), {"test_double": True})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(stage.subprocess, "run", preflight)

    class ChildDouble:
        def __init__(self, command, **kwargs):
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-explicit-test-double"
            assert "start_new_session" not in kwargs
            self.pid, self.returncode = 2000000000 + len(processes), None
            self.role = (
                "host" if command[5] == "abc_bench.fasttd3_vector_host" else "learner"
            )
            output = Path(command[command.index("--output") + 1])
            if self.role == "host":
                write(
                    output / "receipt.json",
                    {
                        "status": "completed",
                        "description": description,
                        "vector_steps": 64,
                        "transitions": 8192,
                        "worker_cleanup": [
                            {
                                "rank": rank,
                                "pid": 1900000000 + rank,
                                "alive": False,
                                "exitcode": 0,
                            }
                            for rank in range(128)
                        ],
                    },
                )
            else:
                output.mkdir()
                checkpoint = output / "original.pt"
                checkpoint.write_bytes(b"explicit non-policy checkpoint double")
                write(
                    output / "receipt.json",
                    {
                        "status": "completed",
                        "controls": 64,
                        "configuration": copy.deepcopy(stage.RECIPE),
                        "actual_updates": {
                            "critic": 105 if bad_counts else 106,
                            "actor": 53,
                        },
                        "finite_gradient_samples": {"critic": 3, "actor": 2},
                        "checkpoint": stage.artifact(checkpoint),
                    },
                )
            processes.append(self)

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            self.returncode = 1 if self.role == "learner" and learner_failure else 0
            return self.returncode

        def terminate(self):
            self.returncode = -15

    monkeypatch.setattr(stage.subprocess, "Popen", ChildDouble)

    def rpc(_path, operation, *, timeout):
        assert operation == "shutdown" and timeout == 30
        processes[0].returncode = 0
        return {"status": "closed"}

    monkeypatch.setattr(stage, "rpc", rpc)
    return processes


def test_success_binds_real_counts_and_both_owned_child_exits(
    inputs, tmp_path, monkeypatch
):
    install_child_doubles(monkeypatch, tmp_path)
    result = stage.run(inputs[0], tmp_path / "run")
    assert result["status"] == "completed"
    assert result["metrics"] == {
        "vector_steps": 64,
        "transitions": 8192,
        "critic_updates": 106,
        "actor_updates": 53,
    }
    assert result["child_exit_codes"] == {"host": 0, "learner": 0}
    assert Path(result["export_check"]["path"]).is_file()
    assert list((tmp_path / "builds").iterdir()) == []


@pytest.mark.parametrize("learner_failure,bad_counts", [(True, False), (False, True)])
def test_failure_cleans_owned_children_and_preserves_failed_receipt(
    inputs, tmp_path, monkeypatch, learner_failure, bad_counts
):
    processes = install_child_doubles(
        monkeypatch, tmp_path, learner_failure=learner_failure, bad_counts=bad_counts
    )
    with pytest.raises(RuntimeError):
        stage.run(inputs[0], tmp_path / "run")
    result = json.loads((tmp_path / "run/receipt.json").read_text())
    assert result["status"] == "failed"
    assert all(child.poll() is not None for child in processes)
    assert list((tmp_path / "builds").iterdir()) == []
