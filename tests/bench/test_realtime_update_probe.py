"""CPU-only boundary checks; child fixtures do not import a learner or JAX."""

import json
import os
import sys
import time
from pathlib import Path

import pytest

from abc_bench import realtime_update_probe as probe


def test_mem_available_requires_exact_field_and_unit(tmp_path):
    path = tmp_path / "meminfo"
    path.write_text("MemFree: 9 kB\nMemAvailable: 12345 kB\n")
    assert probe.host_available(path) == 12345 * 1024
    path.write_text("MemAvailable: 12345 MB\n")
    with pytest.raises(ValueError):
        probe.host_available(path)


def test_memory_environment_removes_conflicting_inherited_values(monkeypatch):
    monkeypatch.setenv("XLA_CLIENT_MEM_FRACTION", "0.2")
    monkeypatch.setenv("XLA_FLAGS", "--unexpected")
    monkeypatch.setenv("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    monkeypatch.setenv("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    monkeypatch.setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1")
    monkeypatch.setenv("JAX_THREEFRY_PARTITIONABLE", "false")
    monkeypatch.setenv("JAX_DEFAULT_PRNG_IMPL", "unsafe_rbg")
    monkeypatch.setenv("JAX_DEBUG_NANS", "true")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "leased-GPU")
    env = probe.memory_environment(
        {"environment": {"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.90"}}
    )
    assert (
        not {
            "XLA_CLIENT_MEM_FRACTION",
            "XLA_FLAGS",
            "TF_GPU_ALLOCATOR",
            "TF_FORCE_GPU_ALLOW_GROWTH",
            "CUDA_MANAGED_FORCE_DEVICE_ALLOC",
            "JAX_DEBUG_NANS",
        }
        & env.keys()
    )
    assert all(env[k] == v for k, v in probe.MEMORY_ENV.items())
    assert all(env[k] == v for k, v in probe.JAX_ENV.items())
    assert env["CUDA_VISIBLE_DEVICES"] == "leased-GPU"


@pytest.fixture
def valid_job(tmp_path, monkeypatch):
    online = tmp_path / "online.json"
    online.write_text(json.dumps({"python": sys.executable}))
    monkeypatch.setattr(
        probe, "load_config", lambda _path: {"online_config": {"path": str(online)}}
    )
    monkeypatch.setattr(probe, "host_available", lambda: probe.MIN_START_BYTES)
    return {
        "module": probe.MODULE,
        "max_wall_s": 2400,
        "task": "r1lite_remote_into_organizer",
        "embodiment": "r1lite_2025",
        "arguments": ["--config", str(tmp_path / "config.json")],
        "python": sys.executable,
    }


def test_exact_parent_bound_accepts(valid_job):
    probe.validate_job(valid_job, 2400)


@pytest.mark.parametrize("seconds", [2399, float("nan"), float("inf"), True])
def test_effective_parent_bound_rejects(valid_job, seconds):
    with pytest.raises(ValueError):
        probe.validate_job(valid_job, seconds)


def test_host_admission_rejects_before_child(valid_job, monkeypatch):
    monkeypatch.setattr(probe, "host_available", lambda: probe.MIN_START_BYTES - 1)
    with pytest.raises(RuntimeError, match="at admission"):
        probe.validate_job(valid_job, 2400)


def test_installed_server_interpreter_required(valid_job):
    valid_job["python"] = "/different/python"
    with pytest.raises(ValueError, match="server interpreter"):
        probe.validate_job(valid_job, 2400)


def test_inherited_task_label_cannot_be_abc(valid_job):
    valid_job["embodiment"] = "native_yam"
    with pytest.raises(ValueError):
        probe.validate_job(valid_job, 2400)


def test_child_clean_completion_has_reaped_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "host_available", lambda: probe.MIN_START_BYTES)
    result = probe.watch_child(
        [sys.executable, "-I", "-c", "print('actual CPU child')"],
        dict(os.environ),
        tmp_path,
        time.monotonic() + 10,
        [False],
    )
    assert result["child_exit_code"] == 0 and result["child_reaped"]
    assert result["child_start_ticks"] > 0 and result["boot_id"]
    assert result["stop_reason"] is None
    assert "actual CPU child" in (tmp_path / "probe.log").read_text()


def test_child_nonzero_is_not_success(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "host_available", lambda: probe.MIN_START_BYTES)
    result = probe.watch_child(
        [sys.executable, "-I", "-c", "raise SystemExit(7)"],
        dict(os.environ),
        tmp_path,
        time.monotonic() + 10,
        [False],
    )
    assert result["child_exit_code"] == 7 and result["child_reaped"]


@pytest.mark.parametrize("reason", ["host", "deadline", "cancel"])
def test_actual_child_stop_is_bounded_and_reaped(tmp_path, monkeypatch, reason):
    calls = [0]

    def available():
        calls[0] += 1
        return (
            probe.MIN_RUN_BYTES - 1
            if reason == "host" and calls[0] > 1
            else probe.MIN_START_BYTES
        )

    monkeypatch.setattr(probe, "host_available", available)
    started = time.monotonic()
    result = probe.watch_child(
        [sys.executable, "-I", "-c", "import time; time.sleep(30)"],
        dict(os.environ),
        tmp_path,
        started - 1 if reason == "deadline" else started + 20,
        [reason == "cancel"],
    )
    assert time.monotonic() - started < 16
    assert result["child_reaped"] and result["child_exit_code"] != 0
    assert (
        result["stop_reason"]
        == {
            "host": "host_available_below_48_GiB",
            "deadline": "worker_deadline",
            "cancel": "cancelled",
        }[reason]
    )
    assert not Path(f"/proc/{result['child_pid']}").exists()


def test_unreadable_monitor_still_reaps_child(tmp_path, monkeypatch):
    calls = [0]

    def unavailable():
        calls[0] += 1
        if calls[0] > 1:
            raise ValueError("host observation lost")
        return probe.MIN_START_BYTES

    monkeypatch.setattr(probe, "host_available", unavailable)
    with pytest.raises(ValueError, match="observation lost"):
        probe.watch_child(
            [sys.executable, "-I", "-c", "import time; time.sleep(30)"],
            dict(os.environ),
            tmp_path,
            time.monotonic() + 20,
            [False],
        )
    receipt = json.loads((tmp_path / "child.json").read_text())
    assert receipt["child_reaped"]
    assert not Path(f"/proc/{receipt['child_pid']}").exists()
