"""CPU-only profile and retained-child checks; no learner, JAX or native imports."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from abc_bench import author_program as author
from abc_bench import runner, subset


def fixture_config(tmp_path, text="print('CPU supervisor fixture')\n"):
    source = tmp_path / "source"
    files = []
    for name, contents in {
        "train_pi_robo.py": text,
        "expo_ft/agents/alg/realtime_expo_ft.py": "# Explicit CPU stand-in\n",
        "expo_ft/agents/alg/rtc.py": "# Explicit CPU stand-in\n",
    }.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
        data = path.read_bytes()
        files.append(
            {
                "path": name,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": files}))
    return {
        "schema_version": 1,
        "program": "train_pi_robo.py",
        "python": sys.executable,
        "source_root": str(source),
        "source_manifest": str(manifest),
        "arguments": [],
        "timeout_s": 5,
        "memory_profile": author.UNIFIED_MEMORY_PROFILE,
        "environment": dict(author.UNIFIED_MEMORY_ENV),
    }


def write_worker(tmp_path, config):
    path = tmp_path / "worker.json"
    path.write_text(json.dumps(config))
    return path


def write_job(tmp_path, worker):
    path = tmp_path / "job.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "label": "CPU profile fixture",
                "python": sys.executable,
                "module": "abc_bench.author_program",
                "arguments": ["--config", str(worker)],
                "max_wall_s": 105,
            }
        )
    )
    return path


def test_profile_keeps_original_timeout_and_cleanup_limits(tmp_path):
    config = fixture_config(tmp_path)
    config["timeout_s"] = 21600
    assert author.load_config(write_worker(tmp_path, config))["timeout_s"] == 21600
    config["timeout_s"] = 21601
    with pytest.raises(ValueError, match="timeout_s"):
        author.load_config(write_worker(tmp_path, config))
    assert author.TRAINING_WATCHDOG_LIMITS["train_pi_robo.py"] == (21600, 100)


@pytest.mark.parametrize(
    "key,value",
    [
        ("XLA_PYTHON_CLIENT_MEM_FRACTION", "1.6"),
        ("XLA_PYTHON_CLIENT_PREALLOCATE", "true"),
        ("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async"),
        ("TF_FORCE_UNIFIED_MEMORY", "false"),
        ("JAX_THREEFRY_PARTITIONABLE", "false"),
    ],
)
def test_changed_profile_refuses_before_canonical_outputs_or_gpu(
    tmp_path, monkeypatch, key, value
):
    config = fixture_config(tmp_path)
    config["environment"][key] = value
    job = write_job(tmp_path, write_worker(tmp_path, config))
    results = tmp_path / "must-not-exist"
    monkeypatch.setattr(runner, "RESULTS", results)
    monkeypatch.setattr(
        subset, "inspect_reserved_gpu", lambda _: pytest.fail("No GPU inspection")
    )
    with pytest.raises(ValueError, match="unified-memory profile"):
        subset.run_job(argparse.Namespace(training_config=job, timeout_seconds=105))
    assert not results.exists()


@pytest.mark.parametrize(
    "change", ["no_profile", "unknown_profile", "offline", "eval", "schema2"]
)
def test_profile_cannot_leak_to_other_programs_or_implicit_environment(
    tmp_path, change
):
    config = fixture_config(tmp_path)
    if change == "no_profile":
        del config["memory_profile"]
    elif change == "unknown_profile":
        config["memory_profile"] = "anything"
    elif change == "schema2":
        config["schema_version"] = 2
    else:
        config["program"] = (
            "train_offline_rtc.py" if change == "offline" else "eval_droid_policy.py"
        )
    with pytest.raises(ValueError, match="unified-memory profile"):
        subset.load_job(write_job(tmp_path, write_worker(tmp_path, config)))


def test_low_host_refuses_before_ledger_gpu_and_child(tmp_path, monkeypatch):
    config = fixture_config(tmp_path)
    job = write_job(tmp_path, write_worker(tmp_path, config))
    results = tmp_path / "must-not-exist"
    monkeypatch.setattr(runner, "RESULTS", results)
    monkeypatch.setattr(
        subset, "inspect_reserved_gpu", lambda _: pytest.fail("No GPU inspection")
    )
    monkeypatch.setattr(
        author, "host_available_bytes", lambda: author.UNIFIED_MEMORY_START_BYTES - 1
    )
    with pytest.raises(RuntimeError, match=">=160 GiB"):
        subset.run_job(argparse.Namespace(training_config=job, timeout_seconds=105))
    monkeypatch.setattr(author, "require_subset_lease", lambda: None)
    with pytest.raises(RuntimeError, match=">=160 GiB"):
        author.run(config, tmp_path / "out")
    assert not results.exists() and not (tmp_path / "out").exists()


def test_short_effective_cli_retains_cleanup_margin(tmp_path, monkeypatch):
    config = fixture_config(tmp_path)
    job = write_job(tmp_path, write_worker(tmp_path, config))
    monkeypatch.setattr(
        author, "host_available_bytes", lambda: author.UNIFIED_MEMORY_START_BYTES
    )
    with pytest.raises(ValueError, match="Effective author training parent"):
        subset.run_job(argparse.Namespace(training_config=job, timeout_seconds=104))


def test_actual_child_receives_explicit_profile_without_inherited_overrides(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(author, "require_subset_lease", lambda: None)
    monkeypatch.setattr(
        author, "host_available_bytes", lambda: author.UNIFIED_MEMORY_START_BYTES
    )
    inherited = {
        "XLA_FLAGS": "--bad",
        "XLA_CLIENT_MEM_FRACTION": ".1",
        "TF_GPU_ALLOCATOR": "bad",
        "TF_FORCE_GPU_ALLOW_GROWTH": "true",
        "CUDA_MANAGED_FORCE_DEVICE_ALLOC": "1",
        "JAX_DEBUG_NANS": "true",
        "JAX_THREEFRY_PARTITIONABLE": "false",
    }
    for key, value in inherited.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "CPU-fixture-lease")
    child = (
        "import os\n"
        f"expected={author.UNIFIED_MEMORY_ENV!r}\n"
        "assert all(os.environ.get(k)==v for k,v in expected.items())\n"
        f"assert not (set({list(inherited)!r})-set(expected)) & os.environ.keys()\n"
        "assert os.environ['CUDA_VISIBLE_DEVICES']=='CPU-fixture-lease'\n"
    )
    receipt = author.run(fixture_config(tmp_path, child), tmp_path / "out")
    assert receipt["status"] == "completed" and receipt["checkpoint_accepted"] is False
    assert receipt["author_shutdown"] == "already_stopped"
    assert receipt["memory_environment"] == author.UNIFIED_MEMORY_ENV
    assert not Path(f"/proc/{receipt['author_pid']}").exists()


@pytest.mark.parametrize("failure", ["floor", "observation_lost"])
def test_actual_child_is_reaped_when_memory_monitor_fails(
    tmp_path, monkeypatch, failure
):
    monkeypatch.setattr(author, "require_subset_lease", lambda: None)
    calls = [0]

    def available():
        calls[0] += 1
        if calls[0] <= 2:
            return author.UNIFIED_MEMORY_START_BYTES
        if failure == "observation_lost":
            raise ValueError("Explicit host observation failure")
        return author.UNIFIED_MEMORY_FLOOR_BYTES - 1

    monkeypatch.setattr(author, "host_available_bytes", available)
    with pytest.raises((ValueError, RuntimeError)):
        author.run(
            fixture_config(tmp_path, "import time\ntime.sleep(30)\n"), tmp_path / "out"
        )
    receipt = json.loads((tmp_path / "out/receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["owned_children_stopped"]
    assert not Path(f"/proc/{receipt['author_pid']}").exists()


def test_legacy_environment_is_unchanged_and_does_not_read_host_memory(
    tmp_path, monkeypatch
):
    config = fixture_config(tmp_path)
    del config["memory_profile"]
    config["environment"] = {"JAX_THREEFRY_PARTITIONABLE": "false"}
    monkeypatch.setattr(
        author, "host_available_bytes", lambda: pytest.fail("Legacy has no host gate")
    )
    env = dict(os.environ, XLA_FLAGS="legacy-value")
    assert author.admit_host_memory(config) is None
    assert author.unified_memory_environment(config, env) is env
    assert (
        author.load_config(write_worker(tmp_path, config))["environment"]
        == config["environment"]
    )


def add_cpu_client(config, text):
    source = Path(config["source_root"])
    manifest = Path(config["source_manifest"])
    entries = json.loads(manifest.read_text())
    for name, contents in {
        "client/__init__.py": "",
        "client/run_client.py": text,
    }.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
        data = path.read_bytes()
        entries["files"].append(
            {
                "path": name,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    manifest.write_text(json.dumps(entries))
    config["client"] = {"python": sys.executable, "arguments": ["--exit-after-session"]}


@pytest.mark.parametrize("failure", ["floor", "value_error", "os_error"])
def test_memory_failure_reaps_actual_author_before_client_cleanup(
    tmp_path, monkeypatch, failure
):
    monkeypatch.setattr(author, "require_subset_lease", lambda: None)
    calls = [0]

    def available():
        calls[0] += 1
        if calls[0] <= 2:
            return author.UNIFIED_MEMORY_START_BYTES
        if failure == "value_error":
            raise ValueError("Explicit host observation failure")
        if failure == "os_error":
            raise OSError("Explicit host observation read failure")
        return author.UNIFIED_MEMORY_FLOOR_BYTES - 1

    monkeypatch.setattr(author, "host_available_bytes", available)
    config = fixture_config(tmp_path, "import time\ntime.sleep(30)\n")
    add_cpu_client(config, "import time\ntime.sleep(30)\n")
    actual_stop = author._stop
    stopped = []

    def observed_stop(process, *, graceful=False):
        assert process is not None
        if graceful:
            assert len(stopped) == 1
            assert stopped[0][1] is False
            assert stopped[0][0].poll() is not None
        outcome = actual_stop(process, graceful=graceful)
        stopped.append((process, graceful))
        return outcome

    monkeypatch.setattr(author, "_stop", observed_stop)
    with pytest.raises((OSError, ValueError, RuntimeError)):
        author.run(config, tmp_path / "out")
    receipt = json.loads((tmp_path / "out/receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["owned_children_stopped"]
    assert receipt["host_memory"]["monitor_failed"] is True
    assert receipt["cleanup_order"] == ["author", "client"]
    assert [process.pid for process, _ in stopped] == [
        receipt["author_pid"],
        receipt["client_pid"],
    ]
    assert [graceful for _, graceful in stopped] == [False, True]
    assert all(process.returncode is not None for process, _ in stopped)
    assert all(not Path(f"/proc/{process.pid}").exists() for process, _ in stopped)


@pytest.mark.parametrize("profile", [False, True])
def test_normal_actual_children_keep_client_first_cleanup(
    tmp_path, monkeypatch, profile
):
    monkeypatch.setattr(author, "require_subset_lease", lambda: None)
    monkeypatch.setattr(
        author, "host_available_bytes", lambda: author.UNIFIED_MEMORY_START_BYTES
    )
    config = fixture_config(tmp_path, "import time\ntime.sleep(0.2)\n")
    if not profile:
        del config["memory_profile"]
        config["environment"] = {}
    add_cpu_client(config, "print('CPU client completed')\n")
    actual_stop = author._stop
    stopped = []

    def observed_stop(process, *, graceful=False):
        assert process is not None
        outcome = actual_stop(process, graceful=graceful)
        stopped.append((process, graceful))
        return outcome

    monkeypatch.setattr(author, "_stop", observed_stop)
    receipt = author.run(config, tmp_path / "out")
    assert receipt["status"] == "completed" and receipt["owned_children_stopped"]
    assert "cleanup_order" not in receipt
    assert not receipt.get("host_memory", {}).get("monitor_failed", False)
    assert [process.pid for process, _ in stopped] == [
        receipt["client_pid"],
        receipt["author_pid"],
    ]
    assert [graceful for _, graceful in stopped] == [True, False]
    assert all(not Path(f"/proc/{process.pid}").exists() for process, _ in stopped)
