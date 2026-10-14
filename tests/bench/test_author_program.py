"""Process supervision tests; these explicit stand-ins never simulate learning."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from abc_bench import author_program


def fixture_config(tmp_path, program_text):
    source = tmp_path / "source"
    files = []
    for name, text in {
        "train_offline_rtc.py": program_text,
        "expo_ft/agents/alg/realtime_expo_ft.py": "# Explicit process-test stand-in\n",
        "expo_ft/agents/alg/rtc.py": "# Explicit process-test stand-in\n",
    }.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
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
        "program": "train_offline_rtc.py",
        "source_root": str(source),
        "source_manifest": str(manifest),
        "python": sys.executable,
        "arguments": [],
        "timeout_s": 5,
    }


def test_zero_exit_is_not_checkpoint_acceptance(tmp_path, monkeypatch):
    monkeypatch.setattr(author_program, "require_subset_lease", lambda: None)
    result = author_program.run(
        fixture_config(tmp_path, "print('process fixture')\n"), tmp_path / "out"
    )
    assert result["status"] == "completed"
    assert result["checkpoint_accepted"] is False
    assert result["owned_children_stopped"] is True


def test_changed_source_refuses_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(author_program, "require_subset_lease", lambda: None)
    config = fixture_config(tmp_path, "print('process fixture')\n")
    Path(config["source_root"], "train_offline_rtc.py").write_text(
        "raise AssertionError('must not run')\n"
    )
    with pytest.raises(ValueError, match="differs"):
        author_program.run(config, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "program,timeout,accepted",
    [
        ("train_offline_rtc.py", 36000, True),
        ("train_offline_rtc.py", 36001, False),
        ("train_offline_rtc.py", float("inf"), False),
        ("train_offline_rtc.py", True, False),
        ("train_pi_robo.py", 36000, False),
        ("eval_droid_policy.py", 36000, False),
    ],
)
def test_extended_watchdog_is_offline_only(tmp_path, program, timeout, accepted):
    config = fixture_config(tmp_path, "raise AssertionError('must not run')\n")
    config.update(program=program, timeout_s=timeout)
    path = tmp_path / "worker.json"
    path.write_text(json.dumps(config))
    if accepted:
        assert author_program.load_config(path)["timeout_s"] == timeout
    else:
        with pytest.raises(ValueError, match="timeout_s"):
            author_program.load_config(path)


def test_extended_offline_job_retains_parent_cleanup_and_original_limits(tmp_path):
    from abc_bench import subset

    config = fixture_config(tmp_path, "raise AssertionError('must not run')\n")
    config["timeout_s"] = 36000
    worker_path = tmp_path / "worker.json"
    worker_path.write_text(json.dumps(config))
    job = {
        "schema_version": 1,
        "label": "offline RTC process-only fixture",
        "python": sys.executable,
        "module": "abc_bench.author_program",
        "arguments": ["--config", str(worker_path)],
        "max_wall_s": 36100,
    }
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(job))
    assert subset.load_job(job_path)["max_wall_s"] == 36100
    import argparse

    # Reject before touching the real ledger or GPU, even if the CLI lowers time.
    with pytest.raises(ValueError, match="Effective RTC parent"):
        subset.run_job(
            argparse.Namespace(training_config=job_path, timeout_seconds=36099)
        )
    for maximum in (120, 7200, 36099, 36101):
        job["max_wall_s"] = maximum
        job_path.write_text(json.dumps(job))
        with pytest.raises(ValueError):
            subset.load_job(job_path)
    job["max_wall_s"] = 36100
    job_path.write_text(json.dumps(job))
    config["program"] = "train_pi_robo.py"
    config["timeout_s"] = 7100
    worker_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="only for original offline RTC"):
        subset.load_job(job_path)


@pytest.mark.parametrize(
    "settings_key", ["REALTIME_EXPO_CATCHER_SETTINGS", "SADHANA_R1LITE_EXPO_SETTINGS"]
)
def test_task_settings_and_original_prng_reach_child(
    tmp_path, monkeypatch, settings_key
):
    monkeypatch.setattr(author_program, "require_subset_lease", lambda: None)
    settings = tmp_path / "catcher.json"
    settings.write_text('{"env_seed": 42}\n')
    config = fixture_config(
        tmp_path,
        "import json,os\nfrom pathlib import Path\n"
        f"settings = Path(os.environ[{settings_key!r}])\n"
        "assert json.loads(settings.read_text())['env_seed'] == 42\n"
        "assert os.environ['JAX_THREEFRY_PARTITIONABLE'] == 'false'\n",
    )
    config["environment"] = {
        settings_key: str(settings),
        "JAX_THREEFRY_PARTITIONABLE": "false",
    }
    config_path = tmp_path / "job.json"
    config_path.write_text(json.dumps(config))
    result = author_program.run(
        author_program.load_config(config_path), tmp_path / "out"
    )
    assert result["status"] == "completed"


def test_catcher_settings_do_not_allow_python_path_override(tmp_path):
    config = fixture_config(tmp_path, "raise AssertionError('must not run')\n")
    config["environment"] = {
        "REALTIME_EXPO_CATCHER_SETTINGS": str(tmp_path / "catcher.json"),
        "PYTHONPATH": str(tmp_path),
    }
    config_path = tmp_path / "job.json"
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="environment keys"):
        author_program.load_config(config_path)


def test_timeout_reaps_original_process(tmp_path, monkeypatch):
    monkeypatch.setattr(author_program, "require_subset_lease", lambda: None)
    config = fixture_config(tmp_path, "import time\ntime.sleep(30)\n")
    config["timeout_s"] = 0.1
    with pytest.raises(TimeoutError):
        author_program.run(config, tmp_path / "out")
    receipt = json.loads((tmp_path / "out/receipt.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["owned_children_stopped"] is True
    assert not Path(f"/proc/{receipt['author_pid']}").exists()


def test_failed_program_stays_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(author_program, "require_subset_lease", lambda: None)
    config = fixture_config(tmp_path, "raise RuntimeError('explicit test failure')\n")
    with pytest.raises(RuntimeError, match="exited 1"):
        author_program.run(config, tmp_path / "out")
    receipt = json.loads((tmp_path / "out/receipt.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["author_exit_code"] == 1
    assert receipt["owned_children_stopped"] is True


def test_graceful_client_interrupt_flushes_exit_cleanup(tmp_path):
    marker = tmp_path / "closed.txt"
    code = (
        "import atexit,time\nfrom pathlib import Path\n"
        f"atexit.register(lambda:Path({str(marker)!r}).write_text('closed'))\n"
        "print('ready',flush=True)\ntime.sleep(60)\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-u", "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        assert child.stdout.readline().strip() == "ready"
        assert author_program._stop(child, graceful=True) == "sigint"
        assert marker.read_text() == "closed"
        assert child.poll() is not None
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
