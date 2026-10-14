"""Installed CPU process and metadata fixtures; no author learner or native job."""

import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from abc_bench import fasttd3_program as worker
from abc_bench.subset import load_job

ORIGINAL_VERIFY_RUNTIME = worker.verify_runtime


def commit_fixture(source):
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=CPU Process Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "--no-gpg-sign",
            "-qm",
            "Explicit metadata/process fixture",
        ],
        check=True,
    )
    return subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()


@pytest.fixture
def config(tmp_path, monkeypatch):
    # The fake train.py is a stdlib process fixture, never native author code.
    source = tmp_path / "source"
    script = source / "fast_td3/train.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "print('Explicit CPU process fixture, no simulator or learner')\n"
    )
    hb = tmp_path / "humanoid-source"
    (hb / "humanoid_bench").mkdir(parents=True)
    (hb / "humanoid_bench/__init__.py").write_text("# Explicit fixture\n")
    commit, hb_commit = commit_fixture(source), commit_fixture(hb)
    monkeypatch.setattr(worker, "AUTHOR_COMMIT", commit)
    monkeypatch.setattr(worker, "HUMANOID_COMMIT", hb_commit)
    monkeypatch.setattr(worker, "require_subset_lease", lambda: None)
    monkeypatch.setattr(
        worker,
        "verify_runtime",
        lambda config: {"scope": "Explicit CPU runtime metadata fixture"},
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-EXPLICIT-CPU-PROCESS-FIXTURE")
    monkeypatch.setenv("MUJOCO_GL", "disable")
    return {
        "schema_version": 1,
        "python": sys.executable,
        "source_root": str(source),
        "source_commit": commit,
        "humanoid_source_root": str(hb),
        "humanoid_source_commit": hb_commit,
        "phase": "original_humanoidbench_train",
        "arguments": ["--env-name", worker.TASK, "--seed", "1"],
        "timeout_s": 3,
    }


def write_config(tmp_path, config):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


def replace_fixture(config, body):
    source = Path(config["source_root"])
    (source / "fast_td3/train.py").write_text(body)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=CPU Process Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "--no-gpg-sign",
            "-qm",
            "Updated explicit process fixture",
        ],
        check=True,
    )
    config["source_commit"] = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()


def test_direct_original_argv_fresh_cwd_uuid_egl_tracking_and_no_score(
    tmp_path, monkeypatch, config
):
    replace_fixture(
        config,
        "import json,os,sys\nfrom pathlib import Path\nfields=['CUDA_VISIBLE_DEVICES','MUJOCO_GL','MUJOCO_EGL_DEVICE_ID','WANDB_MODE','WANDB_DIR','TMPDIR','PATH']\ndata={k:os.environ[k] for k in fields}\ndata.update(argv=sys.argv[1:],cwd=os.getcwd(),pythonpath=os.environ.get('PYTHONPATH'))\nPath('observed.json').write_text(json.dumps(data))\n",
    )
    monkeypatch.setattr(worker, "AUTHOR_COMMIT", config["source_commit"])
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "prohibited-injection"))
    loaded = worker.load_config(write_config(tmp_path, config))
    result = worker.run(loaded, tmp_path / "out")
    observed = json.loads((tmp_path / "out/author-output/observed.json").read_text())
    assert observed["argv"] == config["arguments"]
    assert result["command"][:4] == [
        sys.executable,
        "-u",
        "-B",
        str(Path(config["source_root"], "fast_td3/train.py")),
    ]
    assert observed["cwd"] == str(tmp_path / "out/author-output")
    assert observed["PATH"] == os.environ["PATH"] and observed["pythonpath"] is None
    assert observed["CUDA_VISIBLE_DEVICES"] == "GPU-EXPLICIT-CPU-PROCESS-FIXTURE"
    assert observed["MUJOCO_GL"] == "egl" and observed["MUJOCO_EGL_DEVICE_ID"] == "0"
    assert os.environ["MUJOCO_GL"] == "disable" and observed["WANDB_MODE"] == "offline"
    assert (
        result["status"] == "completed"
        and result["experiment_accepted"] is False
        and result["metrics"] == {}
    )
    assert (
        result["descendant_cleanup_owner"] == "canonical runner"
        and result["direct_child_stopped"]
    )
    assert (
        result["task"] == worker.TASK
        and result["embodiment"] == "unitree_h1_with_hands"
    )
    with pytest.raises(FileExistsError):
        worker.run(loaded, tmp_path / "out")


def test_nonzero_exit_remains_failed(tmp_path, monkeypatch, config):
    replace_fixture(config, "raise SystemExit(75)\n")
    monkeypatch.setattr(worker, "AUTHOR_COMMIT", config["source_commit"])
    with pytest.raises(RuntimeError, match="exited75"):
        worker.run(config, tmp_path / "out")
    receipt = json.loads((tmp_path / "out/receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["author_exit_code"] == 75
    assert receipt["direct_child_stopped"] and not receipt["experiment_accepted"]


def test_timeout_reaps_direct_child(tmp_path, monkeypatch, config):
    replace_fixture(config, "import time\ntime.sleep(30)\n")
    monkeypatch.setattr(worker, "AUTHOR_COMMIT", config["source_commit"])
    config["timeout_s"] = 0.05
    with pytest.raises(subprocess.TimeoutExpired):
        worker.run(config, tmp_path / "out")
    receipt = json.loads((tmp_path / "out/receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["direct_child_stopped"]
    assert not Path(f"/proc/{receipt['author_pid']}").exists()


def test_sigterm_records_failure_and_restores_handler(tmp_path, monkeypatch, config):
    replace_fixture(config, "import time\ntime.sleep(30)\n")
    monkeypatch.setattr(worker, "AUTHOR_COMMIT", config["source_commit"])
    original, timers = subprocess.Popen, []
    previous = signal.getsignal(signal.SIGTERM)

    def start(*args, **kwargs):
        process = original(*args, **kwargs)
        timer = threading.Timer(0.05, lambda: os.kill(os.getpid(), signal.SIGTERM))
        timers.append(timer)
        timer.start()
        return process

    # Source git reads precede Popen; schedule only once the child exists.
    monkeypatch.setattr(
        worker,
        "verify_source",
        lambda _: {"scope": "Explicit cancellation metadata fixture"},
    )
    monkeypatch.setattr(worker.subprocess, "Popen", start)
    with pytest.raises(InterruptedError, match="cancelled"):
        worker.run(config, tmp_path / "out")
    for timer in timers:
        timer.join()
    receipt = json.loads((tmp_path / "out/receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["direct_child_stopped"]
    assert signal.getsignal(signal.SIGTERM) == previous


def test_source_change_and_missing_lease_reject_before_outputs(
    tmp_path, monkeypatch, config
):
    (Path(config["source_root"]) / "shadow.py").write_text(
        "# Must not affect original imports\n"
    )
    with pytest.raises(ValueError, match="clean"):
        worker.run(config, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    monkeypatch.setattr(
        worker,
        "require_subset_lease",
        lambda: (_ for _ in ()).throw(RuntimeError("Missing fixture lease")),
    )
    with pytest.raises(RuntimeError, match="lease"):
        worker.run(config, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "change",
    [
        {"timeout_s": float("inf")},
        {"timeout_s": True},
        {"timeout_s": 28801},
        {"arguments": ["--env-name", "h1hand-stand-v0", "--seed", "1"]},
        {"arguments": ["--env-name", worker.TASK, "--seed", "1", "--num-envs", "1"]},
        {
            "arguments": [
                "--env-name",
                worker.TASK,
                "--seed",
                "1",
                "--total-timesteps",
                "1",
            ]
        },
        {
            "arguments": [
                "--env-name",
                worker.TASK,
                "--seed",
                "1",
                "--checkpoint-path",
                "/tmp/weights.pt",
            ]
        },
        {
            "arguments": [
                "--env-name",
                worker.TASK,
                "--seed",
                "1",
                "--exp-name",
                "../../escape",
            ]
        },
        {"environment": {"CUDA_VISIBLE_DEVICES": "1"}},
    ],
)
def test_method_or_boundary_override_rejected(tmp_path, config, change):
    config.update(change)
    with pytest.raises(ValueError):
        worker.load_config(write_config(tmp_path, config))


def test_narrow_long_admission_and_preserved_existing_bounds(tmp_path, config):
    config["timeout_s"] = 28800
    path = write_config(tmp_path, config)
    job = {
        "schema_version": 1,
        "python": sys.executable,
        "module": "abc_bench.fasttd3_program",
        "arguments": ["--config", str(path)],
        "label": "Explicit metadata fixture",
        "max_wall_s": 28860,
    }
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(job))
    assert load_job(job_path)["max_wall_s"] == 28860
    for value in (28859, 28861):
        job["max_wall_s"] = value
        job_path.write_text(json.dumps(job))
        with pytest.raises(ValueError):
            load_job(job_path)
    job.update(module="abc_bench.author_program", arguments=[], max_wall_s=7201)
    job_path.write_text(json.dumps(job))
    with pytest.raises(ValueError, match="7200"):
        load_job(job_path)


def test_qualification_is_explicitly_separate(tmp_path, config):
    config.update(phase="original_humanoidbench_qualification", timeout_s=1800)
    config["arguments"] += ["--total-timesteps", "12"]
    assert worker.load_config(write_config(tmp_path, config))["phase"].endswith(
        "qualification"
    )
    config["arguments"][-1] = "100000"
    with pytest.raises(ValueError, match="Qualification"):
        worker.load_config(write_config(tmp_path, config))


def test_normal_runtime_metadata_complete_payloads_and_rejection(
    tmp_path, monkeypatch, config
):
    result = {
        "prefix": str(Path(sys.prefix).resolve()),
        "distributions": {},
        "versions": {"torch": "2.6.0+cu124", "tensordict": "0.7.2", "torchrl": "0.7.2"},
    }
    for name, namespace, field, version in (
        ("fast-td3", "fast_td3", "source_root", "0.1.0"),
        ("humanoid-bench", "humanoid_bench", "humanoid_source_root", "0.2"),
    ):
        source = Path(config[field])
        files = {
            str(p.relative_to(source)): {
                "path": str(Path(sys.prefix) / "lib" / str(p.relative_to(source))),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in (source / namespace).rglob("*")
            if p.is_file()
        }
        result["distributions"][name] = {"version": version, "files": files}

    def metadata(command, **kwargs):
        assert command[:4] == [sys.executable, "-I", "-B", "-c"]
        assert "from importlib.metadata import distribution" in command[-1]
        assert "import torch" not in command[-1] and "sys.path" not in command[-1]
        return json.dumps(result)

    monkeypatch.setattr(worker.subprocess, "check_output", metadata)
    assert ORIGINAL_VERIFY_RUNTIME(config)["distributions"]["fast-td3"]["payloads"] == 1
    result["distributions"]["humanoid-bench"]["files"].clear()
    with pytest.raises(ValueError, match="payloads"):
        ORIGINAL_VERIFY_RUNTIME(config)


def test_qualification_parent_bound_not_full_eight_hours(tmp_path, config):
    config.update(phase="original_humanoidbench_qualification", timeout_s=1800)
    config["arguments"] += ["--total-timesteps", "12"]
    path = write_config(tmp_path, config)
    job = {
        "schema_version": 1,
        "python": sys.executable,
        "module": "abc_bench.fasttd3_program",
        "arguments": ["--config", str(path)],
        "label": "Explicit qualification metadata fixture",
        "max_wall_s": 1861,
    }
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(job))
    with pytest.raises(ValueError, match="1860"):
        load_job(job_path)
