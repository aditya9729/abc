"""Explicit CPU fixtures for original-program supervision; no simulator or GPU."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from abc_bench import wbc_program as wbc
from abc_bench.subset import load_job


@pytest.fixture
def setup(tmp_path, monkeypatch):
    source = tmp_path / "source"
    script = source / "scripts/train_feedback_residual.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import pathlib,json,sys\np=pathlib.Path(sys.argv[sys.argv.index('--output')+1]);p.mkdir()\n(p/'training_summary.json').write_text(json.dumps({'status':'complete','optimizer_steps':40}))\n"
    )
    (source / "src/fixture").mkdir(parents=True)
    (source / "src/fixture/__init__.py").write_text("# Explicit CPU fixture\n")
    interpreter = tmp_path / "prefix/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(Path(sys._base_executable).resolve())
    installed = tmp_path / "prefix/lib/python3.12/site-packages/fixture/__init__.py"
    installed.parent.mkdir(parents=True)
    installed.write_bytes((source / "src/fixture/__init__.py").read_bytes())
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "--no-gpg-sign",
            "-qm",
            "CPU fixture",
        ],
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    monkeypatch.setattr(wbc, "AUTHOR_ROOT", source)
    monkeypatch.setattr(wbc, "AUTHOR_COMMIT", commit)
    monkeypatch.setattr(wbc, "AUTHOR_PYTHON", interpreter)
    monkeypatch.setattr(wbc, "require_subset_lease", lambda: None)
    model = tmp_path / "model.xml"
    model.write_text("Explicit non-simulator fixture")
    pins = [interpreter, script, source / "src/fixture/__init__.py", installed, model]
    manifest = tmp_path / "inputs.json"
    manifest.write_text(
        json.dumps({"files": [{"path": str(p), "sha256": wbc.digest(p)} for p in pins]})
    )
    cfg = {
        "schema_version": 1,
        "python": str(interpreter),
        "source_root": str(source),
        "source_commit": commit,
        "program": "pose-ppo",
        "arguments": [
            "--model",
            str(model),
            "--output",
            "{output}",
            "--device",
            "cuda:0",
        ],
        "timeout_s": 3,
        "input_manifest": str(manifest),
        "expected_summary": {"status": "complete", "optimizer_steps": 40},
    }
    return cfg, tmp_path / "config.json", script


def load(setup):
    cfg, path, _ = setup
    path.write_text(json.dumps(cfg))
    return wbc.load_config(path)


def test_canonical_child_and_summary(tmp_path, setup):
    result = wbc.run(load(setup), tmp_path / "out")
    assert result["status"] == "completed" and result["direct_child_stopped"]
    assert result["summary"]["optimizer_steps"] == 40
    assert result["experiment_accepted"] is False
    assert result["descendant_cleanup_owner"] == "canonical runner"
    assert not Path(f"/proc/{result['author_pid']}").exists()
    assert wbc.verify_inputs(setup[0])["files"] == 5


@pytest.mark.parametrize(
    "change",
    [
        {"timeout_s": float("inf")},
        {"timeout_s": True},
        {"timeout_s": 7101},
        {"program": "arbitrary"},
        {"python": "/usr/bin/python"},
        {"arguments": ["--model", "/tmp", "--output", "/tmp", "--device", "cuda:0"]},
        {
            "arguments": [
                "--model",
                "/tmp",
                "--output",
                "{output}",
                "--device",
                "cuda:0",
                "--evil",
                "x",
            ]
        },
        {
            "arguments": [
                "--model",
                "/tmp",
                "--output",
                "{output}",
                "--device",
                "cuda:0",
                "--device",
                "cuda:0",
            ]
        },
        {"expected_summary": {}},
    ],
)
def test_bad_configuration(setup, change):
    setup[0].update(change)
    with pytest.raises((ValueError, FileNotFoundError)):
        load(setup)


def test_parent_cleanup_margin(setup):
    load(setup)
    job = {
        "module": "abc_bench.wbc_program",
        "arguments": ["--config", str(setup[1])],
        "max_wall_s": 103,
    }
    wbc.validate_job(job, 103)
    with pytest.raises(ValueError, match="plus100s"):
        wbc.validate_job(job, 102)


def test_absent_lease_refuses_before_output(tmp_path, setup, monkeypatch):
    def fail():
        raise RuntimeError("No lease fixture")

    monkeypatch.setattr(wbc, "require_subset_lease", fail)
    with pytest.raises(RuntimeError, match="No lease"):
        wbc.run(load(setup), tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_changed_source_and_missing_pin_refuse(tmp_path, setup):
    cfg = load(setup)
    setup[2].write_text("modified\n")
    with pytest.raises(ValueError, match="clean"):
        wbc.run(cfg, tmp_path / "out")
    subprocess.run(
        [
            "git",
            "-C",
            cfg["source_root"],
            "checkout",
            "--",
            "scripts/train_feedback_residual.py",
        ],
        check=True,
    )
    manifest = Path(cfg["input_manifest"])
    data = json.loads(manifest.read_text())
    data["files"].pop()
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="omits"):
        wbc.run(cfg, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_timeout_and_bad_exit_are_preserved(tmp_path, setup, monkeypatch):
    cfg = load(setup)
    monkeypatch.setattr(wbc, "verify_inputs", lambda _: {"fixture": True})
    setup[2].write_text("import time\ntime.sleep(30)\n")
    cfg["timeout_s"] = 0.05
    with pytest.raises(subprocess.TimeoutExpired):
        wbc.run(cfg, tmp_path / "timeout")
    receipt = json.loads((tmp_path / "timeout/receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["direct_child_stopped"]
    assert not Path(f"/proc/{receipt['author_pid']}").exists()
    setup[2].write_text("raise SystemExit(75)\n")
    cfg["timeout_s"] = 3
    with pytest.raises(RuntimeError, match="exited 75"):
        wbc.run(cfg, tmp_path / "exit")
    receipt = json.loads((tmp_path / "exit/receipt.json").read_text())
    assert receipt["author_exit_code"] == 75 and receipt["status"] == "failed"


def test_summary_mismatch_is_not_completion(tmp_path, setup):
    cfg = load(setup)
    cfg["expected_summary"]["optimizer_steps"] = 41
    with pytest.raises(ValueError, match="summary differs"):
        wbc.run(cfg, tmp_path / "out")
    assert json.loads((tmp_path / "out/receipt.json").read_text())["status"] == "failed"


def test_disk_guard(tmp_path, monkeypatch):
    class Low:
        free = 19 * wbc.GIB

    monkeypatch.setattr(wbc.shutil, "disk_usage", lambda _: Low())
    with pytest.raises(RuntimeError, match="disk floor"):
        wbc.storage_check(tmp_path)


def test_pins_rechecked_after_success(tmp_path, setup, monkeypatch):
    cfg = load(setup)
    original = wbc.verify_inputs
    calls = []

    def changed(value):
        calls.append(1)
        if len(calls) > 1:
            raise ValueError("Changed pinned input fixture")
        return original(value)

    monkeypatch.setattr(wbc, "verify_inputs", changed)
    with pytest.raises(ValueError, match="Changed pinned"):
        wbc.run(cfg, tmp_path / "out")
    assert len(calls) == 2


def test_subset_admits_only_validated_worker(tmp_path, setup):
    load(setup)
    job = {
        "schema_version": 1,
        "label": "Explicit CPU process fixture",
        "method": "WBC",
        "task": "fixture",
        "embodiment": "r1lite",
        "python": str(Path(sys.executable)),
        "module": "abc_bench.wbc_program",
        "arguments": ["--config", str(setup[1])],
        "max_wall_s": 103,
    }
    path = tmp_path / "job.json"
    path.write_text(json.dumps(job))
    assert load_job(path)["module"] == "abc_bench.wbc_program"
    job["max_wall_s"] = 102
    path.write_text(json.dumps(job))
    with pytest.raises(ValueError, match="plus100s"):
        load_job(path)


def test_export_evaluator_retains_author_sibling_import(setup):
    cfg = setup[0].copy()
    cfg["program"] = "exported-policy-evaluate"
    argv = wbc.command(cfg, Path("/tmp/explicit-fixture"))
    assert "-I" not in argv
    assert argv[3].endswith("scripts/evaluate_exported_policy.py")
    assert "-I" in wbc.command(setup[0], Path("/tmp/explicit-fixture"))


def test_signal_cancellation_reaps_direct_child(tmp_path, setup, monkeypatch):
    cfg = load(setup)
    monkeypatch.setattr(wbc, "verify_inputs", lambda _: {"fixture": True})
    setup[2].write_text(
        "import os,signal,time\nos.kill(os.getppid(),signal.SIGTERM)\ntime.sleep(30)\n"
    )
    with pytest.raises(RuntimeError, match="received signal"):
        wbc.run(cfg, tmp_path / "out")
    receipt = json.loads((tmp_path / "out/receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["direct_child_stopped"]
    assert not Path(f"/proc/{receipt['author_pid']}").exists()


def test_live_disk_refusal_cancels_child(tmp_path, setup, monkeypatch):
    cfg = load(setup)
    monkeypatch.setattr(wbc, "verify_inputs", lambda _: {"fixture": True})
    setup[2].write_text("import time\ntime.sleep(30)\n")
    calls = []

    def check(path):
        calls.append(1)
        if len(calls) > 1:
            raise RuntimeError("Live disk refusal fixture")
        return {"fixture": True}

    monkeypatch.setattr(wbc, "storage_check", check)
    with pytest.raises(RuntimeError, match="Live disk"):
        wbc.run(cfg, tmp_path / "out")
    receipt = json.loads((tmp_path / "out/receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["direct_child_stopped"]
