"""Explicit CPU fixtures. They do not import a simulator or execute a learner."""

import hashlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from abc_bench import mulligan_program, runner, subset
from abc_bench import mulligan_r1lite_diagnostic as diagnostic

FIXTURE_OPERATOR = """
import argparse, hashlib, json, os
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--plan', type=Path)
p.add_argument('--case')
p.add_argument('--settings', type=Path)
p.add_argument('--output', type=Path)
a = p.parse_args()
a.output.mkdir()
def pin(path):
    raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
report = {'status': 'passed_adapter_diagnostic_pending_canonical_acceptance',
          'case': a.case, 'error': None, 'cleanup_error': None,
          'settings': pin(a.settings), 'operator': pin(Path(__file__).resolve()),
          'plan': pin(a.plan), 'fixture': 'CPU process only; no native data'}
(a.output / 'RESULT.json').write_text(json.dumps(report))
(a.output / 'environment.json').write_text(json.dumps({
    'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES'],
    'OMP_NUM_THREADS': os.environ['OMP_NUM_THREADS'],
    'PYTHONPATH': os.environ.get('PYTHONPATH')}))
"""


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def config(tmp_path, monkeypatch):
    """Synthetic manifests and payloads; never use these as native evidence."""
    monkeypatch.setattr(diagnostic, "WORKSPACE", tmp_path)
    source = tmp_path / "fixture-source"
    source.mkdir()
    (source / "qualify_native.py").write_text(FIXTURE_OPERATOR)
    boundary_bytes = b'"""Synthetic installed package payload."""\n'
    boundary_manifest = write(
        tmp_path / "boundary.json",
        {
            "files": [
                {
                    "path": "src/sadhana_mulligan_boundary/__init__.py",
                    "bytes": len(boundary_bytes),
                    "sha256": hashlib.sha256(boundary_bytes).hexdigest(),
                }
            ]
        },
    )
    monkeypatch.setattr(
        diagnostic, "BOUNDARY_SHA256", diagnostic.pin(boundary_manifest)["sha256"]
    )
    empty = tmp_path / "fixture-input"
    empty.write_bytes(b"Explicit CPU fixture; no task data")
    plan = {
        "boundary_source_manifest": diagnostic.pin(boundary_manifest),
        "boundary_wheel": diagnostic.pin(empty),
        "cpu_qualification": diagnostic.pin(empty),
        "input_pins": [diagnostic.pin(empty)],
        "cases": {"one_slot": {"seeds": [1]}, "two_slots": {"seeds": [1, 2]}},
        "oracles": {str(seed): diagnostic.pin(empty) for seed in (1, 2)},
    }
    write(source / "QUALIFICATION_PLAN.json", plan)
    for name in ("ONE_SLOT_SETTINGS.json", "TWO_SLOTS_SETTINGS.json"):
        write(
            source / name,
            {
                "fixture": True,
                "native_python": "UNRESOLVED",
                "output_dir": "UNRESOLVED",
            },
        )
    runtimes = {}
    for name in ("learner", "native"):
        package = tmp_path / name / "sadhana_mulligan_boundary"
        package.mkdir(parents=True)
        (package / "__init__.py").write_bytes(boundary_bytes)
        python = package.parent / "python-fixture-not-executable"
        python.write_bytes(b"Interpreter identity fixture, never executed")
        runtimes[name] = {"python": diagnostic.pin(python), "package_dir": str(package)}
    result = {
        "schema_version": 1,
        "mode": diagnostic.MODE,
        "python": runtimes["learner"]["python"]["path"],
        "case": "one_slot",
        "timeout_s": 1800,
        "qualification_manifest": {},
        "runtimes": runtimes,
    }
    reseal(result, source, monkeypatch)
    return result


def reseal(config, source, monkeypatch):
    files = []
    for path in sorted(source.iterdir()):
        if path.name != "SOURCE_MANIFEST.json":
            files.append({**diagnostic.pin(path), "path": path.name})
    manifest = write(source / "SOURCE_MANIFEST.json", {"files": files})
    config["qualification_manifest"] = diagnostic.pin(manifest)
    monkeypatch.setattr(
        diagnostic, "QUALIFICATION_SHA256", diagnostic.pin(manifest)["sha256"]
    )


def job(config_path):
    return {
        "schema_version": 1,
        "module": "abc_bench.mulligan_program",
        "python": sys.executable,
        "arguments": ["--config", str(config_path)],
        "label": "Explicit CPU admission fixture",
        "max_wall_s": 1900,
        "task": diagnostic.TASK,
        "embodiment": diagnostic.EMBODIMENT,
    }


@pytest.mark.parametrize("case", ["one_slot", "two_slots"])
def test_exact_mode_uses_existing_entrypoint_and_keeps_allowlist(
    tmp_path, config, case
):
    config["case"] = case
    path = write(tmp_path / "config.json", config)
    assert mulligan_program.load_config(path) == config
    assert (
        subset.load_job(write(tmp_path / "job.json", job(path)))["max_wall_s"] == 1900
    )
    assert "abc_bench.mulligan_r1lite_diagnostic" not in subset.ALLOWED_MODULES
    assert mulligan_program.SHELL_ENTRYPOINTS == {
        "scripts/sim/eval_cell.sh",
        "scripts/sim/train_cell.sh",
        "scripts/sim/train_divl_heads.sh",
    }


@pytest.mark.parametrize(
    "change",
    [
        {"mode": "r1lite-native-training"},
        {"timeout_s": 1799},
        {"timeout_s": float("inf")},
        {"timeout_s": True},
        {"case": "ten_slots"},
        {"environment": {"CUDA_VISIBLE_DEVICES": "1"}},
        {"entrypoint": "unreviewed.py"},
    ],
)
def test_invalid_profile_refused(tmp_path, config, change):
    config.update(change)
    with pytest.raises(ValueError):
        mulligan_program.load_config(write(tmp_path / "config.json", config))


@pytest.mark.parametrize("target", ["script", "settings", "package", "interpreter"])
def test_changed_pinned_payload_refused(tmp_path, config, target):
    source = Path(config["qualification_manifest"]["path"]).parent
    paths = {
        "script": source / "qualify_native.py",
        "settings": source / "ONE_SLOT_SETTINGS.json",
        "package": Path(config["runtimes"]["native"]["package_dir"]) / "__init__.py",
        "interpreter": Path(config["runtimes"]["learner"]["python"]["path"]),
    }
    paths[target].write_bytes(b"Changed explicit CPU fixture")
    with pytest.raises(ValueError, match="changed"):
        diagnostic.load_config(write(tmp_path / "config.json", config))


@pytest.mark.parametrize("timeout", [1899, 0, float("inf"), float("nan"), True])
def test_effective_cli_bound_refused_before_outputs_or_lease(
    tmp_path, monkeypatch, config, timeout
):
    path = write(tmp_path / "config.json", config)
    job_path = write(tmp_path / "job.json", job(path))
    output = tmp_path / "must-not-be-created"
    monkeypatch.setattr(runner, "RESULTS", output)

    def no_gpu(*args):
        pytest.fail("Invalid CLI bound reached GPU inspection")

    monkeypatch.setattr(subset, "inspect_reserved_gpu", no_gpu)
    with pytest.raises(ValueError, match="Effective"):
        subset.run_job(
            SimpleNamespace(training_config=job_path, timeout_seconds=timeout)
        )
    assert not output.exists()


@pytest.mark.parametrize(
    "change",
    [
        {"task": "square_narrow"},
        {"embodiment": "franka_panda"},
        {"max_wall_s": 1899},
        {"max_wall_s": 1901},
        {"max_wall_s": 1900.0},
    ],
)
def test_parent_identity_and_fixed_bound_required(tmp_path, config, change):
    value = job(write(tmp_path / "config.json", config))
    value.update(change)
    with pytest.raises(ValueError):
        subset.load_job(write(tmp_path / "job.json", value))


@pytest.mark.parametrize("corruption", [None, "version", "location", "native_task"])
def test_bounded_stdlib_metadata_probe_requires_pinned_installation(
    monkeypatch, config, corruption
):
    native_versions = {
        "numpy": "1.26.4",
        "mujoco": "3.13.0",
        "leela-sim": "0.7.0",
        "sadhana-r1lite": "0.2.1",
    }
    task_hash = "fixture-native-env-source-sha"
    plan = {
        "fixed_settings": {
            "native_versions": native_versions,
            "native_env_sha256": task_hash,
        }
    }
    calls = []

    def probe(command, **kwargs):
        calls.append(command)
        assert command[1:4] == ["-I", "-B", "-c"]
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert kwargs["timeout"] == 15 and kwargs["check"] is True
        name = (
            "native"
            if command[0] == config["runtimes"]["native"]["python"]["path"]
            else "learner"
        )
        package = Path(config["runtimes"][name]["package_dir"])
        versions = native_versions if name == "native" else {"numpy": "2.2.6"}
        observed = {
            "python": [3, 11] if name == "native" else [3, 12],
            "prefix": str(package.parent),
            "package_dir": str(package),
            "versions": {**versions, "sadhana-mulligan-boundary": "0.1.2"},
        }
        if name == "native":
            observed["native_env_sha256"] = task_hash
            if corruption == "version":
                observed["versions"]["numpy"] = "2.2.6"
            if corruption == "location":
                observed["package_dir"] = str(package.parent / "shadow")
            if corruption == "native_task":
                observed["native_env_sha256"] = "changed"
        return SimpleNamespace(stdout=json.dumps(observed))

    monkeypatch.setattr(diagnostic.subprocess, "run", probe)
    if corruption:
        with pytest.raises(ValueError):
            diagnostic.verify_runtime(config, plan)
    else:
        assert set(diagnostic.verify_runtime(config, plan)) == {"learner", "native"}
    assert len(calls) == 2


def configure_process_fixture(config, monkeypatch):
    config["python"] = sys.executable
    monkeypatch.setattr(
        diagnostic,
        "verify_runtime",
        lambda *args: {"fixture": "No runtime qualification"},
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-CPU-FIXTURE")
    monkeypatch.setenv("PYTHONPATH", "must-not-enter-diagnostic-child")
    seen = []

    def lease():
        seen.append(os.environ["CUDA_VISIBLE_DEVICES"])

    monkeypatch.setattr(subset, "require_subset_lease", lease)
    return seen


def test_fresh_child_literal_command_and_no_learner_score(
    tmp_path, monkeypatch, config
):
    seen = configure_process_fixture(config, monkeypatch)
    output = tmp_path / "canonical"
    output.mkdir()
    write(output / "canonical_receipt.json", {"fixture": "Existing parent bookkeeping"})
    result = diagnostic.run(config, output)
    assert seen == ["GPU-CPU-FIXTURE"]
    assert result["status"] == "completed" and result["metrics"] == {}
    assert result["experiment_accepted"] is False and result["direct_child_stopped"]
    environment = json.loads((output / "diagnostic/environment.json").read_text())
    assert environment == {
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "PYTHONPATH": None,
    }
    settings = json.loads((output / "diagnostic-settings.json").read_text())
    assert settings["output_dir"] == str(output / "diagnostic/native")
    assert settings["native_python"] == config["runtimes"]["native"]["python"]["path"]
    assert result["command"][:4] == [sys.executable, "-I", "-u", "-B"]
    with pytest.raises(ValueError, match="fresh"):
        diagnostic.run(config, output)


def test_native_diagnostic_still_requires_canonical_gpu_lease(
    tmp_path, monkeypatch, config
):
    def missing():
        raise RuntimeError("Explicit absent canonical lease")

    monkeypatch.setattr(subset, "require_subset_lease", missing)
    with pytest.raises(RuntimeError, match="absent canonical lease"):
        diagnostic.run(config, tmp_path / "must-not-exist")
    assert not (tmp_path / "must-not-exist").exists()


def test_exit75_is_failure_not_diagnostic_success(tmp_path, monkeypatch, config):
    configure_process_fixture(config, monkeypatch)
    source = Path(config["qualification_manifest"]["path"]).parent
    (source / "qualify_native.py").write_text("raise SystemExit(75)\n")
    reseal(config, source, monkeypatch)
    with pytest.raises(ValueError, match="exited 75"):
        diagnostic.run(config, tmp_path / "output")
    receipt = json.loads((tmp_path / "output/receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["diagnostic_exit_code"] == 75
    assert receipt["direct_child_stopped"] and not receipt["experiment_accepted"]


def test_real_owned_child_retained_through_timeout_cleanup(
    tmp_path, monkeypatch, config
):
    configure_process_fixture(config, monkeypatch)
    source = Path(config["qualification_manifest"]["path"]).parent
    marker = source / "child-cleaned"
    (source / "qualify_native.py").write_text(
        "import signal, time\nfrom pathlib import Path\n"
        "def stop(signum, frame):\n"
        "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "    time.sleep(0.2)\n"
        f"    Path({str(marker)!r}).write_text('Child completed owned cleanup')\n"
        "    raise SystemExit(75)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        "while True: time.sleep(0.05)\n"
    )
    reseal(config, source, monkeypatch)
    monkeypatch.setattr(diagnostic, "WORKER_SECONDS", 2)
    monkeypatch.setattr(diagnostic, "CHILD_CLEANUP_SECONDS", 5)
    with pytest.raises(subprocess.TimeoutExpired):
        diagnostic.run(config, tmp_path / "output")
    result = json.loads((tmp_path / "output/receipt.json").read_text())
    assert marker.read_text() == "Child completed owned cleanup"
    assert result["cleanup_intervention"] == "diagnostic_SIGTERM"
    assert result["status"] == "failed" and result["direct_child_stopped"]
    assert not Path(f"/proc/{result['diagnostic_pid']}").exists()


@pytest.mark.parametrize("must_kill", [False, True])
def test_wrapper_cancellation_retains_child_until_bounded_reap(
    tmp_path, monkeypatch, config, must_kill
):
    configure_process_fixture(config, monkeypatch)
    waits, actions = [], []

    class Process:
        pid = 123456789
        stopped = False

        def wait(self, timeout=None):
            waits.append(timeout)
            if len(waits) == 1:
                raise InterruptedError("Explicit CPU SIGTERM delivery fixture")
            if len(waits) == 2 and must_kill:
                raise subprocess.TimeoutExpired("cpu-fixture", timeout)
            self.stopped = True
            return -15

        def poll(self):
            return -15 if self.stopped else None

        def terminate(self):
            actions.append("TERM")

        def kill(self):
            actions.append("KILL")

    monkeypatch.setattr(
        diagnostic.subprocess, "Popen", lambda *args, **kwargs: Process()
    )
    with pytest.raises(InterruptedError, match="SIGTERM"):
        diagnostic.run(config, tmp_path / "output")
    assert waits[1:] == ([80, 2] if must_kill else [80])
    assert actions == (["TERM", "KILL"] if must_kill else ["TERM"])
    result = json.loads((tmp_path / "output/receipt.json").read_text())
    assert result["status"] == "failed" and result["direct_child_stopped"]
    assert result["experiment_accepted"] is False


@pytest.mark.parametrize("profile,cleanup", [(None, 5), (diagnostic.MODE, 90)])
def test_cancelled_canonical_group_keeps_square_default_and_r1_grace(
    tmp_path, monkeypatch, profile, cleanup
):
    """Explicit process double: exercise actual execute_command cancellation branch."""
    waits, signals = [], []

    class Process:
        pid = 123456789

        def wait(self, timeout=None):
            waits.append(timeout)
            if len(waits) == 1:
                raise runner.RunCancelled("Explicit CPU cancellation fixture")
            return 0

    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(
        runner.os, "killpg", lambda pid, signum: signals.append((pid, signum))
    )
    with pytest.raises(runner.RunCancelled):
        runner.execute_command(
            ["cpu-fixture"], tmp_path / "log", 1900, cleanup_profile=profile
        )
    assert waits == [1900, cleanup, None]
    assert signals == [(123456789, signal.SIGTERM), (123456789, signal.SIGKILL)]


def test_unknown_cleanup_profile_refused_before_spawn(tmp_path, monkeypatch):
    def no_spawn(*args, **kwargs):
        pytest.fail("Unknown profile reached process creation")

    monkeypatch.setattr(runner.subprocess, "Popen", no_spawn)
    with pytest.raises(ValueError, match="Unknown"):
        runner.execute_command(
            ["cpu-fixture"], tmp_path / "log", 1, cleanup_profile="r1-anything"
        )
    assert not (tmp_path / "log").exists()
