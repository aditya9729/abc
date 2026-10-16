"""CPU process/admission fixtures only; no learner, model, native task or GPU."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from abc_bench import mulligan_program, runner, subset
from abc_bench import mulligan_r1lite_diagnostic as diagnostic
from abc_bench import mulligan_r1lite_training as training

AUTHOR_PREPARATION = Path(
    "/home/user/aditya/RL/builds/other-methods-plan-20261015/author-r1-preparation-v1"
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def config(tmp_path, monkeypatch):
    """Small explicit synthetic pins; none are real qualification evidence."""
    monkeypatch.setattr(diagnostic, "WORKSPACE", tmp_path)
    source = tmp_path / "source-preparation"
    source.mkdir()
    payload = source / "source/mulligan/training/train.py"
    payload.parent.mkdir(parents=True)
    payload.write_text('"""CPU source identity fixture; never execute."""\n')
    boundary_payload = b'"""CPU boundary identity fixture; never execute."""\n'
    boundary = write(
        tmp_path / "boundary.json",
        {
            "files": [
                {
                    "path": "src/sadhana_mulligan_boundary/__init__.py",
                    "bytes": len(boundary_payload),
                    "sha256": hashlib.sha256(boundary_payload).hexdigest(),
                }
            ]
        },
    )
    monkeypatch.setattr(training, "BOUNDARY_SHA256", training.pin(boundary)["sha256"])
    dataset_file = tmp_path / "data/fixture.txt"
    dataset_file.parent.mkdir()
    dataset_file.write_text("Explicit CPU bytes, not LeRobot rows")
    data = write(
        tmp_path / "data/MANIFEST.json",
        {"files": [{**training.pin(dataset_file), "path": "fixture.txt"}]},
    )
    monkeypatch.setattr(training, "DATA_SHA256", training.pin(data)["sha256"])
    task = write(tmp_path / "task.json", {"fixture": True})
    settings = {
        "task_config": str(task),
        "task_config_sha256": training.pin(task)["sha256"],
        "native_python": "UNRESOLVED",
        "output_dir": "UNRESOLVED",
        "split": "TRAIN",
    }
    for stage in ("IDQL", "DIVL"):
        recipe = json.loads((AUTHOR_PREPARATION / f"{stage}_ARGV.json").read_text())
        write(source / f"{stage}_ARGV.json", recipe)
        write(source / f"{stage}_TRAIN_SETTINGS.json", settings)
    write(
        source / "INPUTS.json", {"inputs": [training.pin(boundary), training.pin(data)]}
    )
    manifest = write(
        source / "SOURCE_MANIFEST.json",
        {"files": [training.pin(p) for p in sorted(source.rglob("*")) if p.is_file()]},
    )
    monkeypatch.setattr(
        training, "AUTHOR_MANIFEST_SHA256", training.pin(manifest)["sha256"]
    )
    monkeypatch.setattr(training, "AUTHOR_MANIFEST_PATH", manifest)
    runtimes = {}
    for name in ("learner", "native"):
        package = tmp_path / name / "sadhana_mulligan_boundary"
        package.mkdir(parents=True)
        (package / "__init__.py").write_bytes(boundary_payload)
        python = package.parent / "python-fixture"
        python.write_bytes(b"Interpreter identity fixture, never execute")
        runtimes[name] = {"python": training.pin(python), "package_dir": str(package)}
    installed = tmp_path / "learner/mulligan/training/train.py"
    installed.parent.mkdir(parents=True)
    installed.write_bytes(payload.read_bytes())
    runtimes["learner"]["author_package_dir"] = str(installed.parent.parent)
    cfg = {
        "schema_version": 1,
        "mode": training.MODE,
        "stage": "idql",
        "python": runtimes["learner"]["python"]["path"],
        "timeout_s": 1000,
        "author_manifest": training.pin(manifest),
        "runtimes": runtimes,
        "admission": {},
        "parent": None,
    }
    attest(cfg, tmp_path)
    return cfg


def attest(config, tmp_path):
    _, _, inputs = training.source_inputs(config)
    evidence = write(
        tmp_path / "cpu-evidence.json", {"fixture": "NOT actual qualification"}
    )
    admission = write(
        tmp_path / "admission.json",
        {
            "schema": "sadhana.mulligan_r1_training_admission/1",
            "status": "accepted_for_r1_original_training",
            "stage": config["stage"],
            "task": training.TASK,
            "embodiment": training.EMBODIMENT,
            "training_steps": 150000,
            "timeout_s": config["timeout_s"],
            "evidence": {key: training.pin(evidence) for key in training.GATES},
            "bindings": {
                "author_manifest": config["author_manifest"],
                "runtimes": config["runtimes"],
                "data": inputs["data"],
                "task": inputs["task"],
                "parent": config["parent"],
            },
        },
    )
    config["admission"] = training.pin(admission)


def with_parent(config, tmp_path):
    parent = tmp_path / "accepted/final_model"
    parent.mkdir(parents=True)
    model = parent / "fixture-model.bin"
    model.write_bytes(b"Explicit synthetic fixture; not a model")
    evidence = write(tmp_path / "parent-producer.json", {"fixture": True})
    acceptance = write(
        tmp_path / "parent.json",
        {
            "status": "accepted_r1_idql_parent",
            "task": training.TASK,
            "embodiment": training.EMBODIMENT,
            "training_steps": 150000,
            "policy_type": "idql",
            "checkpoint_root": str(parent),
            "checkpoint_files": [training.pin(model)],
            "evidence": [training.pin(evidence)],
        },
    )
    config.update(stage="divl", parent=training.pin(acceptance))
    attest(config, tmp_path)
    return parent


def job(config_path):
    return {
        "schema_version": 1,
        "module": "abc_bench.mulligan_program",
        "python": sys.executable,
        "arguments": ["--config", str(config_path)],
        "label": "Explicit CPU training-admission fixture",
        "max_wall_s": 1100,
        "task": training.TASK,
        "embodiment": training.EMBODIMENT,
    }


@pytest.mark.parametrize("stage", ["idql", "divl"])
def test_sealed_recipes_only_and_canonical_dispatch(tmp_path, config, stage):
    parent = with_parent(config, tmp_path) if stage == "divl" else None
    path = write(tmp_path / "config.json", config)
    assert mulligan_program.load_config(path) == config
    assert (
        subset.load_job(write(tmp_path / "job.json", job(path)))["max_wall_s"] == 1100
    )
    _, recipe, _ = training.source_inputs(config)
    command = training.command_for(config, recipe, tmp_path / "training", parent)
    assert command[:6] == [
        config["python"],
        "-I",
        "-u",
        "-B",
        "-m",
        "mulligan.training.train",
    ]
    actual = {x.split("=", 1)[0]: x.split("=", 1)[1] for x in command[6:]}
    expected = {x.split("=", 1)[0]: x.split("=", 1)[1] for x in recipe["argv"]}
    expected["--system.checkpoint_dir"] = str(tmp_path / "training/checkpoints")
    if parent:
        expected["--pretrained_artifact"] = str(parent)
        expected["--wandb.run_name"] = f"r1-divl-{tmp_path.name}"
    assert actual == expected
    assert actual["--training.training_steps"] == "150000"
    assert "abc_bench.mulligan_r1lite_training" not in subset.ALLOWED_MODULES
    assert mulligan_program.SHELL_ENTRYPOINTS == {
        "scripts/sim/eval_cell.sh",
        "scripts/sim/train_cell.sh",
        "scripts/sim/train_divl_heads.sh",
    }


@pytest.mark.parametrize(
    "change",
    [
        {"arguments": ["--training.training_steps=1"]},
        {"entrypoint": "arbitrary.module"},
        {"environment": {"CUDA_VISIBLE_DEVICES": "1"}},
        {"stage": "resfit"},
        {"mode": "gpu"},
        {"timeout_s": True},
        {"timeout_s": 0},
        {"timeout_s": float("inf")},
        {"timeout_s": 86401},
    ],
)
def test_invalid_profile_refused(tmp_path, config, change):
    config.update(change)
    with pytest.raises(ValueError):
        training.load_config(write(tmp_path / "config.json", config))


@pytest.mark.parametrize(
    "target",
    [
        "source",
        "recipe",
        "data",
        "task",
        "installed_author",
        "boundary",
        "python",
        "evidence",
    ],
)
def test_changed_pinned_input_refused(tmp_path, config, target):
    source = Path(config["author_manifest"]["path"]).parent
    paths = {
        "source": source / "source/mulligan/training/train.py",
        "recipe": source / "IDQL_ARGV.json",
        "data": tmp_path / "data/fixture.txt",
        "task": tmp_path / "task.json",
        "installed_author": tmp_path / "learner/mulligan/training/train.py",
        "boundary": tmp_path / "native/sadhana_mulligan_boundary/__init__.py",
        "python": Path(config["python"]),
        "evidence": tmp_path / "cpu-evidence.json",
    }
    paths[target].write_bytes(b"changed fixture")
    with pytest.raises(ValueError):
        training.load_config(write(tmp_path / "config.json", config))


@pytest.mark.parametrize(
    "failure", ["missing_gate", "unaccepted", "wrong_runtime", "wrong_timeout"]
)
def test_root_admission_is_bound_to_all_gates(tmp_path, config, failure):
    path = Path(config["admission"]["path"])
    value = json.loads(path.read_text())
    if failure == "missing_gate":
        del value["evidence"]["native_ten"]
    elif failure == "unaccepted":
        value["status"] = "prepared"
    elif failure == "wrong_runtime":
        value["bindings"]["runtimes"]["native"]["python"]["sha256"] = "0" * 64
    else:
        value["timeout_s"] = 999
    config["admission"] = training.pin(write(path, value))
    with pytest.raises(ValueError):
        training.load_config(write(tmp_path / "config.json", config))


@pytest.mark.parametrize(
    "failure", ["wrong_task", "changed_model", "extra_file", "empty_inventory"]
)
def test_divl_parent_requires_accepted_exact_checkpoint(tmp_path, config, failure):
    parent = with_parent(config, tmp_path)
    path = Path(config["parent"]["path"])
    value = json.loads(path.read_text())
    if failure == "wrong_task":
        value["task"] = "SquareNarrow"
    elif failure == "changed_model":
        (parent / "fixture-model.bin").write_bytes(b"changed")
    elif failure == "extra_file":
        (parent / "unknown.bin").write_bytes(b"extra")
    else:
        value["checkpoint_files"] = []
    config["parent"] = training.pin(write(path, value))
    with pytest.raises(ValueError):
        training.parent_checkpoint(config)


@pytest.mark.parametrize("effective", [1099, 0, float("inf"), float("nan"), True])
def test_short_effective_bound_rejected_before_gpu_or_outputs(
    tmp_path, config, monkeypatch, effective
):
    path = write(tmp_path / "config.json", config)
    job_path = write(tmp_path / "job.json", job(path))
    output = tmp_path / "must-not-exist"
    monkeypatch.setattr(runner, "RESULTS", output)
    monkeypatch.setattr(
        subset, "inspect_reserved_gpu", lambda *_: pytest.fail("Reached GPU admission")
    )
    with pytest.raises(ValueError, match="Effective"):
        subset.run_job(
            SimpleNamespace(training_config=job_path, timeout_seconds=effective)
        )
    assert not output.exists()


@pytest.mark.parametrize("delta", [-1, 0, 500])
def test_cleanup_margin_is_minimum_not_guessed_budget(config, delta):
    value = job(Path("/fixture"))
    value["max_wall_s"] = config["timeout_s"] + 100 + delta
    if delta < 0:
        with pytest.raises(ValueError):
            training.check_parent(value, config)
    else:
        training.check_parent(value, config, value["max_wall_s"])


def test_no_lease_prevents_runtime_probe_or_output(tmp_path, config, monkeypatch):
    def absent():
        raise RuntimeError("explicit absent canonical lease")

    monkeypatch.setattr(subset, "require_subset_lease", absent)
    monkeypatch.setattr(
        training, "verify_runtime", lambda *_: pytest.fail("Reached runtime probe")
    )
    with pytest.raises(RuntimeError, match="absent"):
        training.run(config, tmp_path / "must-not-exist")
    assert not (tmp_path / "must-not-exist").exists()


def test_original_square_commands_still_use_unchanged_path():
    cfg = {
        "python": "/fixture/python",
        "source_root": "/fixture/source",
        "arguments": ["recipe", "--seed", "1"],
        "entrypoint": "scripts/sim/train_cell.sh",
    }
    assert mulligan_program.command_for(cfg, Path("/fixture/out")) == [
        "/bin/bash",
        "/fixture/source/scripts/sim/train_cell.sh",
        "recipe",
        "--seed",
        "1",
    ]


def test_cleanup_retains_already_exited_trainer_until_native_peer_exits(
    tmp_path, monkeypatch
):
    """Real stdlib child and orphan peer, in a fresh fixture-only process group."""
    monkeypatch.setattr(training, "CHILD_CLEANUP_SECONDS", 2)
    marker = tmp_path / "peer-finished"
    peer = f"import time; from pathlib import Path; time.sleep(0.4); Path({str(marker)!r}).write_text('finished')"
    code = f"import subprocess,sys; subprocess.Popen([sys.executable,'-c',{peer!r}],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)"
    process = subprocess.Popen(
        [sys.executable, "-I", "-c", code],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    process.wait(timeout=5)
    try:
        result = training.cleanup(process, process.pid)
        assert result["clean"] and result["known_descendants"]
        assert (
            result["cleanup_intervention"] is None and marker.read_text() == "finished"
        )
    finally:
        try:
            os.killpg(process.pid, 9)
        except ProcessLookupError:
            pass


def test_cleanup_timeout_records_survivors_for_canonical_owner(monkeypatch):
    member = {"boot_id": "fixture", "pid": 1, "start_ticks": 9, "pgid": 7, "ppid": 8}
    monkeypatch.setattr(training, "CHILD_CLEANUP_SECONDS", 0)
    monkeypatch.setattr(
        training,
        "observe_descendants",
        lambda _, known: known.update({(1, 9): member}) or [member],
    )
    result = training.cleanup(None, 7)
    assert not result["clean"] and result["remaining_known_descendants"] == [member]
    assert result["known_descendants"] == [member]


@pytest.mark.parametrize(
    "profile,expected",
    [(None, 5), ("r1lite-native-diagnostic", 90), (training.MODE, 90)],
)
def test_canonical_cleanup_grace_scoped_to_named_modes(
    tmp_path, monkeypatch, profile, expected
):
    waits = []

    class Child:
        pid = 700000000

        def wait(self, timeout=None):
            waits.append(timeout)
            return 0

    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **kw: Child())
    monkeypatch.setattr(runner.os, "killpg", lambda *a: None)
    assert (
        runner.execute_command(
            ["explicit CPU double"], tmp_path / "run.log", 1, cleanup_profile=profile
        )
        == 0
    )
    assert waits == [1, expected, None]


def test_detached_descendant_is_retained_after_reparenting_and_pid_reuse_is_ignored(
    monkeypatch,
):
    owner = os.getpid()
    trainer = {
        "boot_id": "fixture",
        "pid": 11,
        "start_ticks": 22,
        "pgid": owner,
        "ppid": owner,
    }
    detached = {
        "boot_id": "fixture",
        "pid": 33,
        "start_ticks": 44,
        "pgid": 33,
        "ppid": 11,
    }
    known = {}
    monkeypatch.setattr(
        training, "process_snapshot", lambda: {11: trainer, 33: detached}
    )
    assert len(training.observe_descendants(owner, known)) == 2
    monkeypatch.setattr(
        training, "process_snapshot", lambda: {33: {**detached, "ppid": 1}}
    )
    assert training.observe_descendants(owner, known) == [detached]
    monkeypatch.setattr(
        training,
        "process_snapshot",
        lambda: {33: {**detached, "ppid": 1, "start_ticks": 99}},
    )
    assert training.observe_descendants(owner, known) == []


@pytest.mark.parametrize("exit_code", [0, 75])
def test_wrapper_receipt_uses_real_cpu_child_but_never_claims_score(
    tmp_path, config, monkeypatch, exit_code
):
    """Replace only the learner process with a declared stdlib fixture child."""
    seen = []
    monkeypatch.setattr(subset, "require_subset_lease", lambda: seen.append("lease"))
    monkeypatch.setattr(training.os, "getpgrp", lambda: os.getpid())
    monkeypatch.setattr(training.os, "killpg", lambda *args: None)
    monkeypatch.setattr(
        training,
        "verify_runtime",
        lambda *_: {"fixture": "No author/native runtime imported"},
    )
    original_popen = subprocess.Popen
    child_code = (
        "import json,os,sys; from pathlib import Path; "
        "p=Path(os.environ['SADHANA_MULLIGAN_SETTINGS']).parent; "
        "(p/'fixture-environment.json').write_text(json.dumps({"
        "'cuda':os.environ['CUDA_VISIBLE_DEVICES'],'path':os.environ.get('PYTHONPATH'),"
        "'wandb':os.environ['WANDB_MODE']})); " + f"sys.exit({exit_code})"
    )

    def fixture_child(command, **kwargs):
        assert command[4:6] == ["-m", "mulligan.training.train"]
        assert not kwargs.get("start_new_session", False)
        return original_popen([sys.executable, "-I", "-c", child_code], **kwargs)

    monkeypatch.setattr(training.subprocess, "Popen", fixture_child)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-CPU-FIXTURE")
    monkeypatch.setenv("PYTHONPATH", "must-not-leak")
    output = tmp_path / "canonical"
    if exit_code:
        with pytest.raises(ValueError, match="exited 75"):
            training.run(config, output)
    else:
        result = training.run(config, output)
        assert result["status"] == "completed"
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == ("failed" if exit_code else "completed")
    assert receipt["experiment_accepted"] is False and receipt["metrics"] == {}
    assert receipt["cleanup"]["clean"] and seen == ["lease"]
    assert json.loads((output / "training/fixture-environment.json").read_text()) == {
        "cuda": "GPU-CPU-FIXTURE",
        "path": None,
        "wandb": "offline",
    }


def test_failure_signals_only_the_owned_group_then_retains_cleanup(monkeypatch):
    signals = []
    monkeypatch.setattr(
        training.os, "killpg", lambda group, sig: signals.append((group, sig))
    )
    monkeypatch.setattr(training, "observe_descendants", lambda group, known: [])
    result = training.cleanup(None, 123, terminate_group=True)
    assert signals == [(123, training.signal.SIGTERM)]
    assert result["group_sigterm_requested"] and result["clean"]


def test_copied_identical_manifest_cannot_select_unpinned_sibling_recipe(
    tmp_path, config
):
    original = Path(config["author_manifest"]["path"])
    copied = tmp_path / "untrusted-siblings/SOURCE_MANIFEST.json"
    copied.parent.mkdir()
    copied.write_bytes(original.read_bytes())
    write(
        copied.parent / "IDQL_ARGV.json",
        {"module": "arbitrary.module", "argv": ["--changed"]},
    )
    config["author_manifest"] = training.pin(copied)
    assert config["author_manifest"]["sha256"] == training.pin(original)["sha256"]
    with pytest.raises(ValueError, match="exact frozen author manifest path"):
        training.load_config(write(tmp_path / "copied-config.json", config))
