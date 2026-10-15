"""CPU admission/process fixtures; original config parser only, no model/native work."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from abc_bench import mulligan_program, runner, subset
from abc_bench import mulligan_r1lite_diagnostic as diagnostic
from abc_bench import mulligan_r1lite_qualification as qualification
from abc_bench import mulligan_r1lite_training as training

M = Path("/home/user/aditya/RL/builds/other-methods-plan-20261015")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    """Explicit CPU-only evidence doubles; real inventory checks remain separately covered."""
    monkeypatch.setattr(diagnostic, "WORKSPACE", tmp_path)
    marker = training.pin(write(tmp_path / "evidence.json", {"fixture": True}))
    manifest = write(tmp_path / "operator/SOURCE_MANIFEST.json", {"files": []})
    monkeypatch.setattr(qualification, "OPERATOR_MANIFEST_PATH", manifest)
    monkeypatch.setattr(
        qualification, "OPERATOR_MANIFEST_SHA256", training.pin(manifest)["sha256"]
    )
    config = {
        "schema_version": 1,
        "mode": qualification.MODE,
        "stage": "idql",
        "phase": "update_restore",
        "python": sys.executable,
        "timeout_s": 1800,
        "author_manifest": marker,
        "runtimes": {"learner": {}, "native": {"python": {"path": sys.executable}}},
        "parent": None,
        "admission": {},
    }
    inputs = {
        "data": marker,
        "task": marker,
        "boundary": {},
        "settings": {
            "split": "TRAIN",
            "seeds": list(range(2000000, 2000020)),
            "record_native_video": False,
        },
    }
    (tmp_path / "source").mkdir()
    monkeypatch.setattr(
        training,
        "source_inputs",
        lambda cfg: (
            tmp_path,
            json.loads(
                (
                    M / f"author-r1-preparation-v1/{cfg['stage'].upper()}_ARGV.json"
                ).read_text()
            ),
            inputs,
        ),
    )
    monkeypatch.setattr(training, "verify_installation_files", lambda *_: None)
    attest(config, inputs, tmp_path)
    return config, inputs


def attest(config, inputs, tmp_path):
    gates = qualification.GATES | (
        {"update_restore"} if config["phase"] == "timing_monitor" else set()
    )
    receipt = {
        "schema": "sadhana.mulligan_r1_qualification_admission/1",
        "status": "accepted_for_short_original_qualification",
        "stage": config["stage"],
        "phase": config["phase"],
        "task": qualification.TASK,
        "embodiment": qualification.EMBODIMENT,
        "timeout_s": 1800,
        "evidence": {key: inputs["data"] for key in gates},
        "bindings": {
            "author_manifest": config["author_manifest"],
            "runtimes": config["runtimes"],
            "data": inputs["data"],
            "task": inputs["task"],
            "parent": config["parent"],
            "operator_manifest": training.pin(qualification.OPERATOR_MANIFEST_PATH),
        },
    }
    config["admission"] = training.pin(write(tmp_path / "admission.json", receipt))


def parent(config, inputs, tmp_path):
    root = tmp_path / "parent/final_model"
    payload = write(root / "model.json", {"explicit": "CPU fixture, never load"})
    receipt = {
        "status": "accepted_r1_idql_diagnostic_parent",
        "policy_type": "idql",
        "training_steps": 2,
        "task": qualification.TASK,
        "embodiment": qualification.EMBODIMENT,
        "bindings": {
            "author_manifest": config["author_manifest"],
            "runtimes": config["runtimes"],
            "data": inputs["data"],
            "task": inputs["task"],
        },
        "evidence": {
            key: inputs["data"] for key in ("model_inspection", "canonical_cleanup")
        },
        "checkpoint_root": str(root),
        "checkpoint_files": [training.pin(payload)],
    }
    config.update(
        stage="divl", parent=training.pin(write(tmp_path / "parent.json", receipt))
    )
    attest(config, inputs, tmp_path)
    return root


def job(path):
    return {
        "schema_version": 1,
        "module": "abc_bench.mulligan_program",
        "python": sys.executable,
        "arguments": ["--config", str(path)],
        "label": "CPU diagnostic admission fixture",
        "max_wall_s": 1900,
        "task": qualification.TASK,
        "embodiment": qualification.EMBODIMENT,
    }


@pytest.mark.parametrize("stage", ["idql", "divl"])
@pytest.mark.parametrize(
    "phase,steps,frequency",
    [("update_restore", 2, "null"), ("timing_monitor", 2000, "2000")],
)
def test_fixed_original_recipe_and_dispatch(
    tmp_path, prepared, stage, phase, steps, frequency
):
    config, inputs = prepared
    checkpoint = parent(config, inputs, tmp_path) if stage == "divl" else None
    config["phase"] = phase
    attest(config, inputs, tmp_path)
    path = write(tmp_path / "config.json", config)
    assert mulligan_program.load_config(path) == config
    assert (
        subset.load_job(write(tmp_path / "job.json", job(path)))["max_wall_s"] == 1900
    )
    _, recipe, _ = training.source_inputs(config)
    output = tmp_path / "qualification"
    original = training.command_for(config, recipe, output, checkpoint)
    command = qualification.command_for(config, recipe, output, checkpoint)
    assert command[:6] == original[:6]
    actual = dict(arg.split("=", 1) for arg in command[6:])
    previous = dict(arg.split("=", 1) for arg in original[6:])
    changed = {key for key in actual if actual[key] != previous.get(key)}
    assert changed <= {
        "--training.training_steps",
        "--training.periodic_checkpoint_freq",
        "--training.resume_checkpoint_freq",
        "--training.save_final_model",
        "--eval.freq",
        "--wandb.run_name",
    }
    assert actual["--training.training_steps"] == str(steps)
    assert actual["--eval.freq"] == frequency
    assert actual["--training.batch_size"] == "1024"
    assert actual["--training.compile_actor"] == (
        "True" if stage == "idql" else "False"
    )
    assert actual["--eval.episodes"] == "20" and actual["--eval.num_envs"] == "10"
    assert "abc_bench.mulligan_r1lite_qualification" not in subset.ALLOWED_MODULES


@pytest.mark.parametrize(
    "change",
    [
        {"arguments": []},
        {"phase": "full"},
        {"stage": "other"},
        {"timeout_s": True},
        {"timeout_s": 1799},
        {"timeout_s": 1801},
    ],
)
def test_invalid_profile_rejected(prepared, tmp_path, change):
    config, _ = prepared
    config.update(change)
    with pytest.raises(ValueError):
        qualification.load_config(write(tmp_path / "config.json", config))


@pytest.mark.parametrize("effective", [1899, float("inf"), float("nan"), True])
def test_effective_bound_before_gpu_and_output(
    prepared, tmp_path, monkeypatch, effective
):
    config, _ = prepared
    path = write(tmp_path / "config.json", config)
    job_path = write(tmp_path / "job.json", job(path))
    monkeypatch.setattr(runner, "RESULTS", tmp_path / "never-created")
    monkeypatch.setattr(
        subset, "inspect_reserved_gpu", lambda *_: pytest.fail("GPU admission reached")
    )
    with pytest.raises(ValueError, match="Effective"):
        subset.run_job(
            SimpleNamespace(training_config=job_path, timeout_seconds=effective)
        )
    assert not (tmp_path / "never-created").exists()


@pytest.mark.parametrize(
    "damage", ["parser_gate", "phase", "runtime", "parent_step", "extra_parent_file"]
)
def test_evidence_and_parent_refusal(prepared, tmp_path, damage):
    config, inputs = prepared
    root = parent(config, inputs, tmp_path)
    if damage.startswith("parent") or damage == "extra_parent_file":
        path = Path(config["parent"]["path"])
        receipt = json.loads(path.read_text())
        if damage == "parent_step":
            receipt["training_steps"] = 150000
            config["parent"] = training.pin(write(path, receipt))
        else:
            (root / "extra").write_text("unlisted")
        with pytest.raises(ValueError):
            qualification.parent_checkpoint(config, inputs)
    else:
        path = Path(config["admission"]["path"])
        receipt = json.loads(path.read_text())
        if damage == "parser_gate":
            del receipt["evidence"]["original_cli_parser"]
        elif damage == "phase":
            receipt["phase"] = "timing_monitor"
        else:
            receipt["bindings"]["runtimes"] = {}
        config["admission"] = training.pin(write(path, receipt))
        with pytest.raises(ValueError):
            qualification.verify_admission(config, inputs)


@pytest.mark.parametrize("stage", ["idql", "divl"])
@pytest.mark.parametrize("phase", ["update_restore", "timing_monitor"])
def test_actual_original_draccus_parser_only(stage, phase, tmp_path):
    """Actual parser, CUDA hidden; config imports Torch precision constants but creates no model."""
    release_path = M / "author-r1-release-02/RELEASE.json"
    assert (
        training.pin(release_path)["sha256"]
        == "7b33d066480bdc0858e0969d3dc583a3e408900d3bbebf92f6f5e1244451b58f"
    )
    release = json.loads(release_path.read_text())
    python = training.verify(release["python"])
    assert python.resolve().is_relative_to(Path("/home/user/aditya"))
    config = {"stage": stage, "phase": phase, "python": str(python)}
    recipe = json.loads(
        (M / f"author-r1-preparation-v1/{stage.upper()}_ARGV.json").read_text()
    )
    command = qualification.command_for(
        config, recipe, tmp_path, tmp_path / "parent" if stage == "divl" else None
    )
    code = (
        "import json,sys; import draccus; from mulligan.configs import TrainConfig; "
        "c=draccus.parse(TrainConfig,args=json.loads(sys.argv[1])); "
        "assert 'mulligan.training.train' not in sys.modules; "
        "assert 'mujoco' not in sys.modules; assert not sys.modules['torch'].cuda.is_initialized(); "
        "print(json.dumps(c.to_dict()))"
    )
    result = subprocess.run(
        [str(python), "-I", "-B", "-c", code, json.dumps(command[6:])],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"},
    )
    parsed = json.loads(result.stdout)
    assert parsed["eval"]["freq"] == (None if phase == "update_restore" else 2000)
    assert parsed["training"]["batch_size"] == 1024
    assert parsed["training"]["training_steps"] == (
        2 if phase == "update_restore" else 2000
    )
    assert parsed["training"]["periodic_checkpoint_freq"] == (
        1 if phase == "update_restore" else 1000
    )
    assert parsed["training"]["resume_checkpoint_freq"] == (
        1 if phase == "update_restore" else 1000
    )
    assert parsed["training"]["compile_actor"] is (stage == "idql")
    assert parsed["training"]["amp_dtype"] == "bfloat16"
    assert parsed["training"]["enable_tf32"] is True


@pytest.mark.parametrize("exit_code", [0, 75])
def test_actual_canonical_receipt_publication(
    tmp_path, prepared, monkeypatch, exit_code
):
    config, _ = prepared
    config_path = write(tmp_path / "config.json", config)
    job_path = write(tmp_path / "job.json", job(config_path))
    results = tmp_path / "results"
    write(
        results / "gpu_budget.json",
        {"gpu": 0, "charged_seconds": 17, "limit_seconds": 5000},
    )
    monkeypatch.setattr(runner, "RESULTS", results)
    monkeypatch.setattr(
        subset, "inspect_reserved_gpu", lambda gpu: {"uuid": "GPU-CPU-FIXTURE"}
    )

    def worker(command, log_path, timeout, **options):
        assert timeout == 1900 and options["cleanup_profile"] == qualification.MODE
        output = Path(command[-1])
        receipt = json.loads((results / "runs" / (output.name + ".json")).read_text())
        assert (
            receipt["task"] == qualification.TASK
            and receipt["embodiment"] == qualification.EMBODIMENT
        )
        assert (
            receipt["phase"] == "original_r1_qualification"
            and receipt["experiment_accepted"] is False
        )
        assert (
            receipt["qualification_stage"] == "idql"
            and receipt["qualification_phase"] == "update_restore"
        )
        assert receipt["metrics"] == {}
        log_path.write_text("Explicit CPU worker double")
        if exit_code == 0:
            write(
                output / "receipt.json",
                {"status": "completed", "metrics": {}, "experiment_accepted": False},
            )
        return exit_code

    monkeypatch.setattr(runner, "execute_command", worker)
    result = subset.run_job(
        SimpleNamespace(training_config=job_path, timeout_seconds=1900)
    )
    assert result["status"] == ("completed" if exit_code == 0 else "failed")
    assert result["experiment_accepted"] is False
    ledger = json.loads((results / "gpu_budget.json").read_text())
    assert "active_parent_pid" not in ledger and ledger["charged_seconds"] >= 17


@pytest.mark.parametrize(
    "phase,error", [("author", True), ("inspection", True), ("none", False)]
)
def test_two_children_cleanup_and_failure_receipt(
    tmp_path, prepared, monkeypatch, phase, error
):
    config, _ = prepared
    monkeypatch.setattr(subset, "require_subset_lease", lambda: None)
    monkeypatch.setattr(qualification.os, "getpgrp", lambda: os.getpid())
    monkeypatch.setattr(
        training, "verify_runtime", lambda *_: {"explicit_cpu_double": True}
    )
    monkeypatch.setattr(training, "observe_descendants", lambda *_: [])
    cleaned = []
    monkeypatch.setattr(
        training,
        "cleanup",
        lambda proc, group, known, **kw: (
            cleaned.append(kw["terminate_group"])
            or {"clean": True, "cleanup_intervention": None}
        ),
    )
    calls = []

    def child(command, **kwargs):
        name = "author" if "mulligan.training.train" in command else "inspection"
        calls.append(name)
        assert not kwargs.get("start_new_session")
        if name == "inspection":
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
            write(
                Path(command[command.index("--output") + 1]),
                {
                    "status": "passed_short_model_inspection_pending_root_acceptance",
                    "stage": "idql",
                    "phase": "update_restore",
                    "steps": 2,
                    "artifact_accepted": False,
                },
            )
        return SimpleNamespace(
            pid=100 + len(calls), poll=lambda: 75 if name == phase else 0
        )

    monkeypatch.setattr(qualification.subprocess, "Popen", child)
    output = tmp_path / "canonical"
    if error:
        with pytest.raises(ValueError, match="exited 75"):
            qualification.run(config, output)
    else:
        assert qualification.run(config, output)["status"] == "completed"
    assert calls == (["author"] if phase == "author" else ["author", "inspection"])
    receipt = json.loads((output / "receipt.json").read_text())
    assert (
        receipt["experiment_accepted"] is False
        and receipt["exact_resume_executed"] is False
    )
    assert cleaned == [error]


def test_ten_native_profile_binds_only_new_manifest(tmp_path):
    config = json.loads(
        (M / "next-native-handoff-20261015/TWO_SLOTS_CONFIG.json").read_text()
    )
    config["case"] = "ten_slots"
    with pytest.raises(ValueError, match="independently reviewed"):
        diagnostic.load_config(write(tmp_path / "old-manifest.json", config))
    config["qualification_manifest"] = diagnostic.pin(
        M / "native-qualification-preparation-v2/SOURCE_MANIFEST.json"
    )
    assert diagnostic.load_config(write(tmp_path / "ten.json", config)) == config
    source, plan, _ = diagnostic.qualification(config)
    assert plan["cases"]["ten_slots"]["slot_seeds"] == [20044, 20045] * 5
    command = diagnostic.command_for(
        config, source, tmp_path / "settings.json", tmp_path / "diagnostic"
    )
    assert command[command.index("--case") + 1] == "ten_slots"


@pytest.mark.parametrize("profile,grace", [(None, 5), (qualification.MODE, 90)])
def test_named_cleanup_grace_preserves_square_default(
    tmp_path, monkeypatch, profile, grace
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
            ["CPU double"], tmp_path / "run.log", 1, cleanup_profile=profile
        )
        == 0
    )
    assert waits == [1, grace, None]


def test_no_lease_prevents_source_probe_and_output(tmp_path, monkeypatch):
    def absent():
        raise RuntimeError("Absent explicit canonical lease")

    monkeypatch.setattr(subset, "require_subset_lease", absent)
    monkeypatch.setattr(
        training, "source_inputs", lambda *_: pytest.fail("Reached source probe")
    )
    with pytest.raises(RuntimeError, match="Absent"):
        qualification.run({}, tmp_path / "never-created")
    assert not (tmp_path / "never-created").exists()
