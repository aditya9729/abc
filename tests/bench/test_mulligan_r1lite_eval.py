"""Explicit CPU admission/process fixtures; no original evaluator, native task or GPU."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from abc_bench import mulligan_program, runner, subset
from abc_bench import mulligan_r1lite_diagnostic as diagnostic
from abc_bench import mulligan_r1lite_eval as evaluation
from abc_bench import mulligan_r1lite_training as training


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, "WORKSPACE", tmp_path)
    marker = write(
        tmp_path / "evidence.json", {"fixture": "CPU only, not actual acceptance"}
    )
    marker_pin = evaluation.pin(marker)
    inputs = {key: marker_pin for key in ("author_manifest", "data", "task")}
    config = {
        "schema_version": 1,
        "mode": evaluation.MODE,
        "method": "idql",
        "num_action_samples": 1,
        "python": sys.executable,
        "timeout_s": 1000,
        "caller_release": marker_pin,
        "runtimes": {"learner": {}, "native": {}},
        "checkpoint_acceptance": {},
        "parent_acceptance": None,
        "admission": {},
    }
    root = tmp_path / "idql/final_model"
    root.mkdir(parents=True)
    for name in ("metadata.json", "stats.json", "policy.pt"):
        (root / name).write_text("explicit synthetic bytes, never loaded")
    checkpoint = {
        "status": "accepted_r1_idql_parent",
        "task": evaluation.TASK,
        "embodiment": evaluation.EMBODIMENT,
        "training_steps": 150000,
        "policy_type": "idql",
        "checkpoint_root": str(root),
        "checkpoint_files": [evaluation.pin(p) for p in sorted(root.iterdir())],
        "bindings": dict(inputs),
        "evidence": [marker_pin],
    }
    config["checkpoint_acceptance"] = evaluation.pin(
        write(tmp_path / "checkpoint.json", checkpoint)
    )
    monkeypatch.setattr(evaluation, "source_inputs", lambda _: inputs)
    attest(config, inputs, tmp_path)
    return config, inputs


def attest(config, inputs, tmp_path):
    bindings = {
        key: config[key]
        for key in (
            "caller_release",
            "runtimes",
            "checkpoint_acceptance",
            "parent_acceptance",
        )
    }
    bindings.update({key: inputs[key] for key in ("author_manifest", "data", "task")})
    admission = {
        "schema": "sadhana.mulligan_r1_frozen_eval_admission/1",
        "status": "accepted_for_r1_frozen_eval",
        "task": evaluation.TASK,
        "embodiment": evaluation.EMBODIMENT,
        "method": config["method"],
        "num_action_samples": config["num_action_samples"],
        "timeout_s": config["timeout_s"],
        "bindings": bindings,
        "evidence": {key: inputs["task"] for key in evaluation.GATES},
    }
    config["admission"] = evaluation.pin(write(tmp_path / "admission.json", admission))


def job(path):
    return {
        "schema_version": 1,
        "module": "abc_bench.mulligan_program",
        "python": sys.executable,
        "arguments": ["--config", str(path)],
        "label": "CPU frozen evaluation fixture",
        "max_wall_s": 1100,
        "task": evaluation.TASK,
        "embodiment": evaluation.EMBODIMENT,
    }


@pytest.mark.parametrize(
    "method,n", [("idql", 1), ("idql", 32), ("divl", 1), ("divl", 32)]
)
def test_exact_profile_dispatch(tmp_path, prepared, method, n):
    config, inputs = prepared
    config.update(method=method, num_action_samples=n)
    if method == "divl":
        parent_pin = config["checkpoint_acceptance"]
        child = json.loads(Path(parent_pin["path"]).read_text())
        child.update(
            status="accepted_r1_divl_checkpoint", policy_type="divl", parent=parent_pin
        )
        config.update(
            parent_acceptance=parent_pin,
            checkpoint_acceptance=evaluation.pin(write(tmp_path / "divl.json", child)),
        )
    attest(config, inputs, tmp_path)
    path = write(tmp_path / "config.json", config)
    assert mulligan_program.load_config(path) == config
    assert (
        subset.load_job(write(tmp_path / "job.json", job(path)))["max_wall_s"] == 1100
    )
    assert evaluation.command_for(config, path) == [
        sys.executable,
        "-I",
        "-u",
        "-B",
        "-m",
        "sadhana_mulligan_eval.cli",
        "--config",
        str(path),
    ]
    assert "sadhana_mulligan_eval.cli" not in subset.ALLOWED_MODULES


@pytest.mark.parametrize(
    "change",
    [
        {"arguments": []},
        {"environment": {}},
        {"method": "resfit"},
        {"num_action_samples": True},
        {"num_action_samples": 2},
        {"timeout_s": True},
        {"timeout_s": 0},
        {"timeout_s": float("inf")},
        {"timeout_s": 86401},
    ],
)
def test_invalid_profile_refused(prepared, change):
    config, _ = prepared
    config.update(change)
    with pytest.raises(ValueError):
        evaluation.validate(config)


@pytest.mark.parametrize(
    "damage",
    [
        "checkpoint_bytes",
        "extra_file",
        "unaccepted",
        "lineage",
        "training_steps",
        "missing_gate",
        "different_n",
    ],
)
def test_real_pin_inventory_and_lineage_refusals(tmp_path, prepared, damage):
    config, inputs = prepared
    checkpoint_path = Path(config["checkpoint_acceptance"]["path"])
    checkpoint = json.loads(checkpoint_path.read_text())
    if damage == "checkpoint_bytes":
        (Path(checkpoint["checkpoint_root"]) / "policy.pt").write_text("changed")
    elif damage == "extra_file":
        (Path(checkpoint["checkpoint_root"]) / "unlisted").write_text("extra")
    elif damage in ("unaccepted", "lineage", "training_steps"):
        if damage == "unaccepted":
            checkpoint["status"] = "prepared"
        if damage == "lineage":
            checkpoint["bindings"]["task"] = {}
        if damage == "training_steps":
            checkpoint["training_steps"] = 1000
        config["checkpoint_acceptance"] = evaluation.pin(
            write(checkpoint_path, checkpoint)
        )
        attest(config, inputs, tmp_path)
    else:
        path = Path(config["admission"]["path"])
        admission = json.loads(path.read_text())
        if damage == "missing_gate":
            del admission["evidence"]["original_restore"]
        else:
            admission["num_action_samples"] = 32
        config["admission"] = evaluation.pin(write(path, admission))
    with pytest.raises(ValueError):
        evaluation.validate(config)


def test_copied_release_cannot_redirect_sibling_inputs(tmp_path):
    copied = tmp_path / "CPU_QUALIFICATION.json"
    copied.write_bytes(evaluation.RELEASE.read_bytes())
    with pytest.raises(ValueError, match="exact installed"):
        evaluation.source_inputs({"caller_release": evaluation.pin(copied)})


@pytest.mark.parametrize("effective", [1099, 0, float("inf"), float("nan"), True])
def test_effective_parent_refusal_before_gpu_or_outputs(
    tmp_path, prepared, monkeypatch, effective
):
    config, _ = prepared
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


def test_no_lease_prevents_runtime_or_outputs(tmp_path, monkeypatch):
    def absent():
        raise RuntimeError("explicit absent canonical lease")

    monkeypatch.setattr(subset, "require_subset_lease", absent)
    monkeypatch.setattr(
        evaluation, "validate", lambda *_: pytest.fail("Reached runtime validation")
    )
    with pytest.raises(RuntimeError, match="absent"):
        evaluation.run({}, tmp_path / "must-not-exist")
    assert not (tmp_path / "must-not-exist").exists()


@pytest.mark.parametrize(
    "profile,grace", [(None, 5), (training.MODE, 90), (evaluation.MODE, 90)]
)
def test_cleanup_grace_is_scoped(tmp_path, monkeypatch, profile, grace):
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


@pytest.mark.parametrize("exit_code", [0, 75])
def test_real_cpu_child_receipt_no_score(tmp_path, monkeypatch, exit_code):
    """Only the wrapper runs. A stdlib child writes an explicit synthetic result."""
    monkeypatch.setattr(diagnostic, "WORKSPACE", tmp_path)
    monkeypatch.setattr(subset, "require_subset_lease", lambda: None)
    monkeypatch.setattr(evaluation.os, "getpgrp", lambda: os.getpid())
    monkeypatch.setattr(training.os, "killpg", lambda *_: None)
    monkeypatch.setattr(evaluation, "validate", lambda _: ({"settings": {}}, {}))
    monkeypatch.setattr(training, "verify_runtime", lambda *_: {"fixture": True})

    def fake_resolved(config, inputs, checkpoint, settings, destination):
        return {
            "schema": "CPU fixture",
            "settings": evaluation.pin(settings),
            "output_dir": str(destination),
        }

    monkeypatch.setattr(evaluation, "caller_config", fake_resolved)
    child_code = """import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]); c=json.loads(p.read_text()); out=Path(c['output_dir']); out.mkdir()
r={'schema':'sadhana.mulligan_frozen_eval_result/1','status':'completed','config':{'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()},'experiment_accepted':False,'method':'idql','num_action_samples':1}
(out/'RESULT.json').write_text(json.dumps(r))
sys.exit(int(sys.argv[2]))
"""
    original_popen = subprocess.Popen

    def child(command, **kwargs):
        assert command[4:6] == ["-m", "sadhana_mulligan_eval.cli"]
        assert not kwargs.get("start_new_session", False)
        return original_popen(
            [sys.executable, "-I", "-B", "-c", child_code, command[-1], str(exit_code)],
            **kwargs,
        )

    monkeypatch.setattr(evaluation.subprocess, "Popen", child)
    config = {
        "python": sys.executable,
        "timeout_s": 5,
        "method": "idql",
        "num_action_samples": 1,
        "runtimes": {"native": {"python": {"path": "/fixture"}}},
        "admission": {},
        "checkpoint_acceptance": {},
    }
    output = tmp_path / "canonical"
    if exit_code:
        with pytest.raises(ValueError, match="exited 75"):
            evaluation.run(config, output)
    else:
        assert evaluation.run(config, output)["status"] == "completed"
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == ("failed" if exit_code else "completed")
    assert receipt["experiment_accepted"] is False and receipt["metrics"] == {}
    assert receipt["cleanup"]["clean"]


def test_resolved_caller_preserves_frozen_recipe(tmp_path):
    template = json.loads((evaluation.PREPARATION / "CONFIG.template.json").read_text())
    learner_root = tmp_path / "learner"
    native_root = tmp_path / "native"
    learner = {
        "python": {"path": str(learner_root / "python")},
        "package_dir": str(learner_root / "sadhana_mulligan_boundary"),
        "author_package_dir": str(learner_root / "mulligan"),
    }
    inventory = []
    for package, names in evaluation.CALLER_MODULES.items():
        for name in names:
            inventory.append(
                {
                    "path": str(learner_root / package / name),
                    "bytes": 0,
                    "sha256": "fixture",
                }
            )
    settings = write(tmp_path / "settings.json", {"fixture": True})
    inputs = {
        "installed": inventory,
        "caller_template": template,
        "boundary": {
            "files": [
                {
                    "path": "src/sadhana_mulligan_boundary/protocol.py",
                    "bytes": 0,
                    "sha256": "fixture",
                }
            ]
        },
    }
    config = {
        "method": "divl",
        "num_action_samples": 32,
        "checkpoint_acceptance": {"fixture": "child"},
        "parent_acceptance": {"fixture": "parent"},
        "runtimes": {
            "learner": learner,
            "native": {
                "python": {"path": str(native_root / "python")},
                "package_dir": str(native_root / "sadhana_mulligan_boundary"),
            },
        },
    }
    checkpoint = {
        "checkpoint_root": str(tmp_path / "final_model"),
        "checkpoint_files": [{"fixture": True}],
    }
    actual = evaluation.caller_config(
        config, inputs, checkpoint, settings, tmp_path / "evaluation"
    )
    for key in (
        "schema",
        "eval_seed",
        "num_envs",
        "num_episodes",
        "max_steps",
        "device",
    ):
        assert actual[key] == template[key]
    assert (
        actual["num_envs"],
        actual["num_episodes"],
        actual["max_steps"],
        actual["eval_seed"],
    ) == (10, 20, 900, 42)
    assert actual["num_action_samples"] == 32 and actual["method"] == "divl"
    assert actual["settings"] == evaluation.pin(settings)
    assert actual["parent_acceptance"] == config["parent_acceptance"]
    assert actual["native_runtime_files"][0] == config["runtimes"]["native"]["python"]
    assert len(actual["installed_modules"]) == 11


def test_original_square_command_unchanged():
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


@pytest.mark.parametrize("exit_code", [0, 75])
def test_actual_canonical_receipt_and_index_use_r1_identity(
    tmp_path, prepared, monkeypatch, exit_code
):
    """Run real receipt construction/publication; GPU and worker calls are explicit doubles."""
    config, _ = prepared
    config_path = write(tmp_path / "config.json", config)
    job_path = write(tmp_path / "job.json", job(config_path))
    results = tmp_path / "results"
    write(
        results / "gpu_budget.json",
        {"gpu": 0, "charged_seconds": 17, "limit_seconds": 5000},
    )
    monkeypatch.setattr(runner, "RESULTS", results)

    def fake_gpu(gpu):
        assert gpu == 0
        return {"uuid": "GPU-EXPLICIT-CPU-FIXTURE"}

    monkeypatch.setattr(subset, "inspect_reserved_gpu", fake_gpu)
    observed = []

    def assert_identity(receipt, status):
        assert receipt["status"] == status
        assert receipt["task"] == "r1lite_remote_into_tray"
        assert receipt["embodiment"] == "r1lite_2025"
        assert receipt["phase"] == "original_r1_frozen_eval"
        assert receipt["experiment_accepted"] is False
        assert receipt["cleanup_profile"] == evaluation.MODE
        assert receipt["method"] == "idql" and receipt["num_action_samples"] == 1
        assert receipt["metrics"] == {}

    def fake_worker(command, log_path, timeout, **options):
        assert options == {
            "gpu_uuid": "GPU-EXPLICIT-CPU-FIXTURE",
            "cleanup_profile": evaluation.MODE,
        }
        assert timeout == 1100
        output = Path(command[-1])
        initial = json.loads((results / "runs" / (output.name + ".json")).read_text())
        assert_identity(initial, "running")
        assert json.loads((results / "results.json").read_text())["runs"] == [initial]
        observed.append(initial)
        log_path.write_text("Explicit CPU worker double; no evaluator or native task")
        if exit_code == 0:
            write(
                output / "receipt.json",
                {"status": "completed", "metrics": {}, "experiment_accepted": False},
            )
        return exit_code

    monkeypatch.setattr(runner, "execute_command", fake_worker)
    result = subset.run_job(
        SimpleNamespace(training_config=job_path, timeout_seconds=1100)
    )
    assert len(observed) == 1
    assert_identity(result, "completed" if exit_code == 0 else "failed")
    assert (
        json.loads((results / "runs" / (result["run_id"] + ".json")).read_text())
        == result
    )
    assert json.loads((results / "results.json").read_text())["runs"] == [result]
    ledger = json.loads((results / "gpu_budget.json").read_text())
    assert ledger["gpu"] == 0 and ledger["charged_seconds"] >= 17
    assert "active_parent_pid" not in ledger and "active_gpu_uuid" not in ledger
