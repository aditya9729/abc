"""Monitored-recipe boundary tests. Model/native execution is excluded."""

import copy
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_mulligan_r1lite_training import (
    config as config,  # noqa: PLC0414 - Re-export the pytest fixture.
)
from test_mulligan_r1lite_training import job as original_job
from test_mulligan_r1lite_training import write

from abc_bench import mulligan_r1lite_training as training
from abc_bench import runner, subset

M = Path("/home/user/aditya/RL/builds/other-methods-plan-20261015")
D = M / "mulligan-monitored-preparation-20261015"


@pytest.fixture
def monitored(tmp_path, monkeypatch, config):
    """Explicit small CPU identity fixture; not actual admission evidence."""
    original = Path(config["author_manifest"]["path"]).parent / "IDQL_ARGV.json"
    recipe = json.loads((D / "MONITORED_RECIPE.json").read_text())
    recipe["base_recipe"] = training.pin(original)
    path = write(tmp_path / "monitored/recipe.json", recipe)
    monkeypatch.setattr(training, "MONITORED_RECIPE_PATH", path)
    monkeypatch.setattr(
        training, "MONITORED_RECIPE_SHA256", training.pin(path)["sha256"]
    )
    config.update(monitor_recipe=training.pin(path), timeout_s=14400)
    admission_path = Path(config["admission"]["path"])
    admission = json.loads(admission_path.read_text())
    admission["timeout_s"] = 14400
    admission["bindings"]["monitor_recipe"] = config["monitor_recipe"]
    config["admission"] = training.pin(write(admission_path, admission))
    return config


def test_exact_monitored_recipe_has_only_two_cli_changes(tmp_path, monitored):
    loaded = training.load_config(write(tmp_path / "config.json", monitored))
    root, recipe, inputs = training.source_inputs(loaded)
    original = json.loads((root / "IDQL_ARGV.json").read_text())
    assert len(original["argv"]) == len(recipe["argv"])
    assert [(a, b) for a, b in zip(original["argv"], recipe["argv"]) if a != b] == [
        (
            "--training.periodic_checkpoint_freq=0",
            "--training.periodic_checkpoint_freq=10000",
        ),
        ("--eval.freq=150000", "--eval.freq=10000"),
    ]
    command = training.command_for(loaded, recipe, tmp_path / "fresh", None)
    assert command[4:6] == ["-m", "mulligan.training.train"]
    assert "--training.training_steps=150000" in command
    assert "--training.batch_size=1024" in command
    assert inputs["settings"]["split"] == "TRAIN"
    assert "--system.checkpoint_dir=" + str(tmp_path / "fresh/checkpoints") in command


@pytest.mark.parametrize(
    "damage", ["copied_path", "wrong_hash", "changed_bytes", "divl"]
)
def test_recipe_refusal_precedes_gpu_admission(
    tmp_path, monitored, monkeypatch, damage
):
    cfg = copy.deepcopy(monitored)
    path = Path(cfg["monitor_recipe"]["path"])
    if damage == "copied_path":
        copied = tmp_path / "untrusted/recipe.json"
        copied.parent.mkdir()
        copied.write_bytes(path.read_bytes())
        cfg["monitor_recipe"] = training.pin(copied)
    elif damage == "wrong_hash":
        cfg["monitor_recipe"]["sha256"] = "0" * 64
    elif damage == "changed_bytes":
        path.write_bytes(path.read_bytes() + b" ")
    else:
        cfg["stage"] = "divl"
    config_path = write(tmp_path / "bad.json", cfg)
    job = original_job(config_path)
    job["max_wall_s"] = 14500
    output = tmp_path / "must-not-exist"
    monkeypatch.setattr(runner, "RESULTS", output)
    monkeypatch.setattr(
        subset, "inspect_reserved_gpu", lambda *_: pytest.fail("GPU reached")
    )
    expected = {
        "copied_path": "exact reviewed monitored",
        "wrong_hash": "exact reviewed monitored",
        "changed_bytes": "Diagnostic input changed",
        "divl": "fresh IDQL only",
    }[damage]
    with pytest.raises(ValueError, match=expected):
        subset.run_job(
            SimpleNamespace(
                training_config=write(tmp_path / "job.json", job), timeout_seconds=14500
            )
        )
    assert not output.exists()


@pytest.mark.parametrize(
    "damage", ["different_batch", "extra_argument", "wrong_module", "different_base"]
)
def test_exact_recipe_contract_refuses_broader_override(
    tmp_path, monitored, monkeypatch, damage
):
    path = Path(monitored["monitor_recipe"]["path"])
    recipe = json.loads(path.read_text())
    if damage == "different_batch":
        recipe["argv"] = [
            "--training.batch_size=32" if a == "--training.batch_size=1024" else a
            for a in recipe["argv"]
        ]
    elif damage == "extra_argument":
        recipe["argv"].append("--training.training_steps=500")
    elif damage == "wrong_module":
        recipe["module"] = "arbitrary.module"
    else:
        recipe["base_recipe"]["sha256"] = "0" * 64
    monitored["monitor_recipe"] = training.pin(write(path, recipe))
    # Authenticate the synthetic bytes, then exercise the independent CLI contract.
    monkeypatch.setattr(
        training, "MONITORED_RECIPE_SHA256", monitored["monitor_recipe"]["sha256"]
    )
    with pytest.raises(ValueError, match="may change only"):
        training.source_inputs(monitored)


@pytest.mark.parametrize("damage", ["missing", "wrong_recipe", "extra_binding"])
def test_root_admission_must_bind_monitored_recipe(tmp_path, monitored, damage):
    path = Path(monitored["admission"]["path"])
    value = json.loads(path.read_text())
    if damage == "missing":
        del value["bindings"]["monitor_recipe"]
    elif damage == "wrong_recipe":
        value["bindings"]["monitor_recipe"]["sha256"] = "0" * 64
    else:
        value["bindings"]["arbitrary_override"] = True
    monitored["admission"] = training.pin(write(path, value))
    with pytest.raises(ValueError, match="Admission bindings"):
        training.load_config(write(tmp_path / "config.json", monitored))


@pytest.mark.parametrize("seconds,valid", [(14500, True), (14499, False)])
def test_monitored_budget_preserves_cleanup_margin(monitored, seconds, valid):
    job = {
        "task": training.TASK,
        "embodiment": training.EMBODIMENT,
        "max_wall_s": 14500,
    }
    if valid:
        training.check_parent(job, monitored, seconds)
    else:
        with pytest.raises(ValueError, match="Effective"):
            training.check_parent(job, monitored, seconds)


def test_old_recipe_and_admission_remain_valid(tmp_path, config):
    loaded = training.load_config(write(tmp_path / "old.json", config))
    root, recipe, _ = training.source_inputs(loaded)
    assert recipe == json.loads((root / "IDQL_ARGV.json").read_text())
    assert "--eval.freq=150000" in recipe["argv"]
    assert "--training.periodic_checkpoint_freq=0" in recipe["argv"]


def test_actual_original_cli_parser_only(tmp_path):
    """Root runs original Draccus with CUDA hidden; no trainer/model/native import."""
    release = M / "author-r1-release-02/RELEASE.json"
    assert (
        training.pin(release)["sha256"]
        == "7b33d066480bdc0858e0969d3dc583a3e408900d3bbebf92f6f5e1244451b58f"
    )
    python = training.verify(json.loads(release.read_text())["python"])
    recipe = json.loads(
        training.verify(training.pin(D / "MONITORED_RECIPE.json")).read_text()
    )
    assert (
        training.pin(D / "MONITORED_RECIPE.json")["sha256"]
        == training.MONITORED_RECIPE_SHA256
    )
    command = training.command_for(
        {"python": str(python)}, recipe, tmp_path / "fresh", None
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
    assert parsed["training"]["training_steps"] == 150000
    assert parsed["training"]["batch_size"] == 1024
    assert parsed["training"]["periodic_checkpoint_freq"] == 10000
    assert parsed["training"]["resume_checkpoint_freq"] == 5000
    assert parsed["eval"]["freq"] == 10000
    assert parsed["eval"]["num_envs"] == 10
    assert parsed["eval"]["episodes"] == 20
    assert parsed["eval"]["eval_num_action_samples"] == [1, 32]
    assert parsed["policy"]["chunk_size"] == 8
    assert parsed["policy"]["n_action_steps"] == 6
    assert parsed["training"]["amp_dtype"] == "bfloat16"
