"""CPU process doubles only: no JAX, physics, learner, or GPU execution."""

import argparse
import json
import os
import sys
from pathlib import Path

import pytest

from abc_bench import author_program, runner, subset


@pytest.fixture
def catcher_config(tmp_path, monkeypatch):
    from test_author_program import fixture_config

    config = fixture_config(tmp_path, "# Unused process fixture\n")
    source = Path(config["source_root"])
    (source / "train_pi_robo.py").write_text("print('CPU process double')\n")
    (source / "client").mkdir()
    (source / "client/__init__.py").write_text("")
    (source / "client/run_client.py").write_text("# CPU process double\n")
    paths = {}
    for name in ("config", "model_config"):
        file = tmp_path / (name + ".py")
        file.write_text("# Explicit metadata boundary double\n")
        paths[name] = str(file)
    monkeypatch.setattr(author_program, "catcher_config_paths", lambda _: paths)
    config.update(
        schema_version=2,
        task_profile=author_program.CATCHER_GPU_PROFILE,
        program="train_pi_robo.py",
        timeout_s=5,
        arguments=[
            "--config=" + paths["model_config"],
            "--config_task",
            paths["config"],
        ],
        client={
            "python": sys.executable,
            "arguments": ["--config-task-path", "realtime_expo_catcher_task.config"],
        },
    )
    return config


def write_config(tmp_path, config):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


@pytest.mark.parametrize(
    "change",
    [
        {"task_profile": "arbitrary-role"},
        {"program": "eval_droid_policy.py"},
        {"client": None},
        {"server_environment": {"CUDA_VISIBLE_DEVICES": "other"}},
        {"environment": {"CUDA_VISIBLE_DEVICES": "other"}},
        {"client": {"python": sys.executable, "arguments": [], "role": "cuda"}},
    ],
)
def test_wrong_role_program_or_override_rejected(tmp_path, catcher_config, change):
    with pytest.raises(ValueError):
        author_program.load_config(write_config(tmp_path, {**catcher_config, **change}))


@pytest.mark.parametrize(
    "location", ["task", "model", "client", "duplicate", "nested_override"]
)
def test_profile_cannot_label_a_droid_dispatch_catcher(
    tmp_path, catcher_config, location
):
    if location == "task":
        catcher_config["arguments"][-1] = str(tmp_path / "droid.py")
    elif location == "model":
        catcher_config["arguments"][0] = "--config=" + str(tmp_path / "droid.py")
    elif location == "client":
        catcher_config["client"]["arguments"][-1] = "configs.task.pick"
    elif location == "duplicate":
        catcher_config["arguments"].append("--config_task=/other.py")
    else:
        catcher_config["arguments"].append("--config_task.env_type=droid")
    with pytest.raises(ValueError):
        author_program.load_config(write_config(tmp_path, catcher_config))


@pytest.mark.parametrize(
    "arguments",
    [
        ["--config_task_path", "realtime_expo_catcher_task.config"],
        ["--config_task_path=realtime_expo_catcher_task.config"],
        ["--config-task-path=realtime_expo_catcher_task.config"],
    ],
)
def test_single_tyro_task_alias_preserves_the_module_value(
    tmp_path, catcher_config, arguments
):
    catcher_config["client"]["arguments"] = arguments
    result = author_program.load_config(write_config(tmp_path, catcher_config))
    assert result["client"]["arguments"] == arguments


@pytest.mark.parametrize(
    "arguments",
    [
        [
            "--config-task-path",
            "realtime_expo_catcher_task.config",
            "--config_task_path",
            "configs.task.pick",
        ],
        [
            "--config_task_path=configs.task.pick",
            "--config-task-path",
            "realtime_expo_catcher_task.config",
        ],
        [
            "--config_task_path",
            "realtime_expo_catcher_task.config",
            "--config-task-path=configs.task.pick",
        ],
        [
            "--config-task-path",
            "realtime_expo_catcher_task.config",
            "--config_task_path=realtime_expo_catcher_task.config",
        ],
    ],
)
def test_tyro_alias_duplicates_cannot_override_routing(
    tmp_path, catcher_config, arguments
):
    catcher_config["client"]["arguments"] = arguments
    with pytest.raises(ValueError, match="exactly one --config-task-path"):
        author_program.load_config(write_config(tmp_path, catcher_config))


def test_single_underscore_alias_cannot_select_droid(tmp_path, catcher_config):
    catcher_config["client"]["arguments"] = ["--config_task_path=configs.task.pick"]
    with pytest.raises(ValueError, match="installed Catcher task module"):
        author_program.load_config(write_config(tmp_path, catcher_config))


@pytest.mark.parametrize("flag", ["model", "task"])
def test_server_configuration_paths_must_be_absolute(
    tmp_path, monkeypatch, catcher_config, flag
):
    monkeypatch.chdir(tmp_path)
    if flag == "model":
        catcher_config["arguments"][0] = "--config=model_config.py"
    else:
        catcher_config["arguments"][-1] = "config.py"
    with pytest.raises(ValueError, match="absolute server"):
        author_program.load_config(write_config(tmp_path, catcher_config))


def test_worker_configuration_path_must_be_absolute(
    tmp_path, monkeypatch, catcher_config
):
    write_config(tmp_path, catcher_config)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="absolute worker"):
        author_program.load_config(Path("config.json"))


@pytest.mark.parametrize("platform", [None, "cpu", "cuda,cpu", "gpu"])
def test_inherited_non_cuda_refuses_before_output(
    tmp_path, monkeypatch, catcher_config, platform
):
    monkeypatch.setattr(author_program, "require_subset_lease", lambda: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "explicit-cpu-double-uuid")
    if platform is None:
        monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    else:
        monkeypatch.setenv("JAX_PLATFORMS", platform)
    with pytest.raises(ValueError, match="inherited"):
        author_program.run(catcher_config, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_uuid_override_is_not_a_role_option(monkeypatch, catcher_config):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "canonical-uuid")
    monkeypatch.setenv("JAX_PLATFORMS", "cuda")
    with pytest.raises(ValueError, match="leased GPU UUID"):
        author_program.role_environments(
            catcher_config,
            {"CUDA_VISIBLE_DEVICES": "another-uuid", "JAX_PLATFORMS": "cuda"},
        )


def test_schema1_keeps_shared_environment_and_short_parent(tmp_path):
    from test_author_program import fixture_config

    env = {"CUDA_VISIBLE_DEVICES": "old", "JAX_PLATFORMS": "cpu"}
    assert author_program.role_environments({"schema_version": 1}, env) == (env, env)
    config = fixture_config(tmp_path, "print('old process')\n")
    path = write_config(tmp_path, config)
    job = {
        "schema_version": 1,
        "module": "abc_bench.author_program",
        "python": sys.executable,
        "label": "historical process",
        "arguments": ["--config", str(path)],
        "max_wall_s": 5,
    }
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(job))
    assert subset.load_job(job_path) == job
    job["arguments"] = ["--help"]
    job_path.write_text(json.dumps(job))
    assert subset.load_job(job_path) == job


def test_actual_cpu_children_receive_separate_role_environments(
    tmp_path, monkeypatch, catcher_config
):
    """The fake UUID is checked by stdlib children; no CUDA package is imported."""
    import hashlib

    source = Path(catcher_config["source_root"])
    (source / "train_pi_robo.py").write_text(
        "import os,time\nfrom pathlib import Path\n"
        "assert os.environ['CUDA_VISIBLE_DEVICES']=='explicit-cpu-double-uuid'\n"
        "assert os.environ['JAX_PLATFORMS']=='cuda'\n"
        "assert os.environ['JAX_PLATFORM_NAME']=='cuda'\n"
        "marker=Path(os.environ['ABC_REALTIME_OUTPUT_DIR'])/'client-ready'\n"
        "for _ in range(100):\n"
        " if marker.exists():break\n"
        " time.sleep(.02)\n"
        "else:raise RuntimeError('client never ready')\n"
    )
    (source / "client/run_client.py").write_text(
        "import atexit,os,time\nfrom pathlib import Path\n"
        "assert os.environ['CUDA_VISIBLE_DEVICES']==''\n"
        "assert os.environ['JAX_PLATFORMS']=='cpu'\n"
        "assert os.environ['JAX_ENABLE_COMPILATION_CACHE']=='false'\n"
        "root=Path(os.environ['ABC_REALTIME_OUTPUT_DIR'])\n"
        "atexit.register(lambda:(root/'client-closed').write_text('closed'))\n"
        "(root/'client-ready').write_text('ready')\n"
        "try:\n"
        " while True:time.sleep(.1)\n"
        "except KeyboardInterrupt:pass\n"
    )
    files = []
    for path in sorted(source.rglob("*.py")):
        data = path.read_bytes()
        files.append(
            {
                "path": str(path.relative_to(source)),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    Path(catcher_config["source_manifest"]).write_text(json.dumps({"files": files}))
    monkeypatch.setattr(author_program, "require_subset_lease", lambda: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "explicit-cpu-double-uuid")
    monkeypatch.setenv("JAX_PLATFORMS", "cuda")
    monkeypatch.setenv("JAX_PLATFORM_NAME", "cpu")
    result = author_program.run(catcher_config, tmp_path / "out")
    assert result["status"] == "completed" and result["owned_children_stopped"]
    assert result["checkpoint_accepted"] is False
    assert (
        result["role_environments"]["author"]["CUDA_VISIBLE_DEVICES"]
        == "explicit-cpu-double-uuid"
    )
    assert result["role_environments"]["client"]["CUDA_VISIBLE_DEVICES"] == ""
    assert result["task"] == "Catcher-v3" and result["embodiment"] == "kinetix_symbolic"
    assert (tmp_path / "out/environment/client-closed").read_text() == "closed"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "explicit-cpu-double-uuid"
    assert os.environ["JAX_PLATFORM_NAME"] == "cpu"


def make_job(tmp_path, config):
    path = write_config(tmp_path, config)
    job = {
        "schema_version": 1,
        "module": "abc_bench.author_program",
        "python": sys.executable,
        "label": "Catcher CPU process double",
        "arguments": ["--config", str(path)],
        "max_wall_s": 105,
    }
    destination = tmp_path / "job.json"
    destination.write_text(json.dumps(job))
    return destination, job


@pytest.mark.parametrize(
    "change", [{"max_wall_s": 104}, {"task": "bottles"}, {"embodiment": "native_yam"}]
)
def test_canonical_parent_rejects_wrong_scope_or_short_bound(
    tmp_path, catcher_config, change
):
    path, job = make_job(tmp_path, catcher_config)
    path.write_text(json.dumps({**job, **change}))
    with pytest.raises(ValueError, match="fixed task identity"):
        subset.load_job(path)


def test_effective_cli_watchdog_cannot_shorten_parent(tmp_path, catcher_config):
    path, _ = make_job(tmp_path, catcher_config)
    with pytest.raises(ValueError, match="Effective Catcher parent"):
        subset.run_job(argparse.Namespace(training_config=path, timeout_seconds=104))


@pytest.mark.parametrize("spelling", ["abbreviated", "duplicate"])
def test_schema2_cannot_hide_from_canonical_profile_detection(
    tmp_path, catcher_config, spelling
):
    path, job = make_job(tmp_path, catcher_config)
    config = job["arguments"][1]
    job["arguments"] = (
        ["--conf", config]
        if spelling == "abbreviated"
        else ["--config", config, "--config", config]
    )
    path.write_text(json.dumps(job))
    with pytest.raises(ValueError, match="one explicit --config"):
        subset.load_job(path)


def test_relative_config_uses_actual_worker_cwd_before_schema2_rejection(
    tmp_path, monkeypatch, catcher_config
):
    path, job = make_job(tmp_path, catcher_config)
    job["arguments"] = ["--config", "config.json"]
    path.write_text(json.dumps(job))
    caller = tmp_path / "caller"
    caller.mkdir()
    (caller / "config.json").write_text('{"schema_version": 1}\n')
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.chdir(caller)
    with pytest.raises(ValueError, match="absolute --config"):
        subset.load_job(path)


def test_schema1_relative_worker_config_keeps_historical_admission(
    tmp_path, monkeypatch, catcher_config
):
    path, job = make_job(tmp_path, catcher_config)
    legacy = {**catcher_config, "schema_version": 1}
    (tmp_path / "config.json").write_text(json.dumps(legacy))
    caller = tmp_path / "caller"
    caller.mkdir()
    (caller / "config.json").write_text(json.dumps(catcher_config))
    job["arguments"] = ["--config", "config.json"]
    job["max_wall_s"] = 5
    path.write_text(json.dumps(job))
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.chdir(caller)
    assert subset.load_job(path) == job


def test_canonical_receipt_uses_validated_catcher_identity(
    tmp_path, monkeypatch, catcher_config
):
    path, _ = make_job(tmp_path, catcher_config)
    monkeypatch.setattr(runner, "RESULTS", tmp_path / "results")
    runner.RESULTS.mkdir()
    (runner.RESULTS / "gpu_budget.json").write_text(
        json.dumps({"gpu": 0, "charged_seconds": 7, "limit_seconds": 1000})
    )
    monkeypatch.setattr(
        subset,
        "inspect_reserved_gpu",
        lambda _: {"uuid": "explicit-cpu-double", "ordinal": 0},
    )

    def cpu_process_double(command, log, timeout, *, gpu_uuid):
        assert gpu_uuid == "explicit-cpu-double" and timeout == 105
        Path(command[-1], "receipt.json").write_text(
            json.dumps({"status": "completed", "metrics": {}})
        )
        return 0

    monkeypatch.setattr(runner, "execute_command", cpu_process_double)
    result = subset.run_job(
        argparse.Namespace(training_config=path, timeout_seconds=105)
    )
    assert result["task"] == "Catcher-v3" and result["embodiment"] == "kinetix_symbolic"
    assert "no frozen score" in result["claim_scope"]
