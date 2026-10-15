"""CPU process fixtures only: no author learner, simulator, or GPU execution."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from abc_bench import mulligan_program as worker
from abc_bench.subset import load_job


@pytest.fixture
def config(tmp_path, monkeypatch):
    source = tmp_path / "source"
    script = source / "scripts/sim/eval_cell.sh"
    script.parent.mkdir(parents=True)
    script.write_text('#!/bin/bash\nexec "$PYTHON" "$MULLIGAN_PROCESS_FIXTURE" "$@"\n')
    recipes = source / "configs/sim/recipes.json"
    recipes.parent.mkdir(parents=True)
    recipes.write_text(
        json.dumps(
            {
                "recipes": [
                    {
                        "id": "square-narrow-r01-mulligan",
                        "task": "square_narrow",
                        "idql_agent": {},
                    }
                ]
            }
        )
    )
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
            "Explicit CPU process fixture",
        ],
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    monkeypatch.setattr(worker, "AUTHOR_COMMIT", commit)
    monkeypatch.setattr(worker, "WORKSPACE", tmp_path)
    monkeypatch.setattr(worker, "require_subset_lease", lambda: None)
    monkeypatch.setattr(
        worker,
        "verify_runtime",
        lambda config: {"scope": "Explicit CPU process fixture, no author runtime"},
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-EXPLICIT-PROCESS-FIXTURE")
    monkeypatch.setenv("MUJOCO_GL", "disable")
    fixture = tmp_path / "process_fixture.py"
    fixture.write_text("print('Explicit CPU process fixture; no simulation')\n")
    monkeypatch.setenv("MULLIGAN_PROCESS_FIXTURE", str(fixture))
    return {
        "schema_version": 1,
        "python": sys.executable,
        "source_root": str(source),
        "source_commit": commit,
        "mode": "gpu",
        "entrypoint": "scripts/sim/eval_cell.sh",
        "arguments": ["square-narrow-r01-mulligan", "--seed", "1"],
        "timeout_s": 3,
    }


def write_config(tmp_path, config):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


def test_original_argv_environment_and_no_scientific_acceptance(
    tmp_path, monkeypatch, config
):
    marker = tmp_path / "must-not-exist"
    config["arguments"] += [f"$(touch {marker})"]
    Path(os.environ["MULLIGAN_PROCESS_FIXTURE"]).write_text(
        "import json,os,sys\nfrom pathlib import Path\n"
        "data={k:os.environ[k] for k in ['CUDA_VISIBLE_DEVICES','MUJOCO_GL','WANDB_MODE','PYTHON','MULLIGAN_OUTPUT_DIR','TMPDIR']}\n"
        "data['argv']=sys.argv[1:]\n"
        "Path(os.environ['MULLIGAN_OUTPUT_DIR'],'observed.json').write_text(json.dumps(data))\n"
    )
    loaded = worker.load_config(write_config(tmp_path, config))
    result = worker.run(loaded, tmp_path / "out")
    observed = json.loads((tmp_path / "out/author-output/observed.json").read_text())
    assert observed["argv"] == config["arguments"]
    assert observed["CUDA_VISIBLE_DEVICES"] == "GPU-EXPLICIT-PROCESS-FIXTURE"
    assert observed["MUJOCO_GL"] == "osmesa" and os.environ["MUJOCO_GL"] == "disable"
    assert observed["WANDB_MODE"] == "offline"
    assert observed["PYTHON"] == sys.executable
    assert Path(observed["TMPDIR"]).is_relative_to(tmp_path / "out")
    assert not marker.exists()
    assert result["status"] == "completed" and result["experiment_accepted"] is False
    assert result["task"] == "SquareNarrow" and result["embodiment"] == "franka_panda"
    assert result["descendant_cleanup_owner"] == "canonical runner"


def test_source_change_and_untracked_module_fail_before_outputs(tmp_path, config):
    source = Path(config["source_root"])
    (source / "shadow.py").write_text("# A module that could shadow author imports\n")
    with pytest.raises(ValueError, match="clean"):
        worker.run(config, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    (source / "shadow.py").unlink()
    (source / "scripts/sim/eval_cell.sh").write_text("exit 0\n")
    with pytest.raises(ValueError, match="clean"):
        worker.run(config, tmp_path / "out")


def test_timeout_and_exit75_remain_failed(tmp_path, config):
    fixture = Path(os.environ["MULLIGAN_PROCESS_FIXTURE"])
    fixture.write_text("import time\ntime.sleep(30)\n")
    config["timeout_s"] = 0.1
    with pytest.raises(subprocess.TimeoutExpired):
        worker.run(config, tmp_path / "timeout")
    result = json.loads((tmp_path / "timeout/receipt.json").read_text())
    assert result["status"] == "failed" and result["direct_child_stopped"]
    assert not Path(f"/proc/{result['author_pid']}").exists()
    fixture.write_text("raise SystemExit(75)\n")
    with pytest.raises(RuntimeError, match="exited75"):
        worker.run(config, tmp_path / "exit75")
    result = json.loads((tmp_path / "exit75/receipt.json").read_text())
    assert result["status"] == "failed" and result["author_exit_code"] == 75
    assert result["experiment_accepted"] is False


def test_gpu_command_requires_existing_lease(tmp_path, monkeypatch, config):
    def absent():
        raise RuntimeError("Explicit missing lease fixture")

    monkeypatch.setattr(worker, "require_subset_lease", absent)
    with pytest.raises(RuntimeError, match="missing lease"):
        worker.run(config, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "change",
    [
        {"timeout_s": float("inf")},
        {"timeout_s": True},
        {"entrypoint": "arbitrary.module"},
        {
            "arguments": [
                "square-narrow-r01-mulligan",
                "--seed",
                "1",
                "--output-dir",
                "/tmp/escape",
            ]
        },
        {"environment": {"CUDA_VISIBLE_DEVICES": "1"}},
        {"arguments": ["square-narrow-r01-mulligan", "--seed", "1", "--env.name=Lift"]},
    ],
)
def test_config_rejects_boundary_changes(tmp_path, config, change):
    config.update(change)
    with pytest.raises(ValueError):
        worker.load_config(write_config(tmp_path, config))


def test_finite_long_job_keeps_author_config_and_generic_bounds(tmp_path, config):
    config["timeout_s"] = 20000
    path = write_config(tmp_path, config)
    job = {
        "schema_version": 1,
        "module": "abc_bench.mulligan_program",
        "python": sys.executable,
        "arguments": ["--config", str(path)],
        "label": "Explicit CPU config test; no execution",
        "max_wall_s": 20060,
    }
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(job))
    assert load_job(job_path)["max_wall_s"] == 20060
    job["max_wall_s"] = 20059
    job_path.write_text(json.dumps(job))
    with pytest.raises(ValueError, match="cleanup"):
        load_job(job_path)
    job.update(module="abc_bench.author_program", max_wall_s=7201, arguments=[])
    job_path.write_text(json.dumps(job))
    with pytest.raises(ValueError, match="one explicit --config"):
        load_job(job_path)
    job["module"] = "abc_bench.upstream_eval"
    job_path.write_text(json.dumps(job))
    with pytest.raises(ValueError, match="7200"):
        load_job(job_path)


def test_cpu_start_design_is_separate_and_never_admitted_as_gpu(tmp_path, config):
    sampler = Path(config["source_root"], "configs/sim/square_narrow/r01_sampler.yaml")
    sampler.parent.mkdir(parents=True)
    sampler.write_text("# Explicit CPU config fixture\n")
    config.update(
        mode="cpu-start-design",
        entrypoint=worker.CPU_MODULE,
        arguments=["--config", str(sampler), "--check"],
    )
    path = write_config(tmp_path, config)
    loaded = worker.load_config(path)
    argv = worker.command_for(loaded, tmp_path / "author-output")
    assert argv[:6] == [sys.executable, "-I", "-u", "-B", "-m", worker.CPU_MODULE]
    assert argv[-2:] == ["--out", str(tmp_path / "author-output")]
    job = {
        "schema_version": 1,
        "module": "abc_bench.mulligan_program",
        "python": sys.executable,
        "arguments": ["--config", str(path)],
        "label": "CPU configuration fixture",
        "max_wall_s": 120,
    }
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(job))
    with pytest.raises(ValueError, match="GPU"):
        load_job(job_path)
