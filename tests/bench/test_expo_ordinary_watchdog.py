"""Finite ordinary EXPO scope only; no worker, model or GPU execution."""

import json
import sys

import pytest

from abc_bench.subset import load_job


def job_file(tmp_path, *, phase="original_expo_task_rl", worker=18000):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "phase": phase,
                "parameters": {
                    "seed": 42,
                    "batch_size": 64,
                    "utd_ratio": 20,
                    "completed_episode_warmup": 10,
                    "updates_per_trigger": 3,
                    "max_controls": 38940,
                    "sampler_timing_plans": 0,
                    "max_seconds": worker,
                    "episode_seeds": list(range(9200, 9211)),
                },
            }
        )
    )
    job = {
        "schema_version": 1,
        "label": "Original ordinary EXPO first trigger",
        "python": sys.executable,
        "module": "expo_abc_adapter.stage",
        "arguments": ["--config", str(config)],
        "max_wall_s": 18300,
    }
    path = tmp_path / "job.json"
    path.write_text(json.dumps(job))
    return path, job, config


def test_explicit_ordinary_scope_has_finite_enclosing_watchdog(tmp_path):
    path, job, _ = job_file(tmp_path)
    assert load_job(path) == job
    job["max_wall_s"] = 18301
    path.write_text(json.dumps(job))
    with pytest.raises(ValueError, match="at most 18300"):
        load_job(path)


@pytest.mark.parametrize(
    "phase",
    [
        "original_pi05_task_sft",
        "original_expo_task_rl_capacity",
        "original_expo_task_rl_update_capacity",
        "original_pi05_task_frozen_eval",
        "original_expo_task_frozen_eval",
    ],
)
def test_other_expo_phases_keep_historical_cap(tmp_path, phase):
    path, job, _ = job_file(tmp_path, phase=phase)
    with pytest.raises(ValueError, match="eleven-episode"):
        load_job(path)
    job["max_wall_s"] = 7200
    path.write_text(json.dumps(job))
    assert load_job(path)["max_wall_s"] == 7200


@pytest.mark.parametrize(
    "change",
    [
        {"batch_size": 32},
        {"utd_ratio": 1},
        {"completed_episode_warmup": 1},
        {"max_seconds": 18001},
        {"max_seconds": True},
    ],
)
def test_changed_recipe_or_worker_budget_cannot_extend_cap(tmp_path, change):
    path, _, config = job_file(tmp_path)
    value = json.loads(config.read_text())
    value["parameters"].update(change)
    config.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="eleven-episode"):
        load_job(path)


def test_fast_campaign_and_other_modules_keep_their_existing_caps(tmp_path):
    path, job, _ = job_file(tmp_path)
    for module, bound in (
        ("abc_bench.fasttd3_campaign_stage", 12600),
        ("abc_author_subset.host", 7200),
    ):
        job.update(module=module, arguments=[], max_wall_s=bound)
        path.write_text(json.dumps(job))
        assert load_job(path)["max_wall_s"] == bound
        job["max_wall_s"] = bound + 1
        path.write_text(json.dumps(job))
        with pytest.raises(ValueError, match=f"at most {bound}"):
            load_job(path)
