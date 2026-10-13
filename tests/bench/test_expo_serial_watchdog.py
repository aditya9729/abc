"""New serial scope only. No worker/model/GPU execution."""

import json

import pytest
from test_expo_ordinary_watchdog import job_file

from abc_bench.subset import load_job


def test_serial_profile_keeps_exact_parent_and_worker_bounds(tmp_path):
    path, job, config = job_file(tmp_path, phase="original_expo_task_rl_serial")
    body = json.loads(config.read_text())
    body["parameters"]["memory_scheduling_profile"] = "original_queue3_serial_update_v1"
    config.write_text(json.dumps(body))
    assert load_job(path) == job
    job["max_wall_s"] = 18301
    path.write_text(json.dumps(job))
    with pytest.raises(ValueError):
        load_job(path)


def test_serial_profile_cannot_extend_other_phases_or_recipes(tmp_path):
    path, job, config = job_file(tmp_path, phase="original_expo_task_rl_serial")
    body = json.loads(config.read_text())
    for change in (
        {},
        {"memory_scheduling_profile": "fallback"},
        {
            "memory_scheduling_profile": "original_queue3_serial_update_v1",
            "updates_per_trigger": 1,
        },
        {
            "memory_scheduling_profile": "original_queue3_serial_update_v1",
            "batch_size": 32,
        },
    ):
        candidate = json.loads(json.dumps(body))
        candidate["parameters"].update(change)
        config.write_text(json.dumps(candidate))
        with pytest.raises(ValueError):
            load_job(path)
    body["parameters"]["memory_scheduling_profile"] = "original_queue3_serial_update_v1"
    for phase in (
        "original_expo_task_rl_serial_capacity",
        "original_expo_task_serial_frozen_eval",
    ):
        body["phase"] = phase
        config.write_text(json.dumps(body))
        with pytest.raises(ValueError):
            load_job(path)
    job["max_wall_s"] = 7200
    path.write_text(json.dumps(job))
    assert load_job(path)["max_wall_s"] == 7200
