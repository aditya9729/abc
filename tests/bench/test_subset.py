"""Boundary failures must be rejected before the GPU or ledger is touched."""

import fcntl
import json
import os
import sys

import pytest

from abc_bench.subset import load_job, require_subset_lease


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": True},
        {"module": "arbitrary.module"},
        {"max_wall_s": 0},
        {"max_wall_s": float("inf")},
        {"max_wall_s": True},
        {"max_wall_s": 7201},
        {"python": "relative/python"},
        {"arguments": ["--output", "/other"]},
        {"arguments": [17]},
    ],
)
def test_invalid_job_is_rejected(tmp_path, change):
    job = {
        "schema_version": 1,
        "label": "original VLA smoke",
        "python": sys.executable,
        "module": "abc_author_subset.host",
        "arguments": ["--mode", "smoke"],
        "max_wall_s": 120,
        **change,
    }
    path = tmp_path / "job.json"
    path.write_text(json.dumps(job))
    with pytest.raises(ValueError):
        load_job(path)


def test_worker_requires_a_held_ancestor_lease(tmp_path, monkeypatch):
    import abc_bench.runner

    monkeypatch.setattr(abc_bench.runner, "RESULTS", tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "test-reserved-uuid")
    monkeypatch.setenv("MUJOCO_GL", "disable")
    (tmp_path / "gpu_budget.json").write_text(
        json.dumps(
            {
                "gpu": 0,
                "active_parent_pid": os.getppid(),
                "active_gpu_uuid": "test-reserved-uuid",
            }
        )
    )
    lock_path = tmp_path / ".gpu-budget.lock"
    lock_path.touch()
    with pytest.raises(RuntimeError, match="not held"):
        require_subset_lease()
    with lock_path.open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        require_subset_lease()
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "wrong-uuid")
        with pytest.raises(RuntimeError, match="visibility differs"):
            require_subset_lease()
