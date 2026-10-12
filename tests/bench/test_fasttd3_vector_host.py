"""Explicit CPU doubles and real owned-child exit regression; no simulator."""

import json
import multiprocessing as mp
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from abc_bench import fasttd3_vector_host as host


def config(profile="cpu_acceptance2"):
    return {
        "schema_version": 1,
        "profile": profile,
        "num_envs": 2 if profile == "cpu_acceptance2" else 128,
        "seed_base": 200000,
        "seed_stride": 128,
        "max_episode_steps": 3540,
        "vector_steps": 64,
        "timeouts": {
            "ready_s": 10,
            "step_s": 10,
            "shutdown_s": 2,
            "wall_s": 60,
            "rpc_s": 10,
        },
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_envs", 1),
        ("num_envs", True),
        ("seed_base", 20260511),
        ("seed_stride", 1),
        ("max_episode_steps", 100),
        ("vector_steps", 65),
    ],
)
def test_configuration_fails_before_world_creation(field, value):
    value_config = config()
    value_config[field] = value
    with pytest.raises(ValueError):
        host.validate_config(value_config)


def test_scratch_training_has_its_own_exact_budget_and_world_profile():
    value = config("scratch_training128")
    value["vector_steps"] = 4000
    assert host.validate_config(value) is value
    value["vector_steps"] = 64
    with pytest.raises(ValueError, match="4000"):
        host.validate_config(value)


@pytest.mark.parametrize("steps", [4000, True, 65])
def test_qualification_cannot_silently_expand_into_training(steps):
    value = config("training128")
    value["vector_steps"] = steps
    with pytest.raises(ValueError):
        host.validate_config(value)


def test_bank_is_unique_and_does_not_overlap_fallback_or_evaluation():
    settings = config("training128")
    normal = {
        host.seed_for(settings, rank, episode)
        for episode in range(781)
        for rank in range(128)
    }
    fallback = {seed + 100000 for seed in normal}
    assert len(normal) == 99968
    assert not normal & fallback
    assert not (normal | fallback) & host.EXCLUDED_SEEDS
    with pytest.raises(ValueError, match="episode_index"):
        host.seed_for(settings, 0, 781)


def test_original_placement_fallback_and_ordinary_arm_carry(monkeypatch, tmp_path):
    class SamplingError(Exception):
        pass

    monkeypatch.setitem(
        sys.modules,
        "abc_sim.randomization",
        SimpleNamespace(RandomizationSamplingError=SamplingError),
    )
    calls, forget = [], []

    def reset(**kwargs):
        calls.append(kwargs)
        if len(calls) < 3:
            raise SamplingError("explicit placement failure double")
        state = {"seed": kwargs["seed"], "metadata": {"bottle_count": 5}}
        return {}, {"randomization": SimpleNamespace(to_dict=lambda: state)}

    world = host.World.__new__(host.World)
    world.config, world.rank, world.output = config(), 1, tmp_path
    world.closed, world.episode_index, world.schema = False, -1, None
    world.env = SimpleNamespace(
        reset=reset, forget_arm_state=lambda: forget.append(True)
    )
    world.description = lambda: {"stable": True}

    def snapshot(stem):
        (tmp_path / f"{stem}-scene.json").write_text("{}\n")
        return {"explicit_snapshot_double": stem, "native_policy_state": [0.0] * 14}

    world.snapshot = snapshot
    world.observation = lambda: [0.0] * 216
    first = world.reset()
    assert [row["seed"] for row in calls] == [200001, 200001, 300001]
    assert len(forget) == 1
    assert all(row["options"] == host.RESET_OPTIONS for row in calls)
    assert [row["status"] for row in first["attempts"]] == [
        "placement_failed",
        "placement_failed",
        "completed",
    ]
    world.reset()
    assert calls[-1]["seed"] == 200129
    assert len(forget) == 1  # Ordinary reset carries arms.


def test_invalid_vector_action_does_not_advance_any_world():
    pool = host.VectorPool.__new__(host.VectorPool)
    pool.config, pool.initial_pending, pool.vector_steps = config(), False, 0
    pool._call = lambda requests: pytest.fail("must validate all rows before dispatch")
    with pytest.raises(ValueError):
        pool.step([[0.0] * 14, [float("nan")] + [0.0] * 13])


def test_terminal_observation_precedes_native_autoreset(tmp_path):
    world = host.World.__new__(host.World)
    world.closed, world.controls, world.episode_steps, world.episode_index = (
        False,
        0,
        0,
        0,
    )
    world.rank, world.actual_seed, world.requested_seed = 0, 200000, 200000
    world.output, world.reset_attempts = tmp_path, []
    world.schema = {"action_schema": {"policy_low": [-2] * 14, "policy_high": [2] * 14}}
    issued = []

    def step(action, render_obs):
        issued.append((action.copy(), render_obs))
        return None, 1.0, True, False, {"task_eval": {"num_bottles_in_bin": 5}}

    world.env = SimpleNamespace(
        step=step,
        data=SimpleNamespace(ctrl=np.ones(14), time=0.034),
        get_state=lambda: np.zeros(14, dtype=np.float32),
    )
    world.observation = lambda: [3.0] * 216
    world.reset = lambda: {
        "obs": [4.0] * 216,
        "requested_seed": 200128,
        "actual_seed": 200128,
    }
    result = world.step([2.0] * 14)
    assert result["terminal_obs"] == [3.0] * 216
    assert result["obs"] == [4.0] * 216
    assert result["terminated"] is True and result["truncated"] is False
    assert result["normalized_clipped"] == 14
    assert np.array_equal(issued[0][0], np.full(14, 2, dtype=np.float32))
    assert issued[0][1] is False


@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "alive", "nonzero", "bool_exit"]
)
def test_cleanup_rejects_incomplete_or_failed_owned_children(change):
    rows = [
        {"rank": i, "pid": 100 + i, "exitcode": 0, "alive": False} for i in range(2)
    ]
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[1]["pid"] = rows[0]["pid"]
    elif change == "alive":
        rows[1]["alive"] = True
    elif change == "nonzero":
        rows[1]["exitcode"] = 7
    else:
        rows[1]["exitcode"] = False
    with pytest.raises(ValueError):
        host.validate_cleanup(rows, 2)


def test_actual_joined_nonzero_child_cannot_be_accepted(tmp_path):
    context = mp.get_context("fork")
    children = [context.Process(target=os._exit, args=(code,)) for code in (0, 7)]
    for child in children:
        child.start()
        child.join(timeout=5)
        assert not child.is_alive()
    pool = host.VectorPool.__new__(host.VectorPool)
    pool.config, pool.workers, pool.connections = config(), children, []
    pool.output, pool.worker_resources = tmp_path, []
    pool.closed, pool.close_error, pool.worker_cleanup = False, None, []
    with pytest.raises(RuntimeError, match="cleanup failed"):
        pool.close()
    assert [row["exitcode"] for row in pool.worker_cleanup] == [0, 7]
    with pytest.raises(RuntimeError, match="cleanup failed"):
        pool.close()  # Failure remains sticky.


def test_response_deadline_is_total_not_per_world(monkeypatch):
    pool = host.VectorPool.__new__(host.VectorPool)
    pool.deadline = time.monotonic() + 100
    connection = SimpleNamespace()
    monkeypatch.setattr(host, "wait", lambda connections, timeout: [])
    with pytest.raises(TimeoutError, match="total response deadline"):
        pool._receive([connection], 1)


def test_cpu_fixture_not_in_public_rpc_contract():
    with pytest.raises(ValueError, match="operation"):
        host.dispatch(
            None, {"schema": host.SCHEMA, "id": 1, "operation": "cpu_fixture"}
        )


def test_server_lease_precedes_pool_native_construction(monkeypatch, tmp_path):
    seen = []

    def rejected_lease():
        seen.append("lease")
        raise RuntimeError("explicit missing canonical lease")

    monkeypatch.setattr(host, "require_subset_lease", rejected_lease)
    monkeypatch.setattr(
        host,
        "VectorPool",
        lambda *a: pytest.fail("no native construction before lease"),
    )
    with pytest.raises(RuntimeError, match="missing canonical lease"):
        host.serve(config("training128"), tmp_path / "socket", tmp_path)
    assert seen == ["lease"]


def test_server_checks_reserved_gpu_lease_then_hides_cuda_before_native_pool(
    monkeypatch, tmp_path
):
    seen = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-explicit-coordinator-double")
    monkeypatch.setenv("MUJOCO_GL", "disable")
    tmp_path.chmod(0o700)

    def lease():
        seen.append(os.environ["CUDA_VISIBLE_DEVICES"])

    def pool(*args):
        seen.append(os.environ["CUDA_VISIBLE_DEVICES"])
        raise RuntimeError("explicit stop before native pool construction")

    monkeypatch.setattr(host, "require_subset_lease", lease)
    monkeypatch.setattr(host, "VectorPool", pool)
    with pytest.raises(RuntimeError, match="stop before native"):
        host.serve(config("training128"), tmp_path / "socket", tmp_path)
    assert seen == ["GPU-explicit-coordinator-double", ""]
    assert json.loads((tmp_path / "receipt.json").read_text())["status"] == "failed"


def test_cpu_cleanup_error_seals_failed_receipt(monkeypatch, tmp_path):
    from abc_bench import fasttd3_vector_cpu

    class PoolDouble:
        vector_steps = 0

        def __init__(self):
            self.worker_cleanup = [
                {"rank": 0, "pid": 101, "exitcode": 7, "alive": False}
            ]

        def describe(self):
            raise RuntimeError("explicit primary double failure")

        def close(self):
            raise RuntimeError("explicit cleanup failure")

    output = tmp_path / "new"
    with pytest.raises(RuntimeError, match="cleanup failure"):
        fasttd3_vector_cpu.run(config(), output, pool_factory=lambda *a: PoolDouble())
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "failed"
    assert "cleanup failure" in receipt["cleanup_error"]
    assert receipt["worker_cleanup"][0]["exitcode"] == 7
