"""Explicit CPU doubles for the captured actor boundary; no simulator or GPU."""

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from abc_bench import fasttd3_captured_host as host
from abc_bench.fasttd3_captured_host import SCHEMA, CapturedFastTask, dispatch
from abc_bench.fasttd3_vector_host import World


class TaskDouble:
    def __init__(self, *, verified=True):
        self.env = SimpleNamespace()
        self.reset_verified = verified
        self.episode_steps = 0
        self.reset_proofs = [
            {"path": "/explicit-reset-double.json", "sha256": "a" * 64, "bytes": 1}
        ]
        self.description = {
            "reset_policy": {"heldout_seed": 9100, "allowed_seeds": [9100]}
        }
        self.episodes = []
        self.calls = []
        self.closed = False
        self.terminated, self.truncated = False, False
        self.close_failure = False

    def reset(self, seed):
        self.calls.append(("reset", seed))
        self.episode_steps = 0
        self.terminated, self.truncated = False, False

    def step(self, action):
        if self.terminated or self.truncated:
            raise ValueError("Original terminal scene is preserved")
        self.calls.append(("step", action))
        self.episode_steps += 1
        if getattr(self, "end", False):
            self.terminated = not getattr(self, "timeout", False)
            self.truncated = not self.terminated
            self.episodes.append(
                {
                    "seed": 9100,
                    "completed": True,
                    "success": self.terminated,
                    "controls": self.episode_steps,
                    "video": {"path": "/explicit-video-double.mp4"},
                }
            )
        return {
            "action_clipping": [],
            "action_applied14": action,
            "reward": float(self.terminated),
            "terminated": self.terminated,
            "truncated": self.truncated,
        }

    def close(self):
        self.calls.append(("close",))
        self.closed = True
        if self.close_failure:
            raise RuntimeError("Explicit native cleanup failure")


@pytest.fixture
def boundary(monkeypatch):
    low = np.array([-2] * 6 + [0] + [-2] * 6 + [0], dtype=float)
    high = np.array([2] * 6 + [1] + [2] * 6 + [1], dtype=float)
    monkeypatch.setattr(
        World,
        "description",
        lambda self: {
            "num_obs": 216,
            "num_actions": 14,
            "action_schema": {"policy_low": low.tolist(), "policy_high": high.tolist()},
        },
    )
    monkeypatch.setattr(
        World, "observation", lambda self: [float(self.task.episode_steps)] * 216
    )
    scene = SimpleNamespace(
        randomization={"explicit": "double"},
        episode={
            "seed": 9100,
            "task": "put_plastic_bottles_in_bin",
            "prompt": "sim throw plastic bottles in bin",
        },
    )
    raw = TaskDouble()
    return CapturedFastTask(raw, scene), raw, scene, low, high


def test_reset_reuses_verified_initial_world_without_sampler_or_render(boundary):
    task, raw, _, _, _ = boundary
    result = task.reset([9100])
    assert raw.calls == []
    assert result["actual_seeds"] == [9100]
    assert np.asarray(result["obs"]).shape == (1, 216)
    task.reset([9100])
    assert raw.calls == [("reset", 9100)]


def test_requires_public_rgb_reset_gate(boundary):
    _, _, scene, _, _ = boundary
    with pytest.raises(RuntimeError, match="live RGB"):
        CapturedFastTask(TaskDouble(verified=False), scene)


@pytest.mark.parametrize("bad", [[9101], [True], 9100, [9100, 9100]])
def test_reset_rejects_other_scene_identity(boundary, bad):
    with pytest.raises(ValueError, match="single captured"):
        boundary[0].reset(bad)


def test_same_normalized_float32_affine_action_boundary(boundary):
    task, raw, _, low, high = boundary
    task.reset([9100])
    requested = np.linspace(-1.5, 1.5, 14, dtype=np.float32)[None]
    result = task.step(requested)
    expected = (low + (np.clip(requested[0], -1, 1) + 1) * 0.5 * (high - low)).astype(
        np.float32
    )
    np.testing.assert_array_equal(
        np.asarray(raw.calls[-1][1], dtype=np.float32), expected
    )
    assert result["physical_actions"] == [expected.tolist()]
    assert result["autoresets"] == [None]


@pytest.mark.parametrize("timeout", [False, True])
def test_native_terminal_timeout_keeps_scene_and_video(boundary, timeout):
    task, raw, _, _, _ = boundary
    task.reset([9100])
    raw.end, raw.timeout = True, timeout
    result = task.step([[0] * 14])
    assert result["terminated"] == [not timeout]
    assert result["truncated"] == [timeout]
    assert result["obs"] == result["terminal_obs"] == [[1.0] * 216]
    assert result["episodes"][0]["video"]["path"] == "/explicit-video-double.mp4"
    assert not any(call[0] == "reset" for call in raw.calls)
    with pytest.raises(ValueError, match="terminal scene"):
        task.step([[0] * 14])


@pytest.mark.parametrize(
    "bad", [[[float("nan")] * 14], [[0] * 13], [[1e100] * 14], [[0] * 14] * 2]
)
def test_bad_action_never_reaches_native_control(boundary, bad):
    task, raw, _, _, _ = boundary
    task.reset([9100])
    with pytest.raises(ValueError):
        task.step(bad)
    assert raw.calls == []


def test_step_requires_explicit_client_reset(boundary):
    with pytest.raises(RuntimeError, match="reset"):
        boundary[0].step([[0] * 14])


def test_cleanup_error_propagates(boundary):
    task, raw, _, _, _ = boundary
    raw.close_failure = True
    with pytest.raises(RuntimeError, match="cleanup"):
        task.close()
    assert task.closed and raw.closed


def test_rpc_does_not_admit_extra_operations_or_fields(boundary):
    task = boundary[0]
    with pytest.raises(ValueError, match="fields"):
        dispatch(
            task,
            {
                "schema": SCHEMA,
                "id": 1,
                "operation": "step",
                "actions": [[0] * 14],
                "fixture": "success",
            },
        )
    with pytest.raises(ValueError, match="schema"):
        dispatch(task, {"schema": "wrong", "id": 1, "operation": "describe"})


@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_wrapper_construction_failure_closes_original_task_and_records_failure(
    monkeypatch, tmp_path, cleanup_failure
):
    output = tmp_path / "host"
    output.mkdir()
    raw = TaskDouble()
    raw.close_failure = cleanup_failure
    scene = SimpleNamespace()
    config = {"host": {"reset_profile": {}}, "timeouts": {"host_wall_s": 60}}
    monkeypatch.setattr(host, "load_config", lambda _: config)
    monkeypatch.setattr(host, "require_subset_lease", lambda: None)
    monkeypatch.setattr(host, "artifact", lambda _: {"explicit": "config pin double"})
    monkeypatch.setitem(
        sys.modules,
        "abc_bench.captured_reset",
        SimpleNamespace(
            CapturedScene=lambda _: scene, make_captured_env=lambda *_: raw.env
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "abc_bench.expoft_host",
        SimpleNamespace(CapturedExpoTask=lambda *_: raw),
    )
    monkeypatch.setattr(
        host,
        "CapturedFastTask",
        lambda *_: (_ for _ in ()).throw(ValueError("Explicit wrapper shape failure")),
    )
    with (
        tempfile.TemporaryDirectory(
            prefix="ft-", dir="/home/user/aditya/RL/builds"
        ) as private,
        pytest.raises((ValueError, RuntimeError)),
    ):
        host.serve(tmp_path / "config.json", Path(private) / "s.sock", output)
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "failed"
    assert raw.closed and raw.calls == [("close",)]
    assert ("cleanup_error" in receipt) is cleanup_failure


def test_missing_lease_rejects_before_any_native_factory_or_output_mutation(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(host, "load_config", lambda _: {"explicit": "config double"})
    monkeypatch.setattr(
        host,
        "require_subset_lease",
        lambda: (_ for _ in ()).throw(RuntimeError("No canonical lease")),
    )
    output = tmp_path / "not-created"
    with pytest.raises(RuntimeError, match="lease"):
        host.serve(tmp_path / "config.json", tmp_path / "s.sock", output)
    assert not output.exists()
