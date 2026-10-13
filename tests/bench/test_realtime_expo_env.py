"""CPU contract doubles, never native physics, cameras, a policy, or training."""

import asyncio
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from abc_bench import realtime_expo_env as boundary


class WorldDouble:
    single_timestep_action_dim = 14
    _control_decimation = 17

    def __init__(self):
        self.robot_names = ["left", "right"]
        self.model = SimpleNamespace(opt=SimpleNamespace(timestep=0.002))
        self.data = SimpleNamespace(time=0.0)
        self.cur_step = 0
        self.initial = np.arange(14, dtype=np.float32) / 10
        self.initial[[6, 13]] = 0.5
        self.actions, self.reset_seeds = [], []
        self.closed = False
        self.success_at = None
        self.horizon = 2

    def get_init_q(self):
        return self.initial.copy()

    def get_state(self):
        return self.initial.copy()

    def get_obs(self):
        return {
            "state": self.get_state(),
            "images": {
                camera: np.broadcast_to(
                    np.array([i, i + 10, i + 20], np.uint8)[:, None, None],
                    (3, 168, 224),
                ).copy()
                for i, camera in enumerate(boundary.CAMERAS)
            },
            "masks": dict.fromkeys(boundary.CAMERAS, True),
            "camera_timestamps": dict.fromkeys(boundary.CAMERAS, self.data.time),
        }

    def evaluate_task(self):
        return SimpleNamespace(
            to_info=lambda **_: {
                "num_active_bottles": 5,
                "success": False,
                "reward": 0.0,
            }
        )

    def step(self, action, *, render_obs):
        assert render_obs is False
        self.actions.append(action.copy())
        self.cur_step += 1
        self.data.time += 0.034
        success = self.cur_step == self.success_at
        truncated = self.cur_step >= self.horizon and not success
        return (
            None,
            1.0 if success else 0.4,
            success,
            truncated,
            {
                "task_success": success,
                "task_eval": {"num_bottles_in_bin": 5 if success else 2},
            },
        )

    def close(self):
        self.closed = True


@pytest.fixture
def fixture(monkeypatch, tmp_path):
    world = WorldDouble()
    calls = []
    monkeypatch.setattr(boundary, "require_subset_lease", lambda: calls.append("lease"))

    def factory(config):
        calls.append(("factory", config))
        return world

    def reset(env, seed):
        assert env is world
        world.reset_seeds.append(seed)
        world.cur_step, world.data.time = 0, 0.0
        return world.get_obs(), {"seed": seed}, seed, [{"seed": seed}]

    monkeypatch.setattr(boundary, "make_original_env", factory)
    monkeypatch.setattr(boundary, "reset_world", reset)
    kwargs = {
        "episode_seeds": [9030, 9031],
        "example_action": world.initial[None],
        "event_log_path": str(tmp_path / "events.jsonl"),
    }
    yield world, kwargs, calls
    if boundary.ABCYamEnv._active is not None:
        boundary.ABCYamEnv._active.close()


def test_original_client_sequence_startup_prefix_actions_and_terminal_noop(fixture):
    world, kwargs, calls = fixture
    env = boundary.ABCYamEnv(**kwargs, env_type="abc_yam", env=None)
    assert calls[0] == "lease" and calls[1][0] == "factory"
    reset = env.reset()
    assert reset["state"].dtype == np.float32
    assert reset["top_image"].shape == (168, 224, 3)
    assert reset["left_wrist_image"][0, 0].tolist() == [1, 11, 21]
    assert reset["right_wrist_image"][0, 0].tolist() == [2, 12, 22]
    # Author train_pi_robo: boot step, observe/info, planned step, terminal
    # observe/info, an extra dispatch, then reset. The adapter adds no reordering.
    assert env.step(world.initial)["physics_advanced"] is True
    before = world.data.time
    env.get_observation()
    assert world.data.time == before
    assert env.get_info_for_step() == (False, False, 0.0, 1.0)
    proposed = np.arange(14, dtype=np.float64) + 20
    assert np.array_equal(
        env.step(proposed)["executed_action"], proposed.astype(np.float32)
    )
    assert world.actions[-1].tolist() == proposed.astype(np.float32).tolist()
    env.get_observation()
    assert env.get_info_for_step() == (True, False, 0.0, 0.0)
    before = world.data.time
    assert env.step(proposed)["physics_advanced"] is False
    assert world.data.time == before and len(world.actions) == 2
    env.reset()
    # Later episode starts directly with the policy, not another startup hold.
    env.step(proposed)
    assert len(world.actions) == 3 and world.reset_seeds == [9030, 9031]
    env.close()
    events = [
        json.loads(line)
        for line in Path(kwargs["event_log_path"]).read_text().splitlines()
    ]
    steps = [event for event in events if event["operation"] == "step"]
    assert [event["actual_controls"] for event in steps] == [1, 2, 3]
    assert all(event["physics_steps_advanced"] == 17 for event in steps)
    terminal = next(event for event in events if event["operation"] == "terminal_noop")
    assert terminal["actual_controls"] == 2 and terminal["physics_advanced"] is False
    assert world.closed
    assert events[-1]["operation"] == "closed"


def test_native_success_maps_sparse_reward_without_fraction_shaping(fixture):
    world, kwargs, _ = fixture
    env = boundary.ABCYamEnv(**kwargs)
    env.reset()
    world.success_at = 1
    env.step(world.initial)
    assert env.get_info_for_step() == (True, True, 1.0, 0.0)


def test_terminal_video_closes_after_final_observation(fixture):
    world, kwargs, _ = fixture
    env = boundary.ABCYamEnv(**kwargs)
    with pytest.raises(RuntimeError, match="Reset ABC"):
        env.get_info_for_step()
    env.reset()
    frames, closes = [], []
    env.writer = SimpleNamespace(
        append_data=lambda frame: frames.append(frame.copy()),
        close=lambda: closes.append(True),
    )
    world.success_at = 1
    env.step(world.initial)
    env.get_observation()
    assert len(frames) == 1 and not closes
    env.get_info_for_step()
    env.get_info_for_step()
    env.close()
    assert closes == [True]


def test_lease_refusal_precedes_factory_and_outputs(fixture, monkeypatch):
    _, kwargs, calls = fixture

    def refuse():
        raise RuntimeError("No active ABC subset coordinator")

    monkeypatch.setattr(boundary, "require_subset_lease", refuse)
    with pytest.raises(RuntimeError, match="No active ABC subset coordinator"):
        boundary.ABCYamEnv(**kwargs)
    assert not calls
    assert not Path(kwargs["event_log_path"]).exists()


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"env_usage": "eval"}, "TRAIN resets only"),
        ({"control_hz": 30.0}, "1/0.034"),
        ({"episode_seeds": [9100]}, "held-out"),
        ({"episode_seeds": [9030, 9030]}, "unique TRAIN"),
        ({"max_episode_steps": 3541}, "native controls"),
        ({"example_action": np.zeros((1, 7))}, "absolute14"),
    ],
)
def test_invalid_task_contract_fails_before_native(fixture, changes, match):
    _, kwargs, calls = fixture
    with pytest.raises(ValueError, match=match):
        boundary.ABCYamEnv(**(kwargs | changes))
    assert not calls


def test_wrong_declared_or_dispatched_startup_hold_fails(fixture):
    world, kwargs, _ = fixture
    with pytest.raises(ValueError, match="native initial hold"):
        boundary.ABCYamEnv(**(kwargs | {"example_action": np.zeros(14)}))
    assert world.closed
    kwargs["event_log_path"] += ".second"
    env = boundary.ABCYamEnv(**kwargs)
    env.reset()
    with pytest.raises(ValueError, match="First author command"):
        env.step(np.zeros(14))
    assert not world.actions


def test_bad_camera_timestamp_never_returns_synthetic_observation(fixture):
    world, kwargs, _ = fixture
    env = boundary.ABCYamEnv(**kwargs)
    env.reset()
    original = world.get_obs

    def bad():
        raw = original()
        raw["camera_timestamps"]["left"] += 0.01
        return raw

    world.get_obs = bad
    with pytest.raises(ValueError, match="synchronized"):
        env.get_observation()


def test_seed_exhaustion_and_second_world_refuse(fixture):
    _, kwargs, _ = fixture
    env = boundary.ABCYamEnv(**kwargs)
    with pytest.raises(RuntimeError, match="One live ABC world"):
        boundary.ABCYamEnv(
            **(kwargs | {"event_log_path": kwargs["event_log_path"] + ".other"})
        )
    env.reset()
    env.reset()
    with pytest.raises(RuntimeError, match="exhausted"):
        env.reset()
    env.close()
    env.close()
    with pytest.raises(RuntimeError, match="closed"):
        env.get_observation()


def test_pristine_author_client_factory_and_rpc_flow(fixture, monkeypatch, caplog):
    """Opt-in real author handler with only its native world replaced by a double."""
    client_path = os.environ.get("EXPO_AUTHOR_CLIENT")
    if client_path is None:
        pytest.skip("Set EXPO_AUTHOR_CLIENT to the pristine author run_client.py")
    import ml_collections
    import websockets.exceptions
    from openpi_client import msgpack_numpy

    world, kwargs, _ = fixture
    spec = importlib.util.spec_from_file_location(
        "_author_expo_client_contract", client_path
    )
    client = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, client)
    spec.loader.exec_module(client)
    config = ml_collections.ConfigDict(
        kwargs
        | {
            "env_name": "abc_rt",
            "env_type": "abc_yam",
            "language_instruction": boundary.PROMPT,
            "env": boundary.ABCYamEnv,
        }
    )
    monkeypatch.setattr(client, "load_task_config", lambda _: config)
    env_id = "abc_rt_train_1"
    requests = [
        {"operation": "create_env", "env_usage": "train"},
        {"operation": "reset", "env_id": env_id},
        {"operation": "step", "env_id": env_id, "action": world.initial.tolist()},
        {"operation": "get_observation", "env_id": env_id},
        # Demonstrate inherited NaN sanitation; the env cannot detect its origin.
        {"operation": "step", "env_id": env_id, "action": [float("nan")] * 14},
        {"operation": "get_observation", "env_id": env_id},
        {"operation": "step", "env_id": env_id, "action": [0.0] * 14},
    ]

    class SocketDouble:
        remote_address = "CPU-double"
        transport = SimpleNamespace(get_extra_info=lambda _: None)

        def __init__(self):
            self.requests = iter(requests)
            self.responses = []

        async def recv(self):
            try:
                return msgpack_numpy.packb(next(self.requests))
            except StopIteration:
                raise websockets.exceptions.ConnectionClosedOK(None, None) from None

        async def send(self, frame):
            self.responses.append(msgpack_numpy.unpackb(frame))

    socket = SocketDouble()
    asyncio.run(client._handle_environment_request(socket))
    assert len(socket.responses) == len(requests)
    assert all(response["status"] == "success" for response in socket.responses)
    assert socket.responses[3]["observation"]["state"].shape == (14,)
    assert socket.responses[5]["done"] is True
    assert socket.responses[5]["reward"] == 0.0
    assert socket.responses[5]["mask"] == 0.0
    assert len(world.actions) == 2 and world.data.time == 0.068
    assert np.array_equal(world.actions[1], np.zeros(14, np.float32))
    assert "NaN/Inf; replacing with zeros" in caplog.text
    assert not any(name == "droid" or name.startswith("droid.") for name in sys.modules)
    assert not any(name in sys.modules for name in ("torch", "jax", "mujoco", "warp"))
    client._env_storage[env_id].close()
