"""Explicit CPU API doubles; these tests do not simulate physics or run a VLA."""

import base64
import copy
import json
import math
import os
import signal
import socket
import struct
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from abc_bench import expoft_host as host


class State14(list):
    dtype = "float32"

    def tolist(self):
        return list(self)


class RGBDouble:
    """Three constant CHW planes with an explicit, testable HWC transpose."""

    shape, dtype = (3, 168, 224), "uint8"

    def __init__(self, channels):
        self.channels = bytes(channels)

    def transpose(self, *axes):
        assert axes == (1, 2, 0)
        return SimpleNamespace(tobytes=lambda *, order: self.channels * (168 * 224))


class OriginalWorldDouble:
    """Original public API double; all observations are synthetic test fixtures."""

    single_timestep_action_dim = 14
    _control_decimation = 17
    prompt = "sim throw plastic bottles in bin"

    def __init__(self, *, horizon=2, success_at=None):
        self.robot_names = ["left", "right"]
        self.model = SimpleNamespace(
            opt=SimpleNamespace(timestep=0.002),
            joint=lambda _: SimpleNamespace(range=[-2.0, 4.0]),
        )
        self.horizon, self.success_at = horizon, success_at
        self.cur_step, self.seed, self.closed, self.forgotten = 0, 0, 0, 0
        self.actions, self.resets = [], []
        self.state = State14([0.0] * 14)
        self.randomization = {
            "seed": 20260511,
            "object_states": {},
            "scale_states": {},
            "metadata": {"bottle_count": 5, "bottle_variants": ["bottle_0"] * 5},
        }
        self.images = {
            name: RGBDouble((i + 1, 100 + i, 200 + i))
            for i, name in enumerate(host.CAMERAS)
        }

    def forget_arm_state(self):
        self.forgotten += 1

    def reset(self, *, seed, options, randomize):
        assert randomize is True
        self.resets.append((seed, copy.deepcopy(options)))
        self.cur_step, self.seed = 0, seed
        self.state = State14([0.0] * 14)
        state = copy.deepcopy(self.randomization)
        state["seed"] = seed
        return self.get_obs(), {"randomization": SimpleNamespace(to_dict=lambda: state)}

    def evaluation(self):
        success = self.success_at is not None and self.cur_step >= self.success_at
        return {
            "reward": float(success),
            "success": success,
            "ever_success": success,
            "num_active_bottles": 5,
            "num_bottles_in_bin": 5 if success else 0,
        }

    def evaluate_task(self):
        return SimpleNamespace(to_info=lambda **_: self.evaluation())

    def get_state(self):
        return self.state

    def capture_state(self):
        return {"time": self.cur_step * 0.034}

    def get_obs(self):
        return {
            "state": self.get_state(),
            "images": self.images,
            "masks": {name: True for name in host.CAMERAS},
            "camera_timestamps": {name: self.cur_step * 0.034 for name in host.CAMERAS},
        }

    def step(self, action, *, render_obs):
        assert render_obs is False
        self.actions.append(list(action))
        self.state = State14(action)
        self.cur_step += 1
        evaluation = self.evaluation()
        terminated = evaluation["success"]
        truncated = self.cur_step >= self.horizon and not terminated
        return (
            None,
            evaluation["reward"],
            terminated,
            truncated,
            {"task_eval": evaluation},
        )

    def close(self):
        self.closed += 1


@pytest.fixture
def configuration(tmp_path):
    world = OriginalWorldDouble()
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "task": "put_plastic_bottles_in_bin",
                "worlds": [
                    {"world_seed": 20260511, "randomization": world.randomization}
                ],
            }
        )
    )
    config = {
        "schema_version": 1,
        "stage": "evaluation",
        "host": {
            "num_envs": 1,
            "initial_seed": 20260511,
            "allowed_seeds": [20260511],
            "max_episode_steps": 2,
            "save_video": False,
            "reference": {
                "summary": str(summary),
                "sha256": host.artifact(summary)["sha256"],
                "world_index": 0,
            },
        },
        "timeouts": {"host_wall_s": 10, "rpc_s": 2, "max_requests": 20},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path, config


def make_task(tmp_path, configuration, *, success_at=None):
    _, config = configuration
    world = OriginalWorldDouble(success_at=success_at)
    return world, host.ExpoTask(world, config, tmp_path)


def test_original_state_rgb_and_chronology(tmp_path, configuration):
    world, task = make_task(tmp_path, configuration)
    result = task.observe_rgb()
    assert result["state14"] == [0.0] * 14
    assert result["control_step"] == result["physics_step"] == 0
    assert task.description["privileged_state"] is False
    assert task.description["autoreset"] is False
    assert task.description["control_hz"] == pytest.approx(29.41176470588235)
    assert task.description["action_schema"]["names"][6::7] == [
        "left_gripper",
        "right_gripper",
    ]
    for i, name in enumerate(host.CAMERAS):
        camera = result["cameras"][name]
        raw = base64.b64decode(camera["data_base64"], validate=True)
        assert camera["shape"] == [168, 224, 3] and camera["dtype"] == "uint8"
        assert raw == bytes((i + 1, 100 + i, 200 + i)) * (168 * 224)
    result = task.step([0.123456789] * 14)
    actual = struct.unpack("<f", struct.pack("<f", 0.123456789))[0]
    assert result["action_applied14"] == world.actions[-1] == [actual] * 14
    assert result["action_clipping"] == []
    assert result["control_step"] == 1 and result["physics_step"] == 17
    assert result["simulation_time_s"] == pytest.approx(0.034)
    task.close()


def test_physical_clipping_records_exact_applied_controls(tmp_path, configuration):
    world, task = make_task(tmp_path, configuration)
    action = [-3.0, 8.0] + [0.0] * 4 + [-0.5] + [0.0] * 6 + [1.5]
    result = task.step(action)
    assert result["action_requested14"] == action
    assert result["action_applied14"] == world.actions[-1]
    assert [item["index"] for item in result["action_clipping"]] == [0, 1, 6, 13]
    assert [world.actions[-1][i] for i in (0, 1, 6, 13)] == [-2.0, 4.0, 0.0, 1.0]
    assert task.clipped_actions == 4
    task.close()


@pytest.mark.parametrize(
    "success_at,terminated,truncated", [(2, True, False), (None, False, True)]
)
def test_terminal_scene_preserved_until_explicit_reset(
    tmp_path, configuration, success_at, terminated, truncated
):
    world, task = make_task(tmp_path, configuration, success_at=success_at)
    task.step([0.1] * 14)
    final = task.step([0.5] * 14)
    assert final["terminated"] is terminated and final["truncated"] is truncated
    assert world.resets == [(20260511, None)]
    assert task.observe_rgb()["state14"] == [0.5] * 14
    with pytest.raises(ValueError, match="final scene is preserved"):
        task.step([0.0] * 14)
    assert len(world.actions) == 2 and task.observe_rgb()["control_step"] == 2
    assert task.episodes[0]["completed"] is True
    assert task.episodes[0]["success"] is terminated
    assert task.reset(20260511)["control_step"] == 0
    assert len(world.resets) == 2
    task.close()
    task.close()
    assert world.closed == 1


@pytest.mark.parametrize(
    "action", [[], [[0.0] * 14], [0.0] * 13, [True] * 14, [math.nan] * 14]
)
def test_invalid_actions_do_not_reach_original_step(tmp_path, configuration, action):
    world, task = make_task(tmp_path, configuration)
    with pytest.raises(ValueError):
        task.step(action)
    assert world.actions == []
    task.close()


def test_training_and_evaluation_domains_are_separate(tmp_path, configuration):
    path, config = configuration
    assert host.load_config(path)["stage"] == "evaluation"
    world, task = make_task(tmp_path, configuration)
    with pytest.raises(ValueError, match="domain"):
        task.reset(9030)
    assert world.resets == [(20260511, None)]
    task.close()
    config["stage"] = "train"
    config["host"].update(initial_seed=9030, allowed_seeds=[9030, 9031])
    path.write_text(json.dumps(config))
    config = host.load_config(path)
    world = OriginalWorldDouble()
    task = host.ExpoTask(world, config, tmp_path)
    assert world.resets == [(9030, host.RESET_OPTIONS)]
    assert (
        task.description["reset_policy"]["distribution"]
        == "original fixed-five diagnostic"
    )
    assert task.reset(9031)["scene_match"] == "fixed_five_train"
    with pytest.raises(ValueError, match="domain"):
        task.reset(20260511)
    task.close()
    config["host"]["allowed_seeds"].append(20260511)
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="held-out"):
        host.load_config(path)


def test_reference_pin_and_native_exact_state_mismatch_rejected(
    tmp_path, configuration
):
    path, config = configuration
    world = OriginalWorldDouble()
    world.randomization["scale_states"] = {"bin_joint": 2.0}
    with pytest.raises(RuntimeError, match="pinned baseline RandomizationState"):
        host.ExpoTask(world, config, tmp_path)
    summary = config["host"]["reference"]["summary"]
    with open(summary, "a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="SHA256"):
        host.load_config(path)


@pytest.mark.parametrize("change", ["task", "world_seed"])
def test_reference_task_and_world_seed_must_match(configuration, change):
    path, config = configuration
    reference = config["host"]["reference"]
    summary_path = Path(reference["summary"])
    summary = json.loads(summary_path.read_text())
    if change == "task":
        summary["task"] = "another_task"
    else:
        summary["worlds"][0]["world_seed"] = 20260512
    summary_path.write_text(json.dumps(summary))
    reference["sha256"] = host.artifact(summary_path)["sha256"]
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="bottle task|world seed"):
        host.load_config(path)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("host", "num_envs", 2),
        ("host", "max_episode_steps", 3541),
        ("host", "initial_seed", True),
        ("host", "allowed_seeds", [20260511, 20260511]),
        ("host", "allowed_seeds", [20260511, 20260512]),
        ("host", "save_video", 1),
        ("timeouts", "rpc_s", 0),
        ("timeouts", "max_requests", True),
    ],
)
def test_unbounded_or_ambiguous_config_rejected(configuration, section, key, value):
    path, config = configuration
    config[section][key] = value
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        host.load_config(path)


def test_bad_camera_and_chronology_fail_explicitly(tmp_path, configuration):
    world, task = make_task(tmp_path, configuration)
    world.images["top"].shape = (3, 224, 168)
    with pytest.raises(RuntimeError, match="Camera top"):
        task.observe_rgb()
    world.images["top"].shape = host.CAMERA_CHW
    world.cur_step += 1
    with pytest.raises(RuntimeError, match="chronology"):
        task.observe_rgb()
    task.close()


@contextmanager
def socket_pair():
    sender, receiver = socket.socketpair()
    with sender, receiver:
        yield sender, receiver


def test_strict_operations_and_transport(tmp_path, configuration, monkeypatch):
    _, task = make_task(tmp_path, configuration)
    request = {"schema": host.SCHEMA, "id": 1, "operation": "describe"}
    response, stop = host.dispatch(task, request)
    assert response["ok"] and not stop and response["id"] == 1
    for update in (
        {"schema": "other"},
        {"id": True},
        {"extra": 1},
        {"operation": "reset"},
    ):
        with pytest.raises(ValueError):
            host.dispatch(task, {**request, **update})
    assert host.dispatch(task, {**request, "operation": "close"})[1]
    with socket_pair() as (sender, receiver):
        sender.sendall(b'{"id":1}\n')
        assert json.loads(host.receive_frame(receiver, time.monotonic() + 1)) == {
            "id": 1
        }
    with socket_pair() as (sender, receiver):
        sender.sendall(b"{}\n{}\n")
        with pytest.raises(ValueError, match="One JSON"):
            host.receive_frame(receiver, time.monotonic() + 1)
    monkeypatch.setattr(host, "MAX_FRAME_BYTES", 8)
    with socket_pair() as (sender, receiver):
        sender.sendall(b"123456789\n")
        with pytest.raises(ValueError, match="4 MiB"):
            host.receive_frame(receiver, time.monotonic() + 1)
    with socket_pair() as (_, receiver), pytest.raises(TimeoutError, match="deadline"):
        host.receive_frame(receiver, time.monotonic() - 1)
    task.close()


def test_lease_precedes_factory_and_failure_receipt(
    tmp_path, configuration, monkeypatch
):
    path, _ = configuration
    events = []

    def reject():
        events.append("lease")
        raise RuntimeError("No live canonical lease")

    monkeypatch.setattr(host, "require_subset_lease", reject)
    monkeypatch.setattr(
        host, "make_original_env", lambda _: events.append("native_factory")
    )
    out = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="lease"):
        host.serve(path, tmp_path / "host.sock", out)
    assert events == ["lease"]
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["metrics"] == {}
    assert "No live canonical lease" in receipt["blockers"][0]


def test_socket_mode_checked_before_factory(
    tmp_path, configuration, monkeypatch, short_private_socket
):
    path, _ = configuration
    monkeypatch.setattr(host, "require_subset_lease", lambda: None)
    monkeypatch.setattr(
        host, "make_original_env", lambda _: pytest.fail("native factory must not run")
    )
    short_private_socket.parent.chmod(0o755)
    with pytest.raises(ValueError, match="0700"):
        host.serve(path, short_private_socket, tmp_path / "failed")


@pytest.fixture
def short_private_socket():
    # Unix paths are capped at 107 bytes; keep fixture sockets in the workspace.
    repository = Path(
        os.environ.get("ABC_BENCH_REPO", str(Path(__file__).resolve().parents[2]))
    )
    builds = repository.parent / "builds"
    builds.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=builds, prefix="exrpc-") as directory:
        yield Path(directory) / "host.sock"


def test_full_cpu_rpc_lifecycle_preserves_terminal_world(
    tmp_path, configuration, monkeypatch, short_private_socket
):
    path, _ = configuration
    world = OriginalWorldDouble(success_at=2)
    monkeypatch.setattr(host, "require_subset_lease", lambda: None)
    monkeypatch.setattr(host, "make_original_env", lambda _: world)
    monkeypatch.setattr(
        host, "native_sources", lambda _: {"fixture": "explicit CPU double"}
    )
    socket_path, output = short_private_socket, tmp_path / "host"
    responses, failures = [], []

    def client():
        try:
            deadline = time.monotonic() + 3
            while not (output / "ready.json").is_file():
                if time.monotonic() >= deadline:
                    raise TimeoutError("CPU fixture host did not start")
                time.sleep(0.005)
            for i, (operation, parameters) in enumerate(
                [
                    ("describe", {}),
                    ("step", {"action": [0.5] * 14}),
                    ("step", {"action": [0.5] * 14}),
                    ("step", {"action": [0.5] * 14}),
                    ("observe_rgb", {}),
                    ("close", {}),
                ],
                start=1,
            ):
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                    connection.settimeout(2)
                    connection.connect(str(socket_path))
                    request = {
                        "schema": host.SCHEMA,
                        "id": i,
                        "operation": operation,
                        **parameters,
                    }
                    connection.sendall(json.dumps(request).encode() + b"\n")
                    responses.append(
                        json.loads(host.receive_frame(connection, time.monotonic() + 2))
                    )
        except (OSError, ValueError, RuntimeError, AssertionError) as error:
            failures.append(error)

    thread = threading.Thread(target=client, daemon=True)
    thread.start()
    host.serve(path, socket_path, output)
    thread.join(timeout=3)
    assert not thread.is_alive() and failures == []
    assert responses[3]["ok"] is False and responses[3]["error"]["type"] == "ValueError"
    assert responses[4]["ok"] and responses[4]["result"]["control_step"] == 2
    assert responses[4]["result"]["terminated"] is True
    assert len(world.resets) == 1 and world.closed == 1
    assert not socket_path.exists()
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "completed" and receipt["metrics"]["controls"] == 2
    assert receipt["metrics"]["completed_episodes"] == 1


def test_wall_deadline_closes_only_owned_world_and_socket(
    tmp_path, configuration, monkeypatch, short_private_socket
):
    path, config = configuration
    config["timeouts"]["host_wall_s"] = 1
    path.write_text(json.dumps(config))
    world = OriginalWorldDouble()
    monkeypatch.setattr(host, "require_subset_lease", lambda: None)
    monkeypatch.setattr(host, "make_original_env", lambda _: world)
    monkeypatch.setattr(
        host, "native_sources", lambda _: {"fixture": "explicit CPU double"}
    )
    output = tmp_path / "deadline"
    original_handler = signal.getsignal(signal.SIGTERM)
    with pytest.raises(host.HostDeadlineError, match="deadline"):
        host.serve(path, short_private_socket, output)
    assert world.closed == 1 and not short_private_socket.exists()
    assert signal.getsignal(signal.SIGTERM) == original_handler
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["metrics"]["controls"] == 0
    assert receipt["elapsed_seconds"] < 3


def test_cancellation_records_receipt_and_closes_owned_world(
    tmp_path, configuration, monkeypatch, short_private_socket
):
    path, _ = configuration
    world = OriginalWorldDouble()
    monkeypatch.setattr(host, "require_subset_lease", lambda: None)
    monkeypatch.setattr(host, "make_original_env", lambda _: world)

    def cancel(_):
        raise KeyboardInterrupt("Explicit CPU cancellation fixture")

    monkeypatch.setattr(host, "native_sources", cancel)
    output = tmp_path / "cancelled"
    with pytest.raises(KeyboardInterrupt, match="fixture"):
        host.serve(path, short_private_socket, output)
    assert world.closed == 1 and not short_private_socket.exists()
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "cancelled" and receipt["metrics"]["controls"] == 0


def test_rpc_deadline_interrupts_native_double_and_records_failure(
    tmp_path, configuration, monkeypatch, short_private_socket
):
    path, config = configuration
    config["timeouts"]["rpc_s"] = 1
    path.write_text(json.dumps(config))
    world = OriginalWorldDouble()
    monkeypatch.setattr(host, "require_subset_lease", lambda: None)
    monkeypatch.setattr(host, "make_original_env", lambda _: world)
    monkeypatch.setattr(
        host, "native_sources", lambda _: {"fixture": "explicit CPU double"}
    )

    def slow_step(*_args, **_kwargs):
        time.sleep(3)
        raise AssertionError("The RPC watchdog must interrupt this synthetic stall")

    monkeypatch.setattr(world, "step", slow_step)
    output = tmp_path / "rpc-deadline"
    results, errors = [], []

    def client():
        try:
            deadline = time.monotonic() + 3
            while not (output / "ready.json").is_file():
                if time.monotonic() >= deadline:
                    raise TimeoutError("CPU fixture host did not start")
                time.sleep(0.005)
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(3)
                connection.connect(str(short_private_socket))
                request = {
                    "schema": host.SCHEMA,
                    "id": 1,
                    "operation": "step",
                    "action": [0.5] * 14,
                }
                connection.sendall(json.dumps(request).encode() + b"\n")
                results.append(
                    json.loads(host.receive_frame(connection, time.monotonic() + 3))
                )
        except (OSError, ValueError, RuntimeError, AssertionError) as error:
            errors.append(error)

    thread = threading.Thread(target=client, daemon=True)
    thread.start()
    with pytest.raises(RuntimeError, match="recorded failure"):
        host.serve(path, short_private_socket, output)
    thread.join(timeout=3)
    assert not thread.is_alive() and errors == []
    assert results[0]["ok"] is False and "deadline" in results[0]["error"]["message"]
    assert world.closed == 1 and not short_private_socket.exists()
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["metrics"]["controls"] == 0
