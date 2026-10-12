"""State/transport/supervision fixtures only; no simulator or learner imports."""

import builtins
import copy
import json
import math
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from abc_bench import fasttd3_host as host


class NativeWorldDouble:
    """Explicit CPU native API double; it does not implement physics."""

    single_timestep_action_dim = 14
    _control_decimation = 17

    def __init__(self, horizon=2, success_at=None):
        self.robot_names = ["left", "right"]
        self.horizon, self.success_at = horizon, success_at
        names = list(host.ACTION_NAMES)
        names = [name.replace("gripper", "left_finger") for name in names]
        names += ["left_right_finger", "right_right_finger", "bin_joint"]
        names += [f"bottle_{i}_joint" for i in range(1, 6)]
        joints = [
            SimpleNamespace(id=i, name=name, range=[-2.0, 4.0])
            for i, name in enumerate(names)
        ]
        self.model = SimpleNamespace(
            nq=58,
            nv=52,
            njnt=len(names),
            nbody=1,
            joint=lambda key: (
                joints[key] if type(key) is int else joints[names.index(key)]
            ),
            body=lambda _: SimpleNamespace(name="world"),
            jnt_qposadr=list(range(len(names))),
            jnt_dofadr=list(range(len(names))),
            jnt_type=[3] * len(names),
            jnt_range=[[-2.0, 4.0]] * len(names),
            body_pos=[[0.0, 0.0, 0.0]],
            body_quat=[[1.0, 0.0, 0.0, 0.0]],
            geom_size=[[1.0, 1.0, 1.0]],
            geom_bodyid=[0],
            opt=SimpleNamespace(timestep=0.002),
        )
        self.steps, self.closed, self.actions, self.resets = 0, 0, [], []
        self.randomization = {
            "seed": 0,
            "object_states": {},
            "scale_states": {},
            "metadata": {"bottle_count": 5, "bottle_variants": ["bottle_0"] * 5},
        }

    def forget_arm_state(self):
        pass

    def reset(self, *, seed, options, randomize):
        assert randomize is True
        self.steps, self.seed = 0, seed
        self.resets.append((seed, options))
        self.randomization = copy.deepcopy(self.randomization)
        self.randomization["seed"] = seed
        value = copy.deepcopy(self.randomization)
        return {}, {"randomization": SimpleNamespace(to_dict=lambda: value)}

    def evaluation(self):
        success = self.success_at is not None and self.steps >= self.success_at
        return {
            "reward": float(success),
            "success": success,
            "ever_success": success,
            "num_active_bottles": 5,
            "num_bottles_in_bin": 5 if success else 0,
        }

    def evaluate_task(self):
        return SimpleNamespace(to_info=lambda **_: self.evaluation())

    def capture_state(self):
        return {
            "qpos": [float(self.seed), float(self.steps)] + [0.0] * 56,
            "qvel": [float(self.steps)] * 52,
        }

    def step(self, actions, *, render_obs):
        assert render_obs is False
        self.actions.append(actions)
        self.steps += 1
        evaluation = self.evaluation()
        terminated = evaluation["success"]
        truncated = self.steps >= self.horizon and not terminated
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
def worker_config(tmp_path):
    config = {
        "schema_version": 1,
        "stage": "train",
        "learner_python": sys.executable,
        "host": {
            "num_envs": 1,
            "initial_seed": 9030,
            "max_episode_steps": 2,
            "save_video": False,
            "reference": None,
        },
        "learner": {
            "steps": 64,
            "seed": 1,
            "layout_seed": 9030,
            "batch_size": 32768,
            "buffer_size": 51200,
            "compile": True,
            "amp": True,
        },
        "timeouts": {
            "host_ready_s": 10,
            "learner_s": 30,
            "shutdown_s": 5,
            "host_wall_s": 60,
            "rpc_s": 10,
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path, config


def state_task(tmp_path, *, success_at=None, config=None):
    env = NativeWorldDouble(success_at=success_at)
    values = {
        "num_envs": 1,
        "initial_seed": 9030,
        "max_episode_steps": 2,
        "save_video": False,
        "reference": None,
    }
    values.update(config or {})
    return env, host.StateTask(env, values, tmp_path)


def test_action_affine_and_geometry_features(tmp_path):
    env, task = state_task(tmp_path)
    values = [-1.0, 0.0, 1.0] * 4 + [-1.0, 1.0]
    result = task.step([values])
    assert env.actions[0][:3] == [-2.0, 1.0, 4.0]
    assert env.actions[0][6] == 0.0 and env.actions[0][13] == 1.0
    assert len(result["obs"][0]) == 58 + 52 + 5 + 5 * 19 + 6
    assert result["obs"][0][110:115] == [1.0] * 5
    assert result["obs"][0][-6:] == [1.0] * 6
    assert task.description["privileged_state"] is True
    assert task.description["observation_schema"]["variant_vocabulary"] == list(
        host.VARIANTS
    )
    task.close()
    task.close()
    assert env.closed == 1


@pytest.mark.parametrize(
    "success_at,terminated,truncated", [(2, True, False), (None, False, True)]
)
def test_terminal_state_precedes_autoreset(tmp_path, success_at, terminated, truncated):
    _, task = state_task(tmp_path, success_at=success_at)
    task.step([[0.0] * 14])
    result = task.step([[0.0] * 14])
    assert result["terminal_obs"][0][:2] == [9030.0, 2.0]
    assert result["obs"][0][:2] == [9031.0, 0.0]
    assert result["terminated"] == [terminated] and result["truncated"] == [truncated]
    assert result["autoreset_seeds"] == [9031]
    assert result["episodes"] == [task.episodes[0]]
    assert result["episodes"][0]["seed"] == 9030
    assert result["episodes"][0]["completed"] is True
    assert result["episodes"][0]["success"] is terminated
    task.close()


def test_action_clipping_is_measured(tmp_path):
    env, task = state_task(tmp_path)
    result = task.step([[2.0] * 14])
    assert result["normalized_action_clipping_count"] == 14
    assert env.actions[0][0] == 4.0 and env.actions[0][6] == 1.0
    task.close()


@pytest.mark.parametrize(
    "actions", [[], [[0.0] * 13], [[True] * 14], [[math.inf] * 14]]
)
def test_invalid_actions_reject_before_native_step(tmp_path, actions):
    env, task = state_task(tmp_path)
    with pytest.raises(ValueError):
        task.step(actions)
    assert env.actions == []
    task.close()


def test_matched_reference_uses_original_default_request_and_verifies_state(tmp_path):
    env = NativeWorldDouble()
    expected = copy.deepcopy(env.randomization)
    expected["seed"] = 20260511
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"worlds": [{"randomization": expected}]}))
    task = host.StateTask(
        env,
        {
            "max_episode_steps": 2,
            "initial_seed": 20260511,
            "save_video": False,
            "reference": {
                "summary": str(summary),
                "sha256": host.artifact(summary)["sha256"],
                "world_index": 0,
            },
        },
        tmp_path,
    )
    assert env.resets == [(20260511, None)]
    assert task.episode["scene_match"] == "matched_baseline"
    assert task.randomization == expected
    task.close()
    changed = copy.deepcopy(expected)
    changed["scale_states"] = {"bin_joint": 1.2}
    summary.write_text(json.dumps({"worlds": [{"randomization": changed}]}))
    with pytest.raises(ValueError, match="pinned baseline RandomizationState"):
        host.StateTask(
            NativeWorldDouble(),
            {
                "max_episode_steps": 2,
                "initial_seed": 20260511,
                "save_video": False,
                "reference": {"summary": str(summary), "world_index": 0},
            },
            tmp_path,
        )


def test_wire_contract_and_request_rejections(tmp_path):
    _, task = state_task(tmp_path)
    request = {"schema": host.SCHEMA, "id": 1, "operation": "describe"}
    result, stop = host.dispatch(task, request)
    assert result["ok"] is True and result["id"] == 1 and not stop
    assert result["result"]["num_actions"] == 14
    for change in (
        {"schema": "other"},
        {"id": True},
        {"operation": "reset"},
        {"extra": 1},
    ):
        with pytest.raises(ValueError):
            host.dispatch(task, {**request, **change})
    assert host.dispatch(task, {**request, "operation": "shutdown"})[1]
    task.close()


def test_newline_frame_requires_bounded_complete_message(monkeypatch):
    sender, receiver = socket.socketpair()
    with sender, receiver:
        sender.sendall(b'{"id":1}\n')
        assert json.loads(host.receive_frame(receiver)) == {"id": 1}
    monkeypatch.setattr(host, "MAX_FRAME_BYTES", 8)
    sender, receiver = socket.socketpair()
    with sender, receiver:
        sender.sendall(b"123456789\n")
        with pytest.raises(ValueError, match="4 MiB"):
            host.receive_frame(receiver)


def test_frame_rejects_coalesced_messages_and_slow_trickle(monkeypatch):
    sender, receiver = socket.socketpair()
    with sender, receiver:
        sender.sendall(b"{}\n{}\n")
        with pytest.raises(ValueError, match="exactly one frame"):
            host.receive_frame(receiver)
    times = iter([0.0, 0.4, 1.1])
    monkeypatch.setattr(host.time, "monotonic", lambda: next(times))
    timeouts = []
    connection = SimpleNamespace(settimeout=timeouts.append, recv=lambda _: b"x")
    with pytest.raises(TimeoutError, match="absolute deadline"):
        host.receive_frame(connection, timeout=1.0)
    assert timeouts == [0.6]


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("host", "num_envs", 2),
        ("host", "max_episode_steps", 3541),
        ("learner", "steps", True),
        ("learner", "compile", 1),
        ("timeouts", "learner_s", math.inf),
        ("timeouts", "host_wall_s", 1),
    ],
)
def test_invalid_worker_config_precedes_lease_and_native(
    worker_config, monkeypatch, tmp_path, section, key, value
):
    path, config = worker_config
    config[section][key] = value
    path.write_text(json.dumps(config))
    monkeypatch.setattr(
        host,
        "require_subset_lease",
        lambda: pytest.fail("Invalid config reached lease"),
    )
    monkeypatch.setattr(
        host, "make_original_env", lambda _: pytest.fail("Native factory was called")
    )
    with pytest.raises(ValueError):
        host.supervise(path, tmp_path / "out")


def test_lease_precedes_every_native_factory(worker_config, monkeypatch, tmp_path):
    path, _ = worker_config

    def rejected():
        raise RuntimeError("Explicit absent lease double")

    monkeypatch.setattr(host, "require_subset_lease", rejected)
    original_import = builtins.__import__

    def no_native(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "mujoco", "abc_sim"}:
            pytest.fail("Native import before lease")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_native)
    with pytest.raises(RuntimeError, match="absent lease"):
        host.supervise(path, tmp_path / "out")
    with pytest.raises(RuntimeError, match="absent lease"):
        host.serve(path, tmp_path / "rpc" / "host.sock", tmp_path / "out")


def test_native_command_keeps_original_profile_and_fresh_output(
    worker_config, tmp_path
):
    _, config = worker_config
    command = host.learner_command(config, tmp_path / "socket", tmp_path / "NEW")
    assert command[:6] == [
        sys.executable,
        "-I",
        "-u",
        "-B",
        "-m",
        "sadhana_fasttd3_abc.cli",
    ]
    assert command[6] == "train"
    assert command[command.index("--steps") + 1] == "64"
    assert command[command.index("--batch-size") + 1] == "32768"
    assert "--compile" in command and "--amp" in command
    assert not (tmp_path / "NEW").exists()


def test_failed_snapshot_retains_failed_receipt(worker_config, monkeypatch, tmp_path):
    path, _ = worker_config
    monkeypatch.setattr(host, "require_subset_lease", lambda: None)

    def failed_snapshot(*_):
        raise RuntimeError("Explicit snapshot failure double")

    monkeypatch.setattr(host, "learner_snapshot", failed_snapshot)
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="snapshot failure"):
        host.supervise(path, out)
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["metrics"] == {}
    assert "snapshot failure" in receipt["blockers"][0]


def test_evaluation_metrics_keep_matched_and_additional_scenes_explicit():
    metrics = host.learner_metrics(
        {
            "episodes": [
                {"scene_match": "matched_baseline"},
                {"scene_match": "fixed_five_diagnostic"},
            ],
            "successes": 1,
            "actor_updates": 0,
            "critic_updates": 0,
        },
        "evaluate",
    )
    assert metrics == {
        "num_worlds": 2,
        "successes": 1,
        "matched_worlds": 1,
        "actor_updates": 0,
        "critic_updates": 0,
    }


def test_eval_summary_pin_is_checked_before_lease(worker_config, monkeypatch, tmp_path):
    path, config = worker_config
    summary = tmp_path / "reference.json"
    summary.write_text("{}")
    config["host"]["reference"] = {
        "summary": str(summary),
        "sha256": "0" * 64,
        "world_index": 0,
    }
    path.write_text(json.dumps(config))
    monkeypatch.setattr(
        host, "require_subset_lease", lambda: pytest.fail("Bad reference reached lease")
    )
    with pytest.raises(ValueError, match="SHA-256 pin"):
        host.supervise(path, tmp_path / "out")


@pytest.mark.parametrize("learner_code", [0, 1])
def test_supervisor_uses_one_process_group_and_reaps_failure(
    worker_config, monkeypatch, tmp_path, learner_code
):
    path, _ = worker_config
    out = tmp_path / "out"
    monkeypatch.setattr(host, "require_subset_lease", lambda: None)
    monkeypatch.setattr(host, "learner_snapshot", lambda *_: {"explicit_double": True})
    children = []

    class ProcessDouble:
        def __init__(self, command, **kwargs):
            assert "start_new_session" not in kwargs and "process_group" not in kwargs
            self.command, self.returncode, self.terminated = command, None, False
            self.is_host = "--serve" in command
            self.output = Path(command[command.index("--output") + 1])
            assert not self.output.exists()
            self.output.mkdir(parents=True)
            if self.is_host:
                (self.output / "ready.json").write_text("{}")
                (self.output / "description.json").write_text("{}")
            else:
                assert kwargs["env"]["WANDB_MODE"] == "disabled"
                assert kwargs["env"]["TORCHINDUCTOR_CACHE_DIR"].startswith(str(out))
                (self.output / "receipt.json").write_text(
                    json.dumps(
                        {
                            "status": "completed",
                            "controls": 64,
                            "actual_updates": {"critic": 42, "actor": 21},
                        }
                    )
                )
            children.append(self)

        def poll(self):
            return self.returncode

        def wait(self, *, timeout):
            assert timeout > 0
            self.returncode = 0 if self.is_host or self.terminated else learner_code
            if self.is_host:
                (self.output / "receipt.json").write_text(
                    json.dumps(
                        {
                            "status": "completed",
                            "metrics": {"explicit_fixture_controls": 64},
                        }
                    )
                )
            return self.returncode

        def terminate(self):
            self.terminated = True

    monkeypatch.setattr(host.subprocess, "Popen", ProcessDouble)
    shutdowns = []
    monkeypatch.setattr(host, "rpc_call", lambda *args: shutdowns.append(args))
    if learner_code:
        with pytest.raises(RuntimeError, match="learner exited 1"):
            host.supervise(path, out)
        assert children[0].terminated
        assert not shutdowns
        receipt = json.loads((out / "receipt.json").read_text())
        assert receipt["status"] == "failed" and receipt["metrics"] == {}
    else:
        receipt = host.supervise(path, out)
        assert receipt["status"] == "completed"
        assert receipt["metrics"] == {
            "controls": 64,
            "critic_updates": 42,
            "actor_updates": 21,
        }
        assert shutdowns[0][1] == "shutdown"
        assert not children[0].terminated
    assert all(child.poll() is not None for child in children)
