"""Tiny real MuJoCo model with explicit collector/task/camera CPU doubles.

These tests exercise native MJB/INTEGRATION and host dispatch. They do not
initialize ABC/VLA, render live cameras or measure a robot task score.
"""

import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from abc_bench import captured_reset as capture
from abc_bench import expoft_host as host
from abc_bench.upstream_eval import artifact

JUDGE = {
    "reward": 0.0,
    "success": False,
    "ever_success": False,
    "num_active_bottles": 5,
    "num_bottles_in_bin": 0,
    "max_bottles_in_bin_so_far": 0,
}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return artifact(path)


def save_array(root, name, value):
    path = root / (name + ".npy")
    np.save(path, value, allow_pickle=False)
    return {**artifact(path), "shape": list(value.shape), "dtype": value.dtype.name}


class TinyNativeEnv:
    """Real MuJoCo state facade; task and RGB are explicitly synthetic."""

    single_timestep_action_dim, _control_decimation = 14, 17
    prompt = "EXPLICIT CPU camera/task double"

    def __init__(self, model):
        self.robot_names = ["left", "right"]
        self.model, self.data = model, mujoco.MjData(model)
        self.cur_step, self.actions, self.closed = 0, [], False
        self.pixel_delta, self.mutate_after_render = 0, False

    def reload_from_model(self, model):
        self.model, self.data = copy.deepcopy(model), mujoco.MjData(model)

    def set_task(self, task):
        assert task == "put_plastic_bottles_in_bin"

    def task_runtime_debug_state(self):
        return None

    def get_state(self):
        result = []
        for arm in self.robot_names:
            result.extend(
                float(self.data.qpos[self.model.joint(f"{arm}_joint{i}").qposadr[0]])
                for i in range(1, 7)
            )
            grip = (
                self.data.qpos[self.model.joint(f"{arm}_left_finger").qposadr[0]]
                / 0.0475
            )
            result.append(float(np.clip(grip, 0, 1)))
        return np.asarray(result, dtype=np.float32)

    def evaluate_task(self):
        return SimpleNamespace(to_info=lambda **kwargs: dict(JUDGE))

    def get_obs(self):
        if self.mutate_after_render:
            self.data.qpos[0] += 1e-6
        images = {
            name: np.zeros((3, 168, 224), dtype=np.uint8) for name in host.CAMERAS
        }
        images["top"][0, 0, 0] = self.pixel_delta
        return {
            "state": self.get_state(),
            "images": images,
            "masks": {name: True for name in host.CAMERAS},
            "camera_timestamps": {name: float(self.data.time) for name in host.CAMERAS},
        }

    def step(self, action, *, render_obs):
        assert render_obs is False
        self.actions.append(list(action))
        values = np.asarray(action, dtype=np.float32).copy()
        values[[6, 13]] *= 0.0475
        self.data.ctrl[:] = values
        for _ in range(17):
            mujoco.mj_step(self.model, self.data)
        self.cur_step += 1
        return None, 0.0, False, self.cur_step >= 3, {"task_eval": dict(JUDGE)}

    def close(self):
        self.closed = True


@pytest.fixture
def captured_fixture(tmp_path, monkeypatch):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert os.environ.get("MUJOCO_GL") == "disable"
    bodies, actuators = [], []
    for arm in ("left", "right"):
        names = [f"{arm}_joint{i}" for i in range(1, 7)] + [
            f"{arm}_left_finger",
            f"{arm}_right_finger",
        ]
        for name in names:
            bodies.append(
                f'<body name="{name}_body" pos="{len(bodies) * 0.05} 0 0"><joint name="{name}" range="-2 4"/><geom type="sphere" size=".01" mass="1"/></body>'
            )
        actuators.extend(
            f'<motor name="{name}" joint="{name}" ctrllimited="true" ctrlrange="-2 4"/>'
            for name in names[:6]
        )
        actuators.append(
            f'<motor name="{arm}_gripper" joint="{arm}_left_finger" ctrllimited="true" ctrlrange="0 .0495"/>'
        )
    for name in ["bin_joint"] + [f"bottle_{i}_joint" for i in range(1, 6)]:
        bodies.append(
            f'<body name="{name}_body" pos="{len(bodies) * 0.05} 0 2"><freejoint name="{name}"/><geom type="sphere" size=".01" mass="1"/></body>'
        )
    bodies.extend(f'<body name="fixed{i}"/>' for i in range(27))
    cameras = "".join(f'<camera name="{name}" pos="0 0 4"/>' for name in host.CAMERAS)
    equalities = "".join(
        f'<joint joint1="{arm}_left_finger" joint2="{arm}_right_finger"/>'
        for arm in ("left", "right")
    )
    xml = f'<mujoco><compiler angle="radian"/><option timestep=".002" gravity="0 0 0"/><worldbody>{"".join(bodies)}{cameras}</worldbody><actuator>{"".join(actuators)}</actuator><equality>{equalities}</equality></mujoco>'
    model = mujoco.MjModel.from_xml_string(xml)
    assert (model.nq, model.nv, model.nu) == (58, 52, 14)
    assert mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION) == 531
    environment = TinyNativeEnv(model)
    environment.data.qpos[:16] = 0.01
    environment.data.qpos[0] = 0.321
    environment.data.ctrl[0] = 0.125
    environment.data.qacc_warmstart[0] = 0.0625
    mujoco.mj_forward(model, environment.data)
    integration = np.empty(531, dtype=np.float64)
    mujoco.mj_getState(
        model, environment.data, integration, mujoco.mjtState.mjSTATE_INTEGRATION
    )
    root = tmp_path / "bench" / "toy-run"
    root.mkdir(parents=True)
    files = {}
    for name, value in (
        ("initial_integration", integration),
        ("qpos", np.tile(environment.data.qpos, (17, 1))),
        ("qvel", np.tile(environment.data.qvel, (17, 1))),
        ("states", np.tile(environment.get_state(), (17, 1))),
        ("sim_times", np.arange(17, dtype=np.float64) * 0.034),
    ):
        files[name] = save_array(root, name, value)
    mjb = root / "scene.mjb"
    mujoco.mj_saveModel(model, str(mjb), None)
    files["compiled_model"] = artifact(mjb)
    scene_xml = root / "scene.xml"
    scene_xml.write_text(xml)
    files["scene_xml"] = artifact(scene_xml)
    randomization = {"seed": 9100, "metadata": {"bottle_count": 5}}
    files["randomization"] = save_json(root / "randomization.json", randomization)
    rgb = root / "rgb24.bin"
    rgb.write_bytes(bytes(17 * 168 * 672 * 3))
    files["rgb24"] = {**artifact(rgb), "shape": [17, 168, 672, 3], "dtype": "uint8"}
    software = save_json(
        root / "software.json",
        {
            "sources": {},
            "distributions": {
                name: importlib.metadata.version(name)
                for name in (
                    "mujoco",
                    "numpy",
                    "abc-minimal",
                    "mujoco-warp",
                    "warp-lang",
                )
            },
        },
    )
    producer = {
        "config": save_json(root / "config.json", {"split": "validation"}),
        "software": software,
        "metadata": save_json(root / "metadata.json", {"fixture": True}),
        "checkpoint": {"fixture": "NO MODEL"},
    }
    reset_group = hashlib.sha256(
        (
            files["scene_xml"]["sha256"] + ":" + files["initial_integration"]["sha256"]
        ).encode()
    ).hexdigest()
    episode = {
        "episode_id": "explicit-cpu-val",
        "split": "validation",
        "seed": 9100,
        "world_index": 0,
        "task": "put_plastic_bottles_in_bin",
        "prompt": environment.prompt,
        "controls": 16,
        "artifacts": files,
        "producer": producer,
        "reset_group_id": reset_group,
        "integration_state_spec": {"name": "mjSTATE_INTEGRATION", "value": 16383},
        "initial_task_eval": dict(JUDGE),
    }
    manifest = save_json(root / "manifest.json", episode)
    summary = save_json(root / "summary.json", {"worlds": [{}]})
    worker = save_json(
        root / "receipt.json",
        {
            "schema_version": 1,
            "status": "completed",
            "phase": "original_policy_demonstration_collection",
            "split": "validation",
            "input_config": producer["config"],
            "checkpoint": producer["checkpoint"],
            "metadata": producer["metadata"],
            "software": software,
            "episodes": [{"manifest": manifest, "admitted": True}],
            "original_summary": summary,
        },
    )
    job = save_json(root / "job.json", {"module": "abc_bench.demo_collect"})
    parent = save_json(
        root.parent / "runs" / "toy-run.json",
        {
            "schema_version": 1,
            "status": "completed",
            "run_id": "toy-run",
            "task": "put_plastic_bottles_in_bin",
            "embodiment": "native_yam",
            "worker_exit_code": 0,
            "worker_receipt_sha256": worker["sha256"],
            "resolved_job_sha256": job["sha256"],
        },
    )
    cohort = {
        "schema": "sadhana.abc.demonstration-cohort/v1",
        "splits": {"train": [], "validation": ["explicit-cpu-val"], "development": []},
        "episodes": [
            {"manifest": manifest, "worker_receipt": worker, "parent_receipt": parent}
        ],
    }
    profile = {
        "schema": capture.RESET_SCHEMA,
        "method": capture.RESET_METHOD,
        "cohort": save_json(root / "cohort.json", cohort),
        "episode_id": "explicit-cpu-val",
    }
    config = {
        "schema_version": 2,
        "stage": "evaluation",
        "host": {
            "num_envs": 1,
            "initial_seed": 9100,
            "allowed_seeds": [9100],
            "max_episode_steps": 3,
            "save_video": False,
            "action_profile": capture.ACTION_PROFILE,
            "reset_profile": profile,
        },
        "timeouts": {"host_wall_s": 30, "rpc_s": 2, "max_requests": 20},
    }
    # Only the producer/task semantics are doubled. MJB/INTEGRATION are real.
    monkeypatch.setattr(
        "abc_bench.demo_cohort.validate_episode", lambda *args: copy.deepcopy(episode)
    )
    output = tmp_path / "host"
    output.mkdir()
    return SimpleNamespace(
        env=environment,
        episode=episode,
        config=config,
        profile=profile,
        cohort=cohort,
        output=output,
        root=root,
        model=model,
    )


def make_task(fixture):
    scene = capture.CapturedScene(fixture.profile)
    return host.CapturedExpoTask(fixture.env, fixture.config, fixture.output, scene)


def test_real_mjb_full_integration_repeated_reset_preserves_arm_ctrl_warmstart(
    captured_fixture,
):
    fixture = captured_fixture
    task = make_task(fixture)
    for _ in range(2):
        fixture.env.data.qpos[0] = -1
        fixture.env.data.ctrl[0] = -1
        fixture.env.data.qacc_warmstart[0] = -1
        result = task.reset(9100)
        assert result["reset_verified"] is True
        assert fixture.env.data.qpos[0] == 0.321
        assert fixture.env.data.ctrl[0] == 0.125
        assert fixture.env.data.qacc_warmstart[0] == 0.0625
        proof = json.loads(Path(result["reset_verification"]["path"]).read_text())
        assert all(
            value["equal"]
            for stage in proof["checks"].values()
            for value in stage.values()
        )
        assert all(value["equal"] for value in proof["camera_checks"].values())
    assert fixture.env.actions == []


@pytest.mark.parametrize("failure", ["pixel", "post_render_state"])
def test_failed_live_render_gates_disable_actions_and_preserve_evidence(
    captured_fixture, failure
):
    fixture = captured_fixture
    task = make_task(fixture)
    if failure == "pixel":
        fixture.env.pixel_delta = 1
    else:
        fixture.env.mutate_after_render = True
    with pytest.raises(RuntimeError, match="differs"):
        task.reset(9100)
    with pytest.raises(RuntimeError, match="require complete"):
        host.dispatch(
            task,
            {"schema": host.SCHEMA, "id": 1, "operation": "step", "action": [0] * 14},
        )
    assert fixture.env.actions == []
    proof = json.loads((fixture.output / "reset-proof-0002.json").read_text())
    assert proof["status"] == "failed" and proof["reset_verified"] is False


def test_raw_profile_forwards_float32_out_of_nominal_commands(captured_fixture):
    task = make_task(captured_fixture)
    action = [0.123456789] * 14
    action[0], action[6], action[13] = 8.0, -0.16, 1.1
    response, _ = host.dispatch(
        task, {"schema": host.SCHEMA, "id": 1, "operation": "step", "action": action}
    )
    result = response["result"]
    assert (
        result["action_applied14"]
        == np.asarray(action, dtype=np.float32).astype(np.float64).tolist()
    )
    assert result["action_clipping"] == []
    assert captured_fixture.env.actions == [result["action_applied14"]]
    assert task.episode["nominal_bound_violation_count"] == 3


def test_wrong_declared_seed_is_rejected_without_resampling(captured_fixture):
    path = captured_fixture.root / "host-config.json"
    save_json(path, captured_fixture.config)
    assert host.load_config(path) == captured_fixture.config
    config = copy.deepcopy(captured_fixture.config)
    config["host"]["initial_seed"] = 9101
    with pytest.raises(ValueError, match="selected actual"):
        capture.validate_config(config)
    task = make_task(captured_fixture)
    with pytest.raises(ValueError, match="selected captured"):
        task.reset(9101)
    assert captured_fixture.env.actions == []


@pytest.mark.parametrize("failure", ["failed_parent", "substituted_worker"])
def test_capture_requires_exact_completed_parent_worker_pair(captured_fixture, failure):
    fixture = captured_fixture
    entry = fixture.cohort["episodes"][0]
    path = Path(entry["parent_receipt"]["path"])
    parent = json.loads(path.read_text())
    if failure == "failed_parent":
        parent["status"] = "failed"
    else:
        parent["worker_receipt_sha256"] = "a" * 64
    entry["parent_receipt"] = save_json(path, parent)
    fixture.profile["cohort"] = save_json(fixture.root / "cohort.json", fixture.cohort)
    with pytest.raises(ValueError):
        capture.CapturedScene(fixture.profile)
    assert fixture.env.actions == []


def test_invalid_numeric_state_is_rejected_before_controls(captured_fixture):
    fixture = captured_fixture
    scene = capture.CapturedScene(fixture.profile)
    scene.files["initial_integration"] = save_array(
        fixture.root, "bad_state", np.zeros(531, np.float32)
    )
    with pytest.raises(ValueError, match="shape/dtype"):
        host.CapturedExpoTask(fixture.env, fixture.config, fixture.output, scene)
    assert fixture.env.actions == []
