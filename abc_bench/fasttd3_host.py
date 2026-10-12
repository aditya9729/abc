"""Lease-guarded original ABC state host and isolated original FastTD3 worker.

The first diagnostic supports one world only. Physics uses the unchanged
sequential CPU MuJoCo environment, as does the stock ABC evaluation path.
MJWarp renders videos. Actor observations include privileged scene state.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from abc_bench.subset import require_subset_lease
from abc_bench.upstream_eval import artifact

SCHEMA = "sadhana.fasttd3.abc.v1"
MAX_FRAME_BYTES = 4 * 1024**2
CAMERAS = ("top", "left", "right")
VARIANTS = tuple(f"bottle_{index}" for index in range(19))
RESET_OPTIONS = {
    "randomization": {
        "bottle_count": 5,
        "randomize_variants": False,
        "randomize_scales": False,
    }
}
ACTION_NAMES = tuple(
    name
    for arm in ("left", "right")
    for name in (*[f"{arm}_joint{index}" for index in range(1, 7)], f"{arm}_gripper")
)


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def finite_json(value: Any) -> None:
    if type(value) is float and not math.isfinite(value):
        raise ValueError("All JSON numbers must be finite")
    if isinstance(value, dict):
        for child in value.values():
            finite_json(child)
    elif isinstance(value, list):
        for child in value:
            finite_json(child)


def bounded_int(value: Any, name: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def load_config(path: Path) -> dict[str, Any]:
    if path.stat().st_size > 1024**2:
        raise ValueError("Worker config must be at most 1 MiB")
    config = json.loads(path.read_text())
    finite_json(config)
    if type(config) is not dict or set(config) != {
        "schema_version",
        "stage",
        "learner_python",
        "host",
        "learner",
        "timeouts",
    }:
        raise ValueError("Expected the complete FastTD3 worker configuration")
    bounded_int(config["schema_version"], "schema_version", 1, 1)
    if config["stage"] not in ("train", "evaluate"):
        raise ValueError("stage must be train or evaluate")
    python = config["learner_python"]
    if (
        type(python) is not str
        or not Path(python).is_absolute()
        or not Path(python).is_file()
    ):
        raise ValueError("learner_python must be an existing absolute interpreter")
    host = config["host"]
    if type(host) is not dict or set(host) != {
        "num_envs",
        "initial_seed",
        "max_episode_steps",
        "save_video",
        "reference",
    }:
        raise ValueError(
            "host requires num_envs, initial_seed, max_episode_steps, save_video"
        )
    bounded_int(host["num_envs"], "host.num_envs", 1, 1)
    bounded_int(host["initial_seed"], "host.initial_seed", 0, 2**31 - 1)
    bounded_int(host["max_episode_steps"], "host.max_episode_steps", 1, 3540)
    if type(host["save_video"]) is not bool:
        raise ValueError("host.save_video must be a boolean")
    reference = host["reference"]
    if reference is not None:
        if type(reference) is not dict or set(reference) != {
            "summary",
            "sha256",
            "world_index",
        }:
            raise ValueError("Reference requires summary, sha256, and world_index")
        source = reference["summary"]
        if (
            type(source) is not str
            or not Path(source).is_absolute()
            or not Path(source).is_file()
        ):
            raise ValueError("Reference summary must be an existing absolute file")
        if Path(source).stat().st_size > MAX_FRAME_BYTES:
            raise ValueError("Reference summary must be at most 4 MiB")
        if artifact(Path(source))["sha256"] != reference["sha256"]:
            raise ValueError("Reference summary differs from its SHA-256 pin")
        summary = json.loads(Path(source).read_text())
        finite_json(summary)
        index = bounded_int(reference["world_index"], "reference.world_index", 0, 4)
        if index >= len(summary["worlds"]):
            raise ValueError("Reference world index does not exist")
        world = summary["worlds"][index]
        state = world["randomization"]
        if (
            summary.get("task") != "put_plastic_bottles_in_bin"
            or world.get("world_seed") != state["seed"]
        ):
            raise ValueError(
                "Reference task or actual world seed differs from its state"
            )
        if state["metadata"].get("bottle_count") != 5:
            raise ValueError("Matched reference requires five active bottles")
        bounded_int(state["seed"], "reference seed", 0, 2**31 - 1)
    learner = config["learner"]
    expected = (
        {"steps", "seed", "layout_seed", "batch_size", "buffer_size", "compile", "amp"}
        if config["stage"] == "train"
        else {"checkpoint", "checkpoint_sha256", "seeds"}
    )
    if type(learner) is not dict or set(learner) != expected:
        raise ValueError("learner fields must match the selected train/evaluate stage")
    if config["stage"] == "train":
        for name, low, high in (
            ("steps", 12, 100000),
            ("seed", 0, 2**31 - 1),
            ("layout_seed", 0, 2**31 - 1),
            ("batch_size", 1, 32768),
            ("buffer_size", 1, 51200),
        ):
            bounded_int(learner[name], f"learner.{name}", low, high)
        for name in ("compile", "amp"):
            if type(learner[name]) is not bool:
                raise ValueError(f"learner.{name} must be a boolean")
    else:
        checkpoint = learner["checkpoint"]
        if (
            type(checkpoint) is not str
            or not Path(checkpoint).is_absolute()
            or not Path(checkpoint).is_file()
        ):
            raise ValueError("Evaluation requires an existing absolute checkpoint")
        pin = learner["checkpoint_sha256"]
        if (
            type(pin) is not str
            or len(pin) != 64
            or any(x not in "0123456789abcdef" for x in pin)
        ):
            raise ValueError("Evaluation requires a lowercase SHA-256 checkpoint pin")
        seeds = learner["seeds"]
        if type(seeds) is not list or not 1 <= len(seeds) <= 5:
            raise ValueError("Evaluation requires one to five explicit layout seeds")
        for seed in seeds:
            bounded_int(seed, "evaluation seed", 0, 2**31 - 1)
        if len(set(seeds)) != len(seeds):
            raise ValueError("Evaluation seeds must be distinct")
    deadlines = config["timeouts"]
    if type(deadlines) is not dict or set(deadlines) != {
        "host_ready_s",
        "learner_s",
        "shutdown_s",
        "host_wall_s",
        "rpc_s",
    }:
        raise ValueError("Explicit finite stage deadlines are required")
    for name, maximum in (
        ("host_ready_s", 600),
        ("learner_s", 6000),
        ("shutdown_s", 30),
        ("host_wall_s", 7200),
        ("rpc_s", 600),
    ):
        bounded_int(deadlines[name], f"timeouts.{name}", 1, maximum)
    if (
        sum(deadlines[name] for name in ("host_ready_s", "learner_s", "shutdown_s"))
        > deadlines["host_wall_s"]
    ):
        raise ValueError("Stage deadlines must fit within host_wall_s")
    return config


def numeric_vector(value: Any, size: int, name: str) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if type(value) is not list or len(value) != size:
        raise ValueError(f"{name} must have shape ({size},)")
    if any(type(item) not in (int, float) or not math.isfinite(item) for item in value):
        raise ValueError(f"{name} must contain finite numbers")
    return [float(item) for item in value]


def plain(value: Any) -> Any:
    return value.tolist() if hasattr(value, "tolist") else value


def reset_original_world(
    env: Any, seed: int, reference_state: dict[str, Any] | None
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """Reset original physics and verify the captured baseline scene when selected.

    The caller must hold the live lease before it constructs ``env``. Matching
    uses the untouched default sampler; a fixed-count request changes its RNG.
    Other seeds use the explicitly different fixed-five diagnostic distribution.
    """
    bounded_int(seed, "reset seed", 0, 2**31 - 1)
    env.forget_arm_state()
    matched = reference_state is not None and seed == reference_state["seed"]
    raw, info = env.reset(
        seed=seed, options=None if matched else RESET_OPTIONS, randomize=True
    )
    randomization = info["randomization"].to_dict()
    finite_json(randomization)
    if matched and randomization != reference_state:
        raise ValueError(
            "Original reset does not reproduce the pinned baseline RandomizationState"
        )
    return raw, randomization, matched


class StateTask:
    """One original MuJoCo world; final state is copied before automatic reset."""

    def __init__(self, env: Any, config: dict[str, Any], output: Path) -> None:
        self.env, self.config, self.output = env, config, output
        self.max_episode_steps = config["max_episode_steps"]
        self.next_seed = config["initial_seed"]
        self.episodes: list[dict[str, Any]] = []
        self.reset_seeds: list[int] = []
        self.controls, self.episode_steps, self.clipped_actions = 0, 0, 0
        self.closed, self.writer, self.episode = False, None, None
        self.description: dict[str, Any] | None = None
        self.randomization: dict[str, Any] = {}
        self.reference = None
        if config["reference"] is not None:
            ref = config["reference"]
            self.reference = json.loads(Path(ref["summary"]).read_text())["worlds"][
                ref["world_index"]
            ]["randomization"]
        self.reset([self.next_seed])

    def _description(self) -> dict[str, Any]:
        model = self.env.model
        joint_names = [model.joint(index).name for index in range(model.njnt)]
        if (
            self.env.robot_names != ["left", "right"]
            or self.env.single_timestep_action_dim != 14
        ):
            raise ValueError("The host requires original left/right YAM 14D control")
        low, high = [], []
        for name in ACTION_NAMES:
            if name.endswith("gripper"):
                low.append(0.0)
                high.append(1.0)
            else:
                limits = numeric_vector(model.joint(name).range, 2, f"{name} limits")
                if limits[0] >= limits[1]:
                    raise ValueError(f"{name} must have physical joint limits")
                low.append(limits[0])
                high.append(limits[1])
        objects = []
        for index in range(1, 6):
            name = f"bottle_{index}"
            joint = model.joint(f"{name}_joint")
            objects.append(
                {
                    "name": name,
                    "joint": joint.name,
                    "qpos_address": int(model.jnt_qposadr[joint.id]),
                    "qvel_address": int(model.jnt_dofadr[joint.id]),
                    "active": True,
                }
            )
        bin_joint = model.joint("bin_joint")
        scene = {
            "joint_names": joint_names,
            "joint_types": plain(model.jnt_type),
            "qpos_addresses": plain(model.jnt_qposadr),
            "qvel_addresses": plain(model.jnt_dofadr),
            "joint_ranges": plain(model.jnt_range),
            "body_names": [model.body(index).name for index in range(model.nbody)],
            "body_pos": plain(model.body_pos),
            "body_quat": plain(model.body_quat),
            "geom_size": plain(model.geom_size),
            "geom_bodyid": plain(model.geom_bodyid),
            "objects": objects,
            "bin_joint": {
                "name": "bin_joint",
                "qpos_address": int(model.jnt_qposadr[bin_joint.id]),
                "qvel_address": int(model.jnt_dofadr[bin_joint.id]),
            },
        }
        nq, nv = int(model.nq), int(model.nv)
        return {
            "num_envs": 1,
            "num_obs": nq + nv + 5 + 5 * len(VARIANTS) + 6,
            "num_actions": 14,
            "max_episode_steps": self.max_episode_steps,
            "privileged_state": True,
            "task": "put_plastic_bottles_in_bin",
            "prompt": "sim throw plastic bottles in bin",
            "physics": "original sequential CPU MuJoCo",
            "physics_dt": float(model.opt.timestep),
            "control_decimation": self.env._control_decimation,
            "camera_backend": "mjwarp",
            "camera_shape": [3, 168, 224],
            "reset_options": RESET_OPTIONS,
            "observation_schema": {
                "format": "sadhana.abc.privileged_state.v1",
                "dtype": "float32",
                "layout": [
                    {"name": "qpos", "offset": 0, "length": nq},
                    {"name": "qvel", "offset": nq, "length": nv},
                    {"name": "active_bottles", "offset": nq + nv, "length": 5},
                    {
                        "name": "bottle_variant_one_hot",
                        "offset": nq + nv + 5,
                        "length": 5 * len(VARIANTS),
                    },
                    {
                        "name": "bottle_and_bin_scales",
                        "offset": nq + nv + 5 + 5 * len(VARIANTS),
                        "length": 6,
                    },
                ],
                "variant_vocabulary": list(VARIANTS),
                "coordinates": "Original MuJoCo generalized coordinates and velocities",
                "units": "SI: metres, radians, metres/second, radians/second",
                "free_joint_quaternion": "wxyz",
                "object_metadata": scene,
            },
            "action_schema": {
                "names": list(ACTION_NAMES),
                "normalized_low": [-1.0] * 14,
                "normalized_high": [1.0] * 14,
                "policy_low": low,
                "policy_high": high,
                "mapping": "low + (clip(u,-1,1)+1)*0.5*(high-low)",
                "joint_units": "radians",
                "gripper_policy_units": "fraction in [0,1]",
                "gripper_physical_range_m": [0.0, 0.0475],
                "gripper_note": "Original environment multiplies fraction by _GRIPPER_CTRL_MAX=0.0475",
            },
        }

    def observation(self) -> list[float]:
        state = self.env.capture_state()
        variants = self.randomization["metadata"]["bottle_variants"]
        if len(variants) != 5 or any(name not in VARIANTS for name in variants):
            raise ValueError(
                "Original bottle variants differ from the pinned vocabulary"
            )
        scales = self.randomization["scale_states"]
        shape_features = [
            float(name == variant) for variant in variants for name in VARIANTS
        ]
        shape_features.extend(
            float(scales.get(name, 1.0))
            for name in (*[f"bottle_{i}_joint" for i in range(1, 6)], "bin_joint")
        )
        finite_json(shape_features)
        return (
            numeric_vector(state["qpos"], int(self.env.model.nq), "qpos")
            + numeric_vector(state["qvel"], int(self.env.model.nv), "qvel")
            + [1.0] * 5
            + shape_features
        )

    def _frame(self, images: dict[str, Any]) -> None:
        if self.writer is None:
            return
        import numpy as np

        self.writer.append_data(
            np.concatenate(
                [images[name].transpose(1, 2, 0) for name in CAMERAS], axis=1
            )
        )

    def _finish(self, *, completed: bool, reason: str) -> dict[str, Any] | None:
        episode, self.episode = self.episode, None
        if episode is None:
            return None
        if self.writer is not None:
            self.writer.close()
            self.writer = None
            episode["video"] = artifact(Path(episode.pop("video_path")))
        episode.update(completed=completed, end_reason=reason)
        if episode["controls"] or completed:
            self.episodes.append(episode)
            write_json(self.output / f"episode-{len(self.episodes):04d}.json", episode)
        return episode

    def reset(self, seeds: Any) -> dict[str, Any]:
        if self.closed:
            raise RuntimeError("ABC state host is closed")
        if type(seeds) is not list or len(seeds) != 1:
            raise ValueError("The first diagnostic accepts exactly one reset seed")
        seed = bounded_int(seeds[0], "reset seed", 0, 2**31 - 1)
        self._finish(completed=False, reason="explicit_reset")
        raw, self.randomization, matched = reset_original_world(
            self.env, seed, self.reference
        )
        evaluation = self.env.evaluate_task().to_info(squeeze=True)
        if evaluation.get("num_active_bottles") != 5:
            raise ValueError("Original scene must contain exactly five active bottles")
        description = self._description()
        if self.description is not None and (
            description["num_obs"] != self.description["num_obs"]
            or description["action_schema"] != self.description["action_schema"]
            or description["observation_schema"]["object_metadata"]["joint_names"]
            != self.description["observation_schema"]["object_metadata"]["joint_names"]
        ):
            raise ValueError(
                "Original physical observation/action schema changed at reset"
            )
        self.description = description
        self.reset_seeds.append(seed)
        self.next_seed = seed + 1
        self.episode_steps = 0
        self.episode = {
            "seed": seed,
            "success": False,
            "completed": False,
            "controls": 0,
            "normalized_action_clipping_count": 0,
            "max_native_reward": float(evaluation["reward"]),
            "final_native_task_eval": evaluation,
            "scene_match": "matched_baseline" if matched else "fixed_five_diagnostic",
            "randomization": self.randomization,
        }
        write_json(
            self.output / f"reset-{len(self.reset_seeds):04d}.json",
            {
                "seed": seed,
                "scene_match": self.episode["scene_match"],
                "randomization": self.randomization,
                "initial_observation": self.observation(),
                "description": description,
            },
        )
        if self.config["save_video"]:
            import imageio.v2 as imageio

            video = self.output / f"reset-{len(self.reset_seeds):04d}-seed-{seed}.mp4"
            self.episode["video_path"] = str(video)
            self.writer = imageio.get_writer(str(video), fps=30, macro_block_size=1)
            self._frame(raw["images"])
        return {
            "obs": [self.observation()],
            "actual_seeds": [seed],
            "object_metadata": description["observation_schema"]["object_metadata"],
        }

    def step(self, actions: Any) -> dict[str, Any]:
        if self.closed or self.episode is None:
            raise RuntimeError("ABC state host has no active episode")
        if type(actions) is not list or len(actions) != 1:
            raise ValueError("Actions must have shape (1,14)")
        normalized = numeric_vector(actions[0], 14, "normalized action")
        schema = self.description["action_schema"]
        clipped = sum(item < -1.0 or item > 1.0 for item in normalized)
        physical = [
            low + (min(1.0, max(-1.0, item)) + 1.0) * 0.5 * (high - low)
            for item, low, high in zip(
                normalized, schema["policy_low"], schema["policy_high"]
            )
        ]
        _, reward, terminated, truncated, info = self.env.step(
            physical, render_obs=False
        )
        if (
            type(terminated) is not bool
            or type(truncated) is not bool
            or terminated
            and truncated
        ):
            raise RuntimeError("Original termination/truncation flags are invalid")
        if type(reward) not in (int, float) or not math.isfinite(reward):
            raise RuntimeError("Original reward is not finite")
        terminal_obs = self.observation()
        self.controls += 1
        self.episode_steps += 1
        self.clipped_actions += clipped
        self.episode["controls"] = self.episode_steps
        self.episode["normalized_action_clipping_count"] += clipped
        evaluation = info["task_eval"]
        self.episode["success"] = bool(evaluation["ever_success"])
        self.episode["max_native_reward"] = max(
            self.episode["max_native_reward"], float(reward)
        )
        self.episode["final_native_task_eval"] = evaluation
        self._frame(self.env.get_obs()["images"] if self.writer is not None else {})
        episodes, reset_seeds, obs = [], [], terminal_obs
        if terminated or truncated:
            episodes.append(
                self._finish(
                    completed=True, reason="success" if terminated else "timeout"
                )
            )
            reset = self.reset([self.next_seed])
            obs, reset_seeds = reset["obs"][0], reset["actual_seeds"]
        return {
            "obs": [obs],
            "terminal_obs": [terminal_obs],
            "rewards": [float(reward)],
            "terminated": [terminated],
            "truncated": [truncated],
            "episodes": episodes,
            "autoreset_seeds": reset_seeds,
            "normalized_action_clipping_count": clipped,
        }

    def close(self) -> None:
        if not self.closed:
            try:
                self._finish(completed=False, reason="host_closed")
            finally:
                self.closed = True
                self.env.close()


def make_original_env(config: dict[str, Any]) -> Any:
    # Called only from the lease-guarded server; no native dependency at module import.
    import abc_sim

    return abc_sim.make_env(
        task="put_plastic_bottles_in_bin",
        prompt="sim throw plastic bottles in bin",
        render_cameras=config["save_video"],
        camera_backend="mjwarp",
        camera_gpu_id=0,
        camera_height=168,
        camera_width=224,
        max_episode_steps=config["max_episode_steps"],
        terminate_on_success=True,
    )


def dispatch(task: StateTask, request: Any) -> tuple[dict[str, Any], bool]:
    if type(request) is not dict or request.get("schema") != SCHEMA:
        raise ValueError("RPC schema must be sadhana.fasttd3.abc.v1")
    bounded_int(request.get("id"), "request id", 1, 2**63 - 1)
    operation = request.get("operation")
    parameters = {
        "describe": set(),
        "reset": {"seeds"},
        "step": {"actions"},
        "shutdown": set(),
    }
    if (
        operation not in parameters
        or set(request) != {"schema", "id", "operation"} | parameters[operation]
    ):
        raise ValueError("RPC operation or fields are invalid")
    if operation == "describe":
        result = task.description
    elif operation == "reset":
        result = task.reset(request["seeds"])
    elif operation == "step":
        result = task.step(request["actions"])
    else:
        result = {"status": "closing"}
    return {
        "schema": SCHEMA,
        "id": request["id"],
        "ok": True,
        "result": result,
    }, operation == "shutdown"


def rpc_call(path: Path, operation: str, timeout: int) -> dict[str, Any]:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout)
        connection.connect(str(path))
        connection.sendall(
            json.dumps({"schema": SCHEMA, "id": 1, "operation": operation}).encode()
            + b"\n"
        )
        result = json.loads(receive_frame(connection, timeout))
    if (
        result.get("schema") != SCHEMA
        or result.get("id") != 1
        or result.get("ok") is not True
    ):
        raise RuntimeError(f"Host rejected supervisor operation {operation}")
    return result["result"]


def receive_frame(connection: socket.socket, timeout: float = 120.0) -> bytes:
    frame = bytearray()
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("RPC frame exceeded its absolute deadline")
        connection.settimeout(remaining)
        chunk = connection.recv(min(65536, MAX_FRAME_BYTES + 1 - len(frame)))
        if not chunk:
            raise ValueError("RPC connection ended before a newline-delimited frame")
        frame.extend(chunk)
        if len(frame) > MAX_FRAME_BYTES:
            raise ValueError("RPC frame exceeds 4 MiB")
        newline = frame.find(b"\n")
        if newline >= 0:
            if newline != len(frame) - 1:
                raise ValueError("One RPC connection must contain exactly one frame")
            return bytes(frame)


def native_sources(output: Path) -> dict[str, Any]:
    import abc_sim

    root = Path(abc_sim.__file__).parent
    sources = [
        root / name
        for name in (
            "__init__.py",
            "env.py",
            "config.py",
            "task_registry.py",
            "task_eval/base.py",
            "task_eval/bottles.py",
            "randomization/core.py",
            "randomization/requests.py",
            "randomization/assets/water_bottles.py",
            "randomization/tasks/bottles.py",
        )
    ]
    assets = [
        path
        for path in (root / "models").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ]
    evidence = {
        "sources": [artifact(path) for path in sources],
        "assets": [artifact(path) for path in sorted(assets)],
    }
    write_json(output / "original_source_assets.json", evidence)
    return artifact(output / "original_source_assets.json")


def serve(config_path: Path, socket_path: Path, output: Path) -> None:
    config = load_config(config_path)
    require_subset_lease()
    if not socket_path.is_absolute() or len(os.fsencode(socket_path)) > 107:
        raise ValueError("Use an absolute Unix socket path of at most 107 bytes")
    private = socket_path.parent
    info = private.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise ValueError("Unix socket directory must be task-owned and mode 0700")
    if socket_path.exists():
        raise ValueError("Refuse a pre-existing socket path")
    output.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "metrics": {},
        "blockers": [],
    }
    write_json(output / "receipt.json", receipt)
    task, bound = None, False
    started = time.monotonic()
    try:
        env = make_original_env(config["host"])
        try:
            task = StateTask(env, config["host"], output)
        except BaseException:
            env.close()
            raise
        receipt["original_sources_assets"] = native_sources(output)
        write_json(output / "description.json", task.description)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(str(socket_path))
            bound = True
            os.chmod(socket_path, 0o600)
            listener.listen(1)
            listener.settimeout(1)
            write_json(
                output / "ready.json",
                {
                    "schema": SCHEMA,
                    "socket": str(socket_path),
                    "description": artifact(output / "description.json"),
                },
            )
            for _ in range(100000):
                if time.monotonic() - started >= config["timeouts"]["host_wall_s"]:
                    raise TimeoutError(
                        "Original ABC host exceeded its finite wall deadline"
                    )
                try:
                    connection, _ = listener.accept()
                except TimeoutError:
                    continue
                request, stop = None, False
                with connection:
                    connection.settimeout(config["timeouts"]["rpc_s"])
                    try:
                        request = json.loads(
                            receive_frame(connection, config["timeouts"]["rpc_s"])
                        )
                        finite_json(request)
                        response, stop = dispatch(task, request)
                    except (ValueError, TypeError, KeyError) as error:
                        response = {
                            "schema": SCHEMA,
                            "id": request.get("id") if type(request) is dict else None,
                            "ok": False,
                            "error": str(error),
                        }
                    body = (
                        json.dumps(
                            response, allow_nan=False, separators=(",", ":")
                        ).encode()
                        + b"\n"
                    )
                    if len(body) > MAX_FRAME_BYTES:
                        raise RuntimeError(
                            "Original ABC response exceeds the 4 MiB limit"
                        )
                    connection.sendall(body)
                if stop:
                    break
            else:
                raise TimeoutError("Original ABC host exceeded its request bound")
        receipt["status"] = "completed"
    except (Exception, KeyboardInterrupt) as error:
        receipt["status"] = (
            "cancelled" if isinstance(error, KeyboardInterrupt) else "failed"
        )
        receipt["blockers"] = [f"{type(error).__name__}: {error}"]
        raise
    finally:
        if task is not None:
            try:
                task.close()
            except (OSError, RuntimeError, ValueError, TypeError) as error:
                receipt["status"] = "failed"
                receipt["blockers"].append(
                    f"Host cleanup {type(error).__name__}: {error}"
                )
            receipt["metrics"] = {
                "controls": task.controls,
                "completed_episodes": sum(e["completed"] for e in task.episodes),
                "normalized_action_clipping_count": task.clipped_actions,
            }
            receipt["reset_seeds"] = task.reset_seeds
            receipt["episodes"] = task.episodes
        if bound:
            socket_path.unlink(missing_ok=True)
        receipt["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "receipt.json", receipt)
    if receipt["status"] == "failed":
        raise RuntimeError("Original ABC host cleanup failed; see receipt.json")


def learner_command(
    config: dict[str, Any], socket_path: Path, output: Path
) -> list[str]:
    learner = config["learner"]
    command = [
        config["learner_python"],
        "-I",
        "-u",
        "-B",
        "-m",
        "sadhana_fasttd3_abc.cli",
        config["stage"],
        "--socket",
        str(socket_path),
        "--output",
        str(output),
        "--device",
        "cuda",
    ]
    if config["stage"] == "train":
        for option, key in (
            ("--steps", "steps"),
            ("--seed", "seed"),
            ("--layout-seed", "layout_seed"),
            ("--batch-size", "batch_size"),
            ("--buffer-size", "buffer_size"),
        ):
            command.extend((option, str(learner[key])))
        command.extend(("--num-envs", "1"))
        for name in ("compile", "amp"):
            if learner[name]:
                command.append("--" + name)
    else:
        command.extend(
            (
                "--checkpoint",
                learner["checkpoint"],
                "--seeds",
                *map(str, learner["seeds"]),
            )
        )
    return command


def learner_snapshot(python: str, output: Path, timeout: int) -> dict[str, Any]:
    code = """import hashlib, importlib.metadata, importlib.util, json, pathlib
result = {'distributions': {}, 'modules': []}
for name in ('torch', 'numpy', 'tensordict', 'fast-td3', 'sadhana-fasttd3-author-abc'):
    result['distributions'][name] = importlib.metadata.version(name)
for name in ('fast_td3', 'sadhana_fasttd3_abc'):
    spec = importlib.util.find_spec(name)
    for directory in spec.submodule_search_locations:
        for path in sorted(pathlib.Path(directory).rglob('*.py')):
            raw = path.read_bytes()
            result['modules'].append({'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
print(json.dumps(result, allow_nan=False))
"""
    result = subprocess.run(
        [python, "-I", "-B", "-c", code],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    snapshot = json.loads(result.stdout)
    write_json(output / "learner_modules.json", snapshot)
    return artifact(output / "learner_modules.json")


def learner_metrics(result: dict[str, Any], stage: str) -> dict[str, Any]:
    """Expose only counters the actual installed learner receipt recorded."""
    if stage == "train":
        metrics = {
            "controls": result["controls"],
            "actor_updates": result["actual_updates"]["actor"],
            "critic_updates": result["actual_updates"]["critic"],
        }
        for name, value in metrics.items():
            bounded_int(value, name, 1, 1000000)
        return metrics
    episodes = result["episodes"]
    return {
        "num_worlds": len(episodes),
        "successes": result["successes"],
        "matched_worlds": sum(
            episode.get("scene_match") == "matched_baseline" for episode in episodes
        ),
        "actor_updates": result["actor_updates"],
        "critic_updates": result["critic_updates"],
    }


def supervise(config_path: Path, output: Path) -> dict[str, Any]:
    config = load_config(config_path)
    require_subset_lease()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "receipt.json").exists():
        raise ValueError("Refuse to overwrite an existing FastTD3 worker receipt")
    private = output / "rpc"
    private.mkdir(mode=0o700)
    socket_path, host_out, learner_out = (
        private / "host.sock",
        output / "host",
        output / "learner",
    )
    if len(os.fsencode(socket_path.resolve())) > 107:
        raise ValueError(
            "Coordinator output path makes the Unix socket longer than 107 bytes"
        )
    write_json(output / "resolved_config.json", config)
    shutil.copyfile(Path(__file__), output / "worker_module.py")
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "algorithm": "original FastTD3 from scratch",
        "phase": config["stage"],
        "claim_scope": "One-world privileged-state diagnostic; not full paper reproduction",
        "input_config": artifact(config_path),
        "resolved_config": artifact(output / "resolved_config.json"),
        "worker_module": artifact(output / "worker_module.py"),
        "metrics": {},
        "blockers": [],
    }
    write_json(output / "receipt.json", receipt)
    started, host, learner = time.monotonic(), None, None
    try:
        if config["stage"] == "evaluate":
            pin = artifact(Path(config["learner"]["checkpoint"]))
            if pin["sha256"] != config["learner"]["checkpoint_sha256"]:
                raise ValueError(
                    "Frozen FastTD3 checkpoint differs from its SHA-256 pin"
                )
            receipt["evaluation_checkpoint"] = pin
        receipt["learner_module_snapshot"] = learner_snapshot(
            config["learner_python"], output, 30
        )
        host_command = [
            sys.executable,
            "-I",
            "-u",
            "-B",
            "-m",
            "abc_bench.fasttd3_host",
            "--serve",
            "--config",
            str((output / "resolved_config.json").resolve()),
            "--socket",
            str(socket_path.resolve()),
            "--output",
            str(host_out.resolve()),
        ]
        command = learner_command(config, socket_path.resolve(), learner_out.resolve())
        receipt["commands"] = {"host": host_command, "learner": command}
        child_env = os.environ.copy()
        child_overrides = {
            "WANDB_MODE": "disabled",
            "TORCHINDUCTOR_CACHE_DIR": str((output / "cache" / "inductor").resolve()),
            "TRITON_CACHE_DIR": str((output / "cache" / "triton").resolve()),
            "CUDA_CACHE_PATH": str((output / "cache" / "cuda").resolve()),
        }
        child_env.update(child_overrides)
        receipt["child_environment_overrides"] = child_overrides
        write_json(output / "receipt.json", receipt)
        with (
            (output / "host.log").open("w") as host_log,
            (output / "learner.log").open("w") as learner_log,
        ):
            # Both children inherit the coordinator process group and GPU visibility.
            host = subprocess.Popen(
                host_command, stdout=host_log, stderr=subprocess.STDOUT
            )
            deadline = time.monotonic() + config["timeouts"]["host_ready_s"]
            while not (host_out / "ready.json").is_file():
                if host.poll() is not None:
                    raise RuntimeError(
                        f"Original ABC host exited {host.returncode}; see host.log"
                    )
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Original ABC host did not become ready by its deadline"
                    )
                time.sleep(0.1)
            receipt["host_description"] = artifact(host_out / "description.json")
            learner = subprocess.Popen(
                command, stdout=learner_log, stderr=subprocess.STDOUT, env=child_env
            )
            code = learner.wait(timeout=config["timeouts"]["learner_s"])
            if code != 0:
                raise RuntimeError(
                    f"Original FastTD3 learner exited {code}; see learner.log"
                )
            child_receipt = learner_out / "receipt.json"
            result = json.loads(child_receipt.read_text())
            if result.get("status") != "completed":
                raise RuntimeError(
                    "Original FastTD3 learner did not complete its declared stage"
                )
            rpc_call(socket_path.resolve(), "shutdown", config["timeouts"]["rpc_s"])
            if host.wait(timeout=config["timeouts"]["shutdown_s"]) != 0:
                raise RuntimeError("Original ABC host did not shut down cleanly")
            host_receipt = json.loads((host_out / "receipt.json").read_text())
            if host_receipt.get("status") != "completed":
                raise RuntimeError("Original ABC host receipt did not complete")
            receipt["learner_receipt"] = artifact(child_receipt)
            receipt["host_receipt"] = artifact(host_out / "receipt.json")
            receipt["metrics"] = learner_metrics(result, config["stage"])
            receipt["host_metrics"] = host_receipt["metrics"]
            receipt["status"] = "completed"
    except (Exception, KeyboardInterrupt) as error:
        receipt["status"] = (
            "cancelled" if isinstance(error, KeyboardInterrupt) else "failed"
        )
        receipt["blockers"] = [f"{type(error).__name__}: {error}"]
        raise
    finally:
        for child in (learner, host):
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
        receipt["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--socket", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.serve:
        if args.socket is None:
            parser.error("Internal server mode requires --socket")
        serve(args.config, args.socket, args.output)
    else:
        if args.socket is not None:
            parser.error("The supervisor owns the socket path")
        supervise(args.config, args.output)


if __name__ == "__main__":
    main()
