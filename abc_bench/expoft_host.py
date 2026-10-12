"""Lease-guarded original ABC RGB host for the separate EXPO-FT runtime.

The host keeps original sequential MuJoCo physics and MJWarp cameras. It adds
bounded RPC, explicit action clipping, and separate training/evaluation resets.
It has no learner, VLA implementation, privileged observation, or auto-reset.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import signal
import socket
import stat
import struct
import time
from pathlib import Path
from typing import Any

from abc_bench.fasttd3_host import (
    ACTION_NAMES,
    RESET_OPTIONS,
    bounded_int,
    finite_json,
    make_original_env,
    native_sources,
    numeric_vector,
    reset_original_world,
    write_json,
)
from abc_bench.subset import require_subset_lease
from abc_bench.upstream_eval import artifact

SCHEMA = "sadhana.expoft.abc.v1"
MAX_FRAME_BYTES = 4 * 1024**2
CAMERAS = ("top", "left", "right")
CAMERA_CHW = (3, 168, 224)
CAMERA_HWC = (168, 224, 3)


class HostDeadlineError(TimeoutError):
    """The wall/RPC watchdog fired; the simulator must not receive more controls."""


def read_reference(reference: Any) -> dict[str, Any]:
    """Read and verify the captured evaluation scene before any native import."""
    if type(reference) is not dict or set(reference) != {
        "summary",
        "sha256",
        "world_index",
    }:
        raise ValueError("reference requires summary, sha256, world_index")
    if type(reference["summary"]) is not str:
        raise ValueError("reference.summary must be an absolute file")
    path = Path(reference["summary"])
    if (
        not path.is_absolute()
        or not path.is_file()
        or path.stat().st_size > 16 * 1024**2
    ):
        raise ValueError("reference.summary must be an absolute file of at most 16 MiB")
    if artifact(path)["sha256"] != reference["sha256"]:
        raise ValueError("The captured baseline summary differs from its SHA256 pin")
    summary = json.loads(path.read_text())
    finite_json(summary)
    if summary.get("task") != "put_plastic_bottles_in_bin":
        raise ValueError("The reference summary must describe the original bottle task")
    index = bounded_int(reference["world_index"], "world_index", 0, 4)
    world = summary["worlds"][index]
    state = world["randomization"]
    if type(state) is not dict or set(state) != {
        "seed",
        "object_states",
        "scale_states",
        "metadata",
    }:
        raise ValueError("The reference must contain a complete RandomizationState")
    bounded_int(state["seed"], "reference seed", 0, 2**31 - 1)
    if type(world.get("world_seed")) is not int or world["world_seed"] != state["seed"]:
        raise ValueError("The selected world seed differs from the captured scene seed")
    if state["metadata"].get("bottle_count") != 5:
        raise ValueError("The captured evaluation scene must contain five bottles")
    return state


def load_config(path: Path) -> dict[str, Any]:
    if path.stat().st_size > 1024**2:
        raise ValueError("The host config exceeds 1 MiB")
    config = json.loads(path.read_text())
    finite_json(config)
    if type(config) is not dict or set(config) != {
        "schema_version",
        "stage",
        "host",
        "timeouts",
    }:
        raise ValueError("Use the explicit EXPO host configuration")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    if config["stage"] not in ("train", "evaluation"):
        raise ValueError("stage must be train or evaluation")
    host = config["host"]
    if type(host) is not dict or set(host) != {
        "num_envs",
        "initial_seed",
        "allowed_seeds",
        "max_episode_steps",
        "save_video",
        "reference",
    }:
        raise ValueError("host fields differ from the explicit one-world contract")
    bounded_int(host["num_envs"], "num_envs", 1, 1)
    bounded_int(host["max_episode_steps"], "max_episode_steps", 1, 3540)
    if type(host["save_video"]) is not bool:
        raise ValueError("save_video must be a boolean")
    seeds = host["allowed_seeds"]
    if type(seeds) is not list or not 1 <= len(seeds) <= 64:
        raise ValueError("allowed_seeds must explicitly contain 1 to 64 seeds")
    for seed in seeds:
        bounded_int(seed, "allowed seed", 0, 2**31 - 1)
    if len(set(seeds)) != len(seeds):
        raise ValueError("allowed_seeds must be unique")
    initial = bounded_int(host["initial_seed"], "initial_seed", 0, 2**31 - 1)
    if initial not in seeds:
        raise ValueError("initial_seed must be explicitly allowed")
    reference = read_reference(host["reference"])
    if config["stage"] == "evaluation":
        if seeds != [reference["seed"]]:
            raise ValueError("Evaluation permits only the pinned captured scene seed")
    elif any(seed < 9030 or seed == reference["seed"] for seed in seeds):
        raise ValueError("Training seeds must be 9030+ and exclude the held-out scene")
    timeouts = config["timeouts"]
    if type(timeouts) is not dict or set(timeouts) != {
        "host_wall_s",
        "rpc_s",
        "max_requests",
    }:
        raise ValueError("Require explicit finite host, RPC, and request bounds")
    bounded_int(timeouts["host_wall_s"], "host_wall_s", 1, 7200)
    bounded_int(timeouts["rpc_s"], "rpc_s", 1, 300)
    bounded_int(timeouts["max_requests"], "max_requests", 1, 100000)
    return config


class ExpoTask:
    """One original world whose final scene remains available until reset."""

    def __init__(self, env: Any, config: dict[str, Any], output: Path):
        self.env, self.config, self.output = env, config, output
        self.reference = read_reference(config["host"]["reference"])
        self.episode: dict[str, Any] | None = None
        self.episodes: list[dict[str, Any]] = []
        self.reset_seeds: list[int] = []
        self.controls = self.clipped_actions = self.episode_steps = 0
        self.closed, self.writer = False, None
        self.reward, self.terminated, self.truncated = 0.0, False, False
        self.description: dict[str, Any] = {}
        self.reset(config["host"]["initial_seed"])

    def _describe(self) -> dict[str, Any]:
        if (
            self.env.robot_names != ["left", "right"]
            or self.env.single_timestep_action_dim != 14
        ):
            raise ValueError("Require original left/right YAM 14D policy coordinates")
        dt = float(self.env.model.opt.timestep)
        decimation = self.env._control_decimation
        if (
            not math.isclose(dt, 0.002)
            or type(decimation) is not int
            or decimation != 17
        ):
            raise ValueError("Original ABC physics must use dt=0.002 and decimation=17")
        low, high = [], []
        for name in ACTION_NAMES:
            limits = (
                [0.0, 1.0]
                if name.endswith("gripper")
                else numeric_vector(
                    self.env.model.joint(name).range, 2, f"{name} limits"
                )
            )
            if limits[0] >= limits[1]:
                raise ValueError(f"{name} requires finite physical limits")
            low.append(limits[0])
            high.append(limits[1])
        host = self.config["host"]
        return {
            "num_envs": 1,
            "num_actions": 14,
            "num_state": 14,
            "max_episode_steps": host["max_episode_steps"],
            "task": "put_plastic_bottles_in_bin",
            "embodiment": "native_yam",
            "prompt": self.env.prompt,
            "physics": "original sequential CPU MuJoCo",
            "camera_backend": "mjwarp",
            "physics_dt": dt,
            "control_decimation": decimation,
            "control_hz": 1.0 / (dt * decimation),
            "autoreset": False,
            "privileged_state": False,
            "state_schema": {
                "names": list(ACTION_NAMES),
                "shape": [14],
                "dtype": "float32",
                "source": "original env.get_state()",
                "gripper_indices": [6, 13],
            },
            "action_schema": {
                "names": list(ACTION_NAMES),
                "shape": [14],
                "mode": "physical_absolute14",
                "dtype": "float32",
                "casting": "Original env.step converts policy coordinates to float32",
                "policy_low": low,
                "policy_high": high,
                "joint_units": "radians",
                "gripper_policy_units": "fraction in [0,1]",
                "gripper_physical_range_m": [0.0, 0.0475],
                "clipping": "Host clips to original joint ranges and gripper[0,1]; every change is recorded",
                "original_difference": "Original ABC policy dispatch does not clip these 14D proposals",
                "replay_action": "action_applied14",
            },
            "camera_schema": {
                "names": list(CAMERAS),
                "shape": list(CAMERA_HWC),
                "dtype": "uint8",
                "encoding": "base64_raw",
                "color": "RGB",
            },
            "reset_policy": {
                "stage": self.config["stage"],
                "allowed_seeds": host["allowed_seeds"],
                "heldout_seed": self.reference["seed"],
                "reference": host["reference"],
                "distribution": "captured baseline default sampler, exact state verification"
                if self.config["stage"] == "evaluation"
                else "original fixed-five diagnostic",
                "train_reset_options": RESET_OPTIONS
                if self.config["stage"] == "train"
                else None,
                "implicit_resampling": False,
            },
            "physics_step_source": "control_step * control_decimation; checked against original capture_state.time",
        }

    def _snapshot(self) -> dict[str, Any]:
        if self.closed or self.episode is None:
            raise RuntimeError("EXPO host has no open scene")
        state = self.env.get_state()
        if str(state.dtype) != "float32":
            raise RuntimeError("Original policy state must be float32")
        state14 = numeric_vector(state, 14, "original policy state")
        if any(not 0.0 <= state14[index] <= 1.0 for index in (6, 13)):
            raise RuntimeError("Original projected gripper state must be in [0,1]")
        sim_time = float(self.env.capture_state()["time"])
        expected_time = self.episode_steps * 0.034
        if self.env.cur_step != self.episode_steps or not math.isclose(
            sim_time, expected_time, rel_tol=0.0, abs_tol=2e-5
        ):
            raise RuntimeError(
                "Original simulator chronology differs from applied controls"
            )
        return {
            "state14": state14,
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "seed": self.episode["seed"],
            "control_step": self.episode_steps,
            "physics_step": self.episode_steps * 17,
            "simulation_time_s": sim_time,
        }

    def _images(self, raw: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
        if (
            numeric_vector(raw["state"], 14, "camera policy state")
            != snapshot["state14"]
        ):
            raise RuntimeError(
                "Camera state differs from the current original policy state"
            )
        images, encoded = raw["images"], {}
        if set(images) != set(CAMERAS) or set(raw["camera_timestamps"]) != set(CAMERAS):
            raise RuntimeError("Require actual original top/left/right RGB cameras")
        for name in CAMERAS:
            image = images[name]
            if (
                tuple(image.shape) != CAMERA_CHW
                or str(image.dtype) != "uint8"
                or raw["masks"].get(name) is not True
            ):
                raise RuntimeError(f"Camera {name} must be valid uint8 CHW[3,168,224]")
            stamp = float(raw["camera_timestamps"][name])
            if not math.isclose(
                stamp, snapshot["simulation_time_s"], rel_tol=0.0, abs_tol=2e-5
            ):
                raise RuntimeError(f"Camera {name} is not at the current physics time")
            payload = image.transpose(1, 2, 0).tobytes(order="C")
            if len(payload) != math.prod(CAMERA_HWC):
                raise RuntimeError(f"Camera {name} raw byte length is invalid")
            encoded[name] = {
                "data_base64": base64.b64encode(payload).decode("ascii"),
                "shape": list(CAMERA_HWC),
                "dtype": "uint8",
                "encoding": "base64_raw",
                "timestamp_s": stamp,
            }
        return encoded

    def _video_frame(self, raw: dict[str, Any]) -> None:
        if self.writer is not None:
            import numpy as np

            self.writer.append_data(
                np.concatenate(
                    [raw["images"][name].transpose(1, 2, 0) for name in CAMERAS], axis=1
                )
            )

    def _finish(self, reason: str) -> None:
        if self.episode is None or self.episode.get("end_reason") is not None:
            return
        if self.writer is not None:
            try:
                self.writer.close()
            finally:
                self.writer = None
            self.episode["video"] = artifact(Path(self.episode.pop("video_path")))
        self.episode["end_reason"] = reason
        self.episode["completed"] = self.terminated or self.truncated
        self.episodes.append(dict(self.episode))
        write_json(self.output / f"episode-{len(self.episodes):04d}.json", self.episode)

    def reset(self, seed: Any) -> dict[str, Any]:
        if self.closed:
            raise RuntimeError("EXPO host is closed")
        seed = bounded_int(seed, "reset seed", 0, 2**31 - 1)
        if seed not in self.config["host"]["allowed_seeds"]:
            raise ValueError(
                "Reset seed is outside the explicit train/evaluation domain"
            )
        self._finish("explicit_reset")
        reference = self.reference if self.config["stage"] == "evaluation" else None
        try:
            raw, randomization, matched = reset_original_world(
                self.env, seed, reference
            )
        except Exception as error:
            raise RuntimeError(f"Original reset failed: {error}") from error
        if matched != (self.config["stage"] == "evaluation"):
            raise RuntimeError("Reset did not honor the explicit scene domain")
        if randomization.get("seed") != seed:
            raise RuntimeError("Original reset resampled the requested seed")
        evaluation = self.env.evaluate_task().to_info(squeeze=True)
        finite_json(evaluation)
        if evaluation.get("num_active_bottles") != 5:
            raise RuntimeError(
                "The original scene must contain exactly five active bottles"
            )
        try:
            description = self._describe()
        except (ValueError, KeyError, TypeError) as error:
            raise RuntimeError(f"Original reset schema is invalid: {error}") from error
        if (
            self.description
            and description["action_schema"] != self.description["action_schema"]
        ):
            raise RuntimeError("Original action bounds changed across resets")
        self.description = description
        self.reset_seeds.append(seed)
        self.episode_steps = 0
        self.reward, self.terminated, self.truncated = (
            float(evaluation["reward"]),
            False,
            False,
        )
        if not math.isfinite(self.reward):
            raise ValueError("Original reset reward is not finite")
        self.episode = {
            "seed": seed,
            "controls": 0,
            "completed": False,
            "success": False,
            "action_clipping_count": 0,
            "max_native_reward": self.reward,
            "scene_match": "matched_baseline" if matched else "fixed_five_train",
            "randomization": randomization,
            "final_native_task_eval": evaluation,
        }
        snapshot = self._snapshot()
        self._images(raw, snapshot)
        write_json(
            self.output / f"reset-{len(self.reset_seeds):04d}.json",
            {
                **snapshot,
                "scene_match": self.episode["scene_match"],
                "randomization": randomization,
            },
        )
        if self.config["host"]["save_video"]:
            import imageio.v2 as imageio

            video = self.output / f"reset-{len(self.reset_seeds):04d}-seed-{seed}.mp4"
            self.episode["video_path"] = str(video)
            self.writer = imageio.get_writer(
                str(video), fps=1 / 0.034, macro_block_size=1
            )
            self._video_frame(raw)
        return {**snapshot, "scene_match": self.episode["scene_match"]}

    def step(self, action: Any) -> dict[str, Any]:
        if self.closed or self.episode is None or self.terminated or self.truncated:
            raise ValueError(
                "Reset is required before another step; final scene is preserved"
            )
        requested = numeric_vector(action, 14, "physical absolute action")
        schema = self.description["action_schema"]
        clipped = [
            min(high, max(low, value))
            for value, low, high in zip(
                requested, schema["policy_low"], schema["policy_high"]
            )
        ]
        applied = list(struct.unpack("<14f", struct.pack("<14f", *clipped)))
        changes = [
            {
                "index": i,
                "name": ACTION_NAMES[i],
                "requested": value,
                "applied": applied[i],
            }
            for i, value in enumerate(requested)
            if value != clipped[i]
        ]
        try:
            _, reward, terminated, truncated, info = self.env.step(
                applied, render_obs=False
            )
        except Exception as error:
            raise RuntimeError(f"Original step failed: {error}") from error
        if type(reward) not in (int, float) or not math.isfinite(reward):
            raise RuntimeError("Original step reward is not finite")
        if (
            type(terminated) is not bool
            or type(truncated) is not bool
            or terminated
            and truncated
        ):
            raise RuntimeError("Original termination/truncation flags are invalid")
        if type(info) is not dict or type(info.get("task_eval")) is not dict:
            raise RuntimeError("Original step must return native task evaluation")
        evaluation = info["task_eval"]
        if type(evaluation.get("success")) is not bool:
            raise RuntimeError("Original native task success must be a boolean")
        finite_json(evaluation)
        next_step = self.episode_steps + 1
        expected_truncation = (
            next_step >= self.config["host"]["max_episode_steps"] and not terminated
        )
        if (
            terminated != bool(evaluation["success"])
            or truncated != expected_truncation
        ):
            raise RuntimeError(
                "Original success/timeout flags disagree with the configured task"
            )
        self.controls += 1
        self.episode_steps += 1
        self.clipped_actions += len(changes)
        self.reward, self.terminated, self.truncated = (
            float(reward),
            terminated,
            truncated,
        )
        snapshot = self._snapshot()
        self.episode.update(
            controls=self.episode_steps,
            action_clipping_count=self.episode["action_clipping_count"] + len(changes),
            success=bool(evaluation["success"]),
            final_native_task_eval=evaluation,
            max_native_reward=max(self.episode["max_native_reward"], self.reward),
        )
        if self.writer is not None:
            raw = self.env.get_obs()
            self._images(raw, snapshot)
            self._video_frame(raw)
        if terminated or truncated:
            self._finish("success" if terminated else "timeout")
        return {
            **snapshot,
            "action_requested14": requested,
            "action_applied14": applied,
            "action_clipping": changes,
            "task_eval": evaluation,
        }

    def observe_rgb(self) -> dict[str, Any]:
        snapshot = self._snapshot()
        return {**snapshot, "cameras": self._images(self.env.get_obs(), snapshot)}

    def close(self) -> None:
        if not self.closed:
            try:
                self._finish("host_closed")
            finally:
                self.closed = True
                self.env.close()


def dispatch(task: ExpoTask, request: Any) -> tuple[dict[str, Any], bool]:
    if type(request) is not dict or request.get("schema") != SCHEMA:
        raise ValueError(f"RPC schema must be {SCHEMA}")
    bounded_int(request.get("id"), "request id", 1, 2**63 - 1)
    operation = request.get("operation")
    fields = {
        "describe": set(),
        "reset": {"seed"},
        "step": {"action"},
        "observe_rgb": set(),
        "close": set(),
    }
    if (
        operation not in fields
        or set(request) != {"schema", "id", "operation"} | fields[operation]
    ):
        raise ValueError("RPC operation or fields are invalid")
    finite_json(request)
    if operation == "describe":
        result = task.description
    elif operation == "reset":
        result = task.reset(request["seed"])
    elif operation == "step":
        result = task.step(request["action"])
    elif operation == "observe_rgb":
        result = task.observe_rgb()
    else:
        result = {"status": "closing"}
    return {
        "schema": SCHEMA,
        "id": request["id"],
        "ok": True,
        "result": result,
    }, operation == "close"


def receive_frame(connection: socket.socket, deadline: float) -> bytes:
    frame = bytearray()
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("RPC frame exceeded its finite deadline")
        connection.settimeout(remaining)
        chunk = connection.recv(min(65536, MAX_FRAME_BYTES + 1 - len(frame)))
        if not chunk:
            raise ValueError("RPC connection ended before a newline-delimited frame")
        frame.extend(chunk)
        if len(frame) > MAX_FRAME_BYTES:
            raise ValueError("RPC frame exceeds 4 MiB")
        if b"\n" in frame:
            if frame.count(b"\n") != 1 or not frame.endswith(b"\n"):
                raise ValueError("One JSON frame is allowed per connection")
            return bytes(frame)


def serve(config_path: Path, socket_path: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if (output / "receipt.json").exists():
        raise ValueError("Refuse to overwrite an existing EXPO host receipt")
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "metrics": {},
        "blockers": [],
    }
    task, bound = None, False
    started = time.monotonic()
    old_handlers: dict[int, Any] = {}
    previous_timer = (0.0, 0.0)
    try:
        config = load_config(config_path)
        receipt.update(
            config=artifact(config_path),
            reference=config["host"]["reference"],
            stage=config["stage"],
        )
        require_subset_lease()  # Must precede factory and every native dependency import.
        if not socket_path.is_absolute() or len(os.fsencode(socket_path)) > 107:
            raise ValueError("Use an absolute Unix socket path of at most 107 bytes")
        private = socket_path.parent.stat()
        if private.st_uid != os.getuid() or stat.S_IMODE(private.st_mode) != 0o700:
            raise ValueError("Unix socket directory must be task-owned and mode 0700")
        if socket_path.exists() or socket_path.is_symlink():
            raise ValueError("Refuse a pre-existing socket path")

        def interrupt(signum: int, _frame: Any) -> None:
            if signum == signal.SIGALRM:
                raise HostDeadlineError(
                    "EXPO ABC host exceeded its finite wall/RPC deadline"
                )
            raise KeyboardInterrupt(f"EXPO ABC host cancelled by signal {signum}")

        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM):
            old_handlers[signum] = signal.signal(signum, interrupt)
        previous_timer = signal.setitimer(
            signal.ITIMER_REAL, config["timeouts"]["host_wall_s"]
        )
        wall_deadline = time.monotonic() + config["timeouts"]["host_wall_s"]
        write_json(output / "receipt.json", receipt)
        env = make_original_env(
            {
                "save_video": True,
                "max_episode_steps": config["host"]["max_episode_steps"],
            }
        )
        try:
            task = ExpoTask(env, config, output)
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
            count, stop = 0, False
            while not stop:
                if count >= config["timeouts"]["max_requests"]:
                    raise TimeoutError(
                        "EXPO ABC host exceeded its finite request bound"
                    )
                try:
                    connection, _ = listener.accept()
                except HostDeadlineError:
                    raise
                except TimeoutError:
                    continue
                count += 1
                require_subset_lease()
                request = None
                with connection:
                    deadline = min(
                        wall_deadline, time.monotonic() + config["timeouts"]["rpc_s"]
                    )
                    signal.setitimer(
                        signal.ITIMER_REAL, max(0.001, deadline - time.monotonic())
                    )
                    try:
                        request = json.loads(receive_frame(connection, deadline))
                        response, stop = dispatch(task, request)
                    except (
                        ValueError,
                        TypeError,
                        KeyError,
                        RuntimeError,
                        TimeoutError,
                    ) as error:
                        response = {
                            "schema": SCHEMA,
                            "id": request.get("id") if type(request) is dict else None,
                            "ok": False,
                            "error": {
                                "type": type(error).__name__,
                                "message": str(error),
                            },
                        }
                        # A native state/chronology failure must not allow more controls.
                        stop = isinstance(error, (RuntimeError, HostDeadlineError))
                        if stop:
                            receipt["blockers"].append(
                                f"{type(error).__name__}: {error}"
                            )
                    payload = (
                        json.dumps(
                            response, allow_nan=False, separators=(",", ":")
                        ).encode()
                        + b"\n"
                    )
                    if len(payload) > MAX_FRAME_BYTES:
                        raise RuntimeError("EXPO host response exceeds the 4 MiB limit")
                    connection.settimeout(max(0.001, deadline - time.monotonic()))
                    connection.sendall(payload)
                    signal.setitimer(
                        signal.ITIMER_REAL, max(0.001, wall_deadline - time.monotonic())
                    )
        receipt["status"] = "failed" if receipt["blockers"] else "completed"
    except (Exception, KeyboardInterrupt) as error:
        receipt["status"] = (
            "cancelled" if isinstance(error, KeyboardInterrupt) else "failed"
        )
        receipt["blockers"].append(f"{type(error).__name__}: {error}")
        raise
    finally:
        if old_handlers:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        try:
            if task is not None:
                try:
                    task.close()
                finally:
                    receipt.update(reset_seeds=task.reset_seeds, episodes=task.episodes)
                    receipt["metrics"] = {
                        "controls": task.controls,
                        "completed_episodes": sum(
                            item["completed"] for item in task.episodes
                        ),
                        "action_clipping_count": task.clipped_actions,
                    }
        except Exception as error:
            receipt["status"] = "failed"
            receipt["blockers"].append(f"Cleanup {type(error).__name__}: {error}")
            raise
        finally:
            if bound:
                socket_path.unlink(missing_ok=True)
            receipt["elapsed_seconds"] = time.monotonic() - started
            write_json(output / "receipt.json", receipt)
    if receipt["status"] != "completed":
        raise RuntimeError("EXPO host stopped after a recorded failure")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    serve(args.config, args.socket, args.output)


if __name__ == "__main__":
    main()
