"""Independent original CPU MuJoCo worlds for the FastTD3 vector boundary.

This module imports no simulator at import time. Native training must hold the
canonical lease. The separate CPU acceptance command hides CUDA and renders no
cameras. The original reward, control step and success judge are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
import resource
import signal
import socket
import stat
import sys
import time
from datetime import datetime, timezone
from multiprocessing.connection import wait
from pathlib import Path
from typing import Any

from abc_bench.fasttd3_host import receive_frame, write_json
from abc_bench.subset import require_subset_lease
from abc_bench.upstream_eval import artifact

SCHEMA = "sadhana.fasttd3.abc.vector.v1"
TASK = "put_plastic_bottles_in_bin"
PROMPT = "sim throw plastic bottles in bin"
NUM_OBS = 216
NUM_ACTIONS = 14
MAX_FRAME_BYTES = 4 * 1024**2
VARIANTS = tuple(f"bottle_{index}" for index in range(19))
CHANNELS = tuple(
    item
    for arm in ("left", "right")
    for item in (*[f"{arm}_joint{index}" for index in range(1, 7)], f"{arm}_gripper")
)
RESET_OPTIONS = {
    "randomization": {
        "bottle_count": 5,
        "randomize_variants": True,
        "randomize_scales": True,
    }
}
EXCLUDED_SEEDS = {20260511, *range(9100, 9103), *range(109100, 109103)}


def integer(value: Any, name: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low},{high}]")
    return value


def validate_config(config: Any) -> dict[str, Any]:
    if type(config) is not dict or set(config) != {
        "schema_version",
        "profile",
        "num_envs",
        "seed_base",
        "seed_stride",
        "max_episode_steps",
        "vector_steps",
        "timeouts",
    }:
        raise ValueError("Expected the complete ABC vector configuration")
    integer(config["schema_version"], "schema_version", 1, 1)
    count = {"training128": 128, "cpu_acceptance2": 2}.get(config["profile"])
    if count is None:
        raise ValueError("profile must be training128 or cpu_acceptance2")
    integer(config["num_envs"], "num_envs", count, count)
    integer(config["seed_base"], "seed_base", 200000, 200000)
    integer(config["seed_stride"], "seed_stride", 128, 128)
    integer(config["max_episode_steps"], "max_episode_steps", 3540, 3540)
    integer(config["vector_steps"], "vector_steps", 2, 64)
    deadlines = config["timeouts"]
    if type(deadlines) is not dict or set(deadlines) != {
        "ready_s",
        "step_s",
        "shutdown_s",
        "wall_s",
        "rpc_s",
    }:
        raise ValueError("Explicit finite vector deadlines are required")
    for key, upper in {
        "ready_s": 600,
        "step_s": 180,
        "shutdown_s": 30,
        "wall_s": 3600,
        "rpc_s": 600,
    }.items():
        integer(deadlines[key], f"timeouts.{key}", 1, upper)
    if deadlines["ready_s"] + deadlines["shutdown_s"] >= deadlines["wall_s"]:
        raise ValueError("World startup and cleanup must fit within wall_s")
    if deadlines["step_s"] > deadlines["rpc_s"]:
        raise ValueError("step_s must fit within the total public RPC deadline")
    return config


def load_config(path: Path) -> dict[str, Any]:
    if path.stat().st_size > 1024**2:
        raise ValueError("Vector config must be at most 1 MiB")
    return validate_config(json.loads(path.read_text()))


def require_cpu() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        raise RuntimeError("ABC physics workers require hidden CUDA")
    if os.environ.get("MUJOCO_GL") != "disable":
        raise RuntimeError("ABC physics workers require MUJOCO_GL=disable")


def seed_for(config: dict[str, Any], rank: int, episode_index: int) -> int:
    integer(rank, "rank", 0, config["num_envs"] - 1)
    # Keep ordinary and +100000 fallback namespaces disjoint. Fail closed if
    # this finite qualification ever exhausts the declared reset bank.
    integer(episode_index, "episode_index", 0, 780)
    seed = config["seed_base"] + rank + config["seed_stride"] * episode_index
    if seed in EXCLUDED_SEEDS or seed + 100000 in EXCLUDED_SEEDS:
        raise ValueError("Training reset seed overlaps a reserved evaluation family")
    return seed


def plain(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {key: plain(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(child) for child in value]
    return value


def native_model_digest(model: Any) -> dict[str, Any]:
    """Hash all public NumPy model arrays, with field, dtype and shape bindings.

    This is a model-array fingerprint, not a claim of deterministic replay or
    an interchangeable binary MJB artifact.
    """
    import numpy as np

    digest = hashlib.sha256()
    fields = []
    for name in sorted(dir(model)):
        value = getattr(model, name)
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(f"{name}:{array.dtype}:{array.shape}".encode("ascii"))
            if array.size:
                digest.update(memoryview(array).cast("B"))
            fields.append(name)
    return {
        "sha256": digest.hexdigest(),
        "fields": fields,
        "format": "native-model-arrays-v1",
    }


class World:
    """One native model/data pair, owned only by its process."""

    def __init__(self, config: dict[str, Any], rank: int, output: Path) -> None:
        require_cpu()
        import abc_sim

        self.config, self.rank, self.output = config, rank, output
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.env = abc_sim.make_env(
            task=TASK,
            prompt=PROMPT,
            render_cameras=False,
            camera_backend="mujoco",
            camera_height=1,
            camera_width=1,
            max_episode_steps=config["max_episode_steps"],
            terminate_on_success=True,
        )
        self.episode_index, self.controls = -1, 0
        self.closed = False
        self.schema = None
        try:
            self.reset()
        except BaseException:
            self.env.close()
            raise

    def description(self) -> dict[str, Any]:
        model = self.env.model
        if self.env.robot_names != ["left", "right"] or self.env.action_dim != 14:
            raise ValueError("Original ABC world does not expose left/right YAM14")
        low, high = [], []
        for name in CHANNELS:
            bounds = (
                [0.0, 1.0]
                if name.endswith("gripper")
                else plain(model.joint(name).range)
            )
            low.append(float(bounds[0]))
            high.append(float(bounds[1]))
        if any(not math.isfinite(x) for x in low + high) or any(
            a >= b for a, b in zip(low, high)
        ):
            raise ValueError("Original action bounds are invalid")
        if int(model.nq) + int(model.nv) + 106 != NUM_OBS:
            raise ValueError("Original randomized five-bottle observation is not216D")
        if float(model.opt.timestep) != 0.002 or self.env._control_decimation != 17:
            raise ValueError("Original ABC physics timestep/decimation changed")
        return {
            "num_obs": NUM_OBS,
            "num_actions": NUM_ACTIONS,
            "nq": int(model.nq),
            "nv": int(model.nv),
            "joint_names": [model.joint(i).name for i in range(model.njnt)],
            "physics_dt": 0.002,
            "control_decimation": 17,
            "action_schema": {
                "names": list(CHANNELS),
                "policy_low": low,
                "policy_high": high,
                "ctrl_indices": [int(model.actuator(name).id) for name in CHANNELS],
                "mapping": "low + (clip(u,-1,1)+1)*0.5*(high-low)",
                "joint_units": "radians",
                "gripper_units": "fraction[0,1]",
                "gripper_physical_range_m": [0.0, 0.0475],
            },
        }

    def observation(self) -> list[float]:
        import numpy as np

        state = self.env.capture_state()
        variants = self.randomization["metadata"]["bottle_variants"]
        if len(variants) != 5 or any(name not in VARIANTS for name in variants):
            raise ValueError("Original variants differ from the declared vocabulary")
        scales = self.randomization["scale_states"]
        values = np.concatenate(
            (
                state["qpos"],
                state["qvel"],
                np.ones(5),
                [float(variant == name) for variant in variants for name in VARIANTS],
                [
                    float(scales.get(name, 1.0))
                    for name in (
                        *[f"bottle_{i}_joint" for i in range(1, 6)],
                        "bin_joint",
                    )
                ],
            )
        ).astype(np.float32)
        if values.shape != (NUM_OBS,) or not np.isfinite(values).all():
            raise ValueError("Original native observation is not finite float32[216]")
        return values.tolist()

    def reset(self) -> dict[str, Any]:
        if self.closed:
            raise RuntimeError("ABC world is closed")
        from abc_sim.randomization import RandomizationSamplingError

        self.episode_index += 1
        requested = seed_for(self.config, self.rank, self.episode_index)
        attempts = []
        for attempt, actual in enumerate((requested, requested, requested + 100000)):
            if actual in EXCLUDED_SEEDS:
                raise ValueError(
                    "Actual reset seed overlaps a reserved evaluation family"
                )
            if attempt == 1:
                # Exact original rollout_worlds failure fallback; ordinary
                # resets keep the carried arm pose.
                self.env.forget_arm_state()
            row = {"seed": actual, "from_init_q": attempt > 0}
            try:
                _, info = self.env.reset(
                    seed=actual, options=RESET_OPTIONS, randomize=True
                )
            except RandomizationSamplingError as error:
                row.update(status="placement_failed", error=str(error))
                attempts.append(row)
                if attempt == 2:
                    write_json(
                        self.output / f"failed-reset-{self.episode_index:04d}.json",
                        {
                            "rank": self.rank,
                            "requested_seed": requested,
                            "attempts": attempts,
                        },
                    )
                    raise
            else:
                row["status"] = "completed"
                attempts.append(row)
                break
        self.randomization = info["randomization"].to_dict()
        if (
            self.randomization["seed"] != actual
            or self.randomization["metadata"].get("bottle_count") != 5
        ):
            raise ValueError("Original reset seed or active bottle count differs")
        schema = self.description()
        if self.schema is not None and schema != self.schema:
            raise ValueError("Original observation or actuator schema changed at reset")
        self.schema = schema
        self.episode_steps = 0
        self.requested_seed, self.actual_seed = requested, actual
        self.reset_attempts = attempts
        snapshot = self.snapshot(f"reset-{self.episode_index:04d}")
        self.initial_obs = self.observation()
        return {
            "obs": self.initial_obs,
            "rank": self.rank,
            "episode_index": self.episode_index,
            "requested_seed": requested,
            "actual_seed": actual,
            "attempts": attempts,
            "scene": artifact(
                self.output / f"reset-{self.episode_index:04d}-scene.json"
            ),
            "native_policy_state": snapshot["native_policy_state"],
        }

    def snapshot(self, stem: str) -> dict[str, Any]:
        import mujoco
        import numpy as np

        xml = self.output / f"{stem}.xml"
        integration = self.output / f"{stem}-integration.npy"
        randomization = self.output / f"{stem}-randomization.json"
        mujoco.mj_saveLastXML(str(xml), self.env.model)
        specification = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(
            mujoco.mj_stateSize(self.env.model, specification), dtype=np.float64
        )
        mujoco.mj_getState(self.env.model, self.env.data, state, specification)
        with integration.open("xb") as stream:
            np.save(stream, state, allow_pickle=False)
        write_json(randomization, self.randomization)
        row = {
            "scene_xml": artifact(xml),
            "created_at": self.created_at,
            "initial_integration": artifact(integration),
            "randomization": artifact(randomization),
            "model_arrays": native_model_digest(self.env.model),
            "rank": self.rank,
            "requested_seed": self.requested_seed,
            "actual_seed": self.actual_seed,
            "episode_index": self.episode_index,
            "native_policy_state": plain(self.env.get_state()),
        }
        write_json(self.output / f"{stem}-scene.json", row)
        return row

    def step(self, normalized: Any) -> dict[str, Any]:
        import numpy as np

        if self.closed:
            raise RuntimeError("ABC world is closed")
        action = np.asarray(normalized)
        if (
            action.shape != (14,)
            or action.dtype.kind not in "fiu"
            or not np.isfinite(action).all()
        ):
            raise ValueError("Normalized action must be finite numeric[14]")
        action = action.astype(np.float32)
        if not np.isfinite(action).all():
            raise ValueError("Action exceeds the float32 boundary")
        schema = self.schema["action_schema"]
        low, high = np.asarray(schema["policy_low"]), np.asarray(schema["policy_high"])
        physical = (low + (np.clip(action, -1, 1) + 1) * 0.5 * (high - low)).astype(
            np.float32
        )
        _, reward, terminated, truncated, info = self.env.step(
            physical, render_obs=False
        )
        terminal = self.observation()
        if (
            type(terminated) is not bool
            or type(truncated) is not bool
            or terminated
            and truncated
        ):
            raise RuntimeError("Original termination/truncation flags are invalid")
        if type(reward) not in (int, float) or not math.isfinite(reward):
            raise RuntimeError("Original native reward is not finite")
        self.controls += 1
        self.episode_steps += 1
        result = {
            "obs": terminal,
            "terminal_obs": terminal,
            "reward": float(reward),
            "terminated": terminated,
            "truncated": truncated,
            "normalized_clipped": int(np.count_nonzero((action < -1) | (action > 1))),
            "physical_action": physical.tolist(),
            "native_ctrl": plain(self.env.data.ctrl),
            "sim_time": float(self.env.data.time),
            "terminal_native_policy_state": plain(self.env.get_state()),
            "episodes": [],
            "autoreset": None,
        }
        if terminated or truncated:
            episode = {
                "rank": self.rank,
                "episode_index": self.episode_index,
                "requested_seed": self.requested_seed,
                "seed": self.actual_seed,
                "controls": self.episode_steps,
                "completed": True,
                "success": terminated,
                "terminated": terminated,
                "truncated": truncated,
                "final_native_task_eval": plain(info["task_eval"]),
                "reset_attempts": self.reset_attempts,
            }
            write_json(self.output / f"episode-{self.episode_index:04d}.json", episode)
            result["episodes"] = [episode]
            reset = self.reset()
            result["obs"], result["autoreset"] = reset["obs"], reset
        return result

    def cpu_fixture(self, mode: str) -> dict[str, Any]:
        """Explicit native-state qualification fixtures; never training RPC ops."""
        if self.config["profile"] != "cpu_acceptance2":
            raise ValueError("Native fixtures are forbidden by the training profile")
        import mujoco
        import numpy as np

        if mode == "timeout":
            self.env.cur_step = self.config["max_episode_steps"] - 1
            self.episode_steps = self.env.cur_step
        elif mode == "success":
            # Use public native coordinates and exact original judge metadata.
            # Stack centres along the bin's local height axis; no judge/reward
            # replacement, no changes to physics or simulation timestep.
            evaluation = self.env.evaluate_task().to_info(squeeze=True)
            address = int(
                self.env.model.jnt_qposadr[self.env.model.joint("bin_joint").id]
            )
            position = self.env.data.qpos[address : address + 3].copy()
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, self.env.data.qpos[address + 3 : address + 7])
            rotation = rotation.reshape(3, 3)
            bottom, top = evaluation["bin_bottom_y"], evaluation["bin_top_y"]
            for index, name in enumerate(evaluation["bottle_names"]):
                adr = int(
                    self.env.model.jnt_qposadr[self.env.model.joint(f"{name}_joint").id]
                )
                center = position + rotation @ np.array(
                    [0, bottom + (index + 1) * (top - bottom) / 6, 0]
                )
                offset = np.asarray(evaluation["bottle_center_offsets"][index])
                self.env.data.qpos[adr : adr + 3] = center - offset
                self.env.data.qpos[adr + 3 : adr + 7] = [1, 0, 0, 0]
            self.env.data.qvel[:] = 0
            mujoco.mj_forward(self.env.model, self.env.data)
        else:
            raise ValueError("Unknown explicit CPU native fixture")
        return {
            "fixture": mode,
            "native_state_modified": True,
            "pre_step_obs": self.observation(),
            "pre_step_judge": plain(self.env.evaluate_task().to_info(squeeze=True)),
        }

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.env.close()
            write_json(
                self.output / "native-close.json",
                {
                    "rank": self.rank,
                    "pid": os.getpid(),
                    "status": "closed",
                    "created_at": self.created_at,
                    "maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "render_cameras": False,
                    "torch_imported": "torch" in sys.modules,
                    "warp_imported": "warp" in sys.modules,
                },
            )


def worker(connection: Any, config: dict[str, Any], rank: int, output: str) -> None:
    world = None
    try:
        world = World(config, rank, Path(output))
        connection.send(
            {
                "ok": True,
                "result": {
                    "description": world.schema,
                    "initial": {
                        "obs": world.initial_obs,
                        "rank": rank,
                        "actual_seed": world.actual_seed,
                        "requested_seed": world.requested_seed,
                    },
                },
            }
        )
        while True:
            request = connection.recv()
            operation = request["operation"]
            if operation == "step":
                result = world.step(request["action"])
            elif operation == "reset":
                result = world.reset()
            elif operation == "cpu_fixture":
                result = world.cpu_fixture(request["mode"])
            elif operation == "close":
                world.close()
                connection.send({"ok": True, "result": {"closed": True}})
                break
            else:
                raise ValueError("Unknown world operation")
            connection.send({"ok": True, "result": result})
    except BaseException as error:
        try:
            connection.send({"ok": False, "error": f"{type(error).__name__}: {error}"})
        except (OSError, EOFError, BrokenPipeError):
            pass
        raise
    finally:
        try:
            if world is not None:
                world.close()
        finally:
            connection.close()


def validate_cleanup(rows: Any, count: int) -> None:
    if type(rows) is not list or len(rows) != count:
        raise ValueError("ABC cleanup inventory is incomplete")
    ranks, pids = set(), set()
    for row in rows:
        if type(row) is not dict or set(row) != {"rank", "pid", "exitcode", "alive"}:
            raise ValueError("ABC cleanup fields differ")
        rank = integer(row["rank"], "cleanup.rank", 0, count - 1)
        pid = integer(row["pid"], "cleanup.pid", 1, 2**31 - 1)
        if rank in ranks or pid in pids:
            raise ValueError("ABC cleanup identity is duplicated")
        if (
            row["alive"] is not False
            or type(row["exitcode"]) is not int
            or row["exitcode"] != 0
        ):
            raise ValueError(f"ABC worker cleanup failed: {row}")
        ranks.add(rank)
        pids.add(pid)
    if ranks != set(range(count)):
        raise ValueError("ABC cleanup inventory misses an owned world")


class VectorPool:
    """Barrier-scheduled independent spawn workers with absolute deadlines."""

    def __init__(
        self, config: dict[str, Any], output: Path, *, context: Any = None
    ) -> None:
        self.config = validate_config(config)
        require_cpu()
        self.output = output
        self.context = context or mp.get_context("spawn")
        self.workers, self.connections = [], []
        self.closed, self.close_error = False, None
        self.worker_cleanup: list[dict[str, Any]] = []
        self.worker_resources: list[dict[str, Any]] = []
        self.vector_steps, self.initial_pending = 0, True
        self.deadline = time.monotonic() + config["timeouts"]["wall_s"]
        try:
            for rank in range(config["num_envs"]):
                directory = output / f"world-{rank:03d}"
                directory.mkdir()
                parent, child = self.context.Pipe()
                process = self.context.Process(
                    target=worker, args=(child, config, rank, str(directory))
                )
                process.start()
                child.close()
                self.workers.append(process)
                self.connections.append(parent)
            ready = self._receive(self.connections, config["timeouts"]["ready_s"])
            self.world_schema = ready[0]["description"]
            if any(row["description"] != self.world_schema for row in ready):
                raise ValueError(
                    "Independent original worlds have different action/observation schemas"
                )
            self.initial = [row["initial"] for row in ready]
        except BaseException as error:
            cleanup_error = None
            try:
                self.close()
            except BaseException as failure:  # noqa: BLE001 - seal cleanup evidence, then re-raise
                cleanup_error = failure
            write_json(
                output / "startup-failure.json",
                {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "cleanup_error": None
                    if cleanup_error is None
                    else str(cleanup_error),
                    "worker_cleanup": self.worker_cleanup,
                },
            )
            raise

    def _receive(self, connections: list[Any], seconds: int) -> list[Any]:
        deadline = min(self.deadline, time.monotonic() + seconds)
        pending, results = dict(enumerate(connections)), {}
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "ABC world pool exceeded its total response deadline"
                )
            ready = wait(list(pending.values()), timeout=remaining)
            if not ready:
                raise TimeoutError(
                    "ABC world pool exceeded its total response deadline"
                )
            for index, connection in list(pending.items()):
                if connection not in ready:
                    continue
                try:
                    response = connection.recv()
                except EOFError as error:
                    raise RuntimeError(
                        f"ABC world {index} exited before responding"
                    ) from error
                if type(response) is not dict or response.get("ok") is not True:
                    raise RuntimeError(
                        f"ABC world {index} rejected operation: {response}"
                    )
                results[index] = response["result"]
                del pending[index]
        return [results[index] for index in range(len(connections))]

    def _call(
        self, requests: list[dict[str, Any]], *, timeout: int | None = None
    ) -> list[Any]:
        if self.closed:
            raise RuntimeError("ABC vector pool is closed")
        if len(requests) != len(self.connections):
            raise ValueError("A barrier requires one request per owned world")
        if time.monotonic() >= self.deadline:
            raise TimeoutError("ABC vector host exceeded its wall deadline")
        for connection, request in zip(self.connections, requests):
            connection.send(request)
        return self._receive(
            self.connections, timeout or self.config["timeouts"]["step_s"]
        )

    def describe(self) -> dict[str, Any]:
        nq, nv = self.world_schema["nq"], self.world_schema["nv"]
        return {
            "num_envs": self.config["num_envs"],
            "num_obs": NUM_OBS,
            "num_actions": NUM_ACTIONS,
            "max_episode_steps": 3540,
            "privileged_state": True,
            "profile": self.config["profile"],
            "task": TASK,
            "prompt": PROMPT,
            "physics": "original sequential CPU MuJoCo per world",
            "physics_dt": 0.002,
            "control_decimation": 17,
            "camera_backend": None,
            "reset_options": RESET_OPTIONS,
            "carried_arm_reset": True,
            "placement_retry": "Original: carried pose, init_q same seed, then seed+100000",
            "reset_seed_bank": {
                "base": 200000,
                "rank_stride": 1,
                "episode_stride": 128,
                "max_episode_index": 780,
                "fallback_offset": 100000,
            },
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
                        "length": 95,
                    },
                    {
                        "name": "bottle_and_bin_scales",
                        "offset": nq + nv + 100,
                        "length": 6,
                    },
                ],
                "variant_vocabulary": list(VARIANTS),
                "quaternion": "wxyz",
                "coordinates": "Original generalized coordinates/velocities in SI units",
            },
            "action_schema": self.world_schema["action_schema"],
            "vector_steps": self.vector_steps,
            "transitions": self.vector_steps * self.config["num_envs"],
            "qualification_scope": "128-world capacity and GPU learner pending",
        }

    def reset(self) -> dict[str, Any]:
        if self.closed:
            raise RuntimeError("ABC vector pool is closed")
        if self.initial_pending:
            self.initial_pending = False
            rows = self.initial
        else:
            rows = self._call([{"operation": "reset"}] * self.config["num_envs"])
        return {
            "obs": [row["obs"] for row in rows],
            "actual_seeds": [row["actual_seed"] for row in rows],
            "requested_seeds": [row["requested_seed"] for row in rows],
        }

    def step(self, actions: Any) -> dict[str, Any]:
        if self.initial_pending:
            raise RuntimeError("Reset must precede vector stepping")
        if self.vector_steps >= self.config["vector_steps"]:
            raise RuntimeError("Finite vector-step budget is exhausted")
        if type(actions) is not list or len(actions) != self.config["num_envs"]:
            raise ValueError("Actions require one numeric14 row per world")
        # Validate every action before dispatch, preventing partial advancement.
        for row in actions:
            if (
                type(row) is not list
                or len(row) != 14
                or any(
                    type(x) not in (int, float)
                    or not math.isfinite(x)
                    or abs(x) > 3.4028234e38
                    for x in row
                )
            ):
                raise ValueError("Actions require finite float32-compatible14 rows")
        rows = self._call([{"operation": "step", "action": row} for row in actions])
        self.vector_steps += 1
        return {
            "obs": [row["obs"] for row in rows],
            "terminal_obs": [row["terminal_obs"] for row in rows],
            "rewards": [row["reward"] for row in rows],
            "terminated": [row["terminated"] for row in rows],
            "truncated": [row["truncated"] for row in rows],
            "episodes": [episode for row in rows for episode in row["episodes"]],
            "autoresets": [
                None
                if row["autoreset"] is None
                else {
                    key: value
                    for key, value in row["autoreset"].items()
                    if key != "obs"
                }
                for row in rows
            ],
            "physical_actions": [row["physical_action"] for row in rows],
            "terminal_native_policy_states": [
                row["terminal_native_policy_state"] for row in rows
            ],
            "native_ctrl": [row["native_ctrl"] for row in rows],
            "sim_times": [row["sim_time"] for row in rows],
            "normalized_action_clipping_count": sum(
                row["normalized_clipped"] for row in rows
            ),
            "vector_steps": self.vector_steps,
            "transitions": self.vector_steps * self.config["num_envs"],
        }

    def cpu_fixtures(self) -> list[dict[str, Any]]:
        if self.config["profile"] != "cpu_acceptance2":
            raise ValueError("CPU fixtures are not a training operation")
        return self._call(
            [
                {"operation": "cpu_fixture", "mode": mode}
                for mode in ("success", "timeout")
            ]
        )

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            failures = []
            deadline = time.monotonic() + self.config["timeouts"]["shutdown_s"]
            for process, connection in zip(self.workers, self.connections):
                if process.is_alive():
                    try:
                        connection.send({"operation": "close"})
                    except (OSError, EOFError, BrokenPipeError) as error:
                        failures.append(str(error))
            for process in self.workers:
                process.join(timeout=max(0, deadline - time.monotonic()))
            for process in self.workers:
                if process.is_alive():
                    failures.append(
                        f"Owned worker {process.pid} exceeded normal-close deadline"
                    )
                    process.terminate()
            deadline = time.monotonic() + 3
            for process in self.workers:
                process.join(timeout=max(0, deadline - time.monotonic()))
                if process.is_alive():
                    process.kill()
                    process.join(timeout=1)
            self.worker_cleanup = [
                {
                    "rank": rank,
                    "pid": process.pid,
                    "exitcode": process.exitcode,
                    "alive": process.is_alive(),
                }
                for rank, process in enumerate(self.workers)
            ]
            for connection in self.connections:
                connection.close()
            for rank, process in enumerate(self.workers):
                path = self.output / f"world-{rank:03d}/native-close.json"
                if path.is_file():
                    evidence = json.loads(path.read_text())
                    if (
                        evidence.get("rank") != rank
                        or evidence.get("pid") != process.pid
                    ):
                        failures.append(
                            f"Native cleanup/resource evidence identity differs for rank{rank}"
                        )
                    self.worker_resources.append(
                        {"evidence": artifact(path), "measurement": evidence}
                    )
            try:
                validate_cleanup(self.worker_cleanup, self.config["num_envs"])
            except ValueError as error:
                failures.append(str(error))
            if failures:
                self.close_error = RuntimeError(
                    "ABC worker cleanup failed: " + "; ".join(failures)
                )
        if self.close_error is not None:
            raise self.close_error


def dispatch(pool: VectorPool, request: Any) -> tuple[dict[str, Any], bool]:
    if type(request) is not dict or request.get("schema") != SCHEMA:
        raise ValueError("ABC vector RPC schema differs")
    integer(request.get("id"), "id", 1, 2**63 - 1)
    operation = request.get("operation")
    fields = {"describe": set(), "reset": set(), "step": {"actions"}, "shutdown": set()}
    if (
        operation not in fields
        or set(request) != {"schema", "id", "operation"} | fields[operation]
    ):
        raise ValueError("ABC vector RPC operation/fields differ")
    if operation == "describe":
        result = pool.describe()
    elif operation == "reset":
        result = pool.reset()
    elif operation == "step":
        result = pool.step(request["actions"])
    else:
        pool.close()
        result = {"status": "closed", "worker_cleanup": pool.worker_cleanup}
    return {
        "schema": SCHEMA,
        "id": request["id"],
        "ok": True,
        "result": result,
    }, operation == "shutdown"


def serve(config: dict[str, Any], socket_path: Path, output: Path) -> None:
    validate_config(config)
    if config["profile"] != "training128":
        raise ValueError("Public native server admits only training128")
    require_subset_lease()
    # Authenticate coordinator visibility and ancestry first. Hide CUDA only
    # in this CPU server process before native imports. The separate learner
    # retains the coordinator's original GPU UUID.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    require_cpu()
    if not output.is_dir() or any(output.iterdir()):
        raise ValueError("Host output must be an existing empty directory")
    if len(os.fsencode(socket_path)) > 107 or socket_path.exists():
        raise ValueError("Host socket must be new and within the Unix path limit")
    parent = socket_path.parent
    if not parent.is_dir() or stat.S_IMODE(parent.stat().st_mode) != 0o700:
        raise ValueError("Host socket parent must be a private0700 directory")
    receipt = {
        "schema_version": 1,
        "phase": "original_fasttd3_abc_vector_host",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "config": config,
        "gpu_used_by_physics": False,
    }
    pool = None
    before = time.monotonic()
    previous = {}

    def cancel(signum, frame):
        raise InterruptedError(f"ABC vector host received signal {signum}")

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, cancel)
        pool = VectorPool(config, output)
        receipt["description"] = pool.describe()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(str(socket_path))
            os.chmod(socket_path, 0o600)
            server.listen(4)
            while True:
                remaining = pool.deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("ABC vector host wall deadline expired")
                # GPU compilation may take longer than an RPC. Bound idle
                # time with the host wall watchdog, not a per-message deadline.
                server.settimeout(min(remaining, 1.0))
                try:
                    accepted = server.accept()[0]
                except socket.timeout:
                    continue
                with accepted as connection:
                    rpc_deadline = min(
                        pool.deadline, time.monotonic() + config["timeouts"]["rpc_s"]
                    )
                    request = json.loads(
                        receive_frame(connection, rpc_deadline - time.monotonic())
                    )
                    response, done = dispatch(pool, request)
                    body = (
                        json.dumps(
                            response, separators=(",", ":"), allow_nan=False
                        ).encode()
                        + b"\n"
                    )
                    if len(body) > MAX_FRAME_BYTES:
                        raise ValueError("ABC vector response exceeds4MiB")
                    remaining = rpc_deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(
                            "ABC vector public RPC exceeded its total deadline"
                        )
                    connection.settimeout(remaining)
                    connection.sendall(body)
                if done:
                    break
        receipt["status"] = "completed"
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        cleanup_error = None
        try:
            if pool is not None:
                pool.close()
        except BaseException as error:  # noqa: BLE001 - cleanup failure must fail the final receipt
            cleanup_error = error
            receipt.update(
                status="failed", cleanup_error=f"{type(error).__name__}: {error}"
            )
        finally:
            if pool is not None:
                receipt.update(
                    worker_cleanup=pool.worker_cleanup,
                    worker_resources=pool.worker_resources,
                    vector_steps=pool.vector_steps,
                    transitions=pool.vector_steps * config["num_envs"],
                )
            elif (output / "startup-failure.json").is_file():
                receipt["startup_failure"] = artifact(output / "startup-failure.json")
            receipt["seconds"] = time.monotonic() - before
            receipt["runtime"] = {
                name: importlib.metadata.version(name)
                for name in ("numpy", "mujoco", "gymnasium", "abc-minimal")
            }
            socket_path.unlink(missing_ok=True)
            write_json(output / "receipt.json", receipt)
            for signum, handler in previous.items():
                signal.signal(signum, handler)
        if cleanup_error is not None:
            raise cleanup_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    serve(load_config(args.config), args.socket, args.output)


if __name__ == "__main__":
    main()
