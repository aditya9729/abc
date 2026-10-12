"""Exact captured VAL model/state reset; no sampler or policy implementation.

Native imports occur only when the lease-guarded host constructs this profile.
The recorded RGB is a comparison authority and is never returned as live RGB.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import stat
import struct
from pathlib import Path
from typing import Any

from abc_bench.fasttd3_host import ACTION_NAMES, bounded_int, finite_json
from abc_bench.upstream_eval import artifact

RESET_SCHEMA = "sadhana.abc.captured-reset/v1"
RESET_METHOD = "captured_demonstration_mjb_integration_v1"
ACTION_PROFILE = "original_float32_no_host_clip_v1"
PROOF_SCHEMA = "sadhana.abc.captured-reset-verification/v1"
CAMERAS = ("top", "left", "right")
IMAGE_SHAPE = (168, 224, 3)


def checked_file(pin: Any, maximum: int) -> Path:
    """Authenticate an ordinary pinned file before any native loader sees it."""
    if (
        type(pin) is not dict
        or not {"path", "bytes", "sha256"} <= pin.keys()
        or type(pin["path"]) is not str
        or type(pin["bytes"]) is not int
        or not 0 < pin["bytes"] <= maximum
        or type(pin["sha256"]) is not str
        or len(pin["sha256"]) != 64
        or any(value not in "0123456789abcdef" for value in pin["sha256"])
    ):
        raise ValueError("Captured inputs require bounded path/bytes/SHA256 pins")
    path = Path(pin["path"])
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not path.exists()
        or not stat.S_ISREG(path.lstat().st_mode)
    ):
        raise ValueError("Captured inputs must be regular absolute non-symlink files")
    actual = artifact(path)
    if any(actual[key] != pin[key] for key in ("path", "bytes", "sha256")):
        raise ValueError(f"Captured input differs from its pin: {path}")
    return path


def checked_json(pin: Any) -> dict[str, Any]:
    value = json.loads(checked_file(pin, 1024**2).read_text())
    finite_json(value)
    if type(value) is not dict:
        raise ValueError("Captured JSON input must be an object")
    return value


def select_episode(profile: Any) -> tuple[dict, dict, dict]:
    """Select the declared VAL identity without native/model imports."""
    if (
        type(profile) is not dict
        or set(profile) != {"schema", "method", "cohort", "episode_id"}
        or profile["schema"] != RESET_SCHEMA
        or profile["method"] != RESET_METHOD
        or type(profile["episode_id"]) is not str
        or not profile["episode_id"]
    ):
        raise ValueError("Use the explicit captured-reset profile")
    cohort = checked_json(profile["cohort"])
    if cohort.get("schema") != "sadhana.abc.demonstration-cohort/v1":
        raise ValueError("Captured reset requires a published demonstration cohort")
    splits = cohort.get("splits")
    if type(splits) is not dict or set(splits) != {
        "train",
        "validation",
        "development",
    }:
        raise ValueError("Captured cohort must declare all whole-episode splits")
    ids = []
    for values in splits.values():
        if type(values) is not list or any(type(value) is not str for value in values):
            raise ValueError("Declared split IDs must be strings")
        ids.extend(values)
    if len(ids) != len(set(ids)) or profile["episode_id"] not in splits["validation"]:
        raise ValueError("Select one unique declared validation episode")
    entries = cohort.get("episodes")
    if type(entries) is not list or not 1 <= len(entries) <= 32:
        raise ValueError("Captured cohort must pin finite episode entries")
    selected, seen = None, set()
    for entry in entries:
        if type(entry) is not dict or set(entry) != {
            "manifest",
            "worker_receipt",
            "parent_receipt",
        }:
            raise ValueError("Captured episode requires manifest/worker/parent pins")
        episode = checked_json(entry["manifest"])
        identity = episode.get("episode_id")
        if type(identity) is not str or identity in seen or identity not in ids:
            raise ValueError("Captured cohort episode IDs differ from declared splits")
        seen.add(identity)
        if identity == profile["episode_id"]:
            selected = entry, episode
    if seen != set(ids) or selected is None or selected[1].get("split") != "validation":
        raise ValueError("The selected capture is not a complete declared VAL episode")
    return cohort, selected[0], selected[1]


def validate_config(config: dict) -> dict:
    if config.get("stage") != "evaluation":
        raise ValueError("Captured schema2 is evaluation only")
    host = config["host"]
    if type(host) is not dict or set(host) != {
        "num_envs",
        "initial_seed",
        "allowed_seeds",
        "max_episode_steps",
        "save_video",
        "action_profile",
        "reset_profile",
    }:
        raise ValueError("Captured schema2 requires explicit reset/action profiles")
    bounded_int(host["num_envs"], "num_envs", 1, 1)
    bounded_int(host["max_episode_steps"], "max_episode_steps", 1, 3540)
    if type(host["save_video"]) is not bool or host["action_profile"] != ACTION_PROFILE:
        raise ValueError("Captured schema2 requires original raw float32 actions")
    _, _, episode = select_episode(host["reset_profile"])
    seed = bounded_int(episode.get("seed"), "captured actual seed", 0, 2**31 - 1)
    if (
        type(host["initial_seed"]) is not int
        or host["initial_seed"] != seed
        or host["allowed_seeds"] != [seed]
        or any(type(value) is not int for value in host["allowed_seeds"])
    ):
        raise ValueError("Captured seeds must identify the selected actual VAL capture")
    timeouts = config["timeouts"]
    if type(timeouts) is not dict or set(timeouts) != {
        "host_wall_s",
        "rpc_s",
        "max_requests",
    }:
        raise ValueError("Captured host needs explicit finite resource bounds")
    for key, maximum in (
        ("host_wall_s", 7200),
        ("rpc_s", 300),
        ("max_requests", 100000),
    ):
        bounded_int(timeouts[key], key, 1, maximum)
    return config


class CapturedScene:
    """One authenticated complete VAL episode, with exact reset comparison data."""

    def __init__(self, profile: dict):
        from abc_bench.demo_cohort import validate_episode
        from abc_bench.upstream_report import coordinated_receipt

        self.profile = profile
        _, self.entry, episode = select_episode(profile)
        worker = checked_json(self.entry["worker_receipt"])
        parent = checked_json(self.entry["parent_receipt"])
        run = Path(self.entry["worker_receipt"]["path"]).parent
        admitted = coordinated_receipt(run)
        expected_parent = run.parent / "runs" / (run.name + ".json")
        job = json.loads((run / "job.json").read_text())
        if (
            admitted is None
            or admitted[1] != self.entry["worker_receipt"]
            or admitted[2] != parent
            or Path(self.entry["parent_receipt"]["path"]) != expected_parent
            or job.get("module") != "abc_bench.demo_collect"
            or worker.get("phase") != "original_policy_demonstration_collection"
            or worker.get("split") != "validation"
            or type(worker.get("schema_version")) is not int
            or worker["schema_version"] != 1
            or type(parent.get("schema_version")) is not int
            or parent["schema_version"] != 1
            or type(parent.get("worker_exit_code")) is not int
            or parent["worker_exit_code"] != 0
        ):
            raise ValueError(
                "Captured episode must bind its completed canonical collector"
            )
        index = bounded_int(episode.get("world_index"), "captured world index", 0, 4)
        entries = worker.get("episodes")
        if (
            type(entries) is not list
            or index >= len(entries)
            or entries[index].get("manifest") != self.entry["manifest"]
            or entries[index].get("admitted") is not True
        ):
            raise ValueError("Completed worker did not admit the selected capture")
        summary = checked_json(worker["original_summary"])
        worlds = summary.get("worlds")
        if type(worlds) is not list or index >= len(worlds):
            raise ValueError("Captured source world is missing")
        self.episode = validate_episode(
            self.entry["manifest"], worlds[index], "validation"
        )
        producer = {
            "config": worker["input_config"],
            "checkpoint": worker["checkpoint"],
            "metadata": worker["metadata"],
            "software": worker["software"],
        }
        if self.episode["producer"] != producer:
            raise ValueError("Captured manifest and completed producer differ")
        self.software = checked_json(worker["software"])
        checked_json(worker["input_config"])
        checked_json(worker["metadata"])
        for item in self.software["sources"].values():
            checked_file(item, 1024**2)
        self.files = self.episode["artifacts"]
        for name, item in self.files.items():
            maximum = (
                256 * 1024**2
                if name == "compiled_model"
                else 64 * 1024
                if name == "initial_integration"
                else 2 * 1024**3
            )
            checked_file(item, maximum)
        if self.episode.get("integration_state_spec") != {
            "name": "mjSTATE_INTEGRATION",
            "value": 16383,
        }:
            raise ValueError(
                "Captured reset requires complete INTEGRATION specification16383"
            )
        self.randomization = checked_json(self.files["randomization"])
        self.source = {
            "cohort": profile["cohort"],
            **self.entry,
            "software": worker["software"],
        }

    def array(self, name: str, shape: tuple, dtype: str):
        import numpy as np

        pin = self.files[name]
        result = np.load(pin["path"], mmap_mode="r", allow_pickle=False)
        if (
            result.shape != shape
            or pin.get("shape") != list(shape)
            or result.dtype != np.dtype(dtype)
            or type(pin.get("dtype")) is not str
            or np.dtype(pin["dtype"]) != np.dtype(dtype)
            or not np.isfinite(result).all()
        ):
            raise ValueError(f"Captured {name} shape/dtype/finite values differ")
        return result

    def recorded_camera_hashes(self) -> dict[str, str]:
        import numpy as np

        with Path(self.files["rgb24"]["path"]).open("rb") as stream:
            payload = stream.read(168 * 672 * 3)
        if len(payload) != 168 * 672 * 3:
            raise ValueError("Captured initial RGB frame is incomplete")
        combined = np.frombuffer(payload, dtype=np.uint8).reshape(168, 672, 3)
        return {
            name: hashlib.sha256(
                combined[:, index * 224 : (index + 1) * 224].tobytes()
            ).hexdigest()
            for index, name in enumerate(CAMERAS)
        }

    def restore(self, env: Any, report: dict) -> None:
        import mujoco

        for key in ("mujoco", "numpy", "abc-minimal", "mujoco-warp", "warp-lang"):
            if importlib.metadata.version(key) != self.software["distributions"][key]:
                raise ValueError(f"Captured runtime version differs: {key}")
        if mujoco.__version__ != "3.8.1":
            raise ValueError("Captured profile requires MuJoCo3.8.1")
        with Path(self.files["compiled_model"]["path"]).open("rb") as stream:
            header = struct.unpack("<5i", stream.read(20))
        if header != (54321, 8, 92, 3008001, 472):
            raise ValueError("Captured MJB precision/version/ABI differs")
        model = mujoco.MjModel.from_binary_path(self.files["compiled_model"]["path"])
        if (
            (model.nq, model.nv, model.nu) != (58, 52, 14)
            or model.nplugin != 0
            or model.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_SLEEP)
        ):
            raise ValueError("Captured model dimensions/plugins/sleeping differ")
        if float(model.opt.timestep) != 0.002 or env._control_decimation != 17:
            raise ValueError("Captured physics clock differs")
        for kind, names in (
            (
                mujoco.mjtObj.mjOBJ_JOINT,
                [f"{arm}_joint{i}" for arm in ("left", "right") for i in range(1, 7)]
                + [f"{arm}_left_finger" for arm in ("left", "right")]
                + [f"bottle_{i}_joint" for i in range(1, 6)],
            ),
            (mujoco.mjtObj.mjOBJ_ACTUATOR, ACTION_NAMES),
            (mujoco.mjtObj.mjOBJ_CAMERA, CAMERAS),
        ):
            if any(mujoco.mj_name2id(model, kind, name) < 0 for name in names):
                raise ValueError("Captured joint/actuator/camera names differ")
        for name in (
            "passive",
            "control",
            "contactfilter",
            "sensor",
            "time",
            "act_dyn",
            "act_gain",
            "act_bias",
        ):
            getter = getattr(mujoco, "get_mjcb_" + name, None)
            if getter is not None and getter() is not None:
                raise ValueError(
                    "Captured profile cannot restore custom callback state"
                )
        env.reload_from_model(model)
        env.set_task(self.episode["task"])
        if env.task_runtime_debug_state() is not None:
            raise ValueError(
                "Captured profile cannot restore Python task runtime state"
            )
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        if int(spec) != 16383 or mujoco.mj_stateSize(env.model, spec) != 531:
            raise ValueError("Captured complete integration size/specification differs")
        state = self.array("initial_integration", (531,), "float64")
        mujoco.mj_setState(env.model, env.data, state, spec)
        env.cur_step = 0
        mujoco.mj_forward(env.model, env.data)
        report["runtime"] = {
            "mujoco": mujoco.__version__,
            "model_dimensions": {"nq": model.nq, "nv": model.nv, "nu": model.nu},
            "integration_spec": int(spec),
            "integration_values": 531,
        }

    def check_physics(self, env: Any) -> dict:
        import mujoco
        import numpy as np

        encoded = np.empty(mujoco.mj_sizeModel(env.model), dtype=np.uint8)
        mujoco.mj_saveModel(env.model, None, encoded)
        actual_model = {
            "bytes": encoded.size,
            "sha256": hashlib.sha256(encoded.tobytes()).hexdigest(),
        }
        expected_model = {
            key: self.files["compiled_model"][key] for key in ("bytes", "sha256")
        }
        integration = np.empty(531, dtype=np.float64)
        mujoco.mj_getState(
            env.model, env.data, integration, mujoco.mjtState.mjSTATE_INTEGRATION
        )
        controls = self.episode["controls"]
        expected = {
            "integration": self.array("initial_integration", (531,), "float64"),
            "qpos": self.array("qpos", (controls + 1, 58), "float64")[0],
            "qvel": self.array("qvel", (controls + 1, 52), "float64")[0],
            "time": self.array("sim_times", (controls + 1,), "float64")[0],
            "state14": self.array("states", (controls + 1, 14), "float32")[0],
        }
        actual = {
            "integration": integration,
            "qpos": env.data.qpos,
            "qvel": env.data.qvel,
            "time": np.asarray(env.data.time),
            "state14": env.get_state(),
        }
        checks = {
            key: {
                "equal": np.asarray(value).dtype == np.asarray(expected[key]).dtype
                and np.asarray(value).shape == np.asarray(expected[key]).shape
                and np.asarray(value).tobytes() == np.asarray(expected[key]).tobytes(),
                "actual_sha256": hashlib.sha256(
                    np.asarray(value).tobytes()
                ).hexdigest(),
                "expected_sha256": hashlib.sha256(
                    np.asarray(expected[key]).tobytes()
                ).hexdigest(),
            }
            for key, value in actual.items()
        }
        judge = env.evaluate_task().to_info(squeeze=True)
        finite_json(judge)
        checks["judge"] = {
            "equal": judge == self.episode["initial_task_eval"],
            "actual": judge,
            "expected": self.episode["initial_task_eval"],
        }
        checks["model"] = {
            "equal": actual_model == expected_model,
            "actual": actual_model,
            "expected": expected_model,
        }
        checks["chronology"] = {
            "equal": env.cur_step == 0 and float(env.data.time) == 0.0,
            "control_step": env.cur_step,
            "simulation_time_s": float(env.data.time),
        }
        return checks


def make_captured_env(scene: CapturedScene, horizon: int):
    import abc_sim

    return abc_sim.make_env(
        task=scene.episode["task"],
        prompt=scene.episode["prompt"],
        render_cameras=True,
        enable_task_randomizer=False,
        camera_backend="mjwarp",
        camera_gpu_id=0,
        camera_height=168,
        camera_width=224,
        max_episode_steps=horizon,
        terminate_on_success=True,
    )
