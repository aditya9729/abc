"""Record the unchanged author ABC-VLA rollout under a canonical subset lease.

The recorder adds no SimTaskEnv observation, evaluator, render, physics, or
policy inference call. It adds pure native get_state reads and reset-state/model
serialization. Numeric traces are bounded by the author horizon; RGB is streamed.
This module does not import a native runtime until admission.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from abc_bench import upstream_eval
from abc_bench.subset import require_subset_lease

SCHEMA = "sadhana.abc.policy-demonstration/v1"
PHASE = "original_policy_demonstration_collection"
CAMERAS = ("top", "left", "right")
TRAIN_REQUEST = {
    "bottle_count": 5,
    "randomize_variants": True,
    "randomize_scales": True,
}
VALIDATION_SEEDS = frozenset(range(9100, 9103))
VALIDATION_ACTUAL_SEEDS = VALIDATION_SEEDS | {
    seed + 100_000 for seed in VALIDATION_SEEDS
}
DEVELOPMENT_SEED = 20260511
MAX_CONTROLS = 3540
UPSTREAM_COMMIT = "d0832d12651d1b260a652861a14648dc5f3660c7"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def _json_copy(value: Any) -> Any:
    """Detach source JSON values without retaining mutable observations."""
    return json.loads(_json_bytes(value))


def check_seed(split: str, seed: int) -> None:
    if type(seed) is not int or not 0 <= seed <= 2**31 - 1:
        raise ValueError("Seeds must be integer values in [0, 2147483647]")
    if split == "development" and seed != DEVELOPMENT_SEED:
        raise ValueError("Development permits only the original seed20260511")
    if split == "validation" and seed not in VALIDATION_SEEDS:
        raise ValueError("Validation permits only seeds9100..9102")
    if split == "train" and seed in VALIDATION_ACTUAL_SEEDS | {DEVELOPMENT_SEED}:
        raise ValueError("TRAIN cannot use the development or validation seeds")


def check_actual_seed(split: str, seed: int, requested_seed: int) -> None:
    """Permit only the original requested seed or its documented +100000 retry."""
    check_seed(split, requested_seed)
    if type(seed) is not int or seed not in {requested_seed, requested_seed + 100_000}:
        raise ValueError(
            "Actual reset seed is not the requested seed or original mapped fallback"
        )
    if split == "development" and seed != DEVELOPMENT_SEED:
        raise ValueError("Development reference must remain exact seed20260511")
    if split == "train" and seed in VALIDATION_ACTUAL_SEEDS | {DEVELOPMENT_SEED}:
        raise ValueError(
            "Actual TRAIN seed overlaps reserved development/validation domains"
        )


def load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the full author recipe before any lease or native import."""
    config = upstream_eval._read_json(path)
    if (
        set(config)
        != {
            "schema_version",
            "checkpoint",
            "metadata",
            "sim_eval",
            "split",
            "max_output_bytes",
        }
        or type(config["schema_version"]) is not int
        or config["schema_version"] != 1
    ):
        raise ValueError("Use the six documented demonstration configuration fields")
    split = config["split"]
    if type(split) is not str or split not in {"train", "validation", "development"}:
        raise ValueError("split must be train, validation, or development")
    cap = config["max_output_bytes"]
    if type(cap) is not int or not 0 < cap <= 5 * 1024**3:
        raise ValueError("max_output_bytes must be a positive integer at most5GiB")
    sim = config["sim_eval"]
    if type(sim) is not dict or set(sim) != set(upstream_eval.RECIPE):
        raise ValueError(
            "sim_eval must contain the complete original qualification recipe"
        )
    for name, expected in upstream_eval.RECIPE.items():
        if name not in {"seed", "num_worlds", "randomization"} and (
            type(sim[name]) is not type(expected) or sim[name] != expected
        ):
            raise ValueError(
                f"sim_eval.{name} must preserve the author recipe: {expected!r}"
            )
    if type(sim["num_worlds"]) is not int or not 1 <= sim["num_worlds"] <= 3:
        raise ValueError("num_worlds must be an integer in [1,3]")
    check_seed(split, sim["seed"])
    for index in range(sim["num_worlds"]):
        check_seed(split, sim["seed"] + index)
    if split == "development":
        if sim["num_worlds"] != 1 or sim["randomization"] is not None:
            raise ValueError(
                "Development uses one world and the original default randomization"
            )
    else:
        if type(sim["randomization"]) is not str:
            raise ValueError(
                "TRAIN/validation randomization must be an author-compatible JSON string"
            )
        request = json.loads(sim["randomization"])
        if (
            type(request) is not dict
            or set(request) != set(TRAIN_REQUEST)
            or any(
                type(request[key]) is not type(value) or request[key] != value
                for key, value in TRAIN_REQUEST.items()
            )
        ):
            raise ValueError("Use count5 with original random variants and scales")
        sim["randomization"] = json.dumps(TRAIN_REQUEST, sort_keys=True)
    # Reuse the original qualification's pinned checkpoint/model admission.
    config["checkpoint"] = upstream_eval._pin(
        config["checkpoint"], "checkpoint", 32 * 1024**3
    )
    config["metadata"] = upstream_eval._pin(config["metadata"], "metadata", 1024**2)
    metadata_path = Path(config["metadata"]["path"])
    if upstream_eval.artifact(metadata_path) != config["metadata"]:
        raise ValueError("Metadata SHA-256 differs from its pin")
    metadata = upstream_eval._read_json(metadata_path)
    if (
        metadata.get("format") != "abc_checkpoint_metadata/v1"
        or type(metadata.get("bytes")) is not int
        or metadata["bytes"] != config["checkpoint"]["bytes"]
        or metadata.get("sha256") != config["checkpoint"]["sha256"]
        or type(metadata.get("max_action_prefix")) is not int
        or metadata["max_action_prefix"] <= 4
    ):
        raise ValueError(
            "Metadata does not admit the checkpoint and original RTC prefix4"
        )
    upstream_eval._model_config(metadata.get("model_config"))
    return config, metadata


class OutputBudget:
    """Bound owned payload writes; also check independently written MP4/XML files.

    Native XML and original FFmpeg writes are external. They are checked at
    the next recorder boundary and at final admission, not intercepted.
    """

    def __init__(self, root: Path, maximum: int):
        self.root, self.maximum = root, maximum
        self.lock = threading.RLock()
        self.reserve = min(128 * 1024, maximum // 2)

    def size(self) -> int:
        return sum(
            path.stat().st_size for path in self.root.rglob("*") if path.is_file()
        )

    def check(self, added: int = 0, *, diagnostic: bool = False) -> None:
        limit = self.maximum if diagnostic else self.maximum - self.reserve
        if self.size() + added > limit:
            raise RuntimeError("Demonstration output byte cap reached")

    def write(self, path: Path, value: bytes, *, diagnostic: bool = False) -> None:
        with self.lock:
            self.check(len(value), diagnostic=diagnostic)
            with path.open("xb") as stream:
                stream.write(value)

    def json(self, path: Path, value: Any, *, diagnostic: bool = False) -> None:
        self.write(path, _json_bytes(value), diagnostic=diagnostic)

    def array(self, path: Path, value: np.ndarray) -> dict[str, Any]:
        buffer = io.BytesIO()
        np.save(buffer, value, allow_pickle=False)
        self.write(path, buffer.getvalue())
        return {
            **upstream_eval.artifact(path),
            "shape": list(value.shape),
            "dtype": value.dtype.name,
        }


class BinaryStream:
    def __init__(self, path: Path, budget: OutputBudget):
        self.path, self.budget = path, budget
        self.stream = path.open("xb", buffering=0)
        self.count = 0

    def append(self, value: bytes) -> int:
        with self.budget.lock:
            self.budget.check(len(value))
            offset = self.stream.tell()
            if self.stream.write(value) != len(value):
                raise OSError("Short demonstration stream write")
            self.count += 1
            return offset

    def close(self) -> None:
        if not self.stream.closed:
            self.stream.close()


def _rgb(images: dict[str, Any]) -> np.ndarray:
    frames = []
    for name in CAMERAS:
        image = np.asarray(images[name])
        if image.dtype != np.uint8 or image.shape != (3, 168, 224):
            raise ValueError(f"{name} must be the original uint8 CHW168x224 RGB image")
        frames.append(image.transpose(1, 2, 0))
    return np.ascontiguousarray(np.concatenate(frames, axis=1))


def _numeric(
    value: Any, dtype: str, shape: tuple[int, ...] | None = None
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype(dtype) or (shape is not None and array.shape != shape):
        raise ValueError(
            f"Expected {dtype} array with shape{shape}, got{array.dtype}/{array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("Non-finite demonstration numeric value")
    return np.array(array, dtype=np.dtype(dtype).newbyteorder("<"), copy=True)


def snapshot_reset(native: Any, episode: Episode) -> None:
    """Copy full native integration state and the randomized scene after reset."""
    import mujoco

    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(native.model, state_spec), dtype=np.float64)
    mujoco.mj_getState(native.model, native.data, state, state_spec)
    episode.pins["initial_integration"] = episode.budget.array(
        episode.root / "initial_integration.npy", _numeric(state, "float64")
    )
    episode.integration_spec = {"name": "mjSTATE_INTEGRATION", "value": int(state_spec)}
    scene = episode.root / "scene.xml"
    episode.budget.check()
    mujoco.mj_saveLastXML(str(scene), native.model)
    episode.budget.check()
    episode.pins["scene_xml"] = upstream_eval.artifact(scene)
    compiled = episode.root / "scene.mjb"
    episode.budget.check(int(mujoco.mj_sizeModel(native.model)))
    mujoco.mj_saveModel(native.model, str(compiled))
    episode.budget.check()
    episode.pins["compiled_model"] = upstream_eval.artifact(compiled)
    episode.actuator_limits = {
        "ctrl_limited": np.asarray(native.model.actuator_ctrllimited)
        .astype(bool)
        .tolist(),
        "ctrl_range": np.asarray(
            native.model.actuator_ctrlrange, dtype=np.float64
        ).tolist(),
        "submitted_ctrl_note": "Native data.ctrl input; not measured force or internally limited effective control",
        "explicit_action_clipping": False,
    }


class Episode:
    """One bounded physical rollout. No RGB observations accumulate in memory."""

    def __init__(
        self, collection: Collection, native: Any, obs: dict[str, Any], seed: int
    ):
        self.collection, self.native, self.budget = (
            collection,
            native,
            collection.budget,
        )
        self.world_index = len(collection.episodes)
        self.episode_id = f"{collection.collection_id}-world_{self.world_index:03d}"
        self.root = collection.root / "episodes" / f"world_{self.world_index:03d}"
        self.root.mkdir(parents=True)
        self.seed, self.requested_seed = (
            seed,
            collection.config["sim_eval"]["seed"] + self.world_index,
        )
        self.reset_attempts = _json_copy(collection.reset_attempts)
        collection.reset_attempts.clear()
        self.randomization = _json_copy(collection.env.randomization.to_dict())
        if self.randomization.get("seed") != seed:
            raise ValueError(
                "Native randomization seed differs from the actual reset seed"
            )
        self.prompt = obs["prompt"]
        self.states: list[np.ndarray] = []
        self.qpos: list[np.ndarray] = []
        self.qvel: list[np.ndarray] = []
        self.sim_times: list[float] = []
        self.actions: list[np.ndarray] = []
        self.ctrl: list[np.ndarray] = []
        self.judges: list[dict[str, Any] | None] = []
        self.initial_eval: dict[str, Any] | None = None
        self.pins: dict[str, Any] = {}
        self.integration_spec: dict[str, Any] = {}
        self.actuator_limits: dict[str, Any] = {}
        self.closed = False
        self.finished = False
        opened = []
        try:
            for attribute, filename in (
                ("rgb", "rgb24.bin"),
                ("request_rgb", "request_rgb24.bin"),
                ("requests", "policy_requests.jsonl"),
                ("provenance", "action_provenance.jsonl"),
            ):
                stream = BinaryStream(self.root / filename, self.budget)
                setattr(self, attribute, stream)
                opened.append(stream)
        except Exception:
            for stream in opened:
                stream.close()
            raise
        self.output_refs: list[dict[str, Any]] = []
        self.observations: dict[int, dict[str, Any]] = {}
        self.frame_index = -1
        collection.episodes.append(self)
        collection.current = self
        self.note_state()
        self.frame(obs["images"])
        self.note_observation(obs, "reset")
        snapshot_reset(native, self)

    @property
    def controls(self) -> int:
        return len(self.actions)

    def note_state(self) -> None:
        self.states.append(_numeric(self.native.get_state(), "float32", (14,)))
        self.qpos.append(_numeric(self.native.data.qpos, "float64"))
        self.qvel.append(_numeric(self.native.data.qvel, "float64"))
        sim_time = float(self.native.data.time)
        if not np.isfinite(sim_time):
            raise ValueError("Non-finite native simulation time")
        self.sim_times.append(sim_time)

    def frame(self, images: dict[str, Any]) -> None:
        if self.frame_index + 1 != self.controls:
            raise RuntimeError(
                "Post-action video cadence must supply exactly one frame per control"
            )
        self.rgb.append(_rgb(images).tobytes())
        self.frame_index += 1

    def note_observation(self, obs: dict[str, Any], source: str) -> None:
        self.observations[id(obs)] = {
            "source": source,
            "control_index": self.controls,
            "sim_time": float(self.native.data.time),
            "object": obs,
            "fingerprint": self.observation_fingerprint(obs),
        }
        while len(self.observations) > 2:
            del self.observations[next(iter(self.observations))]

    @staticmethod
    def observation_fingerprint(obs: dict[str, Any]) -> str:
        digest = hashlib.sha256(_numeric(obs["state"], "float32", (14,)).tobytes())
        digest.update(_rgb(obs["images"]).tobytes())
        digest.update(str(obs["prompt"]).encode())
        return digest.hexdigest()

    def before_infer(
        self, obs: dict[str, Any], noise: Any, prefix: Any, length: int
    ) -> dict[str, Any]:
        with self.budget.lock:
            observation = self.observations.get(id(obs))
            if observation is None or observation["object"] is not obs:
                raise RuntimeError(
                    "Policy request observation bypassed the public recorder"
                )
            if observation["fingerprint"] != self.observation_fingerprint(obs):
                raise RuntimeError("Public observation changed before the sampler call")
            if self.requests.count >= 236:
                raise RuntimeError("Original sampler request count exceeded236")
            frame_index = self.request_rgb.count
            offset = self.request_rgb.append(_rgb(obs["images"]).tobytes())
            return {
                "inference_id": self.requests.count + 1,
                "observation": {
                    name: value
                    for name, value in observation.items()
                    if name != "object"
                },
                "state": _numeric(obs["state"], "float32", (14,)).tolist(),
                "prompt": obs["prompt"],
                "camera_names": list(CAMERAS),
                "image_frame_index": frame_index,
                "image_byte_offset": offset,
                "image_shape_hwc": [168, 672, 3],
                "image_dtype": "uint8",
                "noise": None
                if noise is None
                else _numeric(noise, "float32", (30, 14)).tolist(),
                "action_prefix": None
                if prefix is None
                else _numeric(prefix, "float32", (30, 14)).tolist(),
                "prefix_length": int(length or 0),
                "started_monotonic": time.monotonic(),
                "purpose": "executed_rollout_sampler; internal compile/RTC warmups are excluded",
            }

    def after_infer(self, request: dict[str, Any], result: Any) -> None:
        array = _numeric(result, "float32", (30, 14))
        request.update(
            {"actions": array.tolist(), "finished_monotonic": time.monotonic()}
        )
        original = np.asarray(result)
        with self.budget.lock:
            self.requests.append(_json_bytes(request))
            self.output_refs.append(
                {
                    "pointer": original.__array_interface__["data"][0],
                    "stride": original.strides[0],
                    "rows": original.shape[0],
                    "inference_id": request["inference_id"],
                    "observation_control_index": request["observation"][
                        "control_index"
                    ],
                    "prefix_length": request["prefix_length"],
                    "owner": result,
                    "values": array,
                }
            )

    def action_source(self, action: Any) -> dict[str, Any]:
        pointer = np.asarray(action).__array_interface__["data"][0]
        with self.budget.lock:
            for ref in reversed(self.output_refs):
                delta = pointer - ref["pointer"]
                if ref["stride"] > 0 and delta >= 0 and delta % ref["stride"] == 0:
                    index = delta // ref["stride"]
                    if index < ref["rows"] and np.shares_memory(
                        np.asarray(action), ref["owner"]
                    ):
                        if not np.array_equal(np.asarray(action), ref["values"][index]):
                            raise RuntimeError(
                                "Sampler output row changed before physical control"
                            )
                        return {
                            "control_index": self.controls + 1,
                            "inference_id": ref["inference_id"],
                            "inferred_action_index": index,
                            "observation_control_index": ref[
                                "observation_control_index"
                            ],
                            "prefix_length": ref["prefix_length"],
                        }
        raise RuntimeError(
            "Issued action does not reference a recorded original sampler output"
        )

    def judge(self, value: dict[str, Any]) -> None:
        detached = _json_copy(value)
        if self.controls == 0:
            self.initial_eval = detached
        else:
            self.judges[-1] = {
                "control_index": self.controls,
                "sim_time": self.sim_times[-1],
                "task_eval": detached,
            }

    def finish(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            arrays = {
                "states": np.asarray(self.states, dtype="<f4"),
                "issued_actions": np.asarray(self.actions, dtype="<f4").reshape(-1, 14),
                "sim_times": np.asarray(self.sim_times, dtype="<f8"),
                "qpos": np.asarray(self.qpos, dtype="<f8"),
                "qvel": np.asarray(self.qvel, dtype="<f8"),
                "actuator_ctrl": np.asarray(self.ctrl, dtype="<f8").reshape(
                    -1, self.native.model.nu
                ),
            }
            for name, array in arrays.items():
                self.pins[name] = self.budget.array(self.root / f"{name}.npy", array)
            self.budget.json(self.root / "randomization.json", self.randomization)
            self.pins["randomization"] = upstream_eval.artifact(
                self.root / "randomization.json"
            )
            if any(value is None for value in self.judges):
                raise RuntimeError(
                    "An executed control lacks the original post-control task evaluation"
                )
            self.budget.write(
                self.root / "task_judge.jsonl",
                b"".join(_json_bytes(value) for value in self.judges),
            )
            self.pins["task_judge"] = upstream_eval.artifact(
                self.root / "task_judge.jsonl"
            )
            self.finished = True
        finally:
            for stream in (self.rgb, self.request_rgb, self.requests, self.provenance):
                stream.close()
            for name, stream, shape, dtype in (
                ("rgb24", self.rgb, [self.rgb.count, 168, 672, 3], "uint8"),
                (
                    "request_rgb24",
                    self.request_rgb,
                    [self.request_rgb.count, 168, 672, 3],
                    "uint8",
                ),
                ("policy_requests", self.requests, None, None),
                ("action_provenance", self.provenance, None, None),
            ):
                self.pins[name] = upstream_eval.artifact(stream.path)
                if shape is not None:
                    self.pins[name].update({"shape": shape, "dtype": dtype})

    def manifest(self, world: dict[str, Any] | None = None) -> dict[str, Any]:
        final = (
            self.judges[-1]["task_eval"] if self.judges and self.judges[-1] else None
        )
        count = self.randomization.get("metadata", {}).get("bottle_count")
        current = None if final is None else final.get("num_bottles_in_bin")
        complete = bool(
            self.finished and self.rgb.count == self.controls + 1 and world is not None
        )
        terminated = bool(complete and final.get("ever_success", False))
        admitted = bool(
            complete and count == 5 and current == 5 and final.get("success") is True
        )
        group = None
        if {"scene_xml", "initial_integration"} <= self.pins.keys():
            group = hashlib.sha256(
                (
                    self.pins["scene_xml"]["sha256"]
                    + ":"
                    + self.pins["initial_integration"]["sha256"]
                ).encode("ascii")
            ).hexdigest()
        return {
            "schema": SCHEMA,
            "created_at": self.collection.created_at,
            "episode_id": self.episode_id,
            "world_index": self.world_index,
            "split": self.collection.config["split"],
            "status": "completed" if complete else "partial",
            "controls": self.controls,
            "frame_count": self.rgb.count,
            "seed": self.seed,
            "requested_seed": self.requested_seed,
            "reset_group_id": group,
            "reset_attempts": self.reset_attempts,
            "task": "put_plastic_bottles_in_bin",
            "prompt": self.prompt,
            "policy_channels": [f"left_j{i}" for i in range(1, 7)]
            + ["left_grip"]
            + [f"right_j{i}" for i in range(1, 7)]
            + ["right_grip"],
            "action_units": "joint radians; gripper fraction multiplied by0.0475m in original env",
            "camera_names": list(CAMERAS),
            "image_shape_hwc": [168, 224, 3],
            "physics": {
                "dt": self.collection.physics["physics_dt"],
                "decimation": self.collection.physics["control_decimation"],
                "control_hz": self.collection.physics["control_hz"],
            },
            "integration_state_spec": self.integration_spec,
            "actuator_limits": self.actuator_limits,
            "artifacts": self.pins,
            "initial_task_eval": self.initial_eval,
            "final_task_eval": final,
            "producer": self.collection.producer,
            "admission": {
                "complete_current_success": admitted,
                "active_bottles": count,
                "current_bottles_in_bin": current,
                "terminated": terminated,
                "truncated": bool(
                    complete and not terminated and self.controls == MAX_CONTROLS
                ),
            },
            "human_expert_approved": False,
            "atlas_approved": False,
            "dynamics_validated": False,
            "original_world": world,
        }


class Collection:
    def __init__(self, config: dict[str, Any], root: Path):
        self.config, self.root = config, root
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.collection_id = (
            "demo-" + hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:16]
        )
        self.budget = OutputBudget(root, config["max_output_bytes"])
        self.episodes: list[Episode] = []
        self.current: Episode | None = None
        self.env: Any = None
        self.physics: dict[str, Any] = {}
        self.reset_attempts: list[dict[str, Any]] = []
        self.producer: dict[str, Any] = {}

    def finish_current(self) -> None:
        if self.current is not None:
            episode, self.current = self.current, None
            episode.finish()

    def reconcile(self, summary: dict[str, Any]) -> list[dict[str, Any]]:
        worlds = summary.get("worlds")
        if (
            summary.get("task") != "put_plastic_bottles_in_bin"
            or not isinstance(worlds, list)
            or len(worlds) != self.config["sim_eval"]["num_worlds"]
            or len(worlds) != len(self.episodes)
        ):
            raise RuntimeError(
                "Original summary does not match the captured task/world cohort"
            )
        manifests = []
        for episode, world in zip(self.episodes, worlds, strict=True):
            final = (
                episode.judges[-1]["task_eval"]
                if episode.judges and episode.judges[-1]
                else None
            )
            if (
                not episode.finished
                or episode.rgb.count != episode.controls + 1
                or episode.controls < 1
                or any(
                    (
                        world.get("world_index") != episode.world_index,
                        world.get("world_seed") != episode.seed,
                        world.get("steps") != episode.controls,
                        world.get("randomization") != episode.randomization,
                        world.get("final_task_eval") != final,
                        world.get("final_success") is not final.get("success"),
                    )
                )
            ):
                raise RuntimeError(
                    "Original summary differs from actual captured controls/reset/final evaluation"
                )
            check_actual_seed(
                self.config["split"], episode.seed, episode.requested_seed
            )
            manifests.append((episode, episode.manifest(world)))
        results = []
        for episode, manifest in manifests:
            path = episode.root / "manifest.json"
            self.budget.json(path, manifest, diagnostic=True)
            results.append(
                {
                    "manifest": upstream_eval.artifact(path),
                    "admitted": manifest["admission"]["complete_current_success"],
                }
            )
        return results

    def partial_manifests(self) -> list[dict[str, Any]]:
        results = []
        for episode in self.episodes:
            path = episode.root / "manifest.json"
            if not path.exists():
                self.budget.json(path, episode.manifest(), diagnostic=True)
            results.append(
                {"manifest": upstream_eval.artifact(path), "admitted": False}
            )
        return results


class EnvRecorder:
    """Transparent public SimTaskEnv proxy; .env remains the original raw env."""

    def __init__(self, delegate: Any, collection: Collection):
        self.delegate, self.collection, self.env = delegate, collection, delegate.env
        collection.env = self
        collection.physics = upstream_eval_physics(self.env)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    def reset(self, seed: int, options: dict[str, Any] | None = None) -> Any:
        self.collection.finish_current()
        attempt: dict[str, Any] = {
            "seed": seed,
            "requested_seed": self.collection.config["sim_eval"]["seed"]
            + len(self.collection.episodes),
            "options": _json_copy(options),
        }
        self.collection.reset_attempts.append(attempt)
        try:
            check_actual_seed(
                self.collection.config["split"], seed, attempt["requested_seed"]
            )
            obs = self.delegate.reset(seed=seed, options=options)
        except Exception as error:
            attempt.update(
                {"status": "failed", "error": f"{type(error).__name__}: {error}"}
            )
            raise
        attempt["status"] = "succeeded"
        self.collection.physics = upstream_eval_physics(self.env)
        Episode(self.collection, self.env, obs, seed)
        return obs

    def forget_arm_state(self) -> Any:
        self.collection.reset_attempts.append({"operation": "forget_arm_state"})
        return self.delegate.forget_arm_state()

    def obs(self) -> Any:
        result = self.delegate.obs()
        self._episode().note_observation(result, "obs")
        return result

    def _episode(self) -> Episode:
        if self.collection.current is None:
            raise RuntimeError("No successfully reset demonstration episode")
        return self.collection.current

    def step_one(self, action: Any) -> Any:
        episode = self._episode()
        if episode.controls >= MAX_CONTROLS:
            raise RuntimeError("Original control horizon exceeded")
        if episode.frame_index != episode.controls:
            raise RuntimeError("Missing exact pre-control video frame")
        issued = _numeric(np.asarray(action, dtype=np.float32), "float32", (14,))
        provenance = episode.action_source(action)
        result = self.delegate.step_one(action)
        episode.actions.append(issued)
        episode.ctrl.append(
            _numeric(self.env.data.ctrl, "float64", (self.env.model.nu,))
        )
        episode.judges.append(None)
        episode.note_state()
        episode.provenance.append(_json_bytes(provenance))
        return result

    def evaluate(self) -> Any:
        result = self.delegate.evaluate()
        self._episode().judge(result)
        return result

    def render_cameras(self) -> Any:
        result = self.delegate.render_cameras()
        self._episode().frame(result)
        return result

    def close(self) -> Any:
        try:
            self.collection.finish_current()
        except BaseException:
            self.delegate.close()
            raise
        return self.delegate.close()

    # Source aliases bind base functions. Override all four explicitly.
    step_one_vanilla = step_one
    evaluate_vanilla = evaluate
    obs_vanilla_state = obs
    render_cameras_vanilla_state = render_cameras


def upstream_eval_physics(native: Any) -> dict[str, Any]:
    dt, decimation = float(native.model.opt.timestep), int(native._control_decimation)
    if dt != 0.002 or decimation != 17:
        raise ValueError(
            "Demonstration physics must remain original dt0.002/decimation17"
        )
    return {
        "physics_dt": dt,
        "control_decimation": decimation,
        "control_hz": 1.0 / (dt * decimation),
    }


class PolicyRecorder:
    def __init__(self, delegate: Any, collection: Collection):
        self.delegate, self.collection = delegate, collection

    def __getattr__(self, name: str) -> Any:
        # Compile and RTC warmups remain bound to the original policy object.
        return getattr(self.delegate, name)

    def infer(
        self,
        obs: Any,
        noise: Any = None,
        action_prefix: Any = None,
        prefix_length: Any = 0,
    ) -> Any:
        episode = self.collection.current
        if episode is None:
            raise RuntimeError("Sampler was invoked before successful native reset")
        request = episode.before_infer(obs, noise, action_prefix, prefix_length)
        result = self.delegate.infer(
            obs, noise=noise, action_prefix=action_prefix, prefix_length=prefix_length
        )
        episode.after_infer(request, result)
        return result


@contextlib.contextmanager
def instrument(sim_env_module: Any, eval_module: Any, collection: Collection):
    """Scope constructor interposition to this original run_eval invocation."""
    env_class, policy_class = sim_env_module.SimTaskEnv, eval_module.VLAInferencePolicy

    def make_env(*args: Any, **kwargs: Any) -> EnvRecorder:
        delegate = env_class(*args, **kwargs)
        try:
            return EnvRecorder(delegate, collection)
        except Exception:
            delegate.close()
            raise

    def make_policy(*args: Any, **kwargs: Any) -> PolicyRecorder:
        return PolicyRecorder(policy_class(*args, **kwargs), collection)

    sim_env_module.SimTaskEnv, eval_module.VLAInferencePolicy = make_env, make_policy
    try:
        yield
    finally:
        sim_env_module.SimTaskEnv, eval_module.VLAInferencePolicy = (
            env_class,
            policy_class,
        )


def software_pins(modules: list[Any], created_at: str) -> dict[str, Any]:
    versions = {}
    for name in (
        "abc-minimal",
        "numpy",
        "torch",
        "mujoco",
        "mujoco-warp",
        "warp-lang",
        "imageio",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    return {
        "created_at": created_at,
        "python": platform.python_version(),
        "upstream_reference_commit": UPSTREAM_COMMIT,
        "distributions": versions,
        "sources": {
            module.__name__: upstream_eval.artifact(Path(module.__file__))
            for module in modules
        },
        "recorder": upstream_eval.artifact(Path(__file__)),
    }


def run(config_path: Path, output: Path) -> dict[str, Any]:
    config, metadata = load_config(config_path)
    output = output.resolve()
    if output.exists() and (
        not output.is_dir()
        or any(
            path.name not in {"job.json", "run.log"}
            or path.is_symlink()
            or not path.is_file()
            for path in output.iterdir()
        )
    ):
        raise ValueError(
            "Output may contain only canonical job.json/run.log bookkeeping"
        )
    require_subset_lease()
    output.mkdir(parents=True, exist_ok=True)
    collection = Collection(config, output)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "created_at": collection.created_at,
        "phase": PHASE,
        "status": "running",
        "split": config["split"],
        "input_config": upstream_eval.artifact(config_path),
        "checkpoint": config["checkpoint"],
        "metadata": config["metadata"],
        "episodes": [],
        "blockers": [],
        "claim_scope": "Author ABC-VLA generated simulation trajectories; no human/Atlas/dynamics approval",
        "output_byte_limit": config["max_output_bytes"],
        "byte_limit_scope": "Owned payloads bounded before each write; original MP4/native XML checked at boundaries and final admission",
        "compile_resource_environment": {
            key: os.environ[key]
            for key in upstream_eval.RESOURCE_ENV
            if key in os.environ
        },
    }
    started = time.monotonic()
    collection.budget.json(output / "receipt.json", receipt)
    try:
        if (
            upstream_eval.artifact(Path(config["checkpoint"]["path"]))
            != config["checkpoint"]
        ):
            raise ValueError("Checkpoint SHA-256 differs from its pin")
        from abc_minimal import config as original_config
        from abc_minimal import eval_policy, policy, sim_env
        from abc_sim import env as native_env

        collection.budget.json(
            output / "software.json",
            software_pins(
                [original_config, eval_policy, policy, sim_env, native_env],
                collection.created_at,
            ),
        )
        receipt["software"] = upstream_eval.artifact(output / "software.json")
        collection.producer = {
            "config": receipt["input_config"],
            "checkpoint": config["checkpoint"],
            "metadata": config["metadata"],
            "software": receipt["software"],
        }
        model = metadata["model_config"]
        vla = original_config.VLAModelConfig(
            camera_keys=tuple(model["camera_keys"]),
            backbone=original_config.GemmaVLAConfig(**model["backbone"]),
            dit=original_config.VLADiTConfig(**model["dit"]),
        )
        sim = original_config.SimEvalConfig(
            checkpoint=config["checkpoint"]["path"],
            output_dir=str(output / "original_eval"),
            vla_model=vla,
            **config["sim_eval"],
        )
        collection.budget.json(output / "requested_sim_eval.json", asdict(sim))
        with instrument(sim_env, eval_policy, collection):
            eval_policy.run_eval(sim)
        collection.finish_current()
        summary_path = output / "original_eval" / "summary.json"
        summary = upstream_eval._read_json(summary_path)
        receipt["original_summary"] = upstream_eval.artifact(summary_path)
        receipt["episodes"] = collection.reconcile(summary)
        collection.budget.json(output / "resolved_sim_eval.json", summary["config"])
        receipt["resolved_config"] = upstream_eval.artifact(
            output / "resolved_sim_eval.json"
        )
        receipt["physics"] = collection.physics
        receipt["status"] = "completed"
    except (Exception, KeyboardInterrupt) as error:
        receipt["status"] = (
            "cancelled" if isinstance(error, KeyboardInterrupt) else "failed"
        )
        receipt["blockers"] = [f"{type(error).__name__}: {error}"]
        try:
            collection.finish_current()
            receipt["episodes"] = collection.partial_manifests()
        except Exception as cleanup_error:  # noqa: BLE001 -- preserve native failure and report independent cleanup failure
            receipt["blockers"].append(
                f"Artifact cleanup: {type(cleanup_error).__name__}: {cleanup_error}"
            )
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        receipt["output_bytes_before_final_receipt"] = collection.budget.size()
        temporary = output / "receipt.final.json"
        collection.budget.json(temporary, receipt, diagnostic=True)
        temporary.replace(output / "receipt.json")
        collection.budget.check(diagnostic=True)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == "__main__":
    main()
