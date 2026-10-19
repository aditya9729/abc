"""Finite TRAIN recording around the unchanged original EXPO evaluator.

The qualified Sadhana frame writer and native adapter are imported normally.
No model, simulator, replay buffer or device is created at module import.
"""

from __future__ import annotations

import atexit
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
from time import perf_counter

SCHEMA = "sadhana.expo_train_collection/1"
RESULT_SCHEMA = "sadhana.expo_train_collection_result/1"
EPISODE_SCHEMA = "sadhana.expo_train_episode/1"
PROMPT = "put the remote in the organizer"
ROOT = Path("/home/user/aditya")
EPISODE_BYTES = 250_000_000
METADATA_BYTES = 2_000_000
MODULES = (
    "sadhana_r1lite.recovery_continuation",
    "sadhana_r1lite.recovery_capture",
    "sadhana_r1lite.expo_env",
    "sadhana_r1lite.expo_config",
    "sadhana_r1lite.expo_data",
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pin(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def authenticate(item):
    path = Path(item["path"])
    require(
        path.is_absolute() and path.is_file() and not path.is_symlink(),
        "Regular absolute input required",
    )
    require(pin(path) == item, f"Input changed: {path}")
    return path


def read(item):
    return json.loads(authenticate(item).read_text())


def validate_seeds(seeds, startup):
    require(
        isinstance(seeds, list) and 1 <= len(seeds) <= 4,
        "Declare one to four TRAIN starts",
    )
    require(
        all(type(seed) is int and seed >= 21004 for seed in seeds),
        "Fresh TRAIN starts must be >=21004; DEV/FINAL excluded",
    )
    require(len(set(seeds)) == len(seeds), "Duplicate TRAIN starts")
    require(
        not any(30000 <= seed <= 280001 for seed in seeds),
        "Existing online TRAIN schedule is reserved",
    )
    require(type(startup) is int and startup == 20999, "Use unscored startup20999")


def load_descriptor(path):
    path = Path(path)
    require(
        path.is_absolute() and not path.is_symlink(), "Absolute descriptor required"
    )
    value = json.loads(path.read_text())
    require(
        set(value)
        == {
            "schema",
            "split",
            "mode",
            "seeds",
            "startup_seed",
            "reservation",
            "task_settings",
            "native_config",
            "recording_modules",
            "policy_reference",
            "output_dir",
            "lineage",
        },
        "Unknown collection descriptor fields",
    )
    require(
        value["schema"] == SCHEMA and value["split"] == "TRAIN",
        "TRAIN collection schema required",
    )
    require(
        value["mode"] == "original_base_evaluator",
        "Only the original frozen base-evaluator condition",
    )
    validate_seeds(value["seeds"], value["startup_seed"])
    reservation = read(value["reservation"])
    require(
        reservation
        == {
            "schema": "sadhana.expo_train_seed_reservation/1",
            "owner": "/root",
            "split": "TRAIN",
            "seeds": value["seeds"],
            "startup_seed": value["startup_seed"],
        },
        "Coordinator TRAIN reservation required",
    )
    settings = read(value["task_settings"])
    require(
        settings["schema"] == "sadhana.r1lite_expo_task/1"
        and settings["episode_seeds"] == [value["startup_seed"], *value["seeds"]],
        "Task reset schedule differs",
    )
    require(
        settings.get("record_video") is False and not settings.get("event_log_path"),
        "Recorder owns all media and event output",
    )
    require(
        settings["env_config_path"] == str(authenticate(value["native_config"])),
        "Native task binding differs",
    )
    require(
        set(value["recording_modules"]) == set(MODULES),
        "Pin the existing recording API modules",
    )
    for item in value["recording_modules"].values():
        authenticate(item)
    reference = read(value["policy_reference"])
    require(
        reference["program"] == "eval_droid_policy.py",
        "Original evaluator reference required",
    )
    validate_evaluator_arguments(reference["arguments"], episodes=None)
    require(
        value["lineage"]
        == {
            "source_episode": 12,
            "scene_reference": 11,
            "new_independent_source_lineage": False,
        },
        "Preserve the existing scene ancestry",
    )
    output = Path(value["output_dir"])
    require(
        output.is_absolute()
        and output.resolve().is_relative_to(ROOT)
        and len(str(output.resolve())) <= 512,
        "Output must be under aditya",
    )
    return value


def validate_evaluator_arguments(arguments, *, episodes):
    """Binder check; the recorder alone cannot attest remote server arguments."""
    expected = {
        "only_base_actions": "True",
        "checkpoint_step": "0",
        "seed": "42",
        "replan_steps": "8",
        "delay": "5",
        "sim_latency": "0",
        "config.N": "32",
        "config.n_edit_samples": "32",
    }
    if episodes is not None:
        expected["num_episodes"] = str(episodes)
        expected["save_video"] = "False"
    for key, value in expected.items():
        matches = [
            arg
            for arg in arguments
            if arg.split("=", 1)[0] in (f"--{key}", f"--no{key}")
        ]
        require(
            matches == [f"--{key}={value}"],
            f"Original base evaluator requires one --{key}={value}",
        )


def recording_api(descriptor):
    for name, item in descriptor["recording_modules"].items():
        path = authenticate(item)
        spec = importlib.util.find_spec(name)
        require(
            spec is not None and Path(spec.origin).resolve() == path.resolve(),
            f"Installed recording API differs: {name}",
        )
    return importlib.import_module("sadhana_r1lite.recovery_continuation")


class TrainRecordingEnv:
    """Record T+1 native frames for each T-command episode, including failures."""

    def __init__(self, env, descriptor, descriptor_pin, api, *, observe=None):
        self.env, self.descriptor, self.descriptor_pin, self.api = (
            env,
            descriptor,
            descriptor_pin,
            api,
        )
        self.observe = observe or api.observe_native
        self.output = api.BoundedOutput(
            descriptor["output_dir"], max_bytes=METADATA_BYTES, reserve=100_000
        )
        self.output.json(
            "INPUTS.json",
            {
                "descriptor": descriptor_pin,
                "resolved": descriptor,
                "policy_binding": "declared; actual canonical server arguments require separate root acceptance",
            },
        )
        self.resets = self.total_controls = self.controls = 0
        self.episodes = []
        self.episode_output = self.last_snapshot = None
        self.terminal = self.closed = False
        self.error = self.last_unfinished_proposal = None
        self.started = perf_counter()

    def _fail(self, error):
        self.error = self.error or f"{type(error).__name__}: {error}"[:2000]
        self.close()

    def reset(self):
        try:
            require(
                not self.closed and self.resets <= len(self.descriptor["seeds"]),
                "Reset schedule exhausted",
            )
            require(self.episode_output is None, "Cannot reset an unfinished episode")
            observation = self.env.reset()
            schedule = [self.descriptor["startup_seed"], *self.descriptor["seeds"]]
            require(self.env.seed == schedule[self.resets], "Native reset seed differs")
            require(
                self.env.native_controls == self.total_controls,
                "Reset executed physical controls",
            )
            require(
                list(self.env.get_info_for_step()) == [False, False, 0.0, 1.0],
                "Reset labels differ",
            )
            self.resets += 1
            if self.resets == 1:
                self.output.json(
                    "STARTUP.json",
                    {"seed": self.env.seed, "controls": 0, "scored": False},
                )
                return observation
            self.controls, self.terminal = 0, False
            self.episode_output = self.api.BoundedOutput(
                self.output.root / "episodes" / f"{self.resets - 2:04d}"
            )
            self.episode_output.require_frame()
            self.last_snapshot = self.observe(self.env, 0)
            self.episode_output.frame(
                0,
                self.last_snapshot,
                {
                    "schema": EPISODE_SCHEMA,
                    "kind": "reset",
                    "seed": self.env.seed,
                    "episode_control": 0,
                    "is_hil": False,
                    "teacher_data": False,
                },
            )
            return observation
        except BaseException as error:
            self._fail(error)
            raise

    def step(self, action):
        import numpy as np

        try:
            require(
                not self.closed
                and self.episode_output is not None
                and not self.terminal
                and self.controls < 900,
                "Active scored episode required",
            )
            self.episode_output.require_frame()
            proposed = np.asarray(action, dtype=np.float64)
            require(
                proposed.shape == (14,) and np.isfinite(proposed).all(),
                "Finite physical action14 required",
            )
            before = self.last_snapshot["public"]
            self.last_unfinished_proposal = proposed.tolist()
            reply = self.env.step(proposed)
            require(
                reply.get("terminal_noop") is False, "Terminal no-op is not experience"
            )
            self.controls += 1
            self.total_controls += 1
            require(
                self.env.native_controls == self.total_controls,
                "Native control counter differs",
            )
            info = self.api.plain(self.env.last_native_info)
            require(type(info["success"]) is bool, "Boolean success label required")
            require(info["step"] == self.controls, "Episode control counter differs")
            applied = np.asarray(reply["executed_action"], dtype=np.float64)
            require(
                applied.shape == (14,) and np.isfinite(applied).all(),
                "Invalid applied action",
            )
            self.api.same_array(
                applied,
                np.asarray(info["action_applied"], dtype=np.float64),
                "actual executed command",
            )
            require(
                type(info["terminated"]) is bool
                and type(info["truncated"]) is bool
                and not (info["terminated"] and info["truncated"]),
                "Invalid terminal flags",
            )
            self.terminal = info["terminated"] or info["truncated"]
            labels = list(self.env.get_info_for_step())
            require(
                labels
                == [
                    self.terminal,
                    info["success"],
                    float(info["success"]),
                    float(not info["terminated"]),
                ],
                "Historical reward/mask labels differ",
            )
            snapshot = self.observe(self.env, self.controls)
            require(
                abs(snapshot["public"]["time_s"] - before["time_s"] - 1 / 15) < 1e-8,
                "Native 15Hz clock differs",
            )
            self.episode_output.frame(
                self.controls,
                snapshot,
                {
                    "schema": EPISODE_SCHEMA,
                    "kind": "policy_control",
                    "seed": self.env.seed,
                    "episode_control": self.controls,
                    "state_before": before["state14"],
                    "time_before_s": before["time_s"],
                    "proposed": proposed.tolist(),
                    "applied": applied.tolist(),
                    "native_info": info,
                    "done_success_reward_mask": labels,
                    "is_hil": False,
                    "teacher_data": False,
                },
            )
            self.last_snapshot = snapshot
            self.last_unfinished_proposal = None
            if self.terminal:
                self._finish_episode()
            return reply
        except BaseException as error:
            self._fail(error)
            raise

    def _finish_episode(self):
        output = self.episode_output
        result = {
            "schema": EPISODE_SCHEMA,
            "seed": self.env.seed,
            "controls": self.controls,
            "complete": self.terminal
            and self.error is None
            and self.last_unfinished_proposal is None,
            "labels": list(self.env.get_info_for_step()),
            "files": list(output.files),
            "error": self.error,
            "last_unfinished_proposal": self.last_unfinished_proposal,
        }
        item = output.json("RESULT.json", result, final=True)
        self.episodes.append(item)
        self.episode_output = None

    def get_observation(self):
        return self.env.get_observation()

    def get_info_for_step(self):
        return self.env.get_info_for_step()

    def render(self):
        return self.env.render()

    def close(self):
        if self.closed:
            return
        self.closed = True
        close_error = None
        try:
            if self.episode_output is not None:
                self._finish_episode()
        finally:
            try:
                self.env.close()
            except BaseException as error:  # noqa: BLE001 -- persist, then re-raise close failure
                close_error = error
                self.error = (
                    self.error or f"close {type(error).__name__}: {error}"[:2000]
                )
            complete = (
                self.error is None
                and len(self.episodes) == len(self.descriptor["seeds"])
                and self.resets == len(self.episodes) + 1
                and self.terminal
            )
            self.output.json(
                "RESULT.json",
                {
                    "schema": RESULT_SCHEMA,
                    "status": "completed_candidate"
                    if complete
                    else "incomplete_or_failed",
                    "artifact_accepted": False,
                    "descriptor": self.descriptor_pin,
                    "episodes": self.episodes,
                    "resets": self.resets,
                    "actual_controls": self.total_controls,
                    "error": self.error,
                    "elapsed_seconds": perf_counter() - self.started,
                    "prompt": PROMPT,
                    "split": "TRAIN",
                    "lineage": self.descriptor["lineage"],
                    "scope": "Historical judge only; autonomous starts, no strict recovery, new source lineage, or WBC claim",
                    "max_capture_bytes": METADATA_BYTES
                    + len(self.descriptor["seeds"]) * EPISODE_BYTES,
                },
                final=True,
            )
        if close_error is not None:
            raise close_error


def make_collection_env(**kwargs):
    descriptor_pin = dict(kwargs.pop("collection_descriptor"))
    descriptor = load_descriptor(authenticate(descriptor_pin))
    require(
        kwargs.get("env_usage") == "eval", "Original evaluator transport mode required"
    )
    require(
        list(kwargs.get("episode_seeds", []))
        == [descriptor["startup_seed"], *descriptor["seeds"]],
        "Client schedule differs",
    )
    require(
        kwargs.get("record_video") is False and not kwargs.get("event_log_path"),
        "Disable other output writers",
    )
    for key, expected in (
        ("max_episode_steps", 900),
        ("auto_reset_steps", 901),
        ("control_hz", 15.0),
        ("action_horizon", 16),
        ("env_config_path", descriptor["native_config"]["path"]),
    ):
        require(kwargs.get(key) == expected, f"Task field differs: {key}")
    api = recording_api(descriptor)
    from sadhana_r1lite.expo_env import ExpoR1LiteEnv

    env = ExpoR1LiteEnv(**kwargs)
    try:
        wrapped = TrainRecordingEnv(env, descriptor, descriptor_pin, api)
    except BaseException:
        env.close()
        raise
    atexit.register(wrapped.close)
    return wrapped


def make_collection_task_config(descriptor_path):
    """Client shim calls this; the server retains its original task/data factory."""
    descriptor = load_descriptor(descriptor_path)
    recording_api(descriptor)
    from sadhana_r1lite.expo_config import make_task_config

    config = make_task_config(authenticate(descriptor["task_settings"]))
    require(config.language_instruction == PROMPT, "Task prompt differs")
    config.env = make_collection_env
    config.collection_descriptor = pin(descriptor_path)
    return config
