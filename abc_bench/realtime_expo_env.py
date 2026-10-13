"""ABC environment boundary for the unchanged Real-Time EXPO-FT client.

No inference, learning, control-rate pacing, or autonomous physics lives here.
One nonterminal ``step`` issues one original absolute14 command (17 physics
steps, 0.034 seconds). Native dependencies load only after the canonical lease.
"""

from __future__ import annotations

import atexit
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from abc_bench.expoft_train_profile import check_seed, reset_world
from abc_bench.fasttd3_host import make_original_env
from abc_bench.subset import require_subset_lease

CONTROL_DT = 0.034
CONTROL_HZ = 1.0 / CONTROL_DT
PROMPT = "put plastic bottles in bin"
CAMERAS = {
    "top": "top_image",
    "left": "left_wrist_image",
    "right": "right_wrist_image",
}


def _action(value: Any) -> np.ndarray:
    action = np.asarray(value, dtype=np.float32)
    if action.shape != (14,) or not np.isfinite(action).all():
        raise ValueError("ABC requires finite float32 absolute14 commands")
    # The upstream client reserves an all-minus-one vector and never forwards it.
    if np.allclose(action, -1.0):
        raise ValueError("The author client reserves all-minus-one as no-action")
    return action.copy()


class ABCYamEnv:
    """One native TRAIN world behind the author's environment interface.

    State/actions: left joints1..6, left gripper, right joints1..6, right
    gripper. Joints use radians; grippers retain ABC's policy coordinates.
    Images are actual uint8 HWC RGB[168,224,3]. No command clipping is added.
    ``env_usage='eval'`` is deliberately unavailable until a captured contract
    is supplied; random TRAIN resets are not matched frozen evaluation.
    """

    _active: ABCYamEnv | None = None

    def __init__(
        self,
        *,
        episode_seeds: list[int],
        example_action: Any,
        event_log_path: str,
        control_hz: float = CONTROL_HZ,
        max_episode_steps: int = 3540,
        env_usage: str = "train",
        language_instruction: str = PROMPT,
        video_dir: str = "",
        **task_config: Any,
    ):
        if env_usage != "train":
            raise ValueError("ABCYamEnv currently admits explicit TRAIN resets only")
        if language_instruction != PROMPT:
            raise ValueError(f"ABC task prompt must be {PROMPT!r}")
        if not math.isfinite(control_hz) or not math.isclose(
            control_hz, CONTROL_HZ, rel_tol=0.0, abs_tol=1e-10
        ):
            raise ValueError("ABC control_hz must equal 1/0.034, not nominal30Hz")
        if type(max_episode_steps) is not int or not 1 <= max_episode_steps <= 3540:
            raise ValueError("ABC horizon must be an explicit1..3540 native controls")
        if (
            not isinstance(episode_seeds, (list, tuple))
            or not episode_seeds
            or len(set(episode_seeds)) != len(episode_seeds)
        ):
            raise ValueError("ABC requires an explicit unique TRAIN seed sequence")
        self.seeds = tuple(check_seed(seed) for seed in episode_seeds)
        example = np.asarray(example_action, dtype=np.float32)
        if example.shape == (1, 14):
            example = example[0]
        self.example_action = _action(example)
        log_path = Path(event_log_path)
        if not log_path.is_absolute() or not log_path.parent.is_dir():
            raise ValueError(
                "event_log_path needs an existing absolute parent directory"
            )
        self.video_dir = Path(video_dir) if video_dir else None
        if self.video_dir is not None and (
            not self.video_dir.is_absolute() or not self.video_dir.is_dir()
        ):
            raise ValueError("video_dir must be an existing absolute directory")
        if ABCYamEnv._active is not None:
            raise RuntimeError("One live ABC world per original rollout-client process")
        require_subset_lease()  # Before opening outputs or any native dependency.
        self._events = log_path.open("x", encoding="utf-8")
        self.env = None
        self.closed, self.writer = False, None
        self.episode_index = -1
        self.controls = 0
        self.done = self.success = False
        self.native_reward = 0.0
        self._last_observation = None
        self._video_control = -1
        self._startup_pending = True
        try:
            self.env = make_original_env(
                {"save_video": True, "max_episode_steps": max_episode_steps}
            )
            if (
                self.env.robot_names != ["left", "right"]
                or self.env.single_timestep_action_dim != 14
                or self.env._control_decimation != 17
                or not math.isclose(float(self.env.model.opt.timestep), 0.002)
            ):
                raise ValueError("Require original dual-YAM14/dt0.002/decimation17")
            if not np.array_equal(self.example_action, _action(self.env.get_init_q())):
                raise ValueError(
                    "Startup example_action differs from native initial hold"
                )
            ABCYamEnv._active = self
            self._record("constructed", control_hz=CONTROL_HZ, control_dt=CONTROL_DT)
            atexit.register(self.close)
        except BaseException:
            self.close()
            raise

    def _record(self, operation: str, **fields: Any) -> None:
        record = {
            "schema": "sadhana.abc.realtime-expo-env/v1",
            "operation": operation,
            "monotonic_ns": time.monotonic_ns(),
            "wall_time_ns": time.time_ns(),
            "episode_index": self.episode_index,
            "actual_controls": self.controls,
            **fields,
        }
        if self.env is not None:
            record.update(
                control_step=int(self.env.cur_step),
                simulation_time_s=float(self.env.data.time),
            )
        self._events.write(json.dumps(record, allow_nan=False) + "\n")
        self._events.flush()

    def _open(self) -> None:
        if self.closed:
            raise RuntimeError("ABC environment is closed")
        require_subset_lease()

    def reset(self) -> dict[str, Any]:
        self._open()
        if self.episode_index + 1 >= len(self.seeds):
            raise RuntimeError("ABC explicit TRAIN seed sequence is exhausted")
        self._finish_video()
        requested = self.seeds[self.episode_index + 1]
        raw, randomization, actual, attempts = reset_world(self.env, requested)
        evaluation = self.env.evaluate_task().to_info(squeeze=True)
        if evaluation["num_active_bottles"] != 5 or evaluation["success"]:
            raise RuntimeError(
                "ABC TRAIN reset must start an unsolved five-bottle task"
            )
        self.episode_index += 1
        self.done = self.success = False
        self.native_reward = float(evaluation["reward"])
        self._last_observation = None
        self._video_control = -1
        if self._startup_pending and not np.array_equal(
            self.example_action, _action(self.env.get_state())
        ):
            raise ValueError(
                "Startup reset state differs from the declared native hold"
            )
        if self.video_dir is not None:
            import imageio.v2 as imageio

            path = self.video_dir / f"abc-train-{self.episode_index:04d}.mp4"
            if path.exists():
                raise FileExistsError(f"Refuse existing episode video: {path}")
            self.writer = imageio.get_writer(str(path), fps=CONTROL_HZ)
        self._record(
            "reset",
            requested_seed=requested,
            actual_seed=actual,
            randomization=randomization,
            reset_attempts=attempts,
            native_task_eval=evaluation,
        )
        return self._observation(raw)

    def _observation(self, raw: dict[str, Any]) -> dict[str, Any]:
        state = _action(raw["state"])
        observation = {"state": state, "prompt": PROMPT}
        native_time = float(self.env.data.time)
        for camera, key in CAMERAS.items():
            image = np.asarray(raw["images"][camera])
            timestamp = float(np.asarray(raw["camera_timestamps"][camera]))
            if (
                image.shape != (3, 168, 224)
                or image.dtype != np.uint8
                or raw["masks"][camera] is not True
                or not math.isclose(timestamp, native_time, rel_tol=0, abs_tol=1e-10)
            ):
                raise ValueError(
                    "Require actual native synchronized CHW ABC RGB cameras"
                )
            observation[key] = image.transpose(1, 2, 0).copy()
        self._last_observation = observation
        if self.writer is not None and self._video_control != self.env.cur_step:
            self.writer.append_data(observation["top_image"])
            self._video_control = self.env.cur_step
        return observation

    def get_observation(self) -> dict[str, Any]:
        self._open()
        if self.episode_index < 0:
            raise RuntimeError("Reset ABC before observations or controls")
        before = float(self.env.data.time)
        observation = self._observation(self.env.get_obs())
        if float(self.env.data.time) != before:
            raise RuntimeError("ABC observation advanced native physics")
        self._record("observation", physics_advanced=False)
        return observation

    def get_info_for_step(self) -> tuple[bool, bool, float, float]:
        self._open()
        if self.episode_index < 0:
            raise RuntimeError("Reset ABC before step information")
        if self.done:
            self._finish_video()
        # Match the original DROID environment's sparse success and terminal mask.
        return self.done, self.success, float(self.success), 0.0 if self.done else 1.0

    def step(self, action: Any) -> dict[str, Any]:
        self._open()
        applied = _action(action)
        if self.episode_index < 0:
            raise RuntimeError("Reset ABC before controls")
        if self.done:
            # Original train_pi_robo dispatches once after observing done, before
            # its reset branch. Acknowledge that command; keep the terminal scene.
            self._record(
                "terminal_noop", action14=applied.tolist(), physics_advanced=False
            )
            return {"executed_action": applied, "physics_advanced": False}
        if self._startup_pending and not np.array_equal(applied, self.example_action):
            raise ValueError("First author command must equal the declared native hold")
        before = float(self.env.data.time)
        control_before = int(self.env.cur_step)
        started = time.monotonic_ns()
        _, reward, terminated, truncated, info = self.env.step(
            applied, render_obs=False
        )
        if self.env.cur_step != control_before + 1 or not math.isclose(
            float(self.env.data.time) - before, CONTROL_DT, abs_tol=1e-9
        ):
            raise RuntimeError(
                "One ABC control must advance exactly17 native physics steps"
            )
        self._startup_pending = False
        self.controls += 1
        self.native_reward = float(reward)
        self.success = bool(info["task_success"])
        self.done = bool(terminated or truncated)
        self._record(
            "step",
            started_monotonic_ns=started,
            action14=applied.tolist(),
            physics_advanced=True,
            physics_steps_advanced=17,
            terminated=bool(terminated),
            truncated=bool(truncated),
            native_reward=self.native_reward,
            native_task_eval=info.get("task_eval"),
            author_reward=float(self.success),
        )
        return {"executed_action": applied, "physics_advanced": True}

    def render(self) -> np.ndarray:
        return self.get_observation()["top_image"]

    def _finish_video(self) -> None:
        if self.writer is not None:
            writer, self.writer = self.writer, None
            writer.close()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        atexit.unregister(self.close)
        try:
            self._finish_video()
        finally:
            try:
                if self.env is not None:
                    self.env.close()
                    self._record("closed", video_closed=True, native_closed=True)
            finally:
                self._events.close()
                if ABCYamEnv._active is self:
                    ABCYamEnv._active = None
