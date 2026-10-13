"""Bounded video observer for the unchanged author Square-Narrow grid evaluator.

This module calls ``evaluate_manifest``; it contains no policy or rollout loop.
Its scoped public SyncVectorEnv substitution is single-process instrumentation,
not an asynchronous evaluation. Run in a dedicated author child process.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import pickle
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from abc_bench.upstream_eval import artifact


def _digest(*arrays: Any) -> str:
    result = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        result.update(str((array.dtype.str, array.shape)).encode())
        result.update(array.tobytes())
    return result.hexdigest()


def _state(env: Any) -> str:
    state = env.sim.get_state()
    return _digest(state.qpos, state.qvel, np.float64(state.time))


def _render_guard(env: Any) -> tuple[str, str]:
    # Rendering must not alter the physics state or reset/policy RNG streams.
    rng = getattr(env, "rng", None)
    bit_generator = getattr(rng, "bit_generator", None)
    payload = (
        random.getstate(),
        np.random.get_state(),
        bit_generator.state if bit_generator is not None else None,
    )
    return _state(env), hashlib.sha256(pickle.dumps(payload)).hexdigest()


def _writer(path: Path) -> Any:
    import imageio.v2 as imageio

    return imageio.get_writer(path, fps=20, codec="libx264", macro_block_size=1)


@contextmanager
def _replace(module: Any, name: str, value: Any):
    original = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, original)


class _Stream:
    def __init__(self, directory: Path, point: dict, env: Any, video: bool):
        self.point = point
        self.env = env
        self.length = 0
        self.reward = 0.0
        self.success = False
        self.complete = False
        self.author_outcome = None
        self.frames = 0
        self.path = directory / f"point-{int(point['point_idx']):06d}.mp4"
        self.trace_path = self.path.with_suffix(".jsonl")
        self.trace = self.trace_path.open("x")
        self.writer = None
        try:
            self.writer = _writer(self.path) if video else None
            self._row({"event": "reset", "state": _state(env)})
            self.frame()
        except BaseException:
            self.close()
            raise

    def _row(self, row: dict) -> None:
        self.trace.write(json.dumps(row, allow_nan=False) + "\n")

    def frame(self) -> None:
        if self.writer is None:
            return
        before = _render_guard(self.env)
        frame = self.env.render()
        if _render_guard(self.env) != before:
            raise ValueError("Rendering changed physics state or RNG")
        if frame.dtype != np.uint8 or frame.shape != (240, 320, 3):
            raise ValueError("Original agentview render must be uint8[240,320,3]")
        self.writer.append_data(frame)
        self.frames += 1

    def step(self, action: Any, before: str, result: tuple, max_steps: int) -> None:
        _, reward, done, info = result
        self.length += 1
        self.reward += float(reward)
        self.success = self.success or bool(info.get("success", False))
        self._row(
            {
                "event": "step",
                "step": self.length,
                "action": _digest(action),
                "before": before,
                "after": _state(self.env),
                "reward": float(reward),
                "done": bool(done),
                "success": bool(info.get("success", False)),
            }
        )
        self.frame()
        if done or self.success or self.length >= max_steps:
            self.complete = True
            self.close()

    def close(self) -> None:
        # Each resource gets its close attempt, even if the other close fails.
        try:
            if self.writer is not None:
                writer, self.writer = self.writer, None
                writer.close()
        finally:
            self.trace.close()

    def receipt(self) -> dict:
        return {
            "point": self.point,
            "episode_complete": self.complete,
            "author_outcome": self.author_outcome,
            "observed": {
                "success": int(self.success),
                "reward": self.reward,
                "length": self.length,
            },
            "frames": self.frames,
            "trace": artifact(self.trace_path),
            "video": artifact(self.path) if self.path.exists() else None,
        }


def _observer_class(
    original: type,
    env_module: Any,
    points: list[dict],
    selected: set[int],
    directory: Path,
    max_steps: int,
    video: bool,
    streams: list[_Stream],
) -> type:
    class ObservedSyncVectorEnv(original):
        def __init__(self, env_fns: list, *args: Any, **kwargs: Any):
            selected_slots = {
                i % len(env_fns)
                for i, p in enumerate(points)
                if int(p["point_idx"]) in selected
            }

            def render_factory(factory):
                def make():
                    create = env_module.create_robosuite_env

                    def with_render(*a, **kw):
                        # Forward the author's environment factory unchanged,
                        # except its explicit no-video setting. Physics,
                        # observations, controller and reset flags stay intact.
                        if kw.get("use_render_wrapper") is not False:
                            raise ValueError(
                                "Unexpected author render factory contract"
                            )
                        return create(*a, **{**kw, "use_render_wrapper": True})

                    with _replace(env_module, "create_robosuite_env", with_render):
                        return factory()

                return make

            factories = [
                render_factory(fn) if video and i in selected_slots else fn
                for i, fn in enumerate(env_fns)
            ]
            super().__init__(factories, *args, **kwargs)
            self.cursor = 0
            self.active: dict[int, _Stream] = {}

        def reset_with_placements(self, placements_per_env):
            if self.active or self.cursor + len(self) > len(points):
                raise ValueError("Unexpected author batch/reset sequence")
            result = super().reset_with_placements(placements_per_env)
            batch = points[self.cursor : self.cursor + len(self)]
            self.cursor += len(self)
            for i, point in enumerate(batch):
                if int(point["point_idx"]) in selected:
                    stream = _Stream(directory, point, self.envs[i], video)
                    streams.append(stream)
                    self.active[i] = stream
            return result

        def step(self, actions):
            before = {i: _state(stream.env) for i, stream in self.active.items()}
            result = super().step(actions)
            for i, stream in list(self.active.items()):
                stream.step(
                    actions[i], before[i], tuple(x[i] for x in result), max_steps
                )
                if stream.complete:
                    del self.active[i]
            return result

        def close(self):
            errors = []
            try:
                for stream in getattr(self, "active", {}).values():
                    try:
                        stream.close()
                    except Exception as exc:
                        errors.append(exc)
            finally:
                super().close()
            if errors:
                raise errors[0]

    return ObservedSyncVectorEnv


def evaluate_with_video(
    *, video_points: list[int], record_video: bool = True, **evaluation: Any
) -> dict:
    """Run the original sync evaluator and bind up to three videos to its rows.

    The output directory must be fresh. Resume and shards remain available via
    the original CLI, not this bounded visualization path. ``record_video=False``
    records the same state/action traces without enabling rendering for parity.
    """
    from mulligan.sim import envs, vec_env
    from mulligan.sim.eval import grid_eval

    if record_video and os.environ.get("MUJOCO_GL") != "osmesa":
        raise ValueError("Video recording requires explicit CPU MUJOCO_GL=osmesa")
    if evaluation.get("use_sync_envs") is not True:
        raise ValueError("Video observer requires explicit --use-sync-envs")
    if (
        evaluation.get("shard_idx") is not None
        or evaluation.get("n_shards") is not None
    ):
        raise ValueError("Video observer requires an unsharded fresh evaluation")
    if evaluation.get("eval_seed") is None or evaluation.get("env_seed") is None:
        raise ValueError("Both original eval_seed and env_seed are required")
    if (
        not 1 <= len(video_points) <= 3
        or any(type(point) is not int for point in video_points)
        or len(set(video_points)) != len(video_points)
    ):
        raise ValueError("Select one to three distinct video point indices")
    manifest = grid_eval.load_manifest(evaluation["manifest_path"])
    if manifest.get("task_key", "square_narrow") != "square_narrow":
        raise ValueError("This recording profile supports Square-Narrow only")
    points = manifest["points"]
    selected = set(video_points)
    if not selected <= {int(p["point_idx"]) for p in points}:
        raise ValueError("Video point index is absent from the genuine manifest")
    output = Path(evaluation["output_dir"])
    if output.is_symlink() or (output.exists() and any(output.iterdir())):
        raise ValueError("Video output must be fresh; existing artifacts are immutable")
    directory = output / "videos"
    directory.mkdir(parents=True)
    streams: list[_Stream] = []
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "kind": "original_mulligan_grid_video_observer",
        "status": "failed",
        "author_evaluation_completed": False,
        "video_points": video_points,
        "record_video": record_video,
        "camera": "agentview",
        "fps": 20,
        "render_size": [240, 320],
        "sync_instrumentation": True,
        "asynchronous_timing_equivalence": False,
        "manifest": artifact(Path(evaluation["manifest_path"])),
        "manifest_hash": manifest["manifest_hash"],
        "author_function_sha256": {
            name: hashlib.sha256(
                inspect.getsource(getattr(grid_eval, name)).encode()
            ).hexdigest()
            for name in ("evaluate_manifest", "step_rollouts_from_obs")
        },
    }
    observer = _observer_class(
        vec_env.SyncVectorEnv,
        envs,
        points,
        selected,
        directory,
        evaluation.get("max_steps", 400),
        record_video,
        streams,
    )
    try:
        with _replace(vec_env, "SyncVectorEnv", observer):
            result = grid_eval.evaluate_manifest(**evaluation)
        rows_path = Path(result["point_results_path"])
        rows = json.loads(rows_path.read_text())
        selected_rows = {
            int(row["point_idx"]): row
            for row in rows
            if int(row["point_idx"]) in selected
        }
        if len(streams) != len(selected) or set(selected_rows) != selected:
            raise ValueError("Selected videos lack authoritative author outcomes")
        for stream in streams:
            row = selected_rows[int(stream.point["point_idx"])]
            observed = stream.receipt()
            if not stream.complete or any(
                row[key] != value for key, value in observed["observed"].items()
            ):
                raise ValueError(
                    "Video trajectory differs from authoritative author outcome"
                )
            if record_video and (
                observed["frames"] != stream.length + 1
                or not observed["video"]
                or observed["video"]["bytes"] <= 0
            ):
                raise ValueError("Completed episode lacks a complete non-empty video")
            stream.author_outcome = row
        receipt.update(
            status="completed",
            author_evaluation_completed=True,
            point_results=artifact(rows_path),
            results=artifact(output / "results.json"),
        )
        return result
    except BaseException as exc:
        receipt["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        receipt["episodes"] = [stream.receipt() for stream in streams]
        temporary = directory / "receipt.tmp"
        temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        temporary.replace(directory / "receipt.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", required=True)
    parser.add_argument(
        "--point-manifest", dest="manifest_path", type=Path, required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--num-envs", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-action-samples", type=int, default=32)
    parser.add_argument("--use-sync-envs", action="store_true")
    parser.add_argument("--eval-seed", type=int, required=True)
    parser.add_argument("--env-seed", type=int, required=True)
    parser.add_argument(
        "--video-point", dest="video_points", type=int, action="append", required=True
    )
    parser.add_argument("--trace-only", action="store_true")
    args = vars(parser.parse_args())
    trace_only = args.pop("trace_only")
    evaluate_with_video(record_video=not trace_only, **args)


if __name__ == "__main__":
    main()
