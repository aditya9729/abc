"""Separate bounded native reach diagnostic using public ABC world interfaces."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
import signal
import socket
import stat
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from abc_bench import fasttd3_vector_host as native
from abc_bench.bottle_success_reward import BottlePotentialGeometry
from abc_bench.fasttd3_campaign_host import physics_snapshot
from abc_bench.fasttd3_host import receive_frame, write_json
from abc_bench.fasttd3_reach_contract import (
    ReachEpisode,
    nearest_distance,
    validate_config,
)
from abc_bench.fasttd3_scratch_stage import pin
from abc_bench.subset import require_subset_lease
from abc_bench.upstream_eval import artifact

MAX_FRAME_BYTES = native.MAX_FRAME_BYTES
require_cpu, dispatch = native.require_cpu, native.dispatch


def load_config(path: Path) -> dict:
    if path.stat().st_size > 1024**2:
        raise ValueError("Reach host configuration exceeds1MiB")
    config = validate_config(json.loads(path.read_text()))
    evidence = json.loads(pin(config["selection_evidence"]).read_text())
    if evidence.get("split") != "train" or evidence.get("validation_used") is not False:
        raise ValueError("Reach criterion requires TRAIN-only geometry evidence")
    return config


class ReachWorld(native.World):
    def __init__(self, config: dict, rank: int, output: Path):
        self.geometry = None
        self.completed_episodes = 0
        self.trace = None
        super().__init__(config, rank, output)
        self.trace = (output / "reach-controls.jsonl").open("x")

    def integration(self):
        import mujoco
        import numpy as np

        state = np.empty(
            mujoco.mj_stateSize(self.env.model, mujoco.mjtState.mjSTATE_INTEGRATION)
        )
        mujoco.mj_getState(
            self.env.model, self.env.data, state, mujoco.mjtState.mjSTATE_INTEGRATION
        )
        return state

    def check_physics(self, before=None) -> dict:
        current = physics_snapshot(self.env.data)
        valid = current["time_finite"] and all(current["finite"].values())
        valid = valid and not any(row["number"] for row in current["warnings"].values())
        if before is not None and valid:
            valid = math.isclose(
                current["time"], before["time"] + 0.034, rel_tol=0, abs_tol=1e-9
            )
        if not valid:
            path = self.output / f"physics-failure-{self.controls:08d}.json"
            write_json(
                path,
                {
                    "status": "failed",
                    "rank": self.rank,
                    "controls": self.controls,
                    "before": before,
                    "after": current,
                },
            )
            raise FloatingPointError("Invalid native physics; evidence: " + str(path))
        return current

    def reset(self) -> dict:
        # This documented public attribute only changes task termination. The
        # native five-bottle evaluator and all physics/control settings persist.
        self.env.terminate_on_success = False
        if self.geometry is not None:
            self.geometry.close()
        result = super().reset()
        import mujoco
        import numpy as np

        from abc_sim.task_eval import make_task_evaluator
        from abc_sim.task_specs import get_task_spec

        self.initial_physics = self.check_physics()
        if int(self.env.model.opt.disableflags) & int(
            mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL
        ):
            raise ValueError("Native actuator control clamping is disabled")
        if not np.asarray(self.env.model.actuator_ctrllimited).all():
            raise ValueError(
                "Every native actuator must retain its original control limit"
            )
        evaluator = make_task_evaluator(self.env.model, get_task_spec(native.TASK))
        judge = evaluator.evaluate_qpos_batch(
            np.array(self.env.data.qpos, copy=True)[None]
        ).to_info(squeeze=True)
        self.geometry = BottlePotentialGeometry(self.env.model, judge)
        before = self.integration()
        snapshot = self.geometry.snapshot(self.env.model, self.env.data, judge)
        after = self.integration()
        if before.tobytes() != after.tobytes():
            raise RuntimeError(
                "Reach geometry changed authoritative native INTEGRATION"
            )
        # Independently check the public qpos-only FK against owned native MjData.
        reference = mujoco.MjData(self.env.model)
        reference.qpos[:] = self.env.data.qpos
        mujoco.mj_kinematics(self.env.model, reference)
        indices = [self.env.model.body(name).id for name in judge["bottle_names"]]
        centers = reference.xpos[indices] + np.einsum(
            "bij,bj->bi",
            reference.xmat.reshape(-1, 3, 3)[indices],
            np.asarray(judge["bottle_center_offsets"]),
        )
        sites = [
            reference.site_xpos[self.env.model.site(name).id]
            for name in ("left_grasp_site", "right_grasp_site")
        ]
        if not np.array_equal(centers, snapshot.bottle_centers) or not np.array_equal(
            sites, snapshot.grasp_positions
        ):
            raise RuntimeError(
                "Reach COM/site geometry differs from independent owned FK"
            )
        self.reaching = ReachEpisode(
            nearest_distance(snapshot.bottle_centers, snapshot.grasp_positions)
        )
        self.peak_native_bottles = 0
        write_json(
            self.output / f"reach-reset-{self.episode_index:04d}.json",
            {
                "rank": self.rank,
                "episode_index": self.episode_index,
                "requested_seed": self.requested_seed,
                "seed": self.actual_seed,
                "geometry_preserves_integration": True,
                "geometry_matches_independent_FK": True,
                "integration_sha256": hashlib.sha256(before.tobytes()).hexdigest(),
                "scene": result["scene"],
                "reach": self.reaching.evidence(),
                "physics": self.initial_physics,
            },
        )
        return result

    def step(self, normalized: Any) -> dict:
        import numpy as np

        if self.closed:
            raise RuntimeError("Reach world is closed")
        action = np.asarray(normalized)
        if (
            action.shape != (14,)
            or action.dtype.kind not in "fiu"
            or not np.isfinite(action).all()
        ):
            raise ValueError("Action must be finite normalized numeric14")
        with np.errstate(over="ignore", invalid="ignore"):
            action = action.astype(np.float32)
        if not np.isfinite(action).all():
            raise ValueError("Action exceeds float32 bounds")
        schema = self.schema["action_schema"]
        low, high = np.asarray(schema["policy_low"]), np.asarray(schema["policy_high"])
        physical = (low + (np.clip(action, -1, 1) + 1) * 0.5 * (high - low)).astype(
            np.float32
        )
        before = self.check_physics()
        _, native_reward, native_terminated, truncated, info = self.env.step(
            physical, render_obs=False
        )
        after = self.check_physics(before)
        self.controls += 1
        self.episode_steps += 1
        if native_terminated or bool(truncated) != (
            self.episode_steps == self.config["max_episode_steps"]
        ):
            raise RuntimeError("Reach diagnostic must end only at its fixed horizon")
        judge = native.plain(info["task_eval"])
        if judge["num_active_bottles"] != 5 or not math.isclose(
            float(native_reward),
            judge["num_bottles_in_bin"] / 5,
            rel_tol=0,
            abs_tol=1e-7,
        ):
            raise RuntimeError("Native five-bottle judge/reward changed")
        self.peak_native_bottles = max(
            self.peak_native_bottles, judge["num_bottles_in_bin"]
        )
        state_before_geometry = self.integration()
        snapshot = self.geometry.snapshot(self.env.model, self.env.data, judge)
        if state_before_geometry.tobytes() != self.integration().tobytes():
            raise RuntimeError(
                "Reach geometry changed authoritative native INTEGRATION"
            )
        reward = self.reaching.step(
            nearest_distance(snapshot.bottle_centers, snapshot.grasp_positions)
        )
        self.trace.write(
            json.dumps(
                {
                    "control": self.controls,
                    "episode_index": self.episode_index,
                    "episode_control": self.episode_steps,
                    "distance_m": self.reaching.last_distance,
                    "reward": reward,
                    "hold_controls": self.reaching.hold,
                    "native_five_bottle_eval": judge,
                },
                allow_nan=False,
            )
            + "\n"
        )
        self.trace.flush()
        terminal = self.observation()
        result = {
            "obs": terminal,
            "terminal_obs": terminal,
            "reward": reward,
            "terminated": False,
            "truncated": bool(truncated),
            "normalized_clipped": int(np.count_nonzero((action < -1) | (action > 1))),
            "physical_action": physical.tolist(),
            "native_ctrl": native.plain(self.env.data.ctrl),
            "sim_time": after["time"],
            "terminal_native_policy_state": native.plain(self.env.get_state()),
            "episodes": [],
            "autoreset": None,
        }
        if truncated:
            episode = {
                "rank": self.rank,
                "episode_index": self.episode_index,
                "requested_seed": self.requested_seed,
                "seed": self.actual_seed,
                "controls": self.episode_steps,
                "completed": True,
                "terminated": False,
                "truncated": True,
                "reach_diagnostic": self.reaching.evidence(),
                "reach_spec": self.config["reach"],
                "final_native_task_eval": judge,
                "native_full_success": judge["success"],
                "peak_native_bottles": self.peak_native_bottles,
                "physics": {"initial": self.initial_physics, "final": after},
                "reset_attempts": self.reset_attempts,
            }
            write_json(self.output / f"episode-{self.episode_index:04d}.json", episode)
            self.completed_episodes += 1
            result["episodes"] = [episode]
            reset = self.reset()
            result["obs"], result["autoreset"] = reset["obs"], reset
        return result

    def close(self) -> None:
        try:
            if not self.closed:
                write_json(
                    self.output / "reach-summary.json",
                    {
                        "rank": self.rank,
                        "controls": self.controls,
                        "completed_episodes": self.completed_episodes,
                        "partial_episode": self.reaching.evidence(),
                        "reach_spec": self.config["reach"],
                        "physics": self.check_physics(),
                    },
                )
        finally:
            if self.trace is not None:
                self.trace.close()
                self.trace = None
            if self.geometry is not None:
                self.geometry.close()
                self.geometry = None
            super().close()


def worker(connection: Any, config: dict[str, Any], rank: int, output: str) -> None:
    world = None
    try:
        world = ReachWorld(config, rank, Path(output))
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


class ReachVectorPool(native.VectorPool):
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

    def describe(self) -> dict:
        return {
            **super().describe(),
            "max_episode_steps": self.config["max_episode_steps"],
            "reach_diagnostic": self.config["reach"],
            "diagnostic_stopping": self.config["stopping"],
            "native_terminate_on_success": False,
            "physics_failure_policy": "fail_on_every_warning_nonfinite_or_internal_time_reset",
            "qualification_scope": "reach_diagnostic_native_and_learning_pending",
        }


def serve(config: dict[str, Any], socket_path: Path, output: Path) -> None:
    validate_config(config)
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
        "phase": "fasttd3_abc_reach_diagnostic_native_host",
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
        pool = ReachVectorPool(config, output)
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
                except TimeoutError:
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
