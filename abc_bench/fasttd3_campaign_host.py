"""Separate reward-v2 native world pool; historical host bytes stay unchanged."""

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
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from abc_bench import fasttd3_vector_host as native
from abc_bench.bottle_success_reward import (
    PBRS_PROFILE,
    BottlePotentialGeometry,
    potential,
    reward_spec,
    transition_reward,
)
from abc_bench.fasttd3_campaign_contract import PHASES, PROFILE, campaign_spec
from abc_bench.fasttd3_host import receive_frame, write_json
from abc_bench.subset import require_subset_lease
from abc_bench.upstream_eval import artifact

MAX_FRAME_BYTES = native.MAX_FRAME_BYTES
require_cpu = native.require_cpu
dispatch = native.dispatch


def validate_config(config: Any) -> dict:
    fields = {
        "schema_version",
        "profile",
        "phase",
        "num_envs",
        "seed_base",
        "seed_stride",
        "max_episode_steps",
        "vector_steps",
        "reward",
        "campaign",
        "timeouts",
    }
    if type(config) is not dict or set(config) != fields:
        raise ValueError("Use the complete separate reward-v2 campaign config")
    if config["profile"] != PROFILE or config["phase"] not in PHASES:
        raise ValueError("Reward-v2 has separate runtime_gate/training profiles")
    for key, expected in {
        "schema_version": 1,
        "num_envs": 128,
        "seed_base": 200000,
        "seed_stride": 128,
        "max_episode_steps": 3540,
        "vector_steps": PHASES[config["phase"]],
    }.items():
        native.integer(config[key], key, expected, expected)
    if config["reward"] != reward_spec(config["reward"]["profile"]):
        raise ValueError("Reward specification differs from the versioned equation")
    if config["campaign"] != campaign_spec(config["phase"]):
        raise ValueError("Campaign minimum, patience or improvement thresholds differ")
    limits = config["timeouts"]
    if type(limits) is not dict or set(limits) != {
        "ready_s",
        "step_s",
        "shutdown_s",
        "wall_s",
        "rpc_s",
    }:
        raise ValueError("Require finite native world and RPC watchdogs")
    for key, maximum in {
        "ready_s": 600,
        "step_s": 180,
        "shutdown_s": 30,
        "wall_s": 12600,
        "rpc_s": 600,
    }.items():
        native.integer(limits[key], key, 1, maximum)
    if (
        limits["step_s"] > limits["rpc_s"]
        or limits["ready_s"] + limits["shutdown_s"] >= limits["wall_s"]
    ):
        raise ValueError("Native watchdogs do not fit their containing deadlines")
    return config


def load_config(path: Path) -> dict:
    if path.stat().st_size > 1024**2:
        raise ValueError("Campaign host config exceeds1MiB")
    return validate_config(json.loads(path.read_text()))


def physics_snapshot(data: Any) -> dict:
    """Read public MjData warnings without calls that clear or repair physics."""
    import mujoco
    import numpy as np

    names = (
        "INERTIA",
        "CONTACTFULL",
        "CNSTRFULL",
        "VGEOMFULL",
        "BADQPOS",
        "BADQVEL",
        "BADQACC",
        "BADCTRL",
    )
    warnings = {}
    for name in names:
        index = int(getattr(mujoco.mjtWarning, "mjWARN_" + name))
        warning = data.warning[index]
        warnings[name] = {
            "number": int(warning.number),
            "lastinfo": int(warning.lastinfo),
        }
    time_finite = bool(np.isfinite(data.time))
    return {
        "time": float(data.time) if time_finite else None,
        "time_finite": time_finite,
        "warnings": warnings,
        "finite": {
            name: bool(np.isfinite(getattr(data, name)).all())
            for name in ("qpos", "qvel", "qacc", "ctrl")
        },
    }


class RewardWorld(native.World):
    def __init__(self, config: dict, rank: int, output: Path):
        self.geometry = None
        self.first_geometry_proof = None
        self.reward_profile = config["reward"]["profile"]
        super().__init__(config, rank, output)

    def integration(self):
        import mujoco
        import numpy as np

        specification = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(
            mujoco.mj_stateSize(self.env.model, specification), dtype=np.float64
        )
        mujoco.mj_getState(self.env.model, self.env.data, state, specification)
        return state

    def check_physics(
        self, before: dict | None = None, *, pre_state=None, action=None
    ) -> dict:
        current = physics_snapshot(self.env.data)
        bad = any(
            current["warnings"][key]["number"]
            for key in ("BADQPOS", "BADQVEL", "BADQACC", "BADCTRL")
        )
        bad = bad or not current["time_finite"]
        if before is not None and not bad:
            bad = not math.isclose(
                current["time"], before["time"] + 0.034, rel_tol=0, abs_tol=1e-9
            )
        if bad or not all(current["finite"].values()):
            import mujoco
            import numpy as np

            stem = f"physics-failure-{self.controls:08d}"
            model_path = self.output / (stem + ".mjb")
            mujoco.mj_saveModel(self.env.model, str(model_path))
            states = {}
            for name, state in (("pre", pre_state), ("post", self.integration())):
                if state is not None:
                    state_path = self.output / (stem + f"-{name}-integration.npy")
                    with state_path.open("xb") as stream:
                        np.save(stream, state, allow_pickle=False)
                    states[name] = artifact(state_path)
            path = self.output / f"physics-failure-{self.controls:08d}.json"
            write_json(
                path,
                {
                    "status": "failed",
                    "rank": self.rank,
                    "seed": self.actual_seed,
                    "episode_index": self.episode_index,
                    "completed_controls": self.controls,
                    "attempted_control": self.controls + 1
                    if before is not None
                    else None,
                    "before": before,
                    "after": current,
                    "physical_action14": action,
                    "compiled_model": artifact(model_path),
                    "integration": states,
                    "state_specification": {
                        "name": "mjSTATE_INTEGRATION",
                        "value": 16383,
                    },
                },
            )
            raise FloatingPointError(f"Invalid native physics; evidence: {path}")
        return current

    def reset(self) -> dict:
        if self.geometry is not None:
            self.geometry.close()
            self.geometry = None
        result = super().reset()
        self.initial_physics = self.check_physics()
        self.phi = 0.0
        components = {
            "phi": 0.0,
            "reach": [0.0] * 5,
            "lift_clearance": [0.0] * 5,
            "transport": [0.0] * 5,
        }
        if self.reward_profile == PBRS_PROFILE:
            import numpy as np

            from abc_sim.task_eval import make_task_evaluator
            from abc_sim.task_specs import get_task_spec

            # A separate public evaluator supplies immutable metadata without
            # changing the native evaluator's max/ever-success history.
            evaluator = make_task_evaluator(self.env.model, get_task_spec(native.TASK))
            judge = evaluator.evaluate_qpos_batch(
                np.array(self.env.data.qpos, copy=True)[None]
            ).to_info(squeeze=True)
            self.geometry = BottlePotentialGeometry(self.env.model, judge)
            if self.first_geometry_proof is None:
                components = self.geometry_proof(judge, result)
            else:
                components = potential(
                    self.geometry.snapshot(self.env.model, self.env.data, judge)
                )
            self.phi = components["phi"]
        self.reward_totals = {
            key: 0.0
            for key in (
                "native_fraction",
                "base_reward",
                "shaping_reward",
                "learner_reward",
            )
        }
        self.peak_bottles = 0
        self.initial_phi = self.phi
        self.final_progress = self.progress(components)
        self.peak_progress = self.final_progress.copy()
        return result

    def geometry_proof(self, judge: dict, reset: dict) -> dict:
        """Authenticate first FK with an independent owned reference and state hashes."""
        import mujoco
        import numpy as np

        model, data = self.env.model, self.env.data
        specification = mujoco.mjtState.mjSTATE_INTEGRATION
        before = np.empty(mujoco.mj_stateSize(model, specification), dtype=np.float64)
        after = np.empty_like(before)
        mujoco.mj_getState(model, data, before, specification)
        start = time.perf_counter()
        snapshot = self.geometry.snapshot(model, data, judge)
        latency = time.perf_counter() - start
        model_digest = json.loads(Path(reset["scene"]["path"]).read_text())[
            "model_arrays"
        ]
        contract = self.geometry.contract(
            self.reward_profile, native_model_sha256=model_digest["sha256"]
        )
        reference = mujoco.MjData(model)
        reference.qpos[:] = np.array(data.qpos, copy=True)
        mujoco.mj_kinematics(model, reference)
        geometry = contract["geometry"]
        body_ids, site_ids = geometry["body_ids"], geometry["grasp_site_ids"]
        rotations = np.asarray(reference.xmat).reshape(-1, 3, 3)
        centers = reference.xpos[body_ids] + np.einsum(
            "bij,bj->bi",
            rotations[body_ids],
            np.asarray(geometry["bottle_center_offsets"]),
        )
        com_equal = bool(np.array_equal(centers, snapshot.bottle_centers))
        sites_equal = bool(
            np.array_equal(reference.site_xpos[site_ids], snapshot.grasp_positions)
        )
        mujoco.mj_getState(model, data, after, specification)
        equal = before.tobytes() == after.tobytes()
        components = potential(snapshot)
        proof = {
            "schema": "sadhana.abc.reward-v2-first-native-geometry/v1",
            "rank": self.rank,
            "seed": self.actual_seed,
            "controls": self.controls,
            "episode_index": self.episode_index,
            "status": "completed" if equal and com_equal and sites_equal else "failed",
            "integration_spec": int(specification),
            "integration_bytes": before.nbytes,
            "integration_before_sha256": hashlib.sha256(before.tobytes()).hexdigest(),
            "integration_after_sha256": hashlib.sha256(after.tobytes()).hexdigest(),
            "integration_exact_equal": equal,
            "native_qpos_sha256": hashlib.sha256(
                np.asarray(data.qpos).tobytes()
            ).hexdigest(),
            "qpos_derived_com_exact_equal": com_equal,
            "qpos_derived_sites_exact_equal": sites_equal,
            "snapshot": {
                key: getattr(snapshot, key).tolist()
                for key in (
                    "bottle_centers",
                    "grasp_positions",
                    "bin_opening",
                    "table_point",
                    "table_normal",
                    "bottle_reaches",
                    "in_bin",
                )
            },
            "potential": components,
            "geometry_contract": contract,
            "native_model_arrays": model_digest,
            "snapshot_latency_seconds": latency,
            "scratch_allocation_bytes": None,
            "scratch_memory_scope": "not_exposed_by_published_geometry_API",
            "worker_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "native_step_or_authoritative_forward_calls": 0,
            "initial_judge_source": "separate_public_original_evaluator_on_copied_qpos_once_per_reset",
        }
        path = self.output / "geometry-proof-first.json"
        write_json(path, proof)
        if proof["status"] != "completed":
            raise RuntimeError(f"Read-only actual native geometry gate failed: {path}")
        self.first_geometry_proof = artifact(path)
        return components

    @staticmethod
    def progress(components: dict) -> dict:
        return {
            "phi": float(components["phi"]),
            "reach": sum(components["reach"]) / 5,
            "clearance": sum(components["lift_clearance"]) / 5,
            "lift_transport": sum(
                a * b
                for a, b in zip(components["lift_clearance"], components["transport"])
            )
            / 5,
        }

    def step(self, normalized: Any) -> dict:
        import numpy as np

        if self.closed:
            raise RuntimeError("Reward-v2 world is closed")
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
        pre_state = self.integration()
        _, native_reward, terminated, truncated, info = self.env.step(
            physical, render_obs=False
        )
        after = self.check_physics(
            before, pre_state=pre_state, action=physical.tolist()
        )
        judge = native.plain(info["task_eval"])
        components = None
        if self.geometry is not None:
            components = potential(
                self.geometry.snapshot(self.env.model, self.env.data, judge)
            )
        breakdown = transition_reward(
            self.reward_profile,
            judge,
            terminated=terminated,
            truncated=truncated,
            **(
                {"phi_pre": self.phi, "phi_final": components["phi"]}
                if components is not None
                else {}
            ),
        )
        if not math.isclose(
            float(native_reward), breakdown["native_fraction"], rel_tol=0, abs_tol=1e-7
        ):
            raise RuntimeError("Native judge and original reward disagree")
        if components is not None:
            self.phi = components["phi"]
            self.final_progress = self.progress(components)
            self.peak_progress = {
                key: max(value, self.peak_progress[key])
                for key, value in self.final_progress.items()
            }
        self.peak_bottles = max(self.peak_bottles, judge["num_bottles_in_bin"])
        for key in self.reward_totals:
            self.reward_totals[key] += breakdown[key]
        self.controls += 1
        self.episode_steps += 1
        terminal = self.observation()
        result = {
            "obs": terminal,
            "terminal_obs": terminal,
            "reward": breakdown["learner_reward"],
            "terminated": terminated,
            "truncated": truncated,
            "normalized_clipped": int(np.count_nonzero((action < -1) | (action > 1))),
            "physical_action": physical.tolist(),
            "native_ctrl": native.plain(self.env.data.ctrl),
            "sim_time": after["time"],
            "terminal_native_policy_state": native.plain(self.env.get_state()),
            "episodes": [],
            "autoreset": None,
            "reward_breakdown": breakdown,
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
                "final_native_task_eval": judge,
                "reset_attempts": self.reset_attempts,
                "reward_v2": {
                    "spec": self.config["reward"],
                    "sums": self.reward_totals.copy(),
                    "initial_phi": self.initial_phi,
                    "final_phi": self.phi,
                    "peak_bottles": self.peak_bottles,
                    "final_progress": self.final_progress.copy(),
                    "peak_progress": self.peak_progress.copy(),
                    "final_transition": breakdown,
                },
                "physics": {"initial": self.initial_physics, "final": after},
            }
            write_json(self.output / f"episode-{self.episode_index:04d}.json", episode)
            result["episodes"] = [episode]
            reset = self.reset()
            result["obs"], result["autoreset"] = reset["obs"], reset
        return result

    def close(self) -> None:
        if not self.closed:
            write_json(
                self.output / "reward-summary.json",
                {
                    "rank": self.rank,
                    "controls": self.controls,
                    "completed_episodes": self.episode_index,
                    "partial_episode_controls": self.episode_steps,
                    "partial_episode_sums": self.reward_totals,
                    "reward": self.config["reward"],
                    "first_geometry_proof": self.first_geometry_proof,
                    "physics": physics_snapshot(self.env.data),
                },
            )
        if self.geometry is not None:
            self.geometry.close()
            self.geometry = None
        super().close()


def worker(connection: Any, config: dict[str, Any], rank: int, output: str) -> None:
    world = None
    try:
        world = RewardWorld(config, rank, Path(output))
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


class RewardVectorPool(native.VectorPool):
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
            "reward": self.config["reward"],
            "campaign": self.config["campaign"],
            "physics_failure_policy": "fail_on_BADQPOS_BADQVEL_BADQACC_BADCTRL_nonfinite_or_internal_time_reset",
        }


def serve(config: dict[str, Any], socket_path: Path, output: Path) -> None:
    validate_config(config)
    if config["profile"] != PROFILE:
        raise ValueError("Native server admits only separate reward-v2 profiles")
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
        "phase": "fasttd3_abc_reward_v2_native_host",
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
        pool = RewardVectorPool(config, output)
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
