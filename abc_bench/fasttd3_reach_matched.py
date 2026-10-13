"""Six bounded deterministic TRAIN reach episodes from exact captured resets.

This separate host does not widen the existing VAL captured-reset interface.
Native imports occur only after canonical lease and CPU-environment admission.
"""

from __future__ import annotations

import argparse
import hashlib
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

from abc_bench import fasttd3_vector_host as native
from abc_bench.bottle_success_reward import BottlePotentialGeometry
from abc_bench.captured_reset import checked_file, checked_json
from abc_bench.fasttd3_host import SCHEMA, dispatch, receive_frame, write_json
from abc_bench.fasttd3_reach_contract import (
    HORIZON,
    REACH,
    ReachEpisode,
    nearest_distance,
)
from abc_bench.fasttd3_reach_host import ReachWorld
from abc_bench.subset import require_subset_lease
from abc_bench.upstream_eval import artifact
from abc_bench.upstream_report import coordinated_receipt

PROFILE = "reach_matched_TRAIN3_v1"
ARRAYS = {
    "initial_integration": ((531,), "float64"),
    "qpos": (None, "float64"),
    "qvel": (None, "float64"),
    "states": (None, "float32"),
    "sim_times": (None, "float64"),
}


def load_config(path: Path) -> tuple[dict, list[dict]]:
    if path.is_symlink() or not 0 < path.stat().st_size <= 1024**2:
        raise ValueError("Matched host config requires a regular file at most1MiB")
    value = json.loads(path.read_text())
    if type(value) is not dict or set(value) != {
        "profile",
        "cohort",
        "cases",
        "reach",
        "timeouts",
    }:
        raise ValueError(
            "Use the complete separate matched TRAIN diagnostic configuration"
        )
    if value["profile"] != PROFILE or value["reach"] != REACH:
        raise ValueError("Matched diagnostic profile/reward/horizon changed")
    if value["timeouts"] != {"wall_s": 1200, "rpc_s": 180, "max_requests": 800}:
        raise ValueError("Use the declared bounded six-episode diagnostic watchdogs")
    cohort = checked_json(value["cohort"])
    if cohort.get("schema") != "sadhana.abc.demonstration-cohort/v1":
        raise ValueError("Matched resets require the published demonstration cohort")
    splits = cohort["splits"]
    ids = [identity for split in splits.values() for identity in split]
    if set(splits) != {"train", "validation", "development"} or len(ids) != len(
        set(ids)
    ):
        raise ValueError("Whole-episode split inventory is invalid")
    cases = value["cases"]
    if type(cases) is not list or len(cases) != 3 or len(splits["train"]) != 3:
        raise ValueError(
            "This small diagnostic requires exactly three declared TRAIN captures"
        )
    episodes, identities, seeds = [], [], []
    for entry in cases:
        if entry not in cohort["episodes"]:
            raise ValueError("Case manifest/worker/parent must be pinned by the cohort")
        episode = checked_json(entry["manifest"])
        identity = episode.get("episode_id")
        worker, parent = (
            checked_json(entry["worker_receipt"]),
            checked_json(entry["parent_receipt"]),
        )
        admission = coordinated_receipt(Path(entry["worker_receipt"]["path"]).parent)
        worker_row = next(
            (
                row
                for row in worker.get("episodes", [])
                if row.get("manifest") == entry["manifest"]
            ),
            None,
        )
        if (
            identity not in splits["train"]
            or episode.get("split") != "train"
            or episode.get("status") != "completed"
            or episode.get("task") != native.TASK
            or episode.get("prompt") != native.PROMPT
            or episode["admission"].get("complete_current_success") is not True
            or episode["final_task_eval"].get("num_bottles_in_bin") != 5
            or episode["physics"].get("dt") != 0.002
            or episode["physics"].get("decimation") != 17
            or episode["physics"].get("control_hz") != 1 / 0.034
            or episode["integration_state_spec"].get("value") != 16383
            or type(episode.get("seed")) is not int
            or admission is None
            or admission[1] != entry["worker_receipt"]
            or admission[2] != parent
            or parent.get("worker_exit_code") != 0
            or worker.get("split") != "train"
            or worker.get("phase") != "original_policy_demonstration_collection"
            or worker_row is None
            or worker_row.get("admitted") is not True
        ):
            raise ValueError(
                "Only completed canonically admitted successful TRAIN sources are allowed"
            )
        for name in ("compiled_model", "randomization", *ARRAYS):
            checked_file(episode["artifacts"][name], 256 * 1024**2)
        identities.append(identity)
        seeds.append(episode["seed"])
        episodes.append(episode)
    if set(identities) != set(splits["train"]) or len(set(seeds)) != 3:
        raise ValueError("Matched diagnostic requires each unique TRAIN capture once")
    return value, episodes


def array_inputs(episode: dict) -> dict:
    import numpy as np

    controls = episode["controls"]
    shapes = {
        "qpos": (controls + 1, 58),
        "qvel": (controls + 1, 52),
        "states": (controls + 1, 14),
        "sim_times": (controls + 1,),
    }
    arrays = {}
    for name, (shape, dtype) in ARRAYS.items():
        pin = episode["artifacts"][name]
        value = np.load(
            checked_file(pin, 16 * 1024**2), allow_pickle=False, mmap_mode="r"
        )
        expected = shape or shapes[name]
        if (
            value.shape != expected
            or value.dtype != np.dtype(dtype)
            or pin.get("shape") != list(expected)
            or pin.get("dtype") != dtype
            or not np.isfinite(value).all()
        ):
            raise ValueError("Captured reset array differs: " + name)
        arrays[name] = value
    return arrays


class MatchedWorld(ReachWorld):
    """One camera-free native world; no random sampler and no autoreset."""

    def __init__(self, config: dict, episodes: list[dict], output: Path):
        native.require_cpu()
        import abc_sim

        self.config, self.output, self.rank = config, output, 0
        self.cases = {episode["seed"]: episode for episode in episodes}
        self.controls, self.resets, self.episode_steps = 0, 0, 0
        self.closed, self.active, self.geometry = False, False, None
        self.episodes = []
        self.env = abc_sim.make_env(
            task=native.TASK,
            prompt=native.PROMPT,
            render_cameras=False,
            enable_task_randomizer=False,
            camera_backend="mujoco",
            camera_height=1,
            camera_width=1,
            max_episode_steps=HORIZON,
            terminate_on_success=False,
        )
        try:
            self.restore(episodes[0])
            self.schema = native.World.description(self)
            self.active = (
                False  # The evaluator must request its own authenticated reset.
            )
            self.trace = (output / "controls.jsonl").open("x")
        except BaseException:
            if self.geometry is not None:
                self.geometry.close()
            self.env.close()
            raise

    def restore(self, episode: dict) -> dict:
        import mujoco
        import numpy as np

        files, arrays = episode["artifacts"], array_inputs(episode)
        model_path = checked_file(files["compiled_model"], 256 * 1024**2)
        with model_path.open("rb") as stream:
            header = struct.unpack("<5i", stream.read(20))
        if mujoco.__version__ != "3.8.1" or header != (54321, 8, 92, 3008001, 472):
            raise ValueError("TRAIN captured MuJoCo version/MJB ABI differs")
        model = mujoco.MjModel.from_binary_path(str(model_path))
        if (
            (model.nq, model.nv, model.nu) != (58, 52, 14)
            or model.nplugin
            or model.nmocap
            or model.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_SLEEP)
            or model.opt.disableflags != 0
            or not np.asarray(model.actuator_ctrllimited).all()
            or float(model.opt.timestep) != 0.002
        ):
            raise ValueError("Captured model physics/control/state contract differs")
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
                raise ValueError("Custom callback state cannot be restored")
        if self.geometry is not None:
            self.geometry.close()
            self.geometry = None
        self.env.reload_from_model(model)
        self.env.set_task(native.TASK)
        if self.env.task_runtime_debug_state() is not None:
            raise ValueError("Python task runtime state cannot be restored")
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        if int(spec) != 16383 or mujoco.mj_stateSize(model, spec) != 531:
            raise ValueError("Captured full INTEGRATION specification differs")
        mujoco.mj_setState(model, self.env.data, arrays["initial_integration"], spec)
        self.env.cur_step = 0
        mujoco.mj_forward(model, self.env.data)
        restored = self.integration()
        encoded = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
        mujoco.mj_saveModel(model, None, encoded)
        judge = self.env.evaluate_task().to_info(squeeze=True)
        checks = {
            "integration": restored.tobytes()
            == arrays["initial_integration"].tobytes(),
            "qpos": self.env.data.qpos.tobytes() == arrays["qpos"][0].tobytes(),
            "qvel": self.env.data.qvel.tobytes() == arrays["qvel"][0].tobytes(),
            "state14": self.env.get_state().tobytes() == arrays["states"][0].tobytes(),
            "time": np.asarray(self.env.data.time).tobytes()
            == arrays["sim_times"][0].tobytes(),
            "compiled_model": hashlib.sha256(encoded.tobytes()).hexdigest()
            == files["compiled_model"]["sha256"],
            "native_judge": judge == episode["initial_task_eval"],
        }
        if not all(checks.values()):
            raise RuntimeError("Exact captured TRAIN reset failed: " + repr(checks))
        self.randomization = checked_json(files["randomization"])
        if (
            self.randomization["seed"] != episode["seed"]
            or self.randomization["metadata"]["bottle_count"] != 5
        ):
            raise ValueError("Captured observation metadata differs")
        schema = native.World.description(self)
        if hasattr(self, "schema") and schema != self.schema:
            raise ValueError("Action/observation schema changed between matched cases")
        self.actual_seed, self.initial_physics = episode["seed"], self.check_physics()
        self.geometry = BottlePotentialGeometry(model, judge)
        snapshot = self.geometry.snapshot(model, self.env.data, judge)
        if restored.tobytes() != self.integration().tobytes():
            raise RuntimeError("Read-only reach geometry changed native INTEGRATION")
        reference = mujoco.MjData(model)
        reference.qpos[:] = self.env.data.qpos
        mujoco.mj_kinematics(model, reference)
        indices = [model.body(name).id for name in judge["bottle_names"]]
        centers = reference.xpos[indices] + np.einsum(
            "bij,bj->bi",
            reference.xmat.reshape(-1, 3, 3)[indices],
            np.asarray(judge["bottle_center_offsets"]),
        )
        sites = [
            reference.site_xpos[model.site(name).id]
            for name in ("left_grasp_site", "right_grasp_site")
        ]
        if not np.array_equal(centers, snapshot.bottle_centers) or not np.array_equal(
            sites, snapshot.grasp_positions
        ):
            raise RuntimeError(
                "Captured reach geometry differs from independent owned FK"
            )
        self.reaching = ReachEpisode(
            nearest_distance(snapshot.bottle_centers, snapshot.grasp_positions)
        )
        self.episode_steps, self.peak_native_bottles, self.native_ever_success = (
            0,
            judge["num_bottles_in_bin"],
            judge["success"],
        )
        observation = np.asarray(self.observation(), dtype=np.float32)
        self.reset_proof = {
            "schema": "sadhana.abc.reach-matched-TRAIN-reset/v1",
            "episode_id": episode["episode_id"],
            "seed": episode["seed"],
            "split": "train",
            "cohort": self.config["cohort"],
            "compiled_model": files["compiled_model"],
            "initial_integration": files["initial_integration"],
            "restored_integration_sha256": hashlib.sha256(
                restored.tobytes()
            ).hexdigest(),
            "observation_float32_sha256": hashlib.sha256(
                observation.tobytes()
            ).hexdigest(),
            "initial_distance_m": self.reaching.initial_distance,
            "eligible_initial_distance": self.reaching.eligible,
            "checks": checks,
            "geometry_preserves_integration": True,
            "geometry_matches_independent_FK": True,
        }
        return {
            "obs": [observation.tolist()],
            "actual_seeds": [episode["seed"]],
            "reset_proof": self.reset_proof,
        }

    def reset(self, seeds: Any) -> dict:
        if self.closed or self.resets >= 6 or self.active:
            raise RuntimeError(
                "Matched host permits six complete sequential evaluator resets"
            )
        if (
            type(seeds) is not list
            or len(seeds) != 1
            or type(seeds[0]) is not int
            or seeds[0] not in self.cases
        ):
            raise ValueError("Reset must name one admitted TRAIN captured case")
        result = self.restore(self.cases[seeds[0]])
        write_json(self.output / f"reset-{self.resets:02d}.json", self.reset_proof)
        self.resets += 1
        self.active = True
        return result

    def step(self, actions: Any) -> dict:
        import numpy as np

        if self.closed or not self.active:
            raise RuntimeError(
                "Matched host requires an active explicitly reset episode"
            )
        action = np.asarray(actions)
        if (
            action.shape != (1, 14)
            or action.dtype.kind not in "fiu"
            or not np.isfinite(action).all()
        ):
            raise ValueError("Matched action must be finite normalized numeric[1,14]")
        with np.errstate(over="ignore", invalid="ignore"):
            action = action[0].astype(np.float32)
        if not np.isfinite(action).all():
            raise ValueError("Matched action exceeds float32 range")
        schema = self.schema["action_schema"]
        low, high = np.asarray(schema["policy_low"]), np.asarray(schema["policy_high"])
        physical = (low + (np.clip(action, -1, 1) + 1) * 0.5 * (high - low)).astype(
            np.float32
        )
        before = self.check_physics()
        _, native_reward, terminated, truncated, info = self.env.step(
            physical, render_obs=False
        )
        after = self.check_physics(before)
        self.controls += 1
        self.episode_steps += 1
        if terminated or bool(truncated) != (self.episode_steps == HORIZON):
            raise RuntimeError(
                "Matched reach episode must end only at fixed128-control timeout"
            )
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
        self.native_ever_success = self.native_ever_success or judge["success"]
        integration = self.integration()
        snapshot = self.geometry.snapshot(self.env.model, self.env.data, judge)
        if integration.tobytes() != self.integration().tobytes():
            raise RuntimeError("Reach geometry changed authoritative INTEGRATION")
        reward = self.reaching.step(
            nearest_distance(snapshot.bottle_centers, snapshot.grasp_positions)
        )
        self.trace.write(
            json.dumps(
                {
                    "reset_index": self.resets - 1,
                    "seed": self.actual_seed,
                    "control": self.episode_steps,
                    "normalized_action": action.tolist(),
                    "physical_action": physical.tolist(),
                    "reward": reward,
                    "distance_m": self.reaching.last_distance,
                    "native_five_bottle_eval": judge,
                },
                allow_nan=False,
            )
            + "\n"
        )
        self.trace.flush()
        episodes = []
        if truncated:
            self.active = False
            episode = {
                "seed": self.actual_seed,
                "completed": True,
                "controls": HORIZON,
                "terminated": False,
                "truncated": True,
                "success": judge["success"],  # Public frozen evaluator: native5/5 only.
                "native_full_success": judge["success"],
                "native_full_success_any_control": self.native_ever_success,
                "peak_native_bottles": self.peak_native_bottles,
                "final_native_task_eval": judge,
                "reach_spec": REACH,
                "reach_diagnostic": self.reaching.evidence(),
                "reset_proof": self.reset_proof,
                "physics": {"initial": self.initial_physics, "final": after},
            }
            write_json(self.output / f"episode-{self.resets - 1:02d}.json", episode)
            self.episodes.append(episode)
            episodes = [episode]
        observation = [self.observation()]
        return {
            "obs": observation,
            "terminal_obs": observation,
            "rewards": [reward],
            "terminated": [False],
            "truncated": [bool(truncated)],
            "episodes": episodes,
            "autoresets": [None],
        }

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.trace.close()
            if self.geometry is not None:
                self.geometry.close()
            self.env.close()


class MatchedTask:
    def __init__(self, config: dict, episodes: list[dict], output: Path):
        self.world = MatchedWorld(config, episodes, output)
        self.description = {
            **self.world.schema,
            "profile": PROFILE,
            "num_envs": 1,
            "max_episode_steps": HORIZON,
            "privileged_state": True,
            "camera_backend": None,
            "native_terminate_on_success": False,
            "autoreset": False,
            "reach_spec": REACH,
            "cases": [
                {"seed": e["seed"], "episode_id": e["episode_id"]} for e in episodes
            ],
        }

    def reset(self, seeds: Any) -> dict:
        return self.world.reset(seeds)

    def step(self, actions: Any) -> dict:
        return self.world.step(actions)

    def close(self) -> None:
        self.world.close()


def serve(config_path: Path, socket_path: Path, output: Path) -> None:
    config, episodes = load_config(config_path)
    require_subset_lease()
    # Verify inherited GPU0 ownership first; the native physics process then
    # hides CUDA. No simulator/Torch import precedes this transition.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    native.require_cpu()
    if any(
        os.environ.get(name) != "1"
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    ):
        raise ValueError(
            "Matched native host requires explicit single-thread environment"
        )
    if not socket_path.is_absolute() or len(os.fsencode(socket_path)) > 107:
        raise ValueError("Use an absolute Unix socket path at most107bytes")
    info = socket_path.parent.stat()
    if (
        info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o700
        or socket_path.exists()
    ):
        raise ValueError("Socket directory must be task-owned0700 with no prior socket")
    output.mkdir(parents=True, exist_ok=False)
    task, bound = None, False
    previous = {}
    receipt = {
        "schema": "sadhana.abc.reach-matched-TRAIN-host/v1",
        "status": "running",
        "config": artifact(config_path),
    }
    started = time.monotonic()

    def cancel(signum: int, _frame: Any) -> None:
        raise InterruptedError("Matched native host received signal " + str(signum))

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, cancel)
        task = MatchedTask(config, episodes, output)
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
                    "description": artifact(output / "description.json"),
                },
            )
            requests = 0
            while requests < config["timeouts"]["max_requests"]:
                if time.monotonic() - started >= config["timeouts"]["wall_s"]:
                    raise TimeoutError("Matched TRAIN host exceeded wall deadline")
                try:
                    connection, _ = listener.accept()
                except TimeoutError:
                    continue
                requests += 1
                with connection:
                    request = json.loads(
                        receive_frame(connection, config["timeouts"]["rpc_s"])
                    )
                    response, stop = dispatch(task, request)
                    connection.sendall(
                        json.dumps(response, allow_nan=False).encode() + b"\n"
                    )
                if stop:
                    if (
                        len(task.world.episodes) != 6
                        or task.world.controls != 6 * HORIZON
                    ):
                        raise RuntimeError(
                            "Matched comparison ended without six complete episodes"
                        )
                    receipt["status"] = "completed"
                    break
            else:
                raise RuntimeError("Matched TRAIN host request bound exhausted")
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        cleanup_error = None
        if task is not None:
            receipt.update(
                controls=task.world.controls,
                resets=task.world.resets,
                episodes=task.world.episodes,
            )
            try:
                task.close()
            except (OSError, RuntimeError, ValueError, TypeError) as error:
                cleanup_error = error
                receipt.update(
                    status="failed", cleanup_error=f"{type(error).__name__}: {error}"
                )
        if bound:
            socket_path.unlink(missing_ok=True)
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        receipt["seconds"] = time.monotonic() - started
        write_json(output / "receipt.json", receipt)
        if cleanup_error is not None:
            raise RuntimeError("Matched host cleanup failed") from cleanup_error


def prepare_binding(
    parent_path: Path, worker_path: Path, host_config: Path, output: Path | None
) -> dict:
    """Read completed actual reach evidence before binding either policy export."""
    from abc_bench.fasttd3_reach_stage import verify_result

    config, _ = load_config(host_config)
    parent_pin, worker_pin = artifact(parent_path), artifact(worker_path)
    parent, worker = checked_json(parent_pin), checked_json(worker_pin)
    admission = coordinated_receipt(worker_path.parent)
    if (
        admission is None
        or admission[1] != worker_pin
        or admission[2] != parent
        or parent.get("status") != "completed"
        or parent.get("worker_exit_code") != 0
        or parent.get("worker_receipt_sha256") != worker_pin["sha256"]
        or worker.get("status") != "completed"
        or worker.get("phase") != "fasttd3_reach_diagnostic_training"
        or worker.get("child_exit_codes") != {"host": 0, "learner": 0}
    ):
        raise ValueError(
            "Matched checkpoint binding requires an actual completed canonical reach TRAIN producer"
        )
    producer_config = checked_json(worker["input_config"])
    producer_host = checked_json(worker["host_config"])
    if (
        producer_host.get("phase") != "training"
        or producer_host.get("reach") != config["reach"]
    ):
        raise ValueError("Producer horizon/reward/TRAIN phase differs")
    actual = verify_result(producer_config, producer_host, worker_path.parent)
    if (
        any(worker.get(key) != value for key, value in actual.items())
        or parent.get("metrics") != actual["metrics"]
    ):
        raise ValueError("Producer receipt differs from actual native/learner evidence")
    learner = checked_json(worker["learner_receipt"])
    initial = learner["initial_export"]
    binding = {
        "schema": "sadhana.fasttd3.reach-matched-binding/v1",
        "training_parent": parent_pin,
        "training_worker": worker_pin,
        "learner_receipt": worker["learner_receipt"],
        "diagnostic_receipt": worker["diagnostic_receipt"],
        "initial_metadata": initial["metadata"],
        "initial_checkpoint": initial["checkpoint"],
        "final_checkpoint": worker["checkpoint"],
        "host_config": artifact(host_config),
    }
    if output is None:
        return binding
    if output.exists():
        raise FileExistsError("Refuse an existing matched binding output")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, binding)
    return binding


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--socket", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-binding", action="store_true")
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.prepare_binding:
        if args.parent is None or args.worker is None or args.socket is not None:
            parser.error(
                "Binding preparation requires --parent/--worker and no --socket"
            )
        prepare_binding(args.parent, args.worker, args.config, args.output)
    else:
        if args.socket is None or args.parent is not None or args.worker is not None:
            parser.error(
                "Native server requires --socket and no producer-path arguments"
            )
        serve(args.config, args.socket, args.output)


if __name__ == "__main__":
    main()
