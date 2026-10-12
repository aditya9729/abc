"""Frozen FastTD3 state/action boundary over the public captured VAL reset gate."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import stat
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from abc_bench.fasttd3_host import receive_frame, write_json
from abc_bench.fasttd3_vector_host import MAX_FRAME_BYTES, World, integer
from abc_bench.subset import require_subset_lease
from abc_bench.upstream_eval import artifact

SCHEMA = "sadhana.fasttd3.abc.captured.v1"
ACTION_PROFILE = "normalized14_affine_clip_v1"


def load_config(path: Path) -> dict:
    from abc_bench.expoft_host import load_config as load_captured_config

    config = load_captured_config(path)
    if config["schema_version"] != 2 or config["stage"] != "evaluation":
        raise ValueError(
            "Fast captured evaluation requires the published schema2 VAL profile"
        )
    if config["host"]["max_episode_steps"] != 3540:
        raise ValueError(
            "Frozen VAL must use the complete original3540-control horizon"
        )
    return config


class CapturedFastTask(World):
    """Reuse original privileged observation and the public exact-reset host.

    Only observation/description helpers are inherited. The parent constructor,
    sampler, reset, autoreset and native fixture paths are never called.
    """

    def __init__(self, task: Any, scene: Any):
        self.task, self.capture, self.env = task, scene, task.env
        self.randomization = scene.randomization
        self.schema = self.description()
        self.reset_count = 0
        self.closed = False
        if task.reset_verified is not True:
            raise RuntimeError(
                "Captured VAL is not verified by the public live RGB gate"
            )

    def describe(self) -> dict:
        return {
            **self.schema,
            "num_envs": 1,
            "max_episode_steps": 3540,
            "privileged_state": True,
            "asymmetric_obs": False,
            "autoreset": False,
            "profile": "captured_validation",
            "task": self.capture.episode["task"],
            "prompt": self.capture.episode["prompt"],
            "embodiment": "native_yam",
            "observation_schema": "Same float32[216] qpos/qvel/active/variant/scales as training",
            "action_profile": ACTION_PROFILE,
            "action_difference_from_original_ABC": "Fast normalized14 is clipped and mapped to physical14; source VLA forwards raw physical14",
            "reset_policy": self.task.description["reset_policy"],
            "camera_policy": "Actual live initial RGB gate and original post-control video; no camera inputs to actor",
        }

    def reset(self, seeds: Any) -> dict:
        seed = self.capture.episode["seed"]
        if (
            self.closed
            or type(seeds) is not list
            or seeds != [seed]
            or any(type(x) is not int for x in seeds)
        ):
            raise ValueError("Reset must identify this single captured VAL actual seed")
        if self.reset_count:
            self.task.reset(seed)
        # The public task constructor already restored and verified the first
        # reset. Return that live state without a second restore/render pass.
        if self.task.reset_verified is not True or self.task.episode_steps != 0:
            raise RuntimeError("Captured reset is not a verified initial world")
        self.reset_count += 1
        return {
            "obs": [self.observation()],
            "actual_seeds": [seed],
            "reset_verification": self.task.reset_proofs[-1],
        }

    def step(self, actions: Any) -> dict:
        import numpy as np

        if self.closed or not self.reset_count:
            raise RuntimeError("Captured reset must precede policy controls")
        proposed = np.asarray(actions)
        if (
            proposed.shape != (1, 14)
            or proposed.dtype.kind not in "fiu"
            or not np.isfinite(proposed).all()
        ):
            raise ValueError("Captured Fast actions must be finite numeric[1,14]")
        with np.errstate(over="ignore", invalid="ignore"):
            proposed = proposed.astype(np.float32)[0]
        if not np.isfinite(proposed).all():
            raise ValueError("Captured action exceeds float32 limits")
        schema = self.schema["action_schema"]
        low, high = np.asarray(schema["policy_low"]), np.asarray(schema["policy_high"])
        physical = (low + (np.clip(proposed, -1, 1) + 1) * 0.5 * (high - low)).astype(
            np.float32
        )
        result = self.task.step(physical.tolist())
        if result["action_clipping"] or not np.array_equal(
            np.asarray(result["action_applied14"], dtype=np.float32), physical
        ):
            raise RuntimeError(
                "Captured native command differs from the training action adapter"
            )
        observation = self.observation()
        ended = result["terminated"] or result["truncated"]
        episode = []
        if ended:
            row = dict(self.task.episodes[-1])
            if (
                row.get("completed") is not True
                or row.get("seed") != self.capture.episode["seed"]
            ):
                raise RuntimeError(
                    "Captured episode did not preserve complete native evidence"
                )
            row.update(terminated=result["terminated"], truncated=result["truncated"])
            episode = [row]
        return {
            "obs": [observation],
            "terminal_obs": [observation],
            "rewards": [result["reward"]],
            "terminated": [result["terminated"]],
            "truncated": [result["truncated"]],
            "episodes": episode,
            "autoresets": [None],
            "normalized_clipped": int(
                np.count_nonzero((proposed < -1) | (proposed > 1))
            ),
            "physical_actions": [physical.tolist()],
        }

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.task.close()


def dispatch(task: CapturedFastTask, request: Any) -> tuple[dict, bool]:
    fields = {
        "describe": set(),
        "reset": {"seeds"},
        "step": {"actions"},
        "shutdown": set(),
    }
    if type(request) is not dict or request.get("schema") != SCHEMA:
        raise ValueError("Captured Fast RPC schema differs")
    integer(request.get("id"), "id", 1, 2**63 - 1)
    operation = request.get("operation")
    if (
        operation not in fields
        or set(request) != {"schema", "id", "operation"} | fields[operation]
    ):
        raise ValueError("Captured Fast operation or fields differ")
    if operation == "describe":
        result = task.describe()
    elif operation == "reset":
        result = task.reset(request["seeds"])
    elif operation == "step":
        result = task.step(request["actions"])
    else:
        task.close()
        result = {"status": "closed"}
    return {
        "schema": SCHEMA,
        "id": request["id"],
        "ok": True,
        "result": result,
    }, operation == "shutdown"


def serve(config_path: Path, socket_path: Path, output: Path) -> None:
    config = load_config(config_path)
    require_subset_lease()
    # Camera/native construction stays below the live canonical lease.
    if not output.is_dir() or any(output.iterdir()):
        raise ValueError("Captured host output must be an existing empty directory")
    if len(os.fsencode(socket_path)) > 107 or socket_path.exists():
        raise ValueError("Captured host socket must be new and short")
    if (
        not socket_path.parent.is_dir()
        or stat.S_IMODE(socket_path.parent.stat().st_mode) != 0o700
    ):
        raise ValueError("Captured socket parent must have private0700 mode")
    receipt = {
        "schema_version": 1,
        "phase": "original_fasttd3_captured_validation_host",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "input_config": artifact(config_path),
        "action_profile": ACTION_PROFILE,
        "actual_optimizer_updates": 0,
        "episodes": [],
    }
    task, original_task, env, failure = None, None, None, None
    before = time.monotonic()
    deadline = before + config["timeouts"]["host_wall_s"]
    previous = {}

    def cancel(signum, frame):
        raise InterruptedError(f"Captured Fast host received signal{signum}")

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, cancel)
        from abc_bench.captured_reset import CapturedScene, make_captured_env
        from abc_bench.expoft_host import CapturedExpoTask

        capture = CapturedScene(config["host"]["reset_profile"])
        env = make_captured_env(capture, 3540)
        original_task = CapturedExpoTask(env, config, output, capture)
        task = CapturedFastTask(original_task, capture)
        receipt["description"] = task.describe()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(str(socket_path))
            os.chmod(socket_path, 0o600)
            server.listen(4)
            for _ in range(config["timeouts"]["max_requests"]):
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Captured host exceeded its wall deadline")
                    server.settimeout(min(remaining, 1.0))
                    try:
                        connection = server.accept()[0]
                        break
                    except TimeoutError:
                        continue
                with connection:
                    rpc_deadline = min(
                        deadline, time.monotonic() + config["timeouts"]["rpc_s"]
                    )
                    request = json.loads(
                        receive_frame(connection, rpc_deadline - time.monotonic())
                    )
                    response, done = dispatch(task, request)
                    body = (
                        json.dumps(
                            response, separators=(",", ":"), allow_nan=False
                        ).encode()
                        + b"\n"
                    )
                    remaining = rpc_deadline - time.monotonic()
                    if len(body) > MAX_FRAME_BYTES or remaining <= 0:
                        raise TimeoutError(
                            "Captured response exceeds size or total deadline"
                        )
                    connection.settimeout(remaining)
                    connection.sendall(body)
                if done:
                    break
            else:
                raise RuntimeError(
                    "Captured host exhausted its declared request budget"
                )
        receipt["status"] = "completed"
    except BaseException as error:  # noqa: BLE001 - seal interruption/failure and propagate after cleanup
        failure = error
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        try:
            if task is not None:
                task.close()
            elif original_task is not None:
                original_task.close()
            elif env is not None:
                env.close()
        except BaseException as error:  # noqa: BLE001 - reject failed native cleanup
            failure = error
            receipt.update(
                status="failed", cleanup_error=f"{type(error).__name__}: {error}"
            )
        if task is not None:
            receipt.update(
                episodes=task.task.episodes,
                controls=task.task.controls,
                reset_proofs=task.task.reset_proofs,
            )
        receipt["seconds"] = time.monotonic() - before
        socket_path.unlink(missing_ok=True)
        write_json(output / "receipt.json", receipt)
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    if failure is not None:
        raise failure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    serve(options.config, options.socket, options.output)


if __name__ == "__main__":
    main()
