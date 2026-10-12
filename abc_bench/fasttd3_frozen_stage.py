"""Evaluate an authenticated original FastTD3 export on one captured VAL scene."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import signal
import socket
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from abc_bench import fasttd3_vector_stage as qualification
from abc_bench.fasttd3_host import receive_frame, write_json
from abc_bench.fasttd3_scratch_stage import pin
from abc_bench.subset import require_subset_lease
from abc_bench.upstream_eval import artifact

ROOT = qualification.ROOT
HOST_PYTHON = qualification.HOST_PYTHON
LEARNER_PYTHON = qualification.LEARNER_PYTHON
SCHEMA = "sadhana.fasttd3.abc.captured.v1"
TRAIN_METRICS = {
    "vector_steps": 4000,
    "transitions": 512000,
    "critic_updates": 7978,
    "actor_updates": 3989,
}
ABC_PAYLOADS = {
    "abc_bench/" + name
    for name in (
        "fasttd3_captured_host.py",
        "fasttd3_vector_host.py",
        "captured_reset.py",
        "expoft_host.py",
    )
}
FAST_PAYLOADS = {
    "sadhana_fasttd3_abc/" + name
    for name in (
        "captured.py",
        "environment.py",
        "learner.py",
        "provenance.json",
    )
}


def training_producer(
    parent_pin: dict[str, Any], worker_pin: dict[str, Any]
) -> dict[str, Any]:
    """Bind the actual canonical parent, original export and completed native run."""
    from abc_bench.fasttd3_vector_host import validate_cleanup

    parent = json.loads(pin(parent_pin).read_text())
    worker = json.loads(pin(worker_pin).read_text())
    if (
        parent.get("status") != "completed"
        or parent.get("task") != "put_plastic_bottles_in_bin"
        or parent.get("embodiment") != "native_yam"
        or parent.get("worker_exit_code") != 0
        or parent.get("worker_receipt_sha256") != worker_pin["sha256"]
        or parent.get("metrics") != TRAIN_METRICS
    ):
        raise ValueError(
            "Frozen evaluation requires the completed canonical scratch4000 parent"
        )
    if (
        worker.get("status") != "completed"
        or worker.get("phase") != "original_fasttd3_abc_scratch_training4000"
        or worker.get("metrics") != TRAIN_METRICS
        or worker.get("child_exit_codes") != {"host": 0, "learner": 0}
    ):
        raise ValueError(
            "Scratch4000 worker did not complete native training and cleanup"
        )
    config = json.loads(pin(worker["input_config"]).read_text())
    if config.get("host_config") != worker.get("host_config"):
        raise ValueError("Training worker configuration binding differs")
    host = json.loads(pin(worker["native_receipt"]).read_text())
    learner = json.loads(pin(worker["learner_receipt"]).read_text())
    if (
        host.get("status") != "completed"
        or host.get("vector_steps") != 4000
        or host.get("transitions") != 512000
        or host.get("config", {}).get("profile") != "scratch_training128"
    ):
        raise ValueError("Actual native scratch4000 collector did not complete")
    validate_cleanup(host.get("worker_cleanup"), 128)
    if (
        learner.get("status") != "completed"
        or learner.get("controls") != 4000
        or learner.get("actual_updates") != {"critic": 7978, "actor": 3989}
        or learner.get("checkpoint") != worker.get("checkpoint")
    ):
        raise ValueError("Original learner checkpoint binding differs")
    export = json.loads(pin(worker["export_check"]).read_text())
    if export.get("global_step") != 4000 or not all(
        export.get("state_fingerprints", {}).get(key, {}).get("sha256")
        for key in (
            "actor_state_dict",
            "qnet_state_dict",
            "qnet_target_state_dict",
            "obs_normalizer_state",
            "critic_obs_normalizer_state",
        )
    ):
        raise ValueError("Producer did not authenticate the original4000-step export")
    pin(worker["checkpoint"])
    return worker


def load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    from abc_bench.fasttd3_captured_host import load_config as load_host

    config = json.loads(path.read_text())
    if (
        type(config) is not dict
        or set(config)
        != {
            "schema_version",
            "host_python",
            "learner_python",
            "host_config",
            "training_parent",
            "training_worker",
            "checkpoint",
            "source_pins",
            "limits",
        }
        or type(config["schema_version"]) is not int
        or config["schema_version"] != 1
    ):
        raise ValueError("Use the complete frozen FastTD3 schema1 configuration")
    for key, expected in (
        ("host_python", HOST_PYTHON),
        ("learner_python", LEARNER_PYTHON),
    ):
        if config[key] != str(expected) or not expected.is_file():
            raise ValueError(f"Use the admitted normal installed {key}")
    worker = training_producer(config["training_parent"], config["training_worker"])
    if config["checkpoint"] != worker["checkpoint"]:
        raise ValueError(
            "Frozen checkpoint differs from the authenticated scratch4000 export"
        )
    pin(config["checkpoint"])
    host = load_host(pin(config["host_config"]))
    if (
        host["host"]["allowed_seeds"] not in ([9100], [9101], [9102])
        or host["host"]["save_video"] is not True
    ):
        raise ValueError(
            "Use one declared captured VAL9100/9101/9102 with actual video"
        )
    limits = config["limits"]
    maxima = {"ready_seconds": 300, "learner_seconds": 1400, "shutdown_seconds": 30}
    if (
        type(limits) is not dict
        or set(limits) != set(maxima)
        or any(
            type(limits[key]) is not int or not 1 <= limits[key] <= maximum
            for key, maximum in maxima.items()
        )
    ):
        raise ValueError(
            "Frozen evaluation requires finite ready/learner/shutdown deadlines"
        )
    if sum(limits.values()) + 30 > min(1800, host["timeouts"]["host_wall_s"]):
        raise ValueError(
            "Frozen stage bounds must fit the1800-second native episode job"
        )
    sources = config["source_pins"]
    if type(sources) is not dict or set(sources) != {"abc", "fast_adapter"}:
        raise ValueError(
            "Pin the executed captured host and original actor consumer boundaries"
        )
    for key, expected in (("abc", ABC_PAYLOADS), ("fast_adapter", FAST_PAYLOADS)):
        rows = sources[key]
        if (
            type(rows) is not list
            or len(rows) != len(expected)
            or {row.get("path") for row in rows} != expected
            or any(set(row) != {"path", "bytes", "sha256"} for row in rows)
        ):
            raise ValueError(f"Incomplete {key} source inventory")
    return config, host, worker


def check_description(description: dict[str, Any], host: dict[str, Any]) -> None:
    seed = host["host"]["initial_seed"]
    expected = {
        "num_envs": 1,
        "num_obs": 216,
        "num_actions": 14,
        "max_episode_steps": 3540,
        "profile": "captured_validation",
        "autoreset": False,
        "privileged_state": True,
        "action_profile": "normalized14_affine_clip_v1",
        "embodiment": "native_yam",
    }
    if any(
        type(description.get(key)) is not type(value) or description.get(key) != value
        for key, value in expected.items()
    ):
        raise ValueError("Actual captured Fast policy/task boundary differs")
    reset = description.get("reset_policy", {})
    if (
        reset.get("reset_verified") is not True
        or reset.get("profile") != "captured_demonstration_mjb_integration_v1"
        or reset.get("heldout_seed") != seed
        or reset.get("allowed_seeds") != [seed]
        or reset.get("episode_id") != host["host"]["reset_profile"]["episode_id"]
    ):
        raise ValueError(
            "Captured VAL identity, physics and live RGB were not admitted"
        )
    proof = json.loads(pin(reset["verification"]).read_text())
    if (
        proof.get("status") != "completed"
        or proof.get("reset_verified") is not True
        or proof.get("schema") != "sadhana.abc.captured-reset-verification/v1"
        or proof.get("seed") != seed
        or proof.get("episode_id") != reset["episode_id"]
        or any(
            proof.get(key) != 0
            for key in ("actual_controls", "physics_steps", "sampler_calls")
        )
    ):
        raise ValueError(
            "Initial reset proof does not describe a verified unstepped capture"
        )
    for when in ("before_rgb", "after_rgb"):
        checks = proof.get("checks", {}).get(when, {})
        if set(checks) != {
            "chronology",
            "model",
            "integration",
            "qpos",
            "qvel",
            "state14",
            "time",
            "judge",
        } or any(row.get("equal") is not True for row in checks.values()):
            raise ValueError(
                "Captured full physics equality did not pass before/after live RGB"
            )
    cameras = proof.get("camera_checks", {})
    if set(cameras) != {"left", "right", "top"} or any(
        row.get("equal") is not True
        or row.get("actual_sha256") != row.get("expected_sha256")
        for row in cameras.values()
    ):
        raise ValueError("All three actual live initial RGB hashes must match")


def rpc(path: Path, operation: str, *, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    body = (
        json.dumps({"schema": SCHEMA, "id": 1, "operation": operation}).encode() + b"\n"
    )
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(max(0.001, deadline - time.monotonic()))
        connection.connect(str(path))
        connection.sendall(body)
        value = json.loads(
            receive_frame(connection, max(0.001, deadline - time.monotonic()))
        )
    if (
        value.get("schema") != SCHEMA
        or value.get("id") != 1
        or value.get("ok") is not True
        or type(value.get("result")) is not dict
    ):
        raise RuntimeError(f"Captured host rejected {operation}: {value.get('error')}")
    return value["result"]


def wait_ready(
    process: Any, path: Path, seconds: int, host: dict[str, Any]
) -> dict[str, Any]:
    deadline = time.monotonic() + seconds
    while True:
        if process.poll() is not None:
            raise RuntimeError(
                f"Captured host exited {process.returncode} before readiness"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError("Captured Fast host readiness deadline expired")
        if path.exists():
            try:
                description = rpc(
                    path, "describe", timeout=max(0.001, deadline - time.monotonic())
                )
                check_description(description, host)
                return description
            except (FileNotFoundError, ConnectionRefusedError):
                pass
        time.sleep(0.1)


def evaluate_result(
    config: dict[str, Any], host_config: dict[str, Any], output: Path
) -> dict[str, Any]:
    learner = json.loads((output / "learner/receipt.json").read_text())
    host = json.loads((output / "native_host/receipt.json").read_text())
    if (
        learner.get("status") != "completed"
        or host.get("status") != "completed"
        or learner.get("checkpoint") != config["checkpoint"]
        or host.get("actual_optimizer_updates") != 0
    ):
        raise RuntimeError(
            "Frozen host/consumer did not complete with the exact producer checkpoint"
        )
    pin(learner["checkpoint"])
    policy = learner.get("policy_state_before")
    rng = learner.get("rng_state_before")
    if (
        learner.get("actor_updates") != 0
        or learner.get("critic_updates") != 0
        or type(policy) is not str
        or len(policy) != 64
        or any(character not in "0123456789abcdef" for character in policy)
        or policy != learner.get("policy_state_after")
        or type(rng) is not dict
        or set(rng) != {"torch_cpu", "torch_policy_device"}
        or any(
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in rng.values()
        )
        or rng != learner.get("rng_state_after")
    ):
        raise RuntimeError(
            "Frozen policy, normalization, RNG or optimizer state changed"
        )
    check_description(host["description"], host_config)
    seed = host_config["host"]["initial_seed"]
    if (
        learner.get("evaluation_seeds") != [seed]
        or len(learner.get("episodes", [])) != 1
        or len(host.get("episodes", [])) != 1
    ):
        raise RuntimeError(
            "Frozen result must contain one completed declared VAL episode"
        )
    episode, native = learner["episodes"][0], host["episodes"][0]
    if (
        episode.get("completed") is not True
        or native.get("completed") is not True
        or episode.get("seed") != seed
        or native.get("seed") != seed
        or type(episode.get("success")) is not bool
        or episode.get("success") != native.get("success")
    ):
        raise RuntimeError(
            "Frozen result lacks complete matching native success evidence"
        )
    controls = episode.get("controls")
    if (
        type(controls) is not int
        or not 1 <= controls <= 3540
        or controls != host.get("controls")
        or episode.get("terminated") is not episode["success"]
        or episode.get("truncated") is not (not episode["success"])
        or (not episode["success"] and controls != 3540)
    ):
        raise RuntimeError(
            "Partial controls or inconsistent native termination cannot become a task score"
        )
    judge = episode.get("final_native_task_eval", {})
    if (
        judge != native.get("final_native_task_eval")
        or judge.get("num_active_bottles") != 5
        or type(judge.get("num_bottles_in_bin")) is not int
        or not 0 <= judge["num_bottles_in_bin"] <= 5
        or judge.get("success") is not episode["success"]
        or (episode["success"] and judge["num_bottles_in_bin"] != 5)
    ):
        raise RuntimeError(
            "Frozen score differs from the original native five-bottle judge"
        )
    if (
        len(host.get("reset_proofs", [])) != 1
        or episode.get("reset_verification") != host["reset_proofs"][0]
        or native.get("reset_verification") != episode["reset_verification"]
    ):
        raise RuntimeError(
            "Frozen episode does not bind the one actual initial captured reset"
        )
    video = native.get("video")
    if video != episode.get("video"):
        raise RuntimeError("Frozen learner and host video bindings differ")
    pin(video)
    return {
        "native_receipt": artifact(output / "native_host/receipt.json"),
        "learner_receipt": artifact(output / "learner/receipt.json"),
        "video": video,
        "episode": episode,
        "metrics": {
            "completed_episodes": 1,
            "successes": int(episode["success"]),
            "success_rate": float(episode["success"]),
            "bottles_in_bin": judge["num_bottles_in_bin"],
            "controls": controls,
            "actor_updates": 0,
            "critic_updates": 0,
        },
    }


def run(config_path: Path, output: Path) -> dict[str, Any]:
    require_subset_lease()
    config, host_config, _producer = load_config(config_path)
    distribution = importlib.metadata.distribution("abc-minimal")
    for row in config["source_pins"]["abc"]:
        actual = artifact(Path(distribution.locate_file(row["path"])))
        if any(actual[key] != row[key] for key in ("bytes", "sha256")):
            raise RuntimeError(f"Installed captured ABC payload differs: {row['path']}")
    output.mkdir(parents=True, exist_ok=True)
    if any(
        not path.is_file()
        or path.is_symlink()
        or path.name not in {"job.json", "run.log"}
        for path in output.iterdir()
    ):
        raise FileExistsError(
            "Frozen output may contain only fresh canonical bookkeeping"
        )
    receipt = {
        "schema_version": 1,
        "phase": "original_fasttd3_captured_frozen_evaluation",
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "claim_scope": "One captured VAL scene, original scratch actor frozen; native YAM, no population/paper/R1Lite claim",
        "input_config": artifact(config_path),
        "training_parent": config["training_parent"],
        "training_worker": config["training_worker"],
        "checkpoint": config["checkpoint"],
        "host_config": config["host_config"],
        "process_group": os.getpgrp(),
        "metrics": {},
    }
    host = learner = None
    socket_directory = socket_path = None
    before = time.monotonic()
    previous = {}

    def cancel(signum: int, _frame: Any) -> None:
        raise InterruptedError(f"Frozen Fast stage received signal {signum}")

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, cancel)
        subprocess.run(
            [
                config["learner_python"],
                "-I",
                "-B",
                "-c",
                qualification.PREFLIGHT,
                str(output / "learner_preflight.json"),
                json.dumps(config["source_pins"]["fast_adapter"]),
            ],
            env=qualification.child_environment(cpu=True),
            check=True,
            timeout=30,
        )
        receipt["learner_preflight"] = artifact(output / "learner_preflight.json")
        socket_directory = ROOT / "builds" / ("abcfval-" + uuid.uuid4().hex[:12])
        socket_directory.mkdir(mode=0o700)
        socket_path = socket_directory / "env.sock"
        (output / "native_host").mkdir()
        with (
            (output / "host.log").open("w") as host_log,
            (output / "learner.log").open("w") as learner_log,
        ):
            host = subprocess.Popen(
                [
                    config["host_python"],
                    "-I",
                    "-u",
                    "-B",
                    "-m",
                    "abc_bench.fasttd3_captured_host",
                    "--config",
                    config["host_config"]["path"],
                    "--socket",
                    str(socket_path),
                    "--output",
                    str(output / "native_host"),
                ],
                env=qualification.child_environment(cpu=False),
                stdout=host_log,
                stderr=subprocess.STDOUT,
            )
            if os.getpgid(host.pid) != os.getpgrp():
                raise RuntimeError(
                    "Captured host must inherit the canonical process group"
                )
            receipt["native_environment"] = wait_ready(
                host, socket_path, config["limits"]["ready_seconds"], host_config
            )
            learner = subprocess.Popen(
                [
                    config["learner_python"],
                    "-I",
                    "-u",
                    "-B",
                    "-m",
                    "sadhana_fasttd3_abc.captured",
                    "--socket",
                    str(socket_path),
                    "--checkpoint",
                    config["checkpoint"]["path"],
                    "--output",
                    str(output / "learner"),
                    "--device",
                    "cuda",
                    "--rpc-timeout",
                    "180",
                ],
                env=qualification.child_environment(cpu=False),
                stdout=learner_log,
                stderr=subprocess.STDOUT,
            )
            if os.getpgid(learner.pid) != os.getpgrp():
                raise RuntimeError(
                    "Frozen actor consumer must inherit the canonical process group"
                )
            if learner.wait(timeout=config["limits"]["learner_seconds"]) != 0:
                raise RuntimeError(
                    "Original frozen actor consumer failed; see learner.log"
                )
            rpc(socket_path, "shutdown", timeout=config["limits"]["shutdown_seconds"])
            if host.wait(timeout=config["limits"]["shutdown_seconds"]) != 0:
                raise RuntimeError("Captured host did not exit cleanly")
        receipt.update(
            status="completed", **evaluate_result(config, host_config, output)
        )
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        cleanup_errors = []
        for process in (learner, host):
            try:
                qualification.stop_process(
                    process, config["limits"]["shutdown_seconds"]
                )
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                cleanup_errors.append(str(error))
        if socket_path is not None:
            socket_path.unlink(missing_ok=True)
        if socket_directory is not None:
            socket_directory.rmdir()
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        receipt.update(
            seconds=time.monotonic() - before,
            child_exit_codes={
                "host": None if host is None else host.poll(),
                "learner": None if learner is None else learner.poll(),
            },
        )
        if cleanup_errors:
            receipt.update(status="failed", cleanup_errors=cleanup_errors)
        write_json(output / "receipt.json", receipt)
        if cleanup_errors:
            raise RuntimeError("; ".join(cleanup_errors))
    return receipt


def prepare_config(
    training_parent: Path,
    training_worker: Path,
    host_config: Path,
    source_pins: dict[str, Any],
) -> dict[str, Any]:
    """Create concrete pinned evaluation input only after the producer completes."""
    parent_pin, worker_pin = artifact(training_parent), artifact(training_worker)
    producer = training_producer(parent_pin, worker_pin)
    return {
        "schema_version": 1,
        "host_python": str(HOST_PYTHON),
        "learner_python": str(LEARNER_PYTHON),
        "host_config": artifact(host_config),
        "training_parent": parent_pin,
        "training_worker": worker_pin,
        "checkpoint": producer["checkpoint"],
        "source_pins": source_pins,
        "limits": {
            "ready_seconds": 300,
            "learner_seconds": 1400,
            "shutdown_seconds": 30,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    run(options.config, options.output)


if __name__ == "__main__":
    main()
