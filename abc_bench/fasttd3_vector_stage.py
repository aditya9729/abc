"""Run original FastTD3 with 128 ABC worlds under the canonical GPU lease."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import socket
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from abc_bench.fasttd3_host import receive_frame, write_json
from abc_bench.subset import require_subset_lease
from abc_bench.upstream_eval import artifact

ROOT = Path("/home/user/aditya/RL")
HOST_PYTHON = ROOT / "abc/.venv/bin/python"
LEARNER_PYTHON = ROOT / "runtimes/fasttd3-author-abc-py310/bin/python"
SCHEMA = "sadhana.fasttd3.abc.vector.v1"
TASK = {
    "profile": "training128",
    "num_envs": 128,
    "num_obs": 216,
    "num_actions": 14,
    "max_episode_steps": 3540,
    "task": "put_plastic_bottles_in_bin",
    "privileged_state": True,
    "physics_dt": 0.002,
    "control_decimation": 17,
    "camera_backend": None,
    "carried_arm_reset": True,
}
RECIPE = {
    "env_name": "ABC-FiveBottlesVector-v1",
    "exp_name": "FastTD3ScratchVector128",
    "seed": 1,
    "cuda": True,
    "num_envs": 128,
    "num_eval_envs": 1,
    "total_timesteps": 64,
    "batch_size": 32768,
    "buffer_size": 51200,
    "agent": "fasttd3",
    "amp": True,
    "amp_dtype": "bf16",
    "compile": True,
    "use_wandb": False,
    "eval_interval": 0,
    "render_interval": 0,
    "save_interval": 0,
}

# The public adapter verifies every original installed module. This CPU-only
# preflight also checks the four protected author functions and runtime versions.
PREFLIGHT = """
import ast, hashlib, importlib.metadata, json, sys
from pathlib import Path
from sadhana_fasttd3_abc.learner import verify_source
provenance = verify_source()
runtime = {name: importlib.metadata.version(name) for name in ('torch','tensordict','numpy')}
if runtime != {'torch':'2.6.0+cu124','tensordict':'0.7.2','numpy':'1.26.4'}:
    raise RuntimeError('Original FastTD3 runtime versions differ')
if len(provenance['files']) != 12:
    raise RuntimeError('Original FastTD3 module inventory differs')
distribution = importlib.metadata.distribution('fast-td3')
tree = ast.parse(Path(distribution.locate_file('fast_td3/train.py')).read_text())
expected = provenance['protected_update_functions']
actual = {node.name:hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
          for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name in expected}
if len(expected) != 4 or actual != expected:
    raise RuntimeError('Original FastTD3 protected functions differ')
adapter = importlib.metadata.distribution('sadhana-fasttd3-author-abc')
for row in json.loads(sys.argv[2]):
    path = Path(adapter.locate_file(row['path']))
    if path.stat().st_size != row['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
        raise RuntimeError('Installed FastTD3 adapter differs: '+row['path'])
Path(sys.argv[1]).write_text(json.dumps({'source':provenance,'protected_functions':actual,'runtime':runtime},indent=2)+'\\n')
"""

EXPORT_CHECK = """
import hashlib, json, sys, torch
from pathlib import Path
payload = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
expected = json.loads(sys.argv[3])
if payload['global_step'] != 64 or any(type(payload['args'].get(k)) is not type(v) or payload['args'].get(k) != v for k,v in expected.items()):
    raise RuntimeError('Original checkpoint step or recipe differs')
fingerprints = {}
for name in ('actor_state_dict','qnet_state_dict','qnet_target_state_dict','obs_normalizer_state','critic_obs_normalizer_state'):
    state = payload.get(name)
    if not isinstance(state,dict) or not state or not all(torch.isfinite(t).all() for t in state.values()):
        raise RuntimeError('Original checkpoint has absent/nonfinite '+name)
    digest = hashlib.sha256()
    for key,tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(f'{key}:{array.dtype}:{array.shape}'.encode())
        digest.update(array.tobytes())
    fingerprints[name] = {'sha256':digest.hexdigest(),'tensor_count':len(state)}
Path(sys.argv[2]).write_text(json.dumps({'global_step':64,'state_fingerprints':fingerprints,'scope':'Original policy export; no exact resume state'},indent=2)+'\\n')
"""


def _integer(value: Any, name: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [1, {maximum}]")
    return value


def _pin(value: Any) -> Path:
    if type(value) is not dict or set(value) != {"path", "bytes", "sha256"}:
        raise ValueError("Use a complete path/bytes/sha256 artifact pin")
    path = Path(value["path"])
    if (
        not path.is_absolute()
        or not path.is_file()
        or path.is_symlink()
        or artifact(path) != value
    ):
        raise ValueError("Pinned artifact differs or is not an absolute regular file")
    return path


def check_description(description: dict[str, Any], *, count: int = 128) -> None:
    expected = {
        **TASK,
        "num_envs": count,
        "profile": "training128" if count == 128 else "cpu_acceptance2",
    }
    if any(
        type(description.get(key)) is not type(value) or description.get(key) != value
        for key, value in expected.items()
    ):
        raise ValueError(
            "Native ABC task dimensions, physics or reset semantics differ"
        )
    if description.get("reset_options") != {
        "randomization": {
            "bottle_count": 5,
            "randomize_variants": True,
            "randomize_scales": True,
        }
    } or description.get("reset_seed_bank") != {
        "base": 200000,
        "rank_stride": 1,
        "episode_stride": 128,
        "max_episode_index": 780,
        "fallback_offset": 100000,
    }:
        raise ValueError(
            "Native ABC distribution or independent training seed bank differs"
        )


def load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    from abc_bench.fasttd3_vector_host import load_config as load_host
    from abc_bench.fasttd3_vector_host import validate_cleanup

    config = json.loads(path.read_text())
    if (
        type(config) is not dict
        or set(config)
        != {
            "schema_version",
            "host_python",
            "learner_python",
            "host_config",
            "cpu_admission",
            "source_pins",
            "limits",
        }
        or type(config["schema_version"]) is not int
        or config["schema_version"] != 1
    ):
        raise ValueError("Use the exact ABC FastTD3 stage schema1")
    for key, expected in (
        ("host_python", HOST_PYTHON),
        ("learner_python", LEARNER_PYTHON),
    ):
        if config[key] != str(expected) or not expected.is_file():
            raise ValueError(f"Use the admitted normal installed {key}")
    host_config = load_host(_pin(config["host_config"]))
    if host_config["profile"] != "training128" or host_config["vector_steps"] != 64:
        raise ValueError("This stage admits only ABC128/64 vector qualification")
    limits = config["limits"]
    maxima = {"ready_seconds": 600, "learner_seconds": 1800, "shutdown_seconds": 30}
    if type(limits) is not dict or set(limits) != set(maxima):
        raise ValueError("Explicit finite ready/learner/shutdown bounds are required")
    for key, maximum in maxima.items():
        _integer(limits[key], key, maximum)
    if sum(limits.values()) > host_config["timeouts"]["wall_s"]:
        raise ValueError("Host wall bound must cover startup, learning and shutdown")
    if limits["ready_seconds"] < host_config["timeouts"]["ready_s"]:
        raise ValueError("Stage startup bound must cover the host startup bound")
    evidence = json.loads(_pin(config["cpu_admission"]).read_text())
    if (
        evidence.get("status") != "completed"
        or evidence.get("schema_version") != 1
        or evidence.get("phase") != "abc_fasttd3_vector_native_cpu_acceptance"
        or evidence.get("vector_steps") != 2
        or evidence.get("transitions") != 4
    ):
        raise ValueError(
            "CPU admission must prove the actual two-world native qualification"
        )
    check_description(evidence["description"], count=2)
    validate_cleanup(evidence.get("worker_cleanup"), 2)
    if (
        evidence.get("original_fast_bootstrap_mask") != [False, True]
        or evidence.get("terminal_timeout_step", {}).get("terminated") != [True, False]
        or evidence.get("terminal_timeout_step", {}).get("truncated") != [False, True]
    ):
        raise ValueError(
            "CPU admission did not prove distinct success/timeout bootstrap boundaries"
        )
    carry = evidence.get("carried_arm_acceptance")
    if type(carry) is not list or len(carry) != 2:
        raise ValueError("CPU admission needs both carried-arm reset checks")
    for rank, row in enumerate(carry):
        differences = row.get("absolute_difference")
        if (
            row.get("rank") != rank
            or row.get("same_seed_pose_preserved") is not True
            or row.get("joint_comparison") != "exact_float32"
            or type(differences) is not list
            or len(differences) != 14
        ):
            raise ValueError("CPU admission did not prove the ordinary carried pose")
        if any(
            type(value) not in (int, float)
            or not math.isfinite(value)
            or value < 0
            or value > (2**-22 if index in (6, 13) else 0)
            for index, value in enumerate(differences)
        ):
            raise ValueError(
                "CPU carried joints changed or grip roundtrip exceeded two float32 epsilons"
            )
    resources = evidence.get("worker_resources")
    if type(resources) is not list or len(resources) != 2:
        raise ValueError(
            "CPU admission needs both measured native worker resource receipts"
        )
    for rank, row in enumerate(resources):
        measurement = json.loads(_pin(row["evidence"]).read_text())
        if (
            measurement != row["measurement"]
            or measurement.get("rank") != rank
            or measurement.get("pid") != evidence["worker_cleanup"][rank]["pid"]
            or any(
                measurement.get(key) != value
                for key, value in {
                    "status": "closed",
                    "cuda_visible_devices": "",
                    "torch_imported": False,
                    "warp_imported": False,
                    "render_cameras": False,
                }.items()
            )
        ):
            raise ValueError(
                "CPU worker resource evidence differs or loaded GPU/camera runtimes"
            )
        _integer(measurement.get("maxrss_kib"), "CPU native worker peak RSS", 2**40)
    pins = config["source_pins"]
    if type(pins) is not dict or set(pins) != {"abc_host", "fast_adapter"}:
        raise ValueError("Pin the executed ABC host and Fast adapter payloads")
    host_pin = pins["abc_host"]
    if type(host_pin) is not dict or set(host_pin) != {"bytes", "sha256"}:
        raise ValueError("Pin the host payload bytes/sha256")
    adapter = pins["fast_adapter"]
    required = {
        "sadhana_fasttd3_abc/" + name
        for name in ("vector.py", "environment.py", "learner.py", "provenance.json")
    }
    if (
        type(adapter) is not list
        or len(adapter) != 4
        or {row.get("path") for row in adapter} != required
        or any(set(row) != {"path", "bytes", "sha256"} for row in adapter)
    ):
        raise ValueError(
            "Pin the complete public vector/transport/learner/provenance boundary"
        )
    return config, evidence


def memory_admission(
    evidence: dict[str, Any], *, meminfo: Path = Path("/proc/meminfo")
) -> dict[str, int | str]:
    """Conservative admission estimate; this is not a measured 128-world peak."""
    fields = {
        line.split(":", 1)[0]: int(line.split()[1])
        for line in meminfo.read_text().splitlines()
    }
    peak = max(row["measurement"]["maxrss_kib"] for row in evidence["worker_resources"])
    required = math.ceil(128 * peak * 1.25) + 8 * 1024**2
    available = fields["MemAvailable"]
    if available < required:
        raise MemoryError(
            f"ABC128 requires estimated {required} KiB available; observed {available} KiB"
        )
    return {
        "available_kib": available,
        "required_kib": required,
        "cpu_worker_measured_peak_kib": peak,
        "scope": "128 times two-world maximum RSS with 25% margin plus 8 GiB; not observed aggregate peak",
    }


def child_environment(*, cpu: bool) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        WANDB_MODE="disabled",
        MUJOCO_GL="disable",
        JAX_PLATFORMS="cpu",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    if cpu:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    return environment


def rpc(path: Path, operation: str, *, timeout: float) -> dict[str, Any]:
    """Use the public one-frame RPC without importing Torch into the supervisor."""
    deadline = time.monotonic() + timeout
    request = (
        json.dumps({"schema": SCHEMA, "id": 1, "operation": operation}).encode() + b"\n"
    )
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(max(0.001, deadline - time.monotonic()))
        connection.connect(str(path))
        connection.sendall(request)
        response = json.loads(
            receive_frame(connection, max(0.001, deadline - time.monotonic()))
        )
    if (
        response.get("schema") != SCHEMA
        or response.get("id") != 1
        or response.get("ok") is not True
        or type(response.get("result")) is not dict
    ):
        raise RuntimeError(f"ABC host rejected {operation}: {response.get('error')}")
    return response["result"]


def wait_ready(process: Any, path: Path, seconds: int) -> dict[str, Any]:
    deadline = time.monotonic() + seconds
    while True:
        if process.poll() is not None:
            raise RuntimeError(
                f"ABC host exited {process.returncode} before readiness; see host.log"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError("ABC128 host readiness deadline expired")
        if path.exists():
            try:
                value = rpc(
                    path, "describe", timeout=max(0.001, deadline - time.monotonic())
                )
                break
            except (FileNotFoundError, ConnectionRefusedError):
                pass  # Socket bind can precede listen; describe is read-only.
        time.sleep(0.1)
    check_description(value)
    return value


def stop_process(process: Any, seconds: int) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
        raise RuntimeError(
            "Child required kill; canonical coordinator must reap its owned group"
        )


def run(config_path: Path, output: Path) -> dict[str, Any]:
    require_subset_lease()  # Before mutation, native import, or any child launch.
    config, evidence = load_config(config_path)
    from abc_bench.fasttd3_vector_host import validate_cleanup

    output.mkdir(parents=True, exist_ok=True)
    if any(
        not path.is_file()
        or path.is_symlink()
        or path.name not in {"job.json", "run.log"}
        for path in output.iterdir()
    ):
        raise FileExistsError(
            "Stage output may contain only fresh canonical bookkeeping"
        )
    host_source = artifact(
        Path(
            __import__("abc_bench.fasttd3_vector_host", fromlist=["__file__"]).__file__
        )
    )
    if {key: host_source[key] for key in ("bytes", "sha256")} != config["source_pins"][
        "abc_host"
    ]:
        raise RuntimeError("Installed ABC vector host differs from the sealed source")
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "phase": "original_fasttd3_abc_128_world_qualification",
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "claim_scope": "Original scratch learner in ABC; 64 vector calls, no convergence or frozen task score",
        "input_config": artifact(config_path),
        "cpu_admission": config["cpu_admission"],
        "host_config": config["host_config"],
        "source_pins": config["source_pins"],
        "process_group": os.getpgrp(),
        "gpu_visibility": os.environ["CUDA_VISIBLE_DEVICES"],
        "metrics": {},
    }
    host = learner = None
    socket_directory = socket_path = None
    before = time.monotonic()
    previous = {}

    def cancel(signum: int, _frame: Any) -> None:
        raise InterruptedError(f"FastTD3 ABC stage received signal {signum}")

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, cancel)
        subprocess.run(
            [
                config["learner_python"],
                "-I",
                "-B",
                "-c",
                PREFLIGHT,
                str(output / "learner_preflight.json"),
                json.dumps(config["source_pins"]["fast_adapter"]),
            ],
            env=child_environment(cpu=True),
            check=True,
            timeout=30,
        )
        receipt["learner_preflight"] = artifact(output / "learner_preflight.json")
        receipt["memory_admission"] = memory_admission(evidence)
        socket_directory = ROOT / "builds" / ("abcrpc-" + uuid.uuid4().hex[:12])
        socket_directory.mkdir(mode=0o700)
        socket_path = socket_directory / "env.sock"
        host_output = output / "native_host"
        host_output.mkdir()
        with (
            (output / "host.log").open("w") as host_log,
            (output / "learner.log").open("w") as learner_log,
        ):
            # Preserve the UUID here: the host authenticates the lease before it hides CUDA.
            host = subprocess.Popen(
                [
                    config["host_python"],
                    "-I",
                    "-u",
                    "-B",
                    "-m",
                    "abc_bench.fasttd3_vector_host",
                    "--config",
                    config["host_config"]["path"],
                    "--socket",
                    str(socket_path),
                    "--output",
                    str(host_output),
                ],
                env=child_environment(cpu=False),
                stdout=host_log,
                stderr=subprocess.STDOUT,
            )
            if os.getpgid(host.pid) != os.getpgrp():
                raise RuntimeError("ABC host must inherit the canonical process group")
            receipt["native_environment"] = wait_ready(
                host, socket_path, config["limits"]["ready_seconds"]
            )
            environment = child_environment(cpu=False)
            environment.update(
                TORCHINDUCTOR_CACHE_DIR=str(output / "torchinductor"),
                TRITON_CACHE_DIR=str(output / "triton"),
            )
            learner = subprocess.Popen(
                [
                    config["learner_python"],
                    "-I",
                    "-u",
                    "-B",
                    "-m",
                    "sadhana_fasttd3_abc.vector",
                    "--socket",
                    str(socket_path),
                    "--output",
                    str(output / "learner"),
                    "--steps",
                    "64",
                    "--device",
                    "cuda",
                    "--rpc-timeout",
                    "180",
                ],
                env=environment,
                stdout=learner_log,
                stderr=subprocess.STDOUT,
            )
            if os.getpgid(learner.pid) != os.getpgrp():
                raise RuntimeError(
                    "Original learner must inherit the canonical process group"
                )
            if learner.wait(timeout=config["limits"]["learner_seconds"]) != 0:
                raise RuntimeError("Original learner failed; see learner.log")
            rpc(socket_path, "shutdown", timeout=config["limits"]["shutdown_seconds"])
            if host.wait(timeout=config["limits"]["shutdown_seconds"]) != 0:
                raise RuntimeError("ABC host failed its clean shutdown")
        learner_result = json.loads((output / "learner/receipt.json").read_text())
        host_result = json.loads((host_output / "receipt.json").read_text())
        if (
            learner_result.get("status") != "completed"
            or host_result.get("status") != "completed"
        ):
            raise RuntimeError("Actual host and learner receipts must both complete")
        validate_cleanup(host_result.get("worker_cleanup"), 128)
        if any(
            Path(f"/proc/{row['pid']}").exists()
            for row in host_result["worker_cleanup"]
        ):
            raise RuntimeError("An owned ABC native worker still has a live process")
        if (
            host_result.get("vector_steps") != 64
            or host_result.get("transitions") != 8192
            or learner_result.get("controls") != 64
            or learner_result.get("actual_updates") != {"critic": 106, "actor": 53}
        ):
            raise RuntimeError(
                "Actual native data and original update accounting differ"
            )
        if any(
            type(learner_result["configuration"].get(key)) is not type(value)
            or learner_result["configuration"].get(key) != value
            for key, value in RECIPE.items()
        ):
            raise RuntimeError(
                "Executed original recipe differs from the admitted configuration"
            )
        check_description(host_result["description"])
        _pin(learner_result["checkpoint"])
        for name in ("critic", "actor"):
            if (
                type(learner_result.get("finite_gradient_samples", {}).get(name))
                is not int
                or learner_result["finite_gradient_samples"][name] < 1
            ):
                raise RuntimeError("Original learner did not measure finite gradients")
        subprocess.run(
            [
                config["learner_python"],
                "-I",
                "-B",
                "-c",
                EXPORT_CHECK,
                learner_result["checkpoint"]["path"],
                str(output / "export_check.json"),
                json.dumps(RECIPE),
            ],
            env=child_environment(cpu=True),
            check=True,
            timeout=30,
        )
        receipt.update(
            status="completed",
            learner_receipt=artifact(output / "learner/receipt.json"),
            native_receipt=artifact(host_output / "receipt.json"),
            checkpoint=learner_result["checkpoint"],
            export_check=artifact(output / "export_check.json"),
            export_semantics="Original actor, critic and normalizer export; no optimizer/replay/RNG resume state",
            metrics={
                "vector_steps": 64,
                "transitions": 8192,
                "critic_updates": 106,
                "actor_updates": 53,
            },
        )
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        cleanup_errors = []
        for process in (learner, host):
            try:
                stop_process(process, config["limits"]["shutdown_seconds"])
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    run(options.config, options.output)


if __name__ == "__main__":
    main()
