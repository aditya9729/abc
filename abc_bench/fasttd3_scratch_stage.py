"""Bounded 4000-call original scratch training after native ABC128 admission."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from abc_bench import fasttd3_vector_stage as qualification
from abc_bench.fasttd3_host import write_json
from abc_bench.subset import require_subset_lease
from abc_bench.upstream_eval import artifact

ROOT = qualification.ROOT
HOST_PYTHON = qualification.HOST_PYTHON
LEARNER_PYTHON = qualification.LEARNER_PYTHON
PREFLIGHT = qualification.PREFLIGHT
memory_admission = qualification.memory_admission
child_environment = qualification.child_environment
stop_process = qualification.stop_process
rpc = qualification.rpc
RECIPE = {
    **qualification.RECIPE,
    "total_timesteps": 4000,
    "exp_name": "FastTD3ScratchTraining128",
}

EXPORT_CHECK = """
import hashlib, json, sys, torch
from pathlib import Path
payload = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
expected = json.loads(sys.argv[3])
if payload['global_step'] != expected['total_timesteps'] or any(type(payload['args'].get(k)) is not type(v) or payload['args'].get(k) != v for k,v in expected.items()):
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
Path(sys.argv[2]).write_text(json.dumps({'global_step':expected['total_timesteps'],'state_fingerprints':fingerprints,'scope':'Original policy export; no exact resume state'},indent=2)+'\\n')
"""


def pin(value: Any) -> Path:
    if type(value) is not dict or set(value) != {"path", "bytes", "sha256"}:
        raise ValueError("Pin path, bytes and sha256")
    path = Path(value["path"])
    if (
        not path.is_absolute()
        or not path.is_file()
        or path.is_symlink()
        or artifact(path) != value
    ):
        raise ValueError("Pinned artifact differs")
    return path


def check_description(description: dict[str, Any]) -> None:
    expected = {**qualification.TASK, "profile": "scratch_training128"}
    if any(
        type(description.get(key)) is not type(value) or description.get(key) != value
        for key, value in expected.items()
    ):
        raise ValueError("Actual scratch128 task description differs")
    # Phase is checked above; all common native physics, distribution and seed
    # bank fields use the already qualified public boundary validator.
    qualification.check_description({**description, "profile": "training128"})


def wait_ready(process: Any, path: Path, seconds: int) -> dict[str, Any]:
    deadline = time.monotonic() + seconds
    while True:
        if process.poll() is not None:
            raise RuntimeError(f"ABC host exited {process.returncode} before readiness")
        if time.monotonic() >= deadline:
            raise TimeoutError("Scratch128 readiness deadline expired")
        if path.exists():
            try:
                description = rpc(
                    path, "describe", timeout=max(0.001, deadline - time.monotonic())
                )
                check_description(description)
                return description
            except (FileNotFoundError, ConnectionRefusedError):
                pass
        time.sleep(0.1)


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
            "qualification_config",
            "qualification_parent",
            "qualification_worker",
        }
        or type(config["schema_version"]) is not int
        or config["schema_version"] != 1
    ):
        raise ValueError("Use the exact scratch4000 stage schema1")
    for key, expected in (
        ("host_python", HOST_PYTHON),
        ("learner_python", LEARNER_PYTHON),
    ):
        if config[key] != str(expected) or not expected.is_file():
            raise ValueError(f"Use the admitted normal installed {key}")
    host = load_host(pin(config["host_config"]))
    if host["profile"] != "scratch_training128" or host["vector_steps"] != 4000:
        raise ValueError(
            "Scratch training requires the separate128-world/4000-call profile"
        )
    limits = config["limits"]
    maxima = {"ready_seconds": 600, "learner_seconds": 2700, "shutdown_seconds": 30}
    if (
        type(limits) is not dict
        or set(limits) != set(maxima)
        or any(
            type(limits[key]) is not int or not 1 <= limits[key] <= maximum
            for key, maximum in maxima.items()
        )
    ):
        raise ValueError("Use finite ready/learner/shutdown bounds")
    if (
        sum(limits.values()) > host["timeouts"]["wall_s"]
        or limits["ready_seconds"] < host["timeouts"]["ready_s"]
    ):
        raise ValueError("Stage limits must fit the host's startup and wall bounds")
    qualified_config, evidence = qualification.load_config(
        pin(config["qualification_config"])
    )
    if config["cpu_admission"] != qualified_config["cpu_admission"]:
        raise ValueError("CPU admission differs from the measured native qualification")
    parent = json.loads(pin(config["qualification_parent"]).read_text())
    worker = json.loads(pin(config["qualification_worker"]).read_text())
    if (
        parent.get("status") != "completed"
        or parent.get("task") != "put_plastic_bottles_in_bin"
        or parent.get("embodiment") != "native_yam"
        or parent.get("worker_exit_code") != 0
        or parent.get("worker_receipt_sha256")
        != config["qualification_worker"]["sha256"]
    ):
        raise ValueError("Canonical native ABC capacity parent did not complete")
    if (
        worker.get("status") != "completed"
        or worker.get("phase") != "original_fasttd3_abc_128_world_qualification"
        or worker.get("input_config") != config["qualification_config"]
        or worker.get("metrics")
        != {
            "vector_steps": 64,
            "transitions": 8192,
            "critic_updates": 106,
            "actor_updates": 53,
        }
        or worker.get("child_exit_codes") != {"host": 0, "learner": 0}
    ):
        raise ValueError(
            "Actual native ABC128 qualification did not prove source/data/learner capacity"
        )
    native = json.loads(pin(worker["native_receipt"]).read_text())
    if (
        native.get("status") != "completed"
        or native.get("vector_steps") != 64
        or native.get("transitions") != 8192
    ):
        raise ValueError("Native128 collector qualification differs")
    validate_cleanup(native.get("worker_cleanup"), 128)
    learner = json.loads(pin(worker["learner_receipt"]).read_text())
    if (
        learner.get("status") != "completed"
        or learner.get("controls") != 64
        or learner.get("actual_updates") != {"critic": 106, "actor": 53}
        or learner.get("checkpoint") != worker.get("checkpoint")
    ):
        raise ValueError("Actual original learner qualification differs")
    pin(worker["checkpoint"])
    pins = config["source_pins"]
    if (
        type(pins) is not dict
        or set(pins) != {"abc_host", "fast_adapter"}
        or type(pins["abc_host"]) is not dict
        or set(pins["abc_host"]) != {"bytes", "sha256"}
    ):
        raise ValueError("Pin the executed native host payload")
    required = {
        "sadhana_fasttd3_abc/" + name
        for name in ("vector.py", "environment.py", "learner.py", "provenance.json")
    }
    if (
        type(pins["fast_adapter"]) is not list
        or len(pins["fast_adapter"]) != 4
        or {row.get("path") for row in pins["fast_adapter"]} != required
        or any(set(row) != {"path", "bytes", "sha256"} for row in pins["fast_adapter"])
    ):
        raise ValueError(
            "Pin the exact public vector/transport/learner/provenance payloads"
        )
    return config, evidence


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
        "phase": "original_fasttd3_abc_scratch_training4000",
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "claim_scope": "Original scratch learner in ABC; 4000 vector calls, no convergence or frozen task score",
        "input_config": artifact(config_path),
        "cpu_admission": config["cpu_admission"],
        "qualification_parent": config["qualification_parent"],
        "qualification_worker": config["qualification_worker"],
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
                    "--phase",
                    "training",
                    "--steps",
                    "4000",
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
            host_result.get("vector_steps") != 4000
            or host_result.get("transitions") != 512000
            or learner_result.get("controls") != 4000
            or learner_result.get("actual_updates") != {"critic": 7978, "actor": 3989}
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
        pin(learner_result["checkpoint"])
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
                "vector_steps": 4000,
                "transitions": 512000,
                "critic_updates": 7978,
                "actor_updates": 3989,
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
