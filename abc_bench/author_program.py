"""Supervise original EXPO entrypoints without implementing a training loop."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from abc_bench.subset import require_subset_lease

PROGRAMS = {"train_offline_rtc.py", "train_pi_robo.py", "eval_droid_policy.py"}
CATCHER_GPU_PROFILE = "kinetix_catcher_gpu_learner_cpu_env_v1"
CATCHER_CLEANUP_SECONDS = 100
ENVIRONMENT_KEYS = {
    "ABC_REALTIME_CONFIG",
    "ABC_REALTIME_EXPO_CLIENT_CONFIG",
    "REALTIME_EXPO_CATCHER_SETTINGS",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "JAX_COMPILATION_CACHE_DIR",
    "JAX_THREEFRY_PARTITIONABLE",
    "XDG_CACHE_HOME",
    "OPENPI_DATA_HOME",
    "HF_HOME",
}


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    if (
        config.get("schema_version") not in (1, 2)
        or config.get("program") not in PROGRAMS
    ):
        raise ValueError(
            "Select schema1 or the declared schema2 Catcher process profile"
        )
    for name in ("python", "source_root", "source_manifest"):
        p = Path(config[name])
        if not p.is_absolute() or not p.exists():
            raise ValueError(f"{name} must be an existing absolute path")
    timeout = config.get("timeout_s")
    if (
        type(timeout) not in (int, float)
        or not math.isfinite(timeout)
        or not 0 < timeout <= 7100
    ):
        raise ValueError("timeout_s must be finite and within (0,7100]")
    _arguments(config.get("arguments"))
    env = config.get("environment", {})
    if (
        not isinstance(env, dict)
        or set(env) - ENVIRONMENT_KEYS
        or any(not isinstance(v, str) for v in env.values())
    ):
        raise ValueError(
            "Only declared cache, data and allocator environment keys are permitted"
        )
    client = config.get("client")
    if client is not None:
        if config["program"] == "train_offline_rtc.py":
            raise ValueError(
                "Offline RTC training must not start an environment client"
            )
        python = Path(client["python"])
        if not python.is_absolute() or not python.is_file():
            raise ValueError("Client python must be an existing absolute interpreter")
        _arguments(client.get("arguments"))
    if config["schema_version"] == 2:
        if not path.is_absolute():
            raise ValueError("Schema2 requires an absolute worker configuration path")
        validate_catcher_profile(config)
    return config


def _option(arguments: list[str], name: str) -> str:
    values = []
    for i, argument in enumerate(arguments):
        if argument == name:
            if i + 1 == len(arguments) or arguments[i + 1].startswith("--"):
                raise ValueError(f"Missing value for {name}")
            values.append(arguments[i + 1])
        elif argument.startswith(name + "="):
            values.append(argument[len(name) + 1 :])
    if len(values) != 1 or not values[0]:
        raise ValueError(f"Require exactly one {name}")
    return values[0]


def catcher_config_paths(python: str) -> dict[str, str]:
    """Resolve only normal distribution metadata; do not import the task or JAX."""
    code = """
import base64, hashlib, importlib.metadata as metadata, json, sys
from pathlib import Path
d = metadata.distribution('realtime-expo-catcher-task')
files = {str(f): f for f in d.files or ()}
result = {}
for name in ('config', 'model_config'):
    item = files['realtime_expo_catcher_task/' + name + '.py']
    path = Path(d.locate_file(item)).resolve()
    if not path.is_file() or not path.is_relative_to(Path(sys.prefix).resolve()):
        raise ValueError('Catcher configuration must be normally installed in the selected prefix')
    digest = base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest()).decode().rstrip('=')
    if item.hash is None or item.hash.mode != 'sha256' or item.hash.value != digest:
        raise ValueError('Catcher configuration differs from its normal distribution RECORD')
    result[name] = str(path)
print(json.dumps(result))
"""
    try:
        result = subprocess.run(
            [python, "-I", "-B", "-c", code],
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "JAX_PLATFORMS": "cpu",
                "MUJOCO_GL": "disable",
            },
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ValueError(
            "Cannot authenticate the selected prefix's Catcher configuration metadata"
        ) from error
    return json.loads(result.stdout)


def validate_catcher_profile(config: dict[str, Any]) -> None:
    """Bind the new role profile to installed Catcher task/model entrypoints."""
    if (
        config.get("task_profile") != CATCHER_GPU_PROFILE
        or config.get("program") != "train_pi_robo.py"
        or set(config)
        - {
            "schema_version",
            "task_profile",
            "program",
            "python",
            "source_root",
            "source_manifest",
            "timeout_s",
            "arguments",
            "environment",
            "client",
        }
        or not isinstance(config.get("client"), dict)
        or set(config["client"]) != {"python", "arguments"}
        or config["client"]["python"] != config["python"]
        or set(config.get("environment", {})) - ENVIRONMENT_KEYS
        or any(
            argument.startswith(("--config.", "--config_task."))
            for argument in config["arguments"]
        )
    ):
        raise ValueError(
            "Schema2 requires the fixed Catcher GPU learner/CPU client profile"
        )
    paths = catcher_config_paths(config["python"])
    for flag, key in (("--config", "model_config"), ("--config_task", "config")):
        path = Path(_option(config["arguments"], flag))
        if not path.is_absolute():
            raise ValueError(
                "Schema2 requires absolute server task/model configuration paths"
            )
        if str(path.resolve()) != paths[key]:
            raise ValueError(
                "Schema2 must use the selected prefix's installed Catcher configurations"
            )
    # Tyro accepts underscore and hyphen option spellings. Normalize option names
    # only; module values contain underscores and must remain byte-exact.
    client_arguments = []
    for argument in config["client"]["arguments"]:
        if argument.startswith("--"):
            option, separator, value = argument.partition("=")
            argument = option.replace("_", "-") + separator + value
        client_arguments.append(argument)
    if (
        _option(client_arguments, "--config-task-path")
        != "realtime_expo_catcher_task.config"
    ):
        raise ValueError("Schema2 client must use the installed Catcher task module")


def role_environments(
    config: dict[str, Any], env: dict[str, str]
) -> tuple[dict[str, str], dict[str, str]]:
    """Preserve schema1; isolate native Catcher physics from the leased GPU learner."""
    if config["schema_version"] == 1:
        return env, env
    if (
        os.environ.get("JAX_PLATFORMS") != "cuda"
        or env.get("JAX_PLATFORMS") != "cuda"
        or not env.get("CUDA_VISIBLE_DEVICES")
        or env["CUDA_VISIBLE_DEVICES"] != os.environ.get("CUDA_VISIBLE_DEVICES")
    ):
        raise ValueError(
            "Schema2 requires inherited JAX_PLATFORMS=cuda and the leased GPU UUID"
        )
    # The worker's require_subset_lease already authenticates this exact UUID.
    server_env = {**env, "JAX_PLATFORM_NAME": "cuda"}
    client_env = {
        **env,
        "CUDA_VISIBLE_DEVICES": "",
        "JAX_PLATFORMS": "cpu",
        "JAX_PLATFORM_NAME": "cpu",
        "JAX_ENABLE_COMPILATION_CACHE": "false",
    }
    return server_env, client_env


def _arguments(value: Any) -> None:
    if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
        raise ValueError("Program arguments must be a list of strings")
    if any(v.startswith(("--output", "--overwrite")) for v in value):
        raise ValueError("The supervisor owns output paths; overwriting is forbidden")


def verify_sources(
    root: Path, manifest_path: Path, program: str, *, client: bool
) -> None:
    manifest = json.loads(manifest_path.read_text())
    entries = manifest["files"]
    required = {
        program,
        "expo_ft/agents/alg/realtime_expo_ft.py",
        "expo_ft/agents/alg/rtc.py",
    }
    if client:
        required.add("client/run_client.py")
    seen = set()
    for entry in entries:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or str(relative) in seen:
            raise ValueError("Source manifest paths must be unique and relative")
        seen.add(str(relative))
        path = root / relative
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Source path escapes the author tree")
        content = path.read_bytes()
        if (
            len(content) != entry["bytes"]
            or hashlib.sha256(content).hexdigest() != entry["sha256"]
        ):
            raise ValueError(
                f"Author source differs from declared integration: {relative}"
            )
    if not required <= seen:
        raise ValueError(f"Source manifest omits required files: {required - seen}")


def _stop(process: subprocess.Popen | None, *, graceful: bool = False) -> str:
    if process is None or process.poll() is not None:
        return "already_stopped"
    if graceful:
        # The author may close its socket before the client finishes native cleanup.
        try:
            process.wait(timeout=5)
            return "already_stopped"
        except subprocess.TimeoutExpired:
            pass
        try:
            process.send_signal(signal.SIGINT)
        except ProcessLookupError:
            process.wait(timeout=20)
            return "already_stopped"
        try:
            process.wait(timeout=20)
            return "sigint"
        except subprocess.TimeoutExpired:
            pass
    try:
        process.terminate()
    except ProcessLookupError:
        process.wait(timeout=10)
        return "already_stopped"
    try:
        process.wait(timeout=10)
        return "sigterm"
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            process.wait(timeout=10)
            return "already_stopped"
        process.wait(timeout=10)
        return "sigkill"


def run(config: dict[str, Any], output: Path) -> dict[str, Any]:
    """Own process lifetime only; author programs own all model and data operations."""
    require_subset_lease()
    if config["schema_version"] == 2:
        validate_catcher_profile(config)
        role_environments(config, dict(os.environ))
    source = Path(config["source_root"])
    verify_sources(
        source,
        Path(config["source_manifest"]),
        config["program"],
        client=config.get("client") is not None,
    )
    output.mkdir(parents=True, exist_ok=True)
    learner_output = output / "author-output"
    if learner_output.exists():
        raise FileExistsError("A fresh author-output directory is required")
    (output / "environment").mkdir(exist_ok=False)
    env = {
        **os.environ,
        **config.get("environment", {}),
        "WANDB_MODE": "offline",
        "WANDB_DIR": str(output),
        "PYTHONDONTWRITEBYTECODE": "1",
        "ABC_REALTIME_OUTPUT_DIR": str(output / "environment"),
    }
    # Direct script execution preserves the author's local config imports.
    # No PYTHONPATH injection, runpy wrapper, or substituted update function.
    env.pop("PYTHONPATH", None)
    env.pop("WANDB_DISABLED", None)
    server_env, client_env = role_environments(config, env)
    command = [
        config["python"],
        "-u",
        "-B",
        str(source / config["program"]),
        *config["arguments"],
    ]
    if config["program"] != "eval_droid_policy.py":
        command.append(f"--output_dir={learner_output}")
    result = {
        "schema_version": 1,
        "status": "running",
        "command": command,
        "source_manifest": config["source_manifest"],
        "claim_scope": "Original program execution with declared ABC IO integration; no task-success or paper-reproduction claim",
        "metrics": {},
    }
    if config["schema_version"] == 2:
        result.update(
            task_profile=CATCHER_GPU_PROFILE,
            task="Catcher-v3",
            embodiment="kinetix_symbolic",
            role_environments={
                role: {
                    key: value[key]
                    for key in (
                        "CUDA_VISIBLE_DEVICES",
                        "JAX_PLATFORMS",
                        "JAX_PLATFORM_NAME",
                        "JAX_ENABLE_COMPILATION_CACHE",
                    )
                    if key in value
                }
                for role, value in (("author", server_env), ("client", client_env))
            },
            claim_scope="Original Real-Time EXPO learner with declared Catcher task integration; no frozen score or paper-reproduction claim",
        )
    receipt = output / "receipt.json"

    def publish() -> None:
        tmp = receipt.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        tmp.replace(receipt)

    def cancelled(signum: int, frame: Any) -> None:
        raise InterruptedError(f"Owner cancelled original program with signal {signum}")

    old_handler = signal.signal(signal.SIGTERM, cancelled)
    started = time.monotonic()
    server = client = None
    try:
        publish()
        with (
            (output / "author.log").open("w") as server_log,
            (output / "client.log").open("w") as client_log,
        ):
            server = subprocess.Popen(
                command,
                cwd=source,
                env=server_env,
                stdout=server_log,
                stderr=subprocess.STDOUT,
            )
            result["author_pid"] = server.pid
            if config.get("client") is not None:
                c = config["client"]
                client_command = [
                    c["python"],
                    "-u",
                    "-B",
                    "-m",
                    "client.run_client",
                    *c["arguments"],
                ]
                # The original client is an upstream local module, as in its launcher.
                client = subprocess.Popen(
                    client_command,
                    cwd=source,
                    env=client_env,
                    stdout=client_log,
                    stderr=subprocess.STDOUT,
                )
                result.update(client_pid=client.pid, client_command=client_command)
            publish()
            while server.poll() is None:
                if client is not None and client.poll() is not None:
                    raise RuntimeError(
                        f"Original environment client exited {client.returncode}"
                    )
                if time.monotonic() - started > config["timeout_s"]:
                    raise TimeoutError(
                        "Original author program exceeded the declared wall-time limit"
                    )
                time.sleep(0.25)
            result["author_exit_code"] = server.returncode
            if server.returncode != 0:
                raise RuntimeError(
                    f"Original author program exited {server.returncode}; see author.log"
                )
            result["status"] = "completed"
            # A zero exit is process evidence only. The author catches checkpoint
            # save errors; root must validate checkpoint contents separately.
            result["checkpoint_accepted"] = False
    except (OSError, ValueError, RuntimeError, KeyboardInterrupt) as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            cleanup_errors = []
            for name, process in (("client", client), ("author", server)):
                try:
                    result[f"{name}_shutdown"] = _stop(
                        process, graceful=name == "client"
                    )
                except (OSError, subprocess.TimeoutExpired) as error:
                    cleanup_errors.append(f"{name}: {type(error).__name__}: {error}")
                result[f"{name}_exit_code"] = (
                    None if process is None else process.poll()
                )
            result["client_cleanup_verified"] = False
            if cleanup_errors or (
                client is not None
                and result.get("client_shutdown") in {"sigterm", "sigkill"}
            ):
                result.update(
                    status="failed",
                    cleanup_errors=cleanup_errors,
                    error="Owned process cleanup required escalation or failed",
                )
            elif client is not None and result.get("client_exit_code") != 0:
                result["status"] = "failed"
                result.setdefault(
                    "error",
                    f"Original environment client exited {result['client_exit_code']} during cleanup",
                )
            result["elapsed_s"] = time.monotonic() - started
            result["owned_children_stopped"] = all(
                p is None or p.poll() is not None for p in (client, server)
            )
            publish()
        finally:
            signal.signal(signal.SIGTERM, old_handler)
    if result["status"] != "completed":
        raise RuntimeError(result.get("error", "Original program failed"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(load_config(args.config), args.output)


if __name__ == "__main__":
    main()
