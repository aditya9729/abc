"""Run original Square-Narrow commands; own processes, never learner logic."""

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

AUTHOR_COMMIT = "bc42da1fe42274e41b59e6d5c84d4fceb7ad5ef8"
MAX_WORKER_SECONDS = 86400
CLEANUP_SECONDS = 60
WORKSPACE = Path("/home/user/aditya")
SHELL_ENTRYPOINTS = {
    "scripts/sim/eval_cell.sh",
    "scripts/sim/train_cell.sh",
    "scripts/sim/train_divl_heads.sh",
}
GPU_MODULES = {"abc_bench.mulligan_video"}
CPU_MODULE = "mulligan.sampling.sim_design"
ENVIRONMENT_KEYS = {
    "HF_HOME",
    "HF_LEROBOT_HOME",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "LD_LIBRARY_PATH",
}


def _owned_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.resolve().is_relative_to(WORKSPACE):
        raise ValueError(
            "Source, output and cache paths must stay under /home/user/aditya"
        )
    return path


def load_config(path: Path) -> dict[str, Any]:
    if (
        not path.is_absolute()
        or not path.is_file()
        or not 0 < path.stat().st_size <= 1024**2
    ):
        raise ValueError("Use an absolute regular Mulligan JSON config within1MiB")
    config = json.loads(path.read_text())
    if config.get("mode") == "r1lite-original-qualification":
        from abc_bench.mulligan_r1lite_qualification import (
            load_config as load_qualification,
        )

        return load_qualification(path)
    if config.get("mode") == "r1lite-native-diagnostic":
        from abc_bench.mulligan_r1lite_diagnostic import load_config as load_diagnostic

        return load_diagnostic(path)
    if config.get("mode") == "r1lite-original-training":
        from abc_bench.mulligan_r1lite_training import load_config as load_training

        return load_training(path)
    if config.get("mode") == "r1lite-frozen-eval":
        from abc_bench.mulligan_r1lite_eval import load_config as load_eval

        return load_eval(path)
    required = {
        "schema_version",
        "python",
        "source_root",
        "source_commit",
        "mode",
        "entrypoint",
        "arguments",
        "timeout_s",
    }
    if set(config) - required - {"environment", "binding_pin"} or not required <= set(
        config
    ):
        raise ValueError("Use the explicit Mulligan process config fields")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("Use Mulligan process schema_version1")
    python = Path(config["python"])
    if not python.is_absolute() or not python.is_file():
        raise ValueError("python must name the normally installed author interpreter")
    source = _owned_path(config["source_root"])
    if not source.is_dir() or config["source_commit"] != AUTHOR_COMMIT:
        raise ValueError("Use the pristine released Mulligan bc42da1 checkout")
    timeout = config["timeout_s"]
    if (
        type(timeout) not in (int, float)
        or not math.isfinite(timeout)
        or not 0 < timeout <= MAX_WORKER_SECONDS
    ):
        raise ValueError("timeout_s must be finite in (0,86400]")
    arguments = config["arguments"]
    if type(arguments) is not list or any(type(arg) is not str for arg in arguments):
        raise ValueError("arguments must be a list of literal strings")
    if any(
        arg.startswith(
            (
                "--out",
                "--checkpoint_dir",
                "--checkpoint-dir",
                "--system.checkpoint_dir",
                "--session",
                "--root",
            )
        )
        for arg in arguments
    ):
        raise ValueError("The supervisor owns output paths")
    for flag in (
        "--config",
        "--seed",
        "--env-name",
        "--robot",
        "--env.name",
        "--env.robot",
    ):
        if flag in arguments and arguments.index(flag) + 1 == len(arguments):
            raise ValueError(f"Missing value for{flag}")
    entrypoint, mode = config["entrypoint"], config["mode"]
    if mode == "cpu-start-design":
        if entrypoint != CPU_MODULE or "--check" not in arguments:
            raise ValueError("CPU mode permits only original sim_design --check")
        if "--config" not in arguments:
            raise ValueError("Declare the original Square-Narrow sampler config")
        sampler = (source / arguments[arguments.index("--config") + 1]).resolve()
        if (
            not sampler.is_relative_to(source / "configs/sim/square_narrow")
            or not sampler.is_file()
        ):
            raise ValueError("CPU start design must use a Square-Narrow config")
    elif mode == "gpu":
        if entrypoint not in SHELL_ENTRYPOINTS | GPU_MODULES:
            raise ValueError(
                "Select an original Mulligan sim entrypoint or declared video recorder"
            )
        if entrypoint in SHELL_ENTRYPOINTS:
            recipe = arguments[0] if arguments else ""
            recipes = json.loads((source / "configs/sim/recipes.json").read_text())[
                "recipes"
            ]
            selected = next((r for r in recipes if r["id"] == recipe), {})
            if selected.get("task") != "square_narrow" or "idql_agent" not in selected:
                raise ValueError(
                    "Only original Square-Narrow IDQL/DIVL recipes are admitted"
                )
            if "--seed" not in arguments or arguments[
                arguments.index("--seed") + 1
            ] not in {"1", "2", "3", "4", "5"}:
                raise ValueError("Select one explicit released training seed1..5")
        for flag, expected in (
            ("--env-name", "NutAssemblySquare"),
            ("--robot", "Panda"),
            ("--env.name", "NutAssemblySquare"),
            ("--env.robot", "Panda"),
        ):
            for index, arg in enumerate(arguments):
                if (
                    arg == flag
                    and arguments[index + 1] != expected
                    or arg.startswith(flag + "=")
                    and arg.split("=", 1)[1] != expected
                ):
                    raise ValueError("The admitted task remains Square-Narrow/Panda")
    else:
        raise ValueError("Select gpu or cpu-start-design mode")
    environment = config.get("environment", {})
    if (
        type(environment) is not dict
        or set(environment) - ENVIRONMENT_KEYS
        or any(type(v) is not str for v in environment.values())
    ):
        raise ValueError(
            "Only declared HF/cache and local OSMesa environment keys are permitted"
        )
    for name in ("HF_HOME", "HF_LEROBOT_HOME"):
        if name in environment:
            _owned_path(environment[name])
    if "LD_LIBRARY_PATH" in environment:
        for item in environment["LD_LIBRARY_PATH"].split(":"):
            if not _owned_path(item).is_dir():
                raise ValueError(
                    "OSMesa library paths must be existing task-local directories"
                )
    return config


def verify_source(config: dict[str, Any]) -> None:
    source = Path(config["source_root"])
    git = ["git", "--no-optional-locks", "-C", str(source)]
    head = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    if head != AUTHOR_COMMIT:
        raise ValueError("Mulligan author source commit differs")
    if subprocess.check_output(
        [*git, "status", "--porcelain", "--untracked-files=normal"], text=True
    ).strip():
        raise ValueError("Mulligan author checkout must remain clean")
    if config["entrypoint"] == "abc_bench.mulligan_video":
        pin = config.get("binding_pin", {})
        try:
            path = _owned_path(pin["path"])
            data = path.read_bytes()
            origin = subprocess.check_output(
                [
                    config["python"],
                    "-I",
                    "-B",
                    "-c",
                    "import importlib.util; print(importlib.util.find_spec('abc_bench.mulligan_video').origin)",
                ],
                text=True,
            ).strip()
        except (KeyError, OSError, subprocess.CalledProcessError) as error:
            raise ValueError(
                "A normal installed video binding pin is required"
            ) from error
        if Path(origin).resolve() != path.resolve():
            raise ValueError("Declared video binding is not the installed child module")
        if (
            len(data) != pin["bytes"]
            or hashlib.sha256(data).hexdigest() != pin["sha256"]
        ):
            raise ValueError("Installed video binding differs from its declared pin")


def verify_runtime(config: dict[str, Any]) -> dict[str, Any]:
    """Inspect public distribution metadata, without importing author modules."""
    code = (
        "import hashlib,json,sys\n"
        "from importlib.metadata import distribution\n"
        "d=distribution('mulligan')\n"
        "rows=[]\n"
        "for f in d.files or []:\n"
        " if str(f).startswith('mulligan/') and str(f).endswith('.py'):\n"
        "  p=d.locate_file(f).resolve(); b=p.read_bytes()\n"
        "  rows.append({'name':str(f),'path':str(p),'sha256':hashlib.sha256(b).hexdigest()})\n"
        "print(json.dumps({'prefix':sys.prefix,'version':d.version,'python_files':rows}))\n"
    )
    result = json.loads(
        subprocess.check_output([config["python"], "-I", "-B", "-c", code], text=True)
    )
    prefix, source = Path(result["prefix"]).resolve(), Path(config["source_root"])
    expected = {
        str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (source / "mulligan").rglob("*.py")
    }
    actual = {row["name"]: row["sha256"] for row in result["python_files"]}
    if (
        result["version"] != "0.1.0"
        or not expected
        or actual != expected
        or any(
            not Path(row["path"]).is_relative_to(prefix)
            for row in result["python_files"]
        )
    ):
        raise ValueError(
            "Use a normal installed Mulligan runtime matching every pinned author Python payload"
        )
    return {
        "prefix": str(prefix),
        "version": result["version"],
        "python_payloads": len(actual),
    }


def command_for(config: dict[str, Any], author_output: Path) -> list[str]:
    entrypoint = config["entrypoint"]
    if entrypoint in SHELL_ENTRYPOINTS:
        return [
            "/bin/bash",
            str(Path(config["source_root"]) / entrypoint),
            *config["arguments"],
        ]
    command = [
        config["python"],
        "-I",
        "-u",
        "-B",
        "-m",
        entrypoint,
        *config["arguments"],
    ]
    if entrypoint == CPU_MODULE:
        command += ["--root", config["source_root"]]
    command += [
        "--out" if entrypoint == CPU_MODULE else "--output-dir",
        str(author_output),
    ]
    return command


def run(config: dict[str, Any], output: Path) -> dict[str, Any]:
    if config["mode"] == "r1lite-original-qualification":
        from abc_bench.mulligan_r1lite_qualification import run as run_qualification

        return run_qualification(config, output)
    if config["mode"] == "r1lite-native-diagnostic":
        from abc_bench.mulligan_r1lite_diagnostic import run as run_diagnostic

        return run_diagnostic(config, output)
    if config["mode"] == "r1lite-original-training":
        from abc_bench.mulligan_r1lite_training import run as run_training

        return run_training(config, output)
    if config["mode"] == "r1lite-frozen-eval":
        from abc_bench.mulligan_r1lite_eval import run as run_eval

        return run_eval(config, output)
    if config["mode"] == "gpu":
        require_subset_lease()
    verify_source(config)
    runtime = verify_runtime(config)
    _owned_path(str(output))
    output.mkdir(parents=True, exist_ok=True)
    author_output = output / "author-output"
    author_output.mkdir(exist_ok=False)
    (output / "tmp").mkdir(exist_ok=False)
    (output / "wandb").mkdir(exist_ok=False)
    command = command_for(config, author_output)
    environment = {
        **os.environ,
        **config.get("environment", {}),
        "PYTHON": config["python"],
        "PATH": str(Path(config["python"]).parent)
        + os.pathsep
        + os.environ.get("PATH", ""),
        "MUJOCO_GL": "osmesa",
        "WANDB_MODE": "offline",
        # Keep imageio's FFmpeg encoders in the coordinator's process group.
        "IMAGEIO_FFMPEG_NO_PREVENT_SIGINT": "1",
        "WANDB_DIR": str(output / "wandb"),
        "TMPDIR": str(output / "tmp"),
        "MULLIGAN_OUTPUT_DIR": str(author_output),
        "XDG_CACHE_HOME": str(WORKSPACE / ".cache"),
        "HF_HOME": config.get("environment", {}).get(
            "HF_HOME", str(WORKSPACE / ".cache/huggingface")
        ),
        "HF_LEROBOT_HOME": config.get("environment", {}).get(
            "HF_LEROBOT_HOME", str(WORKSPACE / ".cache/huggingface/lerobot")
        ),
        "TRITON_CACHE_DIR": str(WORKSPACE / ".cache/triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(WORKSPACE / ".cache/torchinductor"),
        "TORCH_HOME": str(WORKSPACE / ".cache/torch"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
    }
    environment.pop("PYTHONPATH", None)
    environment.pop("WANDB_DISABLED", None)
    if config["mode"] == "cpu-start-design":
        environment.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu")
    else:
        environment["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
    result = {
        "schema_version": 1,
        "status": "running",
        "command": command,
        "source_commit": config["source_commit"],
        "mode": config["mode"],
        "runtime": runtime,
        "task": "SquareNarrow",
        "embodiment": "franka_panda",
        "metrics": {},
        "experiment_accepted": False,
        "claim_scope": "Original-command process execution only; native results and recipe fidelity require separate acceptance",
        "descendant_cleanup_owner": "canonical runner"
        if config["mode"] == "gpu"
        else "original CPU start-design module",
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    receipt = output / "receipt.json"

    def publish() -> None:
        temporary = receipt.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        temporary.replace(receipt)

    def cancelled(signum: int, frame: Any) -> None:
        raise InterruptedError(f"Mulligan command cancelled by signal{signum}")

    previous = signal.signal(signal.SIGTERM, cancelled)
    process = None
    started = time.monotonic()
    try:
        publish()
        with (output / "author.log").open("w") as log:
            # Keep descendants in the canonical runner's process group.
            process = subprocess.Popen(
                command,
                cwd=config["source_root"],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            result["author_pid"] = process.pid
            publish()
            code = process.wait(timeout=config["timeout_s"])
            result["author_exit_code"] = code
            if code != 0:
                raise RuntimeError(f"Original Mulligan command exited{code}")
            result["status"] = "completed"
    except (
        OSError,
        ValueError,
        RuntimeError,
        subprocess.TimeoutExpired,
        KeyboardInterrupt,
    ) as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=20)
            result["direct_child_stopped"] = (
                process is None or process.poll() is not None
            )
            result["elapsed_s"] = time.monotonic() - started
            publish()
        finally:
            signal.signal(signal.SIGTERM, previous)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(load_config(args.config), args.output)


if __name__ == "__main__":
    main()
