"""Supervise untouched original FastTD3 Hurdle commands, without learner logic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from abc_bench.subset import require_subset_lease

AUTHOR_COMMIT = "229ed59bbf43ea2f7a2d5d90d1076314839944d7"
HUMANOID_COMMIT = "cb1189039151c8aadaaa987b442da54383c87fab"
TASK = "h1hand-hurdle-v0"
MAX_WORKER_SECONDS = 28800
QUALIFICATION_SECONDS = 1800
CLEANUP_SECONDS = 60
WORKSPACE = Path("/home/user/aditya")


def _owned_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.resolve().is_relative_to(WORKSPACE):
        raise ValueError(
            "FastTD3 source, interpreter and outputs must stay under /home/user/aditya"
        )
    return path


def load_config(path: Path) -> dict[str, Any]:
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
        or not 0 < path.stat().st_size <= 1024**2
    ):
        raise ValueError("Use an absolute regular FastTD3 JSON config within1MiB")
    config = json.loads(path.read_text())
    fields = {
        "schema_version",
        "python",
        "source_root",
        "source_commit",
        "humanoid_source_root",
        "humanoid_source_commit",
        "phase",
        "arguments",
        "timeout_s",
    }
    if type(config) is not dict or set(config) != fields:
        raise ValueError("Use the explicit original FastTD3 process config fields")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("Use original FastTD3 process schema_version1")
    for field in ("python", "source_root", "humanoid_source_root"):
        if type(config[field]) is not str:
            raise ValueError(f"{field} must be an absolute path")
        if field == "python":
            # Normal venv launchers can symlink an immutable base interpreter.
            # Check their owned bin directory here and actual sys.prefix below.
            item = Path(config[field])
            if not item.is_absolute() or not item.parent.resolve().is_relative_to(
                WORKSPACE
            ):
                raise ValueError(
                    "Use a normal interpreter prefix under /home/user/aditya"
                )
        else:
            item = _owned_path(config[field])
        if not (item.is_file() if field == "python" else item.is_dir()):
            raise ValueError(f"Missing original FastTD3 {field}")
    if (
        config["source_commit"] != AUTHOR_COMMIT
        or config["humanoid_source_commit"] != HUMANOID_COMMIT
    ):
        raise ValueError("Use the pinned pristine FastTD3 and HumanoidBench revisions")
    phase = config["phase"]
    if phase not in {
        "original_humanoidbench_train",
        "original_humanoidbench_qualification",
    }:
        raise ValueError(
            "Select original full training or separately labelled qualification"
        )
    arguments = config["arguments"]
    if (
        type(arguments) is not list
        or len(arguments) % 2
        or any(type(x) is not str for x in arguments)
    ):
        raise ValueError("Use literal original flag/value pairs")
    values = dict(zip(arguments[::2], arguments[1::2], strict=True))
    allowed = {"--env-name", "--seed", "--exp-name"}
    if phase == "original_humanoidbench_qualification":
        allowed.add("--total-timesteps")
    if (
        len(values) * 2 != len(arguments)
        or set(values) - allowed
        or not {"--env-name", "--seed"} <= values.keys()
    ):
        raise ValueError(
            "Preserve learner defaults; only task, seed and operational experiment name are admitted"
        )
    if values["--env-name"] != TASK or values["--seed"] not in {
        "1",
        "2",
        "3",
        "4",
        "5",
    }:
        raise ValueError("Select original H1Hand Hurdle and one explicit seed1..5")
    if "--exp-name" in values and not re.fullmatch(
        r"[A-Za-z0-9_.-]{1,100}", values["--exp-name"]
    ):
        raise ValueError(
            "Experiment name must not change the original relative output paths"
        )
    if phase == "original_humanoidbench_qualification" and values.get(
        "--total-timesteps"
    ) not in {"1", "12"}:
        raise ValueError(
            "Qualification permits exactly1 or12 original vector calls, not a benchmark"
        )
    timeout = config["timeout_s"]
    bound = (
        MAX_WORKER_SECONDS
        if phase == "original_humanoidbench_train"
        else QUALIFICATION_SECONDS
    )
    if (
        type(timeout) not in (int, float)
        or not math.isfinite(timeout)
        or not 0 < timeout <= bound
    ):
        raise ValueError(f"Worker timeout must be finite in (0,{bound}]")
    return config


def verify_source(config: dict[str, Any]) -> dict[str, Any]:
    manifests = {}
    for field, revision in (
        ("source_root", AUTHOR_COMMIT),
        ("humanoid_source_root", HUMANOID_COMMIT),
    ):
        source = Path(config[field])
        git = ["git", "--no-optional-locks", "-C", str(source)]
        if (
            subprocess.check_output(
                [*git, "rev-parse", "HEAD"], text=True, timeout=20
            ).strip()
            != revision
        ):
            raise ValueError("Original source HEAD differs")
        if subprocess.check_output(
            [*git, "status", "--porcelain", "--untracked-files=normal"],
            text=True,
            timeout=20,
        ).strip():
            raise ValueError("Original source must remain clean")
        paths = (
            subprocess.check_output([*git, "ls-files", "-z"], timeout=20)
            .decode()
            .split("\0")
        )
        rows = []
        for name in filter(None, paths):
            data = (source / name).read_bytes()
            rows.append((name, len(data), hashlib.sha256(data).hexdigest()))
        manifests[field] = {
            "commit": revision,
            "tracked_files": len(rows),
            "manifest_sha256": hashlib.sha256(
                json.dumps(rows, separators=(",", ":")).encode()
            ).hexdigest(),
        }
    return manifests


def verify_runtime(config: dict[str, Any]) -> dict[str, Any]:
    """Use normal distribution metadata; import no Torch, author or simulator."""
    code = """import hashlib,json,sys
from importlib.metadata import distribution
result={'prefix':sys.prefix,'distributions':{}}
for name,namespace in [('fast-td3','fast_td3/'),('humanoid-bench','humanoid_bench/')]:
 d=distribution(name); rows={}
 for f in d.files or []:
  if str(f).startswith(namespace) and '__pycache__' not in str(f) and not str(f).endswith('.pyc'):
   p=d.locate_file(f).resolve(); rows[str(f)]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 result['distributions'][name]={'version':d.version,'files':rows}
result['versions']={name:distribution(name).version for name in ['torch','tensordict','torchrl','mujoco','gymnasium','stable-baselines3','wandb','jax','jaxlib']}
print(json.dumps(result))
"""
    result = json.loads(
        subprocess.check_output(
            [config["python"], "-I", "-B", "-c", code], text=True, timeout=120
        )
    )
    prefix = Path(result["prefix"]).resolve()
    if not prefix.is_relative_to(WORKSPACE):
        raise ValueError("Use an isolated normal runtime under /home/user/aditya")
    checked = {}
    for name, namespace, field, version in (
        ("fast-td3", "fast_td3", "source_root", "0.1.0"),
        ("humanoid-bench", "humanoid_bench", "humanoid_source_root", "0.2"),
    ):
        source = Path(config[field])
        candidates = (source / namespace).rglob("*.py" if name == "fast-td3" else "*")
        expected = {
            str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in candidates
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
        }
        installed = result["distributions"][name]
        actual = {key: row["sha256"] for key, row in installed["files"].items()}
        if (
            installed["version"] != version
            or not expected
            or actual != expected
            or any(
                not Path(row["path"]).is_relative_to(prefix)
                for row in installed["files"].values()
            )
        ):
            raise ValueError(
                f"Normal installed {name} payloads differ from pristine source"
            )
        checked[name] = {
            "version": version,
            "payloads": len(actual),
            "manifest_sha256": hashlib.sha256(
                json.dumps(sorted(actual.items())).encode()
            ).hexdigest(),
        }
    if (
        result["versions"]["torch"] != "2.6.0+cu124"
        or result["versions"]["tensordict"] != "0.7.2"
        or result["versions"]["torchrl"] != "0.7.2"
    ):
        raise ValueError(
            "Preserve original FastTD3 Torch2.6cu124/TensorDict0.7.2/TorchRL0.7.2 runtime"
        )
    return {
        "prefix": str(prefix),
        "distributions": checked,
        "versions": result["versions"],
        "dependency_limit": "HumanoidBench declares Torch2.3.1; Fast requires2.6.0. Native/EGL/compile qualification remains independent.",
    }


def run(config: dict[str, Any], output: Path) -> dict[str, Any]:
    require_subset_lease()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.startswith("GPU-") or "," in visible:
        raise RuntimeError(
            "Preserve the coordinator's single physical GPU0 UUID visibility"
        )
    sources, runtime = verify_source(config), verify_runtime(config)
    _owned_path(str(output))
    output.mkdir(parents=True, exist_ok=True)
    author_output = output / "author-output"
    author_output.mkdir(exist_ok=False)
    (output / "tmp").mkdir(exist_ok=False)
    (output / "wandb").mkdir(exist_ok=False)
    command = [
        config["python"],
        "-u",
        "-B",
        str(Path(config["source_root"]) / "fast_td3/train.py"),
        *config["arguments"],
    ]
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": visible,
        "MUJOCO_GL": "egl",
        "MUJOCO_EGL_DEVICE_ID": "0",
        "WANDB_MODE": "offline",
        "WANDB_DIR": str(output / "wandb"),
        "WANDB_DATA_DIR": str(output / "wandb/data"),
        "WANDB_CACHE_DIR": str(WORKSPACE / ".cache/wandb"),
        "WANDB_CONFIG_DIR": str(WORKSPACE / ".config/wandb"),
        "TMPDIR": str(output / "tmp"),
        "XDG_CACHE_HOME": str(WORKSPACE / ".cache"),
        "TORCH_HOME": str(WORKSPACE / ".cache/torch"),
        "TRITON_CACHE_DIR": str(WORKSPACE / ".cache/triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(WORKSPACE / ".cache/torchinductor"),
        "HF_HOME": str(WORKSPACE / ".cache/huggingface"),
        "IMAGEIO_USERDIR": str(WORKSPACE / ".cache/imageio"),
        "MPLCONFIGDIR": str(WORKSPACE / ".cache/matplotlib"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "IMAGEIO_FFMPEG_NO_PREVENT_SIGINT": "1",
    }
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment.pop("WANDB_DISABLED", None)
    result = {
        "schema_version": 1,
        "status": "running",
        "command": command,
        "cwd": str(author_output),
        "source_commit": AUTHOR_COMMIT,
        "sources": sources,
        "runtime": runtime,
        "phase": config["phase"],
        "task": TASK,
        "embodiment": "unitree_h1_with_hands",
        "cuda_visible_devices": visible,
        "effective_author_gl": "egl",
        "physical_egl_device": 0,
        "metrics": {},
        "experiment_accepted": False,
        "claim_scope": "Original-command process execution only; checkpoint, native returns, paper fidelity and EGL/compile qualification require separate acceptance",
        "descendant_cleanup_owner": "canonical runner",
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    receipt = output / "receipt.json"

    def publish() -> None:
        temporary = receipt.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        temporary.replace(receipt)

    def cancelled(signum: int, frame: Any) -> None:
        raise InterruptedError(f"Original FastTD3 command cancelled by signal{signum}")

    previous = signal.signal(signal.SIGTERM, cancelled)
    process = None
    started = time.monotonic()
    try:
        publish()
        with (output / "author.log").open("w") as log:
            process = subprocess.Popen(
                command,
                cwd=author_output,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            result["author_pid"] = process.pid
            publish()
            code = process.wait(timeout=config["timeout_s"])
            result["author_exit_code"] = code
            if code != 0:
                raise RuntimeError(f"Original FastTD3 command exited{code}")
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
