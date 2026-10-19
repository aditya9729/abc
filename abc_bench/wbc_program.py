"""Supervise the original R1 Lite feedback and PPO programs under an ABC lease."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import stat
import subprocess
import time
from pathlib import Path

from abc_bench.subset import require_subset_lease

AUTHOR_ROOT = Path("/home/user/aditya/RL/r1pro-rl-body-controller")
AUTHOR_COMMIT = "c437c7183aee296b973bb706432b92e8a1ae1f16"
AUTHOR_PYTHON = Path(
    "/home/user/aditya/RL/builds/r1pro-wbc-local-pilot-20261016-01/prefix/bin/python"
)
PROGRAMS = {
    "feedback-fit": ("-m", "r1lite_wbc.feedback", "fit"),
    "feedback-evaluate": ("-m", "r1lite_wbc.feedback", "evaluate"),
    "pose-ppo": ("scripts/train_feedback_residual.py",),
    "exported-policy-evaluate": ("scripts/evaluate_exported_policy.py",),
}
FLAGS = {
    "feedback-fit": {
        "--dataset",
        "--output",
        "--model",
        "--epochs",
        "--batch-size",
        "--learning-rate",
        "--device",
        "--seed",
    },
    "feedback-evaluate": {
        "--policy",
        "--references",
        "--output",
        "--model",
        "--episodes",
        "--batch-size",
        "--device",
    },
    "pose-ppo": {
        "--config",
        "--model",
        "--references",
        "--nominal-policy",
        "--normalization-dataset",
        "--output",
        "--seed",
        "--device",
    },
}
FLAGS["exported-policy-evaluate"] = (FLAGS["feedback-evaluate"] - {"--episodes"}) | {
    "--seed"
}
PATH_FLAGS = {
    "--dataset",
    "--model",
    "--policy",
    "--references",
    "--config",
    "--nominal-policy",
    "--normalization-dataset",
}
GIB = 1024**3


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    required = {
        "schema_version",
        "python",
        "source_root",
        "source_commit",
        "program",
        "arguments",
        "timeout_s",
        "input_manifest",
        "expected_summary",
    }
    if (
        not required.issubset(cfg)
        or set(cfg)
        - required
        - {"warp_cache_seed", "disk_reserve_bytes", "output_cap_bytes"}
        or type(cfg["schema_version"]) is not int
        or cfg["schema_version"] != 1
    ):
        raise ValueError(
            "Require WBC schema1 fields and supported optional resource/cache fields"
        )
    storage_limits(cfg)
    if (
        cfg["python"] != str(AUTHOR_PYTHON)
        or cfg["source_root"] != str(AUTHOR_ROOT)
        or cfg["source_commit"] != AUTHOR_COMMIT
        or cfg["program"] not in PROGRAMS
    ):
        raise ValueError("Select the pinned original R1 Lite source and runtime")
    timeout = cfg["timeout_s"]
    if (
        type(timeout) not in (int, float)
        or not math.isfinite(timeout)
        or not 0 < timeout <= 7100
    ):
        raise ValueError("Worker timeout must be finite and within (0,7100]")
    args = cfg["arguments"]
    if (
        not isinstance(args, list)
        or len(args) % 2
        or any(not isinstance(v, str) or not v or "\0" in v for v in args)
    ):
        raise ValueError("Arguments must be flag/value pairs")
    options = dict(zip(args[::2], args[1::2], strict=True))
    if len(options) * 2 != len(args) or set(options) - FLAGS[cfg["program"]]:
        raise ValueError("Unknown or duplicate original argument")
    if options.get("--output") != "{output}" or "--model" not in options:
        raise ValueError("Require explicit model and --output {output}")
    device = "cpu" if cfg["program"].endswith("evaluate") else "cuda:0"
    if options.get("--device") != device:
        raise ValueError("Use the explicit original phase device")
    manifest = Path(cfg["input_manifest"])
    if not manifest.is_absolute() or not manifest.is_file():
        raise ValueError("Input manifest must be an existing absolute file")
    for key in PATH_FLAGS & options.keys():
        p = Path(options[key])
        if not p.is_absolute() or not p.exists():
            raise ValueError("Input argument must exist at an absolute path: " + key)
    expected = cfg["expected_summary"]
    if (
        not isinstance(expected, dict)
        or not expected
        or any(
            not isinstance(k, str) or type(v) not in (int, str, bool)
            for k, v in expected.items()
        )
    ):
        raise ValueError("Require explicit scalar expected summary fields")
    cache_seed_info(cfg)
    return cfg


def validate_job(job, parent_timeout=None):
    if job["arguments"][:1] != ["--config"] or len(job["arguments"]) != 2:
        raise ValueError("WBC worker requires one explicit --config")
    cfg = load_config(Path(job["arguments"][1]))
    effective = min(
        job["max_wall_s"],
        parent_timeout if parent_timeout is not None else job["max_wall_s"],
    )
    if effective < cfg["timeout_s"] + 100:
        raise ValueError("WBC parent requires worker plus100s cleanup")


def verify_inputs(cfg):
    root = Path(cfg["source_root"])
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    )
    if commit != AUTHOR_COMMIT or dirty.strip():
        raise ValueError("Original WBC source must remain pinned and clean")
    manifest = Path(cfg["input_manifest"])
    rows = json.loads(manifest.read_text())["files"]
    pins = {row["path"]: row["sha256"] for row in rows}
    if len(pins) != len(rows) or not rows:
        raise ValueError("Input pins must be nonempty and unique")
    for path, expected in pins.items():
        p = Path(path)
        if not p.is_absolute() or digest(p) != expected:
            raise ValueError("Changed pinned input: " + path)
    site = AUTHOR_PYTHON.parent.parent / "lib/python3.12/site-packages"
    required = {AUTHOR_PYTHON}
    for source in (root / "src").rglob("*.py"):
        installed = site / source.relative_to(root / "src")
        required.update((source, installed))
        if digest(source) != digest(installed):
            raise ValueError("Installed WBC source differs: " + str(installed))
    if PROGRAMS[cfg["program"]][0].endswith(".py"):
        required.add(root / PROGRAMS[cfg["program"]][0])
    if cfg["program"] == "exported-policy-evaluate":
        required.add(root / "scripts/evaluate_policy_parallel.py")
    options = dict(zip(cfg["arguments"][::2], cfg["arguments"][1::2], strict=True))
    for key in PATH_FLAGS & options.keys():
        p = Path(options[key])
        required.update(
            x for x in p.rglob("*") if x.is_file()
        ) if p.is_dir() else required.add(p)
    seed = cache_seed_info(cfg)
    if seed is not None:
        required.update([seed["receipt"], seed["canonical_receipt"], *seed["files"]])
    if not required.issubset(map(Path, pins)):
        raise ValueError("Manifest omits source/runtime or direct input dependencies")
    return {"manifest_sha256": digest(manifest), "files": len(rows)}


def storage_limits(cfg=None):
    """Resolve operational byte limits without changing the original program."""
    cfg = {} if cfg is None else cfg
    reserve = cfg.get("disk_reserve_bytes", 20 * GIB)
    cap = cfg.get("output_cap_bytes", 5 * GIB)
    if type(reserve) is not int or reserve < 8 * GIB:
        raise ValueError("WBC disk_reserve_bytes must be an integer of at least 8GiB")
    if type(cap) is not int or not 0 < cap <= 5 * GIB:
        raise ValueError("WBC output_cap_bytes must be an integer within (0,5GiB]")
    return reserve, cap


def storage_check(root, cfg=None):
    reserve, cap = storage_limits(cfg)
    free = shutil.disk_usage(root).free
    size = sum(p.lstat().st_size for p in root.rglob("*") if p.is_file())
    if free < reserve or size > cap:
        raise RuntimeError(
            f"WBC disk floor {reserve} bytes or output ceiling {cap} bytes reached"
        )
    return {
        "free_bytes": free,
        "output_bytes": size,
        "disk_reserve_bytes": reserve,
        "output_cap_bytes": cap,
    }


def cache_seed_info(cfg):
    value = cfg.get("warp_cache_seed")
    if value is None:
        return None
    if not isinstance(value, str) or cfg["program"] != "pose-ppo":
        raise ValueError("Warp seed requires an absolute completed PPO cache path")
    source = Path(value)
    if (
        not source.is_absolute()
        or source != source.resolve()
        or not source.is_dir()
        or source.name != "warp_cache_path"
        or source.parent.name != "cache"
    ):
        raise ValueError("Warp seed must be a completed run/cache/warp_cache_path")
    receipt_path = source.parents[1] / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    run = receipt_path.parent
    canonical_path = run.parent / "runs" / f"{run.name}.json"
    canonical = json.loads(canonical_path.read_text())
    if (
        canonical.get("status") != "completed"
        or canonical.get("worker_exit_code") != 0
        or canonical.get("worker_receipt_sha256") != digest(receipt_path)
    ):
        raise ValueError("Warp seed requires matching completed canonical receipt")
    if (
        receipt.get("schema") != "original_r1lite_wbc_program/1"
        or receipt.get("status") != "completed"
        or receipt.get("program") != "pose-ppo"
        or receipt.get("author_exit_code") != 0
        or receipt.get("direct_child_stopped") is not True
        or any(
            receipt.get("configuration", {}).get(key) != cfg[key]
            for key in ("python", "source_root", "source_commit")
        )
    ):
        raise ValueError("Warp seed requires successful unchanged original PPO runtime")
    files, size = [], 0
    for path in [source, *sorted(source.rglob("*"))]:
        info = path.lstat()
        if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
            raise ValueError(
                "Warp seed tree must contain only regular files/directories"
            )
        if stat.S_ISREG(info.st_mode):
            files.append(path)
            size += info.st_size
    if not files or size > 5 * GIB:
        raise ValueError("Warp seed must be nonempty and at most5GiB")
    return {
        "source": source,
        "receipt": receipt_path,
        "canonical_receipt": canonical_path,
        "files": files,
        "bytes": size,
    }


def copy_cache_seed(cfg, destination, output, started):
    seed = cache_seed_info(cfg)
    if seed is None:
        return None
    destination = Path(destination)
    if any(destination.iterdir()):
        raise FileExistsError("Warp destination cache must be new and empty")
    reserve, cap = storage_limits(cfg)
    usage = storage_check(output, cfg)
    if (
        usage["free_bytes"] - seed["bytes"] < reserve
        or usage["output_bytes"] + seed["bytes"] > cap
    ):
        raise RuntimeError("Warp seed would exceed WBC disk/output limits")
    pins = {
        row["path"]: row["sha256"]
        for row in json.loads(Path(cfg["input_manifest"]).read_text())["files"]
    }
    copied = 0
    for source in seed["files"]:
        target = destination / source.relative_to(seed["source"])
        target.parent.mkdir(parents=True, exist_ok=True)
        remaining = source.stat().st_size
        with source.open("rb") as incoming, target.open("xb") as outgoing:
            while remaining:
                if time.monotonic() - started > cfg["timeout_s"]:
                    raise subprocess.TimeoutExpired(
                        "Warp cache seed copy", cfg["timeout_s"]
                    )
                storage_check(output, cfg)
                chunk = incoming.read(min(1024**2, remaining))
                if not chunk:
                    raise ValueError("Warp seed changed during bounded copy")
                outgoing.write(chunk)
                remaining -= len(chunk)
                copied += len(chunk)
            if incoming.read(1):
                raise ValueError("Warp seed grew during bounded copy")
        if digest(target) != pins.get(str(source)):
            raise ValueError("Copied Warp seed differs from pinned input")
    storage_check(output, cfg)
    return {
        "source": str(seed["source"]),
        "source_receipt_sha256": digest(seed["receipt"]),
        "files_copied": len(seed["files"]),
        "bytes_copied": copied,
        "destination": str(destination),
        "shared_writable_cache": False,
    }


def command(cfg, output):
    prefix = list(PROGRAMS[cfg["program"]])
    if prefix[0].endswith(".py"):
        prefix[0] = str(Path(cfg["source_root"]) / prefix[0])
    return [
        cfg["python"],
        *([] if cfg["program"] == "exported-policy-evaluate" else ["-I"]),
        "-u",
        "-B",
        *prefix,
        *(str(output) if a == "{output}" else a for a in cfg["arguments"]),
    ]


def summary_path(cfg, output):
    return (
        output
        / {
            "feedback-fit": "policy_manifest.json",
            "feedback-evaluate": "report.json",
            "exported-policy-evaluate": "report.json",
            "pose-ppo": "training_summary.json",
        }[cfg["program"]]
    )


def run(cfg, output):
    require_subset_lease()
    inputs = verify_inputs(cfg)
    output = Path(output)
    output.mkdir(
        parents=True, exist_ok=True
    )  # Canonical coordinator owns this directory.
    author = output / "author-output"
    if author.exists() or (output / "receipt.json").exists():
        raise FileExistsError("Preserve the existing WBC execution")
    storage_check(output, cfg)
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME"}}
    env.update(WANDB_MODE="offline", WANDB_DISABLED="true", PYTHONDONTWRITEBYTECODE="1")
    for key in (
        "XDG_CACHE_HOME",
        "WARP_CACHE_PATH",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "CUDA_CACHE_PATH",
        "TMPDIR",
        "TORCH_HOME",
        "WANDB_DIR",
    ):
        path = output / "cache" / key.lower()
        path.mkdir(parents=True)
        env[key] = str(path)
    receipt = {
        "schema": "original_r1lite_wbc_program/1",
        "status": "running",
        "program": cfg["program"],
        "command": command(cfg, author),
        "input_check": inputs,
        "configuration": cfg,
        "worker_sha256": digest(__file__),
        "experiment_accepted": False,
        "descendant_cleanup_owner": "canonical runner",
    }
    write(output / "receipt.json", receipt)
    child = None
    started = time.monotonic()

    def cancelled(signum, frame):
        raise RuntimeError(f"WBC worker received signal {signum}")

    previous = signal.signal(signal.SIGTERM, cancelled)
    try:
        if cfg.get("warp_cache_seed") is not None:
            receipt["warp_cache_seed"] = copy_cache_seed(
                cfg, env["WARP_CACHE_PATH"], output, started
            )
            write(output / "receipt.json", receipt)
        with (output / "author.log").open("w") as log:
            child = subprocess.Popen(
                receipt["command"],
                cwd=cfg["source_root"],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            receipt["author_pid"] = child.pid
            while child.poll() is None:
                receipt["storage"] = storage_check(output, cfg)
                if time.monotonic() - started > cfg["timeout_s"]:
                    raise subprocess.TimeoutExpired(
                        receipt["command"], cfg["timeout_s"]
                    )
                time.sleep(0.2)
            receipt["author_exit_code"] = child.returncode
            if child.returncode:
                raise RuntimeError(f"Original WBC program exited {child.returncode}")
        if verify_inputs(cfg) != inputs:
            raise ValueError("Input manifest changed during execution")
        path = summary_path(cfg, author)
        summary = json.loads(path.read_text())
        if cfg["program"] == "feedback-fit":
            summary = summary["training"]
        if any(summary.get(k) != v for k, v in cfg["expected_summary"].items()):
            raise ValueError("Original summary differs from the declared run")
        receipt.update(
            status="completed",
            summary_sha256=digest(path),
            summary=summary,
            storage=storage_check(output, cfg),
        )
        return receipt
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
        receipt.update(
            elapsed_seconds=time.monotonic() - started,
            direct_child_stopped=child is None or child.poll() is not None,
        )
        write(output / "receipt.json", receipt)
        signal.signal(signal.SIGTERM, previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(load_config(args.config), args.output)))


if __name__ == "__main__":
    main()
