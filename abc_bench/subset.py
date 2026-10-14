"""Launch one small author-method job through the shared GPU 0 reservation."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from abc_bench.resources import inspect_reserved_gpu

ALLOWED_MODULES = {
    "abc_bench.author_program",
    "abc_bench.mulligan_program",
    "abc_bench.fasttd3_program",
    "abc_author_subset.host",
    "abc_bench.subset_worker",
    "abc_bench.upstream_eval",
    "abc_bench.demo_collect",
    "abc_bench.fasttd3_host",
    "abc_bench.fasttd3_vector_stage",
    "abc_bench.fasttd3_scratch_stage",
    "abc_bench.fasttd3_frozen_stage",
    "abc_bench.fasttd3_campaign_stage",
    "abc_bench.fasttd3_reach_stage",
    "abc_bench.fasttd3_reach_matched_stage",
    "abc_bench.fasttd3_compiler_cache_diagnostic",
    "abc_bench.fasttd3_campaign_frozen_stage",
    "expo_abc_adapter.stage",
    "sadhana_fasttd3_hb.stage",
    "nrh.qf3_training",
}


def require_subset_lease() -> None:
    """Admit only descendants of the live coordinator holding the GPU 0 lock."""
    from abc_bench.runner import RESULTS

    ledger = json.loads((RESULTS / "gpu_budget.json").read_text())
    owner = ledger.get("active_parent_pid")
    if type(owner) is not int or owner <= 1:
        raise RuntimeError("No active ABC subset coordinator")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != ledger.get("active_gpu_uuid"):
        raise RuntimeError("GPU visibility differs from the coordinator reservation")
    if os.environ.get("MUJOCO_GL") != "disable" or ledger.get("gpu") != 0:
        raise RuntimeError("The subset requires the reserved GPU 0 and MJWarp")
    current = os.getppid()
    for _ in range(16):
        if current == owner:
            break
        if current <= 1:
            raise RuntimeError("This worker is not a descendant of the coordinator")
        stat = Path(f"/proc/{current}/stat").read_text()
        current = int(stat[stat.rfind(")") + 2 :].split()[1])
    else:
        raise RuntimeError("Coordinator ancestry exceeds the bounded process tree")
    with (RESULTS / ".gpu-budget.lock").open("rb") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        fcntl.flock(lock, fcntl.LOCK_UN)
        raise RuntimeError("The coordinator GPU 0 lease is not held")


def validate_expo_campaign_watchdog(config: dict, maximum: float) -> None:
    """Separate repeated-call campaign; keep the old extended recipe unchanged."""
    parameters = config.get("parameters")
    fixed = {
        "seed": 42,
        "batch_size": 64,
        "utd_ratio": 20,
        "completed_episode_warmup": 10,
        "updates_per_trigger": 3,
        "max_controls": 53100,
        "sampler_timing_plans": 0,
        "max_episode_completions": 15,
        "max_update_calls": 15,
        "max_seconds": 18000,
        "soft_seconds": 12000,
        "futility_start_calls": 6,
        "futility_min_calls": 9,
        "futility_patience": 2,
    }
    if (
        set(config) != {"schema_version", "phase", "parameters"}
        or type(config.get("schema_version")) is not int
        or config["schema_version"] != 1
        or config.get("phase") != "original_expo_task_rl_serial_campaign"
        or type(parameters) is not dict
        or any(
            type(parameters.get(k)) is not int or parameters[k] != v
            for k, v in fixed.items()
        )
        or parameters.get("memory_scheduling_profile")
        != "original_queue3_serial_update_v1"
        or parameters.get("checkpoint_retention") != "terminal_complete_only"
        or parameters["max_seconds"] > maximum - 300
        or type(parameters.get("episode_seeds")) is not list
        or len(parameters["episode_seeds"]) != 15
        or any(type(seed) is not int for seed in parameters["episode_seeds"])
        or len(set(parameters["episode_seeds"])) != 15
    ):
        raise ValueError(
            "Extended EXPO campaign requires fifteen episodes/calls, fixed original recipe, explicit stop/storage bounds and300s cleanup margin"
        )


def load_job(path: Path) -> dict[str, Any]:
    """Require a finite job and a normal installed, explicitly selected worker."""
    job = json.loads(path.read_text())
    if (
        type(job.get("schema_version")) is not int
        or job["schema_version"] != 1
        or job.get("module") not in ALLOWED_MODULES
    ):
        raise ValueError("Use schema_version 1 and an author-subset worker module")
    python = Path(job["python"])
    if not python.is_absolute() or not python.is_file():
        raise ValueError("python must name an existing absolute interpreter")
    arguments = job.get("arguments")
    if not isinstance(arguments, list) or any(type(x) is not str for x in arguments):
        raise ValueError("arguments must be a list of strings")
    if any(x.startswith("--out") for x in arguments):
        raise ValueError("The coordinator owns the output directory")
    if not isinstance(job.get("label"), str) or not job["label"]:
        raise ValueError("A method and stage label is required")
    if "claim_scope" in job and (
        not isinstance(job["claim_scope"], str) or not job["claim_scope"].strip()
    ):
        raise ValueError("claim_scope must be a nonempty description")
    maximum = job.get("max_wall_s")
    maximum_allowed = (
        12600 if job["module"] == "abc_bench.fasttd3_campaign_stage" else 7200
    )
    if job["module"] == "abc_bench.mulligan_program":
        from abc_bench.mulligan_program import (
            CLEANUP_SECONDS,
            MAX_WORKER_SECONDS,
            load_config,
        )

        if len(arguments) != 2 or arguments[0] != "--config":
            raise ValueError("Mulligan jobs require one explicit --config")
        config = load_config(Path(arguments[1]))
        if (
            config["mode"] != "gpu"
            or type(maximum) not in (int, float)
            or config["timeout_s"] + CLEANUP_SECONDS > maximum
        ):
            raise ValueError(
                "Mulligan GPU jobs need the declared worker timeout plus60s cleanup"
            )
        maximum_allowed = MAX_WORKER_SECONDS + CLEANUP_SECONDS
    if job["module"] == "abc_bench.fasttd3_program":
        from abc_bench.fasttd3_program import (
            CLEANUP_SECONDS,
            EGL_RENDER_PHASE,
            EGL_RENDER_SECONDS,
            MAX_WORKER_SECONDS,
            QUALIFICATION_SECONDS,
            load_config,
        )

        if len(arguments) != 2 or arguments[0] != "--config":
            raise ValueError("Original FastTD3 jobs require one explicit --config")
        config = load_config(Path(arguments[1]))
        if (
            type(maximum) not in (int, float)
            or config["timeout_s"] + CLEANUP_SECONDS > maximum
        ):
            raise ValueError(
                "Original FastTD3 jobs need the worker timeout plus60s cleanup"
            )
        maximum_allowed = (
            MAX_WORKER_SECONDS
            if config["phase"] == "original_humanoidbench_train"
            else EGL_RENDER_SECONDS
            if config["phase"] == EGL_RENDER_PHASE
            else QUALIFICATION_SECONDS
        ) + CLEANUP_SECONDS
    if (
        job["module"] == "expo_abc_adapter.stage"
        and type(maximum) in (int, float)
        and math.isfinite(maximum)
        and maximum > 7200
    ):
        if len(arguments) != 2 or arguments[0] != "--config":
            raise ValueError("Extended EXPO jobs require one explicit --config file")
        config_path = Path(arguments[1])
        if (
            not config_path.is_absolute()
            or config_path.is_symlink()
            or not config_path.is_file()
            or not 0 < config_path.stat().st_size <= 1024**2
        ):
            raise ValueError(
                "Extended EXPO config must be an absolute regular JSON file"
            )
        config = json.loads(config_path.read_text())
        parameters = config.get("parameters") if type(config) is dict else None
        if (
            type(config) is dict
            and config.get("phase") == "original_expo_task_rl_serial_campaign"
        ):
            validate_expo_campaign_watchdog(config, maximum)
        else:
            fixed = {
                "seed": 42,
                "batch_size": 64,
                "utd_ratio": 20,
                "completed_episode_warmup": 10,
                "updates_per_trigger": 3,
                "max_controls": 38940,
                "sampler_timing_plans": 0,
            }
            if (
                type(config) is not dict
                or type(config.get("schema_version")) is not int
                or config["schema_version"] != 1
                or (
                    config.get("phase") != "original_expo_task_rl"
                    and not (
                        config.get("phase") == "original_expo_task_rl_serial"
                        and type(parameters) is dict
                        and parameters.get("memory_scheduling_profile")
                        == "original_queue3_serial_update_v1"
                    )
                )
                or type(parameters) is not dict
                or any(
                    type(parameters.get(key)) is not int or parameters[key] != value
                    for key, value in fixed.items()
                )
                or type(parameters.get("max_seconds")) is not int
                or not 1 <= parameters["max_seconds"] <= 18000
                or parameters["max_seconds"] > maximum - 300
                or type(parameters.get("episode_seeds")) is not list
                or len(parameters["episode_seeds"]) != 11
                or any(type(seed) is not int for seed in parameters["episode_seeds"])
                or len(set(parameters["episode_seeds"])) != 11
            ):
                raise ValueError(
                    "Extended EXPO requires the original eleven-episode B64/UTD20 "
                    "phase, an at-most18000s worker and300s parent cleanup margin"
                )
        maximum_allowed = 18300
    if (
        type(maximum) not in (int, float)
        or not math.isfinite(maximum)
        or not 0 < maximum <= maximum_allowed
    ):
        raise ValueError(
            f"max_wall_s must be finite, positive and at most {maximum_allowed}"
        )
    return job


def run_job(args: Any) -> dict[str, Any]:
    """Keep earlier charges; reap the whole owned group before releasing its lock."""
    from abc_bench.runner import RESULTS, execute_command, publish, write_json

    if args.training_config is None:
        raise ValueError("author-subset requires --training-config")
    job = load_job(args.training_config)
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / ".gpu-budget.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger_path = RESULTS / "gpu_budget.json"
        ledger = json.loads(ledger_path.read_text())
        if ledger.get("gpu") != 0 or any(
            key in ledger
            for key in ("active_parent_pid", "active_gpu_uuid", "active_visual_lease")
        ):
            raise RuntimeError("GPU 0 has an active or unresolved campaign owner")
        charged, limit = ledger["charged_seconds"], ledger["limit_seconds"]
        if any(
            type(x) not in (int, float) or not math.isfinite(x) or x < 0
            for x in (charged, limit)
        ):
            raise ValueError(
                "The original ledger must have finite nonnegative charges and credit"
            )
        timeout = min(float(job["max_wall_s"]), float(args.timeout_seconds))
        if timeout <= 0 or not math.isfinite(timeout) or limit - charged < timeout:
            raise RuntimeError(
                "Insufficient finite GPU 0 admission credit for this job"
            )
        binding = inspect_reserved_gpu(0)
        run_id = (
            "author-subset-"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            + "-"
            + uuid.uuid4().hex[:6]
        )
        out = RESULTS / run_id
        out.mkdir()
        # These workers require a directory without coordinator bookkeeping.
        subdirectory = {
            "nrh.qf3_training": "qf3",
            "expo_abc_adapter.stage": "expoft",
        }.get(job["module"])
        worker_out = out / subdirectory if subdirectory is not None else out
        command = [
            job["python"],
            "-I",
            "-u",
            "-B",
            "-m",
            job["module"],
            *job["arguments"],
            "--out" if job["module"] == "nrh.qf3_training" else "--output",
            str(worker_out),
        ]
        write_json(out / "job.json", job)
        receipt = {
            "schema_version": 1,
            "run_id": run_id,
            "algorithm": job["label"],
            "phase": "small_subset",
            "status": "running",
            "task": "h1hand-stand-v0"
            if job["module"] == "sadhana_fasttd3_hb.stage"
            else "h1hand-hurdle-v0"
            if job["module"] == "abc_bench.fasttd3_program"
            else "SquareNarrow"
            if job["module"] == "abc_bench.mulligan_program"
            else "put_plastic_bottles_in_bin",
            "embodiment": "unitree_h1_with_hands"
            if job["module"]
            in {"sadhana_fasttd3_hb.stage", "abc_bench.fasttd3_program"}
            else "franka_panda"
            if job["module"] == "abc_bench.mulligan_program"
            else "native_yam",
            "claim_scope": job.get(
                "claim_scope",
                "Bounded original FastTD3 Stand integration/capacity; no convergence, ABC transfer or paper reproduction claim"
                if job["module"] == "sadhana_fasttd3_hb.stage"
                else "Original FastTD3 Hurdle process execution; native returns, checkpoints and recipe fidelity require separate acceptance"
                if job["module"] == "abc_bench.fasttd3_program"
                else "Original Mulligan process execution; native results and checkpoints require separate acceptance"
                if job["module"] == "abc_bench.mulligan_program"
                else "small ABC-VLA transfer diagnostic; not full paper reproduction",
            ),
            "gpu": binding,
            "command": command,
            "resolved_job_sha256": hashlib.sha256(
                (out / "job.json").read_bytes()
            ).hexdigest(),
            "metrics": {},
            "artifacts": [
                {
                    "label": "Execution log",
                    "path": str((out / "run.log").relative_to(RESULTS)),
                }
            ],
            "blockers": [],
        }
        publish(receipt, RESULTS)
        ledger["charged_seconds"] = charged + timeout
        ledger["active_parent_pid"] = os.getpid()
        ledger["active_gpu_uuid"] = binding["uuid"]
        write_json(ledger_path, ledger)
        started = time.monotonic()
        try:
            code = execute_command(
                command, out / "run.log", timeout, gpu_uuid=binding["uuid"]
            )
            receipt["worker_exit_code"] = code
            if code != 0:
                raise RuntimeError(f"Worker exited {code}; see run.log")
            worker_path = worker_out / "receipt.json"
            worker = json.loads(worker_path.read_text())
            if worker.get("status") != "completed":
                raise RuntimeError("Worker did not finish its declared stage")
            receipt["metrics"] = worker.get("metrics", {})
            receipt["worker_receipt_sha256"] = hashlib.sha256(
                worker_path.read_bytes()
            ).hexdigest()
            receipt["artifacts"].append(
                {
                    "label": "Worker results",
                    "path": str(worker_path.relative_to(RESULTS)),
                }
            )
            receipt["status"] = "completed"
        except (
            OSError,
            ValueError,
            RuntimeError,
            subprocess.TimeoutExpired,
            KeyboardInterrupt,
        ) as error:
            receipt["status"] = "failed"
            receipt["blockers"] = [str(error)]
        finally:
            elapsed = time.monotonic() - started
            ledger["charged_seconds"] = charged + elapsed
            ledger.pop("active_parent_pid", None)
            ledger.pop("active_gpu_uuid", None)
            write_json(ledger_path, ledger)
            receipt["elapsed_seconds"] = elapsed
            publish(receipt, RESULTS)
        return receipt
