"""Launch one small author-method job through the shared GPU 0 reservation."""

from __future__ import annotations

import argparse
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
    "abc_bench.realtime_update_probe",
    "abc_bench.policy_path_probe",
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
    "sadhana_r1lite.fasttd3",
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


def _author_config(job: dict[str, Any]) -> dict[str, Any] | None:
    """Inspect the new schema only; preserve historical schema1 job admission."""
    from abc_bench.author_program import load_config

    arguments = job["arguments"]
    parser = argparse.ArgumentParser(add_help=False, exit_on_error=False)
    parser.add_argument("--config")
    try:
        selected, _ = parser.parse_known_args(arguments)
    except argparse.ArgumentError:
        return None
    if selected.config is None:
        return None
    path = Path(selected.config)
    requested_path = path
    if not path.is_absolute():
        from abc_bench.runner import ROOT

        # execute_command starts the worker at ROOT, regardless of the shell cwd.
        path = ROOT / path
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(value, dict) or value.get("schema_version") != 2:
        return None
    if not requested_path.is_absolute():
        raise ValueError("Schema2 Catcher jobs require an absolute --config path")
    if not (
        len(arguments) == 2
        and arguments[0] == "--config"
        or len(arguments) == 1
        and arguments[0].startswith("--config=")
    ):
        raise ValueError("Schema2 Catcher jobs require one explicit --config")
    return load_config(path)


def _extended_author_training_config(job: dict[str, Any]) -> dict[str, Any] | None:
    """Admit finite original training runs with a program-specific watchdog."""
    from abc_bench.author_program import (
        TRAINING_WATCHDOG_LIMITS,
        load_config,
        memory_profile_requested,
    )

    maximum = job.get("max_wall_s")
    if job["module"] != "abc_bench.author_program":
        return None
    arguments = job["arguments"]
    parser = argparse.ArgumentParser(add_help=False, exit_on_error=False)
    parser.add_argument("--config")
    try:
        selected, _ = parser.parse_known_args(arguments)
        path = Path(selected.config) if selected.config else None
        read_path = path
        if path is not None and not path.is_absolute():
            from abc_bench.runner import ROOT

            read_path = ROOT / path
        value = json.loads(read_path.read_text()) if read_path is not None else None
    except (argparse.ArgumentError, OSError, ValueError):
        path = value = None
    extended_parent = type(maximum) in (int, float) and maximum > 7200
    timeout = value.get("timeout_s") if isinstance(value, dict) else None
    extended_worker = type(timeout) in (int, float) and timeout > 7100
    managed_memory = isinstance(value, dict) and memory_profile_requested(value)
    if not extended_parent and not extended_worker and not managed_memory:
        return None
    if not (
        len(arguments) == 2
        and arguments[0] == "--config"
        or len(arguments) == 1
        and arguments[0].startswith("--config=")
    ):
        raise ValueError("Extended author training jobs require one explicit --config")
    if path is None or not path.is_absolute() or not path.is_file():
        raise ValueError(
            "Extended author training jobs require an absolute config file"
        )
    config = load_config(path)
    if (
        config["schema_version"] != 1
        or config["program"] not in TRAINING_WATCHDOG_LIMITS
    ):
        raise ValueError(
            "Extended admission is only for original schema1 author training"
        )
    return config


def _r1lite_config(job: dict[str, Any]) -> tuple[dict, bytes, bytes]:
    """Validate the cross-package boundary without importing the task runtime."""
    arguments = job["arguments"]
    if len(arguments) != 2 or arguments[0] != "--config":
        raise ValueError("R1 Lite jobs require one explicit --config")

    def read_json(value: Any) -> tuple[dict, bytes]:
        path = Path(value) if type(value) is str else None
        if (
            path is None
            or not path.is_absolute()
            or path.is_symlink()
            or not path.is_file()
            or not 0 < path.stat().st_size <= 1024**2
        ):
            raise ValueError("R1 Lite configs require absolute regular JSON files")
        raw = path.read_bytes()
        config = json.loads(raw)
        if type(config) is not dict:
            raise ValueError("R1 Lite configs must contain JSON objects")
        return config, raw

    config, raw = read_json(arguments[1])
    fixed = {
        "schema_version": 1,
        "seed": 1,
        "phase": "original_fasttd3_r1lite",
        "author_source_commit": "dce87035e2d58608c015f01f3ee332d6ba84a86b",
        "adapter_version": "0.1.0",
    }
    if (
        set(config) != {*fixed, "task_config", "total_timesteps", "timeout_s"}
        or any(
            type(config.get(k)) is not type(v) or config[k] != v
            for k, v in fixed.items()
        )
        or type(config.get("total_timesteps")) is not int
        or not 12 <= config["total_timesteps"] <= 10000
        or type(config.get("timeout_s")) not in (int, float)
        or not math.isfinite(config["timeout_s"])
        or not 0 < config["timeout_s"] <= 7100
        or job.get("task") != "r1lite_remote_into_tray"
        or job.get("embodiment") != "r1lite_2025"
    ):
        raise ValueError(
            "R1 Lite requires the fixed task, embodiment and bounded worker config"
        )
    _, task_raw = read_json(config["task_config"])
    return config, raw, task_raw


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
    for name in ("task", "embodiment"):
        if name in job and (not isinstance(job[name], str) or not job[name].strip()):
            raise ValueError(f"{name} must be a nonempty description")
    maximum = job.get("max_wall_s")
    maximum_allowed = (
        12600 if job["module"] == "abc_bench.fasttd3_campaign_stage" else 7200
    )
    if job["module"] == "sadhana_r1lite.fasttd3":
        config, _, _ = _r1lite_config(job)
        if (
            type(maximum) not in (int, float)
            or not math.isfinite(maximum)
            or config["timeout_s"] + 60 > maximum
        ):
            raise ValueError("R1 Lite jobs need the worker timeout plus 60s cleanup")
    if job["module"] == "abc_bench.author_program":
        from abc_bench.author_program import (
            CATCHER_CLEANUP_SECONDS,
            TRAINING_WATCHDOG_LIMITS,
        )

        training_config = _extended_author_training_config(job)
        if training_config is not None:
            worker_maximum, cleanup_seconds = TRAINING_WATCHDOG_LIMITS[
                training_config["program"]
            ]
            if (
                type(maximum) not in (int, float)
                or not math.isfinite(maximum)
                or training_config["timeout_s"] + cleanup_seconds > maximum
            ):
                raise ValueError(
                    "Extended author training jobs need worker timeout plus 100s cleanup"
                )
            maximum_allowed = worker_maximum + cleanup_seconds

        config = _author_config(job)
        if config is not None and (
            type(maximum) not in (int, float)
            or not math.isfinite(maximum)
            or config["timeout_s"] + CATCHER_CLEANUP_SECONDS > maximum
            or job.get("task", "Catcher-v3") != "Catcher-v3"
            or job.get("embodiment", "kinetix_symbolic") != "kinetix_symbolic"
        ):
            raise ValueError(
                "Catcher GPU jobs require the fixed task identity and worker timeout plus100s cleanup"
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
        if config["mode"] == "r1lite-native-diagnostic":
            from abc_bench.mulligan_r1lite_diagnostic import check_parent

            check_parent(job, config)
            maximum_allowed = 1900
        elif config["mode"] == "r1lite-original-qualification":
            from abc_bench.mulligan_r1lite_qualification import (
                check_parent as check_qualification_parent,
            )

            check_qualification_parent(job, config)
            maximum_allowed = 1900
        elif config["mode"] == "r1lite-original-training":
            from abc_bench.mulligan_r1lite_training import (
                MAX_WORKER_SECONDS as R1_TRAINING_MAX,
            )
            from abc_bench.mulligan_r1lite_training import (
                PARENT_MARGIN_SECONDS,
            )
            from abc_bench.mulligan_r1lite_training import (
                check_parent as check_training_parent,
            )

            check_training_parent(job, config)
            maximum_allowed = R1_TRAINING_MAX + PARENT_MARGIN_SECONDS
        elif config["mode"] == "r1lite-frozen-eval":
            from abc_bench.mulligan_r1lite_eval import (
                MAX_WORKER_SECONDS as R1_EVAL_MAX,
            )
            from abc_bench.mulligan_r1lite_eval import (
                PARENT_MARGIN_SECONDS as R1_EVAL_MARGIN,
            )
            from abc_bench.mulligan_r1lite_eval import (
                check_parent as check_eval_parent,
            )

            check_eval_parent(job, config)
            maximum_allowed = R1_EVAL_MAX + R1_EVAL_MARGIN
        else:
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
        job["module"] == "abc_bench.realtime_update_probe"
        and type(maximum) is int
        and maximum in (7300, 9100)
    ):
        from abc_bench.realtime_update_probe import load_config, validate_job

        # Each sealed100-call phase has its own worker bound plus100 for cleanup.
        # Authenticate its full phase/interpreter/deadline contract before raising
        # this job's cap; unrelated workers and the old ten-call phase stay bounded.
        validate_job(job)
        config = load_config(Path(arguments[1]))
        expected_phase = {
            7300: "historical_online_fixed_data_100",
            9100: "historical_online_fixed_data_100_training",
        }[maximum]
        if config.get("phase") != expected_phase:
            raise ValueError("Extended exception requires its exact sealed hundred-call phase")
        maximum_allowed = maximum
    if (
        type(maximum) not in (int, float)
        or not math.isfinite(maximum)
        or not 0 < maximum <= maximum_allowed
    ):
        raise ValueError(
            f"max_wall_s must be finite, positive and at most {maximum_allowed}"
        )
    if job["module"] == "abc_bench.realtime_update_probe":
        from abc_bench.realtime_update_probe import validate_job

        validate_job(job)
    if job["module"] == "abc_bench.policy_path_probe":
        from abc_bench.policy_path_probe import validate_job

        validate_job(job)
    return job


def run_job(args: Any) -> dict[str, Any]:
    """Keep earlier charges; reap the whole owned group before releasing its lock."""
    from abc_bench.runner import RESULTS, execute_command, publish, write_json

    if args.training_config is None:
        raise ValueError("author-subset requires --training-config")
    job = load_job(args.training_config)
    if job["module"] == "abc_bench.realtime_update_probe":
        from abc_bench.realtime_update_probe import validate_job

        validate_job(job, args.timeout_seconds)
    policy_identity = job["module"] == "abc_bench.policy_path_probe"
    if policy_identity:
        from abc_bench.policy_path_probe import validate_job

        validate_job(job, args.timeout_seconds)
    r1_diagnostic = False
    r1_training = False
    r1_eval = False
    r1_qualification = False
    if job["module"] == "abc_bench.mulligan_program":
        from abc_bench.mulligan_program import load_config as load_mulligan_config

        mulligan_config = load_mulligan_config(Path(job["arguments"][1]))
        r1_diagnostic = mulligan_config["mode"] == "r1lite-native-diagnostic"
        if r1_diagnostic:
            from abc_bench.mulligan_r1lite_diagnostic import check_parent

            # Refuse a shortened CLI bound before creating outputs or taking a lease.
            check_parent(job, mulligan_config, args.timeout_seconds)
        r1_training = mulligan_config["mode"] == "r1lite-original-training"
        if r1_training:
            from abc_bench.mulligan_r1lite_training import (
                check_parent as check_training_parent,
            )

            check_training_parent(job, mulligan_config, args.timeout_seconds)
        r1_qualification = mulligan_config["mode"] == "r1lite-original-qualification"
        if r1_qualification:
            from abc_bench.mulligan_r1lite_qualification import (
                check_parent as check_qualification_parent,
            )

            check_qualification_parent(job, mulligan_config, args.timeout_seconds)
        r1_eval = mulligan_config["mode"] == "r1lite-frozen-eval"
        if r1_eval:
            from abc_bench.mulligan_r1lite_eval import check_parent as check_eval_parent

            check_eval_parent(job, mulligan_config, args.timeout_seconds)
    training_config = _extended_author_training_config(job)
    if training_config is not None:
        from abc_bench.author_program import TRAINING_WATCHDOG_LIMITS, admit_host_memory

        admit_host_memory(training_config)

        _, cleanup_seconds = TRAINING_WATCHDOG_LIMITS[training_config["program"]]
        if (
            type(args.timeout_seconds) not in (int, float)
            or not math.isfinite(args.timeout_seconds)
            or min(job["max_wall_s"], args.timeout_seconds)
            < training_config["timeout_s"] + cleanup_seconds
        ):
            raise ValueError(
                "Effective author training parent timeout must include 100s cleanup"
            )
    r1lite_config = None
    if job["module"] == "sadhana_r1lite.fasttd3":
        r1lite_config, r1lite_raw, task_raw = _r1lite_config(job)
        if (
            type(args.timeout_seconds) not in (int, float)
            or not math.isfinite(args.timeout_seconds)
            or min(job["max_wall_s"], args.timeout_seconds)
            < r1lite_config["timeout_s"] + 60
        ):
            raise ValueError(
                "Effective R1 Lite parent timeout must contain worker plus 60s cleanup"
            )
    catcher_profile = False
    if job["module"] == "abc_bench.author_program":
        from abc_bench.author_program import CATCHER_CLEANUP_SECONDS

        author_config = _author_config(job)
        catcher_profile = author_config is not None
        if (
            catcher_profile
            and min(job["max_wall_s"], args.timeout_seconds)
            < author_config["timeout_s"] + CATCHER_CLEANUP_SECONDS
        ):
            raise ValueError(
                "Effective Catcher parent timeout must contain worker plus100s cleanup"
            )
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
        worker_arguments = job["arguments"]
        config_snapshots = None
        if r1lite_config is not None:
            task_snapshot = (out / "task-config.json").resolve()
            task_snapshot.write_bytes(task_raw)
            task_snapshot.chmod(0o444)
            worker_snapshot = (out / "worker-config.json").resolve()
            write_json(
                worker_snapshot, {**r1lite_config, "task_config": str(task_snapshot)}
            )
            worker_snapshot.chmod(0o444)
            worker_arguments = ["--config", str(worker_snapshot)]
            config_snapshots = {
                "worker_original_path": job["arguments"][1],
                "worker_original_sha256": hashlib.sha256(r1lite_raw).hexdigest(),
                "worker_snapshot_path": str(worker_snapshot),
                "worker_snapshot_sha256": hashlib.sha256(
                    worker_snapshot.read_bytes()
                ).hexdigest(),
                "task_original_path": r1lite_config["task_config"],
                "task_snapshot_path": str(task_snapshot),
                "task_sha256": hashlib.sha256(task_raw).hexdigest(),
            }
        command = [
            job["python"],
            "-I",
            "-u",
            "-B",
            "-m",
            job["module"],
            *worker_arguments,
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
            "task": "Catcher-v3"
            if catcher_profile
            else "r1lite_remote_into_tray"
            if r1lite_config is not None
            else "h1hand-stand-v0"
            if job["module"] == "sadhana_fasttd3_hb.stage"
            else "h1hand-hurdle-v0"
            if job["module"] == "abc_bench.fasttd3_program"
            else "SquareNarrow"
            if job["module"] == "abc_bench.mulligan_program"
            else "put_plastic_bottles_in_bin",
            "embodiment": "kinetix_symbolic"
            if catcher_profile
            else "r1lite_2025"
            if r1lite_config is not None
            else "unitree_h1_with_hands"
            if job["module"]
            in {"sadhana_fasttd3_hb.stage", "abc_bench.fasttd3_program"}
            else "franka_panda"
            if job["module"] == "abc_bench.mulligan_program"
            else "native_yam",
            "claim_scope": job.get(
                "claim_scope",
                "Original Real-Time EXPO learner with declared Catcher integration; no frozen score or paper-reproduction claim"
                if catcher_profile
                else "Original FastTD3 learner with declared R1 Lite task adapter; native results require separate acceptance"
                if r1lite_config is not None
                else "Bounded original FastTD3 Stand integration/capacity; no convergence, ABC transfer or paper reproduction claim"
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
        if policy_identity:
            receipt.update(
                phase="r1lite-policy-path-identity",
                task="r1lite_remote_into_organizer",
                embodiment="r1lite_2025",
                experiment_accepted=False,
                claim_scope="Fixed-input original inference identity audit; no native controls or score",
                cleanup_profile="r1lite-policy-path-identity",
            )
        if r1_diagnostic:
            receipt.update(
                phase="native_adapter_diagnostic",
                task="r1lite_remote_into_tray",
                embodiment="r1lite_2025",
                experiment_accepted=False,
                claim_scope="Sealed R1 adapter diagnostic only; no learner score or scientific acceptance",
                cleanup_profile="r1lite-native-diagnostic",
            )
        if r1_training:
            receipt.update(
                phase="original_r1_training",
                task="r1lite_remote_into_tray",
                embodiment="r1lite_2025",
                experiment_accepted=False,
                claim_scope="Original Mulligan training with declared R1/data substitutions; no automatic checkpoint or score acceptance",
                cleanup_profile="r1lite-original-training",
            )
        if r1_qualification:
            receipt.update(
                phase="original_r1_qualification",
                task="r1lite_remote_into_tray",
                embodiment="r1lite_2025",
                qualification_stage=mulligan_config["stage"],
                qualification_phase=mulligan_config["phase"],
                experiment_accepted=False,
                claim_scope="Short original trainer qualification only; no benchmark or exact-resume claim",
                cleanup_profile="r1lite-original-qualification",
            )
        if r1_eval:
            receipt.update(
                phase="original_r1_frozen_eval",
                task="r1lite_remote_into_tray",
                embodiment="r1lite_2025",
                experiment_accepted=False,
                claim_scope="Original Mulligan frozen DEV20 evaluation; native/media/cleanup and scores require separate root acceptance",
                cleanup_profile="r1lite-frozen-eval",
                method=mulligan_config["method"],
                num_action_samples=mulligan_config["num_action_samples"],
            )
        if config_snapshots is not None:
            receipt["config_snapshots"] = config_snapshots
        if (
            job["module"]
            in {"abc_bench.author_program", "abc_bench.realtime_update_probe"}
            and not catcher_profile
        ):
            for name in ("task", "embodiment"):
                receipt[name] = job.get(name, receipt[name])
        publish(receipt, RESULTS)
        ledger["charged_seconds"] = charged + timeout
        ledger["active_parent_pid"] = os.getpid()
        ledger["active_gpu_uuid"] = binding["uuid"]
        write_json(ledger_path, ledger)
        started = time.monotonic()
        try:
            execution_options = {"gpu_uuid": binding["uuid"]}
            if policy_identity:
                execution_options["cleanup_profile"] = "r1lite-policy-path-identity"
            if r1_diagnostic:
                execution_options["cleanup_profile"] = "r1lite-native-diagnostic"
            if r1_training:
                execution_options["cleanup_profile"] = "r1lite-original-training"
            if r1_qualification:
                execution_options["cleanup_profile"] = "r1lite-original-qualification"
            if r1_eval:
                execution_options["cleanup_profile"] = "r1lite-frozen-eval"
            code = execute_command(
                command, out / "run.log", timeout, **execution_options
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
