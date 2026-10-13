"""Separate canonical CPU resource probe; never a trainer or qualification.

The parent imports no simulator. Each case executes in a fresh owned process.
Native imports occur only after CPU/owner checks. No learner is constructed.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import resource
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from abc_bench.fasttd3_host import write_json
from abc_bench.fasttd3_scratch_stage import pin
from abc_bench.subset import require_subset_lease
from abc_bench.upstream_eval import artifact

MODULE = "abc_bench.fasttd3_compiler_cache_diagnostic"
SCHEMA = "sadhana.fasttd3.compiler-cache-resource-diagnostic/v1"
RANKS, MODES = [0, 55], ["default", "disabled"]
RESETS, CONTROLS = 13, 2
LIMITS = {"case_seconds": 900, "shutdown_seconds": 30, "parent_seconds": 3900}
PAYLOADS = {
    "abc_bench/fasttd3_compiler_cache_diagnostic.py",
    "abc_bench/subset.py",
    "abc_bench/fasttd3_vector_host.py",
    "abc_bench/fasttd3_campaign_host.py",
    "abc_bench/bottle_success_reward.py",
    "abc_bench/fasttd3_campaign_contract.py",
    "abc_bench/fasttd3_campaign_stage.py",
    "abc_bench/fasttd3_vector_stage.py",
    "abc_bench/fasttd3_host.py",
    "abc_bench/fasttd3_scratch_stage.py",
    "abc_bench/upstream_eval.py",
    "abc_sim/__init__.py",
    "abc_sim/env.py",
    "abc_sim/task_eval/__init__.py",
    "abc_sim/task_eval/base.py",
    "abc_sim/task_eval/bottles.py",
    "abc_sim/task_eval/registry.py",
    "abc_sim/task_specs.py",
    "abc_sim/randomization/tasks/bottles.py",
    "abc_sim/randomization/assets/water_bottles.py",
    "abc_sim/randomization/core.py",
}
CLAIM = (
    "Sequential full-task TRAIN rank0/rank55 default/disabled compiler-cache "
    "resource and exact-equivalence diagnostic only; not full64 qualification, "
    "training, policy performance, aggregate RAM or future admission"
)


def installed_sources() -> list[dict]:
    distribution = importlib.metadata.distribution("abc-minimal")
    return [
        {
            "path": name,
            **{
                k: v
                for k, v in artifact(Path(distribution.locate_file(name))).items()
                if k != "path"
            },
        }
        for name in sorted(PAYLOADS)
    ]


def load_config(path: Path) -> tuple[dict, dict, dict]:
    """Load actual published resource evidence; no native/model import."""
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not 0 < path.stat().st_size <= 1024**2
    ):
        raise ValueError("Use an absolute regular diagnostic config at most1MiB")
    value = json.loads(path.read_text())
    expected = {
        "schema",
        "python",
        "full_host",
        "resource_training_config",
        "resource_parent",
        "resource_worker",
        "ranks",
        "cache_modes",
        "resets_per_case",
        "controls_per_reset",
        "normalized_action14",
        "limits",
        "source_pins",
    }
    if type(value) is not dict or set(value) != expected or value["schema"] != SCHEMA:
        raise ValueError("Use the separate compiler-cache diagnostic schema")
    if value["python"] != str(Path(sys.executable).absolute()):
        raise ValueError("Use the explicitly selected normal installed interpreter")
    if (
        value["ranks"] != RANKS
        or any(type(x) is not int for x in value["ranks"])
        or value["cache_modes"] != MODES
        or type(value["resets_per_case"]) is not int
        or value["resets_per_case"] != RESETS
        or type(value["controls_per_reset"]) is not int
        or value["controls_per_reset"] != CONTROLS
        or value["normalized_action14"] != [0.0] * 14
        or any(type(x) is not float for x in value["normalized_action14"])
        or value["limits"] != LIMITS
        or any(type(x) is not int for x in value["limits"].values())
        or value["source_pins"] != installed_sources()
    ):
        raise ValueError(
            "Diagnostic cases, actions, watchdogs or installed source differ"
        )
    from abc_bench import fasttd3_campaign_stage as campaign

    host = campaign.load_host(pin(value["full_host"]))
    if host["phase"] != "runtime_gate" or host["num_envs"] != 64:
        raise ValueError("Keep the unchanged original full64 task/seed configuration")
    old = json.loads(pin(value["resource_training_config"]).read_text())
    old_host = campaign.load_host(pin(old["host_config"]))
    evidence = campaign.runtime_gate(
        value["resource_parent"], value["resource_worker"], old_host, old["source_pins"]
    )
    if len(evidence["worker_resources"]) != 64:
        raise ValueError("Require all64 authentic historical resource receipts")
    for row in evidence["worker_resources"]:
        if json.loads(pin(row["evidence"]).read_text()) != row["measurement"]:
            raise ValueError("Actual native close resource bytes differ")
    return value, host, evidence


def diagnostic_memory_guard(
    evidence: dict, meminfo: Path = Path("/proc/meminfo")
) -> dict:
    """Separate unvalidated case estimate; cannot admit a production stage."""
    peak = max(row["measurement"]["maxrss_kib"] for row in evidence["worker_resources"])
    fields = {
        line.split(":", 1)[0]: int(line.split()[1])
        for line in meminfo.read_text().splitlines()
    }
    required = math.ceil(RESETS * peak * 1.25) + 8 * 1024**2
    if fields["MemAvailable"] < required:
        raise MemoryError(
            f"One-child cache diagnostic estimates {required} KiB; available {fields['MemAvailable']} KiB"
        )
    return {
        "available_kib": fields["MemAvailable"],
        "required_kib": required,
        "historical_worker_lifetime_peak_kib": peak,
        "scope": "UNVALIDATED diagnostic estimate:13*historical max*1.25+8GiB; one live child; not observed aggregate or full64 admission",
        "production_memory_admission_changed": False,
    }


def process_start_ticks(pid: int) -> int:
    line = Path(f"/proc/{pid}/stat").read_text()
    return int(line[line.rfind(")") + 2 :].split()[19])


def require_case_owner(config_path: Path) -> None:
    """Only the immediate live canonical stage may start this internal worker."""
    parent = os.getppid()
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or os.environ.get("MUJOCO_GL") != "disable"
        or any(
            os.environ.get(n) != "1"
            for n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        )
        or os.environ.get("ABC_CACHE_PARENT_PID") != str(parent)
        or os.environ.get("ABC_CACHE_PARENT_START") != str(process_start_ticks(parent))
        or os.environ.get("ABC_CACHE_PROCESS_GROUP") != str(os.getpgrp())
        or os.getpgid(parent) != os.getpgrp()
        or os.environ.get("ABC_CACHE_CONFIG_SHA256") != artifact(config_path)["sha256"]
        or MODULE.encode()
        not in Path(f"/proc/{parent}/cmdline").read_bytes().split(b"\0")
    ):
        raise RuntimeError(
            "Case requires its live owned CPU-only canonical stage parent"
        )


def set_cache_mode(api: Any, mode: str) -> dict:
    """Only call before constructing a model in the fresh child."""
    if mode not in MODES:
        raise ValueError("Unknown compiler-cache mode")
    cache = api.mj_getCache()
    before = int(api.mj_getCacheCapacity(cache))
    occupied = int(api.mj_getCacheSize(cache))
    if before != 500 * 1024**2 or occupied != 0:
        raise RuntimeError("Fresh MuJoCo3.8.1 cache is not empty at default capacity")
    if mode == "disabled" and int(api.mj_setCacheCapacity(cache, 0)) != 0:
        raise RuntimeError("Public compiler-cache disable did not return0")
    return {
        "cache": cache,
        "initial_capacity_bytes": before,
        "initial_occupied_bytes": occupied,
    }


def install_step_observer(world: Any, observe: Callable[[Any], None]) -> None:
    """Observe exactly one public call, before World.step can autoreset.

    Do not catch an original step error, mutate its result or call it again.
    """
    original = world.env.step

    def step(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        observe(result)
        return result

    world.env.step = step


def valid_physics(snapshot: dict) -> None:
    if (
        not snapshot["time_finite"]
        or not all(snapshot["finite"].values())
        or set(snapshot["warnings"])
        != {
            "INERTIA",
            "CONTACTFULL",
            "CNSTRFULL",
            "BADQPOS",
            "BADQVEL",
            "BADQACC",
            "BADCTRL",
        }
        or any(row["number"] != 0 for row in snapshot["warnings"].values())
    ):
        raise FloatingPointError(
            "Diagnostic requires finite physics and all7 warning counts0"
        )


def resource_sample(api: Any, cache: Any, started: float, world: Any = None) -> dict:
    def field(path: str, name: str) -> int:
        lines = Path(path).read_text().splitlines()
        return next(
            int(line.split()[1]) for line in lines if line.startswith(name + ":")
        )

    sizes = {}
    if world is not None:
        for owner, value in (("model", world.env.model), ("data", world.env.data)):
            sizes[owner] = {
                name: int(getattr(value, name))
                for name in ("nbuffer", "narena")
                if hasattr(value, name)
            }
    return {
        "elapsed_seconds": time.monotonic() - started,
        "current_rss_kib": field("/proc/self/status", "VmRSS"),
        "current_pss_kib": field("/proc/self/smaps_rollup", "Pss"),
        "lifetime_peak_rss_kib": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ),
        "cache_capacity_bytes": int(api.mj_getCacheCapacity(cache)),
        "cache_occupied_bytes": int(api.mj_getCacheSize(cache)),
        "public_buffer_sizes_not_total_native_allocation": sizes,
    }


def run_case(config_path: Path, output: Path, rank: int, mode: str) -> dict:
    require_case_owner(
        config_path
    )  # Before native import, cache call or output mutation.
    config, host, _ = load_config(config_path)
    if rank not in RANKS or mode not in MODES:
        raise ValueError("Only the two declared historical TRAIN ranks/modes")
    output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema": SCHEMA,
        "kind": "native_case_resource_only",
        "status": "running",
        "rank": rank,
        "cache_mode": mode,
        "config": artifact(config_path),
        "pid": os.getpid(),
        "process_group": os.getpgrp(),
        "claim_scope": CLAIM,
        "owned_worlds_constructed": 0,
        "records": [],
        "resources": [],
        "training_updates": 0,
        "qualification": False,
        "benchmark_score": None,
    }
    world = cache = api = None
    started = time.monotonic()
    previous = {}

    def cancel(signum: int, _frame: Any) -> None:
        raise InterruptedError(f"Owned native cache case received signal{signum}")

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, cancel)
        import mujoco as api
        import numpy as np
        from abc_sim.task_eval import make_task_evaluator
        from abc_sim.task_specs import get_task_spec

        from abc_bench import fasttd3_campaign_host as native

        if api.__version__ != "3.8.1" or any(
            n in sys.modules for n in ("torch", "jax", "warp")
        ):
            raise RuntimeError(
                "Require original CPU MuJoCo3.8.1 without learner/GPU imports"
            )
        setup = set_cache_mode(api, mode)
        cache = setup.pop("cache")
        receipt["cache_setup_before_any_model"] = setup
        receipt["resources"].append(
            {"point": "before_model", **resource_sample(api, cache, started)}
        )
        world = native.RewardWorld(host, rank, output)
        receipt["owned_worlds_constructed"] = (
            1  # World internally compiles additional originals.
        )

        def capture(label: str, judge: dict, judge_source: str) -> None:
            before = world.integration().tobytes()
            state_path = output / f"{label}-integration.bin"
            state_path.write_bytes(before)
            model = world.env.model
            options = {
                name: native.native.plain(value)
                for name in dir(model.opt)
                if isinstance(
                    value := getattr(model.opt, name), (int, float, np.ndarray)
                )
            }
            for name in (
                "integrator",
                "solver",
                "cone",
                "jacobian",
                "disableflags",
                "enableflags",
            ):
                options[name] = int(getattr(model.opt, name))
            physics = native.physics_snapshot(world.env.data)
            row = {
                "label": label,
                "episode_index": world.episode_index,
                "requested_seed": world.requested_seed,
                "actual_seed": world.actual_seed,
                "reset_attempts": world.reset_attempts,
                "randomization": world.randomization,
                "model_arrays": native.native.native_model_digest(model),
                "model_options": options,
                "integration": artifact(state_path),
                "integration_dtype": "float64",
                "integration_state_spec": int(api.mjtState.mjSTATE_INTEGRATION),
                "state216": world.observation(),
                "native_ctrl": native.native.plain(world.env.data.ctrl),
                "native_judge": judge,
                "judge_source": judge_source,
                "physics": physics,
                "reset_scene_xml": artifact(
                    output / f"reset-{world.episode_index:04d}.xml"
                ),
            }
            receipt["records"].append(row)
            if world.integration().tobytes() != before:
                raise RuntimeError(
                    "Read-only state observation changed native integration"
                )
            valid_physics(physics)
            receipt["resources"].append(
                {"point": label, **resource_sample(api, cache, started, world)}
            )
            # Persist bounded running evidence before another expensive native reset.
            write_json(output / "receipt.json", receipt)

        observed = []

        def observe(result: Any) -> None:
            _, _, _, _, info = result
            label = (
                f"reset-{world.episode_index:04d}-control-{world.episode_steps + 1:02d}"
            )
            capture(
                label,
                native.native.plain(info["task_eval"]),
                "unchanged_public_env_step_returned_task_eval_before_autoreset",
            )
            observed.append(label)

        install_step_observer(world, observe)
        for index in range(RESETS):
            if index:
                world.reset()  # Original reset/fallback/carry semantics; manual lifecycle probe.
            if (
                world.episode_index != index
                or world.requested_seed != 200000 + rank + 128 * index
            ):
                raise RuntimeError("Original requested TRAIN reset sequence differs")
            evaluator = make_task_evaluator(
                world.env.model, get_task_spec(native.native.TASK)
            )
            initial_judge = native.native.plain(
                evaluator.evaluate_qpos_batch(
                    np.array(world.env.data.qpos, copy=True)[None]
                ).to_info(squeeze=True)
            )
            del evaluator  # The observer must not retain an extra old native model.
            capture(
                f"reset-{index:04d}-initial",
                initial_judge,
                "separate_public_original_evaluator_on_copied_qpos",
            )
            for _ in range(CONTROLS):
                before_count = len(observed)
                result = world.step(config["normalized_action14"])
                if len(observed) != before_count + 1:
                    raise RuntimeError(
                        "Public original step was not observed exactly once"
                    )
                receipt["records"][-1]["original_world_result"] = result
                if (
                    result["terminated"]
                    or result["truncated"]
                    or result["autoreset"] is not None
                ):
                    raise RuntimeError(
                        "Unexpected native terminal/autoreset; preserve actual post-control state and fail lifecycle comparison"
                    )
        receipt["status"] = "completed"
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            if world is not None:
                world.close()
            if api is not None and cache is not None:
                receipt["resources"].append(
                    {
                        "point": "after_close_before_exit",
                        **resource_sample(api, cache, started, world),
                    }
                )
        except BaseException as error:
            receipt.update(
                status="failed", cleanup_error=f"{type(error).__name__}: {error}"
            )
            raise
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
            receipt.update(
                seconds=time.monotonic() - started,
                cpu_visibility=os.environ.get("CUDA_VISIBLE_DEVICES"),
                renderer=os.environ.get("MUJOCO_GL"),
            )
            write_json(output / "receipt.json", receipt)
    return receipt


def compare_records(left_records: list[dict], right_records: list[dict]) -> list[dict]:
    """Pure byte/content comparison; this alone admits no native case or stage."""
    differences = []
    for left, right in zip(left_records, right_records, strict=True):
        for row in (left, right):
            valid_physics(row["physics"])
        ignored = {"integration", "reset_scene_xml"}
        if {k: v for k, v in left.items() if k not in ignored} != {
            k: v for k, v in right.items() if k not in ignored
        }:
            differences.append(
                {
                    "label": left["label"],
                    "component": "native_model_options_state_action_reset_reward_judge_physics",
                }
            )
        for key in ignored:
            if pin(left[key]).read_bytes() != pin(right[key]).read_bytes():
                differences.append({"label": left["label"], "component": key})
    return differences


def compare_cases(default: dict, disabled: dict) -> dict:
    """Only complete native resource cases reach the pure byte comparison."""
    for item, mode in ((default, "default"), (disabled, "disabled")):
        if (
            item.get("schema") != SCHEMA
            or item.get("kind") != "native_case_resource_only"
            or item.get("status") != "completed"
            or item.get("cache_mode") != mode
            or item.get("cpu_visibility") != ""
            or item.get("renderer") != "disable"
            or item.get("training_updates") != 0
            or item.get("qualification") is not False
            or len(item.get("records", [])) != RESETS * (CONTROLS + 1)
            or item.get("cleanup_error")
            or item.get("explicit_CPU_fixture")
        ):
            raise ValueError(
                "Require actual complete resource-only native case; partial/fixture is not equivalent"
            )
        samples = item.get("resources", [])
        capacity = 500 * 1024**2 if mode == "default" else 0
        if (
            len(samples) != 2 + RESETS * (CONTROLS + 1)
            or samples[0].get("point") != "before_model"
            or samples[-1].get("point") != "after_close_before_exit"
            or any(
                sample.get("cache_capacity_bytes") != capacity
                or type(sample.get("cache_occupied_bytes")) is not int
                or not 0 <= sample["cache_occupied_bytes"] <= capacity
                or any(
                    type(sample.get(k)) is not int or sample[k] < 0
                    for k in (
                        "current_rss_kib",
                        "current_pss_kib",
                        "lifetime_peak_rss_kib",
                    )
                )
                for sample in samples
            )
        ):
            raise ValueError("Actual per-case public cache/resource evidence differs")
        expected_labels = [
            f"reset-{index:04d}-{point}"
            for index in range(RESETS)
            for point in ("initial", "control-01", "control-02")
        ]
        if [row.get("label") for row in item["records"]] != expected_labels:
            raise ValueError(
                "Actual complete initial/post-control record order differs"
            )
        scene_keys = {
            (
                tuple(row["randomization"]["metadata"]["bottle_variants"]),
                tuple(
                    sorted(
                        (name, round(value, 8))
                        for name, value in row["randomization"]["scale_states"].items()
                    )
                ),
            )
            for row in item["records"]
            if row["label"].endswith("-initial")
        }
        if len(scene_keys) != RESETS:
            raise ValueError(
                "Require13 distinct original scene keys to cross the12-scene LRU lifecycle"
            )
    if (
        default["rank"] != disabled["rank"]
        or default["config"] != disabled["config"]
        or default["pid"] == disabled["pid"]
    ):
        raise ValueError("Paired rank/config or separate process identities differ")
    differences = compare_records(default["records"], disabled["records"])
    peaks = {
        item["cache_mode"]: max(
            row["lifetime_peak_rss_kib"] for row in item["resources"]
        )
        for item in (default, disabled)
    }
    return {
        "rank": default["rank"],
        "exact_equivalent": not differences,
        "records_compared": len(default["records"]),
        "differences": differences,
        "single_process_lifetime_peak_rss_kib": peaks,
        "disabled_minus_default_single_process_peak_kib": peaks["disabled"]
        - peaks["default"],
        "peak_scope": "Separate process lifetime peaks; not aggregate or future64 memory permission",
        "claim_scope": CLAIM,
        "full64_memory_sufficient": None,
    }


def run(config_path: Path, output: Path) -> dict:
    require_subset_lease()  # Before any output mutation or child/native/cache work.
    config, _, evidence = load_config(config_path)
    output.mkdir(parents=True, exist_ok=True)
    if any(
        p.is_symlink() or not p.is_file() or p.name not in {"job.json", "run.log"}
        for p in output.iterdir()
    ):
        raise FileExistsError("Use fresh canonical bookkeeping only")
    receipt = {
        "schema": SCHEMA,
        "kind": "resource_equivalence_only",
        "phase": "compiler_cache_resource_diagnostic",
        "status": "running",
        "input_config": artifact(config_path),
        "source_pins": config["source_pins"],
        "claim_scope": CLAIM,
        "qualification": False,
        "training_producer": False,
        "benchmark_score": None,
        "full64_memory_sufficient": None,
        "maximum_live_native_children": 1,
        "full_host_num_envs": 64,
        "selected_historical_TRAIN_ranks": RANKS,
        "cases": [],
        "comparisons": [],
        "metrics": {"training_updates": 0, "qualified_worlds": 0},
    }
    process = None
    previous = {}
    started = time.monotonic()

    def cancel(signum: int, _frame: Any) -> None:
        raise InterruptedError(f"Canonical cache diagnostic received signal{signum}")

    from abc_bench.fasttd3_vector_stage import child_environment, stop_process

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, cancel)
        for rank in RANKS:
            paired = {}
            for mode in MODES:
                receipt["memory_estimate_before_case"] = diagnostic_memory_guard(
                    evidence
                )
                seconds = min(
                    LIMITS["case_seconds"],
                    LIMITS["parent_seconds"]
                    - (time.monotonic() - started)
                    - LIMITS["shutdown_seconds"],
                )
                if seconds <= 0:
                    raise TimeoutError(
                        "Diagnostic parent deadline exhausted before next case"
                    )
                case_output = output / f"rank-{rank:03d}-{mode}"
                env = child_environment(cpu=True)
                env.update(
                    ABC_CACHE_PARENT_PID=str(os.getpid()),
                    ABC_CACHE_PARENT_START=str(process_start_ticks(os.getpid())),
                    ABC_CACHE_PROCESS_GROUP=str(os.getpgrp()),
                    ABC_CACHE_CONFIG_SHA256=artifact(config_path)["sha256"],
                )
                with (output / f"rank-{rank:03d}-{mode}.log").open("w") as log:
                    process = subprocess.Popen(
                        [
                            config["python"],
                            "-I",
                            "-u",
                            "-B",
                            "-m",
                            MODULE,
                            "--config",
                            str(config_path),
                            "--case-rank",
                            str(rank),
                            "--cache-mode",
                            mode,
                            "--output",
                            str(case_output),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=env,
                    )
                    case = {
                        "rank": rank,
                        "cache_mode": mode,
                        "pid": process.pid,
                        "start_ticks": process_start_ticks(process.pid),
                        "exit_code": None,
                    }
                    receipt["cases"].append(case)
                    if os.getpgid(process.pid) != os.getpgrp():
                        raise RuntimeError(
                            "Owned case must inherit the canonical process group"
                        )
                    case["exit_code"] = process.wait(timeout=seconds)
                    if (case_output / "receipt.json").is_file():
                        case["receipt"] = artifact(case_output / "receipt.json")
                    if case["exit_code"] != 0:
                        raise RuntimeError(
                            "Owned cache case failed; partial native evidence remains unaccepted"
                        )
                    paired[mode] = json.loads(pin(case["receipt"]).read_text())
                    if (
                        paired[mode].get("config") != receipt["input_config"]
                        or paired[mode].get("rank") != rank
                    ):
                        raise ValueError(
                            "Owned case did not bind this exact configuration/rank"
                        )
                    process = None  # This exact child was reaped before another spawn.
            comparison = compare_cases(paired["default"], paired["disabled"])
            receipt["comparisons"].append(comparison)
            if not comparison["exact_equivalent"]:
                raise RuntimeError(
                    "Cache modes differ in actual native bytes or task outputs"
                )
        receipt.update(
            status="completed",
            metrics={
                "training_updates": 0,
                "qualified_worlds": 0,
                "native_cases": 4,
                "native_controls": 4 * RESETS * CONTROLS,
            },
        )
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            stop_process(process, LIMITS["shutdown_seconds"])
        except BaseException as error:
            receipt.update(
                status="failed", cleanup_error=f"{type(error).__name__}: {error}"
            )
            raise
        finally:
            if process is not None and receipt["cases"]:
                receipt["cases"][-1]["exit_code"] = process.poll()
                receipt["cases"][-1]["owned_child_reaped"] = process.poll() is not None
                if (case_output / "receipt.json").is_file():
                    receipt["cases"][-1]["receipt"] = artifact(
                        case_output / "receipt.json"
                    )
            for signum, handler in previous.items():
                signal.signal(signum, handler)
            receipt["seconds"] = time.monotonic() - started
            write_json(output / "receipt.json", receipt)
    return receipt


def prepare_job(
    full_host: Path, resource_config: Path, parent: Path, worker: Path, output: Path
) -> dict:
    """Prepare metadata only; authentic old resources are not training permission."""
    config = {
        "schema": SCHEMA,
        "python": str(Path(sys.executable).absolute()),
        "full_host": artifact(full_host),
        "resource_training_config": artifact(resource_config),
        "resource_parent": artifact(parent),
        "resource_worker": artifact(worker),
        "ranks": RANKS,
        "cache_modes": MODES,
        "resets_per_case": RESETS,
        "controls_per_reset": CONTROLS,
        "normalized_action14": [0.0] * 14,
        "limits": LIMITS,
        "source_pins": installed_sources(),
    }
    output.mkdir(parents=True, exist_ok=False)
    path = output / "config.json"
    write_json(path, config)
    load_config(path)
    job = {
        "schema_version": 1,
        "python": config["python"],
        "module": MODULE,
        "arguments": ["--config", str(path)],
        "label": "Full-task compiler-cache CPU resource diagnostic ranks0/55",
        "max_wall_s": 4200,
        "claim_scope": CLAIM,
    }
    write_json(output / "job.json", job)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=CLAIM)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--case-rank", type=int, choices=RANKS)
    parser.add_argument("--cache-mode", choices=MODES)
    args = parser.parse_args()
    if (args.case_rank is None) != (args.cache_mode is None):
        parser.error("Both internal case arguments are required together")
    if args.case_rank is None:
        run(args.config, args.output)
    else:
        run_case(args.config, args.output, args.case_rank, args.cache_mode)


if __name__ == "__main__":
    main()
