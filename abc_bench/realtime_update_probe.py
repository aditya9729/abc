"""One original full EXPO update on recorded TRAIN data; no online benchmark."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

SCHEMA = "sadhana.real_time_expo_capacity_probe/1"
MODULE = "abc_bench.realtime_update_probe"
WORKER_SECONDS = 2300
PARENT_SECONDS = 2400
GIB = 2**30
MIN_START_BYTES = 160 * GIB
MIN_RUN_BYTES = 48 * GIB
ONLINE_SHA = "ac522ad3fb4adbece7033a87017bdffb1037d88885fe6337c26949d3a672289e"
MEMORY_ENV = {
    "XLA_PYTHON_CLIENT_ALLOCATOR": "bfc",
    "TF_FORCE_UNIFIED_MEMORY": "true",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "XLA_PYTHON_CLIENT_MEM_FRACTION": "1.5",
}
JAX_ENV = {
    "JAX_PLATFORMS": "cuda",
    "JAX_PLATFORM_NAME": "cuda",
    "JAX_DEFAULT_PRNG_IMPL": "threefry2x32",
    "JAX_THREEFRY_PARTITIONABLE": "true",
    "JAX_THREEFRY_GPU_KERNEL_LOWERING": "false",
    "JAX_ENABLE_X64": "false",
}
COUNTERS = {
    "actor_train_state": 1,
    "edit_actor": 1,
    "temp": 1,
    "critic": 20,
    "filter_critic": 20,
    "batch_encoder": 20,
}


def file_pin(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def authenticate(pin: dict[str, Any]) -> Path:
    path = Path(pin["path"])
    if not path.is_absolute() or not path.is_file():
        raise ValueError("Input pin requires an existing absolute file")
    actual = file_pin(path)
    if actual["sha256"] != pin["sha256"] or actual["bytes"] != pin["bytes"]:
        raise ValueError(f"Input identity differs: {path}")
    return path


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def host_available(path: Path = Path("/proc/meminfo")) -> int:
    rows = [line.split() for line in path.read_text().splitlines()]
    values = [
        int(row[1]) * 1024
        for row in rows
        if row[0] == "MemAvailable:" and row[2:] == ["kB"]
    ]
    if len(values) != 1 or values[0] <= 0:
        raise ValueError("Cannot establish positive host MemAvailable")
    return values[0]


def fixed_phase_module(config):
    """Closed dispatch; no phase or module name can execute arbitrary code."""
    if config.get("phase") == "fixed_data_10":
        from abc_bench import fixed_data_learning

        return fixed_data_learning
    if config.get("phase") == "recovery_fixed_data_10":
        from abc_bench import recovery_fixed_data_learning

        return recovery_fixed_data_learning
    if config.get("phase") == "historical_online_fixed_data_10":
        from abc_bench import historical_fixed_data_learning

        return historical_fixed_data_learning
    raise ValueError("Unknown fixed-data phase")


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_absolute():
        raise ValueError("Probe config must be absolute")
    config = json.loads(path.read_text())
    required_fields = {
        "schema",
        "online_config",
        "source_manifest",
        "source_root",
        "runtime_inputs",
        "initialization_acceptance",
        "installed_modules",
        "memory_environment",
        "worker_seconds",
        "parent_seconds",
        "minimum_start_bytes",
        "minimum_run_bytes",
    }
    if set(config) not in (required_fields, required_fields | {"phase", "fixed_data"}):
        raise ValueError("Unexpected capacity-probe fields")
    if "phase" in config:
        fixed_phase_module(config).load_spec(config)
    exact = {
        "schema": SCHEMA,
        "memory_environment": MEMORY_ENV,
        "worker_seconds": WORKER_SECONDS,
        "parent_seconds": PARENT_SECONDS,
        "minimum_start_bytes": MIN_START_BYTES,
        "minimum_run_bytes": MIN_RUN_BYTES,
    }
    if any(type(config[k]) is not type(v) or config[k] != v for k, v in exact.items()):
        raise ValueError("Use the fixed one-update BFC unified-memory qualification")
    if config["online_config"]["sha256"] != ONLINE_SHA:
        raise ValueError("Probe must derive the sealed original online recipe")
    if (
        not isinstance(config["installed_modules"], dict)
        or MODULE not in config["installed_modules"]
    ):
        raise ValueError(
            "Normally installed diagnostic and author modules must be pinned"
        )
    for name in (
        "online_config",
        "source_manifest",
        "runtime_inputs",
        "initialization_acceptance",
    ):
        authenticate(config[name])
    if not Path(config["source_root"]).is_absolute():
        raise ValueError("Original source root must be absolute")
    return config


def validate_job(job: dict[str, Any], effective_seconds: float | None = None) -> None:
    if (
        job.get("module") != MODULE
        or type(job.get("max_wall_s")) is not int
        or job["max_wall_s"] != PARENT_SECONDS
        or job.get("task") != "r1lite_remote_into_organizer"
        or job.get("embodiment") != "r1lite_2025"
        or len(job.get("arguments", [])) != 2
        or job["arguments"][0] != "--config"
    ):
        raise ValueError(
            "Require the explicit R1 one-update diagnostic and 2400s parent"
        )
    config = load_config(Path(job["arguments"][1]))
    online = json.loads(Path(config["online_config"]["path"]).read_text())
    expected_python = (
        config["fixed_data"]["python"]["path"]
        if "phase" in config
        else online["python"]
    )
    if job["python"] != expected_python:
        raise ValueError("Probe must use the pinned original server interpreter")
    if effective_seconds is not None and (
        type(effective_seconds) not in (int, float)
        or not math.isfinite(effective_seconds)
        or effective_seconds < PARENT_SECONDS
    ):
        raise ValueError("Effective parent deadline must preserve 100s cleanup")
    if effective_seconds is not None and host_available() < MIN_START_BYTES:
        raise RuntimeError("Host MemAvailable is below 160 GiB at admission")


def authenticate_inputs(config: dict[str, Any]) -> dict[str, Any]:
    """Verify bytes before importing JAX, including the original params item."""
    root = Path(config["source_root"])
    manifest = json.loads(authenticate(config["source_manifest"]).read_text())
    for pin in manifest["files"]:
        relative = Path(pin["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Original source manifest path escapes its root")
        authenticate({**pin, "path": str(root / relative)})
    runtime = json.loads(authenticate(config["runtime_inputs"]).read_text())
    for pin in runtime["files"]:
        authenticate(pin)
    for pin in config["installed_modules"].values():
        path = authenticate(pin).resolve()
        if not path.is_relative_to(Path(sys.prefix).resolve()):
            raise ValueError("Probe modules must be installed in the server prefix")
    accepted = json.loads(authenticate(config["initialization_acceptance"]).read_text())
    if accepted.get("status") != "read_only_evidence_accepted" or accepted.get(
        "evidence_errors"
    ):
        raise ValueError("Require accepted initialized actor-weight evidence")
    for pin in accepted["weight_loader_files_after"]:
        authenticate(pin)
    if "phase" in config:
        fixed_phase_module(config).authenticate_data(config)
    return {
        "source_files": len(manifest["files"]),
        "runtime_files": len(runtime["files"]),
        "installed_modules": len(config["installed_modules"]),
        "actor_parameter_files": len(accepted["weight_loader_files_after"]),
    }


def checked_import(name: str, config: dict[str, Any]):
    module = importlib.import_module(name)
    expected = config["installed_modules"][name]
    if Path(module.__file__).resolve() != Path(expected["path"]).resolve():
        raise ValueError(f"Module origin differs: {name}")
    authenticate(expected)
    return module


def memory_environment(online: dict[str, Any]) -> dict[str, str]:
    env = dict(os.environ)
    for key in tuple(env):
        if key.startswith(
            ("XLA_", "JAX_", "TF_GPU_", "TF_CUDA_", "TF_FORCE_UNIFIED_")
        ) or key in {
            "TF_FORCE_GPU_ALLOW_GROWTH",
            "CUDA_MANAGED_FORCE_DEVICE_ALLOC",
        }:
            env.pop(key)
    env.update(online["environment"])
    env.update(MEMORY_ENV)
    env.update(JAX_ENV)
    env["WANDB_MODE"] = "offline"
    return env


def watch_child(
    command: list[str],
    env: dict[str, str],
    output: Path,
    deadline: float,
    cancelled: list[bool],
) -> dict[str, Any]:
    """Retain one direct child; never signal an unowned recorded PID."""
    start_available = host_available()
    if start_available < MIN_START_BYTES:
        raise RuntimeError("Host MemAvailable is below 160 GiB; child not started")
    record: dict[str, Any] = {
        "minimum_host_available_bytes": start_available,
        "command": command,
    }
    process = None
    reason = None
    try:
        with (output / "probe.log").open("x") as log:
            process = subprocess.Popen(
                command, env=env, stdout=log, stderr=subprocess.STDOUT
            )
            record["child_pid"] = process.pid
            stat = Path(f"/proc/{process.pid}/stat").read_text()
            record["child_start_ticks"] = int(stat[stat.rfind(")") + 2 :].split()[19])
            record["boot_id"] = (
                Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            )
            write_json(output / "child.json", record)
            while process.poll() is None:
                available = host_available()
                record["minimum_host_available_bytes"] = min(
                    record["minimum_host_available_bytes"], available
                )
                if cancelled[0]:
                    reason = "cancelled"
                elif time.monotonic() >= deadline:
                    reason = "worker_deadline"
                elif available < MIN_RUN_BYTES:
                    reason = "host_available_below_48_GiB"
                if reason:
                    break
                time.sleep(1)
    finally:
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            record["child_exit_code"] = process.returncode
            record["child_reaped"] = process.returncode is not None
            record["stop_reason"] = reason
            write_json(output / "child.json", record)
    return record


def run_update(config: dict[str, Any], output: Path) -> dict[str, Any]:
    """Original construction, data, batch pipeline, and exactly one update."""
    run_started = time.monotonic()
    import jax
    import numpy as np

    original = checked_import("expo_ft.utils.train_utils", config)
    learner = checked_import("expo_ft.agents.alg.realtime_expo_ft", config)
    vla = checked_import("expo_ft.agents.vla.pi05", config)
    replay = checked_import("expo_ft.data.replay_buffer", config)
    batching = checked_import("expo_ft.data.batch_processor", config)
    sharding = checked_import("openpi.training.sharding", config)
    task_module = checked_import("sadhana_r1lite.expo_config", config)
    for name in config["installed_modules"]:
        if name.startswith("configs.model."):
            checked_import(name, config)
    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise RuntimeError("Require exactly the canonically leased GPU0")
    online = json.loads(Path(config["online_config"]["path"]).read_text())
    flags = dict(item.split("=", 1) for item in online["arguments"] if "=" in item)
    model = task_module.make_online_model_config()
    for key, raw in flags.items():
        if key.startswith("--config."):
            try:
                value = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                value = raw
            model[key.removeprefix("--config.")] = value
    model.delay = 5
    if model.model_cls != "RealTimeEXPOFTLearner":
        raise ValueError("Require the original full real-time learner")
    fixed_spec = None
    if "phase" in config:
        fixed_spec = fixed_phase_module(config).load_spec(config)
    task = (
        task_module.make_task_config(fixed_spec["settings"]["path"])
        if fixed_spec
        else task_module.make_task_config()
    )
    dataset = original.load_task_dataset(flags["--dataset_path"], task, num_data=21)
    if len(dataset) != 7069 or len(dataset.episodes) != 21:
        raise ValueError("Require the unchanged 21-episode TRAIN cohort")
    if fixed_spec:
        jax.config.update("jax_compilation_cache_dir", str(output / "jax-cache"))
    else:
        original.set_compilation_cache_dir("sync-RealTimeEXPOFTLearner")
    mesh = sharding.make_mesh(1)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS)
    )
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    actor, actor_state, target_actor, kwargs, metadata = vla.build_pi05(
        model, 42, mesh, data_sharding, replicated, False, task.language_instruction
    )
    example_action = dataset[0]["actions"][np.newaxis]
    rb_args = {
        "config": model,
        "example_action": example_action,
        "capacity": 500000,
        "task_description": task.language_instruction,
        "replan_steps": 8,
        "seed": 42,
        "delay": 5,
        "critic_camera_keys": tuple(task.critic_camera_keys),
    }
    buffer = replay.create_replay_buffer(**rb_args)
    offline = replay.create_replay_buffer(**rb_args)
    processor = batching.BatchProcessor(
        buffer, offline, data_sharding, 64, 20, 0, True, False, dataset=dataset
    )
    example = {
        replay._critic_key_to_storage(k): buffer.dataset_dict[
            replay._critic_key_to_storage(k)
        ][0][np.newaxis]
        for k in task.critic_camera_keys
    }
    example.update(
        state=buffer.dataset_dict["state"][0][np.newaxis],
        actions=buffer.dataset_dict["actions"][0][np.newaxis],
    )
    observation, state, action = buffer.convert_to_critic_format(example)
    actor.action_dim, actor.state_dim = (
        action.squeeze().shape[-1],
        state.squeeze().shape[-1],
    )
    kwargs.update(delay=5, critic_camera_keys=tuple(task.critic_camera_keys))
    agent = learner.load_agent(
        seed=42,
        example_observation=observation.squeeze(),
        example_action=action.squeeze(),
        example_state=state.squeeze(),
        actor=actor,
        actor_train_state=actor_state,
        target_actor_params=target_actor,
        agent_kwargs=kwargs,
        metadata=metadata,
        mesh=mesh,
        data_sharding=data_sharding,
        replicated_sharding=replicated,
        resume=False,
        replan_steps=8,
        default_prompt=task.language_instruction,
        edit_action_xyzg=task.edit_action_xyzg,
    )
    expected_shape = {
        "N": 32,
        "n_edit_samples": 32,
        "filter_N": 32,
        "filter_n_edit": 1,
        "action_horizon": 16,
        "replan_steps": 8,
        "delay": 5,
        "action_dim": 14,
        "state_dim": 32,
        "edit_scale": 0.1,
    }
    if any(getattr(agent, key) != value for key, value in expected_shape.items()):
        raise ValueError(
            "Original initialized learner shape or sampling recipe differs"
        )
    if actor.model_config.action_dim != 32 or not agent.train_base_actor:
        raise ValueError("Require padded action dimension32 and original actor updates")

    def finite_tree(tree) -> dict[str, int]:
        leaves, checked_bytes = 0, 0
        for value in jax.tree_util.tree_leaves(tree):
            host = np.asarray(jax.device_get(value))
            # ml_dtypes.bfloat16 uses dtype.kind='V'; np.isfinite supports it.
            if not np.isfinite(host).all():
                raise ValueError("Nonfinite parameter state")
            leaves += 1
            checked_bytes += host.nbytes
        if not leaves:
            raise ValueError("Expected nonempty parameter state")
        return {"finite_leaves": leaves, "finite_bytes": checked_bytes}

    def parameters_state(state) -> dict[str, Any]:
        result = {name: finite_tree(getattr(state, name).params) for name in COUNTERS}
        result["target_critic"] = finite_tree(state.target_critic.params)
        result["target_actor_params"] = finite_tree(state.target_actor_params)
        return result

    def optimizer_state(state) -> dict[str, Any]:
        result = {}
        for name in COUNTERS:
            branch = getattr(state, name)
            step = np.asarray(jax.device_get(branch.step))
            if step.shape or step.dtype.kind not in "iu":
                raise ValueError(f"Invalid integer optimizer step: {name}")
            counts, leaves, checked_bytes = {}, 0, 0
            for path, value in jax.tree_util.tree_flatten_with_path(branch.opt_state)[
                0
            ]:
                host = np.asarray(jax.device_get(value))
                if not np.isfinite(host).all():
                    raise ValueError(f"Nonfinite optimizer state: {name}")
                leaves += 1
                checked_bytes += host.nbytes
                if any(getattr(key, "name", None) == "count" for key in path):
                    if host.shape or host.dtype.kind not in "iu":
                        raise ValueError("Optimizer count must be an integer scalar")
                    counts[jax.tree_util.keystr(path)] = int(host)
            result[name] = {
                "step": int(step),
                "counts": counts,
                "finite_leaves": leaves,
                "finite_bytes": checked_bytes,
            }
        return result

    before = optimizer_state(agent)
    if any(
        v["step"] != 0 or not v["counts"] or any(v["counts"].values())
        for v in before.values()
    ):
        raise ValueError("Diagnostic requires fresh optimizer states")
    before_parameters = parameters_state(agent)
    write_json(
        output / "before.json", {"optimizers": before, "parameters": before_parameters}
    )
    if fixed_spec:
        fixed = checked_import(fixed_phase_module(config).MODULE, config)
        for name in (
            "sadhana_terminal_critic.contracts",
            "expo_ft.agents.alg.batch_utils",
        ):
            checked_import(name, config)
        agent_holder = [agent]
        del agent, actor_state, target_actor
        return fixed.run(
            agent_holder,
            processor,
            dataset,
            task,
            config,
            output,
            optimizer_state,
            parameters_state,
            time.monotonic() - run_started,
        )
    combine_rng = jax.random.PRNGKey(142)
    with ThreadPoolExecutor(max_workers=1) as prefetch:
        batch, actor_batch, combine_rng = prefetch.submit(
            processor.next_batch, combine_rng
        ).result()
        pending = prefetch.submit(processor.next_batch, combine_rng)
        # Hold the original one-ahead result, including its full device batches.
        pending_result = pending.result()
        jax.block_until_ready((batch, actor_batch, pending_result))
        batch_info = original.get_batch_info(batch)
        if batch["actions"].shape[0] != 1280 or actor_batch["actions"].shape[0] != 64:
            raise ValueError("Original B64/UTD20 batch shapes differ")
        agent = agent.replace(rng=jax.device_put(agent.rng, replicated))
        start = time.monotonic()
        agent, info = agent.update(agent, batch, 20, actor_batch)
        jax.block_until_ready((agent, info))
        elapsed = time.monotonic() - start
        after = optimizer_state(agent)
        after_parameters = parameters_state(agent)
        metrics = {
            key: float(np.asarray(jax.device_get(value))) for key, value in info.items()
        }
        if not metrics or any(not math.isfinite(v) for v in metrics.values()):
            raise ValueError("Update metrics are missing or nonfinite")
        if metrics.get("next_q_nan_ratio") != 0:
            raise ValueError("Original update observed nonfinite target Q values")
        for name, expected in COUNTERS.items():
            if after[name]["step"] != expected or any(
                v != expected for v in after[name]["counts"].values()
            ):
                raise ValueError(f"Original optimizer counters differ: {name}")
    result = {
        "status": "one_original_update_completed",
        "benchmark_accepted": False,
        "online_schedule_executed": False,
        "update_calls": 1,
        "utd_ratio": 20,
        "batch_size": 64,
        "update_seconds": elapsed,
        "before": before,
        "after": after,
        "parameters_before": before_parameters,
        "parameters_after": after_parameters,
        "learner_shape": expected_shape,
        "jax_environment": JAX_ENV,
        "jax_config": {
            "default_prng_impl": jax.config.jax_default_prng_impl,
            "threefry_partitionable": jax.config.jax_threefry_partitionable,
            "threefry_gpu_kernel_lowering": jax.config.jax_threefry_gpu_kernel_lowering,
            "enable_x64": jax.config.jax_enable_x64,
        },
        "finite_metrics": metrics,
        "batch_info": batch_info,
        "gpu_memory_stats_after": jax.devices()[0].memory_stats(),
        "scope": "Demo-seeded capacity diagnostic; no native rollout, trained checkpoint or improvement claim",
    }
    write_json(output / "update.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    from abc_bench.subset import require_subset_lease

    require_subset_lease()
    started = time.monotonic()
    config = load_config(args.config)
    if args.execute:
        if any(os.environ.get(k) != v for k, v in (MEMORY_ENV | JAX_ENV).items()):
            raise ValueError("Child allocator or JAX environment differs")
        if os.environ.get("XLA_CLIENT_MEM_FRACTION"):
            raise ValueError("Conflicting memory fraction alias")
        authenticate_inputs(config)
        run_update(config, args.output)
        return
    if not args.output.is_absolute() or not args.output.is_dir():
        raise ValueError("Canonical coordinator must create the output directory")
    job = json.loads((args.output / "job.json").read_text())
    validate_job(job, PARENT_SECONDS)
    if (
        job["arguments"][1] != str(args.config)
        or (args.output / "receipt.json").exists()
    ):
        raise ValueError("Canonical job or fresh worker receipt differs")
    detail_output = args.output / "probe-artifacts"
    detail_output.mkdir()
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "failed",
        "metrics": {},
        "benchmark_accepted": False,
        "config": file_pin(args.config),
    }
    cancelled = [False]

    def cancel(_signum, _frame):
        cancelled[0] = True

    old_handlers = {
        sig: signal.signal(sig, cancel) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    code = 1
    try:
        online = json.loads(Path(config["online_config"]["path"]).read_text())
        expected_python = (
            config["fixed_data"]["python"]["path"]
            if "phase" in config
            else online["python"]
        )
        if Path(sys.executable).resolve() != Path(expected_python).resolve():
            raise ValueError(
                "Probe supervisor must use the original server interpreter"
            )
        command = [
            sys.executable,
            "-I",
            "-u",
            "-B",
            "-m",
            MODULE,
            "--config",
            str(args.config),
            "--output",
            str(detail_output),
            "--execute",
        ]
        lifecycle = watch_child(
            command,
            memory_environment(online),
            detail_output,
            started + WORKER_SECONDS,
            cancelled,
        )
        receipt["lifecycle"] = lifecycle
        if lifecycle["stop_reason"] or lifecycle["child_exit_code"] != 0:
            raise RuntimeError(f"Capacity child did not complete: {lifecycle}")
        detail = json.loads((detail_output / "update.json").read_text())
        expected_status, expected_calls = "one_original_update_completed", 1
        if "phase" in config:
            fixed = fixed_phase_module(config)
            expected_status, expected_calls = fixed.STATUS, fixed.CALLS
        if (
            detail["status"] != expected_status
            or detail["update_calls"] != expected_calls
        ):
            raise ValueError("Missing complete original update evidence")
        receipt.update(
            status="completed",
            update=file_pin(detail_output / "update.json"),
            metrics={
                "original_update_calls": expected_calls,
                "update_seconds": detail["update_seconds"],
            },
        )
        code = 0
    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
        KeyboardInterrupt,
    ) as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        receipt["elapsed_s"] = time.monotonic() - started
        write_json(args.output / "receipt.json", receipt)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
