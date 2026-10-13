"""Canonical paired TRAIN reach diagnostic; no training or VAL admission changes."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import signal
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from abc_bench import fasttd3_reach_stage as reach
from abc_bench import fasttd3_vector_stage as qualification
from abc_bench.fasttd3_host import rpc_call, write_json
from abc_bench.fasttd3_reach_matched import HORIZON, PROFILE, prepare_binding
from abc_bench.fasttd3_reach_matched import load_config as load_host
from abc_bench.fasttd3_scratch_stage import pin
from abc_bench.subset import require_subset_lease
from abc_bench.upstream_eval import artifact

ROOT, HOST_PYTHON, LEARNER_PYTHON = reach.ROOT, reach.HOST_PYTHON, reach.LEARNER_PYTHON
ABC_PAYLOADS = reach.ABC_PAYLOADS | {
    "abc_bench/fasttd3_reach_matched.py",
    "abc_bench/fasttd3_reach_matched_stage.py",
    "abc_bench/captured_reset.py",
    "abc_bench/upstream_report.py",
}
FAST_PAYLOADS = reach.FAST_PAYLOADS | {"sadhana_fasttd3_abc/reach_matched.py"}
LIMITS = {"ready_seconds": 180, "learner_seconds": 850, "shutdown_seconds": 30}


def validate_sources(value: Any) -> dict:
    if type(value) is not dict or set(value) != {"abc", "fast_adapter"}:
        raise ValueError("Pin the matched ABC and Fast installed boundaries")
    for key, expected in (("abc", ABC_PAYLOADS), ("fast_adapter", FAST_PAYLOADS)):
        rows = value[key]
        if (
            type(rows) is not list
            or len(rows) != len(expected)
            or {row.get("path") for row in rows} != expected
            or any(
                set(row) != {"path", "bytes", "sha256"}
                or type(row["bytes"]) is not int
                or row["bytes"] <= 0
                or type(row["sha256"]) is not str
                or len(row["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in row["sha256"])
                for row in rows
            )
        ):
            raise ValueError("Incomplete/invalid matched source inventory: " + key)
    return value


def load_config(path: Path) -> tuple[dict, dict, dict]:
    if path.is_symlink() or not 0 < path.stat().st_size <= 1024**2:
        raise ValueError("Matched stage config must be regular JSON at most1MiB")
    config = json.loads(path.read_text())
    if (
        type(config) is not dict
        or set(config)
        != {
            "schema_version",
            "host_python",
            "learner_python",
            "binding",
            "resource_parent",
            "resource_worker",
            "source_pins",
            "limits",
        }
        or type(config["schema_version"]) is not int
        or config["schema_version"] != 1
    ):
        raise ValueError("Use the complete separate matched stage schema1")
    if (
        config["host_python"] != str(HOST_PYTHON)
        or config["learner_python"] != str(LEARNER_PYTHON)
        or config["limits"] != LIMITS
    ):
        raise ValueError(
            "Use the normal installed runtimes and bounded paired-stage watchdogs"
        )
    validate_sources(config["source_pins"])
    binding = json.loads(pin(config["binding"]).read_text())
    if binding.get("schema") != "sadhana.fasttd3.reach-matched-binding/v1":
        raise ValueError(
            "Matched stage must bind one actual complete reach TRAIN producer"
        )
    host, _ = load_host(pin(binding["host_config"]))
    # Repeat actual producer admission rather than trusting the prep file alone.
    from abc_bench.fasttd3_reach_matched import prepare_binding as bind

    # prepare_binding can validate without writing when output is None.
    actual = bind(
        Path(binding["training_parent"]["path"]),
        Path(binding["training_worker"]["path"]),
        Path(binding["host_config"]["path"]),
        None,
    )
    if actual != binding:
        raise ValueError("Matched binding differs from the actual complete producer")
    evidence = reach.resource_evidence(
        config["resource_parent"], config["resource_worker"]
    )
    return config, host, evidence


def wait_ready(process: Any, path: Path, seconds: int) -> dict:
    deadline = time.monotonic() + seconds
    while process.poll() is None and time.monotonic() < deadline:
        if path.exists():
            try:
                description = rpc_call(
                    path, "describe", max(0.001, deadline - time.monotonic())
                )
                if (
                    description.get("profile") != PROFILE
                    or description.get("num_envs") != 1
                    or description.get("max_episode_steps") != HORIZON
                ):
                    raise ValueError("Matched native host description differs")
                return description
            except (FileNotFoundError, ConnectionRefusedError):
                pass
        time.sleep(0.1)
    raise RuntimeError("Matched host failed its bounded readiness; see host.log")


def verify_result(config: dict, output: Path) -> dict:
    host = json.loads((output / "native_host/receipt.json").read_text())
    learner = json.loads((output / "learner/receipt.json").read_text())
    comparison = json.loads((output / "learner/comparison.json").read_text())
    verification = json.loads((output / "comparison-check.json").read_text())
    binding = json.loads(pin(config["binding"]).read_text())
    if (
        host.get("status") != "completed"
        or host.get("controls") != 6 * HORIZON
        or host.get("resets") != 6
        or host.get("config") != binding["host_config"]
        or learner.get("status") != "completed"
        or learner.get("binding") != config["binding"]
        or learner.get("comparison") != artifact(output / "learner/comparison.json")
        or verification.get("comparison") != learner["comparison"]
        or verification.get("status") != "completed"
        or verification.get("recomputed_from_frozen_receipts") is not True
        or comparison.get("updates_during_evaluation") != {"actor": 0, "critic": 0}
        or comparison.get("binding") != config["binding"]
    ):
        raise RuntimeError(
            "Matched native/consumer/verification completion binding differs"
        )
    native_rows = host["episodes"]
    policy_rows = [
        row
        for role in ("initial", "final")
        for row in json.loads((output / "learner" / role / "receipt.json").read_text())[
            "episodes"
        ]
    ]
    if len(native_rows) != 6 or any(
        {key: value for key, value in row.items() if key != "return"} != native
        for row, native in zip(policy_rows, native_rows)
    ):
        raise RuntimeError("Actual native and frozen policy episode records differ")
    return {
        "metrics": {
            "matched_TRAIN_cases": 3,
            "frozen_episodes": 6,
            "native_controls": 6 * HORIZON,
            "actor_updates": 0,
            "critic_updates": 0,
        },
        "native_receipt": artifact(output / "native_host/receipt.json"),
        "learner_receipt": artifact(output / "learner/receipt.json"),
        "comparison": learner["comparison"],
        "comparison_check": artifact(output / "comparison-check.json"),
    }


def run(config_path: Path, output: Path) -> dict:
    require_subset_lease()  # Before filesystem mutation, native import or process launch.
    config, _host_config, evidence = load_config(config_path)
    distribution = importlib.metadata.distribution("abc-minimal")
    for row in config["source_pins"]["abc"]:
        actual = artifact(Path(distribution.locate_file(row["path"])))
        if any(actual[key] != row[key] for key in ("bytes", "sha256")):
            raise RuntimeError("Installed matched ABC payload differs: " + row["path"])
    output.mkdir(parents=True, exist_ok=True)
    if any(
        path.is_symlink()
        or not path.is_file()
        or path.name not in {"job.json", "run.log"}
        for path in output.iterdir()
    ):
        raise FileExistsError(
            "Matched output may contain only fresh canonical bookkeeping"
        )
    receipt = {
        "schema_version": 1,
        "phase": "fasttd3_reach_matched_TRAIN_initial_final",
        "status": "running",
        "input_config": artifact(config_path),
        "binding": config["binding"],
        "source_pins": config["source_pins"],
        "process_group": os.getpgrp(),
        "metrics": {},
        "claim_scope": "Three matched TRAIN captured reach cases; true initial/final actor+normalizer; no five-bottle benchmark, VAL, population or paper claim",
    }
    host = learner = None
    socket_dir = socket_path = None
    previous = {}
    started = time.monotonic()

    def cancel(signum: int, _frame: Any) -> None:
        raise InterruptedError("Matched stage received signal " + str(signum))

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, cancel)
        subprocess.run(
            [
                config["learner_python"],
                "-I",
                "-B",
                "-c",
                qualification.PREFLIGHT,
                str(output / "learner-preflight.json"),
                json.dumps(config["source_pins"]["fast_adapter"]),
            ],
            env=qualification.child_environment(cpu=True),
            check=True,
            timeout=30,
        )
        receipt["learner_preflight"] = artifact(output / "learner-preflight.json")
        receipt["memory_admission"] = qualification.memory_admission(evidence, count=64)
        receipt["effective_parallelism"] = {
            "native_worlds": 1,
            "policy_episodes": 6,
            "memory_bound_worlds": 64,
            "memory_scope": "Unchanged public64-world conservative admission reused for this one-world consumer; not measured one-world peak",
        }
        socket_dir = ROOT / "builds" / ("abcrmatch-" + uuid.uuid4().hex[:12])
        socket_dir.mkdir(mode=0o700)
        socket_path = socket_dir / "env.sock"
        binding = json.loads(pin(config["binding"]).read_text())
        with (
            (output / "host.log").open("w") as host_log,
            (output / "learner.log").open("w") as learner_log,
        ):
            host = subprocess.Popen(
                [
                    config["host_python"],
                    "-I",
                    "-u",
                    "-B",
                    "-m",
                    "abc_bench.fasttd3_reach_matched",
                    "--config",
                    binding["host_config"]["path"],
                    "--socket",
                    str(socket_path),
                    "--output",
                    str(output / "native_host"),
                ],
                env=qualification.child_environment(cpu=False),
                stdout=host_log,
                stderr=subprocess.STDOUT,
            )
            if os.getpgid(host.pid) != os.getpgrp():
                raise RuntimeError(
                    "Matched host must inherit the canonical process group"
                )
            receipt["native_environment"] = wait_ready(
                host, socket_path, LIMITS["ready_seconds"]
            )
            learner = subprocess.Popen(
                [
                    config["learner_python"],
                    "-I",
                    "-u",
                    "-B",
                    "-m",
                    "sadhana_fasttd3_abc.reach_matched",
                    "--binding",
                    config["binding"]["path"],
                    "--socket",
                    str(socket_path),
                    "--output",
                    str(output / "learner"),
                    "--device",
                    "cuda",
                ],
                env=qualification.child_environment(cpu=False),
                stdout=learner_log,
                stderr=subprocess.STDOUT,
            )
            if os.getpgid(learner.pid) != os.getpgrp():
                raise RuntimeError(
                    "Matched consumer must inherit the canonical process group"
                )
            if learner.wait(timeout=LIMITS["learner_seconds"]) != 0:
                raise RuntimeError("Matched frozen consumer failed; see learner.log")
            rpc_call(socket_path, "shutdown", LIMITS["shutdown_seconds"])
            if host.wait(timeout=LIMITS["shutdown_seconds"]) != 0:
                raise RuntimeError("Matched native host failed cleanup")
        subprocess.run(
            [
                config["learner_python"],
                "-I",
                "-B",
                "-m",
                "sadhana_fasttd3_abc.reach_matched",
                "--binding",
                config["binding"]["path"],
                "--verify-output",
                str(output / "learner"),
                "--output",
                str(output / "comparison-check.json"),
            ],
            env=qualification.child_environment(cpu=True),
            check=True,
            timeout=30,
        )
        receipt.update(status="completed", **verify_result(config, output))
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        cleanup_errors = []
        for process in (learner, host):
            try:
                qualification.stop_process(process, LIMITS["shutdown_seconds"])
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                cleanup_errors.append(str(error))
        try:
            if socket_path is not None:
                socket_path.unlink(missing_ok=True)
            if socket_dir is not None:
                socket_dir.rmdir()
        except OSError as error:
            cleanup_errors.append(str(error))
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        receipt.update(
            seconds=time.monotonic() - started,
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


def prepare_job(
    parent: Path, worker: Path, host: Path, sources: dict, output: Path
) -> dict:
    validate_sources(sources)
    binding = prepare_binding(parent, worker, host, None)
    producer_config = json.loads(
        pin(
            json.loads(pin(binding["training_worker"]).read_text())["input_config"]
        ).read_text()
    )
    reach.resource_evidence(
        producer_config["resource_parent"], producer_config["resource_worker"]
    )
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "binding.json", binding)
    config = {
        "schema_version": 1,
        "host_python": str(HOST_PYTHON),
        "learner_python": str(LEARNER_PYTHON),
        "binding": artifact(output / "binding.json"),
        "resource_parent": producer_config["resource_parent"],
        "resource_worker": producer_config["resource_worker"],
        "source_pins": sources,
        "limits": LIMITS,
    }
    write_json(output / "config.json", config)
    job = {
        "schema_version": 1,
        "module": "abc_bench.fasttd3_reach_matched_stage",
        "python": str(HOST_PYTHON),
        "arguments": ["--config", str((output / "config.json").resolve())],
        "label": "Original FastTD3 matched initial/final TRAIN reach diagnostic",
        "claim_scope": "Three exact TRAIN captured resets; separate reach and native5/5; no full-task benchmark or held-out/paper claim",
        "max_wall_s": 1260,
    }
    write_json(output / "job.json", job)
    return job


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-job", action="store_true")
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--host-config", type=Path)
    parser.add_argument("--source-pins", type=Path)
    args = parser.parse_args()
    if args.prepare_job:
        if args.config is not None or any(
            value is None
            for value in (args.parent, args.worker, args.host_config, args.source_pins)
        ):
            parser.error(
                "Job preparation needs --parent/--worker/--host-config/--source-pins"
            )
        prepare_job(
            args.parent,
            args.worker,
            args.host_config,
            json.loads(args.source_pins.read_text()),
            args.output,
        )
    else:
        if args.config is None or any(
            value is not None
            for value in (args.parent, args.worker, args.host_config, args.source_pins)
        ):
            parser.error("Canonical execution needs --config only")
        run(args.config, args.output)


if __name__ == "__main__":
    main()
