"""Freeze an authenticated reward-v2 campaign actor on the declared VAL scenes."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import signal
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from abc_bench import fasttd3_frozen_stage as historical
from abc_bench import fasttd3_vector_stage as qualification
from abc_bench.fasttd3_campaign_stage import training_producer
from abc_bench.fasttd3_host import write_json
from abc_bench.fasttd3_scratch_stage import pin
from abc_bench.subset import require_subset_lease
from abc_bench.upstream_eval import artifact

ROOT, HOST_PYTHON, LEARNER_PYTHON = (
    historical.ROOT,
    historical.HOST_PYTHON,
    historical.LEARNER_PYTHON,
)
ABC_PAYLOADS = historical.ABC_PAYLOADS | {
    "abc_bench/fasttd3_campaign_stage.py",
    "abc_bench/fasttd3_campaign_host.py",
    "abc_bench/fasttd3_campaign_contract.py",
    "abc_bench/bottle_success_reward.py",
    "abc_bench/fasttd3_campaign_frozen_stage.py",
}
FAST_PAYLOADS = historical.FAST_PAYLOADS
wait_ready, rpc, evaluate_result = (
    historical.wait_ready,
    historical.rpc,
    historical.evaluate_result,
)


def load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    from abc_bench.fasttd3_captured_host import load_config as load_host

    config = json.loads(path.read_text())
    if (
        type(config) is not dict
        or set(config)
        != {
            "schema_version",
            "host_python",
            "learner_python",
            "host_config",
            "training_parent",
            "training_worker",
            "checkpoint",
            "source_pins",
            "limits",
        }
        or type(config["schema_version"]) is not int
        or config["schema_version"] != 1
    ):
        raise ValueError("Use the complete frozen FastTD3 schema1 configuration")
    for key, expected in (
        ("host_python", HOST_PYTHON),
        ("learner_python", LEARNER_PYTHON),
    ):
        if config[key] != str(expected) or not expected.is_file():
            raise ValueError(f"Use the admitted normal installed {key}")
    worker = training_producer(config["training_parent"], config["training_worker"])
    if config["checkpoint"] != worker["checkpoint"]:
        raise ValueError(
            "Frozen checkpoint differs from the authenticated reward-v2 final export"
        )
    pin(config["checkpoint"])
    host = load_host(pin(config["host_config"]))
    if (
        host["host"]["allowed_seeds"] not in ([9100], [9101], [9102])
        or host["host"]["save_video"] is not True
    ):
        raise ValueError(
            "Use one declared captured VAL9100/9101/9102 with actual video"
        )
    limits = config["limits"]
    maxima = {"ready_seconds": 300, "learner_seconds": 1400, "shutdown_seconds": 30}
    if (
        type(limits) is not dict
        or set(limits) != set(maxima)
        or any(
            type(limits[key]) is not int or not 1 <= limits[key] <= maximum
            for key, maximum in maxima.items()
        )
    ):
        raise ValueError(
            "Frozen evaluation requires finite ready/learner/shutdown deadlines"
        )
    if sum(limits.values()) + 30 > min(1800, host["timeouts"]["host_wall_s"]):
        raise ValueError(
            "Frozen stage bounds must fit the1800-second native episode job"
        )
    sources = config["source_pins"]
    if type(sources) is not dict or set(sources) != {"abc", "fast_adapter"}:
        raise ValueError(
            "Pin the executed captured host and original actor consumer boundaries"
        )
    for key, expected in (("abc", ABC_PAYLOADS), ("fast_adapter", FAST_PAYLOADS)):
        rows = sources[key]
        if (
            type(rows) is not list
            or len(rows) != len(expected)
            or {row.get("path") for row in rows} != expected
            or any(set(row) != {"path", "bytes", "sha256"} for row in rows)
        ):
            raise ValueError(f"Incomplete {key} source inventory")
    return config, host, worker


def run(config_path: Path, output: Path) -> dict[str, Any]:
    require_subset_lease()
    config, host_config, _producer = load_config(config_path)
    distribution = importlib.metadata.distribution("abc-minimal")
    for row in config["source_pins"]["abc"]:
        actual = artifact(Path(distribution.locate_file(row["path"])))
        if any(actual[key] != row[key] for key in ("bytes", "sha256")):
            raise RuntimeError(f"Installed captured ABC payload differs: {row['path']}")
    output.mkdir(parents=True, exist_ok=True)
    if any(
        not path.is_file()
        or path.is_symlink()
        or path.name not in {"job.json", "run.log"}
        for path in output.iterdir()
    ):
        raise FileExistsError(
            "Frozen output may contain only fresh canonical bookkeeping"
        )
    receipt = {
        "schema_version": 1,
        "phase": "fasttd3_reward_v2_captured_frozen_evaluation",
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "claim_scope": "One captured VAL scene, original actor/normalizer frozen after versioned reward-v2 scratch campaign; native YAM, no population/paper/R1Lite claim",
        "input_config": artifact(config_path),
        "training_parent": config["training_parent"],
        "training_worker": config["training_worker"],
        "checkpoint": config["checkpoint"],
        "host_config": config["host_config"],
        "process_group": os.getpgrp(),
        "metrics": {},
    }
    host = learner = None
    socket_directory = socket_path = None
    before = time.monotonic()
    previous = {}

    def cancel(signum: int, _frame: Any) -> None:
        raise InterruptedError(f"Frozen Fast stage received signal {signum}")

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
                str(output / "learner_preflight.json"),
                json.dumps(config["source_pins"]["fast_adapter"]),
            ],
            env=qualification.child_environment(cpu=True),
            check=True,
            timeout=30,
        )
        receipt["learner_preflight"] = artifact(output / "learner_preflight.json")
        socket_directory = ROOT / "builds" / ("abcfval-" + uuid.uuid4().hex[:12])
        socket_directory.mkdir(mode=0o700)
        socket_path = socket_directory / "env.sock"
        (output / "native_host").mkdir()
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
                    "abc_bench.fasttd3_captured_host",
                    "--config",
                    config["host_config"]["path"],
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
                    "Captured host must inherit the canonical process group"
                )
            receipt["native_environment"] = wait_ready(
                host, socket_path, config["limits"]["ready_seconds"], host_config
            )
            learner = subprocess.Popen(
                [
                    config["learner_python"],
                    "-I",
                    "-u",
                    "-B",
                    "-m",
                    "sadhana_fasttd3_abc.captured",
                    "--socket",
                    str(socket_path),
                    "--checkpoint",
                    config["checkpoint"]["path"],
                    "--output",
                    str(output / "learner"),
                    "--device",
                    "cuda",
                    "--rpc-timeout",
                    "180",
                ],
                env=qualification.child_environment(cpu=False),
                stdout=learner_log,
                stderr=subprocess.STDOUT,
            )
            if os.getpgid(learner.pid) != os.getpgrp():
                raise RuntimeError(
                    "Frozen actor consumer must inherit the canonical process group"
                )
            if learner.wait(timeout=config["limits"]["learner_seconds"]) != 0:
                raise RuntimeError(
                    "Original frozen actor consumer failed; see learner.log"
                )
            rpc(socket_path, "shutdown", timeout=config["limits"]["shutdown_seconds"])
            if host.wait(timeout=config["limits"]["shutdown_seconds"]) != 0:
                raise RuntimeError("Captured host did not exit cleanly")
        receipt.update(
            status="completed", **evaluate_result(config, host_config, output)
        )
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        cleanup_errors = []
        for process in (learner, host):
            try:
                qualification.stop_process(
                    process, config["limits"]["shutdown_seconds"]
                )
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


def prepare_config(
    training_parent: Path,
    training_worker: Path,
    host_config: Path,
    source_pins: dict[str, Any],
) -> dict[str, Any]:
    """Create concrete pinned evaluation input only after the producer completes."""
    parent_pin, worker_pin = artifact(training_parent), artifact(training_worker)
    producer = training_producer(parent_pin, worker_pin)
    return {
        "schema_version": 1,
        "host_python": str(HOST_PYTHON),
        "learner_python": str(LEARNER_PYTHON),
        "host_config": artifact(host_config),
        "training_parent": parent_pin,
        "training_worker": worker_pin,
        "checkpoint": producer["checkpoint"],
        "source_pins": source_pins,
        "limits": {
            "ready_seconds": 300,
            "learner_seconds": 1400,
            "shutdown_seconds": 30,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    run(options.config, options.output)


if __name__ == "__main__":
    main()
