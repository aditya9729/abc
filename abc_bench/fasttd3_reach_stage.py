"""Canonical bounded admission for the separate FastTD3 reaching diagnostic."""

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

from abc_bench import fasttd3_vector_stage as qualification
from abc_bench.fasttd3_host import write_json
from abc_bench.fasttd3_reach_contract import PHASES, PROFILE, REACH, STOPPING
from abc_bench.fasttd3_reach_host import load_config as load_host
from abc_bench.fasttd3_scratch_stage import EXPORT_CHECK, pin
from abc_bench.fasttd3_vector_host import validate_cleanup
from abc_bench.subset import require_subset_lease
from abc_bench.upstream_eval import artifact

ROOT, HOST_PYTHON, LEARNER_PYTHON = (
    qualification.ROOT,
    qualification.HOST_PYTHON,
    qualification.LEARNER_PYTHON,
)
PREFLIGHT = qualification.PREFLIGHT
memory_admission, child_environment, stop_process, rpc = (
    qualification.memory_admission,
    qualification.child_environment,
    qualification.stop_process,
    qualification.rpc,
)
ABC_PAYLOADS = {
    "abc_bench/" + name
    for name in (
        "fasttd3_reach_contract.py",
        "fasttd3_reach_host.py",
        "fasttd3_reach_stage.py",
        "fasttd3_vector_host.py",
        "fasttd3_vector_stage.py",
        "fasttd3_scratch_stage.py",
        "fasttd3_campaign_host.py",
        "fasttd3_campaign_contract.py",
        "bottle_success_reward.py",
        "fasttd3_host.py",
        "subset.py",
        "upstream_eval.py",
    )
}
FAST_PAYLOADS = {
    "sadhana_fasttd3_abc/" + name
    for name in (
        "reach.py",
        "reach_contract.py",
        "vector.py",
        "environment.py",
        "learner.py",
        "initial_export.py",
        "provenance.json",
    )
}


def recipe(host: dict, *, actual: int | None = None) -> dict:
    return {
        **qualification.RECIPE,
        "exp_name": "FastTD3ABCReachDiagnostic_" + host["phase"],
        "num_envs": 64,
        "total_timesteps": host["vector_steps"] if actual is None else actual,
    }


def counters(calls: int) -> dict:
    if type(calls) is not int or not 12 <= calls <= PHASES["training"]:
        raise ValueError(
            "Reach accounting requires completed original updates within2048 calls"
        )
    return {
        "vector_steps": calls,
        "transitions": calls * 64,
        "critic_updates": 2 * (calls - 11),
        "actor_updates": calls - 11,
    }


def resource_evidence(parent_pin: dict, worker_pin: dict) -> dict:
    """Authenticate prior64-world resource evidence, not new reward acceptance."""
    parent, worker = (
        json.loads(pin(parent_pin).read_text()),
        json.loads(pin(worker_pin).read_text()),
    )
    if (
        parent.get("status") != "completed"
        or parent.get("embodiment") != "native_yam"
        or parent.get("worker_exit_code") != 0
        or parent.get("worker_receipt_sha256") != worker_pin["sha256"]
        or parent.get("metrics") != counters(64)
    ):
        raise ValueError("Use the actual canonical completed64-world resource producer")
    if (
        worker.get("phase") != "fasttd3_reward_v2_runtime_gate"
        or worker.get("status") != "completed"
        or worker.get("metrics") != counters(64)
        or worker.get("child_exit_codes") != {"host": 0, "learner": 0}
    ):
        raise ValueError("Prior64-world runtime accounting did not complete")
    native = json.loads(pin(worker["native_receipt"]).read_text())
    if (
        native.get("status") != "completed"
        or native.get("transitions") != 4096
        or native.get("vector_steps") != 64
    ):
        raise ValueError("Prior native64-world resource producer differs")
    validate_cleanup(native.get("worker_cleanup"), 64)
    resources = native.get("worker_resources")
    if type(resources) is not list or len(resources) != 64:
        raise ValueError("Require all64 measured native resource receipts")
    ranks = set()
    for row in resources:
        measurement = json.loads(pin(row["evidence"]).read_text())
        if (
            measurement != row["measurement"]
            or measurement.get("status") != "closed"
            or measurement.get("cuda_visible_devices") != ""
            or measurement.get("torch_imported") is not False
            or measurement.get("warp_imported") is not False
            or measurement.get("render_cameras") is not False
            or type(measurement.get("maxrss_kib")) is not int
            or measurement["maxrss_kib"] < 1
        ):
            raise ValueError("Native CPU resource measurement differs")
        ranks.add(measurement["rank"])
    if ranks != set(range(64)):
        raise ValueError("Native CPU resource rank inventory differs")
    return native


def check_description(description: dict, host: dict) -> None:
    expected = {
        "profile": PROFILE,
        "num_envs": 64,
        "num_obs": 216,
        "num_actions": 14,
        "max_episode_steps": 128,
        "task": "put_plastic_bottles_in_bin",
        "privileged_state": True,
        "physics_dt": 0.002,
        "control_decimation": 17,
        "camera_backend": None,
        "carried_arm_reset": True,
        "native_terminate_on_success": False,
        "reach_diagnostic": REACH,
        "diagnostic_stopping": STOPPING,
        "physics_failure_policy": "fail_on_every_warning_nonfinite_or_internal_time_reset",
        "reset_options": {
            "randomization": {
                "bottle_count": 5,
                "randomize_variants": True,
                "randomize_scales": True,
            }
        },
        "reset_seed_bank": {
            "base": 200000,
            "rank_stride": 1,
            "episode_stride": 128,
            "max_episode_index": 780,
            "fallback_offset": 100000,
        },
    }
    if host["reach"] != REACH or any(
        type(description.get(key)) is not type(value) or description.get(key) != value
        for key, value in expected.items()
    ):
        raise ValueError("Actual native reaching diagnostic description differs")
    schema = description.get("action_schema", {})
    if (
        schema.get("mapping") != "low + (clip(u,-1,1)+1)*0.5*(high-low)"
        or len(schema.get("names", [])) != 14
    ):
        raise ValueError("Original14-action mapping differs")


def wait_ready(process: Any, path: Path, seconds: int, host: dict) -> dict:
    deadline = time.monotonic() + seconds
    while True:
        if process.poll() is not None:
            raise RuntimeError(
                f"Reach host exited {process.returncode} before readiness"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError("Reach host readiness deadline expired")
        if path.exists():
            try:
                description = rpc(
                    path, "describe", timeout=max(0.001, deadline - time.monotonic())
                )
                check_description(description, host)
                return description
            except (FileNotFoundError, ConnectionRefusedError):
                pass
        time.sleep(0.1)


def load_config(path: Path) -> tuple[dict, dict]:
    config = json.loads(path.read_text())
    fields = {
        "schema_version",
        "host_python",
        "learner_python",
        "host_config",
        "resource_parent",
        "resource_worker",
        "source_pins",
        "runtime_gate_parent",
        "runtime_gate_worker",
        "limits",
    }
    if (
        type(config) is not dict
        or set(config) != fields
        or type(config["schema_version"]) is not int
        or config["schema_version"] != 1
    ):
        raise ValueError("Use the complete separate reach stage schema1")
    for key, expected in (
        ("host_python", HOST_PYTHON),
        ("learner_python", LEARNER_PYTHON),
    ):
        if config[key] != str(expected) or not expected.is_file():
            raise ValueError("Use the normal installed " + key)
    host = load_host(pin(config["host_config"]))
    evidence = resource_evidence(config["resource_parent"], config["resource_worker"])
    sources = config["source_pins"]
    if type(sources) is not dict or set(sources) != {"abc", "fast_adapter"}:
        raise ValueError("Pin both installed reaching diagnostic boundaries")
    for key, expected in (("abc", ABC_PAYLOADS), ("fast_adapter", FAST_PAYLOADS)):
        rows = sources[key]
        if (
            type(rows) is not list
            or len(rows) != len(expected)
            or {row.get("path") for row in rows} != expected
            or any(set(row) != {"path", "bytes", "sha256"} for row in rows)
        ):
            raise ValueError("Incomplete diagnostic source inventory: " + key)
    limits = config["limits"]
    maxima = {"ready_seconds": 600, "learner_seconds": 2700, "shutdown_seconds": 30}
    if (
        type(limits) is not dict
        or set(limits) != set(maxima)
        or any(
            type(limits[key]) is not int or not 1 <= limits[key] <= maximum
            for key, maximum in maxima.items()
        )
    ):
        raise ValueError("Use the finite reaching diagnostic watchdogs")
    if sum(limits.values()) + 120 > host["timeouts"]["wall_s"]:
        raise ValueError("Preflight/export/children do not fit the host wall bound")
    if host["phase"] == "runtime_gate":
        if (
            config["runtime_gate_parent"] is not None
            or config["runtime_gate_worker"] is not None
        ):
            raise ValueError(
                "A new diagnostic gate cannot claim prior reach acceptance"
            )
    else:
        diagnostic_gate(
            config["runtime_gate_parent"], config["runtime_gate_worker"], host, sources
        )
    return config, evidence


def diagnostic_gate(
    parent_pin: dict, worker_pin: dict, host: dict, sources: dict
) -> dict:
    parent, worker = (
        json.loads(pin(parent_pin).read_text()),
        json.loads(pin(worker_pin).read_text()),
    )
    if (
        parent.get("status") != "completed"
        or parent.get("worker_exit_code") != 0
        or parent.get("worker_receipt_sha256") != worker_pin["sha256"]
        or parent.get("metrics") != counters(PHASES["runtime_gate"])
    ):
        raise ValueError(
            "Training requires an actual completed canonical reach diagnostic gate"
        )
    if (
        worker.get("status") != "completed"
        or worker.get("phase") != "fasttd3_reach_diagnostic_runtime_gate"
        or worker.get("source_pins") != sources
        or worker.get("metrics") != counters(PHASES["runtime_gate"])
        or worker.get("child_exit_codes") != {"host": 0, "learner": 0}
    ):
        raise ValueError("New reach diagnostic gate/source/cleanup differs")
    gate_host = load_host(pin(worker["host_config"]))
    if (
        gate_host["phase"] != "runtime_gate"
        or gate_host["reach"] != host["reach"]
        or gate_host["selection_evidence"] != host["selection_evidence"]
    ):
        raise ValueError("New diagnostic gate used different reach criteria")
    config = json.loads(pin(worker["input_config"]).read_text())
    actual = verify_result(config, gate_host, Path(worker_pin["path"]).parent)
    if any(worker.get(key) != value for key, value in actual.items()):
        raise ValueError(
            "Completed reach gate receipt differs from actual native/learner evidence"
        )
    pin(worker["export_check"])
    return worker


def prepare_training_config(
    gate_config: Path, training_host: Path, parent: Path, worker: Path
) -> dict:
    config, _ = load_config(gate_config)
    host = load_host(training_host)
    if host["phase"] != "training":
        raise ValueError("Use the separate2048-call diagnostic training host")
    diagnostic_gate(artifact(parent), artifact(worker), host, config["source_pins"])
    return {
        **config,
        "host_config": artifact(training_host),
        "runtime_gate_parent": artifact(parent),
        "runtime_gate_worker": artifact(worker),
    }


def verify_result(config: dict, host_config: dict, output: Path) -> dict:
    learner = json.loads((output / "learner/receipt.json").read_text())
    diagnostic = json.loads((output / "learner/reach-diagnostic.json").read_text())
    native = json.loads((output / "native_host/receipt.json").read_text())
    calls = diagnostic.get("controls")
    expected = counters(calls)
    if (
        any(row.get("status") != "completed" for row in (learner, diagnostic, native))
        or native.get("config") != host_config
        or diagnostic.get("phase") != host_config["phase"]
    ):
        raise RuntimeError(
            "Reach host, original learner and diagnostic did not complete"
        )
    validate_cleanup(native.get("worker_cleanup"), 64)
    if any(Path(f"/proc/{row['pid']}").exists() for row in native["worker_cleanup"]):
        raise RuntimeError("An owned native reach worker remains alive")
    if (
        native.get("vector_steps") != calls
        or learner.get("controls") != calls
        or native.get("transitions") != expected["transitions"]
        or diagnostic.get("transitions") != expected["transitions"]
        or learner.get("actual_updates")
        != {"critic": expected["critic_updates"], "actor": expected["actor_updates"]}
        or diagnostic.get("actual_updates") != learner["actual_updates"]
    ):
        raise RuntimeError(
            "Actual diagnostic calls/transitions/original updates differ"
        )
    for values, expected_args in (
        (learner["configuration"], recipe(host_config)),
        (diagnostic["declared_maximum_args"], recipe(host_config)),
        (diagnostic["effective_final_args"], recipe(host_config, actual=calls)),
    ):
        if any(
            type(values.get(key)) is not type(value) or values.get(key) != value
            for key, value in expected_args.items()
        ):
            raise RuntimeError("Original learner defaults or finite loop limit differ")
    if any(
        learner.get("finite_gradient_samples", {}).get(role, 0) < 1
        for role in ("actor", "critic")
    ):
        raise RuntimeError("Original updates lack finite gradient evidence")
    decision = diagnostic.get("stopping", {}).get("decision")
    if (
        type(decision) is not dict
        or decision.get("controls") != calls
        or diagnostic["stopping"].get("spec") != STOPPING
    ):
        raise RuntimeError("Reach diagnostic lacks a declared stop decision")
    if calls == host_config["vector_steps"]:
        if decision.get("reason") != "declared_diagnostic_maximum_completed":
            raise RuntimeError("Reach budget completion reason differs")
    elif (
        host_config["phase"] != "training"
        or calls < 1280
        or calls % 256
        or calls > 2048
        or decision.get("reason") != "qualified_TRAIN_reach_progress_plateau"
        or decision.get("stale_windows") != 3
    ):
        raise RuntimeError(
            "Partial reach run cannot masquerade as a completed diagnostic"
        )
    check_description(native["description"], host_config)
    episodes, summaries = [], []
    for rank in range(64):
        directory = output / "native_host" / f"world-{rank:03d}"
        summary_path = directory / "reach-summary.json"
        summary = json.loads(summary_path.read_text())
        physics = summary["physics"]
        if (
            summary.get("rank") != rank
            or summary.get("controls") != calls
            or summary.get("reach_spec") != REACH
            or not physics["time_finite"]
            or not all(physics["finite"].values())
            or any(row["number"] for row in physics["warnings"].values())
            or any(directory.glob("physics-failure-*.json"))
        ):
            raise RuntimeError("Actual native reach summary/physics differs")
        rows = [
            json.loads(path.read_text())
            for path in sorted(directory.glob("episode-*.json"))
        ]
        if len(rows) != calls // 128 or summary["completed_episodes"] != len(rows):
            raise RuntimeError("Actual completed fixed-horizon reach episodes differ")
        episodes.extend(rows)
        summaries.append(artifact(summary_path))
    if sorted(episodes, key=lambda row: (row["rank"], row["episode_index"])) != sorted(
        diagnostic.get("episodes", []),
        key=lambda row: (row["rank"], row["episode_index"]),
    ):
        raise RuntimeError("Native/client diagnostic episode evidence differs")
    if diagnostic["stopping"].get("completed_episodes_per_rank") != [calls // 128] * 64:
        raise RuntimeError("Diagnostic rank coverage differs")
    if diagnostic.get("checkpoint") != learner.get("checkpoint"):
        raise RuntimeError("Diagnostic checkpoint binding differs")
    initial = learner.get("initial_export", {})
    if (
        initial != diagnostic.get("initial_export")
        or initial.get("schema") != "sadhana.fasttd3.initial-scratch-export/v1"
        or initial.get("status") != "completed"
        or any(
            type(initial.get(key)) is not int or initial[key] != 0
            for key in ("global_step", "controls", "actor_updates", "critic_updates")
        )
        or initial.get("before") != initial.get("after")
        or initial.get("source") != learner.get("source")
    ):
        raise RuntimeError(
            "Reach producer lacks a read-only true initial scratch export"
        )
    pin(initial["checkpoint"])
    metadata = json.loads(pin(initial["metadata"]).read_text())
    if metadata != {key: value for key, value in initial.items() if key != "metadata"}:
        raise RuntimeError("Initial export metadata differs from the producer binding")
    pin(learner["checkpoint"])
    return {
        "metrics": expected,
        "stop_decision": decision,
        "completed_diagnostic_episodes": len(episodes),
        "matched_initial_final_deterministic_actor_comparison": "pending",
        "reach_summaries": summaries,
        "checkpoint": learner["checkpoint"],
        "diagnostic_receipt": artifact(output / "learner/reach-diagnostic.json"),
        "native_receipt": artifact(output / "native_host/receipt.json"),
        "learner_receipt": artifact(output / "learner/receipt.json"),
    }


def run(config_path: Path, output: Path) -> dict[str, Any]:
    require_subset_lease()  # Before mutation, native import, or any child launch.
    config, evidence = load_config(config_path)
    host_config = load_host(pin(config["host_config"]))
    output.mkdir(parents=True, exist_ok=True)
    if any(
        not path.is_file()
        or path.is_symlink()
        or path.name not in {"job.json", "run.log"}
        for path in output.iterdir()
    ):
        raise FileExistsError(
            "Stage output may contain only fresh canonical bookkeeping"
        )
    distribution = importlib.metadata.distribution("abc-minimal")
    for row in config["source_pins"]["abc"]:
        actual = artifact(Path(distribution.locate_file(row["path"])))
        if any(actual[key] != row[key] for key in ("bytes", "sha256")):
            raise RuntimeError(f"Installed campaign payload differs: {row['path']}")
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "phase": "fasttd3_reach_diagnostic_" + host_config["phase"],
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "claim_scope": "Separate dense-reaching diagnostic; original scratch learner; matched initial/final actor comparison pending; no five-bottle task/paper/R1Lite/isolated-learning claim",
        "input_config": artifact(config_path),
        "prior_native_resource_parent": config["resource_parent"],
        "prior_native_resource_worker": config["resource_worker"],
        "host_config": config["host_config"],
        "source_pins": config["source_pins"],
        "process_group": os.getpgrp(),
        "gpu_visibility": os.environ["CUDA_VISIBLE_DEVICES"],
        "declared_maximum_calls": host_config["vector_steps"],
        "reach_spec": host_config["reach"],
        "metrics": {},
    }
    host = learner = None
    socket_directory = socket_path = None
    before = time.monotonic()
    previous = {}

    def cancel(signum: int, _frame: Any) -> None:
        raise InterruptedError(f"FastTD3 ABC stage received signal {signum}")

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, cancel)
        subprocess.run(
            [
                config["learner_python"],
                "-I",
                "-B",
                "-c",
                PREFLIGHT,
                str(output / "learner_preflight.json"),
                json.dumps(config["source_pins"]["fast_adapter"]),
            ],
            env=child_environment(cpu=True),
            check=True,
            timeout=30,
        )
        receipt["learner_preflight"] = artifact(output / "learner_preflight.json")
        receipt["memory_admission"] = memory_admission(evidence, count=64)
        socket_directory = ROOT / "builds" / ("abcrpc-" + uuid.uuid4().hex[:12])
        socket_directory.mkdir(mode=0o700)
        socket_path = socket_directory / "env.sock"
        host_output = output / "native_host"
        host_output.mkdir()
        with (
            (output / "host.log").open("w") as host_log,
            (output / "learner.log").open("w") as learner_log,
        ):
            # Preserve the UUID here: the host authenticates the lease before it hides CUDA.
            host = subprocess.Popen(
                [
                    config["host_python"],
                    "-I",
                    "-u",
                    "-B",
                    "-m",
                    "abc_bench.fasttd3_reach_host",
                    "--config",
                    config["host_config"]["path"],
                    "--socket",
                    str(socket_path),
                    "--output",
                    str(host_output),
                ],
                env=child_environment(cpu=False),
                stdout=host_log,
                stderr=subprocess.STDOUT,
            )
            if os.getpgid(host.pid) != os.getpgrp():
                raise RuntimeError("ABC host must inherit the canonical process group")
            receipt["native_environment"] = wait_ready(
                host, socket_path, config["limits"]["ready_seconds"], host_config
            )
            environment = child_environment(cpu=False)
            environment.update(
                TORCHINDUCTOR_CACHE_DIR=str(output / "torchinductor"),
                TRITON_CACHE_DIR=str(output / "triton"),
            )
            learner = subprocess.Popen(
                [
                    config["learner_python"],
                    "-I",
                    "-u",
                    "-B",
                    "-m",
                    "sadhana_fasttd3_abc.reach",
                    "--socket",
                    str(socket_path),
                    "--output",
                    str(output / "learner"),
                    "--config",
                    config["host_config"]["path"],
                    "--device",
                    "cuda",
                    "--rpc-timeout",
                    "180",
                ],
                env=environment,
                stdout=learner_log,
                stderr=subprocess.STDOUT,
            )
            if os.getpgid(learner.pid) != os.getpgrp():
                raise RuntimeError(
                    "Original learner must inherit the canonical process group"
                )
            if learner.wait(timeout=config["limits"]["learner_seconds"]) != 0:
                raise RuntimeError("Original learner failed; see learner.log")
            rpc(socket_path, "shutdown", timeout=config["limits"]["shutdown_seconds"])
            if host.wait(timeout=config["limits"]["shutdown_seconds"]) != 0:
                raise RuntimeError("ABC host failed its clean shutdown")
        actual = verify_result(config, host_config, output)
        subprocess.run(
            [
                config["learner_python"],
                "-I",
                "-B",
                "-c",
                EXPORT_CHECK,
                actual["checkpoint"]["path"],
                str(output / "export_check.json"),
                json.dumps(
                    recipe(host_config, actual=actual["metrics"]["vector_steps"])
                ),
            ],
            env=child_environment(cpu=True),
            check=True,
            timeout=30,
        )
        receipt.update(
            status="completed",
            **actual,
            export_check=artifact(output / "export_check.json"),
            export_semantics="Original final actor/critic/normalizer export; no optimizer/replay/RNG resume state",
        )
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        cleanup_errors = []
        for process in (learner, host):
            try:
                stop_process(process, config["limits"]["shutdown_seconds"])
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    options = parser.parse_args()
    if options.check_only:
        config, evidence = load_config(options.config)
        write_json(
            options.output,
            {
                "status": "CPU_input_contract_checked",
                "input": artifact(options.config),
                "resource_worker_count": len(evidence["worker_resources"]),
                "native_reach_run": "not_run",
                "learning": "not_run",
                "phase": load_host(pin(config["host_config"]))["phase"],
            },
        )
    else:
        run(options.config, options.output)


if __name__ == "__main__":
    main()
