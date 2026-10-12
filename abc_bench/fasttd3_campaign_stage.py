"""Canonical supervision for a versioned ABC reward-v2 scratch campaign."""

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
from abc_bench.bottle_success_reward import PBRS_PROFILE, reward_config_sha256
from abc_bench.fasttd3_campaign_host import load_config as load_host
from abc_bench.fasttd3_host import write_json
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
        "fasttd3_campaign_host.py",
        "fasttd3_campaign_contract.py",
        "fasttd3_campaign_stage.py",
        "bottle_success_reward.py",
        "fasttd3_vector_host.py",
    )
}
FAST_PAYLOADS = {
    "sadhana_fasttd3_abc/" + name
    for name in (
        "campaign.py",
        "vector.py",
        "environment.py",
        "learner.py",
        "provenance.json",
    )
}


def recipe(host: dict, *, actual: int | None = None) -> dict:
    return {
        **qualification.RECIPE,
        "exp_name": "FastTD3ABCRewardV2_" + host["phase"],
        "total_timesteps": host["vector_steps"] if actual is None else actual,
    }


def counters(calls: int) -> dict:
    if type(calls) is not int or calls < 12:
        raise ValueError(
            "A completed campaign must reach original critic and actor updates"
        )
    return {
        "vector_steps": calls,
        "transitions": 128 * calls,
        "critic_updates": 2 * (calls - 11),
        "actor_updates": calls - 11,
    }


def check_description(description: dict, host: dict) -> None:
    qualification.check_description({**description, "profile": "training128"})
    if (
        description.get("profile") != host["profile"]
        or description.get("reward") != host["reward"]
        or description.get("campaign") != host["campaign"]
    ):
        raise ValueError("Actual native reward/campaign specification differs")
    if (
        description.get("physics_failure_policy")
        != "fail_on_BADQPOS_BADQVEL_BADQACC_BADCTRL_nonfinite_or_internal_time_reset"
    ):
        raise ValueError("New campaign numerical-physics failure gate differs")


def wait_ready(process: Any, path: Path, seconds: int, host: dict) -> dict:
    deadline = time.monotonic() + seconds
    while True:
        if process.poll() is not None:
            raise RuntimeError(
                f"Campaign host exited {process.returncode} before readiness"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError("Reward-v2 host readiness deadline expired")
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


def runtime_gate(parent_pin: dict, worker_pin: dict, host: dict, sources: dict) -> dict:
    parent, worker = (
        json.loads(pin(parent_pin).read_text()),
        json.loads(pin(worker_pin).read_text()),
    )
    if (
        parent.get("status") != "completed"
        or parent.get("embodiment") != "native_yam"
        or parent.get("task") != "put_plastic_bottles_in_bin"
        or parent.get("worker_exit_code") != 0
        or parent.get("worker_receipt_sha256") != worker_pin["sha256"]
        or parent.get("metrics") != counters(64)
    ):
        raise ValueError(
            "Training requires the actual completed canonical reward-v2 runtime gate"
        )
    gate_host = json.loads(pin(worker["host_config"]).read_text())
    if (
        worker.get("phase") != "fasttd3_reward_v2_runtime_gate"
        or worker.get("status") != "completed"
        or worker.get("metrics") != counters(64)
        or worker.get("child_exit_codes") != {"host": 0, "learner": 0}
    ):
        raise ValueError("Reward-v2 runtime gate and cleanup did not complete")
    if (
        gate_host.get("phase") != "runtime_gate"
        or gate_host.get("reward") != host["reward"]
        or worker.get("source_pins") != sources
    ):
        raise ValueError("Runtime gate used a different reward or executed source")
    native = json.loads(pin(worker["native_receipt"]).read_text())
    validate_cleanup(native.get("worker_cleanup"), 128)
    if (
        native.get("vector_steps") != 64
        or native.get("transitions") != 8192
        or native.get("status") != "completed"
    ):
        raise ValueError("Actual reward-v2 native runtime accounting differs")
    pin(worker["checkpoint"])
    return native


def load_config(path: Path) -> tuple[dict, dict]:
    config = json.loads(path.read_text())
    required = {
        "schema_version",
        "host_python",
        "learner_python",
        "host_config",
        "cpu_admission",
        "qualification_config",
        "qualification_parent",
        "qualification_worker",
        "source_pins",
        "runtime_gate_parent",
        "runtime_gate_worker",
        "limits",
    }
    if (
        type(config) is not dict
        or set(config) != required
        or type(config["schema_version"]) is not int
        or config["schema_version"] != 1
    ):
        raise ValueError("Use the complete separate campaign schema1")
    for key, expected in (
        ("host_python", HOST_PYTHON),
        ("learner_python", LEARNER_PYTHON),
    ):
        if config[key] != str(expected) or not expected.is_file():
            raise ValueError(f"Use the normal installed {key}")
    host = load_host(pin(config["host_config"]))
    qualified, evidence = qualification.load_config(pin(config["qualification_config"]))
    if config["cpu_admission"] != qualified["cpu_admission"]:
        raise ValueError("Measured native CPU admission differs")
    parent, worker = (
        json.loads(pin(config["qualification_parent"]).read_text()),
        json.loads(pin(config["qualification_worker"]).read_text()),
    )
    if (
        parent.get("status") != "completed"
        or parent.get("embodiment") != "native_yam"
        or parent.get("task") != "put_plastic_bottles_in_bin"
        or parent.get("worker_exit_code") != 0
        or parent.get("worker_receipt_sha256")
        != config["qualification_worker"]["sha256"]
    ):
        raise ValueError("Original native ABC128 qualification parent did not complete")
    if (
        worker.get("status") != "completed"
        or worker.get("input_config") != config["qualification_config"]
        or worker.get("metrics") != counters(64)
        or worker.get("child_exit_codes") != {"host": 0, "learner": 0}
    ):
        raise ValueError("Original native ABC128 qualification differs")
    validate_cleanup(
        json.loads(pin(worker["native_receipt"]).read_text()).get("worker_cleanup"), 128
    )
    pin(worker["checkpoint"])
    sources = config["source_pins"]
    if type(sources) is not dict or set(sources) != {"abc", "fast_adapter"}:
        raise ValueError("Pin both native reward and author client boundaries")
    for key, expected in (("abc", ABC_PAYLOADS), ("fast_adapter", FAST_PAYLOADS)):
        rows = sources[key]
        if (
            type(rows) is not list
            or len(rows) != len(expected)
            or {row.get("path") for row in rows} != expected
            or any(set(row) != {"path", "bytes", "sha256"} for row in rows)
        ):
            raise ValueError(f"Incomplete {key} source inventory")
    limits = config["limits"]
    maximum = {
        "ready_seconds": 600,
        "learner_seconds": 1500 if host["phase"] == "runtime_gate" else 11400,
        "shutdown_seconds": 30,
    }
    if (
        type(limits) is not dict
        or set(limits) != set(maximum)
        or any(
            type(limits[key]) is not int or not 1 <= limits[key] <= value
            for key, value in maximum.items()
        )
    ):
        raise ValueError("Use finite campaign ready/learner/shutdown watchdogs")
    if sum(limits.values()) + 120 > host["timeouts"]["wall_s"]:
        raise ValueError("Source, export and shutdown bounds do not fit host wall")
    if host["phase"] == "runtime_gate":
        if (
            config["runtime_gate_parent"] is not None
            or config["runtime_gate_worker"] is not None
        ):
            raise ValueError("Short runtime gate cannot claim prior reward-v2 training")
    else:
        gate = runtime_gate(
            config["runtime_gate_parent"], config["runtime_gate_worker"], host, sources
        )
        evidence = {**evidence, "worker_resources": gate["worker_resources"]}
    return config, evidence


def prepare_training_config(
    gate_config: Path, training_host: Path, gate_parent: Path, gate_worker: Path
) -> dict:
    """Bind the long input to a completed native gate before it can be launched."""
    config = json.loads(gate_config.read_text())
    host = load_host(training_host)
    if host["phase"] != "training":
        raise ValueError("Use the distinct full-length training host profile")
    runtime_gate(
        artifact(gate_parent), artifact(gate_worker), host, config["source_pins"]
    )
    return {
        **config,
        "host_config": artifact(training_host),
        "runtime_gate_parent": artifact(gate_parent),
        "runtime_gate_worker": artifact(gate_worker),
        "limits": {
            "ready_seconds": 600,
            "learner_seconds": 11400,
            "shutdown_seconds": 30,
        },
    }


def training_producer(parent_pin: dict, worker_pin: dict) -> dict:
    """Admit a completed campaign export, including actual natural stop evidence."""
    parent, worker = (
        json.loads(pin(parent_pin).read_text()),
        json.loads(pin(worker_pin).read_text()),
    )
    if (
        parent.get("status") != "completed"
        or parent.get("embodiment") != "native_yam"
        or parent.get("task") != "put_plastic_bottles_in_bin"
        or parent.get("worker_exit_code") != 0
        or parent.get("worker_receipt_sha256") != worker_pin["sha256"]
        or worker.get("status") != "completed"
        or worker.get("phase") != "fasttd3_reward_v2_training"
        or worker.get("child_exit_codes") != {"host": 0, "learner": 0}
        or parent.get("metrics") != worker.get("metrics")
    ):
        raise ValueError(
            "Frozen evaluation requires a completed canonical reward-v2 training producer"
        )
    config = json.loads(pin(worker["input_config"]).read_text())
    host = load_host(pin(worker["host_config"]))
    if (
        host["phase"] != "training"
        or config.get("host_config") != worker["host_config"]
    ):
        raise ValueError("Campaign producer configuration binding differs")
    actual = verify_result(config, host, Path(worker_pin["path"]).parent)
    for key, value in actual.items():
        if worker.get(key) != value:
            raise ValueError(
                f"Producer's authenticated native/campaign/export binding differs: {key}"
            )
    export = json.loads(pin(worker["export_check"]).read_text())
    if export.get("global_step") != actual["metrics"]["vector_steps"] or not all(
        export.get("state_fingerprints", {}).get(name, {}).get("sha256")
        for name in (
            "actor_state_dict",
            "qnet_state_dict",
            "qnet_target_state_dict",
            "obs_normalizer_state",
            "critic_obs_normalizer_state",
        )
    ):
        raise ValueError(
            "Producer did not authenticate its actual final original actor/normalizer export"
        )
    return worker


def verify_result(config: dict, host_config: dict, output: Path) -> dict:
    learner = json.loads((output / "learner/receipt.json").read_text())
    campaign = json.loads((output / "learner/campaign.json").read_text())
    host = json.loads((output / "native_host/receipt.json").read_text())
    calls = campaign.get("controls")
    expected = counters(calls)
    if (
        any(row.get("status") != "completed" for row in (learner, campaign, host))
        or campaign.get("phase") != host_config["phase"]
        or host.get("config") != host_config
    ):
        raise RuntimeError(
            "Native host, original learner and campaign did not complete"
        )
    validate_cleanup(host.get("worker_cleanup"), 128)
    if any(Path(f"/proc/{row['pid']}").exists() for row in host["worker_cleanup"]):
        raise RuntimeError("An owned native worker remains alive")
    if (
        host.get("vector_steps") != calls
        or learner.get("controls") != calls
        or host.get("transitions") != expected["transitions"]
        or campaign.get("transitions") != expected["transitions"]
        or learner.get("actual_updates")
        != {"critic": expected["critic_updates"], "actor": expected["actor_updates"]}
        or campaign.get("actual_updates") != learner["actual_updates"]
    ):
        raise RuntimeError("Actual calls/transitions/original updates differ")
    initial, final = recipe(host_config), recipe(host_config, actual=calls)
    for row, expected_args in (
        (learner["configuration"], initial),
        (campaign["declared_maximum_args"], initial),
        (campaign["effective_final_args"], final),
    ):
        if any(
            type(row.get(key)) is not type(value) or row.get(key) != value
            for key, value in expected_args.items()
        ):
            raise RuntimeError(
                "Executed original defaults or effective loop limit differ"
            )
    if any(
        learner.get("finite_gradient_samples", {}).get(role, 0) < 1
        for role in ("actor", "critic")
    ):
        raise RuntimeError("Original updates lack finite gradient evidence")
    decision = campaign.get("stopping", {}).get("decision")
    if (
        type(decision) is not dict
        or decision.get("controls") != calls
        or campaign["stopping"].get("spec") != host_config["campaign"]
    ):
        raise RuntimeError(
            "A completed campaign needs its declared task/budget stop decision"
        )
    if calls == host_config["vector_steps"]:
        reason = (
            "runtime_gate_completed"
            if host_config["phase"] == "runtime_gate"
            else "declared_maximum_completed"
        )
        if decision.get("reason") != reason:
            raise RuntimeError("Budget completion reason differs")
    elif (
        host_config["phase"] != "training"
        or calls < 31860
        or calls % 3540
        or calls > 42480
        or decision.get("reason") != "no_observed_training_task_improvement"
        or decision.get("stale_windows") != 3
    ):
        raise RuntimeError("A partial run cannot masquerade as task-futility stopping")
    if decision.get("reason") == "no_observed_training_task_improvement":
        windows = campaign["stopping"].get("windows", [])
        tail = windows[-3:]
        if len(tail) != 3 or any(
            row.get("controls") != calls - (2 - index) * 3540
            or row.get("complete") is not True
            or row.get("eligible") is not True
            or row.get("meaningful_improvement") != []
            or row.get("stale_windows") != index + 1
            or len(row.get("rank_episode_counts", [])) != 128
            or min(row["rank_episode_counts"]) < 1
            for index, row in enumerate(tail)
        ):
            raise RuntimeError(
                "Task futility needs three complete eligible task windows"
            )
    check_description(host["description"], host_config)
    episodes, summaries, proofs = [], [], []
    for rank in range(128):
        directory = output / "native_host" / f"world-{rank:03d}"
        summary_path = directory / "reward-summary.json"
        summary = json.loads(summary_path.read_text())
        if (
            type(summary.get("rank")) is not int
            or summary.get("rank") != rank
            or type(summary.get("controls")) is not int
            or summary.get("controls") != calls
            or type(summary.get("completed_episodes")) is not int
            or summary.get("reward") != host_config["reward"]
            or summary["physics"].get("time_finite") is not True
            or not all(summary["physics"]["finite"].values())
            or any(
                summary["physics"]["warnings"][name]["number"]
                for name in ("BADQPOS", "BADQVEL", "BADQACC", "BADCTRL")
            )
            or any(directory.glob("physics-failure-*.json"))
        ):
            raise RuntimeError(f"Rank{rank} native physics/reward closure differs")
        rank_episodes = [
            json.loads(path.read_text())
            for path in sorted(directory.glob("episode-*.json"))
        ]
        if len(rank_episodes) != summary["completed_episodes"] or any(
            type(row.get("rank")) is not int
            or row.get("rank") != rank
            or type(row.get("episode_index")) is not int
            or row.get("episode_index") != index
            for index, row in enumerate(rank_episodes)
        ):
            raise RuntimeError("Native completed episode inventory differs")
        episodes.extend(rank_episodes)
        summaries.append(artifact(summary_path))
        if host_config["reward"]["profile"] == PBRS_PROFILE:
            proof_pin = summary["first_geometry_proof"]
            proof = json.loads(pin(proof_pin).read_text())
            if (
                proof.get("status") != "completed"
                or proof.get("rank") != rank
                or proof.get("controls") != 0
                or proof.get("integration_exact_equal") is not True
                or proof.get("integration_before_sha256")
                != proof.get("integration_after_sha256")
                or proof.get("qpos_derived_com_exact_equal") is not True
                or proof.get("qpos_derived_sites_exact_equal") is not True
                or proof["geometry_contract"]["config_sha256"]
                != reward_config_sha256(PBRS_PROFILE)
                or not 0 <= proof["potential"]["phi"] <= 0.25
            ):
                raise RuntimeError(
                    "Actual first native read-only geometry proof did not pass"
                )
            proofs.append(proof_pin)
    sort_key = lambda row: (row["rank"], row["episode_index"])
    if sorted(episodes, key=sort_key) != sorted(
        campaign.get("episodes", []), key=sort_key
    ):
        raise RuntimeError(
            "Client task windows differ from complete native episode records"
        )
    counts = [sum(row["rank"] == rank for row in episodes) for rank in range(128)]
    if decision.get("completed_episodes_per_rank") != counts:
        raise RuntimeError(
            "Task stopping rank coverage differs from actual native episodes"
        )
    if host_config["phase"] == "training" and min(counts) < 6:
        raise RuntimeError(
            "Long campaign did not meet minimum actual episode exploration"
        )
    pin(learner["checkpoint"])
    if campaign.get("checkpoint") != learner["checkpoint"]:
        raise RuntimeError("Campaign final checkpoint binding differs")
    return {
        "metrics": expected,
        "stop_decision": decision,
        "completed_training_episodes": len(episodes),
        "training_successes": sum(row["success"] for row in episodes),
        "reward_summaries": summaries,
        "first_geometry_proofs": proofs,
        "campaign_receipt": artifact(output / "learner/campaign.json"),
        "checkpoint": learner["checkpoint"],
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
        "phase": "fasttd3_reward_v2_" + host_config["phase"],
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "claim_scope": "Versioned ABC reward adaptation; original scratch learner, training-task futility, no convergence/paper/R1Lite claim",
        "input_config": artifact(config_path),
        "cpu_admission": config["cpu_admission"],
        "qualification_parent": config["qualification_parent"],
        "qualification_worker": config["qualification_worker"],
        "host_config": config["host_config"],
        "source_pins": config["source_pins"],
        "process_group": os.getpgrp(),
        "gpu_visibility": os.environ["CUDA_VISIBLE_DEVICES"],
        "declared_maximum_calls": host_config["vector_steps"],
        "reward": host_config["reward"],
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
        receipt["memory_admission"] = memory_admission(evidence)
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
                    "abc_bench.fasttd3_campaign_host",
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
                    "sadhana_fasttd3_abc.campaign",
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
    options = parser.parse_args()
    run(options.config, options.output)


if __name__ == "__main__":
    main()
