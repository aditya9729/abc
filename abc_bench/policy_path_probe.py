"""Admit one fixed original EXPO inference identity diagnostic."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

from abc_bench import failure_candidate_bank, fixed_bank_scoring
from abc_bench import realtime_update_probe as supervision

MODULE = "abc_bench.policy_path_probe"
PROFILE = "r1lite-policy-path-identity"
SCHEMA = "sadhana.policy_path_canonical/1"
REPO = Path("/home/user/aditya/RL/abc")
BUILDS = Path("/home/user/aditya/RL/builds")
GPU_UUID = "GPU-50925bdd-bd56-4b30-4635-00cee7f5c95b"
WORKER_SECONDS = 1800
PARENT_SECONDS = 1900
PAYLOAD_MANIFEST = {
    "path": "/home/user/aditya/RL/builds/r1lite-realtime-expo-20261014/policy-path-equivalence-caller-preparation-v2-20261015/PAYLOAD_MANIFEST.json",
    "bytes": 1178,
    "sha256": "63cc92d1a8565fc52bfd709129279d7e996ae1cbf0ee66c86465d1d43cc8b869",
}
BANK_PAYLOAD_MANIFEST = {
    "path": "/home/user/aditya/RL/builds/r1lite-realtime-expo-20261014/failure-candidate-bank-preparation-20261016/PAYLOAD_MANIFEST.json",
    "bytes": 1467,
    "sha256": "1fb2d9a5aa3a84ca830503bc8039660785df7324ac89909f54dc1e3c41bff051",
}

ACQUISITION_PAYLOAD_MANIFEST = {
    "path": "/home/user/aditya/RL/builds/r1lite-realtime-expo-20261014/acquisition-bank-preparation-20261019/PAYLOAD_MANIFEST.json",
    "bytes": 1436,
    "sha256": "2e032980171ff155676caff1466362035b65ec7d1d5bbae9b39986c1c9338667",
}

SCORING_PAYLOAD_MANIFEST = {
    "path": "/home/user/aditya/RL/builds/r1lite-realtime-expo-20261014/fixed-bank-scoring-preparation-20261019-05/PAYLOAD_MANIFEST.json",
    "bytes": 1710,
    "sha256": "4c5fdbebe20e84d4c38b361734741513ea0cdc7d3c6ef7acaaf423c03668a8f1",
}


def payload_binding(schema):
    if schema == fixed_bank_scoring.CONFIG_SCHEMA:
        return SCORING_PAYLOAD_MANIFEST
    if schema == failure_candidate_bank.ACQUISITION_CONFIG_SCHEMA:
        return ACQUISITION_PAYLOAD_MANIFEST
    if schema == failure_candidate_bank.CONFIG_SCHEMA:
        return BANK_PAYLOAD_MANIFEST
    return PAYLOAD_MANIFEST


def bank_mode(schema):
    return schema in (
        failure_candidate_bank.CONFIG_SCHEMA,
        failure_candidate_bank.ACQUISITION_CONFIG_SCHEMA,
    )


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fresh_output(value):
    path = Path(value)
    require(
        path.is_absolute()
        and path == path.resolve()
        and path.is_relative_to(BUILDS)
        and path != BUILDS
        and not path.exists()
        and not path.is_symlink(),
        "Require a fresh diagnostic output under RL/builds",
    )
    return path


def payload(binding=PAYLOAD_MANIFEST):
    manifest = json.loads(supervision.authenticate(binding).read_text())
    for item in manifest["files"]:
        supervision.authenticate(item)
    config = json.loads(supervision.authenticate(manifest["config"]).read_text())
    for item in config["installed_sources"] + config["inputs"]:
        supervision.authenticate(item)
    python = Path(config["python"]["path"])
    supervision.authenticate(config["python"])
    require(
        python == python.resolve()
        and python.is_relative_to(Path("/home/user/aditya"))
        and not python.is_symlink(),
        "Require the qualified copied numerical interpreter",
    )
    return manifest, config


def load_config(path):
    require(
        path.is_absolute() and path == path.resolve() and path.is_file(),
        "Absolute configuration required",
    )
    config = json.loads(path.read_text())
    require(
        set(config) == {"schema", "output"}
        and config["schema"]
        in (
            SCHEMA,
            failure_candidate_bank.CONFIG_SCHEMA,
            failure_candidate_bank.ACQUISITION_CONFIG_SCHEMA,
            fixed_bank_scoring.CONFIG_SCHEMA,
        )
        and isinstance(config["output"], str),
        "Only the fixed identity diagnostic or fixed failure bank is admitted",
    )
    fresh_output(config["output"])
    if config["schema"] == fixed_bank_scoring.CONFIG_SCHEMA:
        _, inputs = payload(SCORING_PAYLOAD_MANIFEST)
        fixed_bank_scoring.validate_fixed(inputs)
    elif bank_mode(config["schema"]):
        _, inputs = payload(payload_binding(config["schema"]))
        failure_candidate_bank.validate_fixed(inputs)
        expected = (
            failure_candidate_bank.ACQUISITION_INPUT_SCHEMA
            if config["schema"] == failure_candidate_bank.ACQUISITION_CONFIG_SCHEMA
            else failure_candidate_bank.INPUT_SCHEMA
        )
        require(
            inputs["schema"] == expected, "Payload profile differs from requested mode"
        )
    else:
        payload()
    return config


def validate_job(job, effective_seconds=None):
    from abc_bench.runner import RESULTS, ROOT

    require(
        ROOT == REPO and RESULTS == REPO / "outputs/bench", "Use the shared ABC ledger"
    )
    require(
        job.get("module") == MODULE
        and type(job.get("max_wall_s")) is int
        and job["max_wall_s"] == PARENT_SECONDS
        and job.get("task") == "r1lite_remote_into_organizer"
        and job.get("embodiment") == "r1lite_2025"
        and isinstance(job.get("arguments"), list)
        and len(job["arguments"]) == 2
        and job["arguments"][0] == "--config",
        "Require the fixed identity profile and 1900-second parent",
    )
    python = Path(job["python"])
    require(
        python == python.resolve()
        and python.is_relative_to(Path("/home/user/aditya"))
        and python == Path(sys.executable).resolve(),
        "Use this normally installed canonical interpreter",
    )
    config = load_config(Path(job["arguments"][1]))
    if effective_seconds is not None:
        require(
            type(effective_seconds) in (int, float)
            and math.isfinite(effective_seconds)
            and min(job["max_wall_s"], effective_seconds) == PARENT_SECONDS,
            "Effective deadline must retain the100-second margin",
        )
        require(
            supervision.host_available() >= supervision.MIN_START_BYTES,
            "Host available memory below160GiB",
        )
    return config


def diagnostic_command(config, manifest, inputs):
    binding = payload_binding(config.get("schema"))
    return [
        inputs["python"]["path"],
        "-I",
        "-u",
        "-B",
        manifest["caller"]["path"],
        "--manifest",
        binding["path"],
        "--manifest-sha256",
        binding["sha256"],
        "--output",
        config["output"],
    ]


def validate_result(value, mode=SCHEMA):
    if mode == fixed_bank_scoring.CONFIG_SCHEMA:
        fixed_bank_scoring.validate_result(value)
        require(
            value.get("source_manifest") == SCORING_PAYLOAD_MANIFEST,
            "Scoring payload differs",
        )
        return
    if bank_mode(mode):
        failure_candidate_bank.validate_result(value)
        require(
            value.get("schema")
            == (
                failure_candidate_bank.ACQUISITION_RESULT_SCHEMA
                if mode == failure_candidate_bank.ACQUISITION_CONFIG_SCHEMA
                else failure_candidate_bank.RESULT_SCHEMA
            ),
            "Result profile differs",
        )
        require(
            value.get("source_manifest") == payload_binding(mode),
            "Failure-bank payload differs",
        )
        return
    require(
        value.get("schema") == "sadhana.policy_path_identity_result/1"
        and value.get("status") == "policy_path_diagnostic_complete"
        and value.get("artifact_accepted") is False
        and value.get("benchmark_accepted") is False
        and type(value.get("optimizer_steps_performed")) is int
        and value["optimizer_steps_performed"] == 0
        and type(value.get("native_controls")) is int
        and value["native_controls"] == 0
        and value.get("source_manifest") == PAYLOAD_MANIFEST
        and [case["case"] for case in value.get("cases", [])] == ["boot", "prefix"],
        "Require the completed two-case inference-only result; identity findings remain root-reviewed",
    )


def run(config_path, output):
    from abc_bench.subset import require_subset_lease

    started = time.monotonic()
    require_subset_lease()
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == GPU_UUID
        and os.environ.get("ABC_BENCH_REPO") == str(REPO)
        and os.getpgrp() == os.getpid(),
        "Require leased GPU0 and canonical process group",
    )
    config = load_config(config_path)
    is_bank = bank_mode(config["schema"])
    binding = payload_binding(config["schema"])
    manifest, inputs = payload(binding)
    result = {
        "schema_version": 1,
        "status": "failed",
        "phase": (
            fixed_bank_scoring.PROFILE
            if config["schema"] == fixed_bank_scoring.CONFIG_SCHEMA
            else failure_candidate_bank.ACQUISITION_PROFILE
            if config["schema"] == failure_candidate_bank.ACQUISITION_CONFIG_SCHEMA
            else failure_candidate_bank.PROFILE
            if is_bank
            else PROFILE
        ),
        "task": "r1lite_remote_into_organizer",
        "embodiment": "r1lite_2025",
        "experiment_accepted": False,
        "benchmark_accepted": False,
        "metrics": {},
        "config": supervision.file_pin(config_path),
        "payload_manifest": binding,
        "worker_seconds": WORKER_SECONDS,
        "parent_seconds": PARENT_SECONDS,
        "optimizer_steps_performed": 0,
        "native_controls": 0,
    }
    cancelled = [False]

    def cancel(signum, frame):
        cancelled[0] = True

    previous = {
        sig: signal.signal(sig, cancel) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        env = supervision.memory_environment({"environment": inputs["environment"]})
        env["ABC_BENCH_REPO"] = str(REPO)
        env["JAX_COMPILATION_CACHE_DIR"] = str(output / "jax-cache")
        child = supervision.watch_child(
            diagnostic_command(config, manifest, inputs),
            env,
            output,
            started + WORKER_SECONDS,
            cancelled,
        )
        result["child"] = child
        require(
            child["child_exit_code"] == 0
            and child["child_reaped"] is True
            and child["stop_reason"] is None
            and not cancelled[0],
            "Diagnostic child did not complete cleanly",
        )
        result_path = Path(config["output"]) / "RESULT.json"
        value = json.loads(result_path.read_text())
        validate_result(value, config["schema"])
        for expected in value["arrays"].values():
            if expected is not None:
                supervision.authenticate(
                    {key: expected[key] for key in ("path", "bytes", "sha256")}
                )
        payload(binding)
        result.update(
            status="completed",
            diagnostic_result=supervision.file_pin(result_path),
            scope="Fixed-bank scoring completed; root must assess ranking changes."
            if config["schema"] == fixed_bank_scoring.CONFIG_SCHEMA
            else "Failure-bank inference completed; branch outcomes remain unexecuted."
            if is_bank
            else "Inference completed; root must assess identity discrepancies. No task outcome.",
        )
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        result["elapsed_seconds"] = time.monotonic() - started
        supervision.write_json(output / "receipt.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == "__main__":
    main()
