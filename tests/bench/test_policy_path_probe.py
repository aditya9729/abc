"""Pure boundary checks; explicit arrays/process doubles, no model or GPU."""

import copy
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from abc_bench import policy_path_probe as worker

PREPARATION = Path(
    "/home/user/aditya/RL/builds/r1lite-realtime-expo-20261014/policy-path-equivalence-caller-preparation-v2-20261015"
)
spec = importlib.util.spec_from_file_location(
    "identity_diagnostic_boundary", PREPARATION / "identity_path.py"
)
caller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(caller)


def test_fixed_contract_and_bool_refusal():
    config = json.loads((PREPARATION / "PAYLOAD_CONFIG.json").read_text())
    caller.validate_fixed(config)
    for key, value in [
        ("N", 1),
        ("delay", 0),
        ("noise_seed", 42),
        ("observation_indices", [False, 3, 8]),
    ]:
        wrong = copy.deepcopy(config)
        wrong["fixed"][key] = value
        with pytest.raises(ValueError):
            caller.validate_fixed(wrong)


def test_pin_tampering_and_symlink(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"original")
    expected = caller.pin(source)
    caller.authenticate(expected)
    source.write_bytes(b"changed!")
    with pytest.raises(ValueError, match="pin differs"):
        caller.authenticate(expected)
    alias = tmp_path / "alias"
    alias.symlink_to(source)
    with pytest.raises(ValueError, match="Regular absolute"):
        caller.pin(alias)


def test_difference_distinguishes_arithmetic_dtype_shape_and_nonfinite():
    values = np.array([0.1, 0.2], dtype=np.float32)
    assert caller.difference(values, values.copy(), np)["bitwise_equal"]
    shifted = values.copy()
    shifted[0] = np.nextafter(shifted[0], np.float32(1))
    result = caller.difference(values, shifted, np)
    assert not result["bitwise_equal"] and result["max_abs"] > 0
    assert result["first_value_difference"]["index"] == [0]
    assert not caller.difference(values, values.astype(np.float64), np)["bitwise_equal"]
    assert caller.difference(values, values[:1], np)["max_abs"] is None
    assert not caller.difference(values, np.array([np.nan, 0.2]), np)["finite"]


def test_recorder_roundtrip_and_exact_vs_arithmetic(tmp_path):
    record = caller.Recorder(
        tmp_path, np, SimpleNamespace(device_get=lambda value: value)
    )
    value = np.arange(6, dtype=np.float32).reshape(2, 3)
    record.array("raw/input", value)
    np.testing.assert_array_equal(
        np.load(record.arrays["raw/input"]["path"], allow_pickle=False), value
    )
    record.compare("fixed_bank", value, value, exact=True)
    record.compare("batch_arithmetic", value, value + 0.01, exact=False)
    assert record.comparisons[0]["exact_route_check"]
    assert not record.comparisons[1]["exact_route_check"]
    with pytest.raises(ValueError, match="Duplicate"):
        record.array("raw/input", value)
    with pytest.raises(ValueError, match="Nonfinite"):
        record.array("bad", np.array([np.inf]))


def test_fresh_output_and_arbitrary_config_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "BUILDS", tmp_path)
    assert worker.fresh_output(str(tmp_path / "fresh")) == tmp_path / "fresh"
    with pytest.raises(ValueError):
        worker.fresh_output(str(tmp_path))
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ValueError):
        worker.fresh_output(str(existing))
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "schema": worker.SCHEMA,
                "output": str(tmp_path / "new"),
                "command": ["echo"],
            }
        )
    )
    monkeypatch.setattr(
        worker, "payload", lambda: pytest.fail("Invalid config reached payload")
    )
    with pytest.raises(ValueError, match="fixed identity"):
        worker.load_config(config)


def valid_job():
    return {
        "module": worker.MODULE,
        "python": str(Path(sys.executable).resolve()),
        "max_wall_s": 1900,
        "task": "r1lite_remote_into_organizer",
        "embodiment": "r1lite_2025",
        "arguments": ["--config", "/fixed/config.json"],
    }


def test_exact_job_and_effective_parent_margin(monkeypatch):
    monkeypatch.setattr(worker, "load_config", lambda path: {"output": "/fixed/output"})
    monkeypatch.setattr(
        worker.supervision, "host_available", lambda: worker.supervision.MIN_START_BYTES
    )
    assert worker.validate_job(valid_job(), 1900)["output"] == "/fixed/output"
    for change in (
        {"max_wall_s": 1800},
        {"module": "arbitrary"},
        {"task": "other"},
        {"arguments": ["--config", "/fixed/config.json", "--override"]},
    ):
        with pytest.raises(ValueError):
            worker.validate_job({**valid_job(), **change}, 1900)
    for deadline in (1899, float("nan"), True):
        with pytest.raises(ValueError):
            worker.validate_job(valid_job(), deadline)


def test_fixed_command_cannot_select_another_payload():
    manifest = {"caller": {"path": "/fixed/identity_path.py"}}
    inputs = {"python": {"path": "/fixed/python"}}
    result = worker.diagnostic_command({"output": "/fresh/output"}, manifest, inputs)
    assert result[:5] == ["/fixed/python", "-I", "-u", "-B", "/fixed/identity_path.py"]
    assert result[5:] == [
        "--manifest",
        worker.PAYLOAD_MANIFEST["path"],
        "--manifest-sha256",
        worker.PAYLOAD_MANIFEST["sha256"],
        "--output",
        "/fresh/output",
    ]


def result_value():
    return {
        "schema": "sadhana.policy_path_identity_result/1",
        "status": "policy_path_diagnostic_complete",
        "artifact_accepted": False,
        "benchmark_accepted": False,
        "optimizer_steps_performed": 0,
        "native_controls": 0,
        "source_manifest": worker.PAYLOAD_MANIFEST,
        "cases": [{"case": "boot"}, {"case": "prefix"}],
        "arrays": {},
        "first_exact_route_discrepancy": {"name": "explicit diagnostic finding"},
    }


def test_execution_result_does_not_claim_identity_or_task_acceptance():
    worker.validate_result(result_value())
    for field, value in [
        ("artifact_accepted", True),
        ("native_controls", 1),
        ("optimizer_steps_performed", True),
        ("source_manifest", {}),
        ("cases", [{"case": "boot"}]),
    ]:
        wrong = result_value()
        wrong[field] = value
        with pytest.raises(ValueError):
            worker.validate_result(wrong)


def test_worker_reuses_existing_watcher_and_records_completed_inference(
    tmp_path, monkeypatch
):
    from abc_bench import subset

    target = tmp_path / "diagnostic"
    target.mkdir()
    out = tmp_path / "worker"
    out.mkdir()
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    monkeypatch.setattr(subset, "require_subset_lease", lambda: None)
    monkeypatch.setattr(worker.os, "getpgrp", lambda: os.getpid())
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", worker.GPU_UUID)
    monkeypatch.setenv("ABC_BENCH_REPO", str(worker.REPO))
    monkeypatch.setattr(
        worker,
        "load_config",
        lambda path: {"schema": worker.SCHEMA, "output": str(target)},
    )
    monkeypatch.setattr(
        worker,
        "payload",
        lambda binding: (
            {"caller": {"path": "/fixed/caller.py"}},
            {"python": {"path": "/fixed/python"}, "environment": {}},
        ),
    )
    monkeypatch.setattr(worker.supervision, "memory_environment", lambda config: {})
    observed = []

    def watch(command, env, output, deadline, cancelled):
        observed.append((command, env, output, deadline, cancelled))
        (target / "RESULT.json").write_text(json.dumps(result_value()))
        return {"child_exit_code": 0, "child_reaped": True, "stop_reason": None}

    monkeypatch.setattr(worker.supervision, "watch_child", watch)
    value = worker.run(config_path, out)
    assert len(observed) == 1 and observed[0][2] == out
    assert observed[0][1]["ABC_BENCH_REPO"] == str(worker.REPO)
    assert value["status"] == "completed" and value["experiment_accepted"] is False
    assert json.loads((out / "receipt.json").read_text())["native_controls"] == 0
