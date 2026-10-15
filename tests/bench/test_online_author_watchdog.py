"""CPU admission tests for original online training; no learner or GPU is run."""

import argparse
import json
import os
import sys
from pathlib import Path

import pytest

from abc_bench import author_program, runner, subset


def worker_file(tmp_path, *, program="train_pi_robo.py", timeout=21600, schema=1):
    """Only filesystem fixtures; execute_command is replaced in execution tests."""
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"files": []}\n')
    config = {
        "schema_version": schema,
        "program": program,
        "source_root": str(source),
        "source_manifest": str(manifest),
        "python": sys.executable,
        "arguments": [],
        "timeout_s": timeout,
    }
    path = tmp_path / "worker.json"
    path.write_text(json.dumps(config))
    return path


def job_file(
    tmp_path, worker, maximum, *, arguments=None, module="abc_bench.author_program"
):
    job = {
        "schema_version": 1,
        "label": "Explicit CPU-only original online admission fixture",
        "python": sys.executable,
        "module": module,
        "arguments": ["--config", str(worker)] if arguments is None else arguments,
        "max_wall_s": maximum,
    }
    path = tmp_path / "job.json"
    path.write_text(json.dumps(job))
    return path


@pytest.mark.parametrize("timeout", [1, 7100, 7101, 21600, 21600.0])
def test_online_worker_accepts_its_bounded_range(tmp_path, timeout):
    path = worker_file(tmp_path, timeout=timeout)
    assert author_program.load_config(path)["timeout_s"] == timeout


@pytest.mark.parametrize(
    "timeout", [0, -1, True, None, "21600", 21600.1, float("inf"), float("nan")]
)
def test_online_worker_rejects_invalid_or_excessive_timeout(tmp_path, timeout):
    path = worker_file(tmp_path, timeout=timeout)
    with pytest.raises(ValueError, match="timeout_s"):
        author_program.load_config(path)


@pytest.mark.parametrize(
    "program,schema", [("eval_droid_policy.py", 1), ("train_pi_robo.py", 2)]
)
def test_eval_and_catcher_worker_limits_stay_7100(
    tmp_path, program, schema, monkeypatch
):
    # Profile authentication has separate tests. This double isolates timeout admission.
    monkeypatch.setattr(author_program, "validate_catcher_profile", lambda _: None)
    assert (
        author_program.load_config(
            worker_file(tmp_path, program=program, schema=schema, timeout=7100)
        )["timeout_s"]
        == 7100
    )
    with pytest.raises(ValueError, match="timeout_s"):
        author_program.load_config(
            worker_file(tmp_path, program=program, schema=schema, timeout=7101)
        )


@pytest.mark.parametrize(
    "worker_seconds,parent_seconds",
    [(21600, 21700), (7101, 7201), (7100, 7200), (7100, 7201)],
)
def test_online_job_admits_only_with_cleanup_headroom(
    tmp_path, worker_seconds, parent_seconds
):
    worker = worker_file(tmp_path, timeout=worker_seconds)
    path = job_file(tmp_path, worker, parent_seconds)
    assert subset.load_job(path)["max_wall_s"] == parent_seconds


@pytest.mark.parametrize(
    "parent_seconds",
    [120, 7200, 21600, 21699, 21701, None, True, "21700", float("nan"), float("inf")],
)
def test_online_job_rejects_short_or_excessive_parent(tmp_path, parent_seconds):
    path = job_file(tmp_path, worker_file(tmp_path), parent_seconds)
    with pytest.raises(ValueError):
        subset.load_job(path)


def test_long_online_worker_cannot_hide_under_old_parent_cap(tmp_path):
    path = job_file(tmp_path, worker_file(tmp_path, timeout=7101), 7200)
    with pytest.raises(ValueError, match="cleanup"):
        subset.load_job(path)


@pytest.mark.parametrize(
    "program,schema", [("eval_droid_policy.py", 1), ("train_pi_robo.py", 2)]
)
def test_extension_excludes_eval_and_catcher_parents(
    tmp_path, program, schema, monkeypatch
):
    monkeypatch.setattr(author_program, "validate_catcher_profile", lambda _: None)
    worker = worker_file(tmp_path, program=program, schema=schema, timeout=7100)
    with pytest.raises(ValueError, match="only for original schema1 author training"):
        subset.load_job(job_file(tmp_path, worker, 7201))


def test_unrelated_worker_does_not_inherit_online_parent_cap(tmp_path):
    worker = worker_file(tmp_path)
    path = job_file(tmp_path, worker, 21700, module="abc_bench.subset_worker")
    with pytest.raises(ValueError, match="at most 7200"):
        subset.load_job(path)


def test_extended_online_arguments_require_one_absolute_config(tmp_path, monkeypatch):
    worker = worker_file(tmp_path)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    for arguments in (
        ["--config", worker.name],
        ["--config", str(worker), "--extra"],
        ["--config", str(worker), "--config", str(worker)],
    ):
        with pytest.raises(ValueError):
            subset.load_job(job_file(tmp_path, worker, 21700, arguments=arguments))
    path = job_file(tmp_path, worker, 21700, arguments=[f"--config={worker}"])
    assert subset.load_job(path)["max_wall_s"] == 21700


@pytest.mark.parametrize(
    "cli_timeout", [0, 7200, 21699, True, None, "21700", float("nan"), float("inf")]
)
def test_cli_reduction_refuses_before_ledger_or_gpu(tmp_path, monkeypatch, cli_timeout):
    path = job_file(tmp_path, worker_file(tmp_path), 21700)
    results = tmp_path / "must-not-exist"
    monkeypatch.setattr(runner, "RESULTS", results)
    monkeypatch.setattr(
        subset,
        "inspect_reserved_gpu",
        lambda _: pytest.fail("GPU inspection must not run"),
    )
    with pytest.raises(ValueError, match="Effective author training parent"):
        subset.run_job(
            argparse.Namespace(training_config=path, timeout_seconds=cli_timeout)
        )
    assert not results.exists()


@pytest.mark.parametrize("cli_timeout", [21700, 40000])
def test_effective_limit_is_bounded_and_prior_charges_survive(
    tmp_path, monkeypatch, cli_timeout
):
    """Explicit process/GPU doubles exercise canonical accounting and receipts."""
    path = job_file(tmp_path, worker_file(tmp_path), 21700)
    results = tmp_path / "results"
    results.mkdir()
    authorization = {"mode": "explicit_cpu_fixture_uncapped_gpu0"}
    ledger_path = results / "gpu_budget.json"
    ledger_path.write_text(
        json.dumps(
            {
                "gpu": 0,
                "charged_seconds": 17,
                "limit_seconds": 30000,
                "authorization": authorization,
            }
        )
    )
    monkeypatch.setattr(runner, "RESULTS", results)
    monkeypatch.setattr(
        subset,
        "inspect_reserved_gpu",
        lambda ordinal: {"ordinal": ordinal, "uuid": "explicit-cpu-fixture"},
    )
    calls = []

    def execute_double(command, log, timeout, *, gpu_uuid):
        calls.append(timeout)
        assert timeout == 21700
        assert gpu_uuid == "explicit-cpu-fixture"
        ledger = json.loads(ledger_path.read_text())
        assert ledger["charged_seconds"] == 21717
        assert ledger["active_parent_pid"] == os.getpid()
        assert ledger["authorization"] == authorization
        out = Path(command[-1])
        (out / "receipt.json").write_text('{"status": "completed", "metrics": {}}\n')
        return 0

    monkeypatch.setattr(runner, "execute_command", execute_double)
    receipt = subset.run_job(
        argparse.Namespace(training_config=path, timeout_seconds=cli_timeout)
    )
    assert calls == [21700]
    assert receipt["status"] == "completed"
    ledger = json.loads(ledger_path.read_text())
    assert 17 <= ledger["charged_seconds"] < 21717
    assert ledger["limit_seconds"] == 30000
    assert ledger["authorization"] == authorization
    assert "active_parent_pid" not in ledger
    assert "active_gpu_uuid" not in ledger


@pytest.mark.parametrize(
    "change,error",
    [
        ({"gpu": 1}, "GPU 0"),
        ({"active_parent_pid": 17}, "active or unresolved"),
        ({"active_gpu_uuid": "explicit-other-owner"}, "active or unresolved"),
        ({"active_visual_lease": "explicit-other-owner"}, "active or unresolved"),
        ({"charged_seconds": float("nan")}, "finite nonnegative"),
        ({"limit_seconds": float("inf")}, "finite nonnegative"),
        ({"limit_seconds": 21716}, "Insufficient finite"),
    ],
)
def test_extension_keeps_owner_and_finite_credit_guards(
    tmp_path, monkeypatch, change, error
):
    path = job_file(tmp_path, worker_file(tmp_path), 21700)
    results = tmp_path / "results"
    results.mkdir()
    ledger_path = results / "gpu_budget.json"
    ledger_path.write_text(
        json.dumps({"gpu": 0, "charged_seconds": 17, "limit_seconds": 30000, **change})
    )
    original_bytes = ledger_path.read_bytes()
    monkeypatch.setattr(runner, "RESULTS", results)
    monkeypatch.setattr(
        subset,
        "inspect_reserved_gpu",
        lambda _: pytest.fail("GPU inspection must not run"),
    )
    with pytest.raises((ValueError, RuntimeError), match=error):
        subset.run_job(argparse.Namespace(training_config=path, timeout_seconds=21700))
    assert ledger_path.read_bytes() == original_bytes
