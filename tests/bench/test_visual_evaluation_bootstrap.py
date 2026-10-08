"""Owned CPU lifetimes and fake leases only; no GPU/native execution."""

import copy
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from test_visual_evaluation_dispatch import args
from test_visual_evaluation_dispatch import request_factory as _request_factory

from abc_bench import runner
from abc_bench import visual_evaluation_bootstrap as bootstrap
from abc_bench import visual_evaluation_dispatch as visual

request_factory = _request_factory


CONTROLLER = r"""
import json,sys
from pathlib import Path
from abc_bench import runner,visual_evaluation_dispatch as visual
campaign=Path(sys.argv[4]);visual.CANONICAL_CAMPAIGN=campaign;runner.ROOT=campaign.parent.parent
runner.inspect_reserved_gpu=lambda ordinal:{'uuid':'','scope':'owned CPU fixture; no GPU query'}
sys.argv=['abc-bench','--algorithm','visual-method-evaluation','--evaluation-request',sys.argv[1],'--evaluation-request-sha256',sys.argv[2],'--evaluation-request-bytes',sys.argv[3],'--results',str(campaign),'--timeout-seconds',sys.argv[5]]
runner.main()
"""


def wait_path(path, seconds=5):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if path.exists():
            return json.loads(path.read_text())
        time.sleep(0.01)
    raise TimeoutError(str(path))


def wait_run(campaign, seconds=5):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        paths = list(campaign.glob("visual-method-evaluation-*"))
        if paths:
            return paths[0]
        time.sleep(0.01)
    raise TimeoutError("Owned CPU runner did not start")


def reap_owned(ready, deadline):
    while time.monotonic() < deadline:
        try:
            while os.waitpid(-ready["worker"]["pgid"], os.WNOHANG)[0]:
                pass
        except ChildProcessError:
            pass
        try:
            os.waitpid(ready["guardian"]["pid"], os.WNOHANG)
        except ChildProcessError:
            pass
        if not any(
            bootstrap.identity_matches(ready[k]) for k in ("worker", "guardian")
        ):
            return
        time.sleep(0.01)
    raise TimeoutError("Owned CPU descendants did not terminate/reap")


@pytest.mark.parametrize("field", ["start_ticks", "boot_id", "pgid"])
def test_start_identity_alias_never_signals_an_owned_reused_pid(monkeypatch, field):
    token = bootstrap.process_identity(os.getpid())
    token[field] = token[field] + 1 if isinstance(token[field], int) else "another-boot"
    calls = []
    monkeypatch.setattr(os, "kill", lambda *values: calls.append(values))
    monkeypatch.setattr(os, "killpg", lambda *values: calls.append(values))
    assert not bootstrap.signal_owned(token, signal.SIGTERM)
    assert not bootstrap.signal_owned(token, signal.SIGKILL, group=True)
    assert calls == []


def test_hard_parent_death_retains_original_lock_until_owned_group_is_gone(
    request_factory, tmp_path
):
    previous = bootstrap.subreaper(1)
    pin, _ = request_factory(seed=915)
    command = [
        sys.executable,
        "-B",
        "-c",
        CONTROLLER,
        pin["path"],
        pin["sha256"],
        str(pin["bytes"]),
        str(request_factory.campaign),
        "12",
    ]
    ready = None
    parent = None
    try:
        with (tmp_path / "controller.log").open("xb") as log:
            parent = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            run = wait_run(request_factory.campaign)
            ready = wait_path(run / "control/guardian-ready.json")
            armed = wait_path(run / "control/parent-armed.json")
            sleeping = wait_path(run / "evaluation/sleep-ready.json")
            descendant = bootstrap.process_identity(sleeping["descendant_pid"])
            assert descendant["pgid"] == ready["worker"]["pgid"]
            ledger = json.loads(
                (request_factory.campaign / "gpu_budget.json").read_text()
            )
            assert armed["parent"] == bootstrap.process_identity(parent.pid)
            assert ledger["active_visual_lease"]["worker"] == ready["worker"]
            assert ledger["active_visual_lease"]["guardian"] == ready["guardian"]
            assert (
                ledger["active_visual_lease"]["worker_pgid"] == ready["worker"]["pgid"]
            )
            with (
                (request_factory.campaign / ".gpu-budget.lock").open("rb") as lock,
                pytest.raises(BlockingIOError),
            ):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            token = bootstrap.process_identity(parent.pid)
            stale = copy.deepcopy(token)
            stale["start_ticks"] += 1
            assert not bootstrap.signal_owned(stale, signal.SIGKILL)
            assert parent.poll() is None
            assert bootstrap.signal_owned(token, signal.SIGKILL)
            parent.wait(timeout=3)
            assert parent.returncode == -signal.SIGKILL
            deadline = time.monotonic() + 4
            while (
                bootstrap.identity_matches(ready["worker"], live=True)
                and time.monotonic() < deadline
            ):
                with (
                    (request_factory.campaign / ".gpu-budget.lock").open("rb") as lock,
                    pytest.raises(BlockingIOError),
                ):
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                time.sleep(0.01)
            reap_owned(ready, deadline)
            outcome = wait_path(run / "control/guardian-outcome.json")
            after = json.loads(
                (request_factory.campaign / "gpu_budget.json").read_text()
            )
            assert (
                outcome["status"] == "unresolved" and outcome["reason"] == "parent_died"
            )
            assert after["charged_seconds"] == ledger["charged_seconds"]
            assert after["active_visual_lease"]["status"] == "unresolved"
            assert after["active_parent_pid"] == parent.pid
            with (request_factory.campaign / ".gpu-budget.lock").open("rb") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            assert not (run / "evaluation/receipt.json").exists()
            assert not bootstrap.identity_matches(descendant)
    finally:
        if parent is not None and parent.poll() is None:
            bootstrap.signal_owned(
                bootstrap.process_identity(parent.pid), signal.SIGKILL
            )
            parent.wait(timeout=3)
        if ready is not None:
            if bootstrap.identity_matches(ready["worker"]):
                bootstrap.signal_owned(ready["worker"], signal.SIGKILL, group=True)
            if bootstrap.identity_matches(ready["guardian"]):
                bootstrap.signal_owned(ready["guardian"], signal.SIGKILL)
            reap_owned(ready, time.monotonic() + 3)
        bootstrap.subreaper(previous)


def test_watchdog_is_finite_and_retains_unresolved_reservation_without_score(
    request_factory, tmp_path
):
    pin, _ = request_factory(seed=915)
    command = [
        sys.executable,
        "-B",
        "-c",
        CONTROLLER,
        pin["path"],
        pin["sha256"],
        str(pin["bytes"]),
        str(request_factory.campaign),
        "4",
    ]
    with (tmp_path / "watchdog.log").open("xb") as log:
        start = time.monotonic()
        result = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, timeout=9, check=False
        )
    assert result.returncode == 1 and time.monotonic() - start < 8
    run = wait_run(request_factory.campaign)
    receipt = json.loads(
        (request_factory.campaign / "runs" / (run.name + ".json")).read_text()
    )
    assert (
        receipt["status"] == "failed" and "producer_evaluation_metrics" not in receipt
    )
    assert receipt["charge_settlement"] == "full_reservation_unresolved_no_refund"
    after = json.loads((request_factory.campaign / "gpu_budget.json").read_text())
    assert (
        after["charged_seconds"]
        == 10 + after["active_visual_lease"]["reserved_seconds"]
    )
    ready = json.loads((run / "control/guardian-ready.json").read_text())
    assert not any(bootstrap.identity_matches(ready[k]) for k in ("worker", "guardian"))


def test_receipt_mutation_and_failed_resume_are_not_promoted(request_factory):
    pin, _ = request_factory(seed=913)
    result = runner.run_baseline(args(request_factory, pin))
    pin, _ = request_factory(seed=913, resume=[result["worker_receipt"]])
    with pytest.raises(ValueError, match="Failed invocation"):
        visual.load_request(pin, request_factory.campaign)


def test_request_pin_is_rechecked_before_dispatch_and_receipt_read_is_bounded(
    request_factory,
):
    pin, _ = request_factory()
    path = Path(pin["path"])
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="pin changed"):
        runner.run_baseline(args(request_factory, pin))
    assert request_factory.calls == []


@pytest.mark.parametrize("raw", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}', b"[]"])
def test_duplicate_nonfinite_nonobject_json_denied(tmp_path, raw):
    path = tmp_path / "invalid.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        bootstrap.read_json(path)
    with pytest.raises(ValueError, match="byte bound"):
        bootstrap.read_json(path, maximum=1)


def test_first_launch_failure_survives_outcome_publication_failure(
    monkeypatch, tmp_path
):
    first = RuntimeError("owned fixture first launch failure")
    previous_reaper = bootstrap.subreaper()
    previous_term = signal.getsignal(signal.SIGTERM)

    def no_launch(*args, **kwargs):
        raise first

    def no_publication(*args, **kwargs):
        raise OSError("owned fixture outcome publication failure")

    monkeypatch.setattr(subprocess, "Popen", no_launch)
    monkeypatch.setattr(bootstrap, "exclusive_json", no_publication)
    command = ["not-executed-fixture"]
    spec = {"bootstrap_command": command, "control_directory": str(tmp_path)}
    with pytest.raises(RuntimeError) as failure:
        bootstrap.execute_guarded(
            command,
            tmp_path / "not-launched.log",
            1,
            gpu_uuid="",
            spec=spec,
            lease_fd=-1,
        )
    assert failure.value is first
    assert first.__notes__ == [
        "execution outcome publication: owned fixture outcome publication failure"
    ]
    assert bootstrap.subreaper() == previous_reaper
    assert signal.getsignal(signal.SIGTERM) == previous_term
