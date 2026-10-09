"""Owner regressions adapted from sealed peer faults; only stdlib doubles."""

import base64
import csv
import hashlib
import io
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from itertools import combinations
from pathlib import Path

import pytest
from test_visual_evaluation_dispatch import args, write
from test_visual_evaluation_dispatch import request_factory as _factory

from abc_bench import runner
from abc_bench import visual_evaluation_bootstrap as boot

request_factory = _factory
SIGNALS = boot.CANCELLATION_SIGNALS
SETS = [p for size in range(1, len(SIGNALS) + 1) for p in combinations(SIGNALS, size)]


@contextmanager
def caller_state():
    handlers = {s: signal.getsignal(s) for s in SIGNALS}
    mask = signal.pthread_sigmask(signal.SIG_BLOCK, [])
    timer = signal.getitimer(signal.ITIMER_REAL)
    reaper = boot.subreaper()
    try:
        yield handlers, mask, timer, reaper
    finally:
        signal.pthread_sigmask(signal.SIG_BLOCK, SIGNALS)
        for s in SIGNALS:
            if s in signal.sigpending():
                signal.sigwait({s})
        signal.setitimer(signal.ITIMER_REAL, 0)
        for s, h in handlers.items():
            signal.signal(s, h)
        signal.pthread_sigmask(signal.SIG_SETMASK, mask)
        if timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *timer)
        boot.subreaper(reaper)


def state_code(marker):
    return (
        "import json,os,signal;from pathlib import Path;"
        + f"Path({str(marker)!r}).write_text(json.dumps("
        + "{'pid':os.getpid(),'ignored':[int(s) for s in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM) if signal.getsignal(s)==signal.SIG_IGN],"
        + "'blocked':[int(s) for s in signal.pthread_sigmask(signal.SIG_BLOCK,[])]}));\n"
    )


def instrument_worker(value, marker):
    """Reseal ONLY the explicit owned NRH double and its synthetic trust metadata."""
    worker = value["worker"]
    path = Path(worker["nrh_files"]["nrh/visual_method_evaluation.py"]["path"])
    path.write_text(state_code(marker) + path.read_text())
    worker["nrh_files"]["nrh/visual_method_evaluation.py"] = boot.artifact(path)
    site = Path(worker["site_packages"])
    records = list(csv.reader(Path(worker["record"]["path"]).read_text().splitlines()))
    for row in records:
        if row[1]:
            raw = (site / row[0]).read_bytes()
            row[1] = "sha256=" + base64.urlsafe_b64encode(
                hashlib.sha256(raw).digest()
            ).decode().rstrip("=")
            row[2] = str(len(raw))
    stream = io.StringIO()
    csv.writer(stream).writerows(records)
    Path(worker["record"]["path"]).write_text(stream.getvalue())
    worker["record"] = boot.artifact(worker["record"]["path"])
    software = boot.read_json(value["software_binding"]["path"])
    software["nrh_sources"] = {n: p["sha256"] for n, p in worker["nrh_files"].items()}
    value["software_binding"] = write(Path(value["software_binding"]["path"]), software)
    value["identity"]["evaluator_sources"] = {
        n: worker["nrh_files"]["nrh/" + n + ".py"]["sha256"]
        for n in (
            "visual_policy_export",
            "visual_method_evaluation",
            "evaluation_video",
            "qf3_visual_policy",
        )
    }
    native = boot.read_json(value["native_admission"]["path"])
    native.update(
        worker_binding=worker,
        evaluator_sources=value["identity"]["evaluator_sources"],
        fixture_identity=value["identity"],
        protocol_id=hashlib.sha256(boot.canonical(value["identity"])).hexdigest(),
    )
    value["native_admission"] = write(Path(value["native_admission"]["path"]), native)


@pytest.mark.parametrize("route", ["probe", "dispatch"])
@pytest.mark.parametrize("signals", SETS)
@pytest.mark.parametrize("kind", ["ignored", "blocked"])
def test_actual_nested_route_preserves_child_signal_state(
    request_factory, tmp_path, monkeypatch, route, signals, kind
):
    pin, value = request_factory()
    marker = tmp_path / "ACTUAL_CHILD_STATE.json"
    real_popen = subprocess.Popen
    children = []

    def spawn(argv, **kw):
        actual = argv
        if route == "probe" and argv[-1] == boot.INTERPRETER_PROBE:
            actual = [*argv[:-1], state_code(marker) + argv[-1]]
        child = real_popen(actual, **kw)
        children.append(
            {
                "token": boot.process_identity(child.pid),
                "original_argv": argv,
                "probe_metadata_instrumented": actual is not argv,
            }
        )
        return child

    if route == "dispatch":
        instrument_worker(value, marker)
        pin = write(Path(pin["path"]), value)
    monkeypatch.setattr(subprocess, "Popen", spawn)
    with caller_state():
        if kind == "ignored":
            for s in signals:
                signal.signal(s, signal.SIG_IGN)
            signal.pthread_sigmask(signal.SIG_BLOCK, [signal.SIGUSR1])
        else:
            signal.pthread_sigmask(signal.SIG_BLOCK, [*signals, signal.SIGUSR1])
        expected_mask = sorted(
            int(s) for s in signal.pthread_sigmask(signal.SIG_BLOCK, [])
        )
        expected_handlers = {s: signal.getsignal(s) for s in SIGNALS}
        receipt = runner.run_baseline(args(request_factory, pin))
        observed = boot.read_json(marker)
        ledger = boot.read_json(request_factory.campaign / "gpu_budget.json")
        restored = signal.pthread_sigmask(signal.SIG_BLOCK, []) == set(
            map(signal.Signals, expected_mask)
        ) and all(signal.getsignal(s) == h for s, h in expected_handlers.items())
        write(
            tmp_path / "NESTED_CHILD_SIGNAL_WITNESS.json",
            {
                "route": route,
                "kind": kind,
                "signals": list(map(int, signals)),
                "expected_blocked": expected_mask,
                "expected_ignored": list(map(int, signals))
                if kind == "ignored"
                else None,
                "actual_child": observed,
                "caller_restored": restored,
                "receipt_status": receipt["status"],
                "ledger_active": "active_visual_lease" in ledger,
                "owned_children": children,
                "instrumentation": "Fixed site-disabled metadata probe extended with owned signal-state write, or resealed owned stdlib NRH CLI double. Candidate source is unchanged.",
            },
        )
        assert (
            receipt["status"] == "completed"
            and restored
            and "active_visual_lease" not in ledger
        )
        assert observed["blocked"] == expected_mask
        if kind == "ignored":
            assert set(map(int, signals)) <= set(observed["ignored"]), (
                "Original SIG_IGN lost on nested source-owned child exec"
            )
    assert all(not boot.identity_matches(row["token"]) for row in children)


@pytest.mark.parametrize("pending_before", [False, True])
@pytest.mark.parametrize("signals", SETS)
@pytest.mark.parametrize("disposition", ["ignored", "recording"])
@pytest.mark.parametrize("nested", [False, True])
def test_caller_blocked_pending_signal_remains_pending(
    tmp_path, pending_before, signals, disposition, nested, monkeypatch
):
    delivered = []
    with caller_state():
        signal.pthread_sigmask(signal.SIG_BLOCK, [*signals, signal.SIGUSR1])
        for signum in signals:
            signal.signal(
                signum,
                signal.SIG_IGN
                if disposition == "ignored"
                else lambda s, f: delivered.append(s),
            )
        original_mask = signal.pthread_sigmask(signal.SIG_BLOCK, [])
        original_handlers = {s: signal.getsignal(s) for s in SIGNALS}
        real_signal = signal.signal
        changed_blocked = []

        def install(number, handler):
            if number in signals:
                changed_blocked.append(int(number))
            return real_signal(number, handler)

        monkeypatch.setattr(signal, "signal", install)

        def queue():
            for number in signals:
                os.kill(os.getpid(), number)

        if pending_before:
            queue()
        with boot.cancellation_scope(handler=runner._visual_cancelled):
            if nested:
                with boot.cancellation_scope():
                    if not pending_before:
                        queue()
            elif not pending_before:
                queue()
        pending = signal.sigpending()
        write(
            tmp_path / "CALLER_PENDING_WITNESS.json",
            {
                "pending_before_scope": pending_before,
                "signals": list(map(int, signals)),
                "disposition": disposition,
                "nested": nested,
                "caller_mask_unchanged": signal.pthread_sigmask(signal.SIG_BLOCK, [])
                == original_mask,
                "caller_handlers_unchanged": all(
                    signal.getsignal(s) == h for s, h in original_handlers.items()
                ),
                "still_pending": sorted(map(int, pending)),
                "caller_handler_deliveries": list(map(int, delivered)),
                "blocked_handler_changes": changed_blocked,
            },
        )
        assert set(signals) <= pending and delivered == []
        assert changed_blocked == []
        assert signal.pthread_sigmask(signal.SIG_BLOCK, []) == original_mask
        assert all(signal.getsignal(s) == h for s, h in original_handlers.items())
        monkeypatch.setattr(signal, "signal", real_signal)


def test_expired_probe_deadline_refuses_before_child_birth(tmp_path, monkeypatch):
    real = subprocess.Popen
    children = []

    def spawn(*a, **kw):
        child = real(*a, **kw)
        children.append((child, boot.process_identity(child.pid)))
        return child

    monkeypatch.setattr(subprocess, "Popen", spawn)
    with pytest.raises(TimeoutError):
        boot.interpreter_probe(sys.executable, time.monotonic() - 1)
    write(
        tmp_path / "EXPIRED_PROBE_WITNESS.json",
        {
            "children": [
                {
                    "token": t,
                    "exit": c.returncode,
                    "matching_identity_absent": not boot.identity_matches(t),
                }
                for c, t in children
            ],
            "deadline_expired_before_call": True,
        },
    )
    assert children == [], "An expired absolute metadata deadline still starts a child"


@pytest.fixture
def ignored_termination(request):
    with caller_state():
        if request.param:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGALRM, signal.SIG_IGN)
        yield request.param


@pytest.mark.parametrize("ignored_termination", [False, True], indirect=True)
@pytest.mark.parametrize(
    "seam", ["ledger_settlement", "final_publish", "handler_restoration"]
)
@pytest.mark.parametrize("signals", SETS)
def test_late_retained_route_cancellation_cannot_leave_completed_credit(
    request_factory, tmp_path, monkeypatch, seam, signals, ignored_termination
):
    pin, _ = request_factory()
    real_write, real_publish = runner.write_json, runner.publish
    fired = False
    real_signal = signal.signal
    original_handlers = {s: signal.getsignal(s) for s in SIGNALS}

    def fire():
        nonlocal fired
        fired = True
        prior = signal.pthread_sigmask(signal.SIG_BLOCK, signals)
        for number in signals:
            os.kill(os.getpid(), number)
        signal.pthread_sigmask(signal.SIG_SETMASK, prior)

    def install(number, handler):
        result = real_signal(number, handler)
        receipts = list(
            request_factory.campaign.glob("runs/visual-method-evaluation-*.json")
        )
        if (
            seam == "handler_restoration"
            and not fired
            and number == signal.SIGTERM
            and handler == original_handlers[number]
            and receipts
            and boot.read_json(receipts[0])["status"] == "completed"
        ):
            fire()
        return result

    def settle(path, value):
        nonlocal fired
        result = real_write(path, value)
        if (
            seam == "ledger_settlement"
            and Path(path).name == "gpu_budget.json"
            and "active_visual_lease" not in value
            and not fired
        ):
            fire()
        return result

    def publish(receipt, campaign):
        nonlocal fired
        result = real_publish(receipt, campaign)
        if seam == "final_publish" and receipt["status"] == "completed" and not fired:
            fire()
        return result

    monkeypatch.setattr(signal, "signal", install)
    monkeypatch.setattr(runner, "write_json", settle)
    monkeypatch.setattr(runner, "publish", publish)
    with pytest.raises((runner.RunCancelled, KeyboardInterrupt)) as raised:
        runner.run_baseline(args(request_factory, pin))
    receipts = [
        boot.read_json(p)
        for p in request_factory.campaign.glob("runs/visual-method-evaluation-*.json")
    ]
    assert len(receipts) == 1 and fired
    ledger = boot.read_json(request_factory.campaign / "gpu_budget.json")
    receipt = receipts[0]
    write(
        tmp_path / "LATE_ROUTE_CANCELLATION_WITNESS.json",
        {
            "seam": seam,
            "raised": type(raised.value).__name__,
            "signals": list(map(int, signals)),
            "original_termination_ignored": ignored_termination,
            "final_status": receipt["status"],
            "accepted_complete_control_steps": receipt["metrics"].get(
                "accepted_complete_control_steps"
            ),
            "producer_metrics_retained": "producer_evaluation_metrics" in receipt,
            "ledger_active": "active_visual_lease" in ledger,
            "charge_settlement": receipt["charge_settlement"],
            "scope": "Signal delivered and retained inside original route scope, before return; actual synthetic CLI caller fails.",
        },
    )
    assert receipt["status"] not in {"completed", "incomplete"}
    assert receipt["metrics"]["accepted_complete_control_steps"] == 0
    assert receipt["producer_evaluation_metrics"]["full_cohort_score"] == 1.0
    assert receipt["charge_settlement"] == "actual_owned_lease_wall"
    assert "active_visual_lease" not in ledger


def test_probe_deadline_consumed_during_setup_refuses_before_birth(
    tmp_path, monkeypatch
):
    real_clock = time.monotonic
    real_file = boot.tempfile.TemporaryFile
    now = real_clock()
    expired = False

    def setup(*args, **kwargs):
        nonlocal expired
        result = real_file(*args, **kwargs)
        expired = True
        return result

    monkeypatch.setattr(boot.tempfile, "TemporaryFile", setup)
    monkeypatch.setattr(time, "monotonic", lambda: now + 2 if expired else now)
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *a, **k: pytest.fail("expired setup started a child"),
    )
    with pytest.raises(TimeoutError):
        boot.interpreter_probe(sys.executable, now + 1)
    assert expired
    write(
        tmp_path / "SETUP_EXPIRED_WITNESS.json",
        {"no_child": True, "absolute_deadline_extended": False},
    )


@pytest.mark.parametrize("signals", SETS)
def test_signals_after_terminal_cutoff_follow_original_caller(
    request_factory, tmp_path, monkeypatch, signals
):
    pin, _ = request_factory()
    with caller_state():
        delivered = []
        for number in SIGNALS:
            signal.signal(number, lambda s, f: delivered.append(int(s)))
        signal.pthread_sigmask(signal.SIG_BLOCK, [signal.SIGUSR1])
        handlers = {s: signal.getsignal(s) for s in SIGNALS}
        mask = signal.pthread_sigmask(signal.SIG_BLOCK, [])
        real_mask = signal.pthread_sigmask
        fired = False

        def restore(how, requested):
            nonlocal fired
            result = real_mask(how, requested)
            receipts = list(
                request_factory.campaign.glob("runs/visual-method-evaluation-*.json")
            )
            if (
                not fired
                and how == signal.SIG_SETMASK
                and set(requested) == mask
                and all(signal.getsignal(s) == h for s, h in handlers.items())
                and receipts
                and boot.read_json(receipts[0])["status"] == "completed"
            ):
                fired = True
                for number in signals:
                    os.kill(os.getpid(), number)
            return result

        monkeypatch.setattr(signal, "pthread_sigmask", restore)
        receipt = runner.run_baseline(args(request_factory, pin))
        assert fired and set(delivered) == set(map(int, signals))
        assert receipt["status"] == "completed"
        assert receipt["metrics"]["accepted_complete_control_steps"] == 50
        assert real_mask(signal.SIG_BLOCK, []) == mask
        assert all(signal.getsignal(s) == h for s, h in handlers.items())
        write(
            tmp_path / "POST_CUTOFF_CALLER_WITNESS.json",
            {
                "signals": list(map(int, signals)),
                "caller_deliveries": delivered,
                "invocation_raised": False,
                "status": receipt["status"],
                "cutoff": receipt["terminal_publication"]["cancellation_cutoff"],
                "new_signals_after_final_owned_snapshot": True,
            },
        )


@pytest.mark.parametrize("seam", ["before_write", "after_write"])
def test_terminal_publication_error_preserves_primary_and_revokes_credit(
    request_factory, tmp_path, monkeypatch, seam
):
    pin, _ = request_factory()
    real_publish = runner.publish
    first = RuntimeError("owned terminal publication failure")
    failed_once = False

    def publish(receipt, campaign):
        nonlocal failed_once
        if receipt["status"] == "completed" and not failed_once:
            failed_once = True
            if seam == "after_write":
                real_publish(receipt, campaign)
            raise first
        return real_publish(receipt, campaign)

    monkeypatch.setattr(runner, "publish", publish)
    with pytest.raises(RuntimeError) as caught:
        runner.run_baseline(args(request_factory, pin))
    assert caught.value is first
    receipt = boot.read_json(
        next(request_factory.campaign.glob("runs/visual-method-evaluation-*.json"))
    )
    assert receipt["status"] == "failed"
    assert receipt["metrics"]["accepted_complete_control_steps"] == 0
    assert receipt["producer_evaluation_metrics"]["full_cohort_score"] == 1.0
    ledger = boot.read_json(request_factory.campaign / "gpu_budget.json")
    assert "active_visual_lease" not in ledger
    write(
        tmp_path / "PUBLICATION_FAILURE_WITNESS.json",
        {
            "seam": seam,
            "first_identity_preserved": caught.value is first,
            "final_status": receipt["status"],
            "accepted_complete_control_steps": 0,
            "charge_settlement": receipt["charge_settlement"],
        },
    )


def test_terminal_publication_honors_original_absolute_deadline(
    request_factory, tmp_path, monkeypatch
):
    pin, _ = request_factory()
    real_publish, real_clock = runner.publish, time.monotonic
    started = real_clock()
    expired = False

    def clock():
        return started + 100 if expired else real_clock()

    def publish(receipt, campaign):
        nonlocal expired
        result = real_publish(receipt, campaign)
        if receipt["status"] == "completed":
            expired = True
        return result

    monkeypatch.setattr(time, "monotonic", clock)
    monkeypatch.setattr(runner, "publish", publish)
    with pytest.raises(runner.RunCancelled, match="absolute deadline expired"):
        runner.run_baseline(args(request_factory, pin))
    receipt = boot.read_json(
        next(request_factory.campaign.glob("runs/visual-method-evaluation-*.json"))
    )
    assert expired and receipt["status"] == "cancelled"
    assert receipt["metrics"]["accepted_complete_control_steps"] == 0
    write(
        tmp_path / "TERMINAL_DEADLINE_WITNESS.json",
        {
            "original_absolute_deadline_checked_after_publication": True,
            "deadline_restarted": False,
            "accepted_complete_control_steps": 0,
        },
    )
