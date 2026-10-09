"""Terminal restoration regressions with owned standard-library worker doubles."""

import os
import signal
from contextlib import contextmanager
from itertools import combinations

import pytest
from test_visual_evaluation_dispatch import args, write
from test_visual_evaluation_dispatch import request_factory as _factory

from abc_bench import runner
from abc_bench import visual_evaluation_bootstrap as boot

request_factory = _factory
SIGNALS = boot.CANCELLATION_SIGNALS
SETS = [group for size in range(1, 4) for group in combinations(SIGNALS, size)]


@contextmanager
def saved_state():
    numbers = (*SIGNALS, signal.SIGUSR1)
    handlers = {number: signal.getsignal(number) for number in numbers}
    mask = signal.pthread_sigmask(signal.SIG_BLOCK, [])
    timer = signal.getitimer(signal.ITIMER_REAL)
    reaper = boot.subreaper()
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_BLOCK, numbers)
        for number in numbers:
            if number not in mask and number in signal.sigpending():
                signal.sigwait({number})
        signal.setitimer(signal.ITIMER_REAL, 0)
        for number, handler in handlers.items():
            signal.signal(number, handler)
        signal.pthread_sigmask(signal.SIG_SETMASK, mask)
        if timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *timer)
        boot.subreaper(reaper)


def receipt_path(factory):
    return next(factory.campaign.glob("runs/visual-method-evaluation-*.json"))


def refuse_final_mask(factory, monkeypatch, error):
    """Refuse once, before any side effect, after the route's receipt exists."""
    real_mask = signal.pthread_sigmask
    mask = real_mask(signal.SIG_BLOCK, [])
    handlers = {number: signal.getsignal(number) for number in SIGNALS}
    calls = []

    def restore(how, requested):
        receipts = list(factory.campaign.glob("runs/visual-method-evaluation-*.json"))
        if (
            not calls
            and how == signal.SIG_SETMASK
            and set(requested) == mask
            and all(
                signal.getsignal(number) == handler
                for number, handler in handlers.items()
            )
            and receipts
            and "terminal_publication" in boot.read_json(receipts[0])
        ):
            calls.append("refused before side effect")
            raise error
        return real_mask(how, requested)

    monkeypatch.setattr(signal, "pthread_sigmask", restore)
    return calls, real_mask, mask, handlers


@pytest.mark.parametrize("when", ["before_write", "after_write"])
def test_final_restoration_failure_preserves_prior_primary(
    request_factory, tmp_path, monkeypatch, when
):
    pin, _ = request_factory()
    primary = RuntimeError("owned prior terminal publication failure")
    restoration = OSError("owned refused final mask restoration")
    real_publish = runner.publish
    primary_calls = []

    def publish(receipt, campaign):
        if receipt["status"] == "completed" and not primary_calls:
            primary_calls.append(True)
            if when == "after_write":
                real_publish(receipt, campaign)
            raise primary
        return real_publish(receipt, campaign)

    with saved_state():
        calls, _, _, _ = refuse_final_mask(request_factory, monkeypatch, restoration)
        monkeypatch.setattr(runner, "publish", publish)
        with pytest.raises(RuntimeError) as caught:
            runner.run_baseline(args(request_factory, pin))
        receipt = boot.read_json(receipt_path(request_factory))
        ledger = boot.read_json(request_factory.campaign / "gpu_budget.json")
        write(
            tmp_path / "PRIOR_CAUSE_RESTORATION_WITNESS.json",
            {
                "prior_exception_identity_preserved": caught.value is primary,
                "restoration_refusals": calls,
                "primary_notes": primary.__notes__,
                "status": receipt["status"],
                "accepted": receipt["metrics"]["accepted_complete_control_steps"],
                "raw_score": receipt["producer_evaluation_metrics"][
                    "full_cohort_score"
                ],
                "charge_settlement": receipt["charge_settlement"],
                "ledger_active": "active_visual_lease" in ledger,
            },
        )
        assert caught.value is primary and len(calls) == 1
        assert any(str(restoration) in note for note in primary.__notes__)
        assert receipt["terminal_failure_notes"] == primary.__notes__
        assert receipt["status"] == "failed"
        assert receipt["metrics"]["accepted_complete_control_steps"] == 0
        assert receipt["producer_evaluation_metrics"]["full_cohort_score"] == 1.0
        assert receipt["charge_settlement"] == "actual_owned_lease_wall"
        assert "active_visual_lease" not in ledger


@pytest.mark.parametrize("when", ["before_write", "after_write"])
def test_failed_restoration_publication_retains_errors_and_disk_truth(
    request_factory, tmp_path, monkeypatch, when
):
    pin, _ = request_factory()
    primary = OSError("owned refused final mask restoration")
    secondary = OSError("owned failure receipt storage error")
    real_publish = runner.publish
    failure_calls = []

    def publish(receipt, campaign):
        if receipt["status"] == "failed":
            failure_calls.append(True)
            if when == "after_write":
                real_publish(receipt, campaign)
            raise secondary
        return real_publish(receipt, campaign)

    with saved_state():
        calls, _, _, _ = refuse_final_mask(request_factory, monkeypatch, primary)
        monkeypatch.setattr(runner, "publish", publish)
        with pytest.raises(OSError) as caught:
            runner.run_baseline(args(request_factory, pin))
        receipt = boot.read_json(receipt_path(request_factory))
        ledger = boot.read_json(request_factory.campaign / "gpu_budget.json")
        expected_status = "completed" if when == "before_write" else "failed"
        expected_credit = 50 if when == "before_write" else 0
        write(
            tmp_path / "FAILED_PUBLICATION_DISK_TRUTH.json",
            {
                "when": when,
                "first_identity_preserved": caught.value is primary,
                "primary_notes": primary.__notes__,
                "failure_publications": len(failure_calls),
                "saved_status": receipt["status"],
                "saved_credit": receipt["metrics"]["accepted_complete_control_steps"],
                "invocation_raised": True,
                "accepted_invocation": False,
                "charge_settlement": receipt["charge_settlement"],
                "ledger_active": "active_visual_lease" in ledger,
            },
        )
        assert caught.value is primary and len(calls) == 1
        assert len(failure_calls) == 1
        assert any(str(secondary) in note for note in primary.__notes__)
        assert receipt["status"] == expected_status
        assert receipt["metrics"]["accepted_complete_control_steps"] == expected_credit
        assert receipt["producer_evaluation_metrics"]["full_cohort_score"] == 1.0
        assert receipt["charge_settlement"] == "actual_owned_lease_wall"
        assert "active_visual_lease" not in ledger


@pytest.mark.parametrize("targets", SETS)
def test_final_restoration_failure_retains_caller_blocked_pending_bits(
    request_factory, tmp_path, monkeypatch, targets
):
    pin, _ = request_factory()
    primary = OSError("owned refused final mask with caller pending bits")
    with saved_state():
        for number in (*SIGNALS, signal.SIGUSR1):
            signal.signal(number, signal.SIG_IGN)
        signal.pthread_sigmask(signal.SIG_BLOCK, [*targets, signal.SIGUSR1])
        for target in targets:
            os.kill(os.getpid(), target)
        os.kill(os.getpid(), signal.SIGUSR1)
        pending = signal.sigpending()
        calls, real_mask, mask, handlers = refuse_final_mask(
            request_factory, monkeypatch, primary
        )
        with pytest.raises(OSError) as caught:
            runner.run_baseline(args(request_factory, pin))
        receipt = boot.read_json(receipt_path(request_factory))
        actual_mask = real_mask(signal.SIG_BLOCK, [])
        after = signal.sigpending()
        write(
            tmp_path / "REFUSED_MASK_CALLER_PENDING_WITNESS.json",
            {
                "targets": list(map(int, targets)),
                "before": sorted(map(int, pending)),
                "after": sorted(map(int, after)),
                "requested_mask": sorted(map(int, mask)),
                "actual_partial_mask": sorted(map(int, actual_mask)),
                "restoration_refusals": calls,
                "first_identity_preserved": caught.value is primary,
                "handlers_preserved": all(
                    signal.getsignal(number) == handler
                    for number, handler in handlers.items()
                ),
                "status": receipt["status"],
                "accepted": receipt["metrics"]["accepted_complete_control_steps"],
            },
        )
        assert caught.value is primary and len(calls) == 1
        assert after == pending == {*targets, signal.SIGUSR1}
        assert actual_mask == mask | set(SIGNALS)
        assert all(
            signal.getsignal(number) == handler for number, handler in handlers.items()
        )
        assert receipt["status"] == "failed"
        assert receipt["metrics"]["accepted_complete_control_steps"] == 0


def test_raising_restored_caller_handler_withholds_credit_without_recapture(
    request_factory, tmp_path, monkeypatch
):
    pin, _ = request_factory()
    primary = RuntimeError("owned post-cutoff original caller failure")
    with saved_state():
        deliveries = []

        def original(number, frame):
            deliveries.append(int(number))
            raise primary

        signal.signal(signal.SIGTERM, original)
        real_mask = signal.pthread_sigmask
        mask = real_mask(signal.SIG_BLOCK, [])
        handlers = {number: signal.getsignal(number) for number in SIGNALS}
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
                and all(
                    signal.getsignal(number) == handler
                    for number, handler in handlers.items()
                )
                and receipts
            ):
                fired = True
                os.kill(os.getpid(), signal.SIGTERM)
            return result

        monkeypatch.setattr(signal, "pthread_sigmask", restore)
        with pytest.raises(RuntimeError) as caught:
            runner.run_baseline(args(request_factory, pin))
        receipt = boot.read_json(receipt_path(request_factory))
        write(
            tmp_path / "RAISING_CALLER_CUTOFF_WITNESS.json",
            {
                "delivered_by_original_caller": deliveries,
                "first_identity_preserved": caught.value is primary,
                "status": receipt["status"],
                "accepted": receipt["metrics"]["accepted_complete_control_steps"],
                "mask_restored": real_mask(signal.SIG_BLOCK, []) == mask,
                "handlers_preserved": all(
                    signal.getsignal(number) == handler
                    for number, handler in handlers.items()
                ),
            },
        )
        assert fired and deliveries == [int(signal.SIGTERM)] and caught.value is primary
        assert receipt["status"] == "failed"
        assert receipt["metrics"]["accepted_complete_control_steps"] == 0
        assert real_mask(signal.SIG_BLOCK, []) == mask
        assert all(
            signal.getsignal(number) == handler for number, handler in handlers.items()
        )
