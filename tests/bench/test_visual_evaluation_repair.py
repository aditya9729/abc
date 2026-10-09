"""Actual owned cancellation seams and strict startup/timing boundaries."""

import copy
import ctypes
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from test_visual_evaluation_dispatch import args
from test_visual_evaluation_dispatch import request_factory as _request_factory

from abc_bench import runner
from abc_bench import visual_evaluation_bootstrap as boot
from abc_bench import visual_evaluation_dispatch as visual

request_factory = _request_factory


@pytest.mark.parametrize("signum", boot.CANCELLATION_SIGNALS)
@pytest.mark.parametrize("seam", ["birth", "identity", "cleanup", "restore"])
def test_owned_signal_seams_reap_preserve_first_and_restore(
    tmp_path, monkeypatch, signum, seam
):
    originals = {s: signal.getsignal(s) for s in boot.CANCELLATION_SIGNALS}
    caller_handlers = originals.copy()
    mask = signal.pthread_sigmask(signal.SIG_BLOCK, [])
    timer = signal.getitimer(signal.ITIMER_REAL)
    reaper = boot.subreaper()
    real_popen, real_identity, real_send, real_signal = (
        subprocess.Popen,
        boot.process_identity,
        boot.signal_owned,
        signal.signal,
    )
    first = RuntimeError("owned first handshake fault")
    child = token = None
    fired = False
    command = [sys.executable, "-I", "-B", "-c", "import time;time.sleep(90)"]
    spec = {
        "bootstrap_command": command,
        "control_directory": str(tmp_path),
        "cleanup_s": 0.2,
        "cancel_deadline": time.monotonic() + 2,
        "terminate_deadline": time.monotonic() + 3,
        "hard_deadline": time.monotonic() + 4,
    }

    def fire():
        nonlocal fired
        if not fired:
            fired = True
            os.kill(os.getpid(), signum)

    def spawn(*a, **k):
        nonlocal child, token
        child = real_popen(*a, **k)
        token = real_identity(child.pid)
        if seam == "birth":
            fire()
        return child

    def identity(pid):
        if seam == "identity" and child is not None and pid == child.pid:
            fire()
        return real_identity(pid)

    def send(value, number, *, group=False):
        if seam == "cleanup" and number == signal.SIGTERM:
            fire()
        return real_send(value, number, group=group)

    def install(number, handler):
        previous = real_signal(number, handler)
        if seam == "restore" and number == signum and handler == originals[number]:
            fire()
        return previous

    try:
        # A known caller cancellation handler makes every signal test nonfatal.
        for s in boot.CANCELLATION_SIGNALS:
            real_signal(s, signal.default_int_handler)
        expected_handlers = {s: signal.getsignal(s) for s in boot.CANCELLATION_SIGNALS}
        originals = expected_handlers
        monkeypatch.setattr(subprocess, "Popen", spawn)
        monkeypatch.setattr(boot, "process_identity", identity)
        monkeypatch.setattr(boot, "signal_owned", send)
        monkeypatch.setattr(signal, "signal", install)
        monkeypatch.setattr(
            boot, "_wait_ready", lambda *a: (_ for _ in ()).throw(first)
        )
        with (
            (tmp_path / "owned.lock").open("xb") as lock,
            pytest.raises((KeyboardInterrupt, RuntimeError, AttributeError)) as caught,
        ):
            boot.execute_guarded(
                command,
                tmp_path / "owned.log",
                2,
                gpu_uuid="",
                spec=spec,
                lease_fd=lock.fileno(),
            )
        assert fired
        assert child.returncode is not None and not Path(f"/proc/{child.pid}").exists()
        assert not boot.group_exists(token["pgid"])
        assert boot.subreaper() == reaper
        assert signal.pthread_sigmask(signal.SIG_BLOCK, []) == mask
        assert signal.getitimer(signal.ITIMER_REAL) == timer
        assert {
            s: signal.getsignal(s) for s in boot.CANCELLATION_SIGNALS
        } == expected_handlers
        if seam in {"cleanup", "restore"}:
            assert caught.value is first
        else:
            assert isinstance(caught.value, KeyboardInterrupt)
        (tmp_path / "witness.json").write_text(
            json.dumps(
                {
                    "signum": int(signum),
                    "seam": seam,
                    "worker": token,
                    "child_exit": child.returncode,
                    "raised": type(caught.value).__name__,
                    "primary_is_readiness_fault": caught.value is first,
                    "notes": getattr(caught.value, "__notes__", []),
                    "outcome": boot.read_json(tmp_path / "execution-outcome.json"),
                    "caller_state_restored": True,
                    "group_gone": True,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        for s, handler in caller_handlers.items():
            real_signal(s, handler)
        if child is not None and child.poll() is None:
            assert real_identity(child.pid) == token
            os.killpg(token["pgid"], signal.SIGKILL)
            child.wait(timeout=2)
        boot.subreaper(reaper)


@pytest.mark.parametrize("signum", boot.CANCELLATION_SIGNALS)
@pytest.mark.parametrize("seam", ["install", "timer", "restore"])
def test_partial_outer_signal_startup_restores_without_access(
    monkeypatch, signum, seam
):
    from argparse import Namespace

    old = {s: signal.getsignal(s) for s in boot.CANCELLATION_SIGNALS}
    mask = signal.pthread_sigmask(signal.SIG_BLOCK, [])
    timer = signal.getitimer(signal.ITIMER_REAL)
    real_signal, real_timer = signal.signal, signal.setitimer
    fired = False

    def fire():
        nonlocal fired
        if not fired:
            fired = True
            os.kill(os.getpid(), signum)

    def install(s, handler):
        value = real_signal(s, handler)
        if s == signal.SIGTERM and (
            (
                seam == "install"
                and getattr(handler, "_matched_handlers", {}).get(signal.SIGTERM)
                is runner._visual_cancelled
            )
            or (seam == "restore" and handler == old[s])
        ):
            fire()
        return value

    def arm(which, seconds, *a):
        value = real_timer(which, seconds, *a)
        if seam == "timer" and seconds > 0:
            fire()
        return value

    monkeypatch.setattr(signal, "signal", install)
    monkeypatch.setattr(signal, "setitimer", arm)
    # No request/campaign access is reached: restore seam starts with a deliberate argument fault.
    with pytest.raises((KeyboardInterrupt, RuntimeError, AttributeError)) as caught:
        runner.run_visual_evaluation(Namespace(timeout_seconds=5))
    assert fired
    assert {s: signal.getsignal(s) for s in boot.CANCELLATION_SIGNALS} == old
    assert signal.getitimer(signal.ITIMER_REAL) == timer
    assert signal.pthread_sigmask(signal.SIG_BLOCK, []) == mask
    if seam == "restore":
        assert isinstance(caught.value, AttributeError)
    else:
        assert isinstance(caught.value, (KeyboardInterrupt, runner.RunCancelled))


def test_python_and_kernel_threads_refuse_before_mutations(monkeypatch):
    before = boot.subreaper()
    monkeypatch.setattr(threading, "active_count", lambda: 2)
    with pytest.raises(ValueError, match="kernel thread"), boot.cancellation_scope():
        pytest.fail("threaded controller was admitted")
    monkeypatch.undo()
    libc = ctypes.CDLL(None)
    thread = ctypes.c_ulong()
    assert (
        libc.pthread_create(
            ctypes.byref(thread),
            None,
            ctypes.cast(libc.sleep, ctypes.c_void_p),
            ctypes.c_void_p(1),
        )
        == 0
    )
    try:
        assert threading.active_count() == 1
        with (
            pytest.raises(ValueError, match="kernel thread"),
            boot.cancellation_scope(),
        ):
            pytest.fail("untracked kernel thread was admitted")
    finally:
        assert libc.pthread_join(thread, None) == 0
    assert boot.subreaper() == before


@pytest.mark.parametrize("location", ["site", "pth"])
@pytest.mark.parametrize("customize", ["sitecustomize", "usercustomize"])
@pytest.mark.parametrize(
    "form", ["module", "package", "bytecode", "package_bytecode", "extension"]
)
def test_effective_startup_forms_refuse_without_marker(
    request_factory, tmp_path, location, customize, form
):
    import importlib.machinery
    import py_compile

    _, value = request_factory()
    worker = value["worker"]
    site = Path(worker["site_packages"])
    directory = site if location == "site" else tmp_path / "dependency"
    directory.mkdir(exist_ok=True)
    marker = tmp_path / "never-executed"
    source = "from pathlib import Path\nPath(" + repr(str(marker)) + ").touch()\n"
    if form in {"package", "package_bytecode"}:
        package = directory / customize
        package.mkdir()
        (package / "__init__.py").write_text(source)
        if form == "package_bytecode":
            py_compile.compile(
                str(package / "__init__.py"),
                cfile=str(package / "__init__.pyc"),
                doraise=True,
            )
            (package / "__init__.py").unlink()
    elif form == "extension":
        (
            directory / (customize + importlib.machinery.EXTENSION_SUFFIXES[0])
        ).write_bytes(b"owned non-executable extension marker")
    else:
        path = directory / (customize + ".py")
        path.write_text(source)
        if form == "bytecode":
            py_compile.compile(
                str(path), cfile=str(path.with_suffix(".pyc")), doraise=True
            )
            path.unlink()
    if location == "pth":
        pathfile = site / "readonly-owned.pth"
        pathfile.write_text(str(directory) + "\n")
        worker["pth"] = [boot.artifact(pathfile)]
    with pytest.raises(ValueError, match="startup customization"):
        visual._worker(worker, time.monotonic() + 5)
    assert not marker.exists()


def test_exact_nrh_package_origin_denies_extension_substitution(request_factory):
    import importlib.machinery

    _, value = request_factory()
    package = Path(value["worker"]["site_packages"]) / "nrh"
    (package / ("__init__" + importlib.machinery.EXTENSION_SUFFIXES[0])).write_bytes(
        b"owned non-executable extension marker"
    )
    with pytest.raises(ValueError, match="package import origin"):
        visual._worker(value["worker"], time.monotonic() + 5)


@pytest.mark.parametrize("fault", ["malformed", "oversize", "hang", "birth_signal"])
def test_identity_probe_failure_is_bounded_reaped_and_restored(monkeypatch, fault):
    real = subprocess.Popen
    children = []
    old = signal.getsignal(signal.SIGINT)
    before = boot.subreaper()

    def spawn(argv, **kw):
        code = (
            "print('not json')"
            if fault == "malformed"
            else "print('x'*40000)"
            if fault == "oversize"
            else "import time;time.sleep(90)"
            if fault == "hang"
            else boot.INTERPRETER_PROBE
        )
        child = real([sys.executable, "-I", "-S", "-B", "-c", code], **kw)
        children.append(child)
        if fault == "birth_signal":
            os.kill(os.getpid(), signal.SIGINT)
        return child

    monkeypatch.setattr(subprocess, "Popen", spawn)
    with pytest.raises((ValueError, TimeoutError, KeyboardInterrupt)):
        boot.interpreter_probe(sys.executable, time.monotonic() + 0.15)
    assert all(
        c.returncode is not None and not Path(f"/proc/{c.pid}").exists()
        for c in children
    )
    assert signal.getsignal(signal.SIGINT) == old and boot.subreaper() == before


def sample():
    return {
        "iteration": 1,
        "returned_control_index": 1,
        "selection_dispatch": 0.01,
        "step": 0.02,
        "capture": 0.03,
        "whole_loop": 0.07,
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("iteration", True),
        ("iteration", 0),
        ("returned_control_index", True),
        ("returned_control_index", 2),
        ("selection_dispatch", True),
        ("selection_dispatch", "0.1"),
        ("selection_dispatch", -1),
        ("step", float("inf")),
        ("capture", float("nan")),
        ("whole_loop", 0.01),
        ("capture", None),
    ],
)
def test_typed_causal_timing_sample_refuses(field, value):
    value_sample = sample()
    value_sample[field] = value
    row = {
        "complete": True,
        "returned_controls": 1,
        "unknown_physical_attempts": 0,
        "control_wall_samples": [value_sample],
    }
    with pytest.raises(ValueError):
        visual._timing_samples(row)


@pytest.mark.parametrize(
    "phase",
    ["selection_failed", "step_unknown", "post_return_failed", "capture_failed"],
)
def test_legitimate_partial_timing_nulls_preserve_scope(phase):
    s = sample()
    returned = 1
    unknown = 0
    if phase == "selection_failed":
        s.update(returned_control_index=None, step=None, capture=None)
        returned = 0
    elif phase == "step_unknown":
        s.update(returned_control_index=None, capture=None)
        returned, unknown = 0, 1
    elif phase == "post_return_failed":
        s["capture"] = None
    visual._timing_samples(
        {
            "complete": False,
            "returned_controls": returned,
            "unknown_physical_attempts": unknown,
            "control_wall_samples": [s],
        }
    )


@pytest.mark.parametrize("protocol", visual.PROTOCOLS)
def test_all_protocols_timing_and_malformed_sample_parity(request_factory, protocol):
    pin, _ = request_factory(protocol=protocol)
    receipt = runner.run_baseline(args(request_factory, pin))
    request = visual.load_request(pin, visual.CANONICAL_CAMPAIGN)
    worker = boot.read_json(receipt["worker_receipt"]["path"])
    assert visual.summarize_worker(request, worker, 0)["status"] == "completed"
    for malformed in [True, "time", -1, {}, {"iteration": 1}]:
        invalid = copy.deepcopy(worker)
        invalid["episodes"][0]["control_wall_samples"][0] = malformed
        with pytest.raises(ValueError):
            visual.summarize_worker(request, invalid, 0)


SIGNAL_SETS = [
    tuple(s for i, s in enumerate(boot.CANCELLATION_SIGNALS) if bits & (1 << i))
    for bits in range(1, 8)
]


@pytest.mark.parametrize("signals", SIGNAL_SETS)
@pytest.mark.parametrize(
    "seam",
    [
        "birth",
        "identity",
        "cleanup",
        "term_restore",
        "int_restore",
        "alarm_restore",
        "timer_disable",
        "timer_restore",
    ],
)
def test_pending_signal_combinations_close_ownership_and_keep_first(
    tmp_path, monkeypatch, signals, seam
):
    old_handlers = {s: signal.getsignal(s) for s in boot.CANCELLATION_SIGNALS}
    old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, [])
    old_timer = signal.getitimer(signal.ITIMER_REAL)
    old_reaper = boot.subreaper()
    real_signal, real_timer, real_spawn = (
        signal.signal,
        signal.setitimer,
        subprocess.Popen,
    )
    real_identity, real_send = boot.process_identity, boot.signal_owned
    first = RuntimeError("owned primary readiness error")
    child = token = None
    fired = False
    command = [sys.executable, "-I", "-B", "-c", "import time;time.sleep(90)"]
    spec = {
        "bootstrap_command": command,
        "control_directory": str(tmp_path),
        "cleanup_s": 0.1,
        "cancel_deadline": time.monotonic() + 1,
        "terminate_deadline": time.monotonic() + 2,
        "hard_deadline": time.monotonic() + 3,
    }

    def fire():
        nonlocal fired
        if not fired:
            fired = True
            # Queue the complete set atomically to the current thread, then retain delivery.
            mask = signal.pthread_sigmask(signal.SIG_BLOCK, signals)
            for signum in signals:
                os.kill(os.getpid(), signum)
            signal.pthread_sigmask(signal.SIG_SETMASK, mask)

    def spawn(*a, **k):
        nonlocal child, token
        child = real_spawn(*a, **k)
        token = real_identity(child.pid)
        if seam == "birth":
            fire()
        return child

    def identity(pid):
        if seam == "identity" and child is not None and pid == child.pid:
            fire()
        return real_identity(pid)

    restore = {
        "term_restore": signal.SIGTERM,
        "int_restore": signal.SIGINT,
        "alarm_restore": signal.SIGALRM,
    }

    def install(signum, handler):
        value = real_signal(signum, handler)
        if (
            seam in restore
            and signum == restore[seam]
            and handler is signal.default_int_handler
        ):
            fire()
        return value

    def timer(which, value, *a):
        previous = real_timer(which, value, *a)
        if child is not None and (
            (seam == "timer_disable" and value == 0)
            or (seam == "timer_restore" and value > 0)
        ):
            fire()
        return previous

    def send(value, signum, *, group=False):
        if seam == "cleanup" and signum == signal.SIGTERM:
            fire()
        return real_send(value, signum, group=group)

    try:
        for signum in boot.CANCELLATION_SIGNALS:
            real_signal(signum, signal.default_int_handler)
        real_timer(signal.ITIMER_REAL, 20)
        monkeypatch.setattr(subprocess, "Popen", spawn)
        monkeypatch.setattr(boot, "process_identity", identity)
        monkeypatch.setattr(boot, "signal_owned", send)
        monkeypatch.setattr(signal, "signal", install)
        monkeypatch.setattr(signal, "setitimer", timer)
        monkeypatch.setattr(
            boot, "_wait_ready", lambda *a: (_ for _ in ()).throw(first)
        )
        with (
            (tmp_path / "owned.lock").open("xb") as lock,
            pytest.raises((KeyboardInterrupt, RuntimeError, AttributeError)) as caught,
        ):
            boot.execute_guarded(
                command,
                tmp_path / "owned.log",
                1,
                gpu_uuid="",
                spec=spec,
                lease_fd=lock.fileno(),
            )
        assert fired
        assert child.returncode is not None and not Path(f"/proc/{child.pid}").exists()
        assert not boot.group_exists(token["pgid"])
        assert boot.subreaper() == old_reaper
        assert signal.pthread_sigmask(signal.SIG_BLOCK, []) == old_mask
        assert {s: signal.getsignal(s) for s in boot.CANCELLATION_SIGNALS} == {
            s: signal.default_int_handler for s in boot.CANCELLATION_SIGNALS
        }
        assert 0 < signal.getitimer(signal.ITIMER_REAL)[0] <= 20
        if seam in {"birth", "identity"}:
            assert isinstance(caught.value, KeyboardInterrupt)
        else:
            assert caught.value is first
        (tmp_path / "combination-witness.json").write_text(
            json.dumps(
                {
                    "signals": [int(s) for s in signals],
                    "seam": seam,
                    "worker": token,
                    "child_exit": child.returncode,
                    "first_preserved": caught.value is first,
                    "raised": type(caught.value).__name__,
                    "notes": getattr(caught.value, "__notes__", []),
                    "group_gone": True,
                    "caller_state_restored": True,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        real_timer(signal.ITIMER_REAL, 0)
        for signum, handler in old_handlers.items():
            real_signal(signum, handler)
        signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
        if old_timer[0] > 0:
            real_timer(signal.ITIMER_REAL, *old_timer)
        if child is not None and child.poll() is None:
            assert real_identity(child.pid) == token
            os.killpg(token["pgid"], signal.SIGKILL)
            child.wait(timeout=2)
        boot.subreaper(old_reaper)


def test_expired_one_shot_deadline_is_not_rearmed():
    old = signal.getsignal(signal.SIGALRM)
    original_timer = signal.getitimer(signal.ITIMER_REAL)
    try:
        signal.signal(signal.SIGALRM, signal.default_int_handler)
        signal.setitimer(signal.ITIMER_REAL, 0.02)
        with pytest.raises(KeyboardInterrupt), boot.cancellation_scope():
            time.sleep(0.05)
        assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
        if original_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *original_timer)


@pytest.mark.parametrize("signals", SIGNAL_SETS)
@pytest.mark.parametrize(
    "seam",
    [
        "term_install",
        "int_install",
        "alarm_install",
        "timer_install",
        "term_restore",
        "int_restore",
        "alarm_restore",
        "timer_disable",
    ],
)
def test_outer_pending_sets_restore_before_request_access(monkeypatch, signals, seam):
    from argparse import Namespace

    old_handlers = {s: signal.getsignal(s) for s in boot.CANCELLATION_SIGNALS}
    old_timer = signal.getitimer(signal.ITIMER_REAL)
    old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, [])
    old_reaper = boot.subreaper()
    real_signal, real_timer = signal.signal, signal.setitimer
    fired = False
    ordinal = {"term": signal.SIGTERM, "int": signal.SIGINT, "alarm": signal.SIGALRM}

    def fire():
        nonlocal fired
        if not fired:
            fired = True
            mask = signal.pthread_sigmask(signal.SIG_BLOCK, signals)
            for signum in signals:
                os.kill(os.getpid(), signum)
            signal.pthread_sigmask(signal.SIG_SETMASK, mask)

    def install(signum, handler):
        previous = real_signal(signum, handler)
        phase, action = seam.split("_", 1)
        if (
            phase in ordinal
            and signum == ordinal[phase]
            and (
                (action == "install" and hasattr(handler, "_matched_handlers"))
                or (action == "restore" and handler == old_handlers[signum])
            )
        ):
            fire()
        return previous

    def timer(which, value, *a):
        previous = real_timer(which, value, *a)
        if (seam == "timer_install" and value > 0) or (
            seam == "timer_disable" and value == 0
        ):
            fire()
        return previous

    monkeypatch.setattr(signal, "signal", install)
    monkeypatch.setattr(signal, "setitimer", timer)
    with pytest.raises(
        (runner.RunCancelled, KeyboardInterrupt, AttributeError)
    ) as caught:
        runner.run_visual_evaluation(Namespace(timeout_seconds=5))
    assert fired
    assert {s: signal.getsignal(s) for s in boot.CANCELLATION_SIGNALS} == old_handlers
    assert signal.getitimer(signal.ITIMER_REAL) == old_timer
    assert signal.pthread_sigmask(signal.SIG_BLOCK, []) == old_mask
    assert boot.subreaper() == old_reaper
    if seam.endswith("restore") or seam == "timer_disable":
        assert isinstance(caught.value, AttributeError)
    else:
        assert isinstance(caught.value, (runner.RunCancelled, KeyboardInterrupt))
