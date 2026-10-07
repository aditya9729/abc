"""Independent process-lifetime regressions using CPU-only synthetic children."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from abc_bench.runner import execute_command


def _alive(pid: int) -> bool:
    try:
        # A reaped-or-zombie process cannot keep consuming GPU resources.
        state = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1][0]
        return state != "Z"
    except FileNotFoundError:
        return False


def _cleanup(pid: int | None) -> None:
    if pid is not None and _alive(pid):
        os.kill(pid, signal.SIGKILL)


def _wait_stopped(pid: int) -> bool:
    # SIGKILL is asynchronous; allow bounded kernel delivery, never a live survivor.
    deadline = time.monotonic() + 1
    while _alive(pid) and time.monotonic() < deadline:
        time.sleep(0.01)
    return not _alive(pid)


def test_timeout_reaps_synthetic_child(tmp_path):
    log = tmp_path / "child.log"
    code = "import os,time; print(os.getpid(),flush=True); time.sleep(60)"
    pid = None
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            execute_command([sys.executable, "-c", code], log, 0.5)
        pid = int(log.read_text().strip())
        assert _wait_stopped(pid)
    finally:
        if pid is None and log.exists() and log.read_text().strip():
            pid = int(log.read_text().strip())
        _cleanup(pid)


def test_sigterm_parent_reaps_synthetic_child(tmp_path):
    log = tmp_path / "child.log"
    child = "import os,time; print(os.getpid(),flush=True); time.sleep(60)"
    wrapper = (
        "import sys; from pathlib import Path; "
        "from abc_bench.runner import execute_command; "
        "execute_command([sys.executable,'-c',sys.argv[2]],Path(sys.argv[1]),60)"
    )
    parent = subprocess.Popen([sys.executable, "-c", wrapper, str(log), child])
    pid = None
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if log.exists() and log.read_text().strip():
                pid = int(log.read_text().strip())
                break
            if parent.poll() is not None:
                pytest.fail("Synthetic parent exited before child startup")
            time.sleep(0.02)
        assert pid is not None, "Synthetic child did not start"
        parent.send_signal(signal.SIGTERM)
        assert parent.wait(timeout=12) != 0
        assert _wait_stopped(pid)
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=3)
        _cleanup(pid)


def test_timeout_stops_descendant_that_ignores_sigterm(tmp_path):
    log = tmp_path / "family.log"
    descendant = (
        "import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        "print(os.getpid(),flush=True); time.sleep(60)"
    )
    child = (
        "import subprocess,sys,time; "
        "subprocess.Popen([sys.executable,'-c',sys.argv[1]]); time.sleep(60)"
    )
    pid = None
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            execute_command([sys.executable, "-c", child, descendant], log, 0.5)
        pid = int(log.read_text().strip())
        assert _wait_stopped(pid), "Descendant survived process-group cancellation"
    finally:
        if pid is None and log.exists() and log.read_text().strip():
            pid = int(log.read_text().strip())
        _cleanup(pid)


def test_normal_leader_exit_stops_lingering_descendant(tmp_path):
    log = tmp_path / "normal-family.log"
    descendant = (
        "import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        "print(os.getpid(),flush=True); time.sleep(60)"
    )
    child = (
        "import subprocess,sys,time; "
        "subprocess.Popen([sys.executable,'-c',sys.argv[1]]); time.sleep(0.3)"
    )
    pid = None
    try:
        assert execute_command([sys.executable, "-c", child, descendant], log, 3) == 0
        pid = int(log.read_text().strip())
        assert _wait_stopped(pid), "Descendant survived normal leader exit"
    finally:
        if pid is None and log.exists() and log.read_text().strip():
            pid = int(log.read_text().strip())
        _cleanup(pid)


def test_cuda_smoke_rejects_missing_parent_lease_without_loading_model(monkeypatch):
    from abc_bench.update_smoke import require_campaign_lease

    monkeypatch.setattr(os, "getppid", lambda: -999)
    monkeypatch.setattr(Path, "read_text", lambda self: '{"active_parent_pid":123}')
    with pytest.raises(RuntimeError, match="Use python -m abc_bench.runner"):
        require_campaign_lease("cuda:0")


def test_cpu_guard_does_not_read_or_modify_campaign_ledger(monkeypatch):
    from abc_bench.update_smoke import require_campaign_lease

    def unexpected_read(self):
        pytest.fail("CPU guard read the campaign ledger")

    monkeypatch.setattr(Path, "read_text", unexpected_read)
    require_campaign_lease("cpu")


def test_cuda_worker_refuses_another_logical_device(monkeypatch):
    from abc_bench.update_smoke import require_campaign_lease

    monkeypatch.setattr(os, "getppid", lambda: 123)
    monkeypatch.setattr(Path, "read_text", lambda self: '{"active_parent_pid":123}')
    with pytest.raises(RuntimeError, match="cuda:0 only"):
        require_campaign_lease("cuda:1")


def test_cuda_worker_refuses_visibility_outside_physical_reservation(monkeypatch):
    from abc_bench.update_smoke import require_campaign_lease

    monkeypatch.setattr(os, "getppid", lambda: 123)
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self: (
            '{"active_parent_pid":123,"active_gpu_uuid":"GPU-00000000-0000-0000-0000-000000000000"}'
        ),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-other")
    with pytest.raises(RuntimeError, match="physical reservation"):
        require_campaign_lease("cuda:0")


@pytest.mark.parametrize("reservation", [None, "GPU-invalid", "GPU-one,GPU-two"])
def test_cuda_worker_requires_a_valid_physical_reservation(monkeypatch, reservation):
    import json

    from abc_bench.update_smoke import require_campaign_lease

    monkeypatch.setattr(os, "getppid", lambda: 123)
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self: json.dumps(
            {"active_parent_pid": 123, "active_gpu_uuid": reservation}
        ),
    )
    with pytest.raises(RuntimeError, match="GPU UUID reservation"):
        require_campaign_lease("cuda:0")


def test_cuda_worker_refuses_gl_context_profile(monkeypatch):
    from abc_bench.update_smoke import require_campaign_lease

    gpu_uuid = "GPU-00000000-0000-0000-0000-000000000000"
    monkeypatch.setattr(os, "getppid", lambda: 123)
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self: '{"active_parent_pid":123,"active_gpu_uuid":"' + gpu_uuid + '"}',
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", gpu_uuid)
    monkeypatch.setenv("MUJOCO_GL", "egl")
    with pytest.raises(RuntimeError, match="MUJOCO_GL=disable"):
        require_campaign_lease("cuda:0")


def test_child_uses_reserved_uuid_disabled_gl_and_cpu_thread_caps(tmp_path):
    import json

    gpu_uuid = "GPU-00000000-0000-0000-0000-000000000000"
    keys = [
        "CUDA_VISIBLE_DEVICES",
        "MUJOCO_GL",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ]
    code = (
        "import os,json; print(json.dumps({k:os.environ[k] for k in "
        + repr(keys)
        + "}))"
    )
    log = tmp_path / "profile.json"
    assert execute_command([sys.executable, "-c", code], log, 3, gpu_uuid=gpu_uuid) == 0
    assert json.loads(log.read_text()) == {
        "CUDA_VISIBLE_DEVICES": gpu_uuid,
        "MUJOCO_GL": "disable",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
