"""Bounded CPU process regressions; these children do not train or simulate."""

import contextlib
import hashlib
import json
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from abc_bench import author_program


@contextlib.contextmanager
def ready_child(code):
    child = subprocess.Popen(
        [sys.executable, "-I", "-u", "-B", "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        assert select.select([child.stdout], [], [], 5)[0], "CPU fixture did not start"
        assert child.stdout.readline().strip() == "ready"
        yield child
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
        child.stdout.close()


def test_natural_client_exit_in_grace_is_not_signalled(tmp_path):
    interrupted = tmp_path / "interrupted"
    code = (
        "import signal,time\nfrom pathlib import Path\n"
        f"signal.signal(signal.SIGINT,lambda *_:Path({str(interrupted)!r}).write_text('bad'))\n"
        "print('ready',flush=True)\ntime.sleep(.5)\n"
    )
    with ready_child(code) as child:
        assert child.poll() is None
        assert author_program._stop(child, graceful=True) == "already_stopped"
        assert child.returncode == 0
    assert not interrupted.exists()


def test_hanging_client_gets_original_sigint_then_sigterm(tmp_path):
    interrupted = tmp_path / "interrupted"
    code = (
        "import signal,time\nfrom pathlib import Path\n"
        f"signal.signal(signal.SIGINT,lambda *_:Path({str(interrupted)!r}).write_text('seen'))\n"
        "print('ready',flush=True)\ntime.sleep(60)\n"
    )
    with ready_child(code) as child:
        started = time.monotonic()
        assert author_program._stop(child, graceful=True) == "sigterm"
        elapsed = time.monotonic() - started
        assert child.returncode == -signal.SIGTERM
        assert 24 <= elapsed < 40
    assert interrupted.read_text() == "seen"


@pytest.mark.parametrize("graceful", [False, True])
def test_child_disappears_between_poll_and_signal(monkeypatch, graceful):
    with ready_child(
        "import time\nprint('ready',flush=True)\ntime.sleep(60)\n"
    ) as child:
        original_signal = child.send_signal

        def disappear(*_):
            # Reap only this real owned child. The exception makes the race deterministic.
            original_signal(signal.SIGTERM)
            child.wait(timeout=5)
            raise ProcessLookupError("Explicit post-reap process-race fixture")

        monkeypatch.setattr(
            child, "send_signal" if graceful else "terminate", disappear
        )
        assert author_program._stop(child, graceful=graceful) == "already_stopped"
        assert child.returncode == -signal.SIGTERM


def online_config(tmp_path, client_exit):
    source = tmp_path / "source"
    ready = tmp_path / "client-ready"
    done = tmp_path / "author-done"
    server = (
        "import sys,time\nfrom pathlib import Path\n"
        f"ready=Path({str(ready)!r})\nend=time.monotonic()+5\n"
        "while not ready.exists():\n"
        " if time.monotonic()>end: raise RuntimeError('CPU client fixture did not start')\n"
        " time.sleep(.01)\n"
        "out=Path(next(x.split('=',1)[1] for x in sys.argv if x.startswith('--output_dir=')))\n"
        "out.mkdir(parents=True)\n(out/'synthetic-artifact.txt').write_text('not a checkpoint')\n"
        f"Path({str(done)!r}).write_text('done')\n"
    )
    finish = (
        "time.sleep(60)"
        if client_exit == "wait_for_sigint"
        else f"sys.exit({client_exit})"
    )
    client = (
        "import sys,time\nfrom pathlib import Path\n"
        f"Path({str(ready)!r}).write_text('ready')\n"
        f"done=Path({str(done)!r})\nend=time.monotonic()+5\n"
        "while not done.exists():\n"
        " if time.monotonic()>end: raise RuntimeError('CPU author fixture did not finish')\n"
        " time.sleep(.01)\n"
        f"time.sleep(.5)\n{finish}\n"
    )
    files = []
    for name, text in {
        "train_pi_robo.py": server,
        "client/__init__.py": "",
        "client/run_client.py": client,
        "expo_ft/agents/alg/realtime_expo_ft.py": "# CPU process stand-in only\n",
        "expo_ft/agents/alg/rtc.py": "# CPU process stand-in only\n",
    }.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        files.append(
            {
                "path": name,
                "bytes": len(path.read_bytes()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": files}))
    return {
        "schema_version": 1,
        "program": "train_pi_robo.py",
        "source_root": str(source),
        "source_manifest": str(manifest),
        "python": sys.executable,
        "arguments": [],
        "timeout_s": 10,
        "client": {"python": sys.executable, "arguments": []},
    }


def test_natural_client_zero_exit_finishes_process_only(tmp_path, monkeypatch):
    monkeypatch.setattr(author_program, "require_subset_lease", lambda: None)
    result = author_program.run(online_config(tmp_path, 0), tmp_path / "out")
    assert result["status"] == "completed"
    assert result["client_exit_code"] == result["author_exit_code"] == 0
    assert result["client_shutdown"] == "already_stopped"
    assert result["checkpoint_accepted"] is False


def test_supervisor_module_is_normally_installed():
    assert (
        Path(author_program.__file__)
        .resolve()
        .is_relative_to(Path(sys.prefix).resolve())
    )


def test_nonzero_client_exit_keeps_author_artifact_but_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(author_program, "require_subset_lease", lambda: None)
    with pytest.raises(RuntimeError, match="client exited 7"):
        author_program.run(online_config(tmp_path, 7), tmp_path / "out")
    result = json.loads((tmp_path / "out/receipt.json").read_text())
    assert result["status"] == "failed"
    assert result["author_exit_code"] == 0
    assert result["client_exit_code"] == 7
    assert result["owned_children_stopped"] is True
    assert result["checkpoint_accepted"] is False
    assert (tmp_path / "out/author-output/synthetic-artifact.txt").is_file()


def test_supervisor_sigint_client_exit_cannot_complete_job(tmp_path, monkeypatch):
    monkeypatch.setattr(author_program, "require_subset_lease", lambda: None)
    with pytest.raises(RuntimeError, match="client exited -2"):
        author_program.run(online_config(tmp_path, "wait_for_sigint"), tmp_path / "out")
    result = json.loads((tmp_path / "out/receipt.json").read_text())
    assert result["status"] == "failed"
    assert result["author_exit_code"] == 0
    assert result["client_exit_code"] == -signal.SIGINT
    assert result["client_shutdown"] == "sigint"
    assert result["owned_children_stopped"] is True
    assert result["checkpoint_accepted"] is False
    assert (tmp_path / "out/author-output/synthetic-artifact.txt").is_file()
