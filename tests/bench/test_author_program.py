"""Process supervision tests; these explicit stand-ins never simulate learning."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from abc_bench import author_program


def fixture_config(tmp_path, program_text):
    source = tmp_path / "source"
    files = []
    for name, text in {
        "train_offline_rtc.py": program_text,
        "expo_ft/agents/alg/realtime_expo_ft.py": "# Explicit process-test stand-in\n",
        "expo_ft/agents/alg/rtc.py": "# Explicit process-test stand-in\n",
    }.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        data = path.read_bytes()
        files.append(
            {
                "path": name,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": files}))
    return {
        "schema_version": 1,
        "program": "train_offline_rtc.py",
        "source_root": str(source),
        "source_manifest": str(manifest),
        "python": sys.executable,
        "arguments": [],
        "timeout_s": 5,
    }


def test_zero_exit_is_not_checkpoint_acceptance(tmp_path, monkeypatch):
    monkeypatch.setattr(author_program, "require_subset_lease", lambda: None)
    result = author_program.run(
        fixture_config(tmp_path, "print('process fixture')\n"), tmp_path / "out"
    )
    assert result["status"] == "completed"
    assert result["checkpoint_accepted"] is False
    assert result["owned_children_stopped"] is True


def test_changed_source_refuses_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(author_program, "require_subset_lease", lambda: None)
    config = fixture_config(tmp_path, "print('process fixture')\n")
    Path(config["source_root"], "train_offline_rtc.py").write_text(
        "raise AssertionError('must not run')\n"
    )
    with pytest.raises(ValueError, match="differs"):
        author_program.run(config, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_timeout_reaps_original_process(tmp_path, monkeypatch):
    monkeypatch.setattr(author_program, "require_subset_lease", lambda: None)
    config = fixture_config(tmp_path, "import time\ntime.sleep(30)\n")
    config["timeout_s"] = 0.1
    with pytest.raises(TimeoutError):
        author_program.run(config, tmp_path / "out")
    receipt = json.loads((tmp_path / "out/receipt.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["owned_children_stopped"] is True
    assert not Path(f"/proc/{receipt['author_pid']}").exists()


def test_failed_program_stays_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(author_program, "require_subset_lease", lambda: None)
    config = fixture_config(tmp_path, "raise RuntimeError('explicit test failure')\n")
    with pytest.raises(RuntimeError, match="exited 1"):
        author_program.run(config, tmp_path / "out")
    receipt = json.loads((tmp_path / "out/receipt.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["author_exit_code"] == 1
    assert receipt["owned_children_stopped"] is True


def test_graceful_client_interrupt_flushes_exit_cleanup(tmp_path):
    marker = tmp_path / "closed.txt"
    code = (
        "import atexit,time\nfrom pathlib import Path\n"
        f"atexit.register(lambda:Path({str(marker)!r}).write_text('closed'))\n"
        "print('ready',flush=True)\ntime.sleep(60)\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-u", "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        assert child.stdout.readline().strip() == "ready"
        assert author_program._stop(child, graceful=True) == "sigint"
        assert marker.read_text() == "closed"
        assert child.poll() is not None
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
