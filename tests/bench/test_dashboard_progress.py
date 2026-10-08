"""Explicit synthetic heartbeat fixtures test display behavior, not RL results."""

import json
import threading
from pathlib import Path
from urllib.request import urlopen

import pytest

from abc_bench.dashboard import (
    HTML,
    ROLLOUT_LOG_LIMIT,
    export_snapshot,
    load_results,
    make_server,
    rollout_log_progress,
)


def heartbeat(**changes):
    row = {
        "event": "qf3_rollout_progress",
        "namespace": "warmup-3",
        "actual_world_steps": 1000,
        "physics_ticks": 13,
        "completed_episodes": 2,
        "successes": 1,
        "first_placements": 20,
        "active_worlds": 78,
        "elapsed_collector_wall_s": 61.25,
    }
    return {**row, **changes}


def receipt(root: Path, **changes):
    row = {
        "run_id": "synthetic-only",
        "algorithm": "QF3 ABC-VLA train",
        "status": "running",
        "metrics": {},
        "artifacts": [{"label": "Execution log", "path": "run.log"}],
    }
    path = root / "results.json"
    path.write_text(json.dumps({"runs": [{**row, **changes}]}))
    return path


def test_live_diagnostics_preserve_receipt_bytes_and_missing_metrics(tmp_path):
    path = receipt(tmp_path)
    original = path.read_bytes()
    (tmp_path / "run.log").write_text(json.dumps(heartbeat()) + "\n")
    result, _ = load_results(tmp_path)
    row = result["runs"][0]
    assert row["metrics"] == {}
    assert row["status"] == "running"
    progress = row["_dashboard_rollout_progress"]
    assert progress["namespace"] == "warmup-3"
    assert progress["actual_world_steps"] == 1000
    assert progress["completed_episodes"] == 2
    assert progress["active_worlds"] == 78
    assert "per batch" in progress["scope"]
    assert progress["log_modified_at"].endswith("+00:00")
    assert path.read_bytes() == original


def test_tail_is_bounded_and_does_not_use_partial_last_line(tmp_path, monkeypatch):
    path = tmp_path / "run.log"
    row = json.dumps(heartbeat()) + "\n"
    path.write_text("x" * (ROLLOUT_LOG_LIMIT * 3) + "\n" + row + '{"event":')
    original_open = Path.open
    reads = []

    class RecordingStream:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def read(self, size):
            reads.append(size)
            return self.stream.read(size)

    def recorded_open(self, *args, **kwargs):
        return RecordingStream(original_open(self, *args, **kwargs))

    monkeypatch.setattr(Path, "open", recorded_open)
    assert rollout_log_progress(path)["actual_world_steps"] == 1000
    assert reads == [ROLLOUT_LOG_LIMIT]


def test_old_heartbeat_outside_tail_is_not_invented_as_current(tmp_path):
    path = tmp_path / "run.log"
    path.write_text(json.dumps(heartbeat()) + "\n" + "x" * ROLLOUT_LOG_LIMIT + "\n")
    assert rollout_log_progress(path) is None


@pytest.mark.parametrize(
    "changes",
    [
        {"event": "other"},
        {"namespace": "<script>untrusted</script>"},
        {"namespace": "unknown-3"},
        {"actual_world_steps": True},
        {"completed_episodes": -1},
        {"successes": 3},
        {"active_worlds": 0, "completed_episodes": 0},
        {"actual_world_steps": 1},
        {"actual_world_steps": 1041},
        {"elapsed_collector_wall_s": float("inf")},
        {"elapsed_collector_wall_s": -1},
        {"elapsed_collector_wall_s": True},
    ],
)
def test_invalid_latest_record_falls_back_to_last_valid_record(tmp_path, changes):
    path = tmp_path / "run.log"
    path.write_text(
        json.dumps(heartbeat()) + "\n" + json.dumps(heartbeat(**changes)) + "\n"
    )
    assert rollout_log_progress(path)["actual_world_steps"] == 1000


@pytest.mark.parametrize(
    "line",
    [
        "not-json\n",
        "[]\n",
        '{"event":"qf3_rollout_progress","event":"other"}\n',
        "[" * 2000 + "]" * 2000 + "\n",
        "{\xff}\n",
    ],
)
def test_malformed_only_log_is_unavailable(tmp_path, line):
    path = tmp_path / "run.log"
    path.write_bytes(line.encode("latin1"))
    assert rollout_log_progress(path) is None


@pytest.mark.parametrize(
    "namespace", ["train-15", "evaluate-learned-1-0", "evaluate-frozen-0-40"]
)
def test_native_collector_namespaces_are_supported(tmp_path, namespace):
    path = tmp_path / "run.log"
    path.write_text(json.dumps(heartbeat(namespace=namespace)) + "\n")
    assert rollout_log_progress(path)["namespace"] == namespace


def test_missing_terminal_unlisted_and_unrelated_logs_have_no_live_diagnostic(tmp_path):
    receipt(tmp_path)
    assert "_dashboard_rollout_progress" not in load_results(tmp_path)[0]["runs"][0]
    (tmp_path / "run.log").write_text(json.dumps(heartbeat()) + "\n")
    for changes in (
        {"status": "completed"},
        {"status": "failed"},
        {"algorithm": "another-method"},
        {"artifacts": []},
        {"artifacts": [{"label": "Other artifact", "path": "run.log"}]},
    ):
        receipt(tmp_path, _dashboard_rollout_progress=heartbeat(), **changes)
        row = load_results(tmp_path)[0]["runs"][0]
        assert "_dashboard_rollout_progress" not in row


@pytest.mark.parametrize("algorithm", [[], {}, ["QF3 ABC-VLA train"], None, 1])
def test_non_string_algorithm_does_not_crash_receipt_loading(tmp_path, algorithm):
    receipt(tmp_path, algorithm=algorithm)
    (tmp_path / "run.log").write_text(json.dumps(heartbeat()) + "\n")
    data, _ = load_results(tmp_path)
    assert data["runs"][0]["algorithm"] == algorithm
    assert "_dashboard_rollout_progress" not in data["runs"][0]
    assert data["errors"] == []


def test_external_and_symlink_escape_logs_are_not_read(tmp_path):
    outside = tmp_path.parent / (tmp_path.name + "-external.log")
    outside.write_text(json.dumps(heartbeat()) + "\n")
    try:
        (tmp_path / "escape.log").symlink_to(outside)
        for raw in (str(outside), "escape.log", "../" + outside.name):
            receipt(tmp_path, artifacts=[{"label": "Execution log", "path": raw}])
            result, paths = load_results(tmp_path)
            assert paths == {}
            assert "_dashboard_rollout_progress" not in result["runs"][0]
    finally:
        outside.unlink()


def test_http_refresh_and_offline_snapshot_use_same_safe_display_contract(tmp_path):
    receipt(tmp_path)
    path = tmp_path / "run.log"
    path.write_text(json.dumps(heartbeat()) + "\n")
    server = make_server(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/api/results", timeout=3) as response:
            payload = json.load(response)
        snapshot = tmp_path / "snapshot.html"
        export_snapshot(tmp_path, snapshot)
        document = snapshot.read_text()
        embedded, _ = json.JSONDecoder().raw_decode(
            document.split("window.BENCHMARK_SNAPSHOT=", 1)[1]
        )
        assert (
            embedded["runs"][0]["_dashboard_rollout_progress"]
            == payload["runs"][0]["_dashboard_rollout_progress"]
        )
        assert "if(!window.BENCHMARK_SNAPSHOT)setInterval(load,30000)" in HTML
        assert "Last rollout log" in HTML
        assert "textContent=text" in HTML
        path.write_text(json.dumps(heartbeat(actual_world_steps=1020)) + "\n")
        with urlopen(base + "/api/results", timeout=3) as response:
            updated = json.load(response)
        assert (
            updated["runs"][0]["_dashboard_rollout_progress"]["actual_world_steps"]
            == 1020
        )
        assert snapshot.read_text() == document
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
