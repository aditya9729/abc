"""Synthetic display fixtures are not executed learned benchmark evidence."""

import hashlib
import json
import threading
from urllib.request import urlopen

import pytest

from abc_bench.dashboard import (
    export_snapshot,
    load_results,
    load_vla_report,
    make_server,
)


def summary(successes=1):
    return {
        "completed_episodes": 50,
        "successes": successes,
        "native_successes": successes,
        "mean_successful_episode_length": 900.0 if successes else None,
        "failure_inclusive_mean_control_steps": 998.0,
        "native_current_success_only_mean_control_steps": 900.0 if successes else None,
    }


def baseline():
    return {
        "schema_version": 1,
        "status": "validated_frozen_development_evidence",
        "comparison_executed": False,
        "learned_result": None,
        "baseline": {"summary": summary()},
        "verified_artifacts": [{"path": "/must/not/read/weights.pt"}],
    }


def comparison():
    report = baseline()
    report.pop("comparison_executed")
    report.pop("learned_result")
    report.update(
        status="descriptive_fixed_development_comparison",
        learned={"summary": summary()},
    )
    pairing = {
        "counts": {"win": 0, "loss": 0, "both": 1, "neither": 49},
        "common_success_length_changes": [
            {
                "seed": 1000000,
                "baseline_control_steps": 900,
                "learned_control_steps": 850,
                "learned_minus_baseline": -50,
            }
        ],
        "common_success_mean_length_change": -50.0,
    }
    report["history_success_pairing"] = pairing
    report["native_current_success_pairing"] = pairing
    return report


def pinned(tmp_path, report):
    path = tmp_path / "report.json"
    content = json.dumps(report).encode()
    path.write_bytes(content)
    return path, hashlib.sha256(content).hexdigest()


def test_baseline_no_learned_no_artifact_traversal(tmp_path):
    path, pin = pinned(tmp_path, baseline())
    view = load_vla_report(path, pin)
    assert view["status"] == "available"
    assert view["mode"] == "baseline"
    assert view["learned"] is None
    assert view["history_success_pairing"] is None
    assert "/must/not/read" not in json.dumps(view)
    assert view["baseline"]["successes"] == 1


def test_no_report_and_missing_pin_are_unavailable(tmp_path):
    assert load_vla_report()["status"] == "unavailable"
    path, _ = pinned(tmp_path, baseline())
    assert "supplied together" in load_vla_report(path)["error"]


def test_pin_mismatch_and_later_tampering(tmp_path):
    path, pin = pinned(tmp_path, baseline())
    assert "mismatch" in load_vla_report(path, "0" * 64)["error"]
    path.write_text("{}")
    assert "mismatch" in load_vla_report(path, pin)["error"]


def test_full_descriptive_pairing_and_negative_length_change(tmp_path):
    path, pin = pinned(tmp_path, comparison())
    view = load_vla_report(path, pin)
    assert view["mode"] == "comparison"
    assert view["history_success_pairing"]["counts"]["both"] == 1
    assert (
        view["native_current_success_pairing"]["common_success_mean_length_change"]
        == -50
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r.update(status="rejected"),
        lambda r: r.update(schema_version=True),
        lambda r: r["learned"]["summary"].update(completed_episodes=49),
        lambda r: r.pop("learned"),
        lambda r: r.pop("history_success_pairing"),
        lambda r: r["history_success_pairing"]["counts"].update(neither=48),
        lambda r: r["learned"]["summary"].update(successes=2),
        lambda r: r["history_success_pairing"].update(
            common_success_mean_length_change=3
        ),
        lambda r: r["history_success_pairing"]["common_success_length_changes"][
            0
        ].update(seed=9),
    ],
)
def test_partial_or_inconsistent_comparison_not_displayed(tmp_path, mutation):
    report = comparison()
    mutation(report)
    path, pin = pinned(tmp_path, report)
    view = load_vla_report(path, pin)
    assert view["status"] == "unavailable"
    assert view["learned"] is None


@pytest.mark.parametrize(
    "raw",
    [
        b'{"schema_version":1,"schema_version":1}',
        b'{"metric":NaN}',
        b"x" * (4 * 1024 * 1024 + 1),
    ],
)
def test_unsafe_json_is_not_displayed(tmp_path, raw):
    path = tmp_path / "report.json"
    path.write_bytes(raw)
    assert (
        load_vla_report(path, hashlib.sha256(raw).hexdigest())["status"]
        == "unavailable"
    )


def test_frozen_report_cannot_contain_learned(tmp_path):
    report = baseline()
    report["learned"] = {"summary": summary()}
    path, pin = pinned(tmp_path, report)
    assert "cannot include" in load_vla_report(path, pin)["error"]


def test_empty_success_means_and_pairings_are_null(tmp_path):
    report = comparison()
    report["baseline"]["summary"] = summary(0)
    report["learned"]["summary"] = summary(0)
    for key in ("history_success_pairing", "native_current_success_pairing"):
        report[key] = {
            "counts": {"win": 0, "loss": 0, "both": 0, "neither": 50},
            "common_success_length_changes": [],
            "common_success_mean_length_change": None,
        }
    path, pin = pinned(tmp_path, report)
    assert load_vla_report(path, pin)["status"] == "available"


def test_http_and_new_snapshot_preserve_canonical_receipts(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    runs = [{"run_id": str(i), "artifacts": []} for i in range(49)]
    receipt = results / "results.json"
    receipt.write_text(json.dumps({"runs": runs}))
    original = receipt.read_bytes()
    path, pin = pinned(tmp_path, baseline())
    server = make_server(results, port=0, vla_report=path, vla_report_sha256=pin)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/api/vla-comparison", timeout=3) as response:
            assert json.load(response)["baseline"]["successes"] == 1
        with urlopen(base + "/api/results", timeout=3) as response:
            assert len(json.load(response)["runs"]) == 49
        path.write_text("{}")
        with urlopen(base + "/api/vla-comparison", timeout=3) as response:
            assert json.load(response)["status"] == "unavailable"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
    path, pin = pinned(tmp_path, baseline())
    new_snapshot = tmp_path / "new-view.html"
    export_snapshot(results, new_snapshot, vla_report=path, vla_report_sha256=pin)
    html = new_snapshot.read_text()
    assert "window.VLA_COMPARISON_SNAPSHOT=" in html
    assert pin in html
    assert "/must/not/read" not in html
    assert receipt.read_bytes() == original
    assert len(load_results(results)[0]["runs"]) == 49


def test_vla_snapshot_cannot_overwrite_historical_snapshot(tmp_path):
    path, pin = pinned(tmp_path, baseline())
    existing = tmp_path / "historical.html"
    existing.write_text("immutable old snapshot")
    with pytest.raises(FileExistsError):
        export_snapshot(tmp_path, existing, vla_report=path, vla_report_sha256=pin)
    assert existing.read_text() == "immutable old snapshot"


def test_sha_only_export_preserves_historical_snapshot(tmp_path):
    existing = tmp_path / "historical.html"
    existing.write_text("immutable old snapshot")
    with pytest.raises(FileExistsError):
        export_snapshot(tmp_path, existing, vla_report_sha256="0" * 64)
    assert existing.read_text() == "immutable old snapshot"


def test_overflow_number_in_ignored_metadata_is_rejected(tmp_path):
    content = json.dumps(baseline()).removesuffix("}") + ', "unused_metadata": 1e999}'
    path = tmp_path / "report.json"
    raw = content.encode()
    path.write_bytes(raw)
    view = load_vla_report(path, hashlib.sha256(raw).hexdigest())
    assert view["status"] == "unavailable"
    assert "Non-finite" in view["error"]
