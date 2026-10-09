"""Profile transport CPU fixtures; no native worker or GPU execution."""

import cProfile
import hashlib
import json
from argparse import Namespace

import pytest
from abc_bench import runner


def test_fixed_command_and_separate_destination(tmp_path):
    original = [
        "original-python",
        "-m",
        "nrh.resfit_abc_vla_run",
        "--out",
        str(tmp_path / "training"),
    ]
    result = runner.resfit_profile_command(original, tmp_path / "profile")
    assert result == [
        original[0],
        str(runner.RESFIT_PROFILE_LAUNCHER),
        "--profile-directory",
        str((tmp_path / "profile").resolve()),
        *original[3:],
    ]
    assert original[1:3] == ["-m", "nrh.resfit_abc_vla_run"]
    assert (
        result[result.index("--out") + 1]
        != result[result.index("--profile-directory") + 1]
    )


def test_launcher_pin_and_wrong_worker(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="fixed ResFiT"):
        runner.resfit_profile_command(["python", "-m", "other"], tmp_path)
    launcher = tmp_path / "launcher.py"
    launcher.write_text("wrong")
    monkeypatch.setattr(runner, "RESFIT_PROFILE_LAUNCHER", launcher)
    with pytest.raises(ValueError, match="source pin"):
        runner.resfit_profile_command(
            ["python", "-m", "nrh.resfit_abc_vla_run"], tmp_path
        )


def test_other_algorithm_refused_before_results(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "RESULTS", tmp_path / "canonical-double")
    args = Namespace(
        resfit_profile=True, algorithm="qf3-vla", results=tmp_path / "never"
    )
    with pytest.raises(ValueError, match="requires resfit-abc-vla"):
        runner.run_baseline(args)
    assert not args.results.exists()


def test_profile_gate_missing_truncated_and_complete(tmp_path):
    assert runner.resfit_profile_evidence(tmp_path)["complete"] is False
    path = tmp_path / "worker.pstats"
    profiler = cProfile.Profile()
    profiler.runcall(sum, [1, 2])
    profiler.dump_stats(str(path))
    evidence = {
        "complete": True,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "worker_return_code": 2,
    }
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(evidence))
    assert runner.resfit_profile_evidence(tmp_path)["complete"] is True
    path.write_bytes(b"truncated")
    assert runner.resfit_profile_evidence(tmp_path)["complete"] is False
    evidence.update(
        bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )
    receipt.write_text(json.dumps(evidence))
    assert runner.resfit_profile_evidence(tmp_path)["complete"] is False
    evidence["complete"] = False
    receipt.write_text(json.dumps(evidence))
    assert runner.resfit_profile_evidence(tmp_path)["complete"] is False
