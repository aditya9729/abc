"""Real CLI parser dispatch with explicit worker doubles; no model or GPU."""

import json
import sys

import pytest

from abc_bench import runner, subset


def test_author_jobs_do_not_require_a_legacy_checkpoint(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    calls = []

    def run_job(args):
        calls.append(args)
        assert args.algorithm == "author-subset"
        assert args.training_config == tmp_path / "author-job.json"
        return {"status": "completed", "explicit_worker_double": True}

    def reject_gpu(*_args, **_kwargs):
        pytest.fail("CLI parser/dispatch must not request a GPU")

    monkeypatch.setattr(subset, "run_job", run_job)
    monkeypatch.setattr(runner, "inspect_reserved_gpu", reject_gpu)
    monkeypatch.setattr(runner, "run_baseline", reject_gpu)
    # Both the omitted default checkpoint and an explicit missing file must
    # reach the author job dispatcher through the real parser.
    for legacy_args in ([], ["--checkpoint", str(tmp_path / "missing-legacy.pt")]):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "abc_bench.runner",
                "--algorithm",
                "author-subset",
                "--training-config",
                str(tmp_path / "author-job.json"),
                *legacy_args,
            ],
        )
        runner.main()
        assert json.loads(capsys.readouterr().out)["status"] == "completed"
        assert not calls[-1].checkpoint.is_file()
    assert len(calls) == 2


def test_baseline_missing_checkpoint_fails_before_worker_or_gpu(
    tmp_path, monkeypatch, capsys
):
    calls = []

    def forbidden(*_args, **_kwargs):
        calls.append("worker_or_gpu")
        pytest.fail("Missing baseline checkpoint must fail before a worker/GPU")

    monkeypatch.setattr(runner, "run_baseline", forbidden)
    monkeypatch.setattr(runner, "inspect_reserved_gpu", forbidden)
    monkeypatch.setattr(subset, "run_job", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "abc_bench.runner",
            "--algorithm",
            "baseline",
            "--checkpoint",
            str(tmp_path / "missing-baseline.pt"),
        ],
    )
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert "checkpoint missing" in capsys.readouterr().err
    assert calls == []
